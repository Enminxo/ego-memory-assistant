"""Neon's scene camera, live, through the Companion app's WebSocket.

The Companion serves its sensors as RTSP. Its plain RTSP server (port 8086)
sends the video over UDP, and on the office Wi-Fi a keyframe (over 100
packets sent at once) never arrived whole: 0 decodable frames. The same RTSP
also runs inside a WebSocket on port 8686, the one Pupil Labs' Monitor web
app uses. There the video comes interleaved in the TCP connection: 0 packets
lost and 0.07 s median lag (measured 2026-10-26 with facewatch's
feasibility/link_probe.py, on which this module is based).

    ws://<phone>:8686/rtsp-over-websocket   (subprotocol "binary")
      ──▶ RTSP DESCRIBE, SETUP (interleaved=0-1), PLAY, OPTIONS every 30 s
      ◀── "$" channel 0: RTP carrying H.264    channel 1: RTCP sender reports

Two quirks of the phone: SETUP answers with a UDP transport, yet the video
arrives on channel 0 anyway; and the H.264 parameter sets (SPS/PPS) come only
in the SDP, never inside the video, so they are put in front of every
keyframe before decoding.

`SceneStream` keeps the latest decoded frame in a thread, for the live
session and the page's live view. It needs only `websockets` and PyAV; the
Pupil Labs packages (phone status, clock, recording) are added by
neon.NeonLiveSource.
"""

from __future__ import annotations

import asyncio
import base64
import io
import re
import struct
import threading
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

import numpy as np

WS_PORT = 8686
WS_PATH = "/rtsp-over-websocket"
RTP_CHANNEL, RTCP_CHANNEL = 0, 1
CLOCK_RATE = 90_000  # RTP clock of H.264 video
NTP_TO_UNIX = 2_208_988_800
START_CODE = b"\x00\x00\x00\x01"
KEEPALIVE_SEC = 30.0
STALL_SEC = 10.0


class StaleStream(ConnectionError):
    """The stream is open but no recent frame has arrived."""


# --- RTSP inside the WebSocket -------------------------------------------------

@dataclass(frozen=True)
class RtspResponse:
    status: int
    headers: dict[str, str]
    body: str


class StreamParser:
    """Split the byte stream into RTSP responses and interleaved packets ("$")."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, data: bytes) -> list[RtspResponse | tuple[int, bytes]]:
        self._buffer.extend(data)
        items: list[RtspResponse | tuple[int, bytes]] = []
        while self._buffer:
            if self._buffer[0] == 0x24:  # "$", channel, 2-byte length, packet
                if len(self._buffer) < 4:
                    break
                channel = self._buffer[1]
                length = struct.unpack(">H", self._buffer[2:4])[0]
                if len(self._buffer) < 4 + length:
                    break
                items.append((channel, bytes(self._buffer[4:4 + length])))
                del self._buffer[:4 + length]
                continue
            if not self._buffer.startswith(b"RTSP/"[:len(self._buffer)]):
                raise ValueError(f"unexpected data in the stream: {bytes(self._buffer[:40])!r}")
            end = self._buffer.find(b"\r\n\r\n")
            if end < 0:
                break
            lines = self._buffer[:end].decode(errors="replace").split("\r\n")
            headers: dict[str, str] = {}
            for line in lines[1:]:
                key, _, value = line.partition(":")
                headers[key.strip().lower()] = value.strip()
            length = int(headers.get("content-length", "0"))
            if len(self._buffer) < end + 4 + length:
                break
            body = self._buffer[end + 4:end + 4 + length].decode(errors="replace")
            items.append(RtspResponse(int(lines[0].split()[1]), headers, body))
            del self._buffer[:end + 4 + length]
        return items


class RtspClient:
    """RTSP requests, one at a time, over anything that can send bytes."""

    def __init__(self, send: Callable[[bytes], Awaitable[None]]) -> None:
        self.responses: asyncio.Queue[RtspResponse] = asyncio.Queue()
        self.session: str | None = None
        self._send = send
        self._cseq = 0
        self._lock = asyncio.Lock()

    async def request(self, method: str, uri: str,
                      headers: dict[str, str] | None = None) -> RtspResponse:
        async with self._lock:
            self._cseq += 1
            lines = [f"{method} {uri} RTSP/1.0", f"CSeq: {self._cseq}"]
            lines += [f"{key}: {value}" for key, value in (headers or {}).items()]
            if self.session is not None:
                lines.append(f"Session: {self.session}")
            await self._send(("\r\n".join(lines) + "\r\n\r\n").encode())
            response = await asyncio.wait_for(self.responses.get(), timeout=10)
            if response.status != 200:
                raise ConnectionError(f"RTSP {method} {uri} answered {response.status}")
            return response


def parse_sdp(body: str) -> tuple[list[bytes], str]:
    """SPS/PPS from sprop-parameter-sets, and the control of the video track.

    With the microphone on in the Companion app the audio track comes first
    (trackID=0), so both are looked for inside the m=video section.
    """
    video = re.search(r"^m=video.*?(?=^m=|\Z)", body, re.MULTILINE | re.DOTALL)
    if video is None:
        raise ValueError(f"the SDP has no video track:\n{body}")
    section = video.group(0)
    sprop_match = re.search(r"sprop-parameter-sets=([^;\r\n]+)", section)
    sprop = [base64.b64decode(part) for part in sprop_match.group(1).split(",")] if sprop_match else []
    control_match = re.search(r"a=control:(\S+)", section)
    if control_match is None:
        raise ValueError(f"the video track has no a=control:\n{section}")
    return sprop, control_match.group(1)


def parse_sender_reports(packet: bytes) -> list[tuple[int, float]]:
    """(RTP timestamp, Unix time on the phone's clock) of each RTCP sender report."""
    reports = []
    pos = 0
    while pos + 4 <= len(packet):
        packet_type = packet[pos + 1]
        length = (struct.unpack(">H", packet[pos + 2:pos + 4])[0] + 1) * 4
        if packet_type == 200 and pos + 20 <= len(packet):
            ntp_sec, ntp_frac, rtp_ts = struct.unpack(">III", packet[pos + 8:pos + 20])
            reports.append((rtp_ts, ntp_sec - NTP_TO_UNIX + ntp_frac / 2**32))
        pos += length
    return reports


def capture_time(rtp_ts: int, report: tuple[int, float] | None) -> float | None:
    """When a frame was taken (Unix seconds, phone clock), from the last sender report."""
    if report is None:
        return None
    ref_ts, ref_unix = report
    delta = (rtp_ts - ref_ts + 2**31) % 2**32 - 2**31  # the 32-bit RTP clock wraps around
    return ref_unix + delta / CLOCK_RATE


# --- RTP to H.264 frames ---------------------------------------------------------

@dataclass(frozen=True)
class AccessUnit:
    """One whole video frame, as Annex-B H.264."""

    data: bytes
    rtp_ts: int
    key: bool


class H264Depacketizer:
    """RTP packets of one H.264 stream to whole frames, holding back damaged ones.

    Between keyframes every frame is coded as a change to the one before, so
    a lost packet spoils the picture until the next keyframe. A gap in the
    sequence numbers therefore marks the stream dirty, and nothing is handed
    out until a complete keyframe arrives. Over the WebSocket nothing should
    be lost, but a broken picture must never reach the describer.
    """

    def __init__(self) -> None:
        self.lost = 0
        self._expected: int | None = None
        self._ts: int | None = None
        self._nals: list[bytes] = []
        self._fu: bytearray | None = None
        self._damaged = True  # the first frame may have started before we joined
        self._clean = False   # false until the first complete keyframe

    def resync(self) -> None:
        """Hand out nothing until the next complete keyframe (the decoder rejected a frame)."""
        self._clean = False

    def push(self, packet: bytes) -> AccessUnit | None:
        """Take one RTP packet; return the frame it completes, if whole and clean."""
        if len(packet) < 12 or packet[0] >> 6 != 2:
            return None
        csrc_count = packet[0] & 0x0F
        marker = bool(packet[1] & 0x80)
        seq, ts = struct.unpack(">HI", packet[2:8])
        start = 12 + 4 * csrc_count
        if packet[0] & 0x10:  # header extension
            ext_words = struct.unpack(">H", packet[start + 2:start + 4])[0]
            start += 4 + 4 * ext_words
        end = len(packet) - (packet[-1] if packet[0] & 0x20 else 0)  # padding

        gap = 0
        if self._expected is not None:
            gap = (seq - self._expected) & 0xFFFF
            if gap >= 0x8000:  # late or repeated: already counted as lost
                return None
        self._expected = (seq + 1) & 0xFFFF
        if ts != self._ts:  # a new frame
            if self._ts is not None:
                if self._nals or self._fu is not None:  # the last one never got its end
                    self._clean = False
                self._damaged = False
            self._ts, self._nals, self._fu = ts, [], None
        if gap:
            self.lost += gap
            self._damaged = True
            self._clean = False
            self._fu = None  # a NAL missing pieces cannot be put back together
        self._depacketize(packet[start:end])
        return self._finish() if marker else None

    def _depacketize(self, payload: bytes) -> None:
        if not payload:
            return
        nal_type = payload[0] & 0x1F
        if 1 <= nal_type <= 23:  # a single NAL
            self._nals.append(payload)
        elif nal_type == 24:  # STAP-A: several NALs in one packet
            pos = 1
            while pos + 2 <= len(payload):
                size = struct.unpack(">H", payload[pos:pos + 2])[0]
                self._nals.append(payload[pos + 2:pos + 2 + size])
                pos += 2 + size
        elif nal_type == 28 and len(payload) > 1:  # FU-A: one NAL split over packets
            if payload[1] & 0x80:
                self._fu = bytearray([(payload[0] & 0xE0) | (payload[1] & 0x1F)])
            if self._fu is None:
                self._damaged = True  # its start was lost
                return
            self._fu.extend(payload[2:])
            if payload[1] & 0x40:
                self._nals.append(bytes(self._fu))
                self._fu = None

    def _finish(self) -> AccessUnit | None:
        nals = [nal for nal in self._nals if nal]
        key = any(nal[0] & 0x1F == 5 for nal in nals)
        complete = not self._damaged and self._fu is None and bool(nals)
        if key:
            self._clean = complete
        elif not complete:
            self._clean = False
        unit = AccessUnit(b"".join(START_CODE + nal for nal in nals), self._ts or 0, key)
        self._nals, self._fu, self._damaged = [], None, False
        return unit if complete and self._clean else None


# --- Receiving and decoding ---------------------------------------------------------

def ws_url(host: str, port: int = WS_PORT) -> str:
    netloc = f"[{host}]" if ":" in host else host  # an IPv6 address needs brackets
    return f"ws://{netloc}:{port}{WS_PATH}"


async def _keepalive(rtsp: RtspClient, uri: str) -> None:
    while True:
        await asyncio.sleep(KEEPALIVE_SEC)
        await rtsp.request("OPTIONS", uri)


async def receive_scene(host: str, port: int = WS_PORT, stall_sec: float = STALL_SEC,
                        open_timeout: float = 10.0) -> AsyncIterator[tuple[object, float | None]]:
    """Every clean frame of the scene camera: (av.VideoFrame, capture time or None).

    The capture time is on the phone's clock and is known from the first RTCP
    sender report on (about every 3 s). Raises ConnectionError if the phone
    closes the connection, and TimeoutError after `stall_sec` without any video
    data. A slow link that is still sending is not cut: on a poor Wi-Fi the
    first keyframe can take several seconds to arrive, and reconnecting would
    only start the wait over.
    """
    import av
    import websockets

    netloc = f"[{host}]" if ":" in host else host
    uri = f"rtsp://{netloc}:{port}/?camera=world"
    ws = await websockets.connect(ws_url(host, port), subprotocols=["binary"], max_size=None,
                                  open_timeout=open_timeout)
    parser = StreamParser()
    depacketizer = H264Depacketizer()
    rtsp = RtspClient(ws.send)
    units: asyncio.Queue[AccessUnit | None] = asyncio.Queue()
    last_report: list[tuple[int, float]] = []
    last_packet = [time.monotonic()]
    closed: list[BaseException] = []

    async def read() -> None:
        try:
            async for message in ws:
                for item in parser.feed(message if isinstance(message, bytes) else message.encode()):
                    if isinstance(item, RtspResponse):
                        rtsp.responses.put_nowait(item)
                    elif item[0] == RTP_CHANNEL:
                        last_packet[0] = time.monotonic()
                        unit = depacketizer.push(item[1])
                        if unit is not None:
                            units.put_nowait(unit)
                    elif item[0] == RTCP_CHANNEL:
                        reports = parse_sender_reports(item[1])
                        if reports:
                            last_report[:] = reports[-1:]
        except Exception as exc:  # re-raised by the consumer below
            closed.append(exc)
        finally:
            units.put_nowait(None)

    reader = asyncio.create_task(read())
    keepalive: asyncio.Task | None = None
    base = uri
    try:
        describe = await rtsp.request("DESCRIBE", uri, {"Accept": "application/sdp"})
        base = describe.headers.get("content-base", uri)
        sprop, control = parse_sdp(describe.body)
        headers = b"".join(START_CODE + nal for nal in sprop)
        track = control if control.startswith("rtsp://") else base.rstrip("/") + "/" + control
        # The phone answers with a UDP transport but sends on channel 0 anyway.
        setup = await rtsp.request("SETUP", track,
                                   {"Transport": "RTP/AVP/TCP;unicast;interleaved=0-1"})
        rtsp.session = setup.headers.get("session", "").split(";")[0] or None
        await rtsp.request("PLAY", base, {"Range": "npt=0.000-"})
        keepalive = asyncio.create_task(_keepalive(rtsp, base))
        last_packet[0] = time.monotonic()

        decoder = av.CodecContext.create("h264", "r")
        while True:
            try:
                unit = await asyncio.wait_for(units.get(), timeout=1.0)
            except asyncio.TimeoutError:
                if time.monotonic() - last_packet[0] > stall_sec:
                    raise TimeoutError(f"no video data from {ws_url(host, port)} "
                                       f"in {stall_sec:.0f} s") from None
                continue
            if unit is None:
                raise ConnectionError(f"{ws_url(host, port)} closed the connection"
                                      + (f": {closed[0]!r}" if closed else ""))
            try:
                frames = decoder.decode(av.Packet((headers if unit.key else b"") + unit.data))
            except av.error.FFmpegError:  # a frame the decoder rejects: wait for a keyframe
                depacketizer.resync()
                continue
            for frame in frames:
                yield frame, capture_time(unit.rtp_ts, last_report[0] if last_report else None)
    finally:
        if keepalive is not None:
            keepalive.cancel()
        try:
            await asyncio.wait_for(rtsp.request("TEARDOWN", base), timeout=2)
        except Exception:  # closing politely is optional; the socket closes anyway
            pass
        reader.cancel()
        await ws.close()


class SceneStream:
    """The latest frame of the scene camera, read continuously in a thread.

    The stream hands out all 30 frames per second in order, so a consumer
    that wants a frame every couple of seconds would fall further and further
    behind it. A thread drains it instead and keeps the most recent frame,
    decoded to pixels only when asked for. If the connection drops, the
    thread reconnects every `retry_sec` until close().

    latest() raises StaleStream when the newest frame is older than
    `stale_sec`: a picture that stopped changing must not be described as if
    it were the present.
    """

    def __init__(self, host: str, port: int = WS_PORT, stale_sec: float = 5.0,
                 retry_sec: float = 2.0, stall_sec: float = STALL_SEC) -> None:
        self.host = host
        self.port = port
        self.url = ws_url(host, port)
        self.stale_sec = stale_sec
        self.retry_sec = retry_sec
        self.stall_sec = stall_sec
        self.frames_read = 0
        self.reconnects = 0
        self.error: BaseException | None = None  # the last one; the thread keeps trying
        # This machine's clock minus the phone's, for lag(); None if unknown.
        self.clock_offset_s: float | None = None
        self._frame = None
        self._taken: float | None = None
        self._received = 0.0
        self._arrived = 0.0
        self._recent: deque[float] = deque(maxlen=60)
        self._lock = threading.Lock()
        self._first = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="neon-scene", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                asyncio.run(self._stream())
            except Exception as exc:  # reported by latest() and status, not lost in this thread
                self.error = exc
            if self._stop.wait(self.retry_sec):
                break
            self.reconnects += 1

    async def _stream(self) -> None:
        async def receive() -> None:
            async for frame, taken in receive_scene(self.host, self.port, self.stall_sec):
                now = time.monotonic()
                with self._lock:
                    self._frame, self._taken, self._received = frame, taken, now
                    self._arrived = time.time()
                    self._recent.append(now)
                    self.frames_read += 1
                self._first.set()

        async def stopped() -> None:
            while not self._stop.is_set():
                await asyncio.sleep(0.2)

        tasks = [asyncio.create_task(receive()), asyncio.create_task(stopped())]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)  # lets the stream say TEARDOWN
        for task in done:
            task.result()  # re-raise what ended the stream

    def _latest_frame(self, timeout: float) -> tuple[object, float | None]:
        if not self._first.wait(timeout):
            detail = f": {self.error!r}" if self.error is not None else ""
            raise TimeoutError(f"no scene frame from {self.url} in {timeout:.0f} s{detail}")
        with self._lock:
            frame, taken, age = self._frame, self._taken, time.monotonic() - self._received
        if age > self.stale_sec:
            detail = f" ({self.error!r})" if self.error is not None else ""
            raise StaleStream(f"no scene frame for {age:.0f} s{detail}")
        return frame, taken

    def latest(self, timeout: float = 10.0) -> tuple[np.ndarray, float | None]:
        """Most recent frame as RGB, and when it was taken (Unix seconds, phone clock)."""
        frame, taken = self._latest_frame(timeout)
        return frame.to_ndarray(format="rgb24"), taken

    def latest_rgb(self) -> np.ndarray:
        return self.latest()[0]

    def latest_jpeg(self, width: int = 640, quality: int = 75) -> bytes:
        """The most recent frame as a small JPEG, for the page's live view."""
        frame, _ = self._latest_frame(timeout=0.0)
        height = round(frame.height * width / frame.width / 2) * 2
        out = io.BytesIO()
        frame.reformat(width=width, height=height).to_image().save(out, "JPEG", quality=quality)
        return out.getvalue()

    @property
    def fps(self) -> float:
        """Frames per second received over the last couple of seconds."""
        with self._lock:
            recent = [t for t in self._recent if time.monotonic() - t < 2.0]
        return (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) > 1 else 0.0

    def lag(self) -> float | None:
        """Seconds from the newest frame being taken on the phone to its arrival here.

        Over TCP nothing is lost, but when the Wi-Fi carries less than the
        video's bitrate the frames queue up on the phone and arrive late
        (0.07 s on a good morning, 15 to 30 s on a congested afternoon). Past
        about 20 s the WebSocket's keepalive ping, stuck behind the video,
        times out and the connection is reopened.
        """
        with self._lock:
            taken, arrived = self._taken, self._arrived
        if taken is None or self.clock_offset_s is None:
            return None
        return arrived - (taken + self.clock_offset_s)

    def video_age(self) -> float | None:
        """Seconds since the last frame arrived; None before the first one."""
        if not self._first.is_set():
            return None
        with self._lock:
            return time.monotonic() - self._received

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)
