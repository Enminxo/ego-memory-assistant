"""A fake Neon Companion app: RTSP over a WebSocket, streaming real H.264.

It behaves like the phone measured on 2026-10-26: the SDP lists the audio
track first, SPS/PPS travel only in the SDP, SETUP answers with a UDP
transport, and the video comes interleaved on channel 0 anyway, with RTCP
sender reports on channel 1.
"""

from __future__ import annotations

import asyncio
import base64
import re
import struct
import threading
from fractions import Fraction

import av
import numpy as np
from websockets.asyncio.server import serve

NTP_TO_UNIX = 2_208_988_800


def split_nals(data: bytes) -> list[bytes]:
    """NAL units of an Annex-B byte string."""
    return [nal.lstrip(b"\x00") for nal in re.split(b"\x00\x00\x01", data) if nal.strip(b"\x00")]


def encode_flat_frames(values: list[int], width: int = 64, height: int = 48,
                       keyint: int = 10) -> tuple[list[bytes], list[list[bytes]]]:
    """Flat grey frames of the given values as H.264: (SPS and PPS, the NALs of each frame)."""
    codec = av.CodecContext.create("libx264", "w")
    codec.width, codec.height, codec.pix_fmt = width, height, "yuv420p"
    codec.time_base, codec.framerate = Fraction(1, 30), Fraction(30)
    codec.gop_size, codec.max_b_frames = keyint, 0
    codec.options = {"preset": "ultrafast", "tune": "zerolatency"}
    packets = []
    for i, value in enumerate(values):
        frame = av.VideoFrame.from_ndarray(np.full((height, width, 3), value, np.uint8),
                                           format="rgb24").reformat(format="yuv420p")
        frame.pts = i
        packets += codec.encode(frame)
    packets += codec.encode(None)
    sprop: dict[int, bytes] = {}
    frames = []
    for packet in packets:
        nals = split_nals(bytes(packet))
        for nal in nals:
            if nal[0] & 0x1F in (7, 8):  # the phone sends these only in the SDP
                sprop.setdefault(nal[0] & 0x1F, nal)
        frames.append([nal for nal in nals if nal[0] & 0x1F not in (7, 8)])
    return [sprop[7], sprop[8]], frames


def rtp_packets(nals: list[bytes], seq: int, ts: int, max_payload: int) -> list[bytes]:
    """One frame as RTP: small NALs whole, large ones split as FU-A; marker on the last."""
    payloads = []
    for nal in nals:
        if len(nal) <= max_payload:
            payloads.append(nal)
            continue
        indicator = (nal[0] & 0xE0) | 28
        body = nal[1:]
        chunks = [body[i:i + max_payload] for i in range(0, len(body), max_payload)]
        for k, chunk in enumerate(chunks):
            header = (nal[0] & 0x1F) | (0x80 if k == 0 else 0) | (0x40 if k == len(chunks) - 1 else 0)
            payloads.append(bytes([indicator, header]) + chunk)
    return [struct.pack(">BBHII", 0x80, (0x80 if i == len(payloads) - 1 else 0) | 96,
                        (seq + i) & 0xFFFF, ts & 0xFFFFFFFF, 0x4D9692AA) + payload
            for i, payload in enumerate(payloads)]


def sender_report(rtp_ts: int, unix: float) -> bytes:
    ntp = unix + NTP_TO_UNIX
    return struct.pack(">BBHIIIIII", 0x80, 200, 6, 0x4D9692AA, int(ntp),
                       int((ntp % 1) * 2**32), rtp_ts & 0xFFFFFFFF, 0, 0)


def interleave(channel: int, packet: bytes) -> bytes:
    return b"$" + bytes([channel]) + struct.pack(">H", len(packet)) + packet


class FakeCompanion:
    """Serves `frames` in a loop at `fps` on ws://127.0.0.1:<port>/rtsp-over-websocket.

    `close_after` frames into the first connection, the phone hangs up, to
    test reconnecting. `skip_packet` drops the RTP packet with that index of
    the first connection, to test damaged frames.
    """

    def __init__(self, sprop: list[bytes], frames: list[list[bytes]], fps: float = 60.0,
                 max_payload: int = 200, close_after: int | None = None,
                 skip_packet: int | None = None) -> None:
        self.sprop, self.frames, self.fps = sprop, frames, fps
        self.max_payload = max_payload
        self.close_after, self.skip_packet = close_after, skip_packet
        self.connections = 0
        self.requests: list[str] = []
        self.port = 0
        self._ready = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stopped: asyncio.Event | None = None
        self._thread = threading.Thread(target=lambda: asyncio.run(self._main()), daemon=True)

    def __enter__(self) -> FakeCompanion:
        self._thread.start()
        self._ready.wait(5)
        return self

    def __exit__(self, *exc: object) -> None:
        self._loop.call_soon_threadsafe(self._stopped.set)
        self._thread.join(5)

    async def _main(self) -> None:
        self._loop, self._stopped = asyncio.get_running_loop(), asyncio.Event()
        async with serve(self._handle, "127.0.0.1", 0, subprotocols=["binary"]) as server:
            self.port = server.sockets[0].getsockname()[1]
            self._ready.set()
            await self._stopped.wait()

    def _sdp(self) -> str:
        sprop = ",".join(base64.b64encode(nal).decode() for nal in self.sprop)
        return ("v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\ns=Neon\r\nt=0 0\r\n"
                "m=audio 0 RTP/AVP 97\r\na=rtpmap:97 MPEG4-GENERIC/8000/1\r\na=control:trackID=0\r\n"
                "m=video 0 RTP/AVP 96\r\na=rtpmap:96 H264/90000\r\n"
                f"a=fmtp:96 packetization-mode=1;sprop-parameter-sets={sprop}\r\n"
                "a=control:trackID=1\r\n")

    async def _handle(self, ws) -> None:
        self.connections += 1
        first = self.connections == 1
        sender = None
        try:
            async for message in ws:
                text = message.decode() if isinstance(message, bytes) else message
                lines = text.split("\r\n")
                method = lines[0].split()[0]
                self.requests.append(method)
                cseq = next(line.split(":", 1)[1].strip() for line in lines if line.startswith("CSeq"))
                headers, body = [f"CSeq: {cseq}"], ""
                if method == "DESCRIBE":
                    body = self._sdp()
                    headers += ["Content-Base: rtsp://127.0.0.1/?camera=world/",
                                "Content-Type: application/sdp", f"Content-Length: {len(body)}"]
                elif method == "SETUP":
                    headers += ["Transport: RTP/AVP/UDP;unicast;destination=127.0.0.1;"
                                "client_port=0-1;server_port=50000-50001;mode=play",
                                "Session: 12345678;timeout=60"]
                await ws.send(("RTSP/1.0 200 OK\r\n" + "\r\n".join(headers) + "\r\n\r\n" + body).encode())
                if method == "PLAY":
                    sender = asyncio.create_task(self._send(ws, first))
        finally:
            if sender is not None:
                sender.cancel()

    async def _send(self, ws, first: bool) -> None:
        seq, ts, sent, i = 1000, 3_000_000, 0, 0
        loop = asyncio.get_running_loop()
        while True:
            if i % 30 == 0:
                await ws.send(interleave(1, sender_report(ts, loop.time() + 1.7e9)))
            for packet in rtp_packets(self.frames[i % len(self.frames)], seq, ts, self.max_payload):
                if not (first and sent == self.skip_packet):
                    await ws.send(interleave(0, packet))
                seq, sent = seq + 1, sent + 1
            ts += 90_000 // 30
            i += 1
            if first and self.close_after is not None and i == self.close_after:
                await ws.close()
                return
            await asyncio.sleep(1 / self.fps)
