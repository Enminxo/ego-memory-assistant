"""The live diary through the phone: photos pushed by phone_relay.py.

The direct way in (neon_stream.py) brings the whole video over the Wi-Fi and
picks the frames here; on a busy Wi-Fi it fell behind and clips were lost.
The phone relay picks the frames on the phone instead and sends only those,
each one numbered by clip and slot:

    relay ──POST /api/phone/photo?token&clip&slot&taken&pending──▶ app ─▶ PhoneSource
                                                                         └─▶ whole clips ─▶ LiveSession

A clip is complete when its last slot arrives; the relay sends in order, so a
photo of a later clip means the earlier one will get nothing more, and that
clip is dropped and reported rather than described with a hole in it. The
timeline is the phone's: clip k covers seconds [k * clip_sec, (k + 1) * clip_sec)
of the session, whenever its photos happen to arrive.

PhoneSource also answers what the page asks of any live source (the latest
picture, photos per second, lag, age), so the live view works unchanged.
"""

from __future__ import annotations

import io
import json
import queue
import secrets
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from collections.abc import Callable

import numpy as np
from PIL import Image

from egomem.clips import Clip, frame_offsets
from egomem.neon_stream import StaleStream

RELAY_PORT = 8770


class RelayClient:
    """The relay's commands, over HTTP (phone_relay.py on the phone)."""

    def __init__(self, url: str, timeout: float = 5.0) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, body: dict) -> dict:
        request = urllib.request.Request(f"{self.url}{path}", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.URLError as exc:
            raise ConnectionError(f"the phone relay at {self.url} did not answer: {exc}") from exc

    def diary_start(self, port: int, token: str, clip_sec: float, n_frames: int) -> dict:
        """Start sending the diary's photos to this machine's app, on `port`."""
        return self._post("/diary/start", {"port": port, "token": token,
                                           "clip_sec": clip_sec, "n_frames": n_frames})

    def diary_stop(self) -> dict:
        return self._post("/diary/stop", {})

    def say(self, text: str) -> dict:
        return self._post("/say", {"text": text})


class PhoneSource:
    """Photos from the phone relay, put together into the diary's clips.

    `token` tells this session's photos from those of an earlier one the
    relay may still be sending. `relay` is None in the tests.
    """

    reconnects = 0  # the relay retries each photo itself; nothing reconnects here

    def __init__(self, relay: RelayClient | None, clip_sec: float, n_frames: int,
                 stale_sec: float = 10.0, log: Callable[[str], None] = print) -> None:
        self.relay = relay
        self.clip_sec = clip_sec
        self.n_frames = n_frames
        self.stale_sec = stale_sec
        self.log = log
        self.token = secrets.token_hex(8)
        self.error: BaseException | None = None
        self.pending = 0  # photos still waiting on the phone, as of the last one
        self.photos_received = 0
        self.clips: queue.Queue[Clip] = queue.Queue()
        self._building: tuple[int, dict[int, np.ndarray]] | None = None
        self._rgb: np.ndarray | None = None
        self._jpeg: tuple[int, int, bytes] | None = None  # (photo number, width, bytes)
        self._taken = 0.0
        self._arrived = 0.0
        self._received = 0.0
        self._recent: deque[float] = deque(maxlen=8)
        self._lock = threading.Lock()

    def add_photo(self, token: str, clip: int, slot: int, taken: float, pending: int,
                  jpeg: bytes) -> None:
        """Take one photo from the relay; a clip that becomes complete goes to next_clip()."""
        if token != self.token:
            raise PermissionError("this photo belongs to another live session")
        if not 0 <= slot < self.n_frames:
            raise ValueError(f"slot {slot} is outside a clip of {self.n_frames} photos")
        rgb = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"))
        now = time.monotonic()
        with self._lock:
            self._rgb, self._taken, self._received = rgb, taken, now
            self._arrived = time.time()
            self._recent.append(now)
            self.photos_received += 1
            self.pending = pending
            building = self._building
            if building is not None and clip < building[0]:
                return  # a clip already handed over or dropped
            if building is None or clip > building[0]:
                if building is not None:  # the relay has moved on: this one stays incomplete
                    self.log(f"[clip {building[0]:03d}] dropped: {len(building[1])} of "
                             f"{self.n_frames} photos arrived from the phone")
                building = (clip, {})
                self._building = building
            building[1][slot] = rgb  # a photo sent twice (its answer was lost) just replaces itself
            if len(building[1]) < self.n_frames:
                return
            self._building = None
        t0 = clip * self.clip_sec
        self.clips.put(Clip(index=clip, t0=t0, t1=t0 + self.clip_sec,
                            times=[t0 + off for off in frame_offsets(self.clip_sec, self.n_frames)],
                            frames=[building[1][s] for s in range(self.n_frames)]))

    def next_clip(self, timeout: float) -> Clip | None:
        """The next whole clip, or None if none arrives within `timeout` seconds."""
        try:
            return self.clips.get(timeout=timeout)
        except queue.Empty:
            return None

    # --- what the page asks of a live source -------------------------------------

    def latest_rgb(self) -> np.ndarray:
        """The newest photo; raises if none has come or it is older than stale_sec."""
        with self._lock:
            rgb, received = self._rgb, self._received
        if rgb is None:
            raise TimeoutError("no photo from the phone relay yet: are the glasses plugged "
                               "into the phone and the relay running in Termux?")
        age = time.monotonic() - received
        if age > self.stale_sec:
            raise StaleStream(f"no photo from the phone for {age:.0f} s")
        return rgb

    def latest_jpeg(self, width: int = 640, quality: int = 75) -> bytes:
        """The newest photo as a small JPEG, made once per photo for the page's live view."""
        rgb = self.latest_rgb()
        with self._lock:
            number = self.photos_received
            if self._jpeg is not None and self._jpeg[:2] == (number, width):
                return self._jpeg[2]
        height = round(rgb.shape[0] * width / rgb.shape[1] / 2) * 2
        out = io.BytesIO()
        Image.fromarray(rgb).resize((width, height)).save(out, "JPEG", quality=quality)
        with self._lock:
            self._jpeg = (number, width, out.getvalue())
        return out.getvalue()

    @property
    def fps(self) -> float:
        """Photos per second over the last few (about 0.53 for 8 per 15 s)."""
        with self._lock:
            recent = list(self._recent)
        return (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) > 1 else 0.0

    def lag(self) -> float | None:
        """Seconds from the newest photo being taken on the phone to its arrival here.

        Both clocks follow the network time; on 2026-10-27 they were 2 ms apart.
        """
        with self._lock:
            if self._rgb is None:
                return None
            return self._arrived - self._taken

    def video_age(self) -> float | None:
        """Seconds since the last photo arrived; None before the first one."""
        with self._lock:
            if self._rgb is None:
                return None
            return time.monotonic() - self._received

    def close(self) -> None:
        """Tell the relay to stop sending; the phone may already be out of reach."""
        if self.relay is None:
            return
        try:
            self.relay.diary_stop()
        except ConnectionError as exc:
            self.error = exc


def open_phone_source(address: str, clip_sec: float, n_frames: int, app_port: int,
                      relay_port: int = RELAY_PORT,
                      log: Callable[[str], None] = print) -> PhoneSource:
    """Ask the relay on the phone at `address` to send this session's photos to us.

    The relay sends them back to whichever address this request came from,
    on `app_port`, so nothing here needs to know which of this machine's
    addresses the phone can reach.
    """
    netloc = f"[{address}]" if ":" in address else address
    relay = RelayClient(f"http://{netloc}:{relay_port}")
    source = PhoneSource(relay, clip_sec, n_frames, log=log)
    relay.diary_start(app_port, source.token, clip_sec, n_frames)
    return source
