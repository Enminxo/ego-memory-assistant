"""Pupil Labs Neon glasses as the camera of the assistant.

Neon is a sensor, not a computer: the glasses are wired to a companion phone,
which computes gaze, records, and streams over the local network. The
assistant runs on another machine, and there are two ways in:

- Offline: a recording in Neon's native format, copied from the phone, becomes
  a scenario (`egomem-neon import`). egomem-build and egomem-app then work on
  it unchanged.
- Live: `NeonLiveSource` reads the scene camera through the Real-time API, for
  egomem-live (live.py).

Only the scene camera is used for now. Gaze, IMU and audio come in the same
recording and stream; the scenario keeps the scene camera's start time so they
can be aligned with the video later.

The Pupil Labs packages are optional (`uv sync --extra neon`) and imported
inside the functions that need them, so the rest of the project and its tests
run without them.

    egomem-neon import ~/data/neon/2026-10-23_10-00-00 --id kitchen --title "Kitchen"
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

from egomem.config import load_config
from egomem.scenarios import CATEGORIES, Scenario
from egomem.sources import VideoFileSource

DEFAULT_PORT = 8080


# --- Offline: a native recording becomes a scenario ----------------------------

def scene_resample_indices(ts_ns: np.ndarray, fps: float) -> np.ndarray:
    """For each tick of a constant-rate video, the scene frame shown at it.

    Neon timestamps every scene frame on its own clock (the `.time` file next
    to the video), and its reader does not trust the container's timing, so
    the video is rewritten at a constant rate: tick k is at ts_ns[0] + k / fps
    and shows the last frame taken at or before it. A gap in the recording
    becomes a frozen picture of the same length, instead of the rest of the
    video sliding earlier and the diary's times drifting from the real ones.
    """
    ts = np.asarray(ts_ns, dtype=np.int64)
    if ts.size == 0:
        raise ValueError("the recording has no scene frames")
    if np.any(np.diff(ts) < 0):
        raise ValueError("scene timestamps are not sorted")
    n_ticks = int((ts[-1] - ts[0]) * fps / 1e9) + 1
    ticks = ts[0] + np.round(np.arange(n_ticks) * 1e9 / fps).astype(np.int64)
    return np.searchsorted(ts, ticks, side="right") - 1


def import_recording(rec_dir: Path, scene_id: str, title: str, scenarios_dir: Path,
                     root: Path = Path("."), fps: float = 15.0, width: int = 1280,
                     category: str = "other") -> Scenario:
    """Write the scene camera of a recording as data/scenes/<id>.mp4, and its scenario.

    The scenario starts with no questions: the person who recorded it writes
    them, ideally before looking at the diary.
    """
    import pupil_labs.neon_recording as nr

    video_rel = Path("data/scenes") / f"{scene_id}.mp4"
    dst = root / video_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    with nr.open(rec_dir) as rec:
        scene = rec.scene
        ts = np.asarray(scene.time, dtype=np.int64)
        indices = scene_resample_indices(ts, fps)
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{scene.width}x{scene.height}", "-r", str(fps), "-i", "-",
               "-vf", f"scale={width}:-2", "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        last_i, last = -1, b""
        try:
            for i in indices:
                if i != last_i:  # a repeated index is a gap: send the same picture again
                    last = scene[int(i)].rgb.tobytes()
                    last_i = i
                proc.stdin.write(last)
        finally:
            proc.stdin.close()
            if proc.wait() != 0:
                raise RuntimeError(f"ffmpeg failed writing {dst}")
        recording_id = rec.id
    scenario = Scenario(id=scene_id, title=title, category=category, video=str(video_rel),
                        source={"neon_recording_id": recording_id,
                                "scene_start_unix_ns": int(ts[0]), "fps": fps})
    scenario.save(Path(scenarios_dir) / f"{scene_id}.json")
    return scenario


# --- Live: the latest frame of the scene camera --------------------------------

class NeonLiveSource:
    """The latest frame of a Neon's scene camera, read continuously in a thread.

    The Real-time API hands out frames in order, one per call, so a consumer
    that only wants a frame every couple of seconds would fall further and
    further behind the stream. A thread drains it instead and keeps the most
    recent frame, as egoagent's FrameHub does with the Pi camera.

    `address` None searches the local network (mDNS), which only finds the
    phone when both are on the same network segment; otherwise pass the IP
    shown in the Companion app's streaming screen.
    """

    def __init__(self, address: str | None = None, port: int = DEFAULT_PORT,
                 search_sec: float = 10.0) -> None:
        from pupil_labs.realtime_api.simple import Device, discover_one_device

        if address is None:
            device = discover_one_device(max_search_duration_seconds=search_sec)
            if device is None:
                raise ConnectionError("no Neon found on the local network; "
                                      "pass the phone's IP with --neon")
        else:
            device = Device(address, port)
        self.device = device
        self.frames_read = 0
        self._bgr: np.ndarray | None = None
        self._timestamp = 0.0
        self._lock = threading.Lock()
        self._first = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="neon-scene", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            frame = self.device.receive_scene_video_frame(timeout_seconds=1.0)
            if frame is None:
                continue  # nothing within the timeout: check for stop, then wait again
            with self._lock:
                self._bgr, self._timestamp = frame.bgr_pixels, frame.timestamp_unix_seconds
                self.frames_read += 1
            self._first.set()

    def latest(self, timeout: float = 10.0) -> tuple[np.ndarray, float]:
        """Most recent frame as RGB, and when it was taken (Unix seconds, phone clock)."""
        if not self._first.wait(timeout):
            raise TimeoutError(f"no scene frame from the Neon in {timeout:.0f} s")
        with self._lock:
            bgr, timestamp = self._bgr, self._timestamp
        # The API decodes to BGR; the describer and the evidence store want RGB.
        # Converted here, on the few frames used, not on all 30 per second.
        return np.ascontiguousarray(bgr[:, :, ::-1]), timestamp

    def latest_rgb(self) -> np.ndarray:
        return self.latest()[0]

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)
        self.device.close()


def probe(source: NeonLiveSource, seconds: float = 5.0,
          frame_path: Path | None = None) -> dict:
    """Check the connection: which phone, how many frames per second, how late.

    The lag is measured, not assumed: the phone's clock is related to this
    machine's with Pupil Labs' time echo (offset = client - phone), and each
    frame's capture time is compared with the moment it is seen here.
    """
    device = source.device
    frame, _ = source.latest()  # also waits for the stream to start
    estimate = device.estimate_time_offset()
    offset_s = estimate.time_offset_ms.mean / 1000 if estimate is not None else None
    n0, t0 = source.frames_read, time.monotonic()
    lags = []
    while time.monotonic() - t0 < seconds:
        frame, taken = source.latest()
        if offset_s is not None:
            lags.append(time.time() - (taken + offset_s))
        time.sleep(0.2)
    fps = (source.frames_read - n0) / (time.monotonic() - t0)
    if frame_path is not None:
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frame).save(frame_path, quality=90)
    return {"phone": device.phone_name, "battery_percent": device.battery_level_percent,
            "frame": f"{frame.shape[1]}x{frame.shape[0]}", "fps_received": round(fps, 1),
            "clock_offset_ms": round(offset_s * 1000, 1) if offset_s is not None else None,
            "lag_median_s": round(statistics.median(lags), 3) if lags else None,
            "lag_max_s": round(max(lags), 3) if lags else None,
            "frame_saved": str(frame_path) if frame_path is not None else None}


def main() -> None:
    parser = argparse.ArgumentParser(description="Use recordings from Pupil Labs Neon glasses.")
    sub = parser.add_subparsers(dest="command", required=True)
    imp = sub.add_parser("import", help="turn a native Neon recording into a scenario")
    imp.add_argument("recording", type=Path, help="recording folder copied from the phone")
    imp.add_argument("--id", required=True, help="scenario id, e.g. kitchen")
    imp.add_argument("--title", help="shown in the app (default: the id)")
    imp.add_argument("--category", default="other", choices=CATEGORIES)
    imp.add_argument("--fps", type=float, default=15.0, help="frame rate of the written video")
    imp.add_argument("--width", type=int, default=1280, help="width of the written video")
    imp.add_argument("--config", type=Path, default=Path("configs/demo.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    scenarios_dir = Path(config["scenarios_dir"])
    if (scenarios_dir / f"{args.id}.json").exists():
        raise SystemExit(f"scenario {args.id} already exists; pick another --id")
    scenario = import_recording(args.recording, args.id, args.title or args.id, scenarios_dir,
                                fps=args.fps, width=args.width, category=args.category)
    duration = VideoFileSource(Path(scenario.video)).duration
    print(f"{scenario.id}: {scenario.video}, {duration:.0f} s of video. "
          f"Next: add questions to {scenarios_dir / f'{scenario.id}.json'}, "
          f"then egomem-build {scenario.id}")


if __name__ == "__main__":
    main()
