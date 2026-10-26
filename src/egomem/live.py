"""Live session: the glasses' scene camera, described as it arrives.

    Neon ─▶ NeonLiveSource ─▶ sampler thread ─▶ queue ─▶ describer ─▶ diary
            (latest frame)    (wall clock)               (main thread)

The sampler owns the clock: every clip_sec seconds it has taken n_frames at
the same instants egomem-build uses on a file (clips.frame_offsets), and it
hands the clip over without waiting for the describer. If the describer falls
more than `queue_size` clips behind, the new clip is dropped and reported: the
diary must not silently stretch the timeline. A clip is also dropped when the
video has stopped arriving, rather than describing the last picture again as
if it were the present. Frames are only kept as the evidence the app shows,
as in egomem-build.

The session is a scenario without a video (`"live": true`), so egomem-app
shows its diary growing and answers from it. It is started either from the
app's "Go live" button (LiveRunner) or from the terminal:

    egomem-live --neon 192.168.43.1 --probe     # check the connection only
    egomem-live --neon 192.168.43.1             # phone IP from the Companion app
    egomem-live --minutes 10 --record           # search the network, also record on the phone
"""

from __future__ import annotations

import argparse
import json
import queue
import statistics
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Protocol

import numpy as np

from egomem.build_memory import describe_clip
from egomem.clips import Clip, frame_offsets
from egomem.config import load_config, make_describer, memory_dir
from egomem.describer import Describer
from egomem.memory import EvidenceStore, MemoryEntry, TextualMemory, fmt_time
from egomem.neon import DEFAULT_PORT, NeonLiveSource, probe
from egomem.neon_stream import StaleStream
from egomem.scenarios import Scenario


class LiveSource(Protocol):
    """Anything that returns the camera's most recent RGB frame.

    latest_rgb() raises StaleStream when the camera has sent nothing recently.
    """

    def latest_rgb(self) -> np.ndarray: ...


class StreamSource(LiveSource, Protocol):
    """A live source the app can show and report on (neon_stream.SceneStream)."""

    fps: float
    reconnects: int
    error: BaseException | None

    def latest_jpeg(self, width: int = 640, quality: int = 75) -> bytes: ...

    def lag(self) -> float | None: ...

    def video_age(self) -> float | None: ...

    def close(self) -> None: ...


class LiveSession:
    """Sample the source on the wall clock and describe each clip into the diary."""

    def __init__(self, source: LiveSource, describer: Describer, memory: TextualMemory,
                 evidence: EvidenceStore | None, clip_sec: float, n_frames: int,
                 queue_size: int = 2, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], object] | None = None,
                 log: Callable[[str], None] = print) -> None:
        self.source = source
        self.describer = describer
        self.memory = memory
        self.evidence = evidence
        self.clip_sec = clip_sec
        self.n_frames = n_frames
        self.clock = clock
        self.stop_event = threading.Event()
        # Waiting on the stop event rather than time.sleep, so stop() is immediate.
        self.sleep = sleep or self.stop_event.wait
        self.log = log
        self.clips: queue.Queue[Clip | None] = queue.Queue(maxsize=queue_size)
        self.dropped: list[int] = []
        self.written: list[MemoryEntry] = []
        self.error: BaseException | None = None
        self._start = 0.0

    def sample_clip(self, index: int) -> Clip:
        """The frames of clip `index`, each taken at its instant on the session clock."""
        t0 = index * self.clip_sec
        times = [t0 + off for off in frame_offsets(self.clip_sec, self.n_frames)]
        frames = []
        for t in times:
            wait = self._start + t - self.clock()
            if wait > 0:
                self.sleep(wait)
            frames.append(self.source.latest_rgb())
        return Clip(index=index, t0=t0, t1=t0 + self.clip_sec, times=times, frames=frames)

    def _sample(self, n_clips: int | None) -> None:
        index = 0
        try:
            while not self.stop_event.is_set() and (n_clips is None or index < n_clips):
                try:
                    clip = self.sample_clip(index)
                except StaleStream as exc:
                    self.dropped.append(index)
                    self.log(f"[clip {index:03d}] dropped: {exc}")
                    index += 1
                    continue
                if self.stop_event.is_set():
                    break  # stopped halfway through: the clip is incomplete
                try:
                    self.clips.put_nowait(clip)
                except queue.Full:
                    self.dropped.append(index)
                    self.log(f"[clip {index:03d}] dropped: the describer is "
                             f"{self.clips.qsize()} clips behind")
                index += 1
        except BaseException as exc:  # reported by run(), not lost in this thread
            self.error = exc
        finally:
            self.clips.put(None)  # end of stream for the describer

    def run(self, n_clips: int | None = None) -> list[MemoryEntry]:
        """Sample and describe until n_clips are done or stop() is called."""
        self._start = self.clock()
        sampler = threading.Thread(target=self._sample, args=(n_clips,), name="sampler",
                                   daemon=True)
        sampler.start()
        try:
            while (clip := self.clips.get()) is not None:
                entry = describe_clip(self.describer, clip, self.memory, self.evidence)
                self.written.append(entry)
                late = " [longer than the clip]" if entry.latency_s > self.clip_sec else ""
                self.log(f"[clip {entry.index:03d} | {fmt_time(entry.t0)}-{fmt_time(entry.t1)}"
                         f" | {entry.latency_s:.1f}s{late}] {entry.text}")
        finally:
            self.stop_event.set()
            sampler.join(timeout=self.clip_sec)
        if self.error is not None:
            raise self.error
        return self.written

    def stop(self) -> None:
        self.stop_event.set()

    def summary(self) -> str:
        latencies = [e.latency_s for e in self.written]
        if not latencies:
            return f"no clips described, {len(self.dropped)} dropped"
        return (f"{len(latencies)} clips described, {len(self.dropped)} dropped; "
                f"describer median {statistics.median(latencies):.1f} s, "
                f"max {max(latencies):.1f} s per {self.clip_sec:g} s clip")


def open_live_scene(config: dict, scene_id: str, title: str, address: str | None,
                    source: LiveSource, describer: Describer, root: Path = Path("."),
                    tag: str = "default") -> tuple[Scenario, LiveSession]:
    """Write the live scenario and prepare its diary; the session is not started yet."""
    out = root / memory_dir(config, scene_id, tag)
    if (out / "memory.jsonl").exists():
        raise FileExistsError(f"{scene_id} already has a diary; pick another id")
    scenario = Scenario(id=scene_id, title=title, category="other", live=True,
                        source={"neon": address or "discovered",
                                "started": datetime.now().isoformat(timespec="seconds")})
    scenario.save(root / config["scenarios_dir"] / f"{scene_id}.json")

    d = config["describer"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps({"describer": d, "base_url": config["base_url"],
                                              "source": "neon-live"}, indent=1))
    memory = TextualMemory.load(out / "memory.jsonl")
    evidence = EvidenceStore(out / "frames") if d.get("keep_frames") else None
    session = LiveSession(source, describer, memory, evidence, d["clip_sec"], d["n_frames"])
    return scenario, session


def close_live_scene(config: dict, scenario: Scenario, root: Path = Path(".")) -> None:
    """Write the end time, so the app stops showing the session as still being watched."""
    scenario.source["ended"] = datetime.now().isoformat(timespec="seconds")
    scenario.save(root / config["scenarios_dir"] / f"{scenario.id}.json")


def run_live(config: dict, source: NeonLiveSource, scene_id: str, title: str,
             address: str | None, minutes: float | None, record: bool) -> LiveSession:
    """Create the live scenario and its diary, then describe until stopped."""
    try:
        scenario, session = open_live_scene(config, scene_id, title, address, source,
                                            make_describer(config))
    except FileExistsError as exc:
        raise SystemExit(str(exc)) from None
    d = config["describer"]
    n_clips = round(minutes * 60 / d["clip_sec"]) if minutes else None

    if record:
        print(f"recording on the phone: {source.device.recording_start()}")
    print(f"{scene_id}: {d['n_frames']} frames every {d['clip_sec']} s, described by "
          f"{d['model']}. Ctrl+C to stop; ask questions in egomem-app.", flush=True)
    try:
        session.run(n_clips)
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        if record:
            source.device.recording_stop_and_save()
            print("recording saved on the phone")
        close_live_scene(config, scenario)
        print(f"{scene_id}: {session.summary()}")
    return session


class LiveRunner:
    """One live session at a time, started and stopped from the web page.

    Like UploadJobs for uploads: the page asks, and a thread does the slow
    part. It connects to the phone and waits for the first frame before
    writing the scenario, so the tab never opens on a session with no
    picture. The session is the same LiveSession egomem-live runs.

    `open_source(address)` connects to the glasses (NeonLiveSource in the app,
    a fake in the tests) and `make_describer()` gives the describer.
    """

    def __init__(self, config: dict, tag: str, root: Path,
                 open_source: Callable[[str], StreamSource],
                 make_describer: Callable[[], Describer],
                 log: Callable[[str], None] = print, first_frame_sec: float = 30.0) -> None:
        self.config = config
        self.tag = tag
        self.root = Path(root)
        self.open_source = open_source
        self.make_describer = make_describer
        self.log = log
        # Long enough for the stream to reconnect once: after a while idle, the
        # phone's first session sometimes sends no video (seen twice, 2026-10-26).
        self.first_frame_sec = first_frame_sec
        self.state = "idle"  # idle, connecting, running, ended, error
        self.id: str | None = None
        self.title = ""
        self.address = ""
        self.error: str | None = None
        self.source: StreamSource | None = None
        self.session: LiveSession | None = None
        self.scenario: Scenario | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def _new_id(self, now: datetime) -> str:
        base = f"live-{now:%Y%m%d-%H%M}"
        scene_id, n = base, 1
        while ((self.root / self.config["scenarios_dir"] / f"{scene_id}.json").exists()
               or (self.root / memory_dir(self.config, scene_id, self.tag)).exists()):
            n += 1
            scene_id = f"{base}-{n}"
        return scene_id

    def start(self, address: str, title: str = "") -> None:
        if not address:
            raise ValueError("give the phone's address, shown in the Companion app")
        with self._lock:
            if self.state in ("connecting", "running"):
                raise RuntimeError(f"the live session {self.id} is still on; stop it first")
            now = datetime.now()
            self.id = self._new_id(now)
            self.title = title or f"Live {now:%d %b %H:%M}"
            self.address, self.error = address, None
            self.source, self.session, self.scenario = None, None, None
            self.state = "connecting"
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="live", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        try:
            self.source = self.open_source(self.address)
            deadline = time.monotonic() + self.first_frame_sec
            while self.source.video_age() is None and time.monotonic() < deadline:
                if self._stop.wait(0.2):
                    return
            self.source.latest_rgb()  # the first frame, or the reason there is none
            if self._stop.is_set():
                return
            self.scenario, self.session = open_live_scene(
                self.config, self.id, self.title, self.address, self.source,
                self.make_describer(), self.root, self.tag)
            if self._stop.is_set():  # stopped while the scenario was being written
                self.session.stop()
            self.state = "running"
            self.session.log = self.log
            self.session.run()
        except Exception as exc:  # shown on the page
            self.error = str(exc) or repr(exc)
        finally:
            self.mark_ended()
            if self.source is not None:
                self.source.close()
            self.state = "error" if self.error else "ended"
            if self.session is not None:
                self.log(f"{self.id}: {self.session.summary()}")

    def stop(self) -> None:
        self._stop.set()
        if self.session is not None:
            self.session.stop()

    def mark_ended(self) -> None:
        """Write the session's end into its scenario, once.

        Called when the session finishes, and by the app on shutdown in case
        the describer is stuck (a hung model server) and the session never
        returns: the tab must not show it as still being watched forever.
        """
        with self._lock:
            scenario = self.scenario
            if scenario is None or "ended" in scenario.source:
                return
            close_live_scene(self.config, scenario, self.root)

    def wait(self, timeout: float | None = None) -> None:
        """Block until the session thread has finished (for tests and shutdown)."""
        if self._thread is not None:
            self._thread.join(timeout)

    def status(self) -> dict:
        status = {"state": self.state, "id": self.id, "title": self.title,
                  "address": self.address, "error": self.error,
                  "clip_sec": self.config["describer"]["clip_sec"]}
        source, session = self.source, self.session
        if source is not None and self.state in ("connecting", "running"):
            lag, age = source.lag(), source.video_age()
            status.update(fps=round(source.fps, 1), reconnects=source.reconnects,
                          lag_s=round(lag, 2) if lag is not None else None,
                          video_age_s=round(age, 1) if age is not None else None,
                          stream_error=repr(source.error) if source.error else None)
        if session is not None:
            status.update(clips_written=len(session.written), dropped=len(session.dropped))
        return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the diary live from Neon glasses.")
    parser.add_argument("--neon", metavar="IP", help="phone address shown in the Companion "
                        "app; omit to search the local network")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--probe", action="store_true",
                        help="only check the connection: phone, frames per second, lag")
    parser.add_argument("--id", help="scenario id (default live-YYYYMMDD-HHMM)")
    parser.add_argument("--title", help="shown in the app")
    parser.add_argument("--minutes", type=float, help="stop after this long (default: Ctrl+C)")
    parser.add_argument("--record", action="store_true",
                        help="also record on the phone, keeping gaze and audio for later. Off "
                             "by default: the recording may upload to the phone's Pupil Cloud "
                             "workspace")
    parser.add_argument("--config", type=Path, default=Path("configs/demo.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    source = NeonLiveSource(args.neon, args.port)
    try:
        if args.probe:
            for key, value in probe(source, frame_path=Path("logs/neon_probe.jpg")).items():
                print(f"{key}: {value}")
            return
        now = datetime.now()
        run_live(config, source, args.id or f"live-{now:%Y%m%d-%H%M}",
                 args.title or f"Live {now:%d %b %H:%M}", args.neon, args.minutes, args.record)
    finally:
        source.close()


if __name__ == "__main__":
    main()
