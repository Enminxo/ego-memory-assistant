"""Live session: the glasses' scene camera, described as it arrives.

    Neon ─▶ NeonLiveSource ─▶ sampler thread ─▶ queue ─▶ describer ─▶ diary
            (latest frame)    (wall clock)               (main thread)

The sampler owns the clock: every clip_sec seconds it has taken n_frames at
the same instants egomem-build uses on a file (clips.frame_offsets), and it
hands the clip over without waiting for the describer. If the describer falls
more than `queue_size` clips behind, the new clip is dropped and reported: the
diary must not silently stretch the timeline. Frames are only kept as the
evidence the app shows, as in egomem-build.

The session is a scenario without a video (`"live": true`), so egomem-app,
running at the same time, shows its diary growing and answers from it.

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
from egomem.scenarios import Scenario


class LiveSource(Protocol):
    """Anything that returns the camera's most recent RGB frame."""

    def latest_rgb(self) -> np.ndarray: ...


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
                clip = self.sample_clip(index)
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


def run_live(config: dict, source: NeonLiveSource, scene_id: str, title: str,
             address: str | None, minutes: float | None, record: bool) -> LiveSession:
    """Create the live scenario and its diary, then describe until stopped."""
    out = memory_dir(config, scene_id, "default")
    if (out / "memory.jsonl").exists():
        raise SystemExit(f"{scene_id} already has a diary; pick another --id")
    started = datetime.now()
    scenario = Scenario(id=scene_id, title=title, category="other", live=True,
                        source={"neon": address or "discovered",
                                "started": started.isoformat(timespec="seconds")})
    scenario.save(Path(config["scenarios_dir"]) / f"{scene_id}.json")

    d = config["describer"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps({"describer": d, "base_url": config["base_url"],
                                              "source": "neon-live"}, indent=1))
    memory = TextualMemory.load(out / "memory.jsonl")
    evidence = EvidenceStore(out / "frames") if d.get("keep_frames") else None
    session = LiveSession(source, make_describer(config), memory, evidence,
                          d["clip_sec"], d["n_frames"])
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
        print(f"{scene_id}: {session.summary()}")
    return session


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
