import time

import numpy as np
import pytest

from egomem.clips import frame_offsets
from egomem.live import LiveSession
from egomem.llm import ChatResult
from egomem.memory import TextualMemory


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, dt: float) -> None:
        self.now += dt


class ClockSource:
    """The latest frame is a flat image of value int(seconds into the session)."""

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock

    def latest_rgb(self) -> np.ndarray:
        return np.full((2, 2, 3), int(self.clock() - 100.0), dtype=np.uint8)


class StaticSource:
    def latest_rgb(self) -> np.ndarray:
        return np.zeros((2, 2, 3), dtype=np.uint8)


class FakeDescriber:
    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay

    def describe(self, frames, t0, t1) -> ChatResult:
        time.sleep(self.delay)
        return ChatResult(text=f"Clip at {t0:.2f} s.", latency_s=self.delay,
                          prompt_tokens=None, completion_tokens=None)


def quiet(message: str) -> None:
    pass


def test_frames_are_taken_at_the_same_instants_as_offline():
    clock = FakeClock()
    session = LiveSession(ClockSource(clock), FakeDescriber(), TextualMemory(), None,
                          clip_sec=15, n_frames=8, clock=clock, sleep=clock.sleep)
    session._start = clock()
    clip = session.sample_clip(1)
    assert (clip.index, clip.t0, clip.t1) == (1, 15, 30)
    assert clip.times == [15 + off for off in frame_offsets(15, 8)]
    # each frame is the one the camera showed at its instant, not earlier or later
    assert [int(f[0, 0, 0]) for f in clip.frames] == [int(t) for t in clip.times]


def test_every_clip_is_described_in_order(tmp_path):
    memory = TextualMemory(tmp_path / "memory.jsonl")
    session = LiveSession(StaticSource(), FakeDescriber(), memory, None,
                          clip_sec=0.1, n_frames=2, log=quiet)
    entries = session.run(n_clips=3)
    assert [(e.index, e.t0, e.t1, e.n_frames) for e in entries] == [
        (0, 0.0, 0.1, 2), (1, 0.1, 0.2, 2), (2, 0.2, 0.3, 2)]
    assert session.dropped == []
    assert len(TextualMemory.load(tmp_path / "memory.jsonl")) == 3  # written as it goes


def test_a_slow_describer_drops_clips_and_says_so():
    logs: list[str] = []
    session = LiveSession(StaticSource(), FakeDescriber(delay=0.3), TextualMemory(), None,
                          clip_sec=0.05, n_frames=1, queue_size=1, log=logs.append)
    entries = session.run(n_clips=8)
    assert session.dropped  # 0.3 s per 0.05 s clip cannot keep up
    assert sorted([e.index for e in entries] + session.dropped) == list(range(8))
    assert any("dropped" in m for m in logs)


def test_a_failing_camera_ends_the_session_with_its_error():
    class Broken:
        def latest_rgb(self) -> np.ndarray:
            raise TimeoutError("no scene frame")

    session = LiveSession(Broken(), FakeDescriber(), TextualMemory(), None,
                          clip_sec=0.05, n_frames=1, log=quiet)
    with pytest.raises(TimeoutError):
        session.run(n_clips=3)
