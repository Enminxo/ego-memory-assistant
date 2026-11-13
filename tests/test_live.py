import time

import numpy as np
import pytest

from egomem.clips import frame_offsets
from egomem.live import CompanionRecorder, LiveRunner, LiveSession, RecordingError, run_live
from egomem.llm import ChatResult
from egomem.memory import EvidenceStore, TextualMemory
from egomem.neon_stream import StaleStream
from egomem.scenarios import load_scenario


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


def test_run_live_writes_a_live_scenario_and_marks_its_end(tmp_path, monkeypatch):
    monkeypatch.setattr("egomem.live.make_describer", lambda config: FakeDescriber())
    config = {"scenarios_dir": str(tmp_path / "scenarios"), "data_dir": str(tmp_path / "data"),
              "base_url": "http://unused",
              "describer": {"model": "fake", "clip_sec": 0.1, "n_frames": 2, "keep_frames": True}}
    run_live(config, StaticSource(), "live-t", "Live", None, minutes=0.3 / 60, record=False)

    scenario = load_scenario(tmp_path / "scenarios/live-t.json")
    assert scenario.live and scenario.video is None and "ended" in scenario.source
    out = tmp_path / "data/memories/live-t/default"
    assert len(TextualMemory.load(out / "memory.jsonl")) == 3
    assert EvidenceStore(out / "frames").count(0) == 2
    with pytest.raises(SystemExit):  # an existing diary is never appended to by a new session
        run_live(config, StaticSource(), "live-t", "Live", None, minutes=0.3 / 60, record=False)
    assert scenario.demo_questions == []  # no live block in the config: no buttons


def test_a_new_live_session_gets_the_prepared_questions_of_the_config(tmp_path, monkeypatch):
    monkeypatch.setattr("egomem.live.make_describer", lambda config: FakeDescriber())
    questions = [{"text": "¿Dónde dejé las llaves?"},
                 {"text": "¿Cerré la caja?", "hard": True, "label": "difícil"}]
    config = {"scenarios_dir": str(tmp_path / "scenarios"), "data_dir": str(tmp_path / "data"),
              "base_url": "http://unused", "live": {"demo_questions": questions},
              "describer": {"model": "fake", "clip_sec": 0.1, "n_frames": 2, "keep_frames": True}}
    run_live(config, StaticSource(), "live-q", "Stand", None, minutes=0.2 / 60, record=False)

    assert load_scenario(tmp_path / "scenarios/live-q.json").demo_questions == questions


def test_clips_without_recent_video_are_dropped_not_described():
    class Interrupted:
        """The video stops arriving at the first frame of clip 1, then comes back."""

        def __init__(self) -> None:
            self.calls = 0

        def latest_rgb(self) -> np.ndarray:
            self.calls += 1
            if self.calls == 3:
                raise StaleStream("no scene frame for 6 s")
            return np.zeros((2, 2, 3), dtype=np.uint8)

    logs: list[str] = []
    session = LiveSession(Interrupted(), FakeDescriber(), TextualMemory(), None,
                          clip_sec=0.05, n_frames=2, log=logs.append)
    entries = session.run(n_clips=3)
    assert [e.index for e in entries] == [0, 2] and session.dropped == [1]
    assert any("dropped: no scene frame" in m for m in logs)


class FakeStream(StaticSource):
    fps, reconnects, error = 30.0, 0, None

    def __init__(self) -> None:
        self.closed = False

    def latest_jpeg(self, width: int = 640, quality: int = 75) -> bytes:
        return b"\xff\xd8 a jpeg"

    def lag(self) -> float | None:
        return 0.1

    def video_age(self) -> float | None:
        return 0.0

    def close(self) -> None:
        self.closed = True


def live_config() -> dict:
    return {"scenarios_dir": "scenarios", "data_dir": "data", "base_url": "http://unused",
            "describer": {"model": "fake", "clip_sec": 0.1, "n_frames": 2, "keep_frames": True}}


def wait_for(condition, seconds: float = 5.0) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.02)


def test_runner_starts_and_stops_a_session_from_the_page(tmp_path):
    sources: list[FakeStream] = []

    def connect(address: str) -> FakeStream:
        sources.append(FakeStream())
        return sources[-1]

    runner = LiveRunner(live_config(), "default", tmp_path, connect, FakeDescriber, log=quiet)
    runner.start("100.1.2.3", "Kitchen")
    wait_for(lambda: runner.status().get("clips_written", 0) >= 2)
    assert runner.status()["state"] == "running" and runner.status()["fps"] == 30.0
    with pytest.raises(RuntimeError):  # one session at a time
        runner.start("100.1.2.3")
    runner.stop()
    runner.wait(5)

    status = runner.status()
    assert status["state"] == "ended" and status["error"] is None
    scenario = load_scenario(tmp_path / "scenarios" / f"{status['id']}.json")
    assert scenario.live and scenario.title == "Kitchen" and "ended" in scenario.source
    assert scenario.source["neon"] == "100.1.2.3"
    out = tmp_path / "data/memories" / status["id"] / "default"
    assert len(TextualMemory.load(out / "memory.jsonl")) == status["clips_written"]
    assert sources[0].closed

    runner.start("100.1.2.3")  # a new session gets a new id, never the old diary
    wait_for(lambda: runner.state == "running")
    assert runner.id != status["id"]
    runner.stop()
    runner.wait(5)


def test_runner_reports_a_phone_that_does_not_answer_and_opens_no_tab(tmp_path):
    def unreachable(address: str) -> FakeStream:
        raise ConnectionError("no answer from 10.0.0.9")

    runner = LiveRunner(live_config(), "default", tmp_path, unreachable, FakeDescriber, log=quiet)
    runner.start("10.0.0.9")
    runner.wait(5)
    assert runner.status()["state"] == "error"
    assert "no answer" in runner.status()["error"]
    assert not list(tmp_path.glob("scenarios/*.json"))
    with pytest.raises(ValueError):
        runner.start("")


def test_a_stuck_describer_does_not_leave_the_session_open_forever(tmp_path):
    import threading

    release = threading.Event()

    class Stuck(FakeDescriber):
        def describe(self, frames, t0, t1):
            release.wait(10)  # a model server that does not answer
            return super().describe(frames, t0, t1)

    runner = LiveRunner(live_config(), "default", tmp_path, lambda a: FakeStream(), Stuck, log=quiet)
    runner.start("100.1.2.3")
    wait_for(lambda: runner.state == "running")
    runner.stop()
    runner.wait(0.3)  # still stuck in the describer
    runner.mark_ended()  # what the app does on shutdown
    assert "ended" in load_scenario(tmp_path / "scenarios" / f"{runner.id}.json").source
    release.set()
    runner.wait(5)
    assert runner.state == "ended"


def test_stop_while_waiting_for_the_first_picture_opens_no_tab(tmp_path):
    class NoVideoYet(FakeStream):
        def video_age(self) -> float | None:
            return None

    runner = LiveRunner(live_config(), "default", tmp_path, lambda a: NoVideoYet(), FakeDescriber,
                        log=quiet)
    runner.start("100.1.2.3")
    wait_for(lambda: runner.source is not None)
    runner.stop()
    runner.wait(5)
    assert runner.state == "ended" and runner.source.closed
    assert not list(tmp_path.glob("scenarios/*.json"))


class FakeRecorder:
    """The Companion app's recording, as the runner sees it."""

    def __init__(self, refuse: str | None = None) -> None:
        self.refuse = refuse
        self.calls: list[str] = []

    def start(self) -> str:
        self.calls.append("start")
        if self.refuse:
            raise RecordingError(self.refuse)
        return "rec-1"

    def stop_and_save(self) -> None:
        self.calls.append("stop_and_save")


def test_the_phone_records_each_session_from_go_live_to_stop(tmp_path):
    recorders: list[FakeRecorder] = []

    def open_recorder(address: str) -> FakeRecorder:
        assert address == "100.1.2.3"  # the phone the session reads from
        recorders.append(FakeRecorder())
        return recorders[-1]

    runner = LiveRunner(live_config(), "default", tmp_path, lambda a: FakeStream(), FakeDescriber,
                        log=quiet, open_recorder=open_recorder)
    runner.start("100.1.2.3", "Visitor 1")
    wait_for(lambda: runner.status().get("clips_written", 0) >= 1)
    assert runner.status()["recording"] == {"state": "recording", "id": "rec-1", "error": None}
    runner.stop()
    assert recorders[0].calls == ["start", "stop_and_save"]  # saved at Stop, not later
    runner.wait(5)
    assert recorders[0].calls == ["start", "stop_and_save"]  # and only once

    scenario = load_scenario(tmp_path / "scenarios" / f"{runner.id}.json")
    assert scenario.source["phone_recording"] == {"state": "saved", "id": "rec-1", "error": None}

    runner.start("100.1.2.3", "Visitor 2")  # the next session gets its own recording
    wait_for(lambda: runner.state == "running")
    runner.stop()
    runner.wait(5)
    assert len(recorders) == 2 and recorders[1].calls == ["start", "stop_and_save"]


def test_a_phone_that_refuses_to_record_does_not_stop_the_diary(tmp_path):
    recorder = FakeRecorder(refuse="Recording already running")
    runner = LiveRunner(live_config(), "default", tmp_path, lambda a: FakeStream(), FakeDescriber,
                        log=quiet, open_recorder=lambda a: recorder)
    runner.start("100.1.2.3")
    wait_for(lambda: runner.status().get("clips_written", 0) >= 2)
    status = runner.status()
    assert status["state"] == "running" and status["error"] is None
    assert status["recording"]["state"] == "failed"
    assert status["recording"]["error"] == "Recording already running"
    runner.stop()
    runner.wait(5)
    assert recorder.calls == ["start"]  # nothing to stop
    assert runner.state == "ended"


def test_a_phone_that_answers_nonsense_does_not_stop_the_diary(tmp_path):
    class Garbled(FakeRecorder):
        def start(self) -> str:
            raise KeyError("result")  # an answer without the recording's id

    runner = LiveRunner(live_config(), "default", tmp_path, lambda a: FakeStream(), FakeDescriber,
                        log=quiet, open_recorder=lambda a: Garbled())
    runner.start("100.1.2.3")
    wait_for(lambda: runner.status().get("clips_written", 0) >= 1)
    assert runner.status()["recording"]["state"] == "failed"
    runner.stop()
    runner.wait(5)
    assert runner.state == "ended" and runner.error is None


def test_a_session_without_a_recorder_records_nothing(tmp_path):
    runner = LiveRunner(live_config(), "default", tmp_path, lambda a: FakeStream(), FakeDescriber,
                        log=quiet)
    runner.start("100.1.2.3")
    wait_for(lambda: runner.state == "running")
    assert "recording" not in runner.status()
    runner.stop()
    runner.wait(5)
    assert "phone_recording" not in load_scenario(
        tmp_path / "scenarios" / f"{runner.id}.json").source


def test_the_companion_recorder_speaks_the_companion_api():
    """Against a stand-in of the Companion app: same paths, same JSON as pupil_labs.realtime_api."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    seen: list[str] = []
    recording = {"on": False}

    class Companion(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            seen.append(self.path)
            if self.path == "/api/recording:start" and not recording["on"]:
                recording["on"] = True
                code, body = 200, {"message": "Recording started", "result": {"id": "abc-123"}}
            elif self.path == "/api/recording:start":
                code, body = 400, {"message": "Recording already running"}
            elif self.path == "/api/recording:stop_and_save" and recording["on"]:
                recording["on"] = False
                code, body = 200, {"message": "Recording stopped and saved", "result": {}}
            else:
                code, body = 400, {"message": "Recording not running"}
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Companion)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        recorder = CompanionRecorder("127.0.0.1", server.server_address[1])
        assert recorder.start() == "abc-123"
        with pytest.raises(RecordingError, match="Recording already running"):
            recorder.start()
        recorder.stop_and_save()
        with pytest.raises(RecordingError, match="Recording not running"):
            recorder.stop_and_save()
        assert seen == ["/api/recording:start"] * 2 + ["/api/recording:stop_and_save"] * 2
    finally:
        server.shutdown()
        server.server_close()

    closed_port = server.server_address[1]  # nobody listens there any more
    with pytest.raises(RecordingError, match="no answer from the Companion app"):
        CompanionRecorder("127.0.0.1", closed_port, timeout=1.0).start()


def test_a_name_recognized_during_a_clip_goes_into_that_clips_entry(tmp_path):
    runner = LiveRunner(live_config(), "default", tmp_path, lambda a: FakeStream(), FakeDescriber,
                        log=quiet)
    assert runner.note_sighting("Enmin", time.time()) is False  # no session on: not kept
    runner.start("100.1.2.3")
    wait_for(lambda: runner.state == "running" and runner.status().get("clips_written", 0) >= 1)
    # A few clips ahead: with the real describer a sighting is always in before its clip's
    # entry is written (LiveSession.names_in); the fake one here answers at once.
    seen = time.time() + 0.25
    assert runner.note_sighting("Enmin", seen) is True
    assert runner.note_sighting("Enmin", seen + 0.01) is True  # seen again: named once
    wait_for(lambda: runner.status().get("clips_written", 0) >= 8)
    runner.stop()
    runner.wait(5)

    out = tmp_path / "data/memories" / runner.id / "default"
    entries = TextualMemory.load(out / "memory.jsonl").entries()
    named = [e for e in entries if e.people]
    assert len(named) == 1 and named[0].people == ("Enmin",)  # only the clip the sighting fell in
    assert runner.people_between(seen - 1, seen + 1) == ("Enmin",)
    assert runner.people_between(seen + 1, seen + 2) == ()

    runner.start("100.1.2.3")  # a new session starts with nobody seen
    wait_for(lambda: runner.state == "running")
    assert runner.people_between(0, time.time() + 1) == ()
    runner.stop()
    runner.wait(5)
