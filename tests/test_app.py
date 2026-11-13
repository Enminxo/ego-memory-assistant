import json

from fastapi.testclient import TestClient

from egomem.app import create_app, mjpeg
from egomem.memory import MemoryEntry, TextualMemory
from egomem.reader import Answer
from egomem.scenarios import Scenario


class FakeReader:
    def __init__(self):
        self.seen = None

    def answer(self, entries, question):
        self.seen = entries
        return Answer(text="On the table [00:20].", citations=[20], latency_s=0.1,
                      n_entries=len(entries))


def make_app(tmp_path):
    (tmp_path / "scenarios").mkdir()
    (tmp_path / "data/scenes").mkdir(parents=True)
    (tmp_path / "data/scenes/s.mp4").write_bytes(b"0123456789" * 100)
    (tmp_path / "scenarios/s.json").write_text(json.dumps({
        "id": "s", "title": "Scene", "category": "objects", "video": "data/scenes/s.mp4",
        "questions": [], "demo_questions": [{"text": "Where?", "expected": "table"}]}))
    memory = TextualMemory(tmp_path / "data/memories/s/default/memory.jsonl")
    for i in range(3):
        memory.append(MemoryEntry(i, 15 * i, 15 * (i + 1), f"clip {i}", 1.0, 8))
    config = {"scenarios_dir": "scenarios", "data_dir": "data", "describer": {"clip_sec": 15}}
    reader = FakeReader()
    return TestClient(create_app(config, "default", reader=reader, root=tmp_path)), reader


def test_endpoints(tmp_path):
    client, reader = make_app(tmp_path)
    scenarios = client.get("/api/scenarios").json()
    assert scenarios[0]["id"] == "s" and scenarios[0]["n_entries"] == 3
    assert len(client.get("/api/scenarios/s/memory").json()["entries"]) == 3
    assert client.get("/api/scenarios/nope/memory").status_code == 404

    answer = client.post("/api/ask", json={"scenario": "s", "question": "Where?", "t": 31}).json()
    assert [e.index for e in reader.seen] == [0, 1]  # only clips finished by t=31
    assert answer["citations"][0]["index"] == 1
    client.post("/api/ask", json={"scenario": "s", "question": "Where?"})
    assert len(reader.seen) == 3  # no t: whole diary


def test_video_supports_range_requests(tmp_path):
    client, _ = make_app(tmp_path)
    response = client.get("/api/scenarios/s/video", headers={"Range": "bytes=10-19"})
    assert response.status_code == 206
    assert response.content == b"0123456789"


def test_live_scenario_appears_after_start_and_has_no_video(tmp_path):
    client, reader = make_app(tmp_path)
    assert [s["id"] for s in client.get("/api/scenarios").json()] == ["s"]

    # egomem-live creates its scenario while the app is already running
    Scenario(id="live-1", title="Live", category="other", live=True).save(
        tmp_path / "scenarios/live-1.json")
    TextualMemory(tmp_path / "data/memories/live-1/default/memory.jsonl").append(
        MemoryEntry(0, 0, 15, "I put the keys on the table.", 9.0, 8))

    listed = {s["id"]: s for s in client.get("/api/scenarios").json()}
    assert listed["live-1"]["live"] and listed["live-1"]["n_entries"] == 1
    assert not listed["live-1"]["ended"]
    assert not listed["s"]["live"] and not listed["s"]["ended"]
    assert client.get("/api/scenarios/live-1/video").status_code == 404
    client.post("/api/ask", json={"scenario": "live-1", "question": "Where are my keys?"})
    assert len(reader.seen) == 1


def test_live_scenarios_carry_the_stand_hint_and_recordings_do_not(tmp_path):
    (tmp_path / "scenarios").mkdir()
    Scenario(id="live-1", title="Live", category="other", live=True).save(
        tmp_path / "scenarios/live-1.json")
    Scenario(id="rec", title="Recording", category="other", video="data/scenes/rec.mp4").save(
        tmp_path / "scenarios/rec.json")
    config = {"scenarios_dir": "scenarios", "data_dir": "data", "describer": {"clip_sec": 15},
              "live": {"hint": "Ask what you did."}}
    client = TestClient(create_app(config, "default", reader=FakeReader(), root=tmp_path))

    listed = {s["id"]: s for s in client.get("/api/scenarios").json()}
    assert listed["live-1"]["hint"] == "Ask what you did."
    assert listed["rec"]["hint"] == ""


def make_live_app(tmp_path):
    from test_live import FakeDescriber, FakeStream

    config = {"scenarios_dir": "scenarios", "data_dir": "data", "base_url": "http://unused",
              "describer": {"model": "fake", "clip_sec": 0.1, "n_frames": 2, "keep_frames": True}}
    app = create_app(config, "default", reader=FakeReader(), root=tmp_path,
                     live_source=lambda address: FakeStream(), describer=FakeDescriber)
    return TestClient(app), app.state.live


def test_live_session_is_started_and_stopped_from_the_page(tmp_path):
    from test_live import wait_for

    client, live = make_live_app(tmp_path)
    assert client.get("/api/live").json()["state"] == "idle"
    assert client.post("/api/live/start", json={"address": " "}).status_code == 400

    started = client.post("/api/live/start", json={"address": "100.1.2.3", "title": "Desk",
                                                   "via": "direct"}).json()
    assert started["state"] in ("connecting", "running") and started["via"] == "direct"
    wait_for(lambda: live.status().get("clips_written", 0) >= 1)
    assert client.post("/api/live/start", json={"address": "100.1.2.3"}).status_code == 409
    assert client.post("/api/live/start", json={"address": "100.1.2.3",
                                                "via": "carrier pigeon"}).status_code == 400
    listed = {s["id"]: s for s in client.get("/api/scenarios").json()}
    assert listed[started["id"]]["live"] and not listed[started["id"]]["ended"]

    client.post("/api/live/stop")
    live.wait(5)
    assert client.get("/api/live").json()["state"] == "ended"
    assert client.get("/api/scenarios").json()[0]["ended"]
    assert client.get("/api/live/stream.mjpg").status_code == 404  # nothing to show any more


def test_mjpeg_sends_each_picture_as_a_part_and_skips_stale_ones():
    from test_live import FakeStream

    running = iter([True, True, False]).__next__
    parts = list(mjpeg(FakeStream(), running, fps=1000))
    assert len(parts) == 2 and parts[0].startswith(b"--frame\r\nContent-Type: image/jpeg")

    class Stale(FakeStream):
        def latest_jpeg(self, width=640, quality=75):
            from egomem.neon_stream import StaleStream
            raise StaleStream("no scene frame for 9 s")

    assert list(mjpeg(Stale(), iter([True, True, False]).__next__, fps=1000)) == []


def trashed(client, scenario_id: str) -> tuple[int, dict]:
    response = client.delete(f"/api/scenarios/{scenario_id}")
    return response.status_code, response.json()


def test_an_own_live_session_goes_to_the_trash_with_its_diary(tmp_path):
    client, _ = make_app(tmp_path)
    Scenario(id="live-1", title="A test", category="other", live=True,
             source={"ended": "2026-10-27T19:30:00"}).save(tmp_path / "scenarios/live-1.json")
    memory_file = tmp_path / "data/memories/live-1/default/memory.jsonl"
    TextualMemory(memory_file).append(MemoryEntry(0, 0, 15, "I sat at my desk.", 9.0, 8))
    (memory_file.parent / "frames").mkdir()
    (memory_file.parent / "frames/clip0000_0.jpg").write_bytes(b"jpeg")
    diary = memory_file.read_text()
    listed = {s["id"]: s for s in client.get("/api/scenarios").json()}
    assert listed["live-1"]["own"] and not listed["s"]["own"]

    status, body = trashed(client, "live-1")
    assert status == 200
    trash = tmp_path / body["trashed"]
    assert trash.parent == tmp_path / "data/trash" and trash.name.endswith("_live-1")
    assert [s["id"] for s in client.get("/api/scenarios").json()] == ["s"]
    assert not (tmp_path / "scenarios/live-1.json").exists()
    assert not (tmp_path / "data/memories/live-1").exists()
    # the same relative paths inside the trash: moving them back restores the session
    assert (trash / "scenarios/live-1.json").exists()
    assert (trash / "data/memories/live-1/default/memory.jsonl").read_text() == diary
    assert (trash / "data/memories/live-1/default/frames/clip0000_0.jpg").read_bytes() == b"jpeg"


def test_an_upload_goes_to_the_trash_with_its_video_and_its_files(tmp_path):
    client, _ = make_app(tmp_path)
    Scenario(id="upload-window", title="Window", category="other",
             video="data/scenes/upload-window.mp4").save(tmp_path / "scenarios/upload-window.json")
    (tmp_path / "data/scenes/upload-window.mp4").write_bytes(b"video")
    (tmp_path / "data/uploads/upload-window").mkdir(parents=True)
    (tmp_path / "data/uploads/upload-window/raw.mov").write_bytes(b"raw")

    status, body = trashed(client, "upload-window")
    assert status == 200
    trash = tmp_path / body["trashed"]
    assert (trash / "data/scenes/upload-window.mp4").read_bytes() == b"video"
    assert (trash / "data/uploads/upload-window/raw.mov").read_bytes() == b"raw"
    assert not (tmp_path / "data/scenes/upload-window.mp4").exists()
    assert (tmp_path / "data/scenes/s.mp4").exists()  # the demo scene's video is untouched


def test_demo_scenes_and_unknown_ids_cannot_be_deleted(tmp_path):
    client, _ = make_app(tmp_path)
    status, body = trashed(client, "s")
    assert status == 403 and "demo scene" in body["detail"]
    assert (tmp_path / "scenarios/s.json").exists() and (tmp_path / "data/scenes/s.mp4").exists()
    assert not (tmp_path / "data/trash").exists()
    assert trashed(client, "nope")[0] == 404


def test_the_live_session_still_running_cannot_be_deleted(tmp_path):
    from test_live import wait_for

    client, live = make_live_app(tmp_path)
    started = client.post("/api/live/start", json={"address": "100.1.2.3", "via": "direct"}).json()
    wait_for(lambda: live.state == "running")
    status, body = trashed(client, started["id"])
    assert status == 409 and "still on" in body["detail"]
    client.post("/api/live/stop")
    live.wait(5)
    assert trashed(client, started["id"])[0] == 200
    assert started["id"] not in [s["id"] for s in client.get("/api/scenarios").json()]


def test_live_record_in_the_config_records_each_session_on_the_phone(tmp_path):
    from test_live import FakeDescriber, FakeRecorder, FakeStream, wait_for

    recorders: list[FakeRecorder] = []

    def open_recorder(address: str) -> FakeRecorder:
        recorders.append(FakeRecorder())
        return recorders[-1]

    def app_with(record: bool, root):
        config = {"scenarios_dir": "scenarios", "data_dir": "data", "base_url": "http://unused",
                  "describer": {"model": "fake", "clip_sec": 0.1, "n_frames": 2, "keep_frames": True},
                  "live": {"record": record}}
        app = create_app(config, "default", reader=FakeReader(), root=root,
                         live_source=lambda address: FakeStream(), describer=FakeDescriber,
                         live_recorder=open_recorder)
        return TestClient(app), app.state.live

    client, live = app_with(False, tmp_path / "off")
    client.post("/api/live/start", json={"address": "100.1.2.3", "via": "direct"})
    wait_for(lambda: live.state == "running")
    client.post("/api/live/stop")
    live.wait(5)
    assert recorders == []  # live.record off: the phone records nothing

    client, live = app_with(True, tmp_path / "on")
    client.post("/api/live/start", json={"address": "100.1.2.3", "via": "direct"})
    wait_for(lambda: client.get("/api/live").json().get("recording", {}).get("state") == "recording")
    client.post("/api/live/stop")
    live.wait(5)
    assert recorders[0].calls == ["start", "stop_and_save"]


def test_the_greeting_puts_a_name_into_the_live_diary_through_the_app(tmp_path):
    import time

    from test_live import wait_for

    client, live = make_live_app(tmp_path)
    assert client.post("/api/live/people", json={"name": "Enmin", "seen": time.time()}).json() \
        == {"kept": False}  # no session on
    assert client.post("/api/live/people", json={"name": " ", "seen": 0}).status_code == 400

    started = client.post("/api/live/start", json={"address": "100.1.2.3", "via": "direct"}).json()
    wait_for(lambda: live.status().get("clips_written", 0) >= 1)
    seen = time.time() + 0.25  # a few clips ahead, as in test_live
    assert client.post("/api/live/people", json={"name": "Enmin", "seen": seen}).json() \
        == {"kept": True}
    wait_for(lambda: live.status().get("clips_written", 0) >= 8)
    client.post("/api/live/stop")
    live.wait(5)

    entries = client.get(f"/api/scenarios/{started['id']}/memory").json()["entries"]
    assert [e["people"] for e in entries if e["people"]] == [["Enmin"]]
