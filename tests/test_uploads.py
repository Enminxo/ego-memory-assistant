import io
import json
import shutil
import subprocess
import zipfile

import pytest
from fastapi.testclient import TestClient
from test_neon import write_fake_recording

from egomem.app import create_app
from egomem.memory import MemoryEntry, TextualMemory
from egomem.reader import Answer
from egomem.scenarios import load_scenario
from egomem.sources import VideoFileSource
from egomem.uploads import convert_upload, find_recording, safe_relative

S = 1_000_000_000
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


class FakeReader:
    def answer(self, entries, question):
        return Answer(text="Yes [00:10].", citations=[10], latency_s=0.1, n_entries=len(entries))


def make_video(path, seconds, size="160x120"):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
                    "-i", f"testsrc=size={size}:rate=10", "-t", str(seconds),
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


def make_app(tmp_path, fail=False):
    """The app with a fake diary writer: one entry per 15 s of the uploaded video."""
    (tmp_path / "scenarios").mkdir()
    built = []

    def build(scene_id):
        built.append(scene_id)
        if fail:
            raise ConnectionError("describer unreachable")
        scenario = load_scenario(tmp_path / f"scenarios/{scene_id}.json")
        duration = VideoFileSource(tmp_path / scenario.video).duration
        memory = TextualMemory(tmp_path / f"data/memories/{scene_id}/default/memory.jsonl")
        for i in range(int(-(-duration // 15))):
            memory.append(MemoryEntry(i, 15 * i, min(15 * (i + 1), duration), f"clip {i}", 1.0, 8))

    config = {"scenarios_dir": "scenarios", "data_dir": "data", "describer": {"clip_sec": 15}}
    app = create_app(config, "default", reader=FakeReader(), root=tmp_path, builder=build)
    return TestClient(app), app.state.uploads, built


def post(client, files, title=""):
    return client.post("/api/uploads", data={"title": title}, files=[
        ("files", (name, io.BytesIO(data), "application/octet-stream")) for name, data in files])


def recording_files(rec_dir, folder):
    """A recording folder as the browser sends it: one file per entry, named folder/file."""
    return [(f"{folder}/{p.name}", p.read_bytes()) for p in sorted(rec_dir.iterdir())]


def test_safe_relative_stays_inside_the_upload():
    assert str(safe_relative("rec/info.json")) == "rec/info.json"
    assert str(safe_relative("../../etc/passwd")) == "etc/passwd"
    assert str(safe_relative("/abs\\win\\x.mp4")) == "abs/win/x.mp4"
    with pytest.raises(ValueError):
        safe_relative("../..")


@needs_ffmpeg
def test_an_uploaded_video_becomes_a_scene_and_is_described(tmp_path):
    client, uploads, built = make_app(tmp_path)
    video = make_video(tmp_path / "in.mp4", 20).read_bytes()

    job = post(client, [("window.mov", video)], title="Closing the window").json()
    assert job["id"] == "upload-closing-the-window" and job["title"] == "Closing the window"
    uploads.wait_idle()

    [status] = client.get("/api/uploads").json()
    assert status["stage"] == "done", status
    assert status["total_clips"] == 2 and status["clips_done"] == 2  # 0-15 s and 15-20 s
    assert built == [job["id"]]
    scenario = load_scenario(tmp_path / f"scenarios/{job['id']}.json")
    assert scenario.video == f"data/scenes/{job['id']}.mp4"
    assert scenario.source == {"upload": f"data/uploads/{job['id']}", "file": "window.mov"}
    assert VideoFileSource(tmp_path / scenario.video).duration == pytest.approx(20, abs=0.2)
    (tmp_path / "scenarios/zz.json").write_text(json.dumps(
        {"id": "zz", "title": "Demo", "category": "other", "video": "data/scenes/zz.mp4"}))
    assert [s["id"] for s in client.get("/api/scenarios").json()] == ["zz", job["id"]]  # uploads last
    assert client.get(f"/api/scenarios/{job['id']}/video").status_code == 200

    again = post(client, [("window.mov", video)], title="Closing the window").json()
    assert again["id"] == "upload-closing-the-window-2"  # never overwrites an earlier upload
    uploads.wait_idle()


@needs_ffmpeg
def test_reencoding_caps_the_width_and_keeps_the_aspect(tmp_path):
    upload = tmp_path / "data/uploads/u"
    upload.mkdir(parents=True)
    make_video(upload / "big.mp4", 1, size="400x300")
    scenario = convert_upload(upload, "u", "U", tmp_path / "scenarios", root=tmp_path, width=200)
    [frame] = VideoFileSource(tmp_path / scenario.video).frames_at([0.5])
    assert frame.shape == (150, 200, 3)


@needs_ffmpeg
@pytest.mark.parametrize("as_zip", [False, True])
def test_a_native_neon_recording_goes_through_the_importer(tmp_path, as_zip):
    pytest.importorskip("pupil_labs.neon_recording")
    client, uploads, _ = make_app(tmp_path)
    write_fake_recording(tmp_path / "rec", 1_761_000_000 * S)
    files = recording_files(tmp_path / "rec", "2026-10-23_10-00-00")
    if as_zip:  # Pupil Cloud's download nests the recording one folder deeper
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for name, data in files:
                archive.writestr(f"Native Recording Data/{name}", data)
        files = [("export.zip", buffer.getvalue())]

    job = post(client, files, title="Kitchen").json()
    uploads.wait_idle()

    [status] = client.get("/api/uploads").json()
    assert status["stage"] == "done", status
    scenario = load_scenario(tmp_path / f"scenarios/{job['id']}.json")
    assert scenario.source["neon_recording_id"] == "fake-rec"  # read with Neon's timestamps
    assert scenario.source["file"].endswith("2026-10-23_10-00-00")
    assert VideoFileSource(tmp_path / scenario.video).duration == pytest.approx(3.0, abs=0.11)


def test_a_cloud_export_with_info_json_but_no_native_video_is_a_plain_video(tmp_path):
    (tmp_path / "export").mkdir()
    (tmp_path / "export/info.json").write_text("{}")
    (tmp_path / "export/scene.mp4").write_bytes(b"x")
    assert find_recording(tmp_path) == ("video", tmp_path / "export/scene.mp4")


def test_a_zip_reaching_outside_its_folder_is_refused(tmp_path):
    client, uploads, built = make_app(tmp_path)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("../../escaped.mp4", b"x")
    post(client, [("evil.zip", buffer.getvalue())])
    uploads.wait_idle()

    [status] = client.get("/api/uploads").json()
    assert status["stage"] == "error" and "outside" in status["error"]
    assert not list(tmp_path.rglob("escaped.mp4")) and built == []
    assert not list((tmp_path / "scenarios").iterdir())


def test_a_file_that_is_neither_video_nor_recording_is_refused_at_once(tmp_path):
    client, _, _ = make_app(tmp_path)
    response = post(client, [("notes.txt", b"hello")])
    assert response.status_code == 400 and "video" in response.json()["detail"]
    assert client.get("/api/uploads").json() == []
    assert not (tmp_path / "data/uploads").exists()


@needs_ffmpeg
def test_a_failing_describer_is_reported_and_the_next_upload_still_runs(tmp_path):
    client, uploads, built = make_app(tmp_path, fail=True)
    video = make_video(tmp_path / "in.mp4", 2).read_bytes()
    post(client, [("a.mp4", video)], title="a")
    post(client, [("b.mp4", video)], title="b")
    uploads.wait_idle()

    statuses = {s["id"]: s for s in client.get("/api/uploads").json()}
    assert [s["stage"] for s in statuses.values()] == ["error", "error"]
    assert statuses["upload-a"]["error"] == "describer unreachable"
    assert built == ["upload-a", "upload-b"]  # the worker survived the first failure
    # the scene itself is kept: egomem-build can describe it once the model is back
    assert json.loads((tmp_path / "scenarios/upload-a.json").read_text())["id"] == "upload-a"
