"""The phone relay, with a fake app, a fake ffmpeg and a voice that only listens."""

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from egomem.phone_relay import DiarySession, Relay, ffmpeg_command, serve
from test_live import wait_for


class FakeApp:
    """egomem-app's photo endpoint: the first `fail` photos get 503, then `code`."""

    def __init__(self, fail: int = 0, code: int = 200) -> None:
        self.received: list[tuple[dict, bytes]] = []
        self.fail, self.code = fail, code
        app = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers["Content-Length"]))
                query = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
                if app.fail > 0:
                    app.fail -= 1
                    code = 503
                else:
                    code = app.code
                    if code == 200:
                        app.received.append((query, body))
                self.send_response(code)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *args: object) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


@pytest.fixture
def relay(tmp_path):
    said: list[str] = []
    relay = Relay(tmp_path / "relay", said.append, ffmpeg=None)
    relay.said = said
    relay.start()
    server = serve(relay, "127.0.0.1", 0)
    relay.url = f"http://127.0.0.1:{server.server_address[1]}"
    yield relay
    server.shutdown()
    relay.close()


def post(url: str, body: dict) -> dict:
    request = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST")
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read())


def drop_photos(relay: Relay, n: int, tmp_path: Path, run: str = "0000",
                period: float = 15 / 8) -> None:
    """n photos as ffmpeg leaves them, all appearing at once in a run folder."""
    staging = tmp_path / f"staging-{run}"
    staging.mkdir()
    start = time.time() - n * period
    for k in range(n):
        photo = staging / f"f_{k + 1:06d}.jpg"
        photo.write_bytes(f"photo {k}".encode())
        os.utime(photo, (start + k * period, start + k * period))
    staging.rename(relay.diary_dir / run)


def test_photos_keep_their_slot_by_when_they_were_taken():
    session = DiarySession("url", "token", clip_sec=15, n_frames=8)  # one every 1.875 s
    assert [session.place(t) for t in [100.0, 101.9, 103.7, 105.65]] == [0, 1, 2, 3]
    assert session.place(105.7) == 4  # two photos never share a slot
    assert session.place(115.0) == 8  # after a gap the number follows the clock: clip 1


def test_one_decoder_writes_both_the_diary_and_the_face_photo():
    command = ffmpeg_command("ffmpeg", "rtsp://127.0.0.1:8086/?camera=world", Path("d"),
                             Path("f/latest.jpg"), 15, 8, None, 3, 960)
    graph = command[command.index("-filter_complex") + 1]
    assert "fps=8/15[diary]" in graph and "fps=3,scale=960:-2[face]" in graph
    assert command.index("-rtsp_transport") < command.index("-i")
    assert command[-1] == "f/latest.jpg" and "-update" in command
    assert command.count("-atomic_writing") == 2  # a file that exists is a whole JPEG


def test_photos_are_sent_in_order_and_retried_until_they_arrive(relay, tmp_path):
    app = FakeApp(fail=2)  # the Wi-Fi drops twice on the first photo
    post(relay.url + "/diary/start",
         {"port": app.port, "token": "abc", "clip_sec": 15, "n_frames": 8})
    drop_photos(relay, 10, tmp_path)
    wait_for(lambda: len(app.received) == 10, seconds=10)

    places = [(int(q["clip"]), int(q["slot"])) for q, _ in app.received]
    assert places == [(0, slot) for slot in range(8)] + [(1, 0), (1, 1)]
    assert [body for _, body in app.received] == [f"photo {k}".encode() for k in range(10)]
    assert {q["token"] for q, _ in app.received} == {"abc"}
    assert [int(q["pending"]) for q, _ in app.received][:2] == [10, 9]
    wait_for(lambda: relay.photos() == [])
    assert relay.status()["diary"]["sent"] == 10 and relay.status()["upload_error"] is None


def test_an_app_without_the_session_ends_it_on_the_relay(relay, tmp_path):
    app = FakeApp(code=409)
    post(relay.url + "/diary/start", {"port": app.port, "token": "old", "clip_sec": 15,
                                      "n_frames": 8})
    drop_photos(relay, 3, tmp_path)
    wait_for(lambda: relay.status()["diary"] is None)
    wait_for(lambda: relay.photos() == [])  # nobody wants them now
    assert app.received == []


def test_photos_outside_a_session_are_thrown_away(relay, tmp_path):
    drop_photos(relay, 3, tmp_path)
    wait_for(lambda: relay.photos() == [])


def test_a_new_session_never_sends_the_photos_of_the_one_before(relay, tmp_path):
    unreachable = FakeApp(code=503)
    post(relay.url + "/diary/start", {"port": unreachable.port, "token": "first",
                                      "clip_sec": 15, "n_frames": 8})
    drop_photos(relay, 3, tmp_path, run="0000")
    wait_for(lambda: relay.status()["upload_error"] is not None)

    app = FakeApp()
    post(relay.url + "/diary/start", {"port": app.port, "token": "second",
                                      "clip_sec": 15, "n_frames": 8})
    drop_photos(relay, 1, tmp_path, run="0001")
    wait_for(lambda: len(app.received) == 1, seconds=10)
    query, body = app.received[0]
    assert (query["token"], query["clip"], query["slot"], body) == ("second", "0", "0", b"photo 0")


def test_the_latest_face_photo_and_the_voice(relay):
    with pytest.raises(urllib.error.HTTPError) as no_picture:
        urllib.request.urlopen(relay.url + "/latest.jpg", timeout=5)
    assert no_picture.value.code == 503

    seqs = []
    for i, age in enumerate([1.0, 0.5]):
        relay.face_path.write_bytes(f"face {i}".encode())
        os.utime(relay.face_path, (time.time() - age, time.time() - age))
        with urllib.request.urlopen(relay.url + "/latest.jpg", timeout=5) as response:
            assert response.read() == f"face {i}".encode()
            seqs.append(response.headers["X-Frame-Seq"])
    assert seqs[0] != seqs[1]  # how facewatch tells a new picture from the same one

    started = time.monotonic()  # asking for a newer one than the newest waits, then says so
    with pytest.raises(urllib.error.HTTPError) as unchanged:
        urllib.request.urlopen(f"{relay.url}/latest.jpg?after={seqs[1]}", timeout=5)
    assert unchanged.value.code == 304 and time.monotonic() - started > 1.5

    def new_picture() -> None:
        time.sleep(0.3)
        relay.face_path.write_bytes(b"face 2")
    threading.Thread(target=new_picture).start()
    with urllib.request.urlopen(f"{relay.url}/latest.jpg?after={seqs[1]}", timeout=5) as response:
        assert response.read() == b"face 2"  # handed over as soon as it is written

    os.utime(relay.face_path, (time.time() - 60, time.time() - 60))
    with pytest.raises(urllib.error.HTTPError) as stale:
        urllib.request.urlopen(relay.url + "/latest.jpg", timeout=5)
    assert stale.value.code == 503  # a picture from a minute ago is not "now"

    post(relay.url + "/say", {"text": "Hola, Ana"})
    wait_for(lambda: relay.said == ["Hola, Ana"])
    with pytest.raises(urllib.error.HTTPError) as bad:
        post(relay.url + "/say", {})
    assert bad.value.code == 400


def test_ffmpeg_is_restarted_when_it_dies_and_when_the_diary_rate_changes(tmp_path):
    calls = tmp_path / "calls.jsonl"
    fake = tmp_path / "ffmpeg"
    fake.write_text(f"#!{sys.executable}\nimport json, sys, time\n"
                    f"open({str(calls)!r}, 'a').write(json.dumps(sys.argv[1:]) + '\\n')\n"
                    "time.sleep(60)\n")
    fake.chmod(0o755)
    runs = lambda: calls.read_text().splitlines() if calls.exists() else []

    relay = Relay(tmp_path / "relay", lambda text: None, ffmpeg=str(fake))
    relay.start()
    try:
        wait_for(lambda: len(runs()) == 1)
        assert "fps=8/15" in runs()[0]
        relay.diary_start("http://127.0.0.1:9/api/phone/photo", "t", 10, 5)
        wait_for(lambda: len(runs()) == 2)
        assert "fps=5/10" in runs()[1] and relay.ffmpeg_error is None  # asked for, not a failure

        relay._proc.kill()  # the Companion stream went away
        wait_for(lambda: relay.ffmpeg_error is not None)
        assert "exited with code" in relay.ffmpeg_error
        wait_for(lambda: len(runs()) == 3, seconds=10)
        assert len(list(relay.diary_dir.iterdir())) == 3  # each run in its own folder
    finally:
        relay.close()
