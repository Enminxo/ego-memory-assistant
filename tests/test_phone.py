"""The live diary through the phone: photos into clips, and the relay and app together."""

import io
import json
import os
import socket
import threading
import time
import urllib.request

import numpy as np
import pytest
import uvicorn
from fastapi.testclient import TestClient
from PIL import Image

from egomem.app import create_app
from egomem.clips import frame_offsets
from egomem.live import LiveSession
from egomem.memory import TextualMemory
from egomem.neon_stream import StaleStream
from egomem.phone import PhoneSource, open_phone_source
from egomem.phone_relay import Relay, serve
from egomem.scenarios import load_scenario
from test_app import FakeReader
from test_live import FakeDescriber, quiet, wait_for


def jpeg(value: int) -> bytes:
    """A flat grey photo of the given value, as the relay would send it."""
    out = io.BytesIO()
    Image.fromarray(np.full((8, 8, 3), value, np.uint8)).save(out, "JPEG", quality=95)
    return out.getvalue()


def add(source: PhoneSource, clip: int, slot: int, value: int, token: str | None = None,
        pending: int = 0) -> None:
    source.add_photo(token or source.token, clip, slot, time.time() - 0.4, pending, jpeg(value))


def test_a_clip_is_complete_when_its_last_photo_arrives():
    source = PhoneSource(None, clip_sec=15, n_frames=2, log=quiet)
    add(source, 0, 0, 10)
    assert source.next_clip(timeout=0) is None
    add(source, 0, 1, 20)
    clip = source.next_clip(timeout=0)
    assert (clip.index, clip.t0, clip.t1) == (0, 0, 15)
    assert clip.times == frame_offsets(15, 2)  # the same instants as egomem-build
    assert [int(f.mean()) for f in clip.frames] == pytest.approx([10, 20], abs=2)


def test_a_clip_the_phone_could_not_complete_is_dropped_and_said():
    logs: list[str] = []
    source = PhoneSource(None, clip_sec=15, n_frames=2, log=logs.append)
    add(source, 0, 0, 10)
    add(source, 1, 0, 30)  # the relay has moved on: clip 0 will get nothing more
    add(source, 1, 1, 40)
    assert source.next_clip(timeout=0).index == 1
    assert logs == ["[clip 000] dropped: 1 of 2 photos arrived from the phone"]
    add(source, 0, 1, 20)  # a straggler of the dropped clip changes nothing
    assert source.next_clip(timeout=0) is None


def test_photos_of_another_session_or_outside_the_clip_are_refused():
    source = PhoneSource(None, clip_sec=15, n_frames=2, log=quiet)
    with pytest.raises(PermissionError):
        add(source, 0, 0, 10, token="someone else")
    with pytest.raises(ValueError):
        add(source, 0, 2, 10)


def test_the_page_sees_the_newest_photo():
    source = PhoneSource(None, clip_sec=15, n_frames=2, log=quiet)
    with pytest.raises(TimeoutError):
        source.latest_rgb()
    assert source.lag() is None and source.video_age() is None
    for slot in range(2):
        add(source, 0, slot, 50, pending=3)
    assert int(source.latest_rgb().mean()) == pytest.approx(50, abs=2)
    assert source.latest_jpeg(width=4).startswith(b"\xff\xd8")
    assert source.lag() == pytest.approx(0.4, abs=0.2) and source.video_age() < 1
    assert source.fps > 0 and source.pending == 3
    source.stale_sec = 0
    with pytest.raises(StaleStream):
        source.latest_rgb()


def test_a_session_describes_whole_clips_from_the_phone_and_counts_the_gaps():
    source = PhoneSource(None, clip_sec=15, n_frames=2, log=quiet)
    for clip, slots in [(0, [0, 1]), (1, [0]), (2, [0, 1])]:
        for slot in slots:
            add(source, clip, slot, 10 * clip)
    session = LiveSession(source, FakeDescriber(), TextualMemory(), None, clip_sec=15,
                          n_frames=2, log=quiet)
    entries = session.run(n_clips=3)
    assert [(e.index, e.t0) for e in entries] == [(0, 0), (2, 30)]
    assert session.dropped == [1]


def live_config() -> dict:
    return {"scenarios_dir": "scenarios", "data_dir": "data", "base_url": "http://unused",
            "describer": {"model": "fake", "clip_sec": 15, "n_frames": 2, "keep_frames": True}}


def test_the_app_takes_photos_only_from_its_own_phone_session(tmp_path):
    sources: list[PhoneSource] = []

    def open_phone(address: str) -> PhoneSource:
        sources.append(PhoneSource(None, 15, 2, log=quiet))
        return sources[-1]

    app = create_app(live_config(), "default", reader=FakeReader(), root=tmp_path,
                     describer=FakeDescriber, live_phone=open_phone)
    client, live = TestClient(app), app.state.live
    url = "/api/phone/photo?token={}&clip={}&slot={}&taken=0&pending=0"
    assert client.post(url.format("x", 0, 0), content=jpeg(1)).status_code == 409  # no session

    client.post("/api/live/start", json={"address": "100.1.2.3", "title": "Walk"})
    wait_for(lambda: bool(sources))
    token = sources[0].token
    assert client.post(url.format(token, 0, 0), content=jpeg(10)).status_code == 200
    wait_for(lambda: live.state == "running")  # the first photo opens the tab
    assert client.post(url.format("old", 0, 1), content=jpeg(20)).status_code == 409
    assert client.post(url.format(token, 0, 1), content=b"not a jpeg").status_code == 400
    assert client.post(url.format(token, 0, 1), content=jpeg(20)).status_code == 200
    wait_for(lambda: live.status().get("clips_written", 0) == 1)
    status = live.status()
    assert status["via"] == "phone" and status["pending"] == 0

    client.post("/api/live/stop")
    live.wait(5)
    assert client.post(url.format(token, 1, 0), content=jpeg(30)).status_code == 409
    scenario = load_scenario(tmp_path / "scenarios" / f"{status['id']}.json")
    assert scenario.source["via"] == "phone"
    run = json.loads((tmp_path / "data/memories" / status["id"] / "default/run.json").read_text())
    assert run["source"] == "phone-relay"


def post(url: str, body: dict) -> dict:
    request = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def test_photos_from_the_relay_become_diary_entries(tmp_path):
    """The relay and the app over real HTTP: the contract between the phone and the DGX."""
    relay = Relay(tmp_path / "relay", lambda text: None, ffmpeg=None)
    relay.start()
    relay_server = serve(relay, "127.0.0.1", 0)
    relay_port = relay_server.server_address[1]

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    app_port = sock.getsockname()[1]
    app = create_app(live_config(), "default", reader=FakeReader(), root=tmp_path,
                     describer=FakeDescriber, port=app_port,
                     live_phone=lambda address: open_phone_source(
                         address, 15, 2, app_port, relay_port=relay_port, log=quiet))
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        wait_for(lambda: server.started)
        base = f"http://127.0.0.1:{app_port}"
        post(f"{base}/api/live/start", {"address": "127.0.0.1", "title": "Relay", "via": "phone"})
        wait_for(lambda: relay.status()["diary"] is not None)

        staging = tmp_path / "staging"  # four photos as ffmpeg leaves them, 7.5 s apart
        staging.mkdir()
        start = time.time() - 30
        for k in range(4):
            photo = staging / f"f_{k + 1:06d}.jpg"
            photo.write_bytes(jpeg(40 * k))
            os.utime(photo, (start + 7.5 * k, start + 7.5 * k))
        staging.rename(relay.diary_dir / "0000")

        live = app.state.live
        wait_for(lambda: live.status().get("clips_written", 0) == 2, seconds=15)
        assert live.status()["dropped"] == 0
        post(f"{base}/api/live/stop", {})
        live.wait(5)
        assert relay.status()["diary"] is None  # the app told the relay to stop
        out = tmp_path / "data/memories" / live.id / "default"
        entries = TextualMemory.load(out / "memory.jsonl").entries()
        assert [(e.index, e.t0, e.t1) for e in entries] == [(0, 0, 15), (1, 15, 30)]
    finally:
        server.should_exit = True
        thread.join(5)
        relay_server.shutdown()
        relay.close()


def test_the_relay_scales_the_photos_only_when_the_app_asks_for_a_width(tmp_path):
    """configs/demo.yaml live.width reaches the relay; without it the photos stay full size."""
    relay = Relay(tmp_path / "relay", lambda text: None, ffmpeg=None)
    relay.start()
    relay_server = serve(relay, "127.0.0.1", 0)
    try:
        port = relay_server.server_address[1]
        open_phone_source("127.0.0.1", 15, 8, 8090, relay_port=port, log=quiet)
        assert relay.session.width is None
        open_phone_source("127.0.0.1", 15, 8, 8090, relay_port=port, log=quiet, width=720)
        assert relay.session.width == 720 and relay.diary_params == (15, 8, 720)
    finally:
        relay_server.shutdown()
        relay.close()


def test_a_clip_from_the_phone_knows_when_it_happened_on_the_wall_clock():
    """From when its photos were taken: half a photo's share before the first and after the last."""
    source = PhoneSource(None, clip_sec=15, n_frames=2, log=quiet)
    source.add_photo(source.token, 0, 0, 1000.0 + 3.75, 0, jpeg(10))   # the middles of 0-7.5 s
    source.add_photo(source.token, 0, 1, 1000.0 + 11.25, 0, jpeg(20))  # and of 7.5-15 s
    clip = source.next_clip(timeout=0)
    assert (clip.wall_t0, clip.wall_t1) == (1000.0, 1015.0)


def test_the_live_view_shows_the_relays_quick_pictures_each_once(tmp_path):
    """Through the phone, the page's live view follows the relay's 3 pictures per
    second, not the diary's one every 1.9 s; a picture is never sent twice, and
    while the relay has none recent nothing is sent."""
    from egomem.app import mjpeg
    from egomem.phone import RelayClient

    relay = Relay(tmp_path / "relay", lambda text: None, ffmpeg=None)
    relay.start()
    relay_server = serve(relay, "127.0.0.1", 0)
    client = RelayClient(f"http://127.0.0.1:{relay_server.server_address[1]}")
    source = PhoneSource(client, clip_sec=15, n_frames=8, log=quiet)
    got: list[bytes] = []
    stop = threading.Event()

    def watch() -> None:
        for part in mjpeg(source, lambda: not stop.is_set()):
            got.append(part)

    relay.face_path.parent.mkdir(parents=True, exist_ok=True)
    viewer = threading.Thread(target=watch, daemon=True)
    viewer.start()
    try:
        time.sleep(0.8)
        assert got == []  # no picture on the phone yet (the relay answers 503)
        for i in range(3):
            # whole, as ffmpeg's -atomic_writing leaves it: a half-written file read by
            # the relay would count as one more picture (flaky under load, 16 Nov 2026)
            staged = relay.face_path.with_suffix(".tmp")
            staged.write_bytes(f"quick {i}".encode())
            os.replace(staged, relay.face_path)
            wait_for(lambda: len(got) == i + 1)
            assert got[-1].endswith(f"quick {i}\r\n".encode())
        time.sleep(2.5)  # the relay's 2 s wait runs out: 304, nothing new is sent
        assert len(got) == 3
    finally:
        stop.set()
        viewer.join(timeout=5)
        relay_server.shutdown()
        relay.close()


def test_the_companion_monitor_is_served_pointing_at_the_phone(tmp_path):
    """Neon Monitor looks the phone up as neon.local, which browsers on Tailscale cannot
    resolve: /monitor/ serves its page and code with the phone's address instead."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    import egomem.app as app_module

    files = {
        "/": ("text/html", '<script type="module" src="/assets/app.js"></script>'
                           '<link href="/assets/app.css">'),
        "/assets/app.js": ("application/javascript",
                           'const B={BASE:"http://neon.local:8080/api"};img="/assets/x.png"'),
        "/assets/x.png": ("image/png", "PNG neon.local"),
    }

    class Phone(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path not in files:
                self.send_response(404)
                self.end_headers()
                return
            ctype, text = files[self.path]
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.end_headers()
            self.wfile.write(text.encode())

        def log_message(self, *args) -> None:
            pass

    phone = ThreadingHTTPServer(("127.0.0.1", 0), Phone)
    threading.Thread(target=phone.serve_forever, daemon=True).start()
    old_port = app_module.MONITOR_PORT
    app_module.MONITOR_PORT = phone.server_address[1]
    try:
        app = create_app(live_config(), "default", reader=FakeReader(), root=tmp_path)
        client = TestClient(app)
        assert client.get("/monitor/").status_code == 404  # no live session named a phone yet
        app.state.live.address = "127.0.0.1"
        page = client.get("/monitor/").text
        assert 'src="/monitor/assets/app.js"' in page and 'href="/monitor/assets/app.css"' in page
        code = client.get("/monitor/assets/app.js").text
        assert 'BASE:"http://127.0.0.1:8080/api"' in code and '"/monitor/assets/x.png"' in code
        assert client.get("/monitor/assets/x.png").content == b"PNG neon.local"  # images untouched
        assert client.get("/monitor/missing").status_code == 404
    finally:
        app_module.MONITOR_PORT = old_port
        phone.shutdown()
