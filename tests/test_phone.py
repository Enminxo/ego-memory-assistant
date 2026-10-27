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
