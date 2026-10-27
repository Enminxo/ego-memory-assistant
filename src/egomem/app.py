"""Web app: watch the diary being written alongside the video, and ask it questions.

The diary was built offline (egomem-build); the page replays it in sync with
the video, showing only the entries of clips that have already ended. A
question asked at second t is answered from those entries alone, so the replay
behaves like a live run would.

A recording uploaded from the page is described right after it arrives
(uploads.py), and its diary shows up as it grows. "Go live" connects to the
Neon glasses, shows what they see, and writes the diary as they stream
(live.LiveRunner): through the relay on the phone, which posts its photos to
/api/phone/photo (phone.py), or directly over the Companion's WebSocket.

    egomem-app --config configs/demo.yaml --tag default --port 8090
"""

from __future__ import annotations

import argparse
import asyncio
import time
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from egomem.build_memory import build
from egomem.config import load_config, make_describer, make_reader, memory_dir
from egomem.describer import Describer
from egomem.live import LiveRunner, StreamSource
from egomem.memory import EvidenceStore, MemoryEntry, TextualMemory
from egomem.neon_stream import StaleStream
from egomem.phone import PhoneSource, open_phone_source
from egomem.reader import Reader
from egomem.scenarios import Scenario, load_scenarios
from egomem.uploads import UploadJobs

STATIC = Path(__file__).parent / "static"


class AskRequest(BaseModel):
    scenario: str
    question: str
    t: float | None = None  # seconds into the video; None means the whole diary


class LiveStart(BaseModel):
    address: str
    title: str = ""
    via: str = "phone"  # or "direct": the whole video over the Companion's WebSocket


def entry_at(entries: list[MemoryEntry], t: float) -> MemoryEntry | None:
    """The entry whose clip contains second t."""
    for entry in entries:
        if entry.t0 <= t < entry.t1:
            return entry
    return entries[-1] if entries and t >= entries[-1].t1 else None


def open_neon(address: str) -> StreamSource:
    from egomem.neon import NeonLiveSource

    try:
        return NeonLiveSource(address)
    except ModuleNotFoundError as exc:
        raise RuntimeError("live video needs the Pupil Labs packages: uv sync --extra neon") from exc


def mjpeg(source: StreamSource, running: Callable[[], bool], fps: float = 10.0) -> Iterator[bytes]:
    """The live view as MJPEG: one JPEG after another, which an <img> plays as video.

    While no recent frame arrives nothing is sent, so the page keeps the last
    picture and the status line says how old it is.
    """
    while running():
        try:
            jpeg = source.latest_jpeg()
        except (StaleStream, TimeoutError):
            jpeg = None
        if jpeg is not None:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                   + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
        time.sleep(1 / fps)


def create_app(config: dict, tag: str, reader: Reader | None = None,
               root: Path = Path("."), builder: Callable[[str], object] | None = None,
               live_source: Callable[[str], StreamSource] = open_neon,
               describer: Callable[[], Describer] | None = None,
               live_phone: Callable[[str], StreamSource] | None = None,
               port: int = 8090) -> FastAPI:
    """`port` is the one this app listens on: the phone relay sends its photos there."""
    def open_phone(address: str) -> StreamSource:
        d = config["describer"]
        return open_phone_source(address, d["clip_sec"], d["n_frames"], port)

    live = LiveRunner(config, tag, root, live_source,
                      describer or (lambda: make_describer(config)),
                      open_phone=live_phone or open_phone)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        # Ctrl+C on the app ends a live session properly: its scenario is marked ended.
        live.stop()
        await asyncio.to_thread(live.wait, 20)
        live.mark_ended()

    app = FastAPI(title="Episodic memory assistant", lifespan=lifespan)
    app.state.live = live
    reader = reader or make_reader(config)
    # build() resolves its paths from the working directory, which is `root`
    # when the app is started by main().
    uploads = UploadJobs(root / config["scenarios_dir"], root / config["data_dir"] / "uploads",
                         root, builder or (lambda scene_id: build(config, scene_id, tag)),
                         clip_sec=config["describer"]["clip_sec"])
    app.state.uploads = uploads

    def current_scenarios() -> dict[str, Scenario]:
        # Read on every request, like the diaries: a live session (egomem-live)
        # adds its scenario while the app is already running.
        return load_scenarios(root / config["scenarios_dir"])

    def get_scenario(scenario_id: str) -> Scenario:
        scenarios = current_scenarios()
        if scenario_id not in scenarios:
            raise HTTPException(404, f"unknown scenario {scenario_id}")
        return scenarios[scenario_id]

    def run_dir(scenario_id: str) -> Path:
        return root / memory_dir(config, scenario_id, tag)

    def load_memory(scenario_id: str) -> TextualMemory:
        # Read on every request: files are small, and a diary still being
        # built shows up in the page as it grows.
        return TextualMemory.load(run_dir(scenario_id) / "memory.jsonl")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/scenarios")
    def list_scenarios() -> list[dict]:
        return [{"id": s.id, "title": s.title, "category": s.category,
                 "demo_questions": s.demo_questions, "live": s.live,
                 "ended": s.live and "ended" in s.source,
                 "n_entries": len(load_memory(s.id))}
                # Own recordings (uploaded or live) after the demo scenes, each group in file order.
                for s in sorted(current_scenarios().values(),
                                key=lambda s: s.live or s.id.startswith("upload-"))]

    @app.get("/api/scenarios/{scenario_id}/memory")
    def memory(scenario_id: str) -> dict:
        get_scenario(scenario_id)
        store = EvidenceStore(run_dir(scenario_id) / "frames")
        return {"tag": tag,
                "entries": [{"index": e.index, "t0": e.t0, "t1": e.t1, "text": e.text,
                             "latency_s": e.latency_s, "n_frames": store.count(e.index)}
                            for e in load_memory(scenario_id).entries()]}

    @app.get("/api/scenarios/{scenario_id}/video")
    def video(scenario_id: str) -> FileResponse:
        path = get_scenario(scenario_id).video_path(root)
        if path is None:
            raise HTTPException(404, f"{scenario_id} is a live session and has no video")
        if not path.exists():
            raise HTTPException(404, f"video missing: {path}")
        # FileResponse honours Range requests, which the <video> element
        # needs in order to seek.
        return FileResponse(path, media_type="video/mp4")

    @app.get("/api/scenarios/{scenario_id}/frames/{index}/{i}")
    def frame(scenario_id: str, index: int, i: int) -> FileResponse:
        get_scenario(scenario_id)
        path = EvidenceStore(run_dir(scenario_id) / "frames").path(index, i)
        if not path.exists():
            raise HTTPException(404, "frame not kept")
        return FileResponse(path, media_type="image/jpeg")

    @app.post("/api/uploads")
    def upload(files: list[UploadFile] = File(...), title: str = Form("")) -> dict:
        try:
            job = uploads.submit(title, [(f.filename or "upload", f.file) for f in files])
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        return job.as_dict()

    @app.get("/api/uploads")
    def list_uploads() -> list[dict]:
        return [{**job.as_dict(), "clips_done": len(load_memory(job.id)),
                 "clip_sec": config["describer"]["clip_sec"]}
                for job in uploads.jobs.values()]

    @app.post("/api/live/start")
    def live_start(request: LiveStart) -> dict:
        try:
            live.start(request.address.strip(), request.title.strip(), request.via)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        return live.status()

    @app.post("/api/live/stop")
    def live_stop() -> dict:
        live.stop()
        return live.status()

    @app.get("/api/live")
    def live_status() -> dict:
        return live.status()

    @app.post("/api/phone/photo")
    async def phone_photo(request: Request, token: str, clip: int, slot: int, taken: float,
                          pending: int = 0) -> dict:
        """One photo of the live diary from the phone relay (phone_relay.py).

        409 tells the relay to stop sending: there is no such session here (any
        more). 503 asks it to try again: the session is still connecting.
        """
        source = live.source
        if live.state == "connecting" and source is None:
            raise HTTPException(503, "the live session is still starting")
        if not isinstance(source, PhoneSource) or live.state not in ("connecting", "running"):
            raise HTTPException(409, "no live session through the phone in this app")
        jpeg = await request.body()
        try:
            await asyncio.to_thread(source.add_photo, token, clip, slot, taken, pending, jpeg)
        except PermissionError as exc:
            raise HTTPException(409, str(exc))
        except (ValueError, OSError) as exc:  # not a JPEG, or a slot outside the clip
            raise HTTPException(400, str(exc))
        return {"received": source.photos_received}

    @app.get("/api/live/stream.mjpg")
    def live_stream() -> StreamingResponse:
        source = live.source
        if source is None or live.state not in ("connecting", "running"):
            raise HTTPException(404, "no live session is running in this app")
        running = lambda: live.source is source and live.state in ("connecting", "running")
        return StreamingResponse(mjpeg(source, running), headers={"Cache-Control": "no-store"},
                                 media_type="multipart/x-mixed-replace; boundary=frame")

    @app.post("/api/ask")
    def ask(request: AskRequest) -> dict:
        get_scenario(request.scenario)
        memory = load_memory(request.scenario)
        entries = memory.entries() if request.t is None else memory.until(request.t)
        answer = reader.answer(entries, request.question)
        cited = []
        for t in answer.citations:
            entry = entry_at(entries, t)
            if entry is not None and entry.index not in [c["index"] for c in cited]:
                cited.append({"index": entry.index, "t0": entry.t0, "t1": entry.t1, "at": t})
        return {"text": answer.text, "citations": cited,
                "latency_s": round(answer.latency_s, 2), "n_entries": answer.n_entries}

    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Serve the episodic memory demo.")
    parser.add_argument("--config", type=Path, default=Path("configs/demo.yaml"))
    parser.add_argument("--tag", default="default", help="which diary run to show")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    uvicorn.run(create_app(load_config(args.config), args.tag, port=args.port),
                host=args.host, port=args.port)


if __name__ == "__main__":
    main()
