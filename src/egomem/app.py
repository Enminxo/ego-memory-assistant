"""Web app: watch the diary being written alongside the video, and ask it questions.

The diary was built offline (egomem-build); the page replays it in sync with
the video, showing only the entries of clips that have already ended. A
question asked at second t is answered from those entries alone, so the replay
behaves like a live run would.

    egomem-app --config configs/demo.yaml --tag default --port 8090
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from egomem.config import load_config, make_reader, memory_dir
from egomem.memory import EvidenceStore, MemoryEntry, TextualMemory
from egomem.reader import Reader
from egomem.scenarios import Scenario, load_scenarios

STATIC = Path(__file__).parent / "static"


class AskRequest(BaseModel):
    scenario: str
    question: str
    t: float | None = None  # seconds into the video; None means the whole diary


def entry_at(entries: list[MemoryEntry], t: float) -> MemoryEntry | None:
    """The entry whose clip contains second t."""
    for entry in entries:
        if entry.t0 <= t < entry.t1:
            return entry
    return entries[-1] if entries and t >= entries[-1].t1 else None


def create_app(config: dict, tag: str, reader: Reader | None = None,
               root: Path = Path(".")) -> FastAPI:
    app = FastAPI(title="Episodic memory assistant")
    reader = reader or make_reader(config)

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
                for s in current_scenarios().values()]

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
    uvicorn.run(create_app(load_config(args.config), args.tag), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
