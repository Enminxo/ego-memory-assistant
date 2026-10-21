"""Build the diary of a scenario: video -> clips -> describer -> memory.jsonl.

Runs offline and as fast as the describer allows. The app then replays the
diary in sync with the video, which shows the same thing a live run would
without depending on the model keeping up in real time during a meeting.

Resumable: clips already present in memory.jsonl are skipped.

    egomem-build gas --config configs/demo.yaml
    egomem-build gas --config configs/demo.yaml --tag f4 --n-frames 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from egomem.clips import clip_bounds, iter_clips
from egomem.config import load_config, make_describer, memory_dir
from egomem.memory import EvidenceStore, MemoryEntry, TextualMemory, fmt_time
from egomem.scenarios import load_scenario
from egomem.sources import VideoFileSource


def build(config: dict, scenario_id: str, tag: str, limit: int | None = None) -> Path:
    scenario = load_scenario(Path(config["scenarios_dir"]) / f"{scenario_id}.json")
    d = config["describer"]
    out = memory_dir(config, scenario_id, tag)
    out.mkdir(parents=True, exist_ok=True)
    # The settings travel with the diary, so two runs can be told apart later.
    (out / "run.json").write_text(json.dumps({"describer": d, "base_url": config["base_url"]},
                                             indent=1))

    memory = TextualMemory.load(out / "memory.jsonl")
    evidence = EvidenceStore(out / "frames") if d.get("keep_frames") else None
    describer = make_describer(config)
    source = VideoFileSource(Path(scenario.video))

    done = {e.index for e in memory.entries()}
    total = len(clip_bounds(source.duration, d["clip_sec"]))
    written = 0
    for clip in iter_clips(source, d["clip_sec"], d["n_frames"], skip=done):
        if limit is not None and written >= limit:
            break
        result = describer.describe(clip.frames, clip.t0, clip.t1)
        entry = MemoryEntry(index=clip.index, t0=round(clip.t0, 2), t1=round(clip.t1, 2),
                            text=result.text, latency_s=round(result.latency_s, 2),
                            n_frames=len(clip.frames))
        memory.append(entry)
        if evidence is not None:
            evidence.save(clip.index, clip.frames)
        written += 1
        print(f"[{clip.index + 1:3d}/{total}] {fmt_time(clip.t0)} "
              f"({result.latency_s:4.1f}s) {result.text}", flush=True)
    print(f"{scenario_id}/{tag}: {len(memory)}/{total} clips in {out / 'memory.jsonl'}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Describe a scenario's video into a diary.")
    parser.add_argument("scenarios", nargs="+", help="scenario ids, e.g. gas keys")
    parser.add_argument("--config", type=Path, default=Path("configs/demo.yaml"))
    parser.add_argument("--tag", default="default", help="name of this run")
    parser.add_argument("--n-frames", type=int, help="override describer.n_frames")
    parser.add_argument("--seed", type=int, help="override describer.seed")
    parser.add_argument("--limit", type=int, help="describe at most N new clips")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.n_frames is not None:
        config["describer"]["n_frames"] = args.n_frames
    if args.seed is not None:
        config["describer"]["seed"] = args.seed
    for scenario_id in args.scenarios:
        build(config, scenario_id, args.tag, args.limit)


if __name__ == "__main__":
    main()
