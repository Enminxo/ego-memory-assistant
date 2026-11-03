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

from egomem.clips import Clip, clip_bounds, iter_clips
from egomem.config import load_config, make_describer, memory_dir
from egomem.describer import Describer
from egomem.memory import EvidenceStore, MemoryEntry, TextualMemory, fmt_time
from egomem.scenarios import load_scenario
from egomem.sources import VideoFileSource


def describe_clip(describer: Describer, clip: Clip, memory: TextualMemory,
                  evidence: EvidenceStore | None) -> MemoryEntry:
    """Describe one clip and append it to the diary, with its frames as evidence.

    Shared by the offline build and the live session (live.py), so a diary
    written from a file and one written from the glasses are the same thing.
    """
    result = describer.describe(clip.frames, clip.t0, clip.t1)
    entry = MemoryEntry(index=clip.index, t0=round(clip.t0, 2), t1=round(clip.t1, 2),
                        text=result.text, latency_s=round(result.latency_s, 2),
                        n_frames=len(clip.frames))
    # Frames first: the app reads the diary while it grows, and an entry must
    # not show up before the frames it was written from.
    if evidence is not None:
        evidence.save(clip.index, clip.frames)
    memory.append(entry)
    return entry


def build(config: dict, scenario_id: str, tag: str, limit: int | None = None) -> Path:
    scenario = load_scenario(Path(config["scenarios_dir"]) / f"{scenario_id}.json")
    if scenario.video is None:
        raise SystemExit(f"{scenario_id} has no video: it is a live session, "
                         "and egomem-live already wrote its diary")
    d = config["describer"]
    out = memory_dir(config, scenario_id, tag)
    out.mkdir(parents=True, exist_ok=True)
    # The settings travel with the diary, so two runs can be told apart later.
    (out / "run.json").write_text(json.dumps({"describer": d, "base_url": config["base_url"]},
                                             indent=1))

    memory = TextualMemory.load(out / "memory.jsonl")
    evidence = EvidenceStore(out / "frames") if d.get("keep_frames") else None
    describer = make_describer(config)
    # A scenario may point at a segment of a long video instead of a cut file
    # (see scenarios.make_qaego4d_split): source["segment"] = [start, end].
    segment = scenario.source.get("segment")
    if segment is not None:
        source = VideoFileSource(Path(scenario.video), start=segment[0], end=segment[1])
    else:
        source = VideoFileSource(Path(scenario.video))

    done = {e.index for e in memory.entries()}
    total = len(clip_bounds(source.duration, d["clip_sec"]))
    written = 0
    for clip in iter_clips(source, d["clip_sec"], d["n_frames"], skip=done):
        if limit is not None and written >= limit:
            break
        entry = describe_clip(describer, clip, memory, evidence)
        written += 1
        print(f"[{clip.index + 1:3d}/{total}] {fmt_time(clip.t0)} "
              f"({entry.latency_s:4.1f}s) {entry.text}", flush=True)
    print(f"{scenario_id}/{tag}: {len(memory)}/{total} clips in {out / 'memory.jsonl'}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Describe a scenario's video into a diary.")
    parser.add_argument("scenarios", nargs="+", help="scenario ids, e.g. gas keys")
    parser.add_argument("--config", type=Path, default=Path("configs/demo.yaml"))
    parser.add_argument("--tag", default="default", help="name of this run")
    parser.add_argument("--n-frames", type=int, help="override describer.n_frames")
    parser.add_argument("--seed", type=int, help="override describer.seed")
    parser.add_argument("--model", help="override describer.model, e.g. gemma4:e4b")
    parser.add_argument("--prompt-variant", help="override describer.prompt_variant "
                        "(see egomem.prompts.PROMPT_VARIANTS)")
    parser.add_argument("--limit", type=int, help="describe at most N new clips")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.model is not None:
        config["describer"]["model"] = args.model
    if args.prompt_variant is not None:
        config["describer"]["prompt_variant"] = args.prompt_variant
    if args.n_frames is not None:
        config["describer"]["n_frames"] = args.n_frames
    if args.seed is not None:
        config["describer"]["seed"] = args.seed
    for scenario_id in args.scenarios:
        build(config, scenario_id, args.tag, args.limit)


if __name__ == "__main__":
    main()
