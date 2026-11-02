"""Write the ceiling diary of each scenario from Ego4D's human narrations.

Same 15 s windows as the describer (clips.clip_bounds), one entry per window,
tag `narration`. Overwrites previous narration diaries (they are deterministic).

    uv run python experiments/describer_eval/narration_diary.py \
        --config experiments/describer_eval/config.yaml \
        --narrations ~/data/raw/ego4d/v2/annotations/narration.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from egomem.clips import clip_bounds
from egomem.config import load_config, memory_dir
from egomem.memory import MemoryEntry, TextualMemory
from egomem.scenarios import load_scenarios

from narrations import Narrations, to_diary_text

EMPTY = "(nothing narrated)"


def write_narration_diary(config: dict, scenario_id: str, video_uid: str, start: float,
                          end: float, narrations: Narrations, tag: str) -> tuple[int, int]:
    out = memory_dir(config, scenario_id, tag) / "memory.jsonl"
    out.unlink(missing_ok=True)
    memory = TextualMemory(out)
    bounds = clip_bounds(end - start, config["describer"]["clip_sec"])
    empty = 0
    for index, (t0, t1) in enumerate(bounds):
        texts = [to_diary_text(n) for n in narrations.between(video_uid, start + t0, start + t1)]
        texts = [t for t in texts if t]
        if not texts:
            empty += 1
        text = ". ".join(texts) + "." if texts else EMPTY
        memory.append(MemoryEntry(index=index, t0=round(t0, 2), t1=round(t1, 2), text=text,
                                  latency_s=0.0, n_frames=0))
    return len(bounds), empty


def main() -> None:
    parser = argparse.ArgumentParser(description="Ceiling diaries from Ego4D narrations.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--narrations", type=Path, required=True)
    parser.add_argument("--pass-name", default="narration_pass_1")
    parser.add_argument("--tag", default="narration")
    args = parser.parse_args()

    config = load_config(args.config)
    narrations = Narrations.load(args.narrations, args.pass_name)
    total = empty = 0
    for scenario_id, scenario in load_scenarios(Path(config["scenarios_dir"])).items():
        start, end = scenario.source["segment"]
        n, e = write_narration_diary(config, scenario_id, scenario.source["video_uid"],
                                     start, end, narrations, args.tag)
        total += n
        empty += e
    print(f"narration diaries: {total} windows, {empty} with nothing narrated "
          f"({empty / max(total, 1):.1%})")


if __name__ == "__main__":
    main()
