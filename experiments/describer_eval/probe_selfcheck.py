"""Phase 1b-0, follow-up (not pre-registered): does e4b say yes to objects it named itself?

The probe found that e4b says yes to few of the objects it leaves out. If it also says no to
objects it wrote in its own diary entry for the same window, the yes/no format is the
problem (a "no" bias), not perception. Same windows, frames and question as the probe.

    uv run python experiments/describer_eval/probe_selfcheck.py \
        --config experiments/describer_eval/config.yaml \
        --taxonomy ~/data/raw/ego4d/v2/annotations/narration_noun_taxonomy.csv \
        --probe ~/data/ego-memory-assistant/describer_eval/results/probe
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from egomem.clips import frame_offsets
from egomem.config import load_config, memory_dir
from egomem.llm import ChatClient, image_part
from egomem.memory import TextualMemory
from egomem.scenarios import load_scenarios
from egomem.sources import VideoFileSource

from narrations import Taxonomy
from perception_probe import ask, phrases


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe follow-up: e4b asked about its own objects.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True, help="output folder of perception_probe.py")
    parser.add_argument("--model", default="gemma4:e4b")
    parser.add_argument("--tag", default="e4b_noex", help="diary whose own objects are asked")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    tax = Taxonomy.load(args.taxonomy)
    scenarios = load_scenarios(Path(config["scenarios_dir"]))
    rows = [json.loads(l) for l in (args.probe / f"probe_{args.model.replace(':', '_')}.jsonl").read_text().splitlines()]
    rng = random.Random(args.seed)
    client = ChatClient(config["base_url"], args.model)
    n_frames = config["describer"]["n_frames"]
    out = []
    for r in rows:
        entries = {e.index: e for e in TextualMemory.load(memory_dir(config, r["scenario"], args.tag) / "memory.jsonl").entries()}
        e = entries[r["index"]]
        own = sorted({p for p, _ in phrases(e.text, tax)})
        if not own:
            continue
        phrase = rng.choice(own)
        s = scenarios[r["scenario"]]
        source = VideoFileSource(Path(s.video), start=s.source["segment"][0], end=s.source["segment"][1])
        times = [e.t0 + off for off in frame_offsets(e.t1 - e.t0, n_frames)]
        parts = [image_part(f) for f in source.frames_at(times)]
        out.append({"scenario": r["scenario"], "index": r["index"], "own": phrase, "own_answer": ask(client, parts, phrase),
                    "present_answer": r["present_answer"], "absent_answer": r["absent_answer"]})
    with (args.probe / f"selfcheck_{args.model.replace(':', '_')}.jsonl").open("w") as fh:
        for x in out:
            fh.write(json.dumps(x) + "\n")
    n = len(out)
    summary = {"windows": n, "yes_own": sum(x["own_answer"] == "yes" for x in out) / n,
               "yes_present": sum(x["present_answer"] == "yes" for x in out) / n,
               "yes_absent": sum(x["absent_answer"] == "yes" for x in out) / n}
    (args.probe / f"selfcheck_{args.model.replace(':', '_')}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
