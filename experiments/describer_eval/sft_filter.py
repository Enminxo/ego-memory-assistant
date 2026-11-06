"""Phase 1b-F: the teacher data without negation sentences (README.md, phase 1b, second round).

The main SFT copied the teacher's negations ("I did not switch any appliances, taps, doors,
or windows"): 0.1% of the base's windows had one, 6.6% of the student's. This drops every
sentence of a teacher entry that matches prompt_echo.NEGATION, and drops the window if
nothing is left. The frames are shared with the unfiltered data through a symlink.

    uv run python experiments/describer_eval/sft_filter.py \
        --data ~/data/ego-memory-assistant/describer_eval/sft \
        --out ~/data/ego-memory-assistant/describer_eval/sft_noneg
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from prompt_echo import NEGATION

SENTENCE = re.compile(r"(?<=[.!?])\s+")


def strip_negations(text: str) -> str:
    return " ".join(s for s in SENTENCE.split(text) if not NEGATION.search(s)).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Teacher data without negation sentences.")
    parser.add_argument("--data", type=Path, required=True, help="output folder of sft_data.py")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(l) for l in (args.data / "data.jsonl").read_text().splitlines() if l.strip()]
    args.out.mkdir(parents=True, exist_ok=True)
    frames = args.out / "frames"
    if not frames.exists():
        frames.symlink_to((args.data / "frames").resolve(), target_is_directory=True)
    kept, changed, dropped = [], 0, 0
    for r in rows:
        target = strip_negations(r["target"])
        if not target:
            dropped += 1
            continue
        changed += target != r["target"]
        kept.append({**r, "target": target, "target_unfiltered": r["target"]})
    (args.out / "data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in kept))
    report = {"windows": len(rows), "kept": len(kept), "changed": changed, "dropped": dropped,
              "negation_share_before": sum(bool(NEGATION.search(r["target"])) for r in rows) / len(rows),
              "negation_share_after": sum(bool(NEGATION.search(r["target"])) for r in kept) / len(kept),
              "by_split": {s: sum(r["split"] == s for r in kept) for s in ("train", "val")}}
    (args.out / "filter.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
