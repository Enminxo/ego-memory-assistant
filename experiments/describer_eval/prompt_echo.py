"""Phase 1a: how often each diary names the default prompt's example objects, and writes a negation.

The default describer prompt lists "(keys, wallet, phone, glasses, bags)" as personal objects.
On QAEgo4D test_close, gemma4:e4b wrote "keys" in a third of its windows while the human
narration names keys once (README.md, Results). This counts, per diary, the windows that name
any of those objects, those that name keys, and those with a negation such as "I do not
interact with any doors" (which the A0 taxonomy counts as a mention).

    uv run python experiments/describer_eval/prompt_echo.py \
        --config experiments/describer_eval/config.yaml --tags narration e4b e4b_noex g31b \
        --out ~/data/ego-memory-assistant/describer_eval/results/prompt_echo.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from egomem.config import load_config, memory_dir
from egomem.memory import TextualMemory
from egomem.scenarios import load_scenarios

EXAMPLE_OBJECTS = re.compile(r"\b(keys?|wallets?|phones?|glasses|bags?)\b", re.IGNORECASE)
KEYS = re.compile(r"\bkeys?\b", re.IGNORECASE)
NEGATION = re.compile(r"\b(I do not|I did not|I don't|I didn't|not visible|no other|nor do I|nor did I)\b",
                      re.IGNORECASE)


def count(texts: list[str]) -> dict[str, float]:
    n = len(texts)
    out = {"windows": n,
           "example_objects": sum(bool(EXAMPLE_OBJECTS.search(t)) for t in texts),
           "keys": sum(bool(KEYS.search(t)) for t in texts),
           "negation": sum(bool(NEGATION.search(t)) for t in texts)}
    for k in ("example_objects", "keys", "negation"):
        out[f"{k}_share"] = round(out[k] / n, 4) if n else None
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt echo and negations per diary.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tags", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    scenarios = load_scenarios(Path(config["scenarios_dir"]))
    report = {}
    for tag in args.tags:
        texts: list[str] = []
        for scenario_id in scenarios:
            path = memory_dir(config, scenario_id, tag) / "memory.jsonl"
            if path.exists():
                texts += [e.text for e in TextualMemory.load(path).entries()]
        report[tag] = count(texts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
