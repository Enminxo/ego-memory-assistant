"""Score the reader on a scenario's questions, QAEgo4D-Closed style.

Each question becomes a four-option multiple choice (the annotated answer
plus three wrong ones, in a shuffled order fixed by the seed). The reader sees
the diary up to the end of the scenario, as if asked once the video is over.

The number that matters is the gap to the `--no-memory` control: the same
reader with an empty diary. QAEgo4D's wrong options were written
automatically and are often absurd ("on the ceiling", "who cares"), so a
language model can beat the 25% chance level without seeing anything. Only
what the diary adds on top of that control is evidence that the memory works.

    egomem-eval gas keys meds wallet --tag default
    egomem-eval gas keys meds wallet --no-memory
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from egomem.config import load_config, make_reader, memory_dir
from egomem.memory import TextualMemory
from egomem.scenarios import load_scenario


def shuffled_options(question_id: str, answer: str, wrong: list[str],
                     seed: int) -> tuple[list[str], int]:
    """Options in a per-question order that does not depend on run order."""
    options = [answer] + list(wrong[:3])
    rng = random.Random(f"{seed}:{question_id}")
    order = list(range(len(options)))
    rng.shuffle(order)
    return [options[i] for i in order], order.index(0)


def evaluate(config: dict, scenario_ids: list[str], tag: str, no_memory: bool,
             seed: int, out_path: Path) -> dict:
    reader = make_reader(config)
    rows = []
    for scenario_id in scenario_ids:
        scenario = load_scenario(Path(config["scenarios_dir"]) / f"{scenario_id}.json")
        if no_memory:
            entries = []
        else:
            path = memory_dir(config, scenario_id, tag) / "memory.jsonl"
            if not path.exists():
                raise FileNotFoundError(f"no diary at {path}; run egomem-build first")
            entries = TextualMemory.load(path).entries()
        for q in scenario.questions:
            if not q.answer or len(q.wrong_answers) < 3:
                continue
            options, correct = shuffled_options(q.id, q.answer, q.wrong_answers, seed)
            choice, raw = reader.choose(entries, q.question, options)
            rows.append({"scenario": scenario_id, "id": q.id, "category": q.category,
                         "question": q.question, "options": options, "correct": correct,
                         "choice": choice, "raw": raw, "hit": choice == correct})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def acc(subset: list[dict]) -> dict:
        n = len(subset)
        hits = sum(r["hit"] for r in subset)
        return {"n": n, "hits": hits, "accuracy": round(hits / n, 3) if n else None}

    by_category = defaultdict(list)
    by_scenario = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
        by_scenario[row["scenario"]].append(row)
    summary = {"tag": "no-memory" if no_memory else tag, "seed": seed,
               "overall": acc(rows),
               "unparsed": sum(r["choice"] is None for r in rows),
               "by_category": {k: acc(v) for k, v in sorted(by_category.items())},
               "by_scenario": {k: acc(v) for k, v in sorted(by_scenario.items())}}
    out_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Multiple-choice accuracy of the reader.")
    parser.add_argument("scenarios", nargs="+")
    parser.add_argument("--config", type=Path, default=Path("configs/demo.yaml"))
    parser.add_argument("--tag", default="default", help="which diary run to read")
    parser.add_argument("--no-memory", action="store_true", help="control: empty diary")
    parser.add_argument("--seed", type=int, default=0, help="option shuffling seed")
    parser.add_argument("--out", type=Path, help="per-question JSONL (default results/...)")
    args = parser.parse_args()

    config = load_config(args.config)
    name = "no-memory" if args.no_memory else args.tag
    out = args.out or Path("results") / f"eval_{name}_seed{args.seed}.jsonl"
    summary = evaluate(config, args.scenarios, args.tag, args.no_memory, args.seed, out)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
