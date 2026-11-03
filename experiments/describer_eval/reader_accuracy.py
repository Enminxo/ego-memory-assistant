"""B: reader accuracy with each diary, paired on the same questions.

Reads the per-question JSONL written by egomem-eval for each tag and reports
accuracy per tag and the pre-registered differences, with 95% intervals from
a bootstrap over clips (README.md).

    uv run python experiments/describer_eval/reader_accuracy.py \
        --results ~/data/ego-memory-assistant/describer_eval/results
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

# (a, b): report accuracy(a) - accuracy(b). The first one is the primary endpoint.
COMPARISONS: tuple[tuple[str, str], ...] = (
    ("g31b", "no-memory"),
    ("narration", "g31b"),
    ("g31b", "e4b"),
    ("e4b", "no-memory"),
    ("narration", "no-memory"),
    # phase 1a (README): the e4b with the prompt without example objects
    ("g31b", "e4b_noex"),
    ("e4b_noex", "e4b"),
)


def load_hits(path: Path) -> dict[str, tuple[str, str, bool]]:
    """question id -> (clip, category, hit)."""
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        rows[r["id"]] = (r["scenario"], r["category"], bool(r["hit"]))
    return rows


def accuracy(hits: dict[str, tuple[str, str, bool]], ids: list[str]) -> float:
    return sum(hits[i][2] for i in ids) / len(ids)


def bootstrap(a: dict, b: dict, ids_by_clip: dict[str, list[str]], n_boot: int,
              seed: int) -> dict:
    clips = sorted(ids_by_clip)
    all_ids = [i for c in clips for i in ids_by_clip[c]]
    point = accuracy(a, all_ids) - accuracy(b, all_ids)
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        ids = [i for _ in clips for i in ids_by_clip[clips[rng.randrange(len(clips))]]]
        diffs.append(accuracy(a, ids) - accuracy(b, ids))
    diffs.sort()
    return {"diff": round(point, 4),
            "ci95": [round(diffs[int(0.025 * n_boot)], 4), round(diffs[int(0.975 * n_boot) - 1], 4)],
            "share_le_0": sum(d <= 0 for d in diffs) / n_boot}


def main() -> None:
    parser = argparse.ArgumentParser(description="B: reader accuracy per diary.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0, help="option-shuffling seed of the eval")
    parser.add_argument("--boot", type=int, default=1000)
    args = parser.parse_args()

    tags = sorted({t for pair in COMPARISONS for t in pair})
    hits = {t: load_hits(args.results / f"eval_{t}_seed{args.seed}.jsonl") for t in tags
            if (args.results / f"eval_{t}_seed{args.seed}.jsonl").exists()}
    common = sorted(set.intersection(*(set(h) for h in hits.values())))
    ref = next(iter(hits.values()))
    ids_by_clip: dict[str, list[str]] = defaultdict(list)
    ids_by_cat: dict[str, list[str]] = defaultdict(list)
    for i in common:
        ids_by_clip[ref[i][0]].append(i)
        ids_by_cat[ref[i][1]].append(i)

    report = {"questions": len(common), "clips": len(ids_by_clip),
              "accuracy": {t: round(accuracy(h, common), 4) for t, h in hits.items()},
              "by_category": {c: {"n": len(ids), **{t: round(accuracy(h, ids), 4)
                                                    for t, h in hits.items()}}
                              for c, ids in sorted(ids_by_cat.items())},
              "differences": {f"{a} - {b}": bootstrap(hits[a], hits[b], ids_by_clip,
                                                      args.boot, args.seed)
                              for a, b in COMPARISONS if a in hits and b in hits}}
    out = args.results / f"b_reader_accuracy_seed{args.seed}.json"
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
