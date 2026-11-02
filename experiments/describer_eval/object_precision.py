"""A0: do the describers name objects that the narrations of the same window name?

For every window present in both diaries (paired), objects are Ego4D taxonomy
labels found in the text. A label is supported if the narrations between
t0 - margin and t1 + margin name it. Pooled precision and recall per diary,
and the difference between the two diaries with a bootstrap over clips.
Pre-registered endpoint and gate in README.md.

    uv run python experiments/describer_eval/object_precision.py \
        --config experiments/describer_eval/config.yaml \
        --narrations ~/data/raw/ego4d/v2/annotations/narration.json \
        --taxonomy ~/data/raw/ego4d/v2/annotations/narration_noun_taxonomy.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from egomem.config import load_config, memory_dir
from egomem.memory import TextualMemory
from egomem.scenarios import load_scenarios

from narrations import STOP_WORDS, Narrations, Taxonomy, to_diary_text


@dataclass
class Counts:
    """Totals of one diary over one clip."""

    mentioned: int = 0      # labels named by the describer
    supported: int = 0      # ... that the narrations of the window also name
    narrated: int = 0       # labels named by the narrations
    recalled: int = 0       # ... that the describer also names
    windows: int = 0
    latency: float = 0.0
    unsupported: list[dict] = field(default_factory=list)


def score_clip(entries_by_tag: dict[str, dict], video_uid: str, start: float,
               narrations: Narrations, taxonomy: Taxonomy, stop: tuple[str, ...],
               margin: float, clip_uid: str) -> dict[str, Counts]:
    common = set.intersection(*(set(e) for e in entries_by_tag.values()))
    counts = {tag: Counts() for tag in entries_by_tag}
    for index in sorted(common):
        any_entry = next(iter(entries_by_tag.values()))[index]
        t0, t1 = any_entry.t0, any_entry.t1
        narr = narrations.between(video_uid, start + t0 - margin, start + t1 + margin)
        narr_text = ". ".join(to_diary_text(n) for n in narr)
        narr_labels = taxonomy.labels(narr_text, stop)
        for tag, entries in entries_by_tag.items():
            text = entries[index].text
            labels = taxonomy.labels(text, stop)
            c = counts[tag]
            c.windows += 1
            c.latency += entries[index].latency_s
            c.mentioned += len(labels)
            c.supported += len(labels & narr_labels)
            c.narrated += len(narr_labels)
            c.recalled += len(narr_labels & labels)
            for label in sorted(labels - narr_labels):
                c.unsupported.append({"clip_uid": clip_uid, "index": index, "t0": t0,
                                      "t1": t1, "label": label, "text": text,
                                      "narrations": narr_text})
    return counts


def ratio(num: float, den: float) -> float | None:
    return num / den if den else None


def pooled(per_clip: list[Counts]) -> dict:
    m = sum(c.mentioned for c in per_clip)
    s = sum(c.supported for c in per_clip)
    n = sum(c.narrated for c in per_clip)
    r = sum(c.recalled for c in per_clip)
    w = sum(c.windows for c in per_clip)
    lat = sum(c.latency for c in per_clip)
    return {"precision": ratio(s, m), "recall": ratio(r, n),
            "labels_per_window": ratio(m, w), "latency_s_per_window": ratio(lat, w),
            "mentioned": m, "supported": s, "windows": w}


def bootstrap_diff(clips: list[dict[str, Counts]], a: str, b: str, key: str,
                   n_boot: int, seed: int) -> dict:
    """Interval of pooled metric(a) - metric(b), resampling whole clips."""
    def metric(sample: list[dict[str, Counts]], tag: str) -> float:
        return pooled([c[tag] for c in sample])[key]

    point = metric(clips, a) - metric(clips, b)
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        sample = [clips[rng.randrange(len(clips))] for _ in clips]
        diffs.append(metric(sample, a) - metric(sample, b))
    diffs.sort()
    lo, hi = diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot) - 1]
    return {"diff": point, "ci95": [lo, hi], "share_le_0": sum(d <= 0 for d in diffs) / n_boot}


def run(config: dict, tags: list[str], narrations: Narrations, taxonomy: Taxonomy,
        stop: tuple[str, ...], margin: float, n_boot: int, seed: int) -> tuple[dict, list]:
    clips: list[dict[str, Counts]] = []
    for scenario_id, scenario in load_scenarios(Path(config["scenarios_dir"])).items():
        entries_by_tag = {}
        for tag in tags:
            path = memory_dir(config, scenario_id, tag) / "memory.jsonl"
            if not path.exists():
                break
            entries_by_tag[tag] = {e.index: e for e in TextualMemory.load(path).entries()}
        if len(entries_by_tag) < len(tags):
            continue
        start = scenario.source["segment"][0]
        counts = score_clip(entries_by_tag, scenario.source["video_uid"], start, narrations,
                            taxonomy, stop, margin, scenario_id)
        if next(iter(counts.values())).windows:
            clips.append(counts)
    summary = {"clips": len(clips), "margin_s": margin, "stop_words": bool(stop),
               "per_tag": {t: pooled([c[t] for c in clips]) for t in tags}}
    if len(tags) == 2 and clips:
        a, b = tags[1], tags[0]
        summary["diff"] = {f"{a}-{b}": {k: bootstrap_diff(clips, a, b, k, n_boot, seed)
                                        for k in ("precision", "recall")}}
    return summary, clips


def main() -> None:
    parser = argparse.ArgumentParser(description="A0: object precision against narrations.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--narrations", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--tags", nargs="+", default=["e4b", "g31b"],
                        help="diaries to compare; with two, reports tags[1] - tags[0]")
    parser.add_argument("--margin", type=float, default=2.0)
    parser.add_argument("--boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--spot-check", type=int, default=50,
                        help="unsupported mentions per tag written for checking by eye")
    parser.add_argument("--out", type=Path, required=True, help="output folder")
    args = parser.parse_args()

    config = load_config(args.config)
    narrations = Narrations.load(args.narrations)
    taxonomy = Taxonomy.load(args.taxonomy)
    args.out.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, stop in (("primary_stop_words", STOP_WORDS), ("secondary_all_words", ())):
        summary, clips = run(config, args.tags, narrations, taxonomy, stop, args.margin,
                             args.boot, args.seed)
        report[name] = summary
        if name == "primary_stop_words":
            rng = random.Random(args.seed)
            for tag in args.tags:
                pool = [u for c in clips for u in c[tag].unsupported]
                picked = rng.sample(pool, min(args.spot_check, len(pool)))
                with (args.out / f"spot_check_{tag}.csv").open("w", newline="") as fh:
                    writer = csv.DictWriter(fh, fieldnames=list(picked[0]) if picked else ["label"])
                    writer.writeheader()
                    writer.writerows(picked)
    (args.out / "a0_object_precision.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
