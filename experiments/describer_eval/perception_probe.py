"""Phase 1b-0: does the small describer see the objects it leaves out of its diary?

For windows where g31b names an object that the narration also names and e4b_noex does not,
ask a model, with the same 8 frames the describer saw, whether that object is there (answer
constrained to yes or no). Control in the same window: an object from the same pool of
phrases that nobody names anywhere in the clip. See README.md, phase 1b.

    uv run python experiments/describer_eval/perception_probe.py \
        --config experiments/describer_eval/config.yaml \
        --narrations ~/data/raw/ego4d/v2/annotations/narration.json \
        --taxonomy ~/data/raw/ego4d/v2/annotations/narration_noun_taxonomy.csv \
        --out ~/data/ego-memory-assistant/describer_eval/results/probe
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from egomem.clips import frame_offsets
from egomem.config import load_config, memory_dir
from egomem.llm import ChatClient, image_part
from egomem.memory import TextualMemory
from egomem.scenarios import load_scenarios
from egomem.sources import VideoFileSource

from narrations import STOP_WORDS, Narrations, Taxonomy, to_diary_text

QUESTION = ("These {n} frames come from an egocentric video, in chronological order. "
            "Is there any {phrase} visible in these frames? Answer yes or no.")


def phrases(text: str, tax: Taxonomy) -> list[tuple[str, frozenset[str]]]:
    """(phrase as written, its taxonomy labels) for each object word in text, stop words skipped."""
    low = " " + text.lower() + " "
    masked = low
    for w in STOP_WORDS:
        masked = re.sub(r"\b" + re.escape(w) + r"\b", lambda m: " " * len(m.group(0)), masked)
    out = []
    for m in tax.pattern.finditer(masked):
        out.append((low[m.start():m.end()].strip(), frozenset(tax.phrase_to_labels[m.group(1)])))
    return out


def candidates(config: dict, narrations: Narrations, tax: Taxonomy, margin: float) -> list[dict]:
    """Windows with at least one object named by g31b and the narration but not by e4b_noex."""
    out = []
    for sid, scenario in load_scenarios(Path(config["scenarios_dir"])).items():
        diaries = {}
        for tag in ("g31b", "e4b_noex"):
            path = memory_dir(config, sid, tag) / "memory.jsonl"
            diaries[tag] = {e.index: e for e in TextualMemory.load(path).entries()}
        video_uid, (start, end) = scenario.source["video_uid"], scenario.source["segment"]
        clip_narr = tax.labels(". ".join(to_diary_text(n) for n in narrations.between(video_uid, start, end)),
                               STOP_WORDS)
        for index in sorted(set(diaries["g31b"]) & set(diaries["e4b_noex"])):
            g, e = diaries["g31b"][index], diaries["e4b_noex"][index]
            narr = tax.labels(". ".join(to_diary_text(n) for n in
                                        narrations.between(video_uid, start + g.t0 - margin, start + g.t1 + margin)),
                              STOP_WORDS)
            g_ph = phrases(g.text, tax)
            e_labels = tax.labels(e.text, STOP_WORDS)
            present = [(p, ls) for p, ls in g_ph if ls & narr and not ls & e_labels]
            named = set().union(*(ls for _, ls in g_ph)) | e_labels
            out.append({"scenario": sid, "index": index, "t0": g.t0, "t1": g.t1, "video": scenario.video,
                        "segment": [start, end], "present": present, "exclude": named | clip_narr})
    return out


def sample(windows: list[dict], n: int, seed: int) -> list[dict]:
    """n windows with a present object, each paired with an absent control from the same pool."""
    rng = random.Random(seed)
    pool = sorted({(p, ls) for w in windows for p, ls in w["present"]}, key=lambda x: (x[0], sorted(x[1])))
    eligible = [w for w in windows if w["present"]]
    picked = []
    for w in rng.sample(eligible, len(eligible)):
        if len(picked) == n:
            break
        phrase, labels = rng.choice(sorted(w["present"], key=lambda x: x[0]))
        controls = [(p, ls) for p, ls in pool if not ls & w["exclude"] and p != phrase]
        if not controls:
            continue
        absent, _ = rng.choice(controls)
        picked.append({**{k: w[k] for k in ("scenario", "index", "t0", "t1", "video", "segment")},
                       "present": phrase, "absent": absent})
    return picked


def ask(client: ChatClient, parts: list[dict], phrase: str) -> str:
    schema = {"type": "object", "properties": {"answer": {"type": "string", "enum": ["yes", "no"]}},
              "required": ["answer"], "additionalProperties": False}
    content = [{"type": "text", "text": QUESTION.format(n=len(parts), phrase=phrase)}] + parts
    result = client.chat([{"role": "user", "content": content}], max_tokens=16, temperature=0.0, seed=0,
                         response_format={"type": "json_schema",
                                          "json_schema": {"name": "yesno", "strict": True, "schema": schema}})
    try:
        return json.loads(result.text)["answer"]
    except (ValueError, KeyError, TypeError):
        return "unparsed"


def bootstrap(rows: list[dict], n_boot: int, seed: int) -> dict:
    """Paired yes(present) - yes(absent), resampling clips."""
    by_clip: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_clip[r["scenario"]].append(r)
    clips = sorted(by_clip)

    def diff(rs: list[dict]) -> float:
        return sum((r["present_answer"] == "yes") - (r["absent_answer"] == "yes") for r in rs) / len(rs)

    rng = random.Random(seed)
    boots = sorted(diff([r for _ in clips for r in by_clip[clips[rng.randrange(len(clips))]]])
                   for _ in range(n_boot))
    return {"diff": diff(rows), "ci95": [boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1b-0: perception probe.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--narrations", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["gemma4:e4b", "gemma4:31b"])
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--margin", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--boot", type=int, default=1000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    tax = Taxonomy.load(args.taxonomy)
    windows = candidates(config, Narrations.load(args.narrations), tax, args.margin)
    picked = sample(windows, args.n, args.seed)
    print(f"{sum(bool(w['present']) for w in windows)} of {len(windows)} windows have a candidate; "
          f"{len(picked)} sampled", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)

    n_frames = config["describer"]["n_frames"]
    parts_cache: dict[tuple[str, int], list[dict]] = {}
    for w in picked:
        source = VideoFileSource(Path(w["video"]), start=w["segment"][0], end=w["segment"][1])
        times = [w["t0"] + off for off in frame_offsets(w["t1"] - w["t0"], n_frames)]
        parts_cache[(w["scenario"], w["index"])] = [image_part(f) for f in source.frames_at(times)]

    summary = {"windows": len(windows), "eligible": sum(bool(w["present"]) for w in windows),
               "sampled": len(picked), "question": QUESTION, "models": {}}
    for model in args.models:
        client = ChatClient(config["base_url"], model)
        rows = []
        for i, w in enumerate(picked):
            parts = parts_cache[(w["scenario"], w["index"])]
            rows.append({k: w[k] for k in ("scenario", "index", "present", "absent")}
                        | {"present_answer": ask(client, parts, w["present"]),
                           "absent_answer": ask(client, parts, w["absent"])})
            if (i + 1) % 50 == 0:
                print(f"  {model}: {i + 1}/{len(picked)}", flush=True)
        with (args.out / f"probe_{model.replace(':', '_')}.jsonl").open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        n = len(rows)
        summary["models"][model] = {
            "yes_present": sum(r["present_answer"] == "yes" for r in rows) / n,
            "yes_absent": sum(r["absent_answer"] == "yes" for r in rows) / n,
            "unparsed": sum("unparsed" in (r["present_answer"], r["absent_answer"]) for r in rows),
            **bootstrap(rows, args.boot, args.seed)}
        print(model, json.dumps(summary["models"][model]), flush=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
