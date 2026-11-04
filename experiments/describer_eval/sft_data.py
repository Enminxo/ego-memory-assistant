"""Phase 1b-1: teacher data for the SFT of the small describer (README.md, phase 1b).

Samples 3 full 15 s windows per QAEgo4D train clip (seed 0), holds out 10% of the clips for
validation, saves each window's 8 frames as the exact JPEG bytes the describer sends to the
server, and writes the teacher's diary entry for them with the same prompt the student will
get. Resumable: windows already in data.jsonl are skipped.

    uv run python experiments/describer_eval/sft_data.py \
        --config experiments/describer_eval/config.yaml \
        --scenarios ~/data/ego-memory-assistant/describer_eval/scenarios_train \
        --out ~/data/ego-memory-assistant/describer_eval/sft
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

from egomem.clips import clip_bounds, frame_offsets
from egomem.config import load_config
from egomem.describer import trim_incomplete_sentence
from egomem.llm import JPEG_QUALITY, ChatClient
from egomem.memory import fmt_time
from egomem.prompts import describer_prompt
from egomem.scenarios import load_scenarios
from egomem.sources import VideoFileSource


def jpeg_bytes(frame: np.ndarray) -> bytes:
    """The same encoding as egomem.llm.image_part, so training sees what the server saw."""
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def plan(scenarios: dict, clip_sec: float, per_clip: int, val_share: float, seed: int) -> list[dict]:
    """Which windows to describe, and whether each clip is train or validation."""
    rng = random.Random(seed)
    ids = sorted(scenarios)
    val = set(rng.sample(ids, round(val_share * len(ids))))
    rows = []
    for sid in ids:
        s = scenarios[sid]
        start, end = s.source["segment"]
        full = [(i, t0, t1) for i, (t0, t1) in enumerate(clip_bounds(end - start, clip_sec))
                if t1 - t0 >= clip_sec - 1e-6]
        for i, t0, t1 in sorted(rng.sample(full, min(per_clip, len(full)))):
            rows.append({"clip": sid, "index": i, "t0": t0, "t1": t1, "split": "val" if sid in val else "train"})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1b-1: teacher data for the SFT.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scenarios", type=Path, required=True, help="train scenarios directory")
    parser.add_argument("--teacher", default="gemma4:31b")
    parser.add_argument("--prompt-variant", default="no_examples")
    parser.add_argument("--per-clip", type=int, default=3)
    parser.add_argument("--val-share", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, help="describe at most N new windows")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    d = config["describer"]
    scenarios = load_scenarios(args.scenarios)
    rows = plan(scenarios, d["clip_sec"], args.per_clip, args.val_share, args.seed)
    (args.out / "frames").mkdir(parents=True, exist_ok=True)
    (args.out / "plan.json").write_text(json.dumps({"teacher": args.teacher, "prompt_variant": args.prompt_variant,
                                                    "describer": d, "seed": args.seed, "windows": rows}, indent=1))
    data_path = args.out / "data.jsonl"
    done = set()
    if data_path.exists():
        done = {(r["clip"], r["index"]) for r in map(json.loads, data_path.read_text().splitlines())}
    client = ChatClient(config["base_url"], args.teacher)
    n_new = 0
    with data_path.open("a") as fh:
        for k, r in enumerate(rows):
            if (r["clip"], r["index"]) in done:
                continue
            if args.limit is not None and n_new >= args.limit:
                break
            s = scenarios[r["clip"]]
            source = VideoFileSource(Path(s.video), start=s.source["segment"][0], end=s.source["segment"][1])
            times = [r["t0"] + off for off in frame_offsets(r["t1"] - r["t0"], d["n_frames"])]
            paths, parts = [], []
            for j, frame in enumerate(source.frames_at(times)):
                b = jpeg_bytes(frame)
                p = args.out / "frames" / f"{r['clip']}_{r['index']:03d}_{j}.jpg"
                p.write_bytes(b)
                paths.append(str(p.relative_to(args.out)))
                parts.append({"type": "image_url",
                              "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b).decode()}})
            prompt = describer_prompt(fmt_time(r["t0"]), fmt_time(r["t1"]), d["n_frames"], d["max_words"],
                                      variant=args.prompt_variant)
            t = time.perf_counter()
            result = client.chat([{"role": "user", "content": [{"type": "text", "text": prompt}] + parts}],
                                 max_tokens=d["max_tokens"], temperature=d["temperature"], seed=d.get("seed"))
            target = trim_incomplete_sentence(result.text) or "(no description)"
            fh.write(json.dumps({**r, "frames": paths, "prompt": prompt, "target": target,
                                 "latency_s": round(time.perf_counter() - t, 2)}) + "\n")
            fh.flush()
            n_new += 1
            print(f"[{k + 1}/{len(rows)}] {r['split']} {r['clip'][:8]} #{r['index']} "
                  f"({time.perf_counter() - t:4.1f}s) {target[:110]}", flush=True)
    print(f"sft data: {len(done) + n_new}/{len(rows)} windows in {data_path}")


if __name__ == "__main__":
    main()
