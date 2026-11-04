"""Phase 1b-3: save every window's 8 frames as JPEG, for describers that run outside the server.

The frames are the ones egomem-build would send (same windows, same offsets, same JPEG
encoding), so a Hugging Face describer sees what the Ollama describer saw.

    uv run python experiments/describer_eval/dump_frames.py \
        --config experiments/describer_eval/config.yaml \
        --out ~/data/ego-memory-assistant/describer_eval/frames_test_close
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from egomem.clips import clip_bounds, frame_offsets
from egomem.config import load_config
from egomem.memory import fmt_time
from egomem.prompts import describer_prompt
from egomem.scenarios import load_scenarios
from egomem.sources import VideoFileSource

from sft_data import jpeg_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump the describer's frames of every window as JPEG.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scenarios", type=Path, help="default: scenarios_dir of the config")
    parser.add_argument("--prompt-variant", default="no_examples")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    d = config["describer"]
    scenarios = load_scenarios(args.scenarios or Path(config["scenarios_dir"]))
    (args.out / "frames").mkdir(parents=True, exist_ok=True)
    index_path = args.out / "index.jsonl"
    done = set()
    if index_path.exists():
        done = {(r["clip"], r["index"]) for r in map(json.loads, index_path.read_text().splitlines())}
    with index_path.open("a") as fh:
        for sid, s in scenarios.items():
            start, end = s.source["segment"]
            source = VideoFileSource(Path(s.video), start=start, end=end)
            for i, (t0, t1) in enumerate(clip_bounds(source.duration, d["clip_sec"])):
                if (sid, i) in done:
                    continue
                times = [t0 + off for off in frame_offsets(t1 - t0, d["n_frames"])]
                paths = []
                for j, frame in enumerate(source.frames_at(times)):
                    p = args.out / "frames" / f"{sid}_{i:03d}_{j}.jpg"
                    p.write_bytes(jpeg_bytes(frame))
                    paths.append(str(p.relative_to(args.out)))
                prompt = describer_prompt(fmt_time(t0), fmt_time(t1), d["n_frames"], d["max_words"],
                                          variant=args.prompt_variant)
                fh.write(json.dumps({"clip": sid, "index": i, "t0": t0, "t1": t1, "frames": paths,
                                     "prompt": prompt}) + "\n")
            fh.flush()
    print(f"frames of {len(scenarios)} clips in {args.out}")


if __name__ == "__main__":
    main()
