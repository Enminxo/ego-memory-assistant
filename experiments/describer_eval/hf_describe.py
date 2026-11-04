"""Phase 1b-3: write diaries with gemma-4-E4B-it in Hugging Face, with or without the SFT adapter.

Base and student run here with the same settings, so their comparison does not mix in the
4-bit quantization of the Ollama model. Reads the frames and prompts of dump_frames.py and
writes memories/<clip>/<tag>/memory.jsonl in the format egomem-eval reads. Resumable.
Runs in the environment with torch, transformers and peft.

    python hf_describe.py --frames ~/data/ego-memory-assistant/describer_eval/frames_test_close \
        --data-dir ~/data/ego-memory-assistant/describer_eval --tag e4b_hf
    python hf_describe.py ... --tag e4b_sft --adapter ~/data/ego-memory-assistant/describer_eval/sft_run/best
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sft_train, also under python -I
from egomem.describer import trim_incomplete_sentence  # noqa: E402  (numpy and PIL only)

from sft_train import MODEL, messages  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Diaries with the Hugging Face E4B, base or SFT.")
    parser.add_argument("--frames", type=Path, required=True, help="output folder of dump_frames.py")
    parser.add_argument("--data-dir", type=Path, required=True, help="writes memories/<clip>/<tag>/ here")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--adapter", type=Path, help="LoRA adapter folder (sft_train.py best/)")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, help="describe at most N new windows")
    args = parser.parse_args()

    rows = [json.loads(l) for l in (args.frames / "index.jsonl").read_text().splitlines() if l.strip()]
    by_clip: dict[str, set[int]] = defaultdict(set)
    for clip in {r["clip"] for r in rows}:
        path = args.data_dir / "memories" / clip / args.tag / "memory.jsonl"
        if path.exists():
            by_clip[clip] = {json.loads(l)["index"] for l in path.read_text().splitlines() if l.strip()}
    pending = [r for r in rows if r["index"] not in by_clip[r["clip"]]]
    todo = pending[: args.limit]
    print(f"{len(rows) - len(pending)} done, {len(pending)} pending, writing {len(todo)}", flush=True)

    processor = AutoProcessor.from_pretrained(MODEL)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
    if args.adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    settings = {"model": MODEL, "adapter": str(args.adapter) if args.adapter else None, "backend": "transformers",
                "dtype": "bfloat16", "temperature": args.temperature, "max_new_tokens": args.max_new_tokens,
                "seed": args.seed, "frames": str(args.frames)}

    for b in range(0, len(todo), args.batch):
        chunk = todo[b:b + args.batch]
        texts = [processor.apply_chat_template(messages(r["prompt"], len(r["frames"]), None),
                                               add_generation_prompt=True, tokenize=False) for r in chunk]
        images = [[Image.open(args.frames / p).convert("RGB") for p in r["frames"]] for r in chunk]
        enc = processor(text=texts, images=images, padding=True, return_tensors="pt").to(model.device)
        torch.manual_seed(args.seed + b)
        t = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**enc, do_sample=args.temperature > 0, temperature=args.temperature,
                                 max_new_tokens=args.max_new_tokens)
        dt = (time.perf_counter() - t) / len(chunk)
        answers = processor.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for r, a in zip(chunk, answers):
            folder = args.data_dir / "memories" / r["clip"] / args.tag
            folder.mkdir(parents=True, exist_ok=True)
            if not (folder / "run.json").exists():
                (folder / "run.json").write_text(json.dumps({"describer": settings}, indent=1))
            text = trim_incomplete_sentence(a.strip()) or "(no description)"
            with (folder / "memory.jsonl").open("a") as fh:
                fh.write(json.dumps({"index": r["index"], "t0": r["t0"], "t1": r["t1"], "text": text,
                                     "latency_s": round(dt, 2), "n_frames": len(r["frames"])}) + "\n")
        print(f"[{b + len(chunk)}/{len(todo)}] {dt:4.1f}s/window  {chunk[0]['clip'][:8]} #{chunk[0]['index']}: "
              f"{answers[0].strip()[:100]}", flush=True)


if __name__ == "__main__":
    main()
