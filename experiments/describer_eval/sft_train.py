"""Phase 1b-2: LoRA SFT of gemma-4-E4B-it on the teacher's diary entries (README.md, phase 1b).

Runs in an environment with torch, transformers and peft (not the egomem one): it only reads
the JPEG frames and the JSONL written by sft_data.py. LoRA goes on the attention and MLP
projections of the language model; the vision tower, the audio tower and the projectors stay
frozen. The loss is computed on the answer tokens only.

    python sft_train.py --data ~/data/ego-memory-assistant/describer_eval/sft \
        --out ~/data/ego-memory-assistant/describer_eval/sft_run            # full run
    python sft_train.py --data ... --out ... --smoke 4                       # a few steps, then stop
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL = "google/gemma-4-e4b-it"
# Language model only: q/k/v/o and the MLP of every decoder layer (k/v exist in 24 of the 42).
TARGETS = r"model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
# The planned vision arm (README, phase 1b-0): the same projections of the 16 vision encoder layers too.
VISION_TARGETS = r"model\.vision_tower\.encoder\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.linear"


def messages(prompt: str, n_images: int, target: str | None) -> list[dict]:
    """Text first, then the frames, as in the request the describer sends to the server."""
    msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}] + [{"type": "image"}] * n_images}]
    if target is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
    return msgs


def encode(processor, row: dict, root: Path) -> dict:
    """input_ids, attention and pixel tensors, and labels masked to the answer tokens."""
    images = [Image.open(root / p).convert("RGB") for p in row["frames"]]
    prefix = processor.apply_chat_template(messages(row["prompt"], len(images), None), add_generation_prompt=True,
                                           tokenize=False)
    full = processor.apply_chat_template(messages(row["prompt"], len(images), row["target"]), tokenize=False)
    assert full.startswith(prefix), "the chat template does not extend the prompt with the answer"
    enc = processor(text=full, images=images, return_tensors="pt")
    n_prefix = processor(text=prefix, images=images, return_tensors="pt")["input_ids"].shape[1]
    ids = enc["input_ids"]
    labels = ids.clone()
    labels[:, :n_prefix] = -100
    assert (labels != -100).sum() > 0, "no answer tokens"
    return {**enc, "labels": labels}


def load_rows(data: Path) -> dict[str, list[dict]]:
    rows = [json.loads(l) for l in (data / "data.jsonl").read_text().splitlines() if l.strip()]
    return {s: [r for r in rows if r["split"] == s] for s in ("train", "val")}


def evaluate(model, processor, rows: list[dict], root: Path) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for r in rows:
            batch = {k: v.to(model.device) for k, v in encode(processor, r, root).items()}
            out = model(**batch)
            k = int((batch["labels"] != -100).sum())
            total += float(out.loss) * k
            n += k
    model.train()
    return total / n


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1b-2: LoRA SFT of the small describer.")
    parser.add_argument("--data", type=Path, required=True, help="output folder of sft_data.py")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--accum", type=int, default=8, help="windows per optimizer step")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--eval-every", type=int, default=50, help="optimizer steps between validation runs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", type=int, help="run this many optimizer steps on a few windows, then stop")
    parser.add_argument("--no-checkpointing", action="store_true",
                        help="keep activations instead of recomputing them (faster, more memory)")
    parser.add_argument("--vision-lora", action="store_true",
                        help="also put LoRA on the vision encoder (the planned vision arm)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    rows = load_rows(args.data)
    if args.smoke:
        rows = {"train": rows["train"][: args.smoke * args.accum] or rows["train"], "val": rows["val"][:4] or rows["train"][:2]}
    args.out.mkdir(parents=True, exist_ok=True)

    t = time.time()
    processor = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
    if not args.no_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    targets = f"({TARGETS}|{VISION_TARGETS})" if args.vision_lora else TARGETS
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
                                             target_modules=targets, task_type="CAUSAL_LM"))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()  # LoRA weights in fp32, the frozen model in bf16
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable = {n.split(".lora_")[0] for n, p in model.named_parameters() if p.requires_grad}
    allowed = (".language_model.", ".vision_tower.") if args.vision_lora else (".language_model.",)
    assert all(any(a in n for a in allowed) for n in trainable), f"LoRA reached a module outside {allowed}"
    print(f"loaded in {time.time() - t:.0f}s; trainable {n_train / 1e6:.1f}M params in {len(trainable)} modules; "
          f"train {len(rows['train'])} val {len(rows['val'])} windows", flush=True)

    steps_per_epoch = math.ceil(len(rows["train"]) / args.accum)
    total_steps = args.smoke or steps_per_epoch * args.epochs
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    warmup = max(1, round(0.03 * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))
    log = (args.out / "log.jsonl").open("a")
    best = float("inf")
    v0 = evaluate(model, processor, rows["val"], args.data)
    print(f"step 0 val loss {v0:.4f}", flush=True)
    log.write(json.dumps({"step": 0, "val_loss": v0}) + "\n")
    step, t = 0, time.time()
    model.train()
    while step < total_steps:
        order = rng.sample(rows["train"], len(rows["train"]))
        for i in range(0, len(order), args.accum):
            losses = []
            for r in order[i:i + args.accum]:
                batch = {k: v.to(model.device) for k, v in encode(processor, r, args.data).items()}
                loss = model(**batch).loss / len(order[i:i + args.accum])
                loss.backward()
                losses.append(float(loss.detach()) * len(order[i:i + args.accum]))
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            rec = {"step": step, "train_loss": sum(losses) / len(losses), "lr": sched.get_last_lr()[0],
                   "s_per_step": round((time.time() - t) / step, 2),
                   "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
            if step % args.eval_every == 0 or step == total_steps:
                rec["val_loss"] = evaluate(model, processor, rows["val"], args.data)
                if rec["val_loss"] < best:
                    best = rec["val_loss"]
                    model.save_pretrained(args.out / "best")
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if step >= total_steps:
                break
    print(f"done: best val loss {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
