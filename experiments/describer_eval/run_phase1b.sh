#!/bin/bash
# Phase 1b (README.md), main arm, in order: teacher data (resumes), base diaries in Hugging Face,
# LoRA SFT, student diaries, then B with the Gemma reader (seeds 0 and 1) and the Qwen reader,
# A0 and the prompt-echo counts. Training and Hugging Face diaries run in ~/venvs/enc (torch,
# transformers, peft); the rest in the egomem environment.
#
#   setsid nohup bash experiments/describer_eval/run_phase1b.sh > LOG 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."

CONFIG=experiments/describer_eval/config.yaml
QCONFIG=experiments/describer_eval/config_reader_qwen27.yaml
ROOT=$(grep '^data_dir:' "$CONFIG" | awk '{print $2}')
SCENARIOS=$(grep '^scenarios_dir:' "$CONFIG" | awk '{print $2}')
IDS=$(ls "$SCENARIOS" | sed 's/\.json$//')
ANN=$HOME/data/raw/ego4d/v2/annotations
R=$ROOT/results
E=experiments/describer_eval
PY=$HOME/venvs/enc/bin/python
export HF_HUB_OFFLINE=1

echo "=== $(date '+%F %T') 1b-1: teacher data"
uv run python $E/sft_data.py --config "$CONFIG" --scenarios "$ROOT/scenarios_train" --out "$ROOT/sft"

echo "=== $(date '+%F %T') 1b-3: base diaries (e4b_hf)"
$PY -I $E/hf_describe.py --frames "$ROOT/frames_test_close" --data-dir "$ROOT" --tag e4b_hf

echo "=== $(date '+%F %T') 1b-2: LoRA SFT"
$PY -I $E/sft_train.py --data "$ROOT/sft" --out "$ROOT/sft_run" --no-checkpointing --eval-every 100

echo "=== $(date '+%F %T') 1b-3: student diaries (e4b_sft)"
$PY -I $E/hf_describe.py --frames "$ROOT/frames_test_close" --data-dir "$ROOT" --tag e4b_sft \
    --adapter "$ROOT/sft_run/best"

echo "=== $(date '+%F %T') 1b-3: B, Gemma reader, seed 0"
for tag in e4b_hf e4b_sft; do
    uv run egomem-eval $IDS --config "$CONFIG" --tag "$tag" --seed 0 --out "$R/eval_${tag}_seed0.jsonl" > /dev/null
done
uv run python $E/reader_accuracy.py --results "$R" --seed 0 > /dev/null

echo "=== $(date '+%F %T') 1b-3: A0 and prompt echo"
for pair in "e4b_hf e4b_sft" "e4b_sft g31b"; do
    set -- $pair
    uv run python $E/object_precision.py --config "$CONFIG" \
        --narrations "$ANN/narration.json" --taxonomy "$ANN/narration_noun_taxonomy.csv" \
        --tags "$1" "$2" --out "$R/a0_$2_vs_$1" > /dev/null
done
uv run python $E/prompt_echo.py --config "$CONFIG" \
    --tags narration e4b e4b_noex e4b_hf e4b_sft g31b --out "$R/prompt_echo.json" > /dev/null

echo "=== $(date '+%F %T') 1b-3: B, Qwen reader, seed 0"
for tag in e4b_hf e4b_sft; do
    uv run egomem-eval $IDS --config "$QCONFIG" --tag "$tag" --seed 0 \
        --out "$R/reader_qwen27/eval_${tag}_seed0.jsonl" > /dev/null
done
uv run python $E/reader_accuracy.py --results "$R/reader_qwen27" --seed 0 > /dev/null

echo "=== $(date '+%F %T') 1b-3: B, Gemma reader, seed 1"
for tag in e4b_hf e4b_sft; do
    uv run egomem-eval $IDS --config "$CONFIG" --tag "$tag" --seed 1 --out "$R/eval_${tag}_seed1.jsonl" > /dev/null
done
uv run python $E/reader_accuracy.py --results "$R" --seed 1 > /dev/null

echo "=== $(date '+%F %T') phase 1b done"
