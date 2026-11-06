#!/bin/bash
# Phase 1b, second round (README.md): the vision arm, then the filtered teacher, then the
# evaluation of both. All Hugging Face work runs before the Ollama readers, so the training
# does not share memory with the served models.
#
#   setsid nohup bash experiments/describer_eval/run_phase1b2.sh > LOG 2>&1 &
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
TRAIN="--no-checkpointing --eval-every 100 --patience 2"

echo "=== $(date '+%F %T') 1b-V: train the vision arm"
$PY -I $E/sft_train.py --data "$ROOT/sft" --out "$ROOT/sft_run_vis" $TRAIN --vision-lora
echo "=== $(date '+%F %T') 1b-V: diaries (e4b_sft_vis)"
$PY -I $E/hf_describe.py --frames "$ROOT/frames_test_close" --data-dir "$ROOT" --tag e4b_sft_vis \
    --adapter "$ROOT/sft_run_vis/best"

echo "=== $(date '+%F %T') 1b-F: filter the teacher data and train"
uv run python $E/sft_filter.py --data "$ROOT/sft" --out "$ROOT/sft_noneg"
$PY -I $E/sft_train.py --data "$ROOT/sft_noneg" --out "$ROOT/sft_run_noneg" $TRAIN
echo "=== $(date '+%F %T') 1b-F: diaries (e4b_sft_noneg)"
$PY -I $E/hf_describe.py --frames "$ROOT/frames_test_close" --data-dir "$ROOT" --tag e4b_sft_noneg \
    --adapter "$ROOT/sft_run_noneg/best"

TAGS="e4b_sft_vis e4b_sft_noneg"
echo "=== $(date '+%F %T') B, Gemma reader, seed 0"
for tag in $TAGS; do
    uv run egomem-eval $IDS --config "$CONFIG" --tag "$tag" --seed 0 --out "$R/eval_${tag}_seed0.jsonl" > /dev/null
done
uv run python $E/reader_accuracy.py --results "$R" --seed 0 > /dev/null

echo "=== $(date '+%F %T') A0 and prompt echo"
for tag in $TAGS; do
    uv run python $E/object_precision.py --config "$CONFIG" \
        --narrations "$ANN/narration.json" --taxonomy "$ANN/narration_noun_taxonomy.csv" \
        --tags e4b_sft "$tag" --out "$R/a0_${tag}_vs_e4b_sft" > /dev/null
done
uv run python $E/prompt_echo.py --config "$CONFIG" \
    --tags narration e4b e4b_noex e4b_hf e4b_sft e4b_sft_vis e4b_sft_noneg g31b --out "$R/prompt_echo.json" > /dev/null

echo "=== $(date '+%F %T') B, Qwen reader, seed 0"
for tag in $TAGS; do
    uv run egomem-eval $IDS --config "$QCONFIG" --tag "$tag" --seed 0 \
        --out "$R/reader_qwen27/eval_${tag}_seed0.jsonl" > /dev/null
done
uv run python $E/reader_accuracy.py --results "$R/reader_qwen27" --seed 0 > /dev/null

echo "=== $(date '+%F %T') B, Gemma reader, seed 1"
for tag in $TAGS; do
    uv run egomem-eval $IDS --config "$CONFIG" --tag "$tag" --seed 1 --out "$R/eval_${tag}_seed1.jsonl" > /dev/null
done
uv run python $E/reader_accuracy.py --results "$R" --seed 1 > /dev/null

echo "=== $(date '+%F %T') phase 1b second round done"
