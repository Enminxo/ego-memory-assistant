#!/bin/bash
# Phase 1a (README.md), in order of importance:
#   1a-1 e4b_noex diaries, B with the Gemma reader (seed 0), A0 and the prompt-echo counts;
#   1a-3 B with qwen3.8:27b as reader (seed 0) on all five diaries;
#   1a-2 B with the Gemma reader and option seed 1 on all five diaries.
#
#   setsid nohup bash experiments/describer_eval/run_phase1a.sh > LOG 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."

CONFIG=experiments/describer_eval/config.yaml
QCONFIG=experiments/describer_eval/config_reader_qwen27.yaml
ROOT=$(grep '^data_dir:' "$CONFIG" | awk '{print $2}')
SCENARIOS=$(grep '^scenarios_dir:' "$CONFIG" | awk '{print $2}')
IDS=$(ls "$SCENARIOS" | sed 's/\.json$//')
ANN=$HOME/data/raw/ego4d/v2/annotations
R=$ROOT/results
TAGS="narration e4b g31b e4b_noex"

echo "=== $(date '+%F %T') 1a-1: e4b_noex diaries"
for id in $IDS; do
    uv run egomem-build "$id" --config "$CONFIG" --tag e4b_noex --model gemma4:e4b --prompt-variant no_examples
done

echo "=== $(date '+%F %T') 1a-1: Gemma reader on e4b_noex, seed 0"
uv run egomem-eval $IDS --config "$CONFIG" --tag e4b_noex --seed 0 \
    --out "$R/eval_e4b_noex_seed0.jsonl" > /dev/null
uv run python experiments/describer_eval/reader_accuracy.py --results "$R" --seed 0 > /dev/null

echo "=== $(date '+%F %T') 1a-1: A0 and prompt echo"
for pair in "e4b e4b_noex" "e4b_noex g31b"; do
    set -- $pair
    uv run python experiments/describer_eval/object_precision.py --config "$CONFIG" \
        --narrations "$ANN/narration.json" --taxonomy "$ANN/narration_noun_taxonomy.csv" \
        --tags "$1" "$2" --out "$R/a0_$2_vs_$1" > /dev/null
done
uv run python experiments/describer_eval/prompt_echo.py --config "$CONFIG" \
    --tags narration e4b e4b_noex g31b --out "$R/prompt_echo.json" > /dev/null

echo "=== $(date '+%F %T') 1a-3: qwen3.8:27b reader, seed 0"
Q=$R/reader_qwen27
mkdir -p "$Q"
uv run egomem-eval $IDS --config "$QCONFIG" --no-memory --seed 0 \
    --out "$Q/eval_no-memory_seed0.jsonl" > /dev/null
for tag in $TAGS; do
    echo "    $(date '+%T') $tag"
    uv run egomem-eval $IDS --config "$QCONFIG" --tag "$tag" --seed 0 \
        --out "$Q/eval_${tag}_seed0.jsonl" > /dev/null
done
uv run python experiments/describer_eval/reader_accuracy.py --results "$Q" --seed 0 > /dev/null

echo "=== $(date '+%F %T') 1a-2: Gemma reader, option seed 1"
uv run egomem-eval $IDS --config "$CONFIG" --no-memory --seed 1 \
    --out "$R/eval_no-memory_seed1.jsonl" > /dev/null
for tag in $TAGS; do
    echo "    $(date '+%T') $tag"
    uv run egomem-eval $IDS --config "$CONFIG" --tag "$tag" --seed 1 \
        --out "$R/eval_${tag}_seed1.jsonl" > /dev/null
done
uv run python experiments/describer_eval/reader_accuracy.py --results "$R" --seed 1 > /dev/null

echo "=== $(date '+%F %T') phase 1a done"
