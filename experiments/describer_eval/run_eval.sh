#!/bin/bash
# B: score the reader on QAEgo4D test_close with each diary and with no memory.
# Run after run_diaries.sh has finished (the reader and the large describer share the GPU).
#
#   setsid nohup bash experiments/describer_eval/run_eval.sh > LOG 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."

CONFIG=experiments/describer_eval/config.yaml
ROOT=$(grep '^data_dir:' "$CONFIG" | awk '{print $2}')
SCENARIOS=$(grep '^scenarios_dir:' "$CONFIG" | awk '{print $2}')
IDS=$(ls "$SCENARIOS" | sed 's/\.json$//')
mkdir -p "$ROOT/results"

echo "=== $(date '+%F %T') no-memory"
uv run egomem-eval $IDS --config "$CONFIG" --no-memory --seed 0 \
    --out "$ROOT/results/eval_no-memory_seed0.jsonl" > /dev/null
for tag in narration e4b g31b; do
    echo "=== $(date '+%F %T') $tag"
    uv run egomem-eval $IDS --config "$CONFIG" --tag "$tag" --seed 0 \
        --out "$ROOT/results/eval_${tag}_seed0.jsonl" > /dev/null
done
uv run python experiments/describer_eval/reader_accuracy.py --results "$ROOT/results"
echo "=== $(date '+%F %T') done"
