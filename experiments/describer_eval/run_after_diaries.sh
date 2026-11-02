#!/bin/bash
# Wait for run_diaries.sh to finish, then compute A0 and run B, in that order.
#
#   setsid nohup bash experiments/describer_eval/run_after_diaries.sh DIARIES_LOG > LOG 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."

DIARIES_LOG=$1
CONFIG=experiments/describer_eval/config.yaml
ROOT=$(grep '^data_dir:' "$CONFIG" | awk '{print $2}')
ANN=$HOME/data/raw/ego4d/v2/annotations

until grep -q "all scenarios done" "$DIARIES_LOG"; do
    if ! pgrep -f run_diaries.sh > /dev/null; then
        echo "run_diaries.sh is not running and did not finish; stopping" >&2
        exit 1
    fi
    sleep 300
done

echo "=== $(date '+%F %T') A0"
uv run python experiments/describer_eval/object_precision.py --config "$CONFIG" \
    --narrations "$ANN/narration.json" --taxonomy "$ANN/narration_noun_taxonomy.csv" \
    --out "$ROOT/results/a0" > /dev/null
echo "=== $(date '+%F %T') B"
bash experiments/describer_eval/run_eval.sh
echo "=== $(date '+%F %T') A0 and B done"
