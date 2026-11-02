#!/bin/bash
# Write the e4b and g31b diaries of every scenario, clip by clip, so that A0 can be
# read on the first clips while the rest run. Resumable: egomem-build skips the
# windows already in each memory.jsonl, so relaunching continues where it stopped.
#
#   setsid nohup bash experiments/describer_eval/run_diaries.sh > LOG 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."

CONFIG=experiments/describer_eval/config.yaml
SCENARIOS=$(grep '^scenarios_dir:' "$CONFIG" | awk '{print $2}')

for path in "$SCENARIOS"/*.json; do
    id=$(basename "$path" .json)
    echo "=== $(date '+%F %T') $id"
    uv run egomem-build "$id" --config "$CONFIG" --tag e4b --model gemma4:e4b
    uv run egomem-build "$id" --config "$CONFIG" --tag g31b --model gemma4:31b
done
echo "=== $(date '+%F %T') all scenarios done"
