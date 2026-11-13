#!/bin/bash
# Test A (Neon, six diaries), then test B (QAEgo4D test_close at half size) and
# its scoring, as pre-registered in run.py. Resumable: run again to continue.
#
#   experiments/resolution_compare/run_all.sh > ~/data/ego-memory-assistant/resolution_compare/run_all.log 2>&1
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN="uv run python experiments/resolution_compare/run.py"

for seed in 0 1; do
    for width in 1600 720 360; do
        echo "=== $(date '+%F %T') neon w$width s$seed"
        $RUN neon --width "$width" --seed "$seed"
    done
done

echo "=== $(date '+%F %T') ego4d g31b_half"
$RUN ego4d --scale 0.5 --seed 0 --tag g31b_half

# Score with the reader settings of configs/demo.yaml, pointed at the
# describer_eval data, exactly as g31b was scored (option-order seeds 0 and 1).
ROOT="$HOME/data/ego-memory-assistant/describer_eval"
CONFIG="$HOME/data/ego-memory-assistant/resolution_compare/eval_config.yaml"
sed -e "s|^scenarios_dir:.*|scenarios_dir: $ROOT/scenarios|" \
    -e "s|^data_dir:.*|data_dir: $ROOT|" configs/demo.yaml > "$CONFIG"
IDS=$(ls "$ROOT/scenarios" | sed 's/\.json$//')
for seed in 0 1; do
    echo "=== $(date '+%F %T') eval g31b_half seed $seed"
    uv run egomem-eval $IDS --config "$CONFIG" --tag g31b_half --seed "$seed" \
        --out "$ROOT/results/eval_g31b_half_seed${seed}.jsonl" > /dev/null
done
echo "=== $(date '+%F %T') all done"
