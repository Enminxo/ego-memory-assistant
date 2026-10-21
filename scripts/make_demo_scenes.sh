#!/bin/bash
# Build the four demo scenarios from Ego4D + QAEgo4D.
# Paths are those of the DGX Spark; point them to your own copies.
set -euo pipefail
cd "$(dirname "$0")/.."

EGO4D=${EGO4D:-$HOME/data/raw/ego4d}
QAEGO4D=${QAEGO4D:-$HOME/projects/egomem/third_party/groundvqa_data/data/unified}

uv run egomem-scene \
    --spec scripts/demo_scenes.yaml \
    --ego4d-videos "$EGO4D/tasks/nlq/v2/video_540ss" \
    --nlq "$EGO4D/v2/annotations/nlq_train.json" "$EGO4D/v2/annotations/nlq_val.json" \
    --qaego4d "$QAEGO4D"
