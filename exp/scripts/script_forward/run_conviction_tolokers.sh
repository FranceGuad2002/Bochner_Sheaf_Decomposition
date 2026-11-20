#!/bin/bash
# Forward pruning on tolokers alone (the largest graph, and the slowest pruning
# run). Together with run_conviction_pruning.sh this covers every manifest dataset.

set -e

MODEL="GeneralSheaf"
NORMALISED="true"
MAP_TYPE="identity"
CONVICTION_SCORE="last"

echo "=== score=${CONVICTION_SCORE}  intervention=pruning  dataset=tolokers ==="
python src/conviction/conviction_masking_forward.py \
    --model="${MODEL}" \
    --normalised="${NORMALISED}" \
    --map_type="${MAP_TYPE}" \
    --conviction_score="${CONVICTION_SCORE}" \
    --intervention=pruning \
    --dataset=tolokers
