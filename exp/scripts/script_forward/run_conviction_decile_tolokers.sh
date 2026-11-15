#!/bin/bash
# Decile pruning on tolokers alone. Together with run_conviction_decile_pruning.sh
# this covers every manifest dataset.

set -e

MODEL="GeneralSheaf"
NORMALISED="true"
MAP_TYPE="identity"
CONVICTION_SCORE="last"
N_BINS=10

echo "=== score=${CONVICTION_SCORE}  intervention=pruning  dataset=tolokers ==="
python Bochner_code/conviction/conviction_masking_decile.py \
    --model="${MODEL}" \
    --normalised="${NORMALISED}" \
    --map_type="${MAP_TYPE}" \
    --conviction_score="${CONVICTION_SCORE}" \
    --intervention=pruning \
    --n_bins="${N_BINS}" \
    --dataset=tolokers
