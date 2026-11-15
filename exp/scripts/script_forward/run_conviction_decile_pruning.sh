#!/bin/bash
# Decile pruning: conviction_masking_decile.py --intervention=pruning, every
# manifest dataset except tolokers (see run_conviction_decile_tolokers.sh).
# Pruning rebuilds the model per bin, so this is much slower than masking/gating.

set -e

MODEL="GeneralSheaf"
NORMALISED="true"
MAP_TYPE="identity"
CONVICTION_SCORE="last"
N_BINS=10

DATASETS=("amazon_ratings" "citeseer" "cora" "cornell" "minesweeper" "pubmed" "questions" "roman_empire" "wisconsin")

for DS in "${DATASETS[@]}"; do
    echo "=== score=${CONVICTION_SCORE}  intervention=pruning  dataset=${DS} ==="
    python Bochner_code/conviction/conviction_masking_decile.py \
        --model="${MODEL}" \
        --normalised="${NORMALISED}" \
        --map_type="${MAP_TYPE}" \
        --conviction_score="${CONVICTION_SCORE}" \
        --intervention=pruning \
        --n_bins="${N_BINS}" \
        --dataset="${DS}"
done
