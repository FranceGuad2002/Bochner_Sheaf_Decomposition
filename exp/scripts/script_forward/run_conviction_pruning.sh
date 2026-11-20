#!/bin/bash
# Forward pruning: conviction_masking_forward.py --intervention=pruning, every
# manifest dataset except tolokers (see run_conviction_tolokers.sh). Pruning
# rebuilds the model per evaluation, so this is much slower than masking/gating.

set -e

MODEL="GeneralSheaf"
NORMALISED="true"
MAP_TYPE="identity"
CONVICTION_SCORE="last"

DATASETS=("amazon_ratings" "citeseer" "cora" "cornell" "minesweeper" "pubmed" "questions" "roman_empire" "wisconsin")

for DS in "${DATASETS[@]}"; do
    echo "=== score=${CONVICTION_SCORE}  intervention=pruning  dataset=${DS} ==="
    python src/conviction/conviction_masking_forward.py \
        --model="${MODEL}" \
        --normalised="${NORMALISED}" \
        --map_type="${MAP_TYPE}" \
        --conviction_score="${CONVICTION_SCORE}" \
        --intervention=pruning \
        --dataset="${DS}"
done
