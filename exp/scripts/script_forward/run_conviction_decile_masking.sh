#!/bin/bash
# Decile masking: conviction_masking_decile.py --intervention=masking. Ranks the
# training pool by conviction, splits it into N_BINS bins, and masks one bin at a
# time, for every conviction score and both mask modes. Every manifest dataset.

set -e

MODEL="GeneralSheaf"
NORMALISED="true"
MAP_TYPE="identity"
N_BINS=10

MASK_MODES=("zero" "mean")
CONVICTION_SCORES=("last" "avg" "avg_z")
DATASETS=()          # empty = every dataset in Bochner_code/conviction/conviction_manifest.txt

if [ ${#DATASETS[@]} -eq 0 ]; then
    DATASETS=("")
fi

for SCORE in "${CONVICTION_SCORES[@]}"; do
    for MODE in "${MASK_MODES[@]}"; do
        for DS in "${DATASETS[@]}"; do
            DS_ARGS=()
            if [ -n "${DS}" ]; then
                DS_ARGS=("--dataset=${DS}")
            fi

            echo "=== score=${SCORE}  intervention=masking  mask_mode=${MODE}  dataset=${DS:-ALL} ==="
            python Bochner_code/conviction/conviction_masking_decile.py \
                --model="${MODEL}" \
                --normalised="${NORMALISED}" \
                --map_type="${MAP_TYPE}" \
                --conviction_score="${SCORE}" \
                --intervention=masking \
                --mask_mode="${MODE}" \
                --n_bins="${N_BINS}" \
                "${DS_ARGS[@]}"
        done
    done
done
