#!/bin/bash
# Decile gating: conviction_masking_decile.py --intervention=gating. Decile
# gating is always hard (soft gating attenuates the whole pool, so there is
# nothing to bin), and the script has no --gate_mode flag: do not add one.

set -e

MODEL="GeneralSheaf"
NORMALISED="true"
MAP_TYPE="identity"
N_BINS=10

CONVICTION_SCORES=("last")
DATASETS=()          # empty = every dataset in src/conviction/conviction_manifest.txt

if [ ${#DATASETS[@]} -eq 0 ]; then
    DATASETS=("")
fi

for SCORE in "${CONVICTION_SCORES[@]}"; do
    for DS in "${DATASETS[@]}"; do
        DS_ARGS=()
        if [ -n "${DS}" ]; then
            DS_ARGS=("--dataset=${DS}")
        fi

        echo "=== score=${SCORE}  intervention=gating  dataset=${DS:-ALL} ==="
        python src/conviction/conviction_masking_decile.py \
            --model="${MODEL}" \
            --normalised="${NORMALISED}" \
            --map_type="${MAP_TYPE}" \
            --conviction_score="${SCORE}" \
            --intervention=gating \
            --n_bins="${N_BINS}" \
            "${DS_ARGS[@]}"
    done
done
