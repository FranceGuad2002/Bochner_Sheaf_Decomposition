#!/bin/bash
# Forward masking sweep: conviction_masking_forward.py --intervention=masking.
# Masks the top-conviction training nodes at increasing budgets, for every
# conviction score and both mask modes (zero / mean feature replacement), over
# every dataset in Bochner_code/conviction/conviction_manifest.txt.
# Pure forward passes on the frozen models.

set -e

MODEL="GeneralSheaf"
NORMALISED="true"
MAP_TYPE="identity"

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
            python Bochner_code/conviction/conviction_masking_forward.py \
                --model="${MODEL}" \
                --normalised="${NORMALISED}" \
                --map_type="${MAP_TYPE}" \
                --conviction_score="${SCORE}" \
                --intervention=masking \
                --mask_mode="${MODE}" \
                "${DS_ARGS[@]}"
        done
    done
done
