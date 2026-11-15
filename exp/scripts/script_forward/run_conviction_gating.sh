#!/bin/bash
# Forward gating sweep: conviction_masking_forward.py --intervention=gating.
# hard = budget sweep, like masking; soft = the whole training pool attenuated,
# one point per fold. Pure forward passes; every manifest dataset.

set -e

MODEL="GeneralSheaf"
NORMALISED="true"
MAP_TYPE="identity"

GATE_MODES=("hard" "soft")
CONVICTION_SCORES=("last")
DATASETS=()          # empty = every dataset in Bochner_code/conviction/conviction_manifest.txt

if [ ${#DATASETS[@]} -eq 0 ]; then
    DATASETS=("")
fi

for SCORE in "${CONVICTION_SCORES[@]}"; do
    for MODE in "${GATE_MODES[@]}"; do
        for DS in "${DATASETS[@]}"; do
            DS_ARGS=()
            if [ -n "${DS}" ]; then
                DS_ARGS=("--dataset=${DS}")
            fi

            echo "=== score=${SCORE}  intervention=gating  gate_mode=${MODE}  dataset=${DS:-ALL} ==="
            python Bochner_code/conviction/conviction_masking_forward.py \
                --model="${MODEL}" \
                --normalised="${NORMALISED}" \
                --map_type="${MAP_TYPE}" \
                --conviction_score="${SCORE}" \
                --intervention=gating \
                --gate_mode="${MODE}" \
                "${DS_ARGS[@]}"
        done
    done
done
