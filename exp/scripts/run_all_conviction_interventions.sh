#!/bin/bash
# Run all 8 conviction intervention sweeps (script_forward/) one after another.
# Needs the trained models under results/model_weights/ and a GPU for reasonable
# run times. Run from the repo root, with the nsd environment active:
#     bash exp/scripts/run_all_conviction_interventions.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/script_forward"

SCRIPTS=(
    run_conviction_masking.sh
    run_conviction_gating.sh
    run_conviction_tolokers.sh
    run_conviction_pruning.sh
    run_conviction_decile_masking.sh
    run_conviction_decile_gating.sh
    run_conviction_decile_tolokers.sh
    run_conviction_decile_pruning.sh
)

for script in "${SCRIPTS[@]}"; do
    echo "=== ${script} ==="
    bash "$SCRIPT_DIR/$script"
done
echo "Done."
