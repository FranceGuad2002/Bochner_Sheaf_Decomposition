#!/bin/bash
# Cross-dataset summary tables for the conviction experiments — one invocation
# of conviction_summary_tables.py per conviction score (last, avg, avg_z).
# conviction_summary_tables.py aggregates every intervention it finds under a
# given score, so only the score is looped over here.
#
# Reads only the committed per-fold CSVs under
# src/conviction/outputs/per_dataset_tables/ — no
# model, no GPU, finishes in seconds. Run from the repo root:
#     bash exp/scripts/run_conviction_summary.sh
#
# Gating and pruning were swept at score=last only, so the avg and avg_z
# summary tables contain masking rows only — expected, not a bug.
#
# Output goes to src/conviction/outputs/summary_tables/
# — a flat folder holding exactly the table fragments of the paper.
#
# --exclude is what makes these the 8-dataset aggregation the paper reports.
# Running conviction_summary_tables.py without it aggregates all 10 datasets
# and will NOT reproduce the published numbers.

set -e

MODEL="GeneralSheaf"
NORMALISED="true"
SCORES=("last" "avg" "avg_z")

# Datasets excluded from the aggregation: their training pools are so small
# that a 1% budget is a single node and a decile bin is under ten nodes.
EXCLUDE_DATASETS="cornell,wisconsin"

OUT_DIR="src/conviction/outputs/summary_tables"

# The script prints non-ASCII characters; without this a Windows console
# (cp1252) aborts with UnicodeEncodeError.
export PYTHONUTF8=1

for SCORE in "${SCORES[@]}"; do
    echo "=== summarising score=${SCORE} ==="
    python src/conviction/conviction_summary_tables.py \
        --score="${SCORE}" \
        --model="${MODEL}" \
        --normalised="${NORMALISED}" \
        --exclude="${EXCLUDE_DATASETS}" \
        --out_dir="${OUT_DIR}"
done

echo
echo "Done. Tables in: ${OUT_DIR}"
