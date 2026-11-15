#!/bin/bash

# Conviction observational analysis: per-dataset heatmaps and conviction-vs-degree
# scatters, plus the per-fold CSVs those two are averaged from, and finally the
# cross-dataset table built out of them (tab:conv_observational).
#
# Run from the repo root:
#     bash exp/scripts/run_conviction_heatmap.sh
#
# One invocation of conviction_heatmap.py per configuration. Each writes, for
# every dataset it discovers:
#     Bochner_code/conviction/outputs/observational_figures/conviction_heatmap/<model>/normalised_<x>/*.pdf
#     Bochner_code/conviction/outputs/observational_figures/conviction_scatter/<model>/normalised_<x>/*.pdf
#     Bochner_code/conviction/outputs/per_dataset_tables/observational/<model>/normalised_<x>/*_observational_folds.csv
# The PDFs overwrite the existing ones under identical filenames; the CSVs are
# the new part, holding the Spearman/Cohen's-d values one row per fold, before
# the fold averaging the heatmap does. conviction_observational_table.py then
# aggregates them the same way the intervention tables are aggregated: mean over
# folds within a dataset, mean +/- SE across datasets.
#
# CPU only: the node norms are read from results/node_norms/ and no model is
# instantiated. Run time is dominated by networkx betweenness (k=500 sampled
# sources per graph, 10 graphs per configuration).

set -euo pipefail

# ── analysis config ────────────────────────────────────────────────────────────
# One "model:normalised" pair per configuration to analyse. Keep this list and
# the --configs flag passed to conviction_observational_table.py below in sync,
# otherwise the table will be built from fewer configurations than were computed.
CONFIGS=("GeneralSheaf:true")
MAP_TYPE="identity"         # for Joint models only: identity | general | diag | bundle

# Datasets excluded from the cross-dataset table. The per-dataset heatmaps and
# scatters are still produced for them; only the aggregation drops them.
EXCLUDE_DATASETS="cornell,wisconsin"

# Checkpoint columns, and the diffusion layer the table is read at. epoch-0 is
# the untrained control: it separates what the architecture produces from what
# training produces. "last" is X_L, the layer the conviction score is read from.
CHECKPOINTS="epoch-0,best-epoch"
LAYER="last"

OUT_DIR="Bochner_code/conviction/outputs/summary_tables"

# ── 1. per-dataset heatmaps, scatters and per-fold CSVs ────────────────────────
for CONFIG in "${CONFIGS[@]}"; do
    MODEL="${CONFIG%%:*}"
    NORMALISED="${CONFIG##*:}"
    echo
    echo "=============================================================="
    echo "  conviction_heatmap: model=${MODEL}  normalised=${NORMALISED}"
    echo "=============================================================="
    python Bochner_code/conviction/conviction_heatmap.py \
        --model="${MODEL}" \
        --normalised="${NORMALISED}" \
        --map_type="${MAP_TYPE}"
done

# ── 2. cross-dataset observational table ───────────────────────────────────────
echo
echo "=============================================================="
echo "  conviction_observational_table"
echo "=============================================================="
python Bochner_code/conviction/conviction_observational_table.py \
    --configs="$(IFS=,; echo "${CONFIGS[*]}")" \
    --checkpoints="${CHECKPOINTS}" \
    --layer="${LAYER}" \
    --exclude="${EXCLUDE_DATASETS}" \
    --out_dir="${OUT_DIR}"

echo
echo "Done."
echo "  Heatmaps/scatters: Bochner_code/conviction/outputs/observational_figures/"
echo "  Per-fold CSVs:     Bochner_code/conviction/outputs/per_dataset_tables/observational/"
echo "  Table:             ${OUT_DIR}/observational_summary.tex"
