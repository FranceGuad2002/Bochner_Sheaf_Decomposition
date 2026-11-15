#!/bin/bash
# Forman profile, from the saved eigenvalues to the paper's 11 tables. CPU only.
#   1-3: harvest the per-fold, per-layer Wasserstein distances from
#        results/forman_eigs/ into one long CSV, aggregate it into mean +/- SE
#        across datasets, and emit the 8 global tables.
#   4:   the 3 per-dataset tables (all five metrics, one dataset each).
# Needs results/forman_eigs/ (the Zenodo archive) for steps 1 and 4.
#
# Run from the repo root:  bash exp/scripts/run_forman_global.sh
set -e

sep() { echo; echo "=============================================================="; echo "$1"; echo "=============================================================="; }

sep "[1/4]  Harvest raw per-fold Wasserstein values -> long CSV"
python Bochner_code/forman_profile/compute_forman_global.py

sep "[2/4]  Aggregate -> per-(config, checkpoint, depth-bin) mean +/- SE"
python Bochner_code/forman_profile/aggregate_forman_global.py

sep "[3/4]  Emit the 8 global LaTeX tables"
python Bochner_code/forman_profile/generate_forman_tables.py

sep "[4/4]  Per-dataset tables: per-fold CSV -> 3 LaTeX tables"
python Bochner_code/forman_profile/make_forman_perdataset_tables.py --from_eigs

sep "All done."
