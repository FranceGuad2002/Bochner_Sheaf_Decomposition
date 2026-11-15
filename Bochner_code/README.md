# Bochner decomposition for cellular sheaves — reproduction guide

This folder contains everything specific to the paper: the analysis
code, the derived data, and every table the paper reports. The models and training loop
it builds on are in the repository root, a fork of
[twitter-research/neural-sheaf-diffusion](https://github.com/twitter-research/neural-sheaf-diffusion)
(see the root [`README.md`](../README.md) for what the fork changes).

The paper has two experimental parts, and the folders follow them:

| Folder | Content |
|---|---|
| `conviction/` | **Conviction**: node importance in a trained sheaf neural network, measured by intervening on the frozen model (Bochner term) |
| `forman_profile/` | **Forman profile**: per-node spectra of the curvature block, compared with a topological reference (Forman term) |
| [`TRAINING_RUNS.md`](TRAINING_RUNS.md) | every trained model the paper uses, with the exact command that produced it |
| `environments/`, `requirements-*.txt` | the environments described below |

## Three levels of reproduction

| Level | What it reproduces | Needs | Time |
|---|---|---|---|
| **[A](#a-regenerate-every-table-in-the-paper)** | all 23 tables in the paper, from the committed per-fold data | Python and 4 packages | about a minute |
| **[B](#b-rerun-the-analysis-from-the-archived-raw-outputs)** | the per-fold data itself, from the archived outputs of the trained models | the Zenodo archive (6.8 GB) and the full environment | hours; a GPU helps for the conviction interventions |
| **[C](#c-retrain-the-models)** | the trained models | the full environment; a GPU for the large datasets | about a minute per small-dataset model on CPU (Cornell, Wisconsin) |

Run every command **from the repository root**, not from this folder.

> **On Windows, first unpack or clone the repository into a short folder such as `C:\nsd`.**
> Some file paths in this repository are long, and Windows by default refuses paths longer
> than 260 characters including the folder they sit in. In a deep folder (for example
> `C:\Users\<name>\Downloads\...`) those files fail to unpack and the scripts stop with
> `FileNotFoundError`. For a clone: `git clone -c core.longpaths=true <url> C:\nsd`.
> Linux and macOS are not affected.

---

## A. Regenerate every table in the paper

No GPU, no training, no download: the per-fold data the tables are computed from is
committed under `*/outputs/`. Any Python ≥ 3.9 works.

```bash
pip install -r Bochner_code/requirements-tables.txt

# Forman profile: 8 global tables + 3 per-dataset tables
python Bochner_code/forman_profile/aggregate_forman_global.py
python Bochner_code/forman_profile/generate_forman_tables.py
python Bochner_code/forman_profile/make_forman_perdataset_tables.py

# Conviction: 12 tables
bash   exp/scripts/run_conviction_summary.sh
python Bochner_code/conviction/make_perdataset_tables.py
python Bochner_code/conviction/conviction_observational_table.py
```

**Expected result:** every regenerated file is byte-identical to the committed one, so
`git status` reports no change. Each table's numbers equal those of the corresponding table
in the paper; the paper's copies differ only in hand-adjusted captions and cell
shading. [Where each table comes from](#where-each-table-comes-from) maps every paper label
to its file.

---

## B. Rerun the analysis from the archived raw outputs

### 1. Environment

| Machine | Setup |
|---|---|
| Linux with an NVIDIA GPU (what the paper used) | `conda env create -f Bochner_code/environments/environment_nsd_exact.yml` (Python 3.9.9, torch 2.5.1 + CUDA 12.1, PyG 2.6.1) |
| CPU only | the upstream environment plus the analysis libraries:<br>`conda env create -f environment_cpu.yml`<br>`conda activate nsd`<br>`pip install -r Bochner_code/requirements-analysis.txt` |

Both create an environment called `nsd`; activate it before running anything. Training
(level C) reports to Weights & Biases; without an account, run `export WANDB_MODE=disabled`
first.

### 2. Data

The trained models' raw outputs (`results/`, about 29,000 files) are too large for git and
are archived on Zenodo: **https://zenodo.org/records/23080428**

| File | Size | Unpacks to | Needed for |
|---|---|---|---|
| `results_forman_eigs.tar` | 5.5 GB | `results/forman_eigs/` | Forman profile: curvature eigenvalues per fold, layer and checkpoint |
| `results_conviction.tar` | 0.9 GB | `results/model_weights/`, `results/node_*/` (and an unused `results/scalar_curvature_FMG/`) | Conviction: the frozen models, the conviction scores, predictions, confidences and saliencies |
| `results_identity_sheaf.tar` | 0.4 GB | `results/*/<dataset>/IdentitySheaf/` | The identity-sheaf control of the conviction analysis: the same outputs for the 8 identity-sheaf models |

Download the three files into the **repository root** and unpack them there. The archives
store paths as `results/...`, so they land in the right place, and they do not overlap:

```bash
sha256sum -c <<'EOF'
d81ded5226e1cbbff3cfe2de9859fc207304a39b709c188805290e8e51f83fcc  results_forman_eigs.tar
65dcc3d149f175549e4960a95a5ab82d844b33f12a13d55b0e834755e0540629  results_conviction.tar
8ce805359cde46673a32e8d710d2dd8812c2597b88694a1b5858ad9b3546244f  results_identity_sheaf.tar
EOF
tar -xf results_forman_eigs.tar
tar -xf results_conviction.tar
tar -xf results_identity_sheaf.tar
```

You should now have `results/forman_eigs/` and `results/model_weights/` next to `exp/` and
`models/`, each with one sub-folder per dataset (`results/model_weights/cornell/…`).

### 3. Run

```bash
# Forman profile (CPU): eigenvalues -> per-fold CSVs -> the 8 global and 3 per-dataset tables
bash exp/scripts/run_forman_global.sh

# Conviction
bash exp/scripts/run_all_conviction_interventions.sh   # masking, gating, pruning on the frozen models
bash exp/scripts/run_conviction_heatmap.sh             # observational correlations (CPU)
bash exp/scripts/run_conviction_summary.sh             # per-dataset CSVs -> tables (seconds)
python Bochner_code/conviction/make_perdataset_tables.py

# Identity-sheaf control: the same pipeline for --model=IdentitySheaf
python Bochner_code/conviction/conviction_heatmap.py --model=IdentitySheaf --normalised=true --map_type=identity
python Bochner_code/conviction/conviction_masking_forward.py --model=IdentitySheaf --normalised=true \
    --map_type=identity --conviction_score=last --intervention=masking --mask_mode=zero
python Bochner_code/conviction/conviction_observational_table.py --configs=IdentitySheaf:true \
    --tag=observational_summary_identity
python Bochner_code/conviction/conviction_summary_tables.py --score=last --model=IdentitySheaf \
    --normalised=true --exclude=cornell,wisconsin --tag_suffix=_identity
```

These rewrite the committed per-fold CSVs and tables. Which trained models the conviction
scripts load is listed in [`conviction/conviction_manifest.txt`](conviction/conviction_manifest.txt),
one line per model.

---

## C. Retrain the models

[`TRAINING_RUNS.md`](TRAINING_RUNS.md) lists all 53 trained models (10 for conviction, 43
for the Forman profile), what each is used for, and the exact command that produced it. The
same commands are collected in scripts, which write into `results/`; the third trains the
8 identity-sheaf control models of the conviction analysis:

```bash
export WANDB_MODE=disabled
bash exp/scripts/paper/train_conviction_models.sh
bash exp/scripts/paper/train_forman_models.sh
bash exp/scripts/paper/train_identity_sheaf_models.sh
```

Then run level B, step 3. Retrained models are not bit-identical to the archived ones, since
GPU arithmetic and library versions differ, but they agree within the spread across folds.

---

## Where each table comes from

**Forman profile.** All tables are written to `forman_profile/outputs/summary_tables/`.

| Paper label | File | Script |
|---|---|---|
| `tab:global_general_norm` (body and appendix) | `table1_general_norm.tex` | `generate_forman_tables.py` |
| `tab:global_general_unnorm` | `table2_general_unnorm.tex` | 〃 |
| `tab:global_joint_unnorm` | `table3_joint_unnorm.tex` | 〃 |
| `tab:global_joint_norm` | `table4_joint_norm.tex` | 〃 |
| `tab:global_general_other_norm` | `table5_general_other_norm.tex` | 〃 |
| `tab:global_general_other_unnorm` | `table6_general_other_unnorm.tex` | 〃 |
| `tab:global_cross_a` | `table7_cross_a.tex` | 〃 |
| `tab:global_cross_b` | `table8_cross_b.tex` | 〃 |
| `tab:perds_rm_norm` | `perds_rm_norm.tex` | `make_forman_perdataset_tables.py` |
| `tab:perds_citeseer_unnorm` | `perds_citeseer_unnorm.tex` | 〃 |
| `tab:perds_questions_joint_unnorm` | `perds_questions_joint_unnorm.tex` | 〃 |

**Conviction.** All tables are written to `conviction/outputs/summary_tables/`.

| Paper label | File | Script |
|---|---|---|
| `tab:conv_fwd_masking-zero_last` (body) | `forward_masking-zero_…conv-last.tex` | `conviction_summary_tables.py` |
| `tab:conv_fwd_masking-mean_last` | `forward_masking-mean_…conv-last.tex` | 〃 |
| `tab:conv_fwd_masking-zero_avg_z` | `forward_masking-zero_…conv-avg_z.tex` | 〃 |
| `tab:conv_fwd_gating-hard_last` | `forward_gating-hard_…conv-last.tex` | 〃 |
| `tab:conv_soft_last` | `soft_gating_…conv-last.tex` | 〃 |
| `tab:conv_fwd_pruning_last` | `forward_pruning_…conv-last.tex` | 〃 |
| `tab:conv_dec_masking-zero_last` | `decile_masking-zero_…conv-last.tex` | 〃 |
| `tab:conv_prune_edges_last` | `prune_edges_…conv-last.tex` | 〃 |
| `tab:conv_prune_evd_last` | `prune_edges_vs_drop_…conv-last.tex` | 〃 |
| `tab:conv_perds_high` | `perdataset_masking-zero_high.tex` | `make_perdataset_tables.py` |
| `tab:conv_perds_low` | `perdataset_masking-zero_low.tex` | 〃 |
| `tab:conv_observational` | `observational_summary.tex` | `conviction_observational_table.py` |

The same folder holds 11 further tables the paper does not cite: the robustness variants
(the `conv-avg` score, `masking-mean` and other deciles, `prune_components`). The paper's
three figures are theory diagrams and are not produced by code.

---

## Good to know

- **Cornell and Wisconsin are left out of the conviction aggregates**, because their training
  pools are too small for a percentage budget to mean anything (1% is a single node). The
  `--exclude=cornell,wisconsin` flag that does this is set in
  `exp/scripts/run_conviction_summary.sh`, not in the Python script's defaults, so call the
  wrapper rather than the script directly.
- **Windows, levels B and C.** Linux (or WSL) is recommended: the upstream dependency
  `torch-householder` compiles a C++ extension, which on Windows needs the Visual Studio
  C++ Build Tools. Keep the repository in a short folder (see the note above).
- **Not included:** `wandb/` (training logs). No credential is stored anywhere in the repository.
- The joint sheaf diffusion model is not implemented in this public repository; see
  Hernandez Caralt et al. (2024) for details on how to build it. Its archived outputs are on
  Zenodo, so levels A and B still cover the tables that use it.
