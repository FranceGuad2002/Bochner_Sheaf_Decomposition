# Bochner decomposition: how to use this code

This folder holds everything that is specific to the paper: the analysis code, the derived
data, and the tables. The models and the training loop are in the root of the repository,
which is a fork of
[Neural Sheaf Diffusion](https://github.com/twitter-research/neural-sheaf-diffusion); the
[main README](../README.md) lists what we changed there.

The two experiments of the paper each have their own folder:

| Folder | What it contains |
|---|---|
| `conviction/` | the conviction experiment: how much a trained sheaf network relies on each node, measured by intervening on the frozen model (the Bochner term) |
| `forman_profile/` | the Forman profile experiment: the spectra of the curvature block at every node, compared with a topological reference (the Forman term) |
| [`TRAINING_RUNS.md`](TRAINING_RUNS.md) | the models the paper uses, and how to train them |
| `environments/`, `requirements-*.txt` | the environments |

You can use the code in three ways, depending on how far back you want to start:

1. **Regenerate the tables** from the data committed in the repository.
2. **Rerun the analysis** from the outputs of the trained models, which are on Zenodo.
3. **Retrain the models** yourself.

Run every command from the root of the repository, not from this folder.

On Windows, clone the repository into a short folder such as `C:\nsd`
(`git clone -c core.longpaths=true <url> C:\nsd`). Some paths in the repository are long,
and Windows refuses paths over 260 characters by default, so in a deep folder the scripts
stop with `FileNotFoundError`. Linux and macOS are not affected.

---

## 1. Regenerate the tables

The per-fold data the tables are computed from is committed under `*/outputs/`, so all you
need is Python 3.9 or later and four packages:

```bash
pip install -r Bochner_code/requirements-tables.txt

# Forman profile: 8 global tables and 3 per-dataset tables
python Bochner_code/forman_profile/aggregate_forman_global.py
python Bochner_code/forman_profile/generate_forman_tables.py
python Bochner_code/forman_profile/make_forman_perdataset_tables.py

# Conviction: 12 tables
bash   exp/scripts/run_conviction_summary.sh
python Bochner_code/conviction/make_perdataset_tables.py
python Bochner_code/conviction/conviction_observational_table.py
```

The files you get are identical to the committed ones, so `git status` shows no change. The
numbers are those of the paper; only the captions and the cell shading of the paper's
copies were adjusted by hand. [Where each table comes from](#where-each-table-comes-from)
tells you which file is which table.

---

## 2. Rerun the analysis

### The environment

| Setup | Command |
|---|---|
| the environment we used | `conda env create -f Bochner_code/environments/environment_nsd_exact.yml` (Python 3.9.9, torch 2.5.1 with CUDA 12.1, PyG 2.6.1) |
| without CUDA | the upstream environment plus the analysis libraries:<br>`conda env create -f environment_cpu.yml`<br>`conda activate nsd`<br>`pip install -r Bochner_code/requirements-analysis.txt` |

Both create an environment called `nsd`. Training logs go to Weights & Biases; without an
account, run `export WANDB_MODE=disabled` first.

### The data

The raw outputs of the trained models (about 29,000 files) are too large for git, so they
are on Zenodo: https://doi.org/10.5281/zenodo.23080428

| File | Size | Unpacks to | What it is for |
|---|---|---|---|
| `results_forman_eigs.tar` | 5.5 GB | `results/forman_eigs/` | Forman profile: the curvature eigenvalues for every fold, layer and checkpoint |
| `results_conviction.tar` | 0.9 GB | `results/model_weights/`, `results/node_*/` | conviction: the frozen models, and the scores, predictions, confidences and saliencies |
| `results_identity_sheaf.tar` | 0.4 GB | `results/*/<dataset>/IdentitySheaf/` | the same outputs for the eight identity-sheaf models of the control experiment |

Download the three files into the root of the repository. The archives store their paths as
`results/...` and don't overlap, so they unpack to the right place:

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
`models/`, each with one folder per dataset.

### Running the analysis

```bash
# Forman profile: eigenvalues -> per-fold CSVs -> the global and per-dataset tables
bash exp/scripts/run_forman_global.sh

# Conviction
bash exp/scripts/run_all_conviction_interventions.sh   # masking, gating and pruning on the frozen models
bash exp/scripts/run_conviction_heatmap.sh             # observational correlations
bash exp/scripts/run_conviction_summary.sh             # per-dataset CSVs -> tables
python Bochner_code/conviction/make_perdataset_tables.py

# The identity-sheaf control: the same pipeline for --model=IdentitySheaf
python Bochner_code/conviction/conviction_heatmap.py --model=IdentitySheaf --normalised=true --map_type=identity
python Bochner_code/conviction/conviction_masking_forward.py --model=IdentitySheaf --normalised=true \
    --map_type=identity --conviction_score=last --intervention=masking --mask_mode=zero
python Bochner_code/conviction/conviction_observational_table.py --configs=IdentitySheaf:true \
    --tag=observational_summary_identity
python Bochner_code/conviction/conviction_summary_tables.py --score=last --model=IdentitySheaf \
    --normalised=true --exclude=cornell,wisconsin --tag_suffix=_identity
```

This rewrites the committed CSVs and tables. Which trained models the conviction scripts
load is listed in [`conviction/conviction_manifest.txt`](conviction/conviction_manifest.txt).

---

## 3. Retrain the models

[`TRAINING_RUNS.md`](TRAINING_RUNS.md) describes the models and the settings of each. The
commands are collected in three scripts that write into `results/`:

```bash
export WANDB_MODE=disabled
bash exp/scripts/paper/train_conviction_models.sh
bash exp/scripts/paper/train_forman_models.sh
bash exp/scripts/paper/train_identity_sheaf_models.sh
```

Then go on with the analysis in step 2. A retrained model is not bit-identical to the
archived one, since GPU arithmetic and library versions differ, but they agree within the
spread across folds.

---

## Where each table comes from

### Forman profile

The tables are written to `forman_profile/outputs/summary_tables/`.

| Table in the paper | File | Script |
|---|---|---|
| `tab:global_general_norm` | `table1_general_norm.tex` | `generate_forman_tables.py` |
| `tab:global_general_unnorm` | `table2_general_unnorm.tex` | `generate_forman_tables.py` |
| `tab:global_joint_unnorm` | `table3_joint_unnorm.tex` | `generate_forman_tables.py` |
| `tab:global_joint_norm` | `table4_joint_norm.tex` | `generate_forman_tables.py` |
| `tab:global_general_other_norm` | `table5_general_other_norm.tex` | `generate_forman_tables.py` |
| `tab:global_general_other_unnorm` | `table6_general_other_unnorm.tex` | `generate_forman_tables.py` |
| `tab:global_cross_a` | `table7_cross_a.tex` | `generate_forman_tables.py` |
| `tab:global_cross_b` | `table8_cross_b.tex` | `generate_forman_tables.py` |
| `tab:perds_rm_norm` | `perds_rm_norm.tex` | `make_forman_perdataset_tables.py` |
| `tab:perds_citeseer_unnorm` | `perds_citeseer_unnorm.tex` | `make_forman_perdataset_tables.py` |
| `tab:perds_questions_joint_unnorm` | `perds_questions_joint_unnorm.tex` | `make_forman_perdataset_tables.py` |

### Conviction

The tables are written to `conviction/outputs/summary_tables/`.

| Table in the paper | File | Script |
|---|---|---|
| `tab:conv_fwd_masking-zero_last` | `forward_masking-zero_…conv-last.tex` | `conviction_summary_tables.py` |
| `tab:conv_fwd_masking-mean_last` | `forward_masking-mean_…conv-last.tex` | `conviction_summary_tables.py` |
| `tab:conv_fwd_masking-zero_avg_z` | `forward_masking-zero_…conv-avg_z.tex` | `conviction_summary_tables.py` |
| `tab:conv_fwd_gating-hard_last` | `forward_gating-hard_…conv-last.tex` | `conviction_summary_tables.py` |
| `tab:conv_soft_last` | `soft_gating_…conv-last.tex` | `conviction_summary_tables.py` |
| `tab:conv_fwd_pruning_last` | `forward_pruning_…conv-last.tex` | `conviction_summary_tables.py` |
| `tab:conv_dec_masking-zero_last` | `decile_masking-zero_…conv-last.tex` | `conviction_summary_tables.py` |
| `tab:conv_prune_edges_last` | `prune_edges_…conv-last.tex` | `conviction_summary_tables.py` |
| `tab:conv_prune_evd_last` | `prune_edges_vs_drop_…conv-last.tex` | `conviction_summary_tables.py` |
| `tab:conv_perds_high` | `perdataset_masking-zero_high.tex` | `make_perdataset_tables.py` |
| `tab:conv_perds_low` | `perdataset_masking-zero_low.tex` | `make_perdataset_tables.py` |
| `tab:conv_observational` | `observational_summary.tex` | `conviction_observational_table.py` |

The same folder has eleven more tables that the paper doesn't cite (robustness variants
such as the `conv-avg` score and `masking-mean`). The three figures of the paper are
diagrams, and no code produces them.

---

## Good to know

- Cornell and Wisconsin are left out of the conviction aggregates, because their training
  sets are too small for a percentage budget to mean anything (1% is a single node). The
  `--exclude=cornell,wisconsin` flag that does this is set in
  `exp/scripts/run_conviction_summary.sh`, so call that script rather than the Python file
  directly.
- On Windows, steps 2 and 3 are easier under Linux or WSL: `torch-householder` compiles a
  C++ extension, which needs the Visual Studio C++ Build Tools.
- The Weights & Biases logs (`wandb/`) are not included, and no credential is stored in the
  repository.
- The joint sheaf diffusion model is not implemented in this public repository; see
  Hernandez Caralt et al. (2024) for details on how to build it. Its archived outputs are on
  Zenodo, so steps 1 and 2 still cover the tables that use it.
