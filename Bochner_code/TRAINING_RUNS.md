# Training runs

Every trained model whose outputs enter the paper, with the command that produced it.
Training is the only GPU-heavy step. The archived outputs of these runs are on Zenodo (see
[`README.md`, level B](README.md#b-rerun-the-analysis-from-the-archived-raw-outputs)), so
none of this needs re-running to check the paper.

## How to run a command

From the repository root, with the environment active:

```bash
export WANDB_MODE=disabled      # no Weights & Biases account needed
python -m exp.run --dataset=cornell ...
```

Every run is 10-fold cross-validation (`--folds=10`, the splits in `splits/`), 500 epochs,
hidden width 16, seed 43 — all parser defaults except where a flag says otherwise; only
flags that differ from the defaults in `exp/parser.py` are written out. A run writes its
outputs under `results/<kind>/<dataset>/<model>/normalised-<x>/stalk_dim-<d>/<L>-layers/16-hidden/500-epochs/`.

The same commands are collected, one per run, in
[`exp/scripts/paper/train_conviction_models.sh`](../exp/scripts/paper/train_conviction_models.sh) and
[`exp/scripts/paper/train_forman_models.sh`](../exp/scripts/paper/train_forman_models.sh).

## Where the commands come from

- **Conviction models — exact.** Each model directory under `results/model_weights/` stores the
  `training_args.json` the run was launched with (all at code revision `76c20f9`). The commands
  are those arguments.
- **Forman models — from the W&B run configuration.** Several configurations were trained more
  than once, and each rerun overwrites `results/forman_eigs/`, so the run whose eigenvalues are
  archived is the **latest one with `--save_laplacians=True`**; its run id in our (private)
W&B log is given for traceability. The rule
  was checked on the conviction models, where `training_args.json` gives the ground truth: for all
  of them the latest saving run is the exact match.

The two families share the same per-dataset hyperparameters and differ only in model,
normalisation and what they save (`--save_laplacians` / `--save_others` for Forman,
`--save_norms` / `--save_preds` / `--save_saliency` for conviction).

## Conviction models — 10 runs

`GeneralSheaf`, normalised. Loaded by every conviction script through [`conviction/conviction_manifest.txt`](conviction/conviction_manifest.txt); they produce `results/model_weights/` and the `results/node_*` arrays. Cornell and Wisconsin are trained and analysed but excluded from the aggregated tables (their training pools are too small for a percentage budget to be meaningful).

| Dataset | Model | Normalised | d | Layers | Folds | Used in | Source |
|---|---|---|---|---|---|---|---|
| Amazon Ratings | GeneralSheaf | yes | 5 | 5 | 10 | all conviction tables; observational table (General sheaf) | `training_args.json` |
| Citeseer | GeneralSheaf | yes | 5 | 4 | 10 | all conviction tables; observational table (General sheaf) | `training_args.json` |
| Cora | GeneralSheaf | yes | 5 | 3 | 10 | all conviction tables; observational table (General sheaf) | `training_args.json` |
| Cornell | GeneralSheaf | yes | 2 | 2 | 10 | conviction runs; excluded from the aggregated tables (training pool too small) | `training_args.json` |
| Minesweeper | GeneralSheaf | yes | 5 | 6 | 10 | all conviction tables; observational table (General sheaf) | `training_args.json` |
| PubMed | GeneralSheaf | yes | 3 | 4 | 10 | all conviction tables; observational table (General sheaf) | `training_args.json` |
| Questions | GeneralSheaf | yes | 5 | 4 | 10 | all conviction tables; observational table (General sheaf) | `training_args.json` |
| Roman Empire | GeneralSheaf | yes | 5 | 5 | 10 | all conviction tables; observational table (General sheaf) | `training_args.json` |
| Tolokers | GeneralSheaf | yes | 3 | 4 | 10 | all conviction tables; observational table (General sheaf) | `training_args.json` |
| Wisconsin | GeneralSheaf | yes | 2 | 2 | 10 | conviction runs; excluded from the aggregated tables (training pool too small) | `training_args.json` |

<details><summary>Commands</summary>

**Amazon Ratings — GeneralSheaf, normalised, d=5, 5 layers**

```bash
python -m exp.run \
    --dataset=amazon_ratings \
    --model=GeneralSheaf \
    --normalised=True \
    --d=5 \
    --layers=5 \
    --hidden_channels=16 \
    --epochs=500 \
    --weight_decay=0.0006914841722570725 \
    --sheaf_decay=0.00031764232712732976 \
    --dropout=0.3 \
    --input_dropout=0.2 \
    --edge_weights=False \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_laplacians=False
```

**Citeseer — GeneralSheaf, normalised, d=5, 4 layers**

```bash
python -m exp.run \
    --dataset=citeseer \
    --model=GeneralSheaf \
    --normalised=True \
    --d=5 \
    --layers=4 \
    --hidden_channels=16 \
    --epochs=500 \
    --weight_decay=0.0006914841722570725 \
    --sheaf_decay=0.00031764232712732976 \
    --dropout=0.2 \
    --input_dropout=0.2 \
    --edge_weights=False \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_laplacians=False
```

**Cora — GeneralSheaf, normalised, d=5, 3 layers**

```bash
python -m exp.run \
    --dataset=cora \
    --model=GeneralSheaf \
    --normalised=True \
    --d=5 \
    --layers=3 \
    --hidden_channels=16 \
    --epochs=500 \
    --weight_decay=0.0006914841722570725 \
    --sheaf_decay=0.00031764232712732976 \
    --dropout=0.5 \
    --input_dropout=0.2 \
    --edge_weights=False \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_laplacians=False
```

**Cornell — GeneralSheaf, normalised, d=2, 2 layers**

```bash
python -m exp.run \
    --dataset=cornell \
    --model=GeneralSheaf \
    --normalised=True \
    --d=2 \
    --layers=2 \
    --hidden_channels=16 \
    --epochs=500 \
    --lr=0.02 \
    --weight_decay=0.0006914841722570725 \
    --sheaf_decay=0.00031764232712732976 \
    --dropout=0.5 \
    --input_dropout=0.2 \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_laplacians=False
```

**Minesweeper — GeneralSheaf, normalised, d=5, 6 layers**

```bash
python -m exp.run \
    --dataset=minesweeper \
    --model=GeneralSheaf \
    --normalised=True \
    --d=5 \
    --layers=6 \
    --hidden_channels=16 \
    --epochs=500 \
    --weight_decay=0.0006914841722570725 \
    --sheaf_decay=0.00031764232712732976 \
    --dropout=0.3 \
    --input_dropout=0.2 \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_laplacians=False
```

**PubMed — GeneralSheaf, normalised, d=3, 4 layers**

```bash
python -m exp.run \
    --dataset=pubmed \
    --model=GeneralSheaf \
    --normalised=True \
    --d=3 \
    --layers=4 \
    --hidden_channels=16 \
    --epochs=500 \
    --weight_decay=0.0006914841722570725 \
    --sheaf_decay=0.00031764232712732976 \
    --dropout=0.3 \
    --input_dropout=0.2 \
    --edge_weights=False \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_laplacians=False
```

**Questions — GeneralSheaf, normalised, d=5, 4 layers**

```bash
python -m exp.run \
    --dataset=questions \
    --model=GeneralSheaf \
    --normalised=True \
    --d=5 \
    --layers=4 \
    --hidden_channels=16 \
    --epochs=500 \
    --weight_decay=0.0006914841722570725 \
    --sheaf_decay=0.00031764232712732976 \
    --dropout=0.2 \
    --input_dropout=0.2 \
    --edge_weights=False \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_laplacians=False
```

**Roman Empire — GeneralSheaf, normalised, d=5, 5 layers**

```bash
python -m exp.run \
    --dataset=roman_empire \
    --model=GeneralSheaf \
    --normalised=True \
    --d=5 \
    --layers=5 \
    --hidden_channels=16 \
    --epochs=500 \
    --weight_decay=0.0006914841722570725 \
    --sheaf_decay=0.00031764232712732976 \
    --dropout=0.3 \
    --input_dropout=0.2 \
    --edge_weights=False \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_laplacians=False
```

**Tolokers — GeneralSheaf, normalised, d=3, 4 layers**

```bash
python -m exp.run \
    --dataset=tolokers \
    --model=GeneralSheaf \
    --normalised=True \
    --d=3 \
    --layers=4 \
    --hidden_channels=16 \
    --epochs=500 \
    --weight_decay=0.0006914841722570725 \
    --sheaf_decay=0.00031764232712732976 \
    --dropout=0.3 \
    --input_dropout=0.2 \
    --edge_weights=False \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_laplacians=False
```

**Wisconsin — GeneralSheaf, normalised, d=2, 2 layers**

```bash
python -m exp.run \
    --dataset=wisconsin \
    --model=GeneralSheaf \
    --normalised=True \
    --d=2 \
    --layers=2 \
    --hidden_channels=16 \
    --epochs=500 \
    --lr=0.02 \
    --weight_decay=0.0006685729356079199 \
    --sheaf_decay=0.0006685729356079199 \
    --dropout=0.7276458263736642 \
    --edge_weights=False \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_laplacians=False
```

</details>

## Forman profile — GeneralSheaf — 22 runs

Normalised and unnormalised, same hyperparameters as the conviction models. Each run saves the Forman eigenvalues of its own Laplacian and, with `--save_others=True`, of the Laplacian with the opposite normalisation built from the same maps (the `other` Laplacian of Tables 5–8). Cora was trained at two stalk dimensions; both enter the per-dataset mean, and the d=3 run saved no `other` Laplacian.

| Dataset | Model | Normalised | d | Layers | Folds | Used in | Source |
|---|---|---|---|---|---|---|---|
| Amazon Ratings | GeneralSheaf | yes | 5 | 5 | 10 | Tables 1, 6, 7, 8 | W&B `v9rm55yo` |
| Citeseer | GeneralSheaf | yes | 5 | 4 | 10 | Tables 1, 6, 7, 8 | W&B `hkpv9elq` |
| Cora | GeneralSheaf | yes | 3 | 3 | 10 | Tables 1 (no `other` Laplacian saved) | W&B `tr4tw4l0` |
| Cora | GeneralSheaf | yes | 5 | 3 | 10 | Tables 1, 6, 7, 8 | W&B `ioygag7d` |
| Cornell | GeneralSheaf | yes | 2 | 2 | 10 | Tables 1, 6, 7, 8 | W&B `wqiybad0` |
| Minesweeper | GeneralSheaf | yes | 5 | 6 | 10 | Tables 1, 6, 7, 8 | W&B `c71nuxqm` |
| PubMed | GeneralSheaf | yes | 5 | 4 | 10 | Tables 1, 6, 7, 8 | W&B `ofz4n8sg` |
| Questions | GeneralSheaf | yes | 5 | 4 | 10 | Tables 1, 6, 7, 8 | W&B `lxhqql88` |
| Roman Empire | GeneralSheaf | yes | 5 | 5 | 10 | Tables 1, 6, 7, 8 | W&B `8r4ouiso` |
| Tolokers | GeneralSheaf | yes | 3 | 4 | 10 | Tables 1, 6, 7, 8 | W&B `r69zz1gm` |
| Wisconsin | GeneralSheaf | yes | 2 | 2 | 10 | Tables 1, 6, 7, 8 | W&B `l8ub91ey` |
| Amazon Ratings | GeneralSheaf | no | 5 | 5 | 10 | Tables 2, 5, 7, 8 | W&B `5ts2sd9t` |
| Citeseer | GeneralSheaf | no | 5 | 4 | 10 | Tables 2, 5, 7, 8 | W&B `j1118shf` |
| Cora | GeneralSheaf | no | 3 | 3 | 10 | Tables 2 (no `other` Laplacian saved) | W&B `bisrpzjj` |
| Cora | GeneralSheaf | no | 5 | 3 | 10 | Tables 2, 5, 7, 8 | W&B `067vljth` |
| Cornell | GeneralSheaf | no | 2 | 2 | 10 | Tables 2, 5, 7, 8 | W&B `juuqxiyb` |
| Minesweeper | GeneralSheaf | no | 5 | 6 | 10 | Tables 2, 5, 7, 8 | W&B `9296hkij` |
| PubMed | GeneralSheaf | no | 5 | 4 | 10 | Tables 2, 5, 7, 8 | W&B `f0evtq1k` |
| Questions | GeneralSheaf | no | 5 | 4 | 10 | Tables 2, 5, 7, 8 | W&B `6oqt3cct` |
| Roman Empire | GeneralSheaf | no | 5 | 5 | 10 | Tables 2, 5, 7, 8 | W&B `yndbr8bw` |
| Tolokers | GeneralSheaf | no | 3 | 4 | 10 | Tables 2, 5, 7, 8 | W&B `ldrx10r8` |
| Wisconsin | GeneralSheaf | no | 2 | 2 | 10 | Tables 2, 5, 7, 8 | W&B `maai7vkd` |

<details><summary>Commands</summary>

**Amazon Ratings — GeneralSheaf, normalised, d=5, 5 layers**

```bash
python -m exp.run \
    --dataset=amazon_ratings \
    --model=GeneralSheaf \
    --normalised=True \
    --d=5 \
    --layers=5 \
    --hidden_channels=16 \
    --epochs=500 \
    --weight_decay=0.0006914841722570725 \
    --sheaf_decay=0.00031764232712732976 \
    --dropout=0.3 \
    --input_dropout=0.2 \
    --edge_weights=False \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_norms=False \
    --save_others=True
```

**Citeseer — GeneralSheaf, normalised, d=5, 4 layers**

```bash
python -m exp.run \
    --dataset=citeseer \
    --model=GeneralSheaf \
    --normalised=True \
    --d=5 \
    --layers=4 \
    --hidden_channels=16 \
    --epochs=500 \
    --weight_decay=0.0006914841722570725 \
    --sheaf_decay=0.00031764232712732976 \
    --dropout=0.2 \
    --input_dropout=0.2 \
    --edge_weights=False \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_norms=False \
    --save_others=True
```

**Cora — GeneralSheaf, normalised, d=3, 3 layers**

```bash
python -m exp.run \
    --dataset=cora \
    --model=GeneralSheaf \
    --normalised=True \
    --d=3 \
    --layers=3 \
    --hidden_channels=16 \
    --epochs=500 \
    --weight_decay=0.0006914841722570725 \
    --sheaf_decay=0.00031764232712732976 \
    --dropout=0.5 \
    --input_dropout=0.2 \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_norms=False
```

**Cora — GeneralSheaf, normalised, d=5, 3 layers**

```bash
python -m exp.run \
    --dataset=cora \
    --model=GeneralSheaf \
    --normalised=True \
    --d=5 \
    --layers=3 \
    --hidden_channels=16 \
    --epochs=500 \
    --weight_decay=0.0006914841722570725 \
    --sheaf_decay=0.00031764232712732976 \
    --dropout=0.5 \
    --input_dropout=0.2 \
    --edge_weights=False \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_norms=False \
    --save_others=True
```

**Cornell — GeneralSheaf, normalised, d=2, 2 layers**

```bash
python -m exp.run \
    --dataset=cornell \
    --model=GeneralSheaf \
    --normalised=True \
    --d=2 \
    --layers=2 \
    --hidden_channels=16 \
    --epochs=500 \
    --lr=0.02 \
    --weight_decay=0.0006914841722570725 \
    --sheaf_decay=0.00031764232712732976 \
    --dropout=0.5 \
    --input_dropout=0.2 \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_norms=False \
    --save_others=True
```

**Minesweeper — GeneralSheaf, normalised, d=5, 6 layers**

```bash
python -m exp.run \
    --dataset=minesweeper \
    --model=GeneralSheaf \
    --normalised=True \
    --d=5 \
    --layers=6 \
    --hidden_channels=16 \
    --epochs=500 \
    --weight_decay=0.0006914841722570725 \
    --sheaf_decay=0.00031764232712732976 \
    --dropout=0.3 \
    --input_dropout=0.2 \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_norms=False \
    --save_others=True
```

**PubMed — GeneralSheaf, normalised, d=5, 4 layers**

```bash
python -m exp.run \
    --dataset=pubmed \
    --model=GeneralSheaf \
    --normalised=True \
    --d=5 \
    --layers=4 \
    --hidden_channels=16 \
    --epochs=500 \
    --weight_decay=0.0006914841722570725 \
    --sheaf_decay=0.00031764232712732976 \
    --dropout=0.3 \
    --input_dropout=0.2 \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_norms=False \
    --save_others=True
```

**Questions — GeneralSheaf, normalised, d=5, 4 layers**

```bash
python -m exp.run \
    --dataset=questions \
    --model=GeneralSheaf \
    --normalised=True \
    --d=5 \
    --layers=4 \
    --hidden_channels=16 \
    --epochs=500 \
    --weight_decay=0.0006914841722570725 \
    --sheaf_decay=0.00031764232712732976 \
    --dropout=0.2 \
    --input_dropout=0.2 \
    --edge_weights=False \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_norms=False \
    --save_others=True
```

**Roman Empire — GeneralSheaf, normalised, d=5, 5 layers**

```bash
python -m exp.run \
    --dataset=roman_empire \
    --model=GeneralSheaf \
    --normalised=True \
    --d=5 \
    --layers=5 \
    --hidden_channels=16 \
    --epochs=500 \
    --weight_decay=0.0006914841722570725 \
    --sheaf_decay=0.00031764232712732976 \
    --dropout=0.3 \
    --input_dropout=0.2 \
    --edge_weights=False \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_norms=False \
    --save_others=True
```

**Tolokers — GeneralSheaf, normalised, d=3, 4 layers**

```bash
python -m exp.run \
    --dataset=tolokers \
    --model=GeneralSheaf \
    --normalised=True \
    --d=3 \
    --layers=4 \
    --hidden_channels=16 \
    --epochs=500 \
    --weight_decay=0.0006914841722570725 \
    --sheaf_decay=0.00031764232712732976 \
    --dropout=0.3 \
    --input_dropout=0.2 \
    --edge_weights=False \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_norms=False \
    --save_others=True
```

**Wisconsin — GeneralSheaf, normalised, d=2, 2 layers**

```bash
python -m exp.run \
    --dataset=wisconsin \
    --model=GeneralSheaf \
    --normalised=True \
    --d=2 \
    --layers=2 \
    --hidden_channels=16 \
    --epochs=500 \
    --lr=0.02 \
    --weight_decay=0.0006685729356079199 \
    --sheaf_decay=0.0006685729356079199 \
    --dropout=0.7276458263736642 \
    --edge_weights=False \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_norms=False \
    --save_others=True
```

**Amazon Ratings — GeneralSheaf, unnormalised, d=5, 5 layers**

```bash
python -m exp.run \
    --dataset=amazon_ratings \
    --model=GeneralSheaf \
    --normalised=False \
    --d=5 \
    --layers=5 \
    --hidden_channels=16 \
    --epochs=500 \
    --weight_decay=0.0006914841722570725 \
    --sheaf_decay=0.00031764232712732976 \
    --dropout=0.3 \
    --input_dropout=0.2 \
    --edge_weights=False \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_norms=False \
    --save_others=True
```

**Citeseer — GeneralSheaf, unnormalised, d=5, 4 layers**

```bash
python -m exp.run \
    --dataset=citeseer \
    --model=GeneralSheaf \
    --normalised=False \
    --d=5 \
    --layers=4 \
    --hidden_channels=16 \
    --epochs=500 \
    --weight_decay=0.0006914841722570725 \
    --sheaf_decay=0.00031764232712732976 \
    --dropout=0.2 \
    --input_dropout=0.2 \
    --edge_weights=False \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_norms=False \
    --save_others=True
```

**Cora — GeneralSheaf, unnormalised, d=3, 3 layers**

```bash
python -m exp.run \
    --dataset=cora \
    --model=GeneralSheaf \
    --normalised=False \
    --d=3 \
    --layers=3 \
    --hidden_channels=16 \
    --epochs=500 \
    --weight_decay=0.0006914841722570725 \
    --sheaf_decay=0.00031764232712732976 \
    --dropout=0.5 \
    --input_dropout=0.2 \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_norms=False
```

**Cora — GeneralSheaf, unnormalised, d=5, 3 layers**

```bash
python -m exp.run \
    --dataset=cora \
    --model=GeneralSheaf \
    --normalised=False \
    --d=5 \
    --layers=3 \
    --hidden_channels=16 \
    --epochs=500 \
    --weight_decay=0.0006914841722570725 \
    --sheaf_decay=0.00031764232712732976 \
    --dropout=0.5 \
    --input_dropout=0.2 \
    --edge_weights=False \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_norms=False \
    --save_others=True
```

**Cornell — GeneralSheaf, unnormalised, d=2, 2 layers**

```bash
python -m exp.run \
    --dataset=cornell \
    --model=GeneralSheaf \
    --normalised=False \
    --d=2 \
    --layers=2 \
    --hidden_channels=16 \
    --epochs=500 \
    --lr=0.02 \
    --weight_decay=0.0006914841722570725 \
    --sheaf_decay=0.00031764232712732976 \
    --dropout=0.5 \
    --input_dropout=0.2 \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_norms=False \
    --save_others=True
```

**Minesweeper — GeneralSheaf, unnormalised, d=5, 6 layers**

```bash
python -m exp.run \
    --dataset=minesweeper \
    --model=GeneralSheaf \
    --normalised=False \
    --d=5 \
    --layers=6 \
    --hidden_channels=16 \
    --epochs=500 \
    --weight_decay=0.0006914841722570725 \
    --sheaf_decay=0.00031764232712732976 \
    --dropout=0.3 \
    --input_dropout=0.2 \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_norms=False \
    --save_others=True
```

**PubMed — GeneralSheaf, unnormalised, d=5, 4 layers**

```bash
python -m exp.run \
    --dataset=pubmed \
    --model=GeneralSheaf \
    --normalised=False \
    --d=5 \
    --layers=4 \
    --hidden_channels=16 \
    --epochs=500 \
    --weight_decay=0.0006914841722570725 \
    --sheaf_decay=0.00031764232712732976 \
    --dropout=0.3 \
    --input_dropout=0.2 \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_norms=False \
    --save_others=True
```

**Questions — GeneralSheaf, unnormalised, d=5, 4 layers**

```bash
python -m exp.run \
    --dataset=questions \
    --model=GeneralSheaf \
    --normalised=False \
    --d=5 \
    --layers=4 \
    --hidden_channels=16 \
    --epochs=500 \
    --weight_decay=0.0006914841722570725 \
    --sheaf_decay=0.00031764232712732976 \
    --dropout=0.2 \
    --input_dropout=0.2 \
    --edge_weights=False \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_norms=False \
    --save_others=True
```

**Roman Empire — GeneralSheaf, unnormalised, d=5, 5 layers**

```bash
python -m exp.run \
    --dataset=roman_empire \
    --model=GeneralSheaf \
    --normalised=False \
    --d=5 \
    --layers=5 \
    --hidden_channels=16 \
    --epochs=500 \
    --weight_decay=0.0006914841722570725 \
    --sheaf_decay=0.00031764232712732976 \
    --dropout=0.3 \
    --input_dropout=0.2 \
    --edge_weights=False \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_norms=False \
    --save_others=True
```

**Tolokers — GeneralSheaf, unnormalised, d=3, 4 layers**

```bash
python -m exp.run \
    --dataset=tolokers \
    --model=GeneralSheaf \
    --normalised=False \
    --d=3 \
    --layers=4 \
    --hidden_channels=16 \
    --epochs=500 \
    --weight_decay=0.0006914841722570725 \
    --sheaf_decay=0.00031764232712732976 \
    --dropout=0.3 \
    --input_dropout=0.2 \
    --edge_weights=False \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_norms=False \
    --save_others=True
```

**Wisconsin — GeneralSheaf, unnormalised, d=2, 2 layers**

```bash
python -m exp.run \
    --dataset=wisconsin \
    --model=GeneralSheaf \
    --normalised=False \
    --d=2 \
    --layers=2 \
    --hidden_channels=16 \
    --epochs=500 \
    --lr=0.02 \
    --weight_decay=0.0006685729356079199 \
    --sheaf_decay=0.0006685729356079199 \
    --dropout=0.7276458263736642 \
    --edge_weights=False \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_norms=False \
    --save_others=True
```

</details>

## Forman profile — JointSheafParams — 21 runs

The joint sheaf diffusion model is not implemented in this public repository; see
Hernandez Caralt et al. (2024) for details. The runs are listed for completeness, and their
outputs are in the Zenodo record.

Identity first maps (`learn_first_maps=False`), unnormalised (Table 3) and normalised (Table 4). The normalised regime is less stable and was trained at lower stalk dimension and depth; Tolokers is absent from Table 4 because its training was unstable.

| Dataset | Model | Normalised | d | Layers | Folds | Used in | Source |
|---|---|---|---|---|---|---|---|
| Amazon Ratings | JointSheafParams (identity) | no | 5 | 5 | 10 | Table 3 | W&B `n58ha5a3` |
| Citeseer | JointSheafParams (identity) | no | 5 | 4 | 10 | Table 3 | W&B `2mqwrlse` |
| Cora | JointSheafParams (identity) | no | 3 | 3 | 10 | Table 3 | W&B `fdo7sco3` |
| Cornell | JointSheafParams (identity) | no | 2 | 2 | 10 | Table 3 | W&B `0gs8bv9s` |
| Minesweeper | JointSheafParams (identity) | no | 5 | 6 | 10 | Table 3 | W&B `0ckd9wew` |
| PubMed | JointSheafParams (identity) | no | 5 | 4 | 10 | Table 3 | W&B `391o6654` |
| Questions | JointSheafParams (identity) | no | 5 | 4 | 10 | Table 3 | W&B `hcr27k0u` |
| Roman Empire | JointSheafParams (identity) | no | 5 | 5 | 10 | Table 3 | W&B `bjyrv5pr` |
| Tolokers | JointSheafParams (identity) | no | 3 | 4 | 10 | Table 3 | W&B `pg2pc2nt` |
| Wisconsin | JointSheafParams (identity) | no | 2 | 2 | 10 | Table 3 | W&B `d803e282` |
| Amazon Ratings | JointSheafParams (identity) | yes | 2 | 3 | 10 | Table 4 | W&B `6gvi8vhq` |
| Citeseer | JointSheafParams (identity) | yes | 3 | 3 | 10 | Table 4 | W&B `qc1qtxpf` |
| Cora | JointSheafParams (identity) | yes | 2 | 3 | **6 of 10** | Table 4 | W&B `9b7snu6l` |
| Cora | JointSheafParams (identity) | yes | 3 | 3 | **2 of 10** | Table 4 | W&B `rcm4qwl7` |
| Cornell | JointSheafParams (identity) | yes | 2 | 2 | 10 | Table 4 | W&B `bttwczin` |
| Minesweeper | JointSheafParams (identity) | yes | 3 | 3 | 10 | Table 4 | W&B `mhm2ueil` |
| PubMed | JointSheafParams (identity) | yes | 2 | 3 | **2 of 10** | Table 4 | W&B `lo3uzxg8` |
| PubMed | JointSheafParams (identity) | yes | 3 | 3 | **4 of 10** | Table 4 | W&B `052mlwri` |
| Questions | JointSheafParams (identity) | yes | 2 | 4 | **2 of 10** | Table 4 | W&B `42xe15v0` |
| Roman Empire | JointSheafParams (identity) | yes | 3 | 3 | 10 | Table 4 | W&B `o8p949fm` |
| Wisconsin | JointSheafParams (identity) | yes | 2 | 2 | 10 | Table 4 | W&B `pxwt7jmi` |

**Incomplete runs.** Five normalised JointSheafParams runs were interrupted before finishing all ten folds (Cora d=2 and d=3, PubMed d=2 and d=3, Questions). Table 4 uses the folds that completed: each dataset's value is the mean over its available folds, and Cora and PubMed pool their two stalk dimensions. Re-running the commands below trains all ten folds, so a full re-run will not reproduce Table 4's numbers exactly for these three datasets.

<details><summary>Commands</summary>

**Amazon Ratings — JointSheafParams (identity first maps), unnormalised, d=5, 5 layers**

```bash
python -m exp.run \
    --dataset=amazon_ratings \
    --model=JointSheafParams \
    --normalised=False \
    --d=5 \
    --layers=5 \
    --hidden_channels=16 \
    --epochs=500 \
    --weight_decay=0.0006914841722570725 \
    --sheaf_decay=0.00031764232712732976 \
    --dropout=0.3 \
    --input_dropout=0.2 \
    --edge_weights=False \
    --dual_left_linear=True \
    --dual_linear=False \
    --dual_normalised=False \
    --dual_right_linear=True \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_norms=False
```

**Citeseer — JointSheafParams (identity first maps), unnormalised, d=5, 4 layers**

```bash
python -m exp.run \
    --dataset=citeseer \
    --model=JointSheafParams \
    --normalised=False \
    --d=5 \
    --layers=4 \
    --hidden_channels=16 \
    --epochs=500 \
    --weight_decay=0.0006914841722570725 \
    --sheaf_decay=0.00031764232712732976 \
    --dropout=0.2 \
    --input_dropout=0.2 \
    --edge_weights=False \
    --dual_left_linear=True \
    --dual_linear=False \
    --dual_normalised=False \
    --dual_right_linear=True \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_norms=False
```

**Cora — JointSheafParams (identity first maps), unnormalised, d=3, 3 layers**

```bash
python -m exp.run \
    --dataset=cora \
    --model=JointSheafParams \
    --normalised=False \
    --d=3 \
    --layers=3 \
    --hidden_channels=16 \
    --epochs=500 \
    --weight_decay=0.0006914841722570725 \
    --sheaf_decay=0.00031764232712732976 \
    --dropout=0.5 \
    --input_dropout=0.2 \
    --edge_weights=False \
    --dual_left_linear=True \
    --dual_linear=False \
    --dual_normalised=False \
    --dual_right_linear=True \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_norms=False
```

**Cornell — JointSheafParams (identity first maps), unnormalised, d=2, 2 layers**

```bash
python -m exp.run \
    --dataset=cornell \
    --model=JointSheafParams \
    --normalised=False \
    --d=2 \
    --layers=2 \
    --hidden_channels=16 \
    --epochs=500 \
    --lr=0.02 \
    --weight_decay=0.0006914841722570725 \
    --sheaf_decay=0.00031764232712732976 \
    --dropout=0.5 \
    --input_dropout=0.2 \
    --dual_left_linear=True \
    --dual_linear=False \
    --dual_normalised=False \
    --dual_right_linear=True \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_norms=False
```

**Minesweeper — JointSheafParams (identity first maps), unnormalised, d=5, 6 layers**

```bash
python -m exp.run \
    --dataset=minesweeper \
    --model=JointSheafParams \
    --normalised=False \
    --d=5 \
    --layers=6 \
    --hidden_channels=16 \
    --epochs=500 \
    --weight_decay=0.0006914841722570725 \
    --sheaf_decay=0.00031764232712732976 \
    --dropout=0.3 \
    --input_dropout=0.2 \
    --dual_left_linear=True \
    --dual_linear=False \
    --dual_normalised=False \
    --dual_right_linear=True \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_norms=False
```

**PubMed — JointSheafParams (identity first maps), unnormalised, d=5, 4 layers**

```bash
python -m exp.run \
    --dataset=pubmed \
    --model=JointSheafParams \
    --normalised=False \
    --d=5 \
    --layers=4 \
    --hidden_channels=16 \
    --epochs=500 \
    --weight_decay=0.0006914841722570725 \
    --sheaf_decay=0.00031764232712732976 \
    --dropout=0.3 \
    --input_dropout=0.2 \
    --dual_left_linear=True \
    --dual_linear=False \
    --dual_normalised=False \
    --dual_right_linear=True \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_norms=False
```

**Questions — JointSheafParams (identity first maps), unnormalised, d=5, 4 layers**

```bash
python -m exp.run \
    --dataset=questions \
    --model=JointSheafParams \
    --normalised=False \
    --d=5 \
    --layers=4 \
    --hidden_channels=16 \
    --epochs=500 \
    --weight_decay=0.0006914841722570725 \
    --sheaf_decay=0.00031764232712732976 \
    --dropout=0.2 \
    --input_dropout=0.2 \
    --edge_weights=False \
    --dual_left_linear=True \
    --dual_linear=False \
    --dual_normalised=False \
    --dual_right_linear=True \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_norms=False
```

**Roman Empire — JointSheafParams (identity first maps), unnormalised, d=5, 5 layers**

```bash
python -m exp.run \
    --dataset=roman_empire \
    --model=JointSheafParams \
    --normalised=False \
    --d=5 \
    --layers=5 \
    --hidden_channels=16 \
    --epochs=500 \
    --weight_decay=0.0006914841722570725 \
    --sheaf_decay=0.00031764232712732976 \
    --dropout=0.3 \
    --input_dropout=0.2 \
    --edge_weights=False \
    --dual_left_linear=True \
    --dual_linear=False \
    --dual_normalised=False \
    --dual_right_linear=True \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_norms=False
```

**Tolokers — JointSheafParams (identity first maps), unnormalised, d=3, 4 layers**

```bash
python -m exp.run \
    --dataset=tolokers \
    --model=JointSheafParams \
    --normalised=False \
    --d=3 \
    --layers=4 \
    --hidden_channels=16 \
    --epochs=500 \
    --weight_decay=0.0006914841722570725 \
    --sheaf_decay=0.00031764232712732976 \
    --dropout=0.3 \
    --input_dropout=0.2 \
    --edge_weights=False \
    --dual_left_linear=True \
    --dual_linear=False \
    --dual_normalised=False \
    --dual_right_linear=True \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_norms=False
```

**Wisconsin — JointSheafParams (identity first maps), unnormalised, d=2, 2 layers**

```bash
python -m exp.run \
    --dataset=wisconsin \
    --model=JointSheafParams \
    --normalised=False \
    --d=2 \
    --layers=2 \
    --hidden_channels=16 \
    --epochs=500 \
    --lr=0.02 \
    --weight_decay=0.0006685729356079199 \
    --sheaf_decay=0.0006685729356079199 \
    --dropout=0.7276458263736642 \
    --edge_weights=False \
    --dual_left_linear=True \
    --dual_linear=False \
    --dual_normalised=False \
    --dual_right_linear=True \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_norms=False
```

**Amazon Ratings — JointSheafParams (identity first maps), normalised, d=2, 3 layers**

```bash
python -m exp.run \
    --dataset=amazon_ratings \
    --model=JointSheafParams \
    --normalised=True \
    --d=2 \
    --layers=3 \
    --hidden_channels=16 \
    --epochs=500 \
    --weight_decay=0.0006914841722570725 \
    --sheaf_decay=0.00031764232712732976 \
    --dropout=0.3 \
    --input_dropout=0.2 \
    --edge_weights=False \
    --dual_left_linear=True \
    --dual_linear=False \
    --dual_right_linear=True \
    --learnt_map_type=general \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_norms=False
```

**Citeseer — JointSheafParams (identity first maps), normalised, d=3, 3 layers**

```bash
python -m exp.run \
    --dataset=citeseer \
    --model=JointSheafParams \
    --normalised=True \
    --d=3 \
    --layers=3 \
    --hidden_channels=16 \
    --epochs=500 \
    --weight_decay=0.0006914841722570725 \
    --sheaf_decay=0.00031764232712732976 \
    --dropout=0.2 \
    --input_dropout=0.2 \
    --edge_weights=False \
    --dual_left_linear=True \
    --dual_linear=False \
    --dual_right_linear=True \
    --learnt_map_type=general \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_norms=False
```

**Cora — JointSheafParams (identity first maps), normalised, d=2, 3 layers**

```bash
python -m exp.run \
    --dataset=cora \
    --model=JointSheafParams \
    --normalised=True \
    --d=2 \
    --layers=3 \
    --hidden_channels=16 \
    --epochs=500 \
    --weight_decay=0.0006914841722570725 \
    --sheaf_decay=0.00031764232712732976 \
    --dropout=0.5 \
    --input_dropout=0.2 \
    --edge_weights=False \
    --dual_left_linear=True \
    --dual_linear=False \
    --dual_right_linear=True \
    --learnt_map_type=general \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_norms=False
```

**Cora — JointSheafParams (identity first maps), normalised, d=3, 3 layers**

```bash
python -m exp.run \
    --dataset=cora \
    --model=JointSheafParams \
    --normalised=True \
    --d=3 \
    --layers=3 \
    --hidden_channels=16 \
    --epochs=500 \
    --weight_decay=0.0006914841722570725 \
    --sheaf_decay=0.00031764232712732976 \
    --dropout=0.5 \
    --input_dropout=0.2 \
    --edge_weights=False \
    --dual_left_linear=True \
    --dual_linear=False \
    --dual_right_linear=True \
    --learnt_map_type=general \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_norms=False
```

**Cornell — JointSheafParams (identity first maps), normalised, d=2, 2 layers**

```bash
python -m exp.run \
    --dataset=cornell \
    --model=JointSheafParams \
    --normalised=True \
    --d=2 \
    --layers=2 \
    --hidden_channels=16 \
    --epochs=500 \
    --lr=0.02 \
    --weight_decay=0.0006914841722570725 \
    --sheaf_decay=0.00031764232712732976 \
    --dropout=0.5 \
    --input_dropout=0.2 \
    --dual_left_linear=True \
    --dual_linear=False \
    --dual_right_linear=True \
    --learnt_map_type=general \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_norms=False
```

**Minesweeper — JointSheafParams (identity first maps), normalised, d=3, 3 layers**

```bash
python -m exp.run \
    --dataset=minesweeper \
    --model=JointSheafParams \
    --normalised=True \
    --d=3 \
    --layers=3 \
    --hidden_channels=16 \
    --epochs=500 \
    --weight_decay=0.0006914841722570725 \
    --sheaf_decay=0.00031764232712732976 \
    --dropout=0.3 \
    --input_dropout=0.2 \
    --dual_left_linear=True \
    --dual_linear=False \
    --dual_right_linear=True \
    --learnt_map_type=general \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_norms=False
```

**PubMed — JointSheafParams (identity first maps), normalised, d=2, 3 layers**

```bash
python -m exp.run \
    --dataset=pubmed \
    --model=JointSheafParams \
    --normalised=True \
    --d=2 \
    --layers=3 \
    --hidden_channels=16 \
    --epochs=500 \
    --weight_decay=0.0006914841722570725 \
    --sheaf_decay=0.00031764232712732976 \
    --dropout=0.3 \
    --input_dropout=0.2 \
    --dual_left_linear=True \
    --dual_linear=False \
    --dual_right_linear=True \
    --learnt_map_type=general \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_norms=False
```

**PubMed — JointSheafParams (identity first maps), normalised, d=3, 3 layers**

```bash
python -m exp.run \
    --dataset=pubmed \
    --model=JointSheafParams \
    --normalised=True \
    --d=3 \
    --layers=3 \
    --hidden_channels=16 \
    --epochs=500 \
    --weight_decay=0.0006914841722570725 \
    --sheaf_decay=0.00031764232712732976 \
    --dropout=0.3 \
    --input_dropout=0.2 \
    --dual_left_linear=True \
    --dual_linear=False \
    --dual_right_linear=True \
    --learnt_map_type=general \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_norms=False
```

**Questions — JointSheafParams (identity first maps), normalised, d=2, 4 layers**

```bash
python -m exp.run \
    --dataset=questions \
    --model=JointSheafParams \
    --normalised=True \
    --d=2 \
    --layers=4 \
    --hidden_channels=16 \
    --epochs=500 \
    --weight_decay=0.0006914841722570725 \
    --sheaf_decay=0.00031764232712732976 \
    --dropout=0.2 \
    --input_dropout=0.2 \
    --edge_weights=False \
    --dual_left_linear=True \
    --dual_linear=False \
    --dual_right_linear=True \
    --learnt_map_type=general \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_norms=False
```

**Roman Empire — JointSheafParams (identity first maps), normalised, d=3, 3 layers**

```bash
python -m exp.run \
    --dataset=roman_empire \
    --model=JointSheafParams \
    --normalised=True \
    --d=3 \
    --layers=3 \
    --hidden_channels=16 \
    --epochs=500 \
    --weight_decay=0.0006914841722570725 \
    --sheaf_decay=0.00031764232712732976 \
    --dropout=0.3 \
    --input_dropout=0.2 \
    --edge_weights=False \
    --dual_left_linear=True \
    --dual_linear=False \
    --dual_right_linear=True \
    --learnt_map_type=general \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_norms=False
```

**Wisconsin — JointSheafParams (identity first maps), normalised, d=2, 2 layers**

```bash
python -m exp.run \
    --dataset=wisconsin \
    --model=JointSheafParams \
    --normalised=True \
    --d=2 \
    --layers=2 \
    --hidden_channels=16 \
    --epochs=500 \
    --lr=0.02 \
    --weight_decay=0.0006685729356079199 \
    --sheaf_decay=0.0006685729356079199 \
    --dropout=0.7276458263736642 \
    --edge_weights=False \
    --dual_left_linear=True \
    --dual_linear=False \
    --dual_right_linear=True \
    --learnt_map_type=general \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_norms=False
```

</details>
