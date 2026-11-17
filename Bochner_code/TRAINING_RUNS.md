# Training the models

This page lists the models behind the paper and how to train them again. You don't need to
do this to check the paper: the outputs of all these runs are on
[Zenodo](https://doi.org/10.5281/zenodo.23080428), and the
[main guide](README.md) shows how to go from there to every table. Training is only
needed if you want to start from scratch.

## Running a model

From the root of the repository, with the environment active:

```bash
export WANDB_MODE=disabled      # skip Weights & Biases
python -m exp.run --dataset=cornell --model=GeneralSheaf --normalised=True \
    --d=2 --layers=2 --hidden_channels=16 --epochs=500 --lr=0.02 \
    --weight_decay=0.0006914841722570725 --sheaf_decay=0.00031764232712732976 \
    --dropout=0.5 --input_dropout=0.2 \
    --checkpoint_epochs=0,1,5,15,50,200 --save_norms=False --save_others=True
```

This is the Cornell model of the Forman profile experiment.

Unless a flag says otherwise, every run is a 10-fold cross-validation on the splits in
`splits/`, with 500 epochs, hidden width 16 and seed 43. These are the defaults of
`exp/parser.py`, so only the flags that differ from them are written out. The outputs of a
run go to `results/<kind>/<dataset>/<model>/...`, where `<kind>` is the type of array
(model weights, Forman eigenvalues, node norms, and so on).

The exact command of every run is in three scripts, one command per run:

| Script | Runs | What it trains |
|---|---|---|
| [`train_conviction_models.sh`](../exp/scripts/paper/train_conviction_models.sh) | 10 | the networks of the conviction experiment |
| [`train_forman_models.sh`](../exp/scripts/paper/train_forman_models.sh) | 22 | the networks of the Forman profile experiment |
| [`train_identity_sheaf_models.sh`](../exp/scripts/paper/train_identity_sheaf_models.sh) | 8 | the identity-sheaf control of the conviction experiment |

Run one with `bash exp/scripts/paper/<script>`.

## Settings per dataset

All the models are `GeneralSheaf` with a hidden width of 16, a weight decay of 6.9e-4 and
a sheaf decay of 3.2e-4. What changes from one dataset to the next is the stalk dimension
*d*, the number of layers and the dropout:

| Dataset | Layers | Dropout | *d* (conviction) | *d* (Forman) |
|---|---|---|---|---|
| Amazon Ratings | 5 | 0.3 | 5 | 5 |
| Citeseer | 4 | 0.2 | 5 | 5 |
| Cora | 3 | 0.5 | 5 | 3 and 5 |
| Cornell | 2 | 0.5 | 2 | 2 |
| Minesweeper | 6 | 0.3 | 5 | 5 |
| PubMed | 4 | 0.3 | 3 | 5 |
| Questions | 4 | 0.2 | 5 | 5 |
| Roman Empire | 5 | 0.3 | 5 | 5 |
| Tolokers | 4 | 0.3 | 3 | 3 |
| Wisconsin | 2 | 0.73 | 2 | 2 |

The input dropout is 0.2 throughout, except on Wisconsin where it is not set. The two
small WebKB datasets, Cornell and Wisconsin, also use a learning rate of 0.02, and
Wisconsin has its own weight and sheaf decay (6.6e-4). All of this is spelled out in the
scripts.

## The conviction models

Ten normalised `GeneralSheaf` models, one per dataset. The conviction scripts load them
through [`conviction/conviction_manifest.txt`](conviction/conviction_manifest.txt), and
their outputs are the model weights and the per-node arrays used by the conviction
experiment. Cornell and Wisconsin are trained and analysed, but left out of the aggregated
tables, because their training sets are too small for a percentage budget to mean anything
(1% is a single node).

The identity-sheaf control repeats eight of these runs (all but Cornell and Wisconsin)
with `--model=IdentitySheaf --d=1`: the restriction maps are fixed to the identity, so the
network diffuses with the plain graph Laplacian. The rest of the setup is copied from the
conviction run.

## The Forman profile models

22 `GeneralSheaf` models: every dataset trained twice, normalised and unnormalised, with
the same settings as above.

Each run saves the Forman eigenvalues of its own Laplacian. With `--save_others=True` it
also saves those of the Laplacian with the opposite normalisation, built from the same
maps. This is the "other" Laplacian of the cross-comparison tables.

| | Used in |
|---|---|
| Normalised models | Tables 1, 6, 7 and 8 |
| Unnormalised models | Tables 2, 5, 7 and 8 |

## The joint sheaf diffusion models

Tables 3 and 4 also use 21 runs of the joint sheaf diffusion model, trained with the
identity as first maps, unnormalised (Table 3) and normalised (Table 4). That model is not
implemented in this public repository; see Hernandez Caralt et al. (2024) for details. Their
outputs are on Zenodo, so the tables can still be rebuilt, and the commands are kept,
commented out, at the end of `train_forman_models.sh`.

Some of the normalised runs stopped before finishing their ten folds, and Table 4 uses
the folds that completed. Tolokers is missing from Table 4 because its training is
unstable.

## Retraining gives similar, not identical, numbers

A retrained model is not bit-identical to the archived one, since GPU arithmetic and
library versions differ, but they agree within the spread across folds.
