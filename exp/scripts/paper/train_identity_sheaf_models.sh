#!/bin/bash
# Trains the identity-sheaf control models for the conviction analysis.
# Each run is the GeneralSheaf run of train_conviction_models.sh with every flag kept,
# except --model=IdentitySheaf --d=1: restriction maps fixed to the identity, so the
# diffusion operator is the GCN-normalised graph Laplacian. --sheaf_decay is kept for
# a literal copy but has no effect (the model has no sheaf parameters).
# Cornell and Wisconsin are omitted: the paper excludes them from the conviction aggregation.
# Ordered smallest dataset first. Run from the repository root, with the environment active:
#     bash exp/scripts/paper/train_identity_sheaf_models.sh
# Each run is 10-fold cross-validation; it runs on CPU.
set -e
export WANDB_MODE="${WANDB_MODE:-disabled}"

# Cora — IdentitySheaf, normalised, d=1, 3 layers
python -m exp.run \
    --dataset=cora \
    --model=IdentitySheaf \
    --normalised=True \
    --d=1 \
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

# Citeseer — IdentitySheaf, normalised, d=1, 4 layers
python -m exp.run \
    --dataset=citeseer \
    --model=IdentitySheaf \
    --normalised=True \
    --d=1 \
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

# Minesweeper — IdentitySheaf, normalised, d=1, 6 layers
python -m exp.run \
    --dataset=minesweeper \
    --model=IdentitySheaf \
    --normalised=True \
    --d=1 \
    --layers=6 \
    --hidden_channels=16 \
    --epochs=500 \
    --weight_decay=0.0006914841722570725 \
    --sheaf_decay=0.00031764232712732976 \
    --dropout=0.3 \
    --input_dropout=0.2 \
    --checkpoint_epochs=0,1,5,15,50,200 \
    --save_laplacians=False

# PubMed — IdentitySheaf, normalised, d=1, 4 layers
python -m exp.run \
    --dataset=pubmed \
    --model=IdentitySheaf \
    --normalised=True \
    --d=1 \
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

# Tolokers — IdentitySheaf, normalised, d=1, 4 layers
python -m exp.run \
    --dataset=tolokers \
    --model=IdentitySheaf \
    --normalised=True \
    --d=1 \
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

# Roman Empire — IdentitySheaf, normalised, d=1, 5 layers
python -m exp.run \
    --dataset=roman_empire \
    --model=IdentitySheaf \
    --normalised=True \
    --d=1 \
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

# Amazon Ratings — IdentitySheaf, normalised, d=1, 5 layers
python -m exp.run \
    --dataset=amazon_ratings \
    --model=IdentitySheaf \
    --normalised=True \
    --d=1 \
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

# Questions — IdentitySheaf, normalised, d=1, 4 layers
python -m exp.run \
    --dataset=questions \
    --model=IdentitySheaf \
    --normalised=True \
    --d=1 \
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
