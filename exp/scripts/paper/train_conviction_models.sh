#!/bin/bash
# Trains the 10 models the conviction experiments load.
# Generated from src/TRAINING_RUNS.md — see it for where each
# command comes from. Run from the repository root, with the environment active:
#     bash exp/scripts/paper/train_conviction_models.sh
# Each run is 10-fold cross-validation; the large datasets need a GPU.
set -e
export WANDB_MODE="${WANDB_MODE:-disabled}"

# Amazon Ratings — GeneralSheaf, normalised, d=5, 5 layers
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

# Citeseer — GeneralSheaf, normalised, d=5, 4 layers
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

# Cora — GeneralSheaf, normalised, d=5, 3 layers
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

# Cornell — GeneralSheaf, normalised, d=2, 2 layers
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

# Minesweeper — GeneralSheaf, normalised, d=5, 6 layers
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

# PubMed — GeneralSheaf, normalised, d=3, 4 layers
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

# Questions — GeneralSheaf, normalised, d=5, 4 layers
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

# Roman Empire — GeneralSheaf, normalised, d=5, 5 layers
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

# Tolokers — GeneralSheaf, normalised, d=3, 4 layers
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

# Wisconsin — GeneralSheaf, normalised, d=2, 2 layers
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
