#!/bin/bash
# Trains the models behind the Forman-profile tables: the 22 GeneralSheaf runs
# are run below; the 21 JointSheafParams runs are documented but commented out.
# Generated from Bochner_code/TRAINING_RUNS.md — see it for where each
# command comes from. Run from the repository root, with the environment active:
#     bash exp/scripts/paper/train_forman_models.sh
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
    --save_norms=False \
    --save_others=True

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
    --save_norms=False \
    --save_others=True

# Cora — GeneralSheaf, normalised, d=3, 3 layers
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
    --save_norms=False \
    --save_others=True

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
    --save_norms=False \
    --save_others=True

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
    --save_norms=False \
    --save_others=True

# PubMed — GeneralSheaf, normalised, d=5, 4 layers
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
    --save_norms=False \
    --save_others=True

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
    --save_norms=False \
    --save_others=True

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
    --save_norms=False \
    --save_others=True

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
    --save_norms=False \
    --save_others=True

# Amazon Ratings — GeneralSheaf, unnormalised, d=5, 5 layers
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

# Citeseer — GeneralSheaf, unnormalised, d=5, 4 layers
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

# Cora — GeneralSheaf, unnormalised, d=3, 3 layers
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

# Cora — GeneralSheaf, unnormalised, d=5, 3 layers
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

# Cornell — GeneralSheaf, unnormalised, d=2, 2 layers
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

# Minesweeper — GeneralSheaf, unnormalised, d=5, 6 layers
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

# PubMed — GeneralSheaf, unnormalised, d=5, 4 layers
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

# Questions — GeneralSheaf, unnormalised, d=5, 4 layers
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

# Roman Empire — GeneralSheaf, unnormalised, d=5, 5 layers
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

# Tolokers — GeneralSheaf, unnormalised, d=3, 4 layers
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

# Wisconsin — GeneralSheaf, unnormalised, d=2, 2 layers
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

# ---------------------------------------------------------------------------
# The 21 JointSheafParams runs below are recorded for reference only and are
# commented out: the joint sheaf diffusion model (Hernandez Caralt et al., 2024)
# is not included in this release (see models/unavailable.py). Their outputs are
# in the Zenodo record listed in the README, from which the analysis scripts
# rebuild the Joint tables.
# ---------------------------------------------------------------------------

# # Amazon Ratings — JointSheafParams (identity first maps), unnormalised, d=5, 5 layers
# python -m exp.run \
#     --dataset=amazon_ratings \
#     --model=JointSheafParams \
#     --normalised=False \
#     --d=5 \
#     --layers=5 \
#     --hidden_channels=16 \
#     --epochs=500 \
#     --weight_decay=0.0006914841722570725 \
#     --sheaf_decay=0.00031764232712732976 \
#     --dropout=0.3 \
#     --input_dropout=0.2 \
#     --edge_weights=False \
#     --dual_left_linear=True \
#     --dual_linear=False \
#     --dual_normalised=False \
#     --dual_right_linear=True \
#     --checkpoint_epochs=0,1,5,15,50,200 \
#     --save_norms=False

# # Citeseer — JointSheafParams (identity first maps), unnormalised, d=5, 4 layers
# python -m exp.run \
#     --dataset=citeseer \
#     --model=JointSheafParams \
#     --normalised=False \
#     --d=5 \
#     --layers=4 \
#     --hidden_channels=16 \
#     --epochs=500 \
#     --weight_decay=0.0006914841722570725 \
#     --sheaf_decay=0.00031764232712732976 \
#     --dropout=0.2 \
#     --input_dropout=0.2 \
#     --edge_weights=False \
#     --dual_left_linear=True \
#     --dual_linear=False \
#     --dual_normalised=False \
#     --dual_right_linear=True \
#     --checkpoint_epochs=0,1,5,15,50,200 \
#     --save_norms=False

# # Cora — JointSheafParams (identity first maps), unnormalised, d=3, 3 layers
# python -m exp.run \
#     --dataset=cora \
#     --model=JointSheafParams \
#     --normalised=False \
#     --d=3 \
#     --layers=3 \
#     --hidden_channels=16 \
#     --epochs=500 \
#     --weight_decay=0.0006914841722570725 \
#     --sheaf_decay=0.00031764232712732976 \
#     --dropout=0.5 \
#     --input_dropout=0.2 \
#     --edge_weights=False \
#     --dual_left_linear=True \
#     --dual_linear=False \
#     --dual_normalised=False \
#     --dual_right_linear=True \
#     --checkpoint_epochs=0,1,5,15,50,200 \
#     --save_norms=False

# # Cornell — JointSheafParams (identity first maps), unnormalised, d=2, 2 layers
# python -m exp.run \
#     --dataset=cornell \
#     --model=JointSheafParams \
#     --normalised=False \
#     --d=2 \
#     --layers=2 \
#     --hidden_channels=16 \
#     --epochs=500 \
#     --lr=0.02 \
#     --weight_decay=0.0006914841722570725 \
#     --sheaf_decay=0.00031764232712732976 \
#     --dropout=0.5 \
#     --input_dropout=0.2 \
#     --dual_left_linear=True \
#     --dual_linear=False \
#     --dual_normalised=False \
#     --dual_right_linear=True \
#     --checkpoint_epochs=0,1,5,15,50,200 \
#     --save_norms=False

# # Minesweeper — JointSheafParams (identity first maps), unnormalised, d=5, 6 layers
# python -m exp.run \
#     --dataset=minesweeper \
#     --model=JointSheafParams \
#     --normalised=False \
#     --d=5 \
#     --layers=6 \
#     --hidden_channels=16 \
#     --epochs=500 \
#     --weight_decay=0.0006914841722570725 \
#     --sheaf_decay=0.00031764232712732976 \
#     --dropout=0.3 \
#     --input_dropout=0.2 \
#     --dual_left_linear=True \
#     --dual_linear=False \
#     --dual_normalised=False \
#     --dual_right_linear=True \
#     --checkpoint_epochs=0,1,5,15,50,200 \
#     --save_norms=False

# # PubMed — JointSheafParams (identity first maps), unnormalised, d=5, 4 layers
# python -m exp.run \
#     --dataset=pubmed \
#     --model=JointSheafParams \
#     --normalised=False \
#     --d=5 \
#     --layers=4 \
#     --hidden_channels=16 \
#     --epochs=500 \
#     --weight_decay=0.0006914841722570725 \
#     --sheaf_decay=0.00031764232712732976 \
#     --dropout=0.3 \
#     --input_dropout=0.2 \
#     --dual_left_linear=True \
#     --dual_linear=False \
#     --dual_normalised=False \
#     --dual_right_linear=True \
#     --checkpoint_epochs=0,1,5,15,50,200 \
#     --save_norms=False

# # Questions — JointSheafParams (identity first maps), unnormalised, d=5, 4 layers
# python -m exp.run \
#     --dataset=questions \
#     --model=JointSheafParams \
#     --normalised=False \
#     --d=5 \
#     --layers=4 \
#     --hidden_channels=16 \
#     --epochs=500 \
#     --weight_decay=0.0006914841722570725 \
#     --sheaf_decay=0.00031764232712732976 \
#     --dropout=0.2 \
#     --input_dropout=0.2 \
#     --edge_weights=False \
#     --dual_left_linear=True \
#     --dual_linear=False \
#     --dual_normalised=False \
#     --dual_right_linear=True \
#     --checkpoint_epochs=0,1,5,15,50,200 \
#     --save_norms=False

# # Roman Empire — JointSheafParams (identity first maps), unnormalised, d=5, 5 layers
# python -m exp.run \
#     --dataset=roman_empire \
#     --model=JointSheafParams \
#     --normalised=False \
#     --d=5 \
#     --layers=5 \
#     --hidden_channels=16 \
#     --epochs=500 \
#     --weight_decay=0.0006914841722570725 \
#     --sheaf_decay=0.00031764232712732976 \
#     --dropout=0.3 \
#     --input_dropout=0.2 \
#     --edge_weights=False \
#     --dual_left_linear=True \
#     --dual_linear=False \
#     --dual_normalised=False \
#     --dual_right_linear=True \
#     --checkpoint_epochs=0,1,5,15,50,200 \
#     --save_norms=False

# # Tolokers — JointSheafParams (identity first maps), unnormalised, d=3, 4 layers
# python -m exp.run \
#     --dataset=tolokers \
#     --model=JointSheafParams \
#     --normalised=False \
#     --d=3 \
#     --layers=4 \
#     --hidden_channels=16 \
#     --epochs=500 \
#     --weight_decay=0.0006914841722570725 \
#     --sheaf_decay=0.00031764232712732976 \
#     --dropout=0.3 \
#     --input_dropout=0.2 \
#     --edge_weights=False \
#     --dual_left_linear=True \
#     --dual_linear=False \
#     --dual_normalised=False \
#     --dual_right_linear=True \
#     --checkpoint_epochs=0,1,5,15,50,200 \
#     --save_norms=False

# # Wisconsin — JointSheafParams (identity first maps), unnormalised, d=2, 2 layers
# python -m exp.run \
#     --dataset=wisconsin \
#     --model=JointSheafParams \
#     --normalised=False \
#     --d=2 \
#     --layers=2 \
#     --hidden_channels=16 \
#     --epochs=500 \
#     --lr=0.02 \
#     --weight_decay=0.0006685729356079199 \
#     --sheaf_decay=0.0006685729356079199 \
#     --dropout=0.7276458263736642 \
#     --edge_weights=False \
#     --dual_left_linear=True \
#     --dual_linear=False \
#     --dual_normalised=False \
#     --dual_right_linear=True \
#     --checkpoint_epochs=0,1,5,15,50,200 \
#     --save_norms=False

# # Amazon Ratings — JointSheafParams (identity first maps), normalised, d=2, 3 layers
# python -m exp.run \
#     --dataset=amazon_ratings \
#     --model=JointSheafParams \
#     --normalised=True \
#     --d=2 \
#     --layers=3 \
#     --hidden_channels=16 \
#     --epochs=500 \
#     --weight_decay=0.0006914841722570725 \
#     --sheaf_decay=0.00031764232712732976 \
#     --dropout=0.3 \
#     --input_dropout=0.2 \
#     --edge_weights=False \
#     --dual_left_linear=True \
#     --dual_linear=False \
#     --dual_right_linear=True \
#     --learnt_map_type=general \
#     --checkpoint_epochs=0,1,5,15,50,200 \
#     --save_norms=False

# # Citeseer — JointSheafParams (identity first maps), normalised, d=3, 3 layers
# python -m exp.run \
#     --dataset=citeseer \
#     --model=JointSheafParams \
#     --normalised=True \
#     --d=3 \
#     --layers=3 \
#     --hidden_channels=16 \
#     --epochs=500 \
#     --weight_decay=0.0006914841722570725 \
#     --sheaf_decay=0.00031764232712732976 \
#     --dropout=0.2 \
#     --input_dropout=0.2 \
#     --edge_weights=False \
#     --dual_left_linear=True \
#     --dual_linear=False \
#     --dual_right_linear=True \
#     --learnt_map_type=general \
#     --checkpoint_epochs=0,1,5,15,50,200 \
#     --save_norms=False

# # Cora — JointSheafParams (identity first maps), normalised, d=2, 3 layers
# python -m exp.run \
#     --dataset=cora \
#     --model=JointSheafParams \
#     --normalised=True \
#     --d=2 \
#     --layers=3 \
#     --hidden_channels=16 \
#     --epochs=500 \
#     --weight_decay=0.0006914841722570725 \
#     --sheaf_decay=0.00031764232712732976 \
#     --dropout=0.5 \
#     --input_dropout=0.2 \
#     --edge_weights=False \
#     --dual_left_linear=True \
#     --dual_linear=False \
#     --dual_right_linear=True \
#     --learnt_map_type=general \
#     --checkpoint_epochs=0,1,5,15,50,200 \
#     --save_norms=False

# # Cora — JointSheafParams (identity first maps), normalised, d=3, 3 layers
# python -m exp.run \
#     --dataset=cora \
#     --model=JointSheafParams \
#     --normalised=True \
#     --d=3 \
#     --layers=3 \
#     --hidden_channels=16 \
#     --epochs=500 \
#     --weight_decay=0.0006914841722570725 \
#     --sheaf_decay=0.00031764232712732976 \
#     --dropout=0.5 \
#     --input_dropout=0.2 \
#     --edge_weights=False \
#     --dual_left_linear=True \
#     --dual_linear=False \
#     --dual_right_linear=True \
#     --learnt_map_type=general \
#     --checkpoint_epochs=0,1,5,15,50,200 \
#     --save_norms=False

# # Cornell — JointSheafParams (identity first maps), normalised, d=2, 2 layers
# python -m exp.run \
#     --dataset=cornell \
#     --model=JointSheafParams \
#     --normalised=True \
#     --d=2 \
#     --layers=2 \
#     --hidden_channels=16 \
#     --epochs=500 \
#     --lr=0.02 \
#     --weight_decay=0.0006914841722570725 \
#     --sheaf_decay=0.00031764232712732976 \
#     --dropout=0.5 \
#     --input_dropout=0.2 \
#     --dual_left_linear=True \
#     --dual_linear=False \
#     --dual_right_linear=True \
#     --learnt_map_type=general \
#     --checkpoint_epochs=0,1,5,15,50,200 \
#     --save_norms=False

# # Minesweeper — JointSheafParams (identity first maps), normalised, d=3, 3 layers
# python -m exp.run \
#     --dataset=minesweeper \
#     --model=JointSheafParams \
#     --normalised=True \
#     --d=3 \
#     --layers=3 \
#     --hidden_channels=16 \
#     --epochs=500 \
#     --weight_decay=0.0006914841722570725 \
#     --sheaf_decay=0.00031764232712732976 \
#     --dropout=0.3 \
#     --input_dropout=0.2 \
#     --dual_left_linear=True \
#     --dual_linear=False \
#     --dual_right_linear=True \
#     --learnt_map_type=general \
#     --checkpoint_epochs=0,1,5,15,50,200 \
#     --save_norms=False

# # PubMed — JointSheafParams (identity first maps), normalised, d=2, 3 layers
# python -m exp.run \
#     --dataset=pubmed \
#     --model=JointSheafParams \
#     --normalised=True \
#     --d=2 \
#     --layers=3 \
#     --hidden_channels=16 \
#     --epochs=500 \
#     --weight_decay=0.0006914841722570725 \
#     --sheaf_decay=0.00031764232712732976 \
#     --dropout=0.3 \
#     --input_dropout=0.2 \
#     --dual_left_linear=True \
#     --dual_linear=False \
#     --dual_right_linear=True \
#     --learnt_map_type=general \
#     --checkpoint_epochs=0,1,5,15,50,200 \
#     --save_norms=False

# # PubMed — JointSheafParams (identity first maps), normalised, d=3, 3 layers
# python -m exp.run \
#     --dataset=pubmed \
#     --model=JointSheafParams \
#     --normalised=True \
#     --d=3 \
#     --layers=3 \
#     --hidden_channels=16 \
#     --epochs=500 \
#     --weight_decay=0.0006914841722570725 \
#     --sheaf_decay=0.00031764232712732976 \
#     --dropout=0.3 \
#     --input_dropout=0.2 \
#     --dual_left_linear=True \
#     --dual_linear=False \
#     --dual_right_linear=True \
#     --learnt_map_type=general \
#     --checkpoint_epochs=0,1,5,15,50,200 \
#     --save_norms=False

# # Questions — JointSheafParams (identity first maps), normalised, d=2, 4 layers
# python -m exp.run \
#     --dataset=questions \
#     --model=JointSheafParams \
#     --normalised=True \
#     --d=2 \
#     --layers=4 \
#     --hidden_channels=16 \
#     --epochs=500 \
#     --weight_decay=0.0006914841722570725 \
#     --sheaf_decay=0.00031764232712732976 \
#     --dropout=0.2 \
#     --input_dropout=0.2 \
#     --edge_weights=False \
#     --dual_left_linear=True \
#     --dual_linear=False \
#     --dual_right_linear=True \
#     --learnt_map_type=general \
#     --checkpoint_epochs=0,1,5,15,50,200 \
#     --save_norms=False

# # Roman Empire — JointSheafParams (identity first maps), normalised, d=3, 3 layers
# python -m exp.run \
#     --dataset=roman_empire \
#     --model=JointSheafParams \
#     --normalised=True \
#     --d=3 \
#     --layers=3 \
#     --hidden_channels=16 \
#     --epochs=500 \
#     --weight_decay=0.0006914841722570725 \
#     --sheaf_decay=0.00031764232712732976 \
#     --dropout=0.3 \
#     --input_dropout=0.2 \
#     --edge_weights=False \
#     --dual_left_linear=True \
#     --dual_linear=False \
#     --dual_right_linear=True \
#     --learnt_map_type=general \
#     --checkpoint_epochs=0,1,5,15,50,200 \
#     --save_norms=False

# # Wisconsin — JointSheafParams (identity first maps), normalised, d=2, 2 layers
# python -m exp.run \
#     --dataset=wisconsin \
#     --model=JointSheafParams \
#     --normalised=True \
#     --d=2 \
#     --layers=2 \
#     --hidden_channels=16 \
#     --epochs=500 \
#     --lr=0.02 \
#     --weight_decay=0.0006685729356079199 \
#     --sheaf_decay=0.0006685729356079199 \
#     --dropout=0.7276458263736642 \
#     --edge_weights=False \
#     --dual_left_linear=True \
#     --dual_linear=False \
#     --dual_right_linear=True \
#     --learnt_map_type=general \
#     --checkpoint_epochs=0,1,5,15,50,200 \
#     --save_norms=False
