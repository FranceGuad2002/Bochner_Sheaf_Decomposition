# Bochner decomposition for cellular sheaves — code

Code and data for the paper *"A Bochner–Weitzenböck Decomposition for Cellular Sheaf
Laplacians, Curvature and Neural Sheaf Diffusion"*.

This repository is a **fork of [twitter-research/neural-sheaf-diffusion](https://github.com/twitter-research/neural-sheaf-diffusion)**
(Bodnar et al., NeurIPS 2022), whose original README follows below unchanged. The upstream
code provides the sheaf neural network models and the training loop; this fork adds the
instrumentation needed to observe a trained network, and the analysis the paper reports.

## How to use this code

Start at [`Bochner_code/README.md`](Bochner_code/README.md). It explains three levels of
reproduction:

- **A. Regenerate every table in the paper:** about a minute, on any machine, with
  `pip install` of four packages. No GPU, no training, no download.
- **B. Rerun the analysis** from the trained models' raw outputs, archived on
  [Zenodo](https://zenodo.org/records/23080428) (6.8 GB).
- **C. Retrain the models**, using the exact command for each of the 53 models in
  [`TRAINING_RUNS.md`](Bochner_code/TRAINING_RUNS.md), plus the identity-sheaf control
  models.

**On Windows, unpack or clone the repository into a short folder such as `C:\nsd`**: some
file paths are long, and Windows by default refuses paths over 260 characters.

## Layout

```
Bochner_code/                    everything specific to the paper   <- start here
  conviction/                    node importance in a trained network (Bochner term)
  forman_profile/                curvature-block spectra (Forman term)
  TRAINING_RUNS.md               every trained model, with the command that produced it
  environments/                  the exact conda environments the results were produced with
exp/ lib/ models/ utils/         the upstream code, with the additions listed below
exp/scripts/                     upstream example scripts, plus the paper's drivers
                                 (exp/scripts/paper/ for training, the rest for analysis)
datasets/  splits/               benchmark data (shipped here) and the 10-fold splits
results/                         raw outputs of the trained models (on Zenodo, not in git)
```

## What this fork changes in the upstream code

The changes to the upstream code are almost entirely additions, and each modified upstream
file says so in its header. The paper's analysis code (`Bochner_code/`) and the drivers in
`exp/scripts/` (except the upstream example scripts) are new.

| File | What it adds |
|---|---|
| `exp/run.py` | per-epoch checkpoints and the code that saves Forman eigenvalues, node norms, confidences, predictions, saliencies and weights into `results/` |
| `utils/conviction.py` (new) | the shared conviction backend: loading checkpoints, ranking nodes, and the three interventions (masking, gating, pruning) |
| `models/disc_models.py` | the Bochner, flat Bochner and identity-sheaf models; Laplacian and node-norm instrumentation of the diffusion layers; the message-gating hook used by the conviction interventions |
| `models/laplacian_builders.py` | the flat Bochner Laplacian builder and the switch between normalised and unnormalised Laplacians |
| `models/sheaf_models.py` | the per-node and diagonal edge learners used by the Bochner models |
| `utils/heterophilic.py` | loaders for the newer heterophilic benchmarks (Roman Empire, Amazon Ratings, Minesweeper, Tolokers, Questions) |
| `exp/parser.py` | the `--save_*`, `--save_others`, `--maps_lr` and checkpoint flags |
| `models/test_disc_models.py` | tests for the identity-sheaf model |
| `models/sheaf_base.py`, `models/cont_models.py` | small adjustments to the model arguments |

**The upstream sheaf Laplacian construction is unchanged:** `lib/`, `utils/classic.py`,
`definitions.py`, `main.py`, `models/orthogonal.py`, `models/positional_encodings.py`,
`config/`, the environment files and the upstream example scripts are identical to the
original.

## License

The upstream code is Apache 2.0 and remains so, and this fork is distributed under the same
license; see `LICENSE`. The modified upstream files are marked in their headers and listed
above.

---
---

*Original upstream README below.*

[![example workflow](https://github.com/twitter-research/neural-sheaf-diffusion/actions/workflows/python-tests.yml/badge.svg)](https://github.com/twitter-research/neural-sheaf-diffusion/actions) 
[![License: MIT](https://img.shields.io/badge/License-Apache%202-green.svg)](https://github.com/twitter-research/neural-sheaf-diffusion/blob/main/LICENSE)

# Neural Sheaf Diffusion

This repository contains the official code for the paper 
**[Neural Sheaf Diffusion: A Topological Perspective on Heterophily and Oversmoothing in GNNs](https://arxiv.org/abs/2202.04579) (NeurIPS 2022)**.

![Sheaf Neural Networks](figures/sheaf.png)

## Getting started

We used `CUDA 10.2` for this project. To set up the environment, run the following command:

```bash
conda env create --file=environment_gpu.yml
conda activate nsd
```
For using another `CUDA` version, modify the version specified inside `environment_gpu.yml`. If you like to run 
the code in a CPU-only environment, then use `environment_cpu`. 

To make sure that everything is set up correctly, you can run all the tests using:
```bash
pytest -v .
```

## Running experiments

To run the experiments without a Weights & Biases (wandb) account, first disable `wandb` by running `wandb disabled`. 
Then, for instance, to run the training procedure on `texas`, simply run the example script provided:
```commandline
sh ./exp/scripts/run_texas.sh
```

To run the experiments using `wandb`, first create a [Weights & Biases account](https://wandb.ai/site). Then run the following
commands to log in and follow the displayed instructions:
```bash
wandb online
wandb login
```
Then, you can run the example training procedure on `texas` via:
```bash
export ENTITY=<WANDB_ACCOUNT_ID>
sh ./exp/scripts/run_texas.sh
```
Scripts for the other heterophilic datasets are also provided in `exp/scripts`. 

### Hyperparameter Sweeps

To run a hyperparameter sweep, you will need a `wandb` account. Once you have an account, you can run an example
sweep as follows:
```bash
export ENTITY=<WANDB_ACCOUNT_ID>
wandb sweep --project sheaf config/orth_webkb_sweep.yml
```
This will set up the sweep for a discrete bundle model on the WebKB datasets 
as described in the yaml config at `config/orth_webkb_sweep.yml`.

To run the sweep on a single GPU, simply run the command displayed on screen after running the sweep command above. 
If you like to run the sweep on multiple GPUs, then run the following command by typing in the `SWEEP_ID` received above.
```bash
sh run_sweeps.sh <SWEEP_ID>
```
### Datasets

The WebKB (`texas`, `wisconsin`, `cornell`) and `film` datasets are downloaded on the fly. The
[WikipediaNetwork datasets](https://academic.oup.com/comnet/article/9/2/cnab014/6271062) with the Geom-GCN pre-processing 
can be downloaded from the [Geom-GCN repo](https://github.com/graphdml-uiuc-jlu/geom-gcn/tree/f1fc0d14b3b019c562737240d06ec83b07d16a8f/new_data).
The files for the Planetoid datasets can also be found in the [Geom-GCN repo](https://github.com/graphdml-uiuc-jlu/geom-gcn/tree/master/data). 
The downloaded files must be placed into `datasets/<DATASET_NAME>/raw/`.

## Credits

For attribution in academic contexts, please use the bibtex entry below:
```
@inproceedings{
 bodnar2022neural,
 title={Neural Sheaf Diffusion: A Topological Perspective on Heterophily and Oversmoothing in {GNN}s},
 author={Cristian Bodnar and Francesco Di Giovanni and Benjamin Paul Chamberlain and Pietro Li{\`o} and Michael M. Bronstein},
 booktitle={Advances in Neural Information Processing Systems},
 editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
 year={2022},
 url={https://openreview.net/forum?id=vbPsD-BhOZ}
}
```
