# DWMH-SGG: Directed Weighted Motif Hypergraphs with Spectral Propagation for Spatio-Temporal Scene Graph Generation

[![Paper](https://img.shields.io/badge/Paper-Neurocomputing-blue)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

## Overview
DWMH-SGG is a post-hoc, training-free hypergraph refinement framework 
for spatio-temporal scene graph generation. It constructs a directed 
weighted incidence matrix encoding directional role and structural 
centrality, weights hyperedges by prediction confidence, captures 
inter-motif dependencies through a Jaccard co-occurrence tensor, and 
refines relation confidence via closed-form spectral propagation.



## Installation
pip install -r requirements.txt

## Quick Start

### Step 1 — Set up backbone
Follow instructions in backbones/README.md to install 
VidVRD-II and TRACE and generate baseline predictions.

### Step 2 — Run DWMH-SGG on VidVRD
bash scripts/run_vidvrd.sh

### Step 3 — Run on all datasets
bash scripts/run_all_experiments.sh

## Manual Usage
python dwmh_sgg/main.py \
  --prediction outputs/vidvrd/baseline_relation_prediction.json \
  --groundtruth data/vidvrd/annotations/test_gt.json \
  --output outputs/vidvrd/dwmh_refined_prediction.json \
  --tau 3 --beta 0.10 --gamma 0.50 \
  --delta_det 0.40 --delta_tag 0.30

## Hyperparameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| tau | 3 | Min temporal overlap (frames) for motif formation |
| beta | 0.10 | Jaccard co-occurrence influence weight |
| gamma | 0.50 | Laplacian propagation strength |
| delta_det | 0.40 | Score blending weight for detection |
| delta_tag | 0.30 | Score blending weight for tagging |

## Citation
@article{aliloua2025dwmhsgg,
  title={DWMH-SGG: Directed Weighted Motif Hypergraphs with 
         Spectral Propagation for Spatio-Temporal Scene Graph Generation},
  author={Aliloua, Marouane and Bhuyan, Bikram Pratim and 
          Fissounec, Rachida and Ramdane-Cherif, Amar},
  journal={Neurocomputing},
  year={2025}
}
