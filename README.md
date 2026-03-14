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

## Results

### ImageNet-VidVRD (VidVRD-II backbone)
| Method | mAP | R@50 | R@100 | P@1 | P@5 | P@10 |
|--------|-----|------|-------|-----|-----|------|
| VidVRD-II (baseline) | 22.94 | 9.95 | 10.69 | 70.50 | 52.90 | 40.60 |
| DWMH-SGG (ours) | **23.42** | **9.95** | **10.73** | **71.50** | **53.30** | **40.60** |

### VidOR (VidVRD-II backbone)
| Method | mAP | R@50 | R@100 | P@1 | P@5 |
|--------|-----|------|-------|-----|-----|
| VidVRD-II (baseline) | X.XX | X.XX | X.XX | X.XX | X.XX |
| DWMH-SGG (ours) | **X.XX** | **X.XX** | **X.XX** | **X.XX** | **X.XX** |

### Action Genome (TRACE backbone)
| Task | Metric | Baseline | DWMH-SGG |
|------|--------|----------|----------|
| PredCls | R@50 | X.XX | **X.XX** |
| SGGen | R@50 | X.XX | **X.XX** |

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
