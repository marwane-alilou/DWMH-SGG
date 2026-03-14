#!/bin/bash
echo "Running DWMH-SGG on VidVRD..."
bash scripts/run_vidvrd.sh

echo "Running DWMH-SGG on VidOR..."
bash scripts/run_vidor.sh

echo "Running DWMH-SGG on Action Genome..."
bash scripts/run_action_genome.sh

echo "Running ablation study..."
python experiments/ablation_components.py

echo "Running sensitivity analysis..."
python experiments/sensitivity_analysis.py

echo "All experiments complete."
```

---

## Three important notes

**1. Do not commit the data or backbone weights to GitHub.** The `data/` and `outputs/` folders should have only README files explaining what goes there. Add them to `.gitignore`:
```
data/vidvrd/videos/
data/vidor/videos/
data/action_genome/videos/
outputs/vidvrd/baseline_relation_prediction.json
outputs/action_genome/baseline_relation_prediction.json
