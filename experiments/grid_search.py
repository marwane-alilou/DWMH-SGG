"""Grid search over DWMH-SGG hyper-parameters on the VidVRD test set.

Usage (from E:/PHD/ST-SGG_Models/DWMH-SGG/):

    python grid_search.py

Searches:
    tau   in {1, 2, 3}
    beta  in {0.05, 0.1, 0.2}
    gamma in {0.3, 0.5, 0.7, 1.0}
    delta in {0.2, 0.3, 0.4, 0.5}

Total: 3 x 3 x 4 x 4 = 144 configurations.
"""

import itertools
import json
import logging
import os
import sys
import time

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.dirname(BASE_DIR)
PRED_PATH    = os.path.join(ROOT_DIR, "imagenet-vidvrd-baseline-output",
                            "models", "3step_prop_wd0.01", "video_relations.json")
VIDVRD_PATH  = os.path.join(ROOT_DIR, "VidVRD-II")
DATASET_PATH = os.path.join(ROOT_DIR, "imagenet-vidvrd-dataset")

# ── Grid ──────────────────────────────────────────────────────────────────────
TAU_VALUES   = [1, 2, 3]
BETA_VALUES  = [0.05, 0.1, 0.2]
GAMMA_VALUES = [0.3, 0.5, 0.7, 1.0]
DELTA_VALUES = [0.2, 0.3, 0.4, 0.5]

# ── Metric keys (in evaluate_relation output dict) ────────────────────────────
METRIC_KEYS = [
    ("mAP",        "detection mean AP",      100.0),
    ("Recall@10",  "detection recall@10",    100.0),
    ("Recall@20",  "detection recall@20",    100.0),
    ("Recall@50",  "detection recall@50",    100.0),
    ("Recall@100", "detection recall@100",   100.0),
    ("P@1",        "tagging precision@1",    100.0),
    ("P@5",        "tagging precision@5",    100.0),
    ("P@10",       "tagging precision@10",   100.0),
]

logging.basicConfig(
    level=logging.WARNING,   # suppress per-video noise during grid search
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("grid_search")


def load_vidvrd():
    """Import VidVRD-II modules and load the dataset (done once)."""
    if VIDVRD_PATH not in sys.path:
        sys.path.insert(0, VIDVRD_PATH)
    from dataset import VidVRD
    from evaluation import eval_visual_relation
    pred_meta = json.load(open(PRED_PATH))
    normalize_coords = pred_meta["version"] >= "VERSION 2.1"
    dataset = VidVRD(
        DATASET_PATH,
        os.path.join(DATASET_PATH, "videos"),
        ["train", "test"],
        normalize_coords=normalize_coords,
    )
    return dataset, eval_visual_relation


def extract_scores(eval_dict, setting="overall"):
    """Pull the metrics we care about from an evaluate_relation result dict."""
    s = eval_dict.get(setting, {})
    row = {}
    for short, key, mult in METRIC_KEYS:
        row[short] = s.get(key, float("nan")) * mult
    return row


def run_one(prediction, tau, beta, gamma, delta):
    """Run refinement + eval for a single config. Returns metric dict."""
    from dwmh_sgg.refinement import process_prediction

    refined, _ = process_prediction(
        prediction,
        tau=tau,
        beta=beta,
        gamma=gamma,
        delta=delta,
        dry_run=False,
        verbose=False,
    )
    return refined


def main():
    t0 = time.perf_counter()

    print("Loading baseline prediction …")
    with open(PRED_PATH) as f:
        baseline_pred = json.load(f)
    n_videos = len(baseline_pred["results"])
    print(f"  {n_videos} videos loaded.")

    print("Loading VidVRD-II dataset …")
    dataset, eval_visual_relation = load_vidvrd()

    # Pre-build groundtruth dict (shared across all runs)
    print("Building groundtruth …")
    groundtruth = {}
    for vid in dataset.get_index("test"):
        groundtruth[vid] = dataset.get_relation_insts(vid)

    configs = list(itertools.product(TAU_VALUES, BETA_VALUES, GAMMA_VALUES, DELTA_VALUES))
    n_total = len(configs)
    print(f"\nStarting grid search: {n_total} configurations\n")

    results = []  # list of (cfg_dict, metrics_dict)

    for idx, (tau, beta, gamma, delta) in enumerate(configs, 1):
        t_cfg = time.perf_counter()
        refined = run_one(baseline_pred, tau, beta, gamma, delta)
        scores  = eval_visual_relation(groundtruth, refined["results"])
        metrics = extract_scores({"overall": scores})
        cfg = dict(tau=tau, beta=beta, gamma=gamma, delta=delta)
        results.append((cfg, metrics))
        elapsed = time.perf_counter() - t_cfg
        print(
            f"[{idx:3d}/{n_total}]  tau={tau}  beta={beta:.2f}  gamma={gamma:.2f}  delta={delta:.2f}"
            f"  |  mAP={metrics['mAP']:.2f}  P@1={metrics['P@1']:.2f}"
            f"  ({elapsed:.1f}s)"
        )

    total_elapsed = time.perf_counter() - t0
    print(f"\nGrid search complete in {total_elapsed:.1f}s\n")

    # ── Report ────────────────────────────────────────────────────────────────
    metric_names = [m[0] for m in METRIC_KEYS]
    col_w = 9

    def print_table(ranked, title):
        header_cfg = f"{'Rank':<5}  {'tau':>4}  {'beta':>6}  {'gamma':>6}  {'delta':>6}"
        header_met = "  " + "  ".join(f"{m:>{col_w}}" for m in metric_names)
        sep = "-" * (len(header_cfg) + len(header_met))
        print(f"\n{'='*len(sep)}")
        print(f"  {title}")
        print(f"{'='*len(sep)}")
        print(header_cfg + header_met)
        print(sep)
        for rank, (cfg, met) in enumerate(ranked, 1):
            cfg_str = (f"{rank:<5}  {cfg['tau']:>4}  {cfg['beta']:>6.2f}"
                       f"  {cfg['gamma']:>6.2f}  {cfg['delta']:>6.2f}")
            met_str = "  " + "  ".join(
                f"{met.get(m, float('nan')):>{col_w}.2f}" for m in metric_names
            )
            print(cfg_str + met_str)
        print()

    top5_map = sorted(results, key=lambda x: x[1].get("mAP", -1), reverse=True)[:5]
    top5_p1  = sorted(results, key=lambda x: x[1].get("P@1", -1), reverse=True)[:5]

    print_table(top5_map, "TOP-5 CONFIGURATIONS BY mAP")
    print_table(top5_p1,  "TOP-5 CONFIGURATIONS BY P@1")

    # ── Save full results to JSON ─────────────────────────────────────────────
    out_path = os.path.join(BASE_DIR, "grid_search_results.json")
    serializable = [
        {"config": cfg, "metrics": met}
        for cfg, met in results
    ]
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Full results saved to {out_path}")


if __name__ == "__main__":
    main()
