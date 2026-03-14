"""Hyperparameter sensitivity analysis for DWMH-SGG.

Varies one parameter at a time (keeping others at best values) and records
mAP, P@1, Recall@50 for each setting.  Generates individual plots + 2x2
combined figure, and prints a robustness summary table.

Usage (from E:/PHD/ST-SGG_Models/DWMH-SGG/):
    python sensitivity_analysis.py
"""

import csv
import json
import os
import sys
from collections import defaultdict, OrderedDict

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR      = os.path.dirname(BASE_DIR)
BASELINE_PATH = os.path.join(ROOT_DIR, "imagenet-vidvrd-baseline-output",
                             "models", "3step_prop_wd0.01", "video_relations.json")
VIDVRD_PATH   = os.path.join(ROOT_DIR, "VidVRD-II")
DATASET_PATH  = os.path.join(ROOT_DIR, "imagenet-vidvrd-dataset")
OUT_DIR       = BASE_DIR

# ── Best (canonical) hyper-parameters ─────────────────────────────────────────
BEST = dict(tau=3, beta=0.10, gamma=0.50, delta=0.40, delta_tagging=0.30)

# ── Sweep grids ────────────────────────────────────────────────────────────────
SWEEPS = {
    "tau":   [1, 2, 3, 4, 5],
    "gamma": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "delta": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    "beta":  [0.01, 0.05, 0.10, 0.15, 0.20, 0.30],
}

PARAM_LABELS = {
    "tau":   r"$\tau$ (min. temporal overlap, frames)",
    "gamma": r"$\gamma$ (Laplacian smoothing strength)",
    "delta": r"$\delta_{det}$ (detection blending weight)",
    "beta":  r"$\beta$ (Jaccard re-weighting strength)",
}

PARAM_TITLES = {
    "tau":   r"Sensitivity to $\tau$",
    "gamma": r"Sensitivity to $\gamma$",
    "delta": r"Sensitivity to $\delta_{det}$",
    "beta":  r"Sensitivity to $\beta$",
}

METRICS_DISPLAY = {"mAP": "mAP", "P@1": "P@1", "R@50": "Recall@50"}


# ── VidVRD-II imports ──────────────────────────────────────────────────────────
def load_vidvrd():
    if VIDVRD_PATH not in sys.path:
        sys.path.insert(0, VIDVRD_PATH)
    from dataset import VidVRD
    from evaluation.visual_relation_detection import (
        eval_detection_scores, eval_tagging_scores)
    from evaluation.common import voc_ap
    pred_meta = json.load(open(BASELINE_PATH))
    normalize_coords = pred_meta["version"] >= "VERSION 2.1"
    dataset = VidVRD(DATASET_PATH, os.path.join(DATASET_PATH, "videos"),
                     ["train", "test"], normalize_coords=normalize_coords)
    return dataset, eval_detection_scores, eval_tagging_scores, voc_ap


# ── Evaluation helpers ─────────────────────────────────────────────────────────
def swap_scores(results):
    import copy
    out = {}
    for vid, rels in results.items():
        new_rels = []
        for rel in rels:
            if "tagging_score" in rel:
                r = copy.copy(rel)
                r["score"] = rel["tagging_score"]
                new_rels.append(r)
            else:
                new_rels.append(rel)
        out[vid] = new_rels
    return out


def evaluate(groundtruth, det_results, tag_results,
             eval_detection_scores, eval_tagging_scores, voc_ap):
    """Return dict with mAP, P@1, R@50."""
    video_ap   = {}
    tot_scores = defaultdict(list)
    tot_tp     = defaultdict(list)
    prec_at_n  = defaultdict(list)
    tot_gt     = 0
    det_nret   = [50]
    tag_nret   = [1]

    for vid, gt_rels in groundtruth.items():
        if not gt_rels:
            continue
        tot_gt += len(gt_rels)
        dp = det_results.get(vid, [])
        tp = tag_results.get(vid, [])

        det_prec, det_rec, det_sc = eval_detection_scores(gt_rels, dp, 0.5)
        video_ap[vid] = float(voc_ap(det_rec, det_prec))
        is_tp = np.isfinite(det_sc)
        for nre in det_nret:
            cut = min(nre, det_sc.size)
            tot_scores[nre].append(det_sc[:cut])
            tot_tp[nre].append(is_tp[:cut])

        tag_prec, _, _ = eval_tagging_scores(gt_rels, tp, 1)
        prec_at_n[1].append(tag_prec[0])

    mean_ap = float(np.mean(list(video_ap.values()))) * 100.0

    sc   = np.concatenate(tot_scores[50])
    tp50 = np.concatenate(tot_tp[50])
    idx  = np.argsort(sc)[::-1]
    tp50 = tp50[idx]
    cum  = np.cumsum(tp50).astype(np.float32)
    r50  = float(cum[-1] / max(tot_gt, np.finfo(np.float32).eps)) * 100.0

    p1 = float(np.mean(prec_at_n[1])) * 100.0

    return {"mAP": mean_ap, "P@1": p1, "R@50": r50}


# ── Single sweep run ───────────────────────────────────────────────────────────
def run_sweep(param_name, values, baseline_meta, groundtruth,
              eval_detection_scores, eval_tagging_scores, voc_ap):
    from dwmh_sgg.refinement import process_prediction
    records = []
    for val in values:
        kwargs = dict(BEST)
        kwargs[param_name] = val
        refined_meta, _ = process_prediction(
            baseline_meta,
            tau=kwargs["tau"],
            beta=kwargs["beta"],
            gamma=kwargs["gamma"],
            delta=kwargs["delta"],
            delta_tagging=kwargs["delta_tagging"],
            verbose=False,
        )
        res     = refined_meta["results"]
        tag_res = swap_scores(res)
        scores  = evaluate(groundtruth, res, tag_res,
                           eval_detection_scores, eval_tagging_scores, voc_ap)
        records.append({param_name: val, **scores})
        print(f"  {param_name}={val}  mAP={scores['mAP']:.2f}  "
              f"P@1={scores['P@1']:.2f}  R@50={scores['R@50']:.2f}")
    return records


# ── Plotting ───────────────────────────────────────────────────────────────────
METRIC_COLORS = {"mAP": "#1f77b4", "P@1": "#d62728", "R@50": "#2ca02c"}
METRIC_MARKERS = {"mAP": "o", "P@1": "s", "R@50": "^"}


def make_axes(ax, records, param_name, best_val):
    """Draw the three metric lines onto ax."""
    xs = [r[param_name] for r in records]
    for metric in ["mAP", "P@1", "R@50"]:
        ys = [r[metric] for r in records]
        ax.plot(xs, ys,
                color=METRIC_COLORS[metric],
                marker=METRIC_MARKERS[metric],
                markersize=5,
                linewidth=1.8,
                label=METRICS_DISPLAY[metric])

    ax.axvline(best_val, color="#cc0000", linewidth=1.4,
               linestyle="--", label=f"best ({best_val})", zorder=3)

    ax.set_xlabel(PARAM_LABELS[param_name], fontsize=11)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_title(PARAM_TITLES[param_name], fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.85)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_facecolor("white")
    ax.tick_params(labelsize=9)
    if param_name == "tau":
        ax.set_xticks(xs)


def save_individual(records, param_name, best_val):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("white")
    make_axes(ax, records, param_name, best_val)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"sensitivity_{param_name}.png")
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved {out}")


def save_combined(all_records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor("white")

    order = ["tau", "gamma", "delta", "beta"]
    for ax, param_name in zip(axes.flat, order):
        make_axes(ax, all_records[param_name], param_name, BEST[param_name])

    fig.suptitle("DWMH-SGG Hyperparameter Sensitivity Analysis",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "sensitivity_analysis.png")
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nCombined figure saved to {out}")


# ── Summary table ──────────────────────────────────────────────────────────────
def print_summary(all_records):
    col = [14, 20, 12, 12, 12, 12, 12, 12]
    hdr = (f"{'Parameter':<{col[0]}}  {'Range':<{col[1]}}  "
           f"{'Best val':>{col[2]}}  "
           f"{'Best mAP':>{col[3]}}  {'Worst mAP':>{col[4]}}  {'DeltamAP':>{col[5]}}  "
           f"{'Best P@1':>{col[6]}}  {'Delta P@1':>{col[7]}}")
    print()
    print("=" * len(hdr))
    print("  Sensitivity robustness summary")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for param_name in ["tau", "gamma", "delta", "beta"]:
        recs  = all_records[param_name]
        vals  = [r[param_name] for r in recs]
        maps  = [r["mAP"]  for r in recs]
        p1s   = [r["P@1"]  for r in recs]

        best_map_val  = vals[int(np.argmax(maps))]
        best_map      = max(maps)
        worst_map     = min(maps)
        delta_map     = best_map - worst_map
        best_p1_val   = vals[int(np.argmax(p1s))]
        best_p1       = max(p1s)
        worst_p1      = min(p1s)
        delta_p1      = best_p1 - worst_p1

        range_str = f"[{min(vals)}, {max(vals)}]"
        # best value = canonical best
        canonical = BEST[param_name]
        row = (f"{param_name:<{col[0]}}  {range_str:<{col[1]}}  "
               f"{canonical:>{col[2]}}  "
               f"{best_map:>{col[3]}.2f}  {worst_map:>{col[4]}.2f}  "
               f"{delta_map:>{col[5]}.2f}  "
               f"{best_p1:>{col[6]}.2f}  {delta_p1:>{col[7]}.2f}")
        print(row)
        rows.append({
            "param": param_name,
            "range_min": min(vals),
            "range_max": max(vals),
            "best_val": canonical,
            "best_mAP": best_map,
            "worst_mAP": worst_map,
            "delta_mAP": delta_map,
            "best_P1": best_p1,
            "worst_P1": worst_p1,
            "delta_P1": delta_p1,
        })
    print()
    return rows


# ── CSV export ─────────────────────────────────────────────────────────────────
def save_csvs(all_records, summary_rows):
    # Per-parameter sweep CSVs
    for param_name, recs in all_records.items():
        path = os.path.join(OUT_DIR, f"sensitivity_{param_name}.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[param_name, "mAP", "P@1", "R@50"])
            w.writeheader()
            for r in recs:
                w.writerow({k: f"{v:.4f}" if isinstance(v, float) else v
                            for k, v in r.items()})
        print(f"  CSV saved: {path}")

    # Summary CSV
    path = os.path.join(OUT_DIR, "sensitivity_summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        for r in summary_rows:
            w.writerow({k: f"{v:.4f}" if isinstance(v, float) else v
                        for k, v in r.items()})
    print(f"  Summary CSV saved: {path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("Loading dataset...")
    dataset, eval_det, eval_tag, voc_ap = load_vidvrd()
    groundtruth = {vid: dataset.get_relation_insts(vid)
                   for vid in dataset.get_index("test")}

    print("Loading baseline prediction...")
    baseline_meta = json.load(open(BASELINE_PATH))

    all_records = {}

    for param_name, values in SWEEPS.items():
        print(f"\n-- Sweeping {param_name}: {values}")
        recs = run_sweep(param_name, values, baseline_meta, groundtruth,
                         eval_det, eval_tag, voc_ap)
        all_records[param_name] = recs
        save_individual(recs, param_name, BEST[param_name])

    save_combined(all_records)
    summary_rows = print_summary(all_records)
    save_csvs(all_records, summary_rows)


if __name__ == "__main__":
    main()
