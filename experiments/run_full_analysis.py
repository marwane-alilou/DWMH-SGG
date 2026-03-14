"""Full runtime + qualitative analysis for DWMH-SGG on VidVRD test set.

Configuration: tau=3, beta=0.10, gamma=0.50, delta_detection=0.40, delta_tagging=0.30

Outputs:
  runtime_analysis.csv
  runtime_analysis.png    (2x2 combined, 300 DPI)
  qualitative_analysis.txt
  score_distribution.png  (300 DPI)
  predicate_analysis.png  (300 DPI)

Run from:  E:/PHD/ST-SGG_Models/DWMH-SGG/
"""

import copy
import csv
import json
import logging
import os
import sys
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR      = os.path.dirname(BASE_DIR)
BASELINE_PATH = os.path.join(ROOT_DIR, "imagenet-vidvrd-baseline-output",
                             "models", "3step_prop_wd0.01", "video_relations.json")
REFINED_PATH  = os.path.join(ROOT_DIR, "imagenet-vidvrd-baseline-output",
                             "models", "3step_prop_wd0.01", "video_relations_dwmh_final.json")
VIDVRD_PATH   = os.path.join(ROOT_DIR, "VidVRD-II")
DATASET_PATH  = os.path.join(ROOT_DIR, "imagenet-vidvrd-dataset")

# ── Hyper-parameters ──────────────────────────────────────────────────────────
TAU            = 3
BETA           = 0.10
GAMMA          = 0.50
DELTA_DET      = 0.40
DELTA_TAG      = 0.30

# ── Output files ──────────────────────────────────────────────────────────────
CSV_OUT          = os.path.join(BASE_DIR, "runtime_analysis.csv")
RUNTIME_PNG      = os.path.join(BASE_DIR, "runtime_analysis.png")
QUAL_TXT         = os.path.join(BASE_DIR, "qualitative_analysis.txt")
SCORE_DIST_PNG   = os.path.join(BASE_DIR, "score_distribution.png")
PREDICATE_PNG    = os.path.join(BASE_DIR, "predicate_analysis.png")

_EPS = 1e-8

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("run_analysis")


# ─────────────────────────────────────────────────────────────────────────────
# Step A: Import DWMH-SGG modules
# ─────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, BASE_DIR)
from dwmh_sgg.motif_extraction import assign_tracklet_ids, extract_all_motifs
from dwmh_sgg.hypergraph import build_weighted_incidence
from dwmh_sgg.laplacian import init_node_scores, compute_laplacian, solve_laplacian
from dwmh_sgg.refinement import blend_scores


def refine_video_timed(relations, tau, beta, gamma, delta, delta_tagging):
    """Run the DWMH-SGG pipeline for one video, recording per-component timing."""

    if len(relations) <= 1:
        return copy.deepcopy(relations), {
            "skipped": True,
            "n_relations": len(relations),
            "n_tracklets": 0,
            "n_chain_motifs": 0,
            "n_triangle_motifs": 0,
            "n_star_motifs": 0,
            "n_hyperedges": 0,
            "time_motif_extraction_ms": 0.0,
            "time_hypergraph_construction_ms": 0.0,
            "time_laplacian_ms": 0.0,
            "time_refinement_ms": 0.0,
            "time_total_ms": 0.0,
        }

    t_total_start = time.perf_counter()

    # Step 1: Tracklet IDs (part of motif extraction timing)
    t0 = time.perf_counter()
    sub_tids, obj_tids, n_tracklets = assign_tracklet_ids(relations)

    # Step 2: Motif extraction
    chains, triangles, stars = extract_all_motifs(relations, sub_tids, obj_tids, tau)
    all_motifs = chains + triangles + stars
    t_motif = (time.perf_counter() - t0) * 1000.0

    use_sparse = len(relations) > 500

    # Step 3: Hypergraph construction
    t0 = time.perf_counter()
    H_w, hyperedges = build_weighted_incidence(
        relations, sub_tids, obj_tids, all_motifs,
        n_tracklets=n_tracklets, beta=beta, use_sparse=use_sparse,
    )
    t_hg = (time.perf_counter() - t0) * 1000.0

    # Step 4: Laplacian
    t0 = time.perf_counter()
    Delta, d_v = compute_laplacian(H_w, use_sparse=use_sparse)
    alpha = init_node_scores(relations, sub_tids, obj_tids, n_tracklets)
    alpha_prime = solve_laplacian(Delta, alpha, gamma, use_sparse=use_sparse)
    zero_deg = d_v < _EPS
    alpha_prime = np.where(zero_deg, alpha, alpha_prime)
    t_lap = (time.perf_counter() - t0) * 1000.0

    # Step 5: Score blending (refinement)
    t0 = time.perf_counter()
    det_scores, tag_scores = blend_scores(
        relations, sub_tids, obj_tids, alpha_prime, delta, delta_tagging
    )
    refined_relations = copy.deepcopy(relations)
    for i, rel in enumerate(refined_relations):
        rel["score"] = det_scores[i]
        if tag_scores is not None:
            rel["tagging_score"] = tag_scores[i]
    t_ref = (time.perf_counter() - t0) * 1000.0

    t_total = (time.perf_counter() - t_total_start) * 1000.0

    stats = {
        "skipped": False,
        "n_relations":  len(relations),
        "n_tracklets":  n_tracklets,
        "n_chain_motifs":    len(chains),
        "n_triangle_motifs": len(triangles),
        "n_star_motifs":     len(stars),
        "n_hyperedges":      len(hyperedges),
        "time_motif_extraction_ms":       t_motif,
        "time_hypergraph_construction_ms": t_hg,
        "time_laplacian_ms":              t_lap,
        "time_refinement_ms":             t_ref,
        "time_total_ms":                  t_total,
        # For qualitative analysis
        "sub_tids":    sub_tids,
        "obj_tids":    obj_tids,
        "alpha_prime": alpha_prime,
        "original_scores": [float(r["score"]) for r in relations],
        "refined_scores":  det_scores,
        "chains":    chains,
        "triangles": triangles,
        "stars":     stars,
    }
    return refined_relations, stats


# ─────────────────────────────────────────────────────────────────────────────
# Step B: Run refinement on all 200 test videos
# ─────────────────────────────────────────────────────────────────────────────
logger.info("Loading baseline predictions from %s", BASELINE_PATH)
with open(BASELINE_PATH, "r") as f:
    prediction = json.load(f)

video_ids = list(prediction["results"].keys())
logger.info("Found %d videos. Running refinement with tau=%d beta=%.2f gamma=%.2f "
            "delta_det=%.2f delta_tag=%.2f",
            len(video_ids), TAU, BETA, GAMMA, DELTA_DET, DELTA_TAG)

all_stats   = {}
all_refined = {}

for i, vid in enumerate(video_ids):
    if (i + 1) % 20 == 0:
        logger.info("  Processing video %d / %d", i + 1, len(video_ids))
    relations = prediction["results"][vid]
    refined_rels, stats = refine_video_timed(
        relations, tau=TAU, beta=BETA, gamma=GAMMA,
        delta=DELTA_DET, delta_tagging=DELTA_TAG,
    )
    all_stats[vid]   = stats
    all_refined[vid] = refined_rels

logger.info("Refinement complete for %d videos.", len(video_ids))

# Save refined predictions
refined_prediction = {"version": prediction["version"], "results": all_refined}
with open(REFINED_PATH, "w") as f:
    json.dump(refined_prediction, f)
logger.info("Refined predictions saved to %s", REFINED_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# Step C: Save runtime_analysis.csv
# ─────────────────────────────────────────────────────────────────────────────
logger.info("Saving runtime_analysis.csv ...")

csv_cols = [
    "video_id", "num_relations", "num_tracklets",
    "num_chain_motifs", "num_triangle_motifs", "num_star_motifs",
    "num_hyperedges",
    "time_motif_extraction_ms", "time_hypergraph_construction_ms",
    "time_laplacian_ms", "time_refinement_ms", "time_total_ms",
]

with open(CSV_OUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=csv_cols)
    writer.writeheader()
    for vid in video_ids:
        s = all_stats[vid]
        writer.writerow({
            "video_id":                       vid,
            "num_relations":                  s.get("n_relations", 0),
            "num_tracklets":                  s.get("n_tracklets", 0),
            "num_chain_motifs":               s.get("n_chain_motifs", 0),
            "num_triangle_motifs":            s.get("n_triangle_motifs", 0),
            "num_star_motifs":                s.get("n_star_motifs", 0),
            "num_hyperedges":                 s.get("n_hyperedges", 0),
            "time_motif_extraction_ms":       round(s.get("time_motif_extraction_ms", 0.0), 4),
            "time_hypergraph_construction_ms":round(s.get("time_hypergraph_construction_ms", 0.0), 4),
            "time_laplacian_ms":              round(s.get("time_laplacian_ms", 0.0), 4),
            "time_refinement_ms":             round(s.get("time_refinement_ms", 0.0), 4),
            "time_total_ms":                  round(s.get("time_total_ms", 0.0), 4),
        })

logger.info("Saved %s", CSV_OUT)


# ─────────────────────────────────────────────────────────────────────────────
# Step D: Compute runtime summary statistics
# ─────────────────────────────────────────────────────────────────────────────
total_times = np.array([all_stats[v].get("time_total_ms", 0.0) for v in video_ids])
motif_times = np.array([all_stats[v].get("time_motif_extraction_ms", 0.0) for v in video_ids])
hg_times    = np.array([all_stats[v].get("time_hypergraph_construction_ms", 0.0) for v in video_ids])
lap_times   = np.array([all_stats[v].get("time_laplacian_ms", 0.0) for v in video_ids])
ref_times   = np.array([all_stats[v].get("time_refinement_ms", 0.0) for v in video_ids])

mean_total   = float(np.mean(total_times))
median_total = float(np.median(total_times))
max_total    = float(np.max(total_times))
min_total    = float(np.min(total_times))
sum_total_s  = float(np.sum(total_times)) / 1000.0

mean_motif_pct = float(np.mean(motif_times)) / mean_total * 100 if mean_total > 0 else 0
mean_hg_pct    = float(np.mean(hg_times))    / mean_total * 100 if mean_total > 0 else 0
mean_lap_pct   = float(np.mean(lap_times))   / mean_total * 100 if mean_total > 0 else 0
mean_ref_pct   = float(np.mean(ref_times))   / mean_total * 100 if mean_total > 0 else 0

print("\n" + "="*65)
print("  RUNTIME SUMMARY STATISTICS")
print("="*65)
print(f"  {'Mean total processing time per video':<40} {mean_total:>10.2f} ms")
print(f"  {'Median total processing time per video':<40} {median_total:>10.2f} ms")
print(f"  {'Maximum processing time observed':<40} {max_total:>10.2f} ms")
print(f"  {'Minimum processing time observed':<40} {min_total:>10.2f} ms")
print(f"  {'Total processing time for all 200 videos':<40} {sum_total_s:>10.2f} s")
print("-"*65)
print("  Mean time per component (% of total):")
print(f"    {'Motif extraction':<36} {mean_motif_pct:>8.2f} %")
print(f"    {'Hypergraph construction':<36} {mean_hg_pct:>8.2f} %")
print(f"    {'Laplacian (build + solve)':<36} {mean_lap_pct:>8.2f} %")
print(f"    {'Score refinement (blending)':<36} {mean_ref_pct:>8.2f} %")
print("="*65 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Step E: Generate runtime_analysis.png (2x2)
# ─────────────────────────────────────────────────────────────────────────────
logger.info("Generating runtime_analysis.png ...")

n_rels_arr   = np.array([all_stats[v].get("n_relations", 0) for v in video_ids])
n_chain_arr  = np.array([all_stats[v].get("n_chain_motifs", 0) for v in video_ids])
n_tri_arr    = np.array([all_stats[v].get("n_triangle_motifs", 0) for v in video_ids])
n_star_arr   = np.array([all_stats[v].get("n_star_motifs", 0) for v in video_ids])
n_total_motifs = n_chain_arr + n_tri_arr + n_star_arr

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("DWMH-SGG Runtime Analysis on VidVRD Test Set\n"
             r"($\tau$=3, $\beta$=0.10, $\gamma$=0.50, $\delta_{det}$=0.40, $\delta_{tag}$=0.30)",
             fontsize=13, fontweight="bold", y=0.98)

# ── Plot 1: Runtime vs Relation Count ────────────────────────────────────────
ax1 = axes[0, 0]
ax1.scatter(n_rels_arr, total_times, c="lightsteelblue", alpha=0.65,
            edgecolors="steelblue", linewidths=0.4, s=40, zorder=2,
            label="Individual videos")

# Binned average
if len(n_rels_arr) > 0:
    bins = np.percentile(n_rels_arr, np.linspace(0, 100, 11))
    bins = np.unique(bins)
    bin_idx = np.digitize(n_rels_arr, bins)
    bin_x, bin_y = [], []
    for b in range(1, len(bins) + 1):
        mask = bin_idx == b
        if mask.sum() > 0:
            bin_x.append(float(np.mean(n_rels_arr[mask])))
            bin_y.append(float(np.mean(total_times[mask])))
    if bin_x:
        sort_idx = np.argsort(bin_x)
        bin_x = np.array(bin_x)[sort_idx]
        bin_y = np.array(bin_y)[sort_idx]
        ax1.plot(bin_x, bin_y, "o-", color="navy", linewidth=2.0,
                 markersize=6, zorder=3, label="Binned average")

ax1.set_xlabel("Number of Relations", fontsize=11)
ax1.set_ylabel("Processing Time (ms)", fontsize=11)
ax1.set_title("Runtime vs Relation Count", fontsize=12, fontweight="bold")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(left=0)
ax1.set_ylim(bottom=0)

# ── Plot 2: Runtime vs Motif Count ───────────────────────────────────────────
ax2 = axes[0, 1]
ax2.scatter(n_total_motifs, total_times, c="lightcoral", alpha=0.65,
            edgecolors="firebrick", linewidths=0.4, s=40, zorder=2)
ax2.set_xlabel("Total Motifs per Video (chain + triangle + star)", fontsize=11)
ax2.set_ylabel("Total Processing Time (ms)", fontsize=11)
ax2.set_title("Runtime vs Motif Count", fontsize=12, fontweight="bold")
ax2.grid(True, alpha=0.3)
ax2.set_xlim(left=0)
ax2.set_ylim(bottom=0)

# ── Plot 3: Component Time Breakdown (horizontal stacked bar) ─────────────────
ax3 = axes[1, 0]
components  = ["Motif\nExtraction", "Hypergraph\nConstruction",
               "Laplacian\n(build+solve)", "Score\nRefinement"]
mean_times_abs = [float(np.mean(motif_times)), float(np.mean(hg_times)),
                  float(np.mean(lap_times)),   float(np.mean(ref_times))]
total_comp = sum(mean_times_abs)
if total_comp > 0:
    percentages = [t / total_comp * 100 for t in mean_times_abs]
else:
    percentages = [25.0] * 4

colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
left = 0.0
bar_height = 0.5
for j, (comp, pct, col) in enumerate(zip(components, percentages, colors)):
    ax3.barh(0, pct, height=bar_height, left=left, color=col,
             label=f"{comp.replace(chr(10),' ')}: {pct:.1f}%", zorder=2)
    if pct > 4.0:
        ax3.text(left + pct / 2, 0, f"{pct:.1f}%",
                 ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    left += pct

ax3.set_yticks([])
ax3.set_xlabel("Percentage of Total Processing Time (%)", fontsize=11)
ax3.set_title("Mean Time per Component", fontsize=12, fontweight="bold")
ax3.set_xlim(0, 100)
ax3.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=9)
ax3.grid(True, axis="x", alpha=0.3)

# ── Plot 4: Runtime Histogram ─────────────────────────────────────────────────
ax4 = axes[1, 1]
ax4.hist(total_times, bins=25, color="mediumseagreen", edgecolor="white",
         alpha=0.85, zorder=2)
ax4.axvline(mean_total,   color="navy",    linestyle="-",  linewidth=2.0,
            label=f"Mean: {mean_total:.1f} ms", zorder=3)
ax4.axvline(median_total, color="crimson", linestyle="--", linewidth=2.0,
            label=f"Median: {median_total:.1f} ms", zorder=3)
ax4.set_xlabel("Total Processing Time (ms)", fontsize=11)
ax4.set_ylabel("Number of Videos", fontsize=11)
ax4.set_title("Processing Time Distribution", fontsize=12, fontweight="bold")
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(RUNTIME_PNG, dpi=300, bbox_inches="tight")
plt.close()
logger.info("Saved %s", RUNTIME_PNG)


# ─────────────────────────────────────────────────────────────────────────────
# Step F: Load VidVRD evaluator and run per-video evaluation
# ─────────────────────────────────────────────────────────────────────────────
logger.info("Loading VidVRD-II evaluator ...")
if VIDVRD_PATH not in sys.path:
    sys.path.insert(0, VIDVRD_PATH)

from dataset import VidVRD
from evaluation.visual_relation_detection import eval_detection_scores, eval_tagging_scores
from evaluation.common import voc_ap

norm_coords = prediction["version"] >= "VERSION 2.1"
dataset = VidVRD(DATASET_PATH, os.path.join(DATASET_PATH, "videos"),
                 ["train", "test"], normalize_coords=norm_coords)

test_vids  = dataset.get_index("test")
groundtruth = {vid: dataset.get_relation_insts(vid) for vid in test_vids}
logger.info("Ground truth loaded for %d test videos.", len(test_vids))


def per_video_ap(gt_dict, pred_dict, viou_thr=0.5):
    """Compute per-video AP using the VidVRD-II evaluator."""
    results = {}
    for vid, gt_rels in gt_dict.items():
        if not gt_rels:
            results[vid] = 0.0
            continue
        preds = pred_dict.get(vid, [])
        det_prec, det_rec, _ = eval_detection_scores(gt_rels, preds, viou_thr)
        results[vid] = float(voc_ap(det_rec, det_prec))
    return results


logger.info("Computing per-video AP for baseline ...")
baseline_ap = per_video_ap(groundtruth, prediction["results"])

logger.info("Computing per-video AP for DWMH-SGG ...")
dwmh_ap     = per_video_ap(groundtruth, all_refined)

# Compute delta per video
delta_map = {vid: dwmh_ap.get(vid, 0.0) - baseline_ap.get(vid, 0.0)
             for vid in test_vids}


# ─────────────────────────────────────────────────────────────────────────────
# Step G: Qualitative analysis  → qualitative_analysis.txt
# ─────────────────────────────────────────────────────────────────────────────
logger.info("Running qualitative analyses ...")

# Sort videos by delta mAP
sorted_by_delta = sorted(delta_map.items(), key=lambda x: x[1], reverse=True)
top5_improved   = sorted_by_delta[:5]
top5_hurt       = sorted_by_delta[-5:][::-1]   # most hurt first (most negative)

with open(QUAL_TXT, "w", encoding="utf-8") as fout:

    def wl(s=""):
        fout.write(s + "\n")

    wl("=" * 80)
    wl("  DWMH-SGG QUALITATIVE ANALYSIS  —  VidVRD Test Set")
    wl(f"  tau={TAU}  beta={BETA}  gamma={GAMMA}  "
       f"delta_det={DELTA_DET}  delta_tag={DELTA_TAG}")
    wl("=" * 80)

    # ── Analysis 1: Top-5 most improved ──────────────────────────────────────
    wl()
    wl("=" * 80)
    wl("  ANALYSIS 1 — TOP 5 MOST IMPROVED VIDEOS")
    wl("=" * 80)

    for rank, (vid, d) in enumerate(top5_improved, 1):
        wl()
        wl(f"  Rank {rank}: {vid}")
        wl(f"  {'Baseline mAP':>20}: {baseline_ap[vid]:.6f}")
        wl(f"  {'DWMH-SGG mAP':>20}: {dwmh_ap[vid]:.6f}")
        wl(f"  {'Delta':>20}: {d:+.6f}")

        s = all_stats.get(vid, {})
        n_ch = s.get("n_chain_motifs", 0)
        n_tr = s.get("n_triangle_motifs", 0)
        n_st = s.get("n_star_motifs", 0)
        n_he = s.get("n_hyperedges", 0)
        wl(f"  Motifs: chain={n_ch}  triangle={n_tr}  star={n_st}")
        wl(f"  Hyperedges constructed: {n_he}")

        base_rels = prediction["results"].get(vid, [])
        dwmh_rels = all_refined.get(vid, [])

        base_sorted = sorted(base_rels, key=lambda r: -float(r["score"]))[:10]
        dwmh_sorted = sorted(dwmh_rels, key=lambda r: -float(r["score"]))[:10]

        wl()
        wl("  Top 10 BASELINE predictions:")
        wl(f"  {'Rank':<6} {'Triplet':<45} {'Score':>8}")
        wl("  " + "-" * 62)
        for j, r in enumerate(base_sorted, 1):
            trip = "  ".join(str(x) for x in r["triplet"])
            wl(f"  {j:<6} {trip:<45} {float(r['score']):>8.6f}")

        wl()
        wl("  Top 10 DWMH-SGG predictions (refined scores):")
        wl(f"  {'Rank':<6} {'Triplet':<45} {'Refined Score':>13} {'Original Score':>14}")
        wl("  " + "-" * 82)
        # Map original scores by position
        orig_score_map = {i: float(r["score"]) for i, r in enumerate(base_rels)}
        dwmh_score_map = {i: float(r["score"]) for i, r in enumerate(dwmh_rels)}
        dwmh_with_idx = sorted(enumerate(dwmh_rels), key=lambda x: -float(x[1]["score"]))[:10]
        for j, (idx, r) in enumerate(dwmh_with_idx, 1):
            trip = "  ".join(str(x) for x in r["triplet"])
            orig = orig_score_map.get(idx, float("nan"))
            wl(f"  {j:<6} {trip:<45} {float(r['score']):>13.6f} {orig:>14.6f}")

        # Which relations were boosted most
        wl()
        wl("  Most boosted relations (refined - original):")
        boosts = []
        for idx, (br, dr) in enumerate(zip(base_rels, dwmh_rels)):
            diff = float(dr["score"]) - float(br["score"])
            if diff > 1e-6:
                trip = "  ".join(str(x) for x in br["triplet"])
                boosts.append((trip, diff, float(br["score"]), float(dr["score"])))
        boosts.sort(key=lambda x: -x[1])
        if boosts:
            wl(f"  {'Triplet':<45} {'Delta':>8} {'Original':>10} {'Refined':>10}")
            wl("  " + "-" * 77)
            for trip, diff, orig, ref in boosts[:10]:
                wl(f"  {trip:<45} {diff:>+8.6f} {orig:>10.6f} {ref:>10.6f}")
        else:
            wl("  (no relations boosted above threshold)")

    # ── Analysis 2: Top-5 most hurt ───────────────────────────────────────────
    wl()
    wl()
    wl("=" * 80)
    wl("  ANALYSIS 2 — TOP 5 MOST HURT VIDEOS")
    wl("=" * 80)

    for rank, (vid, d) in enumerate(top5_hurt, 1):
        wl()
        wl(f"  Rank {rank}: {vid}")
        wl(f"  {'Baseline mAP':>20}: {baseline_ap[vid]:.6f}")
        wl(f"  {'DWMH-SGG mAP':>20}: {dwmh_ap[vid]:.6f}")
        wl(f"  {'Delta':>20}: {d:+.6f}")

        s = all_stats.get(vid, {})
        n_ch = s.get("n_chain_motifs", 0)
        n_tr = s.get("n_triangle_motifs", 0)
        n_st = s.get("n_star_motifs", 0)
        n_he = s.get("n_hyperedges", 0)
        wl(f"  Motifs: chain={n_ch}  triangle={n_tr}  star={n_st}")
        wl(f"  Hyperedges constructed: {n_he}")

        base_rels = prediction["results"].get(vid, [])
        dwmh_rels = all_refined.get(vid, [])

        base_sorted = sorted(base_rels, key=lambda r: -float(r["score"]))[:10]
        dwmh_sorted = sorted(dwmh_rels, key=lambda r: -float(r["score"]))[:10]

        wl()
        wl("  Top 10 BASELINE predictions:")
        wl(f"  {'Rank':<6} {'Triplet':<45} {'Score':>8}")
        wl("  " + "-" * 62)
        for j, r in enumerate(base_sorted, 1):
            trip = "  ".join(str(x) for x in r["triplet"])
            wl(f"  {j:<6} {trip:<45} {float(r['score']):>8.6f}")

        wl()
        wl("  Top 10 DWMH-SGG predictions (refined scores):")
        wl(f"  {'Rank':<6} {'Triplet':<45} {'Refined Score':>13} {'Original Score':>14}")
        wl("  " + "-" * 82)
        dwmh_with_idx = sorted(enumerate(dwmh_rels), key=lambda x: -float(x[1]["score"]))[:10]
        orig_score_map = {i: float(r["score"]) for i, r in enumerate(base_rels)}
        for j, (idx, r) in enumerate(dwmh_with_idx, 1):
            trip = "  ".join(str(x) for x in r["triplet"])
            orig = orig_score_map.get(idx, float("nan"))
            wl(f"  {j:<6} {trip:<45} {float(r['score']):>13.6f} {orig:>14.6f}")

        # Incorrectly boosted / penalized
        wl()
        wl("  Incorrectly boosted or penalized relations:")
        gt_rels  = groundtruth.get(vid, [])
        gt_trips = set(tuple(r["triplet"]) for r in gt_rels)
        changes = []
        for idx, (br, dr) in enumerate(zip(base_rels, dwmh_rels)):
            diff = float(dr["score"]) - float(br["score"])
            is_correct = tuple(br["triplet"]) in gt_trips
            if abs(diff) > 1e-6:
                changes.append((tuple(br["triplet"]), diff, is_correct,
                                 float(br["score"]), float(dr["score"])))
        changes.sort(key=lambda x: -abs(x[1]))
        if changes:
            wl(f"  {'Triplet':<45} {'Delta':>8} {'GT?':>5} {'Orig':>8} {'Refined':>8}")
            wl("  " + "-" * 82)
            for trip, diff, is_gt, orig, ref in changes[:10]:
                trip_str = "  ".join(str(x) for x in trip)
                gt_flag = "YES" if is_gt else "no"
                wl(f"  {trip_str:<45} {diff:>+8.6f} {gt_flag:>5} {orig:>8.6f} {ref:>8.6f}")

        # Hypothesis
        wl()
        n_rels_vid = s.get("n_relations", 0)
        total_m = n_ch + n_tr + n_st
        if total_m == 0:
            hypothesis = ("No motifs detected; pair-hyperedge smoothing pulled "
                          "high-scoring GT relations toward lower-scored neighbours.")
        elif n_rels_vid == 0:
            hypothesis = "Video had no predicted relations; trivially skipped."
        else:
            density = total_m / max(n_rels_vid, 1)
            if density > 2.0:
                hypothesis = ("High motif density caused aggressive smoothing that "
                              "diluted distinct discriminative scores for GT triplets.")
            else:
                hypothesis = ("Motif structure encoded false co-occurrence patterns, "
                              "boosting incorrect relations and penalising correct ones.")
        wl(f"  Hypothesis: {hypothesis}")

    # ── Analysis 3: Motif statistics ──────────────────────────────────────────
    wl()
    wl()
    wl("=" * 80)
    wl("  ANALYSIS 3 — MOTIF STATISTICS ACROSS FULL TEST SET")
    wl("=" * 80)
    wl()

    chain_counts    = np.array([all_stats[v].get("n_chain_motifs", 0) for v in video_ids])
    tri_counts      = np.array([all_stats[v].get("n_triangle_motifs", 0) for v in video_ids])
    star_counts     = np.array([all_stats[v].get("n_star_motifs", 0) for v in video_ids])
    he_counts       = np.array([all_stats[v].get("n_hyperedges", 0) for v in video_ids])
    tracklet_counts = np.array([all_stats[v].get("n_tracklets", 0) for v in video_ids])

    total_chain  = int(chain_counts.sum())
    total_tri    = int(tri_counts.sum())
    total_star   = int(star_counts.sum())

    zero_chain = int((chain_counts == 0).sum())
    zero_tri   = int((tri_counts == 0).sum())
    zero_star  = int((star_counts == 0).sum())
    zero_any   = int(((chain_counts == 0) & (tri_counts == 0) & (star_counts == 0)).sum())

    wl(f"  {'Motif Type':<20} {'Total':>10} {'Mean/video':>12} {'Median/video':>14} {'Videos w/ 0':>13}")
    wl("  " + "-" * 73)
    wl(f"  {'Chain':<20} {total_chain:>10} {np.mean(chain_counts):>12.2f} {np.median(chain_counts):>14.2f} {zero_chain:>13}")
    wl(f"  {'Triangle':<20} {total_tri:>10} {np.mean(tri_counts):>12.2f} {np.median(tri_counts):>14.2f} {zero_tri:>13}")
    wl(f"  {'Star':<20} {total_star:>10} {np.mean(star_counts):>12.2f} {np.median(star_counts):>14.2f} {zero_star:>13}")
    wl(f"  {'All (any type)':<20} {total_chain+total_tri+total_star:>10} "
       f"{np.mean(chain_counts+tri_counts+star_counts):>12.2f} "
       f"{np.median(chain_counts+tri_counts+star_counts):>14.2f} {zero_any:>13}")
    wl()
    wl(f"  Videos with zero motifs of ANY type: {zero_any}")
    wl(f"  Mean hyperedges per video: {np.mean(he_counts):.2f}")
    wl(f"  Median hyperedges per video: {np.median(he_counts):.2f}")
    wl()

    # Mean tracklets per hyperedge per motif type
    # (track via individual stats)
    chain_tphe, tri_tphe, star_tphe = [], [], []
    for v in video_ids:
        s = all_stats.get(v, {})
        n_ch = s.get("n_chain_motifs", 0)
        n_tr = s.get("n_triangle_motifs", 0)
        n_st = s.get("n_star_motifs", 0)
        n_he = s.get("n_hyperedges", 0)
        n_rel = s.get("n_relations", 0)
        motif_he = n_he - n_rel  # subtract pair hyperedges

        # Approximate per-motif-type: chains→3 nodes, tri→3 nodes, star→variable
        # We report per-motif tracklet counts (how many tracklets involved)
        if n_ch > 0:
            chain_tphe.append(3.0)  # always 3 nodes per chain
        if n_tr > 0:
            tri_tphe.append(3.0)    # always 3 nodes per triangle
        # Stars have variable size — use overall average if possible
        for sv in s.get("stars", []):
            if isinstance(sv, dict):
                star_tphe.append(float(len(sv.get("node_tids", []))))

    wl(f"  Mean tracklets per chain hyperedge:    3.00 (fixed)")
    wl(f"  Mean tracklets per triangle hyperedge: 3.00 (fixed)")
    wl(f"  Mean tracklets per star hyperedge:     "
       f"{np.mean(star_tphe):.2f}" if star_tphe else "  Mean tracklets per star hyperedge:     N/A (no stars found)")

    # ── Analysis 4: Score distribution ───────────────────────────────────────
    wl()
    wl()
    wl("=" * 80)
    wl("  ANALYSIS 4 — SCORE DISTRIBUTION BEFORE vs AFTER REFINEMENT")
    wl("=" * 80)
    wl()

    all_orig_scores = []
    all_ref_scores  = []
    for v in video_ids:
        s = all_stats.get(v, {})
        orig = s.get("original_scores", [])
        ref  = s.get("refined_scores", [])
        all_orig_scores.extend(orig)
        all_ref_scores.extend(ref)

    orig_arr = np.array(all_orig_scores, dtype=np.float64)
    ref_arr  = np.array(all_ref_scores,  dtype=np.float64)

    wl("  Baseline (original) scores:")
    wl(f"    Mean:  {np.mean(orig_arr):.6f}")
    wl(f"    Std:   {np.std(orig_arr):.6f}")
    wl(f"    Min:   {np.min(orig_arr):.6f}")
    wl(f"    Max:   {np.max(orig_arr):.6f}")
    wl()
    wl("  DWMH-SGG (refined) scores:")
    wl(f"    Mean:  {np.mean(ref_arr):.6f}")
    wl(f"    Std:   {np.std(ref_arr):.6f}")
    wl(f"    Min:   {np.min(ref_arr):.6f}")
    wl(f"    Max:   {np.max(ref_arr):.6f}")
    wl()

    thr = 0.001
    diff_arr = ref_arr - orig_arr
    n_boosted   = int((diff_arr >  thr).sum())
    n_penalized = int((diff_arr < -thr).sum())
    n_unchanged = int((np.abs(diff_arr) <= thr).sum())
    total_r = len(diff_arr)

    wl(f"  Relations boosted   (delta > {thr}):  {n_boosted:>6}  ({100*n_boosted/total_r:.1f}%)")
    wl(f"  Relations penalized (delta < -{thr}): {n_penalized:>6}  ({100*n_penalized/total_r:.1f}%)")
    wl(f"  Relations unchanged (|delta| ≤ {thr}): {n_unchanged:>6}  ({100*n_unchanged/total_r:.1f}%)")

    # ── Analysis 5: Per-predicate score change ────────────────────────────────
    wl()
    wl()
    wl("=" * 80)
    wl("  ANALYSIS 5 — PER-PREDICATE MEAN SCORE CHANGE (refined - original)")
    wl("=" * 80)
    wl()

    pred_diffs   = defaultdict(list)
    pred_counts  = defaultdict(int)

    for v in video_ids:
        s = all_stats.get(v, {})
        orig = s.get("original_scores", [])
        ref  = s.get("refined_scores", [])
        base_rels = prediction["results"].get(v, [])
        for i, (br, os_, rs_) in enumerate(zip(base_rels, orig, ref)):
            # triplet = [subject_class, predicate, object_class]
            triplet = br.get("triplet", [])
            if len(triplet) >= 3:
                predicate = str(triplet[1])
                pred_diffs[predicate].append(rs_ - os_)
                pred_counts[predicate] += 1

    pred_mean_change = {p: float(np.mean(v)) for p, v in pred_diffs.items()}
    pred_sorted = sorted(pred_mean_change.items(), key=lambda x: -x[1])

    wl(f"  {'Rank':<6} {'Predicate':<30} {'Mean Score Change':>18} {'Num Instances':>14}")
    wl("  " + "-" * 72)
    for rank, (pred, mc) in enumerate(pred_sorted, 1):
        n_inst = pred_counts[pred]
        wl(f"  {rank:<6} {pred:<30} {mc:>+18.6f} {n_inst:>14}")

    wl()
    wl("=" * 80)
    wl("  END OF QUALITATIVE ANALYSIS")
    wl("=" * 80)

logger.info("Saved %s", QUAL_TXT)


# ─────────────────────────────────────────────────────────────────────────────
# Step H: score_distribution.png
# ─────────────────────────────────────────────────────────────────────────────
logger.info("Generating score_distribution.png ...")

fig2, ax = plt.subplots(figsize=(10, 6))

ax.hist(orig_arr, bins=60, color="steelblue", alpha=0.6, label="Baseline scores",
        density=False, zorder=2)
ax.hist(ref_arr,  bins=60, color="darkorange", alpha=0.6, label="Refined scores (DWMH-SGG)",
        density=False, zorder=2)

ax.axvline(float(np.mean(orig_arr)), color="steelblue",   linestyle="--", linewidth=2.0,
           label=f"Baseline mean: {np.mean(orig_arr):.4f}", zorder=3)
ax.axvline(float(np.mean(ref_arr)),  color="darkorange",  linestyle="--", linewidth=2.0,
           label=f"Refined mean: {np.mean(ref_arr):.4f}", zorder=3)

ax.set_xlabel("Relation Score", fontsize=13)
ax.set_ylabel("Count", fontsize=13)
ax.set_title("Relation Score Distribution Before and After DWMH-SGG Refinement",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(SCORE_DIST_PNG, dpi=300, bbox_inches="tight")
plt.close()
logger.info("Saved %s", SCORE_DIST_PNG)


# ─────────────────────────────────────────────────────────────────────────────
# Step I: predicate_analysis.png
# ─────────────────────────────────────────────────────────────────────────────
logger.info("Generating predicate_analysis.png ...")

pred_names = [p for p, _ in pred_sorted]
pred_vals  = [v for _, v in pred_sorted]
pred_n     = [pred_counts[p] for p in pred_names]

# Color bars: green if positive, red if negative
bar_colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in pred_vals]

fig3, ax = plt.subplots(figsize=(12, max(6, len(pred_names) * 0.45)))
y_pos = np.arange(len(pred_names))[::-1]  # reversed so highest is at top

bars = ax.barh(y_pos, pred_vals, color=bar_colors, edgecolor="white",
               linewidth=0.5, zorder=2)
ax.axvline(0, color="black", linewidth=0.8, zorder=3)

ax.set_yticks(y_pos)
ax.set_yticklabels(pred_names, fontsize=9)
ax.set_xlabel("Mean Score Change (Refined − Original)", fontsize=12)
ax.set_title("Per-Predicate Mean Score Change after DWMH-SGG Refinement\n"
             "(sorted descending; green = boosted, red = penalized)",
             fontsize=12, fontweight="bold")
ax.grid(True, axis="x", alpha=0.3)

# Annotate with count
for bar, n_i in zip(bars, [pred_counts[p] for p in pred_names[::-1]]):
    ax.text(bar.get_width() + 1e-5 * (1 if bar.get_width() >= 0 else -1),
            bar.get_y() + bar.get_height() / 2,
            f"n={n_i}", va="center", ha="left" if bar.get_width() >= 0 else "right",
            fontsize=7, color="dimgray")

plt.tight_layout()
plt.savefig(PREDICATE_PNG, dpi=300, bbox_inches="tight")
plt.close()
logger.info("Saved %s", PREDICATE_PNG)


# ─────────────────────────────────────────────────────────────────────────────
# Final summary print
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("  OUTPUT FILES")
print("=" * 65)
for path in [CSV_OUT, RUNTIME_PNG, QUAL_TXT, SCORE_DIST_PNG, PREDICATE_PNG]:
    exists = os.path.isfile(path)
    size   = os.path.getsize(path) if exists else 0
    status = f"OK  ({size/1024:.1f} KB)" if exists else "MISSING"
    print(f"  {os.path.basename(path):<35} {status}")
print("=" * 65)
print()
print("All analyses complete.")
