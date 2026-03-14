"""Task 3 — Ablation study for DWMH-SGG on VidVRD test set.

Part A — Component ablation (5 configs):
  Config A: VidVRD-II baseline (no refinement)
  Config B: Binary incidence + ST-SHG count-based log boost (no dir/cent/Jaccard/Laplacian)
  Config C: Directed weighted incidence (H_tilde) + direct weighted-sum, no Jaccard, no Laplacian
  Config D: Config C + Jaccard co-occurrence re-weighting, no Laplacian
  Config E: Full DWMH-SGG (Config D + Laplacian spectral propagation)

Part B — Motif-type ablation (7 subsets), using full DWMH-SGG with:
  tau=3, beta=0.10, gamma=0.50, delta_det=0.40, delta_tag=0.30

Usage (from E:/PHD/ST-SGG_Models/DWMH-SGG/):
    python ablation.py
"""

import copy
import json
import math
import os
import sys
from collections import defaultdict, OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR      = os.path.dirname(BASE_DIR)
BASELINE_PATH = os.path.join(ROOT_DIR, "imagenet-vidvrd-baseline-output",
                             "models", "3step_prop_wd0.01", "video_relations.json")
VIDVRD_PATH   = os.path.join(ROOT_DIR, "VidVRD-II")
DATASET_PATH  = os.path.join(ROOT_DIR, "imagenet-vidvrd-dataset")

TAU, BETA, GAMMA, DELTA_DET, DELTA_TAG = 3, 0.10, 0.50, 0.40, 0.30
_EPS = 1e-8

REPORT_METRICS = [
    ("mAP",        "detection mean AP",     100.0),
    ("R@50",       "detection recall@50",   100.0),
    ("R@100",      "detection recall@100",  100.0),
    ("P@1",        "tagging precision@1",   100.0),
    ("P@5",        "tagging precision@5",   100.0),
    ("P@10",       "tagging precision@10",  100.0),
]


# ── VidVRD-II helpers ─────────────────────────────────────────────────────────
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


def evaluate(groundtruth, det_pred, tag_pred,
             eval_det, eval_tag, voc_ap, viou=0.5):
    """Evaluate with separate det/tag prediction dicts; return metric dict."""
    video_ap = {}
    tot_sc   = defaultdict(list)
    tot_tp   = defaultdict(list)
    prec_at  = defaultdict(list)
    tot_gt   = 0
    det_nret = [10, 20, 50, 100]
    tag_nret = [1, 5, 10]
    for vid, gt_rels in groundtruth.items():
        if not gt_rels:
            continue
        tot_gt += len(gt_rels)
        dp = det_pred.get(vid, [])
        tp = tag_pred.get(vid, [])
        det_prec, det_rec, det_sc = eval_det(gt_rels, dp, viou)
        video_ap[vid] = float(voc_ap(det_rec, det_prec))
        is_tp = np.isfinite(det_sc)
        for nre in det_nret:
            cut = min(nre, det_sc.size)
            tot_sc[nre].append(det_sc[:cut])
            tot_tp[nre].append(is_tp[:cut])
        tag_prec, _, _ = eval_tag(gt_rels, tp, max(tag_nret))
        for nre in tag_nret:
            prec_at[nre].append(tag_prec[nre - 1])
    out = OrderedDict()
    out["detection mean AP"] = float(np.mean(list(video_ap.values())))
    for nre in det_nret:
        sc   = np.concatenate(tot_sc[nre])
        tp_a = np.concatenate(tot_tp[nre])
        idx  = np.argsort(sc)[::-1]
        tp_a = tp_a[idx]
        cum  = np.cumsum(tp_a).astype(np.float32)
        rec  = cum / max(tot_gt, np.finfo(np.float32).eps)
        out[f"detection recall@{nre}"] = float(rec[-1])
    for nre in tag_nret:
        out[f"tagging precision@{nre}"] = float(np.mean(prec_at[nre]))
    return out


# ── DWMH-SGG module helpers (imported once) ───────────────────────────────────
def _import_dwmh():
    from dwmh_sgg.motif_extraction import (
        assign_tracklet_ids, extract_chain_motifs,
        extract_triangle_motifs, extract_star_motifs)
    from dwmh_sgg.hypergraph import (
        build_hyperedge_list, build_h_tilde,
        compute_edge_weights, compute_jaccard_matrix)
    from dwmh_sgg.laplacian import (
        init_node_scores, compute_laplacian, solve_laplacian)
    return dict(
        assign_tracklet_ids=assign_tracklet_ids,
        extract_chain=extract_chain_motifs,
        extract_triangle=extract_triangle_motifs,
        extract_star=extract_star_motifs,
        build_hyperedge_list=build_hyperedge_list,
        build_h_tilde=build_h_tilde,
        compute_edge_weights=compute_edge_weights,
        compute_jaccard_matrix=compute_jaccard_matrix,
        init_node_scores=init_node_scores,
        compute_laplacian=compute_laplacian,
        solve_laplacian=solve_laplacian,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Refinement implementations for each ablation config
# ══════════════════════════════════════════════════════════════════════════════

def _blend(orig: float, signal: float, delta: float) -> float:
    return (1.0 - delta) * orig + delta * signal


def refine_config_b(relations, fn, tau, delta_det, delta_tag,
                    motif_types=("chain", "triangle", "star")):
    """Config B: ST-SHG formula — binary incidence, count-based log boost."""
    if len(relations) <= 1:
        return copy.deepcopy(relations)

    sub_tids, obj_tids, _ = fn["assign_tracklet_ids"](relations)

    chains    = fn["extract_chain"](relations, sub_tids, obj_tids, tau) \
        if "chain" in motif_types else []
    triangles = fn["extract_triangle"](relations, sub_tids, obj_tids, tau) \
        if "triangle" in motif_types else []
    stars     = fn["extract_star"](relations, sub_tids, obj_tids, tau) \
        if "star" in motif_types else []

    # Count motif memberships per relation
    support = defaultdict(int)
    for motif in chains + triangles + stars:
        for idx in motif["rel_indices"]:
            support[idx] += 1

    out = copy.deepcopy(relations)
    for i, rel in enumerate(out):
        c    = support.get(i, 0)
        base = float(relations[i]["score"])
        if c == 0:
            boost_det = base * 0.90
            boost_tag = base * 0.90
        else:
            boost_det = base * (1.0 + delta_det * math.log1p(c))
            boost_tag = base * (1.0 + delta_tag * math.log1p(c))
        rel["score"]         = boost_det
        rel["tagging_score"] = boost_tag
    return out


def refine_config_c(relations, fn, tau, delta_det, delta_tag,
                    motif_types=("chain", "triangle", "star")):
    """Config C: H_tilde (direction+centrality) + direct weighted-sum, no Jaccard/Laplacian."""
    if len(relations) <= 1:
        return copy.deepcopy(relations)

    sub_tids, obj_tids, n_tracklets = fn["assign_tracklet_ids"](relations)

    chains    = fn["extract_chain"](relations, sub_tids, obj_tids, tau) \
        if "chain" in motif_types else []
    triangles = fn["extract_triangle"](relations, sub_tids, obj_tids, tau) \
        if "triangle" in motif_types else []
    stars     = fn["extract_star"](relations, sub_tids, obj_tids, tau) \
        if "star" in motif_types else []
    motifs    = chains + triangles + stars

    hyperedges = fn["build_hyperedge_list"](relations, sub_tids, obj_tids, motifs)
    H_tilde    = fn["build_h_tilde"](hyperedges, n_tracklets)  # (V, K)
    w          = fn["compute_edge_weights"](relations, hyperedges)  # (K,)

    # Direct weighted sum per node (no Jaccard, no Laplacian)
    abs_H   = np.abs(H_tilde)
    dv      = abs_H @ np.ones(len(hyperedges))            # (V,) vertex degree
    signal  = (abs_H * w[np.newaxis, :]) @ np.ones(len(hyperedges))  # (V,)
    node_sig = signal / (dv + _EPS)                       # normalised by degree

    out = copy.deepcopy(relations)
    for i, rel in enumerate(out):
        sig = (node_sig[sub_tids[i]] + node_sig[obj_tids[i]]) / 2.0
        rel["score"]         = _blend(float(relations[i]["score"]), sig, delta_det)
        rel["tagging_score"] = _blend(float(relations[i]["score"]), sig, delta_tag)
    return out


def refine_config_d(relations, fn, tau, beta, delta_det, delta_tag,
                    motif_types=("chain", "triangle", "star")):
    """Config D: H_tilde + Jaccard co-occurrence, direct weighted-sum, no Laplacian."""
    if len(relations) <= 1:
        return copy.deepcopy(relations)

    sub_tids, obj_tids, n_tracklets = fn["assign_tracklet_ids"](relations)

    chains    = fn["extract_chain"](relations, sub_tids, obj_tids, tau) \
        if "chain" in motif_types else []
    triangles = fn["extract_triangle"](relations, sub_tids, obj_tids, tau) \
        if "triangle" in motif_types else []
    stars     = fn["extract_star"](relations, sub_tids, obj_tids, tau) \
        if "star" in motif_types else []
    motifs    = chains + triangles + stars

    hyperedges = fn["build_hyperedge_list"](relations, sub_tids, obj_tids, motifs)
    H_tilde    = fn["build_h_tilde"](hyperedges, n_tracklets)
    w          = fn["compute_edge_weights"](relations, hyperedges)

    # Jaccard re-weighting
    T          = fn["compute_jaccard_matrix"](hyperedges)
    w_new      = w * (1.0 + beta * T.sum(axis=1))

    abs_H   = np.abs(H_tilde)
    dv      = abs_H @ np.ones(len(hyperedges))
    signal  = (abs_H * w_new[np.newaxis, :]) @ np.ones(len(hyperedges))
    node_sig = signal / (dv + _EPS)

    out = copy.deepcopy(relations)
    for i, rel in enumerate(out):
        sig = (node_sig[sub_tids[i]] + node_sig[obj_tids[i]]) / 2.0
        rel["score"]         = _blend(float(relations[i]["score"]), sig, delta_det)
        rel["tagging_score"] = _blend(float(relations[i]["score"]), sig, delta_tag)
    return out


def refine_config_e(relations, fn, tau, beta, gamma, delta_det, delta_tag,
                    motif_types=("chain", "triangle", "star")):
    """Config E: Full DWMH-SGG (H_tilde + Jaccard + Laplacian propagation)."""
    if len(relations) <= 1:
        return copy.deepcopy(relations)

    sub_tids, obj_tids, n_tracklets = fn["assign_tracklet_ids"](relations)

    chains    = fn["extract_chain"](relations, sub_tids, obj_tids, tau) \
        if "chain" in motif_types else []
    triangles = fn["extract_triangle"](relations, sub_tids, obj_tids, tau) \
        if "triangle" in motif_types else []
    stars     = fn["extract_star"](relations, sub_tids, obj_tids, tau) \
        if "star" in motif_types else []
    motifs    = chains + triangles + stars

    hyperedges = fn["build_hyperedge_list"](relations, sub_tids, obj_tids, motifs)
    H_tilde    = fn["build_h_tilde"](hyperedges, n_tracklets)
    w          = fn["compute_edge_weights"](relations, hyperedges)
    T          = fn["compute_jaccard_matrix"](hyperedges)
    w_new      = w * (1.0 + beta * T.sum(axis=1))
    H_w        = H_tilde * w_new[np.newaxis, :]

    if H_w.shape[1] == 0:
        return copy.deepcopy(relations)

    Delta, d_v = fn["compute_laplacian"](H_w)
    alpha      = fn["init_node_scores"](relations, sub_tids, obj_tids, n_tracklets)
    zero_deg   = d_v < _EPS
    alpha_prime = fn["solve_laplacian"](Delta, alpha, gamma)
    alpha_prime = np.where(zero_deg, alpha, alpha_prime)

    out = copy.deepcopy(relations)
    for i, rel in enumerate(out):
        sig = (alpha_prime[sub_tids[i]] + alpha_prime[obj_tids[i]]) / 2.0
        rel["score"]         = _blend(float(relations[i]["score"]), sig, delta_det)
        rel["tagging_score"] = _blend(float(relations[i]["score"]), sig, delta_tag)
    return out


def apply_refinement(baseline_results, refine_fn):
    """Apply a per-video refinement function to the full prediction dict."""
    from tqdm import tqdm
    refined = {}
    for vid, rels in tqdm(baseline_results.items(), desc="refining", unit="vid"):
        refined[vid] = refine_fn(rels)
    return refined


def swap_scores(results):
    out = {}
    for vid, rels in results.items():
        new = []
        for rel in rels:
            if "tagging_score" in rel:
                r = copy.copy(rel)
                r["score"] = rel["tagging_score"]
                new.append(r)
            else:
                new.append(rel)
        out[vid] = new
    return out


# ── Pretty-printing ───────────────────────────────────────────────────────────
def print_ablation_table(rows, title):
    """rows: list of (label, scores_dict)"""
    col0  = max(len(r[0]) for r in rows) + 2
    col_w = 9
    metric_names = [m[0] for m in REPORT_METRICS]

    print(f"\n{'=' * (col0 + len(metric_names) * (col_w + 2))}")
    print(f"  {title}")
    print(f"{'=' * (col0 + len(metric_names) * (col_w + 2))}")
    header = f"{'Config':<{col0}}" + "  ".join(f"{m:>{col_w}}" for m in metric_names)
    print(header)
    print("-" * len(header))
    for label, scores in rows:
        vals = "  ".join(
            f"{scores.get(key, float('nan')) * mult:>{col_w}.2f}"
            for _, key, mult in REPORT_METRICS
        )
        print(f"{label:<{col0}}{vals}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading dataset and VidVRD-II modules...")
    dataset, eval_det, eval_tag, voc_ap = load_vidvrd()
    groundtruth = {vid: dataset.get_relation_insts(vid)
                   for vid in dataset.get_index("test")}

    print("Loading baseline prediction...")
    baseline_meta    = json.load(open(BASELINE_PATH))
    baseline_results = baseline_meta["results"]

    print("Importing DWMH-SGG modules...")
    fn = _import_dwmh()

    def eval_config(det_pred, tag_pred, label=""):
        return evaluate(groundtruth, det_pred, tag_pred, eval_det, eval_tag, voc_ap)

    # ══════════════════════════════════════════════════════════════════════════
    # PART A — Component ablation
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  PART A: Component ablation")
    print("=" * 60)

    part_a_rows = []

    # Config A: Baseline
    print("\nConfig A: Baseline (no refinement)...")
    scores_a = eval_config(baseline_results, baseline_results)
    part_a_rows.append(("A: Baseline", scores_a))

    # Config B: ST-SHG log boost
    print("Config B: ST-SHG count-based log boost...")
    b_res = apply_refinement(
        baseline_results,
        lambda rels: refine_config_b(rels, fn, TAU, DELTA_DET, DELTA_TAG)
    )
    scores_b = eval_config(b_res, swap_scores(b_res))
    part_a_rows.append(("B: ST-SHG log boost", scores_b))

    # Config C: H_tilde + direct sum, no Jaccard, no Laplacian
    print("Config C: H_tilde + direct sum (no Jaccard, no Laplacian)...")
    c_res = apply_refinement(
        baseline_results,
        lambda rels: refine_config_c(rels, fn, TAU, DELTA_DET, DELTA_TAG)
    )
    scores_c = eval_config(c_res, swap_scores(c_res))
    part_a_rows.append(("C: H_tilde direct sum", scores_c))

    # Config D: H_tilde + Jaccard, no Laplacian
    print("Config D: H_tilde + Jaccard (no Laplacian)...")
    d_res = apply_refinement(
        baseline_results,
        lambda rels: refine_config_d(rels, fn, TAU, BETA, DELTA_DET, DELTA_TAG)
    )
    scores_d = eval_config(d_res, swap_scores(d_res))
    part_a_rows.append(("D: H_tilde + Jaccard", scores_d))

    # Config E: Full DWMH-SGG
    print("Config E: Full DWMH-SGG (Laplacian)...")
    e_res = apply_refinement(
        baseline_results,
        lambda rels: refine_config_e(rels, fn, TAU, BETA, GAMMA, DELTA_DET, DELTA_TAG)
    )
    scores_e = eval_config(e_res, swap_scores(e_res))
    part_a_rows.append(("E: Full DWMH-SGG", scores_e))

    print_ablation_table(part_a_rows,
                         "Part A — Component ablation "
                         f"(tau={TAU}, beta={BETA}, gamma={GAMMA}, "
                         f"d_det={DELTA_DET}, d_tag={DELTA_TAG})")

    # ══════════════════════════════════════════════════════════════════════════
    # PART B — Motif-type ablation
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  PART B: Motif-type ablation (Full DWMH-SGG config)")
    print("=" * 60)

    motif_subsets = [
        ("Chain only",          ("chain",)),
        ("Triangle only",       ("triangle",)),
        ("Star only",           ("star",)),
        ("Chain + Triangle",    ("chain", "triangle")),
        ("Chain + Star",        ("chain", "star")),
        ("Triangle + Star",     ("triangle", "star")),
        ("All three motifs",    ("chain", "triangle", "star")),
    ]

    part_b_rows = []
    # Baseline row for reference
    part_b_rows.append(("A: Baseline", scores_a))

    for label, mtypes in motif_subsets:
        print(f"{label}...")
        res = apply_refinement(
            baseline_results,
            lambda rels, mt=mtypes: refine_config_e(
                rels, fn, TAU, BETA, GAMMA, DELTA_DET, DELTA_TAG, motif_types=mt)
        )
        sc = eval_config(res, swap_scores(res))
        part_b_rows.append((label, sc))

    print_ablation_table(part_b_rows,
                         "Part B — Motif-type ablation "
                         f"(tau={TAU}, beta={BETA}, gamma={GAMMA}, "
                         f"d_det={DELTA_DET}, d_tag={DELTA_TAG})")


if __name__ == "__main__":
    main()
