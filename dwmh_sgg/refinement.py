"""Score refinement: blends original VidVRD-II scores with hypergraph signals.

For each relation  i  between subject tracklet  s_i  and object tracklet  o_i:

    α_node_i       = ( α'[s_i] + α'[o_i] ) / 2
    final_score_i  = (1 − δ) · original_score_i  +  δ · α_node_i

where α' is the smoothed node score vector from the Laplacian solver.

The output JSON is a deep clone of the input with only the 'score' field updated.
"""

import copy
import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .motif_extraction import assign_tracklet_ids, extract_all_motifs
from .hypergraph import build_weighted_incidence
from .laplacian import init_node_scores, compute_laplacian, solve_laplacian

logger = logging.getLogger(__name__)

_EPS = 1e-8

# Sparse path is activated when a single video has more than this many relations.
_SPARSE_THRESHOLD = 500


def blend_scores(
    relations: List[Dict],
    sub_tids: List[int],
    obj_tids: List[int],
    alpha_prime: np.ndarray,
    delta: float,
    delta_tagging: Optional[float] = None,
) -> Tuple[List[float], Optional[List[float]]]:
    """Compute final blended scores for every relation in a video.

    Args:
        relations:     List of relation dicts with 'score'.
        sub_tids:      Subject tracklet ID per relation.
        obj_tids:      Object  tracklet ID per relation.
        alpha_prime:   Smoothed node score vector, shape (n_tracklets,).
        delta:         Blending weight for detection δ ∈ [0, 1].
        delta_tagging: Optional separate blending weight for tagging.
                       When provided a second list is returned; otherwise None.

    Returns:
        det_scores: Detection-optimised float scores, one per relation.
        tag_scores: Tagging-optimised float scores (or None).
    """
    det_scores: List[float] = []
    tag_scores: Optional[List[float]] = [] if delta_tagging is not None else None
    for i, rel in enumerate(relations):
        node_signal = (alpha_prime[sub_tids[i]] + alpha_prime[obj_tids[i]]) / 2.0
        orig = float(rel["score"])
        det_scores.append((1.0 - delta) * orig + delta * float(node_signal))
        if delta_tagging is not None:
            tag_scores.append(  # type: ignore[union-attr]
                (1.0 - delta_tagging) * orig + delta_tagging * float(node_signal)
            )
    return det_scores, tag_scores


def refine_video(
    relations: List[Dict],
    tau: int,
    beta: float,
    gamma: float,
    delta: float,
    delta_tagging: Optional[float] = None,
    verbose: bool = False,
) -> Tuple[List[Dict], Dict]:
    """Run the full DWMH-SGG refinement pipeline for one video.

    Steps:
      1. Assign tracklet IDs from trajectory fingerprints.
      2. Extract chain / triangle / star motifs.
      3. Build weighted incidence matrix H_w.
      4. Compute Laplacian Δ and initialise node scores α.
      5. Solve (I + γ·Δ) α' = α.
      6. Blend original scores with node signal.

    Edge cases handled without modification:
      - 0 or 1 relations: returned unchanged (no hypergraph to build).
      - 0 motifs: pair hyperedges still used for smoothing.
      - Zero-degree tracklets: α' defaults to their initialised value.

    Args:
        relations: List of relation dicts for this video.
        tau:       Minimum temporal overlap (frames) for motif detection.
        beta:      Jaccard co-occurrence re-weighting strength.
        gamma:     Laplacian smoothing strength.
        delta:     Score blending weight (0 = no refinement, 1 = full).
        delta_tagging: Optional separate blending weight for tagging metrics.
                       When set, each relation also gets a 'tagging_score' key.
        verbose:   Log per-video stats when True.

    Returns:
        refined_relations: Deep-copy of relations with updated 'score' values
                           (and optional 'tagging_score' values).
        stats:             Dict with motif counts and timing info.
    """
    t0 = time.perf_counter()

    # ── Trivial cases ──────────────────────────────────────────────────────────
    if len(relations) <= 1:
        logger.info("  ↳ too few relations (%d), skipping", len(relations))
        return copy.deepcopy(relations), {"skipped": True}

    # ── Step 1: Tracklet identity ──────────────────────────────────────────────
    sub_tids, obj_tids, n_tracklets = assign_tracklet_ids(relations)

    # ── Step 2: Motif extraction ───────────────────────────────────────────────
    chains, triangles, stars = extract_all_motifs(
        relations, sub_tids, obj_tids, tau
    )
    all_motifs = chains + triangles + stars

    stats = {
        "n_relations":  len(relations),
        "n_tracklets":  n_tracklets,
        "n_chain":      len(chains),
        "n_triangle":   len(triangles),
        "n_star":       len(stars),
        "n_hyperedges": len(relations) + len(all_motifs),
        "skipped":      False,
    }

    use_sparse = len(relations) > _SPARSE_THRESHOLD

    # ── Step 3: Weighted incidence matrix ─────────────────────────────────────
    H_w, hyperedges = build_weighted_incidence(
        relations, sub_tids, obj_tids, all_motifs,
        n_tracklets=n_tracklets, beta=beta, use_sparse=use_sparse,
    )

    if H_w.shape[1] == 0:
        logger.warning("  ↳ empty H_w, returning original scores")
        return copy.deepcopy(relations), stats

    # ── Step 4: Laplacian + init scores ───────────────────────────────────────
    Delta, d_v = compute_laplacian(H_w, use_sparse=use_sparse)
    alpha = init_node_scores(relations, sub_tids, obj_tids, n_tracklets)

    # Guard: zero-degree tracklets get no update — their α already has global mean
    zero_deg = d_v < _EPS
    if zero_deg.any():
        logger.debug("  ↳ %d zero-degree tracklets (will keep init score)",
                     zero_deg.sum())

    # ── Step 5: Solve linear system ───────────────────────────────────────────
    alpha_prime = solve_laplacian(Delta, alpha, gamma, use_sparse=use_sparse)

    # Restore zero-degree nodes to their original scores (skip in smoothing)
    alpha_prime = np.where(zero_deg, alpha, alpha_prime)

    # ── Step 6: Blend scores ───────────────────────────────────────────────────
    det_scores, tag_scores = blend_scores(
        relations, sub_tids, obj_tids, alpha_prime, delta, delta_tagging
    )

    # Build output (deep copy to avoid mutating the caller's data)
    refined_relations = copy.deepcopy(relations)
    for i, rel in enumerate(refined_relations):
        rel["score"] = det_scores[i]
        if tag_scores is not None:
            rel["tagging_score"] = tag_scores[i]

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    stats["elapsed_ms"] = elapsed_ms

    if verbose:
        logger.info(
            "  ↳ rels=%d  tracklets=%d  chain=%d  tri=%d  star=%d  "
            "hyperedges=%d  time=%.1f ms",
            stats["n_relations"], n_tracklets,
            stats["n_chain"], stats["n_triangle"], stats["n_star"],
            stats["n_hyperedges"], elapsed_ms,
        )

    return refined_relations, stats


def process_prediction(
    prediction: Dict,
    tau: int = 3,
    beta: float = 0.10,
    gamma: float = 0.50,
    delta: float = 0.40,
    delta_tagging: Optional[float] = 0.30,
    dry_run: bool = False,
    verbose: bool = False,
) -> Tuple[Dict, Dict]:
    """Apply DWMH-SGG refinement to an entire VidVRD-II prediction dict.

    Args:
        prediction: The JSON dict with keys 'version' and 'results'.
        tau:        Minimum temporal overlap for motif extraction.
        beta:       Jaccard re-weighting strength.
        gamma:      Laplacian smoothing strength.
        delta:         Score blending weight for detection.
        delta_tagging: Score blending weight for tagging (stored as 'tagging_score').
        dry_run:       If True, process only the first 10 videos.
        verbose:       Log per-video stats when True.

    Returns:
        refined_prediction: New dict in the same format as input.
        all_stats:          Per-video stats keyed by video ID.
    """
    from tqdm import tqdm

    video_ids = list(prediction["results"].keys())
    if dry_run:
        video_ids = video_ids[:10]
        logger.info("DRY RUN: processing %d / %d videos", len(video_ids),
                    len(prediction["results"]))

    refined_results: Dict = {}
    all_stats: Dict = {}

    for vid in tqdm(video_ids, desc="DWMH-SGG refinement", unit="vid"):
        relations = prediction["results"][vid]
        logger.info("Video %s | %d relations", vid, len(relations))

        refined_rels, stats = refine_video(
            relations, tau=tau, beta=beta, gamma=gamma, delta=delta,
            delta_tagging=delta_tagging, verbose=verbose,
        )
        refined_results[vid] = refined_rels
        all_stats[vid] = stats

        if verbose or logger.isEnabledFor(logging.DEBUG):
            logger.info(
                "  chain=%d  triangle=%d  star=%d  hyperedges=%d  "
                "time=%.1f ms",
                stats.get("n_chain", 0),
                stats.get("n_triangle", 0),
                stats.get("n_star", 0),
                stats.get("n_hyperedges", 0),
                stats.get("elapsed_ms", 0.0),
            )

    refined_prediction = {
        "version": prediction["version"],
        "results": refined_results,
    }
    return refined_prediction, all_stats
