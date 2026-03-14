"""DWMH-SGG main entry point.

Usage (from E:/PHD/ST-SGG_Models/DWMH-SGG/):

    python -m dwmh_sgg.main \\
        --prediction ../imagenet-vidvrd-baseline-output/models/3step_prop_wd0.01/video_relations.json \\
        --output     ../imagenet-vidvrd-baseline-output/models/3step_prop_wd0.01/video_relations_dwmh.json \\
        --tau 1 --beta 0.1 --gamma 0.5 --delta 0.3 \\
        --evaluate --verbose

For a quick sanity check on 10 videos add --dry_run.
"""

import argparse
import json
import logging
import os
import sys
import time

# ── stdlib logging setup ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dwmh_sgg.main")


def parse_args(argv=None):
    """Parse command-line arguments for the DWMH-SGG refinement pipeline."""
    parser = argparse.ArgumentParser(
        description="DWMH-SGG: post-hoc hypergraph score refinement for VidVRD-II.",
    )
    parser.add_argument(
        "--prediction",
        required=True,
        help="Path to the VidVRD-II baseline prediction JSON.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Where to write the refined prediction JSON.",
    )
    parser.add_argument(
        "--groundtruth",
        default=None,
        help="(Unused placeholder; evaluation uses the VidVRD-II evaluator directly.)",
    )
    # ── Algorithm hyper-parameters ────────────────────────────────────────────
    parser.add_argument(
        "--tau", type=int, default=3,
        help="Minimum temporal overlap (frames) for motif detection (default 3).",
    )
    parser.add_argument(
        "--beta", type=float, default=0.10,
        help="Jaccard co-occurrence re-weighting strength β (default 0.10).",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.50,
        help="Laplacian smoothing strength γ (default 0.50).",
    )
    parser.add_argument(
        "--delta", type=float, default=0.40,
        help="Detection score blending weight δ_det ∈ [0,1] (default 0.40).",
    )
    parser.add_argument(
        "--delta_tagging", type=float, default=0.30,
        help="Tagging score blending weight δ_tag ∈ [0,1] (default 0.30). "
             "Stored as 'tagging_score' in output; used for P@k metrics.",
    )
    # ── Run-mode flags ────────────────────────────────────────────────────────
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Process only the first 10 videos (quick sanity check).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-video motif/hyperedge/timing stats.",
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="After refinement, run the VidVRD-II evaluator and print comparison.",
    )
    # ── Paths for evaluation ──────────────────────────────────────────────────
    parser.add_argument(
        "--vidvrd_path",
        default=os.path.join(os.path.dirname(__file__), "..", "..", "VidVRD-II"),
        help="Path to the VidVRD-II directory (needed for --evaluate).",
    )
    parser.add_argument(
        "--dataset_path",
        default=os.path.join(os.path.dirname(__file__), "..", "..", "..",
                             "imagenet-vidvrd-dataset"),
        help="Path to the imagenet-vidvrd-dataset directory.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Entry point for the DWMH-SGG pipeline."""
    args = parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Load baseline prediction ──────────────────────────────────────────────
    pred_path = os.path.abspath(args.prediction)
    logger.info("Loading prediction from %s", pred_path)
    with open(pred_path, "r") as f:
        prediction = json.load(f)
    logger.info("  version=%s  videos=%d", prediction["version"],
                len(prediction["results"]))

    # ── Validate hyper-parameters ─────────────────────────────────────────────
    assert args.tau >= 0,                    "--tau must be >= 0"
    assert 0.0 <= args.delta <= 1.0,         "--delta must be in [0, 1]"
    assert 0.0 <= args.delta_tagging <= 1.0, "--delta_tagging must be in [0, 1]"
    assert args.gamma > 0.0,                 "--gamma must be > 0"
    assert args.beta >= 0.0,                 "--beta must be >= 0"

    logger.info(
        "Hyper-parameters: tau=%d  beta=%.3f  gamma=%.3f  delta_det=%.3f  delta_tag=%.3f",
        args.tau, args.beta, args.gamma, args.delta, args.delta_tagging,
    )

    # ── Run refinement ────────────────────────────────────────────────────────
    from .refinement import process_prediction

    t_start = time.perf_counter()
    refined_prediction, all_stats = process_prediction(
        prediction,
        tau=args.tau,
        beta=args.beta,
        gamma=args.gamma,
        delta=args.delta,
        delta_tagging=args.delta_tagging,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )
    elapsed_total = time.perf_counter() - t_start

    # ── Aggregate stats ───────────────────────────────────────────────────────
    n_vid      = len(all_stats)
    n_chains   = sum(s.get("n_chain", 0)    for s in all_stats.values())
    n_triangles= sum(s.get("n_triangle", 0) for s in all_stats.values())
    n_stars    = sum(s.get("n_star", 0)     for s in all_stats.values())
    n_he       = sum(s.get("n_hyperedges",0) for s in all_stats.values())
    n_skipped  = sum(1 for s in all_stats.values() if s.get("skipped"))

    logger.info(
        "Done: %d videos  (skipped=%d)  chains=%d  triangles=%d  stars=%d  "
        "total_hyperedges=%d  wall=%.1fs",
        n_vid, n_skipped, n_chains, n_triangles, n_stars, n_he, elapsed_total,
    )

    # ── Save output ───────────────────────────────────────────────────────────
    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(refined_prediction, f)
    logger.info("Refined predictions saved to %s", out_path)

    # ── Optional evaluation ───────────────────────────────────────────────────
    if args.evaluate and not args.dry_run:
        from .evaluate import compare_and_print
        compare_and_print(
            baseline_path=pred_path,
            refined_path=out_path,
            vidvrd_path=os.path.abspath(args.vidvrd_path),
            dataset_path=os.path.abspath(args.dataset_path),
        )
    elif args.evaluate and args.dry_run:
        logger.info("Skipping evaluation in dry_run mode.")

    return refined_prediction, all_stats


if __name__ == "__main__":
    main()
