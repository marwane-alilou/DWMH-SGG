"""Evaluation and comparison of baseline vs. DWMH-SGG refined predictions.

Imports the VidVRD-II evaluator, runs it on both prediction JSONs,
and prints a formatted side-by-side comparison table.

Standalone usage (from E:/PHD/ST-SGG_Models/DWMH-SGG/):

    python -m dwmh_sgg.evaluate \\
        --baseline ../imagenet-vidvrd-baseline-output/models/3step_prop_wd0.01/video_relations.json \\
        --refined  ../imagenet-vidvrd-baseline-output/models/3step_prop_wd0.01/video_relations_dwmh.json \\
        --vidvrd_path   ../../VidVRD-II \\
        --dataset_path  ../../imagenet-vidvrd-dataset
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ── Metric display configuration ─────────────────────────────────────────────
#  (display_name, result_key_in_eval_dict, multiplier_to_percent)
_METRICS = [
    ("mAP",        "detection mean AP",      100.0),
    ("Recall@10",  "detection recall@10",    100.0),
    ("Recall@20",  "detection recall@20",    100.0),
    ("Recall@50",  "detection recall@50",    100.0),
    ("Recall@100", "detection recall@100",   100.0),
    ("P@1",        "tagging precision@1",    100.0),
    ("P@5",        "tagging precision@5",    100.0),
    ("P@10",       "tagging precision@10",   100.0),
]


def _load_vidvrd(vidvrd_path: str):
    """Temporarily add the VidVRD-II directory to sys.path and import modules.

    Args:
        vidvrd_path: Absolute path to the VidVRD-II repository.

    Returns:
        (evaluate_relation, VidVRD)  imported from VidVRD-II.
    """
    abs_path = os.path.abspath(vidvrd_path)
    if abs_path not in sys.path:
        sys.path.insert(0, abs_path)
    try:
        from evaluate import evaluate_relation   # VidVRD-II evaluate.py
        from dataset import VidVRD
        return evaluate_relation, VidVRD
    except ImportError as exc:
        raise ImportError(
            f"Could not import VidVRD-II modules from {abs_path}. "
            "Check --vidvrd_path."
        ) from exc


def _swap_tagging_scores(results: Dict) -> Dict:
    """Return a copy of results with 'tagging_score' promoted to 'score'.

    Used so that the standard VidVRD-II evaluator sees the tagging-specific
    blended score when computing P@k metrics.
    """
    import copy
    swapped = {}
    for vid, rels in results.items():
        new_rels = []
        for rel in rels:
            if "tagging_score" in rel:
                r = copy.copy(rel)
                r["score"] = rel["tagging_score"]
                new_rels.append(r)
            else:
                new_rels.append(rel)
        swapped[vid] = new_rels
    return swapped


def run_vidvrd_eval(
    pred_path: str,
    dataset_path: str,
    vidvrd_path: str,
) -> Dict:
    """Load a prediction JSON and evaluate it with the VidVRD-II evaluator.

    When the prediction contains 'tagging_score' fields (dual-delta mode),
    detection metrics use 'score' and tagging metrics use 'tagging_score'.

    Args:
        pred_path:    Path to the prediction JSON file.
        dataset_path: Path to the imagenet-vidvrd-dataset directory.
        vidvrd_path:  Path to the VidVRD-II repository.

    Returns:
        scores: Dict {setting: {metric: float}} as returned by evaluate_relation.
    """
    evaluate_relation, VidVRD = _load_vidvrd(vidvrd_path)

    with open(pred_path, "r") as f:
        pred = json.load(f)

    normalize_coords = pred["version"] >= "VERSION 2.1"
    dataset = VidVRD(
        dataset_path,
        os.path.join(dataset_path, "videos"),
        ["train", "test"],
        normalize_coords=normalize_coords,
    )

    results = pred["results"]

    # Check for dual-delta mode
    first_vid = next(iter(results.values()), [])
    has_tagging_score = bool(first_vid) and "tagging_score" in first_vid[0]

    if not has_tagging_score:
        return evaluate_relation(dataset, "test", results)

    # Dual-delta: run detection eval with 'score', tagging eval with 'tagging_score'
    from collections import defaultdict
    import numpy as np
    from evaluation.visual_relation_detection import (
        eval_detection_scores, eval_tagging_scores)
    from evaluation.common import voc_ap

    tag_results = _swap_tagging_scores(results)

    groundtruth = {vid: dataset.get_relation_insts(vid)
                   for vid in dataset.get_index("test")}

    def _eval_dual(gt, det_pred, tag_pred):
        video_ap = {}
        tot_scores  = defaultdict(list)
        tot_tp      = defaultdict(list)
        prec_at_n   = defaultdict(list)
        tot_gt      = 0
        det_nret = [10, 20, 50, 100]
        tag_nret = [1, 5, 10]
        for vid, gt_rels in gt.items():
            if not gt_rels:
                continue
            tot_gt += len(gt_rels)
            dp = det_pred.get(vid, [])
            tp = tag_pred.get(vid, [])
            det_prec, det_rec, det_sc = eval_detection_scores(gt_rels, dp, 0.5)
            video_ap[vid] = float(voc_ap(det_rec, det_prec))
            is_tp = np.isfinite(det_sc)
            for nre in det_nret:
                cut = min(nre, det_sc.size)
                tot_scores[nre].append(det_sc[:cut])
                tot_tp[nre].append(is_tp[:cut])
            tag_prec, _, _ = eval_tagging_scores(gt_rels, tp, max(tag_nret))
            for nre in tag_nret:
                prec_at_n[nre].append(tag_prec[nre - 1])
        from collections import OrderedDict
        out = OrderedDict()
        out["detection mean AP"] = float(np.mean(list(video_ap.values())))
        for nre in det_nret:
            sc = np.concatenate(tot_scores[nre])
            tp_arr = np.concatenate(tot_tp[nre])
            idx = np.argsort(sc)[::-1]
            tp_arr = tp_arr[idx]
            cum = np.cumsum(tp_arr).astype(np.float32)
            rec = cum / max(tot_gt, np.finfo(np.float32).eps)
            out[f"detection recall@{nre}"] = float(rec[-1])
        for nre in tag_nret:
            out[f"tagging precision@{nre}"] = float(np.mean(prec_at_n[nre]))
        return out

    overall = _eval_dual(groundtruth, results, tag_results)

    # Zero-shot settings
    zeroshot_triplets = dataset.get_triplets("test").difference(
        dataset.get_triplets("train"))

    zs_gt, gzs_gt = {}, {}
    zs_det, zs_tag, gzs_det, gzs_tag = {}, {}, {}, {}
    for vid in dataset.get_index("test"):
        gt_rels = dataset.get_relation_insts(vid)
        zs_rels = [r for r in gt_rels if tuple(r["triplet"]) in zeroshot_triplets]
        if not zs_rels:
            continue
        zs_gt[vid]  = zs_rels
        gzs_gt[vid] = zs_rels
        zs_det[vid]  = [r for r in results.get(vid, [])
                        if tuple(r["triplet"]) in zeroshot_triplets]
        zs_tag[vid]  = [r for r in tag_results.get(vid, [])
                        if tuple(r["triplet"]) in zeroshot_triplets]
        gzs_det[vid] = results.get(vid, [])
        gzs_tag[vid] = tag_results.get(vid, [])

    scores = {"overall": overall}
    if zs_gt:
        scores["zero-shot"]            = _eval_dual(zs_gt,  zs_det,  zs_tag)
        scores["generalized zero-shot"] = _eval_dual(gzs_gt, gzs_det, gzs_tag)
    return scores


def compare_and_print(
    baseline_path: str,
    refined_path: str,
    vidvrd_path: str,
    dataset_path: str,
    setting: str = "overall",
) -> None:
    """Evaluate both JSONs and print a side-by-side comparison table.

    Args:
        baseline_path: Path to the baseline (VidVRD-II) prediction JSON.
        refined_path:  Path to the DWMH-SGG refined prediction JSON.
        vidvrd_path:   Path to the VidVRD-II repository.
        dataset_path:  Path to the imagenet-vidvrd-dataset directory.
        setting:       Which evaluation setting to display (default 'overall').
    """
    logger.info("Evaluating baseline: %s", baseline_path)
    base_scores = run_vidvrd_eval(baseline_path, dataset_path, vidvrd_path)

    logger.info("Evaluating refined:  %s", refined_path)
    ref_scores  = run_vidvrd_eval(refined_path,  dataset_path, vidvrd_path)

    base_setting = base_scores.get(setting, {})
    ref_setting  = ref_scores.get(setting, {})

    # ── Print table ───────────────────────────────────────────────────────────
    col_w = 12
    print()
    print(f"{'Metric':<14}  {'Baseline':>{col_w}}  {'Refined':>{col_w}}  {'Delta':>{col_w}}")
    print("-" * (14 + 3 * (col_w + 2)))

    for display_name, key, mult in _METRICS:
        b_val = base_setting.get(key, float("nan")) * mult
        r_val = ref_setting.get(key, float("nan"))  * mult
        delta = r_val - b_val
        sign  = "+" if delta >= 0 else ""
        print(
            f"{display_name:<14}  {b_val:>{col_w}.2f}  {r_val:>{col_w}.2f}  "
            f"{sign}{delta:>{col_w - 1}.2f}"
        )

    print()
    # ── Zero-shot rows ────────────────────────────────────────────────────────
    for zs_setting in ("zero-shot", "generalized zero-shot"):
        bz = base_scores.get(zs_setting, {})
        rz = ref_scores.get(zs_setting, {})
        if not bz:
            continue
        print(f"[{zs_setting}]")
        for display_name, key, mult in _METRICS[:5]:  # detection metrics only
            b_val = bz.get(key, float("nan")) * mult
            r_val = rz.get(key, float("nan")) * mult
            delta = r_val - b_val
            sign  = "+" if delta >= 0 else ""
            print(
                f"  {display_name:<12}  {b_val:>{col_w}.2f}  {r_val:>{col_w}.2f}  "
                f"{sign}{delta:>{col_w - 1}.2f}"
            )
        print()


def main(argv=None):
    """Standalone entry point for the evaluation / comparison script."""
    parser = argparse.ArgumentParser(
        description="Compare baseline vs. DWMH-SGG refined VidVRD predictions."
    )
    parser.add_argument("--baseline",     required=True,
                        help="Path to the baseline prediction JSON.")
    parser.add_argument("--refined",      required=True,
                        help="Path to the refined prediction JSON.")
    parser.add_argument("--vidvrd_path",
                        default=os.path.join(os.path.dirname(__file__),
                                             "..", "..", "VidVRD-II"),
                        help="Path to the VidVRD-II directory.")
    parser.add_argument("--dataset_path",
                        default=os.path.join(os.path.dirname(__file__),
                                             "..", "..", "..",
                                             "imagenet-vidvrd-dataset"),
                        help="Path to the imagenet-vidvrd-dataset directory.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    compare_and_print(
        baseline_path=os.path.abspath(args.baseline),
        refined_path=os.path.abspath(args.refined),
        vidvrd_path=os.path.abspath(args.vidvrd_path),
        dataset_path=os.path.abspath(args.dataset_path),
    )


if __name__ == "__main__":
    main()
