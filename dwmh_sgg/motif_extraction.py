"""Temporal motif extraction from predicted video relations.

Detects three motif types from the spatio-temporal relation graph:
  - Chain (3-node):   A→B→C  (B is shared relay)
  - Triangle (3-node): A→B, B→C, C→A  (directed cycle)
  - Star (4+-node):   one hub subject → 3+ object peripherals
"""

import logging
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def assign_tracklet_ids(
    relations: List[Dict],
) -> Tuple[List[int], List[int], int]:
    """Assign integer IDs to unique object trajectories across all relations.

    Two trajectory appearances receive the same ID when their
    (fstart, fend, first_bbox_4dec, last_bbox_4dec) fingerprint matches.
    This works because the VidVRD-II association step re-uses the same
    underlying tracklet bounding-box sequence for every relation that
    involves that tracklet.

    Args:
        relations: List of relation dicts with keys
            'duration', 'sub_traj', 'obj_traj'.

    Returns:
        sub_tids:    Subject tracklet ID for each relation (len == len(relations)).
        obj_tids:    Object  tracklet ID for each relation.
        n_tracklets: Total number of unique tracklets discovered.
    """
    fingerprint_to_id: Dict[Tuple, int] = {}
    sub_tids: List[int] = []
    obj_tids: List[int] = []

    def _fp(traj: List, duration: List) -> Tuple:
        """4-decimal rounded (fstart, fend, first_bbox, last_bbox) fingerprint."""
        arr = np.asarray(traj, dtype=np.float32)
        first = tuple(np.round(arr[0], 4).tolist())
        last = tuple(np.round(arr[-1], 4).tolist())
        return (int(duration[0]), int(duration[1]), first, last)

    for rel in relations:
        sf = _fp(rel["sub_traj"], rel["duration"])
        of = _fp(rel["obj_traj"], rel["duration"])

        if sf not in fingerprint_to_id:
            fingerprint_to_id[sf] = len(fingerprint_to_id)
        if of not in fingerprint_to_id:
            fingerprint_to_id[of] = len(fingerprint_to_id)

        sub_tids.append(fingerprint_to_id[sf])
        obj_tids.append(fingerprint_to_id[of])

    return sub_tids, obj_tids, len(fingerprint_to_id)


def temporal_overlap(dur1: List[int], dur2: List[int]) -> int:
    """Return the number of frames shared by two [fstart, fend) intervals.

    Args:
        dur1: [fstart, fend] of the first interval.
        dur2: [fstart, fend] of the second interval.

    Returns:
        Overlap in frames; 0 if the intervals do not overlap.
    """
    return max(0, min(dur1[1], dur2[1]) - max(dur1[0], dur2[0]))


def _multiway_overlap(durations: List[List[int]]) -> int:
    """Return the common temporal extent across all given intervals.

    Args:
        durations: List of [fstart, fend] intervals.

    Returns:
        Frames that are covered by every interval; 0 if no common window.
    """
    if not durations:
        return 0
    common_start = max(d[0] for d in durations)
    common_end = min(d[1] for d in durations)
    return max(0, common_end - common_start)


def extract_chain_motifs(
    relations: List[Dict],
    sub_tids: List[int],
    obj_tids: List[int],
    tau: int,
) -> List[Dict]:
    """Extract 3-node chain motifs  A→B→C  from relation predictions.

    A valid chain requires:
      - Relation i  : sub_tid[i] == A,  obj_tid[i] == B  (A→B)
      - Relation j  : sub_tid[j] == B,  obj_tid[j] == C  (B→C)
      - B is the relay: obj_tid[i] == sub_tid[j]
      - A ≠ B ≠ C ≠ A  (no repeated nodes)
      - temporal_overlap(dur_i, dur_j) ≥ tau

    Each returned motif dict contains:
      motif_type, rel_indices, node_tids (A,B,C),
      roles {tid: role}, temporal_overlap.

    Args:
        relations: List of relation dicts.
        sub_tids:  Subject tracklet IDs per relation.
        obj_tids:  Object  tracklet IDs per relation.
        tau:       Minimum temporal overlap in frames.

    Returns:
        List of chain motif dicts.
    """
    # Index: which relations have each tracklet as object / subject
    as_obj: Dict[int, List[int]] = defaultdict(list)  # tid → [rel_idx where it is obj]
    as_sub: Dict[int, List[int]] = defaultdict(list)  # tid → [rel_idx where it is sub]
    for i, (st, ot) in enumerate(zip(sub_tids, obj_tids)):
        as_sub[st].append(i)
        as_obj[ot].append(i)

    motifs: List[Dict] = []
    seen: set = set()

    relay_tids = set(as_obj.keys()) & set(as_sub.keys())
    for b_tid in relay_tids:
        for i in as_obj[b_tid]:      # relation  A→B  (i is object of B)
            a_tid = sub_tids[i]
            for j in as_sub[b_tid]:  # relation  B→C  (j starts at B)
                if i == j:
                    continue
                c_tid = obj_tids[j]
                # Ensure all three nodes are distinct
                if a_tid == b_tid or b_tid == c_tid or a_tid == c_tid:
                    continue

                ovlp = temporal_overlap(relations[i]["duration"], relations[j]["duration"])
                if ovlp < tau:
                    continue

                key = (i, j)  # directed: order matters
                if key in seen:
                    continue
                seen.add(key)

                motifs.append({
                    "motif_type": "chain",
                    "rel_indices": (i, j),
                    "node_tids": (a_tid, b_tid, c_tid),
                    "roles": {
                        a_tid: "source",   # peripheral, d=+1, s=0.5
                        b_tid: "relay",    # hub,        d=+1, s=1.0
                        c_tid: "sink",     # peripheral, d=-1, s=0.5
                    },
                    "temporal_overlap": ovlp,
                })

    return motifs


def extract_triangle_motifs(
    relations: List[Dict],
    sub_tids: List[int],
    obj_tids: List[int],
    tau: int,
) -> List[Dict]:
    """Extract 3-node directed triangle motifs  A→B, B→C, C→A.

    All three temporal extents must share a common window of ≥ tau frames.
    The canonical key is the sorted triple of relation indices to prevent
    finding the same triangle via different entry edges.

    Args:
        relations: List of relation dicts.
        sub_tids:  Subject tracklet IDs per relation.
        obj_tids:  Object  tracklet IDs per relation.
        tau:       Minimum three-way temporal overlap in frames.

    Returns:
        List of triangle motif dicts.
    """
    # Map (sub_tid, obj_tid) → [relation indices]
    pair_to_rels: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for i, (st, ot) in enumerate(zip(sub_tids, obj_tids)):
        pair_to_rels[(st, ot)].append(i)

    motifs: List[Dict] = []
    seen: set = set()

    for i, (a_tid, b_tid) in enumerate(zip(sub_tids, obj_tids)):
        if a_tid == b_tid:
            continue
        # Look for B→C edges
        for (s2, c_tid), j_list in pair_to_rels.items():
            if s2 != b_tid:
                continue
            if c_tid in (a_tid, b_tid):
                continue
            # Look for C→A  (closing the cycle)
            k_list = pair_to_rels.get((c_tid, a_tid), [])
            if not k_list:
                continue

            for j in j_list:
                for k in k_list:
                    if len({i, j, k}) < 3:
                        continue
                    canon_key = tuple(sorted([i, j, k]))
                    if canon_key in seen:
                        continue

                    ovlp = _multiway_overlap([
                        relations[i]["duration"],
                        relations[j]["duration"],
                        relations[k]["duration"],
                    ])
                    if ovlp < tau:
                        continue

                    seen.add(canon_key)
                    motifs.append({
                        "motif_type": "triangle",
                        "rel_indices": (i, j, k),
                        "node_tids": (a_tid, b_tid, c_tid),
                        "roles": {
                            a_tid: "triangle",  # equal role, d=+1, s=1.0
                            b_tid: "triangle",
                            c_tid: "triangle",
                        },
                        "temporal_overlap": ovlp,
                    })

    return motifs


def extract_star_motifs(
    relations: List[Dict],
    sub_tids: List[int],
    obj_tids: List[int],
    tau: int,
    min_degree: int = 3,
    max_stars_per_center: int = 20,
) -> List[Dict]:
    """Extract star motifs: one hub subject → ≥ min_degree object peripherals.

    For each candidate hub tracklet, the algorithm first tests whether ALL its
    outgoing relations share a common τ-frame window; if so, one large star is
    recorded.  Otherwise it enumerates 3-combinations and keeps every valid one
    up to max_stars_per_center.

    Args:
        relations:            List of relation dicts.
        sub_tids:             Subject tracklet IDs per relation.
        obj_tids:             Object  tracklet IDs per relation.
        tau:                  Minimum temporal overlap in frames.
        min_degree:           Minimum number of peripheral relations (default 3).
        max_stars_per_center: Cap on motifs emitted per center (default 20).

    Returns:
        List of star motif dicts.
    """
    # center_tid → list of relation indices where it appears as subject
    center_rels: Dict[int, List[int]] = defaultdict(list)
    for i, st in enumerate(sub_tids):
        center_rels[st].append(i)

    motifs: List[Dict] = []
    seen: set = set()

    for center_tid, rels_out in center_rels.items():
        # Filter self-relations (sub == obj)
        valid = [(i, obj_tids[i]) for i in rels_out if obj_tids[i] != center_tid]
        if len(valid) < min_degree:
            continue

        n_found = 0

        def _record_star(combo_valid: List[Tuple[int, int]], ovlp: int) -> None:
            nonlocal n_found
            canon = (center_tid, tuple(sorted(v[0] for v in combo_valid)))
            if canon in seen:
                return
            seen.add(canon)
            p_tids = tuple(v[1] for v in combo_valid)
            roles: Dict[int, str] = {center_tid: "hub"}
            for ptid in p_tids:
                roles[ptid] = "peripheral"
            motifs.append({
                "motif_type": "star",
                "rel_indices": tuple(v[0] for v in combo_valid),
                "node_tids": (center_tid,) + p_tids,
                "roles": roles,
                "temporal_overlap": ovlp,
            })
            n_found += 1

        # Try the full set first (cheapest check)
        full_ovlp = _multiway_overlap([relations[i]["duration"] for i, _ in valid])
        if full_ovlp >= tau:
            _record_star(valid, full_ovlp)
        else:
            # Enumerate exactly min_degree combinations
            for combo in combinations(valid, min_degree):
                if n_found >= max_stars_per_center:
                    break
                ovlp = _multiway_overlap([relations[i]["duration"] for i, _ in combo])
                if ovlp >= tau:
                    _record_star(list(combo), ovlp)

    return motifs


def extract_all_motifs(
    relations: List[Dict],
    sub_tids: List[int],
    obj_tids: List[int],
    tau: int,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Extract all three motif types from a video's predicted relations.

    Args:
        relations: List of relation dicts.
        sub_tids:  Subject tracklet IDs per relation.
        obj_tids:  Object  tracklet IDs per relation.
        tau:       Minimum temporal overlap in frames.

    Returns:
        Tuple (chain_motifs, triangle_motifs, star_motifs).
    """
    chains    = extract_chain_motifs(relations, sub_tids, obj_tids, tau)
    triangles = extract_triangle_motifs(relations, sub_tids, obj_tids, tau)
    stars     = extract_star_motifs(relations, sub_tids, obj_tids, tau)
    return chains, triangles, stars
