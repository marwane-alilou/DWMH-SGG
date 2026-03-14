"""Directed Weighted Motif Hypergraph construction.

Implements the 3-step algorithm that produces the weighted incidence matrix H_w:

  Step 1 — Build directed weighted incidence H_tilde  (n_tracklets × K)
            H_tilde(v, e) = r(v,e) · d(v,e) · s(v,e)
            where  r = membership {0,1},
                   d = direction {+1 subject-role, -1 object-role, 0 neutral},
                   s = centrality {relay/hub→1.0, peripheral→0.5, triangle→1.0}

  Step 2 — Hyperedge confidence weights  w(e) = geometric_mean(scores in e)
            H_w = H_tilde ⊙ w   (column-wise scaling)

  Step 3 — Jaccard co-occurrence re-weighting
            w_new(e_k) = w(e_k) · (1 + β · Σ_{j≠k} T(e_k, e_j))
            where  T(j,k) = Jaccard similarity of node sets
            Rebuild H_w with w_new.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_EPS = 1e-8

# ─────────────────────────────────────────────────────────────────────────────
# Role → (direction d, centrality s)  look-up
# ─────────────────────────────────────────────────────────────────────────────
_ROLE_DS: Dict[str, Tuple[float, float]] = {
    # Pair hyperedge roles
    "subject":    (+1.0, 1.0),
    "object":     (-1.0, 1.0),
    # Chain motif roles
    "source":     (+1.0, 0.5),   # peripheral: d=+1, s=0.5
    "relay":      (+1.0, 1.0),   # hub:        d=+1, s=1.0 (relays forward)
    "sink":       (-1.0, 0.5),   # peripheral: d=-1, s=0.5
    # Triangle motif roles (all equal)
    "triangle":   (+1.0, 1.0),
    # Star motif roles
    "hub":        (+1.0, 1.0),
    "peripheral": (-1.0, 0.5),
}


def _h_tilde_entry(tid: int, edge: Dict) -> float:
    """Compute H_tilde(tid, edge) = r·d·s for one (tracklet, hyperedge) pair.

    Args:
        tid:  Tracklet ID.
        edge: Hyperedge dict with 'roles' mapping {tid: role_str}.

    Returns:
        Signed centrality weight; 0.0 if tid is not a member.
    """
    role = edge["roles"].get(tid)
    if role is None:
        return 0.0
    d, s = _ROLE_DS.get(role, (0.0, 1.0))
    return d * s


def build_hyperedge_list(
    relations: List[Dict],
    sub_tids: List[int],
    obj_tids: List[int],
    motifs: List[Dict],
) -> List[Dict]:
    """Create a flat list of all hyperedges (pair edges first, then motif edges).

    Each hyperedge dict exposes:
        edge_type   : 'pair' | 'chain' | 'triangle' | 'star'
        node_tids   : tuple of member tracklet IDs (unique within the edge)
        roles       : {tid: role_str}
        rel_indices : tuple of indices into the relations list

    Args:
        relations:  List of relation dicts.
        sub_tids:   Subject tracklet IDs per relation.
        obj_tids:   Object  tracklet IDs per relation.
        motifs:     Flat list of all extracted motif dicts.

    Returns:
        List of hyperedge dicts ordered (pair edges, motif edges).
    """
    hyperedges: List[Dict] = []

    # ── Pair hyperedges (one per relation) ────────────────────────────────────
    for i in range(len(relations)):
        st, ot = sub_tids[i], obj_tids[i]
        if st == ot:
            # Self-relation: assign neutral subject role to avoid cancellation
            roles = {st: "subject"}
            node_tids = (st,)
        else:
            roles = {st: "subject", ot: "object"}
            node_tids = (st, ot)
        hyperedges.append({
            "edge_type":   "pair",
            "node_tids":   node_tids,
            "roles":       roles,
            "rel_indices": (i,),
        })

    # ── Motif hyperedges ───────────────────────────────────────────────────────
    for motif in motifs:
        hyperedges.append({
            "edge_type":   motif["motif_type"],
            "node_tids":   motif["node_tids"],
            "roles":       motif["roles"],
            "rel_indices": motif["rel_indices"],
        })

    return hyperedges


def build_h_tilde(
    hyperedges: List[Dict],
    n_tracklets: int,
    use_sparse: bool = False,
):
    """Build the signed centrality incidence matrix H_tilde of shape (V, K).

    Args:
        hyperedges:  List of hyperedge dicts.
        n_tracklets: Number of unique tracklets (V).
        use_sparse:  If True return scipy.sparse.csr_matrix; else np.ndarray.

    Returns:
        H_tilde: Matrix of shape (n_tracklets, len(hyperedges)).
    """
    K = len(hyperedges)

    if use_sparse:
        from scipy.sparse import lil_matrix, csr_matrix
        H = lil_matrix((n_tracklets, K), dtype=np.float64)
        for e_idx, edge in enumerate(hyperedges):
            for tid in edge["node_tids"]:
                if 0 <= tid < n_tracklets:
                    val = _h_tilde_entry(tid, edge)
                    if val != 0.0:
                        H[tid, e_idx] = val
        return csr_matrix(H)

    H = np.zeros((n_tracklets, K), dtype=np.float64)
    for e_idx, edge in enumerate(hyperedges):
        for tid in edge["node_tids"]:
            if 0 <= tid < n_tracklets:
                H[tid, e_idx] = _h_tilde_entry(tid, edge)
    return H


def compute_edge_weights(
    relations: List[Dict],
    hyperedges: List[Dict],
) -> np.ndarray:
    """Compute confidence weight w(e) = geometric_mean(scores of relations in e).

    For a pair hyperedge (single relation) this is just that relation's score.
    For motif hyperedges it is the geometric mean over all constituent scores.

    Args:
        relations:  List of relation dicts containing 'score'.
        hyperedges: List of hyperedge dicts containing 'rel_indices'.

    Returns:
        w: Array of shape (K,) with positive confidence weights.
    """
    w = np.empty(len(hyperedges), dtype=np.float64)
    for e_idx, edge in enumerate(hyperedges):
        scores = [max(float(relations[i]["score"]), _EPS) for i in edge["rel_indices"]]
        if len(scores) == 1:
            w[e_idx] = scores[0]
        else:
            # Geometric mean via log space (numerically stable)
            w[e_idx] = float(np.exp(np.mean(np.log(scores))))
    return w


def compute_jaccard_matrix(hyperedges: List[Dict]) -> np.ndarray:
    """Compute pairwise Jaccard similarity between hyperedge node sets.

    T(j, k) = |nodes(e_j) ∩ nodes(e_k)| / |nodes(e_j) ∪ nodes(e_k)|

    The diagonal is zero (an edge is not compared with itself for re-weighting).

    Args:
        hyperedges: List of hyperedge dicts.

    Returns:
        T: Symmetric float64 matrix of shape (K, K).
    """
    K = len(hyperedges)
    node_sets = [set(e["node_tids"]) for e in hyperedges]
    T = np.zeros((K, K), dtype=np.float64)

    for j in range(K):
        for k in range(j + 1, K):
            inter = len(node_sets[j] & node_sets[k])
            if inter == 0:
                continue
            union = len(node_sets[j] | node_sets[k])
            t_jk = inter / max(union, 1)
            T[j, k] = t_jk
            T[k, j] = t_jk

    return T


def build_weighted_incidence(
    relations: List[Dict],
    sub_tids: List[int],
    obj_tids: List[int],
    motifs: List[Dict],
    n_tracklets: int,
    beta: float = 0.1,
    use_sparse: bool = False,
) -> Tuple:
    """Full 3-step construction of the weighted incidence matrix H_w.

    Steps:
      1. Build H_tilde from pair + motif hyperedges.
      2. Compute per-edge confidence weights via geometric mean of scores;
         column-scale H_tilde to get H_w.
      3. Re-weight each column by Jaccard co-occurrence boost, rebuild H_w.

    Args:
        relations:   List of relation dicts.
        sub_tids:    Subject tracklet IDs per relation.
        obj_tids:    Object  tracklet IDs per relation.
        motifs:      All extracted motif dicts (chain + triangle + star).
        n_tracklets: Total number of unique tracklets (V).
        beta:        Jaccard co-occurrence re-weighting strength (β ≥ 0).
        use_sparse:  Use scipy.sparse matrices when True.

    Returns:
        H_w:        Weighted incidence matrix (V × K); dense or sparse.
        hyperedges: List of hyperedge dicts (for downstream use).
    """
    hyperedges = build_hyperedge_list(relations, sub_tids, obj_tids, motifs)
    K = len(hyperedges)

    if K == 0:
        return np.zeros((n_tracklets, 0), dtype=np.float64), hyperedges

    # ── Step 1: directed weighted incidence ───────────────────────────────────
    H_tilde = build_h_tilde(hyperedges, n_tracklets, use_sparse=use_sparse)

    # ── Step 2: confidence weights ────────────────────────────────────────────
    w = compute_edge_weights(relations, hyperedges)

    if use_sparse:
        from scipy.sparse import diags
        H_w = H_tilde @ diags(w)
    else:
        H_w = H_tilde * w[np.newaxis, :]  # broadcast: (V,K) * (1,K)

    # ── Step 3: Jaccard co-occurrence re-weighting ────────────────────────────
    T = compute_jaccard_matrix(hyperedges)
    # sum over j≠k; diagonal of T is already 0, so sum(axis=1) is correct
    jaccard_boost = T.sum(axis=1)                    # shape (K,)
    w_new = w * (1.0 + beta * jaccard_boost)

    if use_sparse:
        from scipy.sparse import diags
        H_w = H_tilde @ diags(w_new)
    else:
        H_w = H_tilde * w_new[np.newaxis, :]

    logger.debug(
        "H_w built: V=%d  K=%d  (pair=%d  motif=%d)  β=%.3f",
        n_tracklets, K, len(relations), len(motifs), beta,
    )
    return H_w, hyperedges
