"""Hypergraph Laplacian computation and smoothed score inference.

Given the weighted incidence matrix H_w (V × K) the algorithm:

  1. Derives degree matrices  D_v  (vertex) and  D_e  (edge) from |H_w|.
  2. Constructs  M = D_v^{-½} H_w D_e^{-½}  so that  Θ = M M^T.
  3. Forms the normalised Laplacian  Δ = I − Θ.
  4. Initialises node scores  α  as the mean prediction score per tracklet.
  5. Solves the linear system  (I + γ·Δ) α' = α  via scipy.linalg.solve
     (with lstsq fallback) to obtain smoothed node scores  α'.

Zero-degree tracklets (not covered by any hyperedge) are detected and
returned unchanged to prevent division by zero.
"""

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

_EPS = 1e-8


def init_node_scores(
    relations: list,
    sub_tids: list,
    obj_tids: list,
    n_tracklets: int,
) -> np.ndarray:
    """Initialise per-tracklet scores as mean relation score over all appearances.

    Every relation contributes its score to both its subject and object tracklet.
    Tracklets that appear in no relation receive the global mean score.

    Args:
        relations:   List of relation dicts containing 'score'.
        sub_tids:    Subject tracklet ID per relation.
        obj_tids:    Object  tracklet ID per relation.
        n_tracklets: Total number of unique tracklets.

    Returns:
        alpha: Float64 array of shape (n_tracklets,) with initial scores.
    """
    score_sum = np.zeros(n_tracklets, dtype=np.float64)
    count     = np.zeros(n_tracklets, dtype=np.int64)

    for i, rel in enumerate(relations):
        s = float(rel["score"])
        score_sum[sub_tids[i]] += s
        count[sub_tids[i]]     += 1
        score_sum[obj_tids[i]] += s
        count[obj_tids[i]]     += 1

    valid = count > 0
    global_mean = (score_sum[valid].sum() / count[valid].sum()) if valid.any() else 0.0

    alpha = np.where(valid, score_sum / np.maximum(count, 1), global_mean)
    return alpha.astype(np.float64)


def compute_laplacian(H_w, use_sparse: bool = False) -> Tuple:
    """Compute the normalised directed hypergraph Laplacian  Δ = I − Θ.

    Algorithm:
        |H_w|  := element-wise absolute value of H_w
        D_v    := diag( |H_w| @ 1_K )          — vertex degree matrix
        D_e    := diag( |H_w|^T @ 1_V )        — edge degree matrix
        M      := D_v^{-½} · H_w · D_e^{-½}   — normalised incidence
        Θ      := M M^T                         — positive semi-definite
        Δ      := I − Θ

    Zero-degree entries in D_v / D_e are regularised by ε = 1e-8.

    Args:
        H_w:        Weighted incidence matrix (V × K); dense np.ndarray or
                    scipy.sparse.csr_matrix.
        use_sparse: Set True when H_w is a sparse matrix.

    Returns:
        Delta:  Laplacian matrix of shape (V, V); dense or sparse.
        d_v:    Raw vertex degree vector (for diagnosing zero-degree nodes).
    """
    V = H_w.shape[0]

    if use_sparse:
        from scipy.sparse import diags, eye

        abs_H = H_w.copy()
        abs_H.data = np.abs(abs_H.data)
        ones_K = np.ones(H_w.shape[1])
        ones_V = np.ones(V)

        d_v = np.asarray(abs_H @ ones_K).flatten()
        d_e = np.asarray(abs_H.T @ ones_V).flatten()

        dv_inv_sqrt = 1.0 / np.sqrt(d_v + _EPS)
        de_inv_sqrt = 1.0 / np.sqrt(d_e + _EPS)

        M = diags(dv_inv_sqrt) @ H_w @ diags(de_inv_sqrt)
        Theta = M @ M.T
        Delta = eye(V, format="csr") - Theta
        return Delta, d_v

    # ── Dense path ─────────────────────────────────────────────────────────────
    abs_H = np.abs(H_w)
    d_v = abs_H @ np.ones(H_w.shape[1])   # (V,)
    d_e = abs_H.T @ np.ones(V)            # (K,)

    dv_inv_sqrt = 1.0 / np.sqrt(d_v + _EPS)  # (V,)
    de_inv_sqrt = 1.0 / np.sqrt(d_e + _EPS)  # (K,)

    # M[i,k] = dv_inv_sqrt[i] * H_w[i,k] * de_inv_sqrt[k]
    M = dv_inv_sqrt[:, np.newaxis] * H_w * de_inv_sqrt[np.newaxis, :]  # (V,K)

    # Θ = M M^T  (symmetric PSD, eigenvalues in [0, 1] when H is properly normed)
    Theta = M @ M.T   # (V, V)

    Delta = np.eye(V, dtype=np.float64) - Theta
    return Delta, d_v


def solve_laplacian(
    Delta,
    alpha: np.ndarray,
    gamma: float,
    use_sparse: bool = False,
) -> np.ndarray:
    """Solve  (I + γ·Δ) α' = α  for the smoothed node score vector α'.

    The system matrix  A = I + γ·Δ  is symmetric positive definite for γ > 0
    because Δ = I − Θ and Θ is PSD with eigenvalues ≤ 1, making A's eigenvalues
    ≥ 1 − γ·(max_eig(Θ) − 1) ≥ γ > 0.

    A scipy.linalg.lstsq fallback is used if solve raises a LinAlgError.

    Args:
        Delta:      Laplacian matrix (V × V), dense or sparse.
        alpha:      Initial node score vector (V,).
        gamma:      Smoothing strength  γ > 0.
        use_sparse: Use scipy.sparse.linalg.spsolve when True.

    Returns:
        alpha_prime: Smoothed node scores (V,); same shape as alpha.
    """
    V = len(alpha)

    if use_sparse:
        from scipy.sparse import eye
        from scipy.sparse.linalg import spsolve, lsqr

        A = eye(V, format="csr") + gamma * Delta
        try:
            x = spsolve(A, alpha)
            if np.any(~np.isfinite(x)):
                raise RuntimeError("spsolve returned non-finite values")
        except Exception as exc:
            logger.warning("spsolve failed (%s); falling back to lsqr.", exc)
            result = lsqr(A, alpha)
            x = result[0]
        return x

    # ── Dense path ─────────────────────────────────────────────────────────────
    from scipy.linalg import solve, lstsq

    A = np.eye(V, dtype=np.float64) + gamma * Delta

    try:
        x = solve(A, alpha, assume_a="pos")
        if not np.all(np.isfinite(x)):
            raise np.linalg.LinAlgError("Non-finite solution from solve.")
    except Exception as exc:
        logger.warning("linalg.solve failed (%s); falling back to lstsq.", exc)
        x, _, _, _ = lstsq(A, alpha)

    return x
