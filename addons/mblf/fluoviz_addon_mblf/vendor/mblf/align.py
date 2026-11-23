"""Identifiability post-processing of factor loadings.

* `varimax_rsp`: Varimax-RSP (Papastamoulis & Ntzoufras 2022) of posterior draws of the stacked basis-coefficient
  loadings: raw varimax of every draw, then signed permutations that match all draws to a common reference.
* `functional_varimax` and `match_columns`: the weighted functional varimax of loading *functions* (columns of F,
  rows = locations of all modes, weights = quadrature weights times mode weights) and the signed matching of columns
  used to compare loadings with each other or with the truth (paper, Appendix "Evaluation procedures").
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


# ------------------------------------------------------------------------------------------------ Varimax-RSP
def varimax(L, tol=1e-6, max_iter=100):
    """Raw varimax rotation of a (p, K) matrix (the algorithm of scikit-learn). Returns (rotated, rotation)."""
    p, K = L.shape
    R = np.eye(K)
    var = 0.0
    for _ in range(max_iter):
        LR = L @ R
        u, s, vt = np.linalg.svd(L.T @ (LR**3 - LR * (LR**2).sum(axis=0) / p))
        R = u @ vt
        if var != 0 and s.sum() < var * (1 + tol):
            break
        var = s.sum()
    return L @ R, R


def varimax_rsp(lambda_draws, max_iter=100):
    """Varimax-RSP of loading draws (chain, draw, P, K), stacked over modes.

    Returns aligned draws (chain, draw, P, K), the varimax rotations R and signed permutations Q (chain, draw, K, K)
    with aligned = Lambda R Q, and the reference loadings (P, K) of the last averaging step.
    """
    lam = np.asarray(lambda_draws, dtype=float)
    C, S, P, K = lam.shape
    flat = lam.reshape(C * S, P, K)
    rotated = np.empty_like(flat)
    rotations = np.empty((C * S, K, K))
    for t in range(C * S):
        rotated[t], rotations[t] = varimax(flat[t])
    T = C * S
    norms = np.sum(rotated**2, axis=1)                                 # (T, K)
    s = np.ones((T, K))
    nu = np.tile(np.arange(K), (T, 1))
    tol = 1e-6 * T * P * K

    def aligned():
        return np.take_along_axis(rotated, nu[:, None, :], axis=2) * s[:, None, :]

    for _ in range(max_iter):
        current = aligned()
        ref = current.mean(axis=0)
        loss = np.sum((current - ref) ** 2)
        cross = np.einsum("tpi,pj->tji", rotated, ref)                 # (T, K_ref, K_draw)
        cost_ref = np.sum(ref**2, axis=0)
        new_loss = 0.0
        for t in range(T):
            cost = cost_ref[:, None] + norms[t][None, :] - 2 * np.abs(cross[t])
            rows, cols = linear_sum_assignment(cost)
            nu[t, rows] = cols
            s[t, rows] = np.where(cross[t, rows, cols] >= 0, 1.0, -1.0)
            new_loss += cost[rows, cols].sum()
        if loss - new_loss < tol:
            break
    Q = np.zeros((T, K, K))
    Q[np.repeat(np.arange(T), K), nu.reshape(-1), np.tile(np.arange(K), T)] = s.reshape(-1)
    return {"aligned": aligned().reshape(C, S, P, K), "rotations": rotations.reshape(C, S, K, K),
            "signed_permutations": Q.reshape(C, S, K, K), "reference": ref}


# ------------------------------------------------------------------------------- weighted functional varimax
def varimax_criterion(F, w):
    """sum_k { sum_i w_i F_ik^4 - (sum_i w_i F_ik^2)^2 }."""
    F2 = F * F
    return float(np.sum(w[:, None] * F2 * F2) - np.sum((w @ F2) ** 2))


def _start(q, r):
    """Deterministic orthogonal start number r (r = 1 is the identity): a product of Givens rotations."""
    Q = np.eye(q)
    if r <= 1:
        return Q
    for a in range(1, q):
        for b in range(a + 1, q + 1):
            th = np.pi * (((r - 1) * (a * 0.6180339887498949 + b * 0.4142135623730951)) % 1)
            G = np.eye(q)
            G[a - 1, a - 1] = G[b - 1, b - 1] = np.cos(th)
            G[a - 1, b - 1], G[b - 1, a - 1] = np.sin(th), -np.sin(th)
            Q = Q @ G
    return Q


def _varimax_run(F, w, Q0, tol, max_sweeps):
    """Pairwise-plane ascent: in the plane of columns (a, b), with z = L_a + i L_b and P = sum w z^4 - (sum w z^2)^2,
    the best angle is arg(P) / 4."""
    q = F.shape[1]
    L, Q = F @ Q0, Q0.copy()
    converged, sweeps = q < 2, 0
    for sweep in range(1, max_sweeps + 1) if q >= 2 else []:
        largest = 0.0
        for a in range(q - 1):
            for b in range(a + 1, q):
                z2 = (L[:, a] + 1j * L[:, b]) ** 2
                P = np.sum(w * z2 * z2) - np.sum(w * z2) ** 2
                if not np.isfinite(abs(P)) or abs(P) <= 1e-14 * np.sum(w * abs(z2) ** 2):
                    continue
                th = np.angle(P) / 4
                c, s = np.cos(th), np.sin(th)
                L[:, [a, b]] = np.column_stack([c * L[:, a] + s * L[:, b], -s * L[:, a] + c * L[:, b]])
                Q[:, [a, b]] = np.column_stack([c * Q[:, a] + s * Q[:, b], -s * Q[:, a] + c * Q[:, b]])
                largest = max(largest, abs(th))
        sweeps = sweep
        if largest < tol:
            converged = True
            break
    return {"rotated": L, "rotation": Q, "criterion": varimax_criterion(L, w), "converged": converged, "sweeps": sweeps}


def match_columns(F, G, w):
    """Signed permutation of the columns of F that best matches G (Hungarian algorithm on |F' W G|).

    Returns permutation (column of F for every column of G), signs, aligned (= F[:, permutation] * signs), cosine of
    every matched pair and the total |inner product|.
    """
    F, G, w = np.asarray(F, float), np.asarray(G, float), np.asarray(w, float)
    S = F.T @ (w[:, None] * G)
    A = np.abs(S)
    rows, cols = linear_sum_assignment(-A)
    perm = np.empty(G.shape[1], dtype=int)
    perm[cols] = rows
    picked = A[perm, np.arange(G.shape[1])]
    signs = np.where(S[perm, np.arange(G.shape[1])] < 0, -1.0, 1.0)
    denom = np.sqrt(w @ F**2)[perm] * np.sqrt(w @ G**2)
    return {"permutation": perm, "signs": signs, "aligned": F[:, perm] * signs,
            "cosine": np.where(denom > 0, picked / np.where(denom > 0, denom, 1), 0.0), "total": float(picked.sum())}


def _distance(F, G, w):
    al = match_columns(F, G, w)["aligned"]
    return float(np.sqrt(np.sum(w[:, None] * (al - G) ** 2) / max(np.sum(w[:, None] * G**2), np.finfo(float).tiny)))


def functional_varimax(F, w, n_starts=10, tol=1e-10, max_sweeps=500, criterion_tol=1e-6, distinct_tol=1e-3):
    """Weighted functional varimax of loading functions F (n, q) with weights w (n,) summing to one.

    Maximises `varimax_criterion` over orthogonal rotations from ``n_starts`` deterministic starts (the identity
    first) and keeps the earliest start within ``criterion_tol`` (relative) of the best criterion. ``ambiguous`` is
    True when another start reaches a criterion that close with a solution that differs by more than
    ``distinct_tol`` (relative weighted RMS after signed matching).
    """
    F, w = np.asarray(F, float), np.asarray(w, float)
    q = F.shape[1]
    runs = [_varimax_run(F, w, _start(q, r), tol, max_sweeps) for r in range(1, n_starts + 1)]
    crit = np.array([r["criterion"] for r in runs])
    top = crit.max()
    best = int(np.flatnonzero(crit >= top - criterion_tol * max(abs(top), np.finfo(float).eps))[0])
    ref = runs[best]
    dist = np.array([0.0 if q < 2 else _distance(r["rotated"], ref["rotated"], w) for r in runs])
    near = (top - crit) <= criterion_tol * max(abs(ref["criterion"]), np.finfo(float).eps)
    reps = []
    for r in runs:
        if not reps or not any(q < 2 or _distance(r["rotated"], x, w) <= distinct_tol for x in reps):
            reps.append(r["rotated"])
    return {"rotated": ref["rotated"], "rotation": ref["rotation"], "criterion": ref["criterion"],
            "criterion_unrotated": varimax_criterion(F, w), "converged": ref["converged"],
            "ambiguous": bool(np.any(near & (dist > distinct_tol))), "n_solutions": len(reps), "best_start": best + 1}
