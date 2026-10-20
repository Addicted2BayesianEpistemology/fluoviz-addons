"""How good is a map?  The standard measures (SPEC section 5).  Pure numpy / scipy.

* quantisation error: the distance of the samples to their best unit;
* topographic error (Kiviluoto 1996): the share of samples whose first and second best units are not neighbours on the lattice;
* topographic product (Bauer & Pawelzik 1992): compares the neighbours of every unit on the lattice with those in the data space; near 0 = the map fits, negative = the map is too small
  in dimension (it folds), positive = too large;
* trustworthiness and continuity (Venna & Kaski 2001): do the k nearest neighbours on the map (or in a scatter) agree with those in the data?
* the Sammon projection of the prototypes, with the number of lattice edges that cross (a folded map).
"""
from __future__ import annotations

import numpy as np
from scipy import optimize

from .metric import pca_basis, weighted_centre


def pairwise(X: np.ndarray) -> np.ndarray:
    """Euclidean distances between the rows of ``X`` (complete)."""
    sq = (X ** 2).sum(axis=1)
    return np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * X @ X.T, 0.0))


def topographic_error(c1: np.ndarray, c2: np.ndarray, adj: np.ndarray, w: np.ndarray | None = None) -> float:
    if adj.shape[0] < 2:
        return 0.0
    bad = ~adj[c1, c2]
    if w is None:
        return float(bad.mean())
    return float((np.asarray(w) * bad).sum() / np.sum(w))


def topographic_product(protos: np.ndarray, lat_dist: np.ndarray) -> float:
    """Bauer & Pawelzik's ``P`` for the prototypes ``(M, P)`` and the lattice distances ``(M, M)``."""
    M = protos.shape[0]
    if M < 3:
        return 0.0
    dV = pairwise(protos)
    dA = np.asarray(lat_dist, float)
    eps = 1e-12
    big = np.inf
    dVm = dV + np.diag(np.full(M, big))
    dAm = dA + np.diag(np.full(M, big))
    nV = np.argsort(dVm, axis=1, kind="stable")[:, :M - 1]
    nA = np.argsort(dAm, axis=1, kind="stable")[:, :M - 1]
    rows = np.arange(M)[:, None]
    q1 = (dV[rows, nA] + eps) / (dV[rows, nV] + eps)
    q2 = (dA[rows, nA] + eps) / (dA[rows, nV] + eps)
    lg = np.log(q1) + np.log(q2)
    cs = np.cumsum(lg, axis=1)
    k = np.arange(1, M)[None, :]
    p3 = cs / (2.0 * k)
    return float(p3.sum() / (M * (M - 1)))


def _ranks(D: np.ndarray) -> np.ndarray:
    """``rank[i, j]`` = position of j among the neighbours of i (1 = nearest), the sample itself excluded (rank 0)."""
    n = D.shape[0]
    Dm = D + np.diag(np.full(n, np.inf))
    order = np.argsort(Dm, axis=1, kind="stable")
    rank = np.zeros((n, n), int)
    rank[np.arange(n)[:, None], order[:, :n - 1]] = np.arange(1, n)[None, :]
    return rank


def trust_continuity(D_high: np.ndarray, D_low: np.ndarray, k: int = 5) -> dict:
    """Trustworthiness and continuity at ``k`` neighbours: ``D_high`` the distances in the data, ``D_low`` those in the picture (map positions or a scatter).  1 = perfect."""
    n = D_high.shape[0]
    k = int(max(1, min(k, (n - 2) // 2)))
    if n < 5:
        return {"k": k, "trustworthiness": float("nan"), "continuity": float("nan")}
    rh, rl = _ranks(D_high), _ranks(D_low)
    norm = 2.0 / (n * k * (2.0 * n - 3.0 * k - 1.0))
    intruder = (rl <= k) & (rh > k) & (rl > 0)
    T = 1.0 - norm * float(((rh - k) * intruder).sum())
    lost = (rh <= k) & (rl > k) & (rh > 0)
    C = 1.0 - norm * float(((rl - k) * lost).sum())
    return {"k": k, "trustworthiness": float(T), "continuity": float(C)}


def pca_scatter(X: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """The first two principal components (scores ``(n, 2)``) of the coordinates ``X`` (NaN treated as the mean)."""
    X = np.asarray(X, float)
    mu = np.nanmean(X, axis=0)
    Xf = np.where(np.isfinite(X), X, mu)
    Xc, mu2, _ = weighted_centre(Xf, w)
    V, _ = pca_basis(Xc, 2)
    S = (Xf - mu2) @ V
    if S.shape[1] < 2:
        S = np.column_stack([S, np.zeros(S.shape[0])])
    return S


def sammon(D: np.ndarray, init: np.ndarray, iters: int = 300) -> tuple:
    """Sammon's mapping of the distances ``D`` ``(M, M)`` to the plane, started from ``init`` ``(M, 2)``: ``(coordinates, stress)``."""
    M = D.shape[0]
    iu = np.triu_indices(M, 1)
    dstar = np.maximum(D[iu], 1e-12)
    c = float(dstar.sum())

    def f(v):
        Y = v.reshape(M, 2)
        d = np.maximum(np.sqrt(((Y[iu[0]] - Y[iu[1]]) ** 2).sum(axis=1)), 1e-12)
        diff = d - dstar
        E = float((diff ** 2 / dstar).sum() / c)
        coef = (2.0 / c) * diff / (dstar * d)
        G = np.zeros((M, 2))
        v12 = (Y[iu[0]] - Y[iu[1]]) * coef[:, None]
        np.add.at(G, iu[0], v12)
        np.add.at(G, iu[1], -v12)
        return E, G.ravel()

    sc = float(np.median(dstar)) / max(float(np.median(np.sqrt(((init[iu[0]] - init[iu[1]]) ** 2).sum(axis=1)))), 1e-12)
    y0 = (np.asarray(init, float) * sc).ravel()
    res = optimize.minimize(f, y0, jac=True, method="L-BFGS-B", options={"maxiter": iters})
    return res.x.reshape(M, 2), float(res.fun)


def edge_crossings(xy: np.ndarray, edges: np.ndarray) -> int:
    """How many pairs of lattice edges cross when the units are drawn at ``xy`` (edges that share a unit do not count): 0 = not folded."""
    if edges.shape[0] < 2:
        return 0
    j = edges[:, 0].astype(int)
    k = edges[:, 1].astype(int)
    A, B = xy[j], xy[k]
    E = edges.shape[0]

    def ccw(P, Q, R):
        return (R[..., 1] - P[..., 1]) * (Q[..., 0] - P[..., 0]) - (Q[..., 1] - P[..., 1]) * (R[..., 0] - P[..., 0])

    a, b = A[:, None, :], B[:, None, :]
    c, d = A[None, :, :], B[None, :, :]
    inter = (ccw(a, c, d) * ccw(b, c, d) < 0) & (ccw(a, b, c) * ccw(a, b, d) < 0)
    share = (j[:, None] == j[None, :]) | (j[:, None] == k[None, :]) | (k[:, None] == j[None, :]) | (k[:, None] == k[None, :])
    inter &= ~share
    return int(np.triu(inter, 1).sum())


def quantisation_summary(qe: np.ndarray, qe_l2: np.ndarray, qe_rel: np.ndarray, w: np.ndarray | None = None) -> dict:
    ww = np.ones(len(qe)) if w is None else np.asarray(w, float)
    s = float(ww.sum())
    return {"qe_mean": float((ww * qe).sum() / s), "qe_median": float(np.median(qe)), "qe_rel_mean": float((ww * qe_rel).sum() / s), "captured": float(1.0 - (ww * qe_rel).sum() / s),
            "qe_l2_mean": float((ww * qe_l2).sum() / s)}
