"""The distance between a sample and a prototype (SPEC section 3).

Every metric of the add-on is a **linear map of the raw coordinates followed by the plain Euclidean distance**:

* ``l2``           the block-weighted L2 distance between the functions (``g = sqrt(w_v)``): the functional SOM of Rossi, Conan-Guez & El Golli (2004);
* ``standardize``  every coordinate divided by its standard deviation, every variable then counting alike (``g_j = sqrt(u_v / K_v) / s_j``);
* ``prior``        a Gaussian-process basis only: ``g_j = sqrt(u_v / K_v) / sqrt(sigma_f^2 lambda_k)``, the RKHS norm of the difference (rough components count most);
* ``mahalanobis``  the L2 gain followed by the **regularised whitening** ``L = U diag(lam^-alpha/2) U' + (I - UU') rho^-alpha/2`` of the covariance of the training coordinates, ``U`` the
  leading ``r`` directions, ``rho`` the total variance beyond them.  ``alpha = 0`` is the L2 metric, ``alpha = 1`` gives every one of the ``r`` directions unit variance and the whole
  remainder one unit of variance in total, so the noise beyond the kept directions can never dominate.

A map is trained on the transformed coordinates; prototypes are transformed back (``L^-1``, ``1/g``) before they are decoded to functions.  This is exact: the batch update of a map
is a weighted mean, which is the minimiser of the neighbourhood-weighted energy under ANY quadratic metric, so only the search for the best unit depends on the metric.

Missing coordinates (NaN) are allowed for the diagonal metrics (partial distances, ``a`` = the weight of each raw coordinate in the squared distance); ``mahalanobis`` and ``prior``
need complete coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from fluoviz_addon_common.serial import allow as _serial_allow

from .spec import METRIC_WORDS, SomError


@_serial_allow
@dataclass
class Metric:
    kind: str
    g: np.ndarray                      # (P,) gain of every raw coordinate
    a: np.ndarray                      # (P,) weight of every raw coordinate in the squared distance (g^2 q): used to scale up a partial distance
    mean: np.ndarray                   # (P,) subtracted (after the gain) before whitening; zeros for the diagonal metrics
    alpha: float = 0.0
    components: int = 0
    U: np.ndarray | None = None        # (P, r)
    lam: np.ndarray | None = None      # (r,)
    rho: float = 1.0
    words: str = ""

    @property
    def whitens(self) -> bool:
        return self.U is not None

    # ------------------------------------------------------------------ the maps
    def transform(self, R: np.ndarray) -> np.ndarray:
        """Raw coordinates ``(n, P)`` (NaN kept for the diagonal metrics) -> the coordinates in which the distance is Euclidean."""
        X = np.asarray(R, float) * self.g
        if not self.whitens:
            return X
        if not np.isfinite(X).all():
            raise SomError("The standardised distance needs every coordinate of every sample: use the plain distance when pixels are missing.")
        y = X - self.mean
        c = y @ self.U
        out = (c * self.lam ** (-self.alpha / 2.0)) @ self.U.T
        if self.U.shape[1] < y.shape[1]:
            out = out + (y - c @ self.U.T) * self.rho ** (-self.alpha / 2.0)
        return out

    def inverse(self, X: np.ndarray) -> np.ndarray:
        """The inverse of :meth:`transform` for prototypes ``(M, P)`` (complete): back to raw coordinates."""
        y = np.asarray(X, float)
        if self.whitens:
            c = y @ self.U
            out = (c * self.lam ** (self.alpha / 2.0)) @ self.U.T
            if self.U.shape[1] < y.shape[1]:
                out = out + (y - c @ self.U.T) * self.rho ** (self.alpha / 2.0)
            y = out + self.mean
        return y / self.g

    def share_weights(self) -> np.ndarray:
        return self.a

    def to_dict(self) -> dict:
        return {"kind": self.kind, "alpha": float(self.alpha), "components": int(self.components), "words": self.words}


# ----------------------------------------------------------------------------- PCA of the training coordinates
def pca_basis(Rc: np.ndarray, kmax: int) -> tuple[np.ndarray, np.ndarray]:
    """Leading right singular vectors ``(P, k)`` and singular values ``(k,)`` of the (weighted, centred) matrix ``Rc`` ``(n, P)``; uses the Gram matrix when ``P > n``."""
    n, P = Rc.shape
    kmax = int(min(kmax, n, P))
    if P <= n:
        _, s, Vt = np.linalg.svd(Rc, full_matrices=False)
        return Vt[:kmax].T, s[:kmax]
    G = Rc @ Rc.T
    w, Uu = np.linalg.eigh(G)
    order = np.argsort(w)[::-1][:kmax]
    w, Uu = np.clip(w[order], 0.0, None), Uu[:, order]
    s = np.sqrt(w)
    ok = s > 1e-12 * max(float(s.max()), 1e-300)
    V = np.zeros((P, kmax))
    V[:, ok] = (Rc.T @ Uu[:, ok]) / s[ok]
    return V, s


def weighted_centre(X: np.ndarray, w: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, float]:
    """``(Xc, mean, dof)``: ``Xc = sqrt(w)(X - mean)`` with the weighted mean and ``dof`` = (sum of weights) - 1 (weights average 1 or ``w`` is None)."""
    X = np.asarray(X, float)
    if w is None:
        w = np.ones(X.shape[0])
    w = np.asarray(w, float)
    mu = (w[:, None] * X).sum(axis=0) / w.sum()
    return np.sqrt(w)[:, None] * (X - mu), mu, float(w.sum() - 1.0)


def parsimonious_components(X: np.ndarray, w: np.ndarray | None = None, folds: int = 5, seed: int = 0, cap: int = 20) -> dict:
    """The number of directions to standardise: the cross-validated reconstruction error (V folds over the samples; the mean and directions come from the training folds) of ``k`` = 0 …
    ``kmax`` components; the smallest ``k`` whose error is within the larger of one standard error and 0.5 % of the error of ``k = 0`` of the best one (the rule of the Functional PCA's
    cross-validation).  At least 2, at most ``min(n - 2, cap)``."""
    X = np.asarray(X, float)
    n = X.shape[0]
    hi = int(max(2, min(n - 2, cap)))
    w = np.ones(n) if w is None else np.asarray(w, float)
    V = int(min(folds, n))
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    e = np.zeros((n, hi + 1))
    for f in range(V):
        test = order[f::V]
        train = np.setdiff1d(np.arange(n), test)
        Tc, mu, _ = weighted_centre(X[train], w[train])
        k = int(min(hi, Tc.shape[0] - 1))
        U, _ = pca_basis(Tc, k)
        E = X[test] - mu
        S = E @ U
        for j in range(hi + 1):
            kk = min(j, U.shape[1])
            res = E - (S[:, :kk] @ U[:, :kk].T if kk else 0.0)
            e[test, j] = (res ** 2).sum(axis=1)
    press = e.mean(axis=0)
    best = int(np.argmin(press))
    se = float(np.std(e[:, best] - 0.0, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    tol = max(se, 0.005 * float(press[0]))
    ok = np.flatnonzero(press <= press[best] + tol)
    r = int(max(2, ok.min() if ok.size else best))
    r = int(min(r, hi))
    return {"components": r, "press": press, "best": best, "k_max": hi}


# ----------------------------------------------------------------------------- building a metric
def _gains(kind: str, R: np.ndarray, l2_gain: np.ndarray, unit_gain: np.ndarray, prior_scale: np.ndarray | None) -> np.ndarray:
    if kind in ("l2", "mahalanobis"):
        return np.asarray(l2_gain, float).copy()
    if kind == "prior":
        if prior_scale is None:
            raise SomError("The Gaussian-process prior distance needs the Gaussian-process basis for every variable.")
        return np.asarray(unit_gain, float) / np.maximum(np.asarray(prior_scale, float), 1e-300)
    with np.errstate(invalid="ignore"):
        s = np.nanstd(R, axis=0, ddof=1)
    s = np.where(np.isfinite(s), s, 0.0)
    floor = 1e-8 * max(float(s.max()), 1e-300)
    return np.asarray(unit_gain, float) / np.maximum(s, floor)


def fit_metric(R: np.ndarray, kind: str, l2_gain: np.ndarray, unit_gain: np.ndarray, q: np.ndarray, prior_scale: np.ndarray | None = None, alpha: float = 1.0,
               components="auto", sample_w: np.ndarray | None = None, seed: int = 0, cv_folds: int = 5) -> Metric:
    """The metric of kind ``kind`` fitted on the training rows ``R`` ``(n, P)`` (raw coordinates).  ``l2_gain`` / ``unit_gain`` / ``prior_scale``: from ``Prepared``; ``q``: the quadrature
    weights of the coordinates (the weights of the coordinates in the distance are ``g^2 q``)."""
    if kind not in METRIC_WORDS:
        raise SomError(f"Unknown distance '{kind}'.")
    R = np.asarray(R, float)
    P = R.shape[1]
    g = _gains(kind, R, l2_gain, unit_gain, prior_scale)
    a = g ** 2 * np.asarray(q, float)
    if kind == "prior" and not np.isfinite(R).all():
        raise SomError("The Gaussian-process prior distance needs every coordinate of every sample.")
    if kind != "mahalanobis":
        words = {"l2": "The plain L2 distance between the functions (the units of the data); the variables are weighted so that each counts alike.",
                 "standardize": "Every coordinate is divided by its standard deviation across the samples, so every pixel or variable counts alike (and every variable counts as much as any other).",
                 "prior": "The distance in the standardised coefficients of the Gaussian-process basis: rough components, which the prior expects to be small, count most."}[kind]
        return Metric(kind, g, a, np.zeros(P), words=words)
    if not np.isfinite(R).all():
        raise SomError("The standardised (Mahalanobis) distance needs every coordinate of every sample: with missing pixels use the plain distance or standardise every pixel.")
    n = R.shape[0]
    Rg = R * g
    Rc, mu, dof = weighted_centre(Rg, sample_w)
    total = float((Rc ** 2).sum() / max(dof, 1.0))
    if components == "auto":
        cv = parsimonious_components(Rg, sample_w, folds=cv_folds, seed=seed)
        r = cv["components"]
    else:
        r = int(components)
    r = int(max(1, min(r, n - 2, P)))
    U, s = pca_basis(Rc, r)
    r = U.shape[1]
    lam = np.maximum(s ** 2 / max(dof, 1.0), 1e-300)
    rho = max(total - float(lam.sum()), 1e-12 * total)
    alpha = float(alpha)
    words = (f"The distance is standardised (regularised Mahalanobis): the {r} leading directions of variation each count as one unit and everything beyond them counts as one unit in total"
             + ("." if alpha >= 1 else f" (partial standardisation: exponent {alpha:g})."))
    return Metric("mahalanobis", g, a, mu, alpha, r, U, lam, rho, words)
