"""The self-organising map itself: lattice, initialisation, batch and sequential training, best-matching units, continuous positions (SPEC section 4).  Pure numpy.

The map works on **transformed coordinates** ``X`` ``(n, P)`` in which the distance is Euclidean (``metric.Metric.transform``); a NaN coordinate is "not observed" and is left out of the
distance, which is scaled up by the share of the metric's weight that was observed (the SOM Toolbox rule), and out of the update, which averages the samples that have the coordinate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .metric import pca_basis, weighted_centre
from .spec import SomError

SQ3 = float(np.sqrt(3.0) / 2.0)


# ----------------------------------------------------------------------------- the lattice
@dataclass
class Lattice:
    rows: int
    cols: int
    kind: str                          # hexagonal | rectangular
    shape: str                         # sheet | toroid
    coords: np.ndarray                 # (M, 2)
    d2: np.ndarray                     # (M, M) squared lattice distance (wrapped on a toroid)
    adj: np.ndarray                    # (M, M) bool: lattice neighbours (distance 1)
    period: tuple | None = None        # (Lx, Ly) of a toroid

    @property
    def M(self) -> int:
        return int(self.rows * self.cols)

    def row_col(self, unit) -> tuple:
        u = np.asarray(unit)
        return u // self.cols, u % self.cols

    def displacement(self, bmu: np.ndarray) -> np.ndarray:
        """``(n, M, 2)``: the vector from the best unit of every sample to every unit (the shortest one around a toroid)."""
        D = self.coords[None, :, :] - self.coords[np.asarray(bmu)][:, None, :]
        if self.shape != "toroid":
            return D
        Lx, Ly = self.period
        best = D.copy()
        bd = (D ** 2).sum(axis=2)
        for sx in (-1, 0, 1):
            for sy in (-1, 0, 1):
                if sx == 0 and sy == 0:
                    continue
                cand = D + np.array([sx * Lx, sy * Ly])
                cd = (cand ** 2).sum(axis=2)
                better = cd < bd
                best[better] = cand[better]
                bd = np.where(better, cd, bd)
        return best


def make_lattice(rows: int, cols: int, kind: str = "hexagonal", shape: str = "sheet") -> Lattice:
    rows, cols = int(rows), int(cols)
    if rows < 1 or cols < 1:
        raise SomError("A map needs at least one row and one column.")
    if kind == "hexagonal" and shape == "toroid" and rows % 2:
        rows += 1                                                   # the offset rows only close up around a toroid when there is an even number of them
    r, c = np.divmod(np.arange(rows * cols), cols)
    if kind == "hexagonal":
        xy = np.column_stack([c + 0.5 * (r % 2), r * SQ3])
        period = (float(cols), float(rows * SQ3))
    else:
        xy = np.column_stack([c.astype(float), r.astype(float)])
        period = (float(cols), float(rows))
    D = xy[:, None, :] - xy[None, :, :]
    d2 = (D ** 2).sum(axis=2)
    if shape == "toroid":
        for sx in (-1, 0, 1):
            for sy in (-1, 0, 1):
                if sx or sy:
                    d2 = np.minimum(d2, ((D + np.array([sx * period[0], sy * period[1]])) ** 2).sum(axis=2))
    adj = (d2 > 1e-12) & (d2 <= 1.0 + 1e-9)
    return Lattice(rows, cols, kind, shape, xy, d2, adj, period if shape == "toroid" else None)


def neighbourhood(d2: np.ndarray, radius: float, kind: str = "gaussian") -> np.ndarray:
    """The neighbourhood function of the squared lattice distances ``d2`` at ``radius`` (sigma)."""
    r2 = float(radius) ** 2
    if kind == "gaussian":
        return np.exp(-d2 / (2.0 * r2))
    if kind == "cutgauss":
        return np.exp(-d2 / (2.0 * r2)) * (d2 <= r2)
    if kind == "bubble":
        return (d2 <= r2).astype(float)
    if kind == "epanechnikov":
        return np.maximum(0.0, 1.0 - d2 / r2)
    raise SomError(f"Unknown neighbourhood function '{kind}'.")


# ----------------------------------------------------------------------------- best units
def best_units(X: np.ndarray, protos: np.ndarray, share_w: np.ndarray | None = None) -> tuple:
    """``(c1, c2, D2)``: the best and second-best unit of every sample and the squared distances ``D2`` ``(n, M)`` (partial distances scaled up by the observed share of ``share_w``)."""
    X = np.asarray(X, float)
    mask = np.isfinite(X)
    complete = bool(mask.all())
    Xz = np.where(mask, X, 0.0)
    sq = (Xz ** 2).sum(axis=1)
    D2 = sq[:, None] - 2.0 * (Xz @ protos.T)
    if complete:
        D2 = D2 + (protos ** 2).sum(axis=1)[None, :]
    else:
        D2 = D2 + mask.astype(float) @ (protos ** 2).T
        if share_w is not None:
            tot = float(np.sum(share_w))
            obs = mask.astype(float) @ np.asarray(share_w, float)
            D2 = D2 * (tot / np.maximum(obs, 1e-300))[:, None]
    D2 = np.maximum(D2, 0.0)
    M = D2.shape[1]
    if M == 1:
        c = np.zeros(D2.shape[0], int)
        return c, c.copy(), D2
    part = np.argpartition(D2, 1, axis=1)[:, :2]
    v = np.take_along_axis(D2, part, axis=1)
    swap = v[:, 1] < v[:, 0]
    c1 = np.where(swap, part[:, 1], part[:, 0])
    c2 = np.where(swap, part[:, 0], part[:, 1])
    return c1, c2, D2


# ----------------------------------------------------------------------------- initialisation
def init_linear(X: np.ndarray, lat: Lattice, w: np.ndarray | None = None) -> np.ndarray:
    """Prototypes spread over the plane of the two leading principal directions of ``X`` (Kohonen 2001, section 3.7), between +/- 2 standard deviations; the longer side of the map
    follows the first direction.  Deterministic."""
    X = np.asarray(X, float)
    mask = np.isfinite(X)
    mu = np.nanmean(X, axis=0) if not mask.all() else (X.mean(axis=0) if w is None else (np.asarray(w)[:, None] * X).sum(axis=0) / np.sum(w))
    Xf = np.where(mask, X, mu)
    Xc, mu2, dof = weighted_centre(Xf, w)
    V, s = pca_basis(Xc, 2)
    sd = s / np.sqrt(max(dof, 1.0))
    k = V.shape[1]
    xy = lat.coords
    span = np.maximum(xy.max(axis=0) - xy.min(axis=0), 1e-12)
    u = 2.0 * (xy - xy.min(axis=0)) / span - 1.0                       # [-1, 1] on both axes
    first = 0 if span[0] >= span[1] else 1
    ua, ub = u[:, first], u[:, 1 - first]
    protos = np.tile(mu2, (lat.M, 1))
    protos = protos + 2.0 * np.outer(ua, V[:, 0] * sd[0])
    if k > 1:
        protos = protos + 2.0 * np.outer(ub, V[:, 1] * sd[1])
    return protos


def init_random(X: np.ndarray, M: int, seed: int = 0) -> np.ndarray:
    X = np.asarray(X, float)
    mu = np.nanmean(X, axis=0)
    Xf = np.where(np.isfinite(X), X, mu)
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=M, replace=X.shape[0] < M)
    return Xf[idx].copy()


# ----------------------------------------------------------------------------- training
@dataclass
class TrainSpec:
    mode: str = "batch"
    init: str = "linear"
    neighbourhood: str = "gaussian"
    epochs_rough: int = 50
    epochs_fine: int = 150
    radius_initial: float | str = "auto"
    radius_final: float = 1.0
    alpha_initial: float = 0.05
    alpha_final: float = 0.01
    seed: int = 0


@dataclass
class TrainedMap:
    lattice: Lattice
    protos: np.ndarray                 # (M, P) transformed coordinates
    history: dict = field(default_factory=dict)
    radii: np.ndarray | None = None
    energy: float = float("nan")


def radius_schedule(lat: Lattice, spec: TrainSpec) -> np.ndarray:
    """One radius per epoch: linear from the initial radius down to a quarter of it (the rough phase), then down to the final radius (the fine phase)."""
    side = max(lat.rows, lat.cols)
    r0 = float(max(1.0, side / 4.0)) if spec.radius_initial == "auto" else float(spec.radius_initial)
    rf = float(spec.radius_final)
    r0 = max(r0, rf)
    mid = max(r0 / 4.0, rf)
    rough = np.linspace(r0, mid, spec.epochs_rough)
    fine = np.linspace(mid, rf, spec.epochs_fine)
    return np.concatenate([rough, fine])


def train(X: np.ndarray, lat: Lattice, spec: TrainSpec, sample_w: np.ndarray | None = None, share_w: np.ndarray | None = None, progress=None, protos0: np.ndarray | None = None) -> TrainedMap:
    """Fit a map to the transformed coordinates ``X`` ``(n, P)`` (NaN = not observed).  ``sample_w`` ``(n,)``: the weight of every sample (mean 1) or None; ``share_w`` ``(P,)``: the weight
    of every coordinate in the distance (for scaling up partial distances).  ``progress(frac, message)`` may raise ``Cancelled``."""
    X = np.asarray(X, float)
    n, P = X.shape
    w = np.ones(n) if sample_w is None else np.asarray(sample_w, float)
    if protos0 is not None:
        protos = np.array(protos0, float)
    elif spec.init == "random":
        protos = init_random(X, lat.M, spec.seed)
    else:
        protos = init_linear(X, lat, w)
    radii = radius_schedule(lat, spec)
    if spec.mode == "sequential":
        return _train_sequential(X, lat, spec, w, share_w, protos, radii, progress)
    mask = np.isfinite(X)
    Xz = np.where(mask, X, 0.0)
    wm = mask * w[:, None]
    wx = Xz * w[:, None]
    hist = {"energy": [], "qe": [], "radius": [], "phase": []}
    ne = len(radii)
    for e, rad in enumerate(radii):
        c, _, D2 = best_units(X, protos, share_w)
        H = neighbourhood(lat.d2[c], rad, spec.neighbourhood)
        hist["energy"].append(float((w * (H * D2).sum(axis=1)).sum() / w.sum()))
        hist["qe"].append(float((w * np.sqrt(D2[np.arange(n), c])).sum() / w.sum()))
        hist["radius"].append(float(rad))
        hist["phase"].append("rough" if e < spec.epochs_rough else "fine")
        num = H.T @ wx
        den = H.T @ wm
        ok = den > 1e-12
        protos = np.where(ok, num / np.where(ok, den, 1.0), protos)
        if progress is not None and (e % 5 == 0 or e == ne - 1):
            progress((e + 1) / ne, f"Training the map, epoch {e + 1} of {ne}")
    c, _, D2 = best_units(X, protos, share_w)
    H = neighbourhood(lat.d2[c], radii[-1], spec.neighbourhood)
    energy = float((w * (H * D2).sum(axis=1)).sum() / w.sum())
    return TrainedMap(lat, protos, hist, radii, energy)


def _train_sequential(X, lat, spec, w, share_w, protos, radii, progress) -> TrainedMap:
    n, P = X.shape
    rng = np.random.default_rng(spec.seed)
    mask = np.isfinite(X)
    Xz = np.where(mask, X, 0.0)
    ne = len(radii)
    alphas = np.linspace(spec.alpha_initial, spec.alpha_final, ne)
    hist = {"energy": [], "qe": [], "radius": [], "phase": []}
    for e in range(ne):
        for i in rng.permutation(n):
            c, _, D2 = best_units(X[i:i + 1], protos, share_w)
            h = neighbourhood(lat.d2[c[0]][None, :], radii[e], spec.neighbourhood)[0]
            step = (alphas[e] * min(1.0, w[i])) * h
            upd = step[:, None] * mask[i][None, :]
            protos = protos + upd * (Xz[i][None, :] - protos)
        c, _, D2 = best_units(X, protos, share_w)
        H = neighbourhood(lat.d2[c], radii[e], spec.neighbourhood)
        hist["energy"].append(float((w * (H * D2).sum(axis=1)).sum() / w.sum()))
        hist["qe"].append(float((w * np.sqrt(D2[np.arange(n), c])).sum() / w.sum()))
        hist["radius"].append(float(radii[e]))
        hist["phase"].append("rough" if e < spec.epochs_rough else "fine")
        if progress is not None and (e % 5 == 0 or e == ne - 1):
            progress((e + 1) / ne, f"Training the map, epoch {e + 1} of {ne}")
    return TrainedMap(lat, protos, hist, radii, hist["energy"][-1])


# ----------------------------------------------------------------------------- positions, hits, U-matrix
def positions(D2: np.ndarray, lat: Lattice, c1: np.ndarray, tau: float) -> np.ndarray:
    """The continuous position ``(n, 2)`` of every sample, in lattice coordinates: the mean of the coordinates of all units weighted by ``exp(-D2 / 2 tau)`` (on a toroid around the best
    unit, the shortest way round).  ``tau`` = the mean squared distance of the training samples to their best unit."""
    tau = max(float(tau), 1e-300)
    z = -D2 / (2.0 * tau)
    z = z - z.max(axis=1, keepdims=True)
    W = np.exp(z)
    W = W / W.sum(axis=1, keepdims=True)
    if lat.shape == "toroid":
        disp = lat.displacement(c1)
        pos = lat.coords[c1] + np.einsum("nm,nmk->nk", W, disp)
    else:
        pos = W @ lat.coords
    return pos


def hits(c1: np.ndarray, M: int, w: np.ndarray | None = None) -> tuple:
    cnt = np.bincount(c1, minlength=M)
    wt = np.bincount(c1, weights=None if w is None else w, minlength=M)
    return cnt, wt


def umatrix(protos: np.ndarray, lat: Lattice) -> tuple:
    """``(u, edges)``: for every unit the mean distance (Euclidean, in the coordinates of ``protos``) to its lattice neighbours, and the list of edges ``(j, k, distance)`` with ``j < k``."""
    M = protos.shape[0]
    sq = (protos ** 2).sum(axis=1)
    Dm = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * protos @ protos.T, 0.0))
    adj = lat.adj
    cnt = adj.sum(axis=1)
    u = np.where(cnt > 0, (Dm * adj).sum(axis=1) / np.maximum(cnt, 1), np.nan)
    jj, kk = np.nonzero(np.triu(adj, 1))
    edges = np.column_stack([jj, kk, Dm[jj, kk]]) if jj.size else np.zeros((0, 3))
    return u, edges, Dm
