"""Fit a map to prepared samples, with its honest checks (SPEC sections 4 and 5).  Pure numpy / scipy; no server.

``fit_map(prep, cfg)`` → :class:`FitResult`: the metric (fitted on the samples), the trained map, the assignment of every sample (best unit, second unit, quantisation error, position),
the units' hits and U-matrix, the k-fold held-out errors (the metric is refitted inside every fold) and the quality numbers.  ``assign`` is the one function a new sample goes through
(the saved model uses it too).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import advice, quality
from . import som as S
from .metric import Metric, fit_metric
from .spec import Config, SomError


@dataclass
class Layout:
    """How the raw coordinates are grouped into variables (everything ``assign`` needs besides the metric and the map)."""
    names: list
    slices: list
    variances: np.ndarray              # (B,) V_v
    q: np.ndarray                      # (P,) quadrature weights of the coordinates

    @classmethod
    def of(cls, prep) -> "Layout":
        return cls(list(prep.names), list(prep.slices()), np.array([b.variance for b in prep.blocks], float), np.asarray(prep.q, float))


def assign(R: np.ndarray, metric: Metric, protos_t: np.ndarray, protos_raw: np.ndarray, layout: Layout) -> dict:
    """Raw coordinates ``R`` ``(n, P)`` (NaN = not observed; rows that are all NaN are unusable) → the best units, quantisation errors and their pieces."""
    R = np.asarray(R, float)
    n = R.shape[0]
    X = metric.transform(R)
    c1, c2, D2 = S.best_units(X, protos_t, metric.a if not metric.whitens else None)
    qe = np.sqrt(D2[np.arange(n), c1])
    diff = R - protos_raw[c1]
    obs = np.isfinite(diff)
    d2 = np.where(obs, diff, 0.0) ** 2
    qb = np.zeros((n, len(layout.names)))
    rel = np.zeros(n)
    for k, sl in enumerate(layout.slices):
        qq = layout.q[sl]
        num = (d2[:, sl]).sum(axis=1)
        have = (obs[:, sl] * qq).sum(axis=1)
        tot = float(qq.sum()) if qq.size else 1.0
        scale = np.where(have > 0, tot / np.maximum(have, 1e-300), 1.0) if not obs[:, sl].all() else 1.0
        # pixel coordinates carry sqrt(q) already, so the squared sum is the mean over the observed part of the domain; scale to the whole domain share
        sq = num * scale
        qb[:, k] = np.sqrt(sq)
        rel += sq / layout.variances[k]
    rel = rel / len(layout.names)
    return {"X": X, "c1": c1, "c2": c2, "D2": D2, "qe": qe, "qe_blocks": qb, "qe_rel": rel}


@dataclass
class FitResult:
    cfg: Config
    lattice: S.Lattice
    metric: Metric
    protos_t: np.ndarray
    protos_raw: np.ndarray
    layout: Layout
    sample_w: np.ndarray | None
    asg: dict
    tau: float
    pos: np.ndarray
    hits: np.ndarray
    hits_w: np.ndarray
    umat: np.ndarray
    edges: np.ndarray
    proto_dist: np.ndarray
    history: dict
    energy: float
    kfold: dict
    quality: dict
    components_cv: dict | None = None
    notes: list = field(default_factory=list)

    @property
    def n(self) -> int:
        return int(self.asg["c1"].shape[0])


def _spec(cfg: Config, seed: int | None = None) -> S.TrainSpec:
    return S.TrainSpec(cfg.mode, cfg.init, cfg.neighbourhood, cfg.epochs_rough, cfg.epochs_fine, cfg.radius_initial, cfg.radius_final, cfg.alpha_initial, cfg.alpha_final,
                       cfg.seed if seed is None else seed)


def _metric(prep, cfg: Config, rows=None, sample_w=None) -> Metric:
    R = prep.R if rows is None else prep.R[rows]
    sw = sample_w if rows is None or sample_w is None else sample_w[rows]
    return fit_metric(R, cfg.metric_kind, prep.l2_gain(), prep.unit_gain(), prep.q, prep.prior_scales(), cfg.alpha, cfg.components, sw, cfg.seed, cv_folds=5)


def _kfold(prep, cfg: Config, lat: S.Lattice, spec: S.TrainSpec, sample_w, layout: Layout, progress=None) -> dict:
    n = prep.n
    V = int(min(cfg.folds, n))
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(n)
    qe = np.full(n, np.nan)
    rel = np.full(n, np.nan)
    qe_l2 = np.full((n, len(layout.names)), np.nan)
    fold = np.full(n, -1)
    empty_hit = np.full(n, False)
    for f in range(V):
        test = np.sort(order[f::V])
        train = np.setdiff1d(np.arange(n), test)
        m = _metric(prep, cfg, train, sample_w)
        Xtr = m.transform(prep.R[train])
        sw = None if sample_w is None else sample_w[train]
        tm = S.train(Xtr, lat, spec, sw, m.a if not m.whitens else None)
        praw = m.inverse(tm.protos)
        a = assign(prep.R[test], m, tm.protos, praw, layout)
        a_tr = assign(prep.R[train], m, tm.protos, praw, layout)
        occupied = np.zeros(lat.M, bool)
        occupied[a_tr["c1"]] = True
        qe[test], rel[test], qe_l2[test], fold[test] = a["qe"], a["qe_rel"], a["qe_blocks"], f
        empty_hit[test] = ~occupied[a["c1"]]
        if progress is not None:
            progress((f + 1) / V, f"Checking on held-out samples, fold {f + 1} of {V}")
    return {"qe": qe, "qe_rel": rel, "qe_blocks": qe_l2, "fold": fold, "folds": V, "empty_unit": empty_hit}


def fit_map(prep, cfg: Config, progress=None, sample_w=None, with_kfold: bool = True, lattice: S.Lattice | None = None) -> FitResult:
    """The whole fit (see the module text)."""
    from fluoviz.analysis.common import Progress
    progress = progress or Progress()
    cfg.check_samples(prep.n)
    n = prep.n
    if sample_w is None:
        sample_w = prep.sample_weights(cfg.gap_weighting)
    progress(0.01, "Fitting the distance")
    metric = _metric(prep, cfg, None, sample_w)
    X = metric.transform(prep.R)
    if lattice is None:
        rows, cols = advice.resolve_size(cfg.size, X, sample_w, cfg.lattice, cfg.shape)
        lat = S.make_lattice(rows, cols, cfg.lattice, cfg.shape)
    else:
        lat = lattice
    spec = _spec(cfg)
    layout = Layout.of(prep)
    progress(0.05, "Training the map")
    tm = S.train(X, lat, spec, sample_w, metric.a if not metric.whitens else None, progress.sub(0.05, 0.40))
    protos_raw = metric.inverse(tm.protos)
    asg = assign(prep.R, metric, tm.protos, protos_raw, layout)
    w = np.ones(n) if sample_w is None else sample_w
    tau = float((w * asg["D2"][np.arange(n), asg["c1"]]).sum() / w.sum())
    pos = S.positions(asg["D2"], lat, asg["c1"], tau)
    cnt, cw = S.hits(asg["c1"], lat.M, sample_w)
    u, edges, Dm = S.umatrix(tm.protos, lat)
    kf = _kfold(prep, cfg, lat, spec, sample_w, layout, progress.sub(0.40, 0.85)) if with_kfold and n >= 6 else {"qe": np.full(n, np.nan), "qe_rel": np.full(n, np.nan),
                                                                                                    "qe_blocks": np.full((n, len(layout.names)), np.nan), "fold": np.full(n, -1), "folds": 0,
                                                                                                    "empty_unit": np.zeros(n, bool)}
    progress(0.86, "Computing the quality numbers")
    q = quality_numbers(prep, cfg, lat, metric, tm.protos, X, asg, pos, sample_w, kf)
    return FitResult(cfg, lat, metric, tm.protos, protos_raw, layout, sample_w, asg, tau, pos, cnt, cw, u, edges, Dm, tm.history, tm.energy, kf, q)


def lattice_pair_distance(pos: np.ndarray, lat: S.Lattice) -> np.ndarray:
    """Distances between positions on the map (the shortest way round on a toroid)."""
    D = pos[:, None, :] - pos[None, :, :]
    d2 = (D ** 2).sum(axis=2)
    if lat.shape == "toroid":
        Lx, Ly = lat.period
        for sx in (-1, 0, 1):
            for sy in (-1, 0, 1):
                if sx or sy:
                    d2 = np.minimum(d2, ((D + np.array([sx * Lx, sy * Ly])) ** 2).sum(axis=2))
    return np.sqrt(d2)


def quality_numbers(prep, cfg: Config, lat: S.Lattice, metric: Metric, protos_t, X, asg, pos, sample_w, kf) -> dict:
    n = prep.n
    c1, c2 = asg["c1"], asg["c2"]
    qs = quality.quantisation_summary(asg["qe"], asg["qe_blocks"].mean(axis=1), asg["qe_rel"], sample_w)
    te = quality.topographic_error(c1, c2, lat.adj, sample_w)
    tp = quality.topographic_product(protos_t, np.sqrt(lat.d2))
    # the picture against the plain PCA plot
    Xf = np.where(np.isfinite(X), X, 0.0)
    Dh = quality.pairwise(Xf)
    tc_map = quality.trust_continuity(Dh, lattice_pair_distance(pos, lat), 5)
    tc_pca = quality.trust_continuity(Dh, quality.pairwise(quality.pca_scatter(X, sample_w)), 5)
    held = kf["qe"]
    has_held = bool(np.isfinite(held).any())
    out = {**qs, "topographic_error": te, "topographic_product": tp, "trust_map": tc_map, "trust_pca": tc_pca,
           "n_units": int(lat.M), "n_empty": int((np.bincount(c1, minlength=lat.M) == 0).sum()), "samples_per_unit": float(n / lat.M)}
    if has_held:
        ok = np.isfinite(held)
        out.update({"heldout_qe_mean": float(np.mean(held[ok])), "heldout_qe_rel_mean": float(np.mean(kf["qe_rel"][ok])), "heldout_over_train": float(np.mean(held[ok]) / max(qs["qe_mean"], 1e-300)),
                    "heldout_captured": float(1.0 - np.mean(kf["qe_rel"][ok])), "heldout_empty_share": float(np.mean(kf["empty_unit"][ok]))})
    # the share of the variation the first two principal components capture, for the comparison
    Xc, _, dof = _centre(X, sample_w)
    tot = float((Xc ** 2).sum())
    from .metric import pca_basis
    V, s = pca_basis(Xc, 2)
    out["pca2_captured"] = float((s ** 2).sum() / tot) if tot > 0 else float("nan")
    return out


def _centre(X, w):
    from .metric import weighted_centre
    mu = np.nanmean(X, axis=0)
    return weighted_centre(np.where(np.isfinite(X), X, mu), w)
