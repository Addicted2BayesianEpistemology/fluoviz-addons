"""The scans: the map size, the stability over random starts, the choice of distance (SPEC section 5).  Each takes a ``Prepared`` and a ``Config`` and returns a plain dictionary."""
from __future__ import annotations

import copy

import numpy as np

from . import advice, diagnostics
from . import som as S
from .fitting import FitResult, fit_map
from .spec import Config, SomError


def _sub(progress, i, n):
    return progress.sub(i / n, (i + 1) / n) if progress is not None else None


def size_candidates(n: int) -> list:
    raw = [n / 8.0, n / 4.0, n / 2.0, 2.0 * np.sqrt(n), 3.0 * np.sqrt(n), 5.0 * np.sqrt(n), 7.0 * np.sqrt(n), 10.0 * np.sqrt(n)]
    cap = min(4 * n, 400)
    out = sorted({int(min(max(round(u), 6), cap)) for u in raw})
    return out[:8]


def size_scan(prep, cfg: Config, units: list | None = None, progress=None) -> dict:
    """Fit maps of several sizes and compare their held-out errors.  The suggested size is the smallest whose held-out error is within one standard error of the best and whose topographic
    error is below 10 % (when any is)."""
    from fluoviz.analysis.common import Progress
    progress = progress or Progress()
    n = prep.n
    units = units or size_candidates(n)
    rows = []
    sw = prep.sample_weights(cfg.gap_weighting)
    base = fit_map(prep, _with(cfg, size="standard"), None, sw, with_kfold=False)
    X = base.metric.transform(prep.R)
    ratio = advice.side_ratio(X, sw)
    for i, u in enumerate(units):
        r, c = advice.shape_for(u, ratio, cfg.lattice, cfg.shape)
        fit = fit_map(prep, _with(cfg, size=[r, c]), _sub(progress, i, len(units)), sw)
        q = fit.quality
        held = fit.kfold["qe_rel"]
        ok = np.isfinite(held)
        rows.append({"units": r * c, "rows": r, "cols": c, "qe_rel": q["qe_rel_mean"], "heldout_qe_rel": float(np.mean(held[ok])) if ok.any() else float("nan"),
                     "heldout_se": float(np.std(held[ok], ddof=1) / np.sqrt(ok.sum())) if ok.sum() > 1 else float("nan"), "topographic_error": q["topographic_error"],
                     "trustworthiness": q["trust_map"]["trustworthiness"], "n_empty": q["n_empty"]})
    rows.sort(key=lambda x: x["units"])
    h = np.array([r["heldout_qe_rel"] for r in rows])
    suggested, reason = None, ""
    if np.isfinite(h).any():
        best = int(np.nanargmin(h))
        tol = h[best] + (rows[best]["heldout_se"] if np.isfinite(rows[best]["heldout_se"]) else 0.0)
        cand = [i for i, r in enumerate(rows) if np.isfinite(h[i]) and h[i] <= tol]
        good = [i for i in cand if rows[i]["topographic_error"] < 0.1]
        pick = min(good or cand, key=lambda i: rows[i]["units"])
        suggested = rows[pick]["units"]
        reason = "" if good else "No map of this range had a topographic error below 10 %, so the topographic error did not decide."
    return {"rows": rows, "suggested": suggested, "verdict": diagnostics.size_scan(rows, suggested, reason)}


def _with(cfg: Config, **kw) -> Config:
    c = copy.deepcopy(cfg)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def _ward_labels(protos: np.ndarray, hits: np.ndarray, k: int) -> np.ndarray:
    """Group labels of the units (Ward on the prototypes of the occupied units; empty units join the nearest occupied one)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    occ = hits > 0
    lab = np.zeros(protos.shape[0], int)
    if occ.sum() >= max(3, k):
        Z = linkage(protos[occ], "ward")
        lab[occ] = fcluster(Z, t=k, criterion="maxclust") - 1
        for j in np.flatnonzero(~occ):
            lab[j] = lab[occ][int(np.argmin(((protos[occ] - protos[j]) ** 2).sum(axis=1)))]
    return lab


def _ari(a: np.ndarray, b: np.ndarray) -> float:
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(a, b))


def stability(prep, cfg: Config, fit: FitResult | None = None, n_runs: int = 10, bootstrap: bool = False, k: int = 4, progress=None) -> dict:
    """Train the map again from random starting points (and, with ``bootstrap``, on resampled samples) and ask which pairs of samples stay together."""
    from fluoviz.analysis.common import Progress
    progress = progress or Progress()
    n = prep.n
    sw = prep.sample_weights(cfg.gap_weighting)
    fit = fit or fit_map(prep, cfg, None, sw, with_kfold=False)
    lat = fit.lattice
    ref_c = fit.asg["c1"]
    together_ref = lat.d2[ref_c][:, ref_c] <= 1.0 + 1e-9                  # same or neighbouring unit on the kept map
    np.fill_diagonal(together_ref, False)
    base = float((lat.d2[np.ix_(ref_c, ref_c)] <= 1.0 + 1e-9)[~np.eye(n, dtype=bool)].mean())
    ref_groups = _ward_labels(fit.protos_t, fit.hits, k)[ref_c]
    acc = np.zeros((n, n))
    aris = []
    rng = np.random.default_rng(cfg.seed + 101)
    X = fit.metric.transform(prep.R)
    for r in range(n_runs):
        spec = S.TrainSpec(cfg.mode, "random", cfg.neighbourhood, cfg.epochs_rough, cfg.epochs_fine, cfg.radius_initial, cfg.radius_final, cfg.alpha_initial, cfg.alpha_final, cfg.seed + 1 + r)
        rows = np.sort(rng.choice(n, size=n, replace=True)) if bootstrap else np.arange(n)
        w = None if sw is None else sw[rows]
        tm = S.train(X[rows], lat, spec, w, fit.metric.a if not fit.metric.whitens else None)
        c, _, _ = S.best_units(X, tm.protos, fit.metric.a if not fit.metric.whitens else None)
        T = lat.d2[c][:, c] <= 1.0 + 1e-9
        acc += T
        cnt, _ = S.hits(c, lat.M)
        aris.append(_ari(ref_groups, _ward_labels(tm.protos, cnt, k)[c]))
        progress((r + 1) / n_runs, f"Training again from another start, run {r + 1} of {n_runs}")
    acc /= max(n_runs, 1)
    pairs = together_ref
    value = float(acc[pairs].mean()) if pairs.any() else float("nan")
    ari = float(np.mean(aris)) if aris else float("nan")
    return {"value": value, "baseline": base, "ari": ari, "ari_runs": aris, "n_runs": n_runs, "bootstrap": bool(bootstrap), "k": k, "verdict": diagnostics.stability(value, base, ari)}


def metric_compare(prep, cfg: Config, k: int = 4, progress=None) -> dict:
    """The same map under the different distances: neighbour preservation, held-out error (in the plain data units, so it is comparable) and the agreement of the groups with the current map."""
    from fluoviz.analysis.common import Progress
    progress = progress or Progress()
    opts = [("l2", 1.0), ("mahalanobis", 1.0), ("mahalanobis", 0.5), ("standardize", 1.0)]
    if prep.prior_scales() is not None:
        opts.insert(3, ("prior", 1.0))
    sw = prep.sample_weights(cfg.gap_weighting)
    rows, ref_labels = [], None
    for i, (kind, alpha) in enumerate(opts):
        c = _with(cfg, metric_kind=kind, alpha=alpha)
        try:
            fit = fit_map(prep, c, _sub(progress, i, len(opts)), sw)
        except SomError as exc:
            rows.append({"kind": kind, "alpha": alpha, "error": str(exc)})
            continue
        q = fit.quality
        lab = _ward_labels(fit.protos_t, fit.hits, k)[fit.asg["c1"]]
        if ref_labels is None:
            ref_labels = lab
        held = fit.kfold["qe_rel"]
        rows.append({"kind": kind, "alpha": alpha, "trustworthiness": q["trust_map"]["trustworthiness"], "continuity": q["trust_map"]["continuity"], "topographic_error": q["topographic_error"],
                     "heldout_qe_rel": float(np.nanmean(held)) if np.isfinite(held).any() else float("nan"), "agreement": _ari(ref_labels, lab), "components": fit.metric.components})
    return {"rows": rows, "k": k, "reference": "l2"}
