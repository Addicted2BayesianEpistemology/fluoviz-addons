"""What the model says about the POPULATION, with posterior uncertainty (SPEC section 4): the loading functions, how the variation of every mode splits into a part the factors
share, a part of each sample of its own, and noise; how much of every point of a mode the OTHER modes (and the covariates) can predict; the correlation between any two points;
the effects of the covariates.  Every quantity is computed at every posterior draw and summarised by its posterior mean and quantiles: a band is never an afterthought.

The predictable fraction (paper, the ``R^2_{1 -> 2}`` of a target given its sources) at a point ``m`` of the target mode is

    R2(m) = [B_T Lambda_T (Psi - V) Lambda_T' B_T']_mm  /  total(m),       total(m) = [B_T Lambda_T Psi Lambda_T' B_T']_mm + [B_T diag(sig2_T) B_T']_mm (+ psi2_T, measured)

with ``Psi = I + beta Cov(x) beta'`` the covariance of the scores over the population and ``V`` the covariance of the scores of a sample whose sources are fully measured (it does
not depend on the measured values: the model is Gaussian).  It is a property of the population, not of a sample, and it does not depend on the orientation of the factors.
"""
from __future__ import annotations

import numpy as np

from . import predict as P
from .posterior import Fitted, MblfView

FAN = (0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)


def _q(x: np.ndarray, probs=FAN, axis: int = 0) -> np.ndarray:
    return np.quantile(x, probs, axis=axis)


def summarise(draws: np.ndarray, level: float = 0.95, simultaneous: bool = True) -> dict:
    """Summary of draws of a function ``(S, M)``: posterior mean, the fan of quantiles, a simultaneous band (``level``), ``p_pos`` = P(value > 0) and the local false sign rate
    ``lfsr = min(P(> 0), P(< 0))`` (small = the sign is certain)."""
    draws = np.asarray(draws, float)
    mean = draws.mean(axis=0)
    out = {"mean": mean, "quantiles": _q(draws) if draws.shape[0] > 1 else np.repeat(mean[None], len(FAN), axis=0), "probs": list(FAN)}
    if draws.shape[0] > 1:
        sd = draws.std(axis=0, ddof=1)
        p_pos = np.mean(draws > 0, axis=0)
        out.update(sd=sd, p_pos=p_pos, lfsr=np.minimum(p_pos, 1 - p_pos))
        if simultaneous:
            c = P.sup_multiplier(draws, mean, sd, level)
            out["band"] = {"level": level, "lower": mean - c * sd, "upper": mean + c * sd, "multiplier": c}
    return out


def interval(x: np.ndarray, level: float = 0.9) -> dict:
    """Mean and central interval of scalar draws ``(S,)``."""
    x = np.asarray(x, float)
    a = (1 - level) / 2
    if x.size == 1:
        return {"mean": float(x[0]), "lower": float(x[0]), "upper": float(x[0]), "level": level}
    return {"mean": float(x.mean()), "lower": float(np.quantile(x, a)), "upper": float(np.quantile(x, 1 - a)), "level": level}


# ----------------------------------------------------------------------------- the factors
def factor_summaries(f: Fitted, idx: np.ndarray, level: float = 0.95) -> list[dict]:
    """Per factor (aligned, ordered by strength): per mode the summary of its loading function (data units: the change of the mode for one standard deviation of the score), and the
    share of the shared variation it carries (posterior mean and 90 % interval)."""
    from .posterior import alignment
    idx = np.asarray(idx, int)
    lam_a = f.draws.lam[idx] @ f.draws.T[idx]
    roots = [m.frame_root() for m in f.modes]
    Z = np.concatenate([np.einsum("ij,sjk->sik", R, lam_a[:, sl, :]) for R, sl in zip(roots, f.offsets())], axis=1)
    strength = np.sum(Z ** 2, axis=1)
    share = strength / np.where(strength.sum(axis=1, keepdims=True) > 0, strength.sum(axis=1, keepdims=True), 1.0)
    out = []
    for k in range(f.K):
        per_mode = []
        for d, m in enumerate(f.modes):
            F = f.loading_functions(d, True, True, idx)[:, k, :]
            per_mode.append({"mode": m.name, **summarise(F, level)})
        out.append({"factor": k + 1, "share": interval(share[:, k]), "modes": per_mode})
    _ = alignment                                                       # (the alignment itself is computed once, at fit time)
    return out


def trilinear_loadings(f: Fitted, idx: np.ndarray, max_draws: int = 200) -> dict:
    """How far the EEM loading of every factor is from ONE emission profile times ONE excitation profile (the trilinear structure of PARAFAC), checked, not imposed: for every
    surface mode and factor, the posterior (at most ``max_draws`` draws) of the trilinear share of FluoViz's ``rank_one`` (the best product on the pixels that exist), and
    the two profiles of the posterior-mean loading.  ``{mode name: [{factor, mean, lower, upper (90 %), share_of_mean, em, ex}]}``."""
    from fluoviz.analysis.fpca.trilinear import rank_one
    from fluoviz.analysis.frozen.kinds.fpca import ElementRuntime
    idx = np.asarray(idx, int)
    sub = idx[np.unique(np.linspace(0, idx.size - 1, min(idx.size, max_draws)).astype(int))]
    out = {}
    for d, m in enumerate(f.modes):
        if m.kind != "surface" or m.element is None:
            continue
        dom = ElementRuntime(m.element).dom
        F = f.loading_functions(d, True, True, sub)                               # (S, K, M)
        S_, K, M = F.shape
        sh = rank_one(dom, F.reshape(-1, M))["share"].reshape(S_, K)
        mean = rank_one(dom, F.mean(axis=0))
        out[m.name] = [{"factor": k + 1, **interval(sh[:, k][np.isfinite(sh[:, k])] if np.isfinite(sh[:, k]).any() else np.array([np.nan])),
                        "share_of_mean": float(mean["share"][k]), "em": mean["em"][k], "ex": mean["ex"][k],
                        "em_axis": np.asarray(dom.axes[0], float), "ex_axis": np.asarray(dom.axes[1], float)} for k in range(K)]
    return out


# ----------------------------------------------------------------------------- how the variation splits, point by point
def variance_parts(f: Fitted, d: int, idx: np.ndarray) -> dict:
    """At every point of mode ``d`` (data units squared, posterior mean and 90 % interval of the shares): ``factors`` (the variation the factors carry, covariates included),
    ``covariates`` (the part due to the covariates), ``residual`` (each sample's own), ``noise`` (measurement); and the shares of the whole mode (domain-weighted)."""
    idx = np.asarray(idx, int)
    m = f.modes[d]
    BL = np.einsum("mp,spk->smk", m.B, f.lam(d, idx=idx))
    Psi = f.psi_cov(idx)
    if Psi is None:
        fac = np.sum(BL ** 2, axis=2)
        cov = np.zeros_like(fac)
    else:
        fac = np.einsum("smk,skl,sml->sm", BL, Psi, BL)
        cov = fac - np.sum(BL ** 2, axis=2)
    res = (m.B ** 2) @ f.sig2(d, idx).T
    res = res.T
    noi = np.repeat(f.psi2(d, idx)[:, None], m.M, axis=1)
    tot = fac + res + noi
    sc2 = m.scale ** 2
    w = m.w
    whole = {k: np.sum(w * v, axis=1) / np.sum(w * tot, axis=1) for k, v in (("factors", fac), ("covariates", cov), ("residual", res), ("noise", noi))}
    return {
        "variance": {k: (v.mean(axis=0) * sc2) for k, v in (("factors", fac), ("covariates", cov), ("residual", res), ("noise", noi), ("total", tot))},
        "share": {k: summarise(v / tot, simultaneous=False) for k, v in (("factors", fac), ("residual", res), ("noise", noi))},
        "whole": {k: interval(v) for k, v in whole.items()},
    }


# ----------------------------------------------------------------------------- what the other modes can tell
def _full_roots(f: Fitted) -> list:
    out = []
    for m in f.modes:
        U, s, Vt = np.linalg.svd(m.B, full_matrices=False)
        out.append(s[:, None] * Vt)
    return out


def predictable(f: Fitted, t: int, modes: list[int], covariates: bool, idx: np.ndarray, roots: list | None = None) -> dict:
    """The predictable fraction ``R2`` of the target ``t`` given fully measured ``modes`` (and the covariates when ``covariates``): per point (posterior mean, 90 % interval) for the
    latent function and for a new measurement, and for the whole mode (domain-weighted).  See the module text."""
    idx = np.asarray(idx, int)
    roots = roots or _full_roots(f)
    S, K = idx.size, f.K
    Psi = f.psi_cov(idx)
    Psi = np.broadcast_to(np.eye(K), (S, K, K)).copy() if Psi is None else Psi
    cov_known = covariates and f.L > 0
    prec = np.broadcast_to(np.eye(K), (S, K, K)).copy() if cov_known else np.linalg.inv(Psi)
    for d in modes:
        R = roots[d]
        G = R @ f.lam(d, idx=idx)
        W = (R[None] * f.sig2(d, idx)[:, None, :]) @ R.T + f.psi2(d, idx)[:, None, None] * np.eye(R.shape[0])
        prec = prec + np.swapaxes(G, 1, 2) @ np.linalg.solve(W, G)
    V = np.linalg.inv(prec) if (modes or cov_known) else Psi
    E = Psi - V                                                       # the explained covariance of the scores
    E = 0.5 * (E + np.swapaxes(E, 1, 2))
    m = f.modes[t]
    BL = np.einsum("mp,spk->smk", m.B, f.lam(t, idx=idx))
    expl = np.einsum("smk,skl,sml->sm", BL, E, BL)
    lat = np.einsum("smk,skl,sml->sm", BL, Psi, BL) + ((m.B ** 2) @ f.sig2(t, idx).T).T
    noisy = lat + f.psi2(t, idx)[:, None]
    w = m.w
    r2l = expl / lat
    r2n = expl / noisy
    return {"latent": summarise(np.clip(r2l, 0, 1), simultaneous=False), "measured": summarise(np.clip(r2n, 0, 1), simultaneous=False),
            "whole_latent": interval(np.sum(w * expl, axis=1) / np.sum(w * lat, axis=1)), "whole_measured": interval(np.sum(w * expl, axis=1) / np.sum(w * noisy, axis=1)),
            "draws_whole_latent": np.sum(w * expl, axis=1) / np.sum(w * lat, axis=1)}


def predictable_matrix(f: Fitted, idx: np.ndarray) -> list[dict]:
    """Every target and every set of sources: the whole-mode predictable fraction (posterior mean, 90 % interval) — the "what tells what" table of the page."""
    from .loo import source_sets
    roots = _full_roots(f)
    out = []
    for t in range(f.D):
        for s in source_sets(f, t):
            r = predictable(f, t, s["modes"], s["covariates"], idx, roots)
            out.append({"target": f.modes[t].name, "sources": [f.modes[d].name for d in s["modes"]], "covariates": s["covariates"], "key": s["key"],
                        "latent": r["whole_latent"], "measured": r["whole_measured"], "draws": r["draws_whole_latent"]})
    return out


# ----------------------------------------------------------------------------- correlations between points
def correlation_row(f: Fitted, a: int, i: int, b: int, idx: np.ndarray) -> dict:
    """The correlation, over the population, between point ``i`` of mode ``a`` and every point of mode ``b`` (latent functions, measurement noise left out): posterior mean, 90 %
    interval, and the probability that it is positive."""
    idx = np.asarray(idx, int)
    Psi = f.psi_cov(idx)
    S, K = idx.size, f.K
    Psi = np.broadcast_to(np.eye(K), (S, K, K)) if Psi is None else Psi
    ma, mb = f.modes[a], f.modes[b]
    la = np.einsum("p,spk->sk", ma.B[i], f.lam(a, idx=idx))
    Lb = np.einsum("mp,spk->smk", mb.B, f.lam(b, idx=idx))
    cov = np.einsum("sk,skl,sml->sm", la, Psi, Lb)
    if a == b:
        cov = cov + np.einsum("p,sp,mp->sm", ma.B[i], f.sig2(a, idx), mb.B)
    va = np.einsum("sk,skl,sl->s", la, Psi, la) + (ma.B[i] ** 2) @ f.sig2(a, idx).T
    vb = np.einsum("smk,skl,sml->sm", Lb, Psi, Lb) + ((mb.B ** 2) @ f.sig2(b, idx).T).T
    r = cov / np.sqrt(np.maximum(va[:, None] * vb, 1e-300))
    out = summarise(r, simultaneous=False)
    return {"mean": out["mean"], "lower": out["quantiles"][1] if r.shape[0] > 1 else out["mean"], "upper": out["quantiles"][5] if r.shape[0] > 1 else out["mean"],
            "p_pos": out.get("p_pos"), "lfsr": out.get("lfsr")}


# ----------------------------------------------------------------------------- the covariates
def covariate_effects(f: Fitted, idx: np.ndarray, level: float = 0.95) -> list[dict]:
    """Per design column of the covariates and per mode: the effect function ``scale * B Lambda beta_l`` (data units: the change of the mode for one standard deviation of a
    number, or for a category against its reference), with bands and sign probabilities; and the share of each mode's variation the covariates explain."""
    if f.L == 0:
        return []
    idx = np.asarray(idx, int)
    b = f.beta(idx)
    out = []
    for l, col in enumerate(f.cov.columns):
        per_mode = []
        for d, m in enumerate(f.modes):
            eff = np.einsum("mp,spk,sk->sm", m.B, f.lam(d, idx=idx), b[:, :, l]) * m.scale[None]
            per_mode.append({"mode": m.name, **summarise(eff, level)})
        out.append({"column": dict(col), "modes": per_mode})
    return out


def covariate_shares(f: Fitted, idx: np.ndarray) -> list[dict]:
    if f.L == 0:
        return []
    return [{"mode": m.name, **variance_parts(f, d, idx)["whole"]["covariates"]} for d, m in enumerate(f.modes)]


# ----------------------------------------------------------------------------- how many factors
def effective_dimension(f: Fitted, idx: np.ndarray) -> dict:
    """mblf's effective dimension (the number of directions needed for 90 / 95 / 99 % of the shared variation, at every draw) with the quadrature weights of the domains, and the
    posterior probability of every number of directions."""
    from ..vendor.mblf.secondary import effective_dimension as ed
    idx = np.asarray(idx, int)
    sub = Fitted(f.modes, f.cov, f.draws.subset(idx), f.Y, f.ids)
    view = MblfView(sub)
    res = ed(view, point_weights=[m.w for m in f.modes], covariates=f.L > 0)
    ranks = res["ranks"]
    probs = []
    for j, row in enumerate(res["table"]):
        vals, counts = np.unique(ranks[:, j], return_counts=True)
        probs.append({"energy": row["energy"], "p": {int(v): float(c / ranks.shape[0]) for v, c in zip(vals, counts)}})
    return {"table": res["table"], "spectrum": res["spectrum"], "shared_fraction": res["shared_fraction"], "probabilities": probs}
