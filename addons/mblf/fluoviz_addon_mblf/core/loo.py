"""Leave-one-sample-out checks from ONE fit (PSIS-LOO; SPEC section 6).

``log p(y_n | parameters)``: the marginal likelihood of a sample's measurements with its scores and residual coefficients integrated out (paper, Theorem 1; mblf
``likelihood.marginal_loglik``), here for ANY subset of its modes and with the covariates known (prior ``N(beta x, I)``) or not (prior ``N(0, I + beta Cov(x) beta')``).

Leaving sample ``n`` out of the training data reweights the draws by ``1 / p(y_n | draw)`` (all its modes: that is what the fit saw); Pareto smoothing makes the weights stable and
its ``k`` says when they are not (``k > 0.7``: the estimate for that sample is unreliable).  With these weights, for a *target* mode and a set of *sources* (other modes, the
covariates), the held-out log predictive density of the target given the sources is

    log p(y_T,n | y_S,n, data without n) = log sum_s w_s(n) p(y_T,n | y_S,n, draw s),    p(y_T | y_S) = p(y_T, y_S) / p(y_S),

so every combination ("the EEM from the absorbance", "the EEM from the absorbance and the covariates", "the EEM from nothing" ...) is compared on the same samples, honestly, without
fitting again.  The same weights give held-out predictive distributions of every measured target value: their PIT (where the value falls in its predictive distribution) gives the
coverage of the 50 / 80 / 95 % intervals, and their means the error.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from ..vendor.mblf.data import mode_summaries
from ..vendor.mblf.diagnostics import psis, relative_eff
from . import predict as P
from .posterior import Fitted

LOG_2PI = np.log(2 * np.pi)


def summaries(f: Fitted, Y: list | None = None) -> list:
    """``mode_summaries`` of every mode (training data by default)."""
    Y = f.Y if Y is None else Y
    return [mode_summaries(m.B, y) for m, y in zip(f.modes, Y)]


def loglik(f: Fitted, summ: list, modes: list[int], idx: np.ndarray, X: np.ndarray | None, covariates: bool, subjects=None) -> np.ndarray:
    """``(S, N)`` log-likelihood of the modes ``modes`` of every sample at the draws ``idx`` (scores and residual coefficients integrated out); ``covariates``: condition the scores
    on ``X`` (else the prior of the scores is that of the population, ``I + beta Cov(x) beta'``).  An empty ``modes`` gives zeros.  ``subjects``: only these columns (the others stay 0)."""
    idx = np.asarray(idx, int)
    S, K = idx.size, f.K
    N = summ[0][0].shape[1]
    out = np.zeros((S, N))
    if not modes:
        return out
    if f.L and covariates:
        mu = np.einsum("skl,nl->snk", f.beta(idx), np.asarray(X, float))
        Lp = None
    else:
        mu = np.zeros((S, N, K))
        Psi = f.psi_cov(idx)
        Lp = None if Psi is None else np.linalg.cholesky(Psi)
    lam = {d: (f.lam(d, idx=idx) if Lp is None else f.lam(d, idx=idx) @ Lp) for d in modes}
    sig2 = {d: f.sig2(d, idx) for d in modes}
    psi = {d: f.psi2(d, idx) for d in modes}
    eye = np.eye(K)
    for n in (range(N) if subjects is None else subjects):
        H = np.broadcast_to(eye, (S, K, K)).copy()
        h = np.zeros((S, K))
        quad = np.zeros(S)
        const = np.zeros(S)
        any_obs = False
        for d in modes:
            T, rss, n_obs, G = summ[d]
            if n_obs[n] == 0:
                continue
            any_obs = True
            R = G[n]
            keep = np.any(R != 0, axis=1)
            R = R[keep]
            p = R.shape[0]
            E = (R[None] * sig2[d][:, None, :]) @ R.T + psi[d][:, None, None] * np.eye(p)
            RL = R @ lam[d]                                                     # (with the covariates unknown, lam = Lambda chol(Psi) and mu = 0)
            r = R @ T[:, n] - np.einsum("spk,sk->sp", RL, mu[:, n])
            sol = np.linalg.solve(E, np.concatenate([RL, r[..., None]], axis=-1))
            quad += np.einsum("sp,sp->s", r, sol[..., K])
            h += np.einsum("spk,sp->sk", RL, sol[..., K])
            H += np.swapaxes(RL, 1, 2) @ sol[..., :K]
            logdet_E = 2 * np.sum(np.log(np.diagonal(np.linalg.cholesky(E), axis1=1, axis2=2)), axis=1)
            q = int(n_obs[n])
            const += -0.5 * q * (LOG_2PI + np.log(psi[d])) + 0.5 * p * np.log(psi[d]) - 0.5 * rss[n] / psi[d] - 0.5 * logdet_E
        if not any_obs:
            continue
        LH = np.linalg.cholesky(H)
        Hinv_h = np.linalg.solve(H, h[..., None])[..., 0]
        out[:, n] = const - np.sum(np.log(np.diagonal(LH, axis1=1, axis2=2)), axis=1) - 0.5 * (quad - np.einsum("sk,sk->s", h, Hinv_h))
    return out


def psis_weights(f: Fitted, ll: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalised log weights ``(S, N)`` that leave each sample out, and the Pareto ``k`` of each sample (``k`` is NaN for a maximum-likelihood fit: one draw, no weights)."""
    S, N = ll.shape
    if S < 20:
        return np.full((S, N), -np.log(S)), np.full(N, np.nan)
    C = f.draws.chains if S % max(f.draws.chains, 1) == 0 else 1
    r_eff = np.clip(relative_eff(ll.reshape(C, S // C, N)), 1e-12, None) if C > 1 or S > 1 else None
    lw, k = psis(-ll, r_eff)
    return lw, k


def loo_table(f: Fitted, idx: np.ndarray, summ: list | None = None) -> dict:
    """PSIS-LOO of the whole model (every mode of every sample; covariates as in the fit): ``elpd``, ``se``, ``pointwise``, ``pareto_k``, ``p_loo``, the joint log-likelihood and
    the weights (kept for the comparisons below)."""
    summ = summ or summaries(f)
    X = f.cov.X if f.cov is not None else None
    ll = loglik(f, summ, list(range(f.D)), idx, X, True)
    lw, k = psis_weights(f, ll)
    pointwise = logsumexp(lw + ll, axis=0)
    lpd = logsumexp(ll, axis=0) - np.log(ll.shape[0])
    n = pointwise.size
    return {"elpd": float(pointwise.sum()), "se": float(np.sqrt(n * np.var(pointwise, ddof=1))) if n > 1 else float("nan"), "pointwise": pointwise, "pareto_k": k,
            "p_loo": float(np.sum(lpd - pointwise)), "ll": ll, "lw": lw, "loo": ll.shape[0] >= 20}


def source_sets(f: Fitted, t: int) -> list[dict]:
    """Every set of sources for the target ``t``: subsets of the other modes, with and without the covariates (when the model has some), the empty set first."""
    others = [d for d in range(f.D) if d != t]
    sets = []
    for mask in range(1 << len(others)):
        mods = [others[i] for i in range(len(others)) if mask >> i & 1]
        for cov in ([False, True] if f.L else [False]):
            names = [f.modes[d].name for d in mods] + (["covariates"] if cov else [])
            sets.append({"modes": mods, "covariates": cov, "key": "+".join(names) or "nothing"})
    return sets


def conditional_elpd(f: Fitted, t: int, sets: list[dict], idx: np.ndarray, base: dict, summ: list | None = None) -> list[dict]:
    """For every source set: the held-out log predictive density of the target given the sources, per sample (only the samples that have the target), its total and standard
    error, and the paired difference against the empty set (and against the best set) — the one-standard-error comparison of the paper."""
    summ = summ or summaries(f)
    X = f.cov.X if f.cov is not None else None
    lw = base["lw"]
    has = summ[t][2] > 0
    out = []
    cache: dict = {}

    def ll_of(mods, cov):
        key = (tuple(sorted(mods)), bool(cov) and f.L > 0)
        if key not in cache:
            cache[key] = loglik(f, summ, list(key[0]), idx, X, key[1])
        return cache[key]

    for s in sets:
        lc = ll_of(s["modes"] + [t], s["covariates"]) - ll_of(s["modes"], s["covariates"])
        pw = logsumexp(lw + lc, axis=0)
        nobs = summ[t][2]
        out.append({**s, "pointwise": np.where(has, pw, np.nan), "per_value": np.where(has, pw / np.maximum(nobs, 1), np.nan)})
    ref = out[0]["pointwise"]
    totals = [float(np.nansum(o["pointwise"])) for o in out]
    best = int(np.argmax(totals))
    for o, tot in zip(out, totals):
        n = int(np.isfinite(o["pointwise"]).sum())
        diff = (o["pointwise"] - ref)[has]
        dbest = (o["pointwise"] - out[best]["pointwise"])[has]
        o.update({"elpd": tot, "n": n, "se": float(np.sqrt(n * np.nanvar(o["pointwise"][has], ddof=1))) if n > 1 else float("nan"),
                  "gain": float(diff.sum()), "gain_se": float(np.sqrt(n * np.var(diff, ddof=1))) if n > 1 else float("nan"),
                  "vs_best": float(dbest.sum()), "vs_best_se": float(np.sqrt(n * np.var(dbest, ddof=1))) if n > 1 else float("nan"),
                  "per_value_mean": float(np.nanmean(o["per_value"])) if n else float("nan")})
    return out


def calibration(f: Fitted, t: int, s: dict, idx: np.ndarray, base: dict, max_points: int = 400, width_points: int = 40, seed: int = 0) -> dict:
    """Held-out (PSIS) predictive distributions of the MEASURED target values of every training sample, given the sources ``s``: the PIT of every value, the coverage of the central
    50 / 80 / 95 % intervals, the root mean square error of the predictive mean, the mean width of the 80 % interval (data units; from at most ``width_points`` points per sample) and
    the mean predictive standard deviation.  ``max_points`` points per sample at most (a random choice, the same for every source set: ``seed``)."""
    rng = np.random.default_rng(seed)
    mode = f.modes[t]
    X = f.cov.X if f.cov is not None else None
    lw = base["lw"]
    W = np.exp(lw)
    pits, errs, widths, sds, per_sample = [], [], [], [], []
    for n in range(f.N):
        y = f.Y[t][:, n]
        pts = np.flatnonzero(np.isfinite(y))
        if pts.size == 0:
            continue
        if pts.size > max_points:
            pts = np.sort(rng.choice(pts, max_points, replace=False))
        src = {d: f.Y[d][:, [n]] for d in s["modes"]}
        eta, V = P.score_posterior(f, src, X[[n]] if (X is not None and s["covariates"]) else None, idx, n=1)
        mom = P.target_moments(f, t, eta, V, idx, pts)
        m = mom["mean"][:, 0]
        sd = np.sqrt(mom["scores"][:, 0] + mom["residual"] + mom["noise"][:, None])
        w = W[:, n]
        pit = P.mixture_cdf(y[pts], m, sd, w)
        mean = (w / w.sum()) @ m
        pits.append(pit)
        err = (mean - y[pts]) * mode.scale[pts]
        errs.append(err)
        sds.append(np.sqrt(np.maximum((w / w.sum()) @ (sd ** 2 + m ** 2) - mean ** 2, 0.0)) * mode.scale[pts])
        sub = pts[:width_points] if pts.size <= width_points else np.sort(rng.choice(pts, width_points, replace=False))
        loc = np.searchsorted(pts, sub)
        q = P.mixture_quantiles(m[:, loc], sd[:, loc], (0.1, 0.9), w)
        widths.append((q[1] - q[0]) * mode.scale[sub])
        per_sample.append({"index": n, "rmse": float(np.sqrt(np.mean(err ** 2))), "coverage80": float(np.mean((pit > 0.1) & (pit < 0.9)))})
    if not pits:
        return {"n_samples": 0}
    pit = np.concatenate(pits)
    err = np.concatenate(errs)
    cover = {str(lv): float(np.mean((pit > (1 - lv) / 2) & (pit < (1 + lv) / 2))) for lv in (0.5, 0.8, 0.95)}
    hist, _ = np.histogram(pit, bins=10, range=(0, 1))
    return {"n_samples": len(pits), "n_values": int(pit.size), "coverage": cover, "pit_hist": (hist / max(pit.size, 1)).tolist(), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mean_width80": float(np.mean(np.concatenate(widths))), "mean_sd": float(np.mean(np.concatenate(sds))), "per_sample": per_sample}
