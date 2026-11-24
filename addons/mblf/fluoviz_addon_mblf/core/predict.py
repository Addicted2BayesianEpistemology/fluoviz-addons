"""Predicting a mode of a sample from what is known about it, with the uncertainty taken apart (SPEC section 5; paper, Section "Regression problems").

**At one draw of the population parameters** everything is Gaussian.  What is known about a sample (its *sources*: any of its modes, observed at any of their points, and its
covariates, or not) gives the posterior of its scores ``eta | sources ~ N(eta_hat, V)`` (precision ``Psi^-1 + sum_d G_d' W_d^-1 G_d`` with ``G_d = R_d Lambda_d``,
``W_d = R_d diag(sig2_d) R_d' + psi2_d I``, ``B_d,obs = U_d R_d``; the prior is ``N(beta x, I)`` when the covariates are known and ``N(beta xbar, I + beta Cov(x) beta')`` when they are not; ``xbar`` = 0 for centred covariates),
and the target mode ``f = B_T theta_T`` then has

    mean      B_T Lambda_T eta_hat
    variance  B_T Lambda_T V Lambda_T' B_T'           (the scores of THIS sample are not exactly known)
            + B_T diag(sig2_T) B_T'                   (the sample's own residual: what no other measurement of it can tell)
            + psi2_T I                                (measurement noise: only for a new noisy measurement, not for the latent function).

**Over the posterior** the prediction is the equal-weight mixture of these Gaussians (the "modular" rule of the paper), whose variance adds a fourth part, the spread of the
conditional means across the draws: the uncertainty about the population parameters (it shrinks with more training samples; it is zero for a maximum-likelihood fit, which
ignores it).  So, at every point, the predictive variance is exactly

    parameters + scores + residual + noise

which is what the page shows as the *uncertainty budget*: what would shrink the uncertainty (more training samples / more or better measurements of the sample / nothing: it is
the sample's own) and what is the instrument.  Bands are quantiles of the mixture (computed exactly by bisection of its distribution function, pointwise), or simultaneous (a band
that contains the whole function with the stated probability: the largest standardised deviation of draws of whole functions).

All arrays are in standardised units unless said otherwise; ``Mode.to_data`` turns them back.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from ..vendor.mblf.data import SINGULAR_RTOL
from .posterior import Fitted

PROBS = (0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)          # the fan: central 50 %, 80 % and 95 % intervals and the median
PARTS = ("parameters", "scores", "residual", "noise")
PART_WORDS = {
    "parameters": "the population parameters (more training samples shrink it)",
    "scores": "the factor scores of this sample (more, or more precise, measurements of the sample shrink it)",
    "residual": "the sample's own variation (what the other measurements cannot tell about it)",
    "noise": "measurement noise (only for a new measurement)",
}


# ----------------------------------------------------------------------------- the scores of samples, given what is known about them
def _rows_root(B: np.ndarray, rows: np.ndarray):
    U, s, Vt = np.linalg.svd(B[rows], full_matrices=False)
    keep = s > SINGULAR_RTOL * s[0] if s.size and s[0] > 0 else np.zeros(s.size, bool)
    return U[:, keep], s[keep][:, None] * Vt[keep]


def score_posterior(f: Fitted, sources: dict, X: np.ndarray | None, idx: np.ndarray, n: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``(eta_hat (S, n, K), V (S, n, K, K))`` of ``n`` samples at the draws ``idx``.

    ``sources`` ``{mode index: (M_d, n) standardised values, NaN = not observed}`` (any subset of the modes; empty = nothing is measured); ``X`` ``(n, L)`` the design of their
    covariates, or None when they are not known (or the model has none).  Samples with the same observed points share the linear algebra."""
    idx = np.asarray(idx, int)
    S, K, L = idx.size, f.K, f.L
    if n is None:
        n = next(iter(sources.values())).shape[1] if sources else (X.shape[0] if X is not None else 1)
    known = L > 0 and X is not None
    if known:
        mu = np.einsum("skl,nl->snk", f.beta(idx), np.asarray(X, float).reshape(n, L))
        Pinv = np.broadcast_to(np.eye(K), (S, K, K)).copy()
    else:
        mu = np.zeros((S, n, K))
        if L > 0:                                                                   # the mean of the population (0 when the covariates are centred on their mean)
            mu += np.einsum("skl,l->sk", f.beta(idx), f.cov.xbar())[:, None, :]
        Psi = f.psi_cov(idx)
        Pinv = np.broadcast_to(np.eye(K), (S, K, K)).copy() if Psi is None else np.linalg.inv(Psi)
    h0 = np.einsum("skl,snl->snk", Pinv, mu)
    eta = np.zeros((S, n, K))
    V = np.zeros((S, n, K, K))
    src = sorted(sources)
    if not src:
        Vg = np.linalg.inv(Pinv)
        eta[:] = mu
        V[:] = Vg[:, None]
        return eta, V
    Y = {d: np.asarray(sources[d], float).reshape(f.modes[d].M, n) for d in src}
    masks = np.vstack([np.isnan(Y[d]) for d in src])
    patterns, which = np.unique(masks.T, axis=0, return_inverse=True)
    which = np.asarray(which).reshape(-1)
    lam = {d: f.lam(d, idx=idx) for d in src}
    sig2 = {d: f.sig2(d, idx) for d in src}
    psi2 = {d: f.psi2(d, idx) for d in src}
    for g in range(patterns.shape[0]):
        cols = np.flatnonzero(which == g)
        prec = Pinv.copy()
        h = h0[:, cols].copy()
        for d in src:
            rows = np.flatnonzero(~np.isnan(Y[d][:, cols[0]]))
            if rows.size == 0:
                continue
            U, R = _rows_root(f.modes[d].B, rows)
            if R.shape[0] == 0:
                continue
            G = R @ lam[d]                                                              # (S, q, K)
            W = (R[None] * sig2[d][:, None, :]) @ R.T + psi2[d][:, None, None] * np.eye(R.shape[0])
            WG = np.linalg.solve(W, G)
            prec += np.swapaxes(G, 1, 2) @ WG
            z = U.T @ Y[d][np.ix_(rows, cols)]                                          # (q, n_g)
            h += np.einsum("sqk,qn->snk", WG, z)
        Vg = np.linalg.inv(prec)
        Vg = 0.5 * (Vg + np.swapaxes(Vg, 1, 2))
        eta[:, cols] = np.einsum("skl,snl->snk", Vg, h)
        V[:, cols] = Vg[:, None]
    return eta, V


def aligned_scores(f: Fitted, eta: np.ndarray, V: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The scores in the aligned orientation of every draw: ``T' eta``, ``T' V T``."""
    T = f.draws.T[np.asarray(idx, int)]
    Tt = np.swapaxes(T, 1, 2)
    return np.einsum("skl,snl->snk", Tt, eta), np.einsum("skl,snlm,smj->snkj", Tt, V, T)


# ----------------------------------------------------------------------------- the target given the scores
def target_moments(f: Fitted, t: int, eta: np.ndarray, V: np.ndarray, idx: np.ndarray, points: np.ndarray | None = None) -> dict:
    """At every draw: the conditional mean of the target mode ``t`` and the three parts of its conditional variance (standardised units).

    ``eta`` ``(S, n, K)``, ``V`` ``(S, n, K, K)``; ``points`` the rows of the target (all when None).  Returns ``mean`` ``(S, n, M)``, ``scores`` ``(S, n, M)``, ``residual``
    ``(S, M)``, ``noise`` ``(S,)``."""
    idx = np.asarray(idx, int)
    B = f.modes[t].B if points is None else f.modes[t].B[points]
    BL = np.einsum("mp,spk->smk", B, f.lam(t, idx=idx))
    mean = np.einsum("smk,snk->snm", BL, eta)
    LV = np.linalg.cholesky(V + 1e-12 * np.eye(V.shape[-1]))
    scores = np.sum(np.einsum("smk,snkj->snmj", BL, LV) ** 2, axis=-1)
    residual = (B ** 2) @ f.sig2(t, idx).T
    return {"mean": mean, "scores": scores, "residual": residual.T, "noise": f.psi2(t, idx), "BL": BL, "LV": LV}


def budget(mom: dict, noisy: bool, j: int = 0, w: np.ndarray | None = None) -> dict:
    """The four parts of the predictive variance of sample ``j`` at every point ``(M,)`` (standardised units squared); ``w`` weights of the draws (equal when None)."""
    m = mom["mean"][:, j]
    S = m.shape[0]
    w = np.full(S, 1.0 / S) if w is None else np.asarray(w, float) / np.sum(w)
    mu = w @ m
    out = {"parameters": w @ (m - mu) ** 2 if S > 1 else np.zeros(m.shape[1]),
           "scores": w @ mom["scores"][:, j], "residual": w @ mom["residual"],
           "noise": np.full(m.shape[1], float(w @ mom["noise"]) if noisy else 0.0)}
    return out


# ----------------------------------------------------------------------------- mixtures of Gaussians
def mixture_quantiles(m: np.ndarray, sd: np.ndarray, probs=PROBS, w: np.ndarray | None = None, iters: int = 22, chunk: int = 4096) -> np.ndarray:
    """Quantiles ``(len(probs), M)`` of the pointwise mixtures ``sum_s w_s N(m[s], sd[s]^2)`` (equal weights when ``w`` is None), by bisection of the distribution function inside
    the mean +- 7 total standard deviations (Chebyshev: every quantile from 1 % to 99 % lies there)."""
    m = np.asarray(m, float)
    sd = np.maximum(np.asarray(sd, float), 1e-300)
    S, M = m.shape
    w = np.full(S, 1.0 / S) if w is None else np.asarray(w, float) / np.sum(w)
    probs = np.asarray(probs, float)
    out = np.empty((probs.size, M))
    for a in range(0, M, chunk):
        mm, ss = m[:, a:a + chunk], sd[:, a:a + chunk]
        mu = w @ mm
        tot = np.sqrt(np.maximum(w @ (ss ** 2 + mm ** 2) - mu ** 2, 0.0)) + 1e-300
        lo = np.broadcast_to(mu - 7 * tot, (probs.size, mu.size)).copy()
        hi = np.broadcast_to(mu + 7 * tot, (probs.size, mu.size)).copy()
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            F = np.einsum("s,sqm->qm", w, ndtr((mid[None] - mm[:, None]) / ss[:, None]))
            below = F < probs[:, None]
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        out[:, a:a + chunk] = 0.5 * (lo + hi)
    return out


def mixture_cdf(y: np.ndarray, m: np.ndarray, sd: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """``P(Y <= y)`` at every point for the mixture (the PIT of an observed value)."""
    S = m.shape[0]
    w = np.full(S, 1.0 / S) if w is None else np.asarray(w, float) / np.sum(w)
    return w @ ndtr((np.asarray(y, float)[None] - m) / np.maximum(sd, 1e-300))


def sup_multiplier(draws: np.ndarray, centre: np.ndarray, sd: np.ndarray, level: float) -> float:
    """The multiplier ``c`` of a simultaneous band ``centre +- c sd`` that contains a ``level`` share of the whole functions ``draws`` ``(J, M)`` (the largest standardised deviation)."""
    z = np.max(np.abs(draws - centre[None]) / np.maximum(sd[None], 1e-300), axis=1)
    return float(np.quantile(z, level)) if z.size > 1 else float("nan")


# ----------------------------------------------------------------------------- one prediction, everything the page shows
def predict(f: Fitted, t: int, sources: dict, X: np.ndarray | None, idx: np.ndarray, noisy: bool = False, level: float = 0.95, seed: int = 0, n_function_draws: int = 2,
            points: np.ndarray | None = None, w: np.ndarray | None = None) -> dict:
    """The prediction of mode ``t`` of ONE sample (``sources`` and ``X`` as in :func:`score_posterior`, with one column / row).

    Returns, in DATA units: ``mean``, ``quantiles`` ``(len(PROBS), M)``, ``band`` (simultaneous ``level`` band, lower / upper), ``budget`` ``{part: (M,)}`` (variances) with
    ``budget_total`` (the share of each part over the domain, weighted by the quadrature weights), ``p_above`` (the probability that the value is above the mean of the training
    samples, at every point), ``scores`` (aligned: mean, sd, central 95 % interval of each factor) and the number of draws.  ``w``: weights of the draws (the leave-one-out weights
    of a training sample: its prediction is then held out); equal when None."""
    idx = np.asarray(idx, int)
    mode = f.modes[t]
    eta, V = score_posterior(f, sources, X, idx, n=1)
    mom = target_moments(f, t, eta, V, idx, points)
    bud = budget(mom, noisy, 0, w)
    m = mom["mean"][:, 0]                                                          # (S, M)
    S0 = m.shape[0]
    wn = np.full(S0, 1.0 / S0) if w is None else np.asarray(w, float) / np.sum(w)
    var_s = mom["scores"][:, 0] + mom["residual"] + (mom["noise"][:, None] if noisy else 0.0)
    sd = np.sqrt(np.maximum(var_s, 1e-300))
    q = mixture_quantiles(m, sd, PROBS, wn)
    centre = wn @ m
    total_sd = np.sqrt(sum(bud.values()))
    rng = np.random.default_rng(seed)
    S, M, K = m.shape[0], m.shape[1], f.K
    B = mode.B if points is None else mode.B[points]
    J = []
    pick = rng.choice(S, size=S, p=wn) if w is not None else np.arange(S)        # draws of whole functions follow the weights
    for _ in range(max(1, n_function_draws)):
        z1 = rng.standard_normal((S, K))
        fx = m[pick] + np.einsum("smk,skj,sj->sm", mom["BL"][pick], mom["LV"][pick, 0], z1)
        fx += (rng.standard_normal((S, mode.p)) * np.sqrt(f.sig2(t, idx)[pick])) @ B.T
        if noisy:
            fx += rng.standard_normal((S, M)) * np.sqrt(mom["noise"][pick])[:, None]
        J.append(fx)
    J = np.vstack(J)
    c = sup_multiplier(J, centre, total_sd, level)
    w = mode.w if points is None else mode.w[points] / mode.w[points].sum()
    tot = {k: float(np.sum(w * v)) for k, v in bud.items()}
    s_all = sum(tot.values()) or 1.0
    sc = mode.scale if points is None else mode.scale[points]
    ea, Va = aligned_scores(f, eta, V, idx)
    sd_k = np.sqrt(np.maximum(np.diagonal(Va[:, 0], axis1=1, axis2=2), 1e-300))
    sq = mixture_quantiles(ea[:, 0], sd_k, (0.025, 0.975), wn)
    e_mean = wn @ ea[:, 0]
    mean_data = (mode.mean if points is None else mode.mean[points]) + centre * sc
    return {
        "mean": mean_data,
        "quantiles": (mode.mean if points is None else mode.mean[points])[None] + q * sc[None],
        "probs": list(PROBS),
        "band": {"level": level, "lower": mean_data - c * total_sd * sc, "upper": mean_data + c * total_sd * sc, "multiplier": c},
        "budget": {k: v * sc ** 2 for k, v in bud.items()},
        "budget_total": {k: v / s_all for k, v in tot.items()},
        "p_above": wn @ ndtr(m / sd),
        "scores": {"mean": e_mean, "sd": np.sqrt(wn @ np.diagonal(Va[:, 0], axis1=1, axis2=2) + wn @ (ea[:, 0] - e_mean) ** 2), "lower": sq[0], "upper": sq[1]},
        "noisy": bool(noisy), "n_draws": int(S),
    }
