"""Small statistics of an autoencoder that the saved kind and the router share: the limit of the reconstruction error, how typical a code is, the KL term and the ELBO.

Pure numpy / scipy.  Nothing here depends on the data of new samples: the numbers come from the training samples and are frozen in the model.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sps

LEVELS = ("95", "99")
Z = {"95": float(sps.norm.ppf(0.95)), "99": float(sps.norm.ppf(0.99))}
MIN_LOG_SPREAD = 0.4                 # the log-errors of ordinary samples are never less spread than this (a chi-square with a few degrees of freedom has more): guards against a tiny limit from few samples


# ----------------------------------------------------------------------------- the reconstruction error limit
def error_limits(errors, fallback=None) -> dict:
    """The 95 % and 99 % points of the reconstruction error of samples the network did NOT train on.

    A reconstruction error is positive and skewed, so it is described on the log scale by its median and a robust spread (1.4826 x the median absolute deviation, never less than
    ``MIN_LOG_SPREAD``): ``limit = exp(median + z x spread)`` with ``z`` 1.645 / 2.326 (the 95 % / 99 % points of a normal distribution).  With fewer than 3 usable values the limits are taken
    from ``fallback`` (the errors of the training samples, which are optimistic) widened by a factor of 2 and the result says so (``source``).

    Returns ``{"95": float, "99": float, "n": int, "median": float, "spread": float, "source": "held-out" | "training", "words": str}``."""
    v = np.asarray(errors, float)
    v = v[np.isfinite(v) & (v > 0)]
    source = "held-out"
    widen = 1.0
    if v.size < 3:
        fb = np.asarray(fallback if fallback is not None else [], float)
        fb = fb[np.isfinite(fb) & (fb > 0)]
        if fb.size < 3:
            return {"95": float("nan"), "99": float("nan"), "n": int(v.size), "median": float("nan"), "spread": float("nan"), "source": "none",
                    "words": "There are too few samples to set a limit for the reconstruction error."}
        v, source, widen = fb, "training", 2.0
    lv = np.log(v)
    med = float(np.median(lv))
    spread = max(1.4826 * float(np.median(np.abs(lv - med))), MIN_LOG_SPREAD)
    out = {lev: float(widen * np.exp(med + Z[lev] * spread)) for lev in LEVELS}
    out.update({"n": int(v.size), "median": float(np.exp(med)), "spread": spread, "source": source})
    if source == "held-out":
        out["words"] = (f"The limit is the 99 % point of the reconstruction errors of {v.size} samples the network was not trained on (typical error {np.exp(med):.3g}); "
                        "about 1 ordinary new sample in 100 is above it.")
    else:
        out["words"] = ("Too few samples were held out to set the limit from them, so it is taken from the errors of the training samples and doubled; "
                        "it is a rough guide until the model has been applied to more samples.")
    return out


# ----------------------------------------------------------------------------- how typical is a code?
def code_stats(codes, codes_sd=None) -> dict:
    """Mean and covariance of the training codes: the distribution that new codes are compared with.  For a variational autoencoder the covariance is that of the *aggregate posterior*
    (covariance of the posterior means plus the mean posterior variance)."""
    Zc = np.asarray(codes, float)
    n, d = Zc.shape
    mean = Zc.mean(axis=0)
    cov = np.atleast_2d(np.cov(Zc, rowvar=False)) if n > 1 else np.zeros((d, d))
    if codes_sd is not None:
        cov = cov + np.diag((np.asarray(codes_sd, float) ** 2).mean(axis=0))
    cov = np.atleast_2d(cov)
    ridge = 1e-8 * max(float(np.trace(cov)) / d, 1e-300)
    return {"mean": mean, "cov": cov + ridge * np.eye(d), "n": int(n), "sd": np.sqrt(np.maximum(np.diag(cov), 0.0))}


def mahalanobis2(codes, stats: dict) -> np.ndarray:
    """Squared Mahalanobis distance of codes ``(n, d)`` from the training codes."""
    D = np.asarray(codes, float) - np.asarray(stats["mean"], float)
    sol = np.linalg.solve(np.asarray(stats["cov"], float), D.T).T
    return np.einsum("ij,ij->i", D, sol)


def typicality(codes, stats: dict) -> np.ndarray:
    """p-value of how typical a code is among the training codes (1 = right in the middle, near 0 = far out).

    For a sample that did not build the distribution, ``n / (n + 1) D² (n - d) / (d (n - 1))`` follows an F distribution with ``d`` and ``n - d`` degrees of freedom when the codes are normal
    (Hotelling's T² for a new observation); with too few training samples the chi-square distribution with ``d`` degrees of freedom is used."""
    D2 = mahalanobis2(codes, stats)
    d, n = int(np.asarray(stats["mean"]).size), int(stats.get("n", 0))
    if n > d + 2:
        f = D2 * n / (n + 1.0) * (n - d) / (d * (n - 1.0))
        return np.where(np.isfinite(f), sps.f.sf(f, d, n - d), np.nan)
    return np.where(np.isfinite(D2), sps.chi2.sf(D2, d), np.nan)


def typicality_cut(p: float, stats: dict) -> float:
    """The squared Mahalanobis distance above which the typicality is below ``p`` (for the plot of the limit)."""
    d, n = int(np.asarray(stats["mean"]).size), int(stats.get("n", 0))
    if p <= 0:
        return float("nan")
    if n > d + 2:
        f = float(sps.f.isf(p, d, n - d))
        return f * (n + 1.0) / n * d * (n - 1.0) / (n - d)
    return float(sps.chi2.isf(p, d))


# ----------------------------------------------------------------------------- the variational terms
def kl_normal(mu, logvar) -> np.ndarray:
    """KL(q(z|x) || N(0, I)) of every sample, in nats."""
    mu, lv = np.asarray(mu, float), np.asarray(logvar, float)
    return 0.5 * (mu ** 2 + np.exp(lv) - 1.0 - lv).sum(axis=1)


def mixture_responsibilities(mu, logvar, pi, pmu, plogvar) -> np.ndarray:
    """Share of every sample in every component of a Gaussian-mixture prior ``(n, K)``, from the posterior mean ``mu`` and variance ``exp(logvar)`` (the variances add)."""
    mu, lv = np.asarray(mu, float), np.asarray(logvar, float)
    pi, pmu, plv = np.asarray(pi, float), np.asarray(pmu, float), np.asarray(plogvar, float)
    var = np.exp(lv)[:, None, :] + np.exp(plv)[None, :, :]
    ll = -0.5 * (((mu[:, None, :] - pmu[None, :, :]) ** 2) / var + np.log(var)).sum(axis=2) + np.log(np.maximum(pi, 1e-300))[None, :]
    ll -= ll.max(axis=1, keepdims=True)
    r = np.exp(ll)
    return r / r.sum(axis=1, keepdims=True)


def kl_mixture(mu, logvar, pi, pmu, plogvar) -> np.ndarray:
    """KL of every sample to the closest component of a Gaussian-mixture prior, plus ``-log`` of that component's weight (an upper bound of the KL to the mixture)."""
    mu, lv = np.asarray(mu, float), np.asarray(logvar, float)
    pi, pmu, plv = np.asarray(pi, float), np.asarray(pmu, float), np.asarray(plogvar, float)
    out = np.full(mu.shape[0], np.inf)
    for k in range(pi.size):
        kl = 0.5 * ((np.exp(lv) + (mu - pmu[k]) ** 2) / np.exp(plv[k]) - 1.0 + plv[k] - lv).sum(axis=1) - np.log(max(float(pi[k]), 1e-300))
        out = np.minimum(out, kl)
    return out


def elbo(error_sum, kl, sigma: float, n_eff: float = 1.0) -> np.ndarray:
    """Evidence lower bound of every sample up to a constant that does not depend on the sample: ``-(N / 2 (error / sigma^2 + log sigma^2) + KL)``, the training loss of the variational
    autoencoder with a Gaussian likelihood (``error`` the reconstruction error summed over the blocks, ``sigma`` the noise SD of the decoder, ``N`` the effective number of coordinates).
    Higher = the network explains the sample better."""
    sig2 = float(sigma) ** 2
    return -(0.5 * float(n_eff) * (np.asarray(error_sum, float) / sig2 + np.log(sig2)) + np.asarray(kl, float))
