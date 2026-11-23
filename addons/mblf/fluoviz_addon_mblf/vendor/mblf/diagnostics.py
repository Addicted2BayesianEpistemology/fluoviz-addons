"""MCMC diagnostics (rank-normalised split R-hat, bulk/tail ESS, E-BFMI; Vehtari et al. 2021, as in the R package
`posterior`) and PSIS-LOO with the one-standard-error rule for the number of factors."""

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm, rankdata


# ---------------------------------------------------------------------------------------------------- R-hat and ESS
def _split(x):
    """(chain, draw, ...) -> (2 chain, draw // 2, ...), dropping the middle draw of odd chains."""
    n = x.shape[1] // 2
    return np.concatenate([x[:, :n], x[:, x.shape[1] - n:]], axis=0)


def _z_scale(x):
    """Rank-normalise over all chains and draws (average ranks for ties)."""
    shape = x.shape
    flat = x.reshape(shape[0] * shape[1], -1)
    r = rankdata(flat, axis=0)
    return norm.ppf((r - 0.375) / (flat.shape[0] + 0.25)).reshape(shape)


def _rhat_basic(x):
    n = x.shape[1]
    B = n * np.var(x.mean(axis=1), axis=0, ddof=1)
    W = np.mean(np.var(x, axis=1, ddof=1), axis=0)
    return np.sqrt((B / W + n - 1) / n)


def _ess_basic(x):
    """ESS of every quantity of x (chain, draw, Q) with Geyer's initial monotone sequence."""
    m, n, Q = x.shape
    xc = x - x.mean(axis=1, keepdims=True)
    size = 2 ** int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(xc, n=size, axis=1)
    acov = np.fft.irfft(f * np.conj(f), n=size, axis=1)[:, :n] / n          # biased autocovariance (chain, lag, Q)
    mean_var = acov[:, 0].mean(axis=0) * n / (n - 1)
    var_plus = mean_var * (n - 1) / n + (np.var(x.mean(axis=1), axis=0, ddof=1) if m > 1 else 0.0)
    rho = 1.0 - (mean_var[None] - acov.mean(axis=0)) / var_plus[None]       # (lag, Q)
    out = np.full(Q, np.nan)
    for q in range(Q):
        if not np.isfinite(var_plus[q]) or var_plus[q] <= 0:
            continue
        r = np.zeros(n)
        t, r[0], even, odd = 0, 1.0, 1.0, rho[1, q]
        r[1] = odd
        while t < n - 5 and np.isfinite(even + odd) and even + odd > 0:
            t += 2
            even, odd = rho[t, q], rho[t + 1, q]
            if even + odd >= 0:
                r[t], r[t + 1] = even, odd
        max_t = t
        if even > 0:
            r[max_t] = even
        t = 0
        while t <= max_t - 4:
            t += 2
            if r[t] + r[t + 1] > r[t - 2] + r[t - 1]:
                r[t] = r[t + 1] = (r[t - 2] + r[t - 1]) / 2
        tau = -1 + 2 * r[:max_t].sum() + r[max_t]
        out[q] = m * n / max(tau, 1 / np.log10(m * n))
    return out


def _as_3d(x):
    x = np.asarray(x, dtype=float)
    return x.reshape(x.shape[0], x.shape[1], -1)


def _ok(x):
    """Quantities with finite, non-constant draws."""
    return np.all(np.isfinite(x), axis=(0, 1)) & (np.ptp(x.reshape(-1, x.shape[2]), axis=0) > 0)


def rhat(x):
    """Rank-normalised split R-hat of every quantity of x (chain, draw, ...): max of bulk and folded (tail) R-hat."""
    x = _as_3d(x)
    out = np.full(x.shape[2], np.nan)
    ok = _ok(x)
    if ok.any():
        y = x[:, :, ok]
        folded = np.abs(y - np.median(y.reshape(-1, y.shape[2]), axis=0))
        out[ok] = np.maximum(_rhat_basic(_z_scale(_split(y))), _rhat_basic(_z_scale(_split(folded))))
    return out


def ess_bulk(x):
    """Bulk effective sample size of every quantity of x (chain, draw, ...)."""
    x = _as_3d(x)
    out = np.full(x.shape[2], np.nan)
    ok = _ok(x)
    if ok.any():
        out[ok] = _ess_basic(_z_scale(_split(x[:, :, ok])))
    return out


def ess_tail(x):
    """Tail effective sample size: the smaller ESS of the indicators of the 5% and 95% quantiles."""
    x = _as_3d(x)
    out = np.full(x.shape[2], np.nan)
    ok = _ok(x)
    if ok.any():
        y = x[:, :, ok]
        flat = y.reshape(-1, y.shape[2])
        e = [_ess_basic(_split((y <= np.quantile(flat, prob, axis=0)).astype(float))) for prob in (0.05, 0.95)]
        out[ok] = np.minimum(*e)
    return out


def ebfmi(energy):
    """E-BFMI of every chain: sum of squared differences of the energy over its sum of squared deviations."""
    energy = np.asarray(energy, dtype=float)
    return np.sum(np.diff(energy, axis=1) ** 2, axis=1) / np.sum((energy - energy.mean(axis=1, keepdims=True)) ** 2, axis=1)


# ------------------------------------------------------------------------------------------------------- PSIS-LOO
def _gpdfit(x):
    """Generalised Pareto fit of the sorted exceedances x (Zhang & Stephens 2009, with the weakly informative prior
    of the R package loo). Returns (k, sigma)."""
    n = x.size
    m = 30 + int(np.floor(np.sqrt(n)))
    jj = np.arange(1, m + 1)
    xstar = x[int(np.floor(n / 4 + 0.5)) - 1]
    theta = 1 / x[-1] + (1 - np.sqrt(m / (jj - 0.5))) / 3 / xstar
    k = np.array([np.mean(np.log1p(-t * x)) for t in theta])
    l_theta = n * (np.log(-theta / k) - k - 1)
    w = np.exp(l_theta - logsumexp(l_theta))
    theta_hat = np.sum(theta * w)
    k = np.mean(np.log1p(-theta_hat * x))
    sigma = -k / theta_hat
    k = (k * n + 5.0) / (n + 10.0)
    return (np.inf if np.isnan(k) else k), sigma


def psis(log_ratios, r_eff=None):
    """Pareto-smoothed importance sampling of every column of log_ratios (draws, n), as in the R package loo.

    ``r_eff`` (scalar or (n,)) is the relative efficiency of the draws (default 1); the tail has
    ceil(min(0.2 S, 3 sqrt(S / r_eff))) draws. Returns normalised log weights (draws, n) and the Pareto k of every
    column.
    """
    log_ratios = np.asarray(log_ratios, dtype=float)
    S, n = log_ratios.shape
    r_eff = np.broadcast_to(1.0 if r_eff is None else np.asarray(r_eff, dtype=float), (n,))
    lw = log_ratios - log_ratios.max(axis=0)
    khat = np.full(n, np.inf)
    for i in range(n):
        tail = int(np.ceil(min(0.2 * S, 3 * np.sqrt(S / r_eff[i]))))
        if tail < 5:
            continue
        order = np.argsort(lw[:, i], kind="stable")
        ids = order[S - tail:]
        x = lw[ids, i]
        if abs(x.max() - x.min()) < np.finfo(float).eps / 100:
            continue
        cutoff = lw[order[S - tail - 1], i]
        k, sigma = _gpdfit(np.exp(x) - np.exp(cutoff))
        khat[i] = k
        if np.isfinite(k):
            p = (np.arange(1, tail + 1) - 0.5) / tail
            lw[ids, i] = np.log(sigma * np.expm1(-k * np.log1p(-p)) / k + np.exp(cutoff))
    lw = np.minimum(lw, 0.0)
    return lw - logsumexp(lw, axis=0), khat


def relative_eff(log_lik):
    """Relative efficiency ESS / S of exp(log_lik) for every observation of an array (chain, draw, n): the ESS of
    the split chains, as posterior::ess_mean, which the R package loo (>= 2.8) uses."""
    x = np.exp(log_lik - log_lik.max(axis=(0, 1)))
    return _ess_basic(_split(x)) / (x.shape[0] * x.shape[1])


def loo(log_lik):
    """PSIS-LOO of pointwise log-likelihoods, an array (chain, draw, n) or (draws, n) (then r_eff = 1).

    Returns a dict with ``elpd`` (total), ``se``, ``pointwise`` (n,), ``pareto_k`` (n,) and ``p_loo``.
    """
    log_lik = np.asarray(log_lik, dtype=float)
    r_eff = None
    if log_lik.ndim == 3:
        r_eff = np.clip(relative_eff(log_lik), 1e-12, None)
        log_lik = log_lik.reshape(-1, log_lik.shape[2])
    lw, k = psis(-log_lik, r_eff)
    pointwise = logsumexp(lw + log_lik, axis=0)
    lpd = logsumexp(log_lik, axis=0) - np.log(log_lik.shape[0])
    n = pointwise.size
    return {"elpd": float(pointwise.sum()), "se": float(np.sqrt(n * np.var(pointwise, ddof=1))), "pointwise": pointwise,
            "pareto_k": k, "p_loo": float(np.sum(lpd - pointwise))}


def log_mean_exp(x, axis=0):
    """log(mean(exp(x))) along an axis, computed stably (the log predictive density of held-out data)."""
    x = np.asarray(x, dtype=float)
    return logsumexp(x, axis=axis) - np.log(x.shape[axis])


def one_se_rule(pointwise):
    """Choose the number of factors by the one-standard-error rule (paper, eq. factor_selection_rule).

    Parameters
    ----------
    pointwise : dict {K: array (n,)}
        Pointwise expected log predictive densities (PSIS-LOO or held-out) of every candidate, same subjects.

    Returns
    -------
    best : the K with the highest total score; chosen : the smallest K whose loss against ``best`` is within one
    paired standard error; table : list of dicts with K, elpd, se, elpd_diff, se_diff.
    """
    Ks = sorted(pointwise)
    elpd = {K: float(np.sum(pointwise[K])) for K in Ks}
    best = max(Ks, key=lambda K: elpd[K])
    table = []
    for K in Ks:
        diff = np.asarray(pointwise[K]) - np.asarray(pointwise[best])
        n = diff.size
        table.append({"K": K, "elpd": elpd[K], "se": float(np.sqrt(n * np.var(pointwise[K], ddof=1))),
                      "elpd_diff": float(diff.sum()), "se_diff": float(np.sqrt(n * np.var(diff, ddof=1)))})
    chosen = min(row["K"] for row in table if -row["elpd_diff"] <= row["se_diff"])
    return best, chosen, table
