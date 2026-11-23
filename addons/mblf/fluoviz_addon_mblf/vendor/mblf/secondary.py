"""Diagnostics of the appendix studies: effective dimension of the shared variation (sensitivity to the basis, the
cap K and the factor selection), projection residuals against projected Gaussian noise, scores of withheld entries,
and the check of the Stan likelihood against the observation-level Gaussian likelihood.

Tables are dicts of equal-length columns (``pandas.DataFrame(table)`` gives a data frame); NaN marks a missing value.
"""

import math

import numpy as np


def shared_spectrum(lam, bases, sigma2, psi2, M=None, mode_weights="standardized", point_weights=None):
    """Eigenvalues of the weighted shared second moment of the loading functions, for every draw.

    With Omega_d the point weights of mode d (default: the grid average 1/M_d), G_d = B_d' Omega_d B_d and the
    second moment M of the scores (default I), mode d has shared variance tr(M Lambda_d' G_d Lambda_d) and total
    variance shared + sum_j sigma2_dj G_d[j, j] + psi2_d sum(Omega_d). The eigenvalues are those of
    Lm' (sum_d c_d Lambda_d' G_d Lambda_d) Lm (M = Lm Lm'), with c_d = 1 / total_d ("standardized") or 1 ("none").

    Parameters: lam list of (S, p_d, K); bases list of (M_d, p_d); sigma2 list of (S, p_d); psi2 (S, D).
    Returns dict with eigenvalues (S, K, descending), total (S,), mode_shared (S, D), mode_total (S, D).
    """
    S, _, K = lam[0].shape
    D = len(lam)
    w = [np.full(B.shape[0], 1.0 / B.shape[0]) if point_weights is None else np.asarray(point_weights[d], float)
         for d, B in enumerate(bases)]
    grams = [B.T @ (wd[:, None] * B) for B, wd in zip(bases, w)]
    M = np.broadcast_to(np.eye(K) if M is None else M, (S, K, K))
    blocks = np.stack([np.einsum("spk,pq,sql->skl", lam[d], grams[d], lam[d]) for d in range(D)])
    shared = np.einsum("skl,dslk->sd", M, blocks)
    total = shared + np.stack([sigma2[d] @ np.diag(grams[d]) + psi2[:, d] * w[d].sum() for d in range(D)], axis=1)
    c = 1.0 / np.where(total > 0, total, 1.0) if mode_weights == "standardized" else np.ones((S, D))
    A = np.einsum("sd,dskl->skl", c, blocks)
    Lm = np.linalg.cholesky(M)
    A = np.swapaxes(Lm, 1, 2) @ A @ Lm
    ev = np.clip(np.linalg.eigvalsh(0.5 * (A + np.swapaxes(A, 1, 2)))[:, ::-1], 0.0, None)
    return {"eigenvalues": ev, "total": np.sum(c * total, axis=1), "mode_shared": shared, "mode_total": total}


def effective_ranks(eigenvalues, energy):
    """Smallest r whose leading eigenvalues reach each energy fraction of their sum (0 when all are 0)."""
    ev = np.asarray(eigenvalues, dtype=float)
    thr = np.atleast_1d(np.asarray(energy, dtype=float))
    tot = ev.sum(axis=-1, keepdims=True)
    cum = np.cumsum(ev, axis=-1) / np.where(tot > 0, tot, 1.0)
    r = np.argmax(cum[..., None, :] >= thr[:, None] - 1e-12, axis=-1) + 1
    return np.where(tot > 0, r, 0)


def effective_dimension(fit, energy=(0.9, 0.95, 0.99), covariates=False, mode_weights="standardized",
                        point_weights=None, level=0.9):
    """Posterior summary of the effective dimension of a `MvtBLF` fit.

    For every draw, the number of eigen-directions of the (weighted, mode-standardised) shared covariance of the
    loading functions needed to reach each energy threshold. ``covariates=True`` uses the second moment
    I + beta S_x beta' of the scores over the covariates of the fit instead of I.

    Returns a dict with ``table`` (per threshold: mean, median, mode, lower, upper of the rank, p_at_cap = P(rank = K)),
    ``spectrum`` (per direction: mean share of the shared variation, its interval, mean fraction of the total
    variance, mean eigenvalue), ``shared_fraction`` (per mode: mean shared fraction of the variance and interval) and
    the draws (``ranks`` (S, n_thresholds), ``share`` (S, K)).
    """
    K, D = fit.K, fit.D
    lam = [fit.loadings(d).reshape(-1, fit.p[d - 1], K) for d in range(1, D + 1)]
    sig = [fit.coefficient_variances(d).reshape(-1, fit.p[d - 1]) for d in range(1, D + 1)]
    psi2 = fit.draws["psi2"].reshape(-1, D)
    M = None
    if covariates and fit.X.shape[1]:
        beta = fit.draws["regr_coeffs"].reshape(-1, K, fit.X.shape[1])
        Sx = fit.X.T @ fit.X / fit.X.shape[0]
        M = np.eye(K) + np.einsum("skl,lm,sjm->skj", beta, Sx, beta)
    spec = shared_spectrum(lam, [m.basis for m in fit.modes], sig, psi2, M, mode_weights, point_weights)
    ev = spec["eigenvalues"]
    ranks = effective_ranks(ev, energy)
    share = ev / np.where(ev.sum(axis=1, keepdims=True) > 0, ev.sum(axis=1, keepdims=True), 1.0)
    q = (1 - level) / 2

    def interval(v):  # NumPy "lower" / "higher" quantiles
        return np.quantile(v, q, axis=0, method="lower"), np.quantile(v, 1 - q, axis=0, method="higher")

    lo, hi = interval(ranks)
    table = []
    for i, e in enumerate(np.atleast_1d(energy)):
        r = ranks[:, i]
        vals, counts = np.unique(r, return_counts=True)
        table.append({"energy": float(e), "mean": float(r.mean()), "median": float(np.median(r)),
                      "mode": int(vals[np.argmax(counts)]), "lower": float(lo[i]), "upper": float(hi[i]),
                      "p_at_cap": float(np.mean(r == K)), "n_components": K})
    slo, shi = interval(share)
    spectrum = [{"direction": k + 1, "share_mean": float(share[:, k].mean()), "share_lower": float(slo[k]),
                 "share_upper": float(shi[k]), "variance_fraction_mean": float(np.mean(ev[:, k] / spec["total"])),
                 "eigenvalue_mean": float(ev[:, k].mean())} for k in range(K)]
    frac = spec["mode_shared"] / spec["mode_total"]
    flo, fhi = interval(frac)
    shared_fraction = [{"mode": d + 1, "shared_fraction_mean": float(frac[:, d].mean()), "lower": float(flo[d]),
                        "upper": float(fhi[d])} for d in range(D)]
    return {"table": table, "spectrum": spectrum, "shared_fraction": shared_fraction, "ranks": ranks, "share": share}


# ------------------------------------------------------------------------------------------- projection residuals
LEVERAGE_TOL = 1e-8


def _patterns(mask):
    """Groups of identical columns of a logical matrix, in the order of first appearance: [(rows, columns)]."""
    groups = {}
    for n in range(mask.shape[1]):
        groups.setdefault(mask[:, n].tobytes(), []).append(n)
    return [(np.flatnonzero(mask[:, cols[0]]), np.array(cols)) for cols in groups.values()]


def projection_diagnostics(fit, mode, rank_tol=None):
    """Projection residuals of every subject of mode ``mode`` (1-based) of a `MvtBLF` fit.

    For subject n with observed basis rows A_n (M_dn x p_d): an orthonormal basis Q_n of the column space of A_n from
    its SVD (singular values above ``rank_tol`` times the largest; default max(M_dn, p_d) times the machine epsilon),
    the projector P_n = Q_n Q_n', the residual R_n = (I - P_n) y_n, its degrees of freedom df_n = M_dn - rank(A_n), the
    leverages h = diag(P_n) and rss_n = ||R_n||^2. Under the model R_n | psi2_d ~ N(0, psi2_d (I - P_n)). Subjects with
    df_n = 0 are not assessable (rss NaN).

    Returns a dict with ``samples`` (columns sample, n_obs, rank, df, rss, rss_per_df, status, assessable), ``groups``
    (one per observation pattern: rows (0-based grid rows), coords, Q, h, members (0-based subjects), R, assessable,
    defined = 1 - h > 1e-8), ``precision`` (values 1/psi2 of the posterior draws, source) and ``mode``.
    """
    m = fit.modes[mode - 1]
    B, Y = m.basis, m.data
    N = Y.shape[1]
    n_obs, rank, df = (np.zeros(N, int) for _ in range(3))
    rss = np.full(N, np.nan)
    groups = []
    for rows, members in _patterns(~np.isnan(Y)):
        A = B[rows]
        U, s, _ = np.linalg.svd(A, full_matrices=False) if rows.size else (np.zeros((0, 0)), np.zeros(0), None)
        tol = max(A.shape) * np.finfo(float).eps if rank_tol is None else rank_tol
        r = int(np.sum(s > tol * s[0])) if s.size and s[0] > 0 else 0
        Q = U[:, :r]
        h = np.sum(Q ** 2, axis=1)
        y = Y[np.ix_(rows, members)]
        R = y - Q @ (Q.T @ y)
        ok = rows.size > 0 and rows.size - r > 0
        if ok:
            rss[members] = np.sum(R ** 2, axis=0)
        n_obs[members], rank[members], df[members] = rows.size, r, rows.size - r
        groups.append(dict(rows=rows, coords=m.coords[rows], Q=Q, h=h, members=members, R=R, assessable=ok,
                           defined=(1 - h) > LEVERAGE_TOL))
    status = np.where(n_obs == 0, "no observations", np.where(df == 0, "not assessable: no residual dimension", "assessable"))
    psi2 = fit.draws["psi2"].reshape(-1, fit.D)[:, mode - 1]
    samples = dict(sample=np.arange(1, N + 1), n_obs=n_obs, rank=rank, df=df, rss=rss,
                   rss_per_df=np.where(df > 0, rss / np.maximum(df, 1), np.nan), status=status,
                   assessable=status == "assessable")
    precision = dict(values=1 / psi2, source=f"posterior draws of psi2 for mode {mode} (precision psi_d = 1 / psi2_d; "
                                             f"{psi2.size} draws)")
    return dict(samples=samples, groups=groups, precision=precision, mode=mode)


def _compare(obs, ref, band):
    """Observed values against their simulated references (rows): mean, sd, central band (type-7 quantiles), mid-rank
    PIT (less + equal/2 + 1/2) / (S + 1), two-sided Monte Carlo p-value and outside the band; ties within 1e-9 times
    the largest absolute value; a degenerate reference (all simulations tied) gives NaN PIT, p-value and outside."""
    n, S = ref.shape
    out = {k: np.full(n, np.nan) for k in ("mean", "lower", "upper", "pit", "p_value", "outside", "sd")}
    probs = [(1 - band) / 2, 1 - (1 - band) / 2]
    for i in range(n):
        r = ref[i]
        if np.isnan(obs[i]) or np.isnan(r).any():
            continue
        tol = 1e-9 * max(abs(obs[i]), np.abs(r).max())
        q = np.quantile(r, probs)
        out["mean"][i], out["lower"][i], out["upper"][i], out["sd"][i] = r.mean(), q[0], q[1], r.std(ddof=1)
        if r.max() - r.min() <= tol:
            continue
        less, equal = np.sum(r < obs[i] - tol), np.sum(np.abs(r - obs[i]) <= tol)
        greater = S - less - equal
        out["pit"][i] = (less + 0.5 * equal + 0.5) / (S + 1)
        out["p_value"][i] = min(1.0, 2 * min(less + equal + 1, greater + equal + 1) / (S + 1))
        out["outside"][i] = float(obs[i] < q[0] - tol or obs[i] > q[1] + tol)
    return out


def _summary_row(table, summary, statistic, observed, expected, sims, band, n):
    """Append one aggregate row: the observed statistic, its expectation under projected noise, and the central band
    and two-sided Monte Carlo p-value of its simulated values (the joint simulations: one precision per simulation)."""
    sims = sims[~np.isnan(sims)]
    lo = hi = p = np.nan
    if not np.isnan(observed) and sims.size >= 2:
        tol = 1e-9 * max(abs(observed), np.abs(sims).max())
        lo, hi = np.quantile(sims, [(1 - band) / 2, 1 - (1 - band) / 2])
        less, equal = np.sum(sims < observed - tol), np.sum(np.abs(sims - observed) <= tol)
        greater = sims.size - less - equal
        p = min(1.0, 2 * min(less + equal + 1, greater + equal + 1) / (sims.size + 1))
    for k, v in dict(summary=summary, statistic=statistic, observed=observed, expected=expected, ref_lower=lo,
                     ref_upper=hi, p_value=p, n=int(n)).items():
        table[k].append(v)


def _share_outside(ref, cmp):
    """Per simulation, the share of rows (with a PIT) outside their band: the joint reference of the observed share."""
    ok = ~np.isnan(cmp["pit"])
    if not ok.any():
        return np.full(ref.shape[1], np.nan)
    r = ref[ok]
    return np.mean((r < cmp["lower"][ok, None]) | (r > cmp["upper"][ok, None]), axis=0)


def projection_reference(diagnostics, n_sim=500, seed=None, band=0.9, precision=None):
    """Projection residuals of `projection_diagnostics` against Gaussian noise passed through the same projections.

    Simulation s draws e ~ N(0, v_s I) for every assessable subject and computes the same summaries from
    (I - P_n) e, with one noise variance v_s = 1/psi_s shared by all subjects: psi_s are the n_sim precision values
    (``precision``: one value or draws; default the posterior draws 1/psi2 of the diagnostics), thinned evenly
    (round(seq(1, S, length.out = n_sim))) or recycled in order. The summaries:

    * residual energy rss_n per subject (energy_* columns; energy_std = (rss_n - mean) / sd of its reference);
    * the variance profile: per grid location the mean of R_n,l^2 over the subjects observing it (mean_sq), with the
      closed form expected = mean(v_s) * mean(1 - h_nl);
    * the lag-1 autocorrelation sum z_i z_i+1 / sum z_i^2 of the standardised residuals z = R / sqrt(1 - h) along the
      ordered locations (first coordinate) of every subject with at least 3 of them.

    PITs are mid-ranks among the simulations (ties within 1e-9 relative); ``outside`` is 1 outside the central
    ``band`` of the reference. ``summary`` aggregates over subjects / locations (share outside, mean standardised
    energy, mean ratio of the profile to its closed form, mean lag-1), each against its joint simulated reference.

    Returns a dict with ``summary`` (summary, statistic, observed, expected, ref_lower, ref_upper, p_value, n),
    ``locations`` (location (1-based grid row), x, n_samples, n_obs, mean_sq, expected, ref_mean, ref_lower, ref_upper,
    pit, p_value, outside), ``samples`` (sample, n_obs, rank, df, status, assessable, rss, energy_mean, energy_lower,
    energy_upper, energy_pit, energy_p_value, energy_outside, energy_std, lag1, lag1_mean, lag1_lower, lag1_upper,
    lag1_pit, lag1_p_value, lag1_outside) and ``precision`` (values used, source).
    """
    dg = diagnostics
    if precision is None:
        values, source = np.asarray(dg["precision"]["values"], float), dg["precision"]["source"]
    else:
        values = np.atleast_1d(np.asarray(precision, float))
        source = (f"supplied precision draws ({values.size} values; precision psi_d = 1 / psi2_d)" if values.size > 1
                  else "supplied fixed precision (psi_d = 1 / psi2_d); no uncertainty in the precision")
    S = values.size
    index = np.round(np.linspace(1, S, n_sim)).astype(int) - 1 if S >= n_sim else np.arange(n_sim) % S
    sd = np.sqrt(1 / values[index])
    groups = [g for g in dg["groups"] if g["assessable"] and g["members"].size]
    if not groups:
        raise ValueError("no subject has a residual dimension (df = 0 everywhere): nothing to calibrate")
    N = dg["samples"]["sample"].size
    keys = list(dict.fromkeys(int(r) for g in groups for r in g["rows"]))       # grid rows, first appearance
    pos = {k: i for i, k in enumerate(keys)}
    count, one_minus_h, coords = np.zeros(len(keys)), np.zeros(len(keys)), np.zeros((len(keys), groups[0]["coords"].shape[1]))
    for g in groups:
        g["key"] = np.array([pos[int(r)] for r in g["rows"]])
        g["order"] = [i for i in np.argsort(g["coords"][:, 0], kind="stable") if g["defined"][i]]
        np.add.at(count, g["key"], g["members"].size)
        np.add.at(one_minus_h, g["key"], g["members"].size * (1 - g["h"]))
        coords[g["key"]] = g["coords"]

    def summaries(residuals, n_cols):
        # residuals(g): M_g x (n_cols n_g), column = simulation * n_g + member
        energy, lag1 = np.full((N, n_cols), np.nan), np.full((N, n_cols), np.nan)
        profile = np.zeros((len(keys), n_cols))
        for g in groups:
            R, ng = residuals(g), g["members"].size
            R2 = R ** 2
            energy[g["members"]] = R2.sum(axis=0).reshape(n_cols, ng).T
            np.add.at(profile, g["key"], R2.reshape(R.shape[0], n_cols, ng).sum(axis=2))
            if len(g["order"]) >= 3:
                Z = R[g["order"]] / np.sqrt(1 - g["h"][g["order"]])[:, None]
                num, den = np.sum(Z[:-1] * Z[1:], axis=0), np.sum(Z ** 2, axis=0)
                lag1[g["members"]] = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan).reshape(n_cols, ng).T
        return energy, profile / count[:, None], lag1

    rng = np.random.default_rng(seed)
    obs = summaries(lambda g: g["R"], 1)

    def projected_noise(g):                                                      # (I - Q Q') e, e ~ N(0, v_s I)
        e = rng.standard_normal((n_sim * g["members"].size, g["rows"].size)).T * np.repeat(sd, g["members"].size)[None, :]
        return e - g["Q"] @ (g["Q"].T @ e)

    sims = summaries(projected_noise, n_sim)
    s = dg["samples"]
    samples = {k: s[k] for k in ("sample", "n_obs", "rank", "df", "status", "assessable", "rss")}
    table = {k: [] for k in ("summary", "statistic", "observed", "expected", "ref_lower", "ref_upper", "p_value", "n")}
    pct = f"{100 * band:g}%"
    # residual energy
    cmp = _compare(obs[0][:, 0], sims[0], band)
    samples.update({f"energy_{k}": cmp[k] for k in ("mean", "lower", "upper", "pit", "p_value", "outside")})
    samples["energy_std"] = (obs[0][:, 0] - cmp["mean"]) / cmp["sd"]
    ok = ~np.isnan(cmp["pit"])
    share = np.mean(cmp["outside"][ok]) if ok.any() else np.nan
    _summary_row(table, "energy", f"share of samples outside their central {pct} reference band", share, 1 - band,
                 _share_outside(sims[0], cmp), band, ok.sum())
    std_sims = np.mean((sims[0][ok] - cmp["mean"][ok, None]) / cmp["sd"][ok, None], axis=0)
    _summary_row(table, "energy", "mean standardised residual energy",
                 np.mean(samples["energy_std"][ok]) if ok.any() else np.nan, 0.0, std_sims, band, ok.sum())
    # variance profile
    cmp = _compare(obs[1][:, 0], sims[1], band)
    expected = np.mean(sd ** 2) * one_minus_h / count
    locations = dict(location=np.array(keys) + 1, x=coords[:, 0])
    locations.update({f"x{j + 1}": coords[:, j] for j in range(1, coords.shape[1])})
    locations.update(n_samples=count.astype(int), n_obs=count.astype(int), mean_sq=obs[1][:, 0], expected=expected,
                     ref_mean=cmp["mean"], ref_lower=cmp["lower"], ref_upper=cmp["upper"], pit=cmp["pit"],
                     p_value=cmp["p_value"], outside=cmp["outside"])
    ok = ~np.isnan(cmp["pit"])
    _summary_row(table, "variance profile", f"share of locations outside their central {pct} reference band",
                 np.mean(cmp["outside"][ok]) if ok.any() else np.nan, 1 - band, _share_outside(sims[1], cmp), band, ok.sum())
    _summary_row(table, "variance profile", "mean ratio of the location-wise mean square to its closed-form expectation",
                 np.mean(obs[1][ok, 0] / expected[ok]), 1.0, np.mean(sims[1][ok] / expected[ok, None], axis=0), band, ok.sum())
    # lag-1 autocorrelation
    cmp = _compare(obs[2][:, 0], sims[2], band)
    samples["lag1"] = obs[2][:, 0]
    samples.update({f"lag1_{k}": cmp[k] for k in ("mean", "lower", "upper", "pit", "p_value", "outside")})
    ok = ~np.isnan(cmp["pit"])
    _summary_row(table, "dependence", f"share of samples whose lag-1 autocorrelation is outside its central {pct} "
                 "reference band", np.mean(cmp["outside"][ok]) if ok.any() else np.nan, 1 - band,
                 _share_outside(sims[2], cmp), band, ok.sum())
    _summary_row(table, "dependence", "mean lag-1 autocorrelation of the standardised residuals",
                 np.mean(samples["lag1"][ok]) if ok.any() else np.nan, np.mean(cmp["mean"][ok]) if ok.any() else np.nan,
                 np.mean(sims[2][ok], axis=0), band, ok.sum())
    summary = {k: np.array(v) for k, v in table.items()}
    return dict(summary=summary, locations=locations, samples=samples,
                precision=dict(values=values[index], source=source))


# ---------------------------------------------------------------------------------------------- withheld entries
def pretty_breaks(hi, n=4):
    """R's pretty(c(0, hi), n) (R_pretty with min.n = n %/% 3, shrink.sml = 0.75, high.u.bias = 1.5,
    u5.bias = 2.75): equally spaced 'round' break points from 0 covering hi."""
    h, h5, min_n, eps = 1.5, 0.5 + 1.5 * 1.5, n // 3, 1e-10
    lo, up = 0.0, float(hi)
    cell = max(abs(lo), abs(up))
    U = (1 + (1 / (1 + h) if h5 >= 1.5 * h + 0.5 else 1.5 / (1 + h5))) * max(1, n) * np.finfo(float).eps
    if (up - lo == 0 and up == 0) or up - lo < cell * U * 3:
        cell = 1.0 if up == 0 else cell
        cell = (9 + cell / 10 if cell > 10 else cell) * 0.75 / (min_n if min_n > 1 else 1)
    else:
        cell = (up - lo) / n if n > 1 else up - lo
    base = 10.0 ** math.floor(math.log10(cell))
    unit = base
    if 2 * base - cell < h * (cell - unit):
        unit = 2 * base
        if 5 * base - cell < h5 * (cell - unit):
            unit = 5 * base
            if 10 * base - cell < h * (cell - unit):
                unit = 10 * base
    ns, nu = math.floor(lo / unit + eps), math.ceil(up / unit - eps)
    while ns * unit > lo + eps * unit:
        ns -= 1
    while nu * unit < up - eps * unit:
        nu += 1
    k = int(0.5 + nu - ns)
    if k < min_n:
        k = min_n - k
        nu, ns = (nu + k // 2, ns - k // 2 - k % 2) if ns >= 0 else (nu + k // 2 + k % 2, ns - k // 2)
        k = min_n
    out = np.linspace(ns * unit, nu * unit, k + 1)
    out[np.abs(out) < 1e-14 * (nu - ns) * unit / k] = 0.0
    return out


def _r_format(x):
    """A number as R's format() writes it (7 significant digits, fixed unless scientific notation is shorter)."""
    if x == 0:
        return "0"
    t = float(f"{x:.7g}")
    k = next(k for k in range(1, 8) if float(f"{t:.{k - 1}e}") == t)
    sci = f"{t:.{k - 1}e}"
    fixed = f"{t:.{max(0, k - 1 - int(sci.split('e')[1]))}f}"
    return fixed if len(fixed) <= len(sci) else sci


def heldout_scores(fit, truth, withheld, level=0.8, seed=None):
    """Scores of the posterior reconstruction of entries withheld from the fit (training subjects).

    Entry (m, n) of mode d, withheld from the data of the fit (NaN there): the latent draws are B_d[m, ] theta_dn^(s)
    of the posterior draws of the coefficients (which condition on the subject's other observations and, through the
    factors, on its other modes); the noisy draws add N(0, psi2_d^(s)) noise (one per draw and entry; seed
    ``seed + 1000 d``). Each target has its posterior mean and central ``level`` interval (type-7 quantiles), scored
    against its truth (squared error, coverage, width). Every entry carries the distance to the nearest observed
    location of the same subject in that mode, binned on ``pretty_breaks`` of the largest distance ((a, b] bins, the
    first closed); a subject without observations gets an infinite distance and the bin "no observed location".

    Parameters: truth dict with "latent" and/or "noisy", each a list (one M_d x N matrix per mode or None);
    withheld list (one M_d x N logical matrix per mode, or None).

    Returns a dict with ``entries`` (mode, subject, location (1-based), x, distance, bin, target, truth, mean, lower,
    upper, width, squared_error, covered), ``by_distance`` (mode, target, bin, n, n_scored, mean_distance, mse,
    coverage, mean_width) and ``overall`` (the same without bin).
    """
    probs = [(1 - level) / 2, (1 + level) / 2]
    C, S_c = fit.draws["psi2"].shape[:2]
    rows = []
    for d in range(1, fit.D + 1):
        w = None if withheld[d - 1] is None else np.asarray(withheld[d - 1], bool)
        if w is None or not w.any():
            continue
        m = fit.modes[d - 1]
        if (w & ~np.isnan(m.data)).any():
            raise ValueError(f"mode {d}: the fit has seen withheld entries (its data are observed there)")
        subj, loc = np.nonzero(w.T)                                         # subject by subject, as R's which()
        # draws ordered draw by draw, the chains alternating
        theta = np.swapaxes(fit.draws["theta"][:, :, fit.rows(d)], 0, 1).reshape(C * S_c, fit.p[d - 1], -1)
        psi2 = np.swapaxes(fit.draws["psi2"], 0, 1).reshape(C * S_c, -1)[:, d - 1]
        latent = np.einsum("ep,spe->se", m.basis[loc], theta[:, :, subj])
        noise = np.random.default_rng(None if seed is None else seed + 1000 * d).standard_normal((loc.size, C * S_c)).T
        observed = ~np.isnan(m.data)
        dist = np.array([np.sqrt(np.min(np.sum((m.coords[observed[:, n]] - m.coords[l]) ** 2, axis=1)))
                         if observed[:, n].any() else np.inf for l, n in zip(loc, subj)])
        for target, draws in (("latent curve", latent), ("noisy observation", latent + noise * np.sqrt(psi2)[:, None])):
            t = (truth or {}).get("latent" if target == "latent curve" else "noisy")
            tv = np.full(loc.size, np.nan) if t is None or t[d - 1] is None else np.asarray(t[d - 1], float)[loc, subj]
            mean = draws.mean(axis=0)
            lo, hi = np.quantile(draws, probs, axis=0)
            rows.append(dict(mode=np.full(loc.size, d), subject=subj + 1, location=loc + 1, x=m.coords[loc, 0],
                             distance=dist, target=np.full(loc.size, target, dtype=object), truth=tv, mean=mean,
                             lower=lo, upper=hi, width=hi - lo, squared_error=(mean - tv) ** 2,
                             covered=np.where(np.isnan(tv), np.nan, (tv >= lo) & (tv <= hi))))
    if not rows:
        raise ValueError("nothing to score: no withheld entry")
    entries = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    finite = np.isfinite(entries["distance"])
    breaks = pretty_breaks(entries["distance"][finite].max()) if finite.any() else np.zeros(0)
    labels = [f"({_r_format(a)}, {_r_format(b)}]" for a, b in zip(breaks[:-1], breaks[1:])]
    if labels:
        labels[0] = "[" + labels[0][1:]
    b = np.clip(np.searchsorted(breaks, entries["distance"], side="left"), 1, max(len(breaks) - 1, 1)) - 1
    entries["bin"] = np.array([labels[i] if f else "no observed location" for i, f in zip(b, finite)], dtype=object)

    def summarise(by):
        key = list(zip(*[entries[k] for k in by]))
        out = {k: [] for k in by + ["n", "n_scored", "mean_distance", "mse", "coverage", "mean_width"]}
        for k in dict.fromkeys(key):
            i = np.array([kk == k for kk in key])
            sc, dd = i & ~np.isnan(entries["truth"]), entries["distance"][i]
            for name, v in zip(by, k):
                out[name].append(v)
            out["n"].append(int(i.sum()))
            out["n_scored"].append(int(sc.sum()))
            out["mean_distance"].append(dd[np.isfinite(dd)].mean() if np.isfinite(dd).any() else np.inf)
            out["mse"].append(entries["squared_error"][sc].mean() if sc.any() else np.nan)
            out["coverage"].append(entries["covered"][sc].mean() if sc.any() else np.nan)
            out["mean_width"].append(entries["width"][i].mean())
        return {k: np.array(v) for k, v in out.items()}

    return dict(entries=entries, by_distance=summarise(["mode", "target", "bin"]), overall=summarise(["mode", "target"]),
                breaks=breaks)


# ------------------------------------------------------------------------------------------------ likelihood check
def check_likelihood(model, modes, X=None, n_points=10, seed=0):
    """Check the Stan program against the observation-level Gaussian likelihood at random parameter values.

    The data enter the program only through inv_gamma(psi2_d | a_d + M_d/2, b_d + RSS_d/2) and the Gram term
    -||G (T - theta)||^2 / (2 psi2_d). With lp(.) the log density of the program (CmdStan log_prob, Jacobian included)
    at the same parameter values, lp_empty that of the same program with no data (sum_M = 0, sum_rss = 0, zero Gram
    values; same non-centring flags), every prior and Jacobian term cancels and

        lp(data) - lp_empty = sum_d sum_n sum_{i observed} log N(y_dni | (B_d theta_dn)_i, psi2_d) - sum_d log C_d,
        log C_d = a_d log b_d - lgamma(a_d) - (M_d/2) log(2 pi) - (a_d + M_d/2) log(b_d + RSS_d/2)
                  + lgamma(a_d + M_d/2),

    a constant of the data (RSS_d: least-squares residuals, computed here with numpy's lstsq; minimum-norm solutions
    for rank-deficient subjects). The log-likelihood is evaluated directly from the raw data in NumPy, theta being
    rebuilt from theta_tilde for the non-centred coefficients. The parameter values are CmdStan's random
    initialisations (uniform(-2, 2) on the unconstrained scale; ``seed``).

    Returns a dict with per point ``stan`` (lp(data) - lp_empty + sum_d log C_d), ``direct`` and ``abs_diff``, the
    ``constant`` sum_d log C_d and ``max_abs_diff``.
    """
    from scipy.special import gammaln

    from .model import stan_model

    data = model.stan_data(modes, X)
    empty = dict(data, sum_M=[0] * data["D"], sum_rss=[0.0] * data["D"], gram_w=np.zeros_like(data["gram_w"]))
    program = stan_model("mblf")
    init = program.sample(data=data, chains=n_points, parallel_chains=1, iter_warmup=0, iter_sampling=1,
                          fixed_param=True, adapt_engaged=False, seed=seed, inits=2.0, show_progress=False)
    names = ["regr_coeffs", "eta", "Lambda", "tau", "theta_tilde", "delta", "psi2"] if data["L"] else \
        ["eta", "Lambda", "tau", "theta_tilde", "delta", "psi2"]
    values = {k: init.stan_variable(k) for k in names}
    const = 0.0
    for d, m in enumerate(modes):
        rss = 0.0
        for n in range(m.N):
            o = ~np.isnan(m.data[:, n])
            if o.any():
                coef = np.linalg.lstsq(m.basis[o], m.data[o, n], rcond=None)[0]
                rss += np.sum((m.data[o, n] - m.basis[o] @ coef) ** 2)
        a, b, M = data["a_psi"][d], data["b_psi"][d], int(np.sum(~np.isnan(m.data)))
        const += (a * np.log(b) - gammaln(a) - 0.5 * M * np.log(2 * np.pi) - (a + M / 2) * np.log(b + rss / 2)
                  + gammaln(a + M / 2))
    nc = np.asarray(data["noncentered"], bool)
    stan, direct = [], []
    for i in range(n_points):
        point = {k: v[i] for k, v in values.items()}
        lp = [float(program.log_prob(params=point, data=x, jacobian=True, sig_figs=18)["lp__"].iloc[0]) for x in (data, empty)]
        stan.append(lp[0] - lp[1] + const)
        theta = point["theta_tilde"].copy()
        theta[nc] = (point["Lambda"] @ point["eta"])[nc] + theta[nc] / np.sqrt(point["tau"][nc, None])
        ll, start = 0.0, 0
        for d, m in enumerate(modes):
            p = m.basis.shape[1]
            mu = m.basis @ theta[start:start + p]
            o = ~np.isnan(m.data)
            s2 = float(np.atleast_1d(point["psi2"])[d])
            ll += np.sum(-0.5 * np.log(2 * np.pi * s2) - (m.data[o] - mu[o]) ** 2 / (2 * s2))
            start += p
        direct.append(ll)
    stan, direct = np.array(stan), np.array(direct)
    return dict(stan=stan, direct=direct, abs_diff=np.abs(stan - direct), constant=const,
                max_abs_diff=float(np.max(np.abs(stan - direct))))
