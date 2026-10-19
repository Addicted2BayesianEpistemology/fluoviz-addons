"""Is the network any good?  The honest checks of SPEC section 5, each with a plain-English verdict.

Everything here is **numpy** (no PyTorch import at module level), so the server can call it; the *scans* (``latent_scan``, ``smoothify_scan``, ``stability_scan``) train several networks
and are therefore jobs for the worker (``worker.run_in_worker("latent_scan", {...})``): they import ``train`` lazily inside the function, in the worker process.

Conventions.  An *error* is the squared distance between a sample and its reconstruction in the metric of the loss, divided by the total variance of the variable and averaged over the variables
(``core.forward.recon_error``): 0 = perfect, 1 = as far as the mean.  The *share of the variation rebuilt* (``evr``, "explained variance") of a set of samples is
``1 - sum(error) / sum(error of the mean)`` and is the number to compare with the explained variance of the PCA / functional PCA.  Numbers about held-out samples are the honest ones.
"""
from __future__ import annotations

import math

import numpy as np

from . import forward as fw
from . import linear as lin
from . import tools
from .spec import AutoencoderError, Config, ModelSpec


def pct(x: float, digits: int = 1) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{100.0 * x:.{digits}f} %"


def evr(err, null) -> float:
    err, null = np.asarray(err, float), np.asarray(null, float)
    ok = np.isfinite(err) & np.isfinite(null)
    s0 = float(null[ok].sum())
    return float(1.0 - err[ok].sum() / s0) if s0 > 0 else float("nan")


def error_se(err, null) -> float:
    """Standard error of ``sum(err) / sum(null)`` (the share of the variation NOT rebuilt) over the samples (delta method)."""
    err, null = np.asarray(err, float), np.asarray(null, float)
    ok = np.isfinite(err) & np.isfinite(null)
    n = int(ok.sum())
    if n < 2:
        return float("nan")
    e, z = err[ok], null[ok]
    r = e.sum() / max(z.sum(), 1e-300)
    return float(np.std(e - r * z, ddof=1) / (math.sqrt(n) * max(z.mean(), 1e-300)))


# ============================================================================= reconstruction quality
def reconstruction(prep, spec: ModelSpec, arrays: dict, rows=None) -> dict:
    """The network applied to every sample of ``prep`` (a ``core.data.Prepared``), in numbers and in functions.

    ``rows``: the samples the *maps* (local R^2, residual RMS) are computed on (default all).  Returns ``{"X_hat", "codes", "logvar", "error" (n,), "error_blocks" (n, nb), "null", "r2" (n,)
    = 1 - error, "evr", "evr_blocks": {name: ...}, "rmse": (n,) (the first variable), "rmse_blocks": {name: (n,)} (L2 distance to the function the network is asked to rebuild, in the units of the
    data after the pre-treatment), "rmse_observed": {name: (n,)} (to the measured values), "functions": {name: (n, m)} (reconstructions), "local_r2": {name: (m,)}, "residual_rms": {name: (m,)}}``."""
    X, M = prep.X, prep.M
    Xh, mu, lv = fw.reconstruct(spec, arrays, X, M)
    tot, per = fw.recon_error(spec, arrays, X, Xh, M)
    null, per0 = fw.null_error(spec, arrays, X, M)
    F = prep.to_functions(Xh)
    target, observed = prep.smoothed(), prep.observed()
    rows = np.arange(prep.n) if rows is None else np.asarray(rows)
    out = {"X_hat": Xh, "codes": mu, "logvar": lv, "error": tot, "error_blocks": per, "null": null, "r2": 1.0 - tot, "evr": evr(tot, null), "functions": F,
           "evr_blocks": {}, "rmse_blocks": {}, "rmse_observed": {}, "local_r2": {}, "residual_rms": {}}
    for j, b in enumerate(prep.blocks):
        w = b.domain.w
        t, f, o = target[b.name], F[b.name], observed[b.name]
        ok = np.isfinite(t)
        d2 = np.where(ok, (t - f) ** 2, 0.0)
        cov = (ok * w).sum(axis=1)
        out["rmse_blocks"][b.name] = np.sqrt((d2 * w).sum(axis=1) / np.maximum(cov, 1e-300))
        out["rmse_observed"][b.name] = np.sqrt(np.nanmean((o - f) ** 2, axis=1))
        out["evr_blocks"][b.name] = float(1.0 - np.nansum(per[:, j]) / max(np.nansum(per0[:, j]), 1e-300))
        r = rows
        res2 = np.where(np.isfinite(t[r]), (t[r] - f[r]) ** 2, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            var = np.nansum((t[r] - np.nanmean(t[r], axis=0)) ** 2, axis=0)
            out["local_r2"][b.name] = np.where(var > 1e-300, 1.0 - np.nansum(res2, axis=0) / var, 0.0)
            out["residual_rms"][b.name] = np.sqrt(np.nanmean(res2, axis=0))
    out["rmse"] = out["rmse_blocks"][prep.blocks[0].name]
    return out


def worst_samples(ids, error, heldout=None, top: int = 10) -> list[dict]:
    """The samples the network rebuilds worst: ``[{"id", "error", "r2", "heldout"}]`` (largest error first)."""
    error = np.asarray(error, float)
    order = np.argsort(-np.where(np.isfinite(error), error, -np.inf))[:top]
    return [{"id": str(ids[i]), "error": float(error[i]), "r2": float(1.0 - error[i]), "heldout": bool(heldout[i]) if heldout is not None else None} for i in order]


# ============================================================================= the linear baseline
def pca_curve(td, split: dict, kmax: int) -> dict:
    """Error of the (functional) PCA with ``k = 0 ... kmax`` components in the metric of the loss, on the same split as the network: ``{"k", "error_val", "error_val_se", "error_train",
    "mode"}`` (errors = share of the variation NOT rebuilt; validation = held-out samples, or out-of-fold for k-fold validation)."""
    n = td.n
    ks = list(range(0, int(kmax) + 1))
    M = td.M
    if split["mode"] == "kfold":
        errs = {k: np.full(n, np.nan) for k in ks}
        for vi in split["folds"]:
            vi = np.asarray(vi)
            tr = np.setdiff1d(np.arange(n), vi)
            m = lin.fit_pca(td.X[tr], None if M is None else M[tr], td.omega, kmax, sample_w=td.sw_of(tr))
            e = lin.pca_errors(m, td.X[vi], None if M is None else M[vi], td.omega, td.blocks, ks)
            for k in ks:
                errs[k][vi] = e[k]
        m = lin.fit_pca(td.X, M, td.omega, kmax, sample_w=td.sw_of())
        et = lin.pca_errors(m, td.X, M, td.omega, td.blocks, ks)
        val_idx = np.arange(n)
        err_tr = {k: et[k] for k in ks}
        tr_idx = np.arange(n)
    else:
        tr_idx, val_idx = np.asarray(split["train"]), np.asarray(split["val"])
        m = lin.fit_pca(td.X[tr_idx], None if M is None else M[tr_idx], td.omega, kmax, sample_w=td.sw_of(tr_idx))
        e = lin.pca_errors(m, td.X, M, td.omega, td.blocks, ks)
        errs = {k: e[k] for k in ks}
        err_tr = errs
    null = errs[0] if split["mode"] != "kfold" else errs[0]
    out = {"k": ks, "error_val": [], "error_val_se": [], "error_train": [], "mode": split["mode"]}
    for k in ks:
        ev, n0 = errs[k][val_idx], null[val_idx]
        out["error_val"].append(float(ev.sum() / max(n0.sum(), 1e-300)))
        out["error_val_se"].append(error_se(ev, n0))
        out["error_train"].append(float(err_tr[k][tr_idx].sum() / max(err_tr[0][tr_idx].sum(), 1e-300)))
    return out


def compare_baseline(result) -> dict:
    """Is the nonlinearity worth it?  The network against the PCA / functional PCA with the same number of components on the same held-out samples.  Returns
    ``{"name", "k", "evr_network", "evr_baseline", "relative_improvement", "z", "better": "network"|"baseline"|"same", "verdict"}`` (paired comparison of the per-sample errors)."""
    s, v = result.samples, result.validation
    held = np.isfinite(s["error_heldout"]) & np.isfinite(s["error_baseline"])
    name, k = v["baseline"]["name"], v["baseline"]["k"]
    e_net, e_pca, null = s["error_heldout"][held], s["error_baseline"][held], s["null"][held]
    n = int(held.sum())
    out = {"name": name, "k": int(k), "evr_network": v["evr"], "evr_baseline": v["baseline"]["evr"], "n": n}
    if n < 5:
        out.update({"relative_improvement": float("nan"), "z": float("nan"), "better": "same",
                    "verdict": f"Only {n} held-out samples: too few to say whether the network is better than the {name} with {k} components."})
        return out
    d = e_net - e_pca
    se = float(np.std(d, ddof=1) / math.sqrt(n))
    z = float(d.mean() / se) if se > 0 else 0.0
    rel = float(1.0 - e_net.sum() / max(e_pca.sum(), 1e-300))
    both = f"On held-out samples the network rebuilds {pct(v['evr'])} of the variation and the {name} with the same {k} component{'s' if k != 1 else ''} {pct(v['baseline']['evr'])}."
    material = abs(v["evr"] - v["baseline"]["evr"]) >= 0.003                     # a difference of less than 0.3 percentage points of the variation does not matter
    if rel > 0.15 and z < -2 and material:
        better = "network"
        text = f"{both} The network makes {100 * rel:.0f} % less error, and the difference is clear: the nonlinearity is worth it for these data."
    elif rel < -0.15 and z > 2 and material:
        better = "baseline"
        text = (f"{both} The network does worse ({100 * -rel:.0f} % more error), and the difference is clear: a straight-line (linear) description is better for these data. "
                f"Use the {name}, or give the network fewer codes, a smaller size or more samples.")
    else:
        better = "same"
        text = (f"{both} The difference is small or within the noise of {n} held-out samples: the data are well described by a linear model, which is simpler, so the {name} is "
                "the safer choice unless you need the network's other tools (codes for classification, generation, anomaly scores).")
    out.update({"relative_improvement": rel, "z": z, "better": better, "verdict": text})
    return out


# ============================================================================= latent-space health
def participation_ratio(v) -> float:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    s2 = float((v ** 2).sum())
    return float(v.sum() ** 2 / s2) if s2 > 0 else 0.0


def latent_health(spec: ModelSpec, arrays: dict, X: np.ndarray, M=None, codes=None, logvar=None) -> dict:
    """The state of the code space.  Returns ``{"kind", "names", "mean", "sd", "correlation" (d, d), "effect_share" (d,), "inactive" (d,) bool, "unused" (d,) bool (effect < 1 %),
    "collapsed" (d,) bool (VAE: KL < 0.05 nats), "kl_per_dim" (d,)|None, "effective_dim", "n_active", "coverage": {...}|None, "words"}``.

    A code is *inactive* when it has (almost) no effect on the reconstruction (< 1 % of the total, from the Jacobian of the decoder) and, for a variational autoencoder, also when its
    KL divergence is below 0.05 nats or the variance of its posterior means below 0.01 (a collapsed code: the posterior equals the prior)."""
    if codes is None:
        codes, logvar = fw.encode(spec, arrays, X, M)
    codes = np.asarray(codes, float)
    d = codes.shape[1]
    mean, sd = codes.mean(axis=0), codes.std(axis=0, ddof=1) if codes.shape[0] > 1 else np.ones(d)
    sens = tools.sensitivity(spec, arrays, codes)
    share = sens["share"]
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(codes.T) if codes.shape[0] > 2 and d > 1 else np.eye(d)
    C = np.where(np.isfinite(C), C, 0.0)
    np.fill_diagonal(C, 1.0)
    unused = share < 0.01                                                # the decoder (almost) ignores the code
    inactive = unused.copy()
    kl_dim = None
    collapsed = np.zeros(d, bool)
    cover = None
    vae = spec.is_vae() and logvar is not None
    if vae:
        kl_dim = 0.5 * (codes ** 2 + np.exp(logvar) - 1.0 - logvar).mean(axis=0)
        collapsed = kl_dim < 0.05                                       # the posterior equals the prior: the code carries nothing
        if not spec.has_mixture():
            inactive = inactive | collapsed | (codes.var(axis=0, ddof=1) < 0.01)
            agg_var = codes.var(axis=0, ddof=1) + np.exp(logvar).mean(axis=0)
            from scipy.stats import chi2
            inside = float(np.mean((codes ** 2).sum(axis=1) <= chi2.ppf(0.95, d)))
            cover = {"mean": mean, "variance": agg_var, "inside95": inside}
    n_act = int((~inactive).sum())
    names = [f"z{j + 1}" for j in range(d)]
    eff = participation_ratio(share)
    corr_max = float(np.max(np.abs(C - np.eye(d)))) if d > 1 else 0.0
    parts = []
    if n_act == d:
        parts.append(f"All {d} codes are used" + (f" (effective number of independent codes: {eff:.1f})." if d > 1 else "."))
    else:
        dead = [names[j] for j in range(d) if inactive[j]]
        parts.append(f"{n_act} of the {d} codes are used. {', '.join(dead)} {'has' if len(dead) == 1 else 'have'} almost no effect on the reconstruction"
                     + (" (the network collapsed " + ("it" if len(dead) == 1 else "them") + " onto the prior)" if vae else "") + f": try {max(n_act, 1)} codes, which is simpler and as good.")
    if d > 1 and corr_max > 0.5:
        j, k = np.unravel_index(np.argmax(np.abs(C - np.eye(d))), C.shape)
        parts.append(f"{names[j]} and {names[k]} are strongly correlated (r = {C[j, k]:.2f}): they partly describe the same thing, which is normal for an autoencoder unless you need independent codes (then raise beta of a variational autoencoder).")
    if cover is not None:
        if cover["inside95"] < 0.85:
            parts.append(f"Only {100 * cover['inside95']:.0f} % of the samples lie inside the 95 % region of the prior: the codes do not fill the prior (new samples drawn from the prior may look unlike real ones). A larger beta or longer warm-up regularises the codes.")
        else:
            parts.append(f"The codes fill the prior well ({100 * cover['inside95']:.0f} % of the samples lie inside its 95 % region): samples drawn from the prior should look realistic.")
    return {"kind": "vae" if vae else "ae", "names": names, "mean": mean, "sd": sd, "correlation": C, "effect_share": share, "inactive": inactive, "unused": unused, "collapsed": collapsed, "kl_per_dim": kl_dim, "effective_dim": eff,
            "n_active": n_act, "coverage": cover, "words": " ".join(parts)}


# ============================================================================= stability across restarts
def _cca(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A, B = A - A.mean(axis=0), B - B.mean(axis=0)
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    return np.clip(np.linalg.svd(Qa.T @ Qb, compute_uv=False), 0.0, 1.0)


def _procrustes(A: np.ndarray, B: np.ndarray) -> float:
    A, B = A - A.mean(axis=0), B - B.mean(axis=0)
    A, B = A / max(np.linalg.norm(A), 1e-300), B / max(np.linalg.norm(B), 1e-300)
    return float(np.linalg.svd(A.T @ B, compute_uv=False).sum() ** 2)


def stability_from_codes(codes: np.ndarray, ref: int = 0) -> dict:
    """How much two runs of the training agree about the codes: for every pair of runs the canonical correlations (the best linear match between the two sets of codes) and the
    Procrustes similarity.  ``codes`` ``(R, n, d)``.  Returns ``{"n_runs", "matrix" (R, R) (mean canonical correlation), "procrustes" (R, R), "per_code" (d,) (how well code j of run ``ref``
    is found, as a multiple correlation, in each other run, averaged), "mean", "min", "verdict"}``."""
    codes = np.asarray(codes, float)
    R, n, d = codes.shape
    if R < 2:
        return {"n_runs": R, "matrix": np.ones((1, 1)), "procrustes": np.ones((1, 1)), "per_code": np.ones(d), "mean": float("nan"), "min": float("nan"),
                "verdict": "Only one training run: stability cannot be judged. Train with more random starts."}
    M = np.eye(R)
    Pm = np.eye(R)
    for a in range(R):
        for b in range(a + 1, R):
            M[a, b] = M[b, a] = float(_cca(codes[a], codes[b]).mean())
            Pm[a, b] = Pm[b, a] = _procrustes(codes[a], codes[b])
    per = np.zeros(d)
    others = [b for b in range(R) if b != ref]
    for j in range(d):
        v = []
        for b in others:
            B = np.hstack([np.ones((n, 1)), codes[b]])
            y = codes[ref][:, j]
            beta, *_ = np.linalg.lstsq(B, y, rcond=None)
            ss = float(((y - y.mean()) ** 2).sum())
            r2 = 1.0 - float(((y - B @ beta) ** 2).sum()) / ss if ss > 0 else 0.0
            v.append(math.sqrt(max(r2, 0.0)))
        per[j] = float(np.mean(v))
    iu = np.triu_indices(R, 1)
    mean, mn = float(M[iu].mean()), float(M[iu].min())
    ok = per >= 0.8
    weak = [f"z{j + 1}" for j in range(d) if not ok[j]]
    if ok.all() and mean >= 0.95:
        verdict = f"The codes are reproducible: another training run finds the same codes (mean canonical correlation {mean:.2f} between {R} runs)."
    elif ok.sum() >= (d + 1) // 2:
        verdict = (f"The codes are only partly reproducible (mean canonical correlation {mean:.2f} between {R} runs)" + (f"; {', '.join(weak)} changed from run to run" if weak else "") +
                   ". Use the codes with care, or fewer codes, more samples or a variational autoencoder with a larger beta.")
    else:
        verdict = (f"The codes are NOT reproducible: other training runs find quite different codes (mean canonical correlation {mean:.2f} between {R} runs). "
                   "Do not interpret individual codes; use fewer codes, a smaller network, more samples or more weight decay.")
    return {"n_runs": R, "matrix": M, "procrustes": Pm, "per_code": per, "mean": mean, "min": mn, "verdict": verdict}


# ============================================================================= smoothness
def curvature(values: np.ndarray, grid: dict) -> np.ndarray:
    """Roughness of functions ``values`` ``(n, m)`` observed at ``grid["idx"]`` of a grid of ``grid["shape"]``: the root mean square of the second differences along each axis (where three
    neighbouring points are observed), per sample ``(n,)``."""
    shape, idx = tuple(grid["shape"]), np.asarray(grid["idx"])
    n = values.shape[0]
    full = np.full((n, int(np.prod(shape))), np.nan)
    full[:, idx] = values
    G = full.reshape(n, *shape)
    tot = np.zeros(n)
    cnt = 0
    for ax in range(1, G.ndim):
        sl = [slice(None)] * G.ndim
        a = np.take(G, range(0, shape[ax - 1] - 2), axis=ax)
        b = np.take(G, range(1, shape[ax - 1] - 1), axis=ax)
        c = np.take(G, range(2, shape[ax - 1]), axis=ax)
        d2 = (a - 2 * b + c) ** 2
        tot += np.nansum(d2.reshape(n, -1), axis=1)
        cnt += int(np.isfinite(d2[0]).sum())
    return np.sqrt(tot / max(cnt, 1))


def roughness(td, X: np.ndarray) -> dict:
    """Roughness of coordinates ``X`` ``(n, P)``: per variable the mean over the samples of the smoothness measure of its block - ``x' Q x / K`` with the representation's own smoothness
    metric (the RKHS norm of the Gaussian-process basis, the P-spline penalty, the cosine shrinkage; about 1 for a typical sample), or, for pixels (no smoothness defined), the root mean
    square curvature (second differences on the grid).  ``{name: float}`` and ``{name: (n,)}`` under ``"per_sample"``."""
    X = np.atleast_2d(np.asarray(X, float))
    qs = td.q_rough if td.q_rough is not None else td.q
    out, per = {}, {}
    for b, q in zip(td.blocks, qs):
        a, c = b["slice"]
        v = X[:, a:c]
        if b["kind"] == "pixels":
            r = curvature(v, td.info["grid"][b["name"]])
        elif q["kind"] == "identity":
            r = (v ** 2).sum(axis=1) / v.shape[1]
        elif q["kind"] == "diag":
            r = ((v ** 2) * q["d"]).sum(axis=1) / v.shape[1]
        else:
            r = ((v @ q["Q"]) * v).sum(axis=1) / v.shape[1]
        per[b["name"]] = r
        out[b["name"]] = float(np.mean(r))
    return {"mean": out, "per_sample": per}


def smoothness_report(td, X_input: np.ndarray, X_recon: np.ndarray, smoothify: float = 0.0, metric: str = "smooth") -> dict:
    """The smoothness check (step 8): per variable the roughness of the reconstructions against that of the samples they rebuild (:func:`roughness`).  Rule: a ratio below 0.9 = smoother than the
    samples, 0.9 to 1.1 = about as rough, above 1.1 = rougher (the network adds wiggles the samples do not have).  Returns ``{"smoothify", "metric", "blocks": [{"name", "measure", "input",
    "reconstruction", "ratio", "verdict"}], "words"}``."""
    rin, rout = roughness(td, X_input)["mean"], roughness(td, X_recon)["mean"]
    rows, parts = [], []
    for b in td.blocks:
        nm = b["name"]
        r_in, r_out = float(rin[nm]), float(rout[nm])
        ratio = r_out / max(r_in, 1e-300)
        what = "curvature (second differences)" if b["kind"] == "pixels" else "smoothness measure of the representation (1 = as rough as a typical sample)"
        word = "smoother than the samples" if ratio < 0.9 else "rougher than the samples" if ratio > 1.1 else "about as rough as the samples"
        rows.append({"name": nm, "measure": what, "input": r_in, "reconstruction": r_out, "ratio": ratio, "verdict": word})
        parts.append(f"The {nm} reconstructions are {word}: {what}, {r_out:.3g} against {r_in:.3g} for the samples ({100 * ratio:.0f} %).")
    parts.append("smoothify is " + ("off." if smoothify <= 0 else f"{smoothify:g} ({'penalty on the roughness of the representation' if metric == 'smooth' else 'plain ridge on the coefficients'})."))
    if smoothify <= 0 and any(r["ratio"] > 1.1 for r in rows):
        parts.append("Try the smoothify scan: a little smoothify usually removes the wiggles at no cost in accuracy.")
    elif smoothify <= 0:
        parts.append("The smoothify scan shows what a little of it costs and gains.")
    return {"smoothify": float(smoothify), "metric": metric, "blocks": rows, "words": " ".join(parts)}


# ============================================================================= calibration (VAE)
def calibration(spec: ModelSpec, arrays: dict, X: np.ndarray, M=None, n_rep: int = 5, seed: int = 0) -> dict:
    """Are the uncertainties of a variational autoencoder believable?  A sample is made up from the code of every sample (the decoding plus noise at the level the network itself estimated,
    ``sigma``), encoded again, and the recovered code is compared with the true one in units of the stated posterior SD: if the SDs are right, 68 % of the errors are inside 1 SD and 95 % inside
    1.96 SD.  Returns ``{"coverage68", "coverage95", "mean_z2", "per_code": [coverage95...], "verdict"}``."""
    if not spec.is_vae() or spec.vae.get("sigma") is None:
        raise AutoencoderError("Calibration needs a variational autoencoder.")
    rng = np.random.default_rng(seed)
    net = fw.Network(spec, arrays)
    mu, _ = net.encode(X, M)
    base = net.decode(mu)
    w = np.maximum(np.asarray(arrays["out.weights"], float), 1e-300)
    sd_x = float(spec.vae["sigma"]) / np.sqrt(spec.input_dim * w)
    zs = []
    for _ in range(int(n_rep)):
        Xn = base + sd_x * rng.standard_normal(base.shape)
        mu2, lv2 = net.encode(Xn)
        zs.append((mu2 - mu) / np.exp(0.5 * lv2))
    z = np.concatenate(zs)
    c68, c95 = float((np.abs(z) < 1.0).mean()), float((np.abs(z) < 1.96).mean())
    per = (np.abs(z) < 1.96).mean(axis=0)
    m2 = float((z ** 2).mean())
    if 0.85 <= c95 <= 0.99:
        v = f"The stated uncertainties are believable: {100 * c95:.0f} % of the code errors fall inside the 95 % bands (should be about 95 %)."
    elif c95 < 0.85:
        v = f"The network is overconfident: only {100 * c95:.0f} % of the code errors fall inside the 95 % bands (should be about 95 %). Do not trust the posterior SDs as error bars; a larger beta or free bits can help."
    else:
        v = f"The network is underconfident: {100 * c95:.0f} % of the code errors fall inside the 95 % bands; its stated uncertainties are larger than needed."
    return {"coverage68": c68, "coverage95": c95, "mean_z2": m2, "per_code": per, "verdict": v}


# ============================================================================= anomalies
def default_limit(errors_heldout, level: float = 0.99, fallback=None) -> dict:
    """The reconstruction-error limit of the saved model: the ``level`` (0.95 or 0.99) point of the errors of samples the network did not train on.  **The same function the saved model uses**
    (``kinds.stats.error_limits``: a log-scale median plus a robust spread), so that the page and the saved kind give one answer.  ``fallback``: the errors of training samples, used (widened)
    when fewer than 3 held-out errors exist.  Returns ``{"limit", "method", "n", "level", "source", "words", "all": {"95", "99"}}``."""
    from ..kinds import stats as kstats
    key = "99" if level >= 0.98 else "95"
    lim = kstats.error_limits(errors_heldout, fallback)
    return {"limit": lim[key], "method": "log-median + robust spread", "n": lim["n"], "level": level, "source": lim["source"], "words": lim["words"], "all": {k: lim[k] for k in kstats.LEVELS}}


def robust_limit(errors, method: str = "mad", c: float = 3.0, level: float = 0.99) -> dict:
    """A threshold for the errors of ALL samples: ``mad`` = median + c MAD of the log errors, back-transformed; ``quantile`` = the ``level`` quantile."""
    e = np.asarray(errors, float)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return {"limit": float("nan"), "method": method}
    if method == "quantile":
        return {"limit": float(np.quantile(e, level)), "method": method, "level": level}
    if method != "mad":
        raise AutoencoderError("The threshold method must be mad or quantile.")
    le = np.log(np.maximum(e, 1e-12))
    med = np.median(le)
    mad = 1.4826 * np.median(np.abs(le - med))
    return {"limit": float(np.exp(med + c * mad)), "method": method, "c": c}


def typicality(codes_ref: np.ndarray, codes: np.ndarray, logvar_ref: np.ndarray | None = None) -> np.ndarray:
    """How typical the codes ``codes`` ``(n, d)`` are among the training codes ``codes_ref``: the p-value of the squared Mahalanobis distance under the aggregate posterior (the covariance
    of the training codes plus, for a variational autoencoder, the mean posterior variance), F-calibrated for a new sample with few training samples.  **The same function the saved model
    uses** (``kinds.stats``).  Small = an unusual code.  ``logvar_ref``: the log posterior variances of the training samples (VAE)."""
    from ..kinds import stats as kstats
    st = kstats.code_stats(np.asarray(codes_ref, float), None if logvar_ref is None else np.exp(0.5 * np.asarray(logvar_ref, float)))
    return kstats.typicality(np.asarray(codes, float), st)


def elbo(spec: ModelSpec, arrays: dict, X: np.ndarray, M=None) -> dict:
    """Per sample of a variational autoencoder: the KL divergence of the code, the reconstruction error, and the ELBO ``-[(N/2)(error / sigma^2 + log sigma^2) + KL]`` (up to a constant,
    evaluated with the reconstruction at the posterior mean; higher = the network explains the sample better).  The KL and the ELBO are those of ``kinds.stats`` (the saved model's)."""
    from ..kinds import stats as kstats
    if not spec.is_vae():
        raise AutoencoderError("The ELBO needs a variational autoencoder.")
    Xh, mu, lv = fw.reconstruct(spec, arrays, X, M)
    err, _ = fw.recon_error(spec, arrays, X, Xh, M)
    if spec.has_mixture():
        kl = kstats.kl_mixture(mu, lv, arrays["prior.pi"], arrays["prior.mu"], arrays["prior.logvar"])
    else:
        kl = kstats.kl_normal(mu, lv)
    sig = max(float(spec.vae.get("sigma") or 1.0), 1e-6)
    return {"kl": kl, "error": err, "elbo": kstats.elbo(err, kl, sig, float(spec.vae.get("n_eff") or spec.input_dim))}


def flag_samples(error: np.ndarray, limit: float, p_typical: np.ndarray | None = None, p_min: float = 0.01) -> tuple[np.ndarray, list[str]]:
    """Status of every sample: 0 inside the limits, 1 reconstruction error above the limit, 2 unusual code, 3 both.  Returns ``(codes, words)``."""
    err = np.asarray(error, float)
    f = np.where(np.isfinite(err) & np.isfinite(limit), err > limit, False).astype(int)
    if p_typical is not None:
        f = f + 2 * (np.asarray(p_typical) < p_min).astype(int)
    words = ["inside the limits", "reconstruction error above the limit", "unusual code", "both"]
    return f, [words[i] for i in f]


# ============================================================================= verdicts
def quality_verdict(evr_val: float, n_val: int, k: int) -> str:
    if not np.isfinite(evr_val):
        return "The quality could not be judged."
    lvl = "excellent" if evr_val >= 0.99 else "good" if evr_val >= 0.95 else "fair" if evr_val >= 0.90 else "poor"
    s = f"On held-out samples the network rebuilds {pct(evr_val)} of the variation ({lvl}; {n_val} samples, {k} code{'s' if k != 1 else ''})."
    if lvl == "poor":
        s += " Try more codes, a larger network or more training epochs, or check whether the samples are very different from each other."
    elif lvl == "fair":
        s += " More codes may help."
    return s


STOP_HINT = "Step 6 · Training settings · Advanced training settings has the controls."


def stop_reason(result) -> str:
    """Why the training of the kept network ended: ``"early"`` (the patience was used up), ``"time"`` (the time limit) or ``"max"`` (all the epochs)."""
    runs = (getattr(result, "restarts", None) or {}).get("runs") or []
    kept = (getattr(result, "restarts", None) or {}).get("kept", 0)
    if runs and 0 <= kept < len(runs) and runs[kept].get("stopped") in ("early", "time", "max"):
        return runs[kept]["stopped"]
    return "early" if result.epochs_run < result.config["epochs"] else "max"


def stopping_words(result, cfg: dict) -> str:
    """One or two sentences on how and why the training ended, with the advanced setting to change when it ended too early (every sentence the training verdict used to have is kept for
    the default settings)."""
    rule = cfg.get("stop_rule", "validation")
    why = stop_reason(result)
    last = result.epochs_run
    restored = cfg.get("restore_best", True)
    kept = f"the weights of epoch {result.best_epoch + 1} were restored" if restored else "the weights of the last epoch were kept"
    if result.best_epoch < 0:
        st = ("The network could not improve on its PCA starting point on held-out samples: the straight-line (linear) part is the answer for these data, and the nonlinear part of the "
              "network added nothing that holds up.")
        if why == "early":
            st += f" Training ended after {cfg['patience']} epochs without an improvement; if the nonlinear part may need longer to catch up with the PCA, raise the patience or the least number of epochs. {STOP_HINT}"
        elif why == "time":
            st += f" Training was stopped at epoch {last} by the time limit of {float(cfg.get('max_minutes') or 0):g} minutes for one network."
        elif rule == "never":
            st += f" Early stopping was off: all {cfg['epochs']} epochs were trained."
        return st
    if result.split["mode"] == "kfold":
        st = f"With k-fold validation the number of epochs ({last}) was taken from the folds."
        if cfg.get("final_epochs", "auto") == "auto":
            st += f" If that is too few for these data, choose the number of epochs of the final networks. {STOP_HINT}"
        else:
            st = f"With k-fold validation the final networks were trained for {last} epochs, as chosen."
        return st
    if why == "time":
        return f"Training was stopped at epoch {last} because the time limit of {float(cfg.get('max_minutes') or 0):g} minutes for one network was reached; {kept}."
    if why == "early":
        loss = "validation" if rule == "validation" else "training"
        st = f"Training stopped at epoch {last} because the {loss} loss had not improved for {cfg['patience']} epochs; {kept}."
        return st + f" To let it train longer, raise the patience or the least number of epochs, or lower the smallest improvement that counts. {STOP_HINT}"
    if rule == "never":
        return f"Early stopping was off: all {cfg['epochs']} epochs were trained; {kept}."
    if result.best_epoch >= 0.9 * cfg["epochs"]:
        return f"Training used all {cfg['epochs']} epochs and the best weights were still near the end (epoch {result.best_epoch + 1}): more epochs may help. Raise the maximum number of epochs."
    return f"Training used all {cfg['epochs']} epochs; the best weights are from epoch {result.best_epoch + 1}."


def training_verdict(result) -> dict:
    """Plain-English verdict on a training run: overfitting, the epoch of the best weights, and the agreement of the random starts.
    Returns ``{"overfit": "none" | "mild" | "clear", "gap", "words", "overfit_words", "stopping_words", "restarts_words", "quality_words"}``."""
    v, rs = result.validation, result.restarts
    et, ev = v["error_train"], v["error"]
    gap = float(v["evr_train"] - v["evr"])
    ratio = ev / max(et, 1e-12)
    if ratio > 3.0 and ev - et > 0.01:
        level = "clear"
        ow = (f"Clear overfitting: the network rebuilds {pct(v['evr_train'])} of the variation of the training samples but only {pct(v['evr'])} of held-out ones, so it partly memorises the samples. "
              "Use fewer codes, a smaller network, more weight decay, dropout or denoising, or more samples; trust the held-out numbers only.")
    elif ratio > 1.6 and ev - et > 0.003:
        level = "mild"
        ow = f"Mild overfitting: {pct(v['evr_train'])} of the variation is rebuilt for training samples and {pct(v['evr'])} for held-out ones. The held-out number is the one to quote."
    else:
        level = "none"
        ow = f"No sign of overfitting: training samples {pct(v['evr_train'])}, held-out samples {pct(v['evr'])}."
    cfg = result.config
    st = stopping_words(result, cfg)
    vals = [r.get("evr_val", r.get("evr_train")) for r in rs["runs"]]
    if rs["n"] > 1:
        spread = max(vals) - min(vals)
        rw = (f"The {rs['n']} random starts agree ({pct(min(vals))} to {pct(max(vals))}); the best was kept." if spread < 0.02 else
              f"The {rs['n']} random starts differ ({pct(min(vals))} to {pct(max(vals))}): the result depends on the start. The best was kept; more samples, a smaller network or more weight decay make it steadier.")
    else:
        rw = "Only one random start was made: its quality may depend on luck."
    qw = quality_verdict(v["evr"], v["n_val"], result.spec.latent_dim)
    return {"overfit": level, "gap": gap, "overfit_words": ow, "stopping_words": st, "restarts_words": rw, "quality_words": qw, "words": " ".join([qw, ow, st, rw])}


# ============================================================================= everything about one trained network
def assess(prep, result, level: float = 0.99, p_min: float = 0.01) -> dict:
    """One call for the page: the numbers and verdicts of a trained network on its data (``prep``: ``core.data.Prepared``, ``result``: ``train.TrainResult``).

    ``{"reconstruction", "baseline", "health", "calibration" (VAE), "training", "limit": {...}, "typicality" (n,), "flag" (n,) int, "flag_words", "elbo" (VAE), "kl" (VAE), "roughness"}``;
    the limit is :func:`default_limit` of the held-out errors, the typicality is :func:`typicality` of the codes against the training codes."""
    spec, arrays, s = result.spec, result.arrays, result.samples
    held = s["heldout"]
    rows = np.flatnonzero(held) if held.sum() >= 5 else None
    rec = reconstruction(prep, spec, arrays, rows)
    td = prep.train_data("smooth")
    health = latent_health(spec, arrays, prep.X, prep.M, rec["codes"], rec["logvar"])
    lim = default_limit(s["error_heldout"], level, s["error"][~held] if (~held).any() else None)
    tr = np.flatnonzero(~held) if (~held).any() else np.arange(prep.n)
    typ = typicality(rec["codes"][tr], rec["codes"], rec["logvar"][tr] if rec["logvar"] is not None else None)
    flag, words = flag_samples(rec["error"], lim["limit"], typ if spec.is_vae() else None, p_min)
    out = {"reconstruction": rec, "baseline": compare_baseline(result), "health": health, "training": training_verdict(result), "limit": lim, "typicality": typ, "flag": flag, "flag_words": words,
           "roughness": {"input": roughness(td, prep.X)["mean"], "reconstruction": roughness(td, rec["X_hat"])["mean"]}}
    if spec.is_vae():
        e = elbo(spec, arrays, prep.X, prep.M)
        out.update({"kl": e["kl"], "elbo": e["elbo"], "calibration": calibration(spec, arrays, prep.X[rows if rows is not None else slice(None)], None if prep.M is None else prep.M[rows if rows is not None else slice(None)])})
    return out


# ============================================================================= scans (jobs for the worker: they train networks)
def _fit(cfg: Config, data, progress=None, history_path=None):
    from . import train                                                  # imported here: only the worker process (and the tests) import PyTorch
    return train.fit(cfg, data, progress, history_path)


def _sub(progress, lo: float, hi: float):
    if progress is None:
        return None
    return progress.sub(lo, hi) if hasattr(progress, "sub") else (lambda f, m="": progress(lo + (hi - lo) * f, m))


def latent_scan(config, data, dims=None, families=None, n_restarts: int = 1, progress=None, history_path=None) -> dict:
    """The validation error for ``d = 1 ... D`` codes for the autoencoder (and the variational autoencoder) and the PCA / functional PCA, on the same split, and the recommended ``d``
    (the smallest within one standard error, or 1 % of the variation, of the best).  ``config``: a ``Config`` (or dict), ``data``: ``Prepared`` or ``TrainData``.  Errors are the share of
    the variation NOT rebuilt on held-out samples.  Returns ``{"scan": "latent", "dims", "families": {"ae": {"val", "val_se", "train"}}, "baseline": {"name", "val", "val_se", "train"},
    "recommended": {"latent_dim", "reason"}, "words"}``."""
    cfg = config if isinstance(config, Config) else Config.from_dict(config)
    td = data.train_data(cfg.smoothify_metric, cfg.gap_weighting) if hasattr(data, "train_data") else data
    dmax = int(min(12, td.P - 1, max(2, td.n // 3)))
    dims = list(range(1, dmax + 1)) if dims is None else [int(x) for x in dims]
    if any(d < 1 or d >= td.P for d in dims):
        raise AutoencoderError("The numbers of codes to scan must be between 1 and the number of inputs minus one.")
    fams = [cfg.family] if families is None else list(families)
    cfg.check_samples(td.n)
    from . import train
    split = train.make_split(td.n, cfg)
    res: dict = {}
    total = len(dims) * len(fams)
    done = 0
    for fam in fams:
        rec = {"val": [], "val_se": [], "train": []}
        for d in dims:
            c = cfg.replace(family=fam, latent_dim=d, n_restarts=n_restarts)
            r = _fit(c, td, _sub(progress, done / total, (done + 1) / total), history_path)
            done += 1
            s = r.samples
            held = np.isfinite(s["error_heldout"])
            rec["val"].append(float(1.0 - r.validation["evr"]))
            rec["val_se"].append(error_se(s["error_heldout"][held], s["null"][held]))
            rec["train"].append(float(1.0 - r.validation["evr_train"]))
        res[fam] = rec
    pc = pca_curve(td, split, max(dims))
    base = {"name": "PCA" if all(b["kind"] == "pixels" for b in td.blocks) else "Functional PCA", "k": dims,
            "val": [pc["error_val"][d] for d in dims], "val_se": [pc["error_val_se"][d] for d in dims], "train": [pc["error_train"][d] for d in dims]}
    first = res[fams[0]]
    val, se = np.asarray(first["val"]), np.asarray(first["val_se"])
    b = int(np.nanargmin(val))
    tol = max(float(se[b]) if np.isfinite(se[b]) else 0.0, 0.01 * float(base["val"][0] if base["val"] else 1.0), 1e-4)
    ok = np.flatnonzero(val <= val[b] + tol)
    rec_d = int(dims[int(ok[0])])
    why = (f"the smallest number of codes whose held-out error ({pct(val[int(ok[0])])} not rebuilt) is within one standard error (or 1 % of the variation) of the best "
           f"({pct(val[b])} with {dims[b]} codes)")
    words = (f"Recommended: {rec_d} code{'s' if rec_d != 1 else ''}, {why}. More codes than that rebuild the held-out samples no better. The {base['name']} needs "
             f"{next((d for d, v in zip(dims, base['val']) if v <= val[b] + tol), dims[-1])} components for the same quality.")
    return {"scan": "latent", "dims": dims, "families": res, "baseline": base, "recommended": {"latent_dim": rec_d, "reason": why}, "words": words}


def smoothify_scan(config, data, values=None, n_restarts: int = 1, progress=None, history_path=None) -> dict:
    """Reconstruction error against roughness for ``smoothify = s`` over a grid (an L-curve), on the same split: the networks are trained with each ``s``, the error is that of held-out
    samples, the roughness that of their reconstructions (:func:`roughness`; the mean over the variables).  The suggested ``s`` is the largest one whose held-out error is within one
    standard error (or 2 % relative) of the error without smoothing.  Returns ``{"scan": "smoothify", "values", "error", "error_se", "roughness", "roughness_input", "suggested", "words"}``."""
    cfg = config if isinstance(config, Config) else Config.from_dict(config)
    td = data.train_data(cfg.smoothify_metric, cfg.gap_weighting) if hasattr(data, "train_data") else data
    cfg.check_samples(td.n)
    vals = [0.0] + list(np.geomspace(1e-4, 10.0, 11)) if values is None else [float(v) for v in values]
    if any(v < 0 for v in vals):
        raise AutoencoderError("smoothify must be 0 or more.")
    err, se, rough = [], [], []
    rin = None
    for i, s in enumerate(vals):
        r = _fit(cfg.replace(smoothify=float(s), n_restarts=n_restarts), td, _sub(progress, i / len(vals), (i + 1) / len(vals)), history_path)
        S = r.samples
        held = np.isfinite(S["error_heldout"])
        err.append(float(1.0 - r.validation["evr"]))
        se.append(error_se(S["error_heldout"][held], S["null"][held]))
        Xh = fw.reconstruct(r.spec, r.arrays, td.X, td.M)[0]
        rr = roughness(td, Xh)["per_sample"]
        rough.append(float(np.mean([np.mean(v[held]) for v in rr.values()])))
        if rin is None:
            ri = roughness(td, td.X)["per_sample"]
            rin = float(np.mean([np.mean(v[held]) for v in ri.values()]))
    err_a, se_a, rough_a = np.asarray(err), np.asarray(se), np.asarray(rough)
    tol = max(float(se_a[0]) if np.isfinite(se_a[0]) else 0.0, 0.02 * err_a[0], 1e-4)
    ok = [i for i in range(len(vals)) if err_a[i] <= err_a[0] + tol]
    sug = float(vals[max(ok, key=lambda i: vals[i])]) if ok else 0.0
    if sug == 0.0:
        words = "Smoothing the reconstructions costs accuracy already at the smallest value tried: keep smoothify at 0 (off)."
    else:
        i = vals.index(sug)
        words = (f"Suggested smoothify: {sug:.3g}. It lowers the roughness of the reconstructions from {rough_a[0]:.3g} to {rough_a[i]:.3g} while the held-out error stays within one standard error "
                 f"({pct(err_a[0])} without, {pct(err_a[i])} with). Larger values smooth more but cost accuracy.")
    return {"scan": "smoothify", "values": [float(v) for v in vals], "error": err, "error_se": se, "roughness": rough, "roughness_input": rin, "suggested": sug, "words": words}


def stability_scan(config, data, n_runs: int = 5, bootstrap: bool = False, progress=None, history_path=None) -> dict:
    """Train ``n_runs`` networks from different random starts (with ``bootstrap``: on bootstrap samples of the training data) and compare the codes of ALL samples (canonical correlations,
    Procrustes) and the reconstructions.  Returns ``{"scan": "stability", "n_runs", "alignment": {"mean", "min", "per_code"}, "recon_spread", "matrix", "procrustes", "verdict", "words"}``."""
    cfg = config if isinstance(config, Config) else Config.from_dict(config)
    td = data.train_data(cfg.smoothify_metric, cfg.gap_weighting) if hasattr(data, "train_data") else data
    cfg.check_samples(td.n)
    n_runs = int(max(2, n_runs))
    codes, recs = [], []
    rng = np.random.default_rng(cfg.seed + 99)
    for i in range(n_runs):
        c = cfg.replace(seed=int(cfg.seed + 1000 + i), n_restarts=1)
        t = td
        if bootstrap:
            t = td.subset(rng.integers(0, td.n, td.n))
        r = _fit(c, t, _sub(progress, i / n_runs, (i + 1) / n_runs), history_path)
        Xh, mu, _ = fw.reconstruct(r.spec, r.arrays, td.X, td.M)
        codes.append(mu)
        recs.append(Xh)
    st = stability_from_codes(np.stack(codes))
    null = float(fw.weighted_error(td.X, np.broadcast_to(td.X.mean(axis=0), td.X.shape), td.omega, td.blocks, td.M)[0].mean())
    sp = []
    for a in range(n_runs):
        for b in range(a + 1, n_runs):
            sp.append(float(fw.weighted_error(recs[a], recs[b], td.omega, td.blocks)[0].mean()) / max(null, 1e-300))
    spread = float(np.mean(sp))
    words = st["verdict"] + f" The reconstructions of different runs differ by {pct(spread)} of the variation between samples."
    return {"scan": "stability", "n_runs": n_runs, "alignment": {"mean": st["mean"], "min": st["min"], "per_code": st["per_code"]}, "recon_spread": spread, "matrix": st["matrix"],
            "procrustes": st["procrustes"], "verdict": st["verdict"], "words": words}
