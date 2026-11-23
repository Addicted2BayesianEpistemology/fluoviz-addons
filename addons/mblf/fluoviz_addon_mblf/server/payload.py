"""What a result ``mblf-N`` holds for the page (``docs/api.md`` section 3), computed once at the end of the fit from the posterior draws; the views of ``views.py`` compute the rest on
demand.  Every function of a mode is sent the way the browser draws it (``to_view``): an EEM as an image ``(n_ex, n_em)`` float32 with NaN outside the domain, a spectrum on its
wavelength grid, the values as they are."""
from __future__ import annotations

import numpy as np

from ..core import checks as CH
from ..core import loo as LOO
from ..core import structure as ST
from ..core.posterior import Fitted, thin

VIEW_DRAWS = 400          # draws used for the summaries of the page (evenly thinned within the chains)


def to_view(mode, arr):
    """Values at the points of a mode ``(..., M)`` -> what the browser draws."""
    a = np.asarray(arr, float)
    if mode.kind == "vector":
        return a.astype(np.float32)
    from fluoviz.analysis.frozen.kinds.fpca import ElementRuntime
    dom = ElementRuntime(mode.element).dom
    g = dom.embed(a)
    if mode.kind == "surface":
        if g.ndim == 2:
            return np.ascontiguousarray(g.T.astype(np.float32))
        return np.ascontiguousarray(np.swapaxes(g, -1, -2).astype(np.float32))
    return g.astype(np.float32)


def mode_info(mode, n_samples: int) -> dict:
    out = {"name": mode.name, "label": mode.label, "kind": mode.kind, "n_points": mode.M, "n_basis": mode.p, "captured": mode.captured, "basis": dict(mode.basis), "units": mode.units,
           "n_samples": n_samples, "mean": to_view(mode, mode.mean)}
    if mode.kind == "surface":
        el = mode.element
        out.update(ex=np.asarray(el.axes[1], float), em=np.asarray(el.axes[0], float), mask=np.ascontiguousarray((~np.asarray(el.obs, bool)).T.astype(np.uint8)))
    elif mode.kind == "curve":
        out.update(x=np.asarray(mode.element.axes[0], float))
    else:
        out.update(columns=list(mode.columns), labels=list(mode.column_labels), sd=np.asarray(mode.scale, float))
    return out


def view_summary(mode, s: dict) -> dict:
    """A function summary of ``structure.summarise`` -> browser form."""
    out = {"mean": to_view(mode, s["mean"]), "quantiles": to_view(mode, s["quantiles"]), "probs": s["probs"]}
    for k in ("sd", "p_pos", "lfsr"):
        if k in s:
            out[k] = to_view(mode, s[k])
    if "band" in s:
        out["band"] = {"level": s["band"]["level"], "multiplier": s["band"]["multiplier"], "lower": to_view(mode, s["band"]["lower"]), "upper": to_view(mode, s["band"]["upper"])}
    return out


def factors_view(f: Fitted, idx) -> list[dict]:
    out = []
    for fac in ST.factor_summaries(f, idx):
        out.append({"factor": fac["factor"], "share": fac["share"],
                    "modes": [{"mode": m["mode"], **view_summary(f.modes[f.index(m["mode"])], m)} for m in fac["modes"]]})
    return out


def variance_view(f: Fitted, idx) -> list[dict]:
    out = []
    for d, m in enumerate(f.modes):
        v = ST.variance_parts(f, d, idx)
        out.append({"mode": m.name, "variance": {k: to_view(m, x) for k, x in v["variance"].items()},
                    "share": {k: {"mean": to_view(m, s["mean"]), "lower": to_view(m, s["quantiles"][1]) if np.ndim(s["quantiles"]) == 2 else None,
                                  "upper": to_view(m, s["quantiles"][5]) if np.ndim(s["quantiles"]) == 2 else None} for k, s in v["share"].items()},
                    "whole": v["whole"]})
    return out


def predictable_view(f: Fitted, t: int, modes: list, covariates: bool, idx) -> dict:
    m = f.modes[t]
    r = ST.predictable(f, t, modes, covariates, idx)
    sl = r["latent"]
    sm = r["measured"]
    q = lambda s, j: to_view(m, s["quantiles"][j]) if np.ndim(s["quantiles"]) == 2 else to_view(m, s["mean"])   # noqa: E731
    return {"target": m.name, "sources": [f.modes[d].name for d in modes], "covariates": bool(covariates and f.L),
            "latent": {"mean": to_view(m, sl["mean"]), "lower": q(sl, 1), "upper": q(sl, 5)}, "measured": {"mean": to_view(m, sm["mean"]), "lower": q(sm, 1), "upper": q(sm, 5)},
            "whole_latent": r["whole_latent"], "whole_measured": r["whole_measured"],
            "hist": np.histogram(np.clip(r["draws_whole_latent"], 0, 1), bins=40, range=(0, 1))[0].astype(float).tolist() if r["draws_whole_latent"].size > 1 else None}


def covariates_view(f: Fitted, idx) -> dict | None:
    if f.L == 0:
        return None
    eff = ST.covariate_effects(f, idx)
    return {"columns": [dict(c) for c in f.cov.columns], "labels": list(f.cov.labels),
            "effects": [{"column": e["column"], "modes": [{"mode": m["mode"], **view_summary(f.modes[f.index(m["mode"])], m)} for m in e["modes"]]} for e in eff],
            "shares": ST.covariate_shares(f, idx)}


def scores_of_training(f: Fitted, idx) -> tuple[np.ndarray, np.ndarray]:
    """Aligned posterior means and standard deviations of the scores of the training samples, from everything each one has (Rao-Blackwellised: the scores are integrated exactly at
    every draw)."""
    from ..core import predict as P
    X = f.cov.X if f.cov is not None else None
    eta, V = P.score_posterior(f, {d: f.Y[d] for d in range(f.D)}, X, idx, n=f.N)
    ea, Va = P.aligned_scores(f, eta, V, idx)
    return ea.mean(axis=0), np.sqrt(np.mean(np.diagonal(Va, axis1=2, axis2=3), axis=0) + ea.var(axis=0))


def surprise_reference(f: Fitted, idx, base: dict) -> tuple[dict, np.ndarray]:
    """For every set of modes a training sample can have: the held-out log predictive density per measured value of the training samples that have them (5 %, 1 % points, median) —
    the reference the saved model compares new samples with."""
    from scipy.special import logsumexp
    summ = LOO.summaries(f)
    X = f.cov.X if f.cov is not None else None
    pres = np.column_stack([s[2] > 0 for s in summ])
    out, values = {}, np.full(f.N, np.nan)
    lw = base["lw"]
    for mask in range(1, 1 << f.D):
        mods = [d for d in range(f.D) if mask >> d & 1]
        have = pres[:, mods].all(axis=1)
        if have.sum() < 3:
            continue
        ll = LOO.loglik(f, summ, mods, idx, X, True)
        nobs = sum(summ[d][2] for d in mods)
        pv = (logsumexp(lw + ll, axis=0)) / np.maximum(nobs, 1)
        key = "+".join(f.modes[d].name for d in mods)
        v = pv[have]
        out[key] = {"q05": float(np.quantile(v, 0.05)), "q01": float(np.quantile(v, 0.01)), "median": float(np.median(v)), "n": int(v.size)}
        if mask == (1 << f.D) - 1:
            values = np.where(have, pv, np.nan)
    return out, values


def assemble(prepared, req, f: Fitted, meta: dict, seconds: float) -> tuple[dict, dict, dict, dict]:
    """``(payload, variables, var_info, summary, reference)`` of a fit (``reference``: what the saved model compares new samples with)."""
    est = req.cfg.estimation
    idx = thin(f.draws.S, VIEW_DRAWS, f.draws.chains)
    diag = meta.get("diagnostics", {})
    verdict = CH.convergence_verdict(diag, est)
    base = LOO.loo_table(f, idx)
    scores, scores_sd = scores_of_training(f, idx)
    pres = prepared.presence()
    summ = LOO.summaries(f)
    nobs = sum(s[2] for s in summ)
    matrix = ST.predictable_matrix(f, idx)
    for row in matrix:
        d = row.pop("draws")
        row["hist"] = np.histogram(np.clip(d, 0, 1), bins=40, range=(0, 1))[0].astype(float).tolist() if d.size > 1 else None
    default_pred = []
    for t in range(f.D):
        others = [d for d in range(f.D) if d != t]
        if others or f.L:
            default_pred.append(predictable_view(f, t, others, True, idx))
    payload = {
        "version": 1,
        "estimation": est, "K": f.K, "n_draws": int(f.draws.S), "n_view_draws": int(idx.size), "chains": int(f.draws.chains),
        "modes": [mode_info(m, int(pres[:, d].sum())) for d, m in enumerate(f.modes)],
        "data": prepared.summary(),
        "diagnostics": {**{k: v for k, v in diag.items() if k != "starts"}, "verdict": verdict, "lp": meta.get("lp"), "starts": diag.get("starts")},
        "factors": factors_view(f, idx),
        "dimension": ST.effective_dimension(f, idx) if f.K > 1 or f.draws.S > 1 else None,
        "variance": variance_view(f, idx),
        "matrix": matrix,
        "predictable": default_pred,
        "covariates": covariates_view(f, idx),
        "loo": {k: base[k] for k in ("elpd", "se", "p_loo", "pointwise", "pareto_k", "loo")} | {"verdict": CH.sample_verdicts(base, f.ids), "per_value": base["pointwise"] / np.maximum(nobs, 1)},
        "samples": {"ids": list(f.ids), "present": pres.astype(np.uint8), "scores": scores, "scores_sd": scores_sd, "n_obs": nobs.astype(float)},
        "basis_check": [{k: (v if k not in ("ratio", "outside") else to_view(f.modes[f.index(b["mode"])], v)) for k, v in b.items()} for b in CH.basis_adequacy(f, idx)],
        "notes": list(prepared.notes),
        "seconds": seconds, "timing": meta.get("info", {}),
        "compare": {},
    }
    surprise, sv = surprise_reference(f, idx, base)
    reference = {"ids": list(f.ids), "scores": scores, "scores_sd": scores_sd, "surprise": surprise, "surprise_values": sv}
    variables, var_info = {}, {}
    for k in range(f.K):
        share = payload["factors"][k]["share"]["mean"]
        variables[f"mblf_f{k + 1}"] = scores[:, k]
        variables[f"mblf_f{k + 1}_sd"] = scores_sd[:, k]
        var_info[f"mblf_f{k + 1}"] = {"kind": "numeric", "label": f"Latent factor {k + 1} (score; {100 * share:.0f} % of the shared variation)",
                                      "description": f"The posterior mean of the score of factor {k + 1}, from everything the sample has: how strongly the pattern of factor {k + 1} is present."}
        var_info[f"mblf_f{k + 1}_sd"] = {"kind": "numeric", "label": f"Latent factor {k + 1} (uncertainty of the score)",
                                         "description": f"The posterior standard deviation of the score of factor {k + 1}: large for samples whose measurements say little about it."}
    variables["mblf_loo"] = base["pointwise"] / np.maximum(nobs, 1)
    var_info["mblf_loo"] = {"kind": "numeric", "label": "Latent factors: how typical (held-out log density per value)",
                            "description": "The leave-one-out log predictive density of the sample's measurements, per measured value: low = the model finds the sample surprising."}
    if np.isfinite(base["pareto_k"]).any():
        variables["mblf_pareto_k"] = base["pareto_k"]
        var_info["mblf_pareto_k"] = {"kind": "numeric", "label": "Latent factors: influence of the sample (Pareto k)",
                                     "description": "Above 0.7 the model changes noticeably without this sample: it is influential or unusual."}
    summary = {"K": f.K, "estimation": est, "modes": [m.name for m in f.modes], "n_samples": f.N, "converged": verdict["level"] != "bad", "verdict": verdict["level"],
               "elpd_loo": base["elpd"], "covariates": [] if f.cov is None else list(f.cov.labels)}
    return payload, variables, var_info, summary, reference
