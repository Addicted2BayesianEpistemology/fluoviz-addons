"""The views of a result computed on demand (``docs/api.md`` section 4): the prediction of one mode of one sample from any choice of sources (in-sample, or held out with the
leave-one-out weights), the predictable fraction for any target and sources, the correlation of one point with a whole mode, the comparison of every source set (a job), the table
of the samples and the exports."""
from __future__ import annotations

import csv
import io
import time

import numpy as np
from fastapi import HTTPException
from scipy.special import logsumexp

from ..core import loo as LOO
from ..core import predict as P
from ..core.posterior import thin
from ..core.spec import MblfError
from ..vendor.mblf.diagnostics import psis
from . import payload as PL
from .results import model_of

VIEW_DRAWS = PL.VIEW_DRAWS


def fitted_of(r):
    return model_of(r).f


def _mode_index(f, name: str) -> int:
    try:
        return f.index(name)
    except KeyError:
        raise HTTPException(400, f"This model has no mode '{name}': it has {', '.join(m.name for m in f.modes)}.") from None


def parse_sources(f, t: int, sources: str | None) -> list[int]:
    """``"auto"`` / None = every other mode; ``"none"`` = nothing; otherwise names separated by commas."""
    if sources in (None, "", "auto"):
        return [d for d in range(f.D) if d != t]
    if sources == "none":
        return []
    out = []
    for s in sources.split(","):
        s = s.strip()
        if not s:
            continue
        d = _mode_index(f, s)
        if d == t:
            raise HTTPException(400, "The mode to predict cannot also be a source: choose the other modes as sources.")
        out.append(d)
    return sorted(set(out))


def _sample_index(f, sid: str) -> int:
    try:
        return f.ids.index(sid)
    except ValueError:
        raise HTTPException(404, f"The sample '{sid}' is not one of the samples of this model.") from None


def held_out_weights(f, n: int, idx: np.ndarray) -> tuple[np.ndarray, float]:
    """The PSIS weights of the draws ``idx`` that leave sample ``n`` out, and their Pareto k (relative efficiency taken as 1)."""
    summ = LOO.summaries(f)
    X = f.cov.X if f.cov is not None else None
    ll = LOO.loglik(f, summ, list(range(f.D)), idx, X, True, subjects=[n])[:, [n]]
    if ll.shape[0] < 20:
        return np.full(ll.shape[0], 1.0 / ll.shape[0]), float("nan")
    lw, k = psis(-ll)
    return np.exp(lw[:, 0] - logsumexp(lw[:, 0])), float(k[0])


def sample_view(p, r, sid: str, target: str | None, sources: str | None, covariates: bool, noisy: bool, held_out: bool) -> dict:
    f = fitted_of(r)
    n = _sample_index(f, sid)
    t = _mode_index(f, target) if target else next((d for d, m in enumerate(f.modes) if m.kind == "surface"), 0)
    src = parse_sources(f, t, sources)
    idx = thin(f.draws.S, VIEW_DRAWS, f.draws.chains)
    w, k = (None, float("nan"))
    if held_out and f.draws.S > 1:
        w, k = held_out_weights(f, n, idx)
    X = f.cov.X[[n]] if (f.cov is not None and covariates) else None
    has = {d: bool(np.isfinite(f.Y[d][:, n]).any()) for d in range(f.D)}
    used = [d for d in src if has[d]]
    pr = P.predict(f, t, {d: f.Y[d][:, [n]] for d in used}, X, idx, noisy=noisy, w=w)
    mode = f.modes[t]
    observed = f.Y[t][:, n]
    obs_data = mode.to_data(observed) if np.isfinite(observed).any() else None
    bt = pr["budget_total"]
    parts = "; ".join(f"{100 * bt[k_]:.0f} % from {P.PART_WORDS[k_]}" for k_ in P.PARTS if bt[k_] > 0.005)
    what = " and ".join([f.modes[d].label for d in used] + (["the covariates"] if X is not None else [])) or "nothing about this sample (the population alone)"
    lacking = [f.modes[d].label for d in src if not has[d]]
    text = (f"The {mode.label} of {sid} predicted from {what}{' (held out: the fit without this sample)' if w is not None else ''}. Its uncertainty comes {parts}." +
            (f" The sample has no {', '.join(lacking)}, so it could not be used." if lacking else ""))
    cover = None
    if obs_data is not None:
        ok = np.isfinite(observed)
        lo, hi = pr["quantiles"][0][ok], pr["quantiles"][-1][ok]
        cover = float(np.mean((obs_data[ok] >= lo) & (obs_data[ok] <= hi)))
    return {"id": sid, "target": mode.name, "sources": [f.modes[d].name for d in used], "covariates": X is not None, "noisy": bool(noisy), "held_out": w is not None, "pareto_k": k,
            "text": text, "observed": None if obs_data is None else PL.to_view(mode, obs_data), "inside95": cover,
            "mean": PL.to_view(mode, pr["mean"]), "quantiles": PL.to_view(mode, pr["quantiles"]), "probs": pr["probs"],
            "band": {"level": pr["band"]["level"], "multiplier": pr["band"]["multiplier"], "lower": PL.to_view(mode, pr["band"]["lower"]), "upper": PL.to_view(mode, pr["band"]["upper"])},
            "budget": {k_: PL.to_view(mode, v) for k_, v in pr["budget"].items()}, "budget_total": bt, "p_above": PL.to_view(mode, pr["p_above"]),
            "scores": pr["scores"], "n_draws": pr["n_draws"],
            "sources_data": [{"mode": f.modes[d].name, "values": PL.to_view(f.modes[d], f.modes[d].to_data(f.Y[d][:, n]))} for d in used]}


def predictable_route(r, target: str, sources: str | None, covariates: bool) -> dict:
    f = fitted_of(r)
    t = _mode_index(f, target)
    return PL.predictable_view(f, t, parse_sources(f, t, sources), covariates, thin(f.draws.S, VIEW_DRAWS, f.draws.chains))


def nearest_point(mode, x: float | None, y: float | None, j: int | None) -> int:
    if mode.kind == "vector":
        if j is None or not 0 <= j < mode.M:
            raise HTTPException(400, f"Choose a variable from 0 to {mode.M - 1}.")
        return int(j)
    c = mode.coords
    if mode.kind == "curve":
        if x is None:
            raise HTTPException(400, "Give the wavelength ('x').")
        return int(np.argmin(np.abs(c[:, 0] - x)))
    if x is None or y is None:
        raise HTTPException(400, "Give the excitation ('x') and the emission ('y') wavelengths.")
    d2 = ((c[:, 1] - x) / max(np.ptp(c[:, 1]), 1e-9)) ** 2 + ((c[:, 0] - y) / max(np.ptp(c[:, 0]), 1e-9)) ** 2          # coords are (emission, excitation)
    return int(np.argmin(d2))


def correlation_route(r, a: str, b: str, x: float | None, y: float | None, j: int | None) -> dict:
    from ..core import structure as ST
    f = fitted_of(r)
    da, db = _mode_index(f, a), _mode_index(f, b)
    i = nearest_point(f.modes[da], x, y, j)
    res = ST.correlation_row(f, da, i, db, thin(f.draws.S, VIEW_DRAWS, f.draws.chains))
    mb = f.modes[db]
    c = f.modes[da].coords[i]
    kind = f.modes[da].kind
    where = ({"em": float(c[0]), "ex": float(c[1])} if kind == "surface" else {"x": float(c[0])} if kind == "curve" else
             {"j": i, "label": f.modes[da].column_labels[i] if f.modes[da].column_labels else ""})
    return {"a": a, "b": b, "point": where, "mean": PL.to_view(mb, res["mean"]), "lower": PL.to_view(mb, res["lower"]), "upper": PL.to_view(mb, res["upper"]),
            "p_pos": None if res["p_pos"] is None else PL.to_view(mb, res["p_pos"]), "lfsr": None if res["lfsr"] is None else PL.to_view(mb, res["lfsr"])}


# ----------------------------------------------------------------------------- the comparison of every source set (a job)
def compare_job(p, r, targets: list[str] | None, max_points: int, progress) -> dict:
    f = fitted_of(r)
    idx = thin(f.draws.S, VIEW_DRAWS, f.draws.chains)
    progress(0.02, "Leaving every sample out once (Pareto-smoothed importance sampling)")
    base = LOO.loo_table(f, idx)
    names = targets or [m.name for m in f.modes]
    out = {}
    for j, name in enumerate(names):
        t = _mode_index(f, name)
        sets = LOO.source_sets(f, t)
        sub = progress.sub(0.05 + 0.95 * j / len(names), 0.05 + 0.95 * (j + 1) / len(names))
        sub(0.0, f"The {f.modes[t].label}: comparing {len(sets)} choices of sources")
        rows = LOO.conditional_elpd(f, t, sets, idx, base)
        for i, row in enumerate(rows):
            sub(0.1 + 0.9 * i / len(rows), f"The {f.modes[t].label} from {row['key'].replace('+', ' + ')}: held-out coverage of the intervals")
            cal = LOO.calibration(f, t, row, idx, base, max_points=max_points)
            row.update({k: v for k, v in cal.items() if k != "per_sample"})
            row["sources"] = [f.modes[d].name for d in row.pop("modes")]
            row["pointwise"] = np.asarray(row["pointwise"], float)
            row.pop("per_value", None)
        best = max(rows, key=lambda x: x["elpd"])
        simplest_ok = min((x for x in rows if -x["vs_best"] <= x["vs_best_se"] or x is best), key=lambda x: (len(x["sources"]) + int(x["covariates"]), -x["elpd"]))
        out[name] = {"rows": rows, "best": best["key"], "one_se": simplest_ok["key"], "n_draws": int(idx.size), "loo": base["loo"], "computed": time.strftime("%Y-%m-%d %H:%M")}
    r.payload.setdefault("compare", {}).update(out)
    p.dirty.add("results")
    return {"compare": out}


# ----------------------------------------------------------------------------- tables and files
def samples_view(r) -> dict:
    pl = r.payload
    s = pl["samples"]
    lo = pl["loo"]
    return {"ids": s["ids"], "present": s["present"], "modes": [m["name"] for m in pl["modes"]], "scores": s["scores"], "scores_sd": s["scores_sd"],
            "loo_per_value": lo["per_value"], "pareto_k": lo["pareto_k"]}


def export(r, what: str) -> tuple[bytes, str, str]:
    pl = r.payload
    buf = io.StringIO()
    w = csv.writer(buf)
    if what == "scores":
        s = pl["samples"]
        K = np.asarray(s["scores"]).shape[1]
        w.writerow(["sample"] + [f"factor{k + 1}" for k in range(K)] + [f"factor{k + 1}_sd" for k in range(K)] + ["loo_per_value", "pareto_k"])
        for i, sid in enumerate(s["ids"]):
            w.writerow([sid] + [f"{x:.6g}" for x in np.asarray(s["scores"])[i]] + [f"{x:.6g}" for x in np.asarray(s["scores_sd"])[i]] +
                       [f"{np.asarray(pl['loo']['per_value'])[i]:.6g}", f"{np.asarray(pl['loo']['pareto_k'])[i]:.4g}"])
        return buf.getvalue().encode(), f"{r.id}-scores.csv", "text/csv"
    if what == "matrix":
        w.writerow(["target", "sources", "covariates", "R2_latent_mean", "R2_latent_lower90", "R2_latent_upper90", "R2_measured_mean"])
        for row in pl["matrix"]:
            w.writerow([row["target"], "+".join(row["sources"]) or "nothing", int(row["covariates"]), f"{row['latent']['mean']:.5g}", f"{row['latent']['lower']:.5g}",
                        f"{row['latent']['upper']:.5g}", f"{row['measured']['mean']:.5g}"])
        return buf.getvalue().encode(), f"{r.id}-what-tells-what.csv", "text/csv"
    if what == "compare":
        w.writerow(["target", "sources", "elpd_loo", "se", "gain_vs_nothing", "gain_se", "coverage50", "coverage80", "coverage95", "rmse", "mean_width80", "n_samples"])
        for tgt, block in (pl.get("compare") or {}).items():
            for row in block["rows"]:
                cov = row.get("coverage") or {}
                w.writerow([tgt, row["key"], f"{row['elpd']:.6g}", f"{row['se']:.4g}", f"{row['gain']:.6g}", f"{row['gain_se']:.4g}", cov.get("0.5", ""), cov.get("0.8", ""),
                            cov.get("0.95", ""), f"{row.get('rmse', float('nan')):.5g}", f"{row.get('mean_width80', float('nan')):.5g}", row.get("n_samples", "")])
        return buf.getvalue().encode(), f"{r.id}-prediction-comparison.csv", "text/csv"
    if what == "loadings":
        f = fitted_of(r)
        idx = thin(f.draws.S, VIEW_DRAWS, f.draws.chains)
        w.writerow(["mode", "point", "x", "y", "factor", "mean", "lower95", "upper95"])
        for d, m in enumerate(f.modes):
            F = f.loading_functions(d, True, True, idx)
            q = np.quantile(F, [0.025, 0.975], axis=0) if F.shape[0] > 1 else np.stack([F[0], F[0]])
            for k in range(f.K):
                for i in range(m.M):
                    c = m.coords[i]
                    w.writerow([m.name, i + 1, f"{c[-1]:.6g}", f"{c[0]:.6g}" if c.size > 1 else "", k + 1, f"{F[:, k, i].mean():.6g}", f"{q[0, k, i]:.6g}", f"{q[1, k, i]:.6g}"])
        return buf.getvalue().encode(), f"{r.id}-loadings.csv", "text/csv"
    raise HTTPException(400, "Choose what to export: scores, matrix, compare or loadings.")


def check_view(p, req, progress) -> dict:
    """What a request would do, without fitting: the modes, their bases, the samples, the covariates, and a rough idea of the time."""
    from . import prep
    try:
        prepared = prep.get_prepared(p, req, progress)
    except MblfError:
        raise
    s = prepared.summary()
    P_ = sum(m.p for m in prepared.modes)
    n_par = P_ * (prepared.N + req.cfg.K + 1) + prepared.N * req.cfg.K
    s["n_parameters"] = int(n_par)
    s["words"] = (f"{prepared.N} samples, {len(prepared.modes)} mode(s), {P_} basis coefficients in total: the sampler explores about {n_par:,} unknowns "
                  f"({'a few minutes' if n_par < 20000 else 'several minutes' if n_par < 60000 else 'a long time: consider fewer basis functions or the quick setting first'}).")
    try:
        req.cfg.check_samples(prepared.N)
        s["problem"] = None
    except MblfError as exc:
        s["problem"] = str(exc)
    return s
