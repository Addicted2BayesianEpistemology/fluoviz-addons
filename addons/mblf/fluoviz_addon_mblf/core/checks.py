"""The honest checks of a fit, in words (SPEC section 6): did the chains converge, is the basis rich enough, which samples does the model describe poorly.  Pure numpy.

* **Convergence** (Bayesian fits): the largest rank-normalised split R-hat and the smallest bulk / tail effective sample size over the rotation-invariant quantities (mblf), the
  divergent transitions, the tree depth, the E-BFMI.  Thresholds of Vehtari et al. (2021): R-hat <= 1.01, ESS >= 400 (100 per chain at least).
* **Basis adequacy** (mblf ``projection_diagnostics`` / ``projection_reference``): what the basis cannot represent of every sample must look like measurement noise passed through the
  same projection — its energy, its profile over the domain and its lag-1 autocorrelation are compared with simulations of noise at the posterior noise level.
* **Samples**: the PSIS-LOO log predictive density of every sample (low = surprising for the model) and its Pareto k (above 0.7 the leave-one-out estimate is unreliable for that sample,
  which itself says the sample is influential).
"""
from __future__ import annotations

import numpy as np

from .posterior import Fitted, MblfView

RHAT_OK, RHAT_BAD = 1.01, 1.05
ESS_OK = 400
K_OK, K_BAD = 0.7, 1.0


def convergence_verdict(diag: dict, estimation: str) -> dict:
    """``{"level": ok | warn | bad, "text": ..., "items": [...]}`` from the diagnostics of the engine."""
    if estimation == "ml":
        items = []
        lvl = "ok"
        if not diag.get("converged", True):
            items.append("The optimiser did not report convergence at the best starting point.")
            lvl = "warn"
        k, n = diag.get("n_starts_at_optimum", 0), diag.get("n_starts", 0)
        if n and k < 2:
            items.append(f"Only {k} of {n} starting points reached the best likelihood: another, better maximum may exist. Use more starting points.")
            lvl = "warn"
        if diag.get("at_variance_floor"):
            items.append(f"{len(diag['at_variance_floor'])} variance(s) sit at their lower bound (for example {diag['at_variance_floor'][0]}): the model has more freedom than the data need.")
        text = ("The maximum-likelihood fit found the same best point from several starts." if lvl == "ok" else "The maximum-likelihood fit may not be the best one: see below.")
        return {"level": lvl, "text": text + " A maximum-likelihood fit treats the population parameters as known: its bands leave out their uncertainty.", "items": items}
    items, lvl = [], "ok"
    rh, eb, et = diag.get("max_rhat", np.nan), diag.get("min_ess_bulk", np.nan), diag.get("min_ess_tail", np.nan)
    nd, ntd = int(diag.get("n_divergent", 0)), int(diag.get("n_max_treedepth", 0))
    if not rh <= RHAT_OK:
        lvl = "bad" if not rh <= RHAT_BAD else "warn"
        worst = max((r for r in diag.get("table", []) if r.get("max_rhat") is not None), key=lambda r: r["max_rhat"], default=None)
        items.append(f"The chains disagree (largest R-hat {rh:.3f}, should be at most {RHAT_OK})" + (f", worst for the {worst['label']}" if worst else "") +
                     ". Run longer (the thorough setting), or lower the largest number of factors.")
    if not min(eb, et) >= ESS_OK:
        lvl = "warn" if lvl == "ok" else lvl
        items.append(f"Few effective draws (smallest effective sample size {min(eb, et):.0f}, should be at least {ESS_OK}): the bands are rough. Run longer.")
    if nd:
        lvl = "bad" if nd > 0.01 * diag.get("n_draws", 1000) * diag.get("n_chains", 1) else ("warn" if lvl == "ok" else lvl)
        items.append(f"{nd} divergent transition(s): the sampler met a region it could not explore, and the posterior may be biased there. Raise the target acceptance rate "
                     "(adapt_delta, for example 0.95).")
    if ntd:
        lvl = "warn" if lvl == "ok" else lvl
        items.append(f"{ntd} iteration(s) hit the largest tree depth: the sampler was slowed down; results are still valid if R-hat and the effective sample size are fine.")
    e = diag.get("ebfmi") or []
    if e and min(e) < 0.3:
        lvl = "warn" if lvl == "ok" else lvl
        items.append(f"Low E-BFMI ({min(e):.2f} in a chain): the sampler explores the scale of the posterior slowly.")
    text = {"ok": "The chains converged: every check of the sampler is fine, so the bands below can be trusted as summaries of the posterior.",
            "warn": "The chains mostly converged, with the warnings below: the results are usable, but the bands are less precise than they look.",
            "bad": "The chains did NOT converge: do not interpret the results yet. See below what to change, and fit again."}[lvl]
    return {"level": lvl, "text": text, "items": items}


def basis_adequacy(f: Fitted, idx: np.ndarray, n_sim: int = 300, seed: int = 0) -> list[dict]:
    """Per functional mode: mblf's projection residuals against projected noise (summary rows, and the location profile for a picture)."""
    from ..vendor.mblf.secondary import projection_diagnostics, projection_reference
    sub = Fitted(f.modes, f.cov, f.draws.subset(np.asarray(idx, int)), f.Y, f.ids)
    view = MblfView(sub)
    out = []
    for d, m in enumerate(f.modes):
        if m.kind == "vector":
            continue
        try:
            dg = projection_diagnostics(view, d + 1)
            sims = int(max(30, min(n_sim, 2e7 / max(m.M * f.N, 1))))          # the simulated residuals of all samples at once stay below ~160 MB
            ref = projection_reference(dg, n_sim=sims, seed=seed, band=0.9)
        except ValueError as exc:
            out.append({"mode": m.name, "available": False, "text": f"Not available for the {m.label}: {exc}."})
            continue
        s = ref["summary"]
        rows = [{"summary": str(a), "statistic": str(b), "observed": float(c), "expected": float(e), "lower": float(lo), "upper": float(hi), "p_value": float(pv), "n": int(n)}
                for a, b, c, e, lo, hi, pv, n in zip(s["summary"], s["statistic"], s["observed"], s["expected"], s["ref_lower"], s["ref_upper"], s["p_value"], s["n"])]
        loc = ref["locations"]
        ratio = np.full(m.M, np.nan)
        ratio[np.asarray(loc["location"], int) - 1] = np.asarray(loc["mean_sq"], float) / np.asarray(loc["expected"], float)
        outside = np.full(m.M, np.nan)
        outside[np.asarray(loc["location"], int) - 1] = np.asarray(loc["outside"], float)
        bad = [r for r in rows if np.isfinite(r["p_value"]) and r["p_value"] < 0.01]
        if bad:
            text = (f"What the basis of the {m.label} leaves out does NOT look like noise ({bad[0]['statistic']}: {bad[0]['observed']:.3g}, expected about {bad[0]['expected']:.3g}). The "
                    "basis is probably too smooth: keep a larger share of the prior variance, or choose shorter length-scales (step 3).")
            lvl = "warn"
        else:
            text = f"What the basis of the {m.label} leaves out looks like measurement noise: the basis is rich enough for these data."
            lvl = "ok"
        out.append({"mode": m.name, "available": True, "level": lvl, "text": text, "rows": rows, "ratio": ratio, "outside": outside})
    return out


def sample_verdicts(loo: dict, ids: list) -> dict:
    """The samples the model finds surprising (lowest held-out log density per measured value) and those whose leave-one-out estimate is unreliable (Pareto k)."""
    k = np.asarray(loo["pareto_k"], float)
    n_bad = int(np.sum(k > K_OK)) if np.isfinite(k).any() else 0
    text = ("Every sample's leave-one-out estimate is reliable (Pareto k at most 0.7)." if n_bad == 0 else
            f"{n_bad} sample(s) have a Pareto k above 0.7: the model changes noticeably without them (they are influential or unusual), so their leave-one-out numbers are rough. "
            "Look at them in the Samples tab.")
    if not np.isfinite(k).any():
        text = "A maximum-likelihood fit has a single set of parameters, so the leave-one-out numbers are in-sample (optimistic) values."
    return {"n_unreliable": n_bad, "text": text}
