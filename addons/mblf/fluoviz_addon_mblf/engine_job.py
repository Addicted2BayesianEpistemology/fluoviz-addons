"""The fit, run INSIDE the private Stan engine (``core/engine.py``): ``python engine_job.py <job folder>``.

Standalone on purpose: it imports nothing of FluoViz or of the add-on, only the vendored ``mblf`` (this folder's ``vendor/``), numpy, scipy and, through mblf, cmdstanpy and xarray.

Input (written by the server): ``input.json`` (``task``: ``bayes`` | ``ml``, the number of factors, the priors, the sampler settings) and ``input.npz`` (per mode ``B<d>`` ``(M, p)``,
``Y<d>`` ``(M, N)`` standardised, ``C<d>`` ``(M, q)`` coordinates; ``X`` ``(N, L)`` covariates).  Output: ``output.npz`` (the draws, thinned per chain to at most ``keep``) and
``output.json`` (diagnostics, timings), written completely before the last progress line ``@@ 1.0 done``; a failure writes ``error.json`` (``message`` in plain English, ``detail``).
Progress: lines ``@@ <fraction> <text>``; CmdStan's own console lines (``Chain [c] Iteration: ...``) are passed through and read by the parent.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor"))


def say(frac: float, text: str) -> None:
    print(f"@@ {frac:.4f} {text}", flush=True)


def thin_per_chain(n_draws: int, n_chains: int, keep: int):
    import numpy as np
    per = max(1, int(keep) // max(n_chains, 1))
    if n_draws <= per:
        return np.arange(n_draws)
    return np.unique(np.round(np.linspace(0, n_draws - 1, per)).astype(int))


def quantity_table(fit) -> list[dict]:
    """Per rotation-invariant quantity: the largest R-hat and the smallest bulk / tail ESS (the convergence checks of the paper's appendix)."""
    import numpy as np
    from mblf import diagnostics as dg
    words = {"lp__": "log density", "psi2": "noise variances", "tau": "residual precisions", "theta": "basis coefficients of the samples", "shared_variance": "shared variance of every coefficient",
             "shared_variance_total": "shared variance of every mode", "factor_strengths": "strengths of the factors", "cross_covariance_norm": "size of the cross-covariance between modes",
             "covariate_effects": "effects of the covariates"}
    rows = []
    for name, v in fit.identified_quantities().items():
        rh, eb, et = dg.rhat(v), dg.ess_bulk(v), dg.ess_tail(v)
        rows.append({"quantity": name, "label": words.get(name, name), "n": int(np.size(rh)), "max_rhat": float(np.nanmax(rh)) if np.isfinite(rh).any() else None,
                     "min_ess_bulk": float(np.nanmin(eb)) if np.isfinite(eb).any() else None, "min_ess_tail": float(np.nanmin(et)) if np.isfinite(et).any() else None})
    return rows


def run_bayes(inp: dict, arrays, job: Path) -> tuple[dict, dict]:
    import numpy as np
    import mblf
    from mblf import diagnostics as dg
    D = int(inp["D"])
    modes = [mblf.FunctionalData(arrays[f"C{d}"], arrays[f"Y{d}"]).set_basis(arrays[f"B{d}"]) for d in range(D)]
    X = arrays["X"] if int(inp["L"]) > 0 else None
    model = mblf.MvtBLF(n_components=int(inp["K"]), **inp["priors"])
    say(0.02, "Compiling the Stan program (the first time only: about a minute)")
    mblf.stan_model("mblf")
    say(0.06, "Starting the chains")
    fit = model.fit(modes, X=X, chains=int(inp["chains"]), iter_warmup=int(inp["warmup"]), iter_sampling=int(inp["draws"]), max_treedepth=int(inp["max_treedepth"]),
                    adapt_delta=float(inp["adapt_delta"]), seed=int(inp["seed"]), output_dir=str(job / "stan"), show_console=True)
    say(0.9, "Checking the convergence of the chains")
    table = quantity_table(fit)
    e = dg.ebfmi(fit.sample_stats["energy"])
    diag = {"n_chains": int(fit.info["chains"]), "n_warmup": int(fit.info["iter_warmup"]), "n_draws": int(fit.draws["lp__"].shape[1]),
            "max_rhat": float(max(r["max_rhat"] for r in table if r["max_rhat"] is not None)),
            "min_ess_bulk": float(min(r["min_ess_bulk"] for r in table if r["min_ess_bulk"] is not None)),
            "min_ess_tail": float(min(r["min_ess_tail"] for r in table if r["min_ess_tail"] is not None)),
            "n_divergent": int(fit.sample_stats["divergent"].sum()), "divergent_per_chain": [int(x) for x in fit.sample_stats["divergent"].sum(axis=1)],
            "n_max_treedepth": int((fit.sample_stats["treedepth"] >= fit.info["max_treedepth"]).sum()), "ebfmi": [float(x) for x in e],
            "stepsize": [float(x) for x in fit.sample_stats["stepsize"][:, -1]], "mean_leapfrog": float(fit.sample_stats["n_leapfrog"].mean()),
            "n_noncentered": int(np.sum(fit.noncentered)), "table": table}
    say(0.95, "Keeping the draws")
    C, S = fit.draws["lp__"].shape
    idx = thin_per_chain(S, C, int(inp["keep"]))
    out = {k: np.asarray(fit.draws[k])[:, idx] for k in ("Lambda", "tau", "psi2", "regr_coeffs", "eta", "lp__")}
    out["lp_full"] = np.asarray(fit.draws["lp__"])
    out["divergent"] = np.asarray(fit.sample_stats["divergent"])[:, idx]
    info = {"seconds_compile": float(fit.info["seconds_prepare_compile"]), "seconds_sampling": float(fit.info["seconds_sampling"]), "seconds_total": float(fit.info["seconds_total"]),
            "kept_per_chain": int(idx.size), "thin": int(S // max(idx.size, 1))}
    return out, {"diagnostics": diag, "info": info}


def run_ml(inp: dict, arrays, job: Path) -> tuple[dict, dict]:
    import numpy as np
    import mblf
    D = int(inp["D"])
    modes = [mblf.FunctionalData(arrays[f"C{d}"], arrays[f"Y{d}"]).set_basis(arrays[f"B{d}"]) for d in range(D)]
    X = arrays["X"] if int(inp["L"]) > 0 else None
    say(0.02, "Compiling the Stan program (the first time only: about a minute)")
    mblf.stan_model("mblf_ml")
    say(0.1, f"Maximising the likelihood from {int(inp['ml_starts'])} starting points")
    res = mblf.fit_ml(modes, X=X, n_components=int(inp["K"]), n_starts=int(inp["ml_starts"]), seed=int(inp["seed"]))
    K, L = int(inp["K"]), int(inp["L"])
    Lam = np.vstack(res.Lambda)
    s2 = np.concatenate(res.sigma2)
    out = {"Lambda": Lam[None, None], "tau": (1.0 / np.maximum(s2, 1e-300))[None, None], "psi2": np.asarray(res.psi2, float)[None, None],
           "regr_coeffs": np.asarray(res.beta, float).reshape(K, L)[None, None], "eta": np.asarray(res.factor_scores, float).T[None, None], "lp__": np.array([[res.loglik]])}
    diag = {"loglik": float(res.loglik), "aic": float(res.aic), "bic": float(res.bic), "n_params": int(res.n_params), "n_effective_params": int(res.n_effective_params),
            "converged": bool(res.converged), "n_starts": len(res.starts), "n_starts_at_optimum": int(res.n_starts_at_optimum), "at_variance_floor": list(res.at_variance_floor),
            "starts": [{k: v for k, v in s.items()} for s in res.starts]}
    return out, {"diagnostics": diag, "info": {"seconds_total": float(res.seconds)}}


def main(job: Path) -> int:
    import numpy as np
    try:
        inp = json.loads((job / "input.json").read_text(encoding="utf-8"))
        arrays = dict(np.load(job / "input.npz"))
        t0 = time.time()
        out, meta = (run_bayes if inp["task"] == "bayes" else run_ml)(inp, arrays, job)
        meta["seconds"] = time.time() - t0
        np.savez(job / "output.npz", **out)
        (job / "output.json").write_text(json.dumps(meta, default=float), encoding="utf-8")
        say(1.0, "done")
        return 0
    except Exception as exc:                                             # noqa: BLE001 - every failure must reach the person as a sentence
        text = traceback.format_exc()
        msg = str(exc)
        if "compil" in text.lower() and ("make" in text.lower() or "error" in text.lower()):
            message = "The Stan program could not be compiled by the engine. Remove the Stan engine and prepare it again; if it fails again, report it."
        elif "Initialization failed" in text or "Rejecting initial value" in text:
            message = "The sampler could not start: the data or the priors are extreme. Check the pre-treatment (very large values?) and the priors."
        else:
            message = f"The fit failed: {msg[:300]}"
        (job / "error.json").write_text(json.dumps({"message": message, "detail": text[-6000:]}), encoding="utf-8")
        print(text, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
