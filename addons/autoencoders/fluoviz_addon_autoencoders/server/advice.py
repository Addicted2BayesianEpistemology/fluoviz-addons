"""``GET /defaults`` and ``POST /check``: facts and advice from the data, and what a request would do (nothing is trained)."""
from __future__ import annotations

import numpy as np
from fastapi import HTTPException

from fluoviz.analysis.fpca.variables import common_samples
from fluoviz.server.analysis import current_view, project

from ..core import advice as CA
from ..core.spec import MIN_SAMPLES, SIZES, Config
from . import prep
from .request import Request

TARGET_PER_CODE = 10                 # advised training samples per code
MAX_LATENT_ADVICE = 8


def components_99(variables: list) -> int | None:
    """Number of principal components that hold 99 % of the variation of the samples (each variable counts equally; gaps filled with the mean): the PCA on the observed pixels."""
    ids = common_samples(variables)
    if len(ids) < 4:
        return None
    blocks = []
    for v in variables:
        v = v.subset(ids)
        Y = np.asarray(v.Y, float)
        mu = np.nanmean(Y, axis=0)
        Y = np.where(np.isfinite(Y), Y, mu) - mu
        Yw = Y * np.sqrt(v.domain.w)
        tot = float((Yw ** 2).sum())
        if tot <= 0:
            continue
        blocks.append(Yw / np.sqrt(tot))
    if not blocks:
        return None
    M = np.hstack(blocks)
    s = np.linalg.svd(M, compute_uv=False)
    c = np.cumsum(s ** 2) / max(float((s ** 2).sum()), 1e-300)
    return int(np.searchsorted(c, 0.99 - 1e-12) + 1)


def components_99_fpca(prepared) -> int | None:
    """The same for the functional PCA of the prepared elements (the smoothed functions)."""
    from fluoviz.analysis.fpca.fit import fit_fpca
    els = [b.element for b in prepared.blocks if b.element is not None]
    if len(els) != len(prepared.blocks):
        return None
    fit = fit_fpca(els, "variance", "auto", 0.99, 40)
    return int(fit.n_components)


def estimate(cfg: Config, n: int, P: int) -> dict:
    sec = CA.estimate_seconds(cfg, n, P)
    return {"seconds": float(sec), "words": CA.words_for_seconds(sec) + " on this computer (early stopping may make it shorter)"}


def warnings_for(cfg: Config, n: int, P: int, params: int, summary: dict | None = None, masked_fraction: float = 0.0) -> tuple[list[str], list[str]]:
    """``(warnings, blocking)`` in plain English."""
    warn, block = [], []
    if n < MIN_SAMPLES:
        block.append(f"Only {n} samples are included: an autoencoder needs at least {MIN_SAMPLES} samples to be trained honestly. Add more samples, or use the PCA / functional PCA pages for very small data sets.")
    elif n < 30:
        warn.append(f"Only {n} samples: the network is checked by cross-validation and its numbers are rough. Expect the codes to be less reproducible than with 100 or more samples.")
    n_tr = n if n < cfg.kfold_below else n - max(5, round(cfg.val_fraction * n))
    if params > n_tr:
        warn.append(f"A network with {params:,} parameters is trained on {n_tr} samples: it can memorise the samples. The small size, weight decay and early stopping are on by default; trust the held-out numbers.")
    per = n_tr / max(cfg.latent_dim, 1)
    if per < TARGET_PER_CODE:
        warn.append(f"Only {per:.1f} training samples per code (at least {TARGET_PER_CODE} are advised): use fewer codes or more samples.")
    if cfg.latent_dim >= P:
        block.append(f"The number of codes ({cfg.latent_dim}) must be smaller than the number of inputs ({P}).")
    if masked_fraction > 0.5:
        warn.append("More than half of the EEM is masked (scatter, regions): the network works on the pixels that remain.")
    return warn, block


def check_view(p, req: Request, progress) -> dict:
    """The job result of ``POST /check``."""
    prepared = prep.get_prepared(p, req, progress)
    cfg = req.cfg
    a = CA.advise(prepared, cfg)
    n, P = prepared.n, prepared.P
    warn, block = list(a["warnings"]), list(a["blocking"])
    if cfg.allow_few_samples and MIN_SAMPLES > n >= 4:
        block = [b for b in block if "needs at least" not in b]
        warn.append(f"Only {n} samples: training goes on because you chose to, but the numbers will be unreliable.")
    for note in prepared.notes:
        if "not observed" in note and note not in warn:
            warn.append(note)
    if cfg.smoothify > 0 and prepared.mode == "pixels":
        warn.append("smoothify has no smoothness defined on the pixels: it acts as a plain ridge penalty on the reconstruction.")
    st = prep.torch_status()
    if not st["available"]:
        block.append(st["words"])
    return {"ok": not block, "n": n, "dropped": [list(x) for x in prepared.dropped], "input": {"variables": prepared.names, "mode": prepared.mode, "n_inputs": P, "blocks": prepared.summary()["blocks"]},
            "network": {"family": cfg.family, "latent_dim": cfg.latent_dim, "hidden": cfg.hidden_widths(), "activation": cfg.activation, "n_params": a["n_params"],
                        "samples_per_param": a["samples_per_param"], "samples_per_latent": a["samples_per_latent"]},
            "validation": a["validation"], "estimate": {"seconds": a["estimate"]["seconds"], "words": a["estimate"]["words"] + " on this computer (early stopping may make it shorter)"},
            "warnings": warn, "blocking": block, "torch": st, "notes": list(prepared.notes), "advice": {"components_99": a["components_99"], "latent_dim": a["latent_dim"], "latent_reason": a["latent_reason"]}}


def defaults_view(p, names: list[str], min_coverage: float, abs_processed: bool, advise: bool) -> dict:
    """The answer of ``GET /defaults``."""
    from fluoviz.server.routers import fpca as R

    out = R.data(advise=advise, min_coverage=min_coverage, abs_processed=abs_processed)
    n_view = int(out["n_samples"])
    want_fl = "fluorescence" in names
    use_abs = "absorbance" in names and bool(out["absorbance"].get("available")) and out["absorbance"].get("n", 0) >= 4
    if not want_fl and not use_abs:
        raise HTTPException(400, "No absorbance spectra are loaded yet (or fewer than 4 samples have one): add them first (Absorbance page or Corrections page), or model the fluorescence.")
    inp_names = (["fluorescence"] if want_fl else []) + (["absorbance"] if use_abs else [])
    gp = {"kind": "gp", "kernel": "matern52", "structure": "separable", "lengths": "auto", "var_fraction": 0.999}
    inp = {"variables": inp_names}
    if want_fl:
        inp["fluorescence"] = {"transform": "none", "min_coverage": min_coverage, "basis": dict(gp), "smoothing": {"method": "ml"}}
    if use_abs:
        inp["absorbance"] = {"transform": "none", "processed": abs_processed, "basis": {"kind": "gp", "lengths": "auto"}, "smoothing": {"method": "ml"}}
    n = (int(out["absorbance"].get("n_common", n_view)) if want_fl else int(out["absorbance"]["n"])) if use_abs else int(out["fluorescence"]["n"])
    variables = prep.make_variables(p, R_check_input(inp))
    k99 = components_99(variables)
    k99f = None
    if advise:
        try:
            req = Request(cfg=Config(input=R_check_input(inp)), input=R_check_input(inp))
            k99f = components_99_fpca(prep.get_prepared(p, req))
        except Exception:                                                              # noqa: BLE001 - the number is advice; failing to compute it must not break the page
            k99f = None
    k = max(2, min(k99 or MAX_LATENT_ADVICE, n // TARGET_PER_CODE, MAX_LATENT_ADVICE))
    size = "small" if n <= 300 else "medium"
    cfg = Config(latent_dim=k, hidden=size, input=R_check_input(inp))
    P_guess = 80 * len(inp_names)                                                   # a guess before the basis exists (about 80 coordinates per variable); `check` gives the exact number
    warn, block = warnings_for(cfg, n, P_guess, CA.count_parameters(P_guess, k, SIZES[size]), masked_fraction=float(out["fluorescence"].get("masked_fraction", 0.0)) if want_fl else 0.0)
    if use_abs and want_fl and out["absorbance"].get("n_common", 0) < n_view:
        warn.append(f"{n_view - int(out['absorbance']['n_common'])} sample(s) have no absorbance spectrum and cannot be used when both variables are modelled.")
    reason = (f"{k99} components describe 99 % of the variation (PCA); one code per ten samples at most (n = {n}) and no more than {MAX_LATENT_ADVICE}." if k99 else
              f"At most {MAX_LATENT_ADVICE} codes, and one per ten samples (n = {n}).")
    request = {"family": "ae", "latent_dim": int(k), "hidden": size, "size": size, "activation": "silu", "smoothify": 0.0, "input": inp}
    return {**out, "n_common": n, "components_99": {"pca": k99, "fpca": k99f},
            "advice": {"latent_dim": int(k), "latent_reason": reason, "size": size, "hidden": list(SIZES[size]), "family": "ae", "estimate": estimate(cfg, n, P_guess),
                       "representation": out.get("advice"), "warnings": warn, "blocking": block},
            "request": request, "torch": prep.torch_status(), "activations": ["silu", "tanh", "gelu", "relu", "softplus", "elu", "linear"], "sizes": {k_: list(v) for k_, v in SIZES.items()},
            "versions": versions()}


def R_check_input(inp: dict) -> dict:
    from .request import check_input
    return check_input(inp)


def versions() -> dict:
    from .. import __version__
    import fluoviz
    return {"addon": __version__, "fluoviz": getattr(fluoviz, "__version__", None)}
