"""Facts and advice from the data (``GET /defaults``) and what a request would do (``POST /check``)."""
from __future__ import annotations

import numpy as np

from fluoviz.analysis.fpca.variables import DEFAULT_MIN_COVERAGE
from fluoviz.server.analysis import current_view

from ..core import advice as A
from ..core.metric import fit_metric
from ..core.spec import METRIC_WORDS, MIN_SAMPLES, Config
from . import prep
from .request import Request, check_input


def recommended_request(variables: list, kind: str = "functional") -> dict:
    """The complete recommended request for ``variables`` (the page's "Run with the recommended settings")."""
    inp = {"kind": kind, "variables": list(variables)}
    for n in variables:
        inp[n] = {"transform": "none", "gaps": "auto", "basis": {"kind": "gp", "lengths": "auto"}, "smoothing": {"method": "ml"}}
        if n == "fluorescence":
            inp[n]["min_coverage"] = DEFAULT_MIN_COVERAGE
        else:
            inp[n]["processed"] = True
    return {"input": inp, "metric": {"kind": "l2"}, "map": {"size": "standard", "lattice": "hexagonal", "shape": "sheet"}, "train": {"mode": "batch", "init": "linear", "neighbourhood": "gaussian"},
            "validation": {"folds": 5, "limit": 99}, "seed": 0}


def defaults_view(p, names: list, min_coverage: float, abs_processed: bool, advise: bool) -> dict:
    ids = list(p.included_ids())
    out: dict = {"n_samples": len(ids), "ids": ids}
    n_common = len(ids)
    try:
        view = current_view()
        out["fluorescence"] = {"available": True, "n": len(view.ids), "grid": [len(view.em), len(view.ex)], "units": view.units, "masked_fraction": float(1 - np.isfinite(view.X[0]).mean())}
        fl_ids = set(view.ids)
    except Exception as exc:                                                  # noqa: BLE001 - fewer than 3 samples and the like
        out["fluorescence"] = {"available": False, "problem": str(getattr(exc, "detail", exc))}
        fl_ids = set()
    if p.abs is not None:
        wl, Aa, aids = p.abs_view()
        out["absorbance"] = {"available": True, "n": len(aids), "n_wl": int(len(wl))}
        ab_ids = set(aids)
    else:
        out["absorbance"] = {"available": False}
        ab_ids = set()
    common = set(ids)
    if "fluorescence" in names:
        common &= fl_ids
    if "absorbance" in names:
        common &= ab_ids
    n = len(common)
    out["n_common"] = n
    blocking, warnings = [], []
    if n < MIN_SAMPLES:
        blocking.append(f"Only {n} samples have everything chosen: a map needs at least {MIN_SAMPLES} (or switch on 'Compute on fewer than {MIN_SAMPLES} samples anyway' in step 6; never fewer than 4).")
    pre = {nm: {"rows": r, "cols": c, "units": r * c, "samples_per_unit": n / (r * c) if n else None, "words": A.PRESET_WORDS[nm]}
           for nm, (r, c) in ((nm, A.shape_for(A.units_for(max(n, 4), nm), 1.5)) for nm in ("compact", "standard", "fine"))}
    advice = {"presets": pre, "side_ratio": None, "warnings": warnings, "blocking": blocking, "metric": "l2", "size": "standard"}
    if n >= 4:
        warnings += A.warnings_for(n, pre["standard"]["rows"], pre["standard"]["cols"], "functional", 200, 1)
    if advise and not blocking:
        body = recommended_request(names)
        if "fluorescence" in names:
            body["input"]["fluorescence"]["min_coverage"] = min_coverage
        if "absorbance" in names:
            body["input"]["absorbance"]["processed"] = abs_processed
        req = _parse(body)
        prepared = prep.get_prepared(p, req)
        X = prepared.R * prepared.l2_gain()
        advice["presets"] = A.presets(X, prepared.sample_weights("observed"))
        advice["side_ratio"] = advice["presets"].pop("side_ratio")
    out["advice"] = advice
    out["request"] = recommended_request(names)
    out["metrics"] = [{"kind": k, "words": w} for k, w in METRIC_WORDS.items()]
    out["sizes"] = {nm: A.PRESET_WORDS[nm] for nm in ("compact", "standard", "fine")}
    return out


def _parse(body):
    from .request import parse_request
    return parse_request(body)


def check_view(p, req: Request, progress) -> dict:
    """What a request would do (nothing is fitted)."""
    progress(0.02, "Preparing the data")
    prepared = prep.get_prepared(p, req, progress.sub(0.0, 0.7))
    cfg = req.cfg
    n = prepared.n
    blocking, warnings = [], []
    try:
        cfg.check_samples(n)
    except Exception as exc:                                                  # noqa: BLE001
        blocking.append(str(exc))
    progress(0.75, "Fitting the distance")
    sw = prepared.sample_weights(cfg.gap_weighting)
    metric = None
    try:
        metric = fit_metric(prepared.R, cfg.metric_kind, prepared.l2_gain(), prepared.unit_gain(), prepared.q, prepared.prior_scales(), cfg.alpha, cfg.components, sw, cfg.seed)
    except Exception as exc:                                                  # noqa: BLE001
        blocking.append(str(exc))
    rows = cols = None
    if metric is not None:
        X = metric.transform(prepared.R)
        try:
            rows, cols = A.resolve_size(cfg.size, X, sw, cfg.lattice, cfg.shape)
        except Exception as exc:                                              # noqa: BLE001
            blocking.append(str(exc))
    epochs = cfg.epochs_rough + cfg.epochs_fine
    M = (rows or 0) * (cols or 0)
    folds = min(cfg.folds, n)
    secs = epochs * n * max(M, 1) * prepared.P * 6.0 / 3e9 * (1 + folds) + 0.5
    est = {"seconds": float(secs), "words": f"about {max(1, round(secs))} second{'s' if round(secs) != 1 else ''} on this computer" if secs < 90 else f"about {secs / 60:.0f} minutes on this computer"}
    if rows:
        warnings += A.warnings_for(n, rows, cols, prepared.mode, epochs, prepared.P)
    missing = prepared.missing_part()
    if missing is not None:
        warnings.append(f"{int((missing > 0.005).sum())} sample(s) lack part of the domain (a region cut out): " + (
            "their coordinates were estimated from the pixels they have and from the other samples, and they count in proportion to what was measured." if prepared.mode != "pixels" else
            "the distance to a prototype uses the pixels they have."))
    return {"ok": not blocking, "n": n, "dropped": [list(d) for d in prepared.dropped],
            "input": {"variables": prepared.names, "mode": prepared.mode, "n_inputs": prepared.P, "blocks": prepared.summary()["blocks"]},
            "metric": None if metric is None else {**metric.to_dict(), "name": METRIC_WORDS[metric.kind]},
            "map": {"rows": rows, "cols": cols, "units": M, "lattice": cfg.lattice, "shape": cfg.shape, "samples_per_unit": (n / M) if M else None},
            "validation": {"folds": folds, "words": f"{folds}-fold validation: the map is fitted {folds} times, each time without one fifth of the samples, and those samples are placed on it."},
            "estimate": est, "warnings": warnings + list(prepared.notes), "blocking": blocking}
