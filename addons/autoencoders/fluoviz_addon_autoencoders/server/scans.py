"""Scans (how many codes?  how much smoothify?  are the codes reproducible?  the linear baselines) and the relation of the codes to other variables.

The scans that train networks run in the worker process (``core.worker``, jobs of ``core.diagnostics``), several networks in ONE process; the baselines and the relation to other variables
are numpy and run in the job thread.  A scan of an existing result can be kept in the result (``payload["scans"][kind]``), so it survives saving the project."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from fastapi import HTTPException

from ..core import diagnostics as DG
from ..core import linear as lin
from ..core import tools as CT
from ..core import worker
from . import prep
from .request import Request, parse_request
from .results import get_result, model_of, stale_words

KINDS = ("latent", "smoothify", "stability", "baseline")
MAX_DIMS = 24
MAX_VALUES = 16


@dataclass
class ScanSpec:
    kind: str
    req: Request
    rid: str | None
    save: bool
    options: dict = field(default_factory=dict)


def _int_list(v, what: str, lo: int, hi: int, maxn: int) -> list[int]:
    if not isinstance(v, (list, tuple)) or not v or len(v) > maxn:
        raise HTTPException(400, f"'{what}' must be a list of 1 to {maxn} whole numbers.")
    out = []
    for x in v:
        if isinstance(x, bool) or not isinstance(x, (int, float)) or abs(float(x) - round(float(x))) > 1e-9 or not lo <= int(round(x)) <= hi:
            raise HTTPException(400, f"'{what}' must hold whole numbers from {lo} to {hi}.")
        out.append(int(round(x)))
    return sorted(set(out))


def parse_scan(p, body) -> ScanSpec:
    """The checked scan request (plain-English 400s)."""
    if not isinstance(body, dict):
        raise HTTPException(400, "The scan must be sent as a group of named values.")
    kind = body.get("scan")
    if kind not in KINDS:
        raise HTTPException(400, f"'scan' must be one of: {', '.join(KINDS)}.")
    rid, request = body.get("rid"), body.get("request")
    if rid is None and request is None:
        raise HTTPException(400, "Say what to scan: give 'rid' (the result whose settings are used) or 'request' (a full set of settings).")
    save = body.get("save", True)
    if not isinstance(save, bool):
        raise HTTPException(400, "'save' must be true or false.")
    if rid is not None:
        if not isinstance(rid, str):
            raise HTTPException(400, "'rid' must be the name of a result.")
        r = get_result(p, rid)
        req = parse_request({**r.params, **({} if request is None else request)})
    else:
        req = parse_request(request)
    opts: dict = {}
    if kind == "latent":
        opts["dims"] = None if body.get("dims") is None else _int_list(body["dims"], "dims", 1, 32, MAX_DIMS)
        fams = body.get("families")
        if fams is not None:
            if not isinstance(fams, list) or not fams or not set(fams) <= {"ae", "vae"}:
                raise HTTPException(400, "'families' must be a list holding ae and / or vae.")
            opts["families"] = list(dict.fromkeys(fams))
    elif kind == "smoothify":
        vals = body.get("values")
        if vals is not None:
            if not isinstance(vals, list) or not vals or len(vals) > MAX_VALUES or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0 for v in vals):
                raise HTTPException(400, f"'values' must be a list of 1 to {MAX_VALUES} numbers, each 0 or more.")
            opts["values"] = sorted({float(v) for v in vals})
    elif kind == "stability":
        n_runs = body.get("n_runs", 5)
        if isinstance(n_runs, bool) or not isinstance(n_runs, int) or not 2 <= n_runs <= 10:
            raise HTTPException(400, "'n_runs' must be a whole number from 2 to 10.")
        boot = body.get("bootstrap", False)
        if not isinstance(boot, bool):
            raise HTTPException(400, "'bootstrap' must be true or false.")
        opts["n_runs"], opts["bootstrap"] = n_runs, boot
    else:
        opts["dims"] = None if body.get("dims") is None else _int_list(body["dims"], "dims", 1, 32, MAX_DIMS)
    if kind in ("latent", "smoothify"):
        nr = body.get("n_restarts", 1)
        if isinstance(nr, bool) or not isinstance(nr, int) or not 1 <= nr <= 5:
            raise HTTPException(400, "'n_restarts' must be a whole number from 1 to 5.")
        opts["n_restarts"] = nr
    return ScanSpec(kind, req, rid, save, opts)


def _plain(o):
    """Numpy scalars -> Python numbers (the scan is kept in a result that is saved as JSON + arrays)."""
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return o


def baseline_scan(td, cfg, dims) -> dict:
    """The linear baselines for every number of components, on the split of the network (no training)."""
    from ..core import train as T                                        # the split only (this module imports PyTorch lazily: nothing is loaded here)
    split = T.make_split(td.n, cfg)
    dims = dims or list(range(1, int(min(12, td.P - 1, max(2, td.n // 3))) + 1))
    dims = [d for d in dims if d < td.P]
    pc = DG.pca_curve(td, split, max(dims))
    name = "PCA" if all(b["kind"] == "pixels" for b in td.blocks) else "Functional PCA"
    val = [pc["error_val"][d] for d in dims]
    best = int(np.nanargmin(val))
    tol = max(0.01 * pc["error_val"][0], pc["error_val_se"][dims[best]] if np.isfinite(pc["error_val_se"][dims[best]]) else 0.0)
    need = next(d for d, v in zip(dims, val) if v <= val[best] + tol)
    return {"scan": "baseline", "name": name, "dims": dims, "val": val, "val_se": [pc["error_val_se"][d] for d in dims], "train": [pc["error_train"][d] for d in dims], "mode": split["mode"],
            "words": f"The {name} needs {need} component{'s' if need != 1 else ''} to get within one standard error (or 1 % of the variation) of its best held-out error ({100 * val[best]:.1f} % of the variation not rebuilt "
                     f"with {dims[best]} components); a network cannot be expected to do much better with fewer codes unless the data are clearly nonlinear."}


def run_scan(p, spec: ScanSpec, progress, history_path) -> dict:
    """The body of the scan job."""
    req = spec.req
    progress(0.01, "Preparing the data")
    prepared = prep.get_prepared(p, req, progress.sub(0.0, 0.1))
    cfg = req.cfg
    cfg.check_samples(prepared.n)
    td = prepared.train_data(cfg.smoothify_metric, cfg.gap_weighting)
    if spec.kind == "baseline":
        out = baseline_scan(td, cfg, spec.options.get("dims"))
    else:
        job = {"latent": "latent_scan", "smoothify": "smoothify_scan", "stability": "stability_scan"}[spec.kind]
        kwargs = {"config": cfg, "data": td, **{k: v for k, v in spec.options.items() if v is not None}}
        out = worker.run_in_worker(job, kwargs, progress=progress.sub(0.1, 0.98), history_path=history_path)
    out = _plain(out)
    out["settings"] = {"latent_dim": cfg.latent_dim, "family": cfg.family, "smoothify": cfg.smoothify, "n": prepared.n, "seed": cfg.seed}
    out["saved_in"] = None
    if spec.kind == "baseline":
        r0 = p.results.get(spec.rid) if spec.rid else None
        q = (r0.payload.get("quality") or {}) if r0 is not None else {}
        out["network"] = {"k": int(r0.payload["latent_dim"]), "val": float(1.0 - q["r2_val"])} if q.get("r2_val") is not None else None      # the network of the result on the same scale (share not rebuilt on held-out samples)
    if spec.rid and spec.save:
        r = p.results.get(spec.rid)
        if r is not None and not p.stale_parts(r):
            r.payload.setdefault("scans", {})[spec.kind] = out
            r._saved = False                                              # write the result again when the project is saved
            p.dirty.add("results")
            out["saved_in"] = spec.rid
    progress(1.0, "Done")
    return out


# ----------------------------------------------------------------------------- the codes against other variables
def relate_view(p, r, body) -> dict:
    """Correlation of the codes with numeric variables of the project and the cross-validated R² of predicting each from the codes, next to the same from as many PCA scores."""
    if not isinstance(body, dict) or not isinstance(body.get("variables"), list) or not body["variables"]:
        raise HTTPException(400, "Choose the variables to compare the codes with ('variables': a list such as meta:DOC, qc:total).")
    keys = [str(k) for k in body["variables"]]
    folds = body.get("folds", 5)
    if isinstance(folds, bool) or not isinstance(folds, int) or not 2 <= folds <= 20:
        raise HTTPException(400, "'folds' must be a whole number from 2 to 20.")
    from fluoviz.server.routers.samples import qc_columns
    qc = qc_columns(p) if any(k.startswith("qc:") for k in keys) else None
    try:
        vals = p.variable_values(keys, qc)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    cat = {v["key"]: v for v in p.variable_catalogue(qc)}
    ids = list(r.sample_ids or [])
    pos = {s: k for k, s in enumerate(vals["ids"])}
    model = model_of(r)
    codes = np.asarray(r.payload["codes"], float)
    targets, labels, skipped = {}, {}, []
    for k in keys:
        col = vals["columns"][k]
        if cat.get(k, {}).get("kind") != "numeric":
            skipped.append(k)
            continue
        y = np.asarray(col, float)
        targets[k] = np.array([y[pos[s]] if s in pos else np.nan for s in ids])
        labels[k] = cat[k]["label"]
    if not targets:
        raise HTTPException(400, "None of these variables is a number, so there is nothing to correlate the codes with" + (f" (not numeric: {', '.join(skipped)})." if skipped else "."))
    pca_scores, note = None, None
    if stale_words(p, r):
        note = "The data changed since this network was trained, so the comparison with the PCA scores is not shown."
    else:
        ev = model.evaluate(prep.apply_data(p, ids, model.names), 0.0)
        pm = lin.fit_pca(ev.X, ev.M, np.asarray(model.arrays["out.weights"], float), model.d)
        pca_scores = pm.project(ev.X, ev.M, model.d)
    res = CT.relate(codes, targets, pca_scores, folds=folds)
    rows = [{"key": k, "label": labels[k], "correlation": res[k]["correlation"], "r2_codes": res[k]["r2_codes"], "r2_pca": res[k]["r2_pca"], "n": res[k]["n"]} for k in targets]
    words = []
    for row in rows:
        if row["r2_codes"] is not None and np.isfinite(row["r2_codes"]):
            w = f"'{row['label']}' can be predicted from the codes with a cross-validated R² of {row['r2_codes']:.2f}"
            if row["r2_pca"] is not None and np.isfinite(row["r2_pca"]):
                w += f" (from the same number of PCA scores: {row['r2_pca']:.2f})"
            words.append(w + ".")
    return {"names": model.code_names, "variables": rows, "skipped": skipped, "words": " ".join(words), "note": note}
