"""The data of the open project as the map sees it: the functional variables (the Functional PCA's own), the prepared coordinates (``core.data.prepare``, cached because the basis and the
smoothing take seconds), a table of other variables, and the samples of a result as an ``ApplyData``.  Modelled on ``server/prep.py`` of the autoencoders add-on."""
from __future__ import annotations

import hashlib
import json
import threading

import numpy as np
from fastapi import HTTPException

from fluoviz.analysis.fpca import absorbance_variable, fluorescence_variable
from fluoviz.analysis.frozen import ApplyData
from fluoviz.core.corrections import absorbance_at
from fluoviz.server.analysis import current_view, project

from ..core import data as cdata
from ..core.spec import SomError
from .request import Request

_CACHE: dict = {"key": None, "prepared": None}
_LOCK = threading.Lock()


def abs_matrix(p, processed: bool):
    wl, A, ids = p.abs_view()
    if processed:
        spec = p.state["corrections"]["ife"]
        A = np.vstack([absorbance_at(wl, row, wl, {**spec, "outside": "edge"}) if np.isfinite(row).sum() >= 2 else row for row in A])
    return wl, A, ids


def make_variables(p, inp: dict) -> list:
    """The functional variables of a request (``inp``: the checked ``input``).  A basis of kind ``pixels`` keeps the gaps inside the domain (the distance ignores them)."""
    out = []
    for n in inp["variables"]:
        v = inp[n]
        if n == "fluorescence":
            view = current_view()
            out.append(fluorescence_variable(view.X, view.ex, view.em, view.ids, transform=v["transform"], min_coverage=v["min_coverage"], units=view.units, complete=False))
        else:
            if p.abs is None:
                raise HTTPException(400, "No absorbance spectra are loaded yet: add them first (Absorbance page or Corrections page).")
            wl, A, ids = abs_matrix(p, v["processed"])
            if len(ids) < 4:
                raise HTTPException(400, f"Only {len(ids)} sample(s) have an absorbance spectrum in the analysis: at least 4 are needed.")
            out.append(absorbance_variable(wl, A, ids, transform=v["transform"], units="absorbance"))
    return out


def ife_of(p, inp: dict) -> dict | None:
    if "absorbance" in inp.get("variables", []) and inp["absorbance"]["processed"]:
        return {**p.state["corrections"]["ife"], "outside": "edge"}
    return None


def table_for(p, keys: list[str]) -> tuple[dict, list]:
    """``({key: (n,) numeric values}, ids)`` of the variables ``keys`` for the included samples (the keys of the variable catalogue: ``res:ID:name``, ``meta:column``, ``qc:key``)."""
    from fluoviz.server.routers.samples import qc_columns
    need_qc = any(k.startswith("qc:") for k in keys)
    try:
        vals = p.variable_values(keys, qc_columns(p) if need_qc else None)["columns"]
    except Exception as exc:                                                  # noqa: BLE001 - ProjectError and the like carry a plain sentence
        raise SomError(str(exc)) from None
    all_ids = list(p.ids)
    pos = {s: k for k, s in enumerate(all_ids)}
    ids = [s for s in p.included_ids() if s in pos]
    cols = {}
    for k in keys:
        v = vals[k]
        arr = np.asarray([np.nan if x is None else x for x in v]) if not isinstance(v, np.ndarray) else v
        try:
            arr = arr.astype(float)
        except (TypeError, ValueError):
            raise SomError(f"The variable '{k}' is not a number (it is a category or text): only numeric variables can be mapped.") from None
        cols[k] = arr[[pos[s] for s in ids]]
    keep = np.isfinite(np.column_stack(list(cols.values()))).any(axis=1)
    ids = [s for s, k in zip(ids, keep) if k]
    cols = {k: v[keep] for k, v in cols.items()}
    return cols, ids


def _key(p, req: Request) -> str:
    h = hashlib.sha1()
    h.update(json.dumps({"input": req.input, "seed": req.cfg.seed, "sig": p.signature(), "res": sorted(p.results)}, sort_keys=True, default=str).encode())
    return h.hexdigest()


def get_prepared(p, req: Request, progress=None):
    """``core.data.Prepared`` of the request on the current data; the last one is kept, so ``fit`` right after ``check`` does not redo the basis and the smoothing."""
    key = _key(p, req)
    with _LOCK:
        if _CACHE["key"] == key and _CACHE["prepared"] is not None:
            return _CACHE["prepared"]
    if req.variables_mode:
        cols, ids = table_for(p, req.input["keys"])
        prepared = cdata.prepare_table(cols, ids, req.input["standardize"], req.cfg.seed)
    else:
        variables = make_variables(p, req.input)
        prepared = cdata.prepare(variables, req.input, progress, seed=req.cfg.seed, ife=ife_of(p, req.input))
    with _LOCK:
        _CACHE["key"], _CACHE["prepared"] = key, prepared
    return prepared


def forget() -> None:
    with _LOCK:
        _CACHE["key"], _CACHE["prepared"] = None, None


def apply_data(p, ids: list[str], names: list[str]) -> ApplyData:
    """The samples ``ids`` of the open project in the state the map of a result (not outdated) reads them."""
    ids = [str(i) for i in ids]
    view = p.view(only=ids)
    A = wl = None
    if "absorbance" in names:
        if p.abs is None:
            raise HTTPException(400, "No absorbance spectra are loaded in this project, but this map reads them.")
        wl, Aall, aids = p.abs_view(include_excluded=True)
        pos = {s: k for k, s in enumerate(aids)}
        A = np.vstack([Aall[pos[s]] if s in pos else np.full(wl.size, np.nan) for s in ids])
    return ApplyData(ids=list(view.ids), X=view.X, ex=view.ex, em=view.em, A=A, abs_wl=wl, units=view.units)
