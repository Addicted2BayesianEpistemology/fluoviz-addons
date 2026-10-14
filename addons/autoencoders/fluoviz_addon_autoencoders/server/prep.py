"""The data of the open project as the network sees it: the functional variables (the Functional PCA's own), the prepared coordinates (``core.data.prepare``, cached because the basis and
the smoothing take seconds), and the samples of a result as an ``ApplyData`` (so that a result is looked at with exactly the code the saved model uses)."""
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
from .request import Request

_CACHE: dict = {"key": None, "prepared": None}
_LOCK = threading.Lock()


def abs_matrix(p, processed: bool):
    """The absorbance spectra of the project as the Functional PCA reads them: ``(wl, A, ids)``, cleaned like the Corrections page does when ``processed``."""
    wl, A, ids = p.abs_view()
    if processed:
        spec = p.state["corrections"]["ife"]
        A = np.vstack([absorbance_at(wl, row, wl, {**spec, "outside": "edge"}) if np.isfinite(row).sum() >= 2 else row for row in A])
    return wl, A, ids


def make_variables(p, inp: dict) -> list:
    """The functional variables of a request (``inp``: the checked ``input``).  Unlike the Functional PCA, a basis of kind ``pixels`` keeps the gaps inside the domain (the network
    ignores them)."""
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
    """The clean-up of the absorbance spectra the data used (``None`` when the absorbance is not used or was not cleaned): kept in the saved model."""
    if "absorbance" in inp["variables"] and inp["absorbance"]["processed"]:
        return {**p.state["corrections"]["ife"], "outside": "edge"}
    return None


def _key(p, req: Request) -> str:
    h = hashlib.sha1()
    h.update(json.dumps({"input": req.input, "seed": req.cfg.seed, "sig": p.signature()}, sort_keys=True, default=str).encode())
    return h.hexdigest()


def get_prepared(p, req: Request, progress=None):
    """``core.data.Prepared`` of the request on the current data; the last one is kept, so ``fit`` right after ``check`` does not redo the basis and the smoothing."""
    key = _key(p, req)
    with _LOCK:
        if _CACHE["key"] == key and _CACHE["prepared"] is not None:
            return _CACHE["prepared"]
    variables = make_variables(p, req.input)
    prepared = cdata.prepare(variables, req.input, progress, seed=req.cfg.seed, ife=ife_of(p, req.input))
    with _LOCK:
        _CACHE["key"], _CACHE["prepared"] = key, prepared
    return prepared


def forget() -> None:
    with _LOCK:
        _CACHE["key"], _CACHE["prepared"] = None, None


def apply_data(p, ids: list[str], names: list[str]) -> ApplyData:
    """The samples ``ids`` of the open project in the state the network of a result (not outdated) reads them: the view's EEMs, the absorbance spectra as the project holds them (the
    saved model cleans them like the training spectra)."""
    ids = [str(i) for i in ids]
    view = p.view(only=ids)
    A = wl = None
    if "absorbance" in names:
        if p.abs is None:
            raise HTTPException(400, "No absorbance spectra are loaded in this project, but this network reads them.")
        wl, Aall, aids = p.abs_view(include_excluded=True)
        pos = {s: k for k, s in enumerate(aids)}
        A = np.vstack([Aall[pos[s]] if s in pos else np.full(wl.size, np.nan) for s in ids])
    return ApplyData(ids=list(view.ids), X=view.X, ex=view.ex, em=view.em, A=A, abs_wl=wl, units=view.units)


def torch_status() -> dict:
    """Is PyTorch installed?  Found without importing it (the server never loads it)."""
    import importlib.util
    ok = importlib.util.find_spec("torch") is not None
    version = None
    if ok:
        try:
            from importlib.metadata import version as _v
            version = _v("torch")
        except Exception:                                                           # noqa: BLE001 - the version is a nicety
            version = None
    return {"available": ok, "version": version,
            "words": "" if ok else "PyTorch is not installed: open Settings, Add-ons, and press Install (or Repair) for Autoencoders."}


def require_torch() -> None:
    st = torch_status()
    if not st["available"]:
        raise HTTPException(400, st["words"])
