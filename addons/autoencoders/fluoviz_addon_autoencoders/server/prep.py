"""The data of the open project as the network sees it: the functional variables (the Functional PCA's own), the prepared coordinates (``core.data.prepare``, cached because the basis and
the smoothing take seconds), and the samples of a result as an ``ApplyData`` (so that a result is looked at with exactly the code the saved model uses).  The part that the self-organising
map add-on shares is in the ``common`` add-on (``fluoviz_addon_common.project_data``)."""
from __future__ import annotations

from fastapi import HTTPException
from fluoviz_addon_common import project_data as C
from fluoviz_addon_common.project_data import abs_matrix, ife_of, make_variables  # noqa: F401  (part of this module's interface)

from fluoviz.analysis.frozen import ApplyData

from ..core import data as cdata
from .request import Request

_CACHE = C.PreparedCache()


def get_prepared(p, req: Request, progress=None):
    """``core.data.Prepared`` of the request on the current data; the last one is kept, so ``fit`` right after ``check`` does not redo the basis and the smoothing."""
    key = C.request_key(p, req.input, req.cfg.seed)
    return _CACHE.get(key, lambda: cdata.prepare(make_variables(p, req.input), req.input, progress, seed=req.cfg.seed, ife=ife_of(p, req.input)))


def forget() -> None:
    _CACHE.forget()


def apply_data(p, ids: list[str], names: list[str]) -> ApplyData:
    """The samples ``ids`` of the open project in the state the network of a result (not outdated) reads them: the view's EEMs, the absorbance spectra as the project holds them (the
    saved model cleans them like the training spectra)."""
    return C.apply_data(p, ids, names, "network")


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
            "words": "" if ok else "PyTorch is not installed: open Settings, Add-ons, and press Install for Autoencoders (if it is already listed as installed, remove it and install it again)."}


def require_torch() -> None:
    st = torch_status()
    if not st["available"]:
        raise HTTPException(400, st["words"])
