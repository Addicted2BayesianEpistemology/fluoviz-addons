"""The data of the open project as a model add-on sees it (server side): the functional variables (the Functional PCA's own), the cache of the prepared coordinates (the basis and the smoothing
take seconds), and the samples of a result as an ``ApplyData`` so that a result is looked at with exactly the code the saved model uses."""
from __future__ import annotations

import hashlib
import json
import threading

import numpy as np
from fastapi import HTTPException

from fluoviz.analysis.fpca import absorbance_variable, fluorescence_variable
from fluoviz.analysis.frozen import ApplyData
from fluoviz.core.corrections import absorbance_at
from fluoviz.server.analysis import current_view


def abs_matrix(p, processed: bool):
    """The absorbance spectra of the project as the Functional PCA reads them: ``(wl, A, ids)``, cleaned like the Corrections page does when ``processed``."""
    wl, A, ids = p.abs_view()
    if processed:
        spec = p.state["corrections"]["ife"]
        A = np.vstack([absorbance_at(wl, row, wl, {**spec, "outside": "edge"}) if np.isfinite(row).sum() >= 2 else row for row in A])
    return wl, A, ids


def make_variables(p, inp: dict, keeps_gaps: str = "the model ignores them") -> list:
    """The functional variables of a request (``inp``: the checked ``input``).  Unlike the Functional PCA, a basis of kind ``pixels`` keeps the gaps inside the domain (``keeps_gaps`` says
    who ignores them; it is only documentation)."""
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
    if "absorbance" in inp.get("variables", []) and inp["absorbance"]["processed"]:
        return {**p.state["corrections"]["ife"], "outside": "edge"}
    return None


def apply_data(p, ids: list[str], names: list[str], what: str = "network") -> ApplyData:
    """The samples ``ids`` of the open project in the state the model of a result (not outdated) reads them: the view's EEMs, the absorbance spectra as the project holds them (the saved
    model cleans them like the training spectra).  ``what`` ("network", "map") is the word of the add-on for its model in the message."""
    ids = [str(i) for i in ids]
    view = p.view(only=ids)
    A = wl = None
    if "absorbance" in names:
        if p.abs is None:
            raise HTTPException(400, f"No absorbance spectra are loaded in this project, but this {what} reads them.")
        wl, Aall, aids = p.abs_view(include_excluded=True)
        pos = {s: k for k, s in enumerate(aids)}
        A = np.vstack([Aall[pos[s]] if s in pos else np.full(wl.size, np.nan) for s in ids])
    return ApplyData(ids=list(view.ids), X=view.X, ex=view.ex, em=view.em, A=A, abs_wl=wl, units=view.units)


# ----------------------------------------------------------------------------- the last prepared data
def request_key(p, inp: dict, seed, **extra) -> str:
    """A key for the request on the current data (``p.signature()``): the same key = the same prepared data."""
    h = hashlib.sha1()
    h.update(json.dumps({"input": inp, "seed": seed, "sig": p.signature(), **extra}, sort_keys=True, default=str).encode())
    return h.hexdigest()


class PreparedCache:
    """The last prepared data is kept, so that ``fit`` right after ``check`` does not redo the basis and the smoothing."""

    def __init__(self):
        self._key = None
        self._value = None
        self._lock = threading.Lock()

    def get(self, key: str, build):
        """The prepared data of ``key``; ``build()`` makes it (outside the lock: it takes seconds) when it is not the last one."""
        with self._lock:
            if self._key == key and self._value is not None:
                return self._value
        value = build()
        with self._lock:
            self._key, self._value = key, value
        return value

    def forget(self) -> None:
        with self._lock:
            self._key, self._value = None, None
