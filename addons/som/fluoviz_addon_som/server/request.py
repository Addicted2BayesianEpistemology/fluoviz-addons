"""The request of a fit, checked in plain English: the map settings are ``core.spec.Config``; the data settings (``input``) are the Functional PCA request, checked by the function the
Functional PCA page uses (``routers/fpca.check_spec``), plus the weights between variables, ``kind`` and, for a map of other variables, ``keys``.  Nothing is computed here."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

from ..core.spec import INPUT_KINDS, Config, SomError

INPUT_KEYS = ("kind", "variables", "fluorescence", "absorbance", "weights", "keys", "standardize")
MAX_TITLE = 120
MAX_KEYS = 60


@dataclass
class Request:
    cfg: Config
    input: dict
    title: str = ""
    replace: str | None = None
    notes: list = field(default_factory=list)

    @property
    def variables_mode(self) -> bool:
        return self.input.get("kind") == "variables"

    def params(self) -> dict:
        """The replayable request (what ``result.params`` holds): ``POST /fit`` with it gives the same map."""
        d = self.cfg.to_dict()
        d["input"] = copy.deepcopy(self.input)
        d["title"] = self.title
        return d


def check_input(raw) -> dict:
    """The data settings of a request (see the module text)."""
    from fluoviz.server.routers import fpca as R

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise SomError("The data settings ('input') must be a group of settings.")
    bad = sorted(set(raw) - set(INPUT_KEYS))
    if bad:
        raise SomError(f"Unknown data setting '{bad[0]}': use kind, variables, fluorescence, absorbance, weights (and keys, standardize for a map of other variables).")
    kind = raw.get("kind")
    if kind is not None and kind not in INPUT_KINDS:
        raise SomError(f"The kind of input must be one of: {', '.join(INPUT_KINDS)}.")
    if kind == "variables":
        keys = raw.get("keys")
        if not isinstance(keys, list) or not keys or not all(isinstance(k, str) and k for k in keys):
            raise SomError("Choose the variables to map: 'keys' must be a list of variable names (for example res:parafac-1:C1_Fmax or meta:DOC).")
        if len(keys) > MAX_KEYS:
            raise SomError(f"At most {MAX_KEYS} variables can be mapped at once.")
        if len(set(keys)) != len(keys):
            raise SomError("A variable is named twice in 'keys'.")
        std = raw.get("standardize", True)
        if not isinstance(std, bool):
            raise SomError("'standardize' must be true or false.")
        return {"kind": "variables", "keys": list(keys), "standardize": std}
    body = {k: copy.deepcopy(raw[k]) for k in ("variables", "fluorescence", "absorbance", "weights") if k in raw}
    names = body.get("variables") or ["fluorescence"]
    if kind == "pixels":
        for n in ([names] if isinstance(names, str) else names):
            if n in ("fluorescence", "absorbance"):
                sub = dict(body.get(n) or {})
                sub["basis"] = {"kind": "pixels"}
                sub["smoothing"] = {"method": "none"}
                body[n] = sub
    spec = R.check_spec(body)
    out = {"variables": list(spec["variables"]), **{n: spec[n] for n in spec["variables"]}, "weights": spec["weights"]}
    all_pix = all(out[n]["basis"]["kind"] == "pixels" for n in out["variables"])
    out["kind"] = "pixels" if all_pix else "functional"
    if kind == "functional" and all_pix:
        raise SomError("The functional version needs a basis (Gaussian process, B-splines or cosines); the pixels themselves are the other version.")
    return out


def parse_request(body) -> Request:
    """``Request`` of a ``check`` / ``fit`` body; a plain-English ``ValueError`` (a ``SomError``) for anything wrong."""
    if not isinstance(body, dict):
        raise SomError("The settings must be sent as a group of named values.")
    b = dict(body)
    title = b.pop("title", "") or ""
    if not isinstance(title, str):
        raise SomError("The title must be text.")
    replace = b.pop("replace", None)
    if replace is not None and not isinstance(replace, str):
        raise SomError("'replace' must be the name of an earlier map result.")
    inp = check_input(b.pop("input", None))
    cfg = Config.from_dict({**b, "input": inp})
    if inp["kind"] == "variables" and cfg.metric_kind == "prior":
        raise SomError("The Gaussian-process prior distance needs functions; use the plain, standardised or every-coordinate distance for a map of other variables.")
    return Request(cfg=cfg, input=inp, title=title.strip()[:MAX_TITLE], replace=replace)
