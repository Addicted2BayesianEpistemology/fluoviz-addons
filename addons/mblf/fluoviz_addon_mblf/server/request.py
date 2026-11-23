"""The request of a fit, checked in plain English: the model and sampler settings are ``core.spec.Config``; the data settings (``input``) are the Functional PCA request for the
fluorescence and the absorbance (checked by the function the Functional PCA page uses, ``routers/fpca.check_spec``; the basis must be the Gaussian-process basis), plus the
``values`` to model, the ``covariates`` and whether samples that lack a mode are used (``partial``).  Nothing is computed here."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

from ..core.spec import Config, MblfError

INPUT_KEYS = ("variables", "fluorescence", "absorbance", "values", "covariates", "partial")
MAX_TITLE = 120
MAX_KEYS = 20


@dataclass
class Request:
    cfg: Config
    input: dict
    title: str = ""
    replace: str | None = None
    notes: list = field(default_factory=list)

    @property
    def names(self) -> list[str]:
        return list(self.input["variables"]) + (["values"] if self.input.get("values") else [])

    def params(self) -> dict:
        """The replayable request (what ``result.params`` holds): ``POST /fit`` with it gives the same model."""
        d = self.cfg.to_dict()
        d["input"] = copy.deepcopy(self.input)
        d["title"] = self.title
        return d


def _keys(raw, what: str) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        raw = raw.get("keys")
    if not isinstance(raw, list) or not all(isinstance(k, str) and k for k in raw):
        raise MblfError(f"The {what} must be a list of variable names (for example meta:DOC or res:parafac-1:C1_Fmax).")
    if len(raw) > MAX_KEYS:
        raise MblfError(f"At most {MAX_KEYS} {what} can be used.")
    if len(set(raw)) != len(raw):
        raise MblfError(f"A variable is named twice in the {what}.")
    return list(raw)


def check_input(raw) -> dict:
    """The data settings of a request (see the module text)."""
    from fluoviz.server.routers import fpca as R

    raw = {} if raw is None else raw
    if not isinstance(raw, dict):
        raise MblfError("The data settings ('input') must be a group of settings.")
    bad = sorted(set(raw) - set(INPUT_KEYS))
    if bad:
        raise MblfError(f"Unknown data setting '{bad[0]}': use {', '.join(INPUT_KEYS)}.")
    names = raw.get("variables")
    names = ["fluorescence"] if names is None else ([names] if isinstance(names, str) else list(names))
    bad = [n for n in names if n not in ("fluorescence", "absorbance")]
    if bad:
        raise MblfError(f"'{bad[0]}' cannot be modelled as a function: the functions are the fluorescence and the absorbance (other numbers go in 'values').")
    values = _keys(raw.get("values"), "values to model")
    covariates = _keys(raw.get("covariates"), "covariates")
    both = set(values) & set(covariates)
    if both:
        raise MblfError(f"The variable {sorted(both)[0]} is both a value to model and a covariate: a variable is either explained by the factors or explains them, not both.")
    if not names and not values:
        raise MblfError("Choose what to model: the fluorescence, the absorbance, other values, or several of them.")
    out: dict = {"variables": [], "values": values, "covariates": covariates}
    if names:
        body = {"variables": names}
        for n in names:
            sub = dict(raw.get(n) or {})
            b = dict(sub.get("basis") or {"kind": "gp"})
            if b.get("kind", "gp") != "gp":
                raise MblfError(f"The multimodal latent factor model needs the Gaussian-process basis for the {n} (the prior of the paper is built on it).")
            sub["basis"] = {**b, "kind": "gp"}
            sub["smoothing"] = {"method": "none"}
            body[n] = sub
        try:
            spec = R.check_spec(body)
        except ValueError as exc:
            raise MblfError(str(exc)) from None
        out["variables"] = list(spec["variables"])
        for n in out["variables"]:
            item = {k: v for k, v in spec[n].items() if k not in ("smoothing", "gaps")}
            b = dict(item["basis"])
            given = dict((raw.get(n) or {}).get("basis") or {})
            b["var_fraction"] = float(given.get("var_fraction", 0.99))
            if not 0.5 <= b["var_fraction"] <= 0.9999:
                raise MblfError("The share of the prior variance the basis keeps must be between 50 % and 99.99 %.")
            b["k_max"] = int(given.get("k_max", 40 if n == "fluorescence" else 25))
            if not 3 <= b["k_max"] <= 150:
                raise MblfError("The largest number of basis functions must be between 3 and 150.")
            item["basis"] = b
            out[n] = item
    partial = raw.get("partial", True)
    if not isinstance(partial, bool):
        raise MblfError("'partial' must be true or false.")
    out["partial"] = partial
    return out


def parse_request(body) -> Request:
    """``Request`` of a ``check`` / ``fit`` body; a plain-English ``MblfError`` for anything wrong."""
    if not isinstance(body, dict):
        raise MblfError("The settings must be sent as a group of named values.")
    b = dict(body)
    title = b.pop("title", "") or ""
    if not isinstance(title, str):
        raise MblfError("The title must be text.")
    replace = b.pop("replace", None)
    if replace is not None and not isinstance(replace, str):
        raise MblfError("'replace' must be the name of an earlier result.")
    inp = check_input(b.pop("input", None))
    cfg = Config.from_dict({**b, "input": inp})
    return Request(cfg=cfg, input=inp, title=title.strip()[:MAX_TITLE], replace=replace)
