"""The request of a fit, checked in plain English: the model and sampler settings are ``core.spec.Config``; the data settings (``input``) are the Functional PCA request for the
fluorescence and the absorbance (checked by the function the Functional PCA page uses, ``routers/fpca.check_spec``; the basis must be the Gaussian-process basis), plus the
``values`` to model, the ``covariates`` and whether samples that lack a mode are used (``partial``).  Nothing is computed here.

Per function, ``basis_from`` says where its basis functions come from: ``gp`` (default: the Karhunen-Loeve eigenfunctions of the Gaussian-process prior, the paper's
construction) or ``fpca`` (the eigenfunctions of a functional PCA of the training samples, smoothed with that Gaussian-process basis).  The size of a basis ("Maximum number of
terms", "Number of terms" of step 3) is limited to 400 by FluoViz's own check; above ``SOFT_MAX`` (150) the request is refused unless ``large_basis`` is true, because the
sampler estimates every coefficient of every sample and such a model is extremely slow, usually without describing the data better.

Per function also how the data are sent to the model: ``log`` (``log10(value + log_offset)``; the offset is chosen from the data when ``log_offset`` is null), ``centre``
(``mean`` / ``median`` / ``none``) and ``scaling`` (``global`` / ``pointwise`` / ``pareto`` / ``none``).  ``log_keys`` lists the values and the covariates (numbers) that enter
as their ``log10``; ``values_basis`` is ``{"kind": "identity" | "pca", "var_fraction", "n_basis"}``; ``var_prep`` ``{key: {"centre", "scaling"}}`` says how a value or a
covariate is centred (``mean`` / ``median`` / ``none``; a category: ``mean`` = centred indicators, ``none`` = 0 / 1) and scaled (``sd`` / ``mad`` / ``none``) before the model
sees it (default: the mean and the SD).  ``basis_from: "fitted"`` takes the eigenfunctions of the functional PCA ``fpca_result`` (a result of the Functional PCA page): the
function then has that analysis' domain, pre-treatment and (absorbance) clean-up, whatever the request says."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

from ..core import modes as MD
from ..core.spec import Config, MblfError

INPUT_KEYS = ("variables", "fluorescence", "absorbance", "values", "covariates", "partial", "large_basis", "log_keys", "values_basis", "var_prep", "fpca_result")
PREP_KEYS = ("centre", "scaling", "log", "log_offset")
BASIS_FROM = ("gp", "fpca", "fitted")
SOFT_MAX = 150
MODE_WORDS = {"fluorescence": "fluorescence (EEM)", "absorbance": "absorbance"}
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
    large = raw.get("large_basis", False)
    if not isinstance(large, bool):
        raise MblfError("'large_basis' must be true or false.")
    log_keys = _keys(raw.get("log_keys"), "variables on the log scale")
    stray = [k for k in log_keys if k not in values and k not in covariates]
    if stray:
        raise MblfError(f"The variable {stray[0]} is on the log scale but is neither a value to model nor a covariate.")
    vb = raw.get("values_basis") or {"kind": "identity"}
    if not isinstance(vb, dict) or vb.get("kind", "identity") not in MD.VALUES_BASES:
        raise MblfError("The basis of the values must be 'identity' (one per value) or 'pca' (principal components).")
    vb = {"kind": vb.get("kind", "identity"), "var_fraction": float(vb.get("var_fraction", 0.95)), "n_basis": vb.get("n_basis")}
    if not 0.5 <= vb["var_fraction"] <= 0.9999:
        raise MblfError("The share of the variance of the values the principal components keep must be between 50 % and 99.99 %.")
    if vb["n_basis"] is not None:
        if isinstance(vb["n_basis"], bool) or not isinstance(vb["n_basis"], int) or not 1 <= vb["n_basis"] <= max(len(values), 1):
            raise MblfError(f"The number of principal components of the values must be a whole number from 1 to {max(len(values), 1)} (the number of values).")
    vp = raw.get("var_prep") or {}
    if not isinstance(vp, dict):
        raise MblfError("'var_prep' must give, per variable, how it is centred and scaled.")
    var_prep = {}
    for k, v in vp.items():
        if k not in values and k not in covariates:
            raise MblfError(f"The variable {k} has a centring or scaling but is neither a value to model nor a covariate.")
        if not isinstance(v, dict) or set(v) - {"centre", "scaling"}:
            raise MblfError(f"How {k} is centred and scaled must be given as 'centre' and 'scaling'.")
        c, sc = v.get("centre", "mean"), v.get("scaling", "sd")
        if c not in MD.VAR_CENTRES:
            raise MblfError(f"The centring of {k} must be one of {', '.join(MD.VAR_CENTRES)}.")
        if sc not in MD.VAR_SCALINGS:
            raise MblfError(f"The scaling of {k} must be one of {', '.join(MD.VAR_SCALINGS)}.")
        if (c, sc) != ("mean", "sd"):
            var_prep[k] = {"centre": c, "scaling": sc}
    out: dict = {"variables": [], "values": values, "covariates": covariates, "large_basis": large, "log_keys": log_keys, "values_basis": vb, "var_prep": var_prep}
    if names:
        body = {"variables": names}
        for n in names:
            sub = dict(raw.get(n) or {})
            sub.pop("basis_from", None)
            for k in PREP_KEYS:
                sub.pop(k, None)
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
                raise MblfError("The share of the variance the basis keeps (step 3) must be between 50 % and 99.99 %.")
            if "k_max" not in given:
                b["k_max"] = 40 if n == "fluorescence" else 25                       # (the add-on's defaults; FluoViz's own are larger)
            big = [(what, int(val)) for what, val in (("maximum number of terms", b.get("k_max")), ("number of terms", b.get("n_basis"))) if val and int(val) > SOFT_MAX]
            if big and not large:
                what, val = big[0]
                raise MblfError(f"The {what} of the {MODE_WORDS[n]} basis (step 3 · Basis) is {val}. Above {SOFT_MAX} the sampler has to estimate every coefficient of every "
                                "sample, so the model becomes extremely slow, and it rarely describes the data better. Lower it, or switch on 'Allow more than "
                                f"{SOFT_MAX} basis functions' in step 3 if you really want it.")
            bf = (raw.get(n) or {}).get("basis_from", "gp")
            if bf not in BASIS_FROM:
                raise MblfError(f"Where the basis functions of the {MODE_WORDS[n]} come from must be 'gp' (the Gaussian-process prior), 'fpca' (a functional PCA of the samples) "
                                "or 'fitted' (a functional PCA already fitted on the Functional PCA page).")
            item["basis"] = b
            item["basis_from"] = bf
            item.update(_prep(n, raw.get(n) or {}))
            out[n] = item
    fr = raw.get("fpca_result")
    if fr is not None and (not isinstance(fr, str) or not fr):
        raise MblfError("'fpca_result' must be the name of a functional PCA result (for example fpca-1).")
    if any(out.get(n, {}).get("basis_from") == "fitted" for n in out["variables"]):
        if not fr:
            raise MblfError("Choose the fitted functional PCA whose eigenfunctions are the basis (step 3).")
        bad = [n for n in out["variables"] if out[n].get("basis_from") == "fitted" and out[n].get("log")]
        if bad:
            raise MblfError(f"The basis of the {MODE_WORDS[bad[0]]} comes from a fitted functional PCA, which did not use a log scale: switch the log off (step 3).")
    out["fpca_result"] = fr if any(out.get(n, {}).get("basis_from") == "fitted" for n in out["variables"]) else None
    partial = raw.get("partial", True)
    if not isinstance(partial, bool):
        raise MblfError("'partial' must be true or false.")
    out["partial"] = partial
    return out


def _prep(n: str, raw: dict) -> dict:
    """How a function is sent to the model (see the module text), checked."""
    centre, scaling, log, off = raw.get("centre", "mean"), raw.get("scaling", "global"), raw.get("log", False), raw.get("log_offset")
    if centre not in MD.CENTRES:
        raise MblfError(f"The centring of the {MODE_WORDS[n]} must be one of {', '.join(MD.CENTRES)}.")
    if scaling not in MD.SCALINGS:
        raise MblfError(f"The scaling of the {MODE_WORDS[n]} must be one of {', '.join(MD.SCALINGS)}.")
    if not isinstance(log, bool):
        raise MblfError(f"The log scale of the {MODE_WORDS[n]} must be true or false.")
    if off is not None:
        if isinstance(off, bool) or not isinstance(off, (int, float)) or not off >= 0 or off != off:
            raise MblfError(f"The offset of the log of the {MODE_WORDS[n]} must be a number of zero or more (or empty: chosen from the data).")
        off = float(off)
    return {"centre": centre, "scaling": scaling, "log": log, "log_offset": off if log else None}


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
