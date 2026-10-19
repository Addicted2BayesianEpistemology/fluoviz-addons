"""The requests of the four tools that show the nonlinear effects of a network (``POST /{rid}/response | bends | local | pair``): every key is checked here and a wrong request is a 400
with a sentence that says what to change.  See ``docs/api.md`` (section 2.15) for what each key means."""
from __future__ import annotations

import numpy as np
from fastapi import HTTPException

from ..core.features import compile_feature, compile_features, default_features
from ..core.nonlinear import MAX_ICE
from ..core.spec import AutoencoderError

SPAN = (0.5, 4.0)
POINTS = (5, 201)
SHOWN = 8                          # the codes the page draws at first (and the codes that get curves of single samples unless the request says which)
GRIDS = (3, 5, 7)
HEAT = (11, 61)
MAX_LOADINGS = 6
MAX_FEATURES = 6


def bad(message: str) -> HTTPException:
    return HTTPException(400, message)


def plain(fn):
    """Run ``fn``; the plain-English refusals of the numerics (``AutoencoderError``) become a 400."""
    def wrapped(*a, **kw):
        try:
            return fn(*a, **kw)
        except AutoencoderError as exc:
            raise bad(str(exc)) from None
    wrapped.__name__ = fn.__name__
    wrapped.__doc__ = fn.__doc__
    return wrapped


def check_keys(body, allowed: set[str]) -> dict:
    if not isinstance(body, dict):
        raise bad("The request must be a group of settings.")
    extra = set(body) - allowed
    if extra:
        raise bad(f"Unknown setting '{sorted(extra)[0]}': the settings of this request are {', '.join(sorted(allowed))}.")
    return body


def number(body: dict, key: str, default, lo: float, hi: float, what: str, integer: bool = False):
    v = body.get(key, default)
    if v is None:
        v = default
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v):
        raise bad(f"'{key}' ({what}) must be a number.")
    if integer and abs(float(v) - round(float(v))) > 1e-9:
        raise bad(f"'{key}' ({what}) must be a whole number from {lo:g} to {hi:g}.")
    if not lo <= float(v) <= hi:
        raise bad(f"'{key}' ({what}) must be {'a whole number ' if integer else 'a number '}from {lo:g} to {hi:g}.")
    return int(round(float(v))) if integer else float(v)


def flag(body: dict, key: str, default: bool) -> bool:
    v = body.get(key, default)
    if v is None:
        return default
    if not isinstance(v, bool):
        raise bad(f"'{key}' must be true or false.")
    return v


def code_number(v, d: int, key: str = "code") -> int:
    """A 1-based code number -> 0-based."""
    if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= d:
        raise bad(f"'{key}' must be a whole number from 1 to {d} (the network has {d} code{'s' if d != 1 else ''}).")
    return v - 1


def sample_numbers(names, ids: list[str], what: str, limit: int) -> list[int]:
    if not isinstance(names, (list, tuple)) or not names:
        raise bad(f"'{what}' must be a list of sample names.")
    if len(names) > limit:
        raise bad(f"At most {limit} samples can be chosen.")
    out = []
    for nm in names:
        if not isinstance(nm, str):
            raise bad(f"'{what}' must be a list of sample names.")
        if nm not in ids:
            raise HTTPException(404, f"'{nm}' is not one of the samples of this network.")
        out.append(ids.index(nm))
    if len(set(out)) != len(out):
        raise bad("A sample can be chosen only once.")
    return out


def traversal(body: dict, span_default: float = 3.0) -> tuple[float, int]:
    return number(body, "span", span_default, *SPAN, "how far each code is moved, in SD of the code"), number(body, "n_points", 41, *POINTS, "positions per curve", integer=True)


def features_of(body: dict, frame, key: str = "features") -> tuple[list, list]:
    """``(specs, compiled features)`` of a request (the default ones when it names none)."""
    specs = body.get(key)
    if specs is None:
        specs = default_features(frame)
    if not isinstance(specs, list):
        raise bad(f"'{key}' must be a list of features, each a group of settings such as {{'kind': 'total'}}.")
    if len(specs) > MAX_FEATURES:
        raise bad(f"At most {MAX_FEATURES} features can be drawn at once.")
    return specs, compile_features(specs, frame)


# ----------------------------------------------------------------------------- the four requests
RESPONSE_KEYS = {"span", "n_points", "features", "overlay", "ice"}


@plain
def parse_response(body: dict, ids: list[str], d: int, frame) -> dict:
    """``{"span", "n_points", "specs", "features", "overlay", "ice"}`` of ``POST /{rid}/response`` (``specs``: the features as asked, ``features``: compiled) with ``ice`` = ``None`` or
    ``{"codes": [0-based], "n": k | "samples": [indices]}``."""
    check_keys(body, RESPONSE_KEYS)
    span, n_points = traversal(body)
    specs, feats = features_of(body, frame)
    out = {"span": span, "n_points": n_points, "specs": specs, "features": feats, "overlay": flag(body, "overlay", True), "ice": None}
    ice = body.get("ice")
    if ice in (None, False):
        return out
    if ice is True:
        ice = {}
    if not isinstance(ice, dict):
        raise bad("'ice' must be true, false or a group of settings (n, samples, codes).")
    check_keys(ice, {"n", "samples", "codes"})
    codes = ice.get("codes")
    if codes is None:
        codes = list(range(1, min(d, SHOWN) + 1))
    if not isinstance(codes, list) or not codes or len(codes) > d:
        raise bad(f"'ice.codes' must be a list of code numbers from 1 to {d}.")
    spec: dict = {"codes": sorted({code_number(c, d, "ice.codes") for c in codes})}
    if ice.get("samples") is not None:
        spec["samples"] = sample_numbers(ice["samples"], ids, "ice.samples", MAX_ICE)
    else:
        spec["n"] = number(ice, "n", MAX_ICE, 1, MAX_ICE, "how many samples", integer=True)
    out["ice"] = spec
    return out


BENDS_KEYS = {"span", "n_points"}


@plain
def parse_bends(body: dict) -> dict:
    check_keys(body, BENDS_KEYS)
    span, n_points = traversal(body)
    return {"span": span, "n_points": n_points}


LOCAL_KEYS = {"code", "samples"}


@plain
def parse_local(body: dict, ids: list[str], d: int) -> dict:
    """``{"code": 0-based, "samples": [indices] | None}`` (``None`` = the samples at the 10th, 50th and 90th percentile of the code)."""
    check_keys(body, LOCAL_KEYS)
    out = {"code": code_number(body.get("code", 1), d), "samples": None}
    if body.get("samples") is not None:
        out["samples"] = sample_numbers(body["samples"], ids, "samples", MAX_LOADINGS)
    return out


PAIR_KEYS = {"codes", "grid", "span", "feature", "heat_points"}


@plain
def parse_pair(body: dict, d: int, frame) -> dict:
    """``{"codes": (i, j) 0-based | None, "grid", "span", "feature", "heat_points"}``; ``None`` codes = the two with the largest nonlinear share."""
    check_keys(body, PAIR_KEYS)
    if d < 2:
        raise bad("Two codes are needed to look at two codes together: this network has only one.")
    codes = body.get("codes")
    if codes is not None:
        if not isinstance(codes, list) or len(codes) != 2:
            raise bad("'codes' must be two code numbers, for example [1, 2].")
        i, j = (code_number(c, d, "codes") for c in codes)
        if i == j:
            raise bad("Choose two different codes.")
        codes = (i, j)
    grid = body.get("grid", 5)
    if isinstance(grid, bool) or grid not in GRIDS:
        raise bad("'grid' (pictures per code) must be 3, 5 or 7.")
    span = number(body, "span", 2.0, *SPAN, "how far each code is moved, in SD of the code")
    spec = body.get("feature")
    feature = compile_feature(spec if spec is not None else {"kind": "moved"}, frame, "f1", pair=True)
    return {"codes": codes, "grid": int(grid), "span": span, "feature": feature, "heat_points": number(body, "heat_points", 31, *HEAT, "points of the heat map per code", integer=True)}
