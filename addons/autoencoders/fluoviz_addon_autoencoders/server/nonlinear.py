"""The four tools that show which nonlinear trend each code fits, as the page reads them (``POST /{rid}/response | bends | local | pair``; ``docs/api.md`` section 2.15).  The numerics are
``core/nonlinear.py`` (pure numpy); this module checks the request (``nlrequest``), finds the network and the training samples, and turns the answers into plain data with the
images in the orientation of the browser and the sentences in plain English."""
from __future__ import annotations

import json
import threading
from collections import OrderedDict

import numpy as np
from fastapi import HTTPException

from ..core import nonlinear as NL
from ..core.features import pixel_text
from . import nlrequest as Q
from . import prep
from .results import axes_of, browser_block, model_of
from .views import code_frame, codes_of

NEARLY_STRAIGHT = 0.05            # a code whose nonlinear share is below this acts like a straight line for the words of the page
SMALL_INTERACTION, LARGE_INTERACTION = 0.05, 0.25          # the words for the interaction index of two codes
SAME_SHAPE, LITTLE_SHAPE, SAME_STRENGTH = 0.005, 0.05, 0.05   # the words for the change of the shape (mean 1 - cos) and of the strength (SD of the relative length) of a local loading
_STATE: OrderedDict = OrderedDict()
_LOCK = threading.Lock()
_KEEP = 2


# ----------------------------------------------------------------------------- the network and the training samples
def decoder_of(r, model) -> NL.Decoder:
    mean, sd = code_frame(r)
    return NL.Decoder.from_model(model, codes_of(r), mean, sd)


class TrainingState:
    """The training samples as the network was trained to rebuild them: the smoothed functions (the measured values for a pixel block), NaN where a sample lacks the values (a region cut
    out of it), and their network coordinates."""

    def __init__(self, functions: dict, coords: np.ndarray):
        self.functions, self.coords = functions, coords


def training_state(p, r, model) -> TrainingState | None:
    """The training samples read from the open project; ``None`` when the data changed since the network was trained (the samples are then not the ones it learned from)."""
    if p.stale_parts(r):
        return None
    key = (r.id, id(model), json.dumps(p.signature(), sort_keys=True, default=str))
    with _LOCK:
        hit = _STATE.get(key)
        if hit is not None:
            _STATE.move_to_end(key)
            return hit
    ids = list(r.sample_ids or [])
    ev = model.evaluate(prep.apply_data(p, ids, model.names), 0.0)
    if not ev.usable.all():
        return None
    smooth = model.frame.to_functions(ev.X)
    funcs = {}
    for b in model.frame.blocks:
        measured = np.asarray(ev.treated[b.name], float)
        funcs[b.name] = measured if b.kind == "pixels" else np.where(np.isfinite(measured), smooth[b.name], np.nan)     # (blank where a sample lacks the values: they were only estimated)
    state = TrainingState(funcs, np.asarray(ev.X, float))
    with _LOCK:
        _STATE[key] = state
        while len(_STATE) > _KEEP:
            _STATE.popitem(last=False)
    return state


OUTDATED = "The data changed since this network was trained, so the training samples are not drawn (use 'Train again with the same settings' to see them)."


def common(model, r) -> dict:
    """What every answer carries: the names and units of the codes and the axes of the pictures."""
    mean, sd = code_frame(r)
    ax = axes_of(model)
    out: dict = {"names": model.code_names, "mean": mean, "sd": sd, "units": ax["units"]}
    for k in ("ex", "em", "mask", "abs_wl"):
        if k in ax:
            out[k] = ax[k]
    return out


def _pct(x: float) -> str:
    """A share as a whole percentage; "under 1 %" for a small one that is not zero (a share written "0 %" would read as exactly none)."""
    p = 100 * x
    return "under 1 %" if 0 < p < 1 else f"{p:.0f} %"


# ----------------------------------------------------------------------------- 1. response curves
def curve_words(f: dict, d: dict, pixel: dict | None) -> str:
    """The sentence under a response curve: which feature, the shape in words, and how far a straight line goes."""
    name = f["short"]
    if pixel:
        name += f" ({pixel_text(pixel)})"
    if d["kind"] == "flat":
        return f"{name}: {d['label']}."
    text = f"{name}: {d['label']}."
    if d.get("sign_label") and f["kind"] != "coordinate":
        text += f" It {d['sign_label']}."
    if d["r2_linear"] is not None and d["kind"] == "linear":
        text += f" A straight line explains {_pct(d['r2_linear'])} of it."
    elif d["r2_linear"] is not None:
        text += f" A straight line explains {_pct(d['r2_linear'])} of it (R² = {d['r2_linear']:.2f}); at its worst the curve is {_pct(d['max_deviation'])} of its range away from the line."
    return text


@Q.plain
def response_view(p, r, body: dict) -> dict:
    model = model_of(r)
    ids = list(r.sample_ids or [])
    req = Q.parse_response(body, ids, model.d, model.frame)
    dec = decoder_of(r, model)
    notes: list[str] = []
    state = None
    if req["overlay"]:
        state = training_state(p, r, model)
        if state is None:
            notes.append(OUTDATED)
    ice = None
    if req["ice"]:
        ice = {"codes": req["ice"]["codes"]}
        ice.update({"samples": req["ice"]["samples"]} if "samples" in req["ice"] else {"n": req["ice"]["n"]})
    res = NL.response_curves(dec, req["features"], req["span"], req["n_points"], ice, state.functions if state else None, state.coords if state else None)
    feats = res["features"]
    out: dict = {"span": req["span"], "n_points": req["n_points"], "t": res["t"], "weights": res["weights"], "specs": req["specs"], "features": feats, "codes": [], "notes": notes,
                 **common(model, r)}
    for j in range(model.d):
        entry: dict = {"code": j + 1, "name": model.code_names[j], "curves": {}, "pixel": {}, "ice": None}
        for f in feats:
            fid = f["id"]
            d = dict(res["description"][fid][j])
            px = res["pixel"][fid][j] if f["per_code"] else None
            if px:
                entry["pixel"][fid] = px
            d["y"] = res["curves"][fid][j]
            d["words"] = curve_words(f, d, px)
            d["turns"] = [{"position": a, "kind": b} for a, b in d["turns"]]
            entry["curves"][fid] = d
        if res["ice"] and j in res["ice"]:
            ic = res["ice"][j]
            entry["ice"] = {"samples": [ids[int(i)] for i in ic["samples"]], "x": ic["x"], "y": ic["y"]}
        out["codes"].append(entry)
    out["overlay"] = None
    if res["overlay"] is not None:
        ov = res["overlay"]
        out["overlay"] = {"ids": ids, "x": ov["x"], "y": ov["y"]}
    return out


# ----------------------------------------------------------------------------- 2. where the response bends
def bends_words(names: list[str], share: np.ndarray, dead: np.ndarray) -> str:
    live = [j for j in range(len(names)) if not dead[j] and np.isfinite(share[j])]
    if not live:
        return "None of the codes changes the output, so there is nothing to be nonlinear."
    order = sorted(live, key=lambda j: -share[j])
    top = order[0]
    parts = [f"{names[top]} bends the most: {_pct(share[top])} of its effect on the picture cannot be described by a straight line."]
    straight = [names[j] for j in live if share[j] < NEARLY_STRAIGHT]
    if straight:
        parts.append(f"{_join(straight)} {'acts' if len(straight) == 1 else 'act'} almost like {'a straight line' if len(straight) == 1 else 'straight lines'} (below {_pct(NEARLY_STRAIGHT)}).")
    if len(live) > 1 and share[top] < NEARLY_STRAIGHT:
        parts = [f"Every code acts almost like a straight line (the largest share is {_pct(share[top])}): this network is nearly linear."]
    ignored = [names[j] for j in range(len(names)) if dead[j]]
    if ignored:
        parts.append(f"{_join(ignored)} {'does' if len(ignored) == 1 else 'do'} not change the output at all.")
    return " ".join(parts)


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


@Q.plain
def bends_view(r, body: dict) -> dict:
    model = model_of(r)
    req = Q.parse_bends(body)
    dec = decoder_of(r, model)
    res = NL.bend_maps(dec, req["span"], req["n_points"])
    out: dict = {"span": req["span"], "n_points": req["n_points"], "t": res["t"], "min_effect": res["min_effect"], "share": res["share"], "effect_share": res["effect_share"],
                 "dead": res["dead"], "nearly_straight": NEARLY_STRAIGHT, **common(model, r)}
    for b in model.frame.blocks:
        blk = res["blocks"][b.name]
        entry = {"nl": browser_block(model, b.name, blk["nl"]), "amplitude": browser_block(model, b.name, blk["amplitude"]), "share": blk["share"], "n_effect": blk["n_effect"],
                 "max_amplitude": blk["max_amplitude"]}
        if b.element.kind != "surface":
            entry["x"] = np.asarray(b.element.axes[0], float)
        out[b.name] = entry
    out["words"] = bends_words(model.code_names, res["share"], res["dead"])
    return out


# ----------------------------------------------------------------------------- 3. local effects
def quantile_label(q: float, name: str) -> str:
    if q == 0.5:
        return f"median of {name}"
    k = round(100 * q)
    suffix = "th" if 10 <= k % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(k % 10, "th")
    return f"{k}{suffix} percentile of {name}"


def local_words(name: str, change: float, size_sd: float, dead: bool) -> str:
    if dead:
        return f"{name} does not change the output, so it has no loading to compare."
    shape = ("The picture of its effect has the same shape for every sample" if change < SAME_SHAPE else
             "The picture of its effect changes a little from sample to sample" if change < LITTLE_SHAPE else "The picture of its effect changes shape from sample to sample")
    strength = (" and about the same strength." if size_sd < SAME_STRENGTH else f" and its strength varies by about {_pct(size_sd)} (one standard deviation).")
    return shape + strength


@Q.plain
def local_view(r, body: dict) -> dict:
    model = model_of(r)
    ids = list(r.sample_ids or [])
    req = Q.parse_local(body, ids, model.d)
    dec = decoder_of(r, model)
    j = req["code"]
    typed = req["samples"] is not None
    idx = np.asarray(req["samples"], int) if typed else NL.loading_samples(dec, j)
    res = NL.local_effects(dec, j, idx)
    name = model.code_names[j]
    pos = dec.position(j)
    quants = NL.LOADINGS
    samples = [{"id": ids[int(i)], "label": ("chosen by you" if typed else quantile_label(quants[k], name)), "position": float(pos[int(i)]), "cosine": None if not np.isfinite(res["cosine"][k]) else float(res["cosine"][k])}
               for k, i in enumerate(idx)]
    out: dict = {"code": j + 1, "name": name, "sample_sd": float(dec.sd[j]), "samples": samples, "norm_at_mean": res["norm_at_mean"], **common(model, r)}
    for b in model.frame.blocks:
        nm = b.name
        entry = {"at_mean": browser_block(model, nm, res["at_mean"][nm]), "samples": browser_block(model, nm, res["samples"][nm]) if idx.size else None,
                 "variability": browser_block(model, nm, res["variability"][nm])}
        if b.element.kind != "surface":
            entry["x"] = np.asarray(b.element.axes[0], float)
        out[nm] = entry
    ch = res["change"]
    out["change"] = {"mean": ch["mean"], "max": ch["max"], "size_mean": ch["size_mean"], "size_sd": ch["size_sd"]}
    dead = not (res["norm_at_mean"] > 0)
    out["words"] = local_words(name, float(ch["mean"][j]) if np.isfinite(ch["mean"][j]) else 0.0, float(ch["size_sd"][j]) if np.isfinite(ch["size_sd"][j]) else 0.0, dead)
    return out


# ----------------------------------------------------------------------------- 4. two codes together
def pair_words(index: float, names: tuple[str, str]) -> str:
    if not np.isfinite(index):
        return f"Neither {names[0]} nor {names[1]} changes the picture over this grid, so there is no interaction to measure."
    if index < SMALL_INTERACTION:
        return f"{names[0]} and {names[1]} act almost independently: the change of the picture is the sum of what each does alone ({_pct(index)} of the change is not)."
    if index < LARGE_INTERACTION:
        return f"{names[0]} and {names[1]} interact a little: {_pct(index)} of the way the picture changes is not the sum of what each does alone."
    return f"{names[0]} and {names[1]} interact strongly: {_pct(index)} of the way the picture changes is not the sum of what each does alone, so what one code does depends on where the other is."


@Q.plain
def pair_view(p, r, body: dict) -> dict:
    model = model_of(r)
    req = Q.parse_pair(body, model.d, model.frame)
    dec = decoder_of(r, model)
    auto = req["codes"] is None
    if auto:
        i, j = NL.default_pair(NL.bend_maps(dec, 3.0, 41)["share"])
    else:
        i, j = req["codes"]
    notes: list[str] = []
    state = training_state(p, r, model)
    if state is None:
        notes.append(OUTDATED)
    res = NL.pair_effects(dec, i, j, req["grid"], req["span"], req["feature"], req["heat_points"], state.functions if state else None, state.coords if state else None)
    names = (model.code_names[i], model.code_names[j])
    g = req["grid"]
    out: dict = {"codes": [i + 1, j + 1], "auto": auto, "grid": g, "span": req["span"], "u": res["u"], "index": res["index"], "interaction_ss": res["interaction_ss"],
                 "variation_ss": res["variation_ss"], "index_by_block": res["index_by_block"], "index_limits": [SMALL_INTERACTION, LARGE_INTERACTION], "words": pair_words(res["index"], names), "notes": notes,
                 "feature": {**req["feature"].info(), **({"where": res["pixel"]} if "pixel" in res else {})}, **common(model, r)}
    gallery, inter = {}, {}
    for b in model.frame.blocks:
        nm = b.name
        G = res["F"][nm].reshape(g * g, -1)                                    # index = position of code j * grid + position of code i
        gal = browser_block(model, nm, G)
        gallery[nm] = {"images": gal} if b.element.kind == "surface" else {"x": np.asarray(b.element.axes[0], float), "series": gal}
        inter[nm] = browser_block(model, nm, res["interaction"][nm])
    out["gallery"], out["interaction"] = gallery, inter
    out["heat"] = {"x": res["v"], "y": res["v"], "z": res["heat"]}
    out["dots"] = {"ids": list(r.sample_ids or []), "x": res["dots"]["x"], "y": res["dots"]["y"], "value": res["dots"]["value"]}
    return out
