"""What the page looks at in a result: one sample against the network, traversals, interpolation, generation, sensitivity and the anomaly list.  Every function takes the result and
returns plain data (images already in the orientation of the browser); the router only checks the request and calls them."""
from __future__ import annotations

import numpy as np
from fastapi import HTTPException

from fluoviz.analysis.frozen import STATUS_WORDS
from fluoviz.analysis.frozen.kinds import _decomp as D

from ..kinds import tools as T
from ..kinds.autoencoder import Judgement, judge
from ..kinds.model import AEModel
from . import prep
from .results import axes_of, block_payload, browser_block, default_options, frozen_of, model_of, require_current

MAX_VALUES = 13
MAX_STEPS = 25


# ----------------------------------------------------------------------------- common bits
def codes_of(r) -> np.ndarray:
    return np.asarray(r.payload["codes"], float)


def code_frame(r) -> tuple[np.ndarray, np.ndarray]:
    """Mean and SD of the training codes (the centre and the unit of the traversals)."""
    return np.asarray(r.payload["code_mean"], float), np.asarray(r.payload["code_sd"], float)


def sample_index(r, sid: str) -> int:
    ids = list(r.sample_ids or [])
    if sid not in ids:
        raise HTTPException(404, f"'{sid}' is not one of the samples of this network.")
    return ids.index(sid)


def split_words(r, i: int) -> str:
    sp = r.payload.get("split") or {}
    flag = np.asarray(sp.get("flag", []))
    fold = sp.get("fold")
    if sp.get("mode") == "kfold":
        f = None if fold is None else int(np.asarray(fold)[i])
        return f"held out in fold {f + 1}" if f is not None and f >= 0 else "held out"
    return "validation" if len(flag) > i and int(flag[i]) else "training"


# ----------------------------------------------------------------------------- one sample
def sample_view(p, r, sid: str, thumb: bool = False) -> dict:
    require_current(p, r, "a single sample")
    i = sample_index(r, sid)
    model = model_of(r)
    ref = frozen_of(r)["reference"]
    data = prep.apply_data(p, [sid], model.names)
    ev = model.evaluate(data, 0.0)
    if not ev.usable[0]:
        raise HTTPException(400, " ".join(ev.why[0]) or "This sample cannot be shown against the network.")
    opt = default_options(model)
    jd = judge(model, ev, ref, opt)
    lim = float(ref["limits"][str(opt["limit"])])
    out: dict = {"id": sid, "title": f"{sid} against the {r.title}", "split": split_words(r, i), "status": str(jd.status[0]), "words": STATUS_WORDS[str(jd.status[0])],
                 "reasons": list(jd.reasons[0]), "code": ev.codes[0], "code_sd": None if ev.sd is None else ev.sd[0], "code_names": model.code_names, "error": float(ev.error[0]),
                 "r2": float(1.0 - ev.error[0]), "rmse": float(ev.rmse[0]), "typicality": float(jd.p[0]), "kl": None if jd.kl is None else float(jd.kl[0]),
                 "elbo": None if jd.elbo is None else float(jd.elbo[0]), "limit": lim, "flag": str(jd.flag[0])}
    mean = model.mean_functions()
    blocks = []
    for k, b in enumerate(model.frame.blocks):
        y, rec = ev.treated[b.name][0], ev.recon[b.name][0]
        resid = np.where(np.isfinite(y), y - rec, np.nan)
        entry = {"name": b.name, "kind": b.element.kind, "units": str(b.element.units or ""), "rmse": float(ev.rmse_blocks[0, k]), "error": float(ev.error_blocks[0, k])}
        if b.element.kind == "surface":
            entry.update({"measured": browser_block(model, b.name, y), "reconstruction": browser_block(model, b.name, rec)})
            if not thumb:
                entry.update({"residual": browser_block(model, b.name, resid), "mean": browser_block(model, b.name, mean[b.name])})
            ax = axes_of(model)
            entry.update({"ex": ax["ex"], "em": ax["em"], "mask": ax["mask"]})
        else:
            entry.update({"x": np.asarray(b.element.axes[0], float), "measured": browser_block(model, b.name, y), "reconstruction": browser_block(model, b.name, rec)})
            if not thumb:
                entry.update({"residual": browser_block(model, b.name, resid), "mean": browser_block(model, b.name, mean[b.name])})
        blocks.append(entry)
    out["blocks"] = blocks
    out["text"] = _sample_text(sid, jd, ev, lim, str(opt["limit"]))
    return out


def _sample_text(sid: str, jd: Judgement, ev, lim: float, lev: str) -> str:
    err = float(ev.error[0])
    if jd.status[0] == "ok":
        return f"{sid} is well rebuilt by the network: its reconstruction error ({D.g(err)}) is inside the {lev} % limit ({D.g(lim)})."
    return f"{sid}: " + " ".join(jd.reasons[0])


# ----------------------------------------------------------------------------- traversals
def check_values(values, name: str = "values", lo: float = -6.0, hi: float = 6.0) -> list[float]:
    if not isinstance(values, (list, tuple)) or not values:
        raise HTTPException(400, f"'{name}' must be a list of numbers (in SD of the training codes).")
    if len(values) > MAX_VALUES:
        raise HTTPException(400, f"At most {MAX_VALUES} values can be decoded at once.")
    out = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v):
            raise HTTPException(400, f"'{name}' must be a list of numbers (in SD of the training codes).")
        if not lo <= float(v) <= hi:
            raise HTTPException(400, f"Each value must lie between {lo:g} and {hi:g} SD of the training codes: farther out the network has never seen anything.")
        out.append(float(v))
    return out


def traverse_view(r, body: dict) -> dict:
    model = model_of(r)
    values = check_values(body.get("values", [-3, -2, -1, 0, 1, 2, 3]))
    base = body.get("base", "mean")
    if base != "mean":
        raise HTTPException(400, "'base' must be 'mean': the other codes are kept at their training mean.")
    which = body.get("code")
    if which is None:
        idx = list(range(model.d))
    else:
        if isinstance(which, bool) or not isinstance(which, int) or not 1 <= which <= model.d:
            raise HTTPException(400, f"'code' must be a whole number from 1 to {model.d} (or null for every code).")
        idx = [which - 1]
    mean, sd = code_frame(r)
    ax = axes_of(model)
    out: dict = {"values": values, "units": ax["units"], "codes": []}
    for k in ("ex", "em", "mask", "abs_wl"):
        if k in ax:
            out[k] = ax[k]
    base_fun = model.decode_functions(mean[None, :])
    out["mean"] = {b.name: browser_block(model, b.name, base_fun[b.name][0]) for b in model.frame.blocks}
    for j in idx:
        t = T.traverse(model, j, values, mean, sd)
        entry = {"code": j + 1, "name": model.code_names[j], "sd": float(sd[j]), "mean": float(mean[j]), "z": t["z"]}
        for b in model.frame.blocks:
            F = t["functions"][b.name]
            diff = F - t["base"][b.name][None, :]
            blk = block_payload(model, b.name, F, difference=diff)
            if b.element.kind == "surface":
                blk.update({"min": float(np.nanmin(F)), "max": float(np.nanmax(F)), "difference_abs_max": float(np.nanmax(np.abs(diff)))})
            entry[b.name] = blk
        out["codes"].append(entry)
    return out


# ----------------------------------------------------------------------------- interpolation
def interpolate_view(p, r, body: dict) -> dict:
    model = model_of(r)
    ids = list(r.sample_ids or [])
    a, b = body.get("a"), body.get("b")
    for nm, v in (("a", a), ("b", b)):
        if not isinstance(v, str):
            raise HTTPException(400, f"Choose the two samples to mix: '{nm}' is missing.")
        if v not in ids:
            raise HTTPException(404, f"'{v}' is not one of the samples of this network.")
    steps = body.get("steps", 9)
    if isinstance(steps, bool) or not isinstance(steps, int) or not 3 <= steps <= MAX_STEPS:
        raise HTTPException(400, f"'steps' must be a whole number from 3 to {MAX_STEPS}.")
    codes = codes_of(r)
    ia, ib = ids.index(a), ids.index(b)
    t = np.linspace(0.0, 1.0, steps)
    dec = T.interpolate(model, codes[ia], codes[ib], t)
    ax = axes_of(model)
    out: dict = {"t": t, "a": {"id": a, "code": codes[ia]}, "b": {"id": b, "code": codes[ib]}, "code_path": dec["path"], "units": ax["units"], "notes": [], "linear": None, "difference": None}
    for k in ("ex", "em", "mask", "abs_wl"):
        if k in ax:
            out[k] = ax[k]
    out["latent"] = {bl.name: block_payload(model, bl.name, dec["functions"][bl.name]) for bl in model.frame.blocks}
    if p.stale_parts(r):
        out["notes"].append("The data changed since this network was trained, so the plain mixture of the two measured samples is not shown.")
        return out
    ev = model.evaluate(prep.apply_data(p, [a, b], model.names), 0.0)
    if not ev.usable.all():
        out["notes"].append("One of the two samples cannot be read against the network, so the plain mixture is not shown.")
        return out
    lin, diff = {}, {}
    for bl in model.frame.blocks:
        Ya, Yb = ev.treated[bl.name][0], ev.treated[bl.name][1]
        L = (1.0 - t)[:, None] * Ya[None, :] + t[:, None] * Yb[None, :]
        lin[bl.name] = block_payload(model, bl.name, L)
        diff[bl.name] = block_payload(model, bl.name, dec["functions"][bl.name] - L)
    out["linear"], out["difference"] = lin, diff
    return out


# ----------------------------------------------------------------------------- generation
def generate_view(r, body: dict) -> dict:
    model = model_of(r)
    if not model.is_vae:
        raise HTTPException(400, "Only variational autoencoders can draw new samples (this network is a deterministic autoencoder). Train a variational autoencoder.")
    n = body.get("n", 6)
    seed = body.get("seed", 1)
    temp = body.get("temperature", 1.0)
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 24:
        raise HTTPException(400, "'n' (how many samples to draw) must be a whole number from 1 to 24.")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 31:
        raise HTTPException(400, "'seed' must be a whole number from 0 to 2147483647.")
    if isinstance(temp, bool) or not isinstance(temp, (int, float)) or not 0.0 <= float(temp) <= 2.0:
        raise HTTPException(400, "'temperature' must be a number from 0 to 2 (1 = the prior itself, smaller = more typical samples).")
    return synthetic(r, model, n, seed, float(temp), images=True)


def synthetic(r, model: AEModel, n: int, seed: int, temperature: float, images: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    Z = T.generate(model, n, rng, temperature)
    mean, sd = code_frame(r)
    idx, dist = T.nearest(codes_of(r), Z, sd)
    ids_real = list(r.sample_ids or [])
    out: dict = {"synthetic": True, "label": "Synthetic samples: made up by the network, not measured", "ids": [f"synthetic-{k + 1}" for k in range(n)], "codes": Z,
                 "nearest": [{"id": ids_real[int(i)], "distance": float(d)} for i, d in zip(idx, dist)], "notes": [], "seed": seed, "temperature": temperature}
    if images:
        F = model.decode_functions(Z)
        ax = axes_of(model)
        out["units"] = ax["units"]
        for k in ("ex", "em", "mask", "abs_wl"):
            if k in ax:
                out[k] = ax[k]
        for b in model.frame.blocks:
            out[b.name] = block_payload(model, b.name, F[b.name])
    return out


# ----------------------------------------------------------------------------- sensitivity
def sensitivity_view(r) -> dict:
    model = model_of(r)
    mean, sd = code_frame(r)
    sens = T.sensitivity(model, mean, sd, codes_of(r))
    ax = axes_of(model)
    out: dict = {"names": model.code_names, "sd": sd, "units": ax["units"]}
    for k in ("ex", "em", "mask", "abs_wl"):
        if k in ax:
            out[k] = ax[k]
    words = []
    for b in model.frame.blocks:
        at, mag = sens["at_mean"][b.name], sens["magnitude"][b.name]
        total = mag.sum(axis=1)
        share = total / total.sum() if total.sum() > 0 else np.zeros_like(total)
        entry = {"at_mean": browser_block(model, b.name, at), "magnitude": browser_block(model, b.name, mag), "share": share}
        if b.element.kind == "surface":
            for j, nm in enumerate(model.code_names):
                where = D.where_text(b.domain.embed(mag[j]), b.element.axes[1], b.element.axes[0])
                if where:
                    words.append(f"Code {nm} changes the fluorescence mostly around {where}.")
        else:
            for j, nm in enumerate(model.code_names):
                if np.isfinite(mag[j]).any():
                    wl = np.asarray(b.element.axes[0], float)[np.asarray(b.element.obs, bool)][int(np.nanargmax(mag[j]))]
                    words.append(f"Code {nm} changes the absorbance mostly around {wl:.0f} nm.")
        out[b.name] = entry
    out["words"] = " ".join(words)
    return out


# ----------------------------------------------------------------------------- the anomaly view
def anomalies_view(p, r, body: dict) -> dict:
    model = model_of(r)
    ref = frozen_of(r)["reference"]
    method = body.get("method", "saved")
    if method not in ("mad", "quantile", "saved"):
        raise HTTPException(400, "'method' must be mad, quantile or saved.")
    c = body.get("c", 3.0)
    level = body.get("level", 0.99)
    top = body.get("top", 12)
    if isinstance(top, bool) or not isinstance(top, int) or not 1 <= top <= 50:
        raise HTTPException(400, "'top' must be a whole number from 1 to 50.")
    err = np.asarray(r.payload["recon"]["error"], float)
    held = np.asarray(r.payload["recon"]["heldout"], bool)
    ref_err = err[np.isfinite(err)] if not held.any() else err[held & np.isfinite(err)]
    if method == "saved":
        lim = float(ref["limits"][str(default_options(model)["limit"])])
        words = f"The limit of the saved model: the {default_options(model)['limit']} % point of the errors of samples the network was not trained on."
    elif method == "mad":
        if isinstance(c, bool) or not isinstance(c, (int, float)) or not 0.5 <= float(c) <= 20:
            raise HTTPException(400, "'c' must be a number from 0.5 to 20.")
        v = np.log(ref_err[ref_err > 0])
        med = float(np.median(v))
        lim = float(np.exp(med + float(c) * 1.4826 * float(np.median(np.abs(v - med)))))
        words = f"The limit is the median of the (log) errors plus {float(c):g} times their robust spread (median absolute deviation)."
    else:
        if isinstance(level, bool) or not isinstance(level, (int, float)) or not 0.5 <= float(level) < 1.0:
            raise HTTPException(400, "'level' must be a number from 0.5 to 0.999.")
        lim = float(np.quantile(ref_err, float(level)))
        words = f"The limit is the {100 * float(level):g} % quantile of the errors."
    typ = np.asarray(r.payload["recon"].get("typicality", np.full(err.size, np.nan)), float)
    flagged = np.flatnonzero(np.isfinite(err) & (err > lim))
    flagged = flagged[np.argsort(-err[flagged])]
    ids = list(r.sample_ids or [])
    out: dict = {"limit": lim, "method": method, "words": words, "n_flagged": int(flagged.size), "flagged": []}
    edges = np.histogram_bin_edges(err[np.isfinite(err)], bins=min(20, max(5, err.size // 3)))
    cnt, _ = np.histogram(err[np.isfinite(err)], bins=edges)
    out["histogram"] = {"edges": edges, "counts": cnt, "limit": lim}
    current = not p.stale_parts(r)
    if not current:
        out["notes"] = ["The data changed since this network was trained, so the residual maps are not shown."]
    for i in flagged[:top]:
        item = {"id": ids[i], "error": float(err[i]), "typicality": None if not np.isfinite(typ[i]) else float(typ[i]),
                "reasons": [f"Its reconstruction error ({D.g(err[i])}) is above the limit ({D.g(lim)})."]}
        if current:
            ev = model.evaluate(prep.apply_data(p, [ids[i]], model.names), 0.0)
            if ev.usable[0]:
                item["residual"] = {b.name: browser_block(model, b.name, np.where(np.isfinite(ev.treated[b.name][0]), ev.treated[b.name][0] - ev.recon[b.name][0], np.nan)) for b in model.frame.blocks}
        out["flagged"].append(item)
    return out
