"""Looking at a result ``autoencoder-N``: finding it, its network (kept in memory after the first use: only numpy arrays), whether it is outdated, and the images in the orientation of
the browser.  The network and everything the page needs live in the result payload, so a project that was closed and opened again behaves exactly like one that was not."""
from __future__ import annotations

import numpy as np
from fastapi import HTTPException

from fluoviz.analysis.frozen import FrozenError
from fluoviz.analysis.frozen.kinds import _decomp as D
from fluoviz.core.project import PART_LABELS
from fluoviz.server.analysis import plot_image, plot_images, plot_mask

from ..kinds.autoencoder import KIND, options_for
from ..kinds.freeze import unpack_frozen
from ..kinds.model import AEModel

RESULT_KIND = "autoencoder"


def get_result(p, rid: str):
    try:
        r = p.get_result(rid)
    except ValueError:
        raise HTTPException(404, f"There is no result '{rid}' (it may have been deleted).") from None
    if r.kind != RESULT_KIND:
        raise HTTPException(400, f"'{rid}' is not an autoencoder result.")
    return r


def model_of(r) -> AEModel:
    """The network of a result (decoded from its payload once, then kept on ``r.obj``; the project file never holds it)."""
    if isinstance(r.obj, AEModel):
        return r.obj
    try:
        _, model = unpack_frozen((r.payload or {}).get("frozen"))
    except FrozenError as exc:
        raise HTTPException(400, str(exc)) from None
    r.obj = model
    return model


def frozen_of(r) -> dict:
    try:
        frozen, _ = unpack_frozen((r.payload or {}).get("frozen"))
    except FrozenError as exc:
        raise HTTPException(400, str(exc)) from None
    return frozen


def stale_words(p, r) -> str:
    stale = p.stale_parts(r)
    return ", ".join(PART_LABELS[x] for x in stale)


def require_current(p, r, what: str = "this") -> None:
    """Refuse (plain English) an endpoint that needs the measured data when the data are no longer those the network was trained on."""
    words = stale_words(p, r)
    if words:
        raise HTTPException(400, f"The data or the corrections changed since this network was trained ({words}), so {what} cannot be shown against the current data. "
                                 "Use 'Train again with the same settings' to train it on the current data.")


def default_options(model: AEModel) -> dict:
    return D.settings(options_for(model.family), None)


# ----------------------------------------------------------------------------- images for the browser
def surface_names(model: AEModel) -> list[str]:
    return [b.name for b in model.frame.blocks if b.element.kind == "surface"]


def browser_block(model: AEModel, name: str, F: np.ndarray):
    """Functions at the observed pixels ``(k, m)`` (or ``(m,)``) of block ``name`` -> what the browser draws: surface ``(k, n_ex, n_em)`` / ``(n_ex, n_em)`` float32 (NaN outside the
    domain), curve ``(k, n_wl)`` / ``(n_wl,)`` float32."""
    b = model.block(name)
    F = np.asarray(F, float)
    single = F.ndim == 1
    G = b.domain.embed(np.atleast_2d(F))
    if b.element.kind == "surface":
        out = plot_images(G)
        return out[0] if single else out
    out = G.astype(np.float32)
    return out[0] if single else out


def axes_of(model: AEModel) -> dict:
    """``ex``, ``em`` / ``abs_wl``, ``mask`` (browser orientation), ``units`` of the blocks of a model."""
    out: dict = {"units": {}}
    for b in model.frame.blocks:
        fe = b.element
        out["units"][b.name] = str(fe.units or "")
        if fe.kind == "surface":
            out["em"], out["ex"] = np.asarray(fe.axes[0], float), np.asarray(fe.axes[1], float)
            out["mask"] = plot_mask(~np.asarray(fe.obs, bool))
        else:
            out["abs_wl"] = np.asarray(fe.axes[0], float)
            out["abs_mask"] = (~np.asarray(fe.obs, bool)).astype(np.uint8)
    return out


def block_payload(model: AEModel, name: str, F: np.ndarray, **extra) -> dict:
    """``{"images": (k, n_ex, n_em)}`` for a surface block, ``{"x": wl, "series": [[...] x k]}`` for a curve block, plus ``extra`` arrays turned the same way."""
    b = model.block(name)
    out: dict = {}
    if b.element.kind == "surface":
        out["images"] = browser_block(model, name, F)
        for k, v in extra.items():
            out[k] = browser_block(model, name, v)
    else:
        out["x"] = np.asarray(b.element.axes[0], float)
        out["series"] = browser_block(model, name, F)
        for k, v in extra.items():
            out[k] = browser_block(model, name, v)
    return out


def turn_for_browser(payload: dict, model: AEModel) -> dict:
    """The stored payload as the page reads it: the frozen weights are left out, images and masks are turned to ``(…, n_ex, n_em)``."""
    out = {k: v for k, v in payload.items() if k != "frozen"}
    imgs = out.get("images")
    if isinstance(imgs, dict):
        out["images"] = {k: (plot_images(v) if isinstance(v, np.ndarray) and v.ndim == 3 else plot_image(v) if isinstance(v, np.ndarray) and v.ndim == 2 else v) for k, v in imgs.items()}
    if isinstance(out.get("mask"), np.ndarray) and out["mask"].ndim == 2:
        out["mask"] = plot_mask(out["mask"])
    q = out.get("quality")
    if isinstance(q, dict):
        q = dict(q)
        for key in ("local_r2", "residual_rms"):
            m = q.get(key)
            if isinstance(m, dict):
                q[key] = {nm: (plot_image(v) if isinstance(v, np.ndarray) and v.ndim == 2 else (np.asarray(v, np.float32) if isinstance(v, np.ndarray) else v)) for nm, v in m.items()}
        out["quality"] = q
    return out
