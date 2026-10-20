"""Looking at a result ``som-N``: finding it, its map (kept in memory after the first use), whether it is outdated, and the images in the orientation of the browser.  Modelled on
``server/results.py`` of the autoencoders add-on."""
from __future__ import annotations

import numpy as np
from fastapi import HTTPException

from fluoviz.analysis.frozen import FrozenError
from fluoviz.core.project import PART_LABELS
from fluoviz.server.analysis import plot_image, plot_images, plot_mask

from ..kinds.freeze import unpack_frozen
from ..kinds.model import SomModel

RESULT_KIND = "som"


def get_result(p, rid: str):
    try:
        r = p.get_result(rid)
    except ValueError:
        raise HTTPException(404, f"There is no result '{rid}' (it may have been deleted).") from None
    if r.kind != RESULT_KIND:
        raise HTTPException(400, f"'{rid}' is not a self-organising map result.")
    return r


def is_variables(r) -> bool:
    return (r.params or {}).get("input", {}).get("kind") == "variables"


def model_of(r) -> SomModel:
    """The map of a result (decoded from its payload once, then kept on ``r.obj``; the project file never holds it)."""
    if isinstance(r.obj, SomModel):
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
    return ", ".join(PART_LABELS[x] for x in p.stale_parts(r))


def require_current(p, r, what: str = "this") -> None:
    words = stale_words(p, r)
    if words:
        raise HTTPException(400, f"The data or the corrections changed since this map was computed ({words}), so {what} cannot be shown against the current data. "
                                 "Use 'Compute again with the same settings' to compute it on the current data.")


def browser_images(domain, F: np.ndarray, surface: bool):
    """Functions at the observed pixels ``(k, m)`` -> what the browser draws: surface ``(k, n_ex, n_em)`` float32 (NaN outside the domain), curve ``(k, n_wl)`` float32."""
    G = domain.embed(np.atleast_2d(np.asarray(F, float)))
    return plot_images(G) if surface else G.astype(np.float32)


def turn_for_browser(payload: dict) -> dict:
    """The stored payload as the page reads it: the frozen weights are left out, images and masks are turned to ``(…, n_ex, n_em)``."""
    out = {k: v for k, v in payload.items() if k != "frozen"}
    for key in ("prototypes", "mean_image"):
        d = out.get(key)
        if isinstance(d, dict):
            out[key] = {nm: (plot_images(v) if isinstance(v, np.ndarray) and v.ndim == 3 else plot_image(v) if isinstance(v, np.ndarray) and v.ndim == 2 else v) for nm, v in d.items()}
    if isinstance(out.get("mask"), np.ndarray) and out["mask"].ndim == 2:
        out["mask"] = plot_mask(out["mask"])
    return out
