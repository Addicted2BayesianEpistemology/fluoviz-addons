"""Looking at a result ``autoencoder-N``: finding it, its network (kept in memory after the first use: only numpy arrays), whether it is outdated, and the images in the orientation of
the browser.  The network and everything the page needs live in the result payload, so a project that was closed and opened again behaves exactly like one that was not.  What the
self-organising map add-on does the same way is in the ``common`` add-on (``fluoviz_addon_common.results``)."""
from __future__ import annotations

import numpy as np
from fluoviz_addon_common import results as C
from fluoviz_addon_common.results import stale_words, turn_images, turn_mask  # noqa: F401  (part of this module's interface)

from fluoviz.analysis.frozen.kinds import _decomp as D
from fluoviz.server.analysis import plot_image, plot_images, plot_mask

from ..kinds.autoencoder import options_for
from ..kinds.freeze import unpack_frozen
from ..kinds.model import AEModel

RESULT_KIND = "autoencoder"

_R = C.Results(RESULT_KIND, a_noun="an autoencoder", noun="network", did="trained", verb="train", again="Train again with the same settings", model_class=AEModel, unpack=unpack_frozen)
get_result, model_of, frozen_of, require_current = _R.get_result, _R.model_of, _R.frozen_of, _R.require_current


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
    if "images" in out:
        out["images"] = turn_images(out["images"])
    if "mask" in out:
        out["mask"] = turn_mask(out["mask"])
    q = out.get("quality")
    if isinstance(q, dict):
        q = dict(q)
        for key in ("local_r2", "residual_rms"):
            m = q.get(key)
            if isinstance(m, dict):
                q[key] = {nm: (plot_image(v) if isinstance(v, np.ndarray) and v.ndim == 2 else (np.asarray(v, np.float32) if isinstance(v, np.ndarray) else v)) for nm, v in m.items()}
        out["quality"] = q
    return out
