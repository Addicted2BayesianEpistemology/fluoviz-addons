"""Looking at a result of a model add-on (server side): finding it, its model (kept in memory after the first use), whether it is outdated, and the images in the orientation of the
browser.  The model and everything the page needs live in the result payload, so a project that was closed and opened again behaves exactly like one that was not."""
from __future__ import annotations

import numpy as np
from fastapi import HTTPException

from fluoviz.analysis.frozen import FrozenError
from fluoviz.core.project import PART_LABELS
from fluoviz.server.analysis import plot_image, plot_images, plot_mask


def stale_words(p, r) -> str:
    """The parts of the project (data, corrections ...) that changed since the result was made, in words ('' = none)."""
    return ", ".join(PART_LABELS[x] for x in p.stale_parts(r))


class Results:
    """How one add-on finds its results: ``kind`` ("som"), ``a_noun`` ("a self-organising map"), ``noun`` ("map"), ``did`` ("computed"), ``verb`` ("compute"), ``again`` (the button that makes it again),
    ``model_class`` (the class of the model), ``unpack`` (``unpack_frozen`` of the add-on)."""

    def __init__(self, kind: str, *, a_noun: str, noun: str, did: str, verb: str, again: str, model_class, unpack):
        self.kind, self.a_noun, self.noun, self.did, self.verb, self.again, self.model_class, self.unpack = kind, a_noun, noun, did, verb, again, model_class, unpack

    def get_result(self, p, rid: str):
        try:
            r = p.get_result(rid)
        except ValueError:
            raise HTTPException(404, f"There is no result '{rid}' (it may have been deleted).") from None
        if r.kind != self.kind:
            raise HTTPException(400, f"'{rid}' is not {self.a_noun} result.")
        return r

    def model_of(self, r):
        """The model of a result (decoded from its payload once, then kept on ``r.obj``; the project file never holds it)."""
        if isinstance(r.obj, self.model_class):
            return r.obj
        try:
            _, model = self.unpack((r.payload or {}).get("frozen"))
        except FrozenError as exc:
            raise HTTPException(400, str(exc)) from None
        r.obj = model
        return model

    def frozen_of(self, r) -> dict:
        try:
            frozen, _ = self.unpack((r.payload or {}).get("frozen"))
        except FrozenError as exc:
            raise HTTPException(400, str(exc)) from None
        return frozen

    def require_current(self, p, r, what: str = "this") -> None:
        """Refuse (plain English) an endpoint that needs the measured data when the data are no longer those the model was made on."""
        words = stale_words(p, r)
        if words:
            raise HTTPException(400, f"The data or the corrections changed since this {self.noun} was {self.did} ({words}), so {what} cannot be shown against the current data. "
                                     f"Use '{self.again}' to {self.verb} it on the current data.")


# ----------------------------------------------------------------------------- images for the browser
def browser_images(domain, F: np.ndarray, surface: bool):
    """Functions at the observed pixels ``(k, m)`` -> what the browser draws: surface ``(k, n_ex, n_em)`` float32 (NaN outside the domain), curve ``(k, n_wl)`` float32."""
    G = domain.embed(np.atleast_2d(np.asarray(F, float)))
    return plot_images(G) if surface else G.astype(np.float32)


def turn_image(v):
    """One entry of a dictionary of images: ``(k, n_ex, n_em)`` / ``(n_ex, n_em)`` arrays are turned to the orientation of the browser, anything else is left as it is."""
    if isinstance(v, np.ndarray) and v.ndim == 3:
        return plot_images(v)
    if isinstance(v, np.ndarray) and v.ndim == 2:
        return plot_image(v)
    return v


def turn_images(d):
    """A ``{name: images}`` dictionary turned for the browser (anything that is not a dictionary is returned as it is)."""
    return {nm: turn_image(v) for nm, v in d.items()} if isinstance(d, dict) else d


def turn_mask(mask):
    """A 2-D mask in the orientation of the browser (anything else is returned as it is)."""
    return plot_mask(mask) if isinstance(mask, np.ndarray) and mask.ndim == 2 else mask
