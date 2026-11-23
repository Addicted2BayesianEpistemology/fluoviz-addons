"""Freeze a fitted model (a result ``mblf-N``) as a saved model of kind ``mblf``.

Everything a saved model needs lives in the result itself (``result.payload["frozen"]``, written by the fit): the modes, the covariates, the posterior draws and the reference of the
training samples.  The steps are those of the ``common`` add-on (``fluoviz_addon_common.freeze``, which also describes the format of ``payload["frozen"]``); this file gives them the
words and the comparison: a few training samples are read again through the data recipe with the frozen numbers, and their scores must be those the result showed.
"""
from __future__ import annotations

import numpy as np
from fluoviz_addon_common import freeze as C

from fluoviz.analysis.frozen import SavedModel

from .mblf_kind import KIND, description_for, options_for, outputs_for
from .model import MblfModel

ADDON_ID = "mblf"
KIT = C.FreezeKit(addon_id=ADDON_ID, kind=KIND, noun="model", did="fitted", again="Fit again with the same settings",
                  unusable="This result cannot be saved as a model (it was made by another version of the add-on).")


def addon_version() -> str:
    from .. import __version__
    return __version__


def kind_label(model: MblfModel) -> str:
    return "Multimodal latent factors (Bayesian)" if model.draws.estimation == "bayes" else "Multimodal latent factors (maximum likelihood)"


def pack_frozen(model_tree: dict, validation: dict, model: MblfModel | None = None) -> dict:
    """The ``payload["frozen"]`` of a result (``model_tree``: see ``model.model_tree``)."""
    model = model or MblfModel(model_tree)
    out = C.pack_frozen(model_tree, model.names, {"ids": list(model.ids), "scores": model.reference.get("scores")}, validation, description_for(model, validation),
                        options=options_for(), outputs=outputs_for(model), kind_label=kind_label(model))
    out["needs"]["meta_columns"] = []          # the covariates and the values are read from the metadata when they are there; a sample without them is still used (see model.py)
    out["optional_meta"] = [k[5:] for k in model.meta_keys()]
    return out


def unpack_frozen(frozen) -> tuple[dict, MblfModel]:
    return C.unpack_frozen(KIT, frozen, lambda tree: MblfModel(tree))


def saved_from_frozen(frozen: dict, title: str, note: str = "", recipe_meta: dict | None = None, source: dict | None = None) -> SavedModel:
    return C.saved_from_frozen(KIT, frozen, title, note, recipe_meta, source, version=addon_version())


def _compare(ev, ref: dict, pos) -> str | None:
    stored = np.asarray(ref["scores"], float)[pos]
    scale = max(float(np.nanmax(np.abs(np.asarray(ref["scores"], float)))), 1e-12)
    err = float(np.nanmax(np.abs(ev.scores - stored))) / scale
    if not err < 1e-5:
        return ("This model cannot be saved: reading its own training samples again through the data recipe gives other factor scores than the result showed "
                f"(the largest difference is {err:.1e} of the largest score). The model would describe new samples wrongly, so it was not saved.")
    return None


def freeze_mblf(p, result, title: str, note: str = "") -> SavedModel:
    return C.freeze_result(KIT, p, result, title, note, unpack=unpack_frozen, compare=_compare, version=addon_version())
