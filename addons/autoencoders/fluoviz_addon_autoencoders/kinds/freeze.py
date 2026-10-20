"""Freeze a trained network (a result ``autoencoder-N``) as a saved model of kind ``autoencoder``.

Everything a saved model needs lives in the result itself (``result.payload["frozen"]``, written by the fit and stored with the project like every result): the weights, how samples
become network input, the reference samples and the limits.  Freezing therefore needs no training and no recomputation of the basis, also after the project was closed and opened again;
it only captures the data recipe of the project (the corrections, scatter treatment and window the training samples went through), checks that the result is not outdated and
re-scores a few training samples through the recipe with the frozen numbers: a model that cannot reproduce its own codes is refused.

The steps and the format of ``payload["frozen"]`` (``pack_frozen`` / ``unpack_frozen`` are its two halves) are those of the ``common`` add-on (``fluoviz_addon_common.freeze``);
this file gives them the words, the options and the comparison of the autoencoder.
"""
from __future__ import annotations

import numpy as np
from fluoviz_addon_common import freeze as C
from fluoviz_addon_common.freeze import FORMAT, N_CHECK, model_meta  # noqa: F401  (part of this module's interface)

from fluoviz.analysis.frozen import SavedModel

from .autoencoder import KIND, LABELS, options_for, outputs_for
from .model import AEModel

ADDON_ID = "autoencoders"
KIT = C.FreezeKit(addon_id=ADDON_ID, kind=KIND, noun="network", did="trained", again="Train again with the same settings",
                  unusable="This network was trained by another version of the Autoencoders add-on and cannot be saved as a model.")


def addon_version() -> str:
    from .. import __version__
    return __version__


def _fluoviz_version() -> str:
    return C.fluoviz_version()


def kind_label(model: AEModel) -> str:
    functional = any(b.kind != "pixels" for b in model.frame.blocks)
    base = LABELS[model.family]
    return f"Functional {base.lower()}" if functional else base


def pack_frozen(model_tree: dict, reference: dict, validation: dict, description: str, model: AEModel | None = None) -> dict:
    """The ``payload["frozen"]`` of a result from the pieces (``model_tree``: see ``model.model_tree``)."""
    model = model or AEModel(model_tree)
    return C.pack_frozen(model_tree, model.names, reference, validation, description, options=options_for(model.family), outputs=outputs_for(model), kind_label=kind_label(model))


def unpack_frozen(frozen) -> tuple[dict, AEModel]:
    """``(frozen, AEModel)`` from ``result.payload["frozen"]`` (checked: a plain-English ``FrozenError`` when it is missing or damaged)."""
    return C.unpack_frozen(KIT, frozen, AEModel)


def saved_from_frozen(frozen: dict, title: str, note: str = "", recipe_meta: dict | None = None, source: dict | None = None) -> SavedModel:
    """The :class:`SavedModel` of a frozen block (no project needed: the recipe and the provenance are given)."""
    return C.saved_from_frozen(KIT, frozen, title, note, recipe_meta, source, version=addon_version())


def _compare(ev, ref: dict, pos) -> str | None:
    """The re-scored training samples must give the codes the result showed."""
    stored = np.asarray(ref["scores"], float)[pos]
    scale = max(float(np.abs(np.asarray(ref["scores"], float)).max()), 1e-300)
    err = float(np.abs(ev.codes - stored).max()) / scale
    if not err < 1e-6:
        return ("This network cannot be saved as a model: scoring its own samples with the frozen numbers does not give the codes it showed "
                f"(the largest difference is {err:.1e} of the largest code). This would give wrong codes for new samples, so the model was not saved.")
    return None


def freeze_autoencoder(p, result, title: str, note: str = "") -> SavedModel:
    """The saved model of a result of the open project ``p``; raises :class:`FrozenError` (plain English) when that is not possible."""
    return C.freeze_result(KIT, p, result, title, note, unpack=unpack_frozen, compare=_compare, version=addon_version())
