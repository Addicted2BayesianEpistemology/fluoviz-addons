"""Freeze a fitted map (a result ``som-N``) as a saved model of kind ``som``.

Everything a saved model needs lives in the result itself (``result.payload["frozen"]``, written by the fit): the map, the metric, how samples become coordinates, the reference samples and
the limits.  Freezing needs no recomputation; it captures the data recipe of the project, checks that the result is not outdated and re-reads a few training samples through the recipe with
the frozen numbers: a model that cannot place its own samples on their own units is refused.  The steps are those of the ``common`` add-on (``fluoviz_addon_common.freeze``, which also
describes the format of ``payload["frozen"]``); this file gives them the words and the comparison of the map.
"""
from __future__ import annotations

import numpy as np
from fluoviz_addon_common import freeze as C
from fluoviz_addon_common.freeze import FORMAT, N_CHECK, model_meta  # noqa: F401  (part of this module's interface)

from fluoviz.analysis.frozen import SavedModel

from .model import SomModel
from .som import KIND, options_for, outputs_for

ADDON_ID = "som"
KIT = C.FreezeKit(addon_id=ADDON_ID, kind=KIND, noun="map", did="computed", again="Compute again with the same settings",
                  unusable="This map cannot be saved as a model (a map of other variables has no model, and a map made by another version of the add-on must be computed again).")


def addon_version() -> str:
    from .. import __version__
    return __version__


def _fluoviz_version() -> str:
    return C.fluoviz_version()


def kind_label(model: SomModel) -> str:
    functional = any(b.kind != "pixels" for b in model.frame.blocks)
    return "Functional self-organising map" if functional else "Self-organising map"


def pack_frozen(model_tree: dict, reference: dict, validation: dict, description: str, model: SomModel | None = None) -> dict:
    """The ``payload["frozen"]`` of a result from the pieces (``model_tree``: see ``model.model_tree``)."""
    model = model or SomModel(model_tree)
    return C.pack_frozen(model_tree, model.names, reference, validation, description, options=options_for(), outputs=outputs_for(model), kind_label=kind_label(model))


def unpack_frozen(frozen) -> tuple[dict, SomModel]:
    """``(frozen, SomModel)`` from ``result.payload["frozen"]`` (checked: a plain-English ``FrozenError`` when it is missing or damaged)."""
    return C.unpack_frozen(KIT, frozen, SomModel)


def saved_from_frozen(frozen: dict, title: str, note: str = "", recipe_meta: dict | None = None, source: dict | None = None) -> SavedModel:
    """The :class:`SavedModel` of a frozen block (no project needed: the recipe and the provenance are given)."""
    return C.saved_from_frozen(KIT, frozen, title, note, recipe_meta, source, version=addon_version())


def _compare(ev, ref: dict, pos) -> str | None:
    """The re-scored training samples must land on the units and at the distances the result showed."""
    stored_qe = np.asarray(ref["qe"], float)[pos]
    stored_unit = np.asarray(ref["unit"], int)[pos]
    scale = max(float(np.nanmax(np.asarray(ref["qe"], float))), 1e-300)
    err = float(np.abs(ev.qe - stored_qe).max()) / scale
    if not (err < 1e-6 and np.array_equal(ev.unit, stored_unit)):
        return ("This map cannot be saved as a model: scoring its own samples with the frozen numbers does not put them on the units they have on the map "
                f"(the largest difference of the distances is {err:.1e} of the largest distance). This would give wrong places for new samples, so the model was not saved.")
    return None


def freeze_som(p, result, title: str, note: str = "") -> SavedModel:
    """The saved model of a result of the open project ``p``; raises :class:`FrozenError` (plain English) when that is not possible."""
    return C.freeze_result(KIT, p, result, title, note, unpack=unpack_frozen, compare=_compare, version=addon_version())
