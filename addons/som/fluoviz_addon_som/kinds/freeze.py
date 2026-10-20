"""Freeze a fitted map (a result ``som-N``) as a saved model of kind ``som``.  Modelled on ``kinds/freeze.py`` of the autoencoders add-on.

Everything a saved model needs lives in the result itself (``result.payload["frozen"]``, written by the fit): the map, the metric, how samples become coordinates, the reference samples and
the limits.  Freezing needs no recomputation; it captures the data recipe of the project, checks that the result is not outdated and re-reads a few training samples through the recipe with
the frozen numbers: a model that cannot place its own samples on their own units is refused.

``payload["frozen"]``::

    {"version": 1, "model": {"tree": <serial tree>, "arrays": {...}}, "reference": {...}, "validation": {...}, "options": [...], "outputs": [...], "description": "...", "kind_label": "...", "needs": {...}}
"""
from __future__ import annotations

import copy

import numpy as np

from fluoviz.analysis.frozen import FrozenError, SavedModel, serial
from fluoviz.core.recipe import RecipeError, capture_recipe, recipe_view

from .model import SomModel
from .som import KIND, options_for, outputs_for

ADDON_ID = "som"
FORMAT = 1
N_CHECK = 6


def addon_version() -> str:
    from .. import __version__
    return __version__


def kind_label(model: SomModel) -> str:
    functional = any(b.kind != "pixels" for b in model.frame.blocks)
    return "Functional self-organising map" if functional else "Self-organising map"


def pack_frozen(model_tree: dict, reference: dict, validation: dict, description: str, model: SomModel | None = None) -> dict:
    model = model or SomModel(model_tree)
    tree, arrays = serial.encode(model_tree)
    return {"version": FORMAT, "model": {"tree": tree, "arrays": arrays}, "reference": reference, "validation": validation, "options": options_for(), "outputs": outputs_for(model),
            "description": description, "kind_label": kind_label(model),
            "needs": {"eem": "fluorescence" in model.names, "absorbance": "absorbance" in model.names, "meta_columns": [], "norm_column": None, "scans": []}}


def unpack_frozen(frozen) -> tuple[dict, SomModel]:
    if not isinstance(frozen, dict) or int(frozen.get("version", 0)) != FORMAT or "model" not in frozen:
        raise FrozenError("This map cannot be saved as a model (a map of other variables has no model, and a map made by another version of the add-on must be computed again). "
                          "Use 'Compute again with the same settings', then save it.")
    try:
        tree = serial.decode(frozen["model"]["tree"], dict(frozen["model"]["arrays"]))
    except FrozenError:
        raise
    except Exception as exc:                                                    # noqa: BLE001
        raise FrozenError(f"The stored map is damaged ({exc}). Use 'Compute again with the same settings', then save it.") from None
    return frozen, SomModel(tree)


def freeze_som(p, result, title: str, note: str = "") -> SavedModel:
    stale = p.stale_parts(result)
    if stale:
        from fluoviz.core.project import PART_LABELS
        raise FrozenError("This map is outdated (" + ", ".join(PART_LABELS[x] for x in stale) + " changed since it was computed). Use 'Compute again with the same settings' first: a saved model "
                          "must describe the data it was computed on.")
    frozen, model = unpack_frozen((result.payload or {}).get("frozen"))
    ids = [str(i) for i in (result.sample_ids or [])]
    if not ids:
        raise FrozenError("This result holds no samples, so it cannot be saved as a model.")
    try:
        recipe = capture_recipe(p, uses_eem="fluorescence" in model.names, uses_abs="absorbance" in model.names, ids=ids)
    except RecipeError as exc:
        raise FrozenError(str(exc)) from None
    ref = frozen["reference"]
    _check_reproduces(p, recipe, model, ref, ids)
    source = {"project": p.name, "result": result.id, "result_title": result.title, "params": copy.deepcopy(result.params), "fluoviz": _fluoviz_version(), "data_sig": copy.deepcopy(result.data_sig)}
    return saved_from_frozen(frozen, title, note, recipe.to_meta(), source)


def saved_from_frozen(frozen: dict, title: str, note: str = "", recipe_meta: dict | None = None, source: dict | None = None) -> SavedModel:
    meta = {"kind_label": frozen["kind_label"], "needs": dict(frozen["needs"]), "model": model_meta(frozen), "reference": frozen["reference"], "validation": dict(frozen["validation"]),
            "options": list(frozen["options"]), "outputs": list(frozen["outputs"]), "description": str(frozen["description"]), "source": source or {},
            "recipe": recipe_meta, "addon": {"id": ADDON_ID, "version": addon_version()}}
    return SavedModel(kind=KIND, title=title, meta=meta, note=note)


def model_meta(frozen: dict) -> dict:
    return serial.decode(frozen["model"]["tree"], dict(frozen["model"]["arrays"]))


def _fluoviz_version() -> str:
    from fluoviz import __version__
    return __version__


def _check_reproduces(p, recipe, model: SomModel, ref: dict, ids: list[str]) -> None:
    """Read a few training samples through the recipe with the frozen numbers: they must land on the units and at the distances the result showed."""
    pos = np.unique(np.linspace(0, len(ids) - 1, min(len(ids), N_CHECK)).round().astype(int))
    pick = [ids[i] for i in pos]
    try:
        data = recipe_view(p, recipe, pick, None)
        ev = model.evaluate(data, 0.0)
    except (RecipeError, FrozenError) as exc:
        raise FrozenError(f"This map cannot be saved as a model: scoring its own samples with the frozen numbers failed ({exc}).") from None
    except Exception as exc:                                                    # noqa: BLE001
        raise FrozenError(f"This map cannot be saved as a model: scoring its own samples with the frozen numbers failed ({exc}).") from None
    if not ev.usable.all():
        raise FrozenError("This map cannot be saved as a model: some of its own training samples cannot be read through the data recipe " + f"({' '.join(ev.why[int(np.flatnonzero(~ev.usable)[0])])}).")
    stored_qe = np.asarray(ref["qe"], float)[pos]
    stored_unit = np.asarray(ref["unit"], int)[pos]
    scale = max(float(np.nanmax(np.asarray(ref["qe"], float))), 1e-300)
    err = float(np.abs(ev.qe - stored_qe).max()) / scale
    if not (err < 1e-6 and np.array_equal(ev.unit, stored_unit)):
        raise FrozenError("This map cannot be saved as a model: scoring its own samples with the frozen numbers does not put them on the units they have on the map "
                          f"(the largest difference of the distances is {err:.1e} of the largest distance). This would give wrong places for new samples, so the model was not saved.")
