"""Freeze a trained network (a result ``autoencoder-N``) as a saved model of kind ``autoencoder``.

Everything a saved model needs lives in the result itself (``result.payload["frozen"]``, written by the fit and stored with the project like every result): the weights, how samples
become network input, the reference samples and the limits.  Freezing therefore needs no training and no recomputation of the basis, also after the project was closed and opened again;
it only captures the data recipe of the project (the corrections, scatter treatment and window the training samples went through), checks that the result is not outdated and
re-scores a few training samples through the recipe with the frozen numbers: a model that cannot reproduce its own codes is refused.

``pack_frozen`` / ``unpack_frozen`` are the two halves of the ``payload["frozen"]`` format::

    {"version": 1, "model": {"tree": <serial tree>, "arrays": {key: ndarray}},          # meta["model"] through fluoviz.analysis.frozen.serial (FrozenElement dataclasses inside)
     "reference": {...}, "validation": {...}, "options": [...], "outputs": [...], "description": "...", "kind_label": "...", "needs": {...}}
"""
from __future__ import annotations

import copy

import numpy as np

from fluoviz.analysis.frozen import FrozenError, SavedModel, serial
from fluoviz.core.recipe import RecipeError, capture_recipe, recipe_view

from .autoencoder import KIND, LABELS, options_for, outputs_for
from .model import AEModel

ADDON_ID = "autoencoders"
FORMAT = 1
N_CHECK = 6


def addon_version() -> str:
    from .. import __version__
    return __version__


def kind_label(model: AEModel) -> str:
    functional = any(b.kind != "pixels" for b in model.frame.blocks)
    base = LABELS[model.family]
    return f"Functional {base.lower()}" if functional else base


def pack_frozen(model_tree: dict, reference: dict, validation: dict, description: str, model: AEModel | None = None) -> dict:
    """The ``payload["frozen"]`` of a result from the pieces (``model_tree``: see ``model.model_tree``)."""
    model = model or AEModel(model_tree)
    tree, arrays = serial.encode(model_tree)
    return {"version": FORMAT, "model": {"tree": tree, "arrays": arrays}, "reference": reference, "validation": validation, "options": options_for(model.family), "outputs": outputs_for(model),
            "description": description, "kind_label": kind_label(model),
            "needs": {"eem": "fluorescence" in model.names, "absorbance": "absorbance" in model.names, "meta_columns": [], "norm_column": None, "scans": []}}


def unpack_frozen(frozen) -> tuple[dict, AEModel]:
    """``(frozen, AEModel)`` from ``result.payload["frozen"]`` (checked: a plain-English ``FrozenError`` when it is missing or damaged)."""
    if not isinstance(frozen, dict) or int(frozen.get("version", 0)) != FORMAT or "model" not in frozen:
        raise FrozenError("This network was trained by another version of the Autoencoders add-on and cannot be saved as a model. Use 'Train again with the same settings', then save it.")
    try:
        tree = serial.decode(frozen["model"]["tree"], dict(frozen["model"]["arrays"]))
    except FrozenError:
        raise
    except Exception as exc:                                                    # noqa: BLE001 - a damaged result must say so politely, whatever the damage
        raise FrozenError(f"The stored network is damaged ({exc}). Use 'Train again with the same settings', then save it.") from None
    return frozen, AEModel(tree)


def freeze_autoencoder(p, result, title: str, note: str = "") -> SavedModel:
    """The saved model of a result of the open project ``p``; raises :class:`FrozenError` (plain English) when that is not possible."""
    stale = p.stale_parts(result)
    if stale:
        from fluoviz.core.project import PART_LABELS
        raise FrozenError("This network is outdated (" + ", ".join(PART_LABELS[x] for x in stale) + " changed since it was trained). Use 'Train again with the same settings' first: a saved model "
                          "must describe the data it was trained on.")
    frozen, model = unpack_frozen((result.payload or {}).get("frozen"))
    ids = [str(i) for i in (result.sample_ids or [])]
    if not ids:
        raise FrozenError("This result holds no samples, so it cannot be saved as a model.")
    uses_eem, uses_abs = "fluorescence" in model.names, "absorbance" in model.names
    try:
        recipe = capture_recipe(p, uses_eem=uses_eem, uses_abs=uses_abs, ids=ids)
    except RecipeError as exc:
        raise FrozenError(str(exc)) from None
    ref = frozen["reference"]
    _check_reproduces(p, recipe, model, ref, ids)
    source = {"project": p.name, "result": result.id, "result_title": result.title, "params": copy.deepcopy(result.params), "fluoviz": _fluoviz_version(), "data_sig": copy.deepcopy(result.data_sig)}
    return saved_from_frozen(frozen, title, note, recipe.to_meta(), source)


def saved_from_frozen(frozen: dict, title: str, note: str = "", recipe_meta: dict | None = None, source: dict | None = None) -> SavedModel:
    """The :class:`SavedModel` of a frozen block (no project needed: the recipe and the provenance are given)."""
    meta = {"kind_label": frozen["kind_label"], "needs": dict(frozen["needs"]), "model": model_meta(frozen), "reference": frozen["reference"], "validation": dict(frozen["validation"]),
            "options": list(frozen["options"]), "outputs": list(frozen["outputs"]), "description": str(frozen["description"]), "source": source or {},
            "recipe": recipe_meta, "addon": {"id": ADDON_ID, "version": addon_version()}}
    return SavedModel(kind=KIND, title=title, meta=meta, note=note)


def model_meta(frozen: dict) -> dict:
    """``meta["model"]`` (the live tree) from the stored frozen block."""
    return serial.decode(frozen["model"]["tree"], dict(frozen["model"]["arrays"]))


def _fluoviz_version() -> str:
    from fluoviz import __version__
    return __version__


def _check_reproduces(p, recipe, model: AEModel, ref: dict, ids: list[str]) -> None:
    """Score a few training samples through the recipe with the frozen numbers: they must give the codes the result showed."""
    pos = np.unique(np.linspace(0, len(ids) - 1, min(len(ids), N_CHECK)).round().astype(int))
    pick = [ids[i] for i in pos]
    try:
        data = recipe_view(p, recipe, pick, None)
        ev = model.evaluate(data, 0.0)
    except (RecipeError, FrozenError) as exc:
        raise FrozenError(f"This network cannot be saved as a model: scoring its own samples with the frozen numbers failed ({exc}).") from None
    except Exception as exc:                                                    # noqa: BLE001
        raise FrozenError(f"This network cannot be saved as a model: scoring its own samples with the frozen numbers failed ({exc}).") from None
    if not ev.usable.all():
        raise FrozenError("This network cannot be saved as a model: some of its own training samples cannot be read through the data recipe " + f"({' '.join(ev.why[int(np.flatnonzero(~ev.usable)[0])])}).")
    stored = np.asarray(ref["scores"], float)[pos]
    scale = max(float(np.abs(np.asarray(ref["scores"], float)).max()), 1e-300)
    err = float(np.abs(ev.codes - stored).max()) / scale
    if not err < 1e-6:
        raise FrozenError("This network cannot be saved as a model: scoring its own samples with the frozen numbers does not give the codes it showed "
                          f"(the largest difference is {err:.1e} of the largest code). This would give wrong codes for new samples, so the model was not saved.")
