"""Freezing a result of a model add-on as a saved model, and reading it back (numpy only; FluoViz's saved-model machinery is imported).

Everything a saved model needs lives in the result itself (``result.payload["frozen"]``, written by the fit and stored with the project like every result): the model, how samples become its
input, the reference samples and the limits.  Freezing therefore needs no recomputation, also after the project was closed and opened again; it only captures the data recipe of the
project (the corrections, scatter treatment and window the training samples went through), checks that the result is not outdated and re-scores a few training samples through the recipe
with the frozen numbers: a model that cannot reproduce its own results is refused.

The format of ``payload["frozen"]`` (the same for every add-on)::

    {"version": 1, "model": {"tree": <serial tree>, "arrays": {key: ndarray}},          # meta["model"] through fluoviz.analysis.frozen.serial (FrozenElement dataclasses inside)
     "reference": {...}, "validation": {...}, "options": [...], "outputs": [...], "description": "...", "kind_label": "...", "needs": {...}}

What differs between add-ons is given by a :class:`FreezeKit` (the words, the id and the kind) and by two functions of the add-on: ``unpack_frozen`` and the comparison of a re-scored
training sample with what the result showed.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from fluoviz.analysis.frozen import FrozenError, SavedModel, serial
from fluoviz.core.recipe import RecipeError, capture_recipe, recipe_view

FORMAT = 1
N_CHECK = 6


@dataclass(frozen=True)
class FreezeKit:
    """What an add-on says about itself: ``addon_id`` ("som"), ``kind`` (the saved kind), ``noun`` ("map"), ``did`` ("computed"), ``again`` (the button that makes the result again) and
    ``unusable`` (the sentence for a result that holds no model of the right version, without the closing advice that the kit adds)."""
    addon_id: str
    kind: str
    noun: str
    did: str
    again: str
    unusable: str


def fluoviz_version() -> str:
    from fluoviz import __version__
    return __version__


def model_meta(frozen: dict) -> dict:
    """``meta["model"]`` (the live tree) from the stored frozen block."""
    return serial.decode(frozen["model"]["tree"], dict(frozen["model"]["arrays"]))


def pack_frozen(model_tree: dict, names, reference: dict, validation: dict, description: str, *, options: list, outputs: list, kind_label: str) -> dict:
    """The ``payload["frozen"]`` of a result from the pieces; ``names`` are the variables the model reads (``fluorescence`` / ``absorbance``)."""
    tree, arrays = serial.encode(model_tree)
    return {"version": FORMAT, "model": {"tree": tree, "arrays": arrays}, "reference": reference, "validation": validation, "options": options, "outputs": outputs,
            "description": description, "kind_label": kind_label,
            "needs": {"eem": "fluorescence" in names, "absorbance": "absorbance" in names, "meta_columns": [], "norm_column": None, "scans": []}}


def unpack_frozen(kit: FreezeKit, frozen, make_model):
    """``(frozen, make_model(tree))`` from ``result.payload["frozen"]`` (checked: a plain-English ``FrozenError`` when it is missing or damaged)."""
    if not isinstance(frozen, dict) or int(frozen.get("version", 0)) != FORMAT or "model" not in frozen:
        raise FrozenError(f"{kit.unusable} Use '{kit.again}', then save it.")
    try:
        tree = serial.decode(frozen["model"]["tree"], dict(frozen["model"]["arrays"]))
    except FrozenError:
        raise
    except Exception as exc:                                                    # noqa: BLE001 - a damaged result must say so politely, whatever the damage
        raise FrozenError(f"The stored {kit.noun} is damaged ({exc}). Use '{kit.again}', then save it.") from None
    return frozen, make_model(tree)


def saved_from_frozen(kit: FreezeKit, frozen: dict, title: str, note: str = "", recipe_meta: dict | None = None, source: dict | None = None, *, version: str) -> SavedModel:
    """The :class:`SavedModel` of a frozen block (no project needed: the recipe and the provenance are given); ``version`` is the version of the add-on."""
    meta = {"kind_label": frozen["kind_label"], "needs": dict(frozen["needs"]), "model": model_meta(frozen), "reference": frozen["reference"], "validation": dict(frozen["validation"]),
            "options": list(frozen["options"]), "outputs": list(frozen["outputs"]), "description": str(frozen["description"]), "source": source or {},
            "recipe": recipe_meta, "addon": {"id": kit.addon_id, "version": version}}
    return SavedModel(kind=kit.kind, title=title, meta=meta, note=note)


def freeze_result(kit: FreezeKit, p, result, title: str, note: str, *, unpack, compare, version: str) -> SavedModel:
    """The saved model of a result of the open project ``p``; raises :class:`FrozenError` (plain English) when that is not possible.

    ``unpack`` is the add-on's ``unpack_frozen``; ``compare(ev, ref, pos)`` checks the re-scored training samples ``pos`` against what the result showed and returns the sentence that says
    why the model cannot be saved, or ``None`` when they agree."""
    stale = p.stale_parts(result)
    if stale:
        from fluoviz.core.project import PART_LABELS
        raise FrozenError(f"This {kit.noun} is outdated (" + ", ".join(PART_LABELS[x] for x in stale) + f" changed since it was {kit.did}). Use '{kit.again}' first: a saved model "
                          f"must describe the data it was {kit.did} on.")
    frozen, model = unpack((result.payload or {}).get("frozen"))
    ids = [str(i) for i in (result.sample_ids or [])]
    if not ids:
        raise FrozenError("This result holds no samples, so it cannot be saved as a model.")
    try:
        recipe = capture_recipe(p, uses_eem="fluorescence" in model.names, uses_abs="absorbance" in model.names, ids=ids)
    except RecipeError as exc:
        raise FrozenError(str(exc)) from None
    check_reproduces(kit, p, recipe, model, frozen["reference"], ids, compare)
    source = {"project": p.name, "result": result.id, "result_title": result.title, "params": copy.deepcopy(result.params), "fluoviz": fluoviz_version(), "data_sig": copy.deepcopy(result.data_sig)}
    return saved_from_frozen(kit, frozen, title, note, recipe.to_meta(), source, version=version)


def check_reproduces(kit: FreezeKit, p, recipe, model, ref: dict, ids: list[str], compare, n_check: int = N_CHECK) -> None:
    """Score a few training samples through the recipe with the frozen numbers: they must give what the result showed (``compare``, see :func:`freeze_result`)."""
    pos = np.unique(np.linspace(0, len(ids) - 1, min(len(ids), n_check)).round().astype(int))
    pick = [ids[i] for i in pos]
    try:
        data = recipe_view(p, recipe, pick, None)
        ev = model.evaluate(data, 0.0)
    except (RecipeError, FrozenError) as exc:
        raise FrozenError(f"This {kit.noun} cannot be saved as a model: scoring its own samples with the frozen numbers failed ({exc}).") from None
    except Exception as exc:                                                    # noqa: BLE001
        raise FrozenError(f"This {kit.noun} cannot be saved as a model: scoring its own samples with the frozen numbers failed ({exc}).") from None
    if not ev.usable.all():
        raise FrozenError(f"This {kit.noun} cannot be saved as a model: some of its own training samples cannot be read through the data recipe "
                          f"({' '.join(ev.why[int(np.flatnonzero(~ev.usable)[0])])}).")
    problem = compare(ev, ref, pos)
    if problem:
        raise FrozenError(problem)
