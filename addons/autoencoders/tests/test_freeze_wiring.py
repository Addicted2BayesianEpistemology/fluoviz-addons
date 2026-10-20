"""Freezing a result as a saved model, WITHOUT PyTorch: the steps are those of the ``common`` add-on, and this checks that the autoencoder gives them its own words, its own comparison and
its own kind.  (The training-based tests of ``test_api_models.py`` are skipped without PyTorch; the project and the data recipe are replaced here by stand-ins.)"""
import numpy as np
import pytest

import fluoviz_addon_autoencoders as addon
from fluoviz.analysis.frozen import FrozenError
from fluoviz_addon_autoencoders.kinds import freeze as F
from fluoviz_addon_common import freeze as C

import kind_support as K


class Recipe:
    def to_meta(self):
        return {"steps": ["stand-in"]}


class Result:
    id, title, params, data_sig = "autoencoder-1", "Autoencoder 1", {"latent_dim": 3}, {"sig": 1}

    def __init__(self, frozen, ids):
        self.payload, self.sample_ids = {"frozen": frozen}, list(ids)


class Project:
    name = "Stand-in"

    def __init__(self, stale=()):
        self._stale = list(stale)

    def stale_parts(self, result):
        return self._stale


@pytest.fixture()
def world(monkeypatch):
    X, _ = K.small_eem(n=24, seed=5)
    prep, _ = K.prepare(X, kind="bspline")
    spec, arrays = K.linear_network(prep, d=3)
    frozen, model, ev = K.make_frozen(prep, spec, arrays)
    ids = list(prep.ids)
    monkeypatch.setattr(C, "capture_recipe", lambda p, uses_eem, uses_abs, ids: Recipe())
    monkeypatch.setattr(C, "recipe_view", lambda p, recipe, pick, extra: K.apply_data(X[[ids.index(i) for i in pick]], ids=list(pick)))
    return {"frozen": frozen, "ids": ids, "X": X}


def test_a_result_is_frozen_with_the_kind_the_provenance_and_the_version_of_the_add_on(world):
    saved = F.freeze_autoencoder(Project(), Result(world["frozen"], world["ids"]), "My network", "a note")
    assert saved.kind == "autoencoder" and saved.title == "My network" and saved.note == "a note"
    m = saved.meta
    assert m["addon"] == {"id": "autoencoders", "version": addon.__version__} and m["recipe"] == {"steps": ["stand-in"]}
    assert m["source"]["project"] == "Stand-in" and m["source"]["result"] == "autoencoder-1" and m["kind_label"] == world["frozen"]["kind_label"]
    assert m["needs"] == {"eem": True, "absorbance": False, "meta_columns": [], "norm_column": None, "scans": []}
    assert saved.meta["options"] == world["frozen"]["options"] and {"spec", "arrays", "frame"} <= set(saved.meta["model"])


def test_the_block_is_unpacked_to_the_network_of_the_result(world):
    frozen, model = F.unpack_frozen(world["frozen"])
    assert frozen is world["frozen"] and model.family == "ae" and model.names == ["fluorescence"]
    assert F.kind_label(model) == world["frozen"]["kind_label"] == "Functional autoencoder"


def test_a_result_whose_codes_cannot_be_reproduced_is_refused_in_the_words_of_the_autoencoder(world):
    bad = dict(world["frozen"])
    bad["reference"] = dict(bad["reference"])
    bad["reference"]["scores"] = np.asarray(bad["reference"]["scores"], float) + 1.0
    with pytest.raises(FrozenError) as e:
        F.freeze_autoencoder(Project(), Result(bad, world["ids"]), "x")
    assert str(e.value).startswith("This network cannot be saved as a model: scoring its own samples with the frozen numbers does not give the codes it showed (the largest difference is ")
    assert str(e.value).endswith("This would give wrong codes for new samples, so the model was not saved.")


def test_an_outdated_result_says_to_train_again(world):
    with pytest.raises(FrozenError, match=r"This network is outdated \(.* changed since it was trained\)\. Use 'Train again with the same settings' first"):
        F.freeze_autoencoder(Project(["corrections"]), Result(world["frozen"], world["ids"]), "x")


def test_a_result_of_another_version_or_without_samples_is_refused(world):
    with pytest.raises(FrozenError, match="This network was trained by another version of the Autoencoders add-on and cannot be saved as a model. Use 'Train again with the same settings', then save it."):
        F.freeze_autoencoder(Project(), Result({"version": 99}, world["ids"]), "x")
    with pytest.raises(FrozenError, match="This result holds no samples"):
        F.freeze_autoencoder(Project(), Result(world["frozen"], []), "x")


def test_the_two_halves_of_the_block_are_the_shared_ones():
    assert F.FORMAT == C.FORMAT == 1 and F.model_meta is C.model_meta and F.N_CHECK == C.N_CHECK
