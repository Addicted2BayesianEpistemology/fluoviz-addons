"""Freezing and reading back a model (``fluoviz_addon_common.freeze``): the format of ``payload["frozen"]``, and the plain-English refusals, in the words of each add-on."""
import numpy as np
import pytest

from fluoviz.analysis.frozen import FrozenError
from fluoviz.core.project import PART_LABELS
from fluoviz_addon_common import freeze as C

KIT = C.FreezeKit(addon_id="toy", kind="toy_model", noun="toy", did="made", again="Make again with the same settings", unusable="This toy was made by another version.")


def tree():
    return {"weights": np.arange(6.0).reshape(2, 3), "name": "x", "n": 3}


def test_a_packed_block_has_the_format_every_add_on_uses():
    f = C.pack_frozen(tree(), ["fluorescence"], {"ids": ["a"]}, {"r2": 0.9}, "A toy.", options=[{"key": "k"}], outputs=[{"key": "o"}], kind_label="Toy")
    assert set(f) == {"version", "model", "reference", "validation", "options", "outputs", "description", "kind_label", "needs"} and f["version"] == C.FORMAT == 1
    assert f["needs"] == {"eem": True, "absorbance": False, "meta_columns": [], "norm_column": None, "scans": []}
    assert set(f["model"]) == {"tree", "arrays"}


def test_what_was_packed_comes_back_unchanged():
    f = C.pack_frozen(tree(), ["fluorescence", "absorbance"], {}, {}, "", options=[], outputs=[], kind_label="Toy")
    got, model = C.unpack_frozen(KIT, f, lambda t: t)
    assert got is f and np.array_equal(model["weights"], tree()["weights"]) and model["name"] == "x" and model["n"] == 3
    assert np.array_equal(C.model_meta(f)["weights"], tree()["weights"]) and f["needs"]["absorbance"] is True


@pytest.mark.parametrize("bad", [None, {}, {"version": 2, "model": {}}, {"version": 1}, "text"])
def test_a_missing_or_foreign_block_says_what_to_do_in_the_words_of_the_add_on(bad):
    with pytest.raises(FrozenError) as e:
        C.unpack_frozen(KIT, bad, lambda t: t)
    assert str(e.value) == "This toy was made by another version. Use 'Make again with the same settings', then save it."


def test_a_damaged_block_is_not_a_traceback():
    f = C.pack_frozen(tree(), [], {}, {}, "", options=[], outputs=[], kind_label="Toy")
    f["model"]["arrays"] = {}                                                  # FluoViz's own reader says plainly that an array is missing: that sentence is kept
    with pytest.raises(FrozenError, match="the file is damaged"):
        C.unpack_frozen(KIT, f, lambda t: t)
    f["model"]["arrays"] = None                                                # anything else that goes wrong while reading is turned into a sentence too
    with pytest.raises(FrozenError, match=r"The stored toy is damaged \(.*\)\. Use 'Make again with the same settings', then save it\."):
        C.unpack_frozen(KIT, f, lambda t: t)


def test_the_saved_model_carries_the_kind_the_version_and_the_provenance():
    f = C.pack_frozen(tree(), ["fluorescence"], {"ids": ["a"]}, {"r2": 0.9}, "A toy.", options=[{"key": "k"}], outputs=[{"key": "o"}], kind_label="Toy")
    saved = C.saved_from_frozen(KIT, f, "My toy", "a note", {"steps": []}, {"project": "p"}, version="3.2.1")
    assert saved.kind == "toy_model" and saved.title == "My toy" and saved.note == "a note"
    m = saved.meta
    assert m["addon"] == {"id": "toy", "version": "3.2.1"} and m["source"] == {"project": "p"} and m["recipe"] == {"steps": []} and m["kind_label"] == "Toy" and m["description"] == "A toy."
    assert np.array_equal(m["model"]["weights"], tree()["weights"]) and m["options"] == [{"key": "k"}] and m["outputs"] == [{"key": "o"}] and m["reference"] == {"ids": ["a"]}


class Project:
    name = "P"

    def __init__(self, stale=()):
        self._stale = list(stale)

    def stale_parts(self, result):
        return self._stale


class Result:
    id, title, params, data_sig = "toy-1", "Toy 1", {"a": 1}, {}

    def __init__(self, payload=None, ids=("a", "b")):
        self.payload, self.sample_ids = payload, list(ids)


def test_an_outdated_result_cannot_be_saved_and_says_which_part_changed():
    part = next(iter(PART_LABELS))
    with pytest.raises(FrozenError) as e:
        C.freeze_result(KIT, Project([part]), Result(), "t", "", unpack=lambda f: (f, None), compare=lambda *a: None, version="1")
    assert str(e.value) == (f"This toy is outdated ({PART_LABELS[part]} changed since it was made). Use 'Make again with the same settings' first: a saved model must describe the data it was made on.")


def test_a_result_without_samples_cannot_be_saved():
    class M:
        names = ["fluorescence"]
    with pytest.raises(FrozenError, match="This result holds no samples"):
        C.freeze_result(KIT, Project(), Result(ids=()), "t", "", unpack=lambda f: (f, M()), compare=lambda *a: None, version="1")
