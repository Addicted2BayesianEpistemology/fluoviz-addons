"""The statuses and the published variables of the saved kind, without the server (SPEC section 7)."""
import numpy as np

import som_support as K
from fluoviz_addon_som.core import fitting
from fluoviz_addon_som.core.spec import Config
from fluoviz_addon_som.kinds import som as KS
from fluoviz_addon_som.kinds.model import SomModel, model_tree

X0, A0 = K.small_eem(n=40, seed=3)


def build(size=(4, 5)):
    prep, inp = K.prepare(X0, kind="bspline")
    cfg = Config.from_dict({"input": inp, "train": {"epochs_rough": 8, "epochs_fine": 20}, "validation": {"folds": 4}, "map": {"size": list(size)}})
    fit = fitting.fit_map(prep, cfg)
    model = SomModel(model_tree(fit, prep.frame, np.full(fit.lattice.M, -1, int)))
    ev = model.evaluate_coordinates(prep.R)
    ref = KS.reference_for(model, prep.ids, ev, fit.kfold["qe"])
    return prep, fit, model, ev, ref


def test_training_samples_through_the_model_land_on_their_units():
    prep, fit, model, ev, ref = build()
    assert np.array_equal(ev.unit, fit.asg["c1"]) and np.allclose(ev.qe, fit.asg["qe"]) and np.allclose(ev.pos, fit.pos)
    assert not ev.empty.any()                                         # a training sample is on a unit that has at least itself


def test_an_empty_unit_makes_the_sample_uncertain_and_a_far_sample_outside():
    prep, fit, model, ev, ref = build((6, 8))
    assert (model.hits == 0).any()
    jd = KS.judge(model, ev, ref, {"limit": "99"})
    assert (jd.status == "ok").sum() >= 30
    ev2 = model.evaluate_coordinates(prep.R)
    empty_unit = int(np.flatnonzero(model.hits == 0)[0])
    ev2.unit[0], ev2.empty[0] = empty_unit, True
    ev2.qe[1] = 1e6
    jd2 = KS.judge(model, ev2, ref, {"limit": "99"})
    assert jd2.status[0] == "uncertain" and "no training sample reached" in jd2.reasons[0][0]
    assert jd2.status[1] == "outside" and "above the 99 % limit" in jd2.reasons[1][0]


def test_unusable_samples_have_nan_outputs():
    prep, fit, model, ev, ref = build()
    ev.usable[3] = False
    ev.why[3] = ["Only 10 % of the fluorescence points the map uses were measured; at least 80 % are needed."]
    jd = KS.judge(model, ev, ref, {"limit": "99"})
    assert jd.status[3] == "unusable"
    var, info = KS.publish(model, ev, jd)
    assert np.isnan(var["som_row"][3]) and var["som_unit"][3] == ""
    assert set(var) == set(info) and "som_cluster" not in var


def test_the_limit_comes_from_held_out_distances_and_is_widened_without_them():
    from fluoviz_addon_som.kinds import stats
    rng = np.random.default_rng(0)
    d = np.sqrt(rng.chisquare(3, 60))
    lim = stats.distance_limits(d)
    assert lim["source"] == "held-out" and lim["95"] < lim["99"] and np.mean(d < lim["95"]) > 0.9
    few = stats.distance_limits([], fallback=d)
    assert few["source"] == "training" and few["99"] > lim["99"]
    assert stats.distance_limits([], fallback=[1.0])["source"] == "none"


def test_published_names_match_the_outputs_of_the_model():
    prep, fit, model, ev, ref = build()
    jd = KS.judge(model, ev, ref, {"limit": "99"})
    var, info = KS.publish(model, ev, jd)
    out = [o["name"] for o in KS.outputs_for(model)]
    assert set(var) == set(out) - {"som_cluster"}
