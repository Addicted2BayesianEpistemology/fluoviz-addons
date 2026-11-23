"""The saved-model side without the Stan engine: the model tree survives FluoViz's serialiser, a model of values reads new samples from the metadata, predicts a missing value with
its interval, and says which samples are unusual."""
import numpy as np
import pandas as pd
import pytest

import mblf_support as T
from fluoviz.analysis.frozen import ApplyData, serial
from fluoviz_addon_mblf.core import loo as LOO
from fluoviz_addon_mblf.kinds.model import MblfModel, model_tree


@pytest.fixture(scope="module")
def values_model():
    modes = [m for m in T.make_modes(values=3) if m.kind == "vector"]
    modes[0].columns = ["meta:a", "meta:b", "meta:c"]
    modes[0].column_labels = ["a", "b", "c"]
    tr = T.truth(modes, K=1, L=0, seed=3)
    tr["lam"] = np.array([[1.0], [0.9], [-0.8]])
    tr["sig2"] = np.full(3, 0.02)
    tr["psi2"] = np.array([0.02])
    Y, X = T.simulate(modes, tr, N=40, seed=4, missing=0.0, X=np.zeros((40, 0)))
    f = T.fitted(modes, tr, Y, X, S=60, L=0)
    base = LOO.loo_table(f, np.arange(f.draws.S))
    from fluoviz_addon_mblf.server.payload import scores_of_training, surprise_reference
    scores, sd = scores_of_training(f, np.arange(f.draws.S))
    sref, sv = surprise_reference(f, np.arange(f.draws.S), base)
    return MblfModel(model_tree(f, {"ids": f.ids, "scores": scores, "scores_sd": sd, "surprise": sref, "surprise_values": sv})), f


def test_the_tree_survives_the_serialiser(values_model):
    m, f = values_model
    tree, arrays = serial.encode(m.to_tree())
    back = MblfModel(serial.decode(tree, arrays))
    assert np.allclose(back.draws.lam, m.draws.lam) and back.modes[0].column_labels == ["a", "b", "c"] and back.reference["surprise"]


def test_new_samples_are_read_from_the_metadata_and_the_missing_value_is_predicted(values_model):
    m, f = values_model
    meta = pd.DataFrame({"a": [1.0, -1.0, 0.0], "b": [0.9, -0.9, np.nan], "c": [-0.8, 0.8, 5.0]}, index=["n1", "n2", "n3"])
    data = ApplyData(ids=["n1", "n2", "n3"], meta=meta)
    ev = m.evaluate(data)
    assert ev.usable.all() and ev.present.all() and np.isfinite(ev.scores).all()
    pr = m.predict(ev, 2, 0, noisy=True)
    assert pr["quantiles"][0][1] < pr["mean"][1] < pr["quantiles"][-1][1]                     # b of n3 (not measured) has an interval
    assert ev.surprise[2] < ev.surprise[0]                                                     # n3 breaks the pattern of the training samples
    from fluoviz_addon_mblf.kinds.mblf_kind import judge
    status, reasons, flag, lim = judge(m, ev, {"limit": "99"})
    assert status[2] == "outside" and status[0] == "ok"
