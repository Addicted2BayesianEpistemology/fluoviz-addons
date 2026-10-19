"""Training when some samples lack part of the domain: the functional versions weigh a sample whose coefficients were estimated in proportion to the share that was measured (the network is
not pulled by what was only estimated), the pixel versions leave the missing pixels out of the loss; with nobody lacking anything nothing changes.  Skipped cleanly when PyTorch is missing."""
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("torch")

from core_support import GP, example_eems, planted                                                           # noqa: E402
from fluoviz_addon_autoencoders.core import data as D, forward as fw, train as T                            # noqa: E402
from fluoviz_addon_autoencoders.core.spec import Config                                                      # noqa: E402
from test_core_gaps import GAP_ROWS, N, cut, fl                                                              # noqa: E402


def same(a, b):
    return all(np.array_equal(a.arrays[k], b.arrays[k]) for k in a.arrays)


def test_the_loss_of_a_sample_is_multiplied_by_its_weight():
    import torch
    A, _ = planted(n=12, P=10, k=2)
    td = D.TrainData.from_array(A)
    loss = T._Loss(td, Config(latent_dim=2), 5, np.ones(td.P))
    x = torch.as_tensor(td.X[:4], dtype=torch.float32)
    xh = x + 0.3
    w = torch.tensor([1.0, 0.5, 0.25, 0.0])
    plain = loss.rec(x, xh, None, None)
    assert torch.allclose(loss.rec(x, xh, None, None, w), plain * w) and loss.rec(x, xh, None, None, w)[3] == 0.0 and plain[3] > 0


def test_weights_of_one_change_nothing_and_a_down_weighted_group_is_followed_less():
    """A linear network is the PCA, so the effect of the weights is exact: the plane it finds is that of the samples it trusts."""
    rng = np.random.default_rng(3)
    W = rng.standard_normal((2, 30))
    v = rng.standard_normal(30)

    def draw(n):
        return rng.standard_normal((n, 2)) @ W + 0.02 * rng.standard_normal((n, 30))

    A, clean = draw(60), draw(40)                                                            # 60 samples to train on, 40 more of the same kind that no network sees
    bad = np.arange(48, 60)                                                                  # the last twelve are off in one more direction: coefficients that were only estimated, and badly
    A[bad] += 3.0 * rng.standard_normal((bad.size, 1)) @ v[None, :]
    cfg = Config(latent_dim=2, activation="linear", epochs=300, patience=300, n_restarts=1, seed=1)
    td = D.TrainData.from_array(A)
    none = T.fit(cfg, td)
    assert same(none, T.fit(cfg, replace(td, sw=np.ones(60))))                               # a weight of one for everybody is no weighting
    w = np.ones(60)
    w[bad] = 0.05
    w /= w.mean()
    weighed = T.fit(cfg, replace(td, sw=w))
    assert not same(none, weighed)

    def err(res):
        e, _ = fw.recon_error(res.spec, res.arrays, clean, fw.reconstruct(res.spec, res.arrays, clean)[0])
        return float(np.mean(e))

    assert err(none) > 0.05 and err(weighed) < 0.1 * err(none)                               # without weights the odd direction takes a code; with them the two real ones are found
    assert any("12 of the 60 samples lack part of the domain" in n for n in weighed.notes) and not any("lack part" in n for n in none.notes)


@pytest.fixture(scope="module")
def prepared():
    X, ex, em, ids, wl, A, _ = example_eems(n=N)
    inp = fl(GP)
    return D.prepare(D.variables_from_arrays(X=cut(X, ex, em), ex=ex, em=em, ids=ids, input=inp), inp, seed=0)


def test_a_functional_fit_weighs_the_samples_that_lack_a_rectangle_and_says_so(prepared):
    cfg = Config(latent_dim=3, epochs=40, patience=40, n_restarts=1, seed=2)
    res = T.fit(cfg, prepared)
    eq = T.fit(Config(latent_dim=3, epochs=40, patience=40, n_restarts=1, seed=2, gap_weighting="equal"), prepared)
    assert any("10 of the 40 samples lack part of the domain" in n and "in proportion to the share that was measured" in n for n in res.notes)
    assert not any("lack part of the domain" in n for n in eq.notes)
    assert not same(res, eq) and res.config["gap_weighting"] == "observed" and eq.config["gap_weighting"] == "equal"
    assert np.isfinite(res.samples["error"]).all() and res.validation["evr"] > 0.5


def test_a_fit_without_gaps_is_the_same_whatever_the_weighting():
    X, ex, em, ids, wl, A, _ = example_eems(n=N)
    inp = fl(GP)
    p = D.prepare(D.variables_from_arrays(X=X, ex=ex, em=em, ids=ids, input=inp), inp, seed=0)
    a = T.fit(Config(latent_dim=3, epochs=30, patience=30, n_restarts=1, seed=2), p)
    b = T.fit(Config(latent_dim=3, epochs=30, patience=30, n_restarts=1, seed=2, gap_weighting="equal"), p)
    assert same(a, b) and not any("lack part" in n for n in a.notes)


def test_a_pixel_fit_ignores_the_missing_pixels_and_needs_no_weights():
    X, ex, em, ids, wl, A, _ = example_eems(n=N)
    inp = fl({"kind": "pixels"})
    p = D.prepare(D.variables_from_arrays(X=cut(X, ex, em), ex=ex, em=em, ids=ids, input=inp), inp, seed=0)
    td = p.train_data()
    assert td.sw is None and td.M is not None and (~td.M[GAP_ROWS]).sum(axis=1).min() > 100
    cfg = Config(latent_dim=3, epochs=60, patience=60, n_restarts=1, seed=2)
    res = T.fit(cfg, p)
    assert not any("lack part" in n for n in res.notes) and np.isfinite(res.samples["error"]).all() and res.validation["evr"] > 0.5
    garbled = D.TrainData(np.where(td.M, td.X, 1.0e6), td.M, td.omega, td.blocks, td.q, td.sd, td.ids, td.info, td.q_rough)       # what stands where nothing was measured is never read
    assert same(res, T.fit(cfg, garbled))
