"""Fluorescence alone, absorbance alone, both; the pixel and the functional paths; gaps (SPEC sections 2 and 3)."""
import numpy as np
import pytest

import som_support as K
from fluoviz_addon_som.core import data as cdata
from fluoviz_addon_som.core import fitting
from fluoviz_addon_som.core.spec import Config, SomError

X0, A0 = K.small_eem(n=24, seed=3)
AB = K.small_abs(A0)


@pytest.mark.parametrize("kind", ["pixels", "bspline", "cosine", "gp"])
@pytest.mark.parametrize("which", ["fluorescence", "absorbance", "both"])
def test_every_choice_of_variables_works_in_every_path(kind, which):
    prep, inp = K.prepare(X0 if which != "absorbance" else None, AB if which != "fluorescence" else None, kind=kind)
    assert prep.n == 24
    assert prep.names == (["fluorescence"] if which == "fluorescence" else ["absorbance"] if which == "absorbance" else ["fluorescence", "absorbance"])
    assert prep.mode == ("pixels" if kind == "pixels" else "functional")
    cfg = Config.from_dict({"input": inp, "train": {"epochs_rough": 5, "epochs_fine": 10}, "map": {"size": [3, 4]}})
    fit = fitting.fit_map(prep, cfg, with_kfold=False)
    assert fit.protos_raw.shape == (12, prep.P)
    funcs = prep.frame.to_functions(fit.protos_raw)
    assert set(funcs) == set(prep.names)
    for b in prep.blocks:
        assert funcs[b.name].shape == (12, b.domain.m) and np.isfinite(funcs[b.name]).all()


def test_the_raw_coordinates_reproduce_the_smoothed_functions_and_the_l2_norm():
    prep, _ = K.prepare(X0, kind="bspline")
    b = prep.blocks[0]
    back = prep.frame.to_functions(prep.R)["fluorescence"]
    smooth = b.element.function()
    assert np.allclose(back, smooth, atol=1e-8)
    var = np.mean((prep.R ** 2).sum(axis=1))
    assert np.isclose(var, b.variance)
    w = np.asarray(b.domain.w)
    l2 = ((back - prep.frame.blocks[0].mean_function()) ** 2 * w).sum(axis=1).mean()
    assert np.isclose(l2, var, rtol=1e-6)


def test_pixel_coordinates_carry_the_quadrature_weights():
    prep, _ = K.prepare(X0, kind="pixels")
    b = prep.blocks[0]
    w = np.asarray(b.domain.w)
    assert np.isclose(prep.q.sum(), 1.0)
    assert np.isclose(np.mean((prep.R ** 2).sum(axis=1)), b.variance)
    back = prep.frame.to_functions(prep.R)["fluorescence"]
    assert np.allclose(back, b.Y, atol=1e-9)
    assert np.isclose(((b.Y - b.ybar) ** 2 * w).sum(axis=1).mean(), b.variance)


@pytest.mark.parametrize("kind", ["pixels", "bspline", "gp"])
def test_new_samples_get_the_training_coordinates(kind):
    prep, _ = K.prepare(X0, AB, kind=kind)
    b = prep.frame.blocks
    values = {"fluorescence": prep.blocks[0].domain.take(X0), "absorbance": prep.blocks[1].domain.take(AB)}
    co = prep.frame.coordinates(values, 0.8)
    assert co["usable"].all()
    ref = prep.R
    both = np.isfinite(ref) & np.isfinite(co["R"])
    assert np.allclose(co["R"][both], ref[both], atol=1e-8)


def test_block_weights_variance_none_and_custom():
    p1, _ = K.prepare(X0, AB, kind="bspline", weights="variance")
    assert np.allclose([b.weight for b in p1.blocks], [1 / b.variance for b in p1.blocks])
    energy = [(p1.R[:, b.sl] ** 2).sum(axis=1).mean() * b.weight for b in p1.blocks]
    assert np.allclose(energy, 1.0)                                   # each variable counts equally
    p2, _ = K.prepare(X0, AB, kind="bspline", weights="none")
    assert np.allclose([b.weight for b in p2.blocks], 1.0)
    p3, _ = K.prepare(X0, AB, kind="bspline", weights={"fluorescence": 1.0, "absorbance": 0.5})
    assert [b.weight for b in p3.blocks] == [1.0, 0.5]
    with pytest.raises(SomError, match="not part of the map"):
        K.prepare(X0, None, kind="bspline", weights={"absorbance": 1.0})


def gappy():
    X = X0.copy()
    for i in range(0, 24, 4):                                          # a rectangle cut out of a quarter of the samples
        X[i, 8:14, 6:10] = np.nan
    return X


def test_samples_that_lack_part_of_the_domain_pixels_keep_nan_and_partial_distances_work():
    X = gappy()
    prep, inp = K.prepare(X, kind="pixels", min_coverage=0.5)
    assert np.isnan(prep.R).any()
    cfg = Config.from_dict({"input": inp, "train": {"epochs_rough": 5, "epochs_fine": 20}, "map": {"size": [3, 4]}})
    fit = fitting.fit_map(prep, cfg, with_kfold=False)
    full, _ = K.prepare(X0, kind="pixels", min_coverage=0.5)
    fit2 = fitting.fit_map(full, cfg, with_kfold=False)
    # a sample with a piece cut out lands on (or next to) the unit of its complete twin
    d = fit.lattice.d2[fit.asg["c1"], fit2.asg["c1"]]
    assert np.mean(d <= 2.0) > 0.7
    mp = prep.missing_part()
    assert mp is not None and mp[0] > 0.005 and mp[1] == 0


def test_samples_that_lack_part_of_the_domain_functional_uses_the_gap_aware_estimate_and_weights():
    X = gappy()
    prep, _ = K.prepare(X, kind="bspline", min_coverage=0.5)
    assert np.isfinite(prep.R).all()
    w = prep.sample_weights("observed")
    assert w is not None and w[0] < w[1] and np.isclose(w.mean(), 1.0)
    assert prep.sample_weights("equal") is None
    clean, _ = K.prepare(X0, kind="bspline", min_coverage=0.5)
    assert clean.sample_weights("observed") is None and clean.missing_part() is None


def test_variables_mode():
    rng = np.random.default_rng(0)
    cols = {"a": rng.standard_normal(30), "b": rng.standard_normal(30) * 5, "c": rng.standard_normal(30)}
    cols["c"][3] = np.nan
    prep = cdata.prepare_table(cols, [f"S{i}" for i in range(30)])
    assert prep.mode == "variables" and prep.frame is None
    assert np.allclose(np.nanstd(prep.R, axis=0, ddof=1), 1.0)
    cfg = Config.from_dict({"train": {"epochs_rough": 3, "epochs_fine": 5}, "map": {"size": [3, 3]}})
    fit = fitting.fit_map(prep, cfg, with_kfold=False)
    assert fit.protos_raw.shape == (9, 3)
    with pytest.raises(SomError):
        prep.to_functions(prep.R)
    with pytest.raises(SomError, match="no value"):
        cdata.prepare_table({"a": np.full(10, np.nan)}, list(range(10)))


def test_too_few_samples_are_refused_in_plain_words():
    cfg = Config.from_dict({})
    with pytest.raises(SomError, match="at least 4"):
        cfg.check_samples(3)
    with pytest.raises(SomError, match="fewer than 8"):
        cfg.check_samples(6)
    Config.from_dict({"allow_few_samples": True}).check_samples(6)
