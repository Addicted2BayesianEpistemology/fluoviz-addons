"""The four tools that show which nonlinear trend each code fits (``core/nonlinear.py``): response curves, where the response bends, local effects, two codes together.
Networks whose decoder is known exactly (``nl_support.build``), so every number can be written down by hand; no PyTorch."""
import numpy as np
import pytest

from nl_support import build, grid_interaction_index, linear_decoder, prep_gp, prep_pixels, standard_codes, weighted_nl
from fluoviz_addon_autoencoders.core import nonlinear as NLN
from fluoviz_addon_autoencoders.core.curves import normal_weights, positions
from fluoviz_addon_autoencoders.core.features import compile_feature, compile_features, default_features, moved_index
from fluoviz_addon_autoencoders.core.nonlinear import Decoder, bend_maps, default_pair, ice_samples, local_effects, loading_samples, pair_effects, response_curves
from fluoviz_addon_autoencoders.core.spec import AutoencoderError

PREPS = [pytest.param(prep_gp, id="functional"), pytest.param(prep_pixels, id="pixels")]
T = positions(3, 41)


def loading_image(dec, pattern):
    """What a unit of unit height draws as a function (the linear part of the map from coordinates to the function): ``(m,)``."""
    blk = dec.block("fluorescence")
    return blk.to_function(pattern[None])[0] - blk.mean_function()


# ----------------------------------------------------------------------------- a decoder that is a straight line
@pytest.mark.parametrize("prep", PREPS)
def test_a_linear_decoder_has_nothing_nonlinear_anywhere(prep):
    dec = linear_decoder(prep(), d=3)
    bm = bend_maps(dec, 3, 41)
    assert (bm["share"] < 1e-12).all() and not bm["dead"].any() and np.isclose(bm["effect_share"].sum(), 1.0)
    nl = bm["blocks"]["fluorescence"]["nl"]
    assert np.nanmax(nl) < 1e-10 and np.isfinite(nl).mean() > 0.5 and (bm["blocks"]["fluorescence"]["share"] < 1e-12).all()      # NL = 0 wherever the code has an effect
    feats = compile_features(default_features(dec.frame), dec.frame)
    rc = response_curves(dec, feats, 3, 41, ice={"codes": [0, 1, 2], "n": 5})
    for f in feats:
        for j in range(3):
            d = rc["description"][f.id][j]
            assert d["kind"] == "linear" and d["r2_linear"] == pytest.approx(1.0, abs=1e-12) and d["max_deviation"] < 1e-10, (f.id, j, d["label"])
        for j in rc["ice"]:                                                                      # the curve of a single sample is the mean curve shifted: they are parallel
            y = rc["ice"][j]["y"][f.id]
            slopes = y[:, -1] - y[:, 0]
            assert np.allclose(slopes, rc["curves"][f.id][j][-1] - rc["curves"][f.id][j][0], rtol=1e-8, atol=1e-10)
    le = local_effects(dec, 1, loading_samples(dec, 1))
    assert (le["change"]["mean"] < 1e-12).all() and (np.abs(le["cosine"] - 1) < 1e-12).all() and (le["change"]["size_sd"] < 1e-9).all() and np.allclose(le["change"]["size_mean"], 1.0)
    assert le["variability"]["fluorescence"].max() < 1e-9 * np.abs(le["at_mean"]["fluorescence"]).max()
    pe = pair_effects(dec, 0, 2, 5, 2.0, feats[1])
    assert abs(pe["index"]) < 1e-12 and np.isfinite(pe["index"]) and max(np.abs(v).max() for v in pe["interaction"].values()) < 1e-10 * np.abs(pe["F"]["fluorescence"]).max()
    assert pe["heat"].shape == (31, 31)


# ----------------------------------------------------------------------------- one bent code
@pytest.mark.parametrize("prep", PREPS)
def test_one_tanh_unit_gives_the_analytic_nonlinear_share_and_words(prep):
    dec, spec, arrays, pat = build(prep(), 3, "tanh", [[0.8, 0, 0]])
    t = positions(3, 41)
    expect = weighted_nl(t, np.tanh(0.8 * t))
    bm = bend_maps(dec, 3, 41)
    assert bm["share"][0] == pytest.approx(expect, abs=1e-9) and bm["blocks"]["fluorescence"]["share"][0] == pytest.approx(expect, abs=1e-9)
    assert np.isnan(bm["share"][1:]).all() and bm["dead"].tolist() == [False, True, True] and bm["effect_share"].tolist() == pytest.approx([1.0, 0.0, 0.0])
    nl = bm["blocks"]["fluorescence"]["nl"]
    assert np.nanmax(np.abs(nl[0] - expect)) < 1e-9 and np.isnan(nl[1:]).all()                    # the same bend at every pixel: the picture is one pattern
    feats = compile_features([{"kind": "total"}, {"kind": "moved"}], dec.frame)
    rc = response_curves(dec, feats, 3, 41)
    for f in feats:
        d = rc["description"][f.id][0]
        assert d["r2_linear"] == pytest.approx(1 - expect, abs=1e-9) and d["kind"] == "s_shaped" and "S-shaped" in d["label"], (f.id, d["label"])
        assert rc["description"][f.id][1]["kind"] == "flat" and rc["description"][f.id][2]["kind"] == "flat"      # the codes the decoder ignores
    # the moved pixel is the one with the largest loading
    A = loading_image(dec, pat[0])
    assert rc["pixel"]["f2"][0]["em"] == pytest.approx(dec.block("fluorescence").domain.coords()[int(np.argmax(np.abs(A))), 0])
    # the curve of the total is its loading times the shape of the unit
    c = A @ dec.block("fluorescence").domain.w
    base = rc["curves"]["f1"][0]
    assert np.allclose(base - base[20], c * (np.tanh(0.8 * t) - np.tanh(0.0)), atol=1e-10)


def test_a_relu_switch_is_a_floor_and_a_straight_part():
    dec, *_ = build(prep_gp(), 3, "relu", [[1.0, 0, 0]])
    rc = response_curves(dec, compile_features([{"kind": "total"}], dec.frame), 3, 41)
    d = rc["description"]["f1"][0]
    assert d["kind"] == "saturating" and "levels off (saturates) below about" in d["label"]
    assert d["r2_linear"] == pytest.approx(1 - weighted_nl(T, np.maximum(T, 0)), abs=1e-9)


def test_the_effect_of_a_code_and_its_share_of_the_total_effect():
    prep = prep_gp()
    rng = np.random.default_rng(3)
    lin = np.zeros((prep.P, 3))
    lin[:, 1] = rng.normal(size=prep.P)
    dec, spec, arrays, pat = build(prep, 3, "tanh", [[0.8, 0, 0]], lin=lin)
    om = dec.omega
    t = positions(3, 41)
    w = normal_weights(t)
    var = lambda y: (w * (y - (w * y).sum()) ** 2).sum()
    effect = np.array([(om * pat[0] ** 2).sum() * var(np.tanh(0.8 * t)), (om * lin[:, 1] ** 2).sum() * var(t), 0.0])
    bm = bend_maps(dec, 3, 41)
    assert bm["effect"][:2] == pytest.approx(effect[:2], rel=1e-9) and bm["effect_share"][:2] == pytest.approx(effect[:2] / effect.sum(), rel=1e-9)
    assert bm["share"][1] < 1e-12 and bm["share"][0] > 0.01 and np.isnan(bm["share"][2])
    nl = bm["blocks"]["fluorescence"]["nl"]
    assert np.nanmax(nl[1]) < 1e-9 and np.nanmin(nl[0]) > 0.01


def test_pixels_that_the_code_hardly_moves_have_no_effect():
    prep = prep_gp()
    dec, spec, arrays, pat = build(prep, 2, "tanh", [[0.8, 0]])
    bm = bend_maps(dec, 3, 41)
    b = bm["blocks"]["fluorescence"]
    amp = b["amplitude"][0]
    quiet = ~(amp >= 0.02 * amp.max())
    assert quiet.any() and np.isnan(b["nl"][0][quiet]).all() and np.isfinite(b["nl"][0][~quiet]).all() and b["n_effect"][0] == (~quiet).sum()
    assert b["max_amplitude"][0] == pytest.approx(amp.max()) and bm["min_effect"] == 0.02
    wide = bend_maps(dec, 3, 41, min_effect=0.5)["blocks"]["fluorescence"]
    assert wide["n_effect"][0] < b["n_effect"][0] and np.nanmax(wide["nl"][0]) <= 1.0 and np.nanmin(b["nl"][0]) >= 0.0
    # the range of the pixel along the traversal
    A = loading_image(dec, pat[0])
    t = positions(3, 41)
    assert np.allclose(amp, np.abs(A) * (np.tanh(0.8 * t[-1]) - np.tanh(0.8 * t[0])), atol=1e-10)


def test_the_share_in_the_metric_of_the_loss_is_the_share_of_the_functions_for_one_block_and_between_them_for_two():
    dec, *_ = build(prep_gp(), 3, "tanh", [[0.8, 0.4, 0], [0, 1.1, 0.3]])
    one = bend_maps(dec, 3, 41)
    assert np.allclose(one["share"], one["blocks"]["fluorescence"]["share"], atol=1e-9, equal_nan=True)
    prep2 = prep_gp(absorbance=True)
    d2, *_ = build(prep2, 3, "tanh", [[0.8, 0.4, 0], [0, 1.1, 0.3]])
    two = bend_maps(d2, 3, 41)
    f, a = two["blocks"]["fluorescence"]["share"], two["blocks"]["absorbance"]["share"]
    lo, hi = np.minimum(f, a), np.maximum(f, a)
    assert (two["share"] >= lo - 1e-9).all() and (two["share"] <= hi + 1e-9).all() and set(two["blocks"]) == {"fluorescence", "absorbance"}
    assert two["blocks"]["absorbance"]["nl"].shape == (3, prep2.frame.block("absorbance").domain.m)


# ----------------------------------------------------------------------------- the features
def test_every_feature_is_what_it_says():
    prep = prep_gp(absorbance=True)
    dec, spec, arrays, pat = build(prep, 3, "tanh", [[0.9, 0.3, 0], [0, 0.7, 0.6]])
    blk = dec.block("fluorescence")
    dom = blk.domain
    co = dom.coords()
    ex0, em0 = co[200, 1], co[200, 0]
    specs = [{"kind": "point", "ex": ex0, "em": em0}, {"kind": "region_mean", "ex": [float(co[:, 1].min()), float(co[:, 1].min()) + 40], "em": [float(co[:, 0].min()), float(co[:, 0].min()) + 60]},
             {"kind": "region_max", "ex": [float(co[:, 1].min()), float(co[:, 1].min()) + 40], "em": [float(co[:, 0].min()), float(co[:, 0].min()) + 60]}, {"kind": "total"},
             {"kind": "absorbance", "wavelength": 300.2}, {"kind": "coordinate", "k": 3}]
    feats = compile_features(specs, dec.frame)
    Z = np.random.default_rng(0).normal(size=(7, 3))
    X, F = dec.decode_functions(Z)
    F0 = F["fluorescence"]
    inside = (co[:, 0] >= specs[1]["em"][0]) & (co[:, 0] <= specs[1]["em"][1]) & (co[:, 1] >= specs[1]["ex"][0]) & (co[:, 1] <= specs[1]["ex"][1])
    assert inside.sum() > 5
    got = [f.evaluate(F, X) for f in feats]
    assert np.allclose(got[0], F0[:, 200]) and np.allclose(got[1], F0[:, inside].mean(axis=1)) and np.allclose(got[2], F0[:, inside].max(axis=1)) and np.allclose(got[3], F0 @ dom.w)
    ab = dec.block("absorbance").domain
    wl = ab.axes[0][np.argmin(np.abs(ab.axes[0] - 300.2))]
    assert np.allclose(got[4], F["absorbance"][:, np.flatnonzero(ab.axes[0][ab.grid_index()[0]] == wl)[0]]) and feats[4].where == {"wavelength": pytest.approx(wl)}
    assert np.allclose(got[5], X[:, 2]) and feats[5].short == "Coefficient 3" and feats[5].units == ""
    assert [f.id for f in feats] == ["f1", "f2", "f3", "f4", "f5", "f6"] and feats[0].units == str(blk.element.units or "") and feats[4].units == "absorbance"
    assert feats[0].where == {"ex": pytest.approx(ex0), "em": pytest.approx(em0)} and feats[1].where["n_pixels"] == inside.sum()
    # the coordinate of the second block counts from the start of its slice
    f_abs = compile_feature({"kind": "coordinate", "block": "absorbance", "k": 2}, dec.frame)
    assert np.allclose(f_abs.evaluate(F, X), X[:, dec.slices[1].start + 1])
    # a total of a function with missing pixels leaves them out; a box with only missing pixels is NaN
    G = F0.copy()
    G[0, :50] = np.nan
    assert np.isclose(feats[3].evaluate({"fluorescence": G})[0], (F0[0, 50:] @ dom.w[50:]) / dom.w[50:].sum()) and np.isfinite(feats[3].evaluate({"fluorescence": G})).all()
    H = F0.copy()
    H[1, inside] = np.nan
    assert np.isnan(feats[1].evaluate({"fluorescence": H})[1]) and np.isfinite(feats[1].evaluate({"fluorescence": H})[0])


def test_the_default_features_and_the_pixel_that_a_code_moves_most():
    prep = prep_gp(absorbance=True)
    d = default_features(prep.frame)
    assert d == [{"kind": "moved", "block": "fluorescence"}, {"kind": "total", "block": "fluorescence"}]
    only_abs = prep_gp(absorbance=True).frame
    F = np.zeros((5, 10))
    F[:, 7] = np.linspace(0, 3, 5)
    F[:, 2] = np.linspace(0, 2, 5)
    assert moved_index(F) == 7
    f = compile_features([{"kind": "moved"}], prep.frame)[0]
    assert f.per_code and "this code moves most" in f.label and f.short == "Most moved pixel"
    with pytest.raises(AutoencoderError, match="found first"):
        f.evaluate({"fluorescence": F})


@pytest.mark.parametrize("spec,word", [
    ({"kind": "magic"}, "must be one of"), ({"kind": "point", "ex": 350}, "emission wavelength must be a number"), ({"kind": "point", "ex": 350, "em": "x"}, "must be a number"),
    ({"kind": "point", "ex": 5000, "em": 400}, "outside the range"), ({"kind": "point", "ex": 330, "em": 300}, "not part of the model"),
    ({"kind": "region_mean", "ex": [300], "em": [300, 400]}, "two numbers"), ({"kind": "region_max", "ex": [250, 252], "em": [291, 292]}, "No pixel of the model lies inside the box"),
    ({"kind": "absorbance", "wavelength": 300}, "does not use the absorbance"), ({"kind": "coordinate", "k": 0}, "from 1 to"), ({"kind": "coordinate", "k": 9999}, "from 1 to"),
    ({"kind": "coordinate", "k": "a"}, "from 1 to"), ({"kind": "total", "block": "fluorescent"}, "has no 'fluorescent'"), ({"kind": "total", "zzz": 1}, "Unknown feature setting 'zzz'"),
    ("total", "group of settings"),
])
def test_a_feature_that_cannot_be_used_says_why_in_plain_english(spec, word):
    with pytest.raises(AutoencoderError, match=word):
        compile_feature(spec, prep_gp().frame)


def test_features_that_the_network_cannot_have():
    with pytest.raises(AutoencoderError, match="works on the pixels"):
        compile_feature({"kind": "coordinate", "k": 1}, prep_pixels().frame)
    with pytest.raises(AutoencoderError, match="at least one feature"):
        compile_features([], prep_gp().frame)
    with pytest.raises(AutoencoderError, match="At most 6"):
        compile_features([{"kind": "total"}] * 7, prep_gp().frame)
    assert compile_feature({"kind": "total"}, prep_gp(absorbance=True).frame).block == "fluorescence" and compile_feature({"kind": "total", "block": "absorbance"}, prep_gp(absorbance=True).frame).label.startswith("Overall absorbance")


# ----------------------------------------------------------------------------- the curves of single samples and the samples around the curve
def test_curves_of_single_samples_are_parallel_when_the_codes_act_separately_and_not_when_they_interact():
    prep = prep_gp()
    sep, *_ = build(prep, 3, "tanh", [[1.0, 0, 0], [0, 1.0, 0]])                            # separate functions of z1 and z2: additive
    tog, *_ = build(prep, 3, "tanh", [[1.0, 1.0, 0], [0.3, 0, 1.0]])                        # the two codes meet in one unit
    feats = compile_features([{"kind": "total"}], sep.frame)
    ys = {}
    for name, dec in (("sep", sep), ("tog", tog)):
        rc = response_curves(dec, feats, 2.5, 21, ice={"codes": [0], "n": 8})
        ice = rc["ice"][0]
        ys[name] = (rc["curves"]["f1"][0], ice["y"]["f1"], ice["samples"], ice["x"])
    mean, yy, idx, x = ys["sep"]
    shifted = yy - yy[:, [10]]
    assert np.allclose(shifted, shifted[0][None, :], atol=1e-10) and np.allclose(shifted[0], mean - mean[10], atol=1e-10)          # additive: the same shape, a different level
    mean, yy, idx, x = ys["tog"]
    shifted = yy - yy[:, [10]]
    assert np.abs(shifted - (mean - mean[10])[None, :]).max() > 1e-3 * np.abs(mean).max() and idx.size == 8 and len(set(idx.tolist())) == 8
    # the samples are chosen over the distribution of the code, each once; typed samples are used as they are
    pos = tog.position(0)
    assert np.all(np.diff(pos[idx]) >= 0) and pos[idx][0] < -1 and pos[idx][-1] > 1
    typed = response_curves(tog, feats, 2.5, 21, ice={"codes": [1], "samples": [3, 5]})["ice"][1]
    assert typed["samples"].tolist() == [3, 5] and np.allclose(typed["x"], tog.position(1)[[3, 5]])
    assert response_curves(tog, feats, 2.5, 21)["ice"] is None and response_curves(tog, feats, 2.5, 21, ice={"codes": []})["ice"] is None


def test_ice_samples_spread_over_the_distribution():
    x = np.random.default_rng(0).normal(size=100)
    idx = ice_samples(x, 12)
    assert idx.size == 12 and len(set(idx.tolist())) == 12 and np.all(np.diff(x[idx]) >= 0)
    ranks = np.argsort(np.argsort(x))[idx]
    assert np.all(np.diff(ranks) >= 8) and ranks[0] < 8 and ranks[-1] > 91                                                  # evenly over the ranks, from near the lowest to near the highest
    assert ice_samples(x, 500).size == 100 and ice_samples(x, 0).size == 0 and np.array_equal(ice_samples(x[:3], 12), np.argsort(x[:3]))


@pytest.mark.parametrize("prep", PREPS)
def test_the_samples_around_the_curve_are_in_standard_deviations_of_their_own_code(prep):
    P = prep()
    codes = standard_codes(30, 3, 7) * np.array([2.0, 0.5, 1.0]) + np.array([1.0, -1.0, 0.0])
    dec, spec, arrays, pat = build(P, 3, "tanh", [[0.9, 0.4, 0]], codes=codes)
    feats = compile_features([{"kind": "total"}, {"kind": "moved"}], dec.frame)
    X = dec.decode(dec.codes)
    Fs = dec.functions(X)
    rc = response_curves(dec, feats, 3, 41, sample_functions=Fs, sample_coords=X)
    ov = rc["overlay"]
    assert np.allclose(ov["x"], (codes - codes.mean(axis=0)) / codes.std(axis=0, ddof=1)) and ov["n"] == 30
    assert np.allclose(ov["y"]["f1"], Fs["fluorescence"] @ dec.block("fluorescence").domain.w) and ov["y"]["f1"].shape == (30,)
    j = 0
    assert ov["y"]["f2"].shape == (3, 30) and np.allclose(ov["y"]["f2"][j], Fs["fluorescence"][:, int(np.argmax(np.abs(loading_image(dec, pat[0]))))])
    # a measured sample lies on the curve of the code when the decoder has no other code: y(x) = curve(x) for a decoder of one code only
    one, *_ = build(P, 1, "tanh", [[0.9]], codes=standard_codes(30, 1, 2))
    X1 = one.decode(one.codes)
    rc1 = response_curves(one, compile_features([{"kind": "total"}], one.frame), 3, 41, sample_functions=one.functions(X1), sample_coords=X1)
    assert np.allclose(rc1["overlay"]["y"]["f1"], np.interp(rc1["overlay"]["x"][:, 0], rc1["t"], rc1["curves"]["f1"][0]), atol=2e-3 * np.ptp(rc1["curves"]["f1"][0]))


# ----------------------------------------------------------------------------- local effects
@pytest.mark.parametrize("prep", PREPS)
def test_the_loading_at_a_sample_is_the_derivative_of_the_decoder_there(prep):
    P = prep()
    codes = standard_codes(40, 3, 3) * np.array([1.3, 1.0, 1.0])
    dec, spec, arrays, pat = build(P, 3, "tanh", [[0.8, 0, 0]], codes=codes)
    A = loading_image(dec, pat[0])
    idx = loading_samples(dec, 0)
    assert idx.size == 3 and len(set(idx.tolist())) == 3
    pos = dec.position(0)
    q = [np.quantile(pos, 0.1), np.quantile(pos, 0.5), np.quantile(pos, 0.9)]
    assert all(abs(pos[i] - qq) <= np.min(np.abs(pos - qq)) + 1e-12 for i, qq in zip(idx, q))                            # the sample nearest to each quantile
    le = local_effects(dec, 0, idx)
    sd = dec.sd[0]
    for k, i in enumerate(idx):
        z = dec.codes[i, 0]
        deriv = 0.8 * (1 - np.tanh(0.8 * z) ** 2) * sd                                                                   # per 1 SD of the code
        assert np.allclose(le["samples"]["fluorescence"][k], A * deriv, rtol=1e-8, atol=1e-12)
    assert np.allclose(le["at_mean"]["fluorescence"], A * 0.8 * dec.sd[0] * (1 - np.tanh(0.8 * dec.mean[0]) ** 2), rtol=1e-8, atol=1e-12)
    assert np.allclose(le["cosine"], 1.0) and abs(le["change"]["mean"][0]) < 1e-12                                       # the same picture, only weaker or stronger
    # the pixel-wise SD over all the samples is the SD of the strength times the picture
    strength = 0.8 * (1 - np.tanh(0.8 * dec.codes[:, 0]) ** 2) * sd
    assert np.allclose(le["variability"]["fluorescence"], np.abs(A) * strength.std(ddof=1), rtol=1e-7, atol=1e-12)
    sizes = strength / strength.mean()
    ratio = strength / (0.8 * (1 - np.tanh(0.8 * dec.mean[0]) ** 2) * sd)
    assert le["change"]["size_mean"][0] == pytest.approx(ratio.mean(), rel=1e-9) and le["change"]["size_sd"][0] == pytest.approx(ratio.std(), rel=1e-7) and sizes.std() > 0.01


def test_a_code_whose_picture_changes_shape_has_a_cosine_below_one():
    prep = prep_gp()
    codes = standard_codes(60, 3, 5)
    dec, spec, arrays, pat = build(prep, 3, "tanh", [[1.2, 0, 0], [2.0, 0, 0]], b0=[0.0, 1.0], codes=codes)               # two pictures that fade in and out at different codes
    le = local_effects(dec, 0, np.arange(60))
    om = dec.omega
    z = dec.codes[:, 0]
    L = (1.2 * (1 - np.tanh(1.2 * z) ** 2))[:, None] * pat[0][None] + (2.0 * (1 - np.tanh(2.0 * z + 1.0) ** 2))[:, None] * pat[1][None]
    L0 = (1.2 * (1 - np.tanh(1.2 * dec.mean[0]) ** 2)) * pat[0] + (2.0 * (1 - np.tanh(2.0 * dec.mean[0] + 1.0) ** 2)) * pat[1]
    cos = (L @ (om * L0)) / (np.sqrt((L * L) @ om) * np.sqrt(L0 @ (om * L0)))
    assert le["cosine"].shape == (60,) and np.allclose(le["cosine"], cos, atol=1e-9)
    assert le["change"]["mean"][0] == pytest.approx((1 - cos).mean(), rel=1e-7) and le["change"]["max"][0] == pytest.approx((1 - cos).max(), rel=1e-7) and le["change"]["mean"][0] > 0.01
    assert np.isnan(le["change"]["mean"][1:]).all()                                                                       # the other codes are ignored by the decoder: no loading to compare
    # the same cosine in the functions of the pixels (the metric of the loss is the L2 metric of the functions): one block, so the weights of the domain
    blk = dec.block("fluorescence")
    LF = np.stack([loading_image(dec, L[i]) for i in (0, 7, 30)])
    L0F = loading_image(dec, L0)
    w = blk.domain.w
    cf = (LF @ (w * L0F)) / (np.sqrt((LF * LF) @ w) * np.sqrt(L0F @ (w * L0F)))
    assert np.allclose(cf, cos[[0, 7, 30]], atol=1e-8)


def test_local_effects_do_not_depend_on_the_size_of_the_batches():
    dec, *_ = build(prep_pixels(), 3, "tanh", [[1.2, 0, 0], [2.0, 0.5, 0]], b0=[0.0, 1.0])
    big = local_effects(dec, 0, [1, 2], floats=NLN.FLOATS_JACOBIAN)
    small = local_effects(dec, 0, [1, 2], floats=dec.P * 3 * 3)                                                           # about three samples at a time
    assert np.allclose(big["variability"]["fluorescence"], small["variability"]["fluorescence"], rtol=1e-9, atol=1e-14)
    assert np.allclose(big["change"]["mean"], small["change"]["mean"], equal_nan=True) and np.allclose(big["change"]["size_sd"], small["change"]["size_sd"], equal_nan=True)


# ----------------------------------------------------------------------------- two codes
def hand_decoder(act, W0, b0=None, **kw):
    return build(prep_gp(), 3, act, W0, b0=b0, **kw)


def test_separate_functions_of_two_codes_do_not_interact():
    dec, *_ = hand_decoder("tanh", [[1.0, 0, 0], [0, 1.0, 0]])
    feats = compile_features(default_features(dec.frame), dec.frame)
    pe = pair_effects(dec, 0, 1, 5, 2.0, feats[1])
    assert abs(pe["index"]) < 1e-12 and np.isfinite(pe["index"]) and pe["interaction"]["fluorescence"].max() < 1e-12 and abs(pe["index_by_block"]["fluorescence"]) < 1e-12
    assert pe["u"].tolist() == pytest.approx([-2, -1, 0, 1, 2]) and pe["F"]["fluorescence"].shape == (5, 5, dec.block("fluorescence").domain.m)


@pytest.mark.parametrize("grid", [3, 5, 7])
def test_codes_that_meet_in_one_unit_interact_by_the_analytic_amount(grid):
    a = 0.9
    dec, spec, arrays, pat = hand_decoder("tanh", [[a, a, 0]])
    pe = pair_effects(dec, 0, 1, grid, 2.0)
    u = positions(2.0, grid)
    G = np.tanh(a * (u[None, :] + u[:, None]))                                                                           # G[b, a]: rows are code 2, columns code 1
    assert pe["index"] == pytest.approx(grid_interaction_index(G), rel=1e-9) and pe["index"] > 0.05
    assert pe["index_by_block"]["fluorescence"] == pytest.approx(pe["index"], rel=1e-9)                                   # one pattern: the same share in the functions
    c = grid // 2
    I = G - G[:, c:c + 1] - G[c:c + 1, :] + G[c, c]
    pat_img = np.abs(loading_image(dec, pat[0]))
    assert np.allclose(pe["interaction"]["fluorescence"], pat_img * np.abs(I).max(), rtol=1e-8, atol=1e-12)


def test_a_relu_unit_gives_the_hand_computed_interaction():
    """relu(z1 + z2) on the grid {-1, 0, 1}: the interaction is -1 where the signs differ (two cells), the variation of the picture is 38/9: the index is 9/19."""
    dec, spec, arrays, pat = build(prep_gp(), 3, "relu", [[1.0, 1.0, 0]])
    pe = pair_effects(dec, 0, 1, 3, 1.0)
    assert pe["index"] == pytest.approx(9 / 19, rel=1e-9) and pe["interaction_ss"] / pe["variation_ss"] == pytest.approx(9 / 19, rel=1e-9)
    om = dec.omega
    assert pe["interaction_ss"] == pytest.approx(2 * (om * pat[0] ** 2).sum(), rel=1e-9) and pe["variation_ss"] == pytest.approx(38 / 9 * (om * pat[0] ** 2).sum(), rel=1e-9)


def test_the_heat_map_is_the_feature_over_the_plane_and_the_dots_are_the_samples():
    codes = standard_codes(25, 3, 4)
    dec, spec, arrays, pat = build(prep_gp(), 3, "tanh", [[0.9, 0.9, 0], [0, 0.5, 1.0]], codes=codes)
    feats = compile_features([{"kind": "total"}], dec.frame)
    X = dec.decode(dec.codes)
    Fs = dec.functions(X)
    pe = pair_effects(dec, 0, 1, 5, 2.0, feats[0], heat_points=9, sample_functions=Fs, sample_coords=X)
    v = pe["v"]
    assert v.tolist() == pytest.approx(np.linspace(-2, 2, 9).tolist()) and pe["heat"].shape == (9, 9)
    for b in (0, 3, 8):                                                                                                   # rows: code 2, columns: code 1
        for a in (0, 5, 8):
            Z = dec.mean.copy()
            Z[0], Z[1] = dec.mean[0] + v[a] * dec.sd[0], dec.mean[1] + v[b] * dec.sd[1]
            assert pe["heat"][b, a] == pytest.approx(feats[0].evaluate(dec.functions(dec.decode(Z[None])))[0], rel=1e-10)
    assert np.allclose(pe["dots"]["x"], dec.position(0)) and np.allclose(pe["dots"]["y"], dec.position(1)) and np.allclose(pe["dots"]["value"], Fs["fluorescence"] @ dec.block("fluorescence").domain.w)
    moved = pair_effects(dec, 0, 1, 5, 2.0, compile_features([{"kind": "moved"}], dec.frame)[0], heat_points=5)
    assert set(moved["pixel"]) == {"ex", "em"} and moved["heat"].shape == (5, 5)
    none = pair_effects(dec, 0, 1, 3, 1.0)
    assert "heat" not in none and none["dots"]["value"] is None


def test_the_pair_needs_two_codes_and_a_grid_of_3_5_or_7():
    dec, *_ = hand_decoder("tanh", [[1.0, 1.0, 0]])
    with pytest.raises(AutoencoderError, match="two different codes"):
        pair_effects(dec, 1, 1)
    with pytest.raises(AutoencoderError, match="3, 5 or 7"):
        pair_effects(dec, 0, 1, grid=4)
    # a picture that does not change at all has no index
    flat, *_ = hand_decoder("tanh", [[0.0, 0.0, 0.0]])
    assert np.isnan(pair_effects(flat, 0, 1, 3, 1.0)["index"])


def test_the_default_pair_is_the_two_codes_with_the_largest_nonlinear_share():
    assert default_pair(np.array([0.1, 0.5, 0.3, 0.9])) == (1, 3)
    assert default_pair(np.array([0.9, 0.5, 0.3, 0.9])) == (0, 3)
    assert default_pair(np.array([np.nan, 0.2, 0.1])) == (1, 2) and default_pair(np.array([0.2, np.nan])) == (0, 1)


def test_a_decoder_is_checked_against_its_codes():
    dec, spec, arrays, pat = hand_decoder("tanh", [[1.0, 0, 0]])
    with pytest.raises(AutoencoderError, match="3 columns"):
        Decoder(spec, arrays, dec.frame, np.zeros((5, 2)))
