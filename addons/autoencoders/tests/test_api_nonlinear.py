"""The four routes that show which nonlinear trend each code fits (``POST /{rid}/response | bends | local | pair``): what they answer, how they agree with the numerics of
``core/nonlinear.py``, the plain-English refusals, and an outdated network.  The training is replaced by a stand-in (``nl_support.use_bending_training``: a network whose decoder bends,
or the exact linear autoencoder), so these tests need no PyTorch; ``test_api_nonlinear_training.py`` does it with networks trained for real."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

import api_support as A
import nl_support as N
from nl_support import RID, arr, decoder, msg, post

from fluoviz_addon_autoencoders.core import nonlinear as NLN
from fluoviz_addon_autoencoders.core.features import compile_features
from fluoviz_addon_autoencoders.server.results import model_of

B = A.BASE
INP = {"variables": ["fluorescence"], "fluorescence": {"transform": "none", "min_coverage": 0.9, "basis": {"kind": "bspline", "n_basis": [8, 6]}, "smoothing": {"method": "gcv"}}}
INP2 = {**INP, "variables": ["fluorescence", "absorbance"], "absorbance": {"transform": "none", "processed": False, "basis": {"kind": "bspline", "n_basis": 12}, "smoothing": {"method": "gcv"}}}


def make_world(tmp_path, monkeypatch, bending: bool = True):
    S, app, active = A.make_app(tmp_path / "ws")
    N.use_bending_training(monkeypatch) if bending else A.use_fake_training(monkeypatch)
    return S, app


@pytest.fixture()
def world(tmp_path, monkeypatch):
    S, app = make_world(tmp_path, monkeypatch)
    with TestClient(app) as c:
        c.S = S
        c.data = A.load_project(S, n=36, absorbance=True)
        yield c
    S.shutdown()


@pytest.fixture()
def straight(tmp_path, monkeypatch):
    S, app = make_world(tmp_path, monkeypatch, bending=False)
    with TestClient(app) as c:
        c.S = S
        c.data = A.load_project(S, n=36, absorbance=True)
        yield c
    S.shutdown()


def fit(c, **body):
    body = {"latent_dim": 3, "input": INP, "epochs": 5, **body}
    j = A.run_job(c, f"{B}/fit", body)
    assert j["status"] == "done", j
    return A.decode(j["result"])


# ----------------------------------------------------------------------------- 1. response curves
def test_the_default_response_curves_are_the_pixel_each_code_moves_most_and_the_overall_intensity(world):
    fit(world)
    o = post(world, "response", {})
    dec, r = decoder(world)
    assert o["span"] == 3.0 and o["n_points"] == 41 and arr(o["t"]).tolist() == pytest.approx(np.linspace(-3, 3, 41).tolist()) and o["names"] == ["z1", "z2", "z3"]
    assert [f["kind"] for f in o["features"]] == ["moved", "total"] and [f["id"] for f in o["features"]] == ["f1", "f2"] and o["features"][0]["per_code"] is True and o["features"][1]["per_code"] is False
    assert o["features"][1]["label"] == "Overall intensity (the average over the whole EEM)" and o["features"][0]["units"] == o["units"]["fluorescence"]
    assert np.allclose(o["sd"], r.payload["code_sd"]) and np.allclose(o["mean"], r.payload["code_mean"])                          # the unit and the centre of the traversals
    assert len(o["codes"]) == 3
    t = arr(o["t"])
    for j, c in enumerate(o["codes"]):
        assert c["code"] == j + 1 and c["name"] == f"z{j + 1}" and set(c["curves"]) == {"f1", "f2"} and set(c["pixel"]) == {"f1"} and c["ice"] is None
        X, F = dec.decode_functions(dec.along(j, t))
        px = c["pixel"]["f1"]
        co = dec.block("fluorescence").domain.coords()
        hit = np.flatnonzero((co[:, 0] == px["em"]) & (co[:, 1] == px["ex"]))[0]
        assert hit == NLN.moved_index(F["fluorescence"]) and np.allclose(c["curves"]["f1"]["y"], F["fluorescence"][:, hit])         # the pixel with the largest range
        assert np.allclose(c["curves"]["f2"]["y"], F["fluorescence"] @ dec.block("fluorescence").domain.w)
        for fid in ("f1", "f2"):
            d = c["curves"][fid]
            assert 0.0 <= d["r2_linear"] <= 1.0 and d["max_deviation"] >= 0 and isinstance(d["words"], str) and d["label"] in d["words"] and len(d["y"]) == 41 and d["kind"] in (
                "linear", "convex", "concave", "s_shaped", "saturating", "peak", "valley", "wiggly", "flat")
    assert any(c["curves"][fid]["kind"] not in ("linear", "flat") for c in o["codes"] for fid in ("f1", "f2"))               # the stand-in network bends
    assert "Most moved pixel (excitation" in o["codes"][0]["curves"]["f1"]["words"] and o["codes"][0]["curves"]["f2"]["words"].startswith("Overall intensity: ")


def test_the_training_samples_are_drawn_around_the_curves_in_the_units_of_the_code(world):
    fit(world)
    o = post(world, "response", {})
    dec, r = decoder(world)
    ov = o["overlay"]
    assert ov["ids"] == r.sample_ids and arr(ov["x"]).shape == (36, 3) and np.allclose(ov["x"], (arr(r.payload["codes"]) - r.payload["code_mean"]) / r.payload["code_sd"])
    assert arr(ov["y"]["f1"]).shape == (3, 36) and arr(ov["y"]["f2"]).shape == (36,)
    # the feature of a sample's smoothed function: the rebuilt function is near it, and it lies among the values of the curve (loosely)
    ev = model_of(r).evaluate(__import__("fluoviz_addon_autoencoders.server.prep", fromlist=["x"]).apply_data(world.S.project, r.sample_ids, ["fluorescence"]), 0.0)
    smooth = model_of(r).frame.to_functions(ev.X)["fluorescence"]
    assert np.allclose(ov["y"]["f2"], smooth @ dec.block("fluorescence").domain.w)
    px = o["codes"][1]["pixel"]["f1"]
    co = dec.block("fluorescence").domain.coords()
    hit = np.flatnonzero((co[:, 0] == px["em"]) & (co[:, 1] == px["ex"]))[0]
    assert np.allclose(arr(ov["y"]["f1"])[1], smooth[:, hit])
    off = post(world, "response", {"overlay": False})
    assert off["overlay"] is None and off["notes"] == []


def test_a_sample_that_lacks_part_of_the_domain_is_drawn_only_where_it_was_measured(world):
    world.S.project.add_region([300.0, 340.0], [400.0, 450.0], samples=[f"S{i:02d}" for i in range(0, 36, 4)])                 # nine samples lack this rectangle
    fit(world, input={**INP, "fluorescence": {**INP["fluorescence"], "min_coverage": 0.5}})
    o = post(world, "response", {"features": [{"kind": "region_mean", "ex": [300, 340], "em": [400, 450]}, {"kind": "total"}]})
    y1, y2 = arr(o["overlay"]["y"]["f1"]).astype(float), arr(o["overlay"]["y"]["f2"]).astype(float)
    own = np.arange(36) % 4 == 0
    assert np.isnan(y1[own]).all() and np.isfinite(y1[~own]).all()              # the box lies inside the rectangle: those samples have nothing to draw there (their values there are only estimates)
    assert np.isfinite(y2).all()                                                # the overall intensity is read from the part that was measured


def test_curves_of_single_samples_are_asked_for_by_code(world):
    fit(world)
    o = post(world, "response", {"ice": {"codes": [1, 3], "n": 5}, "overlay": False})
    dec, r = decoder(world)
    assert o["codes"][1]["ice"] is None
    for j in (0, 2):
        ice = o["codes"][j]["ice"]
        assert len(ice["samples"]) == 5 and len(set(ice["samples"])) == 5 and set(ice["y"]) == {"f1", "f2"} and arr(ice["y"]["f2"]).shape == (5, 41) and arr(ice["x"]).shape == (5,)
        idx = [r.sample_ids.index(s) for s in ice["samples"]]
        assert np.allclose(ice["x"], dec.position(j)[idx]) and np.all(np.diff(arr(ice["x"])) >= 0)
        Z = np.tile(dec.codes[idx[2]], (41, 1))
        Z[:, j] = dec.mean[j] + arr(o["t"]) * dec.sd[j]
        assert np.allclose(arr(ice["y"]["f2"])[2], dec.functions(dec.decode(Z))["fluorescence"] @ dec.block("fluorescence").domain.w)
    typed = post(world, "response", {"ice": {"samples": ["S03", "S09"], "codes": [2]}, "overlay": False})
    assert typed["codes"][1]["ice"]["samples"] == ["S03", "S09"] and typed["codes"][0]["ice"] is None
    every = post(world, "response", {"ice": True, "overlay": False})
    assert all(c["ice"] is not None and len(c["ice"]["samples"]) == 12 for c in every["codes"])                                 # (a network has 8 codes at most drawn at first: here all 3)
    assert post(world, "response", {"ice": False})["codes"][0]["ice"] is None and post(world, "response", {"ice": None})["codes"][0]["ice"] is None


def test_every_kind_of_feature_can_be_asked_for(world):
    fit(world, input=INP2)
    dec, r = decoder(world)
    co = dec.block("fluorescence").domain.coords()
    ex0, em0 = float(co[100, 1]), float(co[100, 0])
    feats = [{"kind": "point", "ex": ex0, "em": em0}, {"kind": "region_mean", "ex": [float(co[:, 1].min()), float(co[:, 1].min()) + 40], "em": [float(co[:, 0].min()), float(co[:, 0].min()) + 60]},
             {"kind": "region_max", "ex": [250, 320], "em": [350, 450]}, {"kind": "total"}, {"kind": "absorbance", "wavelength": 300}, {"kind": "coordinate", "k": 2}, {"kind": "total", "block": "absorbance"}]
    o = post(world, "response", {"features": feats[:6], "n_points": 21, "span": 2})
    assert [f["kind"] for f in o["features"]] == ["point", "region_mean", "region_max", "total", "absorbance", "coordinate"] and o["span"] == 2.0 and len(o["t"]) == 21
    assert o["features"][4]["units"] == "absorbance" and o["features"][5]["units"] == "" and o["features"][0]["where"] == {"ex": pytest.approx(ex0), "em": pytest.approx(em0)}
    assert o["features"][0]["label"].startswith("Intensity at excitation") and o["features"][1]["label"].startswith("Mean intensity in the box") and o["features"][2]["label"].startswith("Highest intensity")
    assert o["features"][4]["label"] == f"Absorbance at {o['features'][4]['where']['wavelength']:g} nm" and o["features"][5]["short"] == "Coefficient 2"
    compiled = compile_features(feats[:6], dec.frame)
    for j in range(3):
        X, F = dec.decode_functions(dec.along(j, arr(o["t"])))
        for k, f in enumerate(compiled):
            assert np.allclose(o["codes"][j]["curves"][f"f{k + 1}"]["y"], f.evaluate(F, X)), (j, f.kind)
    assert not any("pixel" in c and c["pixel"] for c in o["codes"])                                                              # none of these is resolved per code
    ov = o["overlay"]
    assert set(ov["y"]) == {f"f{k}" for k in range(1, 7)} and all(arr(v).shape == (36,) for v in ov["y"].values())
    total_abs = post(world, "response", {"features": [feats[6]]})
    assert total_abs["features"][0]["label"] == "Overall absorbance (the average over all wavelengths)"
    moved_abs = post(world, "response", {"features": [{"kind": "moved", "block": "absorbance"}]})
    assert "wavelength" in moved_abs["codes"][0]["pixel"]["f1"] and moved_abs["features"][0]["label"].startswith("Absorbance at the wavelength")


@pytest.mark.parametrize("body,word,code", [
    ({"span": 9}, "from 0.5 to 4", 400), ({"span": "wide"}, "must be a number", 400), ({"n_points": 3}, "from 5 to 201", 400), ({"n_points": 20.5}, "whole number", 400), ({"n_points": 500}, "from 5 to 201", 400),
    ({"features": []}, "at least one feature", 400), ({"features": "total"}, "must be a list of features", 400), ({"features": [{"kind": "total"}] * 7}, "At most 6", 400),
    ({"features": [{"kind": "wavelet"}]}, "must be one of", 400), ({"features": [{"kind": "point", "ex": 350}]}, "emission wavelength must be a number", 400),
    ({"features": [{"kind": "point", "ex": 5000, "em": 400}]}, "outside the range", 400),     ({"features": [{"kind": "region_mean", "ex": [250], "em": [300, 400]}]}, "two numbers", 400), ({"features": [{"kind": "absorbance", "wavelength": 300}]}, "does not use the absorbance", 400),
    ({"features": [{"kind": "coordinate", "k": 0}]}, "from 1 to", 400), ({"overlay": "yes"}, "true or false", 400), ({"bogus": 1}, "Unknown setting 'bogus'", 400),
    ({"ice": "all"}, "true, false or a group", 400), ({"ice": {"codes": [9]}}, "from 1 to 3", 400), ({"ice": {"codes": []}}, "list of code numbers", 400), ({"ice": {"n": 99}}, "from 1 to 12", 400),
    ({"ice": {"samples": ["nope"]}}, "not one of the samples", 404), ({"ice": {"samples": []}}, "list of sample names", 400), ({"ice": {"samples": ["S01", "S01"]}}, "only once", 400),
    ({"ice": {"samples": [f"S{i:02d}" for i in range(13)]}}, "At most 12", 400), ({"ice": {"zzz": 1}}, "Unknown setting 'zzz'", 400),
])
def test_a_wrong_request_for_response_curves_is_a_sentence_never_a_crash(world, body, word, code):
    fit(world)
    r = post(world, "response", body, ok=False)
    assert r.status_code == code and word in msg(r), (body, r.text)


def test_a_pixel_outside_the_model_is_refused_with_a_sentence(world):
    fit(world)
    dec, r = decoder(world)
    dom = dec.block("fluorescence").domain
    iem, iex = np.argwhere(~dom.obs)[0]                                                                                         # a pixel of the grid that is scatter / below the Rayleigh line
    ex, em = float(dom.axes[1][iex]), float(dom.axes[0][iem])
    bad = post(world, "response", {"features": [{"kind": "point", "ex": ex, "em": em}]}, ok=False)
    assert bad.status_code == 400 and "not part of the model" in msg(bad) and f"excitation {ex:g} nm and emission {em:g} nm" in msg(bad)
    good_em, good_ex = np.argwhere(dom.obs)[0]
    ok = post(world, "response", {"features": [{"kind": "point", "ex": float(dom.axes[1][good_ex]), "em": float(dom.axes[0][good_em])}]})
    assert ok["features"][0]["kind"] == "point"
    emp = post(world, "response", {"features": [{"kind": "region_max", "ex": [ex, ex], "em": [em, em]}]}, ok=False)
    assert emp.status_code == 400 and "No pixel of the model lies inside the box" in msg(emp)


def test_a_missing_result_and_a_foreign_result_are_refused(world):
    fit(world)
    for route in ("response", "bends", "local", "pair"):
        assert post(world, route, {}, rid="autoencoder-9", ok=False).status_code == 404
        other = world.S.project.add_result("pca", "A PCA", {}, {}, {})
        r = post(world, route, {}, rid=other.id, ok=False)
        assert r.status_code == 400 and "not an autoencoder" in msg(r)
    assert post(world, "response", [], ok=False).status_code in (400, 422)


# ----------------------------------------------------------------------------- 2. where the response bends
def test_the_bend_maps_are_images_in_the_orientation_of_the_browser(world):
    fit(world, input=INP2)
    o = post(world, "bends", {})
    dec, r = decoder(world)
    ref = NLN.bend_maps(dec, 3.0, 41)
    assert o["span"] == 3.0 and o["n_points"] == 41 and o["min_effect"] == 0.02 and o["nearly_straight"] == 0.05 and o["names"] == ["z1", "z2", "z3"]
    assert np.allclose(o["share"], ref["share"], equal_nan=True) and np.allclose(o["effect_share"], ref["effect_share"]) and o["dead"] == [False] * 3
    fl, ab = o["fluorescence"], o["absorbance"]
    assert arr(fl["nl"]).shape == (3, 16, 24) and arr(fl["amplitude"]).shape == (3, 16, 24) and arr(ab["nl"]).shape == (3, 40) and arr(ab["amplitude"]).shape == (3, 40) and arr(ab["x"]).shape == (40,)
    nl = arr(fl["nl"])
    assert np.nanmin(nl) >= 0 and np.nanmax(nl) <= 1 and np.isnan(nl).any() and np.isfinite(nl).any()                           # grey pixels: outside the model or without an effect
    mask = np.asarray(o["mask"], bool)
    assert np.isnan(nl[:, mask]).all() and arr(o["ex"]).shape == (16,) and arr(o["em"]).shape == (24,)
    emb = dec.block("fluorescence").domain.embed(ref["blocks"]["fluorescence"]["nl"])                                          # module orientation (n_em, n_ex)
    assert np.allclose(nl, np.transpose(emb, (0, 2, 1)), equal_nan=True, atol=1e-6)
    assert np.allclose(fl["share"], ref["blocks"]["fluorescence"]["share"], equal_nan=True) and np.allclose(fl["n_effect"], ref["blocks"]["fluorescence"]["n_effect"])
    assert all(0 <= s <= 1 for s in o["share"]) and isinstance(o["words"], str) and "bends the most" in o["words"] and o["words"].count("%") >= 1
    narrow = post(world, "bends", {"span": 1, "n_points": 11})
    assert narrow["span"] == 1.0 and len(narrow["t"]) == 11 and arr(narrow["share"]).max() <= arr(o["share"]).max() + 1e-9


def test_a_straight_network_has_nothing_nonlinear_at_all(straight):
    fit(straight)
    o = post(straight, "bends", {})
    assert max(o["share"]) < 1e-9 and np.nanmax(arr(o["fluorescence"]["nl"])) < 1e-6 and o["words"].startswith("Every code acts almost like a straight line")
    resp = post(straight, "response", {})
    assert all(c["curves"][f]["kind"] in ("linear", "flat") and c["curves"][f]["r2_linear"] > 1 - 1e-9 for c in resp["codes"] for f in ("f1", "f2") if c["curves"][f]["kind"] != "flat")
    loc = post(straight, "local", {"code": 1})
    assert np.nanmax(arr(loc["change"]["mean"])) < 1e-9 and np.nanmax(arr(loc["fluorescence"]["variability"])) < 1e-6 * np.nanmax(np.abs(arr(loc["fluorescence"]["at_mean"])))
    assert loc["words"] == "The picture of its effect has the same shape for every sample and about the same strength."
    pair = post(straight, "pair", {})
    assert abs(pair["index"]) < 1e-9 and "act almost independently" in pair["words"]


@pytest.mark.parametrize("body,word", [({"span": 0}, "from 0.5 to 4"), ({"n_points": 4}, "from 5 to 201"), ({"features": []}, "Unknown setting 'features'"), ({"span": True}, "must be a number")])
def test_a_wrong_request_for_the_bend_maps_is_a_sentence(world, body, word):
    fit(world)
    r = post(world, "bends", body, ok=False)
    assert r.status_code == 400 and word in msg(r)


# ----------------------------------------------------------------------------- 3. local effects
def test_local_loadings_at_chosen_samples_and_how_they_vary(world):
    fit(world, input=INP2)
    o = post(world, "local", {"code": 2})
    dec, r = decoder(world)
    idx = NLN.loading_samples(dec, 1)
    ref = NLN.local_effects(dec, 1, idx)
    assert o["code"] == 2 and o["name"] == "z2" and o["sample_sd"] == pytest.approx(dec.sd[1]) and [s["id"] for s in o["samples"]] == [r.sample_ids[i] for i in idx]
    assert [s["label"] for s in o["samples"]] == ["10th percentile of z2", "median of z2", "90th percentile of z2"]
    assert [s["position"] for s in o["samples"]] == pytest.approx(dec.position(1)[idx].tolist()) and o["samples"][0]["position"] < o["samples"][1]["position"] < o["samples"][2]["position"]
    fl, ab = o["fluorescence"], o["absorbance"]
    assert arr(fl["at_mean"]).shape == (16, 24) and arr(fl["samples"]).shape == (3, 16, 24) and arr(fl["variability"]).shape == (16, 24)
    assert arr(ab["at_mean"]).shape == (40,) and arr(ab["samples"]).shape == (3, 40) and arr(ab["variability"]).shape == (40,) and arr(ab["x"]).shape == (40,)
    emb = lambda v: np.transpose(dec.block("fluorescence").domain.embed(v), (0, 2, 1) if np.ndim(v) == 2 else (1, 0))
    assert np.allclose(arr(fl["samples"]), emb(ref["samples"]["fluorescence"]), equal_nan=True, atol=1e-6) and np.allclose(arr(fl["variability"]), emb(ref["variability"]["fluorescence"]), equal_nan=True, atol=1e-6)
    assert np.allclose(arr(fl["at_mean"]), emb(ref["at_mean"]["fluorescence"]), equal_nan=True, atol=1e-6)
    assert [s["cosine"] for s in o["samples"]] == pytest.approx(ref["cosine"].tolist())
    for k in ("mean", "max", "size_mean", "size_sd"):
        assert arr(o["change"][k]).shape == (3,) and np.allclose(o["change"][k], ref["change"][k], equal_nan=True)
    assert np.nanmin(arr(o["change"]["mean"])) >= -1e-12 and isinstance(o["words"], str) and o["words"].startswith("The picture of its effect")
    typed = post(world, "local", {"code": 3, "samples": ["S05", "S11"]})
    assert [s["id"] for s in typed["samples"]] == ["S05", "S11"] and all(s["label"] == "chosen by you" for s in typed["samples"]) and arr(typed["fluorescence"]["samples"]).shape == (2, 16, 24)


@pytest.mark.parametrize("body,word,code", [
    ({"code": 0}, "from 1 to 3", 400), ({"code": 4}, "from 1 to 3", 400), ({"code": "z1"}, "from 1 to 3", 400), ({"code": 1.5}, "from 1 to 3", 400),
    ({"code": 1, "samples": ["nope"]}, "not one of the samples", 404), ({"code": 1, "samples": []}, "list of sample names", 400), ({"code": 1, "samples": [1]}, "list of sample names", 400),
    ({"code": 1, "samples": [f"S{i:02d}" for i in range(7)]}, "At most 6", 400), ({"code": 1, "samples": ["S01", "S01"]}, "only once", 400), ({"codes": [1]}, "Unknown setting 'codes'", 400),
])
def test_a_wrong_request_for_local_effects_is_a_sentence(world, body, word, code):
    fit(world)
    r = post(world, "local", body, ok=False)
    assert r.status_code == code and word in msg(r), (body, r.text)


# ----------------------------------------------------------------------------- 4. two codes together
def test_two_codes_on_a_grid_with_the_interaction_the_heat_map_and_the_samples(world):
    fit(world, input=INP2)
    o = post(world, "pair", {"codes": [1, 3], "grid": 5, "span": 2, "heat_points": 15})
    dec, r = decoder(world)
    assert o["codes"] == [1, 3] and o["auto"] is False and o["grid"] == 5 and o["span"] == 2.0 and arr(o["u"]).tolist() == pytest.approx([-2, -1, 0, 1, 2]) and o["index_limits"] == [0.05, 0.25]
    ref = NLN.pair_effects(dec, 0, 2, 5, 2.0, compile_features([{"kind": "moved"}], dec.frame)[0], 15)
    assert o["index"] == pytest.approx(ref["index"], rel=1e-9) and o["interaction_ss"] == pytest.approx(ref["interaction_ss"], rel=1e-9) and 0 <= o["index"] < 5
    assert isinstance(o["words"], str) and "z1 and z3" in o["words"] and o["index_by_block"].keys() == {"fluorescence", "absorbance"}
    gal = arr(o["gallery"]["fluorescence"]["images"])
    assert gal.shape == (25, 16, 24)                                                                                           # index = position of code 3 * 5 + position of code 1
    centre = gal[2 * 5 + 2]
    mean_fun = dec.functions(dec.decode(dec.mean[None]))["fluorescence"][0]
    assert np.allclose(centre, dec.block("fluorescence").domain.embed(mean_fun).T, equal_nan=True, atol=1e-5)               # the middle picture is the decoding of the mean code
    corner = dec.mean.copy()
    corner[0], corner[2] = dec.mean[0] + 2 * dec.sd[0], dec.mean[2] - 2 * dec.sd[2]
    assert np.allclose(gal[0 * 5 + 4], dec.block("fluorescence").domain.embed(dec.functions(dec.decode(corner[None]))["fluorescence"][0]).T, equal_nan=True, atol=1e-5)    # code 1 at +2 SD, code 3 at -2 SD
    gab = o["gallery"]["absorbance"]
    assert arr(gab["series"]).shape == (25, 40) and arr(gab["x"]).shape == (40,) and arr(o["interaction"]["fluorescence"]).shape == (16, 24) and arr(o["interaction"]["absorbance"]).shape == (40,)
    assert np.nanmin(arr(o["interaction"]["fluorescence"])) >= 0
    heat = o["heat"]
    assert arr(heat["z"]).shape == (15, 15) and arr(heat["x"]).shape == (15,) and np.allclose(heat["z"], ref["heat"]) and o["feature"]["kind"] == "moved" and set(o["feature"]["where"]) == {"ex", "em"}
    dots = o["dots"]
    assert dots["ids"] == r.sample_ids and np.allclose(dots["x"], dec.position(0)) and np.allclose(dots["y"], dec.position(2)) and arr(dots["value"]).shape == (36,)
    assert o["notes"] == []


def test_the_two_codes_are_chosen_by_their_nonlinear_share_when_none_are_named(world):
    fit(world)
    o = post(world, "pair", {})
    dec, r = decoder(world)
    share = NLN.bend_maps(dec, 3.0, 41)["share"]
    i, j = NLN.default_pair(share)
    assert o["auto"] is True and o["codes"] == [i + 1, j + 1] and sorted(np.argsort(-share)[:2].tolist()) == [i, j]
    assert o["grid"] == 5 and o["span"] == 2.0
    for g in (3, 7):
        assert arr(post(world, "pair", {"codes": [1, 2], "grid": g})["gallery"]["fluorescence"]["images"]).shape[0] == g * g
    feat = post(world, "pair", {"codes": [1, 2], "feature": {"kind": "total"}, "heat_points": 11})
    assert feat["feature"]["kind"] == "total" and arr(feat["heat"]["z"]).shape == (11, 11)


@pytest.mark.parametrize("body,word,code", [
    ({"codes": [1]}, "two code numbers", 400), ({"codes": [1, 1]}, "two different codes", 400), ({"codes": [1, 9]}, "from 1 to 3", 400), ({"codes": "z1,z2"}, "two code numbers", 400),
    ({"grid": 4}, "3, 5 or 7", 400), ({"grid": "5"}, "3, 5 or 7", 400), ({"span": 7}, "from 0.5 to 4", 400), ({"heat_points": 5}, "from 11 to 61", 400),
    ({"feature": {"kind": "magic"}}, "must be one of", 400), ({"feature": {"kind": "absorbance", "wavelength": 300}}, "does not use the absorbance", 400), ({"zzz": 1}, "Unknown setting 'zzz'", 400),
])
def test_a_wrong_request_for_two_codes_is_a_sentence(world, body, word, code):
    fit(world)
    r = post(world, "pair", body, ok=False)
    assert r.status_code == code and word in msg(r), (body, r.text)


def test_a_network_with_one_code_has_no_pair(world):
    fit(world, latent_dim=1)
    r = post(world, "pair", {}, ok=False)
    assert r.status_code == 400 and "Two codes are needed" in msg(r)
    assert post(world, "bends", {})["names"] == ["z1"] and post(world, "local", {"code": 1})["name"] == "z1" and post(world, "response", {})["codes"][0]["name"] == "z1"


# ----------------------------------------------------------------------------- an outdated network, a variational autoencoder, an absorbance-only network
def test_an_outdated_network_can_still_be_asked_but_the_samples_are_not_drawn(world):
    fit(world)
    first = post(world, "response", {})
    assert world.put("/api/state/corrections", json={"smoothing": {"enabled": True, "em_nm": 8}}).status_code == 200
    assert world.get(f"{B}/{RID}").json()["result"]["stale"]
    o = post(world, "response", {"ice": {"codes": [1], "n": 4}})
    assert o["overlay"] is None and len(o["notes"]) == 1 and "data changed since this network was trained" in o["notes"][0]
    assert np.allclose(o["codes"][0]["curves"]["f2"]["y"], first["codes"][0]["curves"]["f2"]["y"]) and len(o["codes"][0]["ice"]["samples"]) == 4         # nothing here needs the data
    assert post(world, "bends", {})["words"] and post(world, "local", {"code": 1})["words"]
    p = post(world, "pair", {"codes": [1, 2]})
    assert p["dots"]["value"] is None and len(p["notes"]) == 1 and "data changed" in p["notes"][0] and p["dots"]["ids"]
    off = post(world, "response", {"overlay": False})
    assert off["notes"] == []


def test_a_variational_autoencoder_is_asked_through_its_posterior_means(world):
    fit(world, family="vae")
    r = world.S.project.get_result(RID)
    assert r.payload["family"] == "vae"
    o = post(world, "response", {})
    assert len(o["codes"]) == 3 and np.allclose(arr(o["overlay"]["x"])[:, 0], (arr(r.payload["codes"])[:, 0] - r.payload["code_mean"][0]) / r.payload["code_sd"][0])
    assert post(world, "pair", {})["codes"] and post(world, "bends", {})["share"] and post(world, "local", {"code": 2})["name"] == "z2"


def test_a_network_that_reads_only_the_absorbance_works(tmp_path, monkeypatch):
    S, app = make_world(tmp_path, monkeypatch)
    with TestClient(app) as c:
        c.S = S
        A.load_project(S, n=36, absorbance=True)
        inp = {"variables": ["absorbance"], "absorbance": {"transform": "none", "processed": False, "basis": {"kind": "bspline", "n_basis": 12}, "smoothing": {"method": "gcv"}}}
        j = A.run_job(c, f"{B}/fit", {"latent_dim": 2, "input": inp, "epochs": 5})
        assert j["status"] == "done", j
        o = post(c, "response", {})
        assert [f["kind"] for f in o["features"]] == ["moved", "total"] and o["features"][0]["label"].startswith("Absorbance at the wavelength") and "wavelength" in o["codes"][0]["pixel"]["f1"] and "ex" not in o
        assert post(c, "response", {"features": [{"kind": "absorbance", "wavelength": 300}]})["features"][0]["short"].startswith("Absorbance at")
        b = post(c, "bends", {})
        assert "fluorescence" not in b and arr(b["absorbance"]["nl"]).shape == (2, 40)
        loc = post(c, "local", {"code": 1})
        assert arr(loc["absorbance"]["at_mean"]).shape == (40,) and "fluorescence" not in loc
        pr = post(c, "pair", {})
        assert "series" in pr["gallery"]["absorbance"] and arr(pr["gallery"]["absorbance"]["series"]).shape == (25, 40) and arr(pr["interaction"]["absorbance"]).shape == (40,)
        assert post(c, "response", {"features": [{"kind": "point", "ex": 300, "em": 400}]}, ok=False).status_code == 400
    S.shutdown()


def test_a_pixel_network_can_be_asked_everything_except_coefficients(world):
    px = {"variables": ["fluorescence"], "fluorescence": {"basis": {"kind": "pixels"}, "smoothing": {"method": "none"}}}
    fit(world, input=px)
    assert post(world, "response", {})["codes"][0]["curves"]["f1"]["words"]
    r = post(world, "response", {"features": [{"kind": "coordinate", "k": 1}]}, ok=False)
    assert r.status_code == 400 and "works on the pixels" in msg(r)
    assert arr(post(world, "bends", {})["fluorescence"]["nl"]).shape == (3, 16, 24) and arr(post(world, "pair", {"codes": [1, 2]})["gallery"]["fluorescence"]["images"]).shape[0] == 25


def test_looking_does_not_change_the_network(world):
    fit(world)
    r = world.S.project.get_result(RID)
    before = {k: np.array(v) for k, v in r.payload["frozen"]["model"]["arrays"].items()}
    post(world, "response", {"ice": True})
    post(world, "bends", {})
    post(world, "local", {"code": 2})
    post(world, "pair", {})
    for k, v in r.payload["frozen"]["model"]["arrays"].items():
        assert np.array_equal(before[k], v)
