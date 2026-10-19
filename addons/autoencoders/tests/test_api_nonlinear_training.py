"""The four routes that show the nonlinear effects of the codes (``POST /{rid}/response | bends | local | pair``) on networks trained FOR REAL (PyTorch in the worker process, started by the API):
an autoencoder with fluorescence and absorbance, a variational autoencoder and a pixel network.  The numbers of the routes are compared with independent calculations on the network the server
rebuilds from the saved weights (the same numpy decoder that the page's traversals use; the server never imports PyTorch).  ``test_api_nonlinear.py`` has the many details with a stand-in for
the training; this file shows that they hold for what the training really gives.  Skipped cleanly when PyTorch is not installed."""
import importlib.util

import numpy as np
import pytest
from fastapi.testclient import TestClient

import api_support as A
from nl_support import arr, decoder, msg, post

from fluoviz_addon_autoencoders.core import nonlinear as NLN

pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch is not installed")
B = A.BASE
INP = {"variables": ["fluorescence"], "fluorescence": {"transform": "none", "min_coverage": 0.9, "basis": {"kind": "bspline", "n_basis": [8, 6]}, "smoothing": {"method": "gcv"}}}
INP2 = {**INP, "variables": ["fluorescence", "absorbance"], "absorbance": {"transform": "none", "processed": False, "basis": {"kind": "bspline", "n_basis": 12}, "smoothing": {"method": "gcv"}}}
PIX = {"variables": ["fluorescence"], "fluorescence": {"basis": {"kind": "pixels"}, "smoothing": {"method": "none"}}}
FAST = {"latent_dim": 3, "epochs": 60, "patience": 30, "n_restarts": 1, "val_fraction": 0.2, "seed": 4}
KINDS = {"flat", "wiggly", "peak", "valley", "linear", "s_shaped", "saturating", "convex", "concave"}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """One project and three networks trained for real, shared by the tests of this file: an autoencoder (fluorescence and absorbance), a variational autoencoder and a pixel network."""
    S, app, _ = A.make_app(tmp_path_factory.mktemp("nlreal") / "ws")
    with TestClient(app) as c:
        c.S = S
        c.data = A.load_project(S, n=36, absorbance=True)
        c.rid = {}
        for name, body in (("ae", {"input": INP2}), ("vae", {"family": "vae", "input": INP, "epochs": 80}), ("pixels", {"input": PIX})):
            j = A.run_job(c, f"{B}/fit", {**FAST, **body}, timeout=900)
            assert j["status"] == "done", j
            c.rid[name] = A.decode(j["result"])["rid"]
        yield c
    S.shutdown()


def share_ok(v) -> bool:
    return v is None or (np.isfinite(v) and -1e-9 <= v <= 1 + 1e-9)


def test_response_curves_of_a_trained_network_are_what_the_decoder_gives_along_each_code(world):
    rid = world.rid["ae"]
    dec, r = decoder(world, rid)
    o = post(world, "response", {"ice": {"codes": [1, 2], "n": 4}}, rid)
    assert o["names"] == ["z1", "z2", "z3"] and len(o["codes"]) == 3 and arr(o["t"]).shape == (41,) and o["overlay"] is not None and o["notes"] == []
    assert [f["kind"] for f in o["features"]] == ["moved", "total"] and arr(o["weights"]).sum() == pytest.approx(1.0)
    mean_fun = dec.functions(dec.decode(dec.mean[None]))["fluorescence"][0]
    for j, code in enumerate(o["codes"]):
        moved, total = code["curves"]["f1"], code["curves"]["f2"]
        X, F = dec.decode_functions(dec.along(j, arr(o["t"])))
        hit = NLN.moved_index(F["fluorescence"])
        assert code["pixel"]["f1"] is not None and np.allclose(moved["y"], F["fluorescence"][:, hit])                      # the pixel this code moves most, followed along the code
        assert np.allclose(total["y"], F["fluorescence"] @ dec.block("fluorescence").domain.w)                                # the overall intensity: the quadrature average over the pixels
        assert arr(moved["y"])[20] == pytest.approx(mean_fun[hit], rel=1e-9, abs=1e-9)                                        # at the position 0 the decoding of the mean code
        for cv in (moved, total):
            assert cv["kind"] in KINDS and cv["label"] and cv["words"] and np.isfinite(arr(cv["y"])).all()
            assert cv["r2_linear"] is None or -1e-9 <= cv["r2_linear"] <= 1 + 1e-9
            assert cv["max_deviation"] is None or 0 <= cv["max_deviation"] <= 1 + 1e-9
    assert np.allclose(arr(o["overlay"]["x"]), np.column_stack([dec.position(j) for j in range(3)]))                       # the samples in SD of each code
    ice = o["codes"][0]["ice"]
    assert len(ice["samples"]) == 4 and arr(ice["y"]["f2"]).shape == (4, 41) and o["codes"][2]["ice"] is None


def test_the_bend_maps_and_the_shares_of_a_trained_network(world):
    rid = world.rid["ae"]
    dec, r = decoder(world, rid)
    o = post(world, "bends", {}, rid)
    ref = NLN.bend_maps(dec, 3.0, 41)
    assert np.allclose(o["share"], ref["share"], equal_nan=True) and all(share_ok(s) for s in o["share"]) and o["dead"] == [False] * 3
    assert arr(o["effect_share"]).sum() == pytest.approx(1.0) and o["nearly_straight"] == 0.05 and ("bends the most" in o["words"] or "nearly linear" in o["words"])
    nl = arr(o["fluorescence"]["nl"])
    assert nl.shape == (3, 16, 24) and np.nanmin(nl) >= 0 and np.nanmax(nl) <= 1 and arr(o["absorbance"]["nl"]).shape == (3, 40)
    assert all(share_ok(s) for s in o["fluorescence"]["share"]) and all(share_ok(s) for s in o["absorbance"]["share"])
    assert max(s for s in o["share"] if s is not None) > 0.0                                                                  # a trained decoder is never exactly a straight line


def test_the_local_loading_at_the_mean_code_is_the_derivative_of_the_decoder_and_the_rest_vary_with_the_sample(world):
    rid = world.rid["ae"]
    dec, r = decoder(world, rid)
    o = post(world, "local", {"code": 2}, rid)
    h = 1e-3
    X, F = dec.decode_functions(dec.along(1, np.array([-h, h])))
    fd = (F["fluorescence"][1] - F["fluorescence"][0]) / (2 * h)                                                              # per SD of the code
    want = np.transpose(dec.block("fluorescence").domain.embed(fd))
    got = arr(o["fluorescence"]["at_mean"])
    assert np.allclose(got, want, rtol=1e-3, atol=1e-6 * np.nanmax(np.abs(want)), equal_nan=True)
    assert [s["label"] for s in o["samples"]] == ["10th percentile of z2", "median of z2", "90th percentile of z2"] and all(-1 - 1e-9 <= s["cosine"] <= 1 + 1e-9 for s in o["samples"])
    assert arr(o["fluorescence"]["samples"]).shape == (3, 16, 24) and np.nanmin(arr(o["fluorescence"]["variability"])) >= 0
    assert arr(o["change"]["mean"]).shape == (3,) and np.nanmin(arr(o["change"]["mean"])) >= -1e-12 and isinstance(o["words"], str)


def test_two_codes_of_a_trained_network(world):
    rid = world.rid["ae"]
    dec, r = decoder(world, rid)
    auto = post(world, "pair", {"heat_points": 11}, rid)
    share = NLN.bend_maps(dec, 3.0, 41)["share"]
    top = sorted(np.argsort(-np.nan_to_num(share))[:2] + 1)
    assert auto["auto"] is True and auto["codes"] == [int(top[0]), int(top[1])] and auto["grid"] == 5 and auto["span"] == 2.0
    assert auto["index"] is None or auto["index"] >= 0 and "z" in auto["words"]
    o = post(world, "pair", {"codes": [1, 3], "grid": 3, "heat_points": 11}, rid)
    gal = arr(o["gallery"]["fluorescence"]["images"])
    assert gal.shape == (9, 16, 24) and arr(o["gallery"]["absorbance"]["series"]).shape == (9, 40)
    assert np.allclose(gal[4], np.transpose(dec.block("fluorescence").domain.embed(dec.functions(dec.decode(dec.mean[None]))["fluorescence"][0])), equal_nan=True, atol=1e-6)    # the middle picture
    assert arr(o["heat"]["z"]).shape == (11, 11) and np.nanmin(arr(o["interaction"]["fluorescence"])) >= 0 and arr(o["dots"]["x"]).shape == (36,)


def test_a_variational_autoencoder_is_looked_at_through_its_posterior_means(world):
    rid = world.rid["vae"]
    dec, r = decoder(world, rid)
    assert r.payload["latent"]["kind"] == "vae"
    o = post(world, "response", {}, rid)
    assert np.allclose(o["mean"], r.payload["code_mean"]) and np.allclose(o["sd"], r.payload["code_sd"])
    assert np.allclose(arr(r.payload["codes"])[:, 0], arr(r.variables["z1"]))                                                  # the codes of the page are the posterior means
    assert np.allclose(arr(o["overlay"]["x"]), np.column_stack([dec.position(j) for j in range(3)]))
    b = post(world, "bends", {}, rid)
    assert all(share_ok(s) for s in b["share"]) and isinstance(b["words"], str)
    assert post(world, "local", {"code": 1}, rid)["name"] == "z1" and post(world, "pair", {"heat_points": 11}, rid)["auto"] is True


def test_a_pixel_network_trained_for_real_answers_everything_except_coefficients(world):
    rid = world.rid["pixels"]
    dec, r = decoder(world, rid)
    co = dec.block("fluorescence").domain.coords()
    feats = [{"kind": "moved"}, {"kind": "total"}, {"kind": "point", "ex": float(co[50, 1]), "em": float(co[50, 0])}]
    o = post(world, "response", {"features": feats, "n_points": 21}, rid)
    assert len(o["codes"]) == 3 and all(arr(c["curves"][f"f{k}"]["y"]).shape == (21,) for c in o["codes"] for k in (1, 2, 3))
    bad = post(world, "response", {"features": [{"kind": "coordinate", "k": 1}]}, rid, ok=False)
    assert bad.status_code == 400 and "only for a functional network" in msg(bad) or "exist only for a functional network" in msg(bad)
    b = post(world, "bends", {}, rid)
    assert arr(b["fluorescence"]["nl"]).shape == (3, 16, 24) and all(share_ok(s) for s in b["share"])
    lo = post(world, "local", {"code": 3}, rid)
    assert arr(lo["fluorescence"]["samples"]).shape == (3, 16, 24) and lo["samples"][0]["label"].startswith("10th percentile")
    p = post(world, "pair", {"codes": [2, 3], "grid": 3, "heat_points": 11}, rid)
    assert arr(p["gallery"]["fluorescence"]["images"]).shape == (9, 16, 24)


def test_asking_does_not_change_the_trained_network(world):
    rid = world.rid["ae"]
    before = {k: np.array(v, copy=True) for k, v in world.S.project.get_result(rid).payload["frozen"]["model"]["arrays"].items()}
    codes = np.array(world.S.project.get_result(rid).payload["codes"], copy=True)
    for route, body in (("response", {"ice": True}), ("bends", {}), ("local", {"code": 1}), ("pair", {})):
        post(world, route, body, rid)
    after = world.S.project.get_result(rid).payload["frozen"]["model"]["arrays"]
    assert before.keys() == after.keys() and all(np.array_equal(before[k], np.asarray(after[k])) for k in before)
    assert np.array_equal(codes, np.asarray(world.S.project.get_result(rid).payload["codes"]))


# ----------------------------------------------------------------------------- data that really bends: the code that moves a band bends, the code of the amount does not
@pytest.fixture(scope="module")
def bending(tmp_path_factory):
    """The made-up EEMs of the guide's picture (a band whose emission maximum moves with a hidden condition, and its amount), a network with 2 codes trained for real."""
    import help_data as H
    S, app, _ = A.make_app(tmp_path_factory.mktemp("nlbend") / "ws")
    with TestClient(app) as c:
        c.S = S
        Y, t, a = H.eems_bending()
        S.project.set_eem(Y, H.EX, H.EM, [f"S{i + 1:03d}" for i in range(Y.shape[0])])
        j = A.run_job(c, f"{B}/fit", {"latent_dim": 2, "n_restarts": 1, "seed": 1, "input": {"variables": ["fluorescence"], "fluorescence": H.GP_INPUT["fluorescence"]}}, timeout=900)
        assert j["status"] == "done", j
        c.rid = A.decode(j["result"])["rid"]
        c.t, c.a = t, a
        yield c
    S.shutdown()


def test_a_network_trained_on_bending_data_shows_which_code_bends(bending):
    rid = bending.rid
    dec, r = decoder(bending, rid)
    b = post(bending, "bends", {}, rid)
    share = arr(b["share"])
    hi, lo = int(np.argmax(share)), int(np.argmin(share))
    assert share[hi] > 0.05 > share[lo] and "bends the most" in b["words"]                                                      # one code bends, the other acts like a PCA loading
    o = post(bending, "response", {}, rid)
    bent, straight = o["codes"][hi]["curves"]["f1"], o["codes"][lo]["curves"]["f1"]
    assert bent["kind"] not in ("linear", "flat") and bent["r2_linear"] < 0.97
    assert straight["kind"] in ("linear", "flat") and (straight["kind"] == "flat" or straight["r2_linear"] > 0.98)
    p = post(bending, "pair", {}, rid)
    assert p["auto"] is True and p["codes"] == [1, 2] and p["index"] is not None and p["index"] >= 0
