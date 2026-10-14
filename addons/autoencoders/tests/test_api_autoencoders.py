"""The API of the add-on (``/api/addon/autoencoders``): advice, checks, the fit job and its history, the result for the page, one sample, traversals, interpolation, generation, sensitivity,
anomalies, exports, methods, staleness, plain-English errors.  The training is replaced by the exact linear autoencoder (``api_support.fake_train``) so these tests need no PyTorch;
``test_api_training.py`` trains for real."""
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient

import api_support as A
import kind_support as K

B = A.BASE
INP = {"variables": ["fluorescence"], "fluorescence": {"transform": "none", "min_coverage": 0.9, "basis": {"kind": "bspline", "n_basis": [8, 6]}, "smoothing": {"method": "gcv"}}}
INP2 = {**INP, "variables": ["fluorescence", "absorbance"], "absorbance": {"transform": "none", "processed": False, "basis": {"kind": "bspline", "n_basis": 12}, "smoothing": {"method": "gcv"}}}


@pytest.fixture()
def world(tmp_path, monkeypatch):
    S, app, active = A.make_app(tmp_path / "ws")
    A.use_fake_training(monkeypatch)
    with TestClient(app) as c:
        c.active = active
        c.S = S
        c.data = A.load_project(S, n=36, absorbance=True)
        yield c
    S.shutdown()


def fit(c, **body):
    body = {"latent_dim": 3, "input": INP, "epochs": 5, **body}
    j = A.run_job(c, f"{B}/fit", body)
    assert j["status"] == "done", j
    return A.decode(j["result"])


def msg(r):
    return r.json()["error"]["message"]


# ----------------------------------------------------------------------------- advice and checks
def test_defaults_facts_and_a_request_that_can_be_run(world):
    r = world.get(f"{B}/defaults", params={"variables": "fluorescence,absorbance"})
    assert r.status_code == 200, r.text
    d = A.decode(r.json())
    assert d["n_samples"] == 36 and d["fluorescence"]["available"] and d["absorbance"]["available"] and d["n_common"] == 36
    assert d["components_99"]["pca"] == 3 and d["components_99"]["fpca"] is None
    adv = d["advice"]
    assert adv["latent_dim"] == 3 and adv["size"] == "small" and adv["hidden"] == [64, 32] and adv["family"] == "ae" and adv["estimate"]["seconds"] > 0 and "second" in adv["estimate"]["words"]
    assert "describe 99 %" in adv["latent_reason"] and isinstance(adv["warnings"], list) and adv["blocking"] == []
    assert d["torch"]["available"] and set(d["sizes"]) == {"small", "medium", "large"} and "silu" in d["activations"]
    req = d["request"]
    assert req["input"]["variables"] == ["fluorescence", "absorbance"] and req["input"]["fluorescence"]["basis"]["kind"] == "gp" and req["family"] == "ae" and req["smoothify"] == 0.0
    one = A.decode(world.get(f"{B}/defaults").json())
    assert one["request"]["input"]["variables"] == ["fluorescence"]
    only = A.decode(world.get(f"{B}/defaults", params={"variables": "absorbance"}).json())
    assert only["request"]["input"]["variables"] == ["absorbance"] and "fluorescence" not in only["request"]["input"] and only["n_common"] == 36 and only["components_99"]["pca"] >= 1
    fp = A.decode(world.get(f"{B}/defaults", params={"advise": 1}).json())
    assert fp["components_99"]["fpca"] >= 1 and fp["advice"]["representation"]["fluorescence"]["lengths"]
    for q, word in (({"variables": "magic"}, "fluorescence, absorbance"), ({"variables": ""}, "fluorescence, absorbance"), ({"min_coverage": 5}, "min_coverage")):
        bad = world.get(f"{B}/defaults", params=q)
        assert bad.status_code == 400 and word in msg(bad), (q, bad.text)


def test_defaults_without_data_say_so(tmp_path):
    S, app, _ = A.make_app(tmp_path / "ws2")
    with TestClient(app) as c:
        r = c.get(f"{B}/defaults")
        assert r.status_code == 400 and "No data loaded" in msg(r)
    S.shutdown()


def test_check_reports_the_numbers_a_person_needs(world):
    out = A.done(world, f"{B}/check", {"latent_dim": 3, "input": INP2})
    assert out["ok"] and out["n"] == 36 and out["input"]["variables"] == ["fluorescence", "absorbance"] and out["input"]["mode"] == "functional"
    assert [b["name"] for b in out["input"]["blocks"]] == ["fluorescence", "absorbance"] and out["input"]["n_inputs"] == sum(b["n_inputs"] for b in out["input"]["blocks"])
    net = out["network"]
    assert net["n_params"] > 5000 and net["hidden"] == [64, 32] and 0 < net["samples_per_param"] < 0.01
    assert out["validation"]["mode"] == "holdout" and out["validation"]["n_val"] >= 5 and out["validation"]["n_train"] + out["validation"]["n_val"] == 36 and "held out" in out["validation"]["words"]
    assert any("memorise" in w for w in out["warnings"]) and out["blocking"] == [] and out["estimate"]["seconds"] > 0 and out["torch"]["available"]
    small = A.done(world, f"{B}/check", {"latent_dim": 3, "input": INP, "val_fraction": 0.3})
    assert small["validation"]["n_val"] > out["validation"]["n_val"]


def test_check_with_few_samples_blocks_and_says_why(tmp_path, monkeypatch):
    S, app, _ = A.make_app(tmp_path / "ws3")
    A.use_fake_training(monkeypatch)
    with TestClient(app) as c:
        A.load_project(S, n=9)
        out = A.done(c, f"{B}/check", {"latent_dim": 2, "input": INP})
        assert not out["ok"] and "needs at least 12 samples" in out["blocking"][0]
        j = A.run_job(c, f"{B}/fit", {"latent_dim": 2, "input": INP})
        assert j["status"] == "error" and "at least 12 samples" in j["error"]
        out = A.done(c, f"{B}/fit", {"latent_dim": 2, "input": INP, "allow_few_samples": True, "epochs": 3})               # the person was asked and said yes
        pl = A.decode(c.get(f"{B}/autoencoder-1").json())["payload"]
        assert out["result"]["id"] == "autoencoder-1" and pl["split"]["mode"] == "kfold" and np.asarray(pl["split"]["flag"]).all() and "9 samples" in pl["split"]["words"]
    S.shutdown()


@pytest.mark.parametrize("body,word", [
    ({"latent_dim": 0}, "between 1 and 32"), ({"latent_dim": "many"}, "must be a number"), ({"family": "gan"}, "must be one of"), ({"bogus": 1}, "Unknown setting 'bogus'"),
    ({"activation": "swish"}, "must be one of"), ({"hidden": "huge"}, "network size"), ({"size": "huge"}, "network size"), ({"size": "custom"}, "list of layer widths"),
    ({"dropout": 2}, "dropout"), ({"epochs": 0}, "epochs"), ({"val_fraction": 0.99}, "held out"), ({"vae": {"prior": "uniform"}}, "prior"), ({"vae": {"zzz": 1}}, "Unknown variational"),
    ({"corruption": {"loud": 1}}, "Unknown denoising"), ({"smoothify": -1}, "smoothify"), ({"smoothify_metric": "rough"}, "smoothify_metric"), ({"seed": -3}, "seed"),
    ({"input": {"variables": []}}, "Choose what to analyse"), ({"input": {"fluorescence": {"basis": {"kind": "wavelets"}}}}, "basis must be one of"),
    ({"input": {"fluorescence": {"transform": "magic"}}}, "pre-treatment"), ({"input": {"fluorescence": {"basis": {"kind": "bspline"}, "smoothing": {"method": "ml"}}}}, "Maximum likelihood is only available"),
    ({"input": {"weights": "variance"}}, "Unknown data setting 'weights'"), ({"input": "fluorescence"}, "group of settings"), ({"title": 3}, "title must be text"), ({"replace": 3}, "'replace'"),
])
def test_requests_are_checked_in_plain_english(world, body, word):
    for url in ("check", "fit"):
        r = world.post(f"{B}/{url}", json={"latent_dim": 3, **body})
        assert r.status_code == 400 and word in msg(r), (url, body, r.text)


def test_a_request_for_the_absorbance_without_spectra_is_refused_in_the_job(tmp_path, monkeypatch):
    S, app, _ = A.make_app(tmp_path / "ws4")
    A.use_fake_training(monkeypatch)
    with TestClient(app) as c:
        A.load_project(S, n=20)
        j = A.run_job(c, f"{B}/check", {"input": {"variables": ["absorbance"]}})
        assert j["status"] == "error" and "No absorbance spectra are loaded" in j["error"]
    S.shutdown()


# ----------------------------------------------------------------------------- the fit and its result
def test_the_fit_saves_a_numbered_result_with_variables_and_a_replayable_request(world):
    out = fit(world, family="ae", latent_dim=3, title="My network")
    res = out["result"]
    assert out["rid"] == "autoencoder-1" and res["id"] == "autoencoder-1" and res["kind"] == "autoencoder" and res["title"] == "My network" and res["stale"] == [] and "payload" not in res
    assert res["params"]["latent_dim"] == 3 and res["params"]["size"] == "small" and res["params"]["hidden"] == [64, 32] and res["params"]["input"]["variables"] == ["fluorescence"]
    assert res["summary"]["n"] == 36 and res["summary"]["family"] == "ae" and res["summary"]["r2_val"] > 0.9
    p = world.S.project
    r = p.get_result("autoencoder-1")
    assert set(r.variables) == {"z1", "z2", "z3", "recon_rmse", "recon_r2", "typicality", "flag", "status", "split"} and r.sample_ids == [f"S{i:02d}" for i in range(36)]
    assert r.var_info["z1"]["label"] == "z1 (code 1)" and r.var_info["flag"]["kind"] == "category" and r.var_info["status"]["kind"] == "category"
    cat = world.get("/api/variables").json()["variables"]
    assert any(v["key"] == "res:autoencoder-1:z1" for v in cat) and any(v["key"] == "res:autoencoder-1:recon_r2" for v in cat)
    # the stored request gives the same network again (determinism with a seed)
    again = fit(world, **{k: v for k, v in res["params"].items() if k != "title"})
    assert again["rid"] == "autoencoder-2"
    assert np.allclose(np.asarray(p.get_result("autoencoder-2").variables["z1"]), np.asarray(r.variables["z1"]))
    # train again in place: the same id and title
    rep = fit(world, replace="autoencoder-1")
    assert rep["rid"] == "autoencoder-1" and rep["result"]["title"] == "My network" and len([k for k in p.results if k.startswith("autoencoder")]) == 2
    bad = world.post(f"{B}/fit", json={"replace": "autoencoder-9", "latent_dim": 3, "input": INP})
    assert bad.status_code == 404 and "autoencoder-9" in msg(bad)
    other = p.add_result("pca", "A PCA", {}, {}, {})
    bad = world.post(f"{B}/fit", json={"replace": other.id, "latent_dim": 3, "input": INP})
    assert bad.status_code == 400 and "not an autoencoder" in msg(bad)


def test_the_history_can_be_polled_and_stays_after_the_job(world):
    j = A.run_job(world, f"{B}/fit", {"latent_dim": 3, "input": INP, "epochs": 5})
    jid = j["id"]
    h = world.get(f"{B}/fit/{jid}/history").json()
    assert h["job"] == jid and h["status"] == "done" and h["phase"] == "done" and len(h["epochs"]) == 3 and h["epochs"][0]["epoch"] == 0 and h["next"] >= 4
    assert h["best"]["epoch"] == 2 and h["n_restarts"] == 3 and "event" not in h["epochs"][0]
    again = world.get(f"{B}/fit/{jid}/history", params={"since": h["next"]}).json()
    assert again["epochs"] == [] and again["next"] == h["next"]
    part = world.get(f"{B}/fit/{jid}/history", params={"since": 2}).json()
    assert len(part["epochs"]) < 3 and part["next"] == h["next"]
    assert world.get(f"{B}/fit/nope/history").status_code == 404
    assert world.get(f"{B}/fit/{jid}/history", params={"since": -1}).status_code == 400


def test_the_payload_has_everything_the_page_reads(world):
    fit(world, latent_dim=3)
    r = world.get(f"{B}/autoencoder-1")
    assert r.status_code == 200
    out = A.decode(r.json())
    pl = out["payload"]
    assert out["result"]["id"] == "autoencoder-1" and "payload" not in out["result"] and "frozen" not in pl
    assert pl["n"] == 36 and pl["family"] == "ae" and pl["latent_dim"] == 3 and pl["code_names"] == ["z1", "z2", "z3"] and len(pl["ids"]) == 36
    assert np.asarray(pl["codes"]).shape == (36, 3) and pl["codes_sd"] is None and np.asarray(pl["code_sd"]).shape == (3,)
    assert pl["input"]["mode"] == "functional" and pl["input"]["blocks"][0]["basis"] == "bspline" and pl["network"]["n_params"] > 0 and pl["network"]["spec"]["family"] == "ae"
    sp = pl["split"]
    assert sp["mode"] == "holdout" and sp["n_train"] + sp["n_val"] == 36 and np.asarray(sp["flag"]).sum() == sp["n_val"] and "held out" in sp["words"]
    rc = pl["recon"]
    assert np.allclose(np.asarray(rc["r2"]), 1 - np.asarray(rc["error"])) and rc["limit"] > 0 and rc["limit_level"] == "99" and len(rc["flag"]) == 36 and "99 %" in rc["limit_words"]
    tr = pl["training"]
    assert tr["history"]["epoch"] == [0, 1, 2] and "verdict" in tr and tr["overfit"]["words"] and tr["restarts"][0]["chosen"]
    q = pl["quality"]
    assert q["r2_val"] > 0.9 and q["baseline"]["name"] == "Functional PCA" and q["baseline"]["better"] in ("same", "network", "baseline") and "Functional PCA" in q["baseline"]["verdict"] and len(q["worst"]) == 10 and q["worst"][0]["error"] >= q["worst"][1]["error"]
    assert np.asarray(q["local_r2"]["fluorescence"]).shape == (16, 24) and np.asarray(q["residual_rms"]["fluorescence"]).shape == (16, 24)               # browser orientation (n_ex, n_em)
    la = pl["latent"]
    assert la["kind"] == "ae" and np.asarray(la["correlation"]).shape == (3, 3) and la["effective_dim"] > 0 and la["words"]
    assert pl["smoothness"]["blocks"][0]["name"] == "fluorescence" and "smoothify is off" in pl["smoothness"]["words"]
    assert np.asarray(pl["images"]["mean"]).shape == (1, 16, 24) and np.asarray(pl["images"]["sensitivity"]).shape == (3, 16, 24) and np.asarray(pl["mask"]).shape == (16, 24)
    assert len(pl["ex"]) == 16 and len(pl["em"]) == 24 and pl["units"]
    ci = pl["component_images"]
    assert ci["score_vars"] == ["z1", "z2", "z3"] and np.asarray(ci["images"]).shape == (3, 24, 16) and ci["kind"] == "signed"                                  # the clustering page's contract (module orientation)
    assert pl["limits"][0]["value"] == rc["limit"] and pl["versions"]["seed"] == 0 and isinstance(pl["warnings"], list)
    # the generic route still returns the whole result (the weights included), the add-on's route does not
    full = world.get(f"/api/results/autoencoder-1").json()
    assert "frozen" in full["payload"]
    assert world.get(f"{B}/autoencoder-9").status_code == 404 and "autoencoder-9" in msg(world.get(f"{B}/autoencoder-9"))
    other = world.S.project.add_result("pca", "A PCA", {}, {}, {})
    assert world.get(f"{B}/{other.id}").status_code == 400 and "not an autoencoder" in msg(world.get(f"{B}/{other.id}"))


def test_the_vae_and_two_variables(world):
    out = fit(world, family="vae", latent_dim=3, input=INP2)
    p = world.S.project
    r = p.get_result(out["rid"])
    assert {"z1_sd", "z2_sd", "z3_sd", "kl", "elbo", "z1", "typicality"} <= set(r.variables)
    pl = A.decode(world.get(f"{B}/{out['rid']}").json())["payload"]
    assert pl["family"] == "vae" and np.asarray(pl["codes_sd"]).shape == (36, 3) and pl["latent"]["kind"] == "vae" and np.asarray(pl["latent"]["kl_per_dim"]).shape == (3,)
    assert pl["input"]["variables"] == ["fluorescence", "absorbance"] and np.asarray(pl["curves"]["x"]).shape == (40,) and set(pl["recon"]["error_by_block"]) == {"fluorescence", "absorbance"}
    assert np.asarray(pl["images"]["mean"]).shape == (1, 16, 24)


def test_a_cancelled_fit_saves_nothing(world, monkeypatch):
    import threading
    started = threading.Event()

    def slow(cfg, prepared, history_path, progress):
        started.set()
        for k in range(400):
            progress(k / 400, "Training")
            threading.Event().wait(0.02)
        raise AssertionError("not cancelled")

    A.patch_training(monkeypatch, slow)
    r = world.post(f"{B}/fit", json={"latent_dim": 3, "input": INP})
    jid = r.json()["job_id"]
    assert started.wait(30)
    assert world.post(f"/api/jobs/{jid}/cancel").status_code == 200
    import time
    j = {}
    for _ in range(200):
        j = world.get(f"/api/jobs/{jid}").json()
        if j["status"] not in ("running", "queued"):
            break
        time.sleep(0.05)
    assert j["status"] == "cancelled" and not [k for k in world.S.project.results if k.startswith("autoencoder")]
    assert world.get(f"{B}/fit/{jid}/history").json()["status"] == "cancelled"


# ----------------------------------------------------------------------------- looking at a result
def test_one_sample_against_the_network(world):
    fit(world, latent_dim=3, input=INP2)
    r = world.get(f"{B}/autoencoder-1/sample/S05")
    assert r.status_code == 200, r.text
    s = A.decode(r.json())
    assert s["id"] == "S05" and s["status"] == "ok" and s["words"] == "Fits the model" and s["split"] in ("training", "validation") and len(s["code"]) == 3 and s["code_sd"] is None
    assert 0 <= s["error"] < s["limit"] and s["r2"] == pytest.approx(1 - s["error"]) and s["rmse"] > 0 and s["flag"] == "inside the limits" and "well rebuilt" in s["text"]
    fl, ab = s["blocks"]
    shp = lambda x: np.asarray(x, float).shape
    assert fl["name"] == "fluorescence" and fl["kind"] == "surface" and shp(fl["measured"]) == shp(fl["reconstruction"]) == shp(fl["residual"]) == shp(fl["mean"]) == (16, 24) and shp(fl["mask"]) == (16, 24)
    assert ab["kind"] == "curve" and shp(ab["x"]) == (40,) and shp(ab["measured"]) == (40,) and shp(ab["residual"]) == (40,)
    assert np.isnan(np.asarray(fl["measured"], float)).any() and np.isfinite(np.asarray(fl["measured"], float)).any()
    thumb = A.decode(world.get(f"{B}/autoencoder-1/sample/S05", params={"thumb": 1}).json())
    assert "residual" not in thumb["blocks"][0] and np.asarray(thumb["blocks"][0]["measured"], float).shape == (16, 24)
    bad = world.get(f"{B}/autoencoder-1/sample/nope")
    assert bad.status_code == 404 and "'nope' is not one of the samples" in msg(bad)


def test_traversals_decode_along_a_code_without_the_data(world):
    fit(world, latent_dim=3)
    r = world.post(f"{B}/autoencoder-1/traverse", json={"code": 2, "values": [-2, 0, 2]})
    assert r.status_code == 200, r.text
    t = A.decode(r.json())
    assert t["values"] == [-2.0, 0.0, 2.0] and len(t["codes"]) == 1 and t["codes"][0]["code"] == 2 and t["codes"][0]["name"] == "z2" and t["codes"][0]["sd"] > 0
    f = t["codes"][0]["fluorescence"]
    imgs, diff = np.asarray(f["images"], float), np.asarray(f["difference"], float)
    assert imgs.shape == (3, 16, 24) and diff.shape == (3, 16, 24) and np.nanmax(np.abs(diff[1])) < 1e-9 and f["difference_abs_max"] > 0 and f["max"] > f["min"]
    assert np.allclose(t["codes"][0]["z"], t["codes"][0]["mean"] + np.array([-2, 0, 2]) * t["codes"][0]["sd"]) and np.asarray(t["mean"]["fluorescence"]).shape == (16, 24) and np.asarray(t["ex"]).shape == (16,)
    allc = A.decode(world.post(f"{B}/autoencoder-1/traverse", json={}).json())
    assert [c["code"] for c in allc["codes"]] == [1, 2, 3] and allc["values"] == [-3, -2, -1, 0, 1, 2, 3]
    for body, word in (({"code": 9}, "from 1 to 3"), ({"code": "z1"}, "from 1 to 3"), ({"values": []}, "list of numbers"), ({"values": [9]}, "between -6 and 6"), ({"values": list(range(-6, 8))}, "At most 13"),
                       ({"values": ["a"]}, "list of numbers"), ({"base": "sample"}, "'mean'")):
        bad = world.post(f"{B}/autoencoder-1/traverse", json=body)
        assert bad.status_code == 400 and word in msg(bad), (body, bad.text)


def test_interpolation_next_to_the_linear_mixture(world):
    fit(world, latent_dim=3)
    r = world.post(f"{B}/autoencoder-1/interpolate", json={"a": "S01", "b": "S09", "steps": 5})
    assert r.status_code == 200, r.text
    o = A.decode(r.json())
    assert o["t"] == [0.0, 0.25, 0.5, 0.75, 1.0] and np.asarray(o["code_path"]).shape == (5, 3) and o["a"]["id"] == "S01"
    lat, lin, dif = (np.asarray(o[k]["fluorescence"]["images"], float) for k in ("latent", "linear", "difference"))
    assert lat.shape == lin.shape == dif.shape == (5, 16, 24)
    assert np.allclose(lat - lin, dif, equal_nan=True, atol=1e-5)
    # the ends are the two samples as the network rebuilds them
    s1 = np.asarray(A.decode(world.get(f"{B}/autoencoder-1/sample/S01").json())["blocks"][0]["reconstruction"], float)
    assert np.allclose(lat[0], s1, equal_nan=True, atol=1e-5)
    for body, word in (({"a": "S01"}, "'b' is missing"), ({"a": "S01", "b": "nope"}, "not one of the samples"), ({"a": "S01", "b": "S02", "steps": 2}, "from 3 to 25"), ({"a": 1, "b": "S02"}, "'a' is missing")):
        bad = world.post(f"{B}/autoencoder-1/interpolate", json=body)
        assert bad.status_code in (400, 404) and word in msg(bad), (body, bad.text)


def test_generation_is_for_variational_autoencoders_only(world):
    fit(world, family="ae", latent_dim=3)
    bad = world.post(f"{B}/autoencoder-1/generate", json={"n": 3})
    assert bad.status_code == 400 and "Only variational autoencoders can draw new samples" in msg(bad)
    assert world.get(f"{B}/autoencoder-1/export", params={"what": "synthetic"}).status_code == 400
    fit(world, family="vae", latent_dim=3)
    r = world.post(f"{B}/autoencoder-2/generate", json={"n": 4, "seed": 7, "temperature": 0.5})
    assert r.status_code == 200, r.text
    g = A.decode(r.json())
    assert g["synthetic"] is True and "Synthetic samples" in g["label"] and g["ids"] == [f"synthetic-{k}" for k in range(1, 5)] and np.asarray(g["codes"]).shape == (4, 3)
    assert np.asarray(g["fluorescence"]["images"]).shape == (4, 16, 24) and len(g["nearest"]) == 4 and g["nearest"][0]["id"].startswith("S") and g["nearest"][0]["distance"] >= 0
    again = A.decode(world.post(f"{B}/autoencoder-2/generate", json={"n": 4, "seed": 7, "temperature": 0.5}).json())
    assert np.allclose(again["codes"], g["codes"])                                                           # the same seed, the same samples
    hot = A.decode(world.post(f"{B}/autoencoder-2/generate", json={"n": 4, "seed": 7, "temperature": 1.5}).json())
    assert np.abs(hot["codes"]).max() > np.abs(g["codes"]).max()
    for body, word in (({"n": 0}, "from 1 to 24"), ({"n": 99}, "from 1 to 24"), ({"seed": -1}, "seed"), ({"temperature": 5}, "from 0 to 2")):
        bad = world.post(f"{B}/autoencoder-2/generate", json=body)
        assert bad.status_code == 400 and word in msg(bad), (body, bad.text)


def test_sensitivity_images(world):
    fit(world, latent_dim=3, input=INP2)
    r = world.get(f"{B}/autoencoder-1/sensitivity")
    assert r.status_code == 200, r.text
    s = A.decode(r.json())
    assert s["names"] == ["z1", "z2", "z3"] and len(s["sd"]) == 3
    f = s["fluorescence"]
    at_mean, mag = np.asarray(f["at_mean"], float), np.asarray(f["magnitude"], float)
    assert at_mean.shape == mag.shape == (3, 16, 24) and np.isclose(sum(f["share"]), 1.0) and (np.nan_to_num(mag) >= 0).all()
    assert np.asarray(s["absorbance"]["at_mean"]).shape == (3, 40) and "Code z1 changes the fluorescence mostly around excitation" in s["words"] and "absorbance mostly around" in s["words"]
    # for the exact linear network the effect of a code is the same everywhere: at the mean equals the average magnitude of a signed direction
    assert np.allclose(np.abs(at_mean), mag, equal_nan=True, rtol=1e-3, atol=1e-6)


def test_the_anomaly_view_with_other_thresholds(world):
    fit(world, latent_dim=3)
    out = A.decode(world.post(f"{B}/autoencoder-1/anomalies", json={"method": "saved"}).json())
    assert out["limit"] > 0 and out["words"] and out["histogram"]["limit"] == out["limit"] and len(out["histogram"]["counts"]) == len(out["histogram"]["edges"]) - 1
    tight = A.decode(world.post(f"{B}/autoencoder-1/anomalies", json={"method": "mad", "c": 0.5, "top": 5}).json())
    assert tight["n_flagged"] >= out["n_flagged"] and len(tight["flagged"]) <= 5 and tight["flagged"][0]["error"] >= tight["flagged"][-1]["error"] and np.asarray(tight["flagged"][0]["residual"]["fluorescence"]).shape == (16, 24)
    q = A.decode(world.post(f"{B}/autoencoder-1/anomalies", json={"method": "quantile", "level": 0.9}).json())
    assert 0 < q["n_flagged"] <= 36 * 0.2
    for body, word in (({"method": "magic"}, "mad, quantile or saved"), ({"method": "mad", "c": 99}, "from 0.5 to 20"), ({"method": "quantile", "level": 2}, "0.5 to 0.999"), ({"top": 0}, "from 1 to 50")):
        bad = world.post(f"{B}/autoencoder-1/anomalies", json=body)
        assert bad.status_code == 400 and word in msg(bad), (body, bad.text)


def test_exports(world):
    fit(world, latent_dim=3, family="vae")
    rid = "autoencoder-1"
    codes = world.get(f"{B}/{rid}/export", params={"what": "codes"})
    lines = codes.text.splitlines()
    assert lines[0] == "sample,split,z1,z2,z3,recon_rmse,recon_r2,typicality,flag,z1_sd,z2_sd,z3_sd,kl,elbo,status" and len(lines) == 37 and "autoencoder-1_codes.csv" in codes.headers["content-disposition"]
    assert lines[1].split(",")[0] == "S00" and lines[1].split(",")[1] in ("training", "validation")
    hist = world.get(f"{B}/{rid}/export", params={"what": "history"}).text.splitlines()
    assert hist[0].startswith("restart,fold,epoch,train,val") and len(hist) == 4
    npz = np.load(io.BytesIO(world.get(f"{B}/{rid}/export", params={"what": "reconstruction"}).content))
    assert npz["fluorescence_reconstruction"].shape == (36, 24, 16) and npz["fluorescence_observed"].shape == (36, 24, 16) and npz["fluorescence_residual"].shape == (36, 24, 16) and bool(npz["observed_included"])
    assert npz["fluorescence_mean"].shape == (24, 16) and npz["codes"].shape == (36, 3) and list(npz["ids"])[:2] == ["S00", "S01"] and npz["fluorescence_emission_nm"].shape == (24,)
    syn = world.get(f"{B}/{rid}/export", params={"what": "synthetic", "n": 7, "seed": 3})
    sz = np.load(io.BytesIO(syn.content))
    assert "synthetic" in syn.headers["content-disposition"] and bool(sz["synthetic"]) and sz["fluorescence_synthetic"].shape == (7, 24, 16) and "NOT measured" in str(sz["README"])
    sc = world.get(f"{B}/{rid}/export", params={"what": "synthetic", "n": 5, "format": "csv"}).text.splitlines()
    assert sc[0].startswith("sample,synthetic,z1") and len(sc) == 6 and "made up" in sc[1]
    again = np.load(io.BytesIO(world.get(f"{B}/{rid}/export", params={"what": "synthetic", "n": 7, "seed": 3}).content))
    assert np.allclose(sz["codes"], again["codes"])
    for params, word in (({"what": "pdf"}, "what must be"), ({"what": "synthetic", "n": 0}, "from 1 to 1000"), ({"what": "synthetic", "format": "xml"}, "npz or csv"), ({"what": "synthetic", "temperature": 9}, "from 0 to 2")):
        bad = world.get(f"{B}/{rid}/export", params=params)
        assert bad.status_code == 400 and word in msg(bad), (params, bad.text)


def test_methods_paragraph(world):
    fit(world, latent_dim=3, input=INP2)
    m = world.get(f"{B}/autoencoder-1/methods").json()
    t = m["text"]
    assert "autoencoder was trained on the fluorescence excitation-emission matrices (EEMs) and absorbance spectra of 36 samples" in t and "AdamW" in t and "seed 0" in t and "held out" in t and "Software:" in t
    assert "3 codes" in t and "of the variation" in t and m["references"] and "Loshchilov" in " ".join(m["references"])


# ----------------------------------------------------------------------------- staleness
def test_a_result_that_is_outdated_can_be_looked_at_but_not_trained_silently(world):
    fit(world, latent_dim=3, input=INP2)
    p = world.S.project
    before = np.asarray(p.get_result("autoencoder-1").variables["z1"]).copy()
    assert world.put("/api/state/corrections", json={"smoothing": {"enabled": True, "em_nm": 8}}).status_code == 200
    r = world.get(f"{B}/autoencoder-1")
    assert r.status_code == 200 and r.json()["result"]["stale"]                                           # still readable, says why it is outdated
    bad = world.get(f"{B}/autoencoder-1/sample/S02")
    assert bad.status_code == 400 and "changed since this network was trained" in msg(bad) and "Train again with the same settings" in msg(bad)
    assert world.post(f"{B}/autoencoder-1/anomalies", json={}).status_code == 200 and "residual" not in A.decode(world.post(f"{B}/autoencoder-1/anomalies", json={"method": "mad", "c": 0.5}).json())["flagged"][0]
    assert world.post(f"{B}/autoencoder-1/traverse", json={"code": 1}).status_code == 200                      # these need no data
    assert world.get(f"{B}/autoencoder-1/sensitivity").status_code == 200 and world.get(f"{B}/autoencoder-1/methods").status_code == 200
    ip = A.decode(world.post(f"{B}/autoencoder-1/interpolate", json={"a": "S01", "b": "S02"}).json())
    assert ip["linear"] is None and any("changed since" in n for n in ip["notes"]) and ip["latent"]["fluorescence"]["images"].shape[0] == 9
    npz = np.load(io.BytesIO(world.get(f"{B}/autoencoder-1/export", params={"what": "reconstruction"}).content))
    assert not bool(npz["observed_included"]) and "fluorescence_observed" not in npz.files
    assert world.get(f"{B}/autoencoder-1/export", params={"what": "codes"}).status_code == 200
    # a stale result cannot be saved as a model
    bad = world.post("/api/models", json={"result": "autoencoder-1"})
    assert bad.status_code == 400 and "outdated" in msg(bad)
    # train again with the same settings: a new network on the current data, in place
    params = p.get_result("autoencoder-1").params
    out = fit(world, **{**params, "replace": "autoencoder-1"})
    assert out["result"]["stale"] == [] and out["rid"] == "autoencoder-1"
    assert world.get(f"{B}/autoencoder-1/sample/S02").status_code == 200
    assert not np.allclose(np.asarray(p.get_result("autoencoder-1").variables["z1"]), before)


def test_the_network_is_not_changed_by_looking_at_it(world):
    fit(world, latent_dim=3)
    r = world.S.project.get_result("autoencoder-1")
    arrays = {k: np.array(v) for k, v in r.payload["frozen"]["model"]["arrays"].items()}
    for url in (f"{B}/autoencoder-1", f"{B}/autoencoder-1/sample/S01", f"{B}/autoencoder-1/sensitivity", f"{B}/autoencoder-1/methods"):
        assert world.get(url).status_code == 200
    world.post(f"{B}/autoencoder-1/traverse", json={})
    world.post(f"{B}/autoencoder-1/interpolate", json={"a": "S01", "b": "S02"})
    for k, v in r.payload["frozen"]["model"]["arrays"].items():
        assert np.array_equal(arrays[k], v)


# ----------------------------------------------------------------------------- the framework
def test_the_framework_loads_the_addon_and_its_hooks_work(tmp_path, monkeypatch):
    """Through FluoViz's own loader (``FLUOVIZ_ADDONS_DEV``): the add-on is active, its routes answer, its kind and result kind are known."""
    S, app, active = A.make_app(tmp_path / "wsl", loader=True)
    if not active:
        S.shutdown()
        pytest.skip("the add-on could not be started by the framework in this checkout (see GET /api/addons)")
    with TestClient(app) as c:
        listing = c.get("/api/addons").json()
        mine = next(a for a in listing["installed"] if a["id"] == "autoencoders")
        assert mine["state"] == "active" and mine["result_kinds"][0]["kind"] == "autoencoder" and mine["result_kinds"][0]["saveable"]
        A.load_project(S, n=24)
        d = c.get(f"{B}/defaults")
        assert d.status_code == 200 and d.json()["torch"]["available"] in (True, False)
        from fluoviz.analysis.frozen import KINDS
        from fluoviz.server import jobs, saved_kinds
        assert "autoencoder" in KINDS and saved_kinds.FREEZERS.get("autoencoder") is not None and {"autoencoder", "autoencoder-scan"} <= jobs.HEAVY_KINDS
        assert c.get("/api/addon/other/defaults").status_code in (404, 405)
    S.shutdown()
