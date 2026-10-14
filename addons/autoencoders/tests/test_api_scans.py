"""Scans (latent dimensions, smoothify, stability, baselines) and the codes against other variables, through the API."""
import importlib.util

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api_support as A

B = A.BASE
HAS_TORCH = importlib.util.find_spec("torch") is not None
INP = {"variables": ["fluorescence"], "fluorescence": {"transform": "none", "min_coverage": 0.9, "basis": {"kind": "bspline", "n_basis": [8, 6]}, "smoothing": {"method": "gcv"}}}
FAST = {"latent_dim": 3, "epochs": 30, "patience": 15, "n_restarts": 1}


@pytest.fixture()
def world(tmp_path):
    S, app, _ = A.make_app(tmp_path / "ws")
    with TestClient(app) as c:
        c.S = S
        c.data = A.load_project(S, n=36, absorbance=False)
        yield c
    S.shutdown()


@pytest.fixture()
def fake_world(world, monkeypatch):
    A.use_fake_training(monkeypatch)
    return world


def msg(r):
    return r.json()["error"]["message"]


def test_the_linear_baselines_need_no_network(fake_world):
    out = A.done(fake_world, f"{B}/scan", {"scan": "baseline", "request": {**FAST, "input": INP}, "dims": [1, 2, 3, 4]})
    assert out["scan"] == "baseline" and out["name"] == "Functional PCA" and out["dims"] == [1, 2, 3, 4] and len(out["val"]) == 4 and len(out["train"]) == 4
    assert out["val"][0] > out["val"][2] and out["val"][2] < 0.05 and "component" in out["words"] and out["settings"]["n"] == 36
    kept = A.decode(fake_world.post(f"{B}/scan", json={"scan": "baseline", "request": {**FAST, "input": INP}}).json())
    assert "job_id" in kept


def test_a_scan_of_a_result_is_kept_in_it(fake_world):
    j = A.run_job(fake_world, f"{B}/fit", {**FAST, "input": INP})
    rid = A.decode(j["result"])["rid"]
    out = A.done(fake_world, f"{B}/scan", {"scan": "baseline", "rid": rid, "dims": [1, 2, 3]})
    assert out["saved_in"] == rid
    assert out["network"]["k"] == 3 and 0 <= out["network"]["val"] < 0.1
    pl = A.decode(fake_world.get(f"{B}/{rid}").json())["payload"]
    assert pl["scans"]["baseline"]["dims"] == [1, 2, 3]
    nosave = A.done(fake_world, f"{B}/scan", {"scan": "baseline", "rid": rid, "dims": [1, 2], "save": False})
    assert nosave["saved_in"] is None and A.decode(fake_world.get(f"{B}/{rid}").json())["payload"]["scans"]["baseline"]["dims"] == [1, 2, 3]
    # the saved scan survives saving and opening the project
    from fluoviz.core import persistence
    folder = persistence.save_project(fake_world.S.project, fake_world.S.workspace / "p", everything=True) if hasattr(fake_world.S, "workspace") else None
    if folder is not None:
        q = persistence.load_project(folder)
        assert q.get_result(rid).payload["scans"]["baseline"]["dims"] == [1, 2, 3]


@pytest.mark.parametrize("body,word", [
    ({}, "'scan' must be one of"), ({"scan": "magic"}, "'scan' must be one of"), ({"scan": "latent"}, "give 'rid'"), ({"scan": "latent", "rid": "autoencoder-7"}, "autoencoder-7"),
    ({"scan": "latent", "request": {"latent_dim": 3}, "dims": [0]}, "from 1 to 32"), ({"scan": "latent", "request": {"latent_dim": 3}, "dims": []}, "1 to 24 whole numbers"),
    ({"scan": "latent", "request": {"latent_dim": 3}, "families": ["gan"]}, "ae and / or vae"), ({"scan": "latent", "request": {"latent_dim": 3}, "n_restarts": 9}, "from 1 to 5"),
    ({"scan": "smoothify", "request": {"latent_dim": 3}, "values": [-1]}, "each 0 or more"), ({"scan": "stability", "request": {"latent_dim": 3}, "n_runs": 1}, "from 2 to 10"),
    ({"scan": "stability", "request": {"latent_dim": 3}, "bootstrap": "yes"}, "true or false"), ({"scan": "baseline", "request": {"latent_dim": 0}}, "between 1 and 32"),
    ({"scan": "baseline", "request": {"latent_dim": 3}, "save": 1}, "true or false"),
])
def test_scan_requests_are_checked(fake_world, body, word):
    r = fake_world.post(f"{B}/scan", json=body)
    assert r.status_code in (400, 404) and word in msg(r), (body, r.text)


def test_the_codes_against_other_variables(fake_world):
    X, A_, ids = fake_world.data
    j = A.run_job(fake_world, f"{B}/fit", {**FAST, "input": INP})
    rid = A.decode(j["result"])["rid"]
    p = fake_world.S.project
    p.merge_metadata(pd.DataFrame({"conc": A_[:, 0], "noise": np.random.default_rng(0).normal(size=len(ids)), "site": ["a", "b"] * 18}, index=pd.Index(ids, name="sample")), record=False)
    r = fake_world.post(f"{B}/{rid}/relate", json={"variables": ["meta:conc", "meta:noise", "meta:site"]})
    assert r.status_code == 200, r.text
    o = A.decode(r.json())
    rows = {v["key"]: v for v in o["variables"]}
    assert o["names"] == ["z1", "z2", "z3"] and set(rows) == {"meta:conc", "meta:noise"} and o["skipped"] == ["meta:site"]
    assert rows["meta:conc"]["r2_codes"] > 0.8 and rows["meta:conc"]["r2_pca"] > 0.8 and rows["meta:noise"]["r2_codes"] < 0.3 and rows["meta:conc"]["n"] == 36
    assert max(abs(np.asarray(rows["meta:conc"]["correlation"]))) > 0.5 and "cross-validated R²" in o["words"]
    for body, word in (({}, "Choose the variables"), ({"variables": ["meta:nope"]}, "nope"), ({"variables": ["meta:site"]}, "None of these variables is a number"), ({"variables": ["meta:conc"], "folds": 1}, "from 2 to 20")):
        bad = fake_world.post(f"{B}/{rid}/relate", json=body)
        assert bad.status_code == 400 and word in msg(bad), (body, bad.text)
    assert fake_world.put("/api/state/corrections", json={"smoothing": {"enabled": True, "em_nm": 8}}).status_code == 200
    o2 = A.decode(fake_world.post(f"{B}/{rid}/relate", json={"variables": ["meta:conc"]}).json())
    assert o2["variables"][0]["r2_pca"] is None and "changed since" in o2["note"]


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is not installed")
def test_the_latent_scan_recommends_a_number_of_codes(world):
    out = A.done(world, f"{B}/scan", {"scan": "latent", "request": {**FAST, "epochs": 250, "patience": 100, "input": INP}, "dims": [1, 2, 3, 4]}, timeout=600)
    assert out["scan"] == "latent" and out["dims"] == [1, 2, 3, 4] and set(out["families"]) == {"ae"} and len(out["families"]["ae"]["val"]) == 4
    val = np.asarray(out["families"]["ae"]["val"])
    assert val[0] > val[3] and np.asarray(out["baseline"]["val"])[0] > np.asarray(out["baseline"]["val"])[2] and out["baseline"]["name"] == "Functional PCA" and len(out["baseline"]["val"]) == 4
    assert out["recommended"]["latent_dim"] in (2, 3, 4) and "Recommended" in out["words"]
    both = A.done(world, f"{B}/scan", {"scan": "latent", "request": {**FAST, "input": INP}, "dims": [2, 3], "families": ["ae", "vae"]}, timeout=600)
    assert set(both["families"]) == {"ae", "vae"}


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is not installed")
def test_the_smoothify_and_stability_scans(world):
    inp = {"variables": ["fluorescence"], "fluorescence": {"basis": {"kind": "gp", "lengths": [60.0, 40.0], "k_max": 30}, "smoothing": {"method": "ml"}}}
    sm = A.done(world, f"{B}/scan", {"scan": "smoothify", "request": {**FAST, "input": inp}, "values": [0, 1, 5]}, timeout=600)
    assert sm["values"] == [0.0, 1.0, 5.0] and len(sm["error"]) == 3 and len(sm["roughness"]) == 3 and sm["suggested"] in (0.0, 1.0, 5.0) and sm["words"]
    st = A.done(world, f"{B}/scan", {"scan": "stability", "request": {**FAST, "input": INP}, "n_runs": 2}, timeout=600)
    assert st["n_runs"] == 2 and 0 <= st["alignment"]["mean"] <= 1 + 1e-9 and len(st["alignment"]["per_code"]) == 3 and "reproducible" in st["verdict"]
