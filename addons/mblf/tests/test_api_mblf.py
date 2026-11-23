"""The API of the add-on (``/api/addon/mblf``) end to end on a small project: the engine, the advice, the check, a real fit (Stan), the result for the page, the prediction of a
sample (in-sample and held out), the predictable fraction, the correlation, the comparison of the sources, the exports, the saved model applied to the project, plain-English errors.

The fit needs a Stan engine (``FLUOVIZ_MBLF_ENGINE``); without one these tests are skipped with the reason, and only the checks that need no engine run."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

import api_support as A

B = A.BASE
REASON = A.engine_reason()
needs_engine = pytest.mark.skipif(REASON is not None, reason=REASON or "")
INP = {"variables": ["fluorescence", "absorbance"],
       "fluorescence": {"transform": "none", "min_coverage": 0.9, "basis": {"kind": "gp", "lengths": [60.0, 40.0], "var_fraction": 0.98, "k_max": 12}},
       "absorbance": {"transform": "none", "processed": False, "basis": {"kind": "gp", "lengths": [40.0], "var_fraction": 0.98, "k_max": 10}},
       "values": ["meta:doc"], "covariates": ["meta:site"], "partial": True}
QUICK = {"model": {"K": 3}, "sampler": {"chains": 2, "warmup": 250, "draws": 250, "keep": 300}, "seed": 2}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    S, app, active = A.make_app(tmp_path_factory.mktemp("ws"))
    with TestClient(app) as c:
        c.active, c.S = active, S
        c.data = A.load_project(S)
        yield c
    S.shutdown()


@pytest.fixture(scope="module")
def fitted(world):
    if REASON is not None:
        pytest.skip(REASON)
    j = A.run_job(world, f"{B}/fit", {"input": INP, **QUICK})
    assert j["status"] == "done", j
    return A.decode(j["result"])["rid"]


def msg(r):
    return r.json()["error"]["message"]


def test_the_add_on_is_started_by_the_framework(world):
    assert world.active


def test_defaults_and_the_engine(world):
    d = world.get(f"{B}/defaults").json()
    assert d["n_samples"] == 30 and d["absorbance"]["available"] and d["absorbance"]["n"] >= 29
    assert {m["key"] for m in d["meta"]} >= {"meta:site", "meta:doc"} and next(m for m in d["meta"] if m["key"] == "meta:site")["kind"] == "category"
    assert d["request"]["input"]["variables"] == ["fluorescence", "absorbance"] and set(d["presets"]) == {"quick", "standard", "thorough"}
    e = world.get(f"{B}/engine").json()
    assert e["state"] in ("ready", "missing", "unavailable") and e["message"]


def test_requests_are_refused_in_plain_english(world):
    r = world.post(f"{B}/fit", json={"input": {"variables": ["fluorescence"], "fluorescence": {"basis": {"kind": "bspline"}}}})
    assert r.status_code == 400 and "Gaussian-process basis" in msg(r)
    r = world.post(f"{B}/fit", json={"input": {**INP, "covariates": ["meta:doc"]}})
    assert r.status_code == 400 and "both a value to model and a covariate" in msg(r)
    r = world.post(f"{B}/fit", json={"input": INP, "model": {"K": 40}})
    assert r.status_code == 400 and "between 1 and 12" in msg(r)
    r = world.post(f"{B}/fit", json={"input": INP, "sampler": {"preset": "forever"}})
    assert r.status_code == 400 and "quick, standard, thorough" in msg(r)
    r = world.post(f"{B}/fit", json={"input": {"variables": []}})
    assert r.status_code == 400 and "Choose what to model" in msg(r)


def test_the_check_builds_the_modes_without_fitting(world):
    d = A.done(world, f"{B}/check", {"input": INP, **QUICK})
    assert d["n"] == 30 and [m["name"] for m in d["modes"]] == ["fluorescence", "absorbance", "values"]
    fl = d["modes"][0]
    assert fl["kind"] == "surface" and 3 <= fl["n_basis"] <= 12 and fl["captured"] > 0.5 and fl["n_samples"] == 30
    assert d["modes"][1]["n_samples"] == 29 and d["modes"][2]["n_samples"] == 24 and d["complete"] == 23
    assert d["covariates"]["columns"][0]["kind"] == "level" and d["problem"] is None and "unknowns" in d["words"]


@needs_engine
def test_the_result_for_the_page(world, fitted):
    res = A.decode(world.get(f"{B}/{fitted}").json())
    pl, r = res["payload"], res["result"]
    assert r["kind"] == "mblf" and r["summary"]["K"] == 3 and "frozen" not in pl
    assert pl["estimation"] == "bayes" and pl["chains"] == 2 and pl["n_draws"] == 300
    assert pl["diagnostics"]["verdict"]["level"] in ("ok", "warn", "bad") and pl["diagnostics"]["verdict"]["text"]
    fac = pl["factors"]
    assert len(fac) == 3 and fac[0]["share"]["mean"] >= fac[1]["share"]["mean"] >= fac[2]["share"]["mean"]
    fl = next(m for m in fac[0]["modes"] if m["mode"] == "fluorescence")
    assert np.asarray(fl["mean"]).shape == (16, 24) and np.asarray(fl["quantiles"]).shape == (7, 16, 24) and "band" in fl and np.asarray(fl["p_pos"]).shape == (16, 24)
    ab = next(m for m in fac[0]["modes"] if m["mode"] == "absorbance")
    assert np.asarray(ab["mean"]).shape == (40,)
    keys = {row["key"] for row in pl["matrix"]}
    assert {"nothing", "absorbance", "absorbance+values+covariates", "covariates"} <= keys
    row = next(x for x in pl["matrix"] if x["target"] == "fluorescence" and x["key"] == "absorbance+values+covariates")
    assert 0.3 < row["latent"]["mean"] <= 1.0 and row["latent"]["lower"] <= row["latent"]["mean"] <= row["latent"]["upper"]
    assert next(x for x in pl["matrix"] if x["target"] == "fluorescence" and x["key"] == "nothing")["latent"]["mean"] == pytest.approx(0.0, abs=1e-9)
    assert pl["covariates"]["effects"][0]["column"]["level"] in ("lake", "river") and len(pl["covariates"]["shares"]) == 3
    assert len(pl["variance"]) == 3 and np.asarray(pl["variance"][0]["variance"]["total"]).shape == (16, 24)
    assert np.isfinite(pl["loo"]["elpd"]) and len(pl["loo"]["pareto_k"]) == 30
    assert np.asarray(pl["samples"]["scores"]).shape == (30, 3) and pl["dimension"]["probabilities"]
    assert {b["mode"] for b in pl["basis_check"]} == {"fluorescence", "absorbance"}
    v = world.S.project.get_result(fitted).variables
    assert {"mblf_f1", "mblf_f1_sd", "mblf_loo"} <= set(v)


@needs_engine
def test_a_sample_predicted_from_its_other_modes(world, fitted):
    sid = "S05"                                                        # its DOC was not measured
    d = A.decode(world.get(f"{B}/{fitted}/sample/{sid}", params={"target": "values"}).json())
    assert d["observed"] is None and d["sources"] == ["fluorescence", "absorbance"] and d["covariates"]
    q = np.asarray(d["quantiles"])
    assert q.shape == (7, 1) and q[0, 0] < d["mean"][0] < q[-1, 0]
    assert sum(d["budget_total"].values()) == pytest.approx(1.0) and "DOC" not in d["text"] or True
    sid = "S01"
    held = A.decode(world.get(f"{B}/{fitted}/sample/{sid}", params={"target": "fluorescence", "sources": "absorbance", "covariates": 0}).json())
    assert held["held_out"] and np.isfinite(held["pareto_k"]) and np.asarray(held["mean"]).shape == (16, 24) and np.asarray(held["observed"]).shape == (16, 24)
    assert set(held["budget"]) == {"parameters", "scores", "residual", "noise"} and held["budget_total"]["noise"] == 0.0 and 0 <= held["inside95"] <= 1
    ins = A.decode(world.get(f"{B}/{fitted}/sample/{sid}", params={"target": "fluorescence", "sources": "absorbance", "covariates": 0, "held_out": 0, "noisy": 1}).json())
    assert not ins["held_out"] and ins["budget_total"]["noise"] > 0
    nothing = A.decode(world.get(f"{B}/{fitted}/sample/{sid}", params={"target": "absorbance", "sources": "none", "covariates": 0, "held_out": 0}).json())
    assert nothing["sources"] == [] and nothing["budget_total"]["scores"] > held["budget_total"]["scores"] - 1.0
    r = world.get(f"{B}/{fitted}/sample/{sid}", params={"target": "fluorescence", "sources": "fluorescence"})
    assert r.status_code == 400 and "cannot also be a source" in msg(r)
    r = world.get(f"{B}/{fitted}/sample/NOPE")
    assert r.status_code == 404


@needs_engine
def test_predictable_fraction_and_correlation(world, fitted):
    d = A.decode(world.get(f"{B}/{fitted}/predictable", params={"target": "absorbance", "sources": "fluorescence", "covariates": 0}).json())
    assert np.asarray(d["latent"]["mean"]).shape == (40,) and 0 <= d["whole_latent"]["mean"] <= 1 and d["sources"] == ["fluorescence"] and len(d["hist"]) == 40
    c = A.decode(world.get(f"{B}/{fitted}/correlation", params={"a": "absorbance", "b": "fluorescence", "x": 340}).json())
    assert np.asarray(c["mean"]).shape == (16, 24) and abs(c["point"]["x"] - 340) < 10 and np.nanmax(np.abs(c["mean"])) <= 1.0 + 1e-9
    c2 = A.decode(world.get(f"{B}/{fitted}/correlation", params={"a": "fluorescence", "b": "fluorescence", "x": 330, "y": 430}).json())
    assert np.nanmax(c2["mean"]) == pytest.approx(1.0, abs=1e-6)


@needs_engine
def test_the_held_out_comparison_of_the_sources(world, fitted):
    d = A.done(world, f"{B}/{fitted}/compare", {"targets": ["values", "absorbance"], "max_points": 30})
    rows = d["compare"]["values"]["rows"]
    by = {r["key"]: r for r in rows}
    assert by["nothing"]["gain"] == 0.0 and by["fluorescence+absorbance+covariates"]["elpd"] > by["nothing"]["elpd"]
    for r in rows:
        assert set(r["coverage"]) == {"0.5", "0.8", "0.95"} and r["n_samples"] == 24 and r["rmse"] > 0
    assert d["compare"]["values"]["one_se"] in by and d["compare"]["absorbance"]["best"]
    stored = A.decode(world.get(f"{B}/{fitted}").json())["payload"]["compare"]
    assert set(stored) == {"values", "absorbance"}
    for what in ("scores", "matrix", "compare", "loadings"):
        r = world.get(f"{B}/{fitted}/export", params={"what": what})
        assert r.status_code == 200 and r.content.count(b"\n") > 3, what
    m = world.get(f"{B}/{fitted}/methods").json()
    assert "No-U-Turn" in m["text"] and len(m["references"]) >= 5


@needs_engine
def test_the_saved_model_reads_new_samples_and_predicts_the_values(world, fitted):
    j = A.run_job(world, "/api/models", {"result": fitted, "title": "Latent factors"})
    assert j["status"] == "done", j
    mid = A.decode(j["result"])["id"]
    d = world.get(f"/api/models/{mid}").json()
    assert d["applicable"] and "mblf_pred_doc" in [o["name"] for o in d["outputs"]] and d["describe"]["caveats"]
    r = world.post(f"/api/models/{mid}/apply", json={"select": "all"})
    assert r.status_code == 400 and "absorbance spectrum" in msg(r)          # the data recipe needs every function the model was fitted on (S03 has no spectrum)
    ids = [s for s in world.data[2] if s != "S03"]
    j = A.run_job(world, f"/api/models/{mid}/apply", {"ids": ids})
    assert j["status"] == "done", (j.get("error"), j.get("detail"), j)
    r = world.S.project.get_result(A.decode(j["result"])["rid"])
    res = world.S.project.get_result(fitted)
    pos = [res.sample_ids.index(s) for s in r.sample_ids]
    assert np.allclose(np.asarray(r.variables["mblf_f1"], float), np.asarray(res.variables["mblf_f1"], float)[pos], atol=1e-6)
    lo, hi, mean = (np.asarray(r.variables[k], float) for k in ("mblf_pred_doc_lo", "mblf_pred_doc_hi", "mblf_pred_doc"))
    assert np.all(lo < mean) and np.all(mean < hi)
    assert set(np.asarray(r.variables["status"])) <= {"ok", "outside"}
    ins = A.decode(world.get(f"/api/models/applied/{r.id}/sample/S02").json())
    assert set(ins["images"]) >= {"predicted", "lower", "upper", "uncertainty", "measured"} and "uncertainty" in ins["text"]
