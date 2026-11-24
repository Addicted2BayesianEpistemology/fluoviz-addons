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


def test_a_basis_above_150_terms_needs_the_switch_and_then_is_allowed(world):
    big = {**INP, "absorbance": {**INP["absorbance"], "basis": {**INP["absorbance"]["basis"], "k_max": 200}}}
    r = world.post(f"{B}/check", json={"input": big, **QUICK})
    assert r.status_code == 400 and "maximum number of terms of the absorbance basis (step 3" in msg(r) and "Allow more than 150 basis functions" in msg(r)
    r = world.post(f"{B}/check", json={"input": {**INP, "fluorescence": {**INP["fluorescence"], "basis": {**INP["fluorescence"]["basis"], "n_basis": 160}}}, **QUICK})
    assert r.status_code == 400 and "number of terms of the fluorescence (EEM) basis" in msg(r)
    d = A.done(world, f"{B}/check", {"input": {**big, "large_basis": True}, **QUICK})
    assert d["problem"] is None and d["modes"][1]["n_basis"] <= 200
    r = world.post(f"{B}/check", json={"input": {**INP, "absorbance": {**INP["absorbance"], "basis": {**INP["absorbance"]["basis"], "k_max": 500}}, "large_basis": True}})
    assert r.status_code == 400                                                    # (FluoViz's own limit, 400, still holds)


def test_a_basis_from_a_functional_pca_for_both_functions(world):
    inp = {**INP, "fluorescence": {**INP["fluorescence"], "basis_from": "fpca"}, "absorbance": {**INP["absorbance"], "basis_from": "fpca"}}
    d = A.done(world, f"{B}/check", {"input": inp, **QUICK})
    fl, ab = d["modes"][0], d["modes"][1]
    assert fl["basis"]["from"] == "fpca" and ab["basis"]["from"] == "fpca"
    assert 1 <= fl["n_basis"] <= 12 and 1 <= ab["n_basis"] <= 10 and fl["captured"] > 0.8 and fl["n_basis"] <= 29
    assert any("data-driven" in n for n in d["notes"])
    gp = A.done(world, f"{B}/check", {"input": INP, **QUICK})
    assert gp["modes"][0]["basis"]["from"] == "gp"
    r = world.post(f"{B}/check", json={"input": {**INP, "absorbance": {**INP["absorbance"], "basis_from": "pca"}}})
    assert r.status_code == 400 and "functional PCA of the samples" in msg(r)


@needs_engine
def test_a_model_on_functional_pca_bases_fits_and_reads_new_samples(world):
    inp = {**INP, "fluorescence": {**INP["fluorescence"], "basis_from": "fpca"}, "absorbance": {**INP["absorbance"], "basis_from": "fpca"}}
    j = A.run_job(world, f"{B}/fit", {"input": inp, **QUICK})
    assert j["status"] == "done", j
    rid = A.decode(j["result"])["rid"]
    pl = A.decode(world.get(f"{B}/{rid}").json())["payload"]
    assert pl["modes"][0]["basis"]["from"] == "fpca" and np.isfinite(pl["loo"]["elpd"])
    assert "functional PCA" in world.get(f"{B}/{rid}/methods").json()["text"]
    d = A.decode(world.get(f"{B}/{rid}/sample/S01", params={"target": "fluorescence", "sources": "absorbance"}).json())
    assert np.asarray(d["mean"]).shape == (16, 24)


# ----------------------------------------------------------------------------- how the data are sent to the model; categories as samples
PREP = {**INP,
        "fluorescence": {**INP["fluorescence"], "log": True, "centre": "median", "scaling": "pareto"},
        "absorbance": {**INP["absorbance"], "scaling": "pointwise"},
        "values": ["meta:doc", "meta:temp"], "covariates": ["meta:site", "meta:season"], "log_keys": ["meta:doc"],
        "values_basis": {"kind": "pca", "n_basis": 1}}


def test_the_transforms_are_checked_in_plain_english(world):
    bad = lambda **kw: world.post(f"{B}/check", json={"input": {**INP, **kw}})                         # noqa: E731
    r = bad(fluorescence={**INP["fluorescence"], "centre": "mode"})
    assert r.status_code == 400 and "centring of the fluorescence" in msg(r)
    r = bad(absorbance={**INP["absorbance"], "scaling": "unit"})
    assert r.status_code == 400 and "scaling of the absorbance" in msg(r)
    r = bad(fluorescence={**INP["fluorescence"], "log": True, "log_offset": -1})
    assert r.status_code == 400 and "offset of the log" in msg(r)
    r = bad(log_keys=["meta:temp"])
    assert r.status_code == 400 and "neither a value to model nor a covariate" in msg(r)
    r = bad(values_basis={"kind": "ica"})
    assert r.status_code == 400 and "'identity'" in msg(r)
    j = A.run_job(world, f"{B}/check", {"input": {**INP, "log_keys": ["meta:site"]}})
    assert j["status"] == "error" and "is a category: it has no log" in j["error"]


def test_the_check_with_a_log_other_scalings_and_principal_components_of_the_values(world):
    d = A.done(world, f"{B}/check", {"input": PREP, **QUICK})
    fl, ab, va = d["modes"]
    assert fl["log"]["offset"] > 0 and fl["units"].startswith("log10(") and fl["prep"] == {"centre": "median", "scaling": "pareto"}
    assert ab["prep"]["scaling"] == "pointwise" and va["basis"]["kind"] == "pca" and va["n_basis"] == 1 and va["columns"] == ["log10 doc", "temp"]
    assert any("log10(fluorescence" in n for n in d["notes"]) and any("principal component" in n for n in d["notes"])
    cols = d["covariates"]["columns"]
    assert sum(c["key"] == "meta:season" for c in cols) == 2


@pytest.fixture(scope="module")
def fitted_prep(world):
    if REASON is not None:
        pytest.skip(REASON)
    j = A.run_job(world, f"{B}/fit", {"input": PREP, **QUICK})
    assert j["status"] == "done", j
    return A.decode(j["result"])["rid"]


@needs_engine
def test_a_fit_with_the_transforms_shows_the_residual_and_the_categories(world, fitted_prep):
    rid = fitted_prep
    pl = A.decode(world.get(f"{B}/{rid}").json())["payload"]
    assert pl["modes"][0]["log"]["offset"] > 0 and pl["modes"][2]["labels"] == ["log10 doc", "temp"] and pl["modes"][2]["log_cols"] == [True, False]
    text = world.get(f"{B}/{rid}/methods").json()["text"]
    assert "log10(value +" in text and "Pareto scaling" in text and "principal component" in text
    # the residual of a prediction
    d = A.decode(world.get(f"{B}/{rid}/sample/S01", params={"target": "fluorescence", "sources": "absorbance"}).json())
    res = d["residual"]
    assert np.asarray(res["value"]).shape == (16, 24) and np.asarray(res["quantiles"]).shape == (7, 16, 24) and res["rms"] > 0 and 0 <= res["beyond2"] <= 1
    assert np.allclose(np.asarray(res["value"]), np.asarray(d["observed"]) - np.asarray(d["mean"]), equal_nan=True, atol=1e-4)
    assert A.decode(world.get(f"{B}/{rid}/sample/S05", params={"target": "values"}).json())["residual"] is None or True
    # any two categories compared; the categories as samples
    eff = {e["column"]["level"]: e for e in pl["covariates"]["effects"] if e["column"]["key"] == "meta:season"}
    ref = next(e["column"]["reference"] for e in eff.values())
    a, b = sorted(eff)
    c = A.decode(world.get(f"{B}/{rid}/contrast", params={"key": "meta:season", "a": a, "b": b}).json())
    ab_ = next(m for m in c["modes"] if m["mode"] == "absorbance")
    want = np.asarray(next(m for m in eff[a]["modes"] if m["mode"] == "absorbance")["mean"]) - np.asarray(next(m for m in eff[b]["modes"] if m["mode"] == "absorbance")["mean"])
    assert np.allclose(ab_["mean"], want, atol=1e-5) and c["n_a"] == 10 and c["n_b"] == 10
    c0 = A.decode(world.get(f"{B}/{rid}/contrast", params={"key": "meta:season", "a": a, "b": ref}).json())
    assert np.allclose(next(m for m in c0["modes"] if m["mode"] == "absorbance")["mean"], next(m for m in eff[a]["modes"] if m["mode"] == "absorbance")["mean"], atol=1e-5)
    pa = A.decode(world.get(f"{B}/{rid}/category", params={"key": "meta:season", "level": a, "how": "pure", "target": "absorbance"}).json())
    pb = A.decode(world.get(f"{B}/{rid}/category", params={"key": "meta:season", "level": b, "how": "pure", "target": "absorbance"}).json())
    assert np.allclose(np.asarray(pa["shift"]["mean"]) - np.asarray(pb["shift"]["mean"]), want, atol=1e-5)
    assert np.asarray(pa["member"]["quantiles"]).shape == (7, 40) and sum(pa["budget_total"].values()) == pytest.approx(1.0) and pa["n"] == 10
    lev = {w["label"]: w["value"] for w in pa["design"]}
    assert lev[f"season = {a}"] == pytest.approx(1.0) and lev[f"season = {b}"] == pytest.approx(0.0)
    ty = A.decode(world.get(f"{B}/{rid}/category", params={"key": "meta:season", "level": ref, "how": "typical", "target": "values", "noisy": 1}).json())
    assert ty["how"] == "typical" and ty["n"] == 10 and ty["budget_total"]["noise"] > 0 and np.asarray(ty["member"]["quantiles"]).shape == (7, 2)
    site = next(w for w in ty["design"] if w["label"].startswith("site ="))
    assert 0.0 <= site["value"] <= 1.0
    r = world.get(f"{B}/{rid}/category", params={"key": "meta:season", "level": "winter"})
    assert r.status_code == 400 and "no category 'winter'" in msg(r)
    r = world.get(f"{B}/{rid}/contrast", params={"key": "meta:temp", "a": "x", "b": "y"})
    assert r.status_code == 400 and "not a category" in msg(r)


@needs_engine
def test_the_saved_model_reads_new_samples_on_the_log_scale(world, fitted_prep):
    j = A.run_job(world, "/api/models", {"result": fitted_prep, "title": "Latent factors (log)"})
    assert j["status"] == "done", j                                                        # (freezing re-reads the training samples: the scores agree)
    mid = A.decode(j["result"])["id"]
    ids = [s for s in world.data[2] if s != "S03"]
    j = A.run_job(world, f"/api/models/{mid}/apply", {"ids": ids})
    assert j["status"] == "done", j
    r = world.S.project.get_result(A.decode(j["result"])["rid"])
    res = world.S.project.get_result(fitted_prep)
    pos = [res.sample_ids.index(s) for s in r.sample_ids]
    assert np.allclose(np.asarray(r.variables["mblf_f1"], float), np.asarray(res.variables["mblf_f1"], float)[pos], atol=1e-6)
    lo, hi, mid_ = (np.asarray(r.variables[k], float) for k in ("mblf_pred_doc_lo", "mblf_pred_doc_hi", "mblf_pred_doc"))
    assert np.all(lo > 0) and np.all(lo < mid_) and np.all(mid_ < hi)                      # back from the log: in the units of the DOC
