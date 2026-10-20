"""The API of the add-on (``/api/addon/som``): advice, checks, the fit job, the result for the page, one sample, one unit, the prototypes, the planes, the groups, unusual samples,
trajectories, the projection, exports, methods, staleness, plain-English errors."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

import api_support as A

B = A.BASE
INP = {"variables": ["fluorescence"], "fluorescence": {"transform": "none", "min_coverage": 0.9, "basis": {"kind": "bspline", "n_basis": [8, 6]}, "smoothing": {"method": "gcv"}}}
INP2 = {**INP, "variables": ["fluorescence", "absorbance"], "absorbance": {"transform": "none", "processed": False, "basis": {"kind": "bspline", "n_basis": 12}, "smoothing": {"method": "gcv"}}}
FAST = {"train": {"epochs_rough": 8, "epochs_fine": 20}, "validation": {"folds": 4}, "map": {"size": [4, 5]}}


@pytest.fixture()
def world(tmp_path):
    S, app, active = A.make_app(tmp_path / "ws")
    with TestClient(app) as c:
        c.active = active
        c.S = S
        c.data = A.load_project(S, n=36, absorbance=True)
        import pandas as pd
        p = S.project
        site = np.where(np.arange(36) % 3 == 0, "up", np.where(np.arange(36) % 3 == 1, "mid", "down"))
        p.merge_metadata(pd.DataFrame({"site": site, "doc": np.linspace(1.0, 9.0, 36), "day": np.arange(36) % 12}, index=data_ids(c)))
        yield c
    S.shutdown()


def data_ids(c):
    return c.data[2]


def msg(r):
    return r.json()["error"]["message"]


def fit(c, inp=INP, **body):
    body = {"input": inp, **FAST, **body}
    j = A.run_job(c, f"{B}/fit", body)
    assert j["status"] == "done", j
    return A.decode(j["result"])["rid"]


# ----------------------------------------------------------------------------- advice and checks
def test_the_add_on_is_started_by_the_framework(world):
    assert world.active


def test_defaults_facts_and_a_request_that_can_be_run(world):
    r = world.get(f"{B}/defaults", params={"variables": "fluorescence,absorbance"})
    assert r.status_code == 200, r.text
    d = A.decode(r.json())
    assert d["n_samples"] == 36 and d["fluorescence"]["available"] and d["absorbance"]["available"] and d["n_common"] == 36
    pre = d["advice"]["presets"]
    assert pre["compact"]["units"] < pre["standard"]["units"] < pre["fine"]["units"] and "interpolate" in pre["standard"]["words"] or "interpolat" in pre["standard"]["words"]
    assert d["request"]["input"]["variables"] == ["fluorescence", "absorbance"] and d["request"]["metric"]["kind"] == "l2"
    assert {m["kind"] for m in d["metrics"]} == {"l2", "mahalanobis", "prior", "standardize"}
    adv = A.decode(world.get(f"{B}/defaults", params={"advise": 1}).json())["advice"]
    assert adv["side_ratio"] >= 1.0 and adv["presets"]["standard"]["rows"] >= 2
    only = A.decode(world.get(f"{B}/defaults", params={"variables": "absorbance"}).json())
    assert only["request"]["input"]["variables"] == ["absorbance"] and only["n_common"] == 36
    assert world.get(f"{B}/defaults", params={"variables": "nonsense"}).status_code == 400


def test_check_says_what_a_request_would_do(world):
    j = A.run_job(world, f"{B}/check", {"input": INP2, **FAST, "metric": {"kind": "mahalanobis"}})
    assert j["status"] == "done", j
    c = A.decode(j["result"])
    assert c["ok"] and c["n"] == 36 and c["input"]["mode"] == "functional" and [b["name"] for b in c["input"]["blocks"]] == ["fluorescence", "absorbance"]
    assert c["map"]["rows"] == 4 and c["map"]["cols"] == 5 and c["metric"]["components"] >= 2 and "second" in c["estimate"]["words"]
    assert c["validation"]["folds"] == 4


def test_bad_settings_are_refused_in_plain_english(world):
    for body, text in [({"metric": {"kind": "nope"}}, "distance must be one of"), ({"map": {"size": [1, 9]}}, "at least 2 rows"), ({"input": {"variables": ["fluorescence"], "bogus": 1}}, "Unknown data setting"),
                       ({"input": {"kind": "variables"}}, "Choose the variables"), ({"train": {"mode": "online"}}, "training mode must be one of"), ({"foo": 1}, "Unknown setting"),
                       ({"input": {"kind": "variables", "keys": ["meta:doc"]}, "metric": {"kind": "prior"}}, "Gaussian-process prior distance needs functions")]:
        r = world.post(f"{B}/fit", json={"input": INP, **body} if "input" not in body else body)
        assert r.status_code == 400 and text in msg(r), (body, r.text)
    assert world.post(f"{B}/fit", json={"input": INP, "replace": "som-9"}).status_code == 404
    assert world.get(f"{B}/som-9").status_code == 404
    assert world.get(f"{B}/defaults", params={"min_coverage": 5}).status_code == 400


# ----------------------------------------------------------------------------- the fit and the result
def test_fit_saves_a_result_and_publishes_variables(world):
    j = A.run_job(world, f"{B}/fit", {"input": INP2, **FAST})
    assert j["status"] == "done", j
    res = A.decode(j["result"])
    rid = res["rid"]
    assert rid == "som-1" and res["result"]["kind"] == "som" and res["result"]["summary"]["rows"] == 4 and res["result"]["summary"]["n"] == 36
    r = world.S.project.get_result(rid)
    names = set(r.variables)
    assert names >= {"som_unit", "som_row", "som_col", "som_x", "som_y", "som_qe", "som_qe_rel", "som_qe_rmse_fluorescence", "som_qe_rmse_absorbance", "som_umatrix", "som_cluster", "flag", "status", "heldout_qe"}
    assert r.sample_ids == data_ids(world) and np.isfinite(np.asarray(r.variables["som_x"], float)).all()
    cat = {c["key"] for c in world.get("/api/variables").json()["variables"]} if isinstance(world.get("/api/variables").json(), dict) else {c["key"] for c in world.get("/api/variables").json()}
    assert f"res:{rid}:som_x" in cat and f"res:{rid}:som_cluster" in cat
    got = A.decode(world.get(f"{B}/{rid}").json())
    pl = got["payload"]
    assert pl["map"]["units"] == 20 and len(pl["map"]["unit_names"]) == 20 and pl["samples"]["pos"].shape == (36, 2) and "frozen" not in pl
    assert pl["prototypes"]["fluorescence"].shape == (20, 16, 24) and pl["prototype_curves"]["absorbance"].shape == (20, 40) and pl["mask"].shape == (16, 24)
    assert set(pl["planes"]) >= {"umatrix", "hits", "dmean", "qe_unit", "score1", "score2"}
    assert pl["quality"]["captured"] > 0.5 and pl["verdicts"]["vs_pca"]["verdict"] and pl["clusters"]["k"] >= 2
    assert sum(pl["clusters"]["sizes"]) == 36


def test_pixels_and_every_variable_choice_through_the_api(world):
    for inp in ({"kind": "pixels", "variables": ["fluorescence"], "fluorescence": {"min_coverage": 0.9}}, {"kind": "pixels", "variables": ["absorbance"]},
                {"kind": "pixels", "variables": ["fluorescence", "absorbance"], "fluorescence": {"min_coverage": 0.9}, "absorbance": {"processed": False}},
                {"variables": ["absorbance"], "absorbance": {"processed": False, "basis": {"kind": "bspline", "n_basis": 12}, "smoothing": {"method": "gcv"}}}):
        rid = fit(world, inp)
        pl = A.decode(world.get(f"{B}/{rid}").json())["payload"]
        assert pl["samples"]["unit"].shape == (36,), inp


def test_replace_trains_in_place_and_staleness(world):
    rid = fit(world, title="My map")
    assert world.S.project.get_result(rid).title == "My map"
    j = A.run_job(world, f"{B}/fit", {"input": INP, **FAST, "replace": rid, "seed": 3})
    assert j["status"] == "done" and A.decode(j["result"])["rid"] == rid and world.S.project.get_result(rid).title == "My map"
    X, A_, ids = world.data
    world.S.project.set_excluded([ids[0]], True)
    listing = world.S.project.results_listing()
    assert listing[0]["stale"]
    r = world.get(f"{B}/{rid}/sample/{ids[3]}")
    assert r.status_code == 400 and "Compute again with the same settings" in msg(r)
    assert world.get(f"{B}/{rid}").status_code == 200                  # what is stored can still be looked at


def test_sample_unit_and_prototypes(world):
    rid = fit(world, INP2)
    ids = data_ids(world)
    s = A.decode(world.get(f"{B}/{rid}/sample/{ids[5]}").json())
    assert s["id"] == ids[5] and s["unit_name"].startswith("r") and s["status"] in ("ok", "outside", "uncertain") and [b["name"] for b in s["blocks"]] == ["fluorescence", "absorbance"]
    f = s["blocks"][0]
    assert f["measured"].shape == (16, 24) and f["prototype"].shape == (16, 24) and f["difference"].shape == (16, 24) and s["blocks"][1]["measured"].shape == (40,)
    thumb = A.decode(world.get(f"{B}/{rid}/sample/{ids[5]}", params={"thumb": 1}).json())
    assert "difference" not in thumb["blocks"][0]
    assert world.get(f"{B}/{rid}/sample/nope").status_code == 404
    u = A.decode(world.get(f"{B}/{rid}/unit/{s['unit']}").json())
    assert ids[5] in u["members"] and u["hits"] == len(u["members"]) and u["blocks"][0]["prototype"].shape == (16, 24) and u["blocks"][0]["members_mean"].shape == (16, 24)
    assert world.get(f"{B}/{rid}/unit/999").status_code == 404
    for what in ("prototype", "members", "difference"):
        pr = A.decode(world.get(f"{B}/{rid}/prototypes", params={"what": what}).json())
        assert pr["surfaces"]["fluorescence"].shape == (20, 16, 24) and pr["curves"]["absorbance"].shape == (20, 40), what
    assert world.get(f"{B}/{rid}/prototypes", params={"what": "x"}).status_code == 400


def test_planes_of_every_kind(world):
    rid = fit(world, INP2)
    def plane(**b):
        r = world.post(f"{B}/{rid}/plane", json=b)
        assert r.status_code == 200, (b, r.text)
        return A.decode(r.json())
    assert plane(kind="stored", key="umatrix")["values"].shape == (20,)
    assert plane(kind="total")["values"].shape == (20,)
    assert plane(kind="point", ex=330, em=450)["label"].startswith("Intensity at excitation")
    assert plane(kind="region_mean", ex=[300, 350], em=[400, 480])["values"].shape == (20,)
    assert plane(kind="region_max", ex=[300, 350], em=[400, 480])["values"].shape == (20,)
    assert plane(kind="absorbance", wl=300)["units"] == "absorbance"
    assert plane(kind="index", key="total_fluorescence")["values"].shape == (20,)
    cat = plane(kind="var", key="meta:site")
    assert cat["kind"] == "category" and cat["categories"] == ["down", "mid", "up"] and len(cat["unit_labels"]) == 20
    num = plane(kind="var", key="meta:doc", smooth=True)
    assert num["kind"] == "numeric" and np.isfinite(num["values"]).all()
    assert "var:site" in A.decode(world.get(f"{B}/{rid}").json())["payload"]["planes"]               # computed planes are kept
    r = world.post(f"{B}/{rid}/plane", json={"kind": "point", "ex": 900, "em": 450})
    assert r.status_code == 400 and "outside the measured range" in msg(r)
    assert world.post(f"{B}/{rid}/plane", json={"kind": "bogus"}).status_code == 400
    idx = world.get(f"{B}/{rid}/indices").json()["indices"]
    assert any(i["key"] == "HIX" for i in idx) and any(i["kind"] == "absorbance" for i in idx)
    assert world.get(f"{B}/{rid}/planes").json()["stored"]


def test_groups_cluster_again_and_the_clustering_page_contract(world):
    rid = fit(world)
    g = A.decode(world.get(f"{B}/{rid}/groups").json())
    assert g["k"] >= 2 and g["images"]["mean"].shape[1:] == (16, 24) and g["profiles"] and g["associations"]["variables"]
    out = A.decode(world.post(f"{B}/{rid}/cluster", json={"k": 3, "method": "kmeans"}).json())
    assert out["clusters"]["k"] == 3 and sum(out["clusters"]["sizes"]) == 36
    r = world.S.project.get_result(rid)
    assert len(set(np.asarray(r.variables["som_cluster"]).tolist())) == 3
    assert world.post(f"{B}/{rid}/cluster", json={"method": "bogus"}).status_code == 400


def test_anomalies_trajectories_projection_exports_methods(world):
    rid = fit(world)
    for body in ({"method": "mad", "c": 2.0}, {"method": "quantile", "q": 0.9}, {"method": "limit", "level": 95}):
        a = A.decode(world.post(f"{B}/{rid}/anomalies", json=body).json())
        assert a["histogram"]["counts"].sum() == 36 and isinstance(a["flagged"], list)
    assert world.post(f"{B}/{rid}/anomalies", json={"method": "x"}).status_code == 400
    t = A.decode(world.post(f"{B}/{rid}/trajectories", json={"order_by": "meta:day", "group_by": "meta:site"}).json())
    assert {ln["group"] for ln in t["lines"]} == {"up", "mid", "down"} and all(len(ln["ids"]) >= 2 for ln in t["lines"])
    assert world.post(f"{B}/{rid}/trajectories", json={"group_by": "meta:site"}).status_code == 400
    pj = A.decode(world.get(f"{B}/{rid}/projection").json())
    assert pj["xy"].shape == (20, 2) and pj["words"]
    for what, ext in (("positions", "csv"), ("planes", "csv"), ("clusters", "csv"), ("history", "csv"), ("prototypes", "npz")):
        e = world.get(f"{B}/{rid}/export", params={"what": what})
        assert e.status_code == 200 and e.headers["content-disposition"].endswith(f".{ext}\"") or ext in e.headers["content-disposition"], what
    assert world.get(f"{B}/{rid}/export", params={"what": "zzz"}).status_code == 400
    z = np.load(__import__("io").BytesIO(world.get(f"{B}/{rid}/export", params={"what": "prototypes"}).content))
    assert z["prototype_eems"].shape == (20, 24, 16)
    m = world.get(f"{B}/{rid}/methods").json()
    assert "Kohonen" in m["text"] and "36 samples" in m["text"] and len(m["references"]) >= 8


def test_scans_run_as_jobs(world):
    sc = A.done(world, f"{B}/scan", {"kind": "size", "request": {"input": INP, **FAST}, "units": [8, 16, 30]})
    assert [r["units"] for r in sc["rows"]] == sorted(r["units"] for r in sc["rows"]) and sc["suggested"] and sc["verdict"]
    rid = fit(world)
    st = A.done(world, f"{B}/scan", {"kind": "stability", "result": rid, "n_runs": 3})
    assert 0 <= st["value"] <= 1 and st["verdict"]["verdict"]
    mc = A.done(world, f"{B}/scan", {"kind": "metric", "result": rid})
    assert {r["kind"] for r in mc["rows"]} >= {"l2", "mahalanobis", "standardize"}
    assert world.post(f"{B}/scan", json={"kind": "bogus"}).status_code == 400
    assert world.post(f"{B}/scan", json={"kind": "size", "request": {"input": INP}, "units": [1]}).status_code == 400


def test_too_few_samples_are_refused_in_words(world):
    ids = data_ids(world)
    world.S.project.set_excluded(ids[:31], True)
    j = A.run_job(world, f"{B}/fit", {"input": INP, **FAST, "map": {"size": [2, 2]}})
    assert j["status"] == "error" and "Only 5 samples" in j["error"] and "fewer than 8" in j["error"]
    ok = A.run_job(world, f"{B}/fit", {"input": INP, **FAST, "map": {"size": [2, 2]}, "allow_few_samples": True, "validation": {"folds": 2}})
    assert ok["status"] == "done", ok
    d = A.decode(world.get(f"{B}/defaults").json())
    assert d["n_common"] == 5 and d["advice"]["blocking"]


def test_variables_mode(world):
    rid = fit(world)
    body = {"input": {"kind": "variables", "keys": [f"res:{rid}:som_x", f"res:{rid}:som_y", "meta:doc"]}, "train": {"epochs_rough": 5, "epochs_fine": 10}, "map": {"size": [3, 4]}, "validation": {"folds": 3}}
    j = A.run_job(world, f"{B}/fit", body)
    assert j["status"] == "done", j
    rid2 = A.decode(j["result"])["rid"]
    pl = A.decode(world.get(f"{B}/{rid2}").json())["payload"]
    assert pl["kind"] == "variables" and pl["columns"] == [f"res:{rid}:som_x", f"res:{rid}:som_y", "meta:doc"] and pl["prototype_values"].shape == (12, 3) and "frozen" not in pl
    s = A.decode(world.get(f"{B}/{rid2}/sample/{data_ids(world)[2]}").json())
    assert len(s["variables"]) == 3
    pr = A.decode(world.get(f"{B}/{rid2}/prototypes").json())
    assert pr["surfaces"]["fluorescence"].shape[0] == 12
    assert world.post(f"{B}/{rid2}/plane", json={"kind": "total"}).status_code == 400
    j2 = A.run_job(world, "/api/models", {"result": rid2, "title": "x"})
    assert j2["status"] == "error" and "cannot be saved" in j2["error"]
    j3 = A.run_job(world, f"{B}/fit", {"input": {"kind": "variables", "keys": ["meta:site"]}})
    assert j3["status"] == "error" and "not a number" in j3["error"]
