"""The saved map through the generic models router: freeze, file, apply to the same samples (the positions of the page), to another project with other sample names, to a coarser grid,
planted anomalies, a re-opened project; the model as the first link of an LDA chain."""
import json
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api_support as A
import som_support as K
from fluoviz.core import persistence
from fluoviz.core.project import Project

B = A.BASE
INP = {"variables": ["fluorescence"], "fluorescence": {"transform": "none", "min_coverage": 0.9, "basis": {"kind": "bspline", "n_basis": [8, 6]}, "smoothing": {"method": "gcv"}}}
INP2 = {**INP, "variables": ["fluorescence", "absorbance"], "absorbance": {"transform": "none", "processed": False, "basis": {"kind": "bspline", "n_basis": 12}, "smoothing": {"method": "gcv"}}}
FAST = {"train": {"epochs_rough": 8, "epochs_fine": 20}, "validation": {"folds": 4}, "map": {"size": [4, 5]}}
POS = ("som_x", "som_y")


@pytest.fixture()
def world(tmp_path):
    S, app, _ = A.make_app(tmp_path / "ws")
    with TestClient(app) as c:
        c.S = S
        c.tmp = tmp_path
        c.data = A.load_project(S, n=36, absorbance=False)
        yield c
    S.shutdown()


def fit(c, inp=INP, **body):
    j = A.run_job(c, f"{B}/fit", {"input": inp, **FAST, **body})
    assert j["status"] == "done", j
    return A.decode(j["result"])["rid"]


def save(c, rid, title="Saved map", **body):
    j = A.run_job(c, "/api/models", {"result": rid, "title": title, **body})
    assert j["status"] == "done", j
    return A.decode(j["result"])


def apply(c, mid, body, expect="done"):
    j = A.run_job(c, f"/api/models/{mid}/apply", body)
    assert j["status"] == expect, j
    return A.decode(j["result"]) if expect == "done" else j["error"]


def pos_of(c, applied):
    r = c.S.project.get_result(applied["rid"])
    return np.column_stack([np.asarray(r.variables[n], float) for n in POS]), r


def page_pos(c, rid):
    res = c.S.project.get_result(rid)
    return np.column_stack([np.asarray(res.variables[n], float) for n in POS]), res


def other_project(c, X, ids, ex_sel=slice(None), em_sel=slice(None)):
    q = Project("Second project")
    q.set_eem(X[:, em_sel][:, :, ex_sel], K.EX[ex_sel], K.EM[em_sel], ids)
    c.S.replace_project(q, None)
    return q


def test_the_saved_model_reproduces_the_positions_of_the_page(world):
    rid = fit(world)
    ref, res = page_pos(world, rid)
    out = save(world, rid)
    mid = out["id"]
    rec = out["model"]
    assert rec["kind"] == "som" and rec["kind_label"] == "Functional self-organising map" and rec["title"] == "Saved map" and rec["n_training"] == 36
    d = world.get(f"/api/models/{mid}").json()
    assert d["applicable"] and d["problem"] == "" and [o["name"] for o in d["options"]] == ["limit"] and "som_x" in [o["name"] for o in d["outputs"]]
    assert d["describe"]["training"]["Samples"] == 36 and d["describe"]["caveats"] and d["recipes"] and d["validation"]["captured"] > 0.5
    assert world.S.models.get(mid).meta["addon"] == {"id": "som", "version": json.loads((A.ADDON / "manifest.json").read_text(encoding="utf-8"))["version"]}
    sv = world.get("/api/models/saveable").json()["results"][0]
    assert sv["id"] == rid and sv["kind"] == "som" and sv["saveable"] and sv["saved_as"] == [mid]
    app = apply(world, mid, {"select": "all"})
    pos, r = pos_of(world, app)
    assert np.allclose(pos, ref, rtol=1e-8, atol=1e-10)
    assert np.array_equal(np.asarray(r.variables["som_unit"]), np.asarray(res.variables["som_unit"]))
    assert np.allclose(np.asarray(r.variables["som_qe"], float), np.asarray(res.variables["som_qe"], float), rtol=1e-8)
    assert set(r.variables) >= {"som_unit", "som_row", "som_col", "som_x", "som_y", "som_qe", "som_qe_rel", "som_qe_rmse", "som_umatrix", "som_cluster", "flag", "status", "in_training"}
    got = A.decode(world.get(f"/api/models/applied/{r.id}").json())
    assert set(got["payload"]["status"]) <= {"ok", "outside", "uncertain"} and got["payload"]["plot"]["axes"][0]["name"] == "som_x"
    tight = apply(world, mid, {"select": "all", "options": {"limit": "95"}, "replace": r.id})
    assert tight["rid"] == r.id
    with pytest.raises(AssertionError):
        apply(world, mid, {"select": "all", "options": {"limit": "90"}})
    sid = res.sample_ids[4]
    ins = A.decode(world.get(f"/api/models/applied/{r.id}/sample/{sid}").json())
    assert set(ins["images"]) == {"measured", "prototype", "difference", "mean"} and np.asarray(ins["images"]["measured"]).shape == (16, 24) and sid in ins["text"]
    again = save(world, rid, title="Another name", duplicate=True)
    assert again["model"]["fingerprint"] == rec["fingerprint"]


def test_both_variables_and_pixels_freeze_and_apply(world):
    world.data = A.load_project(world.S, n=36, absorbance=True)
    for inp in (INP2, {"kind": "pixels", "variables": ["fluorescence", "absorbance"], "fluorescence": {"min_coverage": 0.9}, "absorbance": {"processed": False}}):
        rid = fit(world, inp)
        ref, res = page_pos(world, rid)
        mid = save(world, rid, title=f"Map {rid}")["id"]
        pos, r = pos_of(world, apply(world, mid, {"select": "all"}))
        assert np.allclose(pos, ref, rtol=1e-8, atol=1e-10)
        assert "som_qe_rmse_fluorescence" in r.variables and "som_qe_rmse_absorbance" in r.variables


def test_the_model_file_travels(world):
    rid = fit(world)
    ref, _ = page_pos(world, rid)
    mid = save(world, rid)["id"]
    export = world.get(f"/api/models/{mid}/export")
    assert export.status_code == 200
    f = world.tmp / "som.fluoviz-model"
    f.write_bytes(export.content)
    world.delete(f"/api/models/{mid}")
    got = world.post("/api/models/import", json={"path": str(f)})
    assert got.status_code == 200, got.text
    pos, _ = pos_of(world, apply(world, got.json()["id"], {"select": "all"}))
    assert np.allclose(pos, ref, rtol=1e-8, atol=1e-10)


def test_another_project_with_other_sample_names_gives_the_same_positions(world):
    X, A_, ids = world.data
    rid = fit(world)
    ref, _ = page_pos(world, rid)
    mid = save(world, rid)["id"]
    ids2 = [f"new_{i:02d}" for i in range(36)]
    other_project(world, X, ids2)
    pos, r = pos_of(world, apply(world, mid, {"select": "all"}))
    assert r.sample_ids == ids2 and np.allclose(pos, ref, rtol=1e-8, atol=1e-10)
    assert set(np.asarray(r.variables["in_training"])) == {"new"}


def test_a_coarser_grid_is_regridded_and_the_samples_stay_close(world):
    X, A_, ids = world.data
    rid = fit(world)
    ref, res = page_pos(world, rid)
    mid = save(world, rid)["id"]
    other_project(world, X, ids, ex_sel=np.r_[0:15:2, 15], em_sel=np.r_[0:23:2, 23])
    pos, r = pos_of(world, apply(world, mid, {"select": "all"}))
    assert np.mean(np.linalg.norm(pos - ref, axis=1)) < 1.5
    same = np.mean(np.asarray(r.variables["som_unit"]) == np.asarray(res.variables["som_unit"]))
    assert same > 0.4


def test_planted_samples_through_the_router(world):
    X, A_, ids = world.data
    rid = fit(world)
    mid = save(world, rid)["id"]
    Xn, _ = K.small_eem(n=14, seed=21, gaps=False)
    bad = Xn.copy()
    bad[:5] = K.extra_band(Xn[:5], size=10.0, em=430.0, ex=392.0)
    gap = np.where(np.isfinite(bad), bad, np.nan)
    gap[5, :, :10] = np.nan
    gap[6] = np.nan
    other_project(world, gap, [f"N{i:02d}" for i in range(14)])
    out = apply(world, mid, {"select": "all", "min_coverage": 0.8})
    res = world.S.project.get_result(out["rid"])
    st = list(np.asarray(res.payload["status"], str))
    assert st[:5] == ["outside"] * 5 and st[5] == "unusable" and st[6] == "unusable"
    reasons = res.payload["reasons"]
    assert "above the 99 % limit" in reasons[0][0] and "at least 80" in reasons[5][0]
    low = apply(world, mid, {"select": "all", "min_coverage": 0.3})
    assert np.asarray(world.S.project.get_result(low["rid"]).payload["status"], str)[5] != "unusable"
    assert np.isnan(np.asarray(res.variables["som_x"], float)[5])


def test_a_reopened_project_can_freeze_and_apply_without_computing_again(world):
    rid = fit(world)
    p = world.S.project
    ref, _ = page_pos(world, rid)
    folder = persistence.save_project(p, world.tmp / "saved_project", everything=True)
    q = persistence.load_project(folder)
    world.S.replace_project(q, None)
    r2 = q.get_result(rid)
    assert r2.obj is None and np.allclose(np.asarray(r2.variables["som_x"]), ref[:, 0])
    page = world.get(f"{B}/{rid}")
    assert page.status_code == 200 and A.decode(page.json())["payload"]["samples"]["pos"] is not None
    mid = save(world, rid, "After re-opening")["id"]
    pos, _ = pos_of(world, apply(world, mid, {"select": "all"}))
    assert np.allclose(pos, ref, rtol=1e-8, atol=1e-10)
    assert world.get(f"{B}/{rid}/sample/S03").status_code == 200
    assert not list((folder / "results").glob("*.pkl"))


def test_the_groups_set_later_travel_with_the_model(world):
    rid = fit(world)
    world.post(f"{B}/{rid}/cluster", json={"k": 3, "method": "kmeans"})
    mid = save(world, rid)["id"]
    _, r = pos_of(world, apply(world, mid, {"select": "all"}))
    assert len(set(np.asarray(r.variables["som_cluster"]).tolist())) == 3
    res = world.S.project.get_result(rid)
    assert list(np.asarray(r.variables["som_cluster"])) == list(np.asarray(res.variables["som_cluster"]))


def test_an_outdated_result_cannot_be_saved_and_says_why(world):
    rid = fit(world)
    assert world.put("/api/state/corrections", json={"smoothing": {"enabled": True, "em_nm": 8}}).status_code == 200
    sv = world.get("/api/models/saveable").json()["results"][0]
    assert not sv["saveable"]
    bad = world.post("/api/models", json={"result": rid})
    assert bad.status_code == 400 and "outdated" in bad.json()["error"]["message"]
    from fluoviz.analysis.frozen import FrozenError
    from fluoviz_addon_som.kinds import freeze_som
    with pytest.raises(FrozenError, match="Compute again with the same settings"):
        freeze_som(world.S.project, world.S.project.get_result(rid), "x")


def test_a_damaged_result_is_refused_politely(world):
    rid = fit(world)
    r = world.S.project.get_result(rid)
    r.payload["frozen"] = {"version": 99}
    r.obj = None
    j = A.run_job(world, "/api/models", {"result": rid, "title": "x"})
    assert j["status"] == "error" and "Compute again with the same settings" in j["error"]


def test_lda_on_the_positions_embeds_the_map_and_classifies_new_samples(world):
    X, A_, ids = world.data
    rid = fit(world)
    p = world.S.project
    cls = np.where(A_[:, 0] > np.median(A_[:, 0]), "high", "low")
    p.merge_metadata(pd.DataFrame({"group": cls}, index=pd.Index(ids, name="sample")), record=False)
    keys = [f"res:{rid}:som_x", f"res:{rid}:som_y"]
    j = A.run_job(world, "/api/clustering/separation", {"classes": "meta:group", "keys": keys, "method": "lda", "seed": 1})
    assert j["status"] == "done", j
    lid = A.decode(j["result"])["result"]["id"]
    out = save(world, lid, "Group classifier")
    mid = out["id"]
    assert out["model"]["kind"] == "lda" and out["model"]["parts"] == [rid]
    d = world.get(f"/api/models/{mid}").json()
    assert d["applicable"] and d["parts"][0]["kind"] == "som"
    ap = apply(world, mid, {"select": "all"})
    ld = np.asarray(p.get_result(lid).variables["LD1"], float)
    assert np.allclose(np.asarray(world.S.project.get_result(ap["rid"]).variables["LD1"], float), ld, rtol=1e-6, atol=1e-8)
    ids2 = [f"new_{i:02d}" for i in range(36)]
    other_project(world, X, ids2)
    got = {row["id"]: row for row in A.decode(world.get(f"/api/models/applied/{apply(world, mid, {'select': 'all'})['rid']}").json())["table"]["rows"]}
    assert len(got) == 36
    Xn, _ = K.small_eem(n=12, seed=5, gaps=False)
    bad = Xn.copy()
    bad[:3] = K.extra_band(Xn[:3], 10.0, em=430.0, ex=392.0)
    other_project(world, bad, [f"P{i:02d}" for i in range(12)])
    r3 = world.S.project.get_result(apply(world, mid, {"select": "all"})["rid"])
    st = np.asarray(r3.payload["status"], str)
    assert (st[:3] != "ok").any() and (st[3:] == "ok").mean() > 0.6
    flagged = [x for i in range(3) for x in r3.payload["reasons"][i]]
    assert any("model does not describe this sample well" in x for x in flagged)
