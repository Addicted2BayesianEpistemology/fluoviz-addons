"""The saved model of an autoencoder through the generic models router: freeze, file, apply to the same samples (the codes of the page), to another project with other sample names, to a
coarser grid, planted anomalies, gaps; a re-opened project; the model as the first link of an LDA chain."""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

import api_support as A
import kind_support as K
from fluoviz.core import persistence
from fluoviz.core.project import Project

B = A.BASE
INP = {"variables": ["fluorescence"], "fluorescence": {"transform": "none", "min_coverage": 0.9, "basis": {"kind": "bspline", "n_basis": [8, 6]}, "smoothing": {"method": "gcv"}}}
SCATTER = {"rayleigh1": {"enabled": True}, "rayleigh2": {"enabled": True}, "raman1": {"enabled": True}, "raman2": {"enabled": True}, "below_rayleigh": {"enabled": True}, "treatment": "nan"}


@pytest.fixture()
def world(tmp_path, monkeypatch):
    S, app, _ = A.make_app(tmp_path / "ws")
    A.use_fake_training(monkeypatch)
    with TestClient(app) as c:
        c.S = S
        c.tmp = tmp_path
        c.data = A.load_project(S, n=36, absorbance=False)
        yield c
    S.shutdown()


def fit(c, **body):
    body = {"latent_dim": 3, "input": INP, "epochs": 5, **body}
    j = A.run_job(c, f"{B}/fit", body)
    assert j["status"] == "done", j
    return A.decode(j["result"])["rid"]


def save(c, rid, title="Saved AE", **body):
    j = A.run_job(c, "/api/models", {"result": rid, "title": title, **body})
    assert j["status"] == "done", j
    return A.decode(j["result"])


def apply(c, mid, body, expect="done"):
    j = A.run_job(c, f"/api/models/{mid}/apply", body)
    assert j["status"] == expect, j
    return A.decode(j["result"]) if expect == "done" else j["error"]


def codes_of(c, applied, names=("z1", "z2", "z3")):
    r = c.S.project.get_result(applied["rid"])
    return np.column_stack([np.asarray(r.variables[n], float) for n in names]), r


def other_project(c, X, ids, ex_sel=slice(None), em_sel=slice(None)):
    q = Project("Second project")
    q.set_eem(X[:, em_sel][:, :, ex_sel], K.EX[ex_sel], K.EM[em_sel], ids)
    c.S.replace_project(q, None)
    return q


def test_the_saved_model_reproduces_the_codes_of_the_page(world):
    rid = fit(world)
    res = world.S.project.get_result(rid)
    out = save(world, rid)
    mid = out["id"]
    rec = out["model"]
    assert rec["kind"] == "autoencoder" and rec["kind_label"] == "Functional autoencoder" and rec["title"] == "Saved AE" and rec["n_training"] == 36
    d = world.get(f"/api/models/{mid}").json()
    assert d["applicable"] and d["problem"] == "" and [o["name"] for o in d["options"]] == ["limit", "typicality"] and "z1" in [o["name"] for o in d["outputs"]]
    assert d["describe"]["training"]["Samples"] == 36 and d["describe"]["caveats"] and d["recipes"] and d["recipes"][0]["steps"] and d["validation"]["r2_val"] > 0.9
    assert world.S.models.get(mid).meta["addon"] == {"id": "autoencoders", "version": json.loads((A.ADDON / "manifest.json").read_text(encoding="utf-8"))["version"]}
    sv = world.get("/api/models/saveable").json()["results"][0]
    assert sv["id"] == rid and sv["kind"] == "autoencoder" and sv["saveable"] and sv["saved_as"] == [mid]
    chk = world.post(f"/api/models/{mid}/check", json={"select": "all"}).json()
    assert chk["ready"] and chk["samples"]["n"] == 36
    app = apply(world, mid, {"select": "all"})
    codes, r = codes_of(world, app)
    assert np.allclose(codes, np.column_stack([np.asarray(res.variables[f"z{j}"]) for j in (1, 2, 3)]), rtol=1e-8, atol=1e-10)          # the page's codes, to numpy precision
    assert r.sample_ids == res.sample_ids and set(r.variables) >= {"z1", "recon_rmse", "recon_r2", "flag", "status", "in_training"} and set(np.asarray(r.variables["in_training"])) == {"in the fit"}
    got = A.decode(world.get(f"/api/models/applied/{r.id}").json())
    assert set(got["payload"]["status"]) <= {"ok", "outside"} and got["payload"]["plot"]["axes"][0]["name"] == "z1" and got["payload"]["images"]["directions"].shape[0] == 3
    # thresholds are options of the application
    tight = apply(world, mid, {"select": "all", "options": {"limit": "95"}, "replace": r.id})
    assert tight["rid"] == r.id
    with pytest.raises(AssertionError):
        apply(world, mid, {"select": "all", "options": {"limit": "90"}})
    bad = world.post(f"/api/models/{mid}/apply", json={"select": "all", "options": {"typicality": 3}})
    assert bad.status_code == 400 and "at most" in bad.json()["error"]["message"]
    # one sample inspected through the saved model
    sid = res.sample_ids[4]
    ins = A.decode(world.get(f"/api/models/applied/{r.id}/sample/{sid}").json())
    assert set(ins["images"]) == {"measured", "reconstruction", "residual", "mean"} and np.asarray(ins["images"]["measured"]).shape == (16, 24) and sid in ins["text"]
    # freezing twice gives the same model (the fingerprint does not depend on the title)
    again = save(world, rid, title="Another name", duplicate=True)
    assert again["model"]["fingerprint"] == rec["fingerprint"]


def test_the_model_file_travels(world):
    rid = fit(world, family="vae")
    mid = save(world, rid)["id"]
    export = world.get(f"/api/models/{mid}/export")
    assert export.status_code == 200
    f = world.tmp / "ae.fluoviz-model"
    f.write_bytes(export.content)
    world.delete(f"/api/models/{mid}")
    got = world.post("/api/models/import", json={"path": str(f)})
    assert got.status_code == 200, got.text
    mid2 = got.json()["id"]
    codes, _ = codes_of(world, apply(world, mid2, {"select": "all"}))
    res = world.S.project.get_result(rid)
    assert np.allclose(codes, np.column_stack([np.asarray(res.variables[f"z{j}"]) for j in (1, 2, 3)]), rtol=1e-8, atol=1e-10)
    assert "z1_sd" in world.S.project.results["applied-1"].variables


def test_another_project_with_other_sample_names_gives_the_same_codes(world):
    X, A_, ids = world.data
    rid = fit(world)
    res = world.S.project.get_result(rid)
    mid = save(world, rid)["id"]
    ids2 = [f"new_{i:02d}" for i in range(36)]
    other_project(world, X, ids2)
    app = apply(world, mid, {"select": "all"})
    codes, r = codes_of(world, app)
    assert r.sample_ids == ids2 and np.allclose(codes, np.column_stack([np.asarray(res.variables[f"z{j}"]) for j in (1, 2, 3)]), rtol=1e-8, atol=1e-10)
    assert set(np.asarray(r.variables["in_training"])) == {"new"}


def test_a_coarser_grid_is_regridded_and_the_codes_stay_close(world):
    X, A_, ids = world.data
    rid = fit(world)
    res = world.S.project.get_result(rid)
    mid = save(world, rid)["id"]
    other_project(world, X, ids, ex_sel=np.r_[0:15:2, 15], em_sel=np.r_[0:23:2, 23])                 # every other wavelength (the last one kept: the data must cover the model's range)
    codes, r = codes_of(world, apply(world, mid, {"select": "all"}))
    ref = np.column_stack([np.asarray(res.variables[f"z{j}"]) for j in (1, 2, 3)])
    sd = ref.std(axis=0)
    assert (np.abs(codes - ref).max(axis=0) < 0.35 * sd).all() or np.mean(np.abs(codes - ref) / sd) < 0.2
    for j in range(3):
        assert abs(np.corrcoef(codes[:, j], ref[:, j])[0, 1]) > 0.9


def test_planted_samples_and_gaps_through_the_router(world):
    X, A_, ids = world.data
    rid = fit(world)
    mid = save(world, rid)["id"]
    Xn, _ = K.small_eem(n=14, seed=21, gaps=False)
    bad = Xn.copy()
    bad[:5] = K.extra_band(Xn[:5], size=1.0)
    gap = Xn.copy()
    gap[5, :, :10] = np.nan
    gap[6] = np.nan
    ids2 = [f"N{i:02d}" for i in range(14)]
    q = other_project(world, bad, ids2)
    q = other_project(world, np.where(np.isfinite(gap), bad, np.nan), ids2)
    r = world.S.project
    out = apply(world, mid, {"select": "all", "min_coverage": 0.8})
    res = world.S.project.get_result(out["rid"])
    st = list(np.asarray(res.payload["status"], str))
    assert st[:5] == ["outside"] * 5 and st[5] == "unusable" and st[6] == "unusable" and st[7:].count("outside") <= 1
    reasons = res.payload["reasons"]
    assert "above the 99 % limit" in reasons[0][0] and "at least 80" in reasons[5][0] and "at least 80" in reasons[6][0]
    low = apply(world, mid, {"select": "all", "min_coverage": 0.3})
    assert np.asarray(world.S.project.get_result(low["rid"]).payload["status"], str)[5] != "unusable"
    sv = world.post(f"/api/models/applied/{res.id}/save-labels", json={"variable": "flag", "column": "ae_flag"})
    assert sv.status_code == 200 and sv.json()["n_labelled"] >= 5


def test_a_reopened_project_can_freeze_and_apply_without_retraining(world):
    rid = fit(world, input=INP)
    p = world.S.project
    res = p.get_result(rid)
    z = np.asarray(res.variables["z1"]).copy()
    folder = persistence.save_project(p, world.tmp / "saved_project", everything=True)
    q = persistence.load_project(folder)
    world.S.replace_project(q, None)
    r2 = q.get_result(rid)
    assert r2.obj is None and np.allclose(np.asarray(r2.variables["z1"]), z)
    page = world.get(f"{B}/{rid}")
    assert page.status_code == 200 and A.decode(page.json())["payload"]["codes"] is not None
    mid = save(world, rid, "After re-opening")["id"]
    codes, _ = codes_of(world, apply(world, mid, {"select": "all"}))
    assert np.allclose(codes[:, 0], z, rtol=1e-8, atol=1e-10)
    assert world.get(f"{B}/{rid}/sample/S03").status_code == 200 and world.post(f"{B}/{rid}/traverse", json={"code": 1}).status_code == 200
    # the project file holds arrays only: no pickle of the network
    assert not list((folder / "results").glob("*.pkl"))


def test_an_outdated_result_cannot_be_saved_and_says_why(world):
    rid = fit(world)
    assert world.put("/api/state/corrections", json={"smoothing": {"enabled": True, "em_nm": 8}}).status_code == 200
    sv = world.get("/api/models/saveable").json()["results"][0]
    assert not sv["saveable"] and "Recompute it first" in sv["reason"]
    bad = world.post("/api/models", json={"result": rid})
    assert bad.status_code == 400 and "outdated" in bad.json()["error"]["message"]
    from fluoviz_addon_autoencoders.kinds import freeze_autoencoder
    from fluoviz.analysis.frozen import FrozenError
    with pytest.raises(FrozenError, match="Train again with the same settings"):
        freeze_autoencoder(world.S.project, world.S.project.get_result(rid), "x")


def test_a_damaged_result_is_refused_politely(world):
    rid = fit(world)
    r = world.S.project.get_result(rid)
    r.payload["frozen"] = {"version": 99}
    r.obj = None
    bad = world.post("/api/models", json={"result": rid, "title": "x"})
    assert bad.status_code == 200
    j = A.run_job.__wrapped__ if hasattr(A.run_job, "__wrapped__") else None
    import time
    jid = bad.json()["job_id"]
    for _ in range(100):
        j = world.get(f"/api/jobs/{jid}").json()
        if j["status"] not in ("queued", "running"):
            break
        time.sleep(0.05)
    assert j["status"] == "error" and "another version" in j["error"] and "Train again" in j["error"]
    assert world.get(f"{B}/{rid}/traverse").status_code in (404, 405)
    assert world.post(f"{B}/{rid}/traverse", json={}).status_code == 400


# ----------------------------------------------------------------------------- the model as the first link of a classification chain
def test_lda_on_the_codes_embeds_the_network_and_classifies_new_samples(world):
    X, A_, ids = world.data
    rid = fit(world)
    p = world.S.project
    import pandas as pd
    cls = np.where(A_[:, 0] > np.median(A_[:, 0]), "high", "low")
    p.merge_metadata(pd.DataFrame({"group": cls}, index=pd.Index(ids, name="sample")), record=False)
    keys = [f"res:{rid}:z{j}" for j in (1, 2, 3)]
    j = A.run_job(world, "/api/clustering/separation", {"classes": "meta:group", "keys": keys, "method": "lda", "seed": 1})
    assert j["status"] == "done", j
    lid = A.decode(j["result"])["result"]["id"]
    assert lid in [x["id"] for x in world.get("/api/models/saveable").json()["results"]]
    out = save(world, lid, "Group classifier")
    mid = out["id"]
    assert out["model"]["kind"] == "lda" and out["model"]["parts"] == [rid]
    d = world.get(f"/api/models/{mid}").json()
    assert d["applicable"] and d["parts"][0]["kind"] == "autoencoder" and d["recipes"][0]["part"] == rid
    ap = apply(world, mid, {"select": "all"})
    got = A.decode(world.get(f"/api/models/applied/{ap['rid']}").json())
    pred = {row["id"]: row for row in got["table"]["rows"]}
    assert all(r_["in_training"] for r_ in pred.values()) and np.mean([pred[i]["predicted"] == c_ for i, c_ in zip(ids, cls)]) > 0.8
    ld = np.asarray(p.get_result(lid).variables["LD1"], float)
    assert np.allclose(np.asarray(world.S.project.get_result(ap["rid"]).variables["LD1"], float), ld, rtol=1e-6, atol=1e-8)       # the page's discriminant scores, through the embedded network
    # another project, other names: the same classes (the network projects the new samples first)
    ids2 = [f"new_{i:02d}" for i in range(36)]
    other_project(world, X, ids2)
    ap2 = apply(world, mid, {"select": "all"})
    got2 = {row["id"]: row for row in A.decode(world.get(f"/api/models/applied/{ap2['rid']}").json())["table"]["rows"]}
    assert all(got2[n]["predicted"] == pred[o]["predicted"] for o, n in zip(ids, ids2))
    # a planted sample is reported as poorly described by the first link of the chain
    Xn, _ = K.small_eem(n=12, seed=5, gaps=False)
    bad = Xn.copy()
    bad[:3] = K.extra_band(Xn[:3], 1.0)
    other_project(world, bad, [f"P{i:02d}" for i in range(12)])
    ap3 = apply(world, mid, {"select": "all"})
    r3 = world.S.project.get_result(ap3["rid"])
    st = np.asarray(r3.payload["status"], str)
    assert (st[:3] != "ok").all() and any("autoencoder model does not describe this sample well" in x or "Autoencoder model does not describe" in x or "model does not describe this sample well" in x for x in r3.payload["reasons"][0])
