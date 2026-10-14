"""Training for real (PyTorch in a worker process, started by the API): the live history, cancellation, determinism with a seed, the variational and pixel versions, two variables, errors
of the worker in plain English, and the rule that the FluoViz server process never imports PyTorch.  Skipped cleanly when PyTorch is not installed."""
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import api_support as A
import kind_support as K

pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch is not installed")
B = A.BASE
INP = {"variables": ["fluorescence"], "fluorescence": {"transform": "none", "min_coverage": 0.9, "basis": {"kind": "bspline", "n_basis": [8, 6]}, "smoothing": {"method": "gcv"}}}
FAST = {"latent_dim": 3, "epochs": 40, "patience": 20, "n_restarts": 1, "val_fraction": 0.2}


@pytest.fixture()
def world(tmp_path):
    S, app, _ = A.make_app(tmp_path / "ws")
    with TestClient(app) as c:
        c.S = S
        c.tmp = tmp_path
        c.data = A.load_project(S, n=36, absorbance=True)
        yield c
    S.shutdown()


def fit(c, **body):
    j = A.run_job(c, f"{B}/fit", {**FAST, "input": INP, **body}, timeout=600)
    assert j["status"] == "done", j
    return A.decode(j["result"])


def test_a_real_fit_polls_its_history_and_gives_the_numbers_of_the_saved_model(world):
    r = world.post(f"{B}/fit", json={**FAST, "input": INP, "epochs": 150, "patience": 150})
    jid = r.json()["job_id"]
    since, rows, statuses = 0, 0, set()
    t0 = time.time()
    while time.time() - t0 < 300:
        h = world.get(f"{B}/fit/{jid}/history", params={"since": since}).json()
        statuses.add(h["status"])
        rows += len(h["epochs"])
        since = h["next"]
        if world.get(f"/api/jobs/{jid}").json()["status"] in ("done", "error", "cancelled"):
            break
        time.sleep(0.05)
    j = world.get(f"/api/jobs/{jid}").json()
    assert j["status"] == "done", j
    h = world.get(f"{B}/fit/{jid}/history", params={"since": since}).json()
    rows += len(h["epochs"])
    allrows = world.get(f"{B}/fit/{jid}/history").json()
    assert rows == len(allrows["epochs"]) > 20 and "running" in statuses and allrows["status"] == "done" and allrows["epochs"][0]["epoch"] == 0 and allrows["n_restarts"] == 1
    assert {"train", "val", "rec", "lr"} <= set(allrows["epochs"][0]) and allrows["best"]["epoch"] >= -1
    rid = A.decode(j["result"])["rid"]
    res = world.S.project.get_result(rid)
    pl = res.payload
    assert pl["quality"]["r2_val"] > 0.5 and pl["quality"]["baseline"]["name"] == "Functional PCA" and pl["training"]["best_epoch"] >= -1 and pl["versions"]["torch"] and res.summary["n_params"] > 1000
    # the saved model gives the codes of the page for the training samples
    mid = A.decode(A.run_job(world, "/api/models", {"result": rid, "title": "Real"})["result"])["id"]
    j2 = A.run_job(world, f"/api/models/{mid}/apply", {"select": "all"})
    assert j2["status"] == "done"
    ap = world.S.project.get_result(A.decode(j2["result"])["rid"])
    for k in ("z1", "z2", "z3"):
        assert np.allclose(np.asarray(ap.variables[k], float), np.asarray(res.variables[k], float), rtol=1e-8, atol=1e-10)


def test_the_same_seed_gives_the_same_network_and_another_seed_another(world):
    a = fit(world, seed=5)["rid"]
    b = fit(world, seed=5)["rid"]
    c = fit(world, seed=6)["rid"]
    P = world.S.project
    za, zb, zc = (np.asarray(P.get_result(r).variables["z1"], float) for r in (a, b, c))
    assert np.array_equal(za, zb) and not np.allclose(za, zc)
    wa, wb = (P.get_result(r).payload["frozen"]["model"]["arrays"] for r in (a, b))
    assert all(np.array_equal(wa[k], wb[k]) for k in wa)


def test_cancelling_a_real_fit_stops_the_process_and_saves_nothing(world):
    r = world.post(f"{B}/fit", json={**FAST, "input": INP, "epochs": 5000, "patience": 5000})
    jid = r.json()["job_id"]
    t0 = time.time()
    while time.time() - t0 < 120:
        if world.get(f"{B}/fit/{jid}/history").json()["epochs"]:
            break
        time.sleep(0.1)
    else:
        raise AssertionError("the training never started")
    assert world.post(f"/api/jobs/{jid}/cancel").status_code == 200
    for _ in range(300):
        j = world.get(f"/api/jobs/{jid}").json()
        if j["status"] not in ("queued", "running"):
            break
        time.sleep(0.1)
    assert j["status"] == "cancelled" and not [k for k in world.S.project.results if k.startswith("autoencoder")]
    h = world.get(f"{B}/fit/{jid}/history").json()
    assert h["status"] == "cancelled"
    import multiprocessing
    time.sleep(0.5)
    assert not [c for c in multiprocessing.active_children() if c.name == "fluoviz-autoencoder"]            # the worker process is gone


def test_the_variational_pixel_and_two_variable_versions_train(world):
    vae = fit(world, family="vae", epochs=60, patience=30)["rid"]
    r = world.S.project.get_result(vae)
    assert {"z1_sd", "kl", "elbo"} <= set(r.variables) and np.isfinite(np.asarray(r.variables["kl"], float)).all() and r.payload["latent"]["kind"] == "vae"
    gen = world.post(f"{B}/{vae}/generate", json={"n": 3, "seed": 1})
    assert gen.status_code == 200
    pix = fit(world, input={"variables": ["fluorescence"], "fluorescence": {"basis": {"kind": "pixels"}, "smoothing": {"method": "none"}}})["rid"]
    rp = world.S.project.get_result(pix)
    assert rp.payload["mode"] == "pixels" and rp.payload["input"]["blocks"][0]["basis"] == "pixels" and rp.payload["smoothness"]["blocks"][0]["measure"].startswith("curvature")
    two = fit(world, input={**INP, "variables": ["fluorescence", "absorbance"], "absorbance": {"basis": {"kind": "bspline", "n_basis": 12}, "smoothing": {"method": "gcv"}}})["rid"]
    r2 = world.S.project.get_result(two)
    assert r2.payload["input"]["variables"] == ["fluorescence", "absorbance"] and set(r2.payload["recon"]["error_by_block"]) == {"fluorescence", "absorbance"}
    s = A.decode(world.get(f"{B}/{two}/sample/S04").json())
    assert [b["name"] for b in s["blocks"]] == ["fluorescence", "absorbance"]
    smooth = fit(world, smoothify=1.0, input={"variables": ["fluorescence"], "fluorescence": {"basis": {"kind": "gp", "lengths": [60.0, 40.0], "k_max": 30}, "smoothing": {"method": "ml"}}})["rid"]
    sm = world.S.project.get_result(smooth).payload["smoothness"]
    assert sm["smoothify"] == 1.0 and "smoothify is 1" in sm["words"]


def test_a_problem_in_the_worker_is_reported_in_plain_english(world):
    j = A.run_job(world, f"{B}/fit", {**FAST, "latent_dim": 32, "input": {"variables": ["fluorescence"], "fluorescence": {"basis": {"kind": "bspline", "n_basis": [4, 4]}, "smoothing": {"method": "gcv"}}}})
    assert j["status"] == "error" and "must be smaller than the number of inputs" in j["error"]
    assert world.get(f"{B}/fit/{j['id']}/history").json()["status"] == "error"


def test_the_server_never_imports_torch(tmp_path):
    """A server that fits (in a worker), looks at results and applies a saved model has not imported PyTorch: checked in a fresh process."""
    script = tmp_path / "check_no_torch.py"
    script.write_text(f'''
import sys, os, tempfile
sys.path.insert(0, {str(Path(__file__).parent)!r})
import api_support as A
from fastapi.testclient import TestClient
from pathlib import Path


def main():
    S, app, _ = A.make_app(Path(tempfile.mkdtemp()) / "ws")
    with TestClient(app) as c:
        A.load_project(S, n=24)
        inp = {{"variables": ["fluorescence"], "fluorescence": {{"basis": {{"kind": "bspline", "n_basis": [6, 5]}}, "smoothing": {{"method": "gcv"}}}}}}
        d = c.get(A.BASE + "/defaults").json()
        j = A.run_job(c, A.BASE + "/fit", {{"latent_dim": 2, "epochs": 6, "patience": 5, "n_restarts": 1, "input": inp}}, timeout=300)
        assert j["status"] == "done", j
        rid = A.decode(j["result"])["rid"]
        for u in ("", "/sample/S01", "/sensitivity", "/methods"):
            assert c.get(A.BASE + "/" + rid + u).status_code == 200
        c.post(A.BASE + "/" + rid + "/traverse", json={{}})
        jm = A.run_job(c, "/api/models", {{"result": rid, "title": "m"}})
        mid = A.decode(jm["result"])["id"]
        ja = A.run_job(c, "/api/models/" + mid + "/apply", {{"select": "all"}})
        assert ja["status"] == "done"
    bad = sorted(m for m in sys.modules if m == "torch" or m.startswith("torch."))
    print("TORCH_IN_SERVER" if bad else "NO_TORCH_IN_SERVER", len(bad))
    S.shutdown()


if __name__ == "__main__":
    main()
''')
    env = {k: v for k, v in os.environ.items()}
    env.pop("FLUOVIZ_ADDONS_DEV", None)
    env["PYTHONPATH"] = os.pathsep.join(x for x in sys.path if x)
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env, timeout=600)
    assert "NO_TORCH_IN_SERVER" in out.stdout, out.stdout[-1500:] + out.stderr[-3000:]
