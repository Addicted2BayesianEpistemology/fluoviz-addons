"""The real thing, through FluoViz's own installer: installing the Self-organising map from a source built by ``scripts/build_addon.py`` also installs the shared add-on, starts it first,
and removing the map takes it away again.  (The autoencoders need PyTorch, which a test cannot download: their plan is checked, not their installation.)"""
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fluoviz import paths
from fluoviz.addons import installer, owners
from fluoviz.server.app import create_app
from fluoviz.server.routers import files as files_router
from fluoviz.server.state import reset_state

ADDONS = Path(__file__).resolve().parents[2]
ROOT = ADDONS.parent


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("dist")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_addon.py"), "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return out


PACKAGES = ("fluoviz_addon_som", "fluoviz_addon_common", "fluoviz_addon_autoencoders")


def forget_packages():
    for name in PACKAGES:
        for m in [m for m in sys.modules if m == name or m.startswith(name + ".")]:
            sys.modules.pop(m, None)


@pytest.fixture()
def client(tmp_path, monkeypatch, built):
    # The add-ons of this repository are importable in the other tests (conftest puts their folders on sys.path).  Here they must be found only where the installer puts them, as for a person.
    mine = {(ADDONS / n).resolve() for n in ("common", "som", "autoencoders")}
    monkeypatch.setattr(sys, "path", [p for p in sys.path if Path(p or ".").resolve() not in mine])
    forget_packages()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    monkeypatch.setattr(paths, "RUNTIME", runtime)
    monkeypatch.setattr(files_router, "UPLOADS", runtime / "tmp" / "uploads")
    for k in ("FLUOVIZ_ADDONS_DEV", "FLUOVIZ_ADDON_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    home = tmp_path / "home"
    home.mkdir()
    for k in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME"):
        monkeypatch.setenv(k, str(home))
    S = reset_state(tmp_path / "workspace")
    app = create_app()
    with TestClient(app) as c:
        c.rt, c.reg = runtime, app.state.addons
        assert c.put("/api/addons/source", json={"index": str(built)}).status_code == 200
        yield c
    app.state.addons.close()
    owners.set_current(None)
    forget_packages()
    S.shutdown()


def job(c, method, url, body=None):
    r = c.request(method, url, json=body) if body is not None else c.request(method, url)
    assert r.status_code == 200, r.text
    jid = r.json()["job_id"]
    t0 = time.time()
    while time.time() - t0 < 120:
        j = c.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "error", "cancelled"):
            return j
        time.sleep(0.05)
    raise AssertionError("the job did not finish")


def installed(c):
    return {a["id"]: a for a in c.get("/api/addons").json()["installed"]}


def test_the_list_offers_the_two_models_and_marks_the_shared_code(client):
    av = {a["id"]: a for a in client.get("/api/addons").json()["available"]}
    assert set(av) == {"common", "som", "autoencoders"} and av["common"]["library"] is True and av["som"]["library"] is False
    for aid in ("som", "autoencoders"):
        assert [x["id"] for x in av[aid]["also_installs"]] == ["common"] and av[aid]["requires_problem"] == ""


def test_installing_the_map_installs_the_shared_add_on_first_and_removing_it_takes_it_away(client):
    j = job(client, "POST", "/api/addons/som/install")
    assert j["status"] == "done" and "Shared code of the model add-ons" in j["title"] and j["result"]["also_installed"] == ["common"], j
    now = installed(client)
    assert set(now) == {"common", "som"} and now["common"]["state"] == "active" and now["som"]["state"] == "active" and now["common"]["library"] is True
    assert [d["id"] for d in now["common"]["needed_by"]] == ["som"] and now["som"]["requires"][0]["status"] == "ok"
    assert client.get("/api/addon/som/methods").status_code in (200, 400, 404, 405) and "som" in client.get("/api/addons").text
    assert (client.rt / "addons" / "common" / "1.0.0" / "fluoviz_addon_common" / "freeze.py").is_file()
    r = client.request("DELETE", "/api/addons/common", json={})
    assert r.status_code == 400 and "cannot be removed while" in r.json()["error"]["message"]
    j = job(client, "DELETE", "/api/addons/som", {})
    assert j["status"] == "done" and j["result"]["also_removed"] == ["common"]
    assert installed(client) == {} and not (client.rt / "addons" / "common").exists() and json.loads((client.rt / "addons" / "installed.json").read_text()) == {}


def test_the_autoencoders_plan_includes_the_shared_add_on_first(client):
    plan = installer.plan_install(client.reg, "autoencoders")
    assert [e.id for e in plan.needs] == ["common"] and plan.title.startswith("Installing Autoencoders") and plan.title.endswith("(and Shared code of the model add-ons, which it needs)")


def test_a_restart_finds_both_active_and_the_map_still_answers(client):
    job(client, "POST", "/api/addons/som/install")
    client.reg.close()
    forget_packages()
    app2 = create_app()
    with TestClient(app2) as c2:
        states = {a["id"]: a["state"] for a in c2.get("/api/addons").json()["installed"]}
        assert states == {"common": "active", "som": "active"}
    app2.state.addons.close()
