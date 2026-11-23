"""Helpers of the API tests: an app with the add-on wired in (the framework's own loader when the add-on can start, else ``register`` with a small stand-in for ``AddonAPI``), a small
project, jobs, decoding of typed arrays, and whether a Stan engine is available."""
from __future__ import annotations

import base64
import logging
import os
import time
from pathlib import Path

import numpy as np

import mblf_support as K
from fluoviz.analysis.frozen import register_kind
from fluoviz.server.app import create_app
from fluoviz.server.state import reset_state

ADDON = Path(__file__).resolve().parents[1]
BASE = "/api/addon/mblf"
SCATTER = {"rayleigh1": {"enabled": True}, "rayleigh2": {"enabled": True}, "raman1": {"enabled": True}, "raman2": {"enabled": True}, "below_rayleigh": {"enabled": True}, "treatment": "nan"}


def engine_reason() -> str | None:
    """None when the tests can run the Stan engine, else why not (the slow tests are skipped with it)."""
    from fluoviz_addon_mblf.core import engine as E
    if not os.environ.get("FLUOVIZ_MBLF_ENGINE"):
        return "no Stan engine for the tests: set FLUOVIZ_MBLF_ENGINE to a conda environment with cmdstan, cmdstanpy, xarray, numpy and scipy"
    eng = E.locate(Path("."))
    return None if eng.state == "ready" else eng.message


class FakeAPI:
    """The part of ``fluoviz.addons.AddonAPI`` the add-on uses, applied at once."""

    def __init__(self, app, data_dir: Path):
        self.app = app
        self.id, self.version = "mblf", "1.0.0"
        self.dir, self.web_dir, self.data_dir = ADDON, ADDON / "web", data_dir
        self.log = logging.getLogger("test.addon")
        self.kinds, self.result_kinds, self.callbacks = [], [], []

    def add_router(self, router):
        assert router.prefix.startswith(f"/api/addon/{self.id}")
        self.app.include_router(router)

    def add_saved_kind(self, kind, label, applier, result_kinds=(), freezer=None):
        from fluoviz.server import saved_kinds
        saved_kinds.discover()
        register_kind(kind, label, applier)
        for rk in result_kinds:
            saved_kinds.FREEZERS[rk] = freezer
            saved_kinds.SAVEABLE_LABELS[rk] = label
        self.kinds.append(kind)

    def add_result_kind(self, kind, noun, saveable=False):
        self.result_kinds.append((kind, noun, saveable))

    def on_remove(self, fn):
        self.callbacks.append(fn)


def forget_kind() -> None:
    from fluoviz.analysis.frozen import KINDS
    from fluoviz.server import saved_kinds
    KINDS.pop("mblf", None)
    saved_kinds.FREEZERS.pop("mblf", None)
    saved_kinds.SAVEABLE_LABELS.pop("mblf", None)


def make_app(workspace: Path, loader: bool = True):
    forget_kind()
    saved = os.environ.get("FLUOVIZ_ADDONS_DEV")
    if loader:
        os.environ["FLUOVIZ_ADDONS_DEV"] = os.pathsep.join([str(ADDON), str(ADDON.parent / "common")])
    else:
        os.environ.pop("FLUOVIZ_ADDONS_DEV", None)
    try:
        S = reset_state(workspace)
        app = create_app()
    finally:
        if saved is None:
            os.environ.pop("FLUOVIZ_ADDONS_DEV", None)
        else:
            os.environ["FLUOVIZ_ADDONS_DEV"] = saved
    reg = getattr(app.state, "addons", None)
    active = bool(reg and getattr(reg.addons.get("mblf"), "active", False))
    if not active:
        import fluoviz_addon_mblf as addon
        addon.register(FakeAPI(app, workspace / "_mblf_data"))
    return S, app, active


def decode(o):
    if isinstance(o, dict):
        if "$nd" in o:
            dt = {"f4": "<f4", "i4": "<i4", "u1": "u1", "f8": "<f8"}[o["$nd"]]
            return np.frombuffer(base64.b64decode(o["b64"]), dtype=dt).reshape(o["shape"])
        return {k: decode(v) for k, v in o.items()}
    if isinstance(o, list):
        if o and all(isinstance(v, (int, float)) and not isinstance(v, bool) or v is None for v in o):
            return np.array([np.nan if v is None else v for v in o], float)
        return [decode(v) for v in o]
    return o


def load_project(S, n: int = 30, seed: int = 0, absorbance: bool = True, meta: bool = True, scatter: bool = True):
    X, A, ids, md = K.small_project_data(n=n, seed=seed)
    p = S.project
    p.set_eem(X, K.EX, K.EM, ids)
    if absorbance:
        A = A.copy()
        A[3] = np.nan                                                    # one sample without an absorbance spectrum
        p.set_absorbance(K.ABS_WL, A, ids)
    if meta:
        p.merge_metadata(md)
    if scatter:
        p.set_scatter(SCATTER)
    return X, A, ids, md


def run_job(c, url, body, timeout=900, method="post"):
    r = getattr(c, method)(url, json=body)
    assert r.status_code == 200, r.text
    jid = r.json()["job_id"]
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "error", "cancelled"):
            return j
        time.sleep(0.1)
    raise AssertionError("the job did not finish")


def done(c, url, body, timeout=900):
    j = run_job(c, url, body, timeout)
    assert j["status"] == "done", j.get("error")
    return decode(j["result"])
