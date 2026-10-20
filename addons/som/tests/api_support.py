"""Helpers of the API tests: an app with the add-on wired in (the framework's own loader when the add-on can start, else ``register`` with a small stand-in for ``AddonAPI``), a small
project, jobs, decoding of typed arrays."""
from __future__ import annotations

import base64
import logging
import time
from pathlib import Path

import numpy as np

import som_support as K
from fluoviz.analysis.frozen import register_kind
from fluoviz.server.app import create_app
from fluoviz.server.state import reset_state

ADDON = Path(__file__).resolve().parents[1]
BASE = "/api/addon/som"
SCATTER = {"rayleigh1": {"enabled": True}, "rayleigh2": {"enabled": True}, "raman1": {"enabled": True}, "raman2": {"enabled": True}, "below_rayleigh": {"enabled": True}, "treatment": "nan"}


class FakeAPI:
    """The part of ``fluoviz.addons.AddonAPI`` the add-on uses, applied at once."""

    def __init__(self, app):
        self.app = app
        self.id, self.version = "som", "1.0.0"
        self.dir, self.web_dir, self.data_dir = ADDON, ADDON / "web", ADDON / "_data"
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
    KINDS.pop("som", None)
    saved_kinds.FREEZERS.pop("som", None)
    saved_kinds.SAVEABLE_LABELS.pop("som", None)


def make_app(workspace: Path, loader: bool = True):
    import os
    forget_kind()
    saved = os.environ.get("FLUOVIZ_ADDONS_DEV")
    if loader:
        os.environ["FLUOVIZ_ADDONS_DEV"] = str(ADDON)
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
    active = bool(reg and getattr(reg.addons.get("som"), "active", False))
    if not active:
        import fluoviz_addon_som as addon
        addon.register(FakeAPI(app))
    return S, app, active


def decode(o):
    if isinstance(o, dict):
        if "$nd" in o:
            dt = {"f4": "<f4", "i4": "<i4", "u1": "u1"}[o["$nd"]]
            return np.frombuffer(base64.b64decode(o["b64"]), dtype=dt).reshape(o["shape"])
        return {k: decode(v) for k, v in o.items()}
    if isinstance(o, list):
        if o and all(isinstance(v, (int, float)) and not isinstance(v, bool) or v is None for v in o):
            return np.array([np.nan if v is None else v for v in o], float)
        out = [decode(v) for v in o]
        if out and all(isinstance(v, np.ndarray) and v.dtype.kind == "f" for v in out) and len({v.shape for v in out}) == 1:
            return np.stack(out)
        return out
    return o


def load_project(S, n: int = 30, seed: int = 3, absorbance: bool = False, scatter: bool = True):
    """The small synthetic data set in the open project.  Returns ``(X, A, ids)``."""
    X, A = K.small_eem(n=n, seed=seed, gaps=False)
    ids = [f"S{i:02d}" for i in range(n)]
    p = S.project
    p.set_eem(X, K.EX, K.EM, ids)
    if absorbance:
        p.set_absorbance(K.ABS_WL, K.small_abs(A), ids)
    if scatter:
        p.set_scatter(SCATTER)
    return X, A, ids


def run_job(c, url, body, timeout=300, method="post"):
    r = getattr(c, method)(url, json=body)
    assert r.status_code == 200, r.text
    jid = r.json()["job_id"]
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "error", "cancelled"):
            return j
        time.sleep(0.05)
    raise AssertionError("the job did not finish")


def done(c, url, body, timeout=300):
    j = run_job(c, url, body, timeout)
    assert j["status"] == "done", j.get("error")
    return decode(j["result"])
