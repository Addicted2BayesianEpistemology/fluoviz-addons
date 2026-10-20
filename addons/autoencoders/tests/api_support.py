"""Helpers of the API tests: an app with the add-on wired in (the real loader when the add-on can start, else ``register`` with a small stand-in for ``AddonAPI``), a small project, jobs,
decoding of typed arrays, and a fast stand-in for the training (the exact linear autoencoder) so that most tests run without PyTorch."""
from __future__ import annotations

import base64
import logging
import time
from pathlib import Path

import numpy as np

import kind_support as K

from fluoviz.analysis.frozen import register_kind
from fluoviz.server.app import create_app
from fluoviz.server.state import get_state, reset_state

ADDON = Path(__file__).resolve().parents[1]
BASE = "/api/addon/autoencoders"


class FakeAPI:
    """The part of ``fluoviz.addons.AddonAPI`` the add-on uses, applied at once (the real registry applies it after ``register`` returned)."""

    def __init__(self, app):
        self.app = app
        self.id, self.version = "autoencoders", "1.0.0"
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
    """Take the kind and the freezer of an earlier hand-wired app out of the registries (the framework refuses a kind that already exists)."""
    from fluoviz.analysis.frozen import KINDS
    from fluoviz.server import saved_kinds
    KINDS.pop("autoencoder", None)
    saved_kinds.FREEZERS.pop("autoencoder", None)
    saved_kinds.SAVEABLE_LABELS.pop("autoencoder", None)


def patch_training(monkeypatch, fn) -> None:
    """Replace the training of the add-on by ``fn(cfg, prepared, history_path, progress)``.  The framework imports the add-on's modules afresh when it starts the add-on, so the module is
    looked up by name AFTER ``make_app`` (a reference taken earlier would be a stale copy)."""
    import importlib
    monkeypatch.setattr(importlib.import_module("fluoviz_addon_autoencoders.server.trainer"), "train", fn)


def use_fake_training(monkeypatch) -> None:
    patch_training(monkeypatch, fake_train)


def make_app(workspace: Path, loader: bool = True):
    """``create_app()`` with the add-on: started by the framework's own loader (``FLUOVIZ_ADDONS_DEV``), and only when that cannot be done wired by hand with ``FakeAPI``."""
    import os
    forget_kind()
    saved = os.environ.get("FLUOVIZ_ADDONS_DEV")
    if loader:
        os.environ["FLUOVIZ_ADDONS_DEV"] = os.pathsep.join([str(ADDON), str(ADDON.parent / "common")])           # (the shared add-on is loaded too: this one requires it)
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
    active = bool(reg and getattr(reg.addons.get("autoencoders"), "active", False))
    if not active:
        import fluoviz_addon_autoencoders as addon
        addon.register(FakeAPI(app))
    return S, app, active


def decode(o):
    """What the browser does with ``{"$nd": ...}`` objects."""
    if isinstance(o, dict):
        if "$nd" in o:
            dt = {"f4": "<f4", "i4": "<i4", "u1": "u1"}[o["$nd"]]
            return np.frombuffer(base64.b64decode(o["b64"]), dtype=dt).reshape(o["shape"])
        return {k: decode(v) for k, v in o.items()}
    if isinstance(o, list):
        return [decode(v) for v in o]
    return o


def load_project(S, n: int = 30, seed: int = 3, absorbance: bool = False, kind_gaps: bool = True):
    """The small synthetic data set in the open project (the pixels below the Rayleigh line are masked by the scatter treatment).  Returns ``(X, A, ids)``."""
    X, A = K.small_eem(n=n, seed=seed, gaps=False)
    ids = [f"S{i:02d}" for i in range(n)]
    p = S.project
    p.set_eem(X, K.EX, K.EM, ids)
    if absorbance:
        p.set_absorbance(K.ABS_WL, K.small_abs(A), ids)
    if kind_gaps:                                                    # a scatter treatment, as a person would have: the observed pixels become an unstructured subset of the grid
        p.set_scatter({"rayleigh1": {"enabled": True}, "rayleigh2": {"enabled": True}, "raman1": {"enabled": True}, "raman2": {"enabled": True}, "below_rayleigh": {"enabled": True}, "treatment": "nan"})
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


# ----------------------------------------------------------------------------- a fast stand-in for the training (no PyTorch)
def fake_train(cfg, prepared, history_path, progress):
    """The exact linear autoencoder (weighted PCA of the network coordinates) shaped like a ``TrainResult``: what ``trainer.train`` returns, without PyTorch."""
    from fluoviz_addon_autoencoders.core import forward as fw
    from fluoviz_addon_autoencoders.core.train import TrainResult, HistoryWriter, evr, make_split
    d = cfg.latent_dim
    td = prepared.train_data(cfg.smoothify_metric, cfg.gap_weighting)
    n = td.n
    split = make_split(n, cfg)
    tr = np.asarray(split["train"], int)
    spec, arrays = K.linear_network(prepared, d=d, family=cfg.family, sigma=0.1)
    W = HistoryWriter(history_path)
    W.write({"event": "start", "mode": split["mode"], "n": n, "restarts": cfg.n_restarts, "folds": len(split["folds"]), "epochs": cfg.epochs})
    for e in range(3):
        progress((e + 1) / 3, f"Training, start 1 of {cfg.n_restarts}, epoch {e + 1}")
        W.write({"event": "epoch", "restart": 0, "fold": None, "epoch": e, "train": 1.0 / (e + 1), "val": 1.1 / (e + 1), "rec": 1.0 / (e + 1), "rec_val": 1.1 / (e + 1), "kl": None, "kl_val": None,
                 "smooth": None, "smooth_val": None, "beta": None, "sigma2": None, "lr": cfg.lr, "train_batch": 1.0, "best": e == 2})
    W.write({"event": "done"})
    W.close()
    net = fw.Network(spec, arrays)
    mu, lv = net.encode(prepared.X, prepared.M)
    xh = net.decode(mu)
    tot, per = fw.recon_error(spec, arrays, prepared.X, xh, prepared.M)
    null, _ = fw.null_error(spec, arrays, prepared.X, prepared.M)
    held = np.zeros(n, bool)
    if split["mode"] == "holdout":
        held[np.asarray(split["val"], int)] = True
    else:
        held[:] = True
    eh = np.where(held, tot, np.nan)
    from fluoviz_addon_autoencoders.core import linear as lin
    pm = lin.fit_pca(td.X[tr], None if td.M is None else td.M[tr], td.omega, d, sample_w=td.sw_of(tr))
    pe = lin.pca_errors(pm, td.X, td.M, td.omega, td.blocks, [d])[d]
    eb = np.where(held, pe, np.nan)                                                  # the linear baseline on the same held-out samples
    fold = None
    if split["mode"] == "kfold":
        fold = np.zeros(n, int)
        for f, idx in enumerate(split["folds"]):
            fold[np.asarray(idx, int)] = f
    ev_val, ev_tr = evr(tot[held], null[held]), evr(tot[~held] if (~held).any() else tot, null[~held] if (~held).any() else null)
    base = {"name": "PCA" if all(b.kind == "pixels" for b in prepared.blocks) else "Functional PCA", "k": d, "evr": ev_val - 0.0, "evr_train": ev_tr}
    names = prepared.names
    blocks = {nm: {"val": ev_val, "train": ev_tr} for nm in names}
    validation = {"mode": split["mode"], "n_val": int(held.sum()), "n_train": int((~held).sum()), "evr": ev_val, "evr_train": ev_tr, "error": float(tot[held].mean()), "error_train": float(tot.mean()),
                  "evr_blocks": blocks, "baseline": base, "kl": None, "n_eff": 5 if cfg.family == "vae" else None}
    samples = {"codes": mu, "codes_sd": None if lv is None else np.exp(0.5 * lv), "error": tot, "error_blocks": per, "null": null, "heldout": held, "error_heldout": eh, "error_baseline": eb, "fold": fold, "kl": None}
    hist = [{"restart": 0, "fold": None, "epoch": e, "train": 1.0 / (e + 1), "val": 1.1 / (e + 1), "rec": 1.0 / (e + 1), "lr": cfg.lr, "best": e == 2} for e in range(3)]
    restarts = {"n": 1, "runs": [{"restart": 0, "seed": 0, "best_epoch": 2, "epochs_run": 3, "stopped": "max", "chosen": True, "evr_train": ev_tr, "evr_val": ev_val}], "kept": 0, "metric": "evr_val",
                "mean": ev_val, "sd": 0.0, "min": ev_val, "max": ev_val}
    return TrainResult(spec, arrays, hist, validation, samples, restarts, 2, 3, split, cfg.to_dict(), spec.n_parameters(), 0.1 if cfg.family == "vae" else None, 0.01,
                       {"torch": "none", "numpy": np.__version__}, [], np.stack([mu]), [], [])
