"""A fit from start to end, outside the server: prepared modes -> a job of the Stan engine -> the draws -> the alignment -> a :class:`~.posterior.Fitted`.

``fit_model(modes, Y, cov, cfg, data_dir, progress)`` is what the fit job of the server calls, and what a script or a test can call with modes made from arrays.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from . import engine as E
from .posterior import Draws, Fitted, alignment
from .spec import Config, MblfError


def job_input(modes: list, Y: list, cov, cfg: Config) -> tuple[dict, dict]:
    """``(input.json, input.npz arrays)`` of the engine job."""
    D = len(modes)
    arrays = {}
    for d, (m, y) in enumerate(zip(modes, Y)):
        arrays[f"B{d}"] = np.asarray(m.B, float)
        arrays[f"Y{d}"] = np.asarray(y, float)
        arrays[f"C{d}"] = np.asarray(m.coords, float)
    N = Y[0].shape[1]
    arrays["X"] = np.zeros((N, 0)) if cov is None else np.asarray(cov.X, float)
    meta = {"task": cfg.estimation, "K": cfg.K, "priors": dict(cfg.priors), "chains": cfg.chains, "warmup": cfg.warmup, "draws": cfg.draws, "adapt_delta": cfg.adapt_delta,
            "max_treedepth": cfg.max_treedepth, "seed": cfg.seed, "keep": cfg.keep, "ml_starts": cfg.ml_starts, "D": D, "L": 0 if cov is None else int(cov.X.shape[1]),
            "modes": [m.name for m in modes]}
    return meta, arrays


def draws_from(out: dict, modes: list, estimation: str) -> Draws:
    """The engine's draws ``(chain, draw, ...)`` -> flat :class:`Draws` with the alignment."""
    lam = np.asarray(out["Lambda"], float)
    C, s = lam.shape[:2]
    flat = lambda a: np.asarray(a, float).reshape((C * s,) + np.asarray(a).shape[2:])          # noqa: E731
    lam = flat(lam)
    T, _ = alignment(modes, lam)
    return Draws(lam, 1.0 / flat(out["tau"]), flat(out["psi2"]), flat(out["regr_coeffs"]), T, int(C), estimation)


def fit_model(modes: list, Y: list, cov, cfg: Config, data_dir: Path, progress=None, ids: list | None = None, keep_job: bool = False) -> tuple[Fitted, dict]:
    """Sample (or maximise) the model; returns ``(Fitted, meta)`` with ``meta`` = the engine's diagnostics and timings, plus ``lp`` (the log density of every chain, thinned, for a
    picture) and the scores of the training samples as Stan sampled them (``eta``: chain-flattened, ``(S, K, N)``)."""
    eng = E.locate(data_dir)
    if eng.state != "ready":
        raise MblfError(eng.message)
    meta, arrays = job_input(modes, Y, cov, cfg)
    base = Path(data_dir) / "jobs"
    base.mkdir(parents=True, exist_ok=True)
    job = Path(tempfile.mkdtemp(prefix=time.strftime("fit-%Y%m%d-%H%M%S-"), dir=base))
    try:
        (job / "input.json").write_text(json.dumps(meta), encoding="utf-8")
        np.savez(job / "input.npz", **arrays)
        res = E.run_job(eng, Path(data_dir), job, progress)
        with np.load(job / "output.npz") as z:
            out = {k: z[k] for k in z.files}
    finally:
        if not keep_job:
            shutil.rmtree(job, ignore_errors=True)
    draws = draws_from(out, modes, cfg.estimation)
    f = Fitted(modes, cov, draws, [np.asarray(y, float) for y in Y], list(ids or []))
    lp = np.asarray(out.get("lp_full", out["lp__"]), float)
    step = max(1, lp.shape[1] // 300)
    res["lp"] = lp[:, ::step]
    res["eta"] = np.asarray(out["eta"], float).reshape((-1,) + out["eta"].shape[2:])
    if "divergent" in out:
        res["divergent_kept"] = np.asarray(out["divergent"], float).reshape(-1)
    return f, res
