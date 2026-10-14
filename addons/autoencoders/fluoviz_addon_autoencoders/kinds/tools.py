"""What a frozen autoencoder can do beyond coding and decoding: traversals, interpolation, generation and sensitivity, as functions of every block of the model.

All of it is the decoder (numpy), so it works without the data and without PyTorch.  Functions are returned at the observed pixels of each block's domain ``(k, m_b)``, as dictionaries
``{block name: array}``; ``AEModel.embed`` puts them back on the grid of the block (NaN outside the domain, module orientation ``(n_em, n_ex)`` for a surface).
"""
from __future__ import annotations

import numpy as np

from .model import AEModel


def code_sd(codes, floor: float = 1e-9) -> np.ndarray:
    """SD of the training codes (the unit of the traversals); a code that does not vary gets a tiny positive value so nothing divides by zero."""
    c = np.asarray(codes, float)
    return np.maximum(c.std(axis=0, ddof=1) if len(c) > 1 else np.ones(c.shape[1]), floor)


def traverse(model: AEModel, j: int, values, mean: np.ndarray, sd: np.ndarray) -> dict:
    """Decode along code ``j`` (0-based): the other codes stay at ``mean``, code ``j`` takes ``mean_j + v sd_j`` for every ``v`` in ``values``.

    Returns ``{"z": (k,), "functions": {name: (k, m)}, "base": {name: (m,)}}``: ``base`` is the decoding of ``mean`` (what a traversal is compared with)."""
    values = np.asarray(values, float)
    mean = np.asarray(mean, float)
    Z = np.tile(mean, (values.size, 1))
    Z[:, j] = mean[j] + values * sd[j]
    base = model.decode_functions(mean[None, :])
    return {"z": Z[:, j].copy(), "functions": model.decode_functions(Z), "base": {k: v[0] for k, v in base.items()}}


def interpolate(model: AEModel, za: np.ndarray, zb: np.ndarray, t) -> dict:
    """Decode ``(1 - t) za + t zb`` for every ``t``: ``{"path": (k, d), "functions": {name: (k, m)}}``."""
    t = np.asarray(t, float)[:, None]
    path = (1.0 - t) * np.asarray(za, float)[None, :] + t * np.asarray(zb, float)[None, :]
    return {"path": path, "functions": model.decode_functions(path)}


def sensitivity(model: AEModel, mean: np.ndarray, sd: np.ndarray, codes: np.ndarray | None = None, rel_step: float = 0.01) -> dict:
    """What every code does to the output: ``d output / d z_j`` per 1 SD of the code, at the mean code (``at_mean``, signed ``(d, m)`` per block) and, when ``codes`` are given, its average
    magnitude over those samples (``magnitude``).  Central differences of the decoder (exact enough for a smooth network; the step is ``rel_step`` SD)."""
    mean = np.asarray(mean, float)
    d = model.d
    at_mean = {b.name: np.zeros((d, b.domain.m)) for b in model.frame.blocks}
    magnitude = {b.name: np.zeros((d, b.domain.m)) for b in model.frame.blocks} if codes is not None else None
    for j in range(d):
        h = rel_step * sd[j]
        if not h > 0:
            continue
        jobs = [(mean[None, :], False)] + ([(np.asarray(codes, float), True)] if codes is not None else [])
        for Zs, over_samples in jobs:
            up, dn = Zs.copy(), Zs.copy()
            up[:, j] += h
            dn[:, j] -= h
            fu, fd = model.decode_functions(up), model.decode_functions(dn)
            for name in at_mean:
                diff = (fu[name] - fd[name]) / (2.0 * h) * sd[j]
                if over_samples:
                    magnitude[name][j] = np.nanmean(np.abs(diff), axis=0)
                else:
                    at_mean[name][j] = diff[0]
    return {"at_mean": at_mean, "magnitude": magnitude}


def generate(model: AEModel, n: int, rng: np.random.Generator, temperature: float = 1.0) -> np.ndarray:
    """Codes drawn from the prior of a variational autoencoder: ``N(0, I)`` (standard deviation multiplied by ``temperature``), or, for a mixture prior, a draw from its components."""
    d = model.d
    if model.spec.has_mixture() and "prior.pi" in model.arrays:
        pi = np.clip(model.arrays["prior.pi"], 0, None)
        pi = pi / pi.sum()
        k = rng.choice(pi.size, size=n, p=pi)
        sd = np.exp(0.5 * model.arrays["prior.logvar"][k])
        return model.arrays["prior.mu"][k] + temperature * sd * rng.standard_normal((n, d))
    return temperature * rng.standard_normal((n, d))


def nearest(codes: np.ndarray, new: np.ndarray, sd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest training sample of every new code, distance in SD of the codes: ``(index (k,), distance (k,))``."""
    codes = np.asarray(codes, float) / sd
    new = np.asarray(new, float) / sd
    D = ((new[:, None, :] - codes[None, :, :]) ** 2).sum(axis=2)
    idx = np.argmin(D, axis=1)
    return idx, np.sqrt(D[np.arange(new.shape[0]), idx])
