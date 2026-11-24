"""Synthetic data and fitted models for the tests of the numerics: modes with known bases, data drawn from the model, and "posterior draws" made of the true parameters with a
little jitter (so the numerics can be checked against brute-force Gaussian algebra without running Stan)."""
from __future__ import annotations

import numpy as np

from fluoviz_addon_mblf.core.modes import Covariates, Mode
from fluoviz_addon_mblf.core.posterior import Draws, Fitted, alignment


def cos_basis(x: np.ndarray, p: int) -> np.ndarray:
    u = (x - x.min()) / (x.max() - x.min())
    return np.column_stack([np.cos(np.pi * k * u) / (1 + k) for k in range(p)])


def make_modes(M1=30, M2=36, p1=6, p2=7, values=0) -> list[Mode]:
    x = np.linspace(250, 600, M1)
    B1 = cos_basis(x, p1)
    g = np.stack(np.meshgrid(np.linspace(300, 500, 6), np.linspace(250, 400, M2 // 6), indexing="ij"), -1).reshape(-1, 2)
    B2 = np.column_stack([cos_basis(g[:, 0], 4)[:, :min(4, p2)]] + [cos_basis(g[:, 1], p2 - 3)[:, 1:]])[:, :p2]
    modes = [Mode("absorbance", "curve", B1, np.linspace(1, 0.1, M1), np.full(M1, 0.2), np.full(M1, 1 / M1), x[:, None], np.ones(p1), 0.99, {}, None, [], [], "absorbance"),
             Mode("fluorescence", "surface", B2, np.ones(len(g)), np.full(len(g), 3.0), np.full(len(g), 1 / len(g)), g, np.ones(p2), 0.99, {}, None, [], [], "RU")]
    if values:
        modes.append(Mode("values", "vector", np.eye(values), np.zeros(values), np.ones(values), np.full(values, 1 / values), np.arange(1, values + 1.0)[:, None], None, None, {},
                          None, [f"meta:v{i}" for i in range(values)], [f"v{i}" for i in range(values)], ""))
    return modes


def truth(modes, K=2, L=1, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    P = sum(m.p for m in modes)
    return {"lam": rng.normal(size=(P, K)) * 0.8, "sig2": rng.uniform(0.02, 0.1, P), "psi2": rng.uniform(0.01, 0.05, len(modes)), "beta": rng.normal(size=(K, L)) * 0.7}


def simulate(modes, tr, N=40, seed=1, missing=0.1, X=None) -> tuple[list, np.ndarray]:
    rng = np.random.default_rng(seed)
    K, L = tr["beta"].shape
    X = rng.normal(size=(N, L)) if X is None else X
    X = X - X.mean(axis=0)
    eta = X @ tr["beta"].T + rng.normal(size=(N, K))
    Y, a = [], 0
    for d, m in enumerate(modes):
        sl = slice(a, a + m.p)
        a += m.p
        theta = tr["lam"][sl] @ eta.T + rng.normal(size=(m.p, N)) * np.sqrt(tr["sig2"][sl])[:, None]
        y = m.B @ theta + rng.normal(size=(m.M, N)) * np.sqrt(tr["psi2"][d])
        y[rng.random(y.shape) < missing] = np.nan
        Y.append(y)
    return Y, X


def fitted(modes, tr, Y, X, S=60, jitter=0.03, seed=2, chains=2, L=1) -> Fitted:
    rng = np.random.default_rng(seed)
    P, K = tr["lam"].shape
    lam = tr["lam"][None] + jitter * rng.normal(size=(S, P, K))
    sig2 = tr["sig2"][None] * np.exp(jitter * rng.normal(size=(S, P)))
    psi2 = tr["psi2"][None] * np.exp(jitter * rng.normal(size=(S, len(modes))))
    beta = tr["beta"][None] + jitter * rng.normal(size=(S, K, L))
    T, _ = alignment(modes, lam)
    cov = None
    if L:
        cov = Covariates(["meta:x"], ["x"], [{"key": "meta:x", "label": "x", "kind": "numeric", "level": None, "reference": None, "mean": 0.0, "sd": 1.0, "known": []}], X)
    else:
        beta = np.zeros((S, K, 0))
    return Fitted(modes, cov, Draws(lam, sig2, psi2, beta, T, chains, "bayes"), Y, [f"S{i:02d}" for i in range(Y[0].shape[1])])


def joint_cov(f: Fitted, s: int, covariates: bool):
    """The mean ``(sum M,)`` and covariance of all the modes of a sample at draw ``s`` (explicitly), and the stacked loadings on the points: for brute-force checks."""
    K = f.K
    lam = f.draws.lam[s]
    Psi = np.eye(K) if (covariates or f.L == 0) else np.eye(K) + f.draws.beta[s] @ f.cov.cov() @ f.draws.beta[s].T
    BL, Bres, noise = [], [], []
    for d, (m, sl) in enumerate(zip(f.modes, f.offsets())):
        BL.append(m.B @ lam[sl])
        Bres.append(m.B * np.sqrt(f.draws.sig2[s, sl])[None])
        noise.append(np.full(m.M, f.draws.psi2[s, d]))
    BLs = np.vstack(BL)
    from scipy.linalg import block_diag
    R = block_diag(*[b @ b.T for b in Bres])
    C = BLs @ Psi @ BLs.T + R + np.diag(np.concatenate(noise))
    return BLs, Psi, R, np.concatenate(noise), C


# ----------------------------------------------------------------------------- a small project (the API tests)
EX = np.linspace(250.0, 400.0, 16)
EM = np.linspace(300.0, 520.0, 24)
ABS_WL = np.linspace(240.0, 500.0, 40)


def _g(x, c, w):
    return np.exp(-0.5 * ((x - c) / w) ** 2)


def small_project_data(n: int = 30, seed: int = 0, noise: float = 0.01):
    """EEMs ``(n, 24, 16)`` of 3 fluorophores (NaN below the Rayleigh line), absorbance spectra driven by the same concentrations, a laboratory value (DOC) and a site.
    Returns ``(X, A, ids, meta)``."""
    rng = np.random.default_rng(seed)
    em_c, em_w, ex_c, ex_w = [340.0, 430.0, 480.0], [25.0, 35.0, 30.0], [275.0, 330.0, 350.0], [18.0, 25.0, 22.0]
    Bm = np.stack([_g(EM, em_c[f], em_w[f]) for f in range(3)], axis=1)
    Cx = np.stack([_g(EX, ex_c[f], ex_w[f]) for f in range(3)], axis=1)
    site = np.where(np.arange(n) % 2 == 0, "river", "lake")
    conc = rng.gamma(2.0, 1.0, size=(n, 3)) * np.where(site == "river", 1.4, 0.8)[:, None]
    X = np.einsum("nf,mf,pf->nmp", conc, Bm, Cx)
    X = X + noise * X.max() * rng.standard_normal(X.shape)
    X[:, EM[:, None] < EX[None, :] + 10.0] = np.nan
    shapes = np.stack([_g(ABS_WL, c, w) for c, w in ((280.0, 30.0), (340.0, 40.0), (420.0, 50.0))], axis=1)
    A = conc @ shapes.T * 0.05 + 0.002 * rng.standard_normal((n, ABS_WL.size)) + 0.01
    doc = conc @ np.array([1.0, 0.6, 0.3]) + 0.1 * rng.standard_normal(n)
    ids = [f"S{i:02d}" for i in range(n)]
    import pandas as pd
    meta = pd.DataFrame({"site": site, "doc": doc, "temp": np.linspace(5, 20, n), "season": np.array(["spring", "summer", "autumn"])[np.arange(n) % 3]}, index=ids)
    meta.loc[ids[::5], "doc"] = np.nan                                   # the laboratory value was not measured for every sample
    return X, A, ids, meta
