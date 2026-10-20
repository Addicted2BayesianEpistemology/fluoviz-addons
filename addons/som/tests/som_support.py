"""Helpers of the tests: small synthetic EEMs and absorbance spectra, a ``Prepared`` from arrays, requests."""
from __future__ import annotations

import numpy as np

from fluoviz_addon_som.core import data as cdata

EX = np.linspace(250.0, 400.0, 16)
EM = np.linspace(300.0, 520.0, 24)
ABS_WL = np.linspace(240.0, 500.0, 40)


def _g(x, c, w):
    return np.exp(-0.5 * ((x - c) / w) ** 2)


def below_rayleigh() -> np.ndarray:
    return EM[:, None] < EX[None, :] + 10.0


def small_eem(n: int = 30, k: int = 3, noise: float = 0.01, seed: int = 0, gaps: bool = True):
    """``X`` ``(n, 24, 16)`` = sum of ``k`` fluorophores + noise; the pixels below the Rayleigh line are NaN for every sample.  Returns ``(X, A)`` (``A`` the concentrations)."""
    rng = np.random.default_rng(seed)
    em_c, em_w, ex_c, ex_w = [340.0, 430.0, 480.0, 370.0], [25.0, 35.0, 30.0, 20.0], [275.0, 330.0, 350.0, 290.0], [18.0, 25.0, 22.0, 15.0]
    B = np.stack([_g(EM, em_c[f], em_w[f]) for f in range(k)], axis=1)
    C = np.stack([_g(EX, ex_c[f], ex_w[f]) for f in range(k)], axis=1)
    A = rng.gamma(2.0, 1.0, size=(n, k)) * np.exp(0.3 * rng.standard_normal((n, 1)))
    X = np.einsum("nf,mf,pf->nmp", A, B, C)
    X = X + noise * X.max() * rng.standard_normal(X.shape)
    X = np.maximum(X, 0.0)
    if gaps:
        X[:, below_rayleigh()] = np.nan
    return X, A


def small_abs(A: np.ndarray, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shapes = np.stack([_g(ABS_WL, c, w) for c, w in ((280.0, 30.0), (340.0, 40.0), (420.0, 50.0))], axis=1)
    return A[:, :3] @ shapes.T * 0.05 + 0.002 * rng.standard_normal((A.shape[0], ABS_WL.size)) + 0.01


def inputs(kind: str = "bspline", fluorescence: bool = True, absorbance: bool = False, weights="variance", min_coverage: float = 0.9) -> dict:
    """The ``input`` of a request on the small synthetic data."""
    basis = {"pixels": {"kind": "pixels"}, "bspline": {"kind": "bspline", "n_basis": [8, 6]}, "cosine": {"kind": "cosine", "orders": [6, 5]},
             "gp": {"kind": "gp", "lengths": [60.0, 40.0], "k_max": 30}}[kind]
    sm = {"pixels": {"method": "none"}, "bspline": {"method": "gcv"}, "cosine": {"method": "gcv"}, "gp": {"method": "ml"}}[kind]
    out = {"variables": [], "weights": weights}
    if fluorescence:
        out["variables"].append("fluorescence")
        out["fluorescence"] = {"transform": "none", "min_coverage": min_coverage, "basis": basis, "smoothing": sm}
    if absorbance:
        out["variables"].append("absorbance")
        abasis = {"pixels": {"kind": "pixels"}, "gp": {"kind": "gp", "lengths": [40.0], "k_max": 20}}.get(kind, {"kind": "bspline", "n_basis": 12})
        asm = {"pixels": {"method": "none"}, "gp": {"method": "ml"}}.get(kind, {"method": "gcv"})
        out["absorbance"] = {"transform": "none", "processed": False, "basis": abasis, "smoothing": asm}
    return out


def prepare(X=None, A=None, kind: str = "bspline", ids=None, weights="variance", min_coverage: float = 0.9):
    """``core.data.Prepared`` of arrays (fluorescence when ``X`` is given, absorbance when ``A`` is given)."""
    inp = inputs(kind, fluorescence=X is not None, absorbance=A is not None, weights=weights, min_coverage=min_coverage)
    vs = cdata.variables_from_arrays(X=X, ex=EX, em=EM, ids=ids, A=A, wl=ABS_WL if A is not None else None, input=inp)
    return cdata.prepare(vs, inp), inp


def extra_band(X: np.ndarray, size: float = 1.0, em: float = 480.0, ex: float = 290.0) -> np.ndarray:
    """``X`` plus a band that none of the fluorophores has (``size`` x the sample's own maximum): a sample the map does not describe."""
    band = _g(EM[:, None], em, 12.0) * _g(EX[None, :], ex, 10.0)
    return X + size * np.nanmax(X, axis=(-2, -1), keepdims=True) * band
