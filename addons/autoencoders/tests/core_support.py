"""Shared helpers of the tests of the numerical core: the example data set as masked EEMs (the scatter bands removed as an unstructured subset of the grid), prepared for the
network in any representation."""
from __future__ import annotations

import numpy as np

GP = {"kind": "gp", "lengths": "auto"}
PIXELS = {"kind": "pixels"}
EX_COARSE = np.arange(240.0, 455.0, 10.0)
EM_COARSE = np.arange(290.0, 602.0, 4.0)


def example_eems(n: int = 40, noise: float = 0.02, ife: bool = False, outliers: bool = False, seed: int = 7, coarse: bool = True, scatter: bool = True):
    """``(X, ex, em, ids, abs_wl, abs_A, data)`` of the example data set (noise-free fluorescence from the generator plus white noise of ``noise`` x the maximum; with ``ife`` the inner-filter
    attenuation of the measured sample is applied, which makes the fluorescence a NONLINEAR function of the concentrations; the scatter-like pixels are NaN for every sample)."""
    from fluoviz.core.synthetic import make_demo_dataset
    kw = {"ex": EX_COARSE, "em": EM_COARSE} if coarse else {}
    d = make_demo_dataset(n_samples=n, seed=seed, outliers=outliers, **kw)
    X = d.X_true.copy()
    if ife:
        for i in range(X.shape[0]):
            a_ex = np.interp(d.ex, d.abs_wl, d.abs_A[i])[None, :]
            a_em = np.interp(d.em, d.abs_wl, d.abs_A[i])[:, None]
            X[i] = X[i] * 10 ** (-0.5 * (a_ex + a_em))
    E, XX = np.meshgrid(d.em, d.ex, indexing="ij")
    mask = (np.abs(E - XX) < 15) | (np.abs(E - 2 * XX) < 12) | (E < XX)
    X = X + np.random.default_rng(1).normal(0, noise * np.nanmax(X), X.shape)
    if scatter:
        X[:, mask] = np.nan
    return X, d.ex, d.em, d.sample_ids, d.abs_wl, d.abs_A[:-1], d


def prepared(inp: dict, n: int = 40, absorbance: bool = False, seed: int = 0, **kw):
    """``core.data.Prepared`` of the example data: ``inp`` is the ``input`` of a request (``{"fluorescence": {"basis": ...}, ...}``)."""
    from fluoviz_addon_autoencoders.core import data as D
    X, ex, em, ids, wl, A, _ = example_eems(n=n, **kw)
    variables = D.variables_from_arrays(X=X, ex=ex, em=em, ids=ids, A=A if absorbance else None, wl=wl, abs_ids=ids, input=inp)
    return D.prepare(variables, inp, seed=seed)


def planted(n: int = 40, P: int = 30, k: int = 2, noise: float = 0.02, seed: int = 0, nonlinear: float = 0.0):
    """An array of ``n`` samples with exactly ``k`` latent factors (plus ``noise``); with ``nonlinear > 0`` the second factor enters through a sine (a curved manifold)."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, k))
    W = rng.standard_normal((k, P))
    lin = Z @ W
    if nonlinear > 0 and k >= 2:
        lin = lin + nonlinear * np.sin(2.5 * Z[:, [1]]) @ rng.standard_normal((1, P))
    return lin + noise * rng.standard_normal((n, P)), Z


_CACHE: dict = {}


def cached(key, builder):
    """The result of ``builder()`` computed once per test session (the trained networks are shared by the test modules)."""
    if key not in _CACHE:
        _CACHE[key] = builder()
    return _CACHE[key]


def load_factor() -> float:
    """How overloaded the computer is (1 = idle): the speed tests allow this much extra time, because other programs share the machine."""
    import os
    try:
        return max(1.0, os.getloadavg()[0] / float(os.cpu_count() or 1))
    except OSError:
        return 1.0
