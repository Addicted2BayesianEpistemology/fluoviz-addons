"""The functional variables of a request, shared by the model add-ons: how a request names the basis, the smoothing and what to do with gaps, ``Variable`` objects from plain arrays, and
the pieces of "pixels" and "Gaussian-process" variables that do not depend on what the model does with them.

Nothing of FluoViz is copied: the domains, bases, smoothing and the freezing of a representation are those of ``fluoviz.analysis.fpca`` and ``fluoviz.analysis.frozen.kinds.fpca``
(imported inside the functions, so that this module imports with numpy alone).  The add-ons pass their own error class and their own sentence for "there is nothing to learn / show".
"""
from __future__ import annotations

import warnings

import numpy as np

PIXELS = "pixels"


# ----------------------------------------------------------------------------- what a request says
def lookup_basis(inp: dict, name: str, general: tuple = ("basis",)) -> dict | None:
    """The basis settings of variable ``name`` as a request can say them: ``input[name]["basis"]``, or, for every variable at once, ``input[key][name]`` / ``input[key]`` (with a ``kind``)
    for each ``key`` of ``general``.  ``None`` when the request says nothing (the add-on then applies its own default)."""
    sub = inp.get(name)
    if isinstance(sub, dict) and isinstance(sub.get("basis"), dict):
        return dict(sub["basis"])
    for key in general:
        b = inp.get(key)
        if isinstance(b, dict):
            if name in b and isinstance(b[name], dict):
                return dict(b[name])
            if "kind" in b:
                return dict(b)
    return None


def gaps_spec(inp: dict, name: str) -> str:
    """What to do with pixels that only some samples have (``input[name]["gaps"]``: ``auto`` = estimate them from the other samples, ``plain`` = every sample on its own pixels)."""
    sub = inp.get(name)
    return "plain" if isinstance(sub, dict) and sub.get("gaps") == "plain" else "auto"


def smoothing_spec(inp: dict, name: str, general: bool = True) -> dict:
    """The smoothing settings of variable ``name`` (``input[name]["smoothing"]``, or with ``general`` also ``input["smoothing"][name]`` / ``input["smoothing"]`` with a ``method``)."""
    sub = inp.get(name)
    if isinstance(sub, dict) and isinstance(sub.get("smoothing"), dict):
        return dict(sub["smoothing"])
    s = inp.get("smoothing") if general else None
    if isinstance(s, dict):
        if name in s and isinstance(s[name], dict):
            return dict(s[name])
        if "method" in s:
            return dict(s)
    return {}


# ----------------------------------------------------------------------------- variables from arrays
def variables_from_arrays(X=None, ex=None, em=None, ids=None, A=None, wl=None, abs_ids=None, input: dict | None = None, min_coverage: float = 0.5, units: str = "") -> list:
    """``fluorescence`` / ``absorbance`` ``Variable`` objects from arrays: ``X[sample, emission, excitation]`` (NaN = masked) and ``A[sample, wavelength]``.  Pre-treatments and
    ``min_coverage`` are read from ``input`` as in the Functional PCA request (``input[name]["transform"]``)."""
    from fluoviz.analysis.fpca.variables import absorbance_variable, fluorescence_variable
    inp = dict(input or {})
    out = []
    if X is not None:
        sub = inp.get("fluorescence") if isinstance(inp.get("fluorescence"), dict) else {}
        ids = list(ids) if ids is not None else [f"S{i + 1:02d}" for i in range(np.asarray(X).shape[0])]
        out.append(fluorescence_variable(X, ex, em, ids, transform=sub.get("transform", "none"), min_coverage=sub.get("min_coverage", min_coverage), units=units, complete=False))
    if A is not None:
        sub = inp.get("absorbance") if isinstance(inp.get("absorbance"), dict) else {}
        aids = list(abs_ids) if abs_ids is not None else (list(ids) if ids is not None else [f"S{i + 1:02d}" for i in range(np.asarray(A).shape[0])])
        out.append(absorbance_variable(wl, A, aids, transform=sub.get("transform", "none"), units="absorbance"))
    return out


# ----------------------------------------------------------------------------- pixels
def centred_pixels(v, error, nothing: str):
    """The observed pixels of variable ``v`` centred on their mean function: ``(Y, ybar, R, obs, w, msq)`` with ``Y`` the treated values ``(n, m)`` (NaN = not observed), ``ybar`` the mean
    function, ``R = Y - ybar``, ``obs`` which pixels are observed, ``w`` the L2 weights of the domain and ``msq`` the mean squared L2 norm of the centred samples (the variance of the
    variable; every sample is averaged over the part of the domain it has).  Raises ``error`` (the add-on's own error class) in plain words; ``nothing`` ends the sentence that says the
    samples are all alike (``"there is nothing for a map to show."``)."""
    dom = v.domain
    Y = np.asarray(v.Y, float)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                  # ("Mean of empty slice": a pixel that no sample has is refused just below, in words)
        ybar = np.nanmean(Y, axis=0)
    if np.isnan(ybar).any():
        raise error(f"Some points of the {v.name} domain are not measured in any sample: raise the share of samples that must have a point, or mask them.")
    R = Y - ybar
    obs = np.isfinite(R)
    w = np.asarray(dom.w, float)
    cov = (obs * w).sum(axis=1)
    if (cov <= 0).any():
        raise error(f"A sample has no measured {v.name} value inside the domain.")
    ss = np.where(obs, R, 0.0) ** 2 @ w / cov                        # squared L2 norm of every centred sample (average over the part of the domain that was measured)
    msq = float(np.mean(ss))
    if not np.isfinite(msq) or msq <= 0:
        raise error(f"The {v.name} of the samples are all the same: {nothing}")
    return Y, ybar, R, obs, w, msq


def pixel_element(v, ybar: np.ndarray):
    """The ``FrozenElement`` of a variable used on its pixels (no basis, no smoothing): the domain, the pre-treatment and the mean function, as the saved model stores them."""
    from fluoviz.analysis.frozen.kinds.fpca import FrozenElement
    dom = v.domain
    m = int(np.asarray(ybar).size)
    return FrozenElement(name=v.name, kind=dom.kind, units=str(v.units or ""), transform=str(v.transform),
                         asinh_scale=float(v.info.get("asinh_scale", float("nan"))) if v.transform == "asinh" and dom.kind == "surface" else float("nan"),
                         axis_names=list(dom.names), axes=[np.asarray(a, float) for a in dom.axes], obs=np.asarray(dom.obs, bool), w=np.asarray(dom.w, float), measure=float(dom.measure),
                         basis={"kind": PIXELS, "K": m}, B=None, prior_lam=None, smoothing={"criterion": "none", "gamma": "none", "sigma2": float("nan"), "sigma_f2": float("nan")},
                         ybar=ybar, cbar=ybar.copy(), offset=np.zeros(m), ife=None, n_pixels_grid=int(np.prod(dom.shape)))


# ----------------------------------------------------------------------------- functional coordinates
def total_variance(Z: np.ndarray, name: str, error, nothing: str) -> float:
    """The variance of a functional variable: the mean squared L2 norm of the centred samples (the rows of ``Z``, coordinates in the L2 frame).  ``error`` / ``nothing`` as above."""
    msq = float(np.mean((np.asarray(Z, float) ** 2).sum(axis=1)))
    if not np.isfinite(msq) or msq <= 0:
        raise error(f"The {name} functions are all the same: {nothing}")
    return msq


def gp_prior_scale(el, Z: np.ndarray, notes: list) -> np.ndarray:
    """``d = sqrt(sigma_f^2 * lambda)``: the prior standard deviation of every coordinate of a Gaussian-process basis (``el``: the fitted element).  When the amplitude was not estimated by
    maximum likelihood it is taken from the variance of the data and ``notes`` says so."""
    lam = np.asarray(el.rep.prior_lam, float)
    sf2 = float(el.fit.sigma_f2)
    if not np.isfinite(sf2) or sf2 <= 0:
        sf2 = float(np.var(Z, axis=0, ddof=1).sum() / lam.sum())
        notes.append("The amplitude of the Gaussian process was not estimated by maximum likelihood: it was taken from the variance of the data.")
    return np.sqrt(sf2 * lam)
