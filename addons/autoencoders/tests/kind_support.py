"""Helpers of the kind and API tests (``test_kind_*.py``, ``test_api_*.py``): small synthetic EEMs, a network that needs no PyTorch (the exact linear autoencoder = a weighted PCA, or random
weights), the way a result's ``frozen`` block and a saved model are made from it, and the planted samples.

Nothing here trains with PyTorch: the saved kind and the router are tested with networks whose numbers are known, so these tests run (and mean the same) with or without torch."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fluoviz.analysis.common import Progress
from fluoviz.analysis.frozen import ApplyData, SavedModel, resolve_options, thaw

from fluoviz_addon_autoencoders.core import data as cdata
from fluoviz_addon_autoencoders.core import linear
from fluoviz_addon_autoencoders.core.spec import Config, ModelSpec
from fluoviz_addon_autoencoders.kinds import autoencoder as AE
from fluoviz_addon_autoencoders.kinds import register_standalone
from fluoviz_addon_autoencoders.kinds.freeze import pack_frozen, saved_from_frozen
from fluoviz_addon_autoencoders.kinds.model import AEModel, model_tree

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
    """Absorbance spectra ``(n, 40)`` that depend on the same concentrations (so fluorescence and absorbance belong together)."""
    rng = np.random.default_rng(seed)
    shapes = np.stack([_g(ABS_WL, c, w) for c, w in ((280.0, 30.0), (340.0, 40.0), (420.0, 50.0))], axis=1)
    S = A[:, :3] @ shapes.T * 0.05 + 0.002 * rng.standard_normal((A.shape[0], ABS_WL.size))
    return S + 0.01


def extra_band(X: np.ndarray, size: float = 1.0, em: float = 480.0, ex: float = 290.0) -> np.ndarray:
    """``X`` plus a band that none of the fluorophores has (``size`` x the sample's own maximum): a sample the network does not describe."""
    band = _g(EM[:, None], em, 12.0) * _g(EX[None, :], ex, 10.0)
    return X + size * np.nanmax(X, axis=(-2, -1), keepdims=True) * band


def inputs(kind: str = "bspline", absorbance: bool = False) -> dict:
    """The ``input`` of a request on the small synthetic data."""
    basis = {"pixels": {"kind": "pixels"}, "bspline": {"kind": "bspline", "n_basis": [8, 6]}, "cosine": {"kind": "cosine", "orders": [6, 5]},
             "gp": {"kind": "gp", "lengths": [60.0, 40.0], "k_max": 30}}[kind]
    sm = {"pixels": {"method": "none"}, "bspline": {"method": "gcv"}, "cosine": {"method": "gcv"}, "gp": {"method": "ml"}}[kind]
    out = {"variables": ["fluorescence"], "fluorescence": {"transform": "none", "min_coverage": 0.9, "basis": basis, "smoothing": sm}}
    if absorbance:
        out["variables"] = ["fluorescence", "absorbance"]
        out["absorbance"] = {"transform": "none", "processed": False, "basis": {"kind": "bspline", "n_basis": 12} if kind != "pixels" else {"kind": "pixels"},
                             "smoothing": {"method": "gcv"} if kind != "pixels" else {"method": "none"}}
    return out


def prepare(X, A=None, kind: str = "bspline", ids=None):
    """``core.data.Prepared`` of arrays (fluorescence, and absorbance when ``A`` is given)."""
    inp = inputs(kind, absorbance=A is not None)
    vs = cdata.variables_from_arrays(X=X, ex=EX, em=EM, ids=ids, A=A, wl=ABS_WL if A is not None else None, input=inp)
    return cdata.prepare(vs, inp), inp


# ----------------------------------------------------------------------------- networks that need no torch
def linear_network(prep, d: int = 3, family: str = "ae", seed: int = 0, sigma: float = 0.1):
    """The exact linear autoencoder in the metric of the loss (a weighted PCA of the network coordinates): ``(spec, arrays)``.  A VAE gets the same means and a tiny variance."""
    td = prep.train_data()
    pca = linear.fit_pca(prep.X, prep.M, td.omega, d)
    P = prep.P
    cfg = Config(family=family, latent_dim=d, hidden=[], activation="linear", vae={"sigma": sigma} if family == "vae" else {})
    spec = ModelSpec.from_config(cfg, P, td.blocks, sigma=sigma if family == "vae" else None)
    arrays = {k: np.zeros(shp) for k, shp in spec.array_shapes().items()}
    arrays["enc.mu.W"] = pca.Vw * pca.sw[None, :]
    arrays["dec.out.W"] = (pca.Vw / pca.sw[None, :]).T
    arrays["in.mean"] = pca.mean.copy()
    arrays["in.scale"] = np.ones(P)
    arrays["out.weights"] = td.omega.copy()
    if family == "vae":
        arrays["enc.logvar.b"] = np.full(d, -4.0)
    return spec, arrays


def random_network(prep, d: int = 3, family: str = "ae", hidden=(12, 6), activation: str = "silu", seed: int = 0, prior: str = "normal", K: int = 3):
    """A network with random weights (the numbers mean nothing, they only have to be consistent)."""
    rng = np.random.default_rng(seed)
    td = prep.train_data()
    cfg = Config(family=family, latent_dim=d, hidden=list(hidden), activation=activation, vae={"prior": prior, "n_components": K, "sigma": 0.2} if family == "vae" else {})
    spec = ModelSpec.from_config(cfg, prep.P, td.blocks, sigma=0.2 if family == "vae" else None)
    arrays = {k: rng.normal(0, 0.4, shp) for k, shp in spec.array_shapes().items()}
    arrays["in.mean"] = np.zeros(prep.P)
    arrays["in.scale"] = np.ones(prep.P)
    arrays["out.weights"] = td.omega.copy()
    if spec.has_mixture():
        arrays["prior.pi"] = np.full(K, 1.0 / K)
    return spec, arrays


def make_frozen(prep, spec, arrays, ids=None, heldout=None, validation=None):
    """The ``frozen`` block of a result (what ``payload["frozen"]`` holds) for ``prep`` and a network, with the training samples evaluated as the router does."""
    tree = model_tree(spec, arrays, prep.frame)
    model = AEModel(tree)
    ev = model.evaluate_coordinates(prep.X, prep.M, treated={b.name: b.Y for b in prep.blocks})
    ids = list(ids or prep.ids)
    if heldout is None:
        heldout = np.zeros(len(ids), bool)
        heldout[::4] = True                                              # every 4th sample counts as held out
    ref = AE.reference_for(model, ids, heldout, ev)
    val = validation or {"n_samples": len(ids), "r2_val": float(1 - np.nanmean(ev.error[np.asarray(heldout, bool)]))}
    frozen = pack_frozen(tree, ref, val, AE.description_for(model, val), model)
    return frozen, model, ev


def make_saved(prep, spec, arrays, title: str = "test autoencoder", **kw) -> SavedModel:
    frozen, _, _ = make_frozen(prep, spec, arrays, **kw)
    return saved_from_frozen(frozen, title)


def apply_data(X, A=None, ids=None, meta: dict | None = None, units: str = "RU") -> ApplyData:
    n = X.shape[0]
    ids = list(ids) if ids is not None else [f"N{i:02d}" for i in range(n)]
    df = pd.DataFrame(meta, index=ids) if meta else None
    return ApplyData(ids=ids, X=X, ex=EX, em=EM, A=A, abs_wl=ABS_WL if A is not None else None, meta=df, units=units)


def roundtrip(saved: SavedModel) -> SavedModel:
    """Through the file format: encode -> zip -> read -> decode."""
    return SavedModel.from_zip_bytes(saved.to_zip_bytes())


def run(saved: SavedModel, data: ApplyData, min_coverage: float = 0.8, **opts):
    """Apply a (thawed) saved model with the options resolved like the router does, plus ``min_coverage``."""
    register_standalone()
    ap = thaw(saved)
    return ap.apply(data, {**resolve_options(ap.options(), opts), "min_coverage": min_coverage}, Progress())


def column(out, name: str) -> np.ndarray:
    return np.asarray(out.variables[name], float)
