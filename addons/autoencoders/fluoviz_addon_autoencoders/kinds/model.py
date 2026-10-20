"""A frozen autoencoder as a numpy object: how a sample becomes network input, the network itself, and how network output becomes a function again.

``meta["model"]`` of the saved kind (and ``payload["frozen"]["model"]`` of a result) is a plain tree (no class of this add-on), so that a model file can be opened without the add-on
being installed (applying it needs the add-on)::

    {"spec": ModelSpec.to_dict(), "arrays": {"enc.W0": ..., "in.mean": ..., "in.scale": ..., "out.weights": ...},
     "frame": {"blocks": [{"name": "fluorescence", "kind": "gp" | "bspline" | "cosine" | "pixels", "K": 87, "scale": (87,), "element": FrozenElement}, ...]}}

* the **element** is the Functional PCA kind's ``FrozenElement`` (domain, pre-treatment, representation, the smoothing parameters the fit chose, mean function): a new sample is smoothed
  in the stored representation on ITS OWN observed pixels with the stored parameters, which are never estimated again; ``scale`` is ``d`` (``xi = Z / d``) of the SPEC;
* ``core.data.Frame`` turns the values of a sample at the pixels of each domain into network coordinates (functional: frame coordinates of the smoothed centred coefficients
  standardised by ``scale``; pixels: ``(y - mean) / s``, unobserved coordinates 0) and back to functions;
* ``core.forward.Network`` is the numpy network (it standardises the coordinates with ``in.mean`` / ``in.scale`` itself).

Pure numpy; plain English in every text a person reads.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fluoviz.analysis.frozen import FrozenError
from fluoviz.analysis.frozen.kinds import _decomp as D
from fluoviz.analysis.frozen.kinds.fpca import FrozenElement, clean_absorbance
from fluoviz_addon_common.kinds import check_grid as _check_grid

from ..core import forward as fw
from ..core import data as _data
from ..core.spec import AutoencoderError, ModelSpec

Frame = getattr(_data, "AEFrame", None) or _data.Frame                   # (the classes of the frame; named like this in core.data)
FrameBlock = getattr(_data, "AEFrameBlock", None) or _data.FrameBlock

NAMES = ("fluorescence", "absorbance")


# ----------------------------------------------------------------------------- the stored tree
def frame_to_tree(frame: Frame) -> dict:
    return {"blocks": [{"name": b.name, "kind": b.kind, "K": int(b.K), "scale": np.asarray(b.scale, float), "element": b.element} for b in frame.blocks]}


def frame_from_tree(tree: dict) -> Frame:
    try:
        return Frame([FrameBlock(str(b["name"]), str(b["kind"]), int(b["K"]), np.asarray(b["scale"], float), b["element"]) for b in tree["blocks"]])
    except (KeyError, TypeError, ValueError):
        raise FrozenError("This model file is damaged: the description of how samples become network input is incomplete.") from None


def model_tree(spec: ModelSpec, arrays: dict, frame: Frame) -> dict:
    """``meta["model"]`` from the pieces (arrays are copied as float64)."""
    return {"spec": spec.to_dict(), "arrays": {k: np.array(v, float) for k, v in spec.check_arrays(arrays).items()}, "frame": frame_to_tree(frame)}


# ----------------------------------------------------------------------------- what an evaluation gives
@dataclass
class Evaluation:
    """New (or training) samples through the whole chain: values -> coordinates -> codes -> reconstruction -> errors.  Rows of unusable samples are NaN."""
    n: int
    usable: np.ndarray                  # (n,) bool
    why: list                           # per sample: plain-English reasons it cannot be used
    coverage: np.ndarray                # (n, B) share of each block's domain (by area) that is measured
    gaps: np.ndarray                    # (n,) usable samples with some unmeasured pixel
    treated: dict                       # name -> (n, m) pre-treated values at the observed pixels (NaN = not measured)
    X: np.ndarray                       # (n, P) network coordinates (NaN rows when unusable)
    M: np.ndarray | None                # (n, P) observed
    codes: np.ndarray                   # (n, d)
    logvar: np.ndarray | None           # (n, d) VAE
    xhat: np.ndarray                    # (n, P)
    error: np.ndarray                   # (n,) share of the typical variation not rebuilt (sum over blocks of the weighted block errors)
    error_blocks: np.ndarray            # (n, B) each block's part of ``error`` (they add up to it)
    rmse: np.ndarray                    # (n,) in the units of the pre-treated data
    rmse_blocks: np.ndarray             # (n, B)
    recon: dict                         # name -> (n, m) reconstruction at the observed pixels
    notes: list = field(default_factory=list)

    @property
    def sd(self) -> np.ndarray | None:
        return None if self.logvar is None else np.exp(0.5 * self.logvar)


class AEModel:
    """The network and its data description, built from ``meta["model"]`` (see the module text)."""

    def __init__(self, tree: dict):
        if not isinstance(tree, dict) or not {"spec", "arrays", "frame"} <= set(tree):
            raise FrozenError("This model file is damaged: its autoencoder is missing.")
        try:
            self.spec = ModelSpec.from_dict(tree["spec"])
            self.net = fw.Network(self.spec, dict(tree["arrays"]))
        except AutoencoderError as exc:
            raise FrozenError(f"This model file is damaged: {exc}") from None
        self.arrays = self.net.a
        self.frame = frame_from_tree(tree["frame"])
        if self.frame.P != self.spec.input_dim:
            raise FrozenError("This model file is damaged: the data description does not fit the network.")
        got = [(b["name"], int(b["slice"][0]), int(b["slice"][1])) for b in self.spec.blocks]
        want = [(b.name, s.start, s.stop) for b, s in zip(self.frame.blocks, self.frame.slices())]
        if got != want:
            raise FrozenError("This model file is damaged: the blocks of the network do not fit its data description.")
        if np.any(np.asarray(self.arrays["in.scale"]) <= 0):
            raise FrozenError("This model file is damaged: a scale of its inputs is not positive.")
        self.P, self.d = int(self.spec.input_dim), int(self.spec.latent_dim)
        self.family = self.spec.family
        self.slices = self.frame.slices()

    # ------------------------------------------------------------------ names
    @property
    def names(self) -> list[str]:
        return self.frame.names()

    @property
    def code_names(self) -> list[str]:
        return [f"z{j + 1}" for j in range(self.d)]

    @property
    def is_vae(self) -> bool:
        return self.family == "vae"

    def block(self, name: str) -> FrameBlock:
        return self.frame.block(name)

    # ------------------------------------------------------------------ from samples to the pixels of the domains
    def values(self, data) -> dict:
        """``{name: (n, m)}`` the values of every sample at the observed pixels of each domain (NaN = not measured), checked against the grid of the model."""
        out = {}
        for b in self.frame.blocks:
            fe: FrozenElement = b.element
            dom = b.domain
            if b.name == "fluorescence":
                X = D.eem_array(data, tuple(int(np.asarray(a).size) for a in fe.axes), "autoencoder")
                _check_grid(fe, (data.em, data.ex), "fluorescence")
                out[b.name] = dom.take(X)
            else:
                if data.A is None:
                    raise FrozenError("This model uses the absorbance of the samples, but no absorbance was given.")
                A = np.asarray(data.A, float)
                if A.ndim != 2 or A.shape[1] != np.asarray(fe.axes[0]).size:
                    raise FrozenError("The absorbance of the samples is not on the wavelength grid of the model. Apply the model through its own data recipe, which puts the samples on the model's grid.")
                _check_grid(fe, (data.abs_wl,), "absorbance")
                if np.isinf(A).any():
                    raise FrozenError("Some absorbance values are infinite; replace them by missing values first.")
                out[b.name] = dom.take(clean_absorbance(fe, A))
        return out

    # ------------------------------------------------------------------ the whole chain
    def evaluate(self, data, min_coverage: float = 0.8) -> Evaluation:
        """Samples (an ``ApplyData``) -> coordinates -> codes -> reconstruction -> errors."""
        n = data.n
        co = self.frame.coordinates(self.values(data), min_coverage)
        usable = np.asarray(co["usable"], bool)
        return self._chain(n, usable, co["why"], co["coverage"], co["treated"], co["X"], co["M"])

    def evaluate_values(self, values: dict, min_coverage: float = 0.0) -> Evaluation:
        """The same from values already at the pixels of the domains (the pre-treatment is applied here): ``{name: (n, m)}``."""
        n = int(np.asarray(values[self.names[0]]).shape[0])
        co = self.frame.coordinates(values, min_coverage)
        return self._chain(n, np.asarray(co["usable"], bool), co["why"], co["coverage"], co["treated"], co["X"], co["M"])

    def evaluate_coordinates(self, X: np.ndarray, M: np.ndarray | None = None, treated: dict | None = None) -> Evaluation:
        """The chain from network coordinates ``X`` ``(n, P)`` (training data)."""
        n = X.shape[0]
        usable = np.isfinite(X).all(axis=1) if M is None else np.ones(n, bool)
        cover = np.ones((n, len(self.frame.blocks)))
        if treated is None:
            treated = {b.name: np.zeros((n, b.domain.m)) for b in self.frame.blocks}
        return self._chain(n, usable, [[] for _ in range(n)], cover, treated, X, M)

    def _chain(self, n, usable, why, coverage, treated, X, M) -> Evaluation:
        d, P, B = self.d, self.P, len(self.frame.blocks)
        codes = np.full((n, d), np.nan)
        logvar = np.full((n, d), np.nan) if self.is_vae else None
        xhat = np.full((n, P), np.nan)
        err = np.full(n, np.nan)
        errb = np.full((n, B), np.nan)
        rmse = np.full(n, np.nan)
        rmseb = np.full((n, B), np.nan)
        recon = {b.name: np.full((n, b.domain.m), np.nan) for b in self.frame.blocks}
        gaps = np.zeros(n, bool)
        for b in self.frame.blocks:
            gaps |= usable & ~np.isfinite(np.asarray(treated[b.name], float)).all(axis=1)
        if usable.any():
            Xu = np.asarray(X, float)[usable]
            Mu = None if M is None else np.asarray(M, bool)[usable]
            mu, lv = self.net.encode(Xu, Mu)
            xh = self.net.decode(mu)
            tot, per = fw.recon_error(self.spec, self.arrays, Xu, xh, Mu)
            funcs = self.frame.to_functions(xh)
            codes[usable] = mu
            if lv is not None:
                logvar[usable] = lv
            xhat[usable] = xh
            err[usable] = tot
            errb[usable] = per
            sq, cnt = np.zeros(int(usable.sum())), np.zeros(int(usable.sum()))
            for k, b in enumerate(self.frame.blocks):
                recon[b.name][usable] = funcs[b.name]
                Yt = np.asarray(treated[b.name], float)[usable]
                r = np.where(np.isfinite(Yt), (Yt - funcs[b.name]) ** 2, np.nan)
                with np.errstate(invalid="ignore"):
                    rmseb[usable, k] = np.sqrt(np.nanmean(r, axis=1)) if r.shape[1] else np.nan
                sq += np.nansum(r, axis=1)
                cnt += np.isfinite(r).sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                rmse[usable] = np.sqrt(sq / np.where(cnt > 0, cnt, np.nan))
        return Evaluation(n, usable, why, np.asarray(coverage, float), gaps, treated, np.asarray(X, float), None if M is None else np.asarray(M, bool), codes, logvar, xhat, err, errb,
                          rmse, rmseb, recon)

    # ------------------------------------------------------------------ decoding
    def decode_functions(self, Z: np.ndarray) -> dict:
        """Codes ``(k, d)`` -> ``{name: (k, m)}`` functions at the observed pixels of each domain."""
        return self.frame.to_functions(self.net.decode(np.atleast_2d(np.asarray(Z, float))))

    def mean_functions(self) -> dict:
        return self.frame.mean_functions()

    def embed(self, name: str, F: np.ndarray) -> np.ndarray:
        return self.block(name).domain.embed(F)

    def weights_of(self, name: str) -> float:
        return float(self.spec.block(name).get("weight", 1.0))
