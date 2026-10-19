"""From the samples of a FluoViz project to the numbers a network works on, and back to functions (SPEC section 3).

``prepare(variables, input)`` takes the functional variables built by FluoViz (``fluoviz.analysis.fpca.fluorescence_variable`` / ``absorbance_variable``: the domain, the
pre-treatment, NaN = not observed) and the ``input`` of a request (the Functional PCA vocabulary), and returns a :class:`Prepared`:

* ``X`` ``(n, P)``  the **network coordinates**, ``M`` ``(n, P)`` which of them are observed (``None`` = all);
* per variable a :class:`Block`: pixel mode (``kind == "pixels"``: ``x = (y - mean function) / s``, gaps allowed, the loss ignores them) or functional mode (``gp`` /
  ``bspline`` / ``cosine``: the smoothed coefficients of ``fluoviz.analysis.fpca`` moved to the L2 frame ``Z`` and standardised ``xi = Z / d``);
* the loss weights ``omega`` (the L2 metric: ``sum_j omega_j (x_j - xhat_j)^2`` is the squared L2 distance between the functions divided by the total variance of the
  variable and averaged over the variables, so ``1 - L`` is the share of the variation rebuilt), and the smoothness metric ``Q`` of every block for ``smoothify``;
* :class:`AEFrame` = everything that turns functions into coordinates and back (the representation and the smoothing parameters FIXED at training: a new sample is smoothed
  like a training sample; the saved model stores it).

Nothing of FluoViz is copied: the domains, bases, smoothing and the freezing of a representation are those of ``fluoviz.analysis.fpca`` and
``fluoviz.analysis.frozen.kinds.fpca``.  The training itself (``train.py``) works on :class:`TrainData`, which is plain numpy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .spec import AutoencoderError, Config

PIXELS = "pixels"
NAMES = ("fluorescence", "absorbance")


def _serial_allow(cls):
    """Let the dataclass be stored in a saved model (``fluoviz.analysis.frozen.serial``); nothing happens when FluoViz is not importable."""
    try:
        from fluoviz.analysis.frozen import serial
        serial.allow(cls)
    except Exception:                                                  # noqa: BLE001 - tests of the numerics run without the saved-model machinery
        pass
    return cls


# ----------------------------------------------------------------------------- what training needs (numpy only: this is what travels to the worker process)
@dataclass
class TrainData:
    X: np.ndarray                      # (n, P) network coordinates (centred, scaled); 0 where not observed
    M: np.ndarray | None               # (n, P) bool observed; None = every coordinate observed
    omega: np.ndarray                  # (P,) loss weights (the L2 metric, block weights included, adding up to 1 over the blocks)
    blocks: list                       # [{"name", "slice": [a, b], "kind", "weight"}]  (as in ModelSpec.blocks)
    q: list                            # per block: {"kind": "diag", "d": (K,)} | {"kind": "dense", "Q": (K, K)} | {"kind": "identity"}; scaled so that the penalty x'Qx/K of the average sample is 1
    sd: np.ndarray                     # (P,) standard deviation of every coordinate over the samples (the unit of the denoising noise)
    ids: list
    info: dict = field(default_factory=dict)
    q_rough: list | None = None        # the smoothness metric of every block for the roughness measures (always the 'smooth' one, whatever ``q`` is)
    sw: np.ndarray | None = None       # (n,) loss weight of every sample (mean 1): samples that lack part of the domain count in proportion to what was measured; None = all alike

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def P(self) -> int:
        return int(self.X.shape[1])

    @classmethod
    def from_array(cls, X: np.ndarray, M: np.ndarray | None = None, ids: list | None = None, name: str = "x", kind: str = "gp") -> "TrainData":
        """Training data from a plain array ``X`` ``(n, P)`` (NaN or ``M`` False = not observed), for scripts and tests: one block, the columns centred, equal loss weights scaled so that
        the total variance is 1, the identity as smoothness metric (unit penalty)."""
        X = np.asarray(X, float)
        n, P = X.shape
        obs = np.isfinite(X) if M is None else (np.isfinite(X) & np.asarray(M, bool))
        cnt = np.maximum(obs.sum(axis=0), 1)
        mean = np.where(obs, X, 0.0).sum(axis=0) / cnt
        Xc = np.where(obs, X - mean, 0.0)
        tot = float((Xc ** 2).sum(axis=1).mean())
        omega = np.full(P, 1.0 / max(tot, 1e-300))
        q = {"kind": "diag", "d": np.full(P, P / max(float((Xc ** 2).sum(axis=1).mean()), 1e-300))}
        return cls(Xc, None if obs.all() else obs, omega, [{"name": name, "slice": [0, P], "kind": kind, "weight": 1.0}], [q], np.maximum(Xc.std(axis=0, ddof=1), 1e-8),
                   list(ids) if ids is not None else [f"S{i + 1:03d}" for i in range(n)], {"names": [name], "mode": "functional"}, [q])

    def block_sizes(self) -> list[int]:
        return [int(b["slice"][1] - b["slice"][0]) for b in self.blocks]

    def sw_of(self, idx=None):
        """The loss weights of the samples ``idx`` (None = all of them), or None when every sample counts alike."""
        if self.sw is None:
            return None
        return np.asarray(self.sw, float) if idx is None else np.asarray(self.sw, float)[np.asarray(idx)]

    def subset(self, idx) -> "TrainData":
        idx = np.asarray(idx)
        return TrainData(self.X[idx], None if self.M is None else self.M[idx], self.omega, self.blocks, self.q, self.sd, [self.ids[i] for i in idx], self.info, self.q_rough,
                         None if self.sw is None else self.sw[idx])


# ----------------------------------------------------------------------------- functions <-> coordinates
@_serial_allow
@dataclass
class AEFrameBlock:
    """One variable: how a function becomes ``K`` coordinates (and back).  ``element`` is a ``FrozenElement`` of the Functional PCA saved kind (the domain, the
    pre-treatment, the representation, the fixed smoothing, the mean function); ``scale`` is ``d`` (coordinates ``x = z / scale``)."""
    name: str
    kind: str                          # gp | bspline | cosine | pixels
    K: int
    scale: np.ndarray                  # (K,)
    element: object                    # fluoviz.analysis.frozen.kinds.fpca.FrozenElement

    def runtime(self):
        rt = getattr(self, "_rt", None)
        if rt is None:
            from fluoviz.analysis.frozen.kinds.fpca import ElementRuntime
            rt = ElementRuntime(self.element)
            self._rt = rt
        return rt

    @property
    def domain(self):
        return self.runtime().dom

    # ------------------------------------------------------------------ functions -> coordinates
    def treat(self, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The pre-treatment of the training samples applied to values ``Y`` ``(n, m)`` at the observed pixels (NaN = not measured): ``(treated, ok)``."""
        from fluoviz.analysis.frozen.kinds.fpca import pretreat
        return pretreat(self.element, self.domain, np.asarray(Y, float))

    def coordinates(self, Yt: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Network coordinates ``(n, K)`` of already treated samples ``Yt`` ``(n, m)`` (NaN = not observed) and the observed mask (None = all observed).
        Pixels: ``(y - mean) / s``, unobserved coordinates 0.  Functional: the sample is smoothed on ITS OWN observed pixels with the smoothing parameters of the training
        samples (a part it lacks is estimated from the pixels it has and from what the other samples show, as for the training samples, when they had gaps), centred, moved to the
        L2 frame and divided by ``scale``."""
        Yt = np.asarray(Yt, float)
        rt = self.runtime()
        if self.kind == PIXELS:
            R = Yt - rt.ybar
            M = np.isfinite(R)
            return np.where(M, R, 0.0) / self.scale, (None if M.all() else M)
        C = rt.centred_coefficients(Yt)
        return rt.rep.to_frame(C) / self.scale, None

    # ------------------------------------------------------------------ coordinates -> functions
    def to_function(self, Xb: np.ndarray) -> np.ndarray:
        """Functions at the observed pixels ``(n, m)`` (in the units after the pre-treatment) of coordinates ``Xb`` ``(n, K)``."""
        Xb = np.atleast_2d(np.asarray(Xb, float))
        rt = self.runtime()
        if self.kind == PIXELS:
            return rt.ybar + Xb * self.scale
        return rt.rep.evaluate(rt.rep.from_frame(Xb * self.scale) + rt.cbar)

    def mean_function(self) -> np.ndarray:
        rt = self.runtime()
        return rt.ybar.copy() if self.kind == PIXELS else rt.rep.evaluate(rt.cbar[None, :])[0]

    def __getstate__(self):
        d = dict(self.__dict__)
        d.pop("_rt", None)
        return d


@_serial_allow
@dataclass
class AEFrame:
    """All the blocks: what is fixed when the network is trained (the saved model keeps it)."""
    blocks: list                       # [AEFrameBlock]

    @property
    def P(self) -> int:
        return int(sum(b.K for b in self.blocks))

    def slices(self) -> list[slice]:
        out, a = [], 0
        for b in self.blocks:
            out.append(slice(a, a + b.K))
            a += b.K
        return out

    def names(self) -> list[str]:
        return [b.name for b in self.blocks]

    def block(self, name: str) -> AEFrameBlock:
        for b in self.blocks:
            if b.name == name:
                return b
        raise KeyError(name)

    def to_functions(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """``{name: (n, m)}`` the functions (observed pixels, units after the pre-treatment) of network coordinates ``X`` ``(n, P)``."""
        X = np.atleast_2d(np.asarray(X, float))
        return {b.name: b.to_function(X[:, sl]) for b, sl in zip(self.blocks, self.slices())}

    def mean_functions(self) -> dict[str, np.ndarray]:
        return {b.name: b.mean_function() for b in self.blocks}

    def coordinates(self, values: dict, min_coverage: float = 0.8) -> dict:
        """Network coordinates of NEW samples from their raw values ``values[name]`` ``(n, m_domain)`` at the observed pixels of each variable (NaN = not measured).

        Returns ``{"X", "M", "usable" (n,), "why": [[str]], "coverage": (n, n_blocks), "treated": {name: (n, m)}}``; the coordinates of an unusable sample are NaN.  A sample is
        unusable when a variable is missing, its pre-treatment is not defined, or less than ``min_coverage`` of the pixels of the domain are measured."""
        names = self.names()
        missing = [nm for nm in names if nm not in values or values[nm] is None]
        if missing:
            raise AutoencoderError(f"The samples lack the {missing[0]}, which the network needs.")
        n = int(np.asarray(values[names[0]]).shape[0])
        usable = np.ones(n, bool)
        why: list[list[str]] = [[] for _ in range(n)]
        cover = np.ones((n, len(names)))
        treated, Xs, Ms = {}, [], []
        for j, b in enumerate(self.blocks):
            Y = np.asarray(values[b.name], float)
            dom = b.domain
            if Y.ndim != 2 or Y.shape[1] != dom.m:
                raise AutoencoderError(f"The {b.name} of the samples has {Y.shape[-1]} values per sample but the network uses {dom.m} observed points.")
            none = np.isfinite(Y).sum(axis=1) < (3 if b.name == "absorbance" else 1)
            Yt, ok = b.treat(np.where(none[:, None], np.nan, Y))
            cov = (np.isfinite(Yt) * dom.w).sum(axis=1)
            cover[:, j] = cov
            for i in np.flatnonzero(none):
                why[i].append(f"The sample has no {b.name} measured inside the domain of the network.")
            for i in np.flatnonzero(~ok & ~none):
                why[i].append(f"The {b.name} of the sample cannot be pre-treated as the network requires.")
            low = ok & ~none & (cov < min_coverage - 1e-12)
            for i in np.flatnonzero(low):
                why[i].append(f"Only {100 * cov[i]:.0f} % of the {b.name} points the network uses were measured; at least {100 * min_coverage:.0f} % are needed.")
            use = ok & ~none & ~low
            usable &= use
            treated[b.name] = Yt
            xb = np.full((n, b.K), np.nan)
            mb = np.ones((n, b.K), bool)
            if use.any():
                xx, mm = b.coordinates(Yt[use])
                xb[use] = xx
                if mm is not None:
                    mb[use] = mm
            Xs.append(xb)
            Ms.append(mb)
        X, M = np.hstack(Xs), np.hstack(Ms)
        return {"X": X, "M": None if M.all() else M, "usable": usable, "why": why, "coverage": cover, "treated": treated}


# ----------------------------------------------------------------------------- one prepared block
@dataclass
class Block:
    name: str
    kind: str
    sl: slice
    K: int
    domain: object
    units: str
    transform: str
    scale: np.ndarray                  # (K,) d
    omega: np.ndarray                  # (K,) loss weights
    variance: float                    # total variance of the variable (data units^2, mean over the domain)
    q_smooth: dict                     # smoothness metric (see TrainData.q) for smoothify_metric "smooth"
    q_ridge: dict                      # ... for "ridge"
    Y: np.ndarray                      # (n, m) treated values at the observed pixels (NaN = not observed)
    ybar: np.ndarray                   # (m,)
    element: object | None = None      # fluoviz.analysis.fpca.fit.Element (functional)
    rep: object | None = None
    smoothing: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    prior_cumulative: float | None = None
    missing: np.ndarray | None = None  # (n,) the share of the L2 weight of the domain each sample lacks (0 = it has every pixel)


@dataclass
class Prepared:
    ids: list
    blocks: list                       # [Block]
    X: np.ndarray
    M: np.ndarray | None
    frame: AEFrame
    input: dict                        # the resolved data settings (basis and smoothing of every variable)
    sd: np.ndarray
    notes: list = field(default_factory=list)
    dropped: list = field(default_factory=list)
    variables: list = field(default_factory=list)
    seed: int = 0

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def P(self) -> int:
        return int(self.X.shape[1])

    @property
    def mode(self) -> str:
        kinds = {b.kind == PIXELS for b in self.blocks}
        return "pixels" if kinds == {True} else "functional" if kinds == {False} else "mixed"

    @property
    def names(self) -> list[str]:
        return [b.name for b in self.blocks]

    def block(self, name: str) -> Block:
        for b in self.blocks:
            if b.name == name:
                return b
        raise KeyError(name)

    # ------------------------------------------------------------------ for training
    def train_data(self, smoothify_metric: str = "smooth", gap_weighting: str = "observed") -> TrainData:
        omega = np.concatenate([b.omega for b in self.blocks])
        q = [b.q_ridge if smoothify_metric == "ridge" else b.q_smooth for b in self.blocks]
        return TrainData(self.X, self.M, omega, self.model_blocks(), q, self.sd, list(self.ids),
                         {"names": self.names, "mode": self.mode, "n_pixels": {b.name: int(b.domain.m) for b in self.blocks},
                          "grid": {b.name: {"shape": [int(v) for v in b.domain.shape], "idx": np.asarray(b.domain.idx), "kind": b.kind} for b in self.blocks}},
                         [b.q_smooth for b in self.blocks], self.sample_weights(gap_weighting))

    def sample_weights(self, mode: str = "observed") -> np.ndarray | None:
        """The loss weight of every sample ``(n,)`` (mean 1), or None.  A sample that lacks part of the domain of a FUNCTIONAL variable (a rectangle excised from it) has coefficients that were
        estimated from the pixels it has and from the other samples: it is less certain, so it counts in proportion to the share of the domain that was measured (the mean over the functional
        variables; at least 5 %).  Pixel variables do not need it (their loss already ignores the missing pixels).  None when every sample has all it needs, or ``mode`` is ``"equal"``."""
        if mode == "equal":
            return None
        miss = [b.missing for b in self.blocks if b.kind != PIXELS and b.missing is not None]
        if not miss:
            return None
        m = np.mean(miss, axis=0)
        if not (m >= 0.005).any():
            return None
        w = np.clip(1.0 - m, 0.05, 1.0)
        return w / float(w.mean())

    def missing_part(self) -> np.ndarray | None:
        """The share of the domain every sample lacks ``(n,)`` (the mean over the variables), or None when no sample lacks anything worth mentioning (less than half a per cent)."""
        miss = [b.missing for b in self.blocks if b.missing is not None]
        if not miss:
            return None
        m = np.mean(miss, axis=0)
        return m if (m >= 0.005).any() else None

    def model_blocks(self) -> list[dict]:
        return [{"name": b.name, "slice": [int(b.sl.start), int(b.sl.stop)], "kind": b.kind, "weight": 1.0} for b in self.blocks]

    # ------------------------------------------------------------------ functions
    def to_functions(self, Xhat: np.ndarray) -> dict[str, np.ndarray]:
        """``{name: (n, m)}`` the functions at the observed pixels (units after the pre-treatment) of network coordinates ``Xhat`` ``(n, P)``."""
        return self.frame.to_functions(Xhat)

    def observed(self) -> dict[str, np.ndarray]:
        """``{name: (n, m)}`` the treated values that were measured (NaN = not observed)."""
        return {b.name: b.Y for b in self.blocks}

    def smoothed(self) -> dict[str, np.ndarray]:
        """``{name: (n, m)}`` the functions the network is asked to rebuild: the smoothed functions (functional) or the measured values (pixels)."""
        return {b.name: (b.Y if b.element is None else b.element.function()) for b in self.blocks}

    def mean_functions(self) -> dict[str, np.ndarray]:
        return self.frame.mean_functions()

    def embed(self, name: str, F: np.ndarray) -> np.ndarray:
        """Values at the observed pixels ``(..., m)`` -> the grid ``(..., n_em, n_ex)`` (surface) or ``(..., n_wl)`` (curve), NaN outside the domain."""
        return self.block(name).domain.embed(F)

    def summary(self) -> dict:
        """Facts for the page (the ``check`` step): one entry per variable."""
        out = []
        for b in self.blocks:
            d = b.domain
            out.append({"name": b.name, "mode": "pixels" if b.kind == PIXELS else "functional", "basis": b.kind, "n_inputs": int(b.K), "n_pixels": int(d.m),
                        "n_grid": int(np.prod(d.shape)), "units": b.units, "transform": b.transform, "variance": float(b.variance), "smoothing": dict(b.smoothing),
                        "missing_fraction": float(1.0 - np.isfinite(b.Y).mean()), "notes": list(b.notes)})
        return {"n": self.n, "P": self.P, "mode": self.mode, "blocks": out, "notes": list(self.notes), "dropped": [list(d) for d in self.dropped]}


# ----------------------------------------------------------------------------- reading the request
def _basis_spec(inp: dict, name: str) -> dict:
    """The basis settings of variable ``name`` in any of the ways a request can say it (``input[name]["basis"]``, ``input["basis"][name]``, ``input["representation"]``)."""
    sub = inp.get(name)
    if isinstance(sub, dict) and isinstance(sub.get("basis"), dict):
        return dict(sub["basis"])
    b = inp.get("basis")
    if isinstance(b, dict):
        if name in b and isinstance(b[name], dict):
            return dict(b[name])
        if "kind" in b:
            return dict(b)
    r = inp.get("representation")
    if isinstance(r, dict):
        if name in r and isinstance(r[name], dict):
            return dict(r[name])
        if "kind" in r:
            return dict(r)
    if inp.get("functional") is False:
        return {"kind": PIXELS}
    return {}


def _gaps_spec(inp: dict, name: str) -> str:
    """What to do with pixels that only some samples have (``input[name]["gaps"]``: ``auto`` = estimate them from the other samples, ``plain`` = every sample on its own pixels)."""
    sub = inp.get(name)
    return "plain" if isinstance(sub, dict) and sub.get("gaps") == "plain" else "auto"


def _smoothing_spec(inp: dict, name: str) -> dict:
    sub = inp.get(name)
    if isinstance(sub, dict) and isinstance(sub.get("smoothing"), dict):
        return dict(sub["smoothing"])
    s = inp.get("smoothing")
    if isinstance(s, dict):
        if name in s and isinstance(s[name], dict):
            return dict(s[name])
        if "method" in s:
            return dict(s)
    return {}


# ----------------------------------------------------------------------------- the blocks
def _normalise_q(Q: np.ndarray) -> np.ndarray:
    Q = (Q + Q.T) / 2.0
    tr = float(np.trace(Q))
    if not np.isfinite(tr) or tr <= 0:
        return np.eye(Q.shape[0])
    return Q * (Q.shape[0] / tr)


def _penalty(q: dict, X: np.ndarray) -> np.ndarray:
    """``x' Q x / K`` for every row of ``X`` ``(n, K)``."""
    K = X.shape[1]
    if q["kind"] == "identity":
        return (X ** 2).sum(axis=1) / K
    if q["kind"] == "diag":
        return ((X ** 2) * q["d"]).sum(axis=1) / K
    return ((X @ q["Q"]) * X).sum(axis=1) / K


def _unit_q(q: dict, X: np.ndarray) -> dict:
    """The smoothness metric scaled so that the penalty ``x' Q x / K`` of the average sample of the data is 1.  Then ``smoothify = 1`` weighs the penalty of an average sample as much
    as the reconstruction loss of the mean (which is 1), whatever the basis; the roughness numbers read "1 = as rough as a typical sample".  (The shape of ``Q`` is the representation's own:
    the identity for the Gaussian-process basis, the P-spline penalty, the cosine shrinkage; only its overall scale is set here.)"""
    K = X.shape[1]
    if q["kind"] == "identity":
        q = {"kind": "diag", "d": np.ones(K)}
    m = float(np.mean(_penalty(q, X)))
    c = 1.0 / m if np.isfinite(m) and m > 0 else 1.0
    return {"kind": "diag", "d": q["d"] * c} if q["kind"] == "diag" else {"kind": "dense", "Q": q["Q"] * c}


def _pixel_block(v, seed: int) -> tuple:
    from fluoviz.analysis.frozen.kinds.fpca import FrozenElement
    dom = v.domain
    Y = np.asarray(v.Y, float)
    n, m = Y.shape
    with np.errstate(invalid="ignore"):
        ybar = np.nanmean(Y, axis=0)
    if np.isnan(ybar).any():
        raise AutoencoderError(f"Some points of the {v.name} domain are not measured in any sample: raise the share of samples that must have a point, or mask them.")
    R = Y - ybar
    obs = np.isfinite(R)
    w = dom.w
    cov = (obs * w).sum(axis=1)
    if (cov <= 0).any():
        raise AutoencoderError(f"A sample has no measured {v.name} value inside the domain.")
    ss = np.where(obs, R, 0.0) ** 2 @ w / cov                        # squared L2 norm of every centred sample (average over the part of the domain that was measured)
    s = float(np.sqrt(np.mean(ss)))
    if not np.isfinite(s) or s <= 0:
        raise AutoencoderError(f"The {v.name} of the samples are all the same: there is nothing for a network to learn.")
    x = np.where(obs, R, 0.0) / s
    scale = np.full(m, s)
    fe = FrozenElement(name=v.name, kind=dom.kind, units=str(v.units or ""), transform=str(v.transform),
                       asinh_scale=float(v.info.get("asinh_scale", float("nan"))) if v.transform == "asinh" and dom.kind == "surface" else float("nan"),
                       axis_names=list(dom.names), axes=[np.asarray(a, float) for a in dom.axes], obs=np.asarray(dom.obs, bool), w=np.asarray(dom.w, float), measure=float(dom.measure),
                       basis={"kind": PIXELS, "K": int(m)}, B=None, prior_lam=None, smoothing={"criterion": "none", "gamma": "none", "sigma2": float("nan"), "sigma_f2": float("nan")},
                       ybar=ybar, cbar=ybar.copy(), offset=np.zeros(m), ife=None, n_pixels_grid=int(np.prod(dom.shape)))
    return x, (None if obs.all() else obs), scale, w.copy(), s * s, fe, ybar, Y


def _functional_block(el, v, ife):
    """Coordinates of a functional element: ``xi = Z / d``; the weights and the smoothness metrics."""
    from fluoviz.analysis.frozen.kinds.fpca import element_freeze
    rep = el.rep
    Z = np.asarray(el.Z, float)
    K = Z.shape[1]
    msq = float(np.mean((Z ** 2).sum(axis=1)))                           # total variance of the variable: mean squared L2 norm of the centred samples
    if not np.isfinite(msq) or msq <= 0:
        raise AutoencoderError(f"The {v.name} functions are all the same: there is nothing for a network to learn.")
    notes: list = []
    cumulative = None
    if rep.kind == "gp":
        lam = np.asarray(rep.prior_lam, float)
        sf2 = float(el.fit.sigma_f2)
        if not np.isfinite(sf2) or sf2 <= 0:
            sf2 = float(np.var(Z, axis=0, ddof=1).sum() / lam.sum())
            notes.append("The amplitude of the Gaussian process was not estimated by maximum likelihood: it was taken from the variance of the data.")
        d = np.sqrt(sf2 * lam)
        omega = d ** 2 / msq
        q_smooth = {"kind": "identity"}
        q_ridge = {"kind": "diag", "d": d ** 2}
        cumulative = float(rep.kl.cumulative[-1]) if rep.kl is not None else None
    else:
        d = np.full(K, np.sqrt(msq))
        omega = np.ones(K)
        q_ridge = {"kind": "identity"}
        Qr = _roughness_q(el)
        q_smooth = {"kind": "dense", "Q": (Qr + Qr.T) / 2.0}
    Xb = Z / d
    q_smooth, q_ridge = _unit_q(q_smooth, Xb), _unit_q(q_ridge, Xb)
    fe = element_freeze(v.name, el, el, v, ife if v.name == "absorbance" else None)
    return Z / d, d, omega, msq, fe, q_smooth, q_ridge, notes, cumulative


def curvature_operator(domain):
    """Sparse matrix ``A`` ``(rows, m)`` of the second differences (divided by the squared grid step: second derivatives in the physical units of the axes) of a function at the observed
    pixels of ``domain``, along every axis, wherever three neighbouring grid points are all observed.  ``||A f||^2`` is the discrete form of ``integral (f_xx^2 + f_yy^2)``: the
    roughness the P-spline penalty approximates, here measured on the function itself (so that it does not depend on how the basis is parametrised)."""
    from scipy import sparse
    m = domain.m
    pos = np.full(int(np.prod(domain.shape)), -1, dtype=np.int64)
    pos[domain.idx] = np.arange(m)
    gi = domain.grid_index()
    rows, cols, vals = [], [], []
    r0 = 0
    for ax in range(domain.d):
        axis = domain.axes[ax]
        h = float(np.median(np.diff(axis))) if axis.size > 1 else 1.0
        stride = int(np.prod(domain.shape[ax + 1:])) if ax + 1 < domain.d else 1
        i = gi[ax]
        ok = (i > 0) & (i < axis.size - 1)
        p = np.flatnonzero(ok)
        flat = domain.idx[p]
        lo, hi = pos[flat - stride], pos[flat + stride]
        good = (lo >= 0) & (hi >= 0)
        p, lo, hi = p[good], lo[good], hi[good]
        k = np.arange(p.size) + r0
        rows += [k, k, k]
        cols += [lo, p, hi]
        vals += [np.full(p.size, 1.0 / h ** 2), np.full(p.size, -2.0 / h ** 2), np.full(p.size, 1.0 / h ** 2)]
        r0 += p.size
    if r0 == 0:
        return sparse.csr_matrix((1, m))
    return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(r0, m))


def _roughness_q(el) -> np.ndarray:
    """The smoothness metric of a B-spline / cosine representation in its L2 frame: ``Q = T' A'A T`` with ``T = B R^-1`` (the functions of the unit frame coordinates at the pixels)
    and ``A`` the second-difference operator of the domain (:func:`curvature_operator`)."""
    rep = el.rep
    _, Rinv = rep.frame()
    T = rep.B if Rinv is None else rep.B @ Rinv
    A = curvature_operator(el.domain)
    AT = A @ T
    return np.asarray(AT.T @ AT)


# ----------------------------------------------------------------------------- the whole preparation
def prepare(variables: list, input: dict | None = None, progress=None, seed: int = 0, ife: dict | None = None) -> Prepared:
    """The network coordinates of the samples.

    ``variables``: ``fluoviz.analysis.fpca.variables.Variable`` objects (fluorescence and / or absorbance), as the Functional PCA builds them (for a ``pixels`` block build the
    fluorescence variable with ``complete=False`` so that gaps inside the domain stay NaN).  ``input``: the Functional PCA vocabulary, in any of the forms
    ``{"fluorescence": {"basis": {...}, "smoothing": {...}}}`` or ``{"basis": {"fluorescence": {...}}, "smoothing": {...}}``, or ``{"representation": {"kind": "pixels"}}`` for every
    variable; ``basis.kind`` is ``gp`` (default, with ``lengths: "auto" | "ml" | [..]``), ``bspline``, ``cosine`` or ``pixels``.  ``ife``: the clean-up of the absorbance spectra
    the data used (kept for the saved model).  ``progress``: a ``fluoviz.analysis.common.Progress``."""
    from fluoviz.analysis.common import Progress
    from fluoviz.analysis.fpca.fit import prepare_element
    from fluoviz.analysis.fpca.pipeline import resolve_basis
    from fluoviz.analysis.fpca.variables import common_samples

    inp = dict(input or {})
    progress = progress or Progress()
    if not variables:
        raise AutoencoderError("Choose what to analyse: the fluorescence, the absorbance, or both.")
    ids = common_samples(variables)
    if len(ids) < 4:
        raise AutoencoderError(f"Only {len(ids)} sample(s) have every variable chosen: at least 4 are needed.")
    dropped = [d for v in variables for d in v.dropped]
    variables = [v.subset(ids) for v in variables]
    P_ = len(variables)
    n = len(ids)
    blocks: list[Block] = []
    fblocks: list[AEFrameBlock] = []
    parts_x, parts_m, notes = [], [], []
    resolved: dict = {"variables": [v.name for v in variables], "basis": {}, "smoothing": {}, "transform": {}}
    a = 0
    for p, v in enumerate(variables):
        sub = progress.sub(p / P_, (p + 1) / P_)
        tr_want = (inp.get(v.name) or {}).get("transform") if isinstance(inp.get(v.name), dict) else None
        if isinstance(inp.get("transform"), dict):
            tr_want = inp["transform"].get(v.name, tr_want)
        if tr_want is not None and tr_want != v.transform:
            raise AutoencoderError(f"The {v.name} was prepared with the pre-treatment '{v.transform}' but the request asks for '{tr_want}'.")
        bspec = _basis_spec(inp, v.name)
        kind = bspec.get("kind", "gp")
        if kind == PIXELS:
            sub(0.1, f"Preparing the {v.name} pixels")
            x, M, scale, omega_w, msq, fe, ybar, Y = _pixel_block(v, seed)
            K = x.shape[1]
            blk_notes = list(v.notes)
            if M is not None:
                blk_notes.append(f"{100 * (1 - float(M.mean())):.1f} % of the {v.name} values inside the domain are not observed: the network ignores them in the loss.")
            q_pix = _unit_q({"kind": "identity"}, x)
            blk = Block(v.name, kind, slice(a, a + K), K, v.domain, str(v.units or ""), v.transform, scale, omega_w, msq, q_pix, q_pix, Y, ybar,
                        None, None, {"criterion": "none"}, blk_notes, None, v.missing_share())
            resolved["basis"][v.name] = {"kind": PIXELS}
            resolved["smoothing"][v.name] = {}
            fb = AEFrameBlock(v.name, kind, K, scale, fe)
        else:
            sub(0.0, f"Basis for the {v.name}")
            basis, _est = resolve_basis(v, bspec, sub.sub(0.0, 0.3))
            smoothing = _smoothing_spec(inp, v.name)
            sub(0.3, f"Smoothing the {v.name}")
            el = prepare_element(v.name, v.domain, v.Y, basis, smoothing, v.units, sub.sub(0.3, 1.0), seed, _gaps_spec(inp, v.name))
            x, d, omega_w, msq, fe, q_smooth, q_ridge, bn, cum = _functional_block(el, v, ife)
            K = x.shape[1]
            M = None
            fit = el.fit
            sm = {"criterion": fit.criterion, "theta": {k: val for k, val in (fit.theta or {}).items() if k != "scales"}, "sigma2": float(fit.sigma2), "sigma_f2": float(fit.sigma_f2)}
            blk = Block(v.name, kind, slice(a, a + K), K, v.domain, str(v.units or ""), v.transform, d, omega_w, msq, q_smooth, q_ridge, v.Y, el.ybar, el, el.rep, sm,
                        list(el.notes) + [t for t in v.notes if not (el.fit.gap is not None and "are missing in some samples" in t)] + bn, cum, v.missing_share())
            resolved["basis"][v.name] = {k: val for k, val in basis.items()}
            resolved["smoothing"][v.name] = smoothing
            fb = AEFrameBlock(v.name, kind, K, d, fe)
        resolved["transform"][v.name] = v.transform
        blocks.append(blk)
        fblocks.append(fb)
        parts_x.append(x)
        parts_m.append(np.ones_like(x, bool) if M is None else M)
        notes += [f"{v.name}: {t}" for t in blk.notes]
        a += K
    nb = len(blocks)
    for blk in blocks:                                                   # the variables count equally: the weights of the blocks add up to 1
        blk.omega = blk.omega / nb
    X = np.hstack(parts_x)
    Mfull = np.hstack(parts_m)
    M_out = None if Mfull.all() else Mfull
    sd = np.maximum(X.std(axis=0, ddof=1) if n > 1 else np.ones(X.shape[1]), 1e-8)
    progress(1.0, "Data prepared")
    return Prepared(list(ids), blocks, X, M_out, AEFrame(fblocks), resolved, sd, notes, dropped, variables, int(seed))


# ----------------------------------------------------------------------------- convenience: from plain arrays (tests, scripts)
def variables_from_arrays(X=None, ex=None, em=None, ids=None, A=None, wl=None, abs_ids=None, input: dict | None = None, min_coverage: float = 0.5, units: str = "") -> list:
    """``fluorescence`` / ``absorbance`` ``Variable`` objects from arrays: ``X[sample, emission, excitation]`` (NaN = masked) and ``A[sample, wavelength]``.  Pre-treatments and
    ``min_coverage`` are read from ``input`` as in the Functional PCA request (``input[name]["transform"]``)."""
    from fluoviz.analysis.fpca.variables import absorbance_variable, fluorescence_variable
    inp = dict(input or {})
    out = []
    if X is not None:
        sub = inp.get("fluorescence") if isinstance(inp.get("fluorescence"), dict) else {}
        pix = _basis_spec(inp, "fluorescence").get("kind") == PIXELS
        ids = list(ids) if ids is not None else [f"S{i + 1:02d}" for i in range(np.asarray(X).shape[0])]
        out.append(fluorescence_variable(X, ex, em, ids, transform=sub.get("transform", "none"), min_coverage=sub.get("min_coverage", min_coverage), units=units, complete=False if pix else False))
    if A is not None:
        sub = inp.get("absorbance") if isinstance(inp.get("absorbance"), dict) else {}
        aids = list(abs_ids) if abs_ids is not None else (list(ids) if ids is not None else [f"S{i + 1:02d}" for i in range(np.asarray(A).shape[0])])
        out.append(absorbance_variable(wl, A, aids, transform=sub.get("transform", "none"), units="absorbance"))
    return out
