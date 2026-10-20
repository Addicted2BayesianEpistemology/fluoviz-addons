"""From the samples of a FluoViz project to the coordinates a map works on, and back to functions (SPEC sections 2 and 3).

``prepare(variables, input)`` takes the functional variables built by FluoViz (``fluoviz.analysis.fpca.fluorescence_variable`` / ``absorbance_variable``: the domain, the pre-treatment,
NaN = not observed) and the ``input`` of a request (the Functional PCA vocabulary plus ``weights``), and returns a :class:`Prepared`:

* ``R`` ``(n, P)``  the **raw coordinates**, per variable one *block*:
  * functional (``gp`` / ``bspline`` / ``cosine``): the frame coordinates ``Z = C·R_frame`` of the centred smoothed coefficients; the Euclidean distance of two rows is the L2 distance
    between the smooth functions;
  * pixels: ``sqrt(q_j)·(y_j − ȳ_j)`` with ``q_j`` the quadrature weight of the domain (the weights add up to 1, so the squared norm is the mean over the domain); NaN = not observed;
  * variables (a map of arbitrary per-sample variables): the centred columns, NaN = missing.
* the block weights ``w_v`` of the L2 metric, the quadrature weights ``q`` of every coordinate, the prior scales of the Gaussian-process coordinates (for the ``prior`` metric);
* :class:`SomFrame` = everything that turns functions into coordinates and back (the representation and the smoothing parameters FIXED at fit time: a new sample is smoothed like a
  training sample; the saved model keeps it).

Nothing of FluoViz is copied: the domains, bases, smoothing and the freezing of a representation are those of ``fluoviz.analysis.fpca`` and ``fluoviz.analysis.frozen.kinds.fpca``.
The pieces it shares with the autoencoders add-on are in the ``common`` add-on (SPEC section 12).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from fluoviz_addon_common import functional as F
from fluoviz_addon_common.functional import variables_from_arrays  # noqa: F401  (part of this module's interface)
from fluoviz_addon_common.serial import allow as _serial_allow

from .spec import SomError

PIXELS = F.PIXELS
VARIABLES = "variables"
NAMES = ("fluorescence", "absorbance")


# ----------------------------------------------------------------------------- functions <-> coordinates
@_serial_allow
@dataclass
class SomFrameBlock:
    """One variable: how a function becomes ``K`` raw coordinates (and back).  ``element`` is a ``FrozenElement`` of the Functional PCA saved kind (the domain, the pre-treatment, the
    representation, the fixed smoothing, the mean function)."""
    name: str
    kind: str                          # gp | bspline | cosine | pixels
    K: int
    element: object                    # fluoviz.analysis.frozen.kinds.fpca.FrozenElement
    variance: float                    # V_v: mean squared norm of the centred training samples (data units squared)
    prior_scale: np.ndarray | None = None      # (K,) sqrt(sigma_f^2 lambda_k) of a Gaussian-process basis

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

    @property
    def sqrt_q(self) -> np.ndarray:
        """Per coordinate: sqrt of the quadrature weight for a pixel block, 1 for a functional one."""
        return np.sqrt(np.asarray(self.domain.w, float)) if self.kind == PIXELS else np.ones(self.K)

    def treat(self, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from fluoviz.analysis.frozen.kinds.fpca import pretreat
        return pretreat(self.element, self.domain, np.asarray(Y, float))

    def coordinates(self, Yt: np.ndarray) -> np.ndarray:
        """Raw coordinates ``(n, K)`` of already treated samples ``Yt`` ``(n, m)`` (NaN = not observed).  Pixels: ``sqrt(q)(y - mean)`` with NaN kept.  Functional: the sample is smoothed
        on ITS OWN observed pixels with the smoothing parameters of the training samples (a part it lacks is estimated as for training samples that had gaps), centred and moved to the L2
        frame."""
        Yt = np.asarray(Yt, float)
        rt = self.runtime()
        if self.kind == PIXELS:
            return (Yt - rt.ybar) * self.sqrt_q
        return rt.rep.to_frame(rt.centred_coefficients(Yt))

    def to_function(self, Rb: np.ndarray) -> np.ndarray:
        """Functions at the observed pixels ``(n, m)`` (units after the pre-treatment) of raw coordinates ``Rb`` ``(n, K)``."""
        Rb = np.atleast_2d(np.asarray(Rb, float))
        rt = self.runtime()
        if self.kind == PIXELS:
            return rt.ybar + Rb / self.sqrt_q
        return rt.rep.evaluate(rt.rep.from_frame(Rb) + rt.cbar)

    def mean_function(self) -> np.ndarray:
        rt = self.runtime()
        return rt.ybar.copy() if self.kind == PIXELS else rt.rep.evaluate(rt.cbar[None, :])[0]

    def __getstate__(self):
        d = dict(self.__dict__)
        d.pop("_rt", None)
        return d


@_serial_allow
@dataclass
class SomFrame:
    """All the blocks: what is fixed when the map is fitted (the saved model keeps it)."""
    blocks: list                       # [SomFrameBlock]

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

    def block(self, name: str) -> SomFrameBlock:
        for b in self.blocks:
            if b.name == name:
                return b
        raise KeyError(name)

    def sqrt_q(self) -> np.ndarray:
        return np.concatenate([b.sqrt_q for b in self.blocks])

    def to_functions(self, R: np.ndarray) -> dict[str, np.ndarray]:
        """``{name: (n, m)}`` the functions (observed pixels, units after the pre-treatment) of raw coordinates ``R`` ``(n, P)``."""
        R = np.atleast_2d(np.asarray(R, float))
        return {b.name: b.to_function(R[:, sl]) for b, sl in zip(self.blocks, self.slices())}

    def mean_functions(self) -> dict[str, np.ndarray]:
        return {b.name: b.mean_function() for b in self.blocks}

    def coordinates(self, values: dict, min_coverage: float = 0.8) -> dict:
        """Raw coordinates of NEW samples from their values ``values[name]`` ``(n, m_domain)`` at the observed pixels of each variable (NaN = not measured).

        Returns ``{"R", "usable" (n,), "why": [[str]], "coverage": (n, n_blocks), "treated": {name: (n, m)}}``; the coordinates of an unusable sample are NaN.  A sample is unusable
        when a variable is missing, its pre-treatment is not defined, or less than ``min_coverage`` of the pixels of the domain are measured."""
        names = self.names()
        missing = [nm for nm in names if nm not in values or values[nm] is None]
        if missing:
            raise SomError(f"The samples lack the {missing[0]}, which the map needs.")
        n = int(np.asarray(values[names[0]]).shape[0])
        usable = np.ones(n, bool)
        why: list[list[str]] = [[] for _ in range(n)]
        cover = np.ones((n, len(names)))
        treated, parts = {}, []
        for j, b in enumerate(self.blocks):
            Y = np.asarray(values[b.name], float)
            dom = b.domain
            if Y.ndim != 2 or Y.shape[1] != dom.m:
                raise SomError(f"The {b.name} of the samples has {Y.shape[-1]} values per sample but the map uses {dom.m} observed points.")
            none = np.isfinite(Y).sum(axis=1) < (3 if b.name == "absorbance" else 1)
            Yt, ok = b.treat(np.where(none[:, None], np.nan, Y))
            cov = (np.isfinite(Yt) * dom.w).sum(axis=1)
            cover[:, j] = cov
            for i in np.flatnonzero(none):
                why[i].append(f"The sample has no {b.name} measured inside the domain of the map.")
            for i in np.flatnonzero(~ok & ~none):
                why[i].append(f"The {b.name} of the sample cannot be pre-treated as the map requires.")
            low = ok & ~none & (cov < min_coverage - 1e-12)
            for i in np.flatnonzero(low):
                why[i].append(f"Only {100 * cov[i]:.0f} % of the {b.name} points the map uses were measured; at least {100 * min_coverage:.0f} % are needed.")
            use = ok & ~none & ~low
            usable &= use
            treated[b.name] = Yt
            rb = np.full((n, b.K), np.nan)
            if use.any():
                rb[use] = b.coordinates(Yt[use])
            parts.append(rb)
        return {"R": np.hstack(parts), "usable": usable, "why": why, "coverage": cover, "treated": treated}


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
    variance: float                    # total variance V_v
    weight: float                      # w_v of the L2 metric
    user_weight: float                 # u_v (1 unless the person gave a weight)
    Y: np.ndarray | None               # (n, m) treated values at the observed pixels (NaN = not observed)
    ybar: np.ndarray | None
    element: object | None = None      # fluoviz.analysis.fpca.fit.Element (functional)
    rep: object | None = None
    smoothing: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    prior_scale: np.ndarray | None = None
    missing: np.ndarray | None = None  # (n,) share of the L2 weight of the domain each sample lacks
    columns: list | None = None        # variables mode: the names of the columns


@dataclass
class Prepared:
    ids: list
    blocks: list                       # [Block]
    R: np.ndarray                      # (n, P) raw coordinates (NaN = not observed)
    frame: SomFrame | None             # None in variables mode
    q: np.ndarray                      # (P,) quadrature weights of the coordinates (1 for functional and variable coordinates)
    input: dict                        # the resolved data settings
    notes: list = field(default_factory=list)
    dropped: list = field(default_factory=list)
    variables: list = field(default_factory=list)
    seed: int = 0
    mode_kind: str = "functional"

    @property
    def n(self) -> int:
        return int(self.R.shape[0])

    @property
    def P(self) -> int:
        return int(self.R.shape[1])

    @property
    def mode(self) -> str:
        if self.mode_kind == VARIABLES:
            return "variables"
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

    def slices(self) -> list[slice]:
        return [b.sl for b in self.blocks]

    def l2_gain(self) -> np.ndarray:
        """(P,) the gain of the plain (block-weighted) L2 metric: ``sqrt(w_v)`` per coordinate."""
        return np.concatenate([np.full(b.K, np.sqrt(b.weight)) for b in self.blocks])

    def unit_gain(self) -> np.ndarray:
        """(P,) ``sqrt(u_v / K_v)``: every variable counts as much as any other whatever its number of coordinates (used by the standardised metrics)."""
        return np.concatenate([np.full(b.K, np.sqrt(b.user_weight / b.K)) for b in self.blocks])

    def prior_scales(self) -> np.ndarray | None:
        if any(b.prior_scale is None for b in self.blocks):
            return None
        return np.concatenate([b.prior_scale for b in self.blocks])

    def sample_weights(self, mode: str = "observed") -> np.ndarray | None:
        """The weight of every sample ``(n,)`` (mean 1), or None.  A sample that lacks part of the domain of a FUNCTIONAL variable has coefficients that were estimated from the pixels it
        has and from the other samples: it is less certain, so it counts in proportion to the share of the domain that was measured (the mean over the functional variables; at least 5 %).
        Pixel variables do not need it (their distance already ignores the missing pixels).  None when every sample has all it needs, or ``mode`` is ``"equal"``."""
        if mode == "equal":
            return None
        miss = [b.missing for b in self.blocks if b.kind not in (PIXELS, VARIABLES) and b.missing is not None]
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

    def to_functions(self, R: np.ndarray) -> dict[str, np.ndarray]:
        if self.frame is None:
            raise SomError("A map of other variables has no functions to draw.")
        return self.frame.to_functions(R)

    def observed(self) -> dict[str, np.ndarray]:
        return {b.name: b.Y for b in self.blocks if b.Y is not None}

    def summary(self) -> dict:
        out = []
        for b in self.blocks:
            d = b.domain
            out.append({"name": b.name, "mode": "variables" if b.kind == VARIABLES else "pixels" if b.kind == PIXELS else "functional", "basis": b.kind, "n_inputs": int(b.K),
                        "n_pixels": int(d.m) if d is not None else None, "units": b.units, "transform": b.transform, "variance": float(b.variance), "weight": float(b.weight),
                        "smoothing": dict(b.smoothing), "missing_fraction": float(1.0 - np.isfinite(b.Y).mean()) if b.Y is not None else 0.0, "notes": list(b.notes)})
        return {"n": self.n, "P": self.P, "mode": self.mode, "blocks": out, "notes": list(self.notes), "dropped": [list(d) for d in self.dropped]}


# ----------------------------------------------------------------------------- reading the request
def _basis_spec(inp: dict, name: str) -> dict:
    b = F.lookup_basis(inp, name)
    if b is not None:
        return b
    return {"kind": PIXELS} if inp.get("kind") == PIXELS else {}


_gaps_spec = F.gaps_spec


def _smoothing_spec(inp: dict, name: str) -> dict:
    return F.smoothing_spec(inp, name, general=False)


def _user_weights(inp: dict, names: list[str]) -> dict:
    """``(mode, {name: u_v})`` from ``input["weights"]``: ``"variance"`` (default), ``"none"`` or a dict of multipliers."""
    w = inp.get("weights", "variance")
    if isinstance(w, dict):
        bad = [k for k in w if k not in names]
        if bad:
            raise SomError(f"The weight of '{bad[0]}' was given but that variable is not part of the map.")
        out = {}
        for nm in names:
            v = w.get(nm, 1.0)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0:
                raise SomError(f"The weight of the {nm} must be a number from 0 up.")
            out[nm] = float(v)
        if not any(out.values()):
            raise SomError("At least one variable must have a weight above 0.")
        return "custom", out
    if w not in ("variance", "none"):
        raise SomError("The weights between the variables must be 'variance', 'none' or a weight for each variable.")
    return w, {nm: 1.0 for nm in names}


# ----------------------------------------------------------------------------- the blocks
NOTHING = "there is nothing for a map to show."


def _pixel_block(v):
    Y, ybar, Rr, obs, w, msq = F.centred_pixels(v, SomError, NOTHING)
    x = np.where(obs, Rr, np.nan) * np.sqrt(w)
    return x, msq, F.pixel_element(v, ybar), ybar, Y


def _functional_block(el, v, ife):
    from fluoviz.analysis.frozen.kinds.fpca import element_freeze
    Z = np.asarray(el.Z, float)
    msq = F.total_variance(Z, v.name, SomError, NOTHING)                # total variance of the variable: mean squared L2 norm of the centred samples
    notes: list = []
    prior = F.gp_prior_scale(el, Z, notes) if el.rep.kind == "gp" else None
    fe = element_freeze(v.name, el, el, v, ife if v.name == "absorbance" else None)
    return Z, msq, fe, prior, notes


def prepare(variables: list, input: dict | None = None, progress=None, seed: int = 0, ife: dict | None = None) -> Prepared:
    """The coordinates of the samples (see the module text).  ``variables``: ``fluoviz.analysis.fpca.variables.Variable`` objects (fluorescence and / or absorbance; build the
    fluorescence variable with ``complete=False`` so that gaps inside the domain stay NaN).  ``input``: the Functional PCA vocabulary
    ``{"fluorescence": {"basis": {...}, "smoothing": {...}, "gaps": "auto"}, "absorbance": {...}, "weights": "variance"}``; ``basis.kind`` is ``gp`` (default), ``bspline``, ``cosine`` or
    ``pixels`` (``input["kind"] == "pixels"`` says it for every variable).  ``ife``: the clean-up of the absorbance spectra the data used (kept for the saved model)."""
    from fluoviz.analysis.common import Progress
    from fluoviz.analysis.fpca.fit import prepare_element
    from fluoviz.analysis.fpca.pipeline import resolve_basis
    from fluoviz.analysis.fpca.variables import common_samples

    inp = dict(input or {})
    progress = progress or Progress()
    if not variables:
        raise SomError("Choose what to map: the fluorescence, the absorbance, or both.")
    ids = common_samples(variables)
    if len(ids) < 4:
        raise SomError(f"Only {len(ids)} sample(s) have every variable chosen: at least 4 are needed.")
    dropped = [d for v in variables for d in v.dropped]
    variables = [v.subset(ids) for v in variables]
    names = [v.name for v in variables]
    wmode, user = _user_weights(inp, names)
    n = len(ids)
    P_ = len(variables)
    blocks: list[Block] = []
    fblocks: list[SomFrameBlock] = []
    parts, qs, notes = [], [], []
    resolved: dict = {"variables": names, "basis": {}, "smoothing": {}, "transform": {}, "weights": inp.get("weights", "variance")}
    a = 0
    for p, v in enumerate(variables):
        sub = progress.sub(p / P_, (p + 1) / P_)
        bspec = _basis_spec(inp, v.name)
        kind = bspec.get("kind", "gp")
        if kind == PIXELS:
            sub(0.1, f"Preparing the {v.name} pixels")
            x, msq, fe, ybar, Y = _pixel_block(v)
            K = x.shape[1]
            blk_notes = list(v.notes)
            miss = ~np.isfinite(x)
            if miss.any():
                blk_notes.append(f"{100 * float(miss.mean()):.1f} % of the {v.name} values inside the domain are not observed: the distance between a sample and a prototype uses the pixels the sample has.")
            blk = Block(v.name, kind, slice(a, a + K), K, v.domain, str(v.units or ""), v.transform, msq, 1.0, user[v.name], Y, ybar, None, None, {"criterion": "none"}, blk_notes, None,
                        v.missing_share())
            fb = SomFrameBlock(v.name, kind, K, fe, msq, None)
            q = np.asarray(v.domain.w, float)
            resolved["basis"][v.name] = {"kind": PIXELS}
            resolved["smoothing"][v.name] = {}
        else:
            sub(0.0, f"Basis for the {v.name}")
            basis, _est = resolve_basis(v, bspec, sub.sub(0.0, 0.3))
            smoothing = _smoothing_spec(inp, v.name)
            sub(0.3, f"Smoothing the {v.name}")
            el = prepare_element(v.name, v.domain, v.Y, basis, smoothing, v.units, sub.sub(0.3, 1.0), seed, _gaps_spec(inp, v.name))
            x, msq, fe, prior, bn = _functional_block(el, v, ife)
            K = x.shape[1]
            fit = el.fit
            sm = {"criterion": fit.criterion, "theta": {k: val for k, val in (fit.theta or {}).items() if k != "scales"}, "sigma2": float(fit.sigma2), "sigma_f2": float(fit.sigma_f2)}
            blk = Block(v.name, kind, slice(a, a + K), K, v.domain, str(v.units or ""), v.transform, msq, 1.0, user[v.name], v.Y, el.ybar, el, el.rep, sm,
                        list(el.notes) + [t for t in v.notes if not (el.fit.gap is not None and "are missing in some samples" in t)] + bn, prior, v.missing_share())
            fb = SomFrameBlock(v.name, kind, K, fe, msq, prior)
            q = np.ones(K)
            resolved["basis"][v.name] = {k: val for k, val in basis.items()}
            resolved["smoothing"][v.name] = smoothing
        resolved["transform"][v.name] = v.transform
        blocks.append(blk)
        fblocks.append(fb)
        parts.append(x)
        qs.append(q)
        notes += [f"{v.name}: {t}" for t in blk.notes]
        a += K
    for blk in blocks:                                                     # the L2 weights: 1 / V_v ("variance"), 1 ("none") or the person's multipliers in natural units ("custom")
        blk.weight = 1.0 / blk.variance if wmode == "variance" else (1.0 if wmode == "none" else blk.user_weight)
    progress(1.0, "Data prepared")
    return Prepared(list(ids), blocks, np.hstack(parts), SomFrame(fblocks), np.concatenate(qs), resolved, notes, dropped, variables, int(seed))


def prepare_table(columns: dict, ids: list, standardize: bool = True, seed: int = 0) -> Prepared:
    """A map of arbitrary per-sample variables: ``columns`` ``{name: (n,) values}`` (NaN = missing).  The columns are centred; with ``standardize`` (default, the Clustering page's rule)
    they are also divided by their standard deviation (a constant column is only centred)."""
    if not columns:
        raise SomError("Choose at least one variable to map.")
    names = list(columns)
    R = np.column_stack([np.asarray(columns[k], float) for k in names])
    n = R.shape[0]
    if n < 4:
        raise SomError(f"Only {n} sample(s) have the variables chosen: at least 4 are needed.")
    ok = np.isfinite(R).any(axis=0)
    if not ok.all():
        bad = names[int(np.flatnonzero(~ok)[0])]
        raise SomError(f"The variable '{bad}' has no value in any sample.")
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(R, axis=0)
        sd = np.nanstd(R, axis=0, ddof=1)
    R = R - mu
    if standardize:
        R = R / np.where(sd > 0, sd, 1.0)
    P = R.shape[1]
    msq = float(np.mean(np.nansum(R ** 2, axis=1)))
    if not np.isfinite(msq) or msq <= 0:
        raise SomError("The variables chosen are the same in every sample: there is nothing for a map to show.")
    blk = Block("variables", VARIABLES, slice(0, P), P, None, "", "none", msq, 1.0, 1.0, None, None, None, None, {}, [], None, None, names)
    blk.weight = 1.0
    return Prepared(list(ids), [blk], R, None, np.ones(P), {"variables": ["variables"], "keys": names, "standardize": bool(standardize), "weights": "none"}, [], [], [], int(seed), VARIABLES)


# (``variables_from_arrays``, the convenience for tests and scripts, is the shared one imported at the top)
