"""From the samples of a FluoViz project to the *modes* of the multimodal latent factor model, and back (SPEC section 2).

A **mode** is one kind of measurement of every sample: the fluorescence EEM (a surface), the absorbance spectrum (a curve), or a short vector of other numbers (``values``: lab
measurements, indices, scores of another model) that the person wants to model or predict.  For every mode the model needs (mblf, ``FunctionalData``):

* ``Y`` ``(M, N)``  the data at the ``M`` points of the mode (the observed pixels of the domain, or the variables), one column per sample, NaN = not observed.  The data are
  **standardised**: the mean of the training samples is subtracted at every point and the result is divided by ``scale`` (one number for a function: the root mean square of the
  centred samples over the domain; one number per variable for ``values``), so that the default priors of mblf mean the same thing for every mode and every unit;
* ``B`` ``(M, p)``  the basis.  For a function it is the Karhunen-Loeve basis of a Gaussian process on the domain (FluoViz's own, ``fluoviz.analysis.fpca``), each function
  multiplied by the square root of its prior variance (``Phi sqrt(lam)``, the construction of the paper): coefficients of order one then give functions of order one, and the
  number of functions is the smallest that keeps the chosen share of the prior variance (the paper: the share left out is the expected relative truncation error).
  Or (``basis_from = "fpca"``) the eigenfunctions of a functional PCA of the training samples that have the mode (FluoViz's own: the samples are smoothed with the Gaussian-process
  basis by maximum likelihood, missing parts estimated, then the principal components of the smoothed functions), each multiplied by the square root of its eigenvalue in
  standardised units: a data-driven basis with fewer functions for the same variation, built from the same samples the model is then fitted to.  For ``values`` the basis is the
  identity.

Covariates (``Covariates``) are per-sample explanatory variables (metadata): numbers are centred and divided by their standard deviation, a category becomes one centred indicator
per level but the first (the reference).  They shift the mean of the factor scores.

Samples that lack a whole mode stay in the model (their column of that mode is all NaN): the likelihood uses what each sample has, and the model predicts what it lacks.  The
classes of this module are written into saved models (they record their module): do not move them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from fluoviz_addon_common.serial import allow as _serial_allow

from .spec import MblfError

NAMES = ("fluorescence", "absorbance", "values")
LABELS = {"fluorescence": "fluorescence (EEM)", "absorbance": "absorbance spectrum", "values": "values"}
DEFAULT_SHARE = 0.99
K_CAP = {"surface": 40, "curve": 25}
SMOOTH_SHARE = 0.999                   # the Gaussian-process basis that smooths the samples before a functional PCA basis (FluoViz's Functional PCA defaults)
SMOOTH_K = {"surface": 150, "curve": 80}


@_serial_allow
@dataclass
class Mode:
    """One mode of the model; ``element`` (a ``FrozenElement`` of the Functional PCA saved kind, pixels representation) knows the domain and the pre-treatment, so new samples are
    read like the training samples.  ``values`` modes have no element."""
    name: str
    kind: str                          # surface | curve | vector
    B: np.ndarray                      # (M, p) basis at the points of the mode
    mean: np.ndarray                   # (M,) mean of the training samples (units after the pre-treatment)
    scale: np.ndarray                  # (M,) the standardisation: data = mean + scale * standardised
    w: np.ndarray                      # (M,) quadrature weights of the points (sum 1)
    coords: np.ndarray                 # (M, q) coordinates of the points (nm; for values 1..M)
    lam: np.ndarray | None = None      # (p,) prior variances of the Gaussian-process basis functions
    captured: float | None = None      # share of the prior variance the basis keeps
    basis: dict = field(default_factory=dict)          # the resolved basis settings (kernel, lengths ...) and facts
    element: object | None = None      # FrozenElement (surface / curve)
    columns: list = field(default_factory=list)        # values: the variable keys
    column_labels: list = field(default_factory=list)  # values: what a person reads (the variable's own name, without the log)
    units: str = ""
    log: dict | None = None            # functions: {"offset": c}: the model sees log10(value + c) (a value <= -c is not observed); None = as it is
    log_cols: list = field(default_factory=list)       # values: per variable, True = the model sees log10(value) (a value <= 0 is not observed)
    prep: dict = field(default_factory=dict)           # how the data were centred and scaled ({"centre", "scaling"}: see standardise_columns)

    @property
    def M(self) -> int:
        return int(self.B.shape[0])

    @property
    def p(self) -> int:
        return int(self.B.shape[1])

    @property
    def label(self) -> str:
        return LABELS.get(self.name, self.name)

    def gram(self) -> np.ndarray:
        """``B' W B``: the L2 inner products of the basis functions (standardised units)."""
        return self.B.T @ (self.w[:, None] * self.B)

    def frame_root(self) -> np.ndarray:
        """``G^(1/2)`` with ``G = B' W B``: ``frame_root @ coefficients`` has the Euclidean geometry of the L2 geometry of the functions (used to align the factors)."""
        lam, V = np.linalg.eigh(0.5 * (self.gram() + self.gram().T))
        return (V * np.sqrt(np.clip(lam, 0.0, None))) @ V.T

    def forward(self, Y: np.ndarray) -> np.ndarray:
        """``(n, M)`` data after FluoViz's pre-treatment -> the units of the model (``log10`` where asked; a value whose log is not defined becomes NaN = not observed)."""
        Y = np.asarray(Y, float)
        if self.log:
            return log10_or_nan(Y, float(self.log.get("offset", 0.0)))
        if any(self.log_cols or []):
            Y = Y.copy()
            cols = np.flatnonzero(np.asarray(self.log_cols, bool))
            Y[..., cols] = log10_or_nan(Y[..., cols], 0.0)
        return Y

    def backward(self, F: np.ndarray) -> np.ndarray:
        """Units of the model ``(..., M)`` -> the measured units (``10^x - c`` where the model works on a log scale).  Only for single values and quantiles: a mean on the log
        scale is a median (geometric mean) once transformed back."""
        F = np.asarray(F, float)
        if self.log:
            return 10.0 ** F - float(self.log.get("offset", 0.0))
        if any(self.log_cols or []):
            F = F.copy()
            cols = np.flatnonzero(np.asarray(self.log_cols, bool))
            F[..., cols] = 10.0 ** F[..., cols]
        return F

    @property
    def model_labels(self) -> list:
        """values: the name of every variable as the model sees it (``log10 DOC`` for a variable modelled on the log scale)."""
        logs = list(self.log_cols or []) + [False] * (len(self.column_labels) - len(self.log_cols or []))
        return [f"log10 {lab}" if lg else lab for lab, lg in zip(self.column_labels, logs)]

    def standardise(self, Y: np.ndarray) -> np.ndarray:
        """``(n, M)`` data (units after the pre-treatment) -> ``(M, n)`` standardised columns."""
        return ((self.forward(Y) - self.mean) / self.scale).T

    def to_data(self, F: np.ndarray, centred: bool = False) -> np.ndarray:
        """Standardised values ``(..., M)`` -> data units (``centred``: a difference, the mean is not added)."""
        F = np.asarray(F, float)
        return F * self.scale if centred else self.mean + F * self.scale

    def embed(self, f: np.ndarray) -> np.ndarray:
        """Values at the points ``(..., M)`` -> the full grid of the element (NaN outside the domain); values modes are returned as they are."""
        if self.element is None:
            return np.asarray(f, float)
        from fluoviz.analysis.frozen.kinds.fpca import ElementRuntime
        return ElementRuntime(self.element).dom.embed(np.asarray(f, float))


@_serial_allow
@dataclass
class Covariates:
    """The explanatory variables of the scores: ``X`` ``(N, L)`` is the design of the training samples (centred; numbers also standardised)."""
    keys: list                         # the variable keys (meta:site ...)
    labels: list
    columns: list                      # per design column: {"key", "label", "kind": "numeric" | "level", "level", "reference", "mean", "sd"}
    X: np.ndarray

    @property
    def L(self) -> int:
        return int(self.X.shape[1])

    def cov(self) -> np.ndarray:
        """The covariance of the design over the training samples ``(L, L)``."""
        return self.X.T @ self.X / max(self.X.shape[0], 1)

    def design(self, values: dict) -> np.ndarray:
        """The design of NEW samples from ``{key: (n,) values}`` (numbers or category labels; NaN / None = missing: the row is NaN)."""
        n = len(next(iter(values.values()))) if values else 0
        out = np.full((n, self.L), np.nan)
        for j, c in enumerate(self.columns):
            v = values.get(c["key"])
            if v is None:
                raise MblfError(f"The samples lack the covariate {c['label']}, which the model needs.")
            if c["kind"] == "numeric":
                x = np.asarray([np.nan if u is None else u for u in v], float)
                if c.get("log"):
                    x = log10_or_nan(x, 0.0)
                out[:, j] = (x - c["mean"]) / c["sd"]
            else:
                lab = np.array(["" if u is None or (isinstance(u, float) and np.isnan(u)) else str(u) for u in v], dtype=object)
                known = set(c["known"])
                ok = np.array([s in known for s in lab])
                out[:, j] = np.where(ok, (lab == c["level"]).astype(float) - c["mean"], np.nan)
        return out

    def levels(self, key: str) -> tuple[str, list]:
        """A category covariate: ``(reference, [every level, the reference first])``."""
        cols = [c for c in self.columns if c["key"] == key and c["kind"] == "level"]
        if not cols:
            raise MblfError(f"The covariate {key} is not a category of this model.")
        return cols[0]["reference"], [cols[0]["reference"]] + [c["level"] for c in cols]

    def level_rows(self, key: str, level: str) -> np.ndarray:
        """The training samples (rows of ``X``) whose category ``key`` is ``level``: read back from the centred indicators."""
        ref, levels = self.levels(key)
        if level not in levels:
            raise MblfError(f"The covariate {key} has no category '{level}': it has {', '.join(map(str, levels))}.")
        js = [j for j, c in enumerate(self.columns) if c["key"] == key and c["kind"] == "level"]
        ind = np.column_stack([self.X[:, j] + self.columns[j]["mean"] for j in js]) > 0.5
        if level == ref:
            return np.flatnonzero(~ind.any(axis=1))
        return np.flatnonzero(ind[:, [self.columns[j]["level"] for j in js].index(level)])

    def pure(self, key: str, level: str) -> np.ndarray:
        """The design ``(L,)`` of a sample of category ``level`` of ``key`` whose other covariates are at the average of the training samples (0 in the centred design)."""
        self.level_rows(key, level)
        x = np.zeros(self.L)
        for j, c in enumerate(self.columns):
            if c["key"] == key and c["kind"] == "level":
                x[j] = float(c["level"] == level) - c["mean"]
        return x

    def typical(self, key: str, level: str) -> tuple[np.ndarray, int]:
        """The design ``(L,)`` of the typical member of category ``level``: the average design of the training samples of that category (its other covariates at THEIR average
        within the category), and how many there are."""
        rows = self.level_rows(key, level)
        if rows.size == 0:
            raise MblfError(f"No training sample has {key} = {level}.")
        return self.X[rows].mean(axis=0), int(rows.size)


@dataclass
class Prepared:
    """What a fit needs: the modes (standardised data, bases), the covariates, the sample ids, and what was left out."""
    ids: list
    modes: list                        # [Mode]
    Y: list                            # per mode (M, N) standardised data (NaN = not observed)
    cov: Covariates | None
    input: dict
    notes: list = field(default_factory=list)
    dropped: list = field(default_factory=list)          # [(sample id, reason)]

    @property
    def N(self) -> int:
        return len(self.ids)

    @property
    def names(self) -> list[str]:
        return [m.name for m in self.modes]

    def presence(self) -> np.ndarray:
        """``(N, D)`` the sample has at least one observed value of the mode."""
        return np.column_stack([np.isfinite(Y).any(axis=0) for Y in self.Y])

    def summary(self) -> dict:
        pres = self.presence()
        out = []
        for d, (m, Y) in enumerate(zip(self.modes, self.Y)):
            out.append({"name": m.name, "label": m.label, "kind": m.kind, "n_points": m.M, "n_basis": m.p, "captured": m.captured, "basis": dict(m.basis),
                        "n_samples": int(pres[:, d].sum()), "missing_fraction": float(1.0 - np.isfinite(Y[:, pres[:, d]]).mean()) if pres[:, d].any() else 1.0,
                        "units": m.units, "columns": list(m.model_labels if m.kind == "vector" else m.column_labels),
                        "prep": dict(m.prep or {}), "log": dict(m.log) if m.log else None})
        return {"n": self.N, "modes": out, "complete": int(pres.all(axis=1).sum()),
                "covariates": None if self.cov is None else {"keys": list(self.cov.keys), "labels": list(self.cov.labels), "columns": [dict(c) for c in self.cov.columns]},
                "notes": list(self.notes), "dropped": [list(d) for d in self.dropped]}


# ----------------------------------------------------------------------------- one functional mode
CENTRES = {"mean": "the mean of the training samples at every point", "median": "the median of the training samples at every point", "none": "not centred"}
SCALINGS = {"global": "one number: the root mean square of the centred samples over the domain (keeps the shape)",
            "pointwise": "every point divided by its own standard deviation (every point counts the same, noise included)",
            "pareto": "every point divided by the square root of its standard deviation (between the two)",
            "none": "not scaled (the data units; the default priors then assume values of order one)"}


def log10_or_nan(Y: np.ndarray, offset: float) -> np.ndarray:
    """``log10(Y + offset)``, NaN where ``Y + offset <= 0`` (or Y is NaN)."""
    Y = np.asarray(Y, float) + offset
    ok = np.isfinite(Y) & (Y > 0)
    return np.where(ok, np.log10(np.where(ok, Y, 1.0)), np.nan)


def auto_offset(Y: np.ndarray) -> float:
    """The offset of the log of a function when none is given: 1 % of the 99th percentile of the absolute values measured (a value smaller than that is dominated by noise)."""
    a = np.abs(np.asarray(Y, float))
    a = a[np.isfinite(a)]
    q = float(np.quantile(a, 0.99)) if a.size else 1.0
    return 0.01 * q if q > 0 else 1.0


def standardise_columns(Yt: np.ndarray, w: np.ndarray, name: str, centre: str = "mean", scaling: str = "global") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``Yt`` ``(N, M)`` (NaN = not observed; a sample without the mode is a row of NaN) -> ``(centre (M,), Z (M, N), scale (M,))`` with ``Z = (Yt - centre) / scale``.

    ``centre``: ``mean`` (default) / ``median`` of the samples that have each point, or ``none`` (0).  ``scaling``: ``global`` (default: one number, the root mean square of the
    centred samples over the domain, each sample averaged over the part it has), ``pointwise`` (the standard deviation of every point), ``pareto`` (its square root), each made
    so that the standardised data have a root mean square of one, or ``none`` (1: the data units)."""
    import warnings
    has = np.isfinite(Yt).any(axis=1)
    if has.sum() < 2:
        raise MblfError(f"Fewer than 2 samples have the {name}: it cannot be modelled.")
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmedian(Yt[has], axis=0) if centre == "median" else np.nanmean(Yt[has], axis=0)
    if np.isnan(mean).any():
        raise MblfError(f"Some points of the {name} are not measured in any sample: raise the share of samples that must have a point, or mask them.")
    if centre == "none":
        mean = np.zeros_like(mean)
    R = Yt - mean
    obs = np.isfinite(R)

    def rms(Z):
        cov = (obs * w).sum(axis=1)
        ss = np.where(obs, Z, 0.0) ** 2 @ w / np.where(cov > 0, cov, 1.0)
        return float(np.sqrt(np.mean(ss[has])))

    g = rms(R)
    if not np.isfinite(g) or g <= 0:
        raise MblfError(f"The {name} of the samples are all the same: there is nothing for the factors to describe.")
    M = Yt.shape[1]
    if scaling == "none":
        scale = np.ones(M)
    elif scaling in ("pointwise", "pareto"):
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sd = np.sqrt(np.nanmean(R[has] ** 2, axis=0))
        sd = np.maximum(np.nan_to_num(sd, nan=0.0), 1e-3 * g)                     # a point that never varies would be divided by zero
        scale = sd if scaling == "pointwise" else np.sqrt(sd)
        scale = scale * rms(R / scale)                                              # standardised data of root mean square one, like the default
    else:
        scale = np.full(M, g)
    return mean, (R / scale).T, scale


def gp_basis_matrix(Phi: np.ndarray, lam: np.ndarray, total: float, share: float | None) -> tuple[np.ndarray, np.ndarray, float]:
    """``(B, lam_kept, captured)``: the leading functions of a Karhunen-Loeve basis multiplied by the square roots of their prior variances (normalised so that the variances of the
    whole process add up to one), as many as keep ``share`` of the prior variance (all when ``share`` is None)."""
    lam = np.asarray(lam, float)
    total = float(total) if total and np.isfinite(total) and total > 0 else float(lam.sum())
    k = lam.size
    if share is not None:
        k = int(min(lam.size, np.searchsorted(np.cumsum(lam) / total, share - 1e-12) + 1))
    lam_k = lam[:k] / total
    return np.asarray(Phi, float)[:, :k] * np.sqrt(lam_k)[None, :], lam_k, float(lam_k.sum())


def fpca_basis_matrix(eigenfunctions: np.ndarray, eigenvalues: np.ndarray, cumulative: np.ndarray, scale: float, share: float, n_basis: int | None, k_max: int
                      ) -> tuple[np.ndarray, np.ndarray, float]:
    """``(B, lam_kept, captured)`` of a functional PCA basis: ``eigenfunctions`` ``(r, M)`` (L2-orthonormal on the domain), their ``eigenvalues`` (data units squared) and the
    ``cumulative`` share of the variation they hold.  Kept: exactly ``n_basis`` functions, or the fewest that hold ``share`` of the variation, at most ``k_max``; each is
    multiplied by the square root of its eigenvalue in standardised units (``/ scale``), so coefficients of order one give the variation of the samples."""
    r = int(eigenvalues.size)
    if n_basis:
        k = min(int(n_basis), r)
    else:
        k = min(int(np.searchsorted(np.asarray(cumulative, float), share - 1e-12) + 1), int(k_max), r)
    k = max(k, 1)
    lam = np.asarray(eigenvalues, float)[:k] / float(scale) ** 2
    B = np.asarray(eigenfunctions, float)[:k].T * np.sqrt(np.clip(lam, 0.0, None))[None, :]
    return B, lam, float(np.asarray(cumulative, float)[k - 1])


def _fpca_basis(v, have: list, dom, resolved: dict, scale: float, share: float, n_basis, k_max: int, progress, seed: int) -> tuple[np.ndarray, np.ndarray, float, list, int]:
    """The functional PCA basis of a variable from the samples ``have`` (see the module text): ``(B, lam, captured, notes, n_components_available)``.  ``v`` holds the data as
    the model sees them (log, pointwise scaling); ``scale`` the one number that still divides them."""
    from fluoviz.analysis.fpca.fit import fit_fpca, prepare_element
    if len(have) < 4:
        raise MblfError(f"A basis from a functional PCA needs at least 4 samples that have the {v.name}; only {len(have)} have it. Use the Gaussian-process basis (step 3).")
    sub = v.subset(have)
    smooth = {**resolved, "var_fraction": SMOOTH_SHARE, "k_max": max(SMOOTH_K[dom.kind], int(k_max)), "n_basis": None}
    el = prepare_element(v.name, dom, sub.Y, smooth, {"method": "ml"}, v.units, progress, seed, "auto")
    fit = fit_fpca([el], "none", "auto", 0.99, max_components=len(have))
    r = fit.r
    B, lam, captured = fpca_basis_matrix(fit.eigenfunctions(0, r), fit.eigenvalues, fit.cumulative, scale, share, n_basis, k_max)
    notes = list(el.notes)
    k = B.shape[1]
    if n_basis and int(n_basis) > r:
        notes.append(f"A functional PCA of {len(have)} samples has at most {r} components, so the basis has {r} functions, not the {int(n_basis)} asked.")
    elif not n_basis and captured < share - 1e-9:
        notes.append(f"The basis of the {v.name} kept {k} functional components holding {100 * captured:.1f} % of the variation of the smoothed samples, less than the "
                     f"{100 * share:.1f} % asked ({'all the components there are' if k >= r else 'the maximum number of terms'}).")
    notes.append(f"The basis of the {v.name} is data-driven: the first {k} components of a functional PCA of the {len(have)} training samples that have it ({100 * captured:.1f} % of "
                 "their smoothed variation). It is estimated from the same samples the model is fitted to, so the bands are slightly optimistic, and a new sample with a shape none "
                 "of them had is represented only as far as these components allow.")
    return B, lam, captured, notes, r


PREP_DEFAULT = {"centre": "mean", "scaling": "global", "log": False, "log_offset": None}


def functional_mode(v, ids: list, basis_spec: dict | None, progress=None, ife: dict | None = None, basis_from: str = "gp", seed: int = 0, prep: dict | None = None
                    ) -> tuple[Mode, np.ndarray, list]:
    """The mode of a FluoViz ``Variable`` (``fluoviz.analysis.fpca``) on the samples ``ids`` (a sample the variable does not have is a column of NaN).  ``basis_from``: ``gp`` (the
    Gaussian-process prior) or ``fpca`` (a functional PCA of the samples; ``seed`` for its estimation).  ``prep``: ``log`` (the model sees ``log10(value + log_offset)``; the
    offset is 1 % of the 99th percentile of the values when None), ``centre`` and ``scaling`` (``standardise_columns``).  Returns ``(mode, Y, notes)``."""
    from dataclasses import replace

    from fluoviz.analysis.fpca.bases import build_representation
    from fluoviz.analysis.fpca.pipeline import resolve_basis
    from fluoviz_addon_common.functional import pixel_element

    prep = {**PREP_DEFAULT, **(prep or {})}
    dom = v.domain
    pos = {s: i for i, s in enumerate(v.ids)}
    Yt = np.full((len(ids), dom.m), np.nan)
    for k, s in enumerate(ids):
        if s in pos:
            Yt[k] = v.Y[pos[s]]
    w = np.asarray(dom.w, float)
    notes = []
    log = None
    units = str(v.units or "")
    vt = v
    if prep["log"]:
        c = float(prep["log_offset"]) if prep.get("log_offset") is not None else auto_offset(Yt)
        bad = int((np.isfinite(Yt) & (Yt + c <= 0)).sum())
        tot = int(np.isfinite(Yt).sum())
        Yt = log10_or_nan(Yt, c)
        if (~np.isfinite(Yt)).all(axis=1)[np.isin(ids, v.ids)].any():
            raise MblfError(f"With the offset {c:.3g}, the log of the {v.name} of a sample is nowhere defined: raise the offset of the log (step 3).")
        vt = replace(v, Y=log10_or_nan(v.Y, c))
        log = {"offset": c, "auto": prep.get("log_offset") is None}
        units = f"log10({units or 'value'} + {c:.3g})"
        notes.append(f"The model sees log10({v.name} + {c:.3g})" + (" (the offset is 1 % of the 99th percentile of the values)" if log["auto"] else "") +
                     (f"; {bad} of {tot} values ({100 * bad / max(tot, 1):.1f} %) are at or below -{c:.3g}, have no log and are treated as not measured" if bad else "") +
                     (": raise the offset if that is many." if bad > 0.05 * tot else "."))
    mean, Z, scale = standardise_columns(Yt, w, v.name, prep["centre"], prep["scaling"])
    if prep["centre"] != "mean" or prep["scaling"] != "global":
        notes.append(f"The {v.name} is centred on {CENTRES[prep['centre']]} and scaled {SCALINGS[prep['scaling']].split(' (')[0]}.")
    if prep["centre"] == "none":
        notes.append(f"The {v.name} is not centred: the factors and each sample's own part have mean zero, so the mean of the samples has to be carried by them. Use it only for data "
                     "that are already centred.")
    if prep["scaling"] == "none":
        notes.append(f"The {v.name} is not scaled: the default priors assume standardised values of order one (here the root mean square is "
                     f"{float(np.sqrt(np.nanmean(Z ** 2))):.3g}).")
    pointwise = prep["scaling"] in ("pointwise", "pareto")
    vb = replace(vt, Y=vt.Y / scale[None]) if pointwise else vt                    # what the basis is estimated from: the data as the model sees them (up to one number)
    spec = {"kind": "gp", "kernel": "matern52", "structure": "separable", "lengths": "auto", "var_fraction": DEFAULT_SHARE, "k_max": K_CAP[dom.kind], **(basis_spec or {})}
    share = float(spec.get("var_fraction") or DEFAULT_SHARE)
    n_basis = spec.get("n_basis")
    have = [s for s in ids if s in pos]
    sub = vb.subset(have)
    resolved, _ = resolve_basis(sub, spec, progress)
    fe = pixel_element(v, mean)
    if ife is not None and v.name == "absorbance":
        fe.ife = ife
    common = {"kernel": resolved.get("kernel"), "structure": resolved.get("structure"), "lengths": [float(x) for x in resolved.get("lengths", [])],
              "lengths_source": resolved.get("lengths_source"), "share": share}
    pinfo = {"centre": prep["centre"], "scaling": prep["scaling"]}
    if basis_from == "fpca":
        B, lam, captured, bnotes, r = _fpca_basis(vb, have, dom, resolved, 1.0 if pointwise or prep["scaling"] == "none" else float(scale[0]), share, n_basis,
                                                   int(spec.get("k_max") or K_CAP[dom.kind]), progress, seed)
        notes += bnotes
        if prep["centre"] != "mean":                                               # the components describe the samples around THEIR mean: the basis also needs that mean
            with np.errstate(invalid="ignore"):
                zbar = np.nan_to_num(np.nanmean(np.where(np.isfinite(Z), Z, np.nan), axis=1))
            if np.sqrt(np.sum(w * zbar ** 2)) > 1e-6:
                B = np.column_stack([zbar, B])
                lam = np.concatenate([[float(np.sum(w * zbar ** 2))], lam])
                notes.append(f"The basis of the {v.name} also holds the mean of the samples (they are not centred on it).")
        info = {"from": "fpca", **common, "n_basis": int(B.shape[1]), "captured": captured, "n_available": r}
        return Mode(v.name, dom.kind, B, mean, scale, w, dom.coords(), lam, captured, info, fe, [], [], units, log, [], pinfo), Z, notes
    rep = build_representation(dom, {**resolved, "var_fraction": None if n_basis else share, "n_basis": n_basis}, progress)
    if rep.kind != "gp" or rep.prior_lam is None:
        raise MblfError("The multimodal latent factor model needs the Gaussian-process basis (step 3).")
    total = float(rep.params.get("total") or np.sum(rep.prior_lam))
    B, lam, captured = gp_basis_matrix(rep.B, rep.prior_lam, total, None if n_basis else share)
    notes += list(rep.notes)
    if B.shape[1] >= rep.prior_lam.size and not n_basis and captured < share - 1e-9:
        notes.append(f"The basis reached its largest size ({B.shape[1]} functions) and keeps {100 * captured:.1f} % of the prior variance, less than the {100 * share:.1f} % asked: "
                     "raise the largest number of functions, or choose longer length-scales.")
    info = {"from": "gp", **common, "n_basis": int(B.shape[1]), "captured": captured, "approximate": bool(rep.params.get("approximate"))}
    mode = Mode(v.name, dom.kind, B, mean, scale, w, dom.coords(), lam, captured, info, fe, [], [], units, log, [], pinfo)
    return mode, Z, notes


VALUES_BASES = ("identity", "pca")


def values_pca_basis(Z: np.ndarray, share: float, n_basis: int | None) -> tuple[np.ndarray, np.ndarray, float, int]:
    """The principal-component basis of standardised values ``Z`` ``(M, N)`` (NaN = missing): the eigenvectors of their correlation matrix (pairwise complete; made positive
    semi-definite), each times the square root of its eigenvalue; exactly ``n_basis`` of them or the fewest that keep ``share`` of the variance.  ``(B, lam, captured, r)``."""
    M = Z.shape[0]
    C = np.eye(M)
    ok = np.isfinite(Z)
    for i in range(M):
        for j in range(i + 1, M):
            both = ok[i] & ok[j]
            if both.sum() < 3:
                raise MblfError("Two of the values are measured together in fewer than 3 samples: their principal components cannot be estimated. Use one basis function per value.")
            a, b = Z[i, both], Z[j, both]
            C[i, j] = C[j, i] = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0
    lam, V = np.linalg.eigh(C)
    order = np.argsort(lam)[::-1]
    lam, V = np.clip(lam[order], 0.0, None), V[:, order]
    tot = float(lam.sum()) or 1.0
    cum = np.cumsum(lam) / tot
    r = int((lam > 1e-9 * lam[0]).sum())
    k = min(int(n_basis), r) if n_basis else min(int(np.searchsorted(cum, share - 1e-12) + 1), r)
    k = max(k, 1)
    for j in range(k):                                                              # a sign that reads: the largest entry positive
        if V[np.argmax(np.abs(V[:, j])), j] < 0:
            V[:, j] = -V[:, j]
    return V[:, :k] * np.sqrt(lam[:k])[None, :], lam[:k], float(cum[k - 1]), r


def values_mode(columns: dict, labels: dict, ids: list, log_keys: list | None = None, basis: dict | None = None) -> tuple[Mode, np.ndarray, list]:
    """The ``values`` mode: ``columns`` ``{key: (N,) numbers}`` on the samples ``ids`` (NaN = missing).  A variable in ``log_keys`` is modelled as its ``log10`` (a value <= 0 is
    then missing).  Each variable is centred and divided by its standard deviation.  ``basis``: ``{"kind": "identity"}`` (default: every variable is one coefficient) or
    ``{"kind": "pca", "var_fraction", "n_basis"}`` (the principal components of their correlations: fewer coefficients when the values move together)."""
    keys = list(columns)
    if not keys:
        raise MblfError("Choose at least one variable to model.")
    logs = [k in (log_keys or []) for k in keys]
    Yt = np.column_stack([np.asarray(columns[k], float) for k in keys])
    notes = []
    for j, k in enumerate(keys):
        if logs[j]:
            bad = int((np.isfinite(Yt[:, j]) & (Yt[:, j] <= 0)).sum())
            Yt[:, j] = log10_or_nan(Yt[:, j], 0.0)
            if bad:
                notes.append(f"The value {labels.get(k, k)} is modelled on the log scale: {bad} sample(s) with a value of zero or below have no log and count as not measured.")
    has = np.isfinite(Yt)
    for j, k in enumerate(keys):
        if has[:, j].sum() < 3:
            raise MblfError(f"The variable {labels.get(k, k)} has a value in fewer than 3 samples{' (with a log: above zero)' if logs[j] else ''}: it cannot be modelled.")
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(Yt, axis=0)
        sd = np.nanstd(Yt, axis=0, ddof=1)
    if not (sd > 0).all():
        bad = keys[int(np.flatnonzero(~(sd > 0))[0])]
        raise MblfError(f"The variable {labels.get(bad, bad)} has the same value in every sample: there is nothing to model.")
    Z = ((Yt - mean) / sd).T
    M = len(keys)
    b = {"kind": "identity", **(basis or {})}
    if b["kind"] == "pca" and M >= 2:
        share = float(b.get("var_fraction") or 0.95)
        B, lam, captured, r = values_pca_basis(Z, share, b.get("n_basis"))
        info = {"kind": "pca", "from": "pca", "share": share, "n_basis": int(B.shape[1]), "captured": captured, "n_available": r}
        notes.append(f"The values are described by {B.shape[1]} principal component(s) of their correlations ({100 * captured:.1f} % of their variance); what the components leave "
                     "out counts as noise of the values.")
    else:
        if b["kind"] == "pca":
            notes.append("A single value needs no principal components: it is its own basis function.")
        B, lam, captured, info = np.eye(M), None, None, {"kind": "identity", "n_basis": M}
    mode = Mode("values", "vector", B, mean, sd, np.full(M, 1.0 / M), np.arange(1, M + 1, dtype=float)[:, None], lam, captured,
                info, None, keys, [labels.get(k, k) for k in keys], "", None, logs, {"centre": "mean", "scaling": "sd"})
    return mode, Z, notes


# ----------------------------------------------------------------------------- covariates
def covariates_from(columns: dict, labels: dict, ids: list, log_keys: list | None = None) -> tuple[Covariates, np.ndarray, list]:
    """The covariates ``columns`` ``{key: (N,) numbers or category labels}`` -> ``(Covariates, keep (N,), notes)``; ``keep`` is False for samples with a missing covariate (they
    cannot be used: the model conditions the scores on every covariate).  A number in ``log_keys`` enters as its ``log10`` (a sample with a value of zero or below is left out)."""
    keys = list(columns)
    n = len(ids)
    keep = np.ones(n, bool)
    parsed = []
    notes = []
    for k in keys:
        v = columns[k]
        arr = np.asarray([np.nan if u is None else u for u in v], dtype=object)
        try:
            x = arr.astype(float)
            if k in (log_keys or []):
                bad = int((np.isfinite(x) & (x <= 0)).sum())
                x = log10_or_nan(x, 0.0)
                if bad:
                    notes.append(f"The covariate {labels.get(k, k)} enters on the log scale: {bad} sample(s) with a value of zero or below have no log and were left out.")
            parsed.append(("numeric", x))
            keep &= np.isfinite(x)
        except (TypeError, ValueError):
            if k in (log_keys or []):
                raise MblfError(f"The covariate {labels.get(k, k)} is a category: it has no log. Switch its log scale off.") from None
            lab = np.array(["" if u is None or (isinstance(u, float) and np.isnan(u)) else str(u) for u in v], dtype=object)
            parsed.append(("category", lab))
            keep &= lab != ""
    if not keep.all():
        notes.append(f"{int((~keep).sum())} sample(s) lack a covariate and were left out (the model needs every covariate of every sample).")
    cols, design = [], []
    for k, (kind, x) in zip(keys, parsed):
        xk = x[keep]
        lab = labels.get(k, k)
        if kind == "numeric":
            mu, sd = float(np.mean(xk)), float(np.std(xk, ddof=1)) if xk.size > 1 else 0.0
            if not sd > 0:
                notes.append(f"The covariate {lab} has the same value in every sample used: it was left out.")
                continue
            lg = k in (log_keys or [])
            cols.append({"key": k, "label": f"log10 {lab}" if lg else lab, "kind": "numeric", "level": None, "reference": None, "mean": mu, "sd": sd, "known": [], "log": lg})
            design.append((x - mu) / sd)
        else:
            levels = sorted(set(xk.tolist()), key=lambda s: (-(xk == s).sum(), s))
            if len(levels) < 2:
                notes.append(f"The covariate {lab} has the same category in every sample used: it was left out.")
                continue
            if len(levels) > 12:
                raise MblfError(f"The covariate {lab} has {len(levels)} categories: at most 12 can be used (merge some, or choose another variable).")
            ref = levels[0]
            for lev in levels[1:]:
                ind = (xk == lev).astype(float)
                mu = float(ind.mean())
                cols.append({"key": k, "label": f"{lab} = {lev}", "kind": "level", "level": lev, "reference": ref, "mean": mu, "sd": 1.0, "known": list(levels)})
                design.append((x == lev).astype(float) - mu)
    if not cols:
        raise MblfError("None of the covariates chosen varies between the samples: switch the covariates off, or choose others.")
    X = np.column_stack(design)[keep]
    return Covariates(keys, [labels.get(k, k) for k in keys], cols, X), keep, notes
