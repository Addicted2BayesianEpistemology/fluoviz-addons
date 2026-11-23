"""From the samples of a FluoViz project to the *modes* of the multimodal latent factor model, and back (SPEC section 2).

A **mode** is one kind of measurement of every sample: the fluorescence EEM (a surface), the absorbance spectrum (a curve), or a short vector of other numbers (``values``: lab
measurements, indices, scores of another model) that the person wants to model or predict.  For every mode the model needs (mblf, ``FunctionalData``):

* ``Y`` ``(M, N)``  the data at the ``M`` points of the mode (the observed pixels of the domain, or the variables), one column per sample, NaN = not observed.  The data are
  **standardised**: the mean of the training samples is subtracted at every point and the result is divided by ``scale`` (one number for a function: the root mean square of the
  centred samples over the domain; one number per variable for ``values``), so that the default priors of mblf mean the same thing for every mode and every unit;
* ``B`` ``(M, p)``  the basis.  For a function it is the Karhunen-Loeve basis of a Gaussian process on the domain (FluoViz's own, ``fluoviz.analysis.fpca``), each function
  multiplied by the square root of its prior variance (``Phi sqrt(lam)``, the construction of the paper): coefficients of order one then give functions of order one, and the
  number of functions is the smallest that keeps the chosen share of the prior variance (the paper: the share left out is the expected relative truncation error).  For ``values``
  the basis is the identity.

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
    column_labels: list = field(default_factory=list)  # values: what a person reads
    units: str = ""

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

    def standardise(self, Y: np.ndarray) -> np.ndarray:
        """``(n, M)`` data (units after the pre-treatment) -> ``(M, n)`` standardised columns."""
        return ((np.asarray(Y, float) - self.mean) / self.scale).T

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
                out[:, j] = (x - c["mean"]) / c["sd"]
            else:
                lab = np.array(["" if u is None or (isinstance(u, float) and np.isnan(u)) else str(u) for u in v], dtype=object)
                known = set(c["known"])
                ok = np.array([s in known for s in lab])
                out[:, j] = np.where(ok, (lab == c["level"]).astype(float) - c["mean"], np.nan)
        return out


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
                        "units": m.units, "columns": list(m.column_labels)})
        return {"n": self.N, "modes": out, "complete": int(pres.all(axis=1).sum()),
                "covariates": None if self.cov is None else {"keys": list(self.cov.keys), "labels": list(self.cov.labels), "columns": [dict(c) for c in self.cov.columns]},
                "notes": list(self.notes), "dropped": [list(d) for d in self.dropped]}


# ----------------------------------------------------------------------------- one functional mode
def standardise_columns(Yt: np.ndarray, w: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray, float]:
    """``Yt`` ``(N, M)`` (NaN = not observed; a sample without the mode is a row of NaN) -> ``(mean (M,), Z (M, N), scale)``: centred on the mean of the samples that have each point,
    divided by the root mean square of the centred samples over the domain (each sample averaged over the part it has)."""
    has = np.isfinite(Yt).any(axis=1)
    if has.sum() < 2:
        raise MblfError(f"Fewer than 2 samples have the {name}: it cannot be modelled.")
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(Yt[has], axis=0)
    if np.isnan(mean).any():
        raise MblfError(f"Some points of the {name} are not measured in any sample: raise the share of samples that must have a point, or mask them.")
    R = Yt - mean
    obs = np.isfinite(R)
    cov = (obs * w).sum(axis=1)
    ss = np.where(obs, R, 0.0) ** 2 @ w / np.where(cov > 0, cov, 1.0)
    msq = float(np.mean(ss[has]))
    if not np.isfinite(msq) or msq <= 0:
        raise MblfError(f"The {name} of the samples are all the same: there is nothing for the factors to describe.")
    scale = float(np.sqrt(msq))
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


def functional_mode(v, ids: list, basis_spec: dict | None, progress=None, ife: dict | None = None) -> tuple[Mode, np.ndarray, list]:
    """The mode of a FluoViz ``Variable`` (``fluoviz.analysis.fpca``) on the samples ``ids`` (a sample the variable does not have is a column of NaN).  Returns ``(mode, Y, notes)``."""
    from fluoviz.analysis.fpca.bases import build_representation
    from fluoviz.analysis.fpca.pipeline import resolve_basis
    from fluoviz_addon_common.functional import pixel_element

    dom = v.domain
    pos = {s: i for i, s in enumerate(v.ids)}
    Yt = np.full((len(ids), dom.m), np.nan)
    for k, s in enumerate(ids):
        if s in pos:
            Yt[k] = v.Y[pos[s]]
    w = np.asarray(dom.w, float)
    mean, Z, scale = standardise_columns(Yt, w, v.name)
    spec = {"kind": "gp", "kernel": "matern52", "structure": "separable", "lengths": "auto", "var_fraction": DEFAULT_SHARE, "k_max": K_CAP[dom.kind], **(basis_spec or {})}
    share = float(spec.get("var_fraction") or DEFAULT_SHARE)
    n_basis = spec.get("n_basis")
    sub = v.subset([s for s in ids if s in pos])
    resolved, _ = resolve_basis(sub, spec, progress)
    rep = build_representation(dom, {**resolved, "var_fraction": None if n_basis else share, "n_basis": n_basis}, progress)
    if rep.kind != "gp" or rep.prior_lam is None:
        raise MblfError("The multimodal latent factor model needs the Gaussian-process basis (step 3).")
    total = float(rep.params.get("total") or np.sum(rep.prior_lam))
    B, lam, captured = gp_basis_matrix(rep.B, rep.prior_lam, total, None if n_basis else share)
    notes = list(rep.notes)
    if B.shape[1] >= rep.prior_lam.size and not n_basis and captured < share - 1e-9:
        notes.append(f"The basis reached its largest size ({B.shape[1]} functions) and keeps {100 * captured:.1f} % of the prior variance, less than the {100 * share:.1f} % asked: "
                     "raise the largest number of functions, or choose longer length-scales.")
    fe = pixel_element(v, mean)
    if ife is not None and v.name == "absorbance":
        fe.ife = ife
    info = {"kernel": resolved.get("kernel"), "structure": resolved.get("structure"), "lengths": [float(x) for x in resolved.get("lengths", [])],
            "lengths_source": resolved.get("lengths_source"), "share": share, "n_basis": int(B.shape[1]), "captured": captured, "approximate": bool(rep.params.get("approximate"))}
    mode = Mode(v.name, dom.kind, B, mean, np.full(dom.m, scale), w, dom.coords(), lam, captured, info, fe, [], [], str(v.units or ""))
    return mode, Z, notes


def values_mode(columns: dict, labels: dict, ids: list) -> tuple[Mode, np.ndarray, list]:
    """The ``values`` mode: ``columns`` ``{key: (N,) numbers}`` on the samples ``ids`` (NaN = missing).  Each variable is centred and divided by its standard deviation; the basis is
    the identity (every variable is one coefficient)."""
    keys = list(columns)
    if not keys:
        raise MblfError("Choose at least one variable to model.")
    Yt = np.column_stack([np.asarray(columns[k], float) for k in keys])
    notes = []
    has = np.isfinite(Yt)
    for j, k in enumerate(keys):
        if has[:, j].sum() < 3:
            raise MblfError(f"The variable {labels.get(k, k)} has a value in fewer than 3 samples: it cannot be modelled.")
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(Yt, axis=0)
        sd = np.nanstd(Yt, axis=0, ddof=1)
    if not (sd > 0).all():
        bad = keys[int(np.flatnonzero(~(sd > 0))[0])]
        raise MblfError(f"The variable {labels.get(bad, bad)} has the same value in every sample: there is nothing to model.")
    Z = ((Yt - mean) / sd).T
    M = len(keys)
    mode = Mode("values", "vector", np.eye(M), mean, sd, np.full(M, 1.0 / M), np.arange(1, M + 1, dtype=float)[:, None], None, None,
                {"kind": "identity", "n_basis": M}, None, keys, [labels.get(k, k) for k in keys], "")
    return mode, Z, notes


# ----------------------------------------------------------------------------- covariates
def covariates_from(columns: dict, labels: dict, ids: list) -> tuple[Covariates, np.ndarray, list]:
    """The covariates ``columns`` ``{key: (N,) numbers or category labels}`` -> ``(Covariates, keep (N,), notes)``; ``keep`` is False for samples with a missing covariate (they
    cannot be used: the model conditions the scores on every covariate)."""
    keys = list(columns)
    n = len(ids)
    keep = np.ones(n, bool)
    parsed = []
    for k in keys:
        v = columns[k]
        arr = np.asarray([np.nan if u is None else u for u in v], dtype=object)
        try:
            x = arr.astype(float)
            parsed.append(("numeric", x))
            keep &= np.isfinite(x)
        except (TypeError, ValueError):
            lab = np.array(["" if u is None or (isinstance(u, float) and np.isnan(u)) else str(u) for u in v], dtype=object)
            parsed.append(("category", lab))
            keep &= lab != ""
    notes = []
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
            cols.append({"key": k, "label": lab, "kind": "numeric", "level": None, "reference": None, "mean": mu, "sd": sd, "known": []})
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
