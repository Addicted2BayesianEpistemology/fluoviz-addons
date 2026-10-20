"""The settings of a map, checked in plain English (``Config``) and the words the numerics share.

The data settings (``input``) are the Functional PCA request, checked by the function the Functional PCA page uses (``server/request.py``); everything else is here.  Pure Python.
"""
from __future__ import annotations

from dataclasses import dataclass, field

METRICS = ("l2", "mahalanobis", "prior", "standardize")
METRIC_WORDS = {"l2": "plain distance (L2)", "mahalanobis": "standardised distance (regularised Mahalanobis)", "prior": "Gaussian-process prior distance",
                "standardize": "every coordinate standardised"}
LATTICES = ("hexagonal", "rectangular")
SHAPES = ("sheet", "toroid")
SIZES = ("compact", "standard", "fine")
MODES = ("batch", "sequential")
INITS = ("linear", "random")
NEIGHBOURHOODS = ("gaussian", "cutgauss", "bubble", "epanechnikov")
GAP_WEIGHTINGS = ("observed", "equal")
CLUSTER_METHODS = ("hierarchical", "kmeans", "gmm")
MIN_SAMPLES = 8
ABSOLUTE_MIN = 4
MAX_UNITS = 400
INPUT_KINDS = ("functional", "pixels", "variables")


class SomError(ValueError):
    """A problem the person can fix; the message says what to do."""


def _num(d: dict, key: str, default, lo: float, hi: float, what: str, integer: bool = False):
    v = d.get(key, default)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise SomError(f"{what} must be a number.")
    if integer and float(v) != int(v):
        raise SomError(f"{what} must be a whole number.")
    if not lo <= float(v) <= hi:
        raise SomError(f"{what} must be between {lo:g} and {hi:g}.")
    return int(v) if integer else float(v)


def _choice(d: dict, key: str, default: str, options, what: str) -> str:
    v = d.get(key, default)
    if v not in options:
        raise SomError(f"{what} must be one of: {', '.join(options)}.")
    return v


def _group(raw, name: str, allowed: tuple) -> dict:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise SomError(f"'{name}' must be a group of settings.")
    bad = sorted(set(raw) - set(allowed))
    if bad:
        raise SomError(f"Unknown setting '{bad[0]}' in '{name}'. Known settings: {', '.join(allowed)}.")
    return raw


@dataclass
class Config:
    metric_kind: str = "l2"
    alpha: float = 1.0
    components: object = "auto"              # "auto" | int
    size: object = "standard"                # preset | [rows, cols]
    lattice: str = "hexagonal"
    shape: str = "sheet"
    mode: str = "batch"
    init: str = "linear"
    neighbourhood: str = "gaussian"
    epochs_rough: int = 50
    epochs_fine: int = 150
    radius_initial: object = "auto"
    radius_final: float = 1.0
    alpha_initial: float = 0.05
    alpha_final: float = 0.01
    gap_weighting: str = "observed"
    folds: int = 5
    limit: int = 99
    cluster_method: str = "hierarchical"
    cluster_k: object = "auto"
    cluster_k_range: tuple = (2, 10)
    cluster_skip_empty: object = "auto"
    seed: int = 0
    allow_few_samples: bool = False
    input: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ reading
    @classmethod
    def from_dict(cls, b: dict) -> "Config":
        if not isinstance(b, dict):
            raise SomError("The settings must be sent as a group of named values.")
        top = {"input", "metric", "map", "train", "gap_weighting", "validation", "cluster", "seed", "allow_few_samples", "title", "replace"}
        bad = sorted(set(b) - top)
        if bad:
            raise SomError(f"Unknown setting '{bad[0]}'. Known settings: {', '.join(sorted(top - {'title', 'replace'}))}.")
        m = _group(b.get("metric"), "metric", ("kind", "alpha", "components"))
        mp = _group(b.get("map"), "map", ("size", "lattice", "shape"))
        tr = _group(b.get("train"), "train", ("mode", "init", "neighbourhood", "epochs_rough", "epochs_fine", "radius_initial", "radius_final", "alpha_initial", "alpha_final"))
        va = _group(b.get("validation"), "validation", ("folds", "limit"))
        cl = _group(b.get("cluster"), "cluster", ("method", "k", "k_range", "skip_empty"))
        c = cls()
        c.metric_kind = _choice(m, "kind", "l2", METRICS, "The distance")
        c.alpha = _num(m, "alpha", 1.0, 0.0, 1.0, "How far to standardise (alpha)")
        comp = m.get("components", "auto")
        if comp != "auto":
            comp = _num(m, "components", "auto", 1, 200, "The number of components for the standardised distance", integer=True)
        c.components = comp
        size = mp.get("size", "standard")
        if isinstance(size, (list, tuple)):
            if len(size) != 2 or any(isinstance(v, bool) or not isinstance(v, int) for v in size):
                raise SomError("A custom map size must be two whole numbers: the rows and the columns.")
            if min(size) < 2 or size[0] * size[1] > MAX_UNITS:
                raise SomError(f"A map needs at least 2 rows and 2 columns and at most {MAX_UNITS} units.")
            size = [int(size[0]), int(size[1])]
        elif size not in SIZES:
            raise SomError(f"The size of the map must be one of {', '.join(SIZES)} or two numbers [rows, columns].")
        c.size = size
        c.lattice = _choice(mp, "lattice", "hexagonal", LATTICES, "The lattice")
        c.shape = _choice(mp, "shape", "sheet", SHAPES, "The shape of the map")
        c.mode = _choice(tr, "mode", "batch", MODES, "The training mode")
        c.init = _choice(tr, "init", "linear", INITS, "The initialisation")
        c.neighbourhood = _choice(tr, "neighbourhood", "gaussian", NEIGHBOURHOODS, "The neighbourhood function")
        c.epochs_rough = _num(tr, "epochs_rough", 50, 1, 2000, "The epochs of the rough phase", integer=True)
        c.epochs_fine = _num(tr, "epochs_fine", 150, 1, 5000, "The epochs of the fine phase", integer=True)
        ri = tr.get("radius_initial", "auto")
        if ri != "auto":
            ri = _num(tr, "radius_initial", "auto", 0.5, 100, "The initial radius")
        c.radius_initial = ri
        c.radius_final = _num(tr, "radius_final", 1.0, 0.3, 5.0, "The final radius")
        c.alpha_initial = _num(tr, "alpha_initial", 0.05, 1e-4, 1.0, "The initial learning rate")
        c.alpha_final = _num(tr, "alpha_final", 0.01, 1e-5, 1.0, "The final learning rate")
        if c.alpha_final > c.alpha_initial:
            raise SomError("The final learning rate cannot be larger than the initial one.")
        c.gap_weighting = _choice(b, "gap_weighting", "observed", GAP_WEIGHTINGS, "The weighting of samples that lack part of the domain")
        c.folds = _num(va, "folds", 5, 2, 20, "The number of folds", integer=True)
        lim = va.get("limit", 99)
        if lim not in (95, 99, "95", "99"):
            raise SomError("The limit for the quantisation error must be 95 or 99 (per cent).")
        c.limit = int(lim)
        c.cluster_method = _choice(cl, "method", "hierarchical", CLUSTER_METHODS, "The clustering method")
        k = cl.get("k", "auto")
        if k != "auto":
            k = _num(cl, "k", "auto", 2, 50, "The number of groups", integer=True)
        c.cluster_k = k
        kr = cl.get("k_range", [2, 10])
        if not (isinstance(kr, (list, tuple)) and len(kr) == 2 and all(isinstance(v, int) and not isinstance(v, bool) for v in kr) and 2 <= kr[0] <= kr[1] <= 50):
            raise SomError("The range of group numbers must be two whole numbers from 2 to 50, the smaller first.")
        c.cluster_k_range = (int(kr[0]), int(kr[1]))
        se = cl.get("skip_empty", "auto")
        if se not in ("auto", True, False):
            raise SomError("'skip_empty' must be auto, true or false.")
        c.cluster_skip_empty = se
        c.seed = _num(b, "seed", 0, 0, 2 ** 31 - 1, "The seed", integer=True)
        afs = b.get("allow_few_samples", False)
        if not isinstance(afs, bool):
            raise SomError("'allow_few_samples' must be true or false.")
        c.allow_few_samples = afs
        c.input = dict(b.get("input") or {})
        return c

    def to_dict(self) -> dict:
        return {"metric": {"kind": self.metric_kind, "alpha": self.alpha, "components": self.components},
                "map": {"size": self.size, "lattice": self.lattice, "shape": self.shape},
                "train": {"mode": self.mode, "init": self.init, "neighbourhood": self.neighbourhood, "epochs_rough": self.epochs_rough, "epochs_fine": self.epochs_fine,
                          "radius_initial": self.radius_initial, "radius_final": self.radius_final, "alpha_initial": self.alpha_initial, "alpha_final": self.alpha_final},
                "gap_weighting": self.gap_weighting, "validation": {"folds": self.folds, "limit": self.limit},
                "cluster": {"method": self.cluster_method, "k": self.cluster_k, "k_range": list(self.cluster_k_range), "skip_empty": self.cluster_skip_empty},
                "seed": self.seed, "allow_few_samples": self.allow_few_samples, "input": self.input}

    # ------------------------------------------------------------------ checks that need the data
    def check_samples(self, n: int) -> None:
        floor = ABSOLUTE_MIN if self.allow_few_samples else MIN_SAMPLES
        if n < ABSOLUTE_MIN:
            raise SomError(f"Only {n} samples are included: a map needs at least {ABSOLUTE_MIN}.")
        if n < floor:
            raise SomError(f"Only {n} samples are included: a map of fewer than {MIN_SAMPLES} samples shows very little. Add more samples, or switch on 'Compute on fewer than {MIN_SAMPLES} samples anyway'.")
