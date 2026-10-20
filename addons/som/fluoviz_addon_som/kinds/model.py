"""A frozen map as a numpy object: how a sample becomes coordinates, the metric, the prototypes, and the assignment of new samples (modelled on ``kinds/model.py`` of the autoencoders).

``meta["model"]`` of the saved kind (and ``payload["frozen"]["model"]`` of a result) is a plain tree::

    {"map": {"rows", "cols", "lattice", "shape"}, "arrays": {"protos_t", "protos_raw", "hits", "unit_cluster"}, "tau": float,
     "metric": Metric, "frame": SomFrame, "layout": {"variances": (B,)}}

Pure numpy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fluoviz.analysis.frozen import FrozenError
from fluoviz.analysis.frozen.kinds import _decomp as D
from fluoviz.analysis.frozen.kinds.fpca import FrozenElement, clean_absorbance

from ..core import fitting
from ..core import som as S
from ..core.data import SomFrame
from ..core.spec import SomError


def model_tree(fit, frame: SomFrame, unit_cluster: np.ndarray | None = None) -> dict:
    """``meta["model"]`` from a :class:`~..core.fitting.FitResult`."""
    lat = fit.lattice
    M = lat.M
    return {"map": {"rows": int(lat.rows), "cols": int(lat.cols), "lattice": lat.kind, "shape": lat.shape},
            "arrays": {"protos_t": np.array(fit.protos_t, float), "protos_raw": np.array(fit.protos_raw, float), "hits": np.array(fit.hits, int),
                       "unit_cluster": np.full(M, -1, int) if unit_cluster is None else np.asarray(unit_cluster, int)},
            "tau": float(fit.tau), "metric": fit.metric, "frame": frame, "layout": {"variances": np.asarray(fit.layout.variances, float)}}


@dataclass
class Evaluation:
    n: int
    usable: np.ndarray
    why: list
    coverage: np.ndarray
    gaps: np.ndarray
    treated: dict
    R: np.ndarray
    unit: np.ndarray                   # (n,) best unit (-1 when unusable)
    unit2: np.ndarray
    qe: np.ndarray                     # (n,) distance to the best unit in the training metric
    qe_blocks: np.ndarray              # (n, B) distance in the data units of each variable
    qe_rel: np.ndarray                 # (n,) share of the typical variation not captured
    pos: np.ndarray                    # (n, 2) continuous position in lattice coordinates
    empty: np.ndarray                  # (n,) the best unit holds no training sample
    D2: np.ndarray | None = None
    notes: list = field(default_factory=list)


class MapView:
    """What the publishing of variables needs of a map: the lattice, the hits and groups of the units, the names of the variables.  A map of other variables is only this."""

    def __init__(self, lattice, protos_t, hits, unit_cluster, names):
        self.lattice = lattice
        self.protos_t = np.asarray(protos_t, float)
        self.hits = np.asarray(hits, int)
        self.unit_cluster = np.asarray(unit_cluster, int)
        self.names = list(names)
        self.umat, self.edges, _ = S.umatrix(self.protos_t, lattice)

    @property
    def M(self) -> int:
        return self.lattice.M

    def has_clusters(self) -> bool:
        return bool((self.unit_cluster >= 0).any())

    def unit_name(self, u) -> np.ndarray:
        r, c = self.lattice.row_col(np.asarray(u))
        return np.array([f"r{int(a) + 1}c{int(b) + 1}" for a, b in zip(np.atleast_1d(r), np.atleast_1d(c))], dtype=object)


class SomModel(MapView):
    def __init__(self, tree: dict):
        if not isinstance(tree, dict) or not {"map", "arrays", "metric", "frame", "layout"} <= set(tree):
            raise FrozenError("This model file is damaged: its map is missing.")
        try:
            m = tree["map"]
            lattice = S.make_lattice(int(m["rows"]), int(m["cols"]), str(m["lattice"]), str(m["shape"]))
            a = tree["arrays"]
            protos_raw = np.asarray(a["protos_raw"], float)
            self.protos_raw = protos_raw
            self.tau = float(tree["tau"])
            self.metric = tree["metric"]
            self.frame = tree["frame"]
            self.variances = np.asarray(tree["layout"]["variances"], float)
            super().__init__(lattice, a["protos_t"], a["hits"], a["unit_cluster"], self.frame.names())
        except (KeyError, TypeError, ValueError):
            raise FrozenError("This model file is damaged: the description of the map is incomplete.") from None
        if self.frame.P != self.protos_raw.shape[1] or self.protos_raw.shape[0] != self.lattice.M:
            raise FrozenError("This model file is damaged: the prototypes do not fit the map.")
        self.layout = fitting.Layout(self.frame.names(), self.frame.slices(), self.variances, self.frame.sqrt_q() ** 2)

    def block(self, name: str):
        return self.frame.block(name)

    def to_tree(self) -> dict:
        """The plain tree this model was made from (with the current groups of the units)."""
        lat = self.lattice
        return {"map": {"rows": int(lat.rows), "cols": int(lat.cols), "lattice": lat.kind, "shape": lat.shape},
                "arrays": {"protos_t": self.protos_t, "protos_raw": self.protos_raw, "hits": self.hits, "unit_cluster": self.unit_cluster}, "tau": float(self.tau), "metric": self.metric,
                "frame": self.frame, "layout": {"variances": self.variances}}

    # ------------------------------------------------------------------ from samples to the pixels of the domains
    def values(self, data) -> dict:
        out = {}
        for b in self.frame.blocks:
            fe: FrozenElement = b.element
            dom = b.domain
            if b.name == "fluorescence":
                X = D.eem_array(data, tuple(int(np.asarray(a).size) for a in fe.axes), "map")
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

    def evaluate(self, data, min_coverage: float = 0.8) -> Evaluation:
        co = self.frame.coordinates(self.values(data), min_coverage)
        return self._chain(data.n, co["usable"], co["why"], co["coverage"], co["treated"], co["R"])

    def evaluate_coordinates(self, R: np.ndarray, treated: dict | None = None) -> Evaluation:
        n = R.shape[0]
        if treated is None:
            treated = {b.name: np.zeros((n, b.domain.m)) for b in self.frame.blocks}
        usable = np.isfinite(R).any(axis=1)
        return self._chain(n, usable, [[] for _ in range(n)], np.ones((n, len(self.names))), treated, R)

    def _chain(self, n, usable, why, coverage, treated, R) -> Evaluation:
        B = len(self.names)
        unit = np.full(n, -1)
        unit2 = np.full(n, -1)
        qe = np.full(n, np.nan)
        qb = np.full((n, B), np.nan)
        rel = np.full(n, np.nan)
        pos = np.full((n, 2), np.nan)
        empty = np.zeros(n, bool)
        D2all = None
        gaps = np.zeros(n, bool)
        for b in self.frame.blocks:
            gaps |= usable & ~np.isfinite(np.asarray(treated[b.name], float)).all(axis=1)
        if usable.any():
            Ru = np.asarray(R, float)[usable]
            a = fitting.assign(Ru, self.metric, self.protos_t, self.protos_raw, self.layout)
            unit[usable], unit2[usable] = a["c1"], a["c2"]
            qe[usable], qb[usable], rel[usable] = a["qe"], a["qe_blocks"], a["qe_rel"]
            pos[usable] = S.positions(a["D2"], self.lattice, a["c1"], self.tau)
            empty[usable] = self.hits[a["c1"]] == 0
        return Evaluation(n, np.asarray(usable, bool), why, np.asarray(coverage, float), gaps, treated, np.asarray(R, float), unit, unit2, qe, qb, rel, pos, empty, D2all)

    # ------------------------------------------------------------------ pictures
    def prototype_functions(self, units) -> dict:
        return self.frame.to_functions(self.protos_raw[np.asarray(units)])

    def mean_functions(self) -> dict:
        return self.frame.mean_functions()

    def embed(self, name: str, F: np.ndarray) -> np.ndarray:
        return self.block(name).domain.embed(F)


def _check_grid(fe: FrozenElement, grid, what: str) -> None:
    for a, g in zip(fe.axes, grid):
        if g is None or np.asarray(g).shape != np.asarray(a).shape or not np.allclose(np.asarray(g, float), np.asarray(a, float), rtol=0, atol=1e-6):
            raise FrozenError(f"The {what} of the samples is not on the wavelength grid of the model. Apply the model through its own data recipe, which puts the samples on the model's grid.")
