"""The size of the map and the advice the page starts from (SPEC section 4.2).  Pure numpy."""
from __future__ import annotations

import numpy as np

from .metric import pca_basis, weighted_centre
from .spec import MAX_UNITS, MIN_SAMPLES, SomError

PRESET_WORDS = {
    "compact": "Compact: about one unit for every four samples. A map to group the samples: most units stand for real samples.",
    "standard": "Standard: 5 times the square root of the number of samples, the usual rule of the SOM Toolbox. A map to see the data: many units hold no sample and interpolate between their neighbours.",
    "fine": "Fine: twice the standard size. A detailed picture; most units are empty and the held-out error rises when there are few samples.",
}


def units_for(n: int, preset: str) -> int:
    if preset == "compact":
        u = max(6, round(n / 4.0))
    elif preset == "fine":
        u = min(MAX_UNITS, round(10.0 * np.sqrt(n)))
    else:
        u = round(5.0 * np.sqrt(n))
    return int(max(4, min(u, 4 * n, MAX_UNITS)))


def side_ratio(X: np.ndarray, w: np.ndarray | None = None) -> float:
    """``sqrt(lambda_1 / lambda_2)`` of the two largest eigenvalues of the covariance of the coordinates, capped to 1 … 4 (1 when the data have one direction only or are round)."""
    X = np.asarray(X, float)
    mu = np.nanmean(X, axis=0)
    Xf = np.where(np.isfinite(X), X, mu)
    Xc, _, dof = weighted_centre(Xf, w)
    _, s = pca_basis(Xc, 2)
    if s.size < 2 or s[1] <= 1e-12 * max(s[0], 1e-300):
        return 4.0 if s.size and s[0] > 0 else 1.0
    return float(np.clip(s[0] / s[1], 1.0, 4.0))


def shape_for(units: int, ratio: float, lattice: str = "hexagonal", shape: str = "sheet") -> tuple:
    """``(rows, cols)`` with about ``units`` units and the physical aspect ``ratio`` (the longer side holds the first principal direction); a hexagonal row is 0.866 high."""
    k = 0.866 if lattice == "hexagonal" else 1.0
    rows = max(2, int(round(np.sqrt(units / (ratio * k)))))
    cols = max(2, int(round(units / rows)))
    if lattice == "hexagonal" and shape == "toroid" and rows % 2:
        rows += 1
    while rows * cols > MAX_UNITS:
        if cols >= rows:
            cols -= 1
        else:
            rows -= 1
    return rows, cols


def presets(X: np.ndarray, w: np.ndarray | None = None, lattice: str = "hexagonal", shape: str = "sheet") -> dict:
    n = X.shape[0]
    ratio = side_ratio(X, w)
    out = {}
    for name in ("compact", "standard", "fine"):
        u = units_for(n, name)
        r, c = shape_for(u, ratio, lattice, shape)
        out[name] = {"rows": r, "cols": c, "units": r * c, "samples_per_unit": n / (r * c), "words": PRESET_WORDS[name]}
    out["side_ratio"] = ratio
    return out


def resolve_size(size, X: np.ndarray, w: np.ndarray | None, lattice: str, shape: str) -> tuple:
    """``(rows, cols)`` of a request's ``size`` (a preset or ``[rows, cols]``) on the coordinates ``X``."""
    n = X.shape[0]
    if isinstance(size, (list, tuple)):
        rows, cols = int(size[0]), int(size[1])
        if lattice == "hexagonal" and shape == "toroid" and rows % 2:
            rows += 1
    else:
        u = units_for(n, size)
        rows, cols = shape_for(u, side_ratio(X, w), lattice, shape)
    if rows * cols > 4 * n and n >= MIN_SAMPLES:
        raise SomError(f"A map of {rows * cols} units is too many for {n} samples (at most 4 units per sample). Choose a smaller size.")
    return rows, cols


def warnings_for(n: int, rows: int, cols: int, mode: str, epochs: int, p: int) -> list:
    out = []
    u = rows * cols
    if n / u < 1.0:
        out.append(f"{n} samples on {u} units: most units will hold no sample and simply interpolate between their neighbours. That is normal for a map; choose Compact if you want every unit to stand for real samples.")
    if n < 30:
        out.append(f"Only {n} samples: a map of so few samples is a picture, not a model. Read the held-out numbers in step 8 before relying on it.")
    if mode in ("pixels",) and n < p:
        out.append("The map works on the pixels themselves: with fewer samples than pixels, noise in weak pixels counts as much as signal in strong ones; the functional version smooths them first.")
    return out
