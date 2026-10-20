"""The limit of a distance or an error that the saved kind and the router of a model add-on share.  Pure numpy / scipy.

The numbers come from samples the model was NOT trained on, described on the log scale by their median and a robust spread (1.4826 x the median absolute deviation, never below
``MIN_LOG_SPREAD``), because an error or a squared distance is positive and skewed: ``limit = exp(median + z x spread)`` with ``z`` 1.645 / 2.326 (the 95 % / 99 % points of a normal
distribution).  With fewer than 3 usable values the limits are taken from ``fallback`` (the training samples, which are optimistic) and doubled, and the result says so (``source``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sps

LEVELS = ("95", "99")
Z = {"95": float(sps.norm.ppf(0.95)), "99": float(sps.norm.ppf(0.99))}
MIN_LOG_SPREAD = 0.4                 # the log-errors of ordinary samples are never less spread than this (a chi-square with a few degrees of freedom has more): guards against a tiny limit from few samples


@dataclass(frozen=True)
class LimitWords:
    """The sentences that explain a limit in the words of one add-on.  ``held_out`` is formatted with ``n`` (samples) and ``typical`` (the typical value)."""
    too_few: str
    held_out: str
    training: str


def log_limits(values, fallback=None, *, squared: bool = False, words: LimitWords) -> dict:
    """The 95 % and 99 % points of ``values`` (positive numbers: errors, or distances when ``squared``: the squared distance is what is skewed, the limit is a distance again).

    Returns ``{"95": float, "99": float, "n": int, "median": float, "spread": float, "source": "held-out" | "training" | "none", "words": str}``."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v) & (v > 0)]
    source, widen = "held-out", 1.0
    if v.size < 3:
        fb = np.asarray(fallback if fallback is not None else [], float)
        fb = fb[np.isfinite(fb) & (fb > 0)]
        if fb.size < 3:
            return {"95": float("nan"), "99": float("nan"), "n": int(v.size), "median": float("nan"), "spread": float("nan"), "source": "none", "words": words.too_few}
        v, source, widen = fb, "training", 2.0
    lv = np.log(v ** 2 if squared else v)
    med = float(np.median(lv))
    spread = max(1.4826 * float(np.median(np.abs(lv - med))), MIN_LOG_SPREAD)
    back = (lambda x: float(np.sqrt(np.exp(x)))) if squared else (lambda x: float(np.exp(x)))
    out = {lev: float(widen * back(med + Z[lev] * spread)) for lev in LEVELS}
    out.update({"n": int(v.size), "median": back(med), "spread": spread, "source": source})
    out["words"] = words.held_out.format(n=v.size, typical=back(med)) if source == "held-out" else words.training
    return out
