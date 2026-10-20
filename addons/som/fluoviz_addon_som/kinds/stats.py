"""The limit of the quantisation error that the saved kind and the router share (modelled on ``kinds/stats.py`` of the autoencoders add-on).  Pure numpy / scipy.

The limits come from the distances of samples the map was NOT trained on (the out-of-fold distances of the k-fold validation), on the log scale of the SQUARED distance (a squared distance
is positive and skewed): ``limit = sqrt(exp(median + z x spread))`` with a robust spread (1.4826 x the median absolute deviation, never below ``MIN_LOG_SPREAD``).
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sps

LEVELS = ("95", "99")
Z = {"95": float(sps.norm.ppf(0.95)), "99": float(sps.norm.ppf(0.99))}
MIN_LOG_SPREAD = 0.4


def distance_limits(distances, fallback=None) -> dict:
    """``{"95", "99"}`` limits of the distance to the best unit from the held-out distances (with fewer than 3 the training distances are used and the limit is doubled)."""
    d = np.asarray(distances, float)
    d = d[np.isfinite(d) & (d > 0)]
    source, widen = "held-out", 1.0
    if d.size < 3:
        fb = np.asarray(fallback if fallback is not None else [], float)
        fb = fb[np.isfinite(fb) & (fb > 0)]
        if fb.size < 3:
            return {"95": float("nan"), "99": float("nan"), "n": int(d.size), "source": "none", "words": "There are too few samples to set a limit for the distance to the best unit."}
        d, source, widen = fb, "training", 2.0
    lv = np.log(d ** 2)
    med = float(np.median(lv))
    spread = max(1.4826 * float(np.median(np.abs(lv - med))), MIN_LOG_SPREAD)
    out = {lev: float(widen * np.sqrt(np.exp(med + Z[lev] * spread))) for lev in LEVELS}
    out.update({"n": int(d.size), "median": float(np.sqrt(np.exp(med))), "spread": spread, "source": source})
    if source == "held-out":
        out["words"] = (f"The limit is the 99 % point of the distances to the best unit of {d.size} samples the map was not trained on (typical distance {np.sqrt(np.exp(med)):.3g}); "
                        "about 1 ordinary new sample in 100 is above it.")
    else:
        out["words"] = "Too few samples were held out to set the limit from them, so it is taken from the training samples and doubled; it is a rough guide until the model has been applied to more samples."
    return out
