"""The verdicts of the checks, in plain English.  Every threshold of the page lives here; the page only shows the sentences (SPEC section 5)."""
from __future__ import annotations

import numpy as np


def _pct(x: float, digits: int = 0) -> str:
    return f"{100 * x:.{digits}f} %"


def vs_pca(q: dict) -> dict:
    """Does the map keep neighbours together better than a scatter of the first two principal components?"""
    tm, tp = q["trust_map"], q["trust_pca"]
    m = np.nanmean([tm["trustworthiness"], tm["continuity"]])
    p = np.nanmean([tp["trustworthiness"], tp["continuity"]])
    d = float(m - p)
    k = tm["k"]
    nums = (f"With the {k} nearest neighbours of each sample, the map scores {m:.3f} and the scatter of the first two principal components {p:.3f} "
            "(trustworthiness and continuity averaged; 1 = the picture shows exactly the neighbours the data have).")
    if not np.isfinite(d):
        return {"verdict": "There are too few samples to compare the map with a scatter of the first two components.", "level": "info", "map": m, "pca": p, "difference": d}
    if d > 0.02:
        v = f"The map is worth it: it keeps the neighbours of the samples together better than the plain scatter of the first two components. {nums}"
        lvl = "good"
    elif d < -0.02:
        v = f"The map does no better than the scatter of the first two components, which is simpler and shows the same thing; use the scatter unless you need the prototypes or groups. {nums}"
        lvl = "warn"
    else:
        v = f"The map keeps neighbours about as well as the scatter of the first two components. Its value is then the prototypes, the planes and the groups, not a better picture. {nums}"
        lvl = "info"
    return {"verdict": v, "level": lvl, "map": m, "pca": p, "difference": d}


def topographic(q: dict) -> dict:
    te, tp = q["topographic_error"], q["topographic_product"]
    if te < 0.1:
        a = f"Topographic error {_pct(te)}: nearly every sample's two best units are neighbours, so the map is not torn."
        lvl = "good"
    elif te < 0.25:
        a = f"Topographic error {_pct(te)}: some samples have their two best units far apart on the map, which means the map is folded in places. Read distances on it with care."
        lvl = "info"
    else:
        a = f"Topographic error {_pct(te)}: many samples have their two best units far apart, so the map is folded or twisted. A smaller map, a longer training or a different distance may help."
        lvl = "warn"
    if abs(tp) < 0.02:
        b = f" Topographic product {tp:.3f} (near 0): the map has the right dimension for these data."
    elif tp < 0:
        b = f" Topographic product {tp:.3f} (negative): the map is too flat for these data, which are more than two-dimensional, so it folds."
    else:
        b = f" Topographic product {tp:.3f} (positive): the map is larger than these data need; it stretches across them."
    return {"verdict": a + b, "level": lvl}


def heldout(q: dict) -> dict:
    if "heldout_over_train" not in q:
        return {"verdict": "There were too few samples to check the map on samples it was not trained on.", "level": "info"}
    r = q["heldout_over_train"]
    cap = q.get("heldout_captured", float("nan"))
    tail = f" On held-out samples the map captures {_pct(cap)} of the variation, against {_pct(q['captured'])} on the samples it was trained on."
    es = q.get("heldout_empty_share", 0.0)
    if es > 0.2:
        tail += f" {_pct(es)} of the held-out samples landed on a unit that no training sample had, so the map is finer than the samples can support."
    if r < 1.3:
        return {"verdict": f"The map generalises: samples it has not seen sit {r:.2f} times as far from their unit as the training samples do.{tail}", "level": "good", "ratio": r}
    if r < 2.0:
        return {"verdict": f"The map is a little finer than the data support: held-out samples sit {r:.2f} times as far from their unit as the training samples do.{tail}", "level": "info", "ratio": r}
    return {"verdict": f"The map memorises the samples: held-out samples sit {r:.2f} times as far from their unit as the training samples do. Use a smaller map (see the map-size scan).{tail}", "level": "warn", "ratio": r}


def training(history: dict) -> dict:
    e = np.asarray(history.get("energy", []), float)
    if e.size < 10:
        return {"verdict": "The training was too short to judge.", "level": "info"}
    tail = e[-max(5, e.size // 10):]
    change = float(abs(tail[0] - tail[-1]) / max(abs(tail[0]), 1e-300))
    if change < 0.01:
        return {"verdict": "Training has settled: the energy of the map hardly changed over the last tenth of the epochs.", "level": "good", "change": change}
    return {"verdict": f"The energy was still falling by {_pct(change, 1)} at the end of the training: train longer (more epochs in the fine phase).", "level": "warn", "change": change}


def empties(q: dict) -> str:
    e, u = q["n_empty"], q["n_units"]
    return f"{e} of the {u} units hold no sample: they interpolate between their neighbours." if e else f"Every one of the {u} units holds at least one sample."


def all_verdicts(q: dict, history: dict) -> dict:
    return {"vs_pca": vs_pca(q), "topographic": topographic(q), "heldout": heldout(q), "training": training(history), "empty": empties(q)}


def size_scan(rows: list, suggested: int | None, reason: str = "") -> str:
    if suggested is None:
        return "No map size could be recommended: the held-out error could not be computed."
    r = next(x for x in rows if x["units"] == suggested)
    return (f"A map of {r['units']} units ({r['rows']} × {r['cols']}) is the smallest whose held-out error is within one standard error of the best, with a topographic error below 10 % where possible. "
            f"{reason}".strip())


def stability(value: float, baseline: float, ari: float) -> dict:
    if not np.isfinite(value):
        return {"verdict": "The stability could not be computed.", "level": "info"}
    lift = value - baseline
    if value > 0.8 and lift > 0.3:
        return {"verdict": f"The neighbourhoods are reproducible: {_pct(value)} of the pairs of samples that sit on the same or neighbouring units stay together when the map is trained again from other starting points (chance would give {_pct(baseline)}). "
                           f"The groups agree with the first map's groups (adjusted Rand index {ari:.2f}).", "level": "good"}
    if value > 0.6:
        return {"verdict": f"The neighbourhoods are partly reproducible: {_pct(value)} of the pairs that sit together stay together (chance: {_pct(baseline)}). Read the detail of the map with care; the groups agree with adjusted Rand index {ari:.2f}.", "level": "info"}
    return {"verdict": f"The neighbourhoods are not reproducible: only {_pct(value)} of the pairs that sit together stay together when the map is trained again (chance: {_pct(baseline)}). "
                       "A smaller map or more samples would give a map you can trust.", "level": "warn"}
