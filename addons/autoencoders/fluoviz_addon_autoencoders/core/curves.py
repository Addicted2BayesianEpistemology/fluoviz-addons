"""Response curves in numbers and in words (pure numpy): the positions along a code, the straight line that best describes a curve, and the plain-English description of its shape.

A *response curve* is a feature of the decoded function (the intensity at one pixel, the overall intensity ...) at ``n`` positions of one code, from ``-span`` to ``+span``
standard deviations (SD) of that code over the training samples, all the other codes at their mean.  Everything here works on one curve ``y`` at positions ``t``.

The straight line is fitted with weights: the standard normal density at the positions, normalised to add up to 1 (a position near the mean code counts more than one far out,
as the samples do).  ``r2_linear`` is the weighted R^2 of that line, ``max_deviation`` the largest distance of the curve from the line as a share of the range of the curve.

The words come from a small set of rules, applied in this order (the first that fits wins; every threshold is a constant of this module and is quoted by ``docs/api.md``)::

    flat       the range of the curve is below FLAT (2 %) of the largest range among the codes, or is numerically zero      "has almost no effect on this feature"
    wiggly     the curve changes direction two or more times (a change counts when the curve rises / falls by TURN = 10 % of its range on both sides)
    peak       it changes direction once, from rising to falling                                                           "rises then falls (peak at about +x SD)"
    valley     once, from falling to rising                                                                                "falls then rises (lowest at about -x SD)"
    linear     monotone and max_deviation <= LINEAR_TOL (5 %)                                                               "nearly linear"
    s_shaped   monotone, flat at both ends (within PLATEAU = 5 % of the range of the end value over at least a quarter of the positions at each end)
    saturating monotone and flat at one end                                                                                 "levels off (saturates) above / below about x SD"
    s_shaped   monotone, no plateau, the residuals from the line change sign three times or more (|residual| above SIGN_TOL = 2 % of the range counts)
    convex     monotone, the curve lies below its chord in the middle                                                      "has a bend (convex ...)"
    concave    monotone, above its chord                                                                                   "has a bend (concave ...)"

Whatever its shape, a curve that has both signs also says where it crosses zero (``sign_label``: "switches sign near +x SD"); the page shows that for intensities (a negative
intensity is a warning) and not for the coordinates of the network, which are centred and cross zero anyway.
"""
from __future__ import annotations

import numpy as np

FLAT = 0.02                  # no effect: the range of the curve is below this share of the largest range among the codes
LINEAR_TOL = 0.05            # nearly linear: the largest distance from the straight line is below this share of the range
TURN = 0.10                  # a change of direction counts when the curve rises and falls by this share of its range
PLATEAU = 0.05               # a plateau stays within this share of the range of its end value ...
PLATEAU_SHARE = 0.25         # ... over at least this share of the whole range of positions
SIGN_TOL = 0.02              # a residual counts for the sign pattern when it is above this share of the range
NUMERIC_ZERO = 1e-9          # a range below this share of the largest absolute value is zero for practical purposes

LABELS = {
    "flat": "has almost no effect on this feature",
    "linear": "nearly linear",
    "convex": "has a bend (convex: it bends upwards, the slope grows with the code)",
    "concave": "has a bend (concave: it bends downwards, the slope shrinks with the code)",
    "s_shaped": "S-shaped",
    "wiggly": "changes direction several times",
}


def sd_text(x: float) -> str:
    """A position in SD as the page writes it: ``+1.2 SD``, ``−0.5 SD``, ``0 SD`` (a true minus sign, one decimal)."""
    r = round(float(x), 1)
    if r == 0:
        return "0 SD"
    return f"{'+' if r > 0 else '−'}{abs(r):.1f} SD"


# ----------------------------------------------------------------------------- positions and weights
def positions(span: float, n_points: int) -> np.ndarray:
    """``n_points`` positions from ``-span`` to ``+span`` SD (including 0 when ``n_points`` is odd)."""
    return np.linspace(-float(span), float(span), int(n_points))


def normal_weights(t) -> np.ndarray:
    """The standard normal density at the positions ``t``, normalised to add up to 1."""
    t = np.asarray(t, float)
    w = np.exp(-0.5 * t * t)
    return w / w.sum()


# ----------------------------------------------------------------------------- the straight line
def line_fit(t, Y, w=None) -> dict:
    """Weighted least-squares straight line of ``Y`` against ``t``; ``Y`` is ``(n,)`` or ``(n, q)`` (one line per column).

    Returns ``{"slope", "intercept", "sse", "sst", "r2", "weights"}`` (arrays of shape ``(q,)``, floats for a 1-D ``Y``): ``sse`` = weighted sum of the squared residuals,
    ``sst`` = weighted sum of the squares about the weighted mean, ``r2 = 1 - sse / sst`` (NaN where the column is constant up to rounding: ``sst`` and ``sse`` are then 0).  ``w`` defaults to the normal density at ``t``."""
    t = np.asarray(t, float)
    Y = np.asarray(Y, float)
    one = Y.ndim == 1
    Y2 = Y[:, None] if one else Y
    w = normal_weights(t) if w is None else np.asarray(w, float) / np.sum(w)
    tm = float(w @ t)
    tc = t - tm
    stt = float(w @ (tc * tc))
    Ym = w @ Y2
    Yc = Y2 - Ym
    slope = (w * tc) @ Yc / stt if stt > 0 else np.zeros(Y2.shape[1])
    res = Yc - tc[:, None] * slope[None, :]
    sst = w @ (Yc * Yc)
    sse = w @ (res * res)
    live = sst > (NUMERIC_ZERO ** 2) * (w @ (Y2 * Y2))                          # (a column that is constant up to rounding has nothing to explain)
    sst = np.where(live, sst, 0.0)
    sse = np.where(live, sse, 0.0)
    r2 = np.where(live, 1.0 - sse / np.where(live, sst, 1.0), np.nan)
    out = {"slope": slope, "intercept": Ym - slope * tm, "sse": sse, "sst": sst, "r2": r2, "weights": w}
    if one:
        for k in ("slope", "intercept", "sse", "sst", "r2"):
            out[k] = float(out[k][0])
    return out


# ----------------------------------------------------------------------------- the shape in words
def _turning_points(y: np.ndarray, thr: float) -> list[tuple[int, str]]:
    """Interior changes of direction that stand out by at least ``thr`` on both sides: ``[(index, 'max' | 'min'), ...]`` (the zigzag rule: an extreme counts once the
    curve has moved back from it by ``thr``)."""
    n = y.size
    hi = lo = 0
    direction = 0
    cur = 0
    piv: list[tuple[int, str]] = []
    for i in range(1, n):
        if direction == 0:
            if y[i] > y[hi]:
                hi = i
            if y[i] < y[lo]:
                lo = i
            if y[hi] - y[lo] >= thr:
                if lo < hi:
                    direction, cur = 1, hi
                    piv.append((lo, "min"))
                else:
                    direction, cur = -1, lo
                    piv.append((hi, "max"))
        elif direction == 1:
            if y[i] > y[cur]:
                cur = i
            elif y[cur] - y[i] >= thr:
                piv.append((cur, "max"))
                direction, cur = -1, i
        else:
            if y[i] < y[cur]:
                cur = i
            elif y[i] - y[cur] >= thr:
                piv.append((cur, "min"))
                direction, cur = 1, i
    out = []
    for k, (idx, kind) in enumerate(piv):
        if idx <= 0 or idx >= n - 1:
            continue
        if k == 0 and abs(y[idx] - y[0]) < thr:                    # the first extreme must stand out from the start of the curve too
            continue
        out.append((int(idx), kind))
    return out


def _plateau(t: np.ndarray, y: np.ndarray, rng: float) -> dict:
    """Where the curve is flat at its ends: the first position from which it stays within PLATEAU of the range of the last value (right), and the last position up to which it
    stays within that of the first value (left); a plateau must cover at least PLATEAU_SHARE of the whole range of positions."""
    tol = PLATEAU * rng
    n = y.size
    total = float(t[-1] - t[0])
    out: dict = {"right": None, "left": None}
    k = n - 1
    while k > 0 and abs(y[k - 1] - y[-1]) <= tol:
        k -= 1
    if k < n - 1 and (t[-1] - t[k]) >= PLATEAU_SHARE * total - 1e-12:
        out["right"] = float(t[k])
    k = 0
    while k < n - 1 and abs(y[k + 1] - y[0]) <= tol:
        k += 1
    if k > 0 and (t[k] - t[0]) >= PLATEAU_SHARE * total - 1e-12:
        out["left"] = float(t[k])
    return out


def _sign_changes(r: np.ndarray, tol: float) -> int:
    s = np.sign(np.where(np.abs(r) > tol, r, 0.0))
    s = s[s != 0]
    return int((np.diff(s) != 0).sum()) if s.size > 1 else 0


def _zero_crossing(t: np.ndarray, y: np.ndarray) -> float | None:
    """Where a curve that has both signs crosses zero (linear interpolation between the two positions around the first crossing)."""
    sgn = np.sign(y)
    idx = np.flatnonzero(sgn[:-1] * sgn[1:] < 0)
    if idx.size == 0:
        z = np.flatnonzero(sgn == 0)
        return float(t[z[0]]) if z.size and 0 < z[0] < y.size - 1 else None
    i = int(idx[0])
    return float(t[i] - y[i] * (t[i + 1] - t[i]) / (y[i + 1] - y[i]))


def describe_curve(t, y, w=None, reference_range: float | None = None) -> dict:
    """The numbers and the words of one response curve (see the module text for the rules).

    ``reference_range``: the largest range among the codes for the same feature (a curve below FLAT of it has almost no effect); ``None`` = no comparison.
    Returns ``{"kind", "label", "r2_linear", "max_deviation", "range": [lo, hi], "slope", "direction", "position", "turns", "plateau", "sign_change", "sign_label"}``;
    ``slope`` is the slope of the line per SD of the code in the units of the feature, ``direction`` ``'up' | 'down' | None``, ``position`` the place of the peak or valley (SD),
    ``plateau`` ``{'above': x | None, 'below': x | None}`` (the plateau starts at x above / ends at x below), ``sign_change`` the position where a bent curve crosses zero."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    if t.size < 5 or y.size != t.size:
        raise ValueError("A curve needs at least 5 positions.")
    if not np.isfinite(y).all():
        return {"kind": "unknown", "label": "cannot be described (some values are missing)", "r2_linear": None, "max_deviation": None, "range": [None, None], "slope": None,
                "direction": None, "position": None, "turns": [], "plateau": {"above": None, "below": None}, "sign_change": None, "sign_label": None}
    lo, hi = float(y.min()), float(y.max())
    rng = hi - lo
    scale = max(abs(lo), abs(hi))
    fit = line_fit(t, y, w)
    out = {"kind": "flat", "label": LABELS["flat"], "r2_linear": None, "max_deviation": None, "range": [lo, hi], "slope": float(fit["slope"]), "direction": None, "position": None,
           "turns": [], "plateau": {"above": None, "below": None}, "sign_change": None, "sign_label": None}
    if rng <= NUMERIC_ZERO * max(scale, 1e-300) or rng == 0.0:
        return out
    resid = y - (fit["intercept"] + fit["slope"] * t)
    out["r2_linear"] = None if not np.isfinite(fit["r2"]) else float(min(max(fit["r2"], 0.0), 1.0))
    out["max_deviation"] = float(np.abs(resid).max() / rng)
    out["direction"] = "up" if y[-1] > y[0] else "down"
    if reference_range is not None and rng < FLAT * reference_range:
        return out
    if lo < 0.0 < hi and min(-lo, hi) > SIGN_TOL * rng:
        z = _zero_crossing(t, y)
        if z is not None:
            out["sign_change"], out["sign_label"] = z, f"switches sign near {sd_text(z)}"
    turns = _turning_points(y, TURN * rng)
    out["turns"] = [(float(t[i]), k) for i, k in turns]
    if len(turns) >= 2:
        out.update(kind="wiggly", label=LABELS["wiggly"], direction=None)
        return out
    if len(turns) == 1:
        i, kind = turns[0]
        out["position"] = float(t[i])
        if kind == "max":
            out.update(kind="peak", label=f"rises then falls (peak at about {sd_text(t[i])})", direction=None)
        else:
            out.update(kind="valley", label=f"falls then rises (lowest at about {sd_text(t[i])})", direction=None)
        return out
    # monotone from here on
    if out["max_deviation"] <= LINEAR_TOL:
        out.update(kind="linear", label=LABELS["linear"])
        return out
    pl = _plateau(t, y, rng)
    out["plateau"] = {"above": pl["right"], "below": pl["left"]}
    if pl["right"] is not None and pl["left"] is not None:
        out.update(kind="s_shaped", label=f"S-shaped (levels off below about {sd_text(pl['left'])} and above about {sd_text(pl['right'])})")
    elif pl["right"] is not None:
        out.update(kind="saturating", label=f"levels off (saturates) above about {sd_text(pl['right'])}")
    elif pl["left"] is not None:
        out.update(kind="saturating", label=f"levels off (saturates) below about {sd_text(pl['left'])}")
    elif _sign_changes(resid, SIGN_TOL * rng) >= 3:
        out.update(kind="s_shaped", label=LABELS["s_shaped"])
    else:
        mid = float(np.interp(0.5 * (t[0] + t[-1]), t, y))
        chord = 0.5 * (y[0] + y[-1])
        kind = "convex" if mid < chord else "concave"
        out.update(kind=kind, label=LABELS[kind])
    return out
