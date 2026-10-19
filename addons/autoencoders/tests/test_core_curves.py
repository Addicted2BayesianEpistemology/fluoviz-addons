"""Response curves in numbers and words (``core/curves.py``): the positions, the weighted straight line, and the rules that put a curve into plain English.  No PyTorch."""
import numpy as np
import pytest

from fluoviz_addon_autoencoders.core import curves as C
from fluoviz_addon_autoencoders.core.curves import describe_curve, line_fit, normal_weights, positions, sd_text

T = positions(3, 41)


def test_positions_and_weights_follow_the_normal_density():
    assert T[0] == -3 and T[-1] == 3 and T.size == 41 and np.isclose(T[20], 0.0) and np.allclose(np.diff(T), 0.15)
    w = normal_weights(T)
    assert np.isclose(w.sum(), 1.0) and np.allclose(w, w[::-1]) and w[20] == w.max() and w[0] == w.min()
    assert np.isclose(w[20] / w[0], np.exp(4.5))                                            # exp(-t^2 / 2) at 0 against at 3
    assert positions(2, 4).tolist() == pytest.approx([-2, -2 / 3, 2 / 3, 2]) and 0.0 not in positions(1, 4)       # an even number of points has no position at the mean


def test_the_weighted_line_is_the_least_squares_line_with_the_density_as_weights():
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(T.size, 3)) + 2 * T[:, None] * np.array([1.0, -0.5, 0.0])
    w = normal_weights(T)
    fit = line_fit(T, Y)
    for q in range(3):
        c = np.polyfit(T, Y[:, q], 1, w=np.sqrt(w))                                         # numpy's weights multiply the residuals: sqrt of ours
        res = Y[:, q] - np.polyval(c, T)
        sse, sst = (w * res ** 2).sum(), (w * (Y[:, q] - (w * Y[:, q]).sum()) ** 2).sum()
        assert fit["slope"][q] == pytest.approx(c[0], abs=1e-12) and fit["intercept"][q] == pytest.approx(c[1], abs=1e-12)
        assert fit["sse"][q] == pytest.approx(sse, rel=1e-10) and fit["sst"][q] == pytest.approx(sst, rel=1e-10) and fit["r2"][q] == pytest.approx(1 - sse / sst, abs=1e-12)
    one = line_fit(T, 3 * T + 1)
    assert one["slope"] == pytest.approx(3) and one["intercept"] == pytest.approx(1) and one["r2"] == pytest.approx(1.0) and isinstance(one["r2"], float)
    assert np.isnan(line_fit(T, np.full(T.size, 2.0))["r2"])                              # nothing to explain


def test_r2_of_a_curve_with_a_square_term_has_a_closed_form():
    """For y = t + c t^2 and symmetric weights the line is y = t + c E[t^2]: R^2 = Var(t) / (Var(t) + c^2 Var(t^2))."""
    w = normal_weights(T)
    var = lambda x: (w * (x - (w * x).sum()) ** 2).sum()
    for c in (0.05, 0.3, 1.0):
        expected = var(T) / (var(T) + c * c * var(T * T))
        assert line_fit(T, T + c * T ** 2)["r2"] == pytest.approx(expected, abs=1e-12)
        assert describe_curve(T, T + c * T ** 2)["r2_linear"] == pytest.approx(expected, abs=1e-12)
    assert abs(line_fit(T, T ** 2)["r2"]) < 1e-12 and describe_curve(T, T ** 2)["r2_linear"] == pytest.approx(0.0, abs=1e-12)


def test_sd_text_has_a_true_minus_and_one_decimal():
    assert sd_text(1.24) == "+1.2 SD" and sd_text(-0.55) == "−0.6 SD" and sd_text(0.02) == "0 SD" and sd_text(-0.04) == "0 SD" and sd_text(3) == "+3.0 SD"


# (name, the curve, the kind, words that the label must have)
CURVES = [
    ("a straight line", lambda t: 2 + 3 * t, "linear", ["nearly linear"]),
    ("a falling straight line with a ripple", lambda t: 5 - 0.7 * t + 0.01 * np.sin(5 * t), "linear", ["nearly linear"]),
    ("a small square term (4 % of the range)", lambda t: t + 0.03 * t ** 2, "linear", ["nearly linear"]),
    ("a larger square term", lambda t: t + 0.12 * t ** 2, "convex", ["has a bend", "convex", "bends upwards"]),
    ("a concave bend", lambda t: np.tanh(0.3 * t), "concave", ["has a bend", "concave", "bends downwards"]),
    ("a convex bend that has no plateau", lambda t: t + 0.12 * t ** 2 + 4, "convex", ["convex"]),
    ("tanh(t)", np.tanh, "s_shaped", ["S-shaped", "levels off below about −1.5 SD and above about +1.5 SD"]),
    ("tanh(2 t)", lambda t: np.tanh(2 * t), "s_shaped", ["S-shaped", "−0.8 SD", "+0.8 SD"]),
    ("a cubic", lambda t: t ** 3, "s_shaped", ["S-shaped"]),
    ("saturation above", lambda t: 1 - np.exp(-(t + 3)), "saturating", ["levels off (saturates) above about"]),
    ("saturation below (falling)", lambda t: -np.exp(0.6 * t), "saturating", ["levels off (saturates) below about −1.4 SD"]),
    ("a switch (relu)", lambda t: np.maximum(t, 0), "saturating", ["levels off (saturates) below about"]),
    ("a peak", lambda t: np.exp(-0.5 * (t - 0.8) ** 2), "peak", ["rises then falls (peak at about +0.8 SD)"]),
    ("a valley", lambda t: -np.exp(-0.5 * (t + 0.6) ** 2), "valley", ["falls then rises (lowest at about −0.6 SD)"]),
    ("a parabola", lambda t: t ** 2, "valley", ["falls then rises (lowest at about 0 SD)"]),
    ("a wave", lambda t: np.sin(2 * t), "wiggly", ["changes direction several times"]),
    ("a constant", lambda t: np.full_like(t, 4.0), "flat", ["has almost no effect"]),
    ("rounding dust", lambda t: 4.0 + 1e-13 * t, "flat", ["has almost no effect"]),
]


@pytest.mark.parametrize("name,fn,kind,words", CURVES, ids=[c[0] for c in CURVES])
def test_the_words_of_known_curves(name, fn, kind, words):
    d = describe_curve(T, fn(T))
    assert d["kind"] == kind, (name, d["kind"], d["label"])
    for w in words:
        assert w in d["label"], (name, d["label"])


def test_the_numbers_of_a_curve_are_consistent():
    y = np.tanh(T)
    d = describe_curve(T, y)
    res = y - (np.polyval(np.polyfit(T, y, 1, w=np.sqrt(normal_weights(T))), T))
    assert d["max_deviation"] == pytest.approx(np.abs(res).max() / (y.max() - y.min()), rel=1e-9) and d["range"] == [pytest.approx(-np.tanh(3)), pytest.approx(np.tanh(3))]
    assert d["plateau"] == {"above": pytest.approx(1.5), "below": pytest.approx(-1.5)} and d["slope"] > 0 and d["direction"] == "up"
    assert describe_curve(T, -y)["direction"] == "down" and describe_curve(T, -y)["slope"] < 0
    assert describe_curve(T, 2 + 3 * T)["r2_linear"] == pytest.approx(1.0) and describe_curve(T, 2 + 3 * T)["max_deviation"] == pytest.approx(0.0, abs=1e-12)
    p = describe_curve(T, np.exp(-0.5 * (T - 0.8) ** 2))
    assert p["position"] == pytest.approx(0.75) and p["direction"] is None and p["turns"] == [(pytest.approx(0.75), "max")]
    assert describe_curve(T, np.full_like(T, 3.0))["r2_linear"] is None and describe_curve(T, np.full_like(T, 3.0))["max_deviation"] is None


def test_a_curve_below_two_percent_of_the_largest_range_has_almost_no_effect():
    y = np.exp(-0.5 * (T - 0.8) ** 2)                                                       # a peak ...
    assert describe_curve(T, y, reference_range=1.0)["kind"] == "peak"
    assert describe_curve(T, y, reference_range=100.0)["kind"] == "flat"                   # ... of range 1 against a code that moves 100: below 2 %
    assert describe_curve(T, y, reference_range=100.0)["r2_linear"] is not None            # (the numbers are still given)
    rng = y.max() - y.min()
    assert describe_curve(T, y, reference_range=rng)["kind"] == "peak"
    assert C.FLAT == 0.02 and describe_curve(T, y, reference_range=rng / 0.0199)["kind"] == "flat"      # just below 2 % of the largest
    assert describe_curve(T, y, reference_range=rng / 0.0201)["kind"] == "peak"                          # just above


def test_the_thresholds_are_the_documented_ones():
    assert (C.FLAT, C.LINEAR_TOL, C.TURN, C.PLATEAU, C.PLATEAU_SHARE, C.SIGN_TOL) == (0.02, 0.05, 0.10, 0.05, 0.25, 0.02)
    m2 = normal_weights(T) @ T ** 2
    # (1) nearly linear: the largest distance from the line is 5 % of the range.  y = t + c (t^2 - E[t^2]) has the line y = t, the range 6 and the largest distance c (9 - E[t^2])
    for dev, kind in ((0.045, "linear"), (0.055, "convex")):
        y = T + dev * 6.0 / (9.0 - m2) * (T ** 2 - m2)
        d = describe_curve(T, y)
        assert d["max_deviation"] == pytest.approx(dev, rel=1e-9) and d["kind"] == kind, (dev, d["kind"])
    # (2) a change of direction counts when the curve moves back by 10 % of its range: a rise to 1 at t = 0, then a fall by `depth`
    for depth, kind in ((0.08, "linear"), (0.12, "peak")):
        y = np.where(T < 0, (T + 3) / 3, 1 - depth * T / 3)
        d = describe_curve(T, y)
        assert (d["kind"] == "peak") == (kind == "peak"), (depth, d["kind"])
    wavy = np.where(T < -1, (T + 3) / 2, np.where(T < 0, 1 - 0.3 * (T + 1), 0.7 + (T / 3) * 1.3))        # up to 1, down by 0.3 (15 % of the range 2), up to 2: two changes
    assert describe_curve(T, wavy)["kind"] == "wiggly"
    shallow = np.where(T < -1, (T + 3) / 2, np.where(T < 0, 1 - 0.15 * (T + 1), 0.85 + (T / 3) * 1.15))   # the dip is 7.5 % of the range: not a change of direction
    assert describe_curve(T, shallow)["kind"] not in ("wiggly", "peak", "valley")
    # (3) a plateau covers at least a quarter of the positions and stays within 5 % of the range of its end value
    for t0, kind in ((1.0, "saturating"), (2.2, "concave")):
        d = describe_curve(T, np.minimum(T, t0))
        assert d["kind"] == kind, (t0, d["kind"], d["label"])
    assert describe_curve(T, np.minimum(T, 1.0))["plateau"]["above"] == pytest.approx(0.75, abs=0.16)       # (it starts 5 % of the range before the corner)
    # (4) the sign pattern: three changes of sign of the residuals (above 2 % of the range) make a curve S-shaped
    assert describe_curve(T, T ** 3)["kind"] == "s_shaped" and describe_curve(T, T ** 3 + 0 * T)["plateau"] == {"above": None, "below": None}


def test_a_curve_that_has_both_signs_says_where_it_crosses_zero():
    d = describe_curve(T, 2 + 3 * T)
    assert d["sign_change"] == pytest.approx(-2 / 3) and d["sign_label"] == "switches sign near −0.7 SD"
    assert describe_curve(T, 20 + 3 * T)["sign_change"] is None and describe_curve(T, np.tanh(T))["sign_label"] == "switches sign near 0 SD"
    assert describe_curve(T, 100 + 0.001 * T)["sign_change"] is None


def test_few_points_and_missing_values_are_handled():
    with pytest.raises(ValueError, match="at least 5"):
        describe_curve(T[:4], T[:4])
    d = describe_curve(T, np.where(T > 1, np.nan, T))
    assert d["kind"] == "unknown" and "missing" in d["label"] and d["r2_linear"] is None
    short = positions(1, 5)
    assert describe_curve(short, 2 * short)["kind"] == "linear"
