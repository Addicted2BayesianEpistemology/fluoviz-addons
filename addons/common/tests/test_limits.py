"""The limit of an error or a distance on the log scale (``fluoviz_addon_common.limits``): the numbers the two add-ons gave before they shared it."""
import numpy as np
import pytest
from scipy import stats

from fluoviz_addon_common import limits as L

WORDS = L.LimitWords(too_few="few", held_out="held {n} typical {typical:.3g}", training="training")


def reference(values, squared):
    """The formula, written out once more here, independently of the module."""
    v = np.asarray(values, float)
    lv = np.log(v ** 2 if squared else v)
    med = np.median(lv)
    spread = max(1.4826 * np.median(np.abs(lv - med)), 0.4)
    z = {"95": stats.norm.ppf(0.95), "99": stats.norm.ppf(0.99)}
    back = (lambda x: np.sqrt(np.exp(x))) if squared else np.exp
    return {k: float(back(med + z[k] * spread)) for k in z}, float(back(med)), float(spread)


@pytest.mark.parametrize("squared", [False, True])
def test_the_limits_are_the_log_scale_median_plus_a_robust_spread(squared):
    rng = np.random.default_rng(3)
    v = np.exp(rng.normal(0.2, 0.7, 40))
    out = L.log_limits(v, squared=squared, words=WORDS)
    lim, med, spread = reference(v, squared)
    assert out["95"] == pytest.approx(lim["95"]) and out["99"] == pytest.approx(lim["99"]) and out["median"] == pytest.approx(med) and out["spread"] == pytest.approx(spread)
    assert out["n"] == 40 and out["source"] == "held-out" and out["words"] == f"held 40 typical {med:.3g}" and out["95"] < out["99"]


def test_a_tiny_spread_is_raised_to_the_minimum():
    out = L.log_limits(np.full(10, 2.0) * (1 + 1e-9 * np.arange(10)), words=WORDS)
    assert out["spread"] == L.MIN_LOG_SPREAD


def test_too_few_values_use_the_training_values_and_double_the_limit():
    train = np.exp(np.random.default_rng(1).normal(0, 0.5, 30))
    plain = L.log_limits(train, words=WORDS)
    out = L.log_limits([1.0, 2.0], fallback=train, words=WORDS)
    assert out["source"] == "training" and out["words"] == "training" and out["n"] == 30
    assert out["99"] == pytest.approx(2 * plain["99"]) and out["95"] == pytest.approx(2 * plain["95"])


def test_without_enough_values_there_is_no_limit_and_it_says_so():
    out = L.log_limits([1.0], fallback=[1.0, np.nan, -1.0], words=WORDS)
    assert out["source"] == "none" and out["words"] == "few" and np.isnan(out["95"]) and np.isnan(out["99"]) and out["n"] == 1
    assert set(out) == {"95", "99", "n", "median", "spread", "source", "words"}


def test_non_positive_and_non_finite_values_are_ignored():
    v = np.array([1.0, 2.0, 3.0, 4.0, np.nan, np.inf, 0.0, -2.0])
    assert L.log_limits(v, words=WORDS)["n"] == 4
