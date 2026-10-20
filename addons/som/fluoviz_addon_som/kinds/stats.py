"""The limit of the quantisation error that the saved kind and the router share.  Pure numpy / scipy.

The limits come from the distances of samples the map was NOT trained on (the out-of-fold distances of the k-fold validation), on the log scale of the SQUARED distance (a squared distance
is positive and skewed): ``limit = sqrt(exp(median + z x spread))`` with a robust spread (1.4826 x the median absolute deviation, never below ``MIN_LOG_SPREAD``).  The computation is the
one of the ``common`` add-on (``fluoviz_addon_common.limits``); this file gives it the words of the map.
"""
from __future__ import annotations

from fluoviz_addon_common.limits import LEVELS, MIN_LOG_SPREAD, Z, LimitWords, log_limits  # noqa: F401  (LEVELS, Z and MIN_LOG_SPREAD are part of this module's interface)

WORDS = LimitWords(
    too_few="There are too few samples to set a limit for the distance to the best unit.",
    held_out=("The limit is the 99 % point of the distances to the best unit of {n} samples the map was not trained on (typical distance {typical:.3g}); "
              "about 1 ordinary new sample in 100 is above it."),
    training="Too few samples were held out to set the limit from them, so it is taken from the training samples and doubled; it is a rough guide until the model has been applied to more samples.")


def distance_limits(distances, fallback=None) -> dict:
    """``{"95", "99"}`` limits of the distance to the best unit from the held-out distances (with fewer than 3 the training distances are used and the limit is doubled)."""
    return log_limits(distances, fallback, squared=True, words=WORDS)
