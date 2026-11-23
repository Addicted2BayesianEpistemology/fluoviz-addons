"""Looking at a result ``mblf-N``: finding it, its model (kept in memory after the first use), whether it is outdated.  What the other model add-ons do the same way is in the
``common`` add-on (``fluoviz_addon_common.results``)."""
from __future__ import annotations

from fluoviz_addon_common import results as C
from fluoviz_addon_common.results import stale_words  # noqa: F401  (part of this module's interface)

from ..kinds.freeze import unpack_frozen
from ..kinds.model import MblfModel

RESULT_KIND = "mblf"

_R = C.Results(RESULT_KIND, a_noun="a latent factor model", noun="model", did="fitted", verb="fit", again="Fit again with the same settings", model_class=MblfModel,
               unpack=unpack_frozen)
get_result, model_of, frozen_of, require_current = _R.get_result, _R.model_of, _R.frozen_of, _R.require_current


def for_browser(payload: dict) -> dict:
    """The stored payload as the page reads it (the frozen model is left out: it is large and the page never needs it)."""
    return {k: v for k, v in payload.items() if k != "frozen"}
