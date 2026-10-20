"""Looking at a result ``som-N``: finding it, its map (kept in memory after the first use), whether it is outdated, and the images in the orientation of the browser.  What the autoencoders
add-on does the same way is in the ``common`` add-on (``fluoviz_addon_common.results``)."""
from __future__ import annotations

from fluoviz_addon_common import results as C
from fluoviz_addon_common.results import browser_images, stale_words, turn_images, turn_mask  # noqa: F401  (part of this module's interface)

from ..kinds.freeze import unpack_frozen
from ..kinds.model import SomModel

RESULT_KIND = "som"

_R = C.Results(RESULT_KIND, a_noun="a self-organising map", noun="map", did="computed", verb="compute", again="Compute again with the same settings", model_class=SomModel, unpack=unpack_frozen)
get_result, model_of, frozen_of, require_current = _R.get_result, _R.model_of, _R.frozen_of, _R.require_current


def is_variables(r) -> bool:
    return (r.params or {}).get("input", {}).get("kind") == "variables"


def turn_for_browser(payload: dict) -> dict:
    """The stored payload as the page reads it: the frozen weights are left out, images and masks are turned to ``(…, n_ex, n_em)``."""
    out = {k: v for k, v in payload.items() if k != "frozen"}
    for key in ("prototypes", "mean_image"):
        if key in out:
            out[key] = turn_images(out[key])
    if "mask" in out:
        out["mask"] = turn_mask(out["mask"])
    return out
