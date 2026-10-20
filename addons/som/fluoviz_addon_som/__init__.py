"""FluoViz add-on: self-organising maps of EEMs and absorbance spectra (also functional).  See SPEC.md.

``register(api)`` only wires things up.  The add-on needs nothing but numpy and scipy, which FluoViz already has.
"""
from __future__ import annotations

__version__ = "1.0.0"


def register(api) -> None:
    """Routes (``/api/addon/som/...``), the saved-model kind ``som`` and the result kind of the same name."""
    from fluoviz_addon_common.kinds import declare_heavy

    from .kinds import KIND, SomApplier, freeze_som
    from .server.router import router

    declare_heavy(api, {"som-fit", "som-scan"}, "map")
    api.add_router(router)
    api.add_saved_kind(KIND, "Self-organising map", SomApplier, result_kinds=("som",), freezer=freeze_som)
    api.add_result_kind("som", "map", saveable=True)
    api.log.info("self-organising map add-on %s registered", __version__)
