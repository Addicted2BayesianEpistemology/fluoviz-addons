"""FluoViz add-on: self-organising maps of EEMs and absorbance spectra (also functional).  See SPEC.md.

``register(api)`` only wires things up.  The add-on needs nothing but numpy and scipy, which FluoViz already has.
"""
from __future__ import annotations

__version__ = "1.0.0"


def register(api) -> None:
    """Routes (``/api/addon/som/...``), the saved-model kind ``som`` and the result kind of the same name."""
    from .kinds import KIND, SomApplier, freeze_som
    from .server.router import router

    try:                                                     # scans run for a while: they share the small pool of the long jobs, so interactive jobs never queue behind them
        from fluoviz.server import jobs
        jobs.HEAVY_KINDS.update({"som-fit", "som-scan"})
    except Exception:                                        # noqa: BLE001 - a different job runner: the default pool is fine
        api.log.info("the job pool of this FluoViz could not be told about the map jobs")
    api.add_router(router)
    api.add_saved_kind(KIND, "Self-organising map", SomApplier, result_kinds=("som",), freezer=freeze_som)
    api.add_result_kind("som", "map", saveable=True)
    api.log.info("self-organising map add-on %s registered", __version__)
