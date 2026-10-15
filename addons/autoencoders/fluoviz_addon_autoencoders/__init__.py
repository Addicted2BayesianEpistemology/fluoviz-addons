"""FluoViz add-on: autoencoders and variational autoencoders (also functional).  See SPEC.md.

``register(api)`` only wires things up; the heavy libraries (PyTorch) are imported inside the functions that need them (the training process), never in the FluoViz server.
"""
from __future__ import annotations

__version__ = "1.0.1"


def register(api) -> None:
    """Routes (``/api/addon/autoencoders/...``), the saved-model kind ``autoencoder`` and the result kind of the same name."""
    from .kinds import KIND, AutoencoderApplier, freeze_autoencoder
    from .server.router import router

    try:                                                     # fits and scans run for minutes: they share the small pool of the long jobs, so interactive jobs never queue behind them
        from fluoviz.server import jobs
        jobs.HEAVY_KINDS.update({"autoencoder", "autoencoder-scan"})
    except Exception:                                        # noqa: BLE001 - a different job runner: the default pool is fine
        api.log.info("the job pool of this FluoViz could not be told about the autoencoder jobs")
    api.add_router(router)
    api.add_saved_kind(KIND, "Autoencoder", AutoencoderApplier, result_kinds=("autoencoder",), freezer=freeze_autoencoder)
    api.add_result_kind("autoencoder", "autoencoder", saveable=True)
    api.log.info("autoencoders add-on %s registered", __version__)
