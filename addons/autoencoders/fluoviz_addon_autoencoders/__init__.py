"""FluoViz add-on: autoencoders and variational autoencoders (also functional).  See SPEC.md.

``register(api)`` only wires things up; the heavy libraries (PyTorch) are imported inside the functions that need them (the training process), never in the FluoViz server.
"""
from __future__ import annotations

__version__ = "1.1.0"


def register(api) -> None:
    """Routes (``/api/addon/autoencoders/...``), the saved-model kind ``autoencoder`` and the result kind of the same name."""
    from fluoviz_addon_common.kinds import declare_heavy

    from .kinds import KIND, AutoencoderApplier, freeze_autoencoder
    from .server.router import router

    declare_heavy(api, {"autoencoder", "autoencoder-scan"}, "autoencoder")
    api.add_router(router)
    api.add_saved_kind(KIND, "Autoencoder", AutoencoderApplier, result_kinds=("autoencoder",), freezer=freeze_autoencoder)
    api.add_result_kind("autoencoder", "autoencoder", saveable=True)
    api.log.info("autoencoders add-on %s registered", __version__)
