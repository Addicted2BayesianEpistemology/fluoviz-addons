"""The saved-model kind ``autoencoder`` (numpy only: no PyTorch, no server): the applier, the freezer and the network tools.  See ``docs/kind_and_router.md``."""
from __future__ import annotations

from .autoencoder import KIND, AutoencoderApplier
from .freeze import freeze_autoencoder

__all__ = ["KIND", "AutoencoderApplier", "freeze_autoencoder", "register_standalone"]


def register_standalone() -> None:
    """Register the kind with FluoViz without the add-on framework (tests and scripts): ``frozen.register_kind`` + the freezer registry."""
    from fluoviz.analysis.frozen import KINDS, register_kind
    if KIND not in KINDS:
        register_kind(KIND, "Autoencoder", AutoencoderApplier)
    try:
        from fluoviz.server.saved_kinds import FREEZERS, SAVEABLE_LABELS, discover
        discover()
        FREEZERS.setdefault("autoencoder", freeze_autoencoder)
        SAVEABLE_LABELS.setdefault("autoencoder", "Autoencoder")
    except ImportError:                                                                   # pragma: no cover - no server layer in a pure-numpy script
        pass
