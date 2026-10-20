"""The saved-model kind ``autoencoder`` (numpy only: no PyTorch, no server): the applier, the freezer and the network tools.  See ``docs/kind_and_router.md``."""
from __future__ import annotations

from fluoviz_addon_common.kinds import register_standalone as _register

from .autoencoder import KIND, AutoencoderApplier
from .freeze import freeze_autoencoder

__all__ = ["KIND", "AutoencoderApplier", "freeze_autoencoder", "register_standalone"]


def register_standalone() -> None:
    """Register the kind with FluoViz without the add-on framework (tests and scripts): ``frozen.register_kind`` + the freezer registry."""
    _register(KIND, "Autoencoder", AutoencoderApplier, freeze_autoencoder, "autoencoder")
