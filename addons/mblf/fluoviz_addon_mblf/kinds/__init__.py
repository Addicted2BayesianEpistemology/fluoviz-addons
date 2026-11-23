"""The saved-model kind ``mblf`` (numpy only): the applier and the freezer."""
from __future__ import annotations

from fluoviz_addon_common.kinds import register_standalone as _register

from .freeze import freeze_mblf
from .mblf_kind import KIND, MblfApplier

__all__ = ["KIND", "MblfApplier", "freeze_mblf", "register_standalone"]


def register_standalone() -> None:
    """Register the kind with FluoViz without the add-on framework (tests and scripts)."""
    _register(KIND, "Multimodal latent factors", MblfApplier, freeze_mblf)
