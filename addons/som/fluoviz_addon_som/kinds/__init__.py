"""The saved-model kind ``som`` (numpy only): the applier, the freezer.  See ``docs/kind_and_router.md``."""
from __future__ import annotations

from fluoviz_addon_common.kinds import register_standalone as _register

from .freeze import freeze_som
from .som import KIND, SomApplier

__all__ = ["KIND", "SomApplier", "freeze_som", "register_standalone"]


def register_standalone() -> None:
    """Register the kind with FluoViz without the add-on framework (tests and scripts): ``frozen.register_kind`` + the freezer registry."""
    _register(KIND, "Self-organising map", SomApplier, freeze_som)
