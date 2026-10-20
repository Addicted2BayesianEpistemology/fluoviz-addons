"""The saved-model kind ``som`` (numpy only): the applier, the freezer.  See ``docs/kind_and_router.md``."""
from __future__ import annotations

from .freeze import freeze_som
from .som import KIND, SomApplier

__all__ = ["KIND", "SomApplier", "freeze_som", "register_standalone"]


def register_standalone() -> None:
    """Register the kind with FluoViz without the add-on framework (tests and scripts): ``frozen.register_kind`` + the freezer registry."""
    from fluoviz.analysis.frozen import KINDS, register_kind
    if KIND not in KINDS:
        register_kind(KIND, "Self-organising map", SomApplier)
    try:
        from fluoviz.server.saved_kinds import FREEZERS, SAVEABLE_LABELS, discover
        discover()
        FREEZERS.setdefault("som", freeze_som)
        SAVEABLE_LABELS.setdefault("som", "Self-organising map")
    except ImportError:                                                                   # pragma: no cover
        pass
