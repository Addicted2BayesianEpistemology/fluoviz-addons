"""Registering the saved kind and the long jobs of a model add-on."""
from __future__ import annotations


def register_standalone(kind: str, label: str, applier, freezer, result_kind: str | None = None) -> None:
    """Register the kind with FluoViz without the add-on framework (tests and scripts): ``frozen.register_kind`` + the freezer registry."""
    from fluoviz.analysis.frozen import KINDS, register_kind
    if kind not in KINDS:
        register_kind(kind, label, applier)
    try:
        from fluoviz.server.saved_kinds import FREEZERS, SAVEABLE_LABELS, discover
        discover()
        FREEZERS.setdefault(result_kind or kind, freezer)
        SAVEABLE_LABELS.setdefault(result_kind or kind, label)
    except ImportError:                                                                   # pragma: no cover - no server layer in a pure-numpy script
        pass


def declare_heavy(api, kinds: set[str] | tuple[str, ...], what: str) -> None:
    """Fits and scans run for a while: they share the small pool of the long jobs, so interactive jobs never queue behind them."""
    try:
        from fluoviz.server import jobs
        jobs.HEAVY_KINDS.update(set(kinds))
    except Exception:                                        # noqa: BLE001 - a different job runner: the default pool is fine
        api.log.info("the job pool of this FluoViz could not be told about the %s jobs", what)


def check_grid(fe, grid, what: str) -> None:
    """The axes of the samples (``grid``: one array per axis of the element ``fe``) must be those of the model: a plain ``FrozenError`` otherwise."""
    import numpy as np
    from fluoviz.analysis.frozen import FrozenError
    for a, g in zip(fe.axes, grid):
        if g is None or np.asarray(g).shape != np.asarray(a).shape or not np.allclose(np.asarray(g, float), np.asarray(a, float), rtol=0, atol=1e-6):
            raise FrozenError(f"The {what} of the samples is not on the wavelength grid of the model. Apply the model through its own data recipe, which puts the samples on the model's grid.")
