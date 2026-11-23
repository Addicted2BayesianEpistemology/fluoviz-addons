"""FluoViz add-on: Bayesian multimodal latent factors (mblf) for EEMs, absorbance spectra and other values, with covariates.  See SPEC.md.

``register(api)`` only wires things up.  Nothing heavy is imported: the posterior is sampled by CmdStan in a private environment (``core/engine.py``), and everything after the fit is
numpy / scipy (with the vendored ``mblf`` package, ``vendor/``).
"""
from __future__ import annotations

__version__ = "1.0.0"


def register(api) -> None:
    """Routes (``/api/addon/mblf/...``), the saved-model kind ``mblf`` and the result kind of the same name."""
    from fluoviz_addon_common.kinds import declare_heavy

    from .kinds import KIND, MblfApplier, freeze_mblf
    from .server import router as R

    R.DATA_DIR = api.data_dir
    declare_heavy(api, {"mblf-fit", "mblf-engine", "mblf-compare", "mblf-scan"}, "latent factor")
    api.add_router(R.router)
    api.add_saved_kind(KIND, "Multimodal latent factors", MblfApplier, result_kinds=("mblf",), freezer=freeze_mblf)
    api.add_result_kind("mblf", "latent factor model", saveable=True)
    api.log.info("multimodal latent factor add-on %s registered", __version__)
