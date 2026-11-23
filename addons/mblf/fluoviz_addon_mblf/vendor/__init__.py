"""Third-party code carried by the add-on (FluoViz installs conda packages only, and ``mblf`` is not on conda-forge).

``mblf/`` is mblf-python 1.0.0 (https://github.com/Addicted2BayesianEpistemology/mblf-python, commit c83ddb6, MIT licence in ``mblf/LICENSE``), copied UNCHANGED: see ``VENDORED.md``.

Two ways it is used:

* in the FluoViz server, ``fluoviz_addon_mblf.vendor.mblf`` for everything after the fit (alignment, diagnostics, PSIS-LOO, the effective dimension, the projection residuals): numpy and
  scipy only, ``cmdstanpy`` / ``xarray`` are imported by mblf inside the functions that sample, which the server never calls;
* in the private Stan engine (``core/engine.py``), ``engine_job.py`` puts this folder on ``sys.path`` and imports ``mblf`` to sample the posterior with CmdStan.
"""
from pathlib import Path

VENDOR_DIR = Path(__file__).resolve().parent
