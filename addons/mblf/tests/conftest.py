"""Test set-up of the multimodal latent factor add-on: the add-on folder and the shared add-on go on ``sys.path`` (the packages are not installed in the test environment).

The tests that run the Stan engine need one: ``FLUOVIZ_MBLF_ENGINE=<prefix of a conda environment with cmdstan, cmdstanpy, xarray, numpy, scipy>`` (and micromamba, ``FLUOVIZ_MM``, or
FluoViz's own in ``runtime/bin``); without it they are skipped with a plain reason."""
import pathlib
import sys

ADDON = pathlib.Path(__file__).resolve().parents[1]
COMMON = ADDON.parent / "common"
for p in (ADDON, ADDON / "tests", COMMON):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
