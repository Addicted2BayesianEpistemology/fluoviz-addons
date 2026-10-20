"""Test set-up of the Autoencoders add-on: the add-on folder goes on ``sys.path`` (the package is not installed in the test environment).

Tests of the numerics that need PyTorch use ``pytest.importorskip("torch")`` and are skipped cleanly when it is missing (the base FluoViz install has none)."""
import pathlib
import sys

ADDON = pathlib.Path(__file__).resolve().parents[1]
COMMON = ADDON.parent / "common"                      # the shared add-on (``fluoviz_addon_common``): the package is not installed in the test environment either
for _p in (ADDON, COMMON):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


import importlib.util

import pytest

TORCH = importlib.util.find_spec("torch") is not None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Without PyTorch (the base FluoViz install has none) the add-on refuses to train and says so in plain English: a test that only wanted to train is skipped, not failed."""
    outcome = yield
    rep = outcome.get_result()
    if rep.when == "call" and rep.failed and not TORCH and call.excinfo is not None and "PyTorch is not installed" in str(call.excinfo.value):
        rep.outcome = "skipped"
        rep.longrepr = (str(item.fspath), item.location[1] or 0, "Skipped: needs PyTorch (the add-on installs it)")
