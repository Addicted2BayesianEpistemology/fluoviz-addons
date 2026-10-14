"""Test set-up of the Autoencoders add-on: the add-on folder goes on ``sys.path`` (the package is not installed in the test environment).

Tests of the numerics that need PyTorch use ``pytest.importorskip("torch")`` and are skipped cleanly when it is missing (the base FluoViz install has none)."""
import pathlib
import sys

ADDON = pathlib.Path(__file__).resolve().parents[1]
if str(ADDON) not in sys.path:
    sys.path.insert(0, str(ADDON))
