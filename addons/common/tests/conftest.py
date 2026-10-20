"""Test set-up of the shared add-on: its folder goes on ``sys.path`` (the package is not installed in the test environment).  FluoViz itself must be importable
(``PYTHONPATH=<FluoViz folder>`` with its environment, as for the other add-ons)."""
import pathlib
import sys

ADDON = pathlib.Path(__file__).resolve().parents[1]
for p in (ADDON, ADDON / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
