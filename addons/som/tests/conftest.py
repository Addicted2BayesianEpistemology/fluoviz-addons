"""Test set-up of the Self-organising map add-on: the add-on folder goes on ``sys.path`` (the package is not installed in the test environment)."""
import pathlib
import sys

ADDON = pathlib.Path(__file__).resolve().parents[1]
for p in (ADDON, ADDON / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
