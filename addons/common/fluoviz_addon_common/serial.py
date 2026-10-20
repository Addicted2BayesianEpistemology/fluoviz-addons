"""Let a dataclass be stored in a saved model (``fluoviz.analysis.frozen.serial``)."""
from __future__ import annotations


def allow(cls):
    """Class decorator: register ``cls`` with the saved-model serialiser; nothing happens when FluoViz is not importable (tests of the numerics run without the saved-model machinery).

    The class stays where it is defined (a saved model records its module), so moving the decorator here never changes a saved file."""
    try:
        from fluoviz.analysis.frozen import serial
        serial.allow(cls)
    except Exception:                                                  # noqa: BLE001 - numerics tests run without FluoViz's saved-model machinery
        pass
    return cls
