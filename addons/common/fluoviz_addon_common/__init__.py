"""FluoViz add-on ``common``: the code that the model add-ons (Autoencoders, Self-organising map) share.  See ``docs/README.md``.

It has no page, no route and no saved-model kind of its own (``library: true``): FluoViz installs it together with the first add-on that needs it, starts it BEFORE them and removes it
with the last one.  The add-ons import its modules by name (``from fluoviz_addon_common import limits``); its browser files are at ``/addon-web/common/web/...``.

Python modules (each one numpy-only unless it says so):

* ``serial``       let a dataclass be stored in a saved model
* ``limits``       the limit of a distance / an error on the log scale (the 95 % and 99 % lines of the checks)
* ``functional``   the functional variables of a request: basis / smoothing / gaps settings, variables from arrays, pixel and Gaussian-process helpers
* ``project_data`` (server) the project's data as functional variables, the cache of the prepared data, the samples of a result as an ``ApplyData``
* ``results``      (server) finding a result and its model, whether it is outdated, images in the orientation of the browser
* ``freeze``       (kind) freezing a result as a saved model and reading it back
* ``kinds``        registering a saved kind without the add-on framework; the heavy job kinds
* ``routes``       (server) small checks of a request's numbers
"""
from __future__ import annotations

__version__ = "1.0.0"


def register(api) -> None:
    """Nothing to wire up: the other add-ons import this package."""
    api.log.info("common add-on %s ready", __version__)
