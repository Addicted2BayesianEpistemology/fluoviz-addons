"""The request of a fit, checked in plain English: the network settings are the add-on's ``Config``, the data settings (``input``) are the Functional PCA request, checked by the same
function the Functional PCA page uses (``routers/fpca.check_spec``).  Nothing is computed here."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

from ..core.spec import SIZES, AutoencoderError, Config

INPUT_KEYS = ("variables", "fluorescence", "absorbance")
MAX_TITLE = 120


@dataclass
class Request:
    cfg: Config                                  # every network setting, checked (``cfg.input`` = the checked data settings)
    input: dict                                  # {"variables": [...], "fluorescence": {transform, min_coverage, basis, smoothing}, "absorbance": {...}}
    title: str = ""
    replace: str | None = None
    notes: list = field(default_factory=list)

    @property
    def names(self) -> list[str]:
        return list(self.input["variables"])

    def params(self) -> dict:
        """The replayable request (what ``result.params`` holds): ``POST /fit`` with it gives the same network."""
        d = self.cfg.to_dict()
        d["hidden"] = self.cfg.hidden_widths()
        d["size"] = self.size
        d["input"] = copy.deepcopy(self.input)
        d["title"] = self.title
        return d

    @property
    def size(self) -> str:
        w = self.cfg.hidden_widths()
        for name, widths in SIZES.items():
            if list(widths) == w:
                return name
        return "custom"


def check_input(raw) -> dict:
    """The data settings of a request: the Functional PCA request's ``variables`` / ``fluorescence`` / ``absorbance`` blocks, checked as the Functional PCA page checks them."""
    from fluoviz.server.routers import fpca as R

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise AutoencoderError("The data settings ('input') must be a group of settings.")
    bad = sorted(set(raw) - set(INPUT_KEYS))
    if bad:
        raise AutoencoderError(f"Unknown data setting '{bad[0]}': use variables, fluorescence and absorbance (the same as the Functional PCA).")
    spec = R.check_spec({k: raw[k] for k in INPUT_KEYS if k in raw})
    return {"variables": list(spec["variables"]), **{n: spec[n] for n in spec["variables"]}}


def parse_request(body) -> Request:
    """``Request`` of a ``check`` / ``fit`` body; a plain-English ``ValueError`` (an ``AutoencoderError``) for anything wrong."""
    if not isinstance(body, dict):
        raise AutoencoderError("The settings must be sent as a group of named values.")
    b = dict(body)
    title = b.pop("title", "") or ""
    if not isinstance(title, str):
        raise AutoencoderError("The title must be text.")
    replace = b.pop("replace", None)
    if replace is not None and not isinstance(replace, str):
        raise AutoencoderError("'replace' must be the name of an earlier network result.")
    size = b.pop("size", None)
    if size is not None:
        if size == "custom":
            if isinstance(b.get("hidden", "small"), str):
                raise AutoencoderError("A custom size needs the list of layer widths ('hidden').")
        elif size in SIZES:
            b["hidden"] = size
        else:
            raise AutoencoderError(f"The network size must be one of: {', '.join(SIZES)}, custom.")
    inp = check_input(b.pop("input", None))
    cfg = Config.from_dict({**b, "input": inp})
    return Request(cfg=cfg, input=inp, title=title.strip()[:MAX_TITLE], replace=replace)
