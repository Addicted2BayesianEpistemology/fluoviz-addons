"""The scans (jobs ``som-scan``): the map size, the stability, the choice of distance."""
from __future__ import annotations

from fastapi import HTTPException

from ..core import scans as C
from ..core.spec import Config
from .request import Request, parse_request

KINDS = ("size", "stability", "metric")


def parse_scan(p, body: dict) -> tuple:
    """``(kind, request, options)`` of a scan body: ``{"kind": "size", "request": {...}}`` or ``{"kind": ..., "result": "som-1"}`` (the settings of a result)."""
    if not isinstance(body, dict):
        raise HTTPException(400, "The scan must be sent as a group of named values.")
    kind = body.get("kind")
    if kind not in KINDS:
        raise HTTPException(400, f"The kind of scan must be one of: {', '.join(KINDS)}.")
    if body.get("result"):
        try:
            r = p.get_result(str(body["result"]))
        except ValueError:
            raise HTTPException(404, f"There is no result '{body['result']}'.") from None
        raw = dict(r.params or {})
    else:
        raw = dict(body.get("request") or {})
    raw.pop("replace", None)
    req = parse_request(raw)
    opts: dict = {}
    if kind == "size":
        units = body.get("units")
        if units is not None:
            if not (isinstance(units, list) and 2 <= len(units) <= 8 and all(isinstance(u, int) and not isinstance(u, bool) and 4 <= u <= 400 for u in units)):
                raise HTTPException(400, "'units' must be a list of 2 to 8 numbers of units, each from 4 to 400.")
            opts["units"] = units
    elif kind == "stability":
        n = body.get("n_runs", 10)
        if isinstance(n, bool) or not isinstance(n, int) or not 2 <= n <= 50:
            raise HTTPException(400, "'n_runs' must be a whole number from 2 to 50.")
        opts["n_runs"] = n
        opts["bootstrap"] = bool(body.get("bootstrap", False))
    k = body.get("k", 4)
    if isinstance(k, bool) or not isinstance(k, int) or not 2 <= k <= 20:
        raise HTTPException(400, "'k' (the number of groups compared) must be a whole number from 2 to 20.")
    opts["k"] = k
    return kind, req, opts


def run_scan(p, kind: str, req: Request, opts: dict, progress) -> dict:
    from . import prep
    progress(0.02, "Preparing the data")
    prepared = prep.get_prepared(p, req, progress.sub(0.0, 0.2))
    req.cfg.check_samples(prepared.n)
    sub = progress.sub(0.2, 1.0)
    if kind == "size":
        out = C.size_scan(prepared, req.cfg, opts.get("units"), sub)
    elif kind == "stability":
        out = C.stability(prepared, req.cfg, None, opts["n_runs"], opts["bootstrap"], opts["k"], sub)
    else:
        out = C.metric_compare(prepared, req.cfg, opts["k"], sub)
    return {"kind": kind, **out}
