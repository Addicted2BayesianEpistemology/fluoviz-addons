"""The API of the Self-organising map add-on, ``/api/addon/som/...`` (``docs/api.md`` says what every route takes and gives).  The router only checks the request and calls the modules of
this folder; the numerics are ``core`` and ``kinds`` (numpy / scipy).

    GET  /defaults                    facts and advice from the data, a recommended request
    POST /check            (job)      what a request would do (nothing is fitted)
    POST /fit              (job)      fit the map, save the result ``som-N``
    POST /scan             (job)      size | stability | metric
    GET  /{rid}                       the result for the page
    GET  /{rid}/sample/{sid}          one sample against the map
    GET  /{rid}/unit/{index}          one unit
    GET  /{rid}/prototypes            the prototype of every unit as pictures
    POST /{rid}/plane  GET /{rid}/planes  GET /{rid}/indices      planes
    GET  /{rid}/groups  POST /{rid}/cluster                       the groups
    POST /{rid}/trajectories | /anomalies,  GET /{rid}/projection | /methods | /export
"""
from __future__ import annotations

from fastapi import HTTPException
from fastapi.responses import Response

from fluoviz.analysis.fpca.variables import DEFAULT_MIN_COVERAGE
from fluoviz.server.analysis import project, result_public, start_job
from fluoviz.server.encoding import JsonRouter
from fluoviz.server.helpers import attachment_headers

from . import advice, exports, fitjob, scans, views
from .methods import REFERENCES, methods_text
from .request import parse_request
from .results import RESULT_KIND, get_result, turn_for_browser

router = JsonRouter(prefix="/api/addon/som", tags=["som"])


def _num_query(v, what: str, lo: float, hi: float) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise HTTPException(400, f"'{what}' must be a number.") from None
    if not lo <= x <= hi:
        raise HTTPException(400, f"'{what}' must be between {lo:g} and {hi:g}.")
    return x


# ----------------------------------------------------------------------------- advice and checks
@router.get("/defaults")
def defaults(variables: str = "fluorescence", min_coverage: float = DEFAULT_MIN_COVERAGE, abs_processed: bool = True, advise: int = 0):
    p = project()
    names = [v.strip() for v in variables.split(",") if v.strip()]
    if not names or any(v not in ("fluorescence", "absorbance") for v in names):
        raise HTTPException(400, "'variables' must be fluorescence, absorbance or both (separated by a comma).")
    _num_query(min_coverage, "min_coverage", 0.05, 1.0)
    return advice.defaults_view(p, names, float(min_coverage), bool(abs_processed), bool(advise))


@router.post("/check")
def check(body: dict):
    p = project()
    req = parse_request(body)

    def work(progress):
        return advice.check_view(p, req, progress)

    return start_job("som-check", "Checking the settings", work)


# ----------------------------------------------------------------------------- fitting
@router.post("/fit")
def fit(body: dict):
    p = project()
    req = parse_request(body)
    if req.replace is not None:
        get_result(p, req.replace)

    def work(progress):
        return fitjob.run_fit(p, req, progress)

    return start_job("som-fit", req.title or "Computing a self-organising map", work)


@router.post("/scan")
def scan(body: dict):
    p = project()
    kind, req, opts = scans.parse_scan(p, body)

    def work(progress):
        return scans.run_scan(p, kind, req, opts, progress)

    return start_job("som-scan", f"Map scan: {kind}", work)


@router.get("/{rid}")
def result(rid: str):
    p = project()
    r = get_result(p, rid)
    return {"result": result_public(p, r, with_payload=False), "payload": turn_for_browser(r.payload)}


@router.get("/{rid}/sample/{sid:path}")
def sample(rid: str, sid: str, thumb: int = 0):
    p = project()
    return views.sample_view(p, get_result(p, rid), sid, bool(thumb))


@router.get("/{rid}/unit/{index}")
def unit(rid: str, index: int):
    p = project()
    return views.unit_view(p, get_result(p, rid), index)


@router.get("/{rid}/prototypes")
def prototypes(rid: str, what: str = "prototype"):
    p = project()
    return views.prototypes_view(p, get_result(p, rid), what)


@router.post("/{rid}/plane")
def plane(rid: str, body: dict):
    p = project()
    r = get_result(p, rid)
    out = views.plane_view(p, r, body)
    if body.get("kind") != "stored" and body.get("keep", True) and out.get("key"):
        r.payload["planes"][out["key"]] = {k: v for k, v in out.items() if k != "cached"}
        p.dirty.add("results")
    return out


@router.get("/{rid}/planes")
def planes(rid: str):
    return views.planes_list(get_result(project(), rid))


@router.get("/{rid}/indices")
def indices(rid: str):
    return {"indices": views.index_list(get_result(project(), rid))}


@router.get("/{rid}/groups")
def groups(rid: str):
    p = project()
    return views.groups_view(p, get_result(p, rid))


@router.post("/{rid}/cluster")
def cluster(rid: str, body: dict):
    p = project()
    try:
        return views.recluster(p, get_result(p, rid), body)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None


@router.post("/{rid}/trajectories")
def trajectories(rid: str, body: dict):
    p = project()
    return views.trajectories_view(p, get_result(p, rid), body)


@router.get("/{rid}/projection")
def projection(rid: str):
    return views.projection_view(get_result(project(), rid))


@router.post("/{rid}/anomalies")
def anomalies(rid: str, body: dict):
    p = project()
    return views.anomalies_view(p, get_result(p, rid), body)


@router.get("/{rid}/methods")
def methods(rid: str):
    r = get_result(project(), rid)
    return {"text": methods_text(r), "references": REFERENCES}


@router.get("/{rid}/export")
def export(rid: str, what: str = "positions"):
    r = get_result(project(), rid)
    data, name, media = exports.export(r, what)
    return Response(data, media_type=media, headers=attachment_headers(name))
