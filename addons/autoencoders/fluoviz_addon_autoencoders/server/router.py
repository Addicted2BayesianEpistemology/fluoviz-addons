"""The API of the Autoencoders add-on, ``/api/addon/autoencoders/...`` (``docs/api.md`` says what every route takes and gives).  The router only checks the request and calls the modules
of this folder; the numerics are ``core`` (training, in a separate process) and ``kinds`` (numpy).  The FluoViz server process never imports PyTorch.

    GET  /defaults                    facts and advice from the data, a recommended request
    POST /check            (job)      what a request would do (nothing is trained)
    POST /fit              (job)      train the network, save the result ``autoencoder-N``
    GET  /fit/{job}/history           the live training history of a fit / scan
    POST /scan             (job)      latent | smoothify | stability | baseline
    GET  /{rid}                       the result for the page
    GET  /{rid}/sample/{sid}          one sample against the network
    POST /{rid}/traverse | /interpolate | /generate | /anomalies | /relate,  GET /{rid}/sensitivity | /methods | /export
    POST /{rid}/response | /bends | /local | /pair      the nonlinear effects of the codes (``server/nonlinear.py``)
"""
from __future__ import annotations

from fastapi import HTTPException
from fastapi.responses import Response
from fluoviz_addon_common.routes import num_query as _num_query

from fluoviz.analysis.fpca.variables import DEFAULT_MIN_COVERAGE
from fluoviz.server.analysis import project, result_public, start_job
from fluoviz.server.encoding import JsonRouter
from fluoviz.server.helpers import attachment_headers
from fluoviz.server.state import get_state

from . import advice, exports, fitjob, history, nonlinear, prep, scans, views
from .methods import REFERENCES, methods_text
from .request import parse_request
from .results import RESULT_KIND, frozen_of, get_result, model_of, turn_for_browser

router = JsonRouter(prefix="/api/addon/autoencoders", tags=["autoencoders"])


def _bool(v, what: str) -> bool:
    if isinstance(v, bool):
        return v
    raise HTTPException(400, f"'{what}' must be true or false.")


# ----------------------------------------------------------------------------- advice and checks
@router.get("/defaults")
def defaults(variables: str = "fluorescence", min_coverage: float = DEFAULT_MIN_COVERAGE, abs_processed: bool = True, advise: int = 0):
    project()
    names = [v.strip() for v in variables.split(",") if v.strip()]
    bad = [v for v in names if v not in ("fluorescence", "absorbance")]
    if bad or not names:
        raise HTTPException(400, "'variables' must be fluorescence, absorbance or both (separated by a comma).")
    _num_query(min_coverage, "min_coverage", 0.05, 1.0)
    return advice.defaults_view(project(), names, float(min_coverage), bool(abs_processed), bool(advise))


@router.post("/check")
def check(body: dict):
    p = project()
    req = parse_request(body)

    def work(progress):
        return advice.check_view(p, req, progress)

    return start_job("autoencoder-check", "Checking the settings", work)


# ----------------------------------------------------------------------------- training
@router.post("/fit")
def fit(body: dict):
    p = project()
    req = parse_request(body)
    prep.require_torch()
    if req.replace is not None:
        get_result(p, req.replace)
    fh = history.new_fit()

    def work(progress):
        try:
            return fitjob.run_fit(p, req, progress, fh.path)
        finally:
            history.finish(fh)

    job = start_job("autoencoder", req.title or "Training an autoencoder", work)
    history.attach(fh, job["job_id"])
    return job


@router.get("/fit/{job_id}/history")
def fit_history(job_id: str, since: int = 0):
    fit_ = history.FITS.get(job_id)
    if fit_ is None:
        raise HTTPException(404, "There is no training history for this job (it may be too old, or it was not a fit).")
    if since < 0:
        raise HTTPException(400, "'since' must be zero or more.")
    return history.history_view(fit_, get_state().jobs.get(job_id), int(since))


@router.post("/scan")
def scan(body: dict):
    p = project()
    spec = scans.parse_scan(p, body)
    if spec.kind != "baseline":                                      # the linear baselines need no network
        prep.require_torch()
    fh = history.new_fit()

    def work(progress):
        try:
            return scans.run_scan(p, spec, progress, fh.path)
        finally:
            history.finish(fh)

    job = start_job("autoencoder-scan", f"Autoencoder scan: {spec.kind}", work)
    history.attach(fh, job["job_id"])
    return job


# ----------------------------------------------------------------------------- the result
@router.get("/{rid}")
def result(rid: str):
    p = project()
    r = get_result(p, rid)
    model = model_of(r)
    return {"result": result_public(p, r, with_payload=False), "payload": turn_for_browser(r.payload, model)}


@router.get("/{rid}/sample/{sid:path}")
def sample(rid: str, sid: str, thumb: int = 0):
    p = project()
    return views.sample_view(p, get_result(p, rid), sid, bool(thumb))


@router.post("/{rid}/traverse")
def traverse(rid: str, body: dict):
    return views.traverse_view(get_result(project(), rid), body)


@router.post("/{rid}/interpolate")
def interpolate(rid: str, body: dict):
    p = project()
    return views.interpolate_view(p, get_result(p, rid), body)


@router.post("/{rid}/generate")
def generate(rid: str, body: dict):
    return views.generate_view(get_result(project(), rid), body)


@router.get("/{rid}/sensitivity")
def sensitivity(rid: str):
    return views.sensitivity_view(get_result(project(), rid))


@router.post("/{rid}/anomalies")
def anomalies(rid: str, body: dict):
    p = project()
    return views.anomalies_view(p, get_result(p, rid), body)


@router.post("/{rid}/relate")
def relate(rid: str, body: dict):
    p = project()
    return scans.relate_view(p, get_result(p, rid), body)


# ----------------------------------------------------------------------------- which nonlinear trend does each code fit?
@router.post("/{rid}/response")
def response(rid: str, body: dict):
    p = project()
    return nonlinear.response_view(p, get_result(p, rid), body)


@router.post("/{rid}/bends")
def bends(rid: str, body: dict):
    return nonlinear.bends_view(get_result(project(), rid), body)


@router.post("/{rid}/local")
def local(rid: str, body: dict):
    return nonlinear.local_view(get_result(project(), rid), body)


@router.post("/{rid}/pair")
def pair(rid: str, body: dict):
    p = project()
    return nonlinear.pair_view(p, get_result(p, rid), body)


@router.get("/{rid}/methods")
def methods(rid: str):
    r = get_result(project(), rid)
    return {"text": methods_text(r, model_of(r)), "references": REFERENCES}


@router.get("/{rid}/export")
def export(rid: str, what: str = "codes", n: int = 24, seed: int = 1, temperature: float = 1.0, format: str = "npz"):
    p = project()
    r = get_result(p, rid)
    base = exports.slug(r.id, "autoencoder")
    if what == "codes":
        return Response(exports.codes_csv(r), media_type="text/csv", headers=attachment_headers(f"{base}_codes.csv"))
    if what == "history":
        return Response(exports.history_csv(r), media_type="text/csv", headers=attachment_headers(f"{base}_history.csv"))
    if what == "reconstruction":
        return Response(exports.reconstruction_npz(p, r), media_type="application/octet-stream", headers=attachment_headers(f"{base}_reconstruction.npz"))
    if what == "synthetic":
        if not 1 <= n <= 1000:
            raise HTTPException(400, "'n' (how many samples to draw) must be a whole number from 1 to 1000.")
        if format not in ("npz", "csv"):
            raise HTTPException(400, "'format' must be npz or csv.")
        if not 0 <= seed < 2 ** 31:
            raise HTTPException(400, "'seed' must be a whole number from 0 to 2147483647.")
        if not 0.0 <= temperature <= 2.0:
            raise HTTPException(400, "'temperature' must be a number from 0 to 2.")
        data, media, suffix = exports.synthetic_export(r, int(n), int(seed), float(temperature), format)
        return Response(data, media_type=media, headers=attachment_headers(f"{base}_synthetic.{suffix}"))
    raise HTTPException(400, "what must be codes, reconstruction, history or synthetic.")
