"""The API of the multimodal latent factor add-on, ``/api/addon/mblf/...`` (``docs/api.md`` says what every route takes and gives).  The router only checks the request and calls
the modules of this folder; the numerics are ``core`` and ``kinds``.

    GET    /engine                      the Stan engine: ready / missing / unavailable, and why
    POST   /engine/prepare   (job)      prepare it (downloads CmdStan and a compiler once, into the add-on's own folder)
    DELETE /engine                      remove it
    GET    /defaults                    facts about the data and a recommended request
    POST   /check            (job)      what a request would do (the bases are built, nothing is fitted)
    POST   /fit              (job)      fit, save the result ``mblf-N``
    GET    /{rid}                       the result for the page
    GET    /{rid}/sample/{sid}          the prediction of one mode of one sample (sources, covariates, noisy, held out)
    GET    /{rid}/predictable           the predictable fraction of a target from chosen sources, point by point
    GET    /{rid}/correlation           the correlation of one point with every point of a mode
    GET    /{rid}/contrast              the difference of every mode between two categories of a covariate
    GET    /{rid}/category              a category as a sample: its expected function ("pure" or "typical" member) and the prediction of a new member
    POST   /{rid}/compare    (job)      held-out comparison of every choice of sources for the targets
    GET    /{rid}/samples   /methods   /export
"""
from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import Response

from fluoviz.server.analysis import project, result_public, start_job
from fluoviz.server.encoding import JsonRouter
from fluoviz.server.helpers import attachment_headers

from ..core import engine as E
from ..core.spec import PRESETS, PRIORS, MblfError
from . import fitjob, views
from .methods import REFERENCES, methods_text
from .request import parse_request
from .results import for_browser, get_result

router = JsonRouter(prefix="/api/addon/mblf", tags=["mblf"])
DATA_DIR: Path | None = None             # set by register(api): api.data_dir


def data_dir() -> Path:
    if DATA_DIR is None:
        raise HTTPException(500, "The add-on was not started properly (no data folder).")
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    return Path(DATA_DIR)


def _err(fn):
    try:
        return fn()
    except MblfError as exc:
        raise HTTPException(400, str(exc)) from None


# ----------------------------------------------------------------------------- the engine
@router.get("/engine")
def engine(size: int = 0):
    return E.locate(data_dir(), bool(size)).public()


@router.post("/engine/prepare")
def engine_prepare():
    d = data_dir()
    eng = E.locate(d)
    if eng.state == "unavailable":
        raise HTTPException(400, eng.message)

    def work(progress):
        try:
            return E.prepare(d, progress).public()
        except E.EngineError as exc:
            raise ValueError(str(exc)) from None

    return start_job("mblf-engine", "Preparing the Stan engine", work)


@router.delete("/engine")
def engine_remove():
    try:
        E.remove(data_dir())
    except E.EngineError as exc:
        raise HTTPException(400, str(exc)) from None
    return E.locate(data_dir()).public()


# ----------------------------------------------------------------------------- advice and checks
@router.get("/defaults")
def defaults():
    p = project()
    included = set(p.included_ids())
    n_eem = len(included)
    n_abs = 0
    if p.abs is not None:
        _wl, _A, ids = p.abs_view()
        n_abs = len([s for s in ids if s in included])
    meta = []
    for col in list(p.meta.columns):
        kind = p.meta_types.get(col, "text")
        vals = p.meta.loc[[s for s in p.ids if s in included], col] if len(p.meta.index) else []
        n = int(sum(1 for v in vals if v is not None and v == v and v != ""))
        meta.append({"key": f"meta:{col}", "label": col, "kind": "numeric" if kind == "numeric" else "category", "n": n, "levels": None if kind == "numeric" else len(set(map(str, vals)))})
    eng = E.locate(data_dir()).public()
    return {"n_samples": n_eem, "fluorescence": {"available": n_eem > 0, "n": n_eem}, "absorbance": {"available": n_abs > 0, "n": n_abs}, "meta": meta, "engine": eng,
            "presets": PRESETS, "priors": {k: {"default": v[0], "min": v[1], "max": v[2], "label": v[3]} for k, v in PRIORS.items()},
            "request": {"input": {"variables": ["fluorescence", "absorbance"] if n_abs else ["fluorescence"], "values": [], "covariates": [], "partial": True},
                        "model": {"K": 5, "estimation": "bayes"}, "sampler": {"preset": "standard"}, "seed": 1}}


@router.post("/check")
def check(body: dict):
    p = project()
    req = _err(lambda: parse_request(body))

    def work(progress):
        return views.check_view(p, req, progress)

    return start_job("mblf-check", "Checking the settings", work)


# ----------------------------------------------------------------------------- fitting
@router.post("/fit")
def fit(body: dict):
    p = project()
    req = _err(lambda: parse_request(body))
    if req.replace is not None:
        get_result(p, req.replace)
    d = data_dir()
    eng = E.locate(d)
    if eng.state != "ready":
        raise HTTPException(400, eng.message)

    def work(progress):
        try:
            return fitjob.run_fit(p, req, progress, d)
        except E.EngineError as exc:
            raise ValueError(str(exc)) from None

    return start_job("mblf-fit", req.title or "Fitting a multimodal latent factor model", work)


@router.get("/{rid}")
def result(rid: str):
    p = project()
    r = get_result(p, rid)
    return {"result": result_public(p, r, with_payload=False), "payload": for_browser(r.payload)}


@router.get("/{rid}/sample/{sid:path}")
def sample(rid: str, sid: str, target: str | None = None, sources: str | None = "auto", covariates: int = 1, noisy: int = 0, held_out: int = 1):
    p = project()
    return views.sample_view(p, get_result(p, rid), sid, target, sources, bool(covariates), bool(noisy), bool(held_out))


@router.get("/{rid}/predictable")
def predictable(rid: str, target: str, sources: str | None = "auto", covariates: int = 1):
    return views.predictable_route(get_result(project(), rid), target, sources, bool(covariates))


@router.get("/{rid}/contrast")
def contrast(rid: str, key: str, a: str, b: str):
    return views.contrast_route(get_result(project(), rid), key, a, b)


@router.get("/{rid}/category")
def category(rid: str, key: str, level: str, how: str = "pure", target: str | None = None, noisy: int = 0):
    return views.category_route(get_result(project(), rid), key, level, how, target, bool(noisy))


@router.get("/{rid}/correlation")
def correlation(rid: str, a: str, b: str, x: float | None = None, y: float | None = None, j: int | None = None):
    return views.correlation_route(get_result(project(), rid), a, b, x, y, j)


@router.post("/{rid}/compare")
def compare(rid: str, body: dict | None = None):
    p = project()
    r = get_result(p, rid)
    body = body or {}
    targets = body.get("targets")
    if targets is not None and (not isinstance(targets, list) or not all(isinstance(t, str) for t in targets)):
        raise HTTPException(400, "'targets' must be a list of mode names.")
    mp = body.get("max_points", 300)
    if isinstance(mp, bool) or not isinstance(mp, int) or not 10 <= mp <= 5000:
        raise HTTPException(400, "'max_points' must be a whole number from 10 to 5000.")

    def work(progress):
        return views.compare_job(p, r, targets, mp, progress)

    return start_job("mblf-compare", "Comparing what predicts what (held out)", work)


@router.get("/{rid}/samples")
def samples(rid: str):
    return views.samples_view(get_result(project(), rid))


@router.get("/{rid}/methods")
def methods(rid: str):
    r = get_result(project(), rid)
    return {"text": methods_text(r), "references": REFERENCES}


@router.get("/{rid}/export")
def export(rid: str, what: str = "scores"):
    r = get_result(project(), rid)
    data, name, media = views.export(r, what)
    return Response(data, media_type=media, headers=attachment_headers(name))
