"""The fit job: prepare the modes, sample the posterior in the Stan engine, summarise it, check that the saved-model side reads the training samples the same way, and save the
numbered result ``mblf-N``."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from fluoviz.analysis.common import Progress
from fluoviz.server.analysis import result_public, save_result

from ..core import fitting
from ..kinds.freeze import pack_frozen
from ..kinds.model import MblfModel, model_tree
from . import payload as PL
from . import prep
from .request import Request
from .results import RESULT_KIND

CONSISTENCY = 1e-6


def title_for(req: Request, prepared) -> str:
    what = " and ".join(m.label for m in prepared.modes)
    cov = f", covariates {', '.join(prepared.cov.labels)}" if prepared.cov is not None else ""
    est = "" if req.cfg.estimation == "bayes" else " (maximum likelihood)"
    return f"Latent factors of the {what} (up to {req.cfg.K}{cov}){est}"


def check_consistency(p, model: MblfModel, reference: dict) -> str | None:
    """Read a few complete training samples through the data recipe with the frozen numbers: their scores must be those of the fit.  Returns a note when the recipe cannot be used
    (the check is then skipped); raises when they disagree (a fault of the add-on)."""
    from fluoviz.core.recipe import RecipeError, capture_recipe, recipe_view
    pres = np.column_stack([np.isfinite(y).any(axis=0) for y in model.f.Y])
    complete = [i for i in range(len(model.ids)) if pres[i].all()]
    if not complete:
        return None
    pos = np.unique(np.linspace(0, len(complete) - 1, min(6, len(complete))).round().astype(int))
    pick = [complete[k] for k in pos]
    ids = [model.ids[i] for i in pick]
    try:
        recipe = capture_recipe(p, uses_eem="fluorescence" in model.names, uses_abs="absorbance" in model.names, ids=list(model.ids) if "fluorescence" in model.names else None)
        data = recipe_view(p, recipe, ids, None)
    except RecipeError as exc:
        return f"The saved-model check was skipped: {exc}"
    ev = model.evaluate(data, 0.0)
    stored = np.asarray(reference["scores"], float)[pick]
    scale = max(float(np.nanmax(np.abs(reference["scores"]))), 1e-12)
    err = float(np.nanmax(np.abs(ev.scores - stored))) / scale
    if not err < 1e-5:
        raise RuntimeError(f"Internal check failed: reading the training samples through the saved-model path gives other scores (difference {err:.1e}). This is a fault of the add-on: "
                           "please report it.")
    return None


def run_fit(p, req: Request, progress: Progress, data_dir: Path) -> dict:
    """The body of the job.  Returns ``{"result": ..., "rid": ...}``."""
    t0 = time.time()
    sig0 = p.signature()
    progress(0.01, "Preparing the data")
    prepared = prep.get_prepared(p, req, progress.sub(0.0, 0.08))
    req.cfg.check_samples(prepared.N)
    f, meta = fitting.fit_model(prepared.modes, prepared.Y, prepared.cov, req.cfg, data_dir, progress.sub(0.08, 0.85), ids=prepared.ids)
    progress(0.86, "Summarising the posterior: factors, shares of variance, predictions, leave-one-out")
    payload, variables, var_info, summary, reference = PL.assemble(prepared, req, f, meta, time.time() - t0)
    tree = model_tree(f, reference)
    model = MblfModel(tree)
    progress(0.95, "Checking that a saved model would read the samples the same way")
    note = check_consistency(p, model, reference)
    if note:
        payload["notes"].append(note)
    payload["frozen"] = pack_frozen(tree, {"n_samples": prepared.N, "elpd_loo": payload["loo"]["elpd"], "K": f.K, "estimation": req.cfg.estimation}, model)
    if p.signature() != sig0:
        raise ValueError("The data or the corrections changed while the model was fitted, so it was not saved (it would describe data that no longer exist). Fit it again.")
    title = req.title or title_for(req, prepared)
    if req.replace and req.replace in p.results and not req.title:
        title = p.results[req.replace].title
    r = save_result(p, RESULT_KIND, title, req.params(), payload, summary, sample_ids=list(prepared.ids), variables=variables, var_info=var_info, obj=model, data_sig=sig0,
                    rid=req.replace)
    progress(1.0, "Done")
    return {"result": result_public(p, r, with_payload=False), "rid": r.id}
