"""The fit job: prepare the data, train the network in a separate process, check the result with the numpy code the saved model uses, and save the numbered result ``autoencoder-N``."""
from __future__ import annotations

import time

import numpy as np

from fluoviz.analysis.common import Progress
from fluoviz.server.analysis import result_public, save_result

from ..kinds.model import AEModel, model_tree
from . import payload as PL
from . import prep, trainer
from .request import Request
from .results import RESULT_KIND

CONSISTENCY = 1e-6


def default_title(req: Request, model: AEModel) -> str:
    fam = "VAE" if model.is_vae else "AE"
    functional = any(b.kind != "pixels" for b in model.frame.blocks)
    what = " and ".join(model.names)
    return f"{'Functional ' if functional else ''}{fam} of the {what} ({model.d} code{'s' if model.d != 1 else ''})"


def check_consistency(prepared, ev, res) -> None:
    """The training side and the saved-model side must agree: the network coordinates of the training samples read from the project's data through the frozen description (what a new
    sample goes through) equal the ones the network was trained on, and the codes equal the ones the training reported.  A difference is a bug, never silently ignored."""
    X, Xt = ev.X, prepared.X
    scale = max(float(np.nanmax(np.abs(Xt))), 1e-300)
    d = float(np.nanmax(np.abs(np.where(np.isfinite(Xt), Xt, 0.0) - np.where(np.isfinite(X), X, 0.0)))) / scale
    if not d < CONSISTENCY:
        raise RuntimeError(f"Internal check failed: reading the samples through the frozen data description does not give the numbers the network was trained on (difference {d:.1e}). "
                           "This is a fault of the add-on: please report it.")
    c0 = np.asarray(res.samples["codes"], float)
    dc = float(np.nanmax(np.abs(ev.codes - c0))) / max(float(np.nanmax(np.abs(c0))), 1e-300)
    if not dc < CONSISTENCY:
        raise RuntimeError(f"Internal check failed: the numpy network does not reproduce the codes of the training (difference {dc:.1e}). This is a fault of the add-on: please report it.")


def run_fit(p, req: Request, progress: Progress, history_path) -> dict:
    """The body of the job (see the module text).  Returns ``{"result": ..., "rid": ...}``."""
    t0 = time.time()
    sig0 = p.signature()
    progress(0.01, "Preparing the data")
    prepared = prep.get_prepared(p, req, progress.sub(0.0, 0.12))
    req.cfg.check_samples(prepared.n)
    res = trainer.train(req.cfg, prepared, history_path, progress.sub(0.12, 0.92))
    progress(0.93, "Checking the result")
    ids = list(prepared.ids)
    model = AEModel(model_tree(res.spec, res.arrays, prepared.frame))
    ev = model.evaluate(prep.apply_data(p, ids, model.names), 0.0)
    if not ev.usable.all():
        raise RuntimeError("Internal check failed: some training samples cannot be read back from the project. This is a fault of the add-on: please report it.")
    check_consistency(prepared, ev, res)
    progress(0.96, "Computing the checks")
    pl, variables, var_info, summary = PL.assemble(p, req, prepared, res, ids, ev, list(res.notes), time.time() - t0)
    if p.signature() != sig0:
        raise ValueError("The data or the corrections changed while the network was training, so it was not saved (it would describe data that no longer exist). Train it again.")
    title = req.title or default_title(req, model)
    if req.replace and req.replace in p.results and not req.title:
        title = p.results[req.replace].title
    pl["title"] = title
    r = save_result(p, RESULT_KIND, title, req.params(), pl, summary, sample_ids=ids, variables=variables, var_info=var_info, obj=model, data_sig=sig0, rid=req.replace)
    progress(1.0, "Done")
    return {"result": result_public(p, r, with_payload=False), "rid": r.id}
