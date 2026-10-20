"""The fit job: prepare the data, fit the map, group the units, check the result with the numpy code the saved model uses, and save the numbered result ``som-N``."""
from __future__ import annotations

import time

import numpy as np

from fluoviz.analysis.common import Progress
from fluoviz.server.analysis import result_public, save_result

from ..core import fitting, groups
from ..kinds.model import SomModel, model_tree
from . import payload as PL
from . import prep
from .request import Request
from .results import RESULT_KIND

CONSISTENCY = 1e-6


def title_for(req: Request, prepared, fit) -> str:
    what = " and ".join(prepared.names) if not req.variables_mode else f"{len(req.input['keys'])} variables"
    return f"{'Functional ' if prepared.mode in ('functional', 'mixed') else ''}map of the {what} ({fit.lattice.rows} × {fit.lattice.cols} units)"


def check_consistency(prepared, fit, ev) -> None:
    """The fit side and the saved-model side must agree: the training samples read from the project's data through the frozen description land on the same units at the same distances."""
    if not np.array_equal(ev.unit, fit.asg["c1"]):
        raise RuntimeError("Internal check failed: reading the samples through the frozen data description does not put them on the units of the fit. This is a fault of the add-on: please report it.")
    scale = max(float(np.nanmax(fit.asg["qe"])), 1e-300)
    d = float(np.nanmax(np.abs(ev.qe - fit.asg["qe"]))) / scale
    if not d < CONSISTENCY:
        raise RuntimeError(f"Internal check failed: the saved-model pass does not reproduce the distances of the fit (difference {d:.1e}). This is a fault of the add-on: please report it.")


def run_fit(p, req: Request, progress: Progress) -> dict:
    """The body of the job.  Returns ``{"result": ..., "rid": ...}``."""
    t0 = time.time()
    sig0 = p.signature()
    progress(0.01, "Preparing the data")
    prepared = prep.get_prepared(p, req, progress.sub(0.0, 0.15))
    req.cfg.check_samples(prepared.n)
    fit = fitting.fit_map(prepared, req.cfg, progress.sub(0.15, 0.85))
    progress(0.86, "Grouping the units")
    grp = groups.cluster_units(fit.protos_t, fit.asg["c1"], fit.lattice.adj, req.cfg, fit.sample_w)
    progress(0.9, "Checking the result")
    model = None
    if prepared.frame is not None:
        model = SomModel(model_tree(fit, prepared.frame, grp["unit_labels"]))
        ev = model.evaluate(prep.apply_data(p, list(prepared.ids), model.names), 0.0)
        if not ev.usable.all():
            raise RuntimeError("Internal check failed: some training samples cannot be read back from the project. This is a fault of the add-on: please report it.")
        check_consistency(prepared, fit, ev)
    else:
        ev = PL.evaluation_from_fit(fit, prepared)
    progress(0.94, "Computing the checks")
    pl, variables, var_info, summary = PL.assemble(p, req, prepared, fit, grp, ev, model, list(fit.notes), time.time() - t0)
    if p.signature() != sig0:
        raise ValueError("The data or the corrections changed while the map was computed, so it was not saved (it would describe data that no longer exist). Compute it again.")
    title = req.title or title_for(req, prepared, fit)
    if req.replace and req.replace in p.results and not req.title:
        title = p.results[req.replace].title
    r = save_result(p, RESULT_KIND, title, req.params(), pl, summary, sample_ids=list(prepared.ids), variables=variables, var_info=var_info, obj=model, data_sig=sig0, rid=req.replace)
    progress(1.0, "Done")
    return {"result": result_public(p, r, with_payload=False), "rid": r.id}
