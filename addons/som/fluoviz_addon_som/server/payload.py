"""What a fit stores in the result ``som-N`` and publishes as variables (module orientation inside; ``results.turn_for_browser`` turns the images for the page)."""
from __future__ import annotations

import numpy as np

from fluoviz.analysis.frozen.kinds import _decomp as D

from ..core import diagnostics, planes
from ..core.spec import METRIC_WORDS
from ..kinds import som as KS
from ..kinds.freeze import pack_frozen
from ..kinds.model import Evaluation, MapView, SomModel, model_tree


def evaluation_from_fit(fit, prep) -> Evaluation:
    """The training samples as an :class:`Evaluation` (used for a map of other variables, which has no functions to read samples back through)."""
    n = fit.n
    a = fit.asg
    return Evaluation(n, np.ones(n, bool), [[] for _ in range(n)], np.ones((n, len(prep.names))), np.zeros(n, bool), {}, prep.R, a["c1"], a["c2"], a["qe"], a["qe_blocks"], a["qe_rel"], fit.pos,
                      np.zeros(n, bool))


def limits_text(ref: dict, lev: str) -> list:
    return [D.limit_entry(f"Distance to the best unit, {lev} % limit", ref["limits"][lev], ref["limits"].get("words", ""))]


def assemble(p, req, prepared, fit, grp: dict, ev: Evaluation, model, notes: list, seconds_total: float) -> tuple[dict, dict, dict, dict]:
    """``(payload, variables, var_info, summary)`` of a fitted map.  ``ev``: the training samples through the saved-model pass (``SomModel.evaluate`` on the project data) or, for a map
    of other variables, straight from the fit; ``model``: the :class:`SomModel` (None for a map of other variables)."""
    cfg = req.cfg
    n = prepared.n
    ids = list(prepared.ids)
    lat = fit.lattice
    M = lat.M
    kf = fit.kfold
    held = np.asarray(kf["qe"], float)
    unit_cluster = grp["unit_labels"]
    tree = None
    if model is not None:
        model.unit_cluster = np.asarray(unit_cluster, int)
        tree = model_tree(fit, prepared.frame, unit_cluster)
        view = model
    else:
        view = MapView(lat, fit.protos_t, fit.hits, unit_cluster, prepared.names)
    ref = KS.reference_for(view, ids, ev, held)
    opt = D.settings(KS.options_for(), None)
    opt["limit"] = str(cfg.limit)
    jd = KS.judge(view, ev, ref, opt)
    variables, var_info = KS.publish(view, ev, jd)
    variables["heldout_qe"] = held
    var_info["heldout_qe"] = {"kind": "numeric", "label": "Map: distance to the best unit when the sample was held out", "description":
                              "The distance of the sample to the best unit of a map trained WITHOUT it (k-fold validation): an honest measure of how well the map describes a sample it has not seen."}
    lacking = prepared.missing_part()
    if lacking is not None:
        variables["missing_part"] = lacking
        var_info["missing_part"] = {"kind": "numeric", "label": "Map: share of the domain not measured", "description":
                                    "The share of the domain (by area) that was missing for this sample when the map was computed, for example a region excised from it."}
    verd = diagnostics.all_verdicts(fit.quality, fit.history)
    q = dict(fit.quality)
    # ---- the planes
    mean_t = fit.metric.transform(np.zeros((1, prepared.P)))[0]
    pl = planes.basic_planes(fit.protos_t, mean_t, fit.hits, fit.umat, fit.asg["qe"], fit.asg["c1"])
    pl.update(planes.score_planes(prepared.R, fit.protos_raw, prepared.l2_gain(), fit.sample_w, 8))
    scores = planes.sample_scores(prepared.R, prepared.l2_gain(), fit.sample_w, 4)
    # ---- pictures
    payload: dict = {
        "ids": ids, "n": n, "mode": prepared.mode, "variables": list(prepared.names), "title": "", "kind": "variables" if model is None else "functions",
        "input": {"variables": list(prepared.names), "mode": prepared.mode, "n_inputs": prepared.P, "blocks": prepared.summary()["blocks"], "dropped": [list(x) for x in prepared.dropped],
                  "weights": {b.name: float(b.weight) for b in prepared.blocks}},
        "map": {"rows": lat.rows, "cols": lat.cols, "lattice": lat.kind, "shape": lat.shape, "units": M, "coords": lat.coords, "edges": fit.edges, "umatrix": fit.umat,
                "hits": fit.hits.astype(int), "hits_weight": fit.hits_w, "unit_names": [str(x) for x in view.unit_name(np.arange(M))], "period": None if lat.period is None else list(lat.period)},
        "metric": {**fit.metric.to_dict(), "name": METRIC_WORDS[fit.metric.kind], "components_auto": cfg.components == "auto"},
        "samples": {"unit": fit.asg["c1"].astype(int), "unit2": fit.asg["c2"].astype(int), "pos": fit.pos, "qe": fit.asg["qe"], "qe_blocks": fit.asg["qe_blocks"], "qe_rel": fit.asg["qe_rel"],
                    "heldout_qe": held, "heldout_qe_rel": np.asarray(kf["qe_rel"], float), "fold": np.asarray(kf["fold"], int), "flag": jd.flag, "status": jd.status, "scores": scores,
                    "missing": lacking, "weight": fit.sample_w},
        "planes": pl, "quality": q, "verdicts": verd, "training": {"history": {k: list(v) for k, v in fit.history.items()}, "energy": fit.energy, "epochs": len(fit.history.get("energy", []))},
        "kfold": {"folds": int(kf["folds"]), "empty_unit_share": float(np.mean(kf["empty_unit"])) if kf["folds"] else float("nan")},
        "clusters": cluster_payload(grp, cfg), "limits": limits_text(ref, str(cfg.limit)),
        "notes": list(dict.fromkeys(list(prepared.notes))), "warnings": list(dict.fromkeys(list(notes) + list(grp.get("warnings", [])))),
        "versions": {"seed": cfg.seed}, "n_outside": int(jd.err_out.sum() + (jd.empty & ~jd.err_out).sum()),
    }
    if model is None:
        payload["prototype_values"] = np.asarray(fit.protos_raw, np.float32)
        payload["columns"] = list(prepared.blocks[0].columns or [])
        payload["map"]["protos_t"] = np.asarray(fit.protos_t, np.float32)
    payload["map"]["proto_dist"] = np.asarray(fit.proto_dist, np.float32)
    if model is not None:
        funcs = prepared.frame.to_functions(fit.protos_raw)
        mean = prepared.frame.mean_functions()
        payload["prototypes"], payload["prototype_curves"], payload["mean_image"], payload["mean_curve"] = {}, {}, {}, {}
        for b in prepared.frame.blocks:
            fe = b.element
            if fe.kind == "surface":
                payload["prototypes"][b.name] = b.domain.embed(funcs[b.name]).astype(np.float32)
                payload["mean_image"][b.name] = b.domain.embed(mean[b.name])[0].astype(np.float32)
                payload["em"], payload["ex"] = np.asarray(fe.axes[0], float), np.asarray(fe.axes[1], float)
                payload["mask"] = ~np.asarray(fe.obs, bool)
                payload["units"] = str(fe.units or "")
            else:
                payload["prototype_curves"][b.name] = b.domain.embed(funcs[b.name]).astype(np.float32)
                payload["mean_curve"][b.name] = b.domain.embed(mean[b.name])[0].astype(np.float32)
                payload["abs_wl"] = np.asarray(fe.axes[0], float)
        # the contract with Clustering & LDA: a direction of the map's plain L2 PCA as a fluorescence matrix is not offered (the positions are the scores)
        val = {"n_samples": n, "captured": q["captured"], "heldout_captured": q.get("heldout_captured"), "topographic_error": q["topographic_error"], "n_units": M, "n_empty": q["n_empty"]}
        payload["frozen"] = pack_frozen(tree, ref, val, KS.description_for(model, val), model)
    summary = {"n": n, "mode": prepared.mode, "variables": list(prepared.names), "rows": lat.rows, "cols": lat.cols, "units": M, "metric": fit.metric.kind, "captured": q["captured"],
               "heldout_captured": q.get("heldout_captured"), "topographic_error": q["topographic_error"], "trust": q["trust_map"]["trustworthiness"], "trust_pca": q["trust_pca"]["trustworthiness"],
               "verdict": verd["vs_pca"]["verdict"], "n_flagged": payload["n_outside"], "groups": grp["k"], "seed": cfg.seed, "seconds": float(seconds_total)}
    return payload, variables, var_info, summary


def cluster_payload(grp: dict, cfg) -> dict:
    return {"k": int(grp["k"]), "unit_labels": np.asarray(grp["unit_labels"], int), "sample_labels": np.asarray(grp["sample_labels"], int), "sizes": list(grp["sizes"]), "reason": grp["reason"],
            "k_values": list(grp.get("k_values", [])), "silhouette": list(grp.get("silhouette", [])), "skipped_empty": bool(grp.get("skipped_empty")), "warnings": list(grp.get("warnings", [])),
            "method": cfg.cluster_method, "settings": {"method": cfg.cluster_method, "k": cfg.cluster_k, "k_range": list(cfg.cluster_k_range), "skip_empty": cfg.cluster_skip_empty}}
