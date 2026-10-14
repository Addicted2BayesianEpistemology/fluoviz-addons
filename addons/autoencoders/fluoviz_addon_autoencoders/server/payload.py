"""The result of a fit: the payload the page reads (``GET /{rid}``), the frozen block a saved model is made from, the published variables and the words of the verdicts.

Everything is stored in the result (arrays split out when the project is saved, no pickles, no model object): the weights, the codes, the training curves, the checks.  Images and masks are
kept in MODULE orientation ``(n_em, n_ex)``; ``results.turn_for_browser`` turns them for the browser.  Plain English in every verdict and note.
"""
from __future__ import annotations

import numpy as np

from fluoviz.analysis.frozen.kinds import _decomp as D

from ..core import diagnostics as DG
from ..kinds import autoencoder as AE
from ..kinds import stats as S
from ..kinds import tools as T
from ..kinds.freeze import pack_frozen
from ..kinds.model import AEModel, Evaluation, model_tree

MAX_WORST = 10
HISTORY_KEYS = ("restart", "fold", "epoch", "train", "val", "rec", "rec_val", "kl", "kl_val", "smooth", "smooth_val", "beta", "sigma2", "lr")


def _pct(x) -> str:
    return "not available" if x is None or not np.isfinite(float(x)) else f"{100 * float(x):.1f} %"


# ----------------------------------------------------------------------------- training curves and the verdicts (every threshold and verdict wording is ``core.diagnostics``')
def training_block(res, split: dict) -> dict:
    rows = list(res.history)
    cols = {k: [r.get(k) for r in rows] for k in HISTORY_KEYS if any(k in r for r in rows)}
    for k in ("restart", "fold"):
        if k in cols:
            cols[k] = [(-1 if v is None else v) for v in cols[k]]
    tv = DG.training_verdict(res)
    runs = (res.restarts or {}).get("runs", [])
    kept = (res.restarts or {}).get("kept", 0)
    stopped = runs[kept].get("stopped", "max") if runs and kept < len(runs) else "max"
    stopping = tv["stopping_words"]
    spread = {k: (res.restarts or {}).get(k) for k in ("n", "kept", "metric", "mean", "sd", "min", "max")}
    verdict = " ".join([tv["quality_words"], tv["overfit_words"], stopping, tv["restarts_words"]])
    return {"history": cols, "best_epoch": int(res.best_epoch), "epochs_run": int(res.epochs_run), "stopped": stopped,
            "restarts": [{"seed": r.get("seed"), "val": r.get("evr_val", r.get("evr_train")), "best_epoch": r.get("best_epoch"), "chosen": bool(r.get("chosen")), "epochs_run": r.get("epochs_run")} for r in runs],
            "spread": spread, "overfit": {"level": tv["overfit"], "gap": tv["gap"], "words": tv["overfit_words"]}, "stopping": stopping, "restarts_words": tv["restarts_words"],
            "verdict": verdict, "fold_info": list(res.fold_info or []), "seconds": float(res.seconds)}


# ----------------------------------------------------------------------------- quality
def quality_block(res, prepared, model: AEModel, ev: Evaluation, ids: list, heldout: np.ndarray, err_held: np.ndarray) -> dict:
    v = res.validation
    base = v.get("baseline") or {}
    r2_val, r2_tr = v.get("evr"), v.get("evr_train")
    names = model.names
    by_block = {nm: {"val": (v.get("evr_blocks") or {}).get(nm, {}).get("val"), "train": (v.get("evr_blocks") or {}).get(nm, {}).get("train")} for nm in names}
    held = np.flatnonzero(heldout)
    kfold = v.get("mode") == "kfold"
    rows = held if (len(held) >= 5 and not kfold) else np.arange(ev.n)
    rmse_val = float(np.sqrt(np.nanmean(ev.rmse[held] ** 2))) if (not kfold and held.size) else None
    rmse_by_block = {nm: float(np.sqrt(np.nanmean(ev.rmse_blocks[held, k] ** 2))) for k, nm in enumerate(names)} if (not kfold and held.size) else {}
    shown = np.where(np.isfinite(err_held), err_held, ev.error)
    rec = DG.reconstruction(prepared, res.spec, res.arrays, rows)
    local = {nm: prepared.embed(nm, np.asarray(rec["local_r2"][nm], float)) for nm in names}
    rms = {nm: prepared.embed(nm, np.asarray(rec["residual_rms"][nm], float)) for nm in names}
    cmp = DG.compare_baseline(res)
    n_val = int(v.get("n_val", held.size))
    return {"mode": v.get("mode"), "r2_val": r2_val, "r2_train": r2_tr, "r2_by_block": by_block, "rmse_val": rmse_val, "rmse_by_block": rmse_by_block, "rmse_all": float(np.sqrt(np.nanmean(ev.rmse ** 2))),
            "worst": DG.worst_samples(ids, shown, heldout, MAX_WORST),
            "baseline": {"name": cmp["name"], "k": int(cmp["k"]), "r2_val": cmp.get("evr_baseline"), "r2_train": base.get("evr_train"), "difference": None if cmp.get("evr_network") is None or cmp.get("evr_baseline") is None
                         else float(cmp["evr_network"] - cmp["evr_baseline"]), "better": cmp["better"], "relative_improvement": cmp["relative_improvement"], "z": cmp["z"], "verdict": cmp["verdict"]},
            "local_r2": local, "residual_rms": rms, "local_on": "held-out samples" if len(rows) < ev.n else "all samples", "verdict": DG.quality_verdict(float(r2_val), n_val, model.d)}


# ----------------------------------------------------------------------------- the latent space
def latent_block(res, prepared, model: AEModel, ev: Evaluation, ref: dict) -> dict:
    out = dict(DG.latent_health(res.spec, res.arrays, prepared.X, prepared.M, ev.codes, ev.logvar))
    d, n = model.d, ev.n
    p = ref.get("typicality")
    out["typicality"] = {"covariance_ok": bool(np.isfinite(ref["code_stats"]["cov"]).all()), "median": float(np.nanmedian(p)) if p is not None and np.isfinite(p).any() else None,
                         "words": "The typicality of a code is its p-value among the training codes (1 = in the middle, near 0 = far out)."}
    if d > 2 and n > 2:
        Z = ev.codes
        w, V = np.linalg.eigh(np.atleast_2d(np.cov(Z, rowvar=False)))
        order = np.argsort(-w)[:2]
        out["pca"] = {"scores": (Z - Z.mean(axis=0)) @ V[:, order], "explained": w[order] / max(w.sum(), 1e-300)}
    return out


# ----------------------------------------------------------------------------- smoothness
def smoothness_block(prepared, model: AEModel, ev: Evaluation, cfg) -> dict:
    """The smoothness check: ``diagnostics.smoothness_report`` (roughness of the reconstructions against that of the samples, with its own rule and words)."""
    td = prepared.train_data(cfg.smoothify_metric)
    return DG.smoothness_report(td, prepared.X, ev.xhat, float(cfg.smoothify), cfg.smoothify_metric)


# ----------------------------------------------------------------------------- the whole payload
def assemble(p, req, prepared, res, ids: list, ev: Evaluation, notes: list, seconds_total: float) -> tuple[dict, dict, dict, dict]:
    """``(payload, variables, var_info, summary)`` of a trained network.  ``ev``: the training samples through the saved-model pass (``AEModel.evaluate`` on the project data)."""
    cfg = req.cfg
    spec = res.spec
    tree = model_tree(spec, res.arrays, prepared.frame)
    model = AEModel(tree)
    n = len(ids)
    samples = res.samples
    heldout = np.asarray(samples["heldout"], bool)
    err_held = np.asarray(samples["error_heldout"], float)
    ref = AE.reference_for(model, ids, heldout, ev, heldout_error=err_held)
    ref["n_eff"] = (res.validation or {}).get("n_eff")
    opt = D.settings(AE.options_for(model.family), None)
    jd = AE.judge(model, ev, ref, opt)
    variables, var_info = AE.publish(model, ev, jd)
    split_flag = np.where(heldout, "held out", "training")
    variables["split"] = split_flag
    var_info["split"] = {"kind": "category", "label": "Autoencoder: used for training or held out", "description": "Whether the sample was held out to check the network (its numbers are then honest) or used to train it."}
    q = quality_block(res, prepared, model, ev, ids, heldout, err_held)
    tr = training_block(res, res.split)
    lat = latent_block(res, prepared, model, ev, ref)
    smooth = smoothness_block(prepared, model, ev, cfg)
    calib = None
    if model.is_vae and (spec.vae or {}).get("sigma"):
        rows_c = np.flatnonzero(heldout) if heldout.sum() >= 5 else np.arange(n)
        calib = DG.calibration(spec, res.arrays, prepared.X[rows_c], None if prepared.M is None else prepared.M[rows_c])
    mean_code = np.asarray(ref["code_stats"]["mean"], float)
    sd = T.code_sd(ev.codes)
    sens = T.sensitivity(model, mean_code, sd)
    base = model.decode_functions(mean_code[None, :])
    split = res.split
    sp_payload = {"mode": split["mode"], "n_train": int((~heldout).sum()) if split["mode"] == "holdout" else n, "n_val": int(heldout.sum()) if split["mode"] == "holdout" else n,
                  "folds": len(split.get("folds") or []) or None, "flag": heldout.astype(np.uint8), "fold": None if samples.get("fold") is None else np.asarray(samples["fold"], int),
                  "words": (f"{int(heldout.sum())} of {n} samples were held out to check the network." if split["mode"] == "holdout"
                            else f"With {n} samples every sample was rebuilt by a network that had not seen it ({len(split.get('folds') or [])}-fold validation); the final network was trained on all of them.")}
    P, nparam = prepared.P, spec.n_parameters()
    n_train = sp_payload["n_train"] if split["mode"] == "holdout" else n
    warnings = list(notes)
    if nparam > n_train and not any("memorise" in w for w in warnings):
        warnings.append(f"The network has {nparam:,} weights for {n_train} training samples: it can memorise them, so only the held-out numbers are to be trusted.")
    if n_train / max(model.d, 1) < 10:
        warnings.append(f"Only {n_train / max(model.d, 1):.1f} training samples per code (10 or more are advised): the codes may not be reproducible.")
    payload: dict = {
        "ids": [str(i) for i in ids], "n": n, "family": model.family, "latent_dim": model.d, "title": "", "mode": prepared.mode,
        "input": {"variables": list(model.names), "mode": prepared.mode, "n_inputs": P, "blocks": prepared.summary()["blocks"], "weights": {nm: 1.0 for nm in model.names}, "dropped": [list(x) for x in prepared.dropped]},
        "network": {"spec": spec.to_dict(), "n_params": int(nparam), "hidden": list(spec.encoder["hidden"]), "activation": spec.encoder["activation"], "samples_per_param": n_train / max(nparam, 1),
                    "samples_per_latent": n_train / max(model.d, 1)},
        "split": sp_payload, "code_names": model.code_names, "codes": ev.codes, "codes_sd": ev.sd, "code_mean": mean_code, "code_sd": sd,
        "recon": {"error": ev.error, "r2": 1.0 - ev.error, "rmse": ev.rmse, "rmse_by_block": {nm: ev.rmse_blocks[:, k] for k, nm in enumerate(model.names)},
                  "error_by_block": {nm: ev.error_blocks[:, k] for k, nm in enumerate(model.names)}, "heldout": heldout, "limit": float(ref["limits"][str(opt["limit"])]), "limit_level": str(opt["limit"]),
                  "limit_words": ref["limits"].get("words", ""), "limits": {k: ref["limits"][k] for k in ("95", "99")}, "flag": jd.flag, "typicality": jd.p, "kl": jd.kl, "elbo": jd.elbo},
        "training": tr, "quality": q, "latent": lat, "smoothness": smooth, "calibration": calib,
        "limits": [D.limit_entry(f"Reconstruction error, {opt['limit']} % limit", ref["limits"][str(opt["limit"])], ref["limits"].get("words", ""))],
        "notes": list(dict.fromkeys(list(prepared.notes))), "warnings": list(dict.fromkeys(warnings)), "versions": dict(res.versions or {}), "n_outside": int(jd.err_out.sum() + (jd.typ_out & ~jd.err_out).sum()),
    }
    payload["versions"]["seed"] = cfg.seed
    # images of the fluorescence, curves of the absorbance
    ax: dict = {}
    for b in model.frame.blocks:
        fe = b.element
        if fe.kind == "surface":
            payload["images"] = {"mean": b.domain.embed(base[b.name])[0][None], "sensitivity": b.domain.embed(sens["at_mean"][b.name])}
            payload["image_labels"] = {"mean": ["Decoding of the mean code"], "sensitivity": [f"{nm}: change for one SD of the code" for nm in model.code_names]}
            payload["ex"], payload["em"], payload["mask"] = np.asarray(fe.axes[1], float), np.asarray(fe.axes[0], float), ~np.asarray(fe.obs, bool)
            payload["units"] = str(fe.units or "")
            ax["images"] = True
        else:
            series = [{"name": "Decoding of the mean code", "values": b.domain.embed(base[b.name])[0]}]
            series += [{"name": f"{nm}: change for one SD of the code", "values": b.domain.embed(sens["at_mean"][b.name][j])} for j, nm in enumerate(model.code_names)]
            payload["curves"] = {"x": np.asarray(fe.axes[0], float), "label": "Absorbance" + (f" ({fe.units})" if fe.units and fe.units != "absorbance" else ""), "series": series}
    if "images" in payload:
        # the contract with Clustering & LDA: a direction in code space as a fluorescence matrix (a linearisation at the mean code, in the units of the data)
        b = next(bl for bl in model.frame.blocks if bl.element.kind == "surface")
        from fluoviz.server.analysis import component_images_payload
        payload["component_images"] = component_images_payload(
            payload["images"]["sensitivity"] / sd[:, None, None], score_vars=model.code_names, names=[f"{nm} (code {j + 1})" for j, nm in enumerate(model.code_names)], mean_image=payload["images"]["mean"][0],
            kind="signed", ex=payload["ex"], em=payload["em"],
            description="What a unit change of each code does to the fluorescence, a linearisation of the decoder at the mean code (the network is nonlinear: away from the mean the effect differs).")
    # what a saved model is made from
    val = {"n_samples": n, "family": model.family, "latent_dim": model.d, "r2_val": q["r2_val"], "r2_train": q["r2_train"], "baseline": q["baseline"]["name"], "baseline_r2_val": q["baseline"]["r2_val"],
           "n_heldout": int(heldout.sum()), "mode": split["mode"], "error_limit_95": ref["limits"]["95"], "error_limit_99": ref["limits"]["99"], "n_eff": (res.validation or {}).get("n_eff"),
           "n_flagged_in_training": int(jd.err_out.sum() + (jd.typ_out & ~jd.err_out).sum())}
    payload["frozen"] = pack_frozen(tree, ref, val, AE.description_for(model, val), model)
    summary = {"n": n, "family": model.family, "latent_dim": model.d, "hidden": list(spec.encoder["hidden"]), "variables": list(model.names), "mode": prepared.mode, "n_params": int(nparam),
               "r2_val": q["r2_val"], "r2_train": q["r2_train"], "rmse_val": q["rmse_val"], "best_epoch": int(res.best_epoch), "epochs_run": int(res.epochs_run),
               "baseline": {"name": q["baseline"]["name"], "r2_val": q["baseline"]["r2_val"]}, "verdict": q["verdict"], "n_flagged": payload["n_outside"], "seed": cfg.seed, "seconds": float(seconds_total)}
    return payload, variables, var_info, summary
