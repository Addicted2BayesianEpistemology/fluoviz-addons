"""The methods paragraph of a result: what was done, with the numbers, the seed and the versions, in plain English, ready for a report."""
from __future__ import annotations

import numpy as np

from fluoviz.analysis.frozen.kinds import _decomp as D
from fluoviz.analysis.frozen.kinds.fpca import basis_words, smoothing_words

from ..kinds.model import AEModel

REFERENCES = ["Hinton & Salakhutdinov 2006, Science 313:504 (autoencoders).",
              "Kingma & Welling 2014, ICLR (variational autoencoders); Higgins et al. 2017, ICLR (beta-VAE).",
              "Rasmussen & Williams 2006, Gaussian Processes for Machine Learning (Karhunen-Loeve basis, marginal likelihood).",
              "Ramsay & Silverman 2005, Functional Data Analysis, 2nd ed. (smoothing of functional data).",
              "Loshchilov & Hutter 2019, ICLR (AdamW).",
              "Paszke et al. 2019, NeurIPS (PyTorch)."]


def _pct(x) -> str:
    return "not available" if x is None or not np.isfinite(float(x)) else f"{100 * float(x):.1f} %"


def _best_words(e) -> str:
    if e is None:
        return "weights of the best epoch"
    return "the starting weights, which no epoch improved on" if int(e) < 0 else f"epoch {int(e) + 1}"


def _schedule_words(cfg: dict) -> str:
    sch = cfg.get("lr_schedule", "cosine")
    fin = float(cfg.get("lr_final", 0.01))
    if sch == "constant":
        return "kept constant"
    if sch == "plateau":
        return f"halved whenever the loss stopped improving (down to {fin:g} of its start)"
    base = "with cosine decay" if sch == "cosine" else "with cosine decay over the epochs each network was trained for"
    return base if abs(fin - 0.01) < 1e-12 else f"{base} to {fin:g} of its start"


def _stopping_words(cfg: dict) -> str:
    """How long it trained and what stopped it (the defaults read as they always did)."""
    ep, pat, rule = cfg.get("epochs", 600), cfg.get("patience", 60), cfg.get("stop_rule", "validation")
    if rule == "never":
        out = f"{ep} epochs without early stopping"
    else:
        out = f"up to {ep} epochs with early stopping (patience {pat}" + (", on the training loss" if rule == "training" else "")
        if int(cfg.get("min_epochs") or 0) > 0:
            out += f", never before epoch {int(cfg['min_epochs'])}"
        if abs(float(cfg.get("min_delta", 1e-3)) - 1e-3) > 1e-12:
            out += f", an improvement of at least {100 * float(cfg['min_delta']):g} % counting"
        if int(cfg.get("val_window", 1)) > 1:
            out += f", the loss averaged over {int(cfg['val_window'])} epochs"
        out += ")"
    if cfg.get("max_minutes"):
        out += f", at most {float(cfg['max_minutes']):g} minutes per network"
    return out


def _gap_words(r, model: AEModel) -> str:
    """The sentence about samples that lacked part of the domain (a region excised from them): empty when no sample did."""
    miss = (getattr(r, "variables", None) or {}).get("missing_part")
    if miss is None:
        return ""
    m = np.asarray(miss, float)
    k = int((m >= 0.005).sum())
    if not k:
        return ""
    prm = r.params or {}
    inp = prm.get("input") or {}
    func = [b for b in model.frame.blocks if b.kind != "pixels"]
    out = f"{D.plural(k, 'sample')} lacked part of the domain (up to {_pct(float(np.nanmax(m)))} of it)"
    if not func:
        return out + "; the missing pixels were ignored in the loss."
    modes = {(inp.get(b.name) or {}).get("gaps", "auto") for b in func}
    how = ("; the coefficients of these samples were estimated from the pixels they have together with what the other samples show about the part they lack (the gap-aware estimate "
           "of the functional PCA: the smoothness prior of the representation combined with the directions in which the other samples vary)" if modes == {"auto"} else "; the coefficients of these samples were estimated from the pixels they have")
    if prm.get("gap_weighting", "observed") == "observed":
        how += ", and each of them counted in the loss in proportion to the share of the domain that was measured"
    return out + how + "."


def methods_text(r, model: AEModel) -> str:
    """The paragraph; it only reads what the result stores (so it also works for a project that was opened again, and for an outdated result)."""
    pl = r.payload
    prm = r.params or {}
    sp = model.spec
    n = len(r.sample_ids or [])
    fam = {"ae": "deterministic autoencoder", "vae": "variational autoencoder"}[model.family]
    what = D.join_words([{"fluorescence": "fluorescence excitation-emission matrices (EEMs)", "absorbance": "absorbance spectra"}[nm] for nm in model.names])
    parts = [f"A {fam} was trained on the {what} of {n} samples."]
    reads = []
    for b in model.frame.blocks:
        fe = b.element
        pre = {"none": "", "l2": " after division by its L2 norm", "area": " after division by its mean", "max": " after division by its maximum", "sqrt": " after taking its square root",
               "asinh": " after the inverse hyperbolic sine", "log10": " after log10", "derivative1": " after taking its first derivative"}.get(fe.transform, "")
        if b.kind == "pixels":
            reads.append(f"the {b.name}{pre} was used pixel by pixel ({int(np.asarray(fe.obs).sum())} observed points, centred by the training mean function and divided by one scale so that "
                         "unmeasured pixels were ignored in the loss)")
        else:
            reads.append(f"the {b.name}{pre} was written as a smooth function in {basis_words(fe)}; the smoothing ({smoothing_words(fe)}) was estimated once from the training samples and never "
                         f"again, and the coefficients were centred, moved to an L2-orthonormal frame and standardised ({b.K} coordinates)")
    joined = "; ".join(reads)
    parts.append(joined[:1].upper() + joined[1:] + ".")
    gap = _gap_words(r, model)
    if gap:
        parts.append(gap)
    enc = " - ".join(str(w) for w in [sp.input_dim, *sp.encoder["hidden"], model.d])
    skip = (" A linear shortcut ran beside the encoder and the decoder and training started from the PCA solution of the training samples." if getattr(sp, "skip", False) else "")
    parts.append(f"The network had {D.plural(model.d, 'code')} (layers {enc}, {sp.encoder['activation']} activation, the decoder mirroring the encoder) and {sp.n_parameters():,} weights.{skip}")
    loss = ("The loss was the squared L2 distance between a sample and its reconstruction divided by the total variance of its variable (so that one minus it is the share of the variation "
            "rebuilt)")
    s = float(prm.get("smoothify") or 0.0)
    if model.is_vae:
        v = prm.get("vae") or {}
        loss += (f", scaled by a Gaussian noise variance ({'learned' if v.get('sigma', 'learned') == 'learned' else 'fixed'}) and added to a Kullback-Leibler term with weight beta = "
                 f"{float(v.get('beta', 1.0)):g} (prior: {'a Gaussian mixture' if sp.has_mixture() else 'standard normal'})")
    if s > 0 and any(b.kind != "pixels" for b in model.frame.blocks):
        loss += f"; a smoothness penalty on the coefficients the decoder produces (smoothify = {s:g}, {'the smoothness metric of the representation' if prm.get('smoothify_metric', 'smooth') == 'smooth' else 'a plain ridge'}) made the reconstructions smoother"
    parts.append(loss + ".")
    cfg = prm
    split = pl.get("split") or {}
    val = (f"{split.get('n_val')} of the samples were held out for validation" if split.get("mode") == "holdout"
           else f"{cfg.get('folds', 5)}-fold cross-validation was used for the validation numbers and the final network was trained on all samples")
    parts.append(f"Training used AdamW (learning rate {float(cfg.get('lr', 1e-3)):g} {_schedule_words(cfg)}, weight decay {float(cfg.get('weight_decay', 1e-3)):g}"
                 + (f", dropout {float(cfg.get('dropout')):g}" if float(cfg.get("dropout") or 0) > 0 else "") + f"), {_stopping_words(cfg)}, "
                 f"{D.plural(int(cfg.get('n_restarts', 1)), 'random start')}, seed {cfg.get('seed', 0)}; {val}. "
                 + (f"The best weights were kept ({_best_words((pl.get('training') or {}).get('best_epoch'))})." if cfg.get("restore_best", True) else "The weights of the last epoch were kept."))
    q = pl.get("quality") or {}
    if q.get("r2_val") is not None:
        base = q.get("baseline") or {}
        parts.append(f"On samples the network had not seen it rebuilt {_pct(q.get('r2_val'))} of the variation" + (f" ({base.get('name', 'PCA')} with {base.get('k')} components: {_pct(base.get('r2_val'))})" if base else "") + ".")
    ver = pl.get("versions") or {}
    parts.append("Software: " + ", ".join(f"{k} {v}" for k, v in ver.items() if v and str(v).lower() != "none" and k in ("addon", "fluoviz", "torch", "numpy")) + ".")
    return " ".join(x for x in parts if x)
