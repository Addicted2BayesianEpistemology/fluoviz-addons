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
    parts.append(f"Training used AdamW (learning rate {float(cfg.get('lr', 1e-3)):g} with cosine decay, weight decay {float(cfg.get('weight_decay', 1e-3)):g}"
                 + (f", dropout {float(cfg.get('dropout')):g}" if float(cfg.get("dropout") or 0) > 0 else "") + f"), up to {cfg.get('epochs', 600)} epochs with early stopping (patience {cfg.get('patience', 60)}), "
                 f"{D.plural(int(cfg.get('n_restarts', 1)), 'random start')}, seed {cfg.get('seed', 0)}; {val}. The best weights were kept ({_best_words((pl.get('training') or {}).get('best_epoch'))}).")
    q = pl.get("quality") or {}
    if q.get("r2_val") is not None:
        base = q.get("baseline") or {}
        parts.append(f"On samples the network had not seen it rebuilt {_pct(q.get('r2_val'))} of the variation" + (f" ({base.get('name', 'PCA')} with {base.get('k')} components: {_pct(base.get('r2_val'))})" if base else "") + ".")
    ver = pl.get("versions") or {}
    parts.append("Software: " + ", ".join(f"{k} {v}" for k, v in ver.items() if v and str(v).lower() != "none" and k in ("addon", "fluoviz", "torch", "numpy")) + ".")
    return " ".join(x for x in parts if x)
