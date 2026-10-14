"""The saved model of kind ``autoencoder``: freeze a trained network (AE / VAE, functional or on the pixels) and apply it to NEW samples without training anything again.

What is frozen (``meta["model"]``, see :mod:`.model`): the network (``ModelSpec`` + arrays), and how a sample becomes network input, exactly as the training samples were prepared:
the domain, the pre-treatment, the representation and the smoothing parameters the fit chose (never estimated again for a new sample), the mean function and the standardisation.
``meta["reference"]`` holds the training samples (codes, errors, which were held out) and the limits.

A new sample is read at the pixels of the domain, pre-treated, smoothed on ITS OWN observed pixels in the stored representation (functional) or read as it is (pixels, gaps allowed),
standardised, encoded (the code of a variational autoencoder is its posterior mean), decoded, and compared with itself: the **reconstruction error** is the share of the typical variation
between training samples that the network does not rebuild (``1 -`` it is the fraction of variance explained).  Status:

* ``unusable``: a variable is missing, a pre-treatment is not defined, or fewer than ``min_coverage`` of a domain is measured (all outputs NaN);
* ``outside``: the reconstruction error is above the limit of the model (the 95 % or 99 % point of the errors of samples the network was NOT trained on: ``stats.error_limits``) or the
  code is unusual (typicality below the threshold; on by default for a variational autoencoder, off for a deterministic one).

Published variables: ``z1…zd`` (the code), ``recon_rmse`` (data units), ``recon_r2`` (``1 - error``), ``typicality`` (p-value of the code), ``flag`` (words), ``status``; a variational
autoencoder also ``z1_sd…`` (at most 12), ``kl`` and ``elbo``; a mixture prior ``cluster`` and ``membership``.  Plain English in everything a person reads.
"""
from __future__ import annotations

import numpy as np

from fluoviz.analysis.frozen import Applier, ApplyData, ApplyOutput, FrozenError, SavedModel, Statuses
from fluoviz.analysis.frozen.kinds import _decomp as D
from fluoviz.analysis.frozen.kinds.fpca import FrozenElement, basis_words, smoothing_words

from ..core import forward as fw
from . import stats as S
from . import tools as T
from .model import AEModel, Evaluation

KIND = "autoencoder"
LABELS = {"ae": "Autoencoder", "vae": "Variational autoencoder"}
FLAG_WORDS = {"ok": "inside the limits", "error": "reconstruction error above the limit", "code": "unusual code", "both": "reconstruction error above the limit and unusual code",
              "unusable": "cannot be used"}
MAX_SD_VARIABLES = 12


# ----------------------------------------------------------------------------- what a person can change when applying
def options_for(family: str) -> list[dict]:
    vae = family == "vae"
    return [D.choice_option("limit", "Limit for the reconstruction error", "99", [{"value": "95", "label": "95 %"}, {"value": "99", "label": "99 %"}],
                            "A new sample the network rebuilds worse than this limit is reported as 'Poorly described by the model'.  The limit is the 95 % or 99 % point of the errors of samples "
                            "the network was not trained on: 99 % flags only clear departures (about 1 ordinary sample in 100), 95 % is stricter (about 1 in 20)."),
            D.number_option("typicality", "Smallest typicality of the code" + ("" if vae else " (0 = ignore the code)"), 0.01 if vae else 0.0, 0.0, 0.5,
                            ("A new sample whose code is less typical than this (a p-value: 1 = right in the middle of the training codes, near 0 = far out) is reported as 'Poorly described by the "
                             "model', even when its reconstruction looks fine: the network has never seen codes like this.  0.01 flags about 1 ordinary sample in 100; 0 switches the check off."
                             if vae else
                             "A deterministic autoencoder has no distribution of codes, so this check is off (0).  Set it above 0 to also flag samples whose code lies far from the training codes "
                             "(a p-value of how typical the code is among the training codes: 0.01 flags about 1 ordinary sample in 100)."))]


def outputs_for(model: AEModel) -> list[dict]:
    d = model.d
    vae = model.is_vae
    kind = LABELS[model.family].lower()
    out = [{"name": f"z{j + 1}", "label": f"z{j + 1} (code {j + 1})", "kind": "numeric",
            "description": f"Code {j + 1} of the {kind}: one of the {d} numbers the network uses to describe a sample" + (" (the mean of its posterior distribution)." if vae else ".")}
           for j in range(d)]
    out += [{"name": "recon_rmse", "label": "Autoencoder reconstruction error (RMSE, data units)", "kind": "numeric",
             "description": "Root mean square difference between the measured values and the network's reconstruction, in the units of the (pre-treated) data."},
            {"name": "recon_r2", "label": "Autoencoder reconstruction R² (share of the variation rebuilt)", "kind": "numeric",
             "description": "1 minus the reconstruction error: the share of the typical variation between the training samples that the network rebuilds for this sample (1 = perfectly)."},
            {"name": "typicality", "label": "Typicality of the code (p-value)", "kind": "numeric",
             "description": "How ordinary the code is among the training codes: near 1 = typical, near 0 = far from where the training samples lie."},
            {"name": "flag", "label": "Autoencoder anomaly status", "kind": "category",
             "description": "Whether the reconstruction error and the code are inside the limits of the model."}]
    if vae:
        out += [{"name": f"z{j + 1}_sd", "label": f"Uncertainty of z{j + 1} (posterior SD)", "kind": "numeric",
                 "description": f"Standard deviation of the posterior distribution of code {j + 1}: how uncertain the network is about it."} for j in range(min(d, MAX_SD_VARIABLES))]
        out += [{"name": "kl", "label": "KL divergence (how unusual the code is)", "kind": "numeric",
                 "description": "Kullback-Leibler divergence of the code's posterior from the prior, in nats: high = the network needs a code far from the typical ones."},
                {"name": "elbo", "label": "Evidence lower bound (higher = better explained)", "kind": "numeric",
                 "description": "Minus the training loss of the variational autoencoder (the reconstruction error scaled by the noise level, plus the KL term), up to a constant: higher = the network explains the sample better."}]
        if model.spec.has_mixture():
            out += [{"name": "cluster", "label": "Cluster of the mixture prior", "kind": "category", "description": "The cluster of the mixture prior that the code most resembles."},
                    {"name": "membership", "label": "Membership of its cluster", "kind": "numeric", "description": "Probability of the sample's cluster (1 = no doubt)."}]
    out.append({"name": "status", "label": "Status", "kind": "category", "description": "Fits the model / Poorly described by the model / Cannot be used."})
    return out


# ----------------------------------------------------------------------------- judging samples
class Judgement:
    """Status, reasons and the per-sample numbers that depend on the limits (see :func:`judge`)."""

    def __init__(self, n: int):
        self.n = n
        self.status = np.array([], str)
        self.reasons: list = []
        self.flag = np.array([], str)
        self.d2 = np.full(n, np.nan)
        self.p = np.full(n, np.nan)
        self.kl: np.ndarray | None = None
        self.elbo: np.ndarray | None = None
        self.membership: np.ndarray | None = None
        self.membership_index: np.ndarray | None = None
        self.err_out = np.zeros(n, bool)
        self.typ_out = np.zeros(n, bool)


def _which(names: list[str], share: np.ndarray) -> str:
    """', mostly the absorbance (78 % of the misfit)': which variable is rebuilt worst."""
    if len(names) < 2:
        return ""
    j = int(np.argmax(share))
    if share[j] >= 0.6:
        return f", mostly the {names[j]} ({100 * share[j]:.0f} % of the misfit)"
    return f", with the misfit shared between the {D.join_words(names)} ({', '.join(f'{100 * x:.0f} %' for x in share)})"


def judge(model: AEModel, ev: Evaluation, ref: dict, opts: dict) -> Judgement:
    """Status and reasons of evaluated samples against the limits frozen in the reference ``ref`` and the thresholds ``opts`` (``limit``, ``typicality``)."""
    n = ev.n
    jd = Judgement(n)
    lev = str(opts["limit"])
    lim = float(ref["limits"][lev])
    thr = float(opts.get("typicality", 0.0))
    u = ev.usable
    names = model.names
    if u.any():
        codes = ev.codes[u]
        if model.spec.has_mixture():
            d2, p = _mixture_typicality(model, codes, ev.logvar[u] if ev.logvar is not None else None)
        else:
            d2, p = S.mahalanobis2(codes, ref["code_stats"]), S.typicality(codes, ref["code_stats"])
        jd.d2[u], jd.p[u] = d2, p
        if model.is_vae:
            lv = ev.logvar[u]
            if model.spec.has_mixture():
                kl = S.kl_mixture(codes, lv, model.arrays["prior.pi"], model.arrays["prior.mu"], model.arrays["prior.logvar"])
                jd.membership = np.full(n, np.nan)
                jd.membership_index = np.full(n, -1)
                mem = fw.memberships(model.spec, model.arrays, codes, lv)
                jd.membership[u] = mem.max(axis=1)
                jd.membership_index[u] = mem.argmax(axis=1)
            else:
                kl = S.kl_normal(codes, lv)
            jd.kl = np.full(n, np.nan)
            jd.kl[u] = kl
            sigma = (model.spec.vae or {}).get("sigma")
            if sigma:
                jd.elbo = np.full(n, np.nan)
                jd.elbo[u] = S.elbo(ev.error[u], kl, sigma, float((model.spec.vae or {}).get("n_eff") or ref.get("n_eff") or 1.0))
    with np.errstate(invalid="ignore"):
        jd.err_out = u & np.isfinite(lim) & (ev.error > lim)
        jd.typ_out = u & (thr > 0) & np.isfinite(jd.p) & (jd.p < thr)
    st = Statuses(n)
    st.flag(~u, "unusable", lambda i: " ".join(ev.why[i]) or "This sample cannot be used.")
    share = ev.error_blocks / np.where(ev.error > 0, ev.error, np.nan)[:, None]
    st.flag(jd.err_out, "outside", lambda i: f"Its reconstruction error ({D.g(ev.error[i])}) is above the {lev} % limit ({D.g(lim)}): the network does not rebuild this sample well, it contains "
                                             f"something the training samples did not have{_which(names, np.nan_to_num(share[i]))}, or it was measured differently.")
    st.flag(jd.typ_out, "outside", lambda i: f"Its code is unusual (typicality {D.g(jd.p[i])}, below {D.g(thr)}): the network gives it a code far from where the training samples lie, so the "
                                             "code may be less meaningful even if the reconstruction looks fine.")
    st.flag(ev.gaps & u, "ok", lambda i: "Some pixels of this sample were not measured: it was smoothed (or read) on the pixels it has, so its code is less certain "
                                         f"({D.percent(1 - float(np.nanmin(ev.coverage[i])), 2)} of the domain missing).")
    jd.status = st.array()
    jd.reasons = st.reasons
    flag = np.where(jd.err_out & jd.typ_out, FLAG_WORDS["both"], np.where(jd.err_out, FLAG_WORDS["error"], np.where(jd.typ_out, FLAG_WORDS["code"], FLAG_WORDS["ok"])))
    jd.flag = np.where(u, flag, FLAG_WORDS["unusable"]).astype(str)
    return jd


def _mixture_typicality(model: AEModel, codes: np.ndarray, logvar: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """With a mixture prior a code is typical when it lies near one of the clusters: the smallest squared distance to a cluster (in the cluster's own SD, the posterior SD added)."""
    from scipy import stats as sps
    cm, clv = model.arrays["prior.mu"], model.arrays["prior.logvar"]
    var = np.exp(clv)[None] + (0.0 if logvar is None else np.exp(logvar)[:, None, :])
    d2k = (((codes[:, None, :] - cm[None]) ** 2) / var).sum(axis=2)
    d2 = d2k.min(axis=1)
    return d2, sps.chi2.sf(d2, model.d)


def publish(model: AEModel, ev: Evaluation, jd: Judgement) -> tuple[dict, dict]:
    """The variables of an application / of the result: ``(variables, var_info)``."""
    var: dict = {}
    for j, nm in enumerate(model.code_names):
        var[nm] = ev.codes[:, j]
    var["recon_rmse"] = ev.rmse
    var["recon_r2"] = 1.0 - ev.error
    var["typicality"] = jd.p
    var["flag"] = jd.flag
    if model.is_vae:
        for j in range(min(model.d, MAX_SD_VARIABLES)):
            var[f"z{j + 1}_sd"] = ev.sd[:, j]
        var["kl"] = jd.kl
        var["elbo"] = jd.elbo if jd.elbo is not None else np.full(ev.n, np.nan)
        if model.spec.has_mixture():
            lab = np.array([f"cluster {int(k) + 1}" if k >= 0 else "" for k in jd.membership_index], dtype=object)
            var["cluster"] = np.where(ev.usable, lab, None)
            var["membership"] = jd.membership
    var["status"] = jd.status
    info = {o["name"]: {"kind": o["kind"], "label": o["label"], "description": o["description"]} for o in outputs_for(model)}
    return {k: v for k, v in var.items() if k in info}, info


# ----------------------------------------------------------------------------- the reference: the training samples as the limits see them
def reference_for(model: AEModel, ids: list, heldout, ev: Evaluation, heldout_error=None) -> dict:
    """The ``reference`` of a saved model from the evaluation of the training samples: their codes and errors, which of them the network did not train on (``heldout``: those give the
    limit of the reconstruction error; ``heldout_error``: their errors when they are not those of this network, as with k-fold validation) and the distribution of the codes."""
    u = ev.usable
    heldout = np.asarray(heldout, bool)
    held_err = ev.error[heldout & u] if heldout_error is None else np.asarray(heldout_error, float)[heldout]       # the errors of samples the network did not train on (out-of-fold for k-fold)
    limits = S.error_limits(held_err, fallback=ev.error[u])
    stats = S.code_stats(ev.codes[u], None if ev.logvar is None else np.exp(0.5 * ev.logvar[u]))
    d2 = np.full(ev.n, np.nan)
    p = np.full(ev.n, np.nan)
    if u.any():
        if model.spec.has_mixture():
            d2[u], p[u] = _mixture_typicality(model, ev.codes[u], ev.logvar[u] if ev.logvar is not None else None)
        else:
            d2[u], p[u] = S.mahalanobis2(ev.codes[u], stats), S.typicality(ev.codes[u], stats)
    names = model.code_names
    return {"ids": [str(i) for i in ids], "heldout": heldout, "score_names": names, "score_labels": [f"{nm} (code {j + 1})" for j, nm in enumerate(names)], "scores": ev.codes.copy(),
            "scores_sd": None if ev.logvar is None else np.exp(0.5 * ev.logvar), "error": ev.error.copy(), "rmse": ev.rmse.copy(), "d2": d2, "typicality": p, "limits": limits, "code_stats": stats}


# ----------------------------------------------------------------------------- words
def block_words(fe: FrozenElement, mode: str) -> str:
    if mode == "pixels":
        return "the observed pixels themselves (no smoothing; unmeasured pixels are ignored)"
    return f"{basis_words(fe)}, smoothed with a {smoothing_words(fe)}"


def description_for(model: AEModel, validation: dict | None = None, title_n: int | None = None) -> str:
    sp = model.spec
    v = validation or {}
    fam = {"ae": "autoencoder", "vae": "variational autoencoder"}[model.family]
    what = D.join_words([{"fluorescence": "fluorescence matrices (EEMs)", "absorbance": "absorbance spectra"}[n] for n in model.names])
    n = v.get("n_samples") or title_n
    reading = "; ".join(f"the {b.name} is read as {block_words(b.element, 'pixels' if b.kind == 'pixels' else 'functional')}" for b in model.frame.blocks)
    s = (f"A {fam} for the {what}{f' of {n} samples' if n else ''}: a neural network that learns {D.plural(model.d, 'code')} (numbers) for each sample and rebuilds the sample from them. "
         f"The network has {fw_params(sp)} weights ({' - '.join(str(w) for w in [sp.input_dim, *sp.encoder['hidden'], model.d])} encoder, mirrored decoder, {sp.encoder['activation']} activation); {reading}. ")
    if model.is_vae:
        s += ("Being variational, it also learns what a typical code looks like (the codes are drawn from a distribution), which is used to judge whether a new sample's code is ordinary. ")
    if v.get("r2_val") is not None:
        s += f"On samples it was not trained on it rebuilt {100 * float(v['r2_val']):.1f} % of the variation"
        if v.get("baseline_r2_val") is not None:
            s += f" ({v.get('baseline', 'PCA')} with the same number of components: {100 * float(v['baseline_r2_val']):.1f} %)"
        s += ". "
    s += ("For each new sample the model brings it into the same state as the training samples (the same smoothing, centring and scaling, never estimated again from the new samples), gives its "
          f"{D.plural(model.d, 'code')} and how well the network rebuilds it, and compares that with the limits of the training samples.")
    return s


def fw_params(sp) -> int:
    return int(sp.n_parameters())


# ----------------------------------------------------------------------------- the applier
class AutoencoderApplier(Applier):
    def __init__(self, saved: SavedModel):
        super().__init__(saved)
        m = saved.meta
        self.model = AEModel(m.get("model"))
        self.ref = m.get("reference") or {}
        for key in ("limits", "code_stats"):
            if key not in self.ref:
                raise FrozenError("This model file is damaged: its limits are missing.")

    # ------------------------------------------------------------------ facts for the page
    def options(self) -> list[dict]:
        return [dict(o) for o in (self.saved.meta.get("options") or options_for(self.model.family))]

    def _limits(self, opts: dict) -> list[dict]:
        lev, thr = str(opts["limit"]), float(opts.get("typicality", 0.0))
        lims = self.ref["limits"]
        out = [D.limit_entry(f"Reconstruction error, {lev} % limit", lims[lev],
                             f"{lims.get('words', '')} A sample above it is poorly rebuilt by the network (the error is the share of the typical variation between samples that is not rebuilt).")]
        if thr > 0:
            out.append(D.limit_entry("Smallest typicality of the code", thr, "A sample whose code is less typical than this among the training codes is poorly described by the model."))
        return out

    def describe(self) -> dict:
        m, ref = self.model, self.ref
        sp = m.spec
        v = self.saved.meta.get("validation") or {}
        training = {"Samples": int(v.get("n_samples", len(ref.get("ids", [])))), "Kind": LABELS[m.family], "Codes": m.d,
                    "Network": f"{sp.n_parameters()} weights, layers {' - '.join(str(w) for w in [sp.input_dim, *sp.encoder['hidden'], m.d])}, {sp.encoder['activation']}"}
        for b in m.frame.blocks:
            training[f"{b.name.capitalize()}: pixels in the domain"] = int(np.asarray(b.element.obs).sum())
            training[f"{b.name.capitalize()}: how it is read"] = block_words(b.element, "pixels" if b.kind == "pixels" else "functional")
            tw = {"none": "", "l2": "divided by its L2 norm (shape only)", "area": "divided by its mean over the domain", "max": "divided by its maximum", "sqrt": "square-rooted",
                  "asinh": "transformed by the inverse hyperbolic sine", "log10": "transformed by log10", "derivative1": "replaced by its first derivative (the slope)"}.get(b.element.transform, "")
            if tw:
                training[f"{b.name.capitalize()}: before smoothing"] = tw
        if v.get("r2_val") is not None:
            training["Variation rebuilt on held-out samples"] = f"{100 * float(v['r2_val']):.1f} %"
        if v.get("baseline_r2_val") is not None:
            training[f"{v.get('baseline', 'PCA')} with the same number of components"] = f"{100 * float(v['baseline_r2_val']):.1f} %"
        inputs = []
        for b in m.frame.blocks:
            if b.name == "fluorescence":
                inputs.append("The fluorescence matrix (EEM) of every sample, on the wavelength grid of the model.")
            else:
                inputs.append("The absorbance spectrum of every sample, on the wavelength grid of the model" + ("; the model cleans it like the training spectra (baseline offset, negative values)." if b.element.ife else "."))
        if len(m.frame.blocks) > 1:
            inputs.append("A sample needs BOTH: a sample without an absorbance spectrum cannot be used.")
        caveats = ["Every new sample is brought to the network's input exactly like the training samples (the same smoothing parameters, centring and scaling); nothing is estimated again from the new samples.",
                   "The reconstruction error limit comes from samples the network was not trained on; with a small training set it is only a rough guide.",
                   "A nonlinear network rebuilds well what resembles its training samples: a sample of another kind is reported as poorly described, which is the point of the check, but a "
                   "sample can also be flagged because it was measured under other conditions."]
        if m.is_vae:
            caveats.append("The code of a variational autoencoder is the mean of its posterior distribution; the typicality check compares it with the codes of the training samples.")
        if any(b.kind == "pixels" for b in m.frame.blocks):
            caveats.append("A model on the pixels themselves ignores unmeasured pixels, but a sample with many gaps has a less certain code.")
        else:
            caveats.append("A sample with missing pixels is smoothed on the pixels it has; the model fills the gaps with its own smooth shape, so its code is less certain.")
        defaults = D.settings(self.options(), None)
        return D.describe_block(self.saved, inputs, training, self._limits(defaults), caveats)

    # ------------------------------------------------------------------ apply
    def apply(self, data: ApplyData, options: dict, progress) -> ApplyOutput:
        m, ref = self.model, self.ref
        opt = D.settings(self.options(), options)
        lev, thr = str(opt["limit"]), float(opt["typicality"])
        progress(0.03, "reading the samples")
        ev = m.evaluate(data, opt["min_coverage"])
        progress(0.6, "judging the samples")
        jd = judge(m, ev, ref, opt)
        variables, info = publish(m, ev, jd)
        progress(0.85, "preparing the plots")
        payload = self._payload(data, ev, jd, opt)
        status = jd.status
        summary = D.summary_block(status, n_outside_error=int(jd.err_out.sum()), n_outside_code=int(jd.typ_out.sum()), limit=lev, n_codes=m.d, family=m.family, variables=m.names)
        text = self._text(data, status, opt)
        progress(1.0, "done")
        return ApplyOutput(variables=variables, var_info=info, status=status, reasons=jd.reasons, payload=payload, summary=summary, text=text)

    def _payload(self, data: ApplyData, ev: Evaluation, jd: Judgement, opt: dict) -> dict:
        m, ref = self.model, self.ref
        lev, thr = str(opt["limit"]), float(opt["typicality"])
        names = m.code_names
        k = min(m.d, 8)
        labels = [f"{nm} (code {j + 1})" for j, nm in enumerate(names)]
        ref_codes = None if ref.get("scores") is None else np.asarray(ref["scores"], float)
        plot = D.plot_block(f"{names[0]} against {names[1]}" if m.d > 1 else names[0], names[:k], labels[:k], ev.codes[:, :k], None if ref_codes is None else ref_codes[:, :k], ref.get("ids"))
        lim = float(ref["limits"][lev])
        cut = S.typicality_cut(thr, ref["code_stats"]) if thr > 0 and not m.spec.has_mixture() else None
        diag = [D.diagnostic("Reconstruction error against how unusual the code is",
                             "Each point is a sample.  The reconstruction error (across) is the share of the typical variation between samples that the network does not rebuild; the distance of the "
                             "code (up) says how far the code lies from the codes of the training samples.  A new sample beyond a dotted line is poorly described by the model; the grey points are "
                             "the training samples.", "Reconstruction error", ev.error, ref.get("error"), lim, f"{lev} % limit",
                             "Distance of the code from the training codes", jd.d2, ref.get("d2"), cut, "typicality limit")]
        mean = np.asarray(ref["code_stats"]["mean"], float)
        sd = T.code_sd(ref["scores"]) if ref.get("scores") is not None else np.asarray(ref["code_stats"]["sd"], float)
        sens = T.sensitivity(m, mean, sd)
        base = m.decode_functions(mean[None, :])
        payload: dict = {"plot": plot, "diagnostics": diag, "limits": self._limits(opt), "notes": list(data.notes), "n_codes": m.d, "code_names": names,
                         "recon": {"error": ev.error, "rmse": ev.rmse}}
        for b in m.frame.blocks:
            fe = b.element
            if fe.kind == "surface":
                payload["images"] = {"mean": m.embed(b.name, base[b.name])[0][None], "directions": m.embed(b.name, sens["at_mean"][b.name])}
                payload["image_labels"] = {"mean": ["Decoding of the mean code"], "directions": [f"{nm}: change for one SD of the code" for nm in names]}
                payload["ex"], payload["em"], payload["mask"] = np.asarray(fe.axes[1], float), np.asarray(fe.axes[0], float), ~np.asarray(fe.obs, bool)
            else:
                series = [{"name": "Decoding of the mean code", "values": m.embed(b.name, base[b.name])[0]}]
                series += [{"name": f"{nm}: change for one SD of the code", "values": m.embed(b.name, sens["at_mean"][b.name][j])} for j, nm in enumerate(names)]
                payload["curves"] = {"x": np.asarray(fe.axes[0], float), "label": "Absorbance" + (f" ({fe.units})" if fe.units and fe.units != "absorbance" else ""), "series": series}
        if not np.isfinite(lim):
            payload["notes"].append("The model has no limit for the reconstruction error (too few samples were held out): samples are judged by their code only.")
        return payload

    def _text(self, data: ApplyData, status: np.ndarray, opt: dict) -> str:
        m, ref = self.model, self.ref
        lev = str(opt["limit"])
        lim = float(ref["limits"][lev])
        fam = {"ae": "autoencoder", "vae": "variational autoencoder"}[m.family]
        reading = D.join_words([f"the {b.name} as {block_words(b.element, 'pixels' if b.kind == 'pixels' else 'functional')}" for b in m.frame.blocks])
        thr = float(opt["typicality"])
        return (f"The saved {fam} '{self.saved.title}' ({D.plural(m.d, 'code')}, trained on {D.plural(len(ref.get('ids', [])), 'sample')}) was applied to {D.plural(data.n, 'new sample')}. "
                f"{D.counts_sentence(status)}  Method: each new sample was read exactly like the training samples ({reading}; the smoothing parameters, centring and scaling of the training "
                f"samples were used and nothing was estimated from the new samples), encoded into {D.plural(m.d, 'code')} and rebuilt by the decoder; the reconstruction error was compared "
                f"with the {lev} % limit of the samples the network was not trained on ({D.g(lim)})" + (f" and the typicality of the code with the threshold {D.g(thr)}" if thr > 0 else "") + ".")

    # ------------------------------------------------------------------ inspect
    def inspect(self, data: ApplyData, options: dict, index: int) -> dict:
        m, ref = self.model, self.ref
        opt = D.settings(self.options(), options)
        lev = str(opt["limit"])
        i = int(index)
        if not 0 <= i < data.n:
            raise FrozenError("There is no such sample in this application.")
        one = data.subset([i])
        ev = m.evaluate(one, opt["min_coverage"])
        if not ev.usable[0]:
            raise FrozenError(" ".join(ev.why[0]) or "This sample cannot be shown against the model.")
        jd = judge(m, ev, ref, opt)
        images: dict = {}
        labels: dict = {}
        curves = None
        mask = ex = em = None
        mean = m.mean_functions()
        for b in m.frame.blocks:
            fe = b.element
            y = ev.treated[b.name][0]
            rec = ev.recon[b.name][0]
            if fe.kind == "surface":
                images.update({"measured": m.embed(b.name, y), "reconstruction": m.embed(b.name, rec), "residual": m.embed(b.name, np.where(np.isfinite(y), y - rec, np.nan)),
                               "mean": m.embed(b.name, mean[b.name])})
                labels.update({"measured": "Measured (as the model sees it, after the pre-treatment)", "reconstruction": "Rebuilt by the network from its code",
                               "residual": "Measured minus rebuilt (what the network misses)", "mean": "Mean of the training samples"})
                ex, em, mask = np.asarray(fe.axes[1], float), np.asarray(fe.axes[0], float), ~np.asarray(fe.obs, bool)
            else:
                curves = {"x": np.asarray(fe.axes[0], float), "label": "Absorbance" + (" (after the pre-treatment)" if fe.transform != "none" else ""),
                          "series": [{"name": "Measured", "values": m.embed(b.name, y)}, {"name": "Rebuilt by the network", "values": m.embed(b.name, rec)},
                                     {"name": "Residual", "values": m.embed(b.name, np.where(np.isfinite(y), y - rec, np.nan))}, {"name": "Mean of the training samples", "values": m.embed(b.name, mean[b.name])}]}
        err = float(ev.error[0])
        lim = float(ref["limits"][lev])
        bits = []
        if jd.err_out[0]:
            share = np.nan_to_num(ev.error_blocks[0] / ev.error[0]) if ev.error[0] > 0 else np.zeros(len(m.names))
            bits.append(f"the network does not rebuild it well (error {D.g(err)}, {lev} % limit {D.g(lim)}){_which(m.names, share)}")
        if jd.typ_out[0]:
            bits.append(f"its code is unusual (typicality {D.g(jd.p[0])})")
        text = f"{data.ids[i]}: " + ("; and ".join(bits) + "." if bits else f"well rebuilt by the network (error {D.g(err)}, inside the {lev} % limit {D.g(lim)}).")
        facts = {"code": ev.codes[0], "error": err, "r2": 1.0 - err, "rmse": float(ev.rmse[0]), "limit": lim, "level": lev, "typicality": float(jd.p[0])}
        out = D.inspect_block(images, labels, text, signed=("residual",), ex=ex, em=em, mask=mask, units=data.units or "", facts=facts, title=f"{data.ids[i]} against the autoencoder")
        if curves is not None:
            out["curves"] = curves
        return out
