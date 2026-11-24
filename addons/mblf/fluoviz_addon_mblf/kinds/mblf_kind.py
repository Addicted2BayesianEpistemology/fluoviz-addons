"""The saved model of kind ``mblf``: apply a fitted multimodal latent factor model to NEW samples without fitting anything again.

For every new sample the model reads what it has (any of the modes; its covariates when the metadata hold them), and gives:

* its factor scores (aligned, as on the result's page) with their posterior standard deviations;
* for every value the model was fitted on (``values`` mode, for example a laboratory measurement): the predicted value and its 95 % predictive interval — from the fluorescence and / or
  absorbance of the sample (and its covariates), with every source of uncertainty included (the population parameters, the scores, the sample's own variation, the measurement noise);
* ``mblf_surprise``: how typical the sample is for the model (its log predictive density per measured value; low = unusual), against the same number for the training samples
  that had the same modes (held out, PSIS): ``outside`` below the 1 % (or 5 %) point;
* ``inspect``: the predicted EEM / spectrum of a sample (what it lacks, or everything) with its bands and the uncertainty budget.

Plain English in everything a person reads.
"""
from __future__ import annotations

import re

import numpy as np

from fluoviz.analysis.frozen import Applier, ApplyData, ApplyOutput, FrozenError, SavedModel, Statuses
from fluoviz.analysis.frozen.kinds import _decomp as D

from ..core import predict as P
from .model import Evaluation, MblfModel

KIND = "mblf"
FLAG_WORDS = {"ok": "typical for the model", "outside": "unusual for the model", "unusable": "cannot be used"}


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")[:40] or "value"


def options_for() -> list[dict]:
    return [D.choice_option("limit", "Limit for an unusual sample", "99", [{"value": "95", "label": "95 %"}, {"value": "99", "label": "99 %"}],
                            "A new sample whose measurements are less probable under the model than those of all but 1 % (99 %) or 5 % (95 %) of the training samples is reported as "
                            "'Poorly described by the model'. The training samples are scored held out (leave-one-out), so the limit is honest.")]


def outputs_for(model: MblfModel) -> list[dict]:
    out = []
    for k in range(model.K):
        out.append({"name": f"mblf_f{k + 1}", "label": f"Latent factor {k + 1} (score)", "kind": "numeric",
                    "description": f"The score of the sample on factor {k + 1} (the posterior mean, in the orientation of the result's page): how strongly the pattern of factor {k + 1} is present."})
        out.append({"name": f"mblf_f{k + 1}_sd", "label": f"Latent factor {k + 1} (uncertainty of the score)", "kind": "numeric",
                    "description": f"The posterior standard deviation of the score on factor {k + 1}: large when the sample's measurements say little about this factor."})
    for m in model.modes:
        if m.kind == "vector":
            logs = list(m.log_cols or []) + [False] * m.M
            for j, lab in enumerate(m.column_labels):
                s = slug(lab)
                what = ("the median of the predictive distribution: the model works on the log of the value, and the value is given back in its own units" if logs[j] else
                        "the mean of the predictive distribution")
                out += [{"name": f"mblf_pred_{s}", "label": f"Predicted {lab}", "kind": "numeric",
                         "description": f"The value of {lab} the model predicts from the sample's other measurements ({what})."},
                        {"name": f"mblf_pred_{s}_lo", "label": f"Predicted {lab}, lower 95 % limit", "kind": "numeric",
                         "description": f"The lower end of the 95 % predictive interval of {lab}: every source of uncertainty is included."},
                        {"name": f"mblf_pred_{s}_hi", "label": f"Predicted {lab}, upper 95 % limit", "kind": "numeric",
                         "description": f"The upper end of the 95 % predictive interval of {lab}."}]
    out += [{"name": "mblf_surprise", "label": "Latent factors: how typical (log density per value)", "kind": "numeric",
             "description": "The log predictive density of the sample's measurements per measured value: higher = more typical; compare with the training samples."},
            {"name": "mblf_measured", "label": "Latent factors: what was measured", "kind": "category", "description": "Which of the model's modes the sample has."},
            {"name": "flag", "label": "Latent factors status in words", "kind": "category", "description": "Whether the sample is typical for the model."},
            {"name": "status", "label": "Status", "kind": "category", "description": "Fits the model / Poorly described by the model / Cannot be used."}]
    return out


def judge(model: MblfModel, ev: Evaluation, opts: dict) -> tuple[np.ndarray, list, np.ndarray, np.ndarray]:
    lev = str(opts["limit"])
    ref = model.reference.get("surprise") or {}
    n = ev.n
    lim = np.full(n, np.nan)
    for i in range(n):
        r = ref.get(model.pattern_key(ev.present[i]))
        if r:
            lim[i] = float(r["q01" if lev == "99" else "q05"])
    with np.errstate(invalid="ignore"):
        out = ev.usable & np.isfinite(lim) & (ev.surprise < lim)
    st = Statuses(n)
    st.flag(~ev.usable, "unusable", lambda i: " ".join(ev.why[i]) or "This sample cannot be used.")
    st.flag(out, "outside", lambda i: f"Its measurements are less probable under the model ({D.g(ev.surprise[i])} per value) than those of {'99' if lev == '99' else '95'} % of the training "
                                      f"samples with the same measurements (limit {D.g(lim[i])}): it contains something the training samples did not, or it was measured differently.")
    if ev.X is not None:
        unknown = ev.usable & ~np.isfinite(ev.X).all(axis=1)
        st.flag(unknown & ~out, "ok", lambda i: "Its covariates are not known (the metadata lack them), so it was scored with the covariates of the whole population: its predictions are wider.")
    flag = np.where(~ev.usable, FLAG_WORDS["unusable"], np.where(out, FLAG_WORDS["outside"], FLAG_WORDS["ok"]))
    return st.array(), st.reasons, flag.astype(str), lim


def publish(model: MblfModel, ev: Evaluation, status, flag) -> tuple[dict, dict]:
    var: dict = {}
    for k in range(model.K):
        var[f"mblf_f{k + 1}"] = ev.scores[:, k]
        var[f"mblf_f{k + 1}_sd"] = ev.scores_sd[:, k]
    for t, m in enumerate(model.modes):
        if m.kind != "vector":
            continue
        mean = np.full((ev.n, m.M), np.nan)
        lo, hi = mean.copy(), mean.copy()
        for i in np.flatnonzero(ev.usable):
            pr = model.predict(ev, i, t, noisy=True)
            centre = np.where(np.asarray(list(m.log_cols or []) + [False] * (m.M - len(m.log_cols or [])), bool), pr["quantiles"][3], pr["mean"])
            mean[i], lo[i], hi[i] = m.backward(centre), m.backward(pr["quantiles"][0]), m.backward(pr["quantiles"][-1])
        for j, lab in enumerate(m.column_labels):
            s = slug(lab)
            var[f"mblf_pred_{s}"], var[f"mblf_pred_{s}_lo"], var[f"mblf_pred_{s}_hi"] = mean[:, j], lo[:, j], hi[:, j]
    var["mblf_surprise"] = ev.surprise
    var["mblf_measured"] = np.array([model.pattern_key(r) or "nothing" for r in ev.present], dtype=object).astype(str)
    var["flag"] = flag
    var["status"] = status
    info = {o["name"]: {"kind": o["kind"], "label": o["label"], "description": o["description"]} for o in outputs_for(model)}
    return {k: v for k, v in var.items() if k in info}, info


def description_for(model: MblfModel, validation: dict | None = None) -> str:
    v = validation or {}
    what = D.join_words([m.label for m in model.modes])
    cov = "" if model.cov is None else f" The factor scores depend on the covariates {D.join_words(model.cov.labels)}."
    est = "Bayesian (the posterior of every parameter is kept)" if model.draws.estimation == "bayes" else "maximum likelihood (the parameters are treated as known)"
    return (f"A multimodal latent factor model of the {what}{f' of {v['n_samples']} samples' if v.get('n_samples') else ''}, with up to {model.K} shared factors, fitted by {est}.{cov} "
            "For each new sample the model reads the measurements it has exactly like the training samples (the same pre-treatment, mean and scale; nothing is estimated from the new "
            "samples), finds the scores of its factors, predicts what it lacks with every source of uncertainty, and compares how probable its measurements are with the training samples.")


class MblfApplier(Applier):
    def __init__(self, saved: SavedModel):
        super().__init__(saved)
        self.model = MblfModel(saved.meta.get("model"))

    def options(self) -> list[dict]:
        return [dict(o) for o in (self.saved.meta.get("options") or options_for())]

    def describe(self) -> dict:
        m = self.model
        v = self.saved.meta.get("validation") or {}
        training = {"Samples": int(v.get("n_samples", len(m.ids))), "Modes": D.join_words([x.label for x in m.modes]), "Largest number of factors": m.K,
                    "Estimation": "Bayesian posterior (MCMC)" if m.draws.estimation == "bayes" else "maximum likelihood", "Posterior draws used for new samples": int(m.idx.size)}
        if m.cov is not None:
            training["Covariates"] = D.join_words(m.cov.labels)
        for x in m.modes:
            if x.kind != "vector":
                training[f"{x.label.capitalize()}: basis"] = f"{x.p} Gaussian-process functions ({100 * (x.captured or 0):.1f} % of the prior variance)"
        inputs = [("The fluorescence matrix (EEM) of every sample, on the wavelength grid of the model." if x.name == "fluorescence" else
                   "The absorbance spectrum of every sample, on the wavelength grid of the model; the model cleans it like the training spectra." if x.name == "absorbance" else
                   f"The values {D.join_words(x.column_labels)} when the metadata hold them (otherwise they are predicted).") for x in m.modes]
        inputs.append("A sample needs at least ONE of these; what it lacks is predicted.")
        if m.cov is not None:
            inputs.append(f"The covariates {D.join_words(m.cov.labels)} from the metadata (a sample without them is scored with the covariates unknown).")
        caveats = ["Every new sample is read exactly like the training samples; nothing is estimated again from the new samples.",
                   "The predictions average over the posterior of the training fit (the 'modular' rule of the method): a new sample informs its own scores, not the population parameters.",
                   "Predicted values are only as good as the link the training samples showed: see the page of the result (Prediction tab) for the held-out coverage of the intervals."]
        lims = [D.limit_entry(f"How typical, {lev} % limit (all modes measured)", (m.reference.get("surprise") or {}).get(m.pattern_key([True] * len(m.modes)), {}).get("q01" if lev == "99" else "q05"),
                              "The log predictive density per value below which a sample is reported as unusual.") for lev in ("99",)]
        return D.describe_block(self.saved, inputs, training, lims, caveats)

    def apply(self, data: ApplyData, options: dict, progress) -> ApplyOutput:
        m = self.model
        opt = D.settings(self.options(), options)
        progress(0.03, "reading the samples")
        ev = m.evaluate(data, opt.get("min_coverage", 0.5))
        progress(0.5, "judging the samples")
        status, reasons, flag, lim = judge(m, ev, opt)
        progress(0.6, "predicting what the samples lack")
        variables, info = publish(m, ev, status, flag)
        plot = D.plot_block("Scores of the factors", [f"mblf_f{k + 1}" for k in range(m.K)], [f"Factor {k + 1}" for k in range(m.K)], ev.scores,
                            np.asarray(m.reference.get("scores"), float) if m.reference.get("scores") is not None else None, m.reference.get("ids"))
        diag = [D.diagnostic("How typical the samples are", "Each point is a sample: its log predictive density per measured value (higher = more typical). The grey points are the training "
                             "samples, scored held out.", "Log density per value", ev.surprise, m.reference.get("surprise_values"), float(np.nanmedian(lim)) if np.isfinite(lim).any() else None,
                             f"{opt['limit']} % limit")]
        payload = {"plot": plot, "diagnostics": diag, "notes": list(data.notes)}
        summary = D.summary_block(status, n_unusual=int(np.sum(status == "outside")), limit=str(opt["limit"]), modes=[x.name for x in m.modes])
        text = (f"The saved model '{self.saved.title}' (multimodal latent factors of the {D.join_words([x.label for x in m.modes])}) was applied to {D.plural(data.n, 'new sample')}. "
                f"{D.counts_sentence(status)}  Method: each sample was read like the training samples, its factor scores were computed from what it has (with "
                f"{int(m.idx.size)} posterior draws), what it lacks was predicted, and the probability of its measurements was compared with the training samples (held out).")
        progress(1.0, "done")
        return ApplyOutput(variables=variables, var_info=info, status=status, reasons=reasons, payload=payload, summary=summary, text=text)

    def inspect(self, data: ApplyData, options: dict, index: int) -> dict:
        m = self.model
        opt = D.settings(self.options(), options)
        i = int(index)
        if not 0 <= i < data.n:
            raise FrozenError("There is no such sample in this application.")
        sub = data.subset([i])
        ev = m.evaluate(sub, opt.get("min_coverage", 0.5))
        if not ev.usable[0]:
            raise FrozenError(" ".join(ev.why[0]) or "This sample cannot be shown against the model.")
        lacking = [t for t in range(len(m.modes)) if not ev.present[0, t] and m.modes[t].kind != "vector"]
        t = lacking[0] if lacking else next((t for t, x in enumerate(m.modes) if x.kind == "surface"), 0)
        mode = m.modes[t]
        pr = m.predict(ev, 0, t)
        bud = pr["budget_total"]
        words = "; ".join(f"{100 * bud[k]:.0f} % from {P.PART_WORDS[k]}" for k in P.PARTS if bud[k] > 0.005)
        measured = ev.present[0, t]
        text = (f"{data.ids[i]}: the {mode.label} {'as measured and as predicted from the other measurements' if measured else 'predicted from what the sample has'}"
                f" ({m.pattern_key(ev.present[0]) or 'nothing'}). The uncertainty of the prediction comes {words}.")
        if mode.kind == "surface":
            el = mode.element
            from fluoviz.analysis.frozen.kinds.fpca import ElementRuntime
            dom = ElementRuntime(el).dom
            sd = np.sqrt(sum(pr["budget"].values()))
            images = {"predicted": dom.embed(pr["mean"]), "lower": dom.embed(pr["quantiles"][0]), "upper": dom.embed(pr["quantiles"][-1]), "uncertainty": dom.embed(sd)}
            labels = {"predicted": "Predicted (posterior mean)", "lower": "Lower 95 % limit", "upper": "Upper 95 % limit", "uncertainty": "Predictive standard deviation"}
            if measured:
                images["measured"] = dom.embed(mode.to_data(ev.Y[t][:, 0]))
                labels["measured"] = "Measured (after the pre-treatment)"
            return D.inspect_block(images, labels, text, ex=np.asarray(el.axes[1], float), em=np.asarray(el.axes[0], float), mask=~np.asarray(el.obs, bool), units=mode.units,
                                   facts={"budget": bud}, title=f"{data.ids[i]}: the {mode.label}")
        out = D.inspect_block({}, {}, text, facts={"budget": bud}, title=f"{data.ids[i]}: the {mode.label}")
        x = np.asarray(mode.element.axes[0], float) if mode.element is not None else np.arange(1, mode.M + 1.0)
        emb = mode.embed
        series = [{"name": "Predicted (posterior mean)", "values": emb(pr["mean"])}, {"name": "Lower 95 % limit", "values": emb(pr["quantiles"][0])},
                  {"name": "Upper 95 % limit", "values": emb(pr["quantiles"][-1])}]
        if measured:
            series.insert(0, {"name": "Measured", "values": emb(mode.to_data(ev.Y[t][:, 0]))})
        out["curves"] = {"x": x, "label": mode.label.capitalize(), "series": series}
        return out
