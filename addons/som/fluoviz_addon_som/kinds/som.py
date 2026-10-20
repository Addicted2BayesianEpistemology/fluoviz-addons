"""The saved model of kind ``som``: apply a fitted map to NEW samples (position, unit, quantisation error, group, status) without fitting anything again.

A new sample is read at the pixels of the domain, pre-treated, smoothed on ITS OWN observed pixels in the stored representation (functional) or read as it is (pixels, gaps allowed),
transformed by the stored metric, and assigned to its best unit.  Status:

* ``unusable``: a variable is missing, a pre-treatment is not defined, or fewer than ``min_coverage`` of a domain is measured (all outputs NaN);
* ``outside``: the distance to the best unit is above the limit of the model (the 95 % or 99 % point of the distances of samples the map was NOT trained on);
* ``uncertain``: the best unit is an EMPTY unit (no training sample landed on it): the sample sits between what the map knows.

Published variables: ``som_unit`` ``som_row`` ``som_col`` ``som_x`` ``som_y`` ``som_qe`` ``som_qe_rel`` ``som_qe_rmse`` (one variable) or ``som_qe_rmse_<variable>`` ``som_umatrix``
``som_cluster`` (once the units were grouped) ``flag`` ``status``.  Plain English in everything a person reads.  Modelled on ``kinds/autoencoder.py`` of the autoencoders add-on.
"""
from __future__ import annotations

import numpy as np

from fluoviz.analysis.frozen import Applier, ApplyData, ApplyOutput, FrozenError, SavedModel, Statuses
from fluoviz.analysis.frozen.kinds import _decomp as D
from fluoviz.analysis.frozen.kinds.fpca import basis_words, smoothing_words

from ..core.spec import METRIC_WORDS
from .model import Evaluation, SomModel

KIND = "som"
FLAG_WORDS = {"ok": "inside the limit", "error": "far from every unit of the map", "empty": "on a unit no training sample reached", "unusable": "cannot be used"}


def options_for() -> list[dict]:
    return [D.choice_option("limit", "Limit for the distance to the best unit", "99", [{"value": "95", "label": "95 %"}, {"value": "99", "label": "99 %"}],
                            "A new sample that lies farther from its best unit than this limit is reported as 'Poorly described by the model'. The limit is the 95 % or 99 % point of the distances "
                            "of samples the map was not trained on: 99 % flags only clear departures (about 1 ordinary sample in 100), 95 % is stricter (about 1 in 20).")]


def outputs_for(model: SomModel) -> list[dict]:
    blocks = model.names
    out = [{"name": "som_unit", "label": "Map unit (row and column)", "kind": "category", "description": "The unit of the map whose prototype is closest to the sample, written r3c5 for row 3, column 5."},
           {"name": "som_row", "label": "Map row of the unit", "kind": "numeric", "description": "The row (1 = first) of the unit the sample falls on."},
           {"name": "som_col", "label": "Map column of the unit", "kind": "numeric", "description": "The column (1 = first) of the unit the sample falls on."},
           {"name": "som_x", "label": "Map position, across", "kind": "numeric",
            "description": "The position of the sample between the units, across the map (in unit spacings): it sits between the units it resembles. Use it, with the down position, to plot or cluster the samples on the map."},
           {"name": "som_y", "label": "Map position, down", "kind": "numeric", "description": "The position of the sample between the units, down the map (in unit spacings; a row of a hexagonal map is 0.87 high)."},
           {"name": "som_qe", "label": "Map quantisation error (distance to the best unit)", "kind": "numeric",
            "description": "The distance between the sample and the prototype of its best unit, in the distance the map was trained with."},
           {"name": "som_qe_rel", "label": "Map quantisation error, share of the variation", "kind": "numeric",
            "description": "The share of the typical variation between samples that the prototype of the sample's unit does not capture (0 = the sample is its prototype, about 1 = as far as the average sample)."}]
    if len(blocks) == 1:
        out.append({"name": "som_qe_rmse", "label": "Map quantisation error (RMS, data units)", "kind": "numeric",
                    "description": "The root mean square difference between the sample and its prototype, in the units of the (pre-treated) data."})
    else:
        out += [{"name": f"som_qe_rmse_{b}", "label": f"Map quantisation error of the {b} (RMS, data units)", "kind": "numeric",
                 "description": f"The root mean square difference between the {b} of the sample and of its prototype, in the units of the (pre-treated) data."} for b in blocks]
    out.append({"name": "som_umatrix", "label": "Map U-matrix value at the unit", "kind": "numeric",
                "description": "How different the neighbouring prototypes are around the unit: high = the sample sits on a ridge between two regions of the map."})
    if model.has_clusters():
        out.append({"name": "som_cluster", "label": "Map group", "kind": "category", "description": "The group of units (second-level clustering of the prototypes) the sample's unit belongs to."})
    out += [{"name": "flag", "label": "Map status in words", "kind": "category", "description": "Whether the sample is inside the limit of the map and on a unit the training samples reached."},
            {"name": "status", "label": "Status", "kind": "category", "description": "Fits the model / Poorly described by the model / Cannot be used."}]
    return out


class Judgement:
    def __init__(self, n: int):
        self.n = n
        self.status = np.array([], str)
        self.reasons: list = []
        self.flag = np.array([], str)
        self.err_out = np.zeros(n, bool)
        self.empty = np.zeros(n, bool)


def judge(model: SomModel, ev: Evaluation, ref: dict, opts: dict) -> Judgement:
    n = ev.n
    jd = Judgement(n)
    lev = str(opts["limit"])
    lim = float(ref["limits"][lev])
    u = ev.usable
    with np.errstate(invalid="ignore"):
        jd.err_out = u & np.isfinite(lim) & (ev.qe > lim)
    jd.empty = u & ev.empty
    st = Statuses(n)
    st.flag(~u, "unusable", lambda i: " ".join(ev.why[i]) or "This sample cannot be used.")
    st.flag(jd.err_out, "outside", lambda i: f"Its distance to the best unit ({D.g(ev.qe[i])}) is above the {lev} % limit ({D.g(lim)}): the map has no unit that stands for this sample, it contains "
                                             "something the training samples did not have, or it was measured differently.")
    st.flag(jd.empty & ~jd.err_out, "uncertain", lambda i: f"Its best unit ({model.unit_name(ev.unit[i])[0]}) is a unit no training sample reached: the sample sits between what the map knows, so its place on "
                                                            "the map is less certain.")
    st.flag(ev.gaps & u, "ok", lambda i: "Some pixels of this sample were not measured: its place on the map was worked out from the pixels it has"
                                         f" ({D.percent(1 - float(np.nanmin(ev.coverage[i])), 2)} of the domain missing).")
    jd.status = st.array()
    jd.reasons = st.reasons
    flag = np.where(jd.err_out, FLAG_WORDS["error"], np.where(jd.empty, FLAG_WORDS["empty"], FLAG_WORDS["ok"]))
    jd.flag = np.where(u, flag, FLAG_WORDS["unusable"]).astype(str)
    return jd


def publish(model: SomModel, ev: Evaluation, jd: Judgement) -> tuple[dict, dict]:
    u = ev.usable
    r, c = model.lattice.row_col(np.where(u, ev.unit, 0))
    var: dict = {}
    var["som_unit"] = np.where(u, model.unit_name(np.where(u, ev.unit, 0)).astype(str), "").astype(str)
    var["som_row"] = np.where(u, r + 1.0, np.nan)
    var["som_col"] = np.where(u, c + 1.0, np.nan)
    var["som_x"], var["som_y"] = ev.pos[:, 0], ev.pos[:, 1]
    var["som_qe"] = ev.qe
    var["som_qe_rel"] = ev.qe_rel
    if len(model.names) == 1:
        var["som_qe_rmse"] = ev.qe_blocks[:, 0]
    else:
        for k, b in enumerate(model.names):
            var[f"som_qe_rmse_{b}"] = ev.qe_blocks[:, k]
    var["som_umatrix"] = np.where(u, model.umat[np.where(u, ev.unit, 0)], np.nan)
    if model.has_clusters():
        cl = model.unit_cluster[np.where(u, ev.unit, 0)]
        var["som_cluster"] = np.where(u & (cl >= 0), np.array([f"group {int(k) + 1}" for k in cl]), "").astype(str)
    var["flag"] = jd.flag
    var["status"] = jd.status
    info = {o["name"]: {"kind": o["kind"], "label": o["label"], "description": o["description"]} for o in outputs_for(model)}
    return {k: v for k, v in var.items() if k in info}, info


def reference_for(model: SomModel, ids: list, ev: Evaluation, heldout_qe: np.ndarray | None) -> dict:
    """The ``reference`` of a saved model: the training samples' positions and distances, and the limits (from the out-of-fold distances when there are any)."""
    from . import stats as S
    held = np.asarray(heldout_qe, float) if heldout_qe is not None else np.full(ev.n, np.nan)
    limits = S.distance_limits(held[np.isfinite(held)], fallback=ev.qe[ev.usable])
    return {"ids": [str(i) for i in ids], "score_names": ["som_x", "som_y"], "score_labels": ["Map position, across", "Map position, down"], "scores": ev.pos.copy(), "unit": ev.unit.copy(),
            "qe": ev.qe.copy(), "heldout_qe": held, "limits": limits}


def block_words(fe, mode: str) -> str:
    if mode == "pixels":
        return "the observed pixels themselves (no smoothing; unmeasured pixels are left out of the distance)"
    return f"{basis_words(fe)}, smoothed with a {smoothing_words(fe)}"


def description_for(model: SomModel, validation: dict | None = None) -> str:
    v = validation or {}
    what = D.join_words([{"fluorescence": "fluorescence matrices (EEMs)", "absorbance": "absorbance spectra"}[nm] for nm in model.names])
    lat = model.lattice
    reading = "; ".join(f"the {b.name} is read as {block_words(b.element, 'pixels' if b.kind == 'pixels' else 'functional')}" for b in model.frame.blocks)
    s = (f"A self-organising map of the {what}{f' of {v['n_samples']} samples' if v.get('n_samples') else ''}: {lat.rows} × {lat.cols} {lat.kind} units{', wrapping around' if lat.shape == 'toroid' else ''}, "
         f"each holding a prototype. Samples are compared with the prototypes with the {METRIC_WORDS[model.metric.kind]}; {reading}. ")
    if v.get("captured") is not None:
        s += f"The map captures {100 * float(v['captured']):.1f} % of the variation between the samples"
        if v.get("heldout_captured") is not None:
            s += f" ({100 * float(v['heldout_captured']):.1f} % on samples it was not trained on)"
        s += ". "
    s += ("For each new sample the model brings it into the same state as the training samples (the same smoothing and centring, never estimated again from the new samples), finds its best unit and its "
          "position between the units, and compares its distance to the unit with the limit of the training samples.")
    return s


class SomApplier(Applier):
    def __init__(self, saved: SavedModel):
        super().__init__(saved)
        m = saved.meta
        self.model = SomModel(m.get("model"))
        self.ref = m.get("reference") or {}
        if "limits" not in self.ref:
            raise FrozenError("This model file is damaged: its limits are missing.")

    def options(self) -> list[dict]:
        return [dict(o) for o in (self.saved.meta.get("options") or options_for())]

    def _limits(self, opts: dict) -> list[dict]:
        lev = str(opts["limit"])
        lims = self.ref["limits"]
        return [D.limit_entry(f"Distance to the best unit, {lev} % limit", lims[lev], f"{lims.get('words', '')} A sample above it is poorly described by the map.")]

    def describe(self) -> dict:
        m, ref = self.model, self.ref
        v = self.saved.meta.get("validation") or {}
        lat = m.lattice
        training = {"Samples": int(v.get("n_samples", len(ref.get("ids", [])))), "Map": f"{lat.rows} × {lat.cols} {lat.kind} units ({lat.shape})", "Distance": METRIC_WORDS[m.metric.kind],
                    "Units with at least one sample": f"{int((m.hits > 0).sum())} of {m.M}"}
        for b in m.frame.blocks:
            training[f"{b.name.capitalize()}: how it is read"] = block_words(b.element, "pixels" if b.kind == "pixels" else "functional")
        if v.get("captured") is not None:
            training["Variation captured by the map"] = f"{100 * float(v['captured']):.1f} %"
        if v.get("heldout_captured") is not None:
            training["Variation captured on held-out samples"] = f"{100 * float(v['heldout_captured']):.1f} %"
        inputs = []
        for b in m.frame.blocks:
            inputs.append("The fluorescence matrix (EEM) of every sample, on the wavelength grid of the model." if b.name == "fluorescence" else
                          "The absorbance spectrum of every sample, on the wavelength grid of the model" + ("; the model cleans it like the training spectra." if b.element.ife else "."))
        if len(m.frame.blocks) > 1:
            inputs.append("A sample needs BOTH: a sample without an absorbance spectrum cannot be used.")
        caveats = ["Every new sample is brought to the map's input exactly like the training samples (the same smoothing parameters and centring); nothing is estimated again from the new samples.",
                   "The limit for the distance to the best unit comes from samples the map was not trained on; with a small training set it is only a rough guide.",
                   "A map describes well what resembles its training samples: a sample measured under other conditions is reported as poorly described, which is the point of the check, "
                   "but a sample can also be flagged because it was measured differently.",
                   "A new sample that falls on a unit no training sample reached is reported as uncertain: its place on the map is an interpolation."]
        defaults = D.settings(self.options(), None)
        return D.describe_block(self.saved, inputs, training, self._limits(defaults), caveats)

    def apply(self, data: ApplyData, options: dict, progress) -> ApplyOutput:
        m, ref = self.model, self.ref
        opt = D.settings(self.options(), options)
        lev = str(opt["limit"])
        progress(0.03, "reading the samples")
        ev = m.evaluate(data, opt["min_coverage"])
        progress(0.6, "judging the samples")
        jd = judge(m, ev, ref, opt)
        variables, info = publish(m, ev, jd)
        progress(0.85, "preparing the plots")
        payload = self._payload(data, ev, jd, opt)
        status = jd.status
        summary = D.summary_block(status, n_outside_distance=int(jd.err_out.sum()), n_empty_unit=int(jd.empty.sum()), limit=lev, map=[m.lattice.rows, m.lattice.cols], variables=m.names)
        text = self._text(data, status, opt)
        progress(1.0, "done")
        return ApplyOutput(variables=variables, var_info=info, status=status, reasons=jd.reasons, payload=payload, summary=summary, text=text)

    def _payload(self, data: ApplyData, ev: Evaluation, jd: Judgement, opt: dict) -> dict:
        m, ref = self.model, self.ref
        lev = str(opt["limit"])
        lim = float(ref["limits"][lev])
        plot = D.plot_block("Position on the map", ["som_x", "som_y"], ["Map position, across", "Map position, down"], ev.pos, None if ref.get("scores") is None else np.asarray(ref["scores"], float),
                            ref.get("ids"))
        diag = [D.diagnostic("Distance to the best unit", "Each point is a sample; the distance to the prototype of its best unit is shown across. A new sample beyond the dotted line is poorly described by the "
                             "map; the grey points are the training samples.", "Distance to the best unit", ev.qe, ref.get("qe"), lim, f"{lev} % limit")]
        payload: dict = {"plot": plot, "diagnostics": diag, "limits": self._limits(opt), "notes": list(data.notes), "map": [m.lattice.rows, m.lattice.cols]}
        mean = m.mean_functions()
        for b in m.frame.blocks:
            fe = b.element
            if fe.kind == "surface":
                payload["images"] = {"mean": m.embed(b.name, mean[b.name])[None]}
                payload["image_labels"] = {"mean": ["Mean of the training samples"]}
                payload["ex"], payload["em"], payload["mask"] = np.asarray(fe.axes[1], float), np.asarray(fe.axes[0], float), ~np.asarray(fe.obs, bool)
            else:
                payload["curves"] = {"x": np.asarray(fe.axes[0], float), "label": "Absorbance", "series": [{"name": "Mean of the training samples", "values": m.embed(b.name, mean[b.name])}]}
        if not np.isfinite(lim):
            payload["notes"].append("The model has no limit for the distance to the best unit (too few samples were held out): samples are judged by the unit they fall on only.")
        return payload

    def _text(self, data: ApplyData, status: np.ndarray, opt: dict) -> str:
        m, ref = self.model, self.ref
        lev = str(opt["limit"])
        lim = float(ref["limits"][lev])
        reading = D.join_words([f"the {b.name} as {block_words(b.element, 'pixels' if b.kind == 'pixels' else 'functional')}" for b in m.frame.blocks])
        return (f"The saved map '{self.saved.title}' ({m.lattice.rows} × {m.lattice.cols} units, trained on {D.plural(len(ref.get('ids', [])), 'sample')}) was applied to {D.plural(data.n, 'new sample')}. "
                f"{D.counts_sentence(status)}  Method: each new sample was read exactly like the training samples ({reading}; the smoothing parameters and centring of the training samples were used and "
                f"nothing was estimated from the new samples), compared with the prototypes with the {METRIC_WORDS[m.metric.kind]}, and placed on its best unit; its distance to the unit was compared with "
                f"the {lev} % limit of the samples the map was not trained on ({D.g(lim)}).")

    def inspect(self, data: ApplyData, options: dict, index: int) -> dict:
        m, ref = self.model, self.ref
        opt = D.settings(self.options(), options)
        lev = str(opt["limit"])
        i = int(index)
        if not 0 <= i < data.n:
            raise FrozenError("There is no such sample in this application.")
        ev = m.evaluate(data.subset([i]), opt["min_coverage"])
        if not ev.usable[0]:
            raise FrozenError(" ".join(ev.why[0]) or "This sample cannot be shown against the model.")
        jd = judge(m, ev, ref, opt)
        u = int(ev.unit[0])
        protos = m.prototype_functions([u])
        mean = m.mean_functions()
        images, labels, curves, ex, em, mask = {}, {}, None, None, None, None
        for b in m.frame.blocks:
            fe = b.element
            y = ev.treated[b.name][0]
            pr = protos[b.name][0]
            if fe.kind == "surface":
                images.update({"measured": m.embed(b.name, y), "prototype": m.embed(b.name, pr), "difference": m.embed(b.name, np.where(np.isfinite(y), y - pr, np.nan)), "mean": m.embed(b.name, mean[b.name])})
                labels.update({"measured": "Measured (as the model sees it, after the pre-treatment)", "prototype": f"Prototype of unit {m.unit_name(u)[0]}",
                               "difference": "Measured minus prototype", "mean": "Mean of the training samples"})
                ex, em, mask = np.asarray(fe.axes[1], float), np.asarray(fe.axes[0], float), ~np.asarray(fe.obs, bool)
            else:
                curves = {"x": np.asarray(fe.axes[0], float), "label": "Absorbance", "series": [{"name": "Measured", "values": m.embed(b.name, y)}, {"name": f"Prototype of unit {m.unit_name(u)[0]}",
                          "values": m.embed(b.name, pr)}, {"name": "Mean of the training samples", "values": m.embed(b.name, mean[b.name])}]}
        lim = float(ref["limits"][lev])
        unit = m.unit_name(u)[0]
        if jd.status[0] == "ok":
            text = f"{data.ids[i]} falls on unit {unit}, at distance {D.g(ev.qe[0])} from its prototype (inside the {lev} % limit {D.g(lim)})."
        else:
            text = f"{data.ids[i]}: " + " ".join(jd.reasons[0])
        facts = {"unit": unit, "position": ev.pos[0], "distance": float(ev.qe[0]), "limit": lim, "level": lev}
        out = D.inspect_block(images, labels, text, signed=("difference",), ex=ex, em=em, mask=mask, units=data.units or "", facts=facts, title=f"{data.ids[i]} against the map")
        if curves is not None:
            out["curves"] = curves
        return out
