// Step 7 of the Multimodal latent factors page: predicting one mode of a sample from what is known about it, with every source of uncertainty, and checking the predictions held out.
//   * One sample: choose the sample, what to predict and from what (any of its other modes, its covariates or not); held out (the model without this sample, by Pareto-smoothed
//     importance sampling) or not; the latent function or a new measurement.  The prediction is drawn with its fan of bands, the measured values on top, its uncertainty budget,
//     and the factor scores of the sample with their intervals.
//   * Every choice of sources compared held out: the log predictive density, the coverage of the 50 / 80 / 95 % intervals, the error and the width, the calibration curve and the
//     histogram of the PIT values.
import { computed, watch } from 'vue';
import { Icon } from '/static/js/icons.js';
import { Callout, Seg, Switch, Field } from '/static/js/components/Common.js';
import { JobBar } from '/static/js/components/JobBar.js';
import { EEMView } from '/static/js/components/EEMView.js';
import { PlotlyChart, ChartCard } from '/static/js/plot.js';
import { FpcaFrame } from '/static/js/components/FpcaStepper.js';
import { FpcaExplain } from '/static/js/components/FpcaCommon.js';
import { tokens, themeState } from '/static/js/theme.js';
import { fp, mbJobs, MODE_NICE } from './MblfState.js';
import { samplePrediction, runCompare } from './MblfActions.js';
import { MblfVerdict, MblfSection, MblfModeView, MblfBudget, nd, img, arr, modeOf, modeLabel, sourceWords, pct, num } from './MblfCommon.js';
import { MblfGate, MblfStaleNote } from './MblfResults.js';

const MblfOneSample = {
  name: 'MblfOneSample',
  components: { Icon, Callout, Seg, Switch, Field, EEMView, PlotlyChart, ChartCard, MblfModeView, MblfBudget, MblfVerdict },
  setup() {
    const p = computed(() => fp.mb.res.p);
    const P = computed(() => fp.mb.pred);
    const ids = computed(() => p.value.samples.ids);
    const D = computed(() => p.value.modes.length);
    const pres = computed(() => nd(p.value.samples.present));
    const has = (i, d) => !!pres.value.data[i * D.value + d];
    const sample = computed(() => (P.value.sample && ids.value.includes(P.value.sample) ? P.value.sample : ids.value[0]));
    const si = computed(() => ids.value.indexOf(sample.value));
    const targets = computed(() => p.value.modes.map((m) => ({ value: m.name, label: MODE_NICE[m.name] || m.name })));
    const target = computed(() => P.value.target || (p.value.modes.find((m) => m.kind === 'surface') || p.value.modes[0]).name);
    const others = computed(() => p.value.modes.map((m, d) => ({ ...m, d })).filter((m) => m.name !== target.value));
    const chosen = computed(() => (P.value.sources === null ? others.value.map((m) => m.name) : P.value.sources.filter((s) => others.value.some((m) => m.name === s))));
    const toggle = (name) => {
      const cur = chosen.value.slice(), i = cur.indexOf(name);
      if (i >= 0) cur.splice(i, 1); else cur.push(name);
      P.value.sources = cur;
    };
    const hasCov = computed(() => !!p.value.covariates);
    const bayes = computed(() => p.value.estimation === 'bayes');
    const pred = computed(() => samplePrediction({ sample: sample.value, target: target.value, sources: chosen.value.length ? chosen.value.join(',') : 'none', covariates: hasCov.value && P.value.covariates,
      noisy: P.value.noisy, heldOut: bayes.value && P.value.heldOut }));
    const R = computed(() => (pred.value && pred.value.data) || null);
    const mode = computed(() => modeOf(fp.mb.res, target.value));
    const summary = computed(() => (R.value ? { mean: R.value.mean, quantiles: R.value.quantiles, band: R.value.band } : null));
    const scoreTraces = computed(() => {
      themeState.version;
      if (!R.value) return [];
      const s = R.value.scores, t = tokens(), K = s.mean.length || nd(s.mean).data.length;
      const m = arr(nd(s.mean)), lo = arr(nd(s.lower)), hi = arr(nd(s.upper));
      const x = Array.from({ length: K }, (_, k) => `Factor ${k + 1}`);
      return [{ x, y: m, type: 'scatter', mode: 'markers', marker: { size: 10, color: t.series[0] }, name: 'Posterior mean, 95 % interval',
        error_y: { type: 'data', symmetric: false, array: hi.map((v, i) => v - m[i]), arrayminus: m.map((v, i) => v - lo[i]), color: t.series[0], thickness: 1.5, width: 6 } }];
    });
    const scoreLayout = { yaxis: { title: 'Score', zeroline: true }, margin: { t: 10 }, showlegend: false };
    const mask = computed(() => (mode.value && mode.value.mask ? img(nd(mode.value.mask)) : null));
    const pAbove = computed(() => (R.value && mode.value.kind === 'surface' ? img(nd(R.value.p_above)) : null));
    const observedImg = computed(() => (R.value && R.value.observed && mode.value.kind === 'surface' ? img(nd(R.value.observed)) : null));
    const presence = (i) => p.value.modes.filter((_, d) => has(i, d)).map((m) => MODE_NICE[m.name][0]).join('');
    const sampleOptions = computed(() => ids.value.map((id, i) => ({ id, label: `${id}  [${presence(i)}]` })));
    const hasMode = (name) => has(si.value, p.value.modes.findIndex((m) => m.name === name));
    return { p, P, sample, sampleOptions, targets, target, others, chosen, toggle, hasCov, bayes, pred, R, mode, summary, scoreTraces, scoreLayout, mask, pAbove, observedImg, hasMode,
      sourceWords, pct, num, modeLabel, MODE_NICE };
  },
  template: `
  <div class="col gap-12" data-mblf-one>
    <div class="mblf-pred-controls">
      <Field label="Sample" help="The letters say what the sample has: F = fluorescence, A = absorbance, V = values."><select class="select sm" :value="sample" @change="P.sample = $event.target.value" data-mblf-sample><option v-for="o in sampleOptions" :key="o.id" :value="o.id">{{ o.label }}</option></select></Field>
      <Field label="Predict"><Seg :model-value="target" :options="targets" small @update:modelValue="(v) => { P.target = v; P.sources = null; }" data-mblf-pred-target/></Field>
      <Field label="From" help="What is known about the sample. A mode the sample lacks cannot be used (it is greyed out).">
        <div class="row wrap"><label v-for="m in others" :key="m.name" class="check" :class="{ muted: !hasMode(m.name) }"><input type="checkbox" :checked="chosen.includes(m.name)" @change="toggle(m.name)" :data-mblf-source="m.name"/> {{ MODE_NICE[m.name] }}{{ hasMode(m.name) ? '' : ' (not measured)' }}</label>
          <label v-if="hasCov" class="check"><input type="checkbox" v-model="P.covariates" data-mblf-source="covariates"/> Covariates</label></div>
      </Field>
      <div class="col gap-4">
        <Switch v-if="bayes" v-model="P.heldOut" label="Held out (as if the model had never seen this sample)" data-mblf-heldout/>
        <Switch v-model="P.noisy" label="A new measurement (include the noise of the instrument)" data-mblf-noisy/>
      </div>
    </div>
    <p class="muted small" v-if="pred && pred.busy">Predicting …</p>
    <Callout kind="critical" v-if="pred && pred.error">{{ pred.error }}</Callout>
    <template v-if="R">
      <MblfVerdict level="info" data-mblf-pred-text>{{ R.text }}<span v-if="R.inside95 !== null && R.inside95 !== undefined">&#32;{{ pct(R.inside95, 0) }} of the measured values lie inside the 95 % interval{{ R.held_out ? '' : ' (not held out: optimistic)' }}.</span><span v-if="R.held_out && R.pareto_k > 0.7"> The held-out weights are unreliable for this sample (Pareto k {{ num(R.pareto_k, 2) }}): it is influential.</span></MblfVerdict>
      <div class="mblf-grid">
        <div>
          <MblfModeView :mode="mode" :summary="summary" :observed="mode.kind !== 'surface' ? R.observed : null" :title="(R.noisy ? 'A new measurement of the ' : 'The ') + mode.label + ' predicted' + (mode.kind === 'curve' ? ' (fan: 50, 80, 95 %; dashed: the whole curve)' : mode.kind === 'vector' ? ' (95 % and 50 % intervals)' : '')" :units="mode.units" :stats="['mean', 'lower', 'upper', 'width']" :export-name="'prediction-' + sample"/>
        </div>
        <div v-if="mode.kind === 'surface'" class="col gap-8">
          <div v-if="observedImg"><div class="xs"><b>Measured</b></div><EEMView :ex="mode.ex" :em="mode.em" :z="observedImg" :mask="mask" compact :height="230"/></div>
          <div><div class="xs"><b>Probability that the sample is above the mean of the training samples</b> (near 1 = surely above, near 0 = surely below)</div><EEMView :ex="mode.ex" :em="mode.em" :z="pAbove" :mask="mask" kind="diverging" :zrange="[0, 1]" units="probability" compact :height="230"/></div>
        </div>
      </div>
      <MblfBudget :mode="mode" :budget="R.budget" :total="R.budget_total" :noisy="R.noisy"/>
      <ChartCard title="Factor scores of this sample, given what is known about it" help="The posterior of the scores of the sample from the chosen sources (aligned as in the Factors tab). Wide intervals: the sources say little about that factor." :data="scoreTraces" flat>
        <PlotlyChart :data="scoreTraces" :layout="scoreLayout" :height="220"/>
      </ChartCard>
    </template>
  </div>`,
};

const MblfCompare = {
  name: 'MblfCompare',
  components: { Icon, Callout, Seg, JobBar, PlotlyChart, ChartCard, MblfVerdict },
  setup() {
    const p = computed(() => fp.mb.res.p);
    const C = computed(() => fp.mb.cmp);
    const targets = computed(() => p.value.modes.map((m) => ({ value: m.name, label: MODE_NICE[m.name] || m.name })));
    const target = computed(() => C.value.target || (p.value.modes.find((m) => m.kind === 'vector') || p.value.modes[0]).name);
    const block = computed(() => (p.value.compare || {})[target.value] || null);
    const rows = computed(() => (block.value ? block.value.rows.slice().sort((a, b) => b.elpd - a.elpd) : []));
    const running = computed(() => mbJobs.compare.state.running);
    const run = () => runCompare([target.value]);
    const sel = computed(() => rows.value.find((r) => r.key === C.value.row) || rows.value[0] || null);
    const nominal = [0.5, 0.8, 0.95];
    const calTraces = computed(() => {
      themeState.version;
      const t = tokens();
      const out = [{ x: [0, 1], y: [0, 1], type: 'scatter', mode: 'lines', line: { color: t.text3, dash: 'dot', width: 1 }, name: 'Perfect calibration', hoverinfo: 'skip' }];
      rows.value.forEach((r, i) => out.push({ x: nominal, y: nominal.map((l) => r.coverage[String(l)]), type: 'scatter', mode: 'lines+markers', name: sourceWords(r.sources, r.covariates),
        line: { width: r === sel.value ? 3 : 1.5, color: t.series[i % 8] }, marker: { size: r === sel.value ? 9 : 6, color: t.series[i % 8] } }));
      return out;
    });
    const calLayout = { xaxis: { title: 'Nominal coverage of the interval', range: [0.4, 1], tickformat: '.0%' }, yaxis: { title: 'Measured values inside (held out)', range: [0, 1], tickformat: '.0%' }, margin: { t: 10 }, legend: { orientation: 'h', y: -0.35 } };
    const pitTraces = computed(() => {
      themeState.version;
      if (!sel.value || !sel.value.pit_hist) return [];
      const t = tokens();
      return [{ x: sel.value.pit_hist.map((_, i) => (i + 0.5) / 10), y: sel.value.pit_hist, type: 'bar', marker: { color: t.series[0] }, name: 'PIT', hovertemplate: '%{y:.1%}<extra></extra>' },
        { x: [0, 1], y: [0.1, 0.1], type: 'scatter', mode: 'lines', line: { color: t.text3, dash: 'dot' }, name: 'Flat = calibrated', hoverinfo: 'skip' }];
    });
    const pitLayout = { xaxis: { title: 'Where the measured value falls in its predictive distribution (PIT)', range: [0, 1] }, yaxis: { title: 'Share of the values', tickformat: '.0%' }, margin: { t: 10 }, bargap: 0.05, showlegend: false };
    const coverClass = (r, l) => { const c = r.coverage[String(l)]; return Math.abs(c - l) > 0.1 ? 'bad' : Math.abs(c - l) > 0.05 ? 'warn' : 'ok'; };
    const verdict = computed(() => {
      if (!block.value || !rows.value.length) return null;
      const best = rows.value[0], simple = rows.value.find((r) => r.key === block.value.one_se) || best, none = rows.value.find((r) => r.key === 'nothing');
      const w = none && none.mean_width80 ? 1 - simple.mean_width80 / none.mean_width80 : null;
      let txt = `Held out, the ${modeLabel(modeOf(fp.mb.res, target.value)).toLowerCase()} is best predicted from ${sourceWords(best.sources, best.covariates)}.`;
      if (simple.key !== best.key) txt += ` The simplest choice within one standard error of the best is ${sourceWords(simple.sources, simple.covariates)}.`;
      if (w !== null && Number.isFinite(w)) txt += ` It makes the 80 % intervals ${pct(Math.abs(w), 0)} ${w >= 0 ? 'narrower' : 'wider'} than knowing nothing about the sample.`;
      const c95 = simple.coverage['0.95'];
      txt += ` Its 95 % intervals contain ${pct(c95, 0)} of the measured values${Math.abs(c95 - 0.95) <= 0.05 ? ': the uncertainty is honest.' : c95 < 0.95 ? ': the intervals are too narrow, the model is over-confident here.' : ': the intervals are wider than they need to be.'}`;
      return txt;
    });
    return { p, C, targets, target, block, rows, running, run, sel, calTraces, calLayout, pitTraces, pitLayout, coverClass, verdict, mbJobs, sourceWords, pct, num };
  },
  template: `
  <div class="col gap-12" data-mblf-compare>
    <p class="fp-note">Every choice of sources is tried on every training sample that has the target, each time as if the sample had been left out of the fit (Pareto-smoothed importance sampling: no refitting). The log predictive density scores the whole predictive distribution (higher is better; the gain over "nothing" is paired, with its standard error); the coverage says whether the intervals contain as many measured values as they claim. Without a full posterior (maximum likelihood) these numbers are in-sample and optimistic.</p>
    <div class="row wrap"><span class="small">Target</span><Seg :model-value="target" :options="targets" small @update:modelValue="(v) => { C.target = v; C.row = null; }" data-mblf-cmp-target/>
      <button type="button" class="btn primary sm" :disabled="running" @click="run" data-mblf-compare-run><Icon name="play" size="15"/>{{ block ? 'Compare again' : 'Compare every choice of sources' }}</button>
      <button type="button" class="btn danger sm" v-if="running" @click="mbJobs.compare.cancel()">Cancel</button></div>
    <JobBar :job="mbJobs.compare" label="Comparing the sources (held out)"/>
    <template v-if="block">
      <MblfVerdict level="info" v-if="verdict" data-mblf-cmp-verdict>{{ verdict }}</MblfVerdict>
      <div class="table-wrap"><table class="table mblf-cmp-table" data-mblf-cmp-table>
        <thead><tr><th>From</th><th>Log density per sample</th><th>Gain over nothing (± SE)</th><th>Inside 50 %</th><th>Inside 80 %</th><th>Inside 95 %</th><th>Error (RMS)</th><th>Width of the 80 % interval</th></tr></thead>
        <tbody><tr v-for="r in rows" :key="r.key" :class="{ on: sel && r.key === sel.key }" style="cursor:pointer" @click="C.row = r.key" :data-key="r.key">
          <td>{{ sourceWords(r.sources, r.covariates) }} <span class="chip xs" v-if="r.key === block.best">best</span> <span class="chip xs" v-if="r.key === block.one_se && r.key !== block.best">simplest within 1 SE</span></td>
          <td>{{ num(r.elpd / r.n, 3) }}</td><td>{{ num(r.gain, 3) }} ± {{ num(r.gain_se, 2) }}</td>
          <td :class="coverClass(r, 0.5)">{{ pct(r.coverage['0.5'], 0) }}</td><td :class="coverClass(r, 0.8)">{{ pct(r.coverage['0.8'], 0) }}</td><td :class="coverClass(r, 0.95)">{{ pct(r.coverage['0.95'], 0) }}</td>
          <td>{{ num(r.rmse, 3) }}</td><td>{{ num(r.mean_width80, 3) }}</td></tr></tbody></table></div>
      <div class="fp-two">
        <ChartCard title="Calibration of the intervals (held out)" help="For each choice of sources, the share of measured values inside the 50, 80 and 95 % intervals. On the dotted line the intervals are honest; below it they are too narrow." :data="calTraces" flat><PlotlyChart :data="calTraces" :layout="calLayout" :height="300"/></ChartCard>
        <ChartCard v-if="sel" :title="'PIT of the values predicted from ' + sourceWords(sel.sources, sel.covariates)" help="Where every measured value falls in its own predictive distribution. Flat = calibrated; U-shaped = intervals too narrow; a hump in the middle = too wide; tilted = biased." :data="pitTraces" flat><PlotlyChart :data="pitTraces" :layout="pitLayout" :height="300"/></ChartCard>
      </div>
      <p class="muted xs">Computed {{ block.computed }} from {{ block.n_draws }} posterior draws.</p>
    </template>
  </div>`,
};

export const MblfPredict = {
  name: 'MblfPredict',
  components: { Icon, Seg, FpcaFrame, FpcaExplain, MblfGate, MblfStaleNote, MblfOneSample, MblfCompare },
  setup() {
    const A = computed(() => fp.mb);
    const ready = computed(() => !!(A.value.res && A.value.resRid));
    const view = computed({ get: () => A.value.view.predTab || 'one', set: (v) => { A.value.view.predTab = v; } });
    const TABS = [{ value: 'one', label: 'One sample' }, { value: 'compare', label: 'Every choice of sources, held out' }];
    return { A, ready, view, TABS,
      DOES: 'Predicts one measurement of a sample from the others, the way the method prescribes: at every posterior draw the prediction is Gaussian given the sample’s sources; the draws are averaged. Its uncertainty is split into four parts you can act on: the population parameters (more training samples), the factor scores of the sample (more or better measurements of it), the sample’s own residual (nothing: no other measurement can tell it), and the measurement noise.',
      CHECK: 'That the held-out intervals contain about as many measured values as they claim (the comparison: 50, 80 and 95 %), before you trust a prediction for a sample that lacks a measurement. A choice of sources that is not clearly better than a simpler one is not worth its cost.' };
  },
  template: `
  <FpcaFrame :n="7" next-label="Checks and use" :no-next="!ready" :next-disabled="!ready">
    <FpcaExplain :does="DOES" :check="CHECK"/>
    <MblfGate/>
    <template v-if="ready">
      <MblfStaleNote/>
      <Seg v-model="view" :options="TABS" data-mblf-pred-tabs/>
      <MblfOneSample v-if="view === 'one'"/>
      <MblfCompare v-else/>
    </template>
  </FpcaFrame>`,
};
