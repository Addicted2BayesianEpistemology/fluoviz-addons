// Step 6 of the Multimodal latent factors page: the population.  Tabs: Overview (did it converge, how many factors, how well each mode is described), Factors (the loading
// functions with their bands), What tells what (how much of each mode the others predict, for every choice of sources, point by point; the correlation of any point with a mode),
// How the variation splits, Covariates (their effects with bands), Samples (the scores with their uncertainty).  Also `MblfGate` and `MblfStaleNote` for steps 6 to 8.
import { computed, watch } from 'vue';
import { Icon } from '/static/js/icons.js';
import { Callout, Tabs, Seg, Switch, Field } from '/static/js/components/Common.js';
import { EEMView } from '/static/js/components/EEMView.js';
import { PlotlyChart, ChartCard } from '/static/js/plot.js';
import { FpcaFrame } from '/static/js/components/FpcaStepper.js';
import { FpcaExplain } from '/static/js/components/FpcaCommon.js';
import { tokens, themeState, withAlpha } from '/static/js/theme.js';
import { fp, ctx, mbJobs, MODE_NICE } from './MblfState.js';
import { predictable, correlation } from './MblfActions.js';
import { MblfStat, MblfVerdict, MblfSection, MblfModeView, MblfInterval, nd, row, img, arr, modeOf, modeLabel, sourceWords, pct, nTxt, num, ivlPct } from './MblfCommon.js';

export const STALE_WORDS = { raw: 'the raw data', corrections: 'the corrections', masks: 'the scatter removal or excised regions', window: 'the focus window', samples: 'the sample selection', meta: 'the metadata' };

export const MblfStaleNote = {
  name: 'MblfStaleNote',
  components: { Icon },
  setup() {
    const stale = computed(() => (fp.mb.res ? fp.mb.res.result.stale || [] : []));
    return { stale, what: computed(() => stale.value.map((s) => STALE_WORDS[s] || s).join(', ')) };
  },
  template: `<div class="callout warning" v-if="stale.length" data-mblf-stale><Icon name="exclamation-triangle"/><div class="grow"><b>The data changed since this model was fitted</b> ({{ what }}). Everything stored with the model can still be looked at, but a saved model cannot be made from it. Use <b>Fit again with the same settings</b> in the bar above.</div></div>`,
};

export const MblfGate = {
  name: 'MblfGate',
  components: { Icon, Callout },
  setup() { return { A: computed(() => fp.mb), fitting: computed(() => mbJobs.fit.state.running), goStep: ctx.goStep }; },
  template: `
  <div data-mblf-gate>
    <div class="fp-empty" v-if="!A.resBusy && !A.resError && !(A.res && A.resRid)" data-mblf-noresult>
      <p v-if="fitting"><b>The model is being fitted.</b> The results appear here when it is done.</p>
      <p v-else>No model has been fitted yet. Choose the settings and press <b>Fit the model</b> in step 5.</p>
      <div class="row" style="justify-content:center"><button type="button" class="btn" @click="goStep(5)"><Icon name="arrow-left" size="15"/>Go to step 5 · Fit</button></div>
    </div>
    <div class="fp-wait muted" v-else-if="A.resBusy && !A.res">Opening the model …</div>
    <Callout kind="critical" v-else-if="A.resError" data-mblf-result-error><b>This model could not be opened.</b> {{ A.resError }}</Callout>
  </div>`,
};

// ---------------------------------------------------------------- overview
const MblfOverview = {
  name: 'MblfOverview',
  components: { MblfStat, MblfVerdict, MblfInterval, PlotlyChart, ChartCard, Callout },
  setup() {
    const p = computed(() => fp.mb.res.p);
    const verdict = computed(() => p.value.diagnostics.verdict);
    const dim = computed(() => p.value.dimension);
    const dimTraces = computed(() => {
      themeState.version;
      if (!dim.value) return [];
      const t = tokens();
      return dim.value.probabilities.map((r, i) => {
        const ks = Object.keys(r.p).map(Number).sort((a, b) => a - b);
        return { x: ks.map(String), y: ks.map((k) => r.p[k]), type: 'bar', name: `${Math.round(100 * r.energy)} % of the shared variation`, marker: { color: t.series[i] } };
      });
    });
    const dimLayout = { barmode: 'group', xaxis: { title: 'Number of directions', type: 'category' }, yaxis: { title: 'Posterior probability', range: [0, 1] }, margin: { t: 10 }, legend: { orientation: 'h', y: -0.3 } };
    const atCap = computed(() => (dim.value ? dim.value.table.find((r) => r.energy >= 0.95 && r.p_at_cap > 0.2) : null));
    const shares = computed(() => (dim.value ? dim.value.shared_fraction : []));
    const modes = computed(() => p.value.modes);
    return { p, verdict, dim, dimTraces, dimLayout, atCap, shares, modes, pct, nTxt, num, modeLabel };
  },
  template: `
  <div class="col gap-12" data-mblf-overview>
    <MblfVerdict :level="verdict.level === 'ok' ? 'good' : verdict.level === 'warn' ? 'warning' : 'critical'" title="Convergence.">{{ verdict.text }} <span v-if="verdict.items.length">{{ verdict.items[0] }}</span></MblfVerdict>
    <div class="fp-stats">
      <MblfStat label="Samples" :value="nTxt(p.data.n)" :sub="nTxt(p.data.complete) + ' with every mode'"/>
      <MblfStat label="Posterior draws" :value="p.estimation === 'bayes' ? nTxt(p.n_draws) : 'one point'" :sub="p.estimation === 'bayes' ? p.chains + ' chains' : 'maximum likelihood'"/>
      <MblfStat label="Held-out fit (elpd)" :value="num(p.loo.elpd, 4)" :sub="'± ' + num(p.loo.se, 2) + (p.loo.loo ? ' (leave-one-out)' : ' (in-sample)')" help="The expected log predictive density of every sample when it is left out (Pareto-smoothed importance sampling). Higher is better; compare models of the SAME data only."/>
      <MblfStat label="Factors used (95 % of the shared variation)" :value="dim ? dim.table.find((r) => r.energy === 0.95).mode : '–'" :sub="dim ? 'of at most ' + p.K + ' (most probable)' : ''" help="The number of directions the posterior needs for 95 % of the shared variation (its mode over the draws). The prior shrinks the others."/>
    </div>
    <Callout kind="warning" v-if="atCap" data-mblf-at-cap>The posterior often needs ALL {{ p.K }} factors (probability {{ pct(atCap.p_at_cap, 0) }}): the cap is probably too low. Fit again with a larger number of factors.</Callout>
    <div class="fp-two">
      <ChartCard v-if="dim" title="How many factors the data use" help="For every posterior draw, the number of directions needed to reach 90, 95 and 99 % of the shared variation: its posterior distribution. A spread over several numbers is honest uncertainty about the number of factors." :data="dimTraces" flat>
        <PlotlyChart :data="dimTraces" :layout="dimLayout" :height="260"/>
      </ChartCard>
      <div class="card flat"><div class="card-head"><h3>How much of each mode the factors carry</h3></div>
        <div class="card-body col gap-8">
          <p class="fp-note">The share of the variation of each mode that the shared factors carry (the rest is each sample's own part and the noise), with its 90 % credible interval.</p>
          <div v-for="(s, i) in shares" :key="i" class="row mblf-share-row"><span class="mblf-share-label">{{ modeLabel(modes[s.mode - 1]) }}</span><MblfInterval :value="{ mean: s.shared_fraction_mean, lower: s.lower, upper: s.upper }"/></div>
        </div></div>
    </div>
    <Callout kind="info" v-for="n in p.notes" :key="n">{{ n }}</Callout>
  </div>`,
};

// ---------------------------------------------------------------- factors
const MblfFactors = {
  name: 'MblfFactors',
  components: { Seg, Switch, MblfModeView, MblfInterval, Callout },
  setup() {
    const p = computed(() => fp.mb.res.p);
    const V = computed(() => fp.mb.view);
    const options = computed(() => p.value.factors.map((f, i) => ({ value: i, label: `Factor ${f.factor} (${Math.round(100 * f.share.mean)} %)`, title: `Carries ${ivlPct(f.share)} of the shared variation` })));
    const fac = computed(() => p.value.factors[Math.min(V.value.factor, p.value.factors.length - 1)]);
    const weak = computed(() => fac.value.share.upper < 0.02);
    return { p, V, options, fac, weak, modeOf: (n) => modeOf(fp.mb.res, n), modeLabel, ivlPct };
  },
  template: `
  <div class="col gap-12" data-mblf-factors>
    <p class="fp-note">A factor is a pattern shared by all the modes: one loading function per mode, the change of that mode for one standard deviation of the factor's score. The bands are posterior credible bands after aligning the factors across the draws (varimax, then matched in sign and order): dark = 50 %, lighter = 80 % and 95 % at every point; the dashed lines hold the WHOLE curve with 95 % probability. For an EEM choose what the picture shows: the probability of a positive value, or the mean only where the sign is sure, tell where a pattern is credibly present.</p>
    <div class="row wrap"><Seg v-model="V.factor" :options="options" small data-mblf-factor/><span class="grow"></span><Switch v-model="V.simultaneous" label="Band for the whole curve" data-mblf-simultaneous/></div>
    <Callout kind="info" v-if="weak">This factor carries less than 2 % of the shared variation even at the top of its interval: the prior has switched it off. It is shown for completeness.</Callout>
    <p class="small">Factor {{ fac.factor }} carries <b>{{ ivlPct(fac.share) }}</b> of the shared variation (posterior mean and 90 % interval).</p>
    <div class="mblf-grid">
      <MblfModeView v-for="m in fac.modes" :key="m.mode" :mode="modeOf(m.mode)" :summary="m" :title="modeLabel(modeOf(m.mode))" signed :simultaneous="V.simultaneous" :units="modeOf(m.mode).units" :export-name="'factor' + fac.factor + '-' + m.mode"/>
    </div>
  </div>`,
};

// ---------------------------------------------------------------- what tells what
const MblfTells = {
  name: 'MblfTells',
  components: { Seg, Field, Switch, MblfModeView, MblfInterval, PlotlyChart, ChartCard, EEMView, Callout, Icon },
  setup() {
    const p = computed(() => fp.mb.res.p);
    const V = computed(() => fp.mb.view);
    const targets = computed(() => p.value.modes.map((m) => ({ value: m.name, label: MODE_NICE[m.name] || m.name })));
    watch(() => fp.mb.resRid, () => { V.value.target = null; V.value.combo = null; }, { immediate: false });
    const target = computed(() => V.value.target || p.value.modes[0].name);
    const rows = computed(() => p.value.matrix.filter((r) => r.target === target.value).slice().sort((a, b) => b.latent.mean - a.latent.mean));
    const combo = computed(() => rows.value.find((r) => r.key === V.value.combo) || rows.value.find((r) => r.key !== 'nothing') || rows.value[0]);
    const detail = computed(() => (combo.value && combo.value.key !== 'nothing' ? predictable(target.value, combo.value.sources.length ? combo.value.sources.join(',') : 'none', combo.value.covariates) : null));
    const mode = computed(() => modeOf(fp.mb.res, target.value));
    const histTraces = computed(() => {
      themeState.version;
      const h = combo.value && combo.value.hist;
      if (!h) return [];
      const x = h.map((_, i) => (i + 0.5) / h.length);
      return [{ x, y: h, type: 'bar', marker: { color: tokens().series[0] }, name: 'Posterior draws', hovertemplate: 'R² %{x:.2f}: %{y} draws<extra></extra>' }];
    });
    const histLayout = { xaxis: { title: 'Share of the variation between samples predicted', range: [0, 1], tickformat: '.0%' }, yaxis: { title: 'Draws' }, margin: { t: 10 }, bargap: 0.02, showlegend: false };
    const pick = (k) => { V.value.combo = k; };
    // the correlation explorer
    const corrA = computed(() => V.value.corrA || p.value.modes[0].name);
    const corrB = computed(() => V.value.corrB || (p.value.modes[1] || p.value.modes[0]).name);
    const modeA = computed(() => modeOf(fp.mb.res, corrA.value));
    const corr = computed(() => {
      const a = modeA.value;
      if (!a) return null;
      if (a.kind === 'surface' && (V.value.corrX === null || V.value.corrY === null)) return null;
      if (a.kind === 'curve' && V.value.corrX === null) return null;
      return correlation(corrA.value, corrB.value, a.kind === 'vector' ? null : V.value.corrX, a.kind === 'surface' ? V.value.corrY : null, a.kind === 'vector' ? V.value.corrJ : null);
    });
    const corrSummary = computed(() => (corr.value && corr.value.data ? { mean: corr.value.data.mean, quantiles: [corr.value.data.lower, corr.value.data.lower, corr.value.data.lower, corr.value.data.mean, corr.value.data.upper, corr.value.data.upper, corr.value.data.upper], p_pos: corr.value.data.p_pos, lfsr: corr.value.data.lfsr } : null));
    const meanA = computed(() => (modeA.value && modeA.value.kind === 'surface' ? img(nd(modeA.value.mean)) : null));
    const maskA = computed(() => (modeA.value && modeA.value.mask ? img(nd(modeA.value.mask)) : null));
    const onPick = (e) => { if (e && e.ex !== undefined) { V.value.corrX = e.ex; V.value.corrY = e.em; } };
    const modeOptions = computed(() => p.value.modes.map((m) => ({ value: m.name, label: MODE_NICE[m.name] || m.name })));
    const setA = (v) => { V.value.corrA = v; V.value.corrX = null; V.value.corrY = null; };
    return { p, V, targets, target, rows, combo, detail, mode, histTraces, histLayout, pick, sourceWords, pct, ivlPct, corrA, corrB, modeA, corr, corrSummary, meanA, maskA, onPick, modeOptions, setA, modeOf: (n) => modeOf(fp.mb.res, n), modeLabel };
  },
  template: `
  <div class="col gap-12" data-mblf-tells>
    <p class="fp-note">How much of the variation of a mode between samples the OTHER measurements (and the covariates) can predict: the predictable fraction R², a property of the population computed at every posterior draw — the bars are its posterior mean and 90 % interval. "Nothing" is 0 by definition; the covariates alone predict what they shift. This answers "is it worth measuring X to know Y?" before any sample is predicted; step 7 checks it held out.</p>
    <div class="row wrap"><span class="small">Predict</span><Seg :model-value="target" :options="targets" small @update:modelValue="(v) => { V.target = v; V.combo = null; }" data-mblf-target/></div>
    <table class="table mblf-tells-table" data-mblf-matrix>
      <thead><tr><th>From</th><th>Predicted share of the latent variation (90 % interval)</th><th>Of a new measurement</th></tr></thead>
      <tbody><tr v-for="r in rows" :key="r.key" :class="{ on: combo && r.key === combo.key }" @click="pick(r.key)" :data-key="r.key" style="cursor:pointer">
        <td>{{ sourceWords(r.sources, r.covariates) }}</td><td><MblfInterval :value="r.latent"/></td><td class="small">{{ pct(r.measured.mean, 0) }}</td></tr></tbody>
    </table>
    <template v-if="combo && combo.key !== 'nothing'">
      <h3 class="fp-h3">{{ modeLabel(mode) }} from {{ sourceWords(combo.sources, combo.covariates) }}: {{ ivlPct(combo.latent) }}</h3>
      <div class="fp-two">
        <ChartCard title="Posterior of the predicted share" help="Every posterior draw gives one value of R² for the whole mode: their histogram is the uncertainty about how predictable this mode is." :data="histTraces" flat><PlotlyChart :data="histTraces" :layout="histLayout" :height="230"/></ChartCard>
        <div>
          <p class="muted small" v-if="detail && detail.busy">Computing the predictable fraction point by point …</p>
          <Callout kind="critical" v-if="detail && detail.error">{{ detail.error }}</Callout>
          <MblfModeView v-if="detail && detail.data" :mode="mode" :summary="{ mean: detail.data.latent.mean, quantiles: [detail.data.latent.lower, detail.data.latent.lower, detail.data.latent.lower, detail.data.latent.mean, detail.data.latent.upper, detail.data.latent.upper, detail.data.latent.upper] }"
                        :stats="['mean', 'lower', 'upper', 'width']" title="Predicted share, point by point (posterior mean; the limits are the 90 % interval)" units="share" :simultaneous="false" compact/>
        </div>
      </div>
    </template>
    <h3 class="fp-h3">Correlation of one point with a whole mode</h3>
    <p class="fp-note">Choose a point (click the EEM, or type a wavelength): the picture shows how strongly every point of the other mode varies with it across the population, with its uncertainty. A credible link has a probability of a positive (or negative) correlation near 1 (or 0).</p>
    <div class="row wrap">
      <Field label="Point of"><select class="select sm" :value="corrA" @change="setA($event.target.value)" data-mblf-corr-a><option v-for="o in modeOptions" :key="o.value" :value="o.value">{{ o.label }}</option></select></Field>
      <Field label="Against"><select class="select sm" :value="corrB" @change="V.corrB = $event.target.value" data-mblf-corr-b><option v-for="o in modeOptions" :key="o.value" :value="o.value">{{ o.label }}</option></select></Field>
      <Field v-if="modeA && modeA.kind === 'curve'" label="Wavelength (nm)"><input class="input sm" type="number" v-model.number="V.corrX" data-mblf-corr-x/></Field>
      <Field v-if="modeA && modeA.kind === 'vector'" label="Variable"><select class="select sm" v-model.number="V.corrJ"><option v-for="(l, j) in modeA.labels" :key="l" :value="j">{{ l }}</option></select></Field>
      <span v-if="modeA && modeA.kind === 'surface'" class="small muted">{{ V.corrX !== null ? 'Excitation ' + V.corrX.toFixed(0) + ' nm, emission ' + V.corrY.toFixed(0) + ' nm' : 'Click a point of the EEM below.' }}</span>
    </div>
    <div class="fp-two">
      <div v-if="modeA && modeA.kind === 'surface'"><EEMView :ex="modeA.ex" :em="modeA.em" :z="meanA" :mask="maskA" :height="320" marginals="none" :crosshair="V.corrX !== null ? { ex: V.corrX, em: V.corrY } : null" @click="onPick" @update:crosshair="onPick" title="Mean EEM: click a point"/></div>
      <div v-if="corr">
        <p class="muted small" v-if="corr.busy">Computing the correlations …</p>
        <Callout kind="critical" v-if="corr.error">{{ corr.error }}</Callout>
        <MblfModeView v-if="corrSummary" :mode="modeOf(corrB)" :summary="corrSummary" signed :stats="['mean', 'lower', 'upper', 'p_pos', 'sure']" title="Correlation (posterior mean; limits = 90 % interval)" units="correlation" :simultaneous="false"/>
      </div>
    </div>
  </div>`,
};

// ---------------------------------------------------------------- how the variation splits
const MblfVariance = {
  name: 'MblfVariance',
  components: { MblfModeView, MblfInterval, Seg },
  setup() {
    const p = computed(() => fp.mb.res.p);
    const part = computed({ get: () => fp.mb.view.part || 'factors', set: (v) => { fp.mb.view.part = v; } });
    const PARTS_V = [{ value: 'factors', label: 'Shared factors' }, { value: 'residual', label: 'Each sample’s own' }, { value: 'noise', label: 'Measurement noise' }];
    const view = (v) => {
      const s = v.share[part.value];
      return { mean: s.mean, quantiles: [s.lower || s.mean, s.lower || s.mean, s.lower || s.mean, s.mean, s.upper || s.mean, s.upper || s.mean, s.upper || s.mean] };
    };
    return { p, part, PARTS_V, view, modeOf: (n) => modeOf(fp.mb.res, n), modeLabel };
  },
  template: `
  <div class="col gap-12" data-mblf-variance>
    <p class="fp-note">At every point, the variation between samples splits into what the shared factors carry (the part the modes have in common, covariates included), what is each sample's own (smooth, but specific to the sample and the mode), and measurement noise. The shares below are posterior means with 90 % intervals.</p>
    <table class="table"><thead><tr><th>Mode</th><th>Shared factors</th><th>Each sample's own</th><th>Noise</th></tr></thead>
      <tbody><tr v-for="v in p.variance" :key="v.mode"><td>{{ modeLabel(modeOf(v.mode)) }}</td><td><MblfInterval :value="v.whole.factors"/></td><td><MblfInterval :value="v.whole.residual"/></td><td><MblfInterval :value="v.whole.noise"/></td></tr></tbody></table>
    <div class="row wrap"><span class="small">Show</span><Seg v-model="part" :options="PARTS_V" small data-mblf-part/></div>
    <div class="mblf-grid"><MblfModeView v-for="v in p.variance" :key="v.mode" :mode="modeOf(v.mode)" :summary="view(v)" :title="modeLabel(modeOf(v.mode)) + ': share, point by point'" :stats="['mean', 'lower', 'upper', 'width']" units="share" :simultaneous="false"/></div>
  </div>`,
};

// ---------------------------------------------------------------- covariates
const MblfCovariates = {
  name: 'MblfCovariates',
  components: { Seg, Switch, MblfModeView, MblfInterval, Callout },
  setup() {
    const p = computed(() => fp.mb.res.p);
    const C = computed(() => p.value.covariates);
    const which = computed({ get: () => Math.min(fp.mb.view.cov || 0, (C.value ? C.value.effects.length : 1) - 1), set: (v) => { fp.mb.view.cov = v; } });
    const options = computed(() => (C.value ? C.value.effects.map((e, i) => ({ value: i, label: e.column.label })) : []));
    const eff = computed(() => (C.value ? C.value.effects[which.value] : null));
    const words = computed(() => {
      const c = eff.value && eff.value.column;
      if (!c) return '';
      return c.kind === 'numeric' ? `The change of every mode when ${c.label} is one standard deviation (${c.sd.toPrecision(3)}) higher, everything else equal.`
        : `The difference of every mode between ${c.label.split(' = ')[0]} = ${c.level} and the reference ${c.reference}, everything else equal.`;
    });
    return { p, C, which, options, eff, words, modeOf: (n) => modeOf(fp.mb.res, n), modeLabel };
  },
  template: `
  <div class="col gap-12" data-mblf-covariates>
    <Callout kind="info" v-if="!C">This model has no covariates. Add metadata as covariates in step 1 to see how they shift the factors, wavelength by wavelength.</Callout>
    <template v-else>
      <p class="fp-note">The covariates shift the factor scores; their effect on a mode is a function, the same for every sample, which does not depend on how the factors are oriented. The bands are credible bands; for an EEM, "P(value > 0)" and "Mean where the sign is sure" show where the effect is credibly positive or negative.</p>
      <table class="table"><thead><tr><th>Mode</th><th>Share of the variation the covariates explain (90 % interval)</th></tr></thead>
        <tbody><tr v-for="s in C.shares" :key="s.mode"><td>{{ modeLabel(modeOf(s.mode)) }}</td><td><MblfInterval :value="s"/></td></tr></tbody></table>
      <div class="row wrap"><Seg v-model="which" :options="options" small data-mblf-cov/></div>
      <p class="small">{{ words }}</p>
      <div class="mblf-grid"><MblfModeView v-for="m in eff.modes" :key="m.mode" :mode="modeOf(m.mode)" :summary="m" :title="modeLabel(modeOf(m.mode))" signed :units="modeOf(m.mode).units" :export-name="'effect-' + m.mode"/></div>
    </template>
  </div>`,
};

// ---------------------------------------------------------------- samples
const MblfSamples = {
  name: 'MblfSamples',
  components: { Seg, PlotlyChart, ChartCard, Field },
  setup() {
    const p = computed(() => fp.mb.res.p);
    const K = computed(() => p.value.K);
    const fx = computed({ get: () => fp.mb.view.fx ?? 0, set: (v) => { fp.mb.view.fx = v; } });
    const fy = computed({ get: () => fp.mb.view.fy ?? Math.min(1, K.value - 1), set: (v) => { fp.mb.view.fy = v; } });
    const opts = computed(() => Array.from({ length: K.value }, (_, k) => ({ value: k, label: `Factor ${k + 1}` })));
    const traces = computed(() => {
      themeState.version;
      const t = tokens(), s = p.value.samples, S = nd(s.scores), SD = nd(s.scores_sd), P = nd(s.present), D = p.value.modes.length;
      const n = s.ids.length, k = S.shape[1];
      const g = (M, i, j) => M.data[i * k + j];
      const groups = {};
      for (let i = 0; i < n; i++) {
        const have = p.value.modes.filter((_, d) => P.data[i * D + d]).map((m) => MODE_NICE[m.name]).join(' + ') || 'nothing';
        (groups[have] = groups[have] || []).push(i);
      }
      return Object.entries(groups).map(([name, idx], c) => ({
        x: idx.map((i) => g(S, i, fx.value)), y: idx.map((i) => g(S, i, fy.value)), text: idx.map((i) => s.ids[i]), type: 'scatter', mode: 'markers', name: `Has ${name}`,
        marker: { size: 9, color: t.series[c % 8], line: { width: 1.5, color: t.surface } },
        error_x: { type: 'data', array: idx.map((i) => 2 * g(SD, i, fx.value)), color: withAlpha(t.series[c % 8], 0.45), thickness: 1, width: 0 },
        error_y: { type: 'data', array: idx.map((i) => 2 * g(SD, i, fy.value)), color: withAlpha(t.series[c % 8], 0.45), thickness: 1, width: 0 },
        hovertemplate: '%{text}<extra></extra>' }));
    });
    const layout = computed(() => ({ xaxis: { title: `Factor ${fx.value + 1} score` }, yaxis: { title: `Factor ${fy.value + 1} score` }, margin: { t: 10 }, legend: { orientation: 'h', y: -0.2 } }));
    const table = computed(() => {
      const s = p.value.samples, L = nd(p.value.loo.per_value), Kp = nd(p.value.loo.pareto_k);
      const rows = s.ids.map((id, i) => [id, Number.isFinite(L.data[i]) ? Number(L.data[i].toFixed(3)) : null, Number.isFinite(Kp.data[i]) ? Number(Kp.data[i].toFixed(2)) : null]);
      rows.sort((a, b) => (a[1] ?? 0) - (b[1] ?? 0));
      return { columns: ['Sample', 'Held-out log density per value', 'Pareto k'], rows };
    });
    return { p, fx, fy, opts, traces, layout, table, K };
  },
  template: `
  <div class="col gap-12" data-mblf-samples>
    <p class="fp-note">Every sample's factor scores, from everything it has (posterior means; the bars are ± 2 posterior standard deviations). A sample that lacks a mode has wider bars: the measurement it lacks would have told more about its factors. The scores are also variables of FluoViz (step 8).</p>
    <div class="row wrap" v-if="K > 1"><Field label="Across"><select class="select sm" v-model.number="fx"><option v-for="o in opts" :key="o.value" :value="o.value">{{ o.label }}</option></select></Field>
      <Field label="Up"><select class="select sm" v-model.number="fy"><option v-for="o in opts" :key="o.value" :value="o.value">{{ o.label }}</option></select></Field></div>
    <ChartCard title="Scores of the samples, with their uncertainty" :data="traces" flat><PlotlyChart :data="traces" :layout="layout" :height="420"/></ChartCard>
    <ChartCard title="The samples the model finds most surprising first" help="The leave-one-out log predictive density of each sample, per measured value (low = surprising), and its Pareto k (above 0.7: the sample is influential, and its held-out number is rough)." :table="table" flat>
      <div class="table-wrap" style="max-height:320px"><table class="table"><thead><tr><th v-for="c in table.columns" :key="c">{{ c }}</th></tr></thead><tbody><tr v-for="r in table.rows" :key="r[0]"><td>{{ r[0] }}</td><td>{{ r[1] ?? '–' }}</td><td :class="{ bad: r[2] > 0.7 }">{{ r[2] ?? '–' }}</td></tr></tbody></table></div>
    </ChartCard>
  </div>`,
};

export const MblfResults = {
  name: 'MblfResults',
  components: { Icon, Tabs, FpcaFrame, FpcaExplain, MblfGate, MblfStaleNote, MblfOverview, MblfFactors, MblfTells, MblfVariance, MblfCovariates, MblfSamples },
  setup() {
    const A = computed(() => fp.mb);
    const ready = computed(() => !!(A.value.res && A.value.resRid));
    const tabs = computed(() => [{ id: 'overview', label: 'Overview' }, { id: 'factors', label: 'Factors' }, { id: 'tells', label: 'What tells what' }, { id: 'variance', label: 'How the variation splits' },
      { id: 'covariates', label: 'Covariates', dot: !!(A.value.res && A.value.res.p.covariates) }, { id: 'samples', label: 'Samples' }]);
    return { A, ready, tabs,
      DOES: 'Shows what the model learned about the population, every number with its posterior uncertainty: whether the sampler converged and how many factors the data use; the factors (one loading function per mode) with credible bands; how much of each mode the others can predict; how the variation of every wavelength or pixel splits; the effects of the covariates; the scores of the samples.',
      CHECK: 'Read the convergence first (Overview): if it is not fine, the rest is not yet trustworthy. Then whether the factors are credibly different from zero where you read them (the bands, or P(value > 0) on an EEM), and whether "What tells what" supports the use you have in mind; step 7 then checks the predictions on held-out samples.' };
  },
  template: `
  <FpcaFrame :n="6" next-label="Prediction and uncertainty" :no-next="!ready" :next-disabled="!ready">
    <FpcaExplain :does="DOES" :check="CHECK"/>
    <MblfGate/>
    <template v-if="ready">
      <MblfStaleNote/>
      <Tabs v-model="A.tab6" :tabs="tabs" data-mblf-tabs6/>
      <MblfOverview v-if="A.tab6 === 'overview'"/>
      <MblfFactors v-else-if="A.tab6 === 'factors'"/>
      <MblfTells v-else-if="A.tab6 === 'tells'"/>
      <MblfVariance v-else-if="A.tab6 === 'variance'"/>
      <MblfCovariates v-else-if="A.tab6 === 'covariates'"/>
      <MblfSamples v-else-if="A.tab6 === 'samples'"/>
    </template>
  </FpcaFrame>`,
};
