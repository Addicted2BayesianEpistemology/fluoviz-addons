// Shared pieces of the Multimodal latent factors page: arrays of the server as the charts want them, the number card / verdict / titled block (the `common` add-on), and the three
// ways uncertainty is drawn everywhere on the page:
//   * a function of a mode with its posterior or predictive distribution (`MblfModeView`): a spectrum as a FAN (the median, nested 50 / 80 / 95 % bands, optionally the simultaneous
//     band that holds the whole curve), an EEM as a picture of the statistic chosen (mean, a limit, the width of the interval, the probability of a positive value, or the mean
//     only where its sign is certain), values as points with their 50 % and 95 % intervals;
//   * the uncertainty budget of a prediction (`MblfBudget`): what share of the predictive variance comes from the population parameters, the factor scores of the sample, the
//     sample's own variation and the measurement noise, point by point and over the whole mode;
//   * one number with its interval (`MblfInterval`).
import { computed, ref } from 'vue';
import { Icon } from '/static/js/icons.js';
import { Seg, Switch, HelpTip } from '/static/js/components/Common.js';
import { EEMView } from '/static/js/components/EEMView.js';
import { PlotlyChart, ChartCard } from '/static/js/plot.js';
import { tokens, themeState, withAlpha } from '/static/js/theme.js';
import { fmt } from '/static/js/util.js';
import { makeParts } from '/addon-web/common/web/parts.js';
import { pct, nTxt, plural, listWords } from '/addon-web/common/web/words.js';
import { PARTS, PART_LABELS, PART_WORDS, fp } from './MblfState.js';

export { fmt, Icon, pct, nTxt, plural, listWords };

const parts = makeParts({ name: 'Mblf', prefix: 'mblf' });
export const MblfStat = parts.Stat;
export const MblfVerdict = parts.Verdict;
export const MblfSection = parts.Section;
export const num = (x, d = 3) => (x === null || x === undefined || Number.isNaN(x) ? '–' : fmt(x, d));
export const ivl = (o, d = 2, scale = 1) => (o ? `${(o.mean * scale).toFixed(d)} (${(o.lower * scale).toFixed(d)} to ${(o.upper * scale).toFixed(d)})` : '–');
export const ivlPct = (o, d = 0) => (o ? `${(100 * o.mean).toFixed(d)} % (${(100 * o.lower).toFixed(d)} to ${(100 * o.upper).toFixed(d)} %)` : '–');

// ---------------------------------------------------------------- arrays (small arrays arrive as nested lists with null, large ones as typed arrays with .shape)
/** Any server array -> {data: Float64Array (NaN for missing), shape}. */
export function nd(a) {
  if (a === null || a === undefined) return null;
  if (ArrayBuffer.isView(a)) return { data: a instanceof Float64Array ? a : Float64Array.from(a), shape: a.shape ? a.shape.slice() : [a.length] };
  if (a.data && a.shape) return a;
  if (Array.isArray(a) && a.length && a.some((x) => x && (ArrayBuffer.isView(x) || (Array.isArray(x) && x.some((y) => ArrayBuffer.isView(y)))))) {
    const parts = a.map(nd);                                                     // a list of arrays (typed or nested): stacked along a new first axis
    const size = parts[0].data.length, data = new Float64Array(size * parts.length);
    parts.forEach((p, i) => data.set(p.data, i * size));
    return { data, shape: [parts.length, ...parts[0].shape] };
  }
  const shape = [];
  let x = a;
  while (Array.isArray(x)) { shape.push(x.length); x = x[0]; }
  const flat = Array.isArray(a) ? a.flat(Infinity) : [a];
  const data = new Float64Array(flat.length);
  for (let i = 0; i < flat.length; i++) { const v = flat[i]; data[i] = v === null || v === undefined ? NaN : Number(v); }
  return { data, shape };
}
/** Row `i` of the first axis of an array (a sub-array with the remaining shape). */
export function row(n, i) {
  if (!n) return null;
  const size = n.shape.slice(1).reduce((s, v) => s * v, 1);
  return { data: n.data.subarray(i * size, (i + 1) * size), shape: n.shape.slice(1) };
}
/** An image for EEMView (a Float32Array with .shape [n_ex, n_em]). */
export function img(n) {
  if (!n) return null;
  const o = Float32Array.from(n.data);
  o.shape = n.shape.slice(-2);
  return o;
}
export const arr = (n) => (n ? Array.from(n.data, (v) => (Number.isFinite(v) ? v : null)) : []);
export function mapArr(n, f) { if (!n) return null; const o = new Float64Array(n.data.length); for (let i = 0; i < o.length; i++) o[i] = f(n.data[i], i); return { data: o, shape: n.shape.slice() }; }
export function zip2(a, b, f) { const o = new Float64Array(a.data.length); for (let i = 0; i < o.length; i++) o[i] = f(a.data[i], b.data[i], i); return { data: o, shape: a.shape.slice() }; }
export const modeOf = (res, name) => (res ? res.p.modes.find((m) => m.name === name) : null);
export const modeLabel = (m) => (m ? (m.name === 'values' ? `Values (${(m.labels || []).join(', ')})` : m.label.charAt(0).toUpperCase() + m.label.slice(1)) : '');
export const sourceWords = (sources, cov) => listWords([...(sources || []).map((s) => ({ fluorescence: 'the EEM', absorbance: 'the absorbance', values: 'the values' }[s] || s)), ...(cov ? ['the covariates'] : [])]) || 'nothing (the population alone)';

// ---------------------------------------------------------------- a fan chart (a spectrum with nested bands)
/** The traces of a fan: q = the 7 quantiles (2.5, 10, 25, 50, 75, 90, 97.5 %), optional simultaneous band and observed values. */
export function fanTraces(x, q, { mean = null, band = null, observed = null, name = 'Posterior', colour = null, showBand = true } = {}) {
  const t = tokens(), c = colour || t.series[0];
  const qs = [0, 1, 2, 3, 4, 5, 6].map((i) => arr(row(q, i)));
  const out = [];
  const area = (lo, hi, a, label) => {
    out.push({ x, y: hi, type: 'scatter', mode: 'lines', line: { width: 0, color: c }, hoverinfo: 'skip', showlegend: false });
    out.push({ x, y: lo, type: 'scatter', mode: 'lines', line: { width: 0, color: c }, fill: 'tonexty', fillcolor: withAlpha(c, a), name: label, hoverinfo: 'skip' });
  };
  area(qs[0], qs[6], 0.16, '95 % interval');
  area(qs[1], qs[5], 0.22, '80 % interval');
  area(qs[2], qs[4], 0.3, '50 % interval');
  out.push({ x, y: mean ? arr(mean) : qs[3], type: 'scatter', mode: 'lines', line: { width: 2, color: c }, name: mean ? `${name} mean` : `${name} median` });
  if (band && showBand) {
    out.push({ x, y: arr(band.lower), type: 'scatter', mode: 'lines', line: { width: 1.2, color: c, dash: 'dash' }, name: `${Math.round(100 * band.level)} % band for the whole curve` });
    out.push({ x, y: arr(band.upper), type: 'scatter', mode: 'lines', line: { width: 1.2, color: c, dash: 'dash' }, showlegend: false, hoverinfo: 'skip' });
  }
  if (observed) out.push({ x, y: arr(observed), type: 'scatter', mode: 'markers', marker: { size: 5, color: t.text, line: { width: 0 } }, name: 'Measured' });
  return out;
}

export const MblfFan = {
  name: 'MblfFan',
  components: { PlotlyChart, ChartCard },
  props: { x: { required: true }, summary: { type: Object, required: true }, observed: { default: null }, title: String, help: String, xTitle: { type: String, default: 'Wavelength (nm)' },
    yTitle: { type: String, default: '' }, zero: Boolean, height: { type: Number, default: 300 }, simultaneous: { type: Boolean, default: true }, useMean: Boolean },
  setup(props) {
    const traces = computed(() => {
      themeState.version;
      const s = props.summary;
      return fanTraces(Array.from(props.x), nd(s.quantiles), { mean: props.useMean ? nd(s.mean) : null, band: s.band ? { level: s.band.level, lower: nd(s.band.lower), upper: nd(s.band.upper) } : null,
        observed: nd(props.observed), showBand: props.simultaneous });
    });
    const layout = computed(() => {
      themeState.version;
      const shapes = props.zero ? [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 0, y1: 0, line: { color: tokens().text3, width: 1, dash: 'dot' } }] : [];
      return { xaxis: { title: props.xTitle }, yaxis: { title: props.yTitle }, shapes, legend: { orientation: 'h', y: -0.25 }, margin: { t: 10, r: 10 } };
    });
    return { traces, layout };
  },
  template: `<ChartCard :title="title" :help="help" :data="traces" flat><PlotlyChart :data="traces" :layout="layout" :height="height"/></ChartCard>`,
};

// ---------------------------------------------------------------- values: points with intervals
// Values can differ in scale by orders of magnitude (a total fluorescence next to a pH): by default every value is drawn in standard deviations of that value (from its mean
// for a value, as it is for a change), or each in its own units on its own panel.  Shares, correlations and probabilities are drawn as they are.
export const VALUE_SCALES = [{ value: 'sd', label: 'In SDs of each value', title: 'Every value in standard deviations of that value over the training samples: values of very different size side by side' },
  { value: 'own', label: 'Own units, one panel each', title: 'Every value in its own units, on its own axis' }];
export const MblfValuesChart = {
  name: 'MblfValuesChart',
  components: { PlotlyChart, ChartCard, Seg },
  props: { labels: { type: Array, required: true }, summary: { type: Object, required: true }, observed: { default: null }, title: String, help: String, yTitle: { type: String, default: '' }, zero: Boolean, height: { type: Number, default: 260 },
    mode: { type: Object, default: null }, quantity: { type: String, default: 'absolute' } },
  setup(props) {
    const how = computed({ get: () => fp.mb.view.valScale || 'sd', set: (v) => { fp.mb.view.valScale = v; } });
    const can = computed(() => !!(props.mode && props.mode.sd && props.quantity !== 'ratio' && props.labels.length > 1));
    const used = computed(() => (can.value ? how.value : 'raw'));
    const sd = computed(() => (props.mode && props.mode.sd ? arr(nd(props.mode.sd)) : []));
    const mu = computed(() => (props.mode && props.mode.mean ? arr(nd(props.mode.mean)) : []));
    const conv = (v, j) => (v === null || used.value !== 'sd' ? v : props.quantity === 'centred' ? v / sd.value[j] : (v - mu.value[j]) / sd.value[j]);
    const traces = computed(() => {
      themeState.version;
      const t = tokens(), q = nd(props.summary.quantiles), L = props.labels;
      const g = (i) => arr(row(q, i));
      const mid = g(3), lo95 = g(0), hi95 = g(6), lo50 = g(2), hi50 = g(4), obs = props.observed ? arr(nd(props.observed)) : null;
      const panels = used.value === 'own' ? L.map((_, j) => [j]) : [L.map((_, j) => j)];
      const out = [];
      panels.forEach((js, pi) => {
        const ax = used.value === 'own' ? { xaxis: `x${pi ? pi + 1 : ''}`, yaxis: `y${pi ? pi + 1 : ''}` } : {};
        const X = js.map((j) => L[j]), c = (a) => js.map((j) => conv(a[j], j)), raw = (a) => js.map((j) => a[j]);
        const m = c(mid), h95 = c(hi95), l95 = c(lo95), h50 = c(hi50), l50 = c(lo50);
        out.push({ x: X, y: m, customdata: raw(mid), type: 'scatter', mode: 'markers', marker: { size: 10, color: t.series[0] }, name: 'Median, 95 % interval', showlegend: pi === 0, ...ax,
          hovertemplate: '%{x}: %{customdata:.4g}<extra></extra>',
          error_y: { type: 'data', symmetric: false, array: h95.map((v, i) => v - m[i]), arrayminus: m.map((v, i) => v - l95[i]), thickness: 1.5, width: 6, color: t.series[0] } });
        out.push({ x: X, y: m, type: 'scatter', mode: 'markers', marker: { size: 1, color: t.series[0] }, name: '50 % interval', hoverinfo: 'skip', showlegend: pi === 0, ...ax,
          error_y: { type: 'data', symmetric: false, array: h50.map((v, i) => v - m[i]), arrayminus: m.map((v, i) => v - l50[i]), thickness: 5, width: 0, color: withAlpha(t.series[0], 0.55) } });
        if (obs) out.push({ x: X, y: c(obs), customdata: raw(obs), type: 'scatter', mode: 'markers', marker: { size: 11, symbol: 'diamond-open', color: t.text, line: { width: 2 } }, name: 'Measured', showlegend: pi === 0, ...ax,
          hovertemplate: '%{x} measured: %{customdata:.4g}<extra></extra>' });
      });
      return out;
    });
    const layout = computed(() => {
      const lay = { margin: { t: 10, r: 10 }, legend: { orientation: 'h', y: -0.3 } };
      if (used.value === 'own') {
        const n = props.labels.length;
        lay.grid = { rows: 1, columns: n, pattern: 'independent', xgap: 0.35 };
        for (let j = 0; j < n; j++) lay[`yaxis${j ? j + 1 : ''}`] = { zeroline: props.zero || props.quantity === 'centred', title: j === 0 ? props.yTitle : '' };
      } else {
        const t = used.value === 'sd' ? (props.quantity === 'centred' ? 'Standard deviations of each value' : 'Standard deviations from the mean of each value') : props.yTitle;
        lay.yaxis = { title: t, zeroline: props.zero || used.value === 'sd' };
      }
      return lay;
    });
    return { traces, layout, how, can, VALUE_SCALES };
  },
  template: `<ChartCard :title="title" :help="help" :data="traces" flat><div class="row wrap" v-if="can" style="margin-bottom:4px"><Seg v-model="how" :options="VALUE_SCALES" small data-mblf-valscale/></div><PlotlyChart :data="traces" :layout="layout" :height="height"/></ChartCard>`,
};

// ---------------------------------------------------------------- one mode, whatever its kind
export const STATS = [
  { value: 'mean', label: 'Mean', title: 'The posterior (or predictive) mean at every point' },
  { value: 'lower', label: '2.5 %', title: 'The lower end of the central 95 % interval at every point' },
  { value: 'upper', label: '97.5 %', title: 'The upper end of the central 95 % interval at every point' },
  { value: 'width', label: 'Width', title: 'How uncertain the value is at every point (upper minus lower limit)' },
  { value: 'p_pos', label: 'P(> 0)', title: 'The posterior probability that the value is positive: near 1 = surely positive, near 0 = surely negative, near 0.5 = the sign is not known' },
  { value: 'sure', label: 'Sure sign', title: 'The mean, only where the probability of the opposite sign is below 5 % (local false sign rate); grey elsewhere' },
];

export const MblfModeView = {
  name: 'MblfModeView',
  components: { EEMView, MblfFan, MblfValuesChart, Seg, HelpTip },
  props: { mode: { type: Object, required: true }, summary: { type: Object, required: true }, observed: { default: null }, title: { type: String, default: '' }, help: { type: String, default: '' },
    signed: Boolean, stats: { type: Array, default: null }, simultaneous: { type: Boolean, default: true }, height: { type: Number, default: 300 }, units: { type: String, default: '' }, exportName: { type: String, default: '' },
    compact: Boolean, quantity: { type: String, default: null } },
  setup(props) {
    const stat = ref('mean');
    const qty = computed(() => props.quantity || (['share', 'correlation', 'probability'].includes(props.units) ? 'ratio' : props.signed ? 'centred' : 'absolute'));
    const options = computed(() => STATS.filter((s) => (props.stats ? props.stats.includes(s.value) : true)).filter((s) => (s.value === 'p_pos' || s.value === 'sure' ? !!props.summary.p_pos : true)));
    const picture = computed(() => {
      const s = props.summary, q = nd(s.quantiles);
      if (stat.value === 'mean') return img(nd(s.mean));
      if (stat.value === 'lower') return img(row(q, 0));
      if (stat.value === 'upper') return img(row(q, 6));
      if (stat.value === 'width') return img(zip2(row(q, 6), row(q, 0), (a, b) => a - b));
      if (stat.value === 'p_pos') return img(nd(s.p_pos));
      const lf = nd(s.lfsr);
      return img(zip2(nd(s.mean), lf, (m, l) => (l < 0.05 ? m : NaN)));
    });
    const kind = computed(() => (stat.value === 'p_pos' ? 'diverging' : stat.value === 'width' ? 'sequential' : (props.signed ? 'diverging' : 'sequential')));
    const zrange = computed(() => (stat.value === 'p_pos' ? [0, 1] : null));
    const mask = computed(() => (props.mode.mask ? img(nd(props.mode.mask)) : null));
    const caption = computed(() => (options.value.find((o) => o.value === stat.value) || {}).title || '');
    return { stat, options, picture, kind, zrange, mask, caption, qty };
  },
  template: `
  <div class="mblf-modeview" :data-mblf-mode="mode.name">
    <template v-if="mode.kind === 'surface'">
      <div class="row wrap mblf-mv-head"><b class="small" v-if="title">{{ title }}</b><HelpTip v-if="help" :text="help"/><span class="grow"></span><Seg v-model="stat" :options="options" small data-mblf-stat/></div>
      <EEMView :ex="mode.ex" :em="mode.em" :z="picture" :mask="mask" :kind="kind" :zrange="zrange" :units="stat === 'p_pos' ? 'probability' : units" :height="compact ? 260 : 420" :compact="compact" :export-name="exportName" marginals="none"/>
      <p class="muted xs mblf-caption">{{ caption }}.</p>
    </template>
    <MblfFan v-else-if="mode.kind === 'curve'" :x="mode.x" :summary="summary" :observed="observed" :title="title" :help="help" :y-title="units" :zero="signed" :height="height" :simultaneous="simultaneous"/>
    <MblfValuesChart v-else :labels="mode.labels" :summary="summary" :observed="observed" :title="title" :help="help" :zero="signed" :mode="mode" :quantity="qty" :y-title="units"/>
  </div>`,
};

// ---------------------------------------------------------------- the uncertainty budget
export const MblfBudget = {
  name: 'MblfBudget',
  components: { PlotlyChart, ChartCard, EEMView, Icon },
  props: { mode: { type: Object, required: true }, budget: { type: Object, required: true }, total: { type: Object, required: true }, noisy: Boolean },
  setup(props) {
    const shown = computed(() => PARTS.filter((p) => props.noisy || p !== 'noise'));
    const colours = computed(() => { themeState.version; const t = tokens(); return { parameters: t.series[3], scores: t.series[0], residual: t.series[1], noise: t.series[2] }; });
    const shares = computed(() => {
      const b = Object.fromEntries(shown.value.map((p) => [p, nd(props.budget[p])]));
      const n = b[shown.value[0]].data.length, tot = new Float64Array(n);
      for (const p of shown.value) for (let i = 0; i < n; i++) tot[i] += b[p].data[i];
      const out = {};
      for (const p of shown.value) out[p] = { data: Float64Array.from(b[p].data, (v, i) => (tot[i] > 0 ? v / tot[i] : NaN)), shape: b[p].shape };
      out.sd = { data: Float64Array.from(tot, (v) => Math.sqrt(v)), shape: b[shown.value[0]].shape };
      return out;
    });
    const bar = computed(() => {
      themeState.version;
      return shown.value.map((p) => ({ x: [props.total[p]], y: ['whole mode'], type: 'bar', orientation: 'h', name: PART_LABELS[p], marker: { color: colours.value[p] },
        hovertemplate: `${PART_LABELS[p]}: %{x:.1%}<extra></extra>` }));
    });
    const barLayout = { barmode: 'stack', xaxis: { range: [0, 1], tickformat: '.0%', title: 'Share of the predictive variance' }, yaxis: { visible: false }, margin: { t: 6, b: 40, l: 10, r: 10 }, legend: { orientation: 'h', y: -0.9 }, showlegend: true };
    const curve = computed(() => {
      themeState.version;
      const x = props.mode.kind === 'curve' ? Array.from(props.mode.x) : props.mode.labels;
      return shown.value.map((p) => ({ x, y: arr(shares.value[p]), type: 'scatter', mode: 'lines', stackgroup: 'one', name: PART_LABELS[p], line: { width: 0.5, color: colours.value[p] },
        fillcolor: withAlpha(colours.value[p], 0.75), hovertemplate: `${PART_LABELS[p]}: %{y:.0%}<extra></extra>` }));
    });
    const curveLayout = computed(() => ({ yaxis: { range: [0, 1], tickformat: '.0%', title: 'Share of the variance' }, xaxis: { title: props.mode.kind === 'curve' ? 'Wavelength (nm)' : '' }, margin: { t: 10, r: 10 }, legend: { orientation: 'h', y: -0.3 } }));
    const big = computed(() => shown.value.reduce((a, p) => (props.total[p] > props.total[a] ? p : a), shown.value[0]));
    const mask = computed(() => (props.mode.mask ? img(nd(props.mode.mask)) : null));
    const pic = (p) => img(shares.value[p]);
    return { shown, bar, barLayout, curve, curveLayout, big, PART_LABELS, PART_WORDS, pct, mask, pic, shares, img };
  },
  template: `
  <div class="mblf-budget" data-mblf-budget>
    <p class="small"><b>Where the uncertainty comes from.</b> The predictive variance is the sum of {{ shown.length }} parts; over the whole mode the largest is <b>{{ PART_LABELS[big].toLowerCase() }}</b> ({{ pct(total[big], 0) }}): {{ PART_WORDS[big] }}.</p>
    <PlotlyChart :data="bar" :layout="barLayout" :height="130" data-mblf-budget-bar/>
    <ChartCard v-if="mode.kind !== 'surface'" title="Share of each part, point by point" :data="curve" flat><PlotlyChart :data="curve" :layout="curveLayout" :height="260"/></ChartCard>
    <div v-else class="mblf-budget-maps">
      <div v-for="p in shown" :key="p" class="mblf-budget-map"><div class="xs"><b>{{ PART_LABELS[p] }}</b> (share of the variance)</div>
        <EEMView :ex="mode.ex" :em="mode.em" :z="pic(p)" :mask="mask" :zrange="[0, 1]" units="share" compact :height="220"/></div>
    </div>
    <ul class="mblf-parts small">
      <li v-for="p in shown" :key="p"><b>{{ PART_LABELS[p] }}</b> ({{ pct(total[p], 0) }}): {{ PART_WORDS[p] }}.</li>
    </ul>
  </div>`,
};

/** One number with its credible interval, as a small horizontal bar (for shares and fractions in [0, 1]). */
export const MblfInterval = {
  name: 'MblfInterval',
  props: { value: { type: Object, required: true }, percent: { type: Boolean, default: true } },
  setup(props) {
    const w = (x) => `${Math.max(0, Math.min(100, 100 * x)).toFixed(1)}%`;
    return { w };
  },
  template: `
  <span class="mblf-ivl" :title="'Posterior mean ' + (percent ? (100 * value.mean).toFixed(1) + ' %' : value.mean.toFixed(3)) + ', 90 % interval ' + (percent ? (100 * value.lower).toFixed(1) + ' to ' + (100 * value.upper).toFixed(1) + ' %' : value.lower.toFixed(3) + ' to ' + value.upper.toFixed(3))">
    <span class="track"><span class="range" :style="{ left: w(value.lower), width: 'calc(' + w(value.upper) + ' - ' + w(value.lower) + ')' }"></span><span class="dot" :style="{ left: w(value.mean) }"></span></span>
    <span class="v">{{ percent ? (100 * value.mean).toFixed(0) + ' %' : value.mean.toFixed(2) }}</span>
  </span>`,
};

export { Switch };
