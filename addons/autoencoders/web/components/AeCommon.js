// Small pieces shared by the steps of the Autoencoders page: a number card, a verdict notice, a titled block, a strip of EEM pictures drawn as ONE chart (so that it has the
// .rds / .fig buttons, a shared colour scale and a table view), a histogram, and the plain-wording helpers.
import { ref, computed, onMounted, onBeforeUnmount } from 'vue';
import { Icon } from '/static/js/icons.js';
import { PlotlyChart, ChartCard } from '/static/js/plot.js';
import { HelpTip } from '/static/js/components/Common.js';
import { tokens, themeState, withAlpha, sequentialScale, divergingScale } from '/static/js/theme.js';
import { prefs } from '/static/js/prefs.js';
import { toRows, autoRange } from '/static/js/eem.js';
import { fmt } from '/static/js/util.js';

export { fmt, Icon };

// ---------------------------------------------------------------- plain wording
export const pct = (x, d = 1) => (Number.isFinite(x) ? `${(100 * x).toFixed(d)} %` : '–');
export const nTxt = (n) => (Number.isFinite(n) ? String(Math.round(n)).replace(/\B(?=(\d{3})+(?!\d))/g, ' ') : '–');
export const plural = (n, one, many) => `${nTxt(n)} ${n === 1 ? one : (many || `${one}s`)}`;
export const listWords = (a) => (a.length <= 1 ? a.join('') : `${a.slice(0, -1).join(', ')} and ${a[a.length - 1]}`);
/** The units of the answers: a text, or {fluorescence: 'Raman units (RU)', absorbance: ...} (one entry per variable). */
export const unitsOf = (u, name = 'fluorescence') => (u && typeof u === 'object' ? (u[name] ?? Object.values(u)[0] ?? '') : (u || ''));
/** 'Raman units (RU)' -> 'RU' */
export const shortUnits = (u) => { const t = String(unitsOf(u)).trim(); const i = t.lastIndexOf('('); return t.endsWith(')') && i >= 0 && t.length - i > 2 ? t.slice(i + 1, -1).trim() : t; };
export const sdText = (v) => (v === 0 ? 'mean' : `${v > 0 ? '+' : '−'}${Math.abs(v)} SD`);
/** The words of the status of a sample for the chips. */
export const STATUS_LABEL = { ok: 'Fits the network', outside: 'Poorly described by the network', unusable: 'Cannot be used' };

/** True while the window is narrow (phones, split screens): the strips use fewer pictures per row. */
export function useNarrow(px = 760) {
  const narrow = ref(typeof window !== 'undefined' && window.matchMedia ? window.matchMedia(`(max-width: ${px}px)`).matches : false);
  let mq = null;
  const on = (e) => { narrow.value = e.matches; };
  onMounted(() => { if (window.matchMedia) { mq = window.matchMedia(`(max-width: ${px}px)`); narrow.value = mq.matches; mq.addEventListener('change', on); } });
  onBeforeUnmount(() => { if (mq) mq.removeEventListener('change', on); });
  return narrow;
}

// ---------------------------------------------------------------- one number
export const AeStat = {
  name: 'AeStat',
  props: { label: String, value: [String, Number], sub: { type: String, default: '' }, level: { type: String, default: '' }, help: { type: String, default: '' } },
  components: { HelpTip },
  template: `
  <div class="fp-stat ae-stat" :data-level="level">
    <span class="l">{{ label }}<HelpTip v-if="help" :text="help"/></span>
    <span class="v" :class="level">{{ value }}</span>
    <span class="s" v-if="sub">{{ sub }}</span>
  </div>`,
};

/** A verdict in words: the server's sentence (or the page's), with the colour of how good it is.  level: good | info | warning | critical */
export const AeVerdict = {
  name: 'AeVerdict',
  components: { Icon },
  props: { level: { type: String, default: 'info' }, title: { type: String, default: '' } },
  template: `<div class="callout" :class="level === 'good' ? 'good' : level === 'warning' ? 'warning' : level === 'critical' ? 'critical' : 'info'" data-ae-verdict :data-level="level"><Icon :name="level === 'good' ? 'check-circle' : level === 'info' ? 'info-circle' : 'exclamation-triangle'"/><div class="grow"><b v-if="title">{{ title }} </b><slot/></div></div>`,
};

/** A block of a step: a heading, one sentence about what it is, then the controls. */
export const AeSection = {
  name: 'AeSection',
  components: { HelpTip },
  props: { title: String, sentence: { type: String, default: '' }, help: { type: String, default: '' } },
  template: `<section class="ae-section"><h3 class="fp-h3">{{ title }}<HelpTip v-if="help" :text="help"/></h3><p class="fp-note" v-if="sentence">{{ sentence }}</p><slot/></section>`,
};

// ---------------------------------------------------------------- a strip (or a small grid) of EEM pictures as ONE chart
// The pictures follow Settings > Appearance > Axes of fluorescence maps of FluoViz like every EEM picture: emission across and excitation up, or the other way round.
// (A FluoViz without that setting has no `prefs.eemTranspose`: the usual layout.  The images stay [n_ex, n_em]; only what is drawn is turned.)
/** Rows of `rows` turned: out[j][i] = rows[i][j]. */
const transposeRows = (rows) => {
  const nr = rows.length, nc = nr ? rows[0].length : 0, out = new Array(nc);
  for (let j = 0; j < nc; j++) { const r = new Float32Array(nr); for (let i = 0; i < nr; i++) r[i] = rows[i][j]; out[j] = r; }
  return out;
};
/** `images`: typed arrays with .shape [n_ex, n_em] (null = nothing there); `labels`: one title per picture; `kind`: sequential | diverging (signed, symmetric around 0);
 *  `mask`: Uint8Array [n_ex, n_em], 1 = not in the model (grey).  One colour scale for all the pictures (`zrange` or the range of all of them). */
export const AeStrip = {
  name: 'AeStrip',
  components: { PlotlyChart },
  props: {
    images: { type: Array, required: true }, labels: { type: Array, default: () => [] }, ex: { required: true }, em: { required: true }, mask: { default: null },
    kind: { type: String, default: 'sequential' }, zrange: { type: Array, default: null }, units: { type: String, default: '' }, cols: { type: Number, default: 7 }, rowHeight: { type: Number, default: 150 },
    highlight: { type: Number, default: -1 },
    masks: { type: Array, default: null },                      // one mask (like `mask`) for each picture; a picture without one uses `mask`
    nticks: { type: Number, default: 3 },                       // ticks on each axis (2 for a grid of many small pictures)
  },
  setup(props) {
    const nImg = computed(() => props.images.length);
    const cols = computed(() => Math.max(1, Math.min(props.cols, nImg.value)));
    const rows = computed(() => Math.ceil(nImg.value / cols.value));
    const range = computed(() => {
      if (props.zrange) return props.zrange;
      let lo = Infinity, hi = -Infinity;
      for (const im of props.images) { if (!im) continue; const r = autoRange(toRows(im), props.kind, 0.001, 0.999); if (r[0] < lo) lo = r[0]; if (r[1] > hi) hi = r[1]; }
      if (props.kind === 'diverging') { const m = Math.max(Math.abs(lo), Math.abs(hi)) || 1; return [-m, m]; }
      return lo === Infinity ? [0, 1] : [Math.min(0, lo), hi > 0 ? hi : 1];
    });
    const height = computed(() => rows.value * props.rowHeight + 54);
    const layoutOf = () => {
      const t = tokens(), k = cols.value, R = rows.value;
      const gx = 0.012, topPad = 22 / height.value, ph = (1 - 0) / R, tp = !!prefs.eemTranspose;
      const acrossName = tp ? 'Excitation (nm)' : 'Emission (nm)', upName = tp ? 'Emission (nm)' : 'Excitation (nm)';
      const L = { margin: { l: 44, r: 62, t: 6, b: 38 }, annotations: [], showlegend: false, hovermode: 'closest' };
      const ax = (extra) => ({ showline: true, linecolor: t.lineStrong, gridcolor: 'rgba(0,0,0,0)', tickfont: { size: 9.5, color: t.text3, family: t.font }, ticks: 'outside', ticklen: 2, tickcolor: t.lineStrong, zeroline: false, ...extra });
      const em = props.em, ex = props.ex, xRange = tp ? [ex[0], ex[ex.length - 1]] : [em[0], em[em.length - 1]], yRange = tp ? [em[0], em[em.length - 1]] : [ex[0], ex[ex.length - 1]];
      for (let i = 0; i < nImg.value; i++) {
        const r = Math.floor(i / k), c = i % k, n = i + 1;
        const x0 = c / k + gx, x1 = (c + 1) / k - gx, y1 = 1 - r * ph - topPad, y0 = 1 - (r + 1) * ph + (r === R - 1 ? 0 : 0.012);
        const xk = i === 0 ? 'xaxis' : `xaxis${n}`, yk = i === 0 ? 'yaxis' : `yaxis${n}`;
        L[xk] = ax({ domain: [x0, x1], anchor: i === 0 ? 'y' : `y${n}`, range: xRange, nticks: props.nticks, ...(props.nticks < 3 ? { tickangle: 0 } : {}), showticklabels: r === R - 1, ...(i === 0 ? {} : { matches: 'x' }),
          title: r === R - 1 && c === Math.floor((k - 1) / 2) && k > 1 ? { text: acrossName, font: { size: 11, color: t.text2 }, standoff: 4 } : (r === R - 1 && k === 1 ? { text: acrossName, font: { size: 11, color: t.text2 }, standoff: 4 } : undefined) });
        L[yk] = ax({ domain: [y0, y1], anchor: i === 0 ? 'x' : `x${n}`, range: yRange, nticks: props.nticks, showticklabels: c === 0, ...(i === 0 ? {} : { matches: 'y' }),
          title: c === 0 && r === 0 ? { text: upName, font: { size: 11, color: t.text2 }, standoff: 2 } : undefined });
        const lab = props.labels[i];
        if (lab) L.annotations.push({ xref: 'paper', yref: 'paper', x: (x0 + x1) / 2, y: y1, xanchor: 'center', yanchor: 'bottom', text: props.highlight === i ? `<b>${lab}</b>` : lab, showarrow: false, font: { size: 11, color: props.highlight === i ? t.text : t.text2, family: t.font } });
      }
      return L;
    };
    const traces = computed(() => {
      themeState.version;
      const t = tokens(), [lo, hi] = range.value, k = cols.value;
      const scaleDef = props.kind === 'diverging' ? divergingScale(prefs.div) : sequentialScale(prefs.seq);
      const out = [], em = Array.from(props.em), ex = Array.from(props.ex);
      const nem = em.length, tp = !!prefs.eemTranspose, hx = tp ? ex : em, hy = tp ? em : ex;
      const maskRows = (m) => { const rows = new Array(ex.length); for (let i = 0; i < ex.length; i++) { const r = new Float32Array(nem); for (let j = 0; j < nem; j++) r[j] = m[i * nem + j] ? 1 : NaN; rows[i] = r; } return tp ? transposeRows(rows) : rows; };
      const shared = props.mask ? maskRows(props.mask) : null;
      props.images.forEach((im, i) => {
        if (!im) return;
        const mz = props.masks && props.masks[i] ? maskRows(props.masks[i]) : shared;            // (a picture may have a grey of its own: pixels without an effect)
        const n = i + 1, xa = i === 0 ? 'x' : `x${n}`, ya = i === 0 ? 'y' : `y${n}`;
        const last = i === props.images.length - 1;
        out.push({ type: 'heatmap', x: hx, y: hy, z: tp ? transposeRows(toRows(im)) : toRows(im), zmin: lo, zmax: hi, colorscale: scaleDef, xaxis: xa, yaxis: ya, zsmooth: false, hoverongaps: false, name: props.labels[i] || '',
          showscale: last, colorbar: last ? { thickness: 10, len: 0.9, x: 1.01, xanchor: 'left', outlinewidth: 0, tickfont: { size: 10, color: t.text3, family: t.font }, title: { text: props.units, side: 'top', font: { size: 11, color: t.text2 } } } : undefined,
          hovertemplate: `${(props.labels[i] || '').replace(/<[^>]*>/g, '')}<br>${tp ? 'Ex %{x:.0f} nm · Em %{y:.0f} nm' : 'Em %{x:.0f} nm · Ex %{y:.0f} nm'}<br><b>%{z:.4g}</b> ${props.units}<extra></extra>` });
        if (mz) out.push({ type: 'heatmap', x: hx, y: hy, z: mz, zmin: 0, zmax: 1, showscale: false, hoverinfo: 'skip', xaxis: xa, yaxis: ya, zsmooth: false, colorscale: [[0, withAlpha(t.text3, t.dark ? 0.34 : 0.38)], [1, withAlpha(t.text3, t.dark ? 0.34 : 0.38)]] });
      });
      void k;
      return out;
    });
    const layout = computed(() => { themeState.version; return layoutOf(); });
    return { traces, layout, height };
  },
  template: `<PlotlyChart :data="traces" :layout="layout" :height="height" :config="{ displayModeBar: false }"/>`,
};

/** A card with a strip in it (title, help, table view / PNG / CSV / .rds / .fig). */
export const AeStripCard = {
  name: 'AeStripCard',
  components: { ChartCard, AeStrip },
  props: { title: String, help: String, filename: String, ...AeStrip.props },
  // (a strip is an image: the card keeps the buttons for R and MATLAB; the numbers themselves are in the downloads of step 9)
  setup(props) { const strip = computed(() => { const { title, help, filename, ...rest } = props; return rest; }); return { strip }; },
  template: `<ChartCard :title="title" :help="help" :filename="filename" no-export figures><AeStrip v-bind="strip"/><slot/></ChartCard>`,
};

// ---------------------------------------------------------------- a histogram with a limit
/** `edges` (n+1) and `counts` (n); `limit` a vertical dashed line named `limitLabel`; `held` (counts of the held-out samples, optional) is drawn over the others. */
export const AeHistogram = {
  name: 'AeHistogram',
  components: { PlotlyChart, ChartCard },
  props: { title: String, help: String, edges: { type: Array, required: true }, counts: { type: Array, required: true }, held: { type: Array, default: null }, limit: { type: Number, default: null },
    limitLabel: { type: String, default: 'limit' }, xTitle: { type: String, default: '' }, filename: { type: String, default: 'histogram' }, height: { type: Number, default: 260 }, logX: Boolean },
  setup(props) {
    const tx = (v) => (props.logX ? Math.log10(Math.max(v, 1e-12)) : v);                  // a logarithmic axis: the bars are drawn at the logarithms (bars have a width in data units)
    const mids = computed(() => props.counts.map((_, i) => (tx(props.edges[i]) + tx(props.edges[i + 1])) / 2));
    const widths = computed(() => props.counts.map((_, i) => (tx(props.edges[i + 1]) - tx(props.edges[i])) * 0.94));
    const traces = computed(() => {
      themeState.version;
      const t = tokens(), out = [], lab = (x) => (props.logX ? fmt(10 ** x, 3) : fmt(x, 3));
      out.push({ type: 'bar', x: mids.value, y: props.counts, width: widths.value, name: props.held ? 'training samples' : 'samples', marker: { color: props.held ? withAlpha(t.series[0], 0.55) : t.series[0], line: { color: t.surface, width: 1 } },
        customdata: mids.value.map(lab), hovertemplate: '%{customdata}: %{y} samples<extra></extra>' });
      if (props.held) out.push({ type: 'bar', x: mids.value, y: props.held, width: widths.value, name: 'held-out samples', marker: { color: t.series[5], line: { color: t.surface, width: 1 } }, customdata: mids.value.map(lab), hovertemplate: '%{customdata}: %{y} held-out samples<extra></extra>' });
      return out;
    });
    const ticks = computed(() => {
      if (!props.logX) return {};
      const lo = Math.floor(tx(props.edges[0])), hi = Math.ceil(tx(props.edges[props.edges.length - 1])), vals = [], text = [];
      for (let k = lo; k <= hi; k++) { vals.push(k); text.push(fmt(10 ** k, 3)); }
      return { tickmode: 'array', tickvals: vals, ticktext: text };
    });
    const layout = computed(() => {
      themeState.version;
      const t = tokens(), lim = Number.isFinite(props.limit) ? tx(props.limit) : null;
      const L = { height: props.height, margin: { l: 52, r: 14, t: 16, b: 48 }, barmode: 'overlay', showlegend: !!props.held, legend: { orientation: 'h', y: 1.14, x: 0 },
        xaxis: { title: { text: props.xTitle }, ...ticks.value }, yaxis: { title: { text: 'Samples' }, rangemode: 'tozero' }, shapes: [], annotations: [] };
      if (lim !== null) {
        L.shapes.push({ type: 'line', xref: 'x', yref: 'paper', x0: lim, x1: lim, y0: 0, y1: 1, line: { color: t.critical, width: 1.6, dash: 'dash' } });
        L.annotations.push({ xref: 'x', yref: 'paper', x: lim, y: 1, xanchor: 'left', yanchor: 'top', text: ` ${props.limitLabel}`, showarrow: false, font: { size: 11, color: t.text2, family: t.font } });
      }
      return L;
    });
    const table = computed(() => ({ columns: ['from', 'to', 'samples', ...(props.held ? ['held-out samples'] : [])], rows: props.counts.map((c, i) => [props.edges[i], props.edges[i + 1], c, ...(props.held ? [props.held[i]] : [])]) }));
    return { traces, layout, table };
  },
  template: `<ChartCard :title="title" :help="help" :table="table" :filename="filename"><PlotlyChart :data="traces" :layout="layout" :height="height"/></ChartCard>`,
};

// ---------------------------------------------------------------- tabs that can be switched off (with the reason) and are operable with the arrow keys
export const AeTabs = {
  name: 'AeTabs',
  props: { modelValue: String, tabs: { type: Array, required: true }, label: { type: String, default: 'Views' }, hook: { type: String, default: 'data-ae-tab' } },     // [{id, label, disabled, why}]
  emits: ['update:modelValue'],
  setup(props, { emit }) {
    function key(e, i) {
      const step = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0; if (!step) return;
      e.preventDefault();
      const list = props.tabs;
      for (let k = 1; k <= list.length; k++) { const t = list[(i + step * k + list.length * 4) % list.length]; if (!t.disabled) { emit('update:modelValue', t.id); const el = e.currentTarget.parentElement.querySelector(`[data-tab-id="${t.id}"]`); if (el) el.focus(); return; } }
    }
    return { key };
  },
  template: `
  <div class="tabs ae-tabs" role="tablist" :aria-label="label">
    <button v-for="(t, i) in tabs" :key="t.id" type="button" class="tab" :class="{ on: t.id === modelValue, off: t.disabled }" role="tab" :aria-selected="t.id === modelValue ? 'true' : 'false'" :aria-disabled="t.disabled ? 'true' : null"
            :tabindex="t.id === modelValue ? 0 : -1" :data-tab-id="t.id" :data-ae-tab="t.id" :title="t.disabled ? t.why : ''" @click="!t.disabled && $emit('update:modelValue', t.id)" @keydown="key($event, i)">{{ t.label }}</button>
  </div>`,
};
