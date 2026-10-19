// Tab "Nonlinear effects", section "Where it bends": at every pixel the response to a code is followed along the traversal and compared with the best straight line.  One EEM per code
// (0 = a straight line, 1 = nothing about the response is linear, read on the colour bar; a grey pixel is outside the model or hardly changes), a bar for the share of each code's effect that no straight line describes, and for the
// absorbance a curve of the same index against the wavelength.
import { computed, watch, onMounted } from 'vue';
import { Field, NumField, Callout } from '/static/js/components/Common.js';
import { PlotlyChart, ChartCard } from '/static/js/plot.js';
import { sliceImage, vec, maskArr } from '/static/js/components/DecompCommon.js';
import { tokens, themeState, withAlpha } from '/static/js/theme.js';
import { debounce, fmt } from '/static/js/util.js';
import { fp } from './AeState.js';
import { loadBends } from './AeNlActions.js';
import { LIMITS, SHOWN } from './AeNlState.js';
import { pctText } from './AeNlCommon.js';
import { AeStripCard, useNarrow, shortUnits, unitsOf } from './AeCommon.js';

/** The index (0 to 1) and the size of the effect of one code on the absorbance, against the wavelength. */
export const AeNlBendCurve = {
  name: 'AeNlBendCurve',
  components: { ChartCard, PlotlyChart },
  props: { answer: { type: Object, required: true }, j: { type: Number, required: true } },
  setup(props) {
    const P = computed(() => props.answer), name = computed(() => P.value.names[props.j]);
    const row = (m, j) => Array.from(m.data.subarray(j * m.k, (j + 1) * m.k), (v) => (Number.isFinite(v) ? v : null));
    const traces = computed(() => {
      themeState.version;
      const t = tokens(), x = Array.from(P.value.ab.x), ab = P.value.ab, top = ab.maxAmp[props.j] || 1;
      const amp = row(ab.amp, props.j).map((v) => (v === null ? null : v / top));
      return [{ type: 'scatter', mode: 'lines', x, y: amp, fill: 'tozeroy', fillcolor: withAlpha(t.text3, 0.18), line: { color: withAlpha(t.text3, 0.7), width: 1, dash: 'dot' }, name: 'Size of the effect (relative)', hovertemplate: '%{x:.0f} nm: %{y:.0%} of the largest effect<extra>size of the effect</extra>' },
        { type: 'scatter', mode: 'lines', x, y: row(ab.nl, props.j), line: { color: t.series[0], width: 2.4 }, name: 'Nonlinearity (NL)', connectgaps: false, hovertemplate: '%{x:.0f} nm: NL = %{y:.2f}<extra>where it bends</extra>' }];
    });
    const layout = computed(() => ({ height: 210, margin: { l: 52, r: 12, t: 8, b: 74 }, showlegend: true, legend: { orientation: 'h', y: -0.34, x: 0, font: { size: 10.5 } }, hovermode: 'x unified',
      xaxis: { title: { text: 'Wavelength (nm)' } }, yaxis: { title: { text: 'NL (0 to 1)' }, range: [0, 1.04], zeroline: false } }));
    const table = computed(() => { const ab = P.value.ab; return { columns: ['wavelength (nm)', 'NL', 'size of the effect'], rows: Array.from(ab.x, (w, i) => [w, ab.nl.data[props.j * ab.nl.k + i], ab.amp.data[props.j * ab.amp.k + i]]) }; });
    return { name, traces, layout, table };
  },
  template: `
  <div :data-ae-nl-bend-curve="j + 1"><ChartCard :title="name + ': absorbance'" :table="table" :filename="'autoencoder-bends-absorbance-' + name"
    help="Where along the wavelengths the response of the absorbance to this code bends: 0 is a straight line, 1 is nothing like one. The grey area is the size of the effect (relative to its largest value); where it is small the index means little."><PlotlyChart :data="traces" :layout="layout" :height="210"/></ChartCard></div>`,
};

export const AeNlBends = {
  name: 'AeNlBends',
  components: { Field, NumField, Callout, PlotlyChart, ChartCard, AeStripCard, AeNlBendCurve },
  setup() {
    const A = computed(() => fp.ae), R = computed(() => fp.ae.res), N = computed(() => fp.ae.nl);
    const narrow = useNarrow();
    const slot = computed(() => A.value.extra.nlBends || null);
    const P = computed(() => (slot.value ? slot.value.data : null));
    const go = debounce(() => loadBends(), 350);
    onMounted(() => loadBends());
    watch(() => [N.value.span, N.value.n_points, A.value.resRid], () => go());
    const d = computed(() => (R.value ? R.value.d : 0));
    const nShown = computed(() => (N.value.showAll ? d.value : Math.min(d.value, SHOWN)));
    const show = computed({ get: () => N.value.bendsShow, set: (v) => { N.value.bendsShow = v; } });
    const strip = computed(() => {
      const p = P.value; if (!p || !p.fl) return null;
      const fl = p.fl, st = show.value === 'nl' ? fl.nl : fl.amp, model = p.mask ? maskArr(p.mask) : null, n = nShown.value, units = shortUnits(unitsOf(p.units));
      const imgs = [], masks = [], labels = [];
      for (let i = 0; i < n; i++) {
        const im = sliceImage(st, i); imgs.push(im);
        if (show.value === 'nl') { const m = new Uint8Array(im.length); for (let k = 0; k < im.length; k++) m[k] = (model && model[k]) || !Number.isFinite(im[k]) ? 1 : 0; masks.push(m); } else masks.push(model);
        labels.push(show.value === 'nl' ? `${p.names[i]} · ${p.dead[i] ? 'no effect' : pctText(p.share[i])}` : `${p.names[i]} · ${p.dead[i] ? 'no effect' : fmt(fl.maxAmp[i], 3)}`);
      }
      const top = Math.max(...Array.from(fl.maxAmp).slice(0, n)) || 1;
      return { imgs, masks, labels, ex: vec(p.ex), em: vec(p.em), mask: model, zrange: show.value === 'nl' ? [0, 1] : [0, top], units: show.value === 'nl' ? 'NL' : units };
    });
    const cols = computed(() => Math.max(1, Math.min(nShown.value, narrow.value ? 2 : 4)));
    // the bars: the share of each code's effect that is not a straight line
    const bars = computed(() => {
      themeState.version;
      const p = P.value; if (!p) return [];
      const t = tokens();
      return [{ type: 'bar', x: p.names, y: Array.from(p.share, (v, i) => (p.dead[i] || !Number.isFinite(v) ? 0 : v)), marker: { color: Array.from(p.share, (v, i) => (p.dead[i] ? withAlpha(t.text3, 0.35) : t.series[0])), line: { color: t.surface, width: 1 } },
        text: Array.from(p.share, (v, i) => (p.dead[i] ? 'no effect' : pctText(v))), textposition: 'outside', cliponaxis: false, textfont: { size: 11, color: t.text2 },
        customdata: Array.from(p.share, (v, i) => [p.dead[i] ? 'none' : pctText(p.effectShare[i])]),
        hovertemplate: '%{x}: %{text} of its effect is not a straight line<br>this code carries %{customdata[0]} of the total effect of the codes<extra></extra>', name: 'nonlinear share' }];
    });
    const barLayout = computed(() => {
      themeState.version;
      const t = tokens(), p = P.value, top = p ? Math.max(0, ...Array.from(p.share, (v, i) => (p.dead[i] || !Number.isFinite(v) ? 0 : v))) : 0;
      return { height: 250, margin: { l: 60, r: 14, t: 24, b: 46 }, xaxis: { title: { text: 'Code' } }, yaxis: { title: { text: 'Share that is not a straight line' }, tickformat: '.0%', range: [0, Math.min(1.12, Math.max(0.2, 1.3 * top))], zeroline: false },
        shapes: [{ type: 'line', xref: 'paper', yref: 'y', x0: 0, x1: 1, y0: p ? p.nearly_straight : 0, y1: p ? p.nearly_straight : 0, layer: 'below', line: { color: t.lineStrong, width: 1.2, dash: 'dot' } }] };
    });
    const barTable = computed(() => { const p = P.value; return p ? { columns: ['code', 'nonlinear share', 'share of the total effect of the codes', ...(p.fl ? ['fluorescence: nonlinear share'] : []), ...(p.ab ? ['absorbance: nonlinear share'] : [])],
      rows: p.names.map((n, i) => [n, p.dead[i] ? null : p.share[i], p.effectShare[i], ...(p.fl ? [p.fl.share[i]] : []), ...(p.ab ? [p.ab.share[i]] : [])]) } : { columns: [], rows: [] }; });
    const curves = computed(() => (P.value && P.value.ab ? Array.from({ length: nShown.value }, (_, i) => i) : []));
    const hidden = computed(() => Math.max(0, d.value - nShown.value));
    return { A, R, N, slot, P, d, narrow, show, strip, cols, bars, barLayout, barTable, curves, hidden, LIMITS, SHOWN, pctText };
  },
  template: `
  <div class="col gap-12" data-ae-nl-panel="bends">
    <p class="fp-note" data-ae-nl-explain>For every code, each pixel of the EEM is followed along the code and compared with the best straight line: the pixel scores 0 where it moves in a straight line like a PCA loading and 1 where nothing about it is linear (the colour bar says which colour is which), and the bars give the share of each code's whole effect that no straight line describes.</p>
    <div class="card"><div class="card-body" style="padding:12px 16px"><div class="row wrap" style="gap:12px 26px;align-items:flex-end">
      <Field label="Range (± SD)" help="How far each code is moved from its average, in standard deviations (SD) of the code over the training samples. 0.5 to 4."><NumField v-model="N.span" :min="LIMITS.span[0]" :max="LIMITS.span[1]" :step="0.5" small data-ae-nl-span/></Field>
      <Field label="Points per curve" help="How many positions are decoded along each code. 5 to 201."><NumField v-model="N.n_points" :min="LIMITS.points[0]" :max="LIMITS.points[1]" :step="2" small data-ae-nl-points/></Field>
      <div class="col" style="gap:4px" v-if="P && P.fl"><span class="lbl">Show</span>
        <div class="seg ae-seg-wrap" role="group" data-ae-nl-bends-show><button type="button" :class="{ on: show === 'nl' }" title="Where the response of each pixel bends away from a straight line" @click="show = 'nl'">Where it bends</button>
          <button type="button" :class="{ on: show === 'amplitude' }" title="How much each pixel changes along the code (the range of its response)" @click="show = 'amplitude'">Size of the effect</button></div></div>
    </div></div></div>
    <Callout kind="critical" v-if="slot && slot.error" data-ae-nl-error>{{ slot.error }}</Callout>
    <div class="fp-wait muted" v-if="slot && slot.busy && !P">Following every pixel along every code …</div>
    <template v-if="P">
      <p class="fp-note" data-ae-nl-words>{{ P.words }}</p>
      <div :class="{ 'fp-dim': slot && slot.busy }" class="col gap-12">
        <div v-if="strip" data-ae-nl-strip>
          <AeStripCard :title="show === 'nl' ? 'Where each code bends the picture (NL = 1 − R² of the best straight line, 0 to 1)' : 'How much each code changes each pixel (range along the code)'" :images="strip.imgs" :labels="strip.labels" :ex="strip.ex" :em="strip.em"
                       :mask="strip.mask" :masks="strip.masks" :zrange="strip.zrange" :units="strip.units" :cols="cols" :row-height="narrow ? 120 : 150" :filename="'autoencoder-' + (show === 'nl' ? 'bends' : 'effect-size')"
                       :help="show === 'nl' ? 'NL is 0 where the response of the pixel to the code is a straight line and close to 1 where it is far from one. The same colour scale is used for every code.' : 'The range of the response of each pixel along the code, in the units of the data. The same colour scale is used for every code.'"/>
          <p class="fp-note" data-ae-nl-colour-note v-if="show === 'nl'">A pixel scores <b>0</b> (the bottom of the colour bar) where its response to this code is a straight line, as a PCA loading would be, and up to <b>1</b> (the top) where nothing about it is linear. <b>Grey</b> pixels are outside the model, or this code changes them by less than {{ Math.round(100 * P.min_effect) }} % of its largest effect, so the index would mean nothing there.</p>
          <p class="fp-note" v-else>The higher on the colour bar, the more this code changes the pixel between the two ends of its range. Grey pixels are outside the model.</p>
        </div>
        <ChartCard title="Share of the effect of each code that is not a straight line" :table="barTable" filename="autoencoder-nonlinear-share"
                   help="For each code: the part of its effect on the whole picture (measured the way the network was trained) that no straight line can describe. 0 % acts like a PCA loading, 100 % is entirely nonlinear."><PlotlyChart :data="bars" :layout="barLayout" :height="250" data-ae-nl-share-chart/>
          <p class="fp-note muted" style="padding:0 14px 10px" data-ae-nl-share-note>The dotted line is {{ pctText(P.nearly_straight) }}: a code below it acts almost like a straight line. A code that does not change the output at all has no share (it is marked "no effect").</p></ChartCard>
        <div class="ae-nl-grid" v-if="curves.length"><AeNlBendCurve v-for="i in curves" :key="i" :answer="P" :j="i"/></div>
        <div class="row wrap" v-if="hidden"><button type="button" class="btn sm" data-ae-nl-all title="Draw the pictures of the remaining codes too." @click="N.showAll = true">Show the other {{ hidden }} codes</button></div>
        <div class="row wrap" v-else-if="N.showAll && d > SHOWN"><button type="button" class="btn sm quiet" title="Draw only the pictures of the first codes." @click="N.showAll = false">Show only the first {{ SHOWN }} codes</button></div>
      </div>
    </template>
  </div>`,
};
