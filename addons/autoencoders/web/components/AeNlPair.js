// Tab "Nonlinear effects", section "Two codes": two codes moved together on a grid (all the others at their average).  If they act independently the picture is the sum of what each does
// alone; the interaction image and index say how much of it is not.  A heat map shows one feature over the plane of the two codes with the training samples as dots.
import { computed, watch, onMounted } from 'vue';
import { Field, NumField, Callout } from '/static/js/components/Common.js';
import { PlotlyChart, ChartCard } from '/static/js/plot.js';
import { sliceImage, vec, maskArr } from '/static/js/components/DecompCommon.js';
import { tokens, themeState, withAlpha, sequentialScale } from '/static/js/theme.js';
import { prefs } from '/static/js/prefs.js';
import { debounce, fmt } from '/static/js/util.js';
import { fp } from './AeState.js';
import { loadPair } from './AeNlActions.js';
import { LIMITS } from './AeNlState.js';
import { AeNlFeatureEditor, sdText, pctText } from './AeNlCommon.js';
import { AeStripCard, AeStat, useNarrow, shortUnits, unitsOf } from './AeCommon.js';

/** A grid of small curves drawn as ONE chart (so that it has the table view and the .rds / .fig buttons): `series` is {data, n, k} with grid x grid curves, rows = code j. */
export const AeNlCurveGrid = {
  name: 'AeNlCurveGrid',
  components: { PlotlyChart, ChartCard },
  props: { x: { required: true }, series: { type: Object, required: true }, grid: { type: Number, required: true }, u: { required: true }, names: { type: Array, required: true }, title: String, yTitle: String, filename: String },
  setup(props) {
    const g = computed(() => props.grid), height = computed(() => 70 * g.value + 70);
    const range = computed(() => { let lo = Infinity, hi = -Infinity; const d = props.series.data; for (let i = 0; i < d.length; i++) { const v = d[i]; if (v === v) { if (v < lo) lo = v; if (v > hi) hi = v; } } return lo === Infinity ? [0, 1] : [lo, hi + (hi - lo) * 0.05]; });
    const traces = computed(() => {
      themeState.version;
      const t = tokens(), G = g.value, x = Array.from(props.x), S = props.series, out = [];
      for (let r = 0; r < G; r++) for (let c = 0; c < G; c++) {
        const b = G - 1 - r, a = c, k = b * G + a, n = r * G + c + 1;
        out.push({ type: 'scatter', mode: 'lines', x, y: Array.from(S.data.subarray(k * S.k, (k + 1) * S.k), (v) => (Number.isFinite(v) ? v : null)), xaxis: n === 1 ? 'x' : `x${n}`, yaxis: n === 1 ? 'y' : `y${n}`,
          line: { color: a === (G >> 1) && b === (G >> 1) ? t.text : t.series[0], width: a === (G >> 1) && b === (G >> 1) ? 2 : 1.4 }, name: `${props.names[0]} ${sdText(props.u[a])}, ${props.names[1]} ${sdText(props.u[b])}`, hovertemplate: '%{x:.0f} nm: %{y:.4g}<extra>%{fullData.name}</extra>' });
      }
      return out;
    });
    const layout = computed(() => {
      themeState.version;
      const t = tokens(), G = g.value, L = { height: height.value, margin: { l: 44, r: 10, t: 22, b: 40 }, showlegend: false, annotations: [] };
      const ax = (extra) => ({ showline: true, linecolor: t.lineStrong, gridcolor: 'rgba(0,0,0,0)', tickfont: { size: 9, color: t.text3, family: t.font }, ticks: 'outside', ticklen: 2, zeroline: false, nticks: 3, ...extra });
      for (let r = 0; r < G; r++) for (let c = 0; c < G; c++) {
        const n = r * G + c + 1, x0 = c / G + 0.01, x1 = (c + 1) / G - 0.01, y1 = 1 - r / G - 0.02, y0 = 1 - (r + 1) / G + 0.04;
        L[n === 1 ? 'xaxis' : `xaxis${n}`] = ax({ domain: [x0, x1], anchor: n === 1 ? 'y' : `y${n}`, showticklabels: r === G - 1, ...(n === 1 ? {} : { matches: 'x' }), title: r === G - 1 && c === (G >> 1) ? { text: 'Wavelength (nm)', font: { size: 10.5, color: t.text2 }, standoff: 2 } : undefined });
        L[n === 1 ? 'yaxis' : `yaxis${n}`] = ax({ domain: [y0, y1], anchor: n === 1 ? 'x' : `x${n}`, showticklabels: c === 0, range: range.value, ...(n === 1 ? {} : { matches: 'y' }) });
        const b = G - 1 - r, a = c;
        L.annotations.push({ xref: 'paper', yref: 'paper', x: (x0 + x1) / 2, y: y1, xanchor: 'center', yanchor: 'bottom', text: `${sdText(props.u[a])} · ${sdText(props.u[b])}`, showarrow: false, font: { size: G > 5 ? 9 : 10.5, color: t.text2, family: t.font } });
      }
      return L;
    });
    const table = computed(() => { const S = props.series; return { columns: ['wavelength (nm)', ...Array.from({ length: S.n }, (_, k) => `${props.names[0]} ${sdText(props.u[k % props.grid])}, ${props.names[1]} ${sdText(props.u[Math.floor(k / props.grid)])}`)], rows: Array.from(props.x, (w, i) => [w, ...Array.from({ length: S.n }, (_, k) => S.data[k * S.k + i])]) }; });
    return { traces, layout, table, height };
  },
  template: `<ChartCard :title="title" :table="table" :filename="filename" help="One small chart for each pair of positions: the first code across, the second code up. The middle one (the thicker line) is the average code."><PlotlyChart :data="traces" :layout="layout" :height="height"/></ChartCard>`,
};

export const AeNlPair = {
  name: 'AeNlPair',
  components: { Field, NumField, Callout, PlotlyChart, ChartCard, AeStripCard, AeStat, AeNlFeatureEditor, AeNlCurveGrid },
  setup() {
    const A = computed(() => fp.ae), R = computed(() => fp.ae.res), N = computed(() => fp.ae.nl), Pr = computed(() => fp.ae.nl.pair);
    const narrow = useNarrow();
    const slot = computed(() => A.value.extra.nlPair || null);
    const P = computed(() => (slot.value ? slot.value.data : null));
    const d = computed(() => (R.value ? R.value.d : 0));
    const go = debounce(() => loadPair(), 350);
    onMounted(() => { fix(); loadPair(); });
    function fix() { const p = Pr.value, n = d.value; if (p.i && (p.i > n || p.j > n)) { p.i = null; p.j = null; } }
    watch(() => [Pr.value.i, Pr.value.j, Pr.value.grid, Pr.value.span, Pr.value.feature, A.value.resRid], () => { fix(); go(); }, { deep: true });
    const names = computed(() => (R.value ? R.value.codeNames : []));
    const ci = computed({ get: () => Pr.value.i || (P.value ? P.value.codes[0] : 1), set: (v) => { const j = Pr.value.j || (P.value ? P.value.codes[1] : 2); Pr.value.i = v; Pr.value.j = j; } });
    const cj = computed({ get: () => Pr.value.j || (P.value ? P.value.codes[1] : 2), set: (v) => { const i = Pr.value.i || (P.value ? P.value.codes[0] : 1); Pr.value.j = v; Pr.value.i = i; } });
    const same = computed(() => ci.value === cj.value);
    const pairNames = computed(() => (P.value ? [P.value.names[P.value.codes[0] - 1], P.value.names[P.value.codes[1] - 1]] : ['', '']));
    const pos = (v) => { const r = Math.round(Math.abs(v) * 10) / 10; return r === 0 ? '0' : `${v > 0 ? '+' : '−'}${Number.isInteger(r) ? r.toFixed(0) : r.toFixed(1)}`; };      // (a short label: -2, 0, +1.5)
    const gallery = computed(() => {
      const p = P.value; if (!p || !p.fl) return null;
      const G = p.grid, imgs = [], labels = [], [n1, n2] = pairNames.value;
      for (let r = 0; r < G; r++) for (let c = 0; c < G; c++) {
        const b = G - 1 - r, a = c;                                        // the highest position of the second code at the top
        imgs.push(sliceImage(p.fl.images, b * G + a));
        labels.push(G <= 3 && !narrow.value ? `${n1} ${sdText(p.u[a])} · ${n2} ${sdText(p.u[b])}` : `${pos(p.u[a])}, ${pos(p.u[b])}`);
      }
      let lo = Infinity, hi = -Infinity;
      for (const im of imgs) for (let i = 0; i < im.length; i++) { const v = im[i]; if (v === v) { if (v < lo) lo = v; if (v > hi) hi = v; } }
      return { imgs, labels, ex: vec(p.ex), em: vec(p.em), mask: p.mask ? maskArr(p.mask) : null, zrange: [Math.min(0, lo), hi > 0 ? hi : 1], units: shortUnits(unitsOf(p.units)), centre: (G >> 1) * G + (G >> 1) };
    });
    const interaction = computed(() => {
      const p = P.value; if (!p || !p.fl) return null;
      const im = p.fl.interaction; let hi = 0; for (let i = 0; i < im.length; i++) if (im[i] > hi) hi = im[i];
      return { img: im, zrange: [0, hi || 1] };
    });
    const heat = computed(() => {
      themeState.version;
      const p = P.value; if (!p) return { data: [], layout: {} };
      const t = tokens(), h = p.heatV, f = p.feature, dots = p.dotsV, span = p.span, nz = h.z.n, k = h.z.k, rows = Array.from({ length: nz }, (_, i) => Array.from(h.z.data.subarray(i * k, (i + 1) * k), (v) => (Number.isFinite(v) ? v : null)));
      let lo = Infinity, hi = -Infinity; for (const v of h.z.data) if (v === v) { if (v < lo) lo = v; if (v > hi) hi = v; }
      const inside = dots.x.map((x, i) => Math.abs(x) <= span && Math.abs(dots.y[i]) <= span);
      const idx = dots.x.map((_, i) => i).filter((i) => inside[i]);
      const colored = !!dots.value;
      const data = [{ type: 'heatmap', x: Array.from(h.x), y: Array.from(h.y), z: rows, zmin: lo, zmax: hi, colorscale: sequentialScale(prefs.seq), zsmooth: 'best', hoverongaps: false,
        colorbar: { thickness: 10, len: 0.9, outlinewidth: 0, tickfont: { size: 10, color: t.text3, family: t.font }, title: { text: f.units || '', side: 'top', font: { size: 11, color: t.text2 } } },
        hovertemplate: `${pairNames.value[0]} = %{x:+.1f} SD, ${pairNames.value[1]} = %{y:+.1f} SD<br>${f.short}: %{z:.4g} ${f.units || ''}<extra></extra>` },
        { type: 'scatter', mode: 'markers', x: idx.map((i) => dots.x[i]), y: idx.map((i) => dots.y[i]), customdata: idx.map((i) => [dots.ids[i], colored && Number.isFinite(dots.value[i]) ? dots.value[i] : null]), name: 'Training samples',
          marker: colored ? { size: 9, color: idx.map((i) => dots.value[i]), cmin: lo, cmax: hi, colorscale: sequentialScale(prefs.seq), line: { color: t.dark ? t.text : t.heritage, width: 1.6 } } : { size: 8, color: withAlpha(t.text, 0.55), line: { color: t.surface, width: 1.5 } },
          hovertemplate: colored ? '%{customdata[0]}<br>measured: %{customdata[1]:.4g}<extra>sample</extra>' : '%{customdata[0]}<extra>sample</extra>' }];
      const layout = { height: 400, margin: { l: 62, r: 14, t: 10, b: 52 }, xaxis: { title: { text: `${pairNames.value[0]} (SD from its mean)` }, range: [-span, span], zeroline: false }, yaxis: { title: { text: `${pairNames.value[1]} (SD from its mean)` }, range: [-span, span], zeroline: false },
        showlegend: false };
      return { data, layout, outside: dots.x.length - idx.length, colored };
    });
    const heatTable = computed(() => { const p = P.value; if (!p) return { columns: [], rows: [] }; const h = p.heatV; return { columns: [`${pairNames.value[0]} (SD)`, `${pairNames.value[1]} (SD)`, p.feature.short + (p.feature.units ? ` (${p.feature.units})` : '')],
      rows: Array.from({ length: h.z.n * h.z.k }, (_, q) => [h.x[q % h.z.k], h.y[Math.floor(q / h.z.k)], h.z.data[q]]) }; });
    const featureText = computed(() => { const p = P.value; if (!p) return ''; const f = p.feature; return f.kind === 'moved' && f.where && f.where.ex !== undefined ? `${f.label}: excitation ${fmt(f.where.ex, 0)} nm, emission ${fmt(f.where.em, 0)} nm` : f.label; });
    const level = computed(() => { const p = P.value; if (!p || !Number.isFinite(p.index)) return ''; return p.index < p.index_limits[0] ? 'ok' : p.index >= p.index_limits[1] ? 'bad' : ''; });
    function setFeature(spec) { Pr.value.feature = spec.kind === 'moved' ? null : spec; }
    return { A, R, N, Pr, P, slot, d, names, ci, cj, same, pairNames, gallery, interaction, heat, heatTable, featureText, level, narrow, setFeature, LIMITS, fmt, pctText, sdText };
  },
  template: `
  <div class="col gap-12" data-ae-nl-panel="pair">
    <p class="fp-note" data-ae-nl-explain>Two codes are moved together on a grid while all the others stay at their average: if they act independently the picture at each grid point is the sum of what each code does alone, and the interaction image and index show how much of the change is not.</p>
    <Callout kind="info" v-if="d < 2" data-ae-nl-pair-none>This network has only one code, so there are no two codes to look at together.</Callout>
    <template v-else>
    <div class="card"><div class="card-body col gap-12" style="padding:12px 16px">
      <div class="row wrap" style="gap:12px 22px;align-items:flex-end">
        <Field label="First code (across)" help="The code that changes from left to right in the pictures."><select class="select sm" v-model.number="ci" data-ae-nl-pair-i><option v-for="(n, i) in names" :key="n" :value="i + 1">{{ n }}</option></select></Field>
        <Field label="Second code (up)" help="The code that changes from bottom to top in the pictures."><select class="select sm" v-model.number="cj" data-ae-nl-pair-j><option v-for="(n, i) in names" :key="n" :value="i + 1">{{ n }}</option></select></Field>
        <div class="col" style="gap:4px"><span class="lbl">Pictures per code</span>
          <div class="seg ae-seg-wrap" role="group" data-ae-nl-pair-grid><button type="button" v-for="g in LIMITS.grid" :key="g" :class="{ on: Pr.grid === g }" :title="g + ' x ' + g + ' pictures'" @click="Pr.grid = g">{{ g }}</button></div></div>
        <Field label="Range (± SD)" help="How far each of the two codes is moved from its average, in standard deviations of the code. 0.5 to 4."><NumField v-model="Pr.span" :min="LIMITS.span[0]" :max="LIMITS.span[1]" :step="0.5" small data-ae-nl-pair-span/></Field>
      </div>
      <Callout kind="warning" v-if="same">Choose two different codes.</Callout>
      <div class="col gap-8"><div class="lbl">Feature for the heat map</div>
        <p class="fp-note" data-ae-nl-pair-feature>{{ featureText || 'The pixel the two codes move most (found from the pictures below).' }}</p>
        <AeNlFeatureEditor mode="single" offer-moved add-label="Use this feature" @add="setFeature"/></div>
    </div></div>
    <Callout kind="critical" v-if="slot && slot.error" data-ae-nl-error>{{ slot.error }}
      <button type="button" class="btn xs" v-if="Pr.feature" style="margin-left:8px" @click="Pr.feature = null">Use the pixel the codes move most</button></Callout>
    <div class="fp-wait muted" v-if="slot && slot.busy && !P">Decoding the grid …</div>
    <template v-if="P && !same">
      <Callout kind="info" v-for="n in P.notes" :key="n" data-ae-nl-note>{{ n }}</Callout>
      <div :class="{ 'fp-dim': slot && slot.busy }" class="col gap-12">
        <div class="fp-cards4"><AeStat label="Interaction index" :value="Number.isFinite(P.index) ? (P.index < 0.005 ? '0.00' : fmt(P.index, 2)) : '–'" :level="level" sub="0 = the two codes act independently" data-ae-nl-index
                                help="The part of the way the picture changes over the grid that is not the sum of what each code does alone, as a share of the whole change (measured the way the network was trained)."/>
          <AeStat label="Codes" :value="pairNames[0] + ' and ' + pairNames[1]" :sub="'grid of ' + P.grid + ' x ' + P.grid + ', ± ' + P.span + ' SD'"/></div>
        <p class="fp-note" data-ae-nl-pair-words>{{ P.words }}</p>
        <div v-if="gallery" data-ae-nl-gallery><AeStripCard :title="pairNames[0] + ' across, ' + pairNames[1] + ' up (positions in SD of each code)'" :images="gallery.imgs" :labels="gallery.labels" :ex="gallery.ex" :em="gallery.em" :mask="gallery.mask" :zrange="gallery.zrange"
            :units="gallery.units" :cols="P.grid" :nticks="P.grid >= 7 ? 2 : 3" :row-height="narrow ? Math.round(300 / P.grid) + 20 : Math.round(560 / P.grid) + 20" :highlight="gallery.centre" filename="autoencoder-pair-gallery"
            help="The decoded EEM for each pair of positions of the two codes. The middle picture (in bold) is the decoding of the average code. The same colour scale is used for all of them."/>
          <p class="fp-note muted" data-ae-nl-gallery-note>The two numbers above each picture are the positions of {{ pairNames[0] }} (across) and of {{ pairNames[1] }} (up), in standard deviations of each code from its average; 0, 0 is the average code.</p></div>
        <AeNlCurveGrid v-if="P.ab" :x="P.ab.x" :series="P.ab.series" :grid="P.grid" :u="P.u" :names="pairNames" :title="'Absorbance on the grid of ' + pairNames[0] + ' and ' + pairNames[1]" filename="autoencoder-pair-absorbance"/>
        <div v-if="interaction" data-ae-nl-interaction><AeStripCard :title="'Where the effect is not the sum of the two separate effects'" :images="[interaction.img]" :labels="[]" :ex="gallery.ex" :em="gallery.em" :mask="gallery.mask" :zrange="interaction.zrange"
            :units="gallery.units" :cols="1" :row-height="220" filename="autoencoder-pair-interaction"
            help="At each pixel, the largest (in absolute value) difference over the grid between the picture and the sum of the effects of the two codes taken one at a time. Low on the colour bar = the two codes add up there; high = they interact."/></div>
        <ChartCard v-if="P.ab" title="Interaction of the two codes on the absorbance" :table="{ columns: ['wavelength (nm)', 'largest interaction'], rows: Array.from(P.ab.x, (w, i) => [w, P.ab.interaction[i]]) }" filename="autoencoder-pair-interaction-absorbance"
                   help="The largest difference over the grid between the absorbance and the sum of the effects of the two codes taken one at a time, at each wavelength."><PlotlyChart :data="[{ type: 'scatter', mode: 'lines', x: Array.from(P.ab.x), y: Array.from(P.ab.interaction, (v) => (Number.isFinite(v) ? v : null)), line: { width: 2.2 }, name: 'interaction' }]"
                   :layout="{ height: 200, margin: { l: 62, r: 14, t: 8, b: 46 }, xaxis: { title: { text: 'Wavelength (nm)' } }, yaxis: { title: { text: 'Largest interaction' }, rangemode: 'tozero' } }" :height="200"/></ChartCard>
        <ChartCard :title="featureText ? 'Feature over the two codes: ' + P.feature.short : 'Feature over the two codes'" :table="heatTable" filename="autoencoder-pair-heatmap"
                   help="The chosen feature of the decoded picture at every position of the two codes (colour), with your own samples as dots at the positions of their codes. If the dots are coloured, the colour is the measured value of the feature for that sample: dots that match the background colour say that the map agrees with the data.">
          <PlotlyChart :data="heat.data" :layout="heat.layout" :height="400" data-ae-nl-heat/>
          <p class="fp-note muted" style="padding:0 14px 10px">The dots are your samples{{ heat.colored ? ', coloured by the value of the feature measured on each sample' : '' }}.{{ heat.outside ? ' ' + heat.outside + ' sample' + (heat.outside === 1 ? ' lies' : 's lie') + ' beyond the range shown.' : '' }}</p></ChartCard>
      </div>
    </template>
    </template>
  </div>`,
};
