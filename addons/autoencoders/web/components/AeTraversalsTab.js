// Step 7, tab "Traversals": what every code does.  Each code is moved from -3 to +3 standard deviations (of the codes of the training samples) while the others stay at their mean,
// and the network decodes every position: one strip of EEMs per code (and the absorbance curves), next to "Sensitivity of the output to each code".
import { ref, computed, watch, onMounted } from 'vue';
import { Icon } from '/static/js/icons.js';
import { Seg, Callout } from '/static/js/components/Common.js';
import { PlotlyChart, ChartCard } from '/static/js/plot.js';
import { vec, stack, sliceImage, maskArr } from '/static/js/components/DecompCommon.js';
import { tokens, themeState, lutFor, sequentialScale } from '/static/js/theme.js';
import { prefs } from '/static/js/prefs.js';
import { fmt } from '/static/js/util.js';
import { fp } from './AeState.js';
import { loadTraverse, loadSensitivity } from './AeActions.js';
import { AeStripCard, AeSection, useNarrow, sdText, shortUnits, unitsOf, pct } from './AeCommon.js';

const SHOWN = 8;                                                      // codes drawn at first; the rest on request

/** Several curves of one quantity along the positions of a code (one colour per position, the mean position drawn thick). */
export const AeCurveSet = {
  name: 'AeCurveSet',
  components: { PlotlyChart, ChartCard },
  props: { title: String, x: { required: true }, series: { type: Array, required: true }, labels: { type: Array, required: true }, units: { type: String, default: '' }, yTitle: { type: String, default: '' }, mid: { type: Number, default: -1 }, signed: Boolean, filename: String, height: { type: Number, default: 250 } },
  setup(props) {
    const traces = computed(() => {
      themeState.version;
      const t = tokens(), lut = lutFor(sequentialScale(prefs.seq)), k = props.series.length;
      return props.series.map((y, i) => {
        const q = Math.round((k === 1 ? 0.5 : i / (k - 1)) * 255) * 3, isMid = i === props.mid;
        return { type: 'scatter', mode: 'lines', x: Array.from(props.x), y: Array.from(y, (v) => (Number.isFinite(v) ? v : null)), name: props.labels[i], line: { color: isMid ? t.text : `rgb(${lut[q]},${lut[q + 1]},${lut[q + 2]})`, width: isMid ? 2.6 : 1.8 },
          hovertemplate: `${props.labels[i]}<br>%{x:.0f} nm: %{y:.4g}<extra></extra>` };
      });
    });
    const layout = computed(() => ({ height: props.height, margin: { l: 58, r: 14, t: 10, b: 74 }, showlegend: true, legend: { orientation: 'h', y: -0.32, x: 0, font: { size: 10.5 } }, hovermode: 'x unified', xaxis: { title: { text: 'Wavelength (nm)' } }, yaxis: { title: { text: props.yTitle || props.units }, zeroline: props.signed } }));
    const table = computed(() => ({ columns: ['wavelength (nm)', ...props.labels], rows: Array.from(props.x, (w, j) => [w, ...props.series.map((s) => s[j])]) }));
    return { traces, layout, table };
  },
  template: `<ChartCard :title="title" :table="table" :filename="filename"><PlotlyChart :data="traces" :layout="layout" :height="height"/></ChartCard>`,
};

export const AeTraversalsTab = {
  name: 'AeTraversalsTab',
  components: { Icon, Seg, Callout, PlotlyChart, ChartCard, AeStripCard, AeSection, AeCurveSet },
  setup() {
    const A = computed(() => fp.ae), R = computed(() => fp.ae.res);
    const T = computed(() => A.value.extra.trav || null), S = computed(() => A.value.extra.sens || null);
    const state = computed(() => A.value.travView);
    const narrow = useNarrow();
    const load = () => { loadTraverse(state.value.range); loadSensitivity(); };
    onMounted(load);
    watch(() => [state.value.range, A.value.resRid], load);
    const data = computed(() => (T.value ? T.value.data : null));
    const values = computed(() => (data.value ? data.value.values : []));
    const mid = computed(() => values.value.indexOf(0));
    const ex = computed(() => (data.value ? vec(data.value.ex) : null)), em = computed(() => (data.value ? vec(data.value.em) : null));
    const mask = computed(() => (data.value && data.value.mask ? maskArr(data.value.mask) : null));
    const units = computed(() => (data.value ? unitsOf(data.value.units) : ''));
    const hasFl = computed(() => !!(data.value && data.value.codes.length && data.value.codes[0].fluorescence));
    const hasAb = computed(() => !!(data.value && data.value.codes.length && data.value.codes[0].absorbance));
    const range = computed(() => {
      const d = data.value; if (!d || !hasFl.value) return null;
      if (state.value.show === 'difference') { const m = Math.max(...d.codes.map((c) => c.fluorescence.difference_abs_max || 0)) || 1; return [-m, m]; }
      return [Math.min(...d.codes.map((c) => c.fluorescence.min)), Math.max(...d.codes.map((c) => c.fluorescence.max))];
    });
    const strips = computed(() => {
      const d = data.value; if (!d || !hasFl.value) return [];
      return d.codes.map((c, j) => {
        const st = stack(state.value.show === 'difference' ? c.fluorescence.difference : c.fluorescence.images);
        const imgs = values.value.map((_, i) => sliceImage(st, i));
        const labels = values.value.map((v) => sdText(v));
        return { j, name: c.name, sd: c.sd, imgs, labels };
      });
    });
    const shown = computed(() => (state.value.all ? strips.value : strips.value.slice(0, SHOWN)));
    const curves = computed(() => {
      const d = data.value; if (!d || !hasAb.value) return [];
      return d.codes.map((c) => ({ name: c.name, x: vec(d.codes[0].absorbance.x), series: (state.value.show === 'difference' ? c.absorbance.difference : c.absorbance.series).map((s) => vec(s)) }));
    });
    const cols = computed(() => (narrow.value ? Math.min(4, values.value.length || 4) : values.value.length || 7));
    const labels = computed(() => values.value.map((v) => sdText(v)));
    const sensData = computed(() => (S.value ? S.value.data : null));
    const sens = computed(() => {
      const s = sensData.value; if (!s || !s.fluorescence) return null;
      const st = stack(s.fluorescence.at_mean), mag = stack(s.fluorescence.magnitude);
      const m = Math.max(...Array.from(st).map((v) => (Number.isFinite(v) ? Math.abs(v) : 0))) || 1;
      return { imgs: s.names.map((_, i) => sliceImage(st, i)), mag: s.names.map((_, i) => sliceImage(mag, i)), labels: s.names.map((n, i) => `${n}${s.fluorescence.share ? ` · ${pct(s.fluorescence.share[i], 0)}` : ''}`), range: [-m, m], ex: vec(s.ex), em: vec(s.em), mask: s.mask ? maskArr(s.mask) : null };
    });
    const shareTraces = computed(() => {
      themeState.version;
      const s = sensData.value; if (!s || !s.fluorescence || !s.fluorescence.share) return [];
      const t = tokens();
      return [{ type: 'bar', x: s.names, y: s.fluorescence.share, marker: { color: t.series[0], line: { color: t.surface, width: 1 } }, hovertemplate: '%{x}: %{y:.1%} of the total effect<extra></extra>', name: 'share' }];
    });
    const shareLayout = computed(() => ({ height: 220, margin: { l: 56, r: 14, t: 10, b: 44 }, xaxis: { title: { text: 'Code' } }, yaxis: { title: { text: 'Share of the total effect' }, tickformat: '.0%', rangemode: 'tozero' } }));
    const shareTable = computed(() => { const s = sensData.value; return s && s.fluorescence && s.fluorescence.share ? { columns: ['code', 'share of the total effect'], rows: s.names.map((n, i) => [n, s.fluorescence.share[i]]) } : { columns: [], rows: [] }; });
    return { A, R, T, S, state, data, values, mid, ex, em, mask, units, hasFl, hasAb, range, shown, strips, curves, cols, labels, sens, sensData, shareTraces, shareLayout, shareTable, shortUnits, SHOWN, narrow, fmt };
  },
  template: `
  <div class="col gap-12" data-ae-traversals-tab>
    <div class="card"><div class="card-body" style="padding:12px 16px"><div class="row wrap" style="gap:12px 26px;align-items:center">
      <div class="col" style="gap:4px"><span class="lbl">Range</span><Seg v-model="state.range" :options="[{ value: 1, label: '±1 SD' }, { value: 2, label: '±2 SD' }, { value: 3, label: '±3 SD' }]" data-ae-trav-range/></div>
      <div class="col" style="gap:4px"><span class="lbl">Show</span><Seg v-model="state.show" :options="[{ value: 'images', label: 'Images' }, { value: 'difference', label: 'Difference to the mean' }]" data-ae-trav-show/></div>
      <span class="muted xs ae-grow">SD = standard deviation of the code over the training samples. The picture in the middle is the network's decoding of the mean code; all the other codes stay at their mean.</span>
    </div></div></div>
    <Callout kind="critical" v-if="T && T.error" data-ae-trav-error>{{ T.error }}</Callout>
    <div class="fp-wait muted" v-if="T && T.busy && !data">Decoding the codes …</div>
    <div :class="{ 'fp-dim': T && T.busy }" v-if="data" class="col gap-12">
      <template v-if="hasFl">
        <div class="ae-trav" v-for="s in shown" :key="s.name" :data-ae-traversal="s.j + 1">
          <AeStripCard :title="s.name + (state.show === 'difference' ? ': difference to the mean' : ': from ' + labels[0] + ' to ' + labels[labels.length - 1])" :images="s.imgs" :labels="s.labels" :ex="ex" :em="em" :mask="mask"
                       :kind="state.show === 'difference' ? 'diverging' : 'sequential'" :zrange="range" :units="shortUnits(units)" :cols="cols" :row-height="narrow ? 120 : 150" :highlight="mid" :filename="'autoencoder-traversal-' + s.name"
                       help="The decoded EEM for each position of this one code. The grey pixels are not part of the model. The same colour scale is used for every code."/>
        </div>
        <button type="button" class="btn sm" v-if="strips.length > SHOWN && !state.all" data-ae-trav-all @click="state.all = true">Show the other {{ strips.length - SHOWN }} codes</button>
      </template>
      <template v-if="hasAb">
        <div class="fp-grid2"><AeCurveSet v-for="c in curves.slice(0, state.all ? 99 : SHOWN)" :key="c.name" :title="c.name + ': absorbance'" :x="c.x" :series="c.series" :labels="labels" :mid="mid" :signed="state.show === 'difference'" y-title="Absorbance" :filename="'autoencoder-traversal-absorbance-' + c.name"/></div>
      </template>
    </div>

    <AeSection title="Sensitivity of the output to each code" sentence="What a small change of each code does to the EEM, at the mean code (blue: the intensity goes up; red: it goes down). The percentages are the share of the total effect of all the codes.">
      <Callout kind="critical" v-if="S && S.error">{{ S.error }}</Callout>
      <div class="fp-wait muted" v-if="S && S.busy && !sensData">Computing the sensitivity …</div>
      <div v-if="sensData" data-ae-sensitivity class="col gap-12">
        <p class="fp-note" v-if="sensData.words">{{ sensData.words }}</p>
        <AeStripCard v-if="sens" title="Change of the EEM per 1 SD of each code" :images="sens.imgs" :labels="sens.labels" :ex="sens.ex" :em="sens.em" :mask="sens.mask" kind="diverging" :zrange="sens.range" :units="shortUnits(sensData.units)"
                     :cols="Math.min(sens.imgs.length, 4)" :row-height="170" filename="autoencoder-sensitivity" help="The derivative of the rebuilt EEM with respect to each code, at the mean code, for a change of one standard deviation of the code: the nonlinear counterpart of a loading."/>
        <ChartCard v-if="shareTraces.length" title="Share of the total effect of each code" :table="shareTable" filename="autoencoder-sensitivity-share"><PlotlyChart :data="shareTraces" :layout="shareLayout" :height="220"/></ChartCard>
      </div>
    </AeSection>
  </div>`,
};
