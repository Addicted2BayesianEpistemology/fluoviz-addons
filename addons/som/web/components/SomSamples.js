// Step 7, tab "Samples": the table of the samples (the worst described first), and for the chosen sample its measured picture, the prototype of its unit and the difference; the
// card "Unusual samples" with its threshold methods.
import { ref, computed, watch } from 'vue';
import { Icon } from '/static/js/icons.js';
import { Field, NumField, Callout, Seg } from '/static/js/components/Common.js';
import { PlotlyChart, ChartCard } from '/static/js/plot.js';
import { EEMView } from '/static/js/components/EEMView.js';
import { vec, image, maskArr, sharedRange } from '/static/js/components/DecompCommon.js';
import { tokens, themeState } from '/static/js/theme.js';
import { fmt } from '/static/js/util.js';
import { fp } from './SomState.js';
import { loadSample, runAnomalies } from './SomActions.js';
import { pct, STATUS_LABEL, SomVerdict } from './SomCommon.js';

export const SomAnomalies = {
  name: 'SomAnomalies',
  components: { Icon, Field, NumField, Callout, Seg, PlotlyChart, ChartCard },
  emits: ['sample'],
  setup(props, { emit }) {
    const A = computed(() => fp.som), V = computed(() => fp.som.anomView);
    const slot = computed(() => (A.value.extra.anomKey ? A.value.extra[A.value.extra.anomKey] : null));
    const d = computed(() => (slot.value ? slot.value.data : null));
    const run = () => runAnomalies({ method: V.value.method, ...(V.value.method === 'mad' ? { c: Number(V.value.c) } : V.value.method === 'quantile' ? { q: Number(V.value.q) } : { level: Number(fp.spec.map.limit) }) });
    const traces = computed(() => {
      themeState.version;
      const h = d.value && d.value.histogram; if (!h) return [];
      const counts = Array.from(vec(h.counts)), edges = Array.from(vec(h.edges)), t = tokens();
      return [{ type: 'bar', x: counts.map((_, i) => (edges[i] + edges[i + 1]) / 2), y: counts, width: counts.map((_, i) => edges[i + 1] - edges[i]), marker: { color: t.series[0] }, hovertemplate: '%{x:.3g}: %{y} samples<extra></extra>' }];
    });
    const layout = computed(() => {
      const lim = d.value ? d.value.limit : null;
      return { height: 230, margin: { l: 46, r: 14, t: 10, b: 50 }, xaxis: { title: { text: 'Distance to the best unit' } }, yaxis: { title: { text: 'Samples' } }, bargap: 0.04,
        shapes: Number.isFinite(lim) ? [{ type: 'line', x0: lim, x1: lim, yref: 'paper', y0: 0, y1: 1, line: { color: tokens().critical, width: 2, dash: 'dot' } }] : [] };
    });
    return { A, V, slot, d, run, traces, layout, emit, fmt };
  },
  template: `
  <div class="card" data-som-anomalies><div class="card-head"><h3>Unusual samples</h3></div><div class="card-body col gap-12">
    <p class="fp-note">Which samples does the map describe worst? The distance of a sample to its best unit is compared with a limit. When the map was checked on held-out samples, their out-of-fold distances are used (they are honest); otherwise the distances of the training samples.</p>
    <div class="row wrap" style="gap:12px 22px;align-items:flex-end">
      <Field label="Method" style="width:300px"><select class="select sm" v-model="V.method" data-som-anom-method><option value="mad">Median plus c × MAD</option><option value="quantile">Quantile of the distances</option><option value="limit">Limit of the saved model</option></select></Field>
      <Field v-if="V.method === 'mad'" label="c" style="width:100px"><NumField v-model="V.c" :min="0.5" :max="20" :step="0.5" small/></Field>
      <Field v-if="V.method === 'quantile'" label="Quantile" style="width:120px"><NumField v-model="V.q" :min="0.5" :max="0.999" :step="0.01" small/></Field>
      <button type="button" class="btn" data-som-anom-go :disabled="slot && slot.busy" @click="run"><Icon name="search-alt" size="16"/>Find them</button>
    </div>
    <Callout kind="critical" v-if="slot && slot.error">{{ slot.error }}</Callout>
    <template v-if="d">
      <p class="fp-note">{{ d.words }} The limit is {{ fmt(d.limit, 3) }}; {{ d.flagged.length }} sample{{ d.flagged.length === 1 ? '' : 's' }} lie{{ d.flagged.length === 1 ? 's' : '' }} above it.</p>
      <ChartCard title="Distances of the samples to their best unit" :data="traces" filename="som-distances"><PlotlyChart :data="traces" :layout="layout" :height="230"/></ChartCard>
      <div class="row wrap" v-if="d.flagged.length"><button type="button" class="chip som-member" v-for="f in d.flagged" :key="f.id" :title="'Unit ' + f.unit + ', distance ' + fmt(f.qe, 3)" @click="emit('sample', f.id)">{{ f.id }}</button></div>
    </template>
  </div></div>`,
};

export const SomSamplesTab = {
  name: 'SomSamplesTab',
  components: { Icon, Field, Callout, Seg, PlotlyChart, ChartCard, EEMView, SomAnomalies, SomVerdict },
  setup() {
    const A = computed(() => fp.som), R = computed(() => fp.som.res), V = computed(() => fp.som.view);
    const rows = computed(() => {
      const r = R.value; if (!r) return [];
      return r.ids.map((id, i) => ({ id, i, unit: r.unitNames[r.unit[i]], qe: r.qe[i], held: r.held[i], rel: r.qeRel[i], status: r.status[i], flag: r.flag[i], missing: r.missing ? r.missing[i] : 0, group: r.clusters.sampleLabels[i] }));
    });
    const counts = computed(() => ({ all: rows.value.length, bad: rows.value.filter((x) => x.status !== 'ok').length, lack: rows.value.filter((x) => x.missing > 0.005).length }));
    const shown = computed(() => {
      let l = rows.value;
      if (V.value.filter === 'bad') l = l.filter((x) => x.status !== 'ok'); else if (V.value.filter === 'lack') l = l.filter((x) => x.missing > 0.005);
      const f = String(V.value.find || '').trim().toLowerCase(); if (f) l = l.filter((x) => x.id.toLowerCase().includes(f));
      return l.slice().sort((a, b) => (Number.isFinite(b.qe) ? b.qe : -1) - (Number.isFinite(a.qe) ? a.qe : -1));
    });
    watch(() => V.value.sample, (id) => { if (id) loadSample(id); }, { immediate: true });
    const slot = computed(() => { const id = V.value.sample, rid = A.value.resRid; return id && rid && A.value.extra.samples ? A.value.extra.samples[`${rid}|${id}|f`] : null; });
    const d = computed(() => (slot.value ? slot.value.data : null));
    const blocks = computed(() => (d.value && d.value.blocks ? d.value.blocks.map((b) => ({ ...b, surf: b.kind === 'surface', m: b.kind === 'surface' ? image(b.measured) : vec(b.measured), p: b.kind === 'surface' ? image(b.prototype) : vec(b.prototype),
      df: b.difference ? (b.kind === 'surface' ? image(b.difference) : vec(b.difference)) : null, x: b.x ? vec(b.x) : null, ex: b.ex ? vec(b.ex) : null, em: b.em ? vec(b.em) : null, mask: b.mask ? maskArr(b.mask) : null })) : []));
    const rangeOf = (b) => sharedRange([b.m, b.p], 'sequential');
    const curveTraces = (b) => { themeState.version; const t = tokens(); return [{ type: 'scatter', mode: 'lines', x: Array.from(b.x), y: Array.from(b.m, (v) => (Number.isFinite(v) ? v : null)), name: 'Measured', line: { color: t.series[0], width: 2 } },
      { type: 'scatter', mode: 'lines', x: Array.from(b.x), y: Array.from(b.p, (v) => (Number.isFinite(v) ? v : null)), name: 'Prototype of the unit', line: { color: t.series[1], width: 2, dash: 'dash' } }]; };
    const curveLayout = { height: 240, margin: { l: 54, r: 14, t: 10, b: 62 }, showlegend: true, legend: { orientation: 'h', y: -0.3, x: 0 }, xaxis: { title: { text: 'Wavelength (nm)' } } };
    function open(id) { V.value.sample = id; }
    const sample = computed(() => V.value.sample);
    return { A, R, V, rows, counts, shown, slot, d, blocks, rangeOf, curveTraces, curveLayout, open, sample, STATUS_LABEL, pct, fmt };
  },
  template: `
  <div class="col gap-12" data-som-samples-tab>
    <div class="som-samples-grid">
      <div class="card"><div class="card-head"><h3>The samples</h3></div><div class="card-body col gap-8">
        <div class="row wrap"><Seg v-model="V.filter" :options="[{ value: 'all', label: 'All (' + counts.all + ')' }, { value: 'bad', label: 'Poorly described (' + counts.bad + ')' }, { value: 'lack', label: 'Lack part of the domain (' + counts.lack + ')' }]" small data-som-sample-filter/>
          <input class="input sm" style="max-width:180px" v-model="V.find" placeholder="Find a sample" aria-label="Find a sample" data-som-sample-find/></div>
        <div class="table-wrap" style="max-height:420px"><table class="table"><thead><tr><th>Sample</th><th>Unit</th><th>Distance</th><th>Held out</th><th>Status</th></tr></thead>
          <tbody><tr v-for="x in shown" :key="x.id" :class="{ sel: x.id === sample }" style="cursor:pointer" :data-som-sample-row="x.id" @click="open(x.id)"><td>{{ x.id }}<span class="muted xs" v-if="x.missing > 0.005" data-som-sample-missing-share><br>{{ Math.round(x.missing * 100) }} % missing</span></td><td>{{ x.unit }}</td>
            <td class="num">{{ fmt(x.qe, 3) }}</td><td class="num">{{ fmt(x.held, 3) }}</td><td>{{ STATUS_LABEL[x.status] || x.status }}</td></tr></tbody></table></div>
      </div></div>
      <div class="card" data-som-sample-card><div class="card-head"><h3>{{ sample || 'Choose a sample' }}</h3></div><div class="card-body col gap-12">
        <p class="fp-note" v-if="!sample">Click a sample in the table (or a dot on the map) to see it against the prototype of its unit.</p>
        <p class="muted small" v-if="slot && slot.busy">Reading the sample …</p>
        <Callout kind="critical" v-if="slot && slot.error" data-som-sample-error>{{ slot.error }}</Callout>
        <template v-if="d">
          <SomVerdict :level="d.status === 'ok' ? 'good' : d.status === 'uncertain' ? 'info' : 'warn'" data-som-sample-text>{{ d.text }}</SomVerdict>
          <p class="fp-note" v-if="d.missing_words" data-som-sample-missing>{{ d.missing_words }}</p>
          <template v-for="b in blocks" :key="b.name">
            <div class="som-sample-pics" v-if="b.surf && b.ex">
              <div><div class="eyebrow">Measured</div><EEMView :ex="b.ex" :em="b.em" :z="b.m" :mask="b.mask" :zrange="rangeOf(b)" :compact="true" :height="220" :units="b.units"/></div>
              <div><div class="eyebrow">Prototype of the unit {{ d.unit_name }}</div><EEMView :ex="b.ex" :em="b.em" :z="b.p" :mask="b.mask" :zrange="rangeOf(b)" :compact="true" :height="220"/></div>
              <div v-if="b.df"><div class="eyebrow">Difference</div><EEMView :ex="b.ex" :em="b.em" :z="b.df" :mask="b.mask" kind="diverging" :compact="true" :height="220"/></div>
            </div>
            <ChartCard v-if="!b.surf && b.x" title="Absorbance against the prototype" :filename="'som-sample-' + b.name"><PlotlyChart :data="curveTraces(b)" :layout="curveLayout" :height="240"/></ChartCard>
          </template>
          <div class="table-wrap" v-if="d.variables"><table class="table"><thead><tr><th>Variable</th><th>Sample</th><th>Prototype</th></tr></thead><tbody><tr v-for="v in d.variables" :key="v.name"><td>{{ v.name }}</td><td class="num">{{ fmt(v.value, 4) }}</td><td class="num">{{ fmt(v.prototype, 4) }}</td></tr></tbody></table></div>
        </template>
      </div></div>
    </div>
    <SomAnomalies @sample="open"/>
  </div>`,
};
