// Step 7, tab "Map": the grid of units coloured by a plane, the samples as dots at their position on the map, borders between groups, the card of the sample under the pointer.
import { ref, shallowRef, computed, watch, onMounted } from 'vue';
import { Icon } from '/static/js/icons.js';
import { Field, Switch, Callout, Seg } from '/static/js/components/Common.js';
import { ChartCard } from '/static/js/plot.js';
import { VarSelect } from '/static/js/components/VarSelect.js';
import { EEMThumb } from '/static/js/components/EEMThumb.js';
import { catalogue, loadCatalogue, fetchValues, labelOf, kindOf } from '/static/js/variables.js';
import { valuesForIds } from '/static/js/components/DecompCommon.js';
import { tokens, themeState, lutFor, sequentialScale, divergingScale, categoryStyle } from '/static/js/theme.js';
import { prefs } from '/static/js/prefs.js';
import { fmt } from '/static/js/util.js';
import { fp } from './SomState.js';
import { loadGallery, loadSample } from './SomActions.js';
import { SomGrid } from './SomGrid.js';
import { SomUnitCard } from './SomPrototypes.js';
import { numericColours, groupColour, pct, STATUS_LABEL } from './SomCommon.js';

/** The planes that can colour the map: the stored ones and the ones the person added in the tab Planes. */
export function planeOptions() {
  const R = fp.som.res; if (!R) return [];
  const out = Object.entries(R.planes).map(([key, p]) => ({ key, label: p.label, plane: p }));
  for (const p of fp.som.planesAdded) if (!out.some((o) => o.key === p.key)) out.push({ key: p.key, label: p.label, plane: p });
  return out;
}
/** A deterministic small shift so that samples on the same unit do not hide each other. */
const jitter = (i, k) => { const x = Math.sin((i + 1) * 12.9898 + k * 78.233) * 43758.5453; return (x - Math.floor(x) - 0.5) * 0.5; };

export const SomMapTab = {
  name: 'SomMapTab',
  components: { Icon, Field, Switch, Callout, Seg, ChartCard, VarSelect, EEMThumb, SomGrid, SomUnitCard },
  setup() {
    const A = computed(() => fp.som), R = computed(() => fp.som.res), V = computed(() => fp.som.view);
    const colorVals = shallowRef(null), hoverId = ref(null), hoverUnit = ref(null);
    onMounted(() => { loadCatalogue(); loadGallery(); });
    const options = computed(() => planeOptions());
    const plane = computed(() => { const o = options.value.find((x) => x.key === V.value.plane); return o ? o.plane : (options.value[0] ? options.value[0].plane : null); });
    watch(options, (o) => { if (o.length && !o.some((x) => x.key === V.value.plane)) V.value.plane = o[0].key; }, { immediate: true });
    const ids = computed(() => (R.value ? R.value.ids : []));
    const keys = computed(() => [V.value.color, V.value.symbol].filter(Boolean));
    watch([keys, () => ids.value.length], async () => {
      if (!keys.value.length) { colorVals.value = null; return; }
      try { colorVals.value = await fetchValues(keys.value); } catch { colorVals.value = null; }
    }, { immediate: true });
    watch(() => catalogue.value.length, () => { if (keys.value.length) fetchValues(keys.value).then((v) => { colorVals.value = v; }).catch(() => {}); });
    const statusKey = computed(() => (A.value.resRid ? `res:${A.value.resRid}:status` : null));
    const mapProps = computed(() => { const r = R.value; return r ? { M: r.M, rows: r.rows, cols: r.cols, lattice: r.lattice, shape: r.shape, coords: r.coords, edges: r.edges } : null; });
    const dots = computed(() => {
      themeState.version;
      const r = R.value; if (!r) return null;
      const t = tokens(), n = r.n, x = [], y = [], text = [], colors = [], symbols = [];
      const cv = V.value.color && colorVals.value ? valuesForIds(colorVals.value, r.ids, V.value.color) : null;
      const sv = V.value.symbol && colorVals.value ? valuesForIds(colorVals.value, r.ids, V.value.symbol) : null;
      let cinfo = null;
      if (cv && kindOf(V.value.color) === 'numeric') cinfo = numericColours(cv.map((v) => (v === null ? NaN : v)), false);
      for (let i = 0; i < n; i++) {
        const u = r.unit[i];
        if (V.value.place === 'unit') { x.push(r.coords.data[u * 2] + jitter(i, 1)); y.push(r.coords.data[u * 2 + 1] + jitter(i, 2)); }
        else { x.push(r.pos.data[i * 2]); y.push(r.pos.data[i * 2 + 1]); }
        let c = t.text;
        if (cv) c = cinfo ? (cinfo.rgb(cv[i] === null ? NaN : cv[i]) || t.other) : categoryStyle(V.value.color, cv[i]).color;
        colors.push(c);
        symbols.push(sv ? categoryStyle(V.value.symbol, sv[i]).symbol : 'circle');
        text.push(`${r.ids[i]}<br>unit ${r.unitNames[u]}<br>${STATUS_LABEL[r.status[i]] || r.status[i]}${cv && cv[i] !== null ? `<br>${labelOf(V.value.color)}: ${typeof cv[i] === 'number' ? fmt(cv[i], 4) : cv[i]}` : ''}`);
      }
      return { x, y, ids: r.ids.slice(), colors, symbols, text, name: 'Samples' };
    });
    const hoverTexts = computed(() => { const r = R.value; return r ? Array.from({ length: r.M }, (_, j) => `Unit ${r.unitNames[j]}<br>${r.hits[j]} sample${r.hits[j] === 1 ? '' : 's'}${plane.value ? `<br>${plane.value.label}: ${plane.value.kind === 'category' ? ((plane.value.categories || [])[plane.value.v[j]] || '–') : (Number.isFinite(plane.value.v[j]) ? fmt(plane.value.v[j], 4) : '–')}` : ''}`) : []; });
    const mapHeight = computed(() => { const r = R.value; if (!r) return 420; const rowsH = r.rows * (r.lattice === 'hexagonal' ? 0.866 : 1) + 1.4; return Math.round(Math.min(560, Math.max(260, 66 * rowsH + 50))); });
    const groups = computed(() => (V.value.borders && R.value && R.value.clusters.k > 1 ? R.value.clusters.unitLabels : null));
    const shownId = computed(() => hoverId.value || V.value.sample);
    watch(shownId, (id) => { if (id) loadSample(id, true); }, { immediate: true });
    const facts = computed(() => { const id = shownId.value, rid = A.value.resRid; const s = id && rid && A.value.extra.samples ? A.value.extra.samples[`${rid}|${id}|t`] : null; return s && s.data ? s.data : null; });
    const gallery = computed(() => (A.value.gallery && A.value.gallery.data) || null);
    const thumb = computed(() => {
      themeState.version;
      const g = gallery.value, id = shownId.value; if (!g || !id || !g.index.has(id)) return null;
      const i = g.index.get(id), nex = g.z.shape[1], nem = g.z.shape[2];
      return { z: g.z, nex, nem, offset: i * nex * nem, vmin: g.vmin, vmax: g.vmax, lut: lutFor(g.kind === 'diverging' ? divergingScale(prefs.div) : sequentialScale(prefs.seq)) };
    });
    function pickUnit(j) { V.value.unit = j; }
    function pickDot(id) { V.value.sample = id; }
    function openSample() { if (V.value.sample) A.value.tab7 = 'samples'; }
    const planeKind = computed(() => (plane.value ? plane.value.kind : 'numeric'));
    const legend = computed(() => (plane.value && plane.value.kind === 'category' ? (plane.value.categories || []).map((c, i) => ({ c, color: tokens().series[i % 8] })) : []));
    const groupLegend = computed(() => (groups.value ? Array.from({ length: R.value.clusters.k }, (_, g) => ({ g, color: groupColour(g), n: R.value.clusters.sizes[g] })) : []));
    return { A, R, V, options, plane, mapProps, mapHeight, dots, hoverTexts, groups, shownId, facts, thumb, hoverId, hoverUnit, pickUnit, pickDot, openSample, planeKind, legend, groupLegend, STATUS_LABEL, pct, fmt, planeOptions };
  },
  template: `
  <div class="col gap-12" data-som-map-tab>
    <div class="card"><div class="card-body" style="padding:12px 16px"><div class="row wrap" style="gap:12px 22px;align-items:flex-end">
      <Field label="Colour the units by" style="width:300px" help="U-matrix: ridges where the prototypes differ a lot. Hits: how many samples fall on each unit. Principal directions: where each unit lies along the main directions of variation. More planes (pixels, indices, metadata) can be added in the tab Planes.">
        <select class="select sm" v-model="V.plane" data-som-plane><option v-for="o in options" :key="o.key" :value="o.key">{{ o.label }}</option></select></Field>
      <Field label="Place the samples" style="width:240px"><Seg v-model="V.place" :options="[{ value: 'position', label: 'At their position' }, { value: 'unit', label: 'On their unit' }]" small data-som-place/></Field>
      <Field label="Colour by" style="width:230px" help="Colour the sample dots by any metadata column, extracted feature or result of another model."><VarSelect v-model="V.color" small none-label="Nothing (one colour)"/></Field>
      <Field label="Symbol by" style="width:210px"><VarSelect v-model="V.symbol" small :kinds="['category']" none-label="Nothing"/></Field>
      <div class="col" style="gap:6px;padding-bottom:4px"><Switch v-model="V.borders" label="Draw the borders between groups"/></div>
    </div></div></div>
    <div class="som-map-grid">
      <div class="som-map-plot" data-som-map-chart>
        <ChartCard :title="plane ? plane.label : 'The map'" :table="{ columns: ['unit', 'samples', plane ? plane.label : 'value'], rows: R ? Array.from({ length: R.M }, (_, j) => [R.unitNames[j], R.hits[j], plane && plane.kind === 'numeric' ? plane.v[j] : null]) : [] }" filename="som-map"
                   :help="plane ? plane.words : ''">
          <SomGrid v-if="mapProps" :map="mapProps" :plane="plane" :dots="dots" :groups="groups" :selected="V.unit" :hover-texts="hoverTexts" :height="mapHeight" @unit="pickUnit" @dot="pickDot" @hover-dot="hoverId = $event"/>
          <template #legend>
            <div class="som-legend row wrap" v-if="legend.length || groupLegend.length" style="padding:6px 14px">
              <span v-for="l in legend" :key="l.c" class="som-key"><i :style="{ background: l.color }"></i>{{ l.c }}</span>
              <span v-if="groupLegend.length" class="muted xs">Thick lines separate the groups ({{ groupLegend.length }}).</span>
            </div>
          </template>
        </ChartCard>
        <p class="fp-note">The coloured hexagons are the units of the map. The dots are the samples: with <b>At their position</b> a sample sits between the units it resembles. Click a unit to open its card, or a dot to select the sample.</p>
      </div>
      <div class="card som-hover" data-som-hover :data-id="shownId || ''" aria-live="polite">
        <div class="card-head"><h3>{{ shownId ? shownId : 'Hover over a dot' }}</h3></div>
        <div class="card-body col gap-8">
          <p class="fp-note" v-if="!shownId">Move the pointer over a dot to see the sample, and click it to select it.</p>
          <template v-else>
            <div class="som-thumb" v-if="thumb"><EEMThumb :z="thumb.z" :nex="thumb.nex" :nem="thumb.nem" :offset="thumb.offset" :vmin="thumb.vmin" :vmax="thumb.vmax" :lut="thumb.lut" style="width:100%;height:auto"/></div>
            <dl class="fp-facts" v-if="facts"><dt>Unit</dt><dd>{{ facts.unit_name }}</dd><dt>Status</dt><dd :data-status="facts.status">{{ STATUS_LABEL[facts.status] || facts.status }}</dd><dt>Not captured</dt><dd>{{ pct(facts.qe_rel, 1) }} of the variation</dd></dl>
            <p class="fp-note muted" v-if="V.sample === shownId">Selected.</p>
            <button type="button" class="btn xs" v-if="V.sample === shownId" data-som-open-sample @click="openSample"><Icon name="arrow-right" size="14"/>Open in Samples</button>
          </template>
        </div>
      </div>
    </div>
    <SomUnitCard v-if="V.unit !== null && V.unit !== undefined" :index="V.unit" @sample="pickDot($event); openSample()"/>
  </div>`,
};
