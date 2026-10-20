// Step 7, tab "Prototypes": the prototype of every unit as a small picture on the grid of the map, and the card of one unit (its prototype next to the mean of its samples).
import { ref, computed, watch, onMounted } from 'vue';
import { Icon } from '/static/js/icons.js';
import { Seg, Callout, Empty } from '/static/js/components/Common.js';
import { EEMThumb } from '/static/js/components/EEMThumb.js';
import { EEMView } from '/static/js/components/EEMView.js';
import { vec, image, maskArr, sliceImage, sharedRange } from '/static/js/components/DecompCommon.js';
import { tokens, themeState, lutFor, sequentialScale, divergingScale } from '/static/js/theme.js';
import { prefs } from '/static/js/prefs.js';
import { fmt } from '/static/js/util.js';
import { fp } from './SomState.js';
import { loadPrototypes, loadUnit } from './SomActions.js';
import { groupColour, nTxt, pct } from './SomCommon.js';

/** A tiny line chart of one curve (a prototype absorbance spectrum). */
export const SomSpark = {
  name: 'SomSpark',
  props: { y: { required: true }, lo: Number, hi: Number, color: { type: String, default: '' } },
  setup(props) {
    const path = computed(() => {
      const y = props.y, n = y.length, span = (props.hi - props.lo) || 1; let d = '', pen = false;
      for (let i = 0; i < n; i++) {
        const v = y[i];
        if (!(v === v)) { pen = false; continue; }
        d += `${pen ? 'L' : 'M'}${(i / Math.max(1, n - 1) * 100).toFixed(1)},${(30 - ((v - props.lo) / span) * 28 - 1).toFixed(1)}`; pen = true;
      }
      return d;
    });
    return { path, tokens };
  },
  template: `<svg viewBox="0 0 100 30" preserveAspectRatio="none" class="som-spark" aria-hidden="true"><path :d="path" fill="none" :stroke="color || tokens().series[0]" stroke-width="1.6" vector-effect="non-scaling-stroke"/></svg>`,
};

/** The unit card: the prototype and the mean of the samples that fall on the unit, side by side, with the numbers of the unit. */
export const SomUnitCard = {
  name: 'SomUnitCard',
  components: { Icon, Callout, EEMView, SomSpark },
  props: { index: { type: Number, required: true } },
  emits: ['sample'],
  setup(props, { emit }) {
    const slot = computed(() => { const A = fp.som; return A.resRid && A.extra.units ? A.extra.units[`${A.resRid}|${props.index}`] : null; });
    watch(() => props.index, (i) => { if (i !== null && i !== undefined) loadUnit(i); }, { immediate: true });
    const d = computed(() => (slot.value ? slot.value.data : null));
    const R = computed(() => fp.som.res);
    const axes = computed(() => (d.value && d.value.ex ? { ex: vec(d.value.ex), em: vec(d.value.em), mask: d.value.mask ? maskArr(d.value.mask) : null } : null));
    const blocks = computed(() => (d.value && d.value.blocks ? d.value.blocks.map((b) => ({ ...b, protoI: b.kind === 'surface' ? image(b.prototype) : vec(b.prototype), memI: b.members_mean ? (b.kind === 'surface' ? image(b.members_mean) : vec(b.members_mean)) : null,
      diffI: b.difference ? (b.kind === 'surface' ? image(b.difference) : vec(b.difference)) : null, x: b.x ? vec(b.x) : null })) : []));
    const range = computed(() => { const a = []; for (const b of blocks.value) if (b.kind === 'surface') { a.push(b.protoI); if (b.memI) a.push(b.memI); } return a.length ? sharedRange(a, 'sequential') : null; });
    const planeRows = computed(() => (d.value ? Object.entries(d.value.planes).filter(([, v]) => v !== null).slice(0, 12).map(([k, v]) => ({ key: k, label: d.value.plane_labels[k] || k, value: v })) : []));
    return { slot, d, R, axes, blocks, range, planeRows, emit, fmt, pct, nTxt, groupColour, fp };
  },
  template: `
  <div class="card som-unit-card" data-som-unit-card :data-unit="index">
    <div class="card-head"><h3>Unit {{ d ? d.name : index + 1 }}</h3><span class="muted xs" v-if="d">row {{ d.row }}, column {{ d.col }} · group {{ d.cluster + 1 }}</span></div>
    <div class="card-body col gap-12">
      <p class="muted small" v-if="slot && slot.busy">Reading the unit …</p>
      <Callout kind="critical" v-if="slot && slot.error">{{ slot.error }}</Callout>
      <template v-if="d">
        <p class="fp-note">{{ d.hits === 0 ? 'No sample has this unit as its best unit: its prototype interpolates between its neighbours.' : (d.hits === 1 ? '1 sample falls on this unit.' : d.hits + ' samples fall on this unit.') }}</p>
        <div class="som-unit-pics" v-if="blocks.length">
          <template v-for="b in blocks" :key="b.name">
            <div v-if="b.kind === 'surface' && axes" class="som-unit-pic"><div class="eyebrow">Prototype</div>
              <EEMView :ex="axes.ex" :em="axes.em" :z="b.protoI" :mask="axes.mask" :zrange="range" :compact="true" :height="240" :units="d.units || ''"/></div>
            <div v-if="b.kind === 'surface' && axes && b.memI" class="som-unit-pic"><div class="eyebrow">Mean of the samples of the unit</div>
              <EEMView :ex="axes.ex" :em="axes.em" :z="b.memI" :mask="axes.mask" :zrange="range" :compact="true" :height="240"/></div>
            <div v-if="b.kind === 'surface' && axes && b.diffI" class="som-unit-pic"><div class="eyebrow">Mean of the samples minus prototype</div>
              <EEMView :ex="axes.ex" :em="axes.em" :z="b.diffI" :mask="axes.mask" kind="diverging" :compact="true" :height="240"/></div>
            <div v-if="b.kind === 'curve'" class="som-unit-pic"><div class="eyebrow">Absorbance: prototype (and mean of the samples)</div>
              <SomSpark :y="b.protoI" :lo="Math.min(...b.protoI.filter(Number.isFinite), ...(b.memI ? b.memI.filter(Number.isFinite) : []))" :hi="Math.max(...b.protoI.filter(Number.isFinite), ...(b.memI ? b.memI.filter(Number.isFinite) : []))"/>
              <SomSpark v-if="b.memI" :y="b.memI" :lo="Math.min(...b.protoI.filter(Number.isFinite), ...b.memI.filter(Number.isFinite))" :hi="Math.max(...b.protoI.filter(Number.isFinite), ...b.memI.filter(Number.isFinite))" color="#999"/></div>
          </template>
        </div>
        <div v-if="d.variables" class="table-wrap"><table class="table"><thead><tr><th>Variable</th><th>Value of the prototype</th></tr></thead><tbody><tr v-for="v in d.variables" :key="v.name"><td>{{ v.name }}</td><td class="num">{{ fmt(v.prototype, 4) }}</td></tr></tbody></table></div>
        <div v-if="d.members.length"><div class="eyebrow">Samples on the unit</div>
          <div class="row wrap"><button type="button" class="chip som-member" v-for="(id, i) in d.members" :key="id" :title="'Distance to the prototype ' + fmt(d.qe_members[i], 3)" @click="emit('sample', id)">{{ id }}</button></div></div>
        <details class="fp-more" v-if="planeRows.length"><summary>The value of the planes at this unit</summary>
          <dl class="fp-facts"><template v-for="r in planeRows" :key="r.key"><dt>{{ r.label }}</dt><dd>{{ fmt(r.value, 4) }}</dd></template></dl></details>
      </template>
    </div>
  </div>`,
};

export const SomPrototypesTab = {
  name: 'SomPrototypesTab',
  components: { Icon, Seg, Callout, Empty, EEMThumb, SomSpark, SomUnitCard },
  setup() {
    const A = computed(() => fp.som), R = computed(() => fp.som.res), V = computed(() => fp.som.view);
    const slot = computed(() => (A.value.resRid && A.value.extra.protos ? A.value.extra.protos[`${A.value.resRid}|${V.value.proto}`] : null));
    watch(() => [A.value.resRid, V.value.proto], () => { if (A.value.resRid) loadPrototypes(V.value.proto); }, { immediate: true });
    const d = computed(() => (slot.value ? slot.value.data : null));
    const surfaceName = computed(() => (d.value ? Object.keys(d.value.stacks)[0] || null : null));
    const stk = computed(() => (surfaceName.value ? d.value.stacks[surfaceName.value] : null));
    const curveName = computed(() => (d.value ? Object.keys(d.value.curveMats)[0] || null : null));
    const crv = computed(() => (curveName.value ? d.value.curveMats[curveName.value] : null));
    const signed = computed(() => V.value.proto === 'difference');
    const range = computed(() => {
      const s = stk.value; if (!s) return null;
      const [M, r, c] = s.shape, per = r * c;
      const own = V.value.protoScale === 'own';
      if (own) return null;
      let lo = Infinity, hi = -Infinity;
      for (let i = 0; i < s.length; i++) { const v = s[i]; if (v === v) { if (v < lo) lo = v; if (v > hi) hi = v; } }
      if (lo === Infinity) return [0, 1];
      if (signed.value) { const m = Math.max(Math.abs(lo), Math.abs(hi)) || 1; return [-m, m]; }
      return [Math.min(0, lo), hi > 0 ? hi : 1];
    });
    const lut = computed(() => { themeState.version; return lutFor(signed.value ? divergingScale(prefs.div) : sequentialScale(prefs.seq)); });
    function unitRange(j) {
      const s = stk.value, per = s.shape[1] * s.shape[2]; let lo = Infinity, hi = -Infinity;
      for (let i = j * per; i < (j + 1) * per; i++) { const v = s[i]; if (v === v) { if (v < lo) lo = v; if (v > hi) hi = v; } }
      return lo === Infinity ? [0, 1] : (signed.value ? [-Math.max(Math.abs(lo), Math.abs(hi)), Math.max(Math.abs(lo), Math.abs(hi))] : [Math.min(0, lo), hi > 0 ? hi : 1]);
    }
    const units = computed(() => (R.value ? Array.from({ length: R.value.M }, (_, j) => j) : []));
    const hex = computed(() => !!R.value && R.value.lattice === 'hexagonal');
    const gridStyle = computed(() => { const r = R.value; return r ? { display: 'grid', gridTemplateColumns: `repeat(${hex.value ? 2 * r.cols + 1 : r.cols}, 1fr)`, gap: '4px 0' } : {}; });
    // a hexagonal map: every tile spans two columns and the odd rows start one column in (half a tile); a rectangular map: one column per tile
    const tileStyle = (j) => {
      const r = R.value; if (!r) return {};
      if (!hex.value) return { padding: '0 2px' };
      const first = j % r.cols === 0, odd = Math.floor(j / r.cols) % 2 === 1;
      return { gridColumn: first ? `${odd ? 2 : 1} / span 2` : 'span 2', padding: '0 2px' };
    };
    const curveRange = computed(() => { const c = crv.value; if (!c) return [0, 1]; let lo = Infinity, hi = -Infinity; for (const v of c.data) if (v === v) { if (v < lo) lo = v; if (v > hi) hi = v; } return lo === Infinity ? [0, 1] : [lo, hi]; });
    const curveRow = (j) => { const c = crv.value; return Array.from(c.data.subarray(j * c.k, (j + 1) * c.k)); };
    function select(j) { V.value.unit = j; }
    const dims = computed(() => (stk.value ? { nex: stk.value.shape[1], nem: stk.value.shape[2] } : null));
    const hits = computed(() => (R.value ? R.value.hits : []));
    function openSample(id) { V.value.sample = id; A.value.tab7 = 'samples'; }
    return { A, R, V, slot, d, stk, crv, range, lut, unitRange, units, gridStyle, tileStyle, curveRange, curveRow, select, dims, hits, signed, openSample, groupColour, fp, surfaceName, curveName };
  },
  template: `
  <div class="col gap-12" data-som-prototypes-tab>
    <div class="card"><div class="card-body" style="padding:12px 16px"><div class="row wrap" style="gap:12px 22px;align-items:flex-end">
      <div class="field"><label>Show</label><Seg v-model="V.proto" :options="[{ value: 'prototype', label: 'The prototype' }, { value: 'members', label: 'The mean of the samples' }, { value: 'difference', label: 'Samples minus prototype' }]" small data-som-proto-what/></div>
      <div class="field"><label>Colour scale</label><Seg v-model="V.protoScale" :options="[{ value: 'shared', label: 'The same for all units' }, { value: 'own', label: 'Own for each unit' }]" small data-som-proto-scale/></div>
    </div></div></div>
    <p class="fp-note">Each small picture is the prototype of one unit; neighbouring units hold similar pictures. The number in a corner is how many samples fall on the unit; units with a dashed outline hold no sample. Click a unit to open its card.</p>
    <p class="muted small" v-if="slot && slot.busy">Reading the pictures …</p>
    <Callout kind="critical" v-if="slot && slot.error" data-som-proto-error>{{ slot.error }}</Callout>
    <div class="som-protos" v-if="d && (stk || crv)" data-som-protos :data-units="R.M" :style="gridStyle">
      <button type="button" class="som-proto" v-for="j in units" :key="j" :style="tileStyle(j)" :class="{ 'no-hits': hits[j] === 0, sel: V.unit === j }" :title="R.unitNames[j] + ': ' + hits[j] + ' sample' + (hits[j] === 1 ? '' : 's')"
              :data-unit="j" :aria-label="'Unit ' + R.unitNames[j]" @click="select(j)">
        <span class="som-proto-pic" :style="{ outlineColor: groupColour(R.clusters.unitLabels[j]) }">
          <EEMThumb v-if="stk" :z="stk" :nex="dims.nex" :nem="dims.nem" :offset="j * dims.nex * dims.nem" :vmin="range ? range[0] : unitRange(j)[0]" :vmax="range ? range[1] : unitRange(j)[1]" :lut="lut" style="width:100%;height:auto;display:block"/>
          <SomSpark v-else-if="crv" :y="curveRow(j)" :lo="curveRange[0]" :hi="curveRange[1]"/>
        </span>
        <span class="som-proto-n">{{ hits[j] }}</span>
      </button>
    </div>
    <p class="fp-note muted" v-if="d && d.note">{{ d.note }}</p>
    <p class="fp-note muted" v-if="d && stk && crv">Absorbance prototypes are drawn as small curves in a second grid below the pictures of the EEMs.</p>
    <div class="som-protos" v-if="d && stk && crv" data-som-protos-curves :style="gridStyle">
      <button type="button" class="som-proto" v-for="j in units" :key="j" :style="tileStyle(j)" :class="{ 'no-hits': hits[j] === 0 }" @click="select(j)"><span class="som-proto-pic"><SomSpark :y="curveRow(j)" :lo="curveRange[0]" :hi="curveRange[1]"/></span></button>
    </div>
    <SomUnitCard v-if="V.unit !== null && V.unit !== undefined" :index="V.unit" @sample="openSample"/>
  </div>`,
};
