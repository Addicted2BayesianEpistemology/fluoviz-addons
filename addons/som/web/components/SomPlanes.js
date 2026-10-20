// Step 7, tab "Planes": the same map drawn for several quantities side by side (the component planes of the literature): the principal directions, pixels and boxes of the prototype
// EEM, fluorescence and absorbance indices of the prototypes, and the mean of any variable of the samples of each unit (a category as a label map).
import { ref, computed, watch, onMounted } from 'vue';
import { Icon } from '/static/js/icons.js';
import { Field, Switch, Callout, Seg } from '/static/js/components/Common.js';
import { ChartCard } from '/static/js/plot.js';
import { VarSelect } from '/static/js/components/VarSelect.js';
import { EEMView } from '/static/js/components/EEMView.js';
import { loadCatalogue } from '/static/js/variables.js';
import { vec, image, maskArr } from '/static/js/components/DecompCommon.js';
import { fmt } from '/static/js/util.js';
import { fp } from './SomState.js';
import { loadPlane, loadIndices } from './SomActions.js';
import { SomGrid } from './SomGrid.js';
import { planeOptions } from './SomMapTab.js';

export const SomPlanesTab = {
  name: 'SomPlanesTab',
  components: { Icon, Field, Switch, Callout, Seg, ChartCard, VarSelect, SomGrid, EEMView },
  setup() {
    const A = computed(() => fp.som), R = computed(() => fp.som.res);
    const shown = ref(['umatrix', 'hits']);
    const kind = ref('index'), index = ref('HIX'), pixel = ref({ ex: 330, em: 450 }), box = ref({ ex0: 300, ex1: 350, em0: 400, em1: 480, mode: 'region_mean' }), wl = ref(300), varKey = ref(null), smooth = ref(false);
    const error = ref(''), busy = ref(false), shared = ref(false);
    onMounted(() => { loadCatalogue(); loadIndices(); });
    const idx = computed(() => { const s = A.value.extra[`indices|${A.value.resRid}`]; return s && s.data ? s.data.indices : []; });
    const groups = computed(() => { const m = new Map(); for (const i of idx.value) { if (!m.has(i.group)) m.set(i.group, []); m.get(i.group).push(i); } return [...m.entries()]; });
    const all = computed(() => planeOptions());
    const mapProps = computed(() => { const r = R.value; return r ? { M: r.M, rows: r.rows, cols: r.cols, lattice: r.lattice, shape: r.shape, coords: r.coords, edges: r.edges } : null; });
    const planes = computed(() => shown.value.map((k) => all.value.find((o) => o.key === k)).filter(Boolean).map((o) => o.plane));
    function toggle(k) { const i = shown.value.indexOf(k); if (i >= 0) shown.value.splice(i, 1); else shown.value.push(k); }
    async function add() {
      error.value = ''; busy.value = true;
      let body;
      if (kind.value === 'index') body = { kind: 'index', key: index.value };
      else if (kind.value === 'point') body = { kind: 'point', ex: Number(pixel.value.ex), em: Number(pixel.value.em) };
      else if (kind.value === 'box') body = { kind: box.value.mode, ex: [Number(box.value.ex0), Number(box.value.ex1)], em: [Number(box.value.em0), Number(box.value.em1)] };
      else if (kind.value === 'total') body = { kind: 'total' };
      else if (kind.value === 'absorbance') body = { kind: 'absorbance', wl: Number(wl.value) };
      else { if (!varKey.value) { error.value = 'Choose a variable first.'; busy.value = false; return; } body = { kind: 'var', key: varKey.value, smooth: smooth.value }; }
      const slot = await loadPlane(body);
      busy.value = false;
      if (!slot || slot.error || !slot.data) { error.value = (slot && slot.error) || 'The plane could not be computed.'; return; }
      const p = slot.data;
      if (!A.value.planesAdded.some((x) => x.key === p.key)) A.value.planesAdded.push(p);
      if (!shown.value.includes(p.key)) shown.value.push(p.key);
    }
    const hasEem = computed(() => !!(R.value && R.value.protos));
    const hasAbs = computed(() => !!(R.value && R.value.protoCurves));
    const kinds = computed(() => {
      const out = [];
      if (hasEem.value) out.push({ value: 'index', label: 'An index of the prototypes' }, { value: 'point', label: 'Intensity at a pixel' }, { value: 'box', label: 'Intensity in a box' }, { value: 'total', label: 'Overall intensity' });
      if (hasAbs.value) out.push({ value: 'absorbance', label: 'Absorbance at a wavelength' });
      out.push({ value: 'var', label: 'A variable of the samples' });
      return out;
    });
    watch(kinds, (k) => { if (k.length && !k.some((x) => x.value === kind.value)) kind.value = k[0].value; }, { immediate: true });
    // the picture of the mean EEM, to read off a pixel or a box
    const mean = computed(() => (R.value && R.value.p.mean_image && hasEem.value ? image(Object.values(R.value.p.mean_image)[0]) : null));
    const ax = computed(() => (R.value && R.value.p.ex ? { ex: vec(R.value.p.ex), em: vec(R.value.p.em), mask: R.value.p.mask ? maskArr(R.value.p.mask) : null } : null));
    function pick(e) { if (e && Number.isFinite(e.ex) && Number.isFinite(e.em)) { pixel.value = { ex: Math.round(e.ex), em: Math.round(e.em) }; kind.value = 'point'; } }
    return { A, R, shown, kind, kinds, index, groups, pixel, box, wl, varKey, smooth, error, busy, all, mapProps, planes, toggle, add, hasEem, mean, ax, pick, shared, fmt };
  },
  template: `
  <div class="col gap-12" data-som-planes-tab>
    <p class="fp-note">A plane colours the units by one quantity. Planes that look alike say that the quantities rise and fall together across the map; a plane that looks like the hits is just where the samples are. Tick the planes to draw them side by side.</p>
    <div class="card"><div class="card-head"><h3>Planes to show</h3></div><div class="card-body col gap-12">
      <div class="row wrap"><label class="chip som-plane-pick" v-for="o in all" :key="o.key" :class="{ on: shown.includes(o.key) }" :data-plane="o.key"><input type="checkbox" :checked="shown.includes(o.key)" @change="toggle(o.key)"/> {{ o.label }}</label></div>
      <div class="row wrap" style="gap:12px 20px;align-items:flex-end">
        <Field label="Add a plane" style="width:260px"><select class="select sm" v-model="kind" data-som-plane-kind><option v-for="k in kinds" :key="k.value" :value="k.value">{{ k.label }}</option></select></Field>
        <Field v-if="kind === 'index'" label="Index" style="width:300px" help="An index of a prototype (an average, possibly smoothed or shape-only) is not the index of a sample: read it as where on the map the quantity is high."><select class="select sm" v-model="index" data-som-plane-index>
          <optgroup v-for="[g, items] in groups" :key="g" :label="g"><option v-for="i in items" :key="i.key" :value="i.key">{{ i.name }}</option></optgroup></select></Field>
        <template v-if="kind === 'point'"><Field label="Excitation (nm)" style="width:120px"><input class="input sm" type="number" v-model.number="pixel.ex" data-som-px-ex/></Field><Field label="Emission (nm)" style="width:120px"><input class="input sm" type="number" v-model.number="pixel.em" data-som-px-em/></Field></template>
        <template v-if="kind === 'box'"><Field label="Statistic" style="width:150px"><select class="select sm" v-model="box.mode"><option value="region_mean">Mean</option><option value="region_max">Highest</option></select></Field>
          <Field label="Excitation from–to (nm)" style="width:200px"><div class="pair"><input class="input sm" type="number" v-model.number="box.ex0"/><input class="input sm" type="number" v-model.number="box.ex1"/></div></Field>
          <Field label="Emission from–to (nm)" style="width:200px"><div class="pair"><input class="input sm" type="number" v-model.number="box.em0"/><input class="input sm" type="number" v-model.number="box.em1"/></div></Field></template>
        <Field v-if="kind === 'absorbance'" label="Wavelength (nm)" style="width:140px"><input class="input sm" type="number" v-model.number="wl" data-som-wl/></Field>
        <template v-if="kind === 'var'"><Field label="Variable" style="width:300px"><VarSelect v-model="varKey" small none-label="Choose a variable …" data-som-var/></Field><Switch v-model="smooth" label="Smooth over neighbouring units"/></template>
        <button type="button" class="btn" data-som-add-plane :disabled="busy" @click="add"><Icon name="plus" size="16"/>Add this plane</button>
      </div>
      <Callout kind="critical" v-if="error" data-som-plane-error>{{ error }}</Callout>
      <details class="fp-more" v-if="kind === 'point' && mean && ax"><summary>Pick a pixel on the picture of the mean EEM</summary>
        <EEMView :ex="ax.ex" :em="ax.em" :z="mean" :mask="ax.mask" :units="R.p.units || ''" :selectable="false" :toolbar="false" :marginals="'none'" :height="320" @click="pick"/></details>
    </div></div>
    <div class="som-planes-grid" data-som-planes>
      <ChartCard v-for="p in planes" :key="p.key" :title="p.label" :help="p.words" :filename="'som-plane-' + p.key" :table="{ columns: ['unit', 'value'], rows: R.unitNames.map((u, j) => [u, p.kind === 'numeric' ? (Number.isFinite(p.v[j]) ? p.v[j] : null) : ((p.categories || [])[p.v[j]] || '')]) }">
        <SomGrid :map="mapProps" :plane="p" :groups="null" :height="300"/>
      </ChartCard>
    </div>
    <Callout kind="info" v-if="!planes.length">Tick at least one plane to draw it.</Callout>
  </div>`,
};
