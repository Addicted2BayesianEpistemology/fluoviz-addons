// Step 7, tab "Latent space": every sample as a point in the space of the codes (any two codes, or the two main directions of the codes), coloured by any variable, hover cards
// with a small picture of the EEM, a click selects a sample; below it the codes against other variables.
import { ref, shallowRef, computed, watch, onMounted, onBeforeUnmount, onActivated } from 'vue';
import { Icon } from '/static/js/icons.js';
import { Field, Switch, Callout, Empty } from '/static/js/components/Common.js';
import { PlotlyChart, ChartCard } from '/static/js/plot.js';
import { ScatterPlot } from '/static/js/components/ScatterPlot.js';
import { VarSelect } from '/static/js/components/VarSelect.js';
import { EEMThumb } from '/static/js/components/EEMThumb.js';
import { catalogue, loadCatalogue, fetchValues, labelOf, kindOf } from '/static/js/variables.js';
import { column, mat, valuesForIds, rolesOf } from '/static/js/components/DecompCommon.js';
import { tokens, themeState, withAlpha, lutFor, sequentialScale, divergingScale } from '/static/js/theme.js';
import { prefs } from '/static/js/prefs.js';
import { fmt } from '/static/js/util.js';
import { fp } from './AeState.js';
import { loadGallery, loadSample, runRelate } from './AeActions.js';
import { pct, STATUS_LABEL } from './AeCommon.js';

const SHOW_ELLIPSES_MAX = 300;                                   // (the uncertainty ellipses are one drawing each: not for thousands of samples)

export const AeRelate = {
  name: 'AeRelate',
  components: { Icon, Field, Callout, VarSelect, PlotlyChart, ChartCard },
  setup() {
    const A = computed(() => fp.ae), R = computed(() => fp.ae.res);
    const pick = ref(null);
    const vars = computed(() => A.value.latent.relate);
    const slot = computed(() => A.value.extra.relate || null);
    const ans = computed(() => (slot.value ? slot.value.data : null));
    function add(key) { if (key && !vars.value.includes(key)) vars.value.push(key); pick.value = null; }
    function remove(key) { const i = vars.value.indexOf(key); if (i >= 0) vars.value.splice(i, 1); }
    const run = () => runRelate(vars.value);
    const traces = computed(() => {
      themeState.version;
      const a = ans.value, r = R.value; if (!a || !r) return [];
      const rows = r.codeNames.map((_, j) => a.variables.map((v) => (v.correlation && Number.isFinite(v.correlation[j]) ? v.correlation[j] : null)));
      return [{ type: 'heatmap', x: a.variables.map((v) => v.label), y: r.codeNames, z: rows, zmin: -1, zmax: 1, colorscale: divergingScale(prefs.div), xgap: 2, ygap: 2,
        colorbar: { thickness: 10, len: 0.9, outlinewidth: 0, title: { text: 'correlation', side: 'right', font: { size: 11 } }, tickfont: { size: 10 } }, hovertemplate: '%{y} and %{x}: correlation %{z:.2f}<extra></extra>' }];
    });
    const layout = computed(() => ({ height: 120 + 28 * (R.value ? R.value.codeNames.length : 3), margin: { l: 56, r: 14, t: 10, b: 90 }, yaxis: { autorange: 'reversed' }, xaxis: { tickangle: -30 } }));
    const table = computed(() => {
      const a = ans.value; if (!a) return { columns: [], rows: [] };
      return { columns: ['variable', 'samples', 'R² predicted from the codes', 'R² predicted from PCA scores'], rows: a.variables.map((v) => [v.label, v.n, v.r2_codes, v.r2_pca]) };
    });
    return { A, R, pick, vars, slot, ans, add, remove, run, traces, layout, table, labelOf, pct, fmt };
  },
  template: `
  <div class="card" data-ae-relate>
    <div class="card-head"><h3>Codes against other variables</h3></div>
    <div class="card-body col gap-12">
      <p class="fp-note">Which of your variables do the codes follow? For every variable you add, the page shows the correlation with each code and how well the variable can be predicted from the codes (cross-validated), next to the same from the same number of PCA scores.</p>
      <div class="row wrap">
        <Field label="Add a variable" style="min-width:240px"><VarSelect :model-value="pick" :kinds="['numeric']" none-label="Choose a variable …" small @update:modelValue="add"/></Field>
        <button type="button" class="btn" data-ae-relate-go :disabled="!vars.length || (slot && slot.busy)" @click="run"><Icon name="play" size="16"/>Relate the codes</button>
      </div>
      <div class="row wrap" v-if="vars.length"><span class="chip" v-for="v in vars" :key="v">{{ labelOf(v) }}<span class="x" role="button" tabindex="0" :aria-label="'Remove ' + labelOf(v)" @click="remove(v)" @keydown.enter="remove(v)">×</span></span></div>
      <Callout kind="critical" v-if="slot && slot.error">{{ slot.error }}</Callout>
      <div :class="{ 'fp-dim': slot && slot.busy }" v-if="ans">
        <p class="fp-note" v-if="ans.words">{{ ans.words }}</p>
        <p class="fp-note muted" v-if="ans.note">{{ ans.note }}</p>
        <p class="fp-note muted" v-if="ans.skipped && ans.skipped.length">Left out because they are not numbers: {{ ans.skipped.map(labelOf).join(', ') }}.</p>
        <ChartCard title="Correlation of the codes with the variables" :table="table" filename="autoencoder-codes-vs-variables"><PlotlyChart :data="traces" :layout="layout" :height="layout.height"/></ChartCard>
      </div>
    </div>
  </div>`,
};

export const AeLatentTab = {
  name: 'AeLatentTab',
  components: { Icon, Field, Switch, Callout, Empty, ScatterPlot, VarSelect, EEMThumb, AeRelate },
  setup() {
    const A = computed(() => fp.ae), R = computed(() => fp.ae.res), L = computed(() => fp.ae.latent);
    const root = ref(null), hoverId = ref(null);
    const colorVals = shallowRef(null);
    onMounted(() => { loadCatalogue(); loadGallery(); });
    const columns = computed(() => {
      const r = R.value; if (!r) return [];
      const out = r.codeNames.map((nm, j) => ({ key: nm, label: nm, values: column(r.codes, j) }));
      const pc = r.p.latent && r.p.latent.pca;
      if (r.d > 2 && pc && pc.scores) {
        const M = mat(pc.scores), ex = pc.explained || [];
        out.push({ key: 'pc1', label: `Codes: first main direction${Number.isFinite(ex[0]) ? ` (${pct(ex[0], 0)})` : ''}`, values: column(M, 0) }, { key: 'pc2', label: `Codes: second main direction${Number.isFinite(ex[1]) ? ` (${pct(ex[1], 0)})` : ''}`, values: column(M, 1) });
      }
      return out;
    });
    const col = (key) => columns.value.find((c) => c.key === key) || columns.value[0];
    // axes that no longer exist (another network with fewer codes) fall back to the first ones
    watch(columns, (c) => { if (c.length && !c.some((x) => x.key === L.value.x)) L.value.x = c[0].key; if (c.length > 1 && !c.some((x) => x.key === L.value.y)) L.value.y = c[1].key; if (c.length === 1) L.value.y = c[0].key; }, { immediate: true });
    const ids = computed(() => (R.value ? R.value.ids : []));
    const roles = computed(() => rolesOf(ids.value));
    const keys = computed(() => [L.value.color, L.value.symbol].filter(Boolean));
    watch([keys, () => ids.value.length], async () => {
      if (!keys.value.length) { colorVals.value = null; return; }
      try { colorVals.value = await fetchValues(keys.value); } catch { colorVals.value = null; }
    }, { immediate: true });
    watch(() => catalogue.value.length, () => { if (keys.value.length) fetchValues(keys.value).then((v) => { colorVals.value = v; }).catch(() => {}); });
    // the split (training / held out) is a published variable of the network: the marker shape says which samples the network did not see
    const splitKey = computed(() => (A.value.resRid ? `res:${A.value.resRid}:split` : null));
    watch([splitKey, () => catalogue.value.length], () => { if (splitKey.value && !L.value.symbol && catalogue.value.some((v) => v.key === splitKey.value)) L.value.symbol = splitKey.value; }, { immediate: true });
    const spec = (key) => (!key || !colorVals.value ? null : { key, label: labelOf(key), kind: kindOf(key), values: valuesForIds(colorVals.value, ids.value, key) });
    const colorSpec = computed(() => spec(L.value.color));
    const symbolSpec = computed(() => { const s = spec(L.value.symbol); return s && s.kind !== 'numeric' ? s : null; });
    const xs = computed(() => Array.from(col(L.value.x).values)), ys = computed(() => Array.from(col(L.value.y).values));

    // the uncertainty of every code of a variational autoencoder: one ellipse per sample (only when both axes are codes)
    const shapes = computed(() => {
      themeState.version;
      const r = R.value; if (!r || !r.codesSd || !L.value.sd || ids.value.length > SHOW_ELLIPSES_MAX) return [];
      const jx = r.codeNames.indexOf(L.value.x), jy = r.codeNames.indexOf(L.value.y); if (jx < 0 || jy < 0) return [];
      const t = tokens(), out = [], d = r.codesSd.k;
      for (let i = 0; i < r.n; i++) {
        const sx = r.codesSd.data[i * d + jx], sy = r.codesSd.data[i * d + jy];
        if (!(sx > 0 && sy > 0)) continue;
        const cx = r.codes.data[i * r.codes.k + jx], cy = r.codes.data[i * r.codes.k + jy];
        out.push({ type: 'circle', xref: 'x', yref: 'y', x0: cx - sx, x1: cx + sx, y0: cy - sy, y1: cy + sy, line: { color: withAlpha(t.series[0], 0.45), width: 1 }, layer: 'below' });
      }
      return out;
    });
    const ellipseNote = computed(() => (L.value.sd && ids.value.length > SHOW_ELLIPSES_MAX ? `The uncertainty is drawn for up to ${SHOW_ELLIPSES_MAX} samples.` : ''));

    // hover: the card at the right shows the sample the pointer is on (else the selected one)
    function attach() {
      const gd = root.value && root.value.querySelector('.js-plotly-plot');
      if (!gd || typeof gd.on !== 'function') return false;
      if (gd.__aeHover) return true;
      gd.__aeHover = true;
      gd.on('plotly_hover', (ev) => { const p = ev && ev.points && ev.points.find((q) => q.customdata); if (p) hoverId.value = p.customdata[0]; });
      gd.on('plotly_unhover', () => { hoverId.value = null; });
      return true;
    }
    let timer = null;
    function tryAttach() { clearInterval(timer); let n = 0; timer = setInterval(() => { if (attach() || ++n > 60) clearInterval(timer); }, 200); }
    onMounted(tryAttach); onActivated(tryAttach);
    watch(() => [columns.value.length, colorSpec.value && colorSpec.value.key], () => tryAttach());
    onBeforeUnmount(() => clearInterval(timer));

    const shownId = computed(() => hoverId.value || A.value.sample);
    watch(shownId, (id) => { if (id) loadSample(id, true); }, { immediate: true });
    const facts = computed(() => { const id = shownId.value, rid = A.value.resRid; const s = id && rid && A.value.extra.samples ? A.value.extra.samples[`${rid}|${id}|t`] : null; return s && s.data ? s.data : null; });
    const gallery = computed(() => (A.value.gallery && A.value.gallery.data) || null);
    const thumb = computed(() => {
      themeState.version;
      const g = gallery.value, id = shownId.value; if (!g || !id || !g.index.has(id)) return null;
      const i = g.index.get(id), nex = g.z.shape[1], nem = g.z.shape[2];
      return { z: g.z, nex, nem, offset: i * nex * nem, vmin: g.vmin, vmax: g.vmax, lut: lutFor(g.kind === 'diverging' ? divergingScale(prefs.div) : sequentialScale(prefs.seq)) };
    });
    const codeText = computed(() => (facts.value && facts.value.code ? facts.value.code.map((v, j) => `${R.value ? R.value.codeNames[j] : `z${j + 1}`} = ${fmt(v, 3)}`).join(', ') : ''));
    function select(id) { A.value.sample = id; }
    function openSample() { if (A.value.sample) A.value.tab7 = 'samples'; }
    return { A, R, L, root, columns, ids, roles, colorSpec, symbolSpec, xs, ys, shapes, ellipseNote, shownId, facts, thumb, codeText, select, openSample, hoverId, STATUS_LABEL, pct, fmt };
  },
  template: `
  <div class="col gap-12" data-ae-latent-tab ref="root">
    <div class="card"><div class="card-body" style="padding:12px 16px"><div class="row wrap" style="gap:12px 22px;align-items:flex-end">
      <Field label="Horizontal axis" style="width:230px"><select class="select sm" v-model="L.x" data-ae-axis-x><option v-for="c in columns" :key="c.key" :value="c.key">{{ c.label }}</option></select></Field>
      <Field label="Vertical axis" style="width:230px"><select class="select sm" v-model="L.y" data-ae-axis-y><option v-for="c in columns" :key="c.key" :value="c.key">{{ c.label }}</option></select></Field>
      <Field label="Colour by" help="Colour the points by any metadata column, extracted feature or result of another model: the quickest way to see what a code means." style="width:240px"><VarSelect v-model="L.color" small none-label="Nothing (one colour)"/></Field>
      <Field label="Symbol by" help="A second category drawn as marker shapes. The default is whether the network was trained on the sample or not." style="width:220px"><VarSelect v-model="L.symbol" small :kinds="['category']" none-label="Nothing"/></Field>
      <div class="col" style="gap:6px;padding-bottom:4px" v-if="R && R.codesSd"><Switch v-model="L.sd" label="Show the uncertainty of each code"/><span class="muted xs" v-if="ellipseNote">{{ ellipseNote }}</span></div>
    </div></div></div>
    <div class="ae-latent-grid">
      <div class="ae-latent-plot" data-ae-latent-chart>
        <ScatterPlot :show-stats="false" :selectable="false" :ids="ids" :x="xs" :y="ys" :roles="roles" :x-label="(columns.find((c) => c.key === L.x) || {}).label" :y-label="(columns.find((c) => c.key === L.y) || {}).label"
                     :color="colorSpec" :symbol="symbolSpec" :highlight="A.sample ? [A.sample] : []" :shapes="shapes" zero-lines :height="440" title="Codes of the samples" filename="autoencoder-codes"
                     help="One point per sample. Samples with similar codes are described in a similar way by the network. Hover to see the sample, click to select it." @click="select"/>
      </div>
      <div class="card ae-hover" data-ae-hover :data-id="shownId || ''" aria-live="polite">
        <div class="card-head"><h3>{{ shownId ? shownId : 'Hover over a point' }}</h3></div>
        <div class="card-body col gap-8">
          <p class="fp-note" v-if="!shownId">Move the pointer over a point to see the sample, and click it to select it.</p>
          <template v-else>
            <div class="ae-thumb" v-if="thumb"><EEMThumb :z="thumb.z" :nex="thumb.nex" :nem="thumb.nem" :offset="thumb.offset" :vmin="thumb.vmin" :vmax="thumb.vmax" :lut="thumb.lut"/></div>
            <dl class="fp-facts" v-if="facts"><dt>Used for</dt><dd>{{ facts.split }}</dd><dt>Status</dt><dd :data-status="facts.status">{{ STATUS_LABEL[facts.status] || facts.status }}</dd><dt>Not rebuilt</dt><dd>{{ pct(facts.error, 1) }} of the variation</dd></dl>
            <p class="fp-note" v-if="codeText">{{ codeText }}</p>
            <p class="fp-note muted" v-if="A.sample === shownId">Selected.</p>
            <button type="button" class="btn sm" data-ae-open-sample v-if="A.sample" @click="openSample"><Icon name="arrow-right" size="15"/>Open in Samples</button>
          </template>
        </div>
      </div>
    </div>
    <AeRelate/>
  </div>`,
};
