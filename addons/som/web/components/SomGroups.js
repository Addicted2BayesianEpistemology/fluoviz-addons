// Step 7, tab "Groups": the units grouped (two-level clustering of the prototypes, Vesanto and Alhoniemi 2000) and what the groups are: sizes, mean EEMs, profiles, links to the metadata.
import { ref, computed, watch, onMounted } from 'vue';
import { Icon } from '/static/js/icons.js';
import { Field, NumField, Callout, Seg } from '/static/js/components/Common.js';
import { ChartCard } from '/static/js/plot.js';
import { EEMThumb } from '/static/js/components/EEMThumb.js';
import { lutFor, sequentialScale, divergingScale, themeState } from '/static/js/theme.js';
import { prefs } from '/static/js/prefs.js';
import { go } from '/static/js/router.js';
import { fmt } from '/static/js/util.js';
import { fp } from './SomState.js';
import { loadGroups, runCluster } from './SomActions.js';
import { SomGrid } from './SomGrid.js';
import { groupColour, nTxt, SomVerdict } from './SomCommon.js';

export const SomGroupsTab = {
  name: 'SomGroupsTab',
  components: { Icon, Field, NumField, Callout, Seg, ChartCard, EEMThumb, SomGrid, SomVerdict },
  setup() {
    const A = computed(() => fp.som), R = computed(() => fp.som.res), G = computed(() => fp.som.groupsView);
    const cur = computed(() => (R.value ? R.value.clusters : null));
    const mapProps = computed(() => { const r = R.value; return r ? { M: r.M, rows: r.rows, cols: r.cols, lattice: r.lattice, shape: r.shape, coords: r.coords, edges: r.edges } : null; });
    const plane = computed(() => {
      const c = cur.value; if (!c) return null;
      return { key: 'groups', label: 'Group of the unit', kind: 'category', v: c.unitLabels, categories: Array.from({ length: c.k }, (_, g) => `Group ${g + 1}`), words: 'The group each unit belongs to.' };
    });
    watch(() => A.value.resRid, () => { if (A.value.resRid) loadGroups(); }, { immediate: true });
    const slot = computed(() => (A.value.resRid ? A.value.extra[`groups|${A.value.resRid}`] : null));
    const d = computed(() => (slot.value ? slot.value.data : null));
    onMounted(() => { const c = fp.som.res && fp.som.res.p.clusters; if (c) { G.value.method = c.settings.method; G.value.k = c.settings.k === 'auto' ? 'auto' : 'n'; if (typeof c.settings.k === 'number') G.value.k_n = c.settings.k; G.value.skip_empty = c.settings.skip_empty; } });
    async function recompute() {
      await runCluster({ method: G.value.method, k: G.value.k === 'auto' ? 'auto' : Math.round(Number(G.value.k_n)), skip_empty: G.value.skip_empty });
    }
    const keys = computed(() => (A.value.resRid ? [`res:${A.value.resRid}:som_x`, `res:${A.value.resRid}:som_y`] : []));
    const openCluster = () => go('/cluster', { keys: keys.value.join(',') });
    const openSeparation = () => go('/cluster', { keys: keys.value.join(','), tab: 'separation' });
    const lut = computed(() => { themeState.version; return lutFor(sequentialScale(prefs.seq)); });
    const lutD = computed(() => { themeState.version; return lutFor(divergingScale(prefs.div)); });
    const im = computed(() => (d.value && d.value.images ? d.value.images : null));
    const imRange = computed(() => {
      const i = im.value; if (!i) return null; let lo = Infinity, hi = -Infinity;
      for (let k = 0; k < i.mean.length; k++) { const v = i.mean[k]; if (v === v) { if (v < lo) lo = v; if (v > hi) hi = v; } }
      return lo === Infinity ? [0, 1] : [Math.min(0, lo), hi];
    });
    const zRange = computed(() => {
      const i = im.value; if (!i || !i.z) return null; let m = 0;
      for (let k = 0; k < i.z.length; k++) { const v = i.z[k]; if (v === v && Math.abs(v) > m) m = Math.abs(v); }
      return [-(m || 1), m || 1];
    });
    const profileRows = computed(() => {
      const p = d.value && d.value.profiles; if (!p) return [];
      return (p.variables || []).map((v, j) => ({ name: v, z: (p.cluster_ids || []).map((_, g) => (p.z_vs_overall ? p.z_vs_overall[g][j] : null)) }));
    });
    const assoc = computed(() => (d.value && d.value.associations ? (d.value.associations.variables || []).slice(0, 12) : []));
    return { A, R, G, cur, mapProps, plane, slot, d, recompute, openCluster, openSeparation, im, imRange, zRange, lut, lutD, profileRows, assoc, groupColour, nTxt, fmt };
  },
  template: `
  <div class="col gap-12" data-som-groups-tab>
    <p class="fp-note">The prototypes of the units are grouped (second-level clustering), and every sample takes the group of its unit. Groups are numbered by the number of samples they hold. Empty units hold no sample, so by default they are left out of the clustering when most of the map is empty.</p>
    <div class="card"><div class="card-head"><h3>Cluster the map</h3></div><div class="card-body"><div class="row wrap" style="gap:12px 22px;align-items:flex-end">
      <Field label="Method" style="width:240px"><select class="select sm" v-model="G.method" data-som-group-method><option value="hierarchical">Ward clustering (hierarchical)</option><option value="kmeans">k-means</option><option value="gmm">Gaussian mixture</option></select></Field>
      <Field label="Number of groups" style="width:240px"><Seg v-model="G.k" :options="[{ value: 'auto', label: 'From the data' }, { value: 'n', label: 'I choose' }]" small/><NumField v-if="G.k !== 'auto'" v-model="G.k_n" :min="2" :max="50" small data-som-group-k/></Field>
      <Field label="Empty units" style="width:260px"><select class="select sm" v-model="G.skip_empty"><option value="auto">Leave out when most are empty</option><option :value="true">Always leave out</option><option :value="false">Keep</option></select></Field>
      <button type="button" class="btn" data-som-cluster-go :disabled="A.extra.clusterBusy" @click="recompute"><Icon name="play" size="16"/>Cluster the map</button>
    </div>
    <Callout kind="critical" v-if="A.extra.clusterError">{{ A.extra.clusterError }}</Callout></div></div>
    <SomVerdict level="info" v-if="cur" data-som-group-reason>{{ cur.k }} group{{ cur.k === 1 ? '' : 's' }}. {{ cur.reason }}</SomVerdict>
    <Callout kind="warning" v-for="w in (cur ? cur.warnings : [])" :key="w">{{ w }}</Callout>
    <div class="som-groups-grid" v-if="cur && plane">
      <ChartCard title="Groups on the map" filename="som-groups"><SomGrid :map="mapProps" :plane="plane" :groups="cur.k > 1 ? cur.unitLabels : null" :height="360"/></ChartCard>
      <div class="card"><div class="card-head"><h3>Sizes</h3></div><div class="card-body"><table class="table"><thead><tr><th>Group</th><th>Samples</th></tr></thead>
        <tbody><tr v-for="(n, g) in cur.sizes" :key="g"><td><span class="som-key"><i :style="{ background: groupColour(g) }"></i>Group {{ g + 1 }}</span></td><td class="num">{{ n }}</td></tr></tbody></table>
        <div class="row wrap" style="margin-top:12px"><button type="button" class="btn" data-som-open-cluster @click="openCluster"><Icon name="arrow-right" size="16"/>Open in Clustering &amp; LDA</button>
          <button type="button" class="btn" data-som-open-separation @click="openSeparation"><Icon name="arrow-right" size="16"/>Separate the groups (LDA)</button></div>
        <p class="fp-note muted">The map positions (map position across, map position down) are published variables: Clustering &amp; LDA can group or separate the samples on them. The groups are published as the variable Map group.</p></div></div>
    </div>
    <p class="muted small" v-if="slot && slot.busy">Reading the interpretation …</p>
    <Callout kind="critical" v-if="slot && slot.error">{{ slot.error }}</Callout>
    <template v-if="d">
      <div class="card" v-if="im" data-som-group-images><div class="card-head"><h3>Mean EEM of each group</h3></div><div class="card-body">
        <div class="som-group-strip"><div v-for="(n, g) in im.n_samples" :key="g" class="som-group-pic"><div class="eyebrow"><span class="som-key"><i :style="{ background: groupColour(g) }"></i>Group {{ g + 1 }} ({{ n }})</span></div>
          <EEMThumb :z="im.mean" :nex="im.mean.shape[1]" :nem="im.mean.shape[2]" :offset="g * im.mean.shape[1] * im.mean.shape[2]" :vmin="imRange[0]" :vmax="imRange[1]" :lut="lut" style="width:100%;height:auto"/></div></div>
        <div class="eyebrow" style="margin-top:12px" v-if="im.z">Where each group differs from all the others, relative to the scatter (z-map)</div>
        <div class="som-group-strip" v-if="im.z"><div v-for="(n, g) in im.n_samples" :key="'z' + g" class="som-group-pic"><div class="eyebrow">Group {{ g + 1 }}</div>
          <EEMThumb :z="im.z" :nex="im.z.shape[1]" :nem="im.z.shape[2]" :offset="g * im.z.shape[1] * im.z.shape[2]" :vmin="zRange[0]" :vmax="zRange[1]" :lut="lutD" style="width:100%;height:auto"/></div></div>
      </div></div>
      <Callout kind="info" v-if="d.images_note">{{ d.images_note }}</Callout>
      <div class="card" v-if="profileRows.length"><div class="card-head"><h3>What characterises each group</h3></div><div class="card-body"><div class="table-wrap"><table class="table"><thead><tr><th>Quantity</th><th v-for="(n, g) in cur.sizes" :key="g">Group {{ g + 1 }}</th></tr></thead>
        <tbody><tr v-for="r in profileRows" :key="r.name"><td>{{ r.name }}</td><td class="num" v-for="(z, g) in r.z" :key="g">{{ fmt(z, 2) }}</td></tr></tbody></table></div>
        <p class="fp-note muted">The mean of each group in standard deviations from the overall mean (0 = average; positive = higher in this group).</p></div></div>
      <div class="card" v-if="assoc.length"><div class="card-head"><h3>Links to the metadata</h3></div><div class="card-body"><div class="table-wrap"><table class="table"><thead><tr><th>Variable</th><th>Test</th><th>Effect size</th><th>p (adjusted)</th></tr></thead>
        <tbody><tr v-for="a in assoc" :key="a.name"><td>{{ a.name }}</td><td>{{ a.test }}</td><td class="num">{{ fmt(a.effect_size, 2) }} ({{ a.effect }})</td><td class="num">{{ fmt(a.p_adj, 3) }}</td></tr></tbody></table></div></div></div>
    </template>
  </div>`,
};
