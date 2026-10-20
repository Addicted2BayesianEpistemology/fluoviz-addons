// Step 7, tab "Projection": the prototypes drawn by their true distances (Sammon mapping), with the edges of the lattice: a folded map shows as crossing edges.
import { computed, watch } from 'vue';
import { Callout } from '/static/js/components/Common.js';
import { PlotlyChart, ChartCard } from '/static/js/plot.js';
import { tokens, themeState } from '/static/js/theme.js';
import { vec, mat } from '/static/js/components/DecompCommon.js';
import { fp } from './SomState.js';
import { loadProjection } from './SomActions.js';
import { SomVerdict } from './SomCommon.js';

export const SomProjectionTab = {
  name: 'SomProjectionTab',
  components: { Callout, PlotlyChart, ChartCard, SomVerdict },
  setup() {
    const A = computed(() => fp.som), R = computed(() => fp.som.res);
    watch(() => A.value.resRid, (r) => { if (r) loadProjection(); }, { immediate: true });
    const slot = computed(() => (A.value.resRid ? A.value.extra[`proj|${A.value.resRid}`] : null));
    const d = computed(() => (slot.value ? slot.value.data : null));
    const traces = computed(() => {
      themeState.version;
      const p = d.value, r = R.value; if (!p || !r) return [];
      const t = tokens(), xy = mat(p.xy), E = mat(p.edges), hits = vec(p.hits);
      const x = [], y = [];
      for (let e = 0; e < E.n; e++) { const a = E.data[e * 2], b = E.data[e * 2 + 1]; x.push(xy.data[a * 2], xy.data[b * 2], null); y.push(xy.data[a * 2 + 1], xy.data[b * 2 + 1], null); }
      const ux = [], uy = [], size = [], text = [];
      for (let j = 0; j < xy.n; j++) { ux.push(xy.data[j * 2]); uy.push(xy.data[j * 2 + 1]); size.push(8 + 4 * Math.sqrt(hits[j])); text.push(`Unit ${r.unitNames[j]}: ${hits[j]} sample${hits[j] === 1 ? '' : 's'}`); }
      return [{ type: 'scatter', mode: 'lines', x, y, line: { color: t.text3 || t.line, width: 1.2 }, hoverinfo: 'skip', showlegend: false },
        { type: 'scatter', mode: 'markers', x: ux, y: uy, text, hoverinfo: 'text', showlegend: false, marker: { size, color: t.series[0], line: { color: t.surface, width: 1 } } }];
    });
    const layout = { margin: { l: 10, r: 10, t: 10, b: 10 }, xaxis: { visible: false }, yaxis: { visible: false, scaleanchor: 'x', scaleratio: 1 }, hovermode: 'closest' };
    const table = computed(() => { const p = d.value, r = R.value; if (!p || !r) return { columns: [], rows: [] }; const xy = mat(p.xy); return { columns: ['unit', 'x', 'y'], rows: r.unitNames.map((u, j) => [u, xy.data[j * 2], xy.data[j * 2 + 1]]) }; });
    return { slot, d, traces, layout, table };
  },
  template: `
  <div class="col gap-12" data-som-projection-tab>
    <p class="fp-note">Each circle is a unit, drawn where its prototype lies when the true distances between all the prototypes are respected as well as possible on a flat page (Sammon mapping); larger circles hold more samples. The lines join neighbouring units of the map. If the lines cross, the map is folded there.</p>
    <p class="muted small" v-if="slot && slot.busy">Drawing …</p>
    <Callout kind="critical" v-if="slot && slot.error">{{ slot.error }}</Callout>
    <template v-if="d">
      <SomVerdict :level="d.crossings === 0 ? 'good' : 'warn'" data-som-projection-words>{{ d.words }}</SomVerdict>
      <ChartCard title="The prototypes by their true distances" :table="table" filename="som-projection"><PlotlyChart :data="traces" :layout="layout" :height="460"/></ChartCard>
    </template>
  </div>`,
};
