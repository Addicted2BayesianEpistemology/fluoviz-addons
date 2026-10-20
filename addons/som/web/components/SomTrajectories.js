// Step 7, tab "Trajectories": the path of the samples over the map when they are put in order (by a date, a time, a treatment stage), one line for each group of samples.
import { computed, onMounted } from 'vue';
import { mat } from '/static/js/components/DecompCommon.js';
import { Icon } from '/static/js/icons.js';
import { Field, Callout } from '/static/js/components/Common.js';
import { ChartCard } from '/static/js/plot.js';
import { VarSelect } from '/static/js/components/VarSelect.js';
import { loadCatalogue } from '/static/js/variables.js';
import { tokens } from '/static/js/theme.js';
import { fp } from './SomState.js';
import { runTrajectories } from './SomActions.js';
import { SomGrid } from './SomGrid.js';

export const SomTrajectoriesTab = {
  name: 'SomTrajectoriesTab',
  components: { Icon, Field, Callout, ChartCard, VarSelect, SomGrid },
  setup() {
    const A = computed(() => fp.som), R = computed(() => fp.som.res), T = computed(() => fp.som.trajView);
    onMounted(() => loadCatalogue());
    const mapProps = computed(() => { const r = R.value; return r ? { M: r.M, rows: r.rows, cols: r.cols, lattice: r.lattice, shape: r.shape, coords: r.coords, edges: r.edges } : null; });
    const plane = computed(() => (R.value ? R.value.planes.hits : null));
    const slot = computed(() => (A.value.extra.trajKey ? A.value.extra[A.value.extra.trajKey] : null));
    const data = computed(() => (slot.value ? slot.value.data : null));
    const lines = computed(() => {
      const d = data.value; if (!d) return [];
      const t = tokens();
      return d.lines.map((ln, i) => {
        const P = mat(ln.pos), x = [], y = [];
        for (let k = 0; k < P.n; k++) { x.push(P.data[k * 2]); y.push(P.data[k * 2 + 1]); }
        return { name: ln.group, x, y, color: t.series[i % 8], text: ln.ids.map((id, k) => `${ln.group}: ${id} (${k + 1} of ${ln.ids.length})`) };
      });
    });
    const run = () => runTrajectories({ order_by: T.value.order, group_by: T.value.group });
    return { A, R, T, mapProps, plane, slot, data, lines, run };
  },
  template: `
  <div class="col gap-12" data-som-trajectories-tab>
    <p class="fp-note">Put the samples in order (by a date, a time, a treatment stage) and draw, for each group of samples, the path they follow over the map. A hollow circle marks the first sample of a path and a square the last. Short steps mean the samples change little; a long jump means the sample moved to another region of the map.</p>
    <div class="card"><div class="card-body"><div class="row wrap" style="gap:12px 22px;align-items:flex-end">
      <Field label="Order by" style="width:280px" help="A number or a date: the samples of a path are joined in increasing order of this variable."><VarSelect v-model="T.order" small none-label="Choose a variable …" data-som-traj-order/></Field>
      <Field label="One line per" style="width:280px" help="A category: every value of it gets its own path (for example a site). Nothing: one path for all the samples."><VarSelect v-model="T.group" small :kinds="['category']" none-label="Nothing (one line)" data-som-traj-group/></Field>
      <button type="button" class="btn" data-som-traj-go :disabled="!T.order || (slot && slot.busy)" @click="run"><Icon name="play" size="16"/>Draw the paths</button>
    </div>
    <Callout kind="critical" v-if="slot && slot.error" data-som-traj-error>{{ slot.error }}</Callout></div></div>
    <ChartCard title="Paths over the map" filename="som-trajectories" v-if="mapProps && plane"><SomGrid :map="mapProps" :plane="plane" :lines="lines" :show-colourbar="false" :height="460"/></ChartCard>
  </div>`,
};
