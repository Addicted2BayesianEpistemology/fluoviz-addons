// Step 8 of the Self-organising map page: how far the map can be trusted. Tabs: Quality, Held-out error, Map size, Stability, Distances, Training.
import { computed } from 'vue';
import { Icon } from '/static/js/icons.js';
import { Callout, Tabs, NumField, Field, Switch } from '/static/js/components/Common.js';
import { PlotlyChart, ChartCard } from '/static/js/plot.js';
import { FpcaFrame } from '/static/js/components/FpcaStepper.js';
import { FpcaExplain } from '/static/js/components/FpcaCommon.js';
import { tokens, themeState } from '/static/js/theme.js';
import { fmt } from '/static/js/util.js';
import { fp, somJobs } from './SomState.js';
import { runScan, heavyBusy } from './SomActions.js';
import { SomStat, SomVerdict, pct, nTxt } from './SomCommon.js';
import { SomGate, SomStaleNote } from './SomResults.js';
import { SomSizeChart } from './SomStepMap.js';

export const SomChecks = {
  name: 'SomChecks',
  components: { Icon, Callout, Tabs, NumField, Field, Switch, PlotlyChart, ChartCard, FpcaFrame, FpcaExplain, SomGate, SomStaleNote, SomStat, SomVerdict, SomSizeChart },
  setup() {
    const A = computed(() => fp.som), R = computed(() => fp.som.res), SV = computed(() => fp.som.scanView);
    const ready = computed(() => !!(R.value && A.value.resRid));
    const q = computed(() => (R.value ? R.value.quality : null)), v = computed(() => (R.value ? R.value.verdicts : null));
    const tabs = [{ id: 'quality', label: 'Quality' }, { id: 'heldout', label: 'Held-out error' }, { id: 'size', label: 'Map size' }, { id: 'stability', label: 'Stability' }, { id: 'distances', label: 'Distances' }, { id: 'training', label: 'Training' }];
    const scan = (k) => computed(() => (A.value.scans[k] && A.value.scans[k].rid === A.value.resRid ? A.value.scans[k] : null));
    const sizeScan = scan('size'), stabScan = scan('stability'), metScan = scan('metric');
    const busy = computed(() => somJobs.scan.state.running || heavyBusy());
    const runSize = () => runScan('size', { source: 'result' });
    const runStab = () => runScan('stability', { source: 'result', options: { n_runs: Number(SV.value.n_runs), bootstrap: !!SV.value.bootstrap, k: Number(SV.value.k) } });
    const runMet = () => runScan('metric', { source: 'result', options: { k: Number(SV.value.k) } });
    // the held-out distances against the in-sample ones
    const heldTraces = computed(() => {
      themeState.version;
      const r = R.value; if (!r) return [];
      const t = tokens(), x = Array.from(r.qe), y = Array.from(r.held), text = r.ids, ok = y.map((u) => Number.isFinite(u));
      const hi = Math.max(...x.filter(Number.isFinite), ...y.filter(Number.isFinite), 0);
      return [{ type: 'scatter', mode: 'lines', x: [0, hi], y: [0, hi], line: { color: t.line, dash: 'dot' }, hoverinfo: 'skip', showlegend: false },
        { type: 'scatter', mode: 'markers', x: x.filter((_, i) => ok[i]), y: y.filter((_, i) => ok[i]), text: text.filter((_, i) => ok[i]), marker: { size: 8, color: t.series[0], line: { color: t.surface, width: 1 } }, hovertemplate: '%{text}<br>trained on: %{x:.3g}<br>left out: %{y:.3g}<extra></extra>', showlegend: false }];
    });
    const heldLayout = { height: 340, margin: { l: 60, r: 14, t: 10, b: 54 }, xaxis: { title: { text: 'Distance to the best unit, sample used for the map' } }, yaxis: { title: { text: 'Distance to the best unit, sample left out' } }, hovermode: 'closest' };
    const heldTable = computed(() => { const r = R.value; return { columns: ['sample', 'unit', 'distance (trained on)', 'distance (left out)'], rows: r ? r.ids.map((id, i) => [id, r.unitNames[r.unit[i]], r.qe[i], r.held[i]]) : [] }; });
    const histTraces = computed(() => {
      themeState.version;
      const h = R.value && R.value.history; if (!h) return [];
      const t = tokens(), e = h.energy, n = e.length, x = Array.from({ length: n }, (_, i) => i + 1);
      const phases = h.phase; const split = phases.indexOf('fine');
      return [{ type: 'scatter', mode: 'lines', x, y: e, name: 'Energy of the map', line: { color: t.series[0], width: 2 }, hovertemplate: 'epoch %{x}: %{y:.4g}<extra></extra>' },
        { type: 'scatter', mode: 'lines', x, y: h.qe, name: 'Mean distance to the best unit', yaxis: 'y2', line: { color: t.series[1], width: 2, dash: 'dash' }, hovertemplate: 'epoch %{x}: %{y:.4g}<extra></extra>' },
        ...(split > 0 ? [{ type: 'scatter', mode: 'lines', x: [split + 0.5, split + 0.5], y: [Math.min(...e), Math.max(...e)], line: { color: t.text2 || t.line, dash: 'dot', width: 1.5 }, hoverinfo: 'skip', showlegend: false }] : [])];
    });
    const histLayout = { height: 320, margin: { l: 60, r: 60, t: 10, b: 84 }, showlegend: true, legend: { orientation: 'h', y: -0.3, x: 0 }, xaxis: { title: { text: 'Epoch (rough phase, then fine phase)' } }, yaxis: { title: { text: 'Energy' } }, yaxis2: { overlaying: 'y', side: 'right', title: { text: 'Distance' } } };
    const histTable = computed(() => { const h = R.value && R.value.history; return { columns: ['epoch', 'phase', 'radius', 'energy', 'mean distance'], rows: h ? h.energy.map((e, i) => [i + 1, h.phase[i], h.radius[i], e, h.qe[i]]) : [] }; });
    return { A, R, SV, ready, q, v, tabs, sizeScan, stabScan, metScan, busy, runSize, runStab, runMet, heldTraces, heldLayout, heldTable, histTraces, histLayout, histTable, pct, nTxt, fmt,
      DOES: 'Puts numbers on how far the map can be trusted. The quality numbers say how well the prototypes capture the samples and whether the map keeps neighbours together; the held-out error is the honest one, measured on samples the map was not trained on; the scans show what a different size, a different start or a different distance would have given.',
      CHECK: 'That the held-out error is not much larger than the error on the training samples (otherwise the map is finer than the samples can support), that the topographic error is small, and that the map keeps neighbours together at least as well as the scatter of the first two components.' };
  },
  template: `
  <FpcaFrame :n="8" next-label="Use" :no-next="!ready" :next-disabled="!ready">
    <FpcaExplain :does="DOES" :check="CHECK"/>
    <SomGate/>
    <template v-if="ready">
      <SomStaleNote/>
      <Tabs v-model="A.tab8" :tabs="tabs" data-som-tabs8/>
      <div v-if="A.tab8 === 'quality'" class="col gap-12" data-som-quality>
        <div class="fp-stats">
          <SomStat label="Captured (trained on)" :value="pct(q.captured, 1)" :sub="'mean distance ' + fmt(q.qe_mean, 3)"/>
          <SomStat label="Captured (left out)" :value="q.heldout_captured !== undefined ? pct(q.heldout_captured, 1) : '–'" :sub="q.heldout_over_train !== undefined ? fmt(q.heldout_over_train, 2) + ' times the distance' : ''"/>
          <SomStat label="Topographic error" :value="pct(q.topographic_error, 0)" help="The share of samples whose two best units are not neighbours: 0 % means the map is not torn."/>
          <SomStat label="Topographic product" :value="fmt(q.topographic_product, 3)" help="Near 0: the map has the right dimension. Negative: too flat (it folds). Positive: larger than the data need."/>
          <SomStat label="Trustworthiness" :value="fmt(q.trust_map.trustworthiness, 3)" :sub="'scatter of the first two components: ' + fmt(q.trust_pca.trustworthiness, 3)" help="1 = the nearest neighbours of each sample on the picture are its nearest neighbours in the data."/>
          <SomStat label="Continuity" :value="fmt(q.trust_map.continuity, 3)" :sub="'scatter: ' + fmt(q.trust_pca.continuity, 3)" help="1 = the nearest neighbours in the data stay neighbours on the picture."/>
        </div>
        <SomVerdict :level="v.vs_pca.level" data-som-v-pca>{{ v.vs_pca.verdict }}</SomVerdict>
        <SomVerdict :level="v.topographic.level" data-som-v-topo>{{ v.topographic.verdict }}</SomVerdict>
        <SomVerdict :level="v.heldout.level" data-som-v-held>{{ v.heldout.verdict }}</SomVerdict>
        <p class="fp-note">{{ v.empty }} With {{ q.n_units }} units for {{ R.n }} samples there are {{ fmt(q.samples_per_unit, 2) }} samples per unit.</p>
        <Callout kind="warning" v-for="w in R.p.warnings" :key="w">{{ w }}</Callout>
        <p class="fp-note muted" v-for="n in R.p.notes" :key="n">{{ n }}</p>
      </div>
      <div v-else-if="A.tab8 === 'heldout'" class="col gap-12" data-som-heldout>
        <p class="fp-note">The map was fitted {{ R.p.kfold.folds }} times; each time a part of the samples was left out and then placed on the map. The distance of such a sample to its best unit is an honest measure. Samples above the diagonal are described worse by a map that has not seen them.</p>
        <SomVerdict :level="v.heldout.level">{{ v.heldout.verdict }}</SomVerdict>
        <ChartCard title="Distance to the best unit: trained on and left out" :table="heldTable" filename="som-heldout"><PlotlyChart :data="heldTraces" :layout="heldLayout" :height="340"/></ChartCard>
      </div>
      <div v-else-if="A.tab8 === 'size'" class="col gap-12" data-som-size-tab>
        <p class="fp-note">The same settings with maps of several sizes: the smallest map whose error on samples left out is within one standard error of the best is suggested. Fitting 6 to 8 maps takes a little while.</p>
        <div class="row wrap"><button type="button" class="btn" data-som-run-size :disabled="busy" @click="runSize"><Icon name="chart-line" size="16"/>Map-size scan</button><span class="muted xs" v-if="sizeScan && sizeScan.queued">Another computation is running: this one starts right after it.</span></div>
        <Callout kind="critical" v-if="sizeScan && sizeScan.error">{{ sizeScan.error }}</Callout>
        <template v-if="sizeScan && sizeScan.ans"><SomVerdict level="info">{{ sizeScan.ans.verdict }}</SomVerdict><SomSizeChart :rows="sizeScan.ans.rows" :suggested="sizeScan.ans.suggested"/></template>
      </div>
      <div v-else-if="A.tab8 === 'stability'" class="col gap-12" data-som-stability>
        <p class="fp-note">The map is trained again from other random starting points (and, if you like, on resampled samples). Pairs of samples that sit together on the map you see should stay together; the groups should stay the same.</p>
        <div class="row wrap" style="gap:12px 22px;align-items:flex-end">
          <Field label="Number of runs" style="width:140px"><NumField v-model="SV.n_runs" :min="2" :max="50" small/></Field>
          <Field label="Groups compared" style="width:150px"><NumField v-model="SV.k" :min="2" :max="20" small/></Field>
          <Switch v-model="SV.bootstrap" label="Resample the samples (bootstrap)"/>
          <button type="button" class="btn" data-som-run-stability :disabled="busy" @click="runStab"><Icon name="play" size="16"/>Check the stability</button>
        </div>
        <Callout kind="critical" v-if="stabScan && stabScan.error">{{ stabScan.error }}</Callout>
        <template v-if="stabScan && stabScan.ans">
          <SomVerdict :level="stabScan.ans.verdict.level" data-som-v-stability>{{ stabScan.ans.verdict.verdict }}</SomVerdict>
          <div class="fp-stats"><SomStat label="Pairs that stay together" :value="pct(stabScan.ans.value, 0)" :sub="'chance: ' + pct(stabScan.ans.baseline, 0)"/><SomStat label="Agreement of the groups" :value="fmt(stabScan.ans.ari, 2)" sub="adjusted Rand index"/><SomStat label="Runs" :value="stabScan.ans.n_runs"/></div>
        </template>
      </div>
      <div v-else-if="A.tab8 === 'distances'" class="col gap-12" data-som-distances>
        <p class="fp-note">The same map under each distance: how well it keeps neighbours together, how large the error on samples left out is (in plain data units, so the distances can be compared), and whether the groups agree with the groups of the plain distance.</p>
        <div class="row wrap"><button type="button" class="btn" data-som-run-metric :disabled="busy" @click="runMet"><Icon name="chart-line" size="16"/>Compare the distances</button></div>
        <Callout kind="critical" v-if="metScan && metScan.error">{{ metScan.error }}</Callout>
        <div class="table-wrap" v-if="metScan && metScan.ans"><table class="table"><thead><tr><th>Distance</th><th>Trustworthiness</th><th>Continuity</th><th>Topographic error</th><th>Not captured, held out</th><th>Groups agree</th></tr></thead>
          <tbody><tr v-for="r in metScan.ans.rows" :key="r.kind + r.alpha"><td>{{ r.kind }}<span v-if="r.kind === 'mahalanobis'"> (exponent {{ r.alpha }})</span></td>
            <template v-if="r.error"><td colspan="5">{{ r.error }}</td></template><template v-else><td class="num">{{ fmt(r.trustworthiness, 3) }}</td><td class="num">{{ fmt(r.continuity, 3) }}</td><td class="num">{{ pct(r.topographic_error, 0) }}</td><td class="num">{{ fmt(r.heldout_qe_rel, 3) }}</td><td class="num">{{ fmt(r.agreement, 2) }}</td></template></tr></tbody></table></div>
      </div>
      <div v-else-if="A.tab8 === 'training'" class="col gap-12" data-som-training>
        <SomVerdict :level="v.training.level" data-som-v-training>{{ v.training.verdict }}</SomVerdict>
        <ChartCard title="The energy of the map during training" :table="histTable" filename="som-training"><PlotlyChart :data="histTraces" :layout="histLayout" :height="320"/></ChartCard>
        <p class="fp-note muted">The first epochs (left of the dotted line) are the rough phase, with a large radius: they put the map in order. The fine phase fits the prototypes. The energy is the distance of the samples to the prototypes, weighted by the neighbourhood.</p>
      </div>
    </template>
  </FpcaFrame>`,
};
