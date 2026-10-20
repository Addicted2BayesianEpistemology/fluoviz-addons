// Step 5 of the Self-organising map page: the distance between samples, the size and the shape of the map, the advanced training settings, and the map in numbers.
import { computed, watch, onMounted } from 'vue';
import { Icon } from '/static/js/icons.js';
import { Field, NumField, Callout, Switch, Seg } from '/static/js/components/Common.js';
import { PlotlyChart, ChartCard } from '/static/js/plot.js';
import { FpcaFrame } from '/static/js/components/FpcaStepper.js';
import { FpcaExplain } from '/static/js/components/FpcaCommon.js';
import { tokens, themeState } from '/static/js/theme.js';
import { debounce, fmt } from '/static/js/util.js';
import { fp, ctx, somJobs, requestBody, METRICS, isFunctional, variablesOf } from './SomState.js';
import { loadDefaults, runMapCheck, runScan, heavyBusy } from './SomActions.js';
import { nTxt, pct, SomSection, SomVerdict } from './SomCommon.js';

/** The chart of the map-size scan: error on samples left out, against the number of units. */
export const SomSizeChart = {
  name: 'SomSizeChart',
  components: { PlotlyChart, ChartCard },
  props: { rows: { type: Array, required: true }, suggested: { type: Number, default: null }, height: { type: Number, default: 300 } },
  setup(props) {
    const traces = computed(() => {
      themeState.version;
      const t = tokens(), r = props.rows;
      return [
        { type: 'scatter', mode: 'lines+markers', x: r.map((x) => x.units), y: r.map((x) => x.qe_rel), name: 'Samples the map was trained on', line: { color: t.series[0], width: 2 }, marker: { size: 7, symbol: 'circle' },
          hovertemplate: '%{x} units: %{y:.3f}<extra>trained on</extra>' },
        { type: 'scatter', mode: 'lines+markers', x: r.map((x) => x.units), y: r.map((x) => x.heldout_qe_rel), name: 'Samples left out (honest)', line: { color: t.series[1], width: 2, dash: 'dash' }, marker: { size: 8, symbol: 'diamond' },
          error_y: { type: 'data', array: r.map((x) => x.heldout_se || 0), visible: true, thickness: 1, width: 3, color: t.series[1] }, hovertemplate: '%{x} units: %{y:.3f}<extra>left out</extra>' },
        ...(props.suggested ? [{ type: 'scatter', mode: 'markers', x: [props.suggested], y: [(r.find((x) => x.units === props.suggested) || {}).heldout_qe_rel], name: 'Suggested', marker: { size: 16, symbol: 'circle-open', color: t.text, line: { width: 3, color: t.text } }, hoverinfo: 'skip' }] : []),
      ];
    });
    const layout = computed(() => ({ height: props.height, margin: { l: 60, r: 14, t: 10, b: 84 }, showlegend: true, legend: { orientation: 'h', y: -0.34, x: 0, font: { size: 11 } }, xaxis: { title: { text: 'Number of units' } }, yaxis: { title: { text: 'Share of the variation not captured' }, rangemode: 'tozero' } }));
    const table = computed(() => ({ columns: ['units', 'rows', 'columns', 'not captured (trained on)', 'not captured (left out)', 'topographic error', 'empty units'], rows: props.rows.map((x) => [x.units, x.rows, x.cols, x.qe_rel, x.heldout_qe_rel, x.topographic_error, x.n_empty]) }));
    return { traces, layout, table };
  },
  template: `<ChartCard title="Error against the size of the map" :table="table" filename="som-size-scan"><PlotlyChart :data="traces" :layout="layout" :height="height"/></ChartCard>`,
};

export const SomStepMap = {
  name: 'SomStepMap',
  components: { Icon, Field, NumField, Callout, Switch, Seg, FpcaFrame, FpcaExplain, SomSection, SomVerdict, SomSizeChart },
  setup() {
    const m = computed(() => fp.spec.map);
    const A = computed(() => fp.som);
    const check = computed(() => A.value.check);
    const presets = computed(() => (A.value.defaults && A.value.defaults.advice ? A.value.defaults.advice.presets : null));
    const gp = computed(() => !m.value.other.on && isFunctional(fp.spec) && variablesOf(fp.spec).every((n) => fp.spec[n].basis.kind === 'gp'));
    const metrics = computed(() => METRICS.map((x) => ({ ...x, disabled: (x.value === 'prior' && !gp.value) || (x.value === 'prior' && m.value.other.on),
      why: x.value === 'prior' && !gp.value ? 'Needs the Gaussian-process basis for every variable (step 3).' : '' })));
    const ans = computed(() => (check.value.ans && check.value.key === JSON.stringify(requestBody()) ? check.value.ans : check.value.ans));
    const sizeScan = computed(() => A.value.scans.size || null);
    const metricScan = computed(() => A.value.scans.metric || null);
    const scanBusy = computed(() => somJobs.scan.state.running);
    function pickMetric(v) { const x = metrics.value.find((q) => q.value === v); if (x && !x.disabled) m.value.metric = v; }
    function pickSize(v) { m.value.size = v; }
    function useSize(row) { m.value.size = 'custom'; m.value.rows = row.rows; m.value.cols = row.cols; }
    const soon = debounce(() => { runMapCheck(); }, 700);
    onMounted(() => { loadDefaults(); soon(); });
    watch(() => JSON.stringify(requestBody()), () => soon());
    const sizeSentence = computed(() => {
      const a = ans.value; if (!a || !a.map || !a.map.rows) return '';
      return `${a.map.rows} rows × ${a.map.cols} columns = ${a.map.units} units for ${a.n} samples (${fmt(a.map.samples_per_unit, 2)} samples per unit).`;
    });
    return { fp, m, A, check, presets, gp, metrics, ans, sizeScan, metricScan, scanBusy, pickMetric, pickSize, useSize, runScan, heavyBusy, nTxt, pct, sizeSentence, fmt, ctx,
      DOES: 'Chooses how samples are compared, how many units the map has and how they are arranged. The distance decides which samples count as similar; the size decides how finely the map divides the samples; training then pulls the prototypes towards the samples while keeping neighbouring units alike.',
      CHECK: 'That the number of units fits your number of samples: with far more units than samples, most units hold no sample and simply interpolate between their neighbours (fine for a picture, but the held-out numbers in step 8 will tell you if it is too fine). The map-size scan finds a size from your data.' };
  },
  template: `
  <FpcaFrame :n="5" class="som-step5">
    <FpcaExplain :does="DOES" :check="CHECK"/>
    <div class="fp-two fp-two-ctl">
      <div class="col gap-12">
        <SomSection title="Distance between samples" sentence="How two samples are compared. The plain distance is the usual one; the standardised distances give subtle differences more weight.">
          <div class="fp-cards" role="radiogroup" aria-label="Distance between samples">
            <div v-for="x in metrics" :key="x.value" class="fp-card" :class="{ on: m.metric === x.value, off: x.disabled }" role="radio" :aria-checked="m.metric === x.value ? 'true' : 'false'" tabindex="0"
                 :aria-disabled="x.disabled ? 'true' : null" :data-som-metric="x.value" :title="x.why" @click="pickMetric(x.value)" @keydown.enter.prevent="pickMetric(x.value)" @keydown.space.prevent="pickMetric(x.value)">
              <span class="fp-card-head"><b>{{ x.label }}</b><Icon v-if="m.metric === x.value" name="check-circle" size="18" class="fp-tick"/></span>
              <span class="fp-card-text">{{ x.text }}</span>
              <span class="fp-card-why" v-if="x.disabled">{{ x.why }}</span>
            </div>
          </div>
          <div class="pair" v-if="m.metric === 'mahalanobis'" data-som-mahalanobis>
            <Field label="Directions to standardise" help="How many leading directions of variation count as one unit each. Everything beyond them counts as one unit in total.">
              <Seg v-model="m.components" :options="[{ value: 'auto', label: 'From the data' }, { value: 'n', label: 'I choose' }]" small/>
              <NumField v-if="m.components !== 'auto'" v-model="m.components_n" :min="1" :max="200" :step="1" small/>
              <div class="hint" v-else-if="ans && ans.metric && ans.metric.components">The data advise {{ ans.metric.components }} (the smallest number that rebuilds held-out samples as well as the best).</div>
            </Field>
            <Field label="How far to standardise" help="0 is the plain distance; 1 gives every one of the leading directions the same weight; in between is a compromise.">
              <NumField v-model="m.alpha" :min="0" :max="1" :step="0.1" small data-som-alpha/>
            </Field>
          </div>
          <div class="row wrap">
            <button type="button" class="btn" data-som-metric-scan :disabled="scanBusy || heavyBusy()" @click="runScan('metric', { source: 'request' })"><Icon name="chart-line" size="16"/>Compare the distances</button>
            <span class="muted xs" v-if="metricScan && metricScan.queued">Another computation is running: this one starts right after it.</span>
          </div>
          <Callout kind="critical" v-if="metricScan && metricScan.error">{{ metricScan.error }}</Callout>
          <div class="table-wrap" v-if="metricScan && metricScan.ans" data-som-metric-table>
            <table class="table"><thead><tr><th>Distance</th><th>Neighbours kept (trustworthiness)</th><th>Topographic error</th><th>Share not captured on held-out samples</th><th>Groups agree with the plain distance</th></tr></thead>
              <tbody><tr v-for="r in metricScan.ans.rows" :key="r.kind + r.alpha"><td>{{ r.kind }}<span v-if="r.kind === 'mahalanobis'"> (exponent {{ r.alpha }})</span></td>
                <template v-if="r.error"><td colspan="4">{{ r.error }}</td></template>
                <template v-else><td class="num">{{ fmt(r.trustworthiness, 3) }}</td><td class="num">{{ pct(r.topographic_error, 0) }}</td><td class="num">{{ fmt(r.heldout_qe_rel, 3) }}</td><td class="num">{{ fmt(r.agreement, 2) }}</td></template></tr></tbody></table>
            <p class="fp-note muted">Adjusted Rand index of the groups against those of the plain distance (1 = the same groups). The held-out error is measured in the plain data units, so the distances can be compared.</p>
          </div>
        </SomSection>

        <SomSection title="Size of the map" sentence="How many units the map has. With more units than samples, most units interpolate between their neighbours.">
          <div class="fp-cards" role="radiogroup" aria-label="Size of the map">
            <div v-for="(label, key) in { compact: 'Compact', standard: 'Standard', fine: 'Fine' }" :key="key" class="fp-card" :class="{ on: m.size === key }" role="radio" :aria-checked="m.size === key ? 'true' : 'false'" tabindex="0"
                 :data-som-size="key" @click="pickSize(key)" @keydown.enter.prevent="pickSize(key)" @keydown.space.prevent="pickSize(key)">
              <span class="fp-card-head"><b>{{ label }}</b><Icon v-if="m.size === key" name="check-circle" size="18" class="fp-tick"/></span>
              <span class="fp-card-info" v-if="presets && presets[key]">{{ presets[key].rows }} × {{ presets[key].cols }} = {{ presets[key].units }} units</span>
              <span class="fp-card-text" v-if="presets && presets[key]">{{ presets[key].words }}</span>
            </div>
            <div class="fp-card" :class="{ on: m.size === 'custom' }" role="radio" :aria-checked="m.size === 'custom' ? 'true' : 'false'" tabindex="0" data-som-size="custom" @click="pickSize('custom')" @keydown.enter.prevent="pickSize('custom')">
              <span class="fp-card-head"><b>Custom</b><Icon v-if="m.size === 'custom'" name="check-circle" size="18" class="fp-tick"/></span>
              <span class="fp-card-text">Choose the number of rows and columns yourself.</span>
            </div>
          </div>
          <div class="pair" v-if="m.size === 'custom'"><Field label="Rows"><NumField v-model="m.rows" :min="2" :max="200" small data-som-rows/></Field><Field label="Columns"><NumField v-model="m.cols" :min="2" :max="200" small data-som-cols/></Field></div>
          <p class="fp-note" v-if="sizeSentence" data-som-size-sentence>{{ sizeSentence }}<span v-if="ans && ans.map && ans.map.samples_per_unit < 1"> With fewer samples than units, most units will hold no sample: they interpolate between their neighbours. That is normal for a map; choose Compact if you want every unit to stand for real samples.</span></p>
          <div class="row wrap">
            <button type="button" class="btn" data-som-size-scan :disabled="scanBusy || heavyBusy()" @click="runScan('size', { source: 'request' })"><Icon name="chart-line" size="16"/>Map-size scan</button>
            <span class="muted xs" v-if="sizeScan && sizeScan.queued">Another computation is running: this one starts right after it.</span>
          </div>
          <Callout kind="critical" v-if="sizeScan && sizeScan.error">{{ sizeScan.error }}</Callout>
          <template v-if="sizeScan && sizeScan.ans">
            <SomVerdict level="info">{{ sizeScan.ans.verdict }}</SomVerdict>
            <SomSizeChart :rows="sizeScan.ans.rows" :suggested="sizeScan.ans.suggested"/>
            <div class="row wrap" v-if="sizeScan.ans.suggested"><button type="button" class="btn" data-som-use-size @click="useSize(sizeScan.ans.rows.find((x) => x.units === sizeScan.ans.suggested))">Use {{ sizeScan.ans.suggested }} units</button></div>
          </template>
        </SomSection>

        <SomSection title="Shape of the map" sentence="Hexagons have six neighbours and give the most even picture. A map that wraps around has no border: use it when the samples have no natural edge.">
          <div class="pair">
            <Field label="Lattice"><Seg v-model="m.lattice" :options="[{ value: 'hexagonal', label: 'Hexagonal' }, { value: 'rectangular', label: 'Rectangular' }]" small data-som-lattice/></Field>
            <Field label="Edges"><Seg v-model="m.shape" :options="[{ value: 'sheet', label: 'Flat' }, { value: 'toroid', label: 'Wraps around (toroid)' }]" small data-som-shape/></Field>
          </div>
        </SomSection>

        <details class="fp-more" data-som-advanced><summary>Advanced training settings</summary>
          <div class="col gap-12" style="padding-top:8px">
            <p class="fp-note">The defaults are the standard way to train a map (Kohonen 2013, the SOM Toolbox): leave them alone unless the held-out numbers or the training curve in step 8 ask for a change.</p>
            <div class="pair">
              <Field label="Training mode" help="Batch is deterministic and fast. Sequential shows one sample at a time, as in the first publications, and depends on the random seed."><Seg v-model="m.mode" :options="[{ value: 'batch', label: 'Batch' }, { value: 'sequential', label: 'Sequential' }]" small data-som-mode/></Field>
              <Field label="Starting point" help="Linear: the prototypes start spread over the plane of the first two principal directions (deterministic). Random: they start at random samples."><Seg v-model="m.init" :options="[{ value: 'linear', label: 'Linear' }, { value: 'random', label: 'Random' }]" small data-som-init/></Field>
            </div>
            <Field label="Neighbourhood function" help="How strongly a unit near the winner is pulled along. Gaussian is the usual choice."><select class="select sm" v-model="m.neighbourhood" data-som-neighbourhood><option value="gaussian">Gaussian</option><option value="cutgauss">Gaussian, cut off at the radius</option><option value="bubble">Bubble (all or nothing)</option><option value="epanechnikov">Epanechnikov</option></select></Field>
            <div class="pair">
              <Field label="Epochs, rough phase" help="The first phase, with a large radius, puts the map in order."><NumField v-model="m.epochs_rough" :min="1" :max="2000" small data-som-epochs-rough/></Field>
              <Field label="Epochs, fine phase" help="The second phase, with a small radius, fits the prototypes."><NumField v-model="m.epochs_fine" :min="1" :max="5000" small data-som-epochs-fine/></Field>
            </div>
            <div class="pair">
              <Field label="Starting radius" help="How far the pull reaches at the start, in lattice steps. Automatic: a quarter of the longer side."><Seg v-model="m.radius_initial" :options="[{ value: 'auto', label: 'Automatic' }, { value: 'n', label: 'I choose' }]" small/><NumField v-if="m.radius_initial !== 'auto'" v-model="m.radius_initial_n" :min="0.5" :max="100" :step="0.5" small/></Field>
              <Field label="Final radius" help="How far the pull reaches at the end. 1 is the usual value."><NumField v-model="m.radius_final" :min="0.3" :max="5" :step="0.1" small data-som-radius-final/></Field>
            </div>
            <div class="pair" v-if="m.mode === 'sequential'">
              <Field label="Learning rate, start"><NumField v-model="m.alpha_initial" :min="0.0001" :max="1" :step="0.01" small/></Field>
              <Field label="Learning rate, end"><NumField v-model="m.alpha_final" :min="0.00001" :max="1" :step="0.01" small/></Field>
            </div>
            <Field label="Samples that lack part of the domain" help="A sample with a rectangle cut out has coefficients estimated from the pixels it has and from the other samples. Weighted, it counts in proportion to what was measured.">
              <Seg v-model="m.gap_weighting" :options="[{ value: 'observed', label: 'Weigh by what was measured' }, { value: 'equal', label: 'Count every sample alike' }]" small data-som-gap-weighting/>
            </Field>
          </div>
        </details>
      </div>

      <div class="col gap-12">
        <div class="card flat fp-summary" data-som-numbers>
          <div class="card-head"><h3>The map in numbers</h3></div>
          <div class="card-body">
            <p class="muted small" v-if="check.busy && !ans">Working it out …</p>
            <Callout kind="critical" v-if="check.error">{{ check.error }}</Callout>
            <template v-if="ans">
              <dl class="fp-facts">
                <dt>Samples</dt><dd>{{ nTxt(ans.n) }}</dd>
                <dt>Coordinates</dt><dd>{{ nTxt(ans.input.n_inputs) }} ({{ ans.input.mode }})</dd>
                <template v-if="ans.map && ans.map.rows"><dt>Map</dt><dd>{{ ans.map.rows }} × {{ ans.map.cols }} = {{ ans.map.units }} units</dd><dt>Samples per unit</dt><dd>{{ fmt(ans.map.samples_per_unit, 2) }}</dd></template>
                <dt v-if="ans.metric">Distance</dt><dd v-if="ans.metric">{{ ans.metric.name }}<span v-if="ans.metric.components"> ({{ ans.metric.components }} directions)</span></dd>
                <dt>Checking</dt><dd>{{ ans.validation.words }}</dd>
                <dt>Time</dt><dd>{{ ans.estimate.words }}</dd>
              </dl>
              <Callout kind="warning" v-for="w in ans.warnings" :key="w">{{ w }}</Callout>
              <Callout kind="critical" v-for="b in ans.blocking" :key="b">{{ b }}</Callout>
            </template>
          </div>
        </div>
      </div>
    </div>
  </FpcaFrame>`,
};
