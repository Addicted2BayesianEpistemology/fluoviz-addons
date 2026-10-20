// Step 6 of the Self-organising map page: what will be done (in words), the settings of the check, the button that computes the map, and the way to cancel.
import { computed, watch, onMounted } from 'vue';
import { Icon } from '/static/js/icons.js';
import { Field, NumField, Callout, Switch } from '/static/js/components/Common.js';
import { JobBar } from '/static/js/components/JobBar.js';
import { FpcaFrame } from '/static/js/components/FpcaStepper.js';
import { FpcaExplain } from '/static/js/components/FpcaCommon.js';
import { KERNEL_NAMES, dimOf } from '/static/js/components/FpcaState.js';
import { preName } from '/static/js/components/FpcaStepWhat.js';
import { debounce, fmt } from '/static/js/util.js';
import { fp, ctx, somJobs, requestBody, sampleCount, variablesOf, isFunctional, METRICS, FEW } from './SomState.js';
import { runFit, cancelHeavy, loadDefaults, runMapCheck, dropQueued } from './SomActions.js';
import { nTxt } from './SomCommon.js';

const NICE = { fluorescence: 'Fluorescence', absorbance: 'Absorbance' };

/** The choices of steps 1 to 6 in plain words, one line each. */
export function describeSom(spec = fp.spec, data = fp.data, check = fp.som.check.ans) {
  const m = spec.map, out = [];
  if (m.other.on) {
    out.push(`Variables: ${m.other.keys.length} per-sample variable${m.other.keys.length === 1 ? '' : 's'}${m.other.standardize ? ', standardised' : ''}`);
  } else {
    for (const name of variablesOf(spec)) {
      const v = spec[name], b = v.basis, d = dimOf(name);
      const px = data ? (name === 'fluorescence' ? data.fluorescence.n_pixels : data.absorbance.n_wl) : null;
      const parts = [px ? `${nTxt(px)} ${name === 'fluorescence' ? 'pixels' : 'wavelengths'}` : (name === 'fluorescence' ? 'pixels' : 'wavelengths')];
      if (v.transform !== 'none') parts.push(`pre-treatment: ${preName(name, v.transform).toLowerCase()}`);
      if (b.kind === 'pixels') parts.push('the observed values themselves (pixel version: no basis, no smoothing)');
      else {
        parts.push(b.kind === 'gp' ? `Gaussian-process basis (${KERNEL_NAMES[b.kernel]}${d === 2 ? `, ${b.structure}` : ''})` : b.kind === 'bspline' ? `B-splines (${d === 2 ? `${b.bs[0]} × ${b.bs[1]} per axis` : b.bs[0]})` : `cosines (${d === 2 ? `${b.cos[0]} × ${b.cos[1]} per axis` : b.cos[0]})`);
        const s = v.smooth;
        parts.push(s.method === 'ml' ? 'maximum-likelihood smoothing' : s.method === 'gcv' ? 'smoothing by generalised cross-validation' : s.method === 'fixed' ? 'fixed smoothing' : 'no smoothing');
      }
      out.push(`${NICE[name]}: ${parts.join(', ')}`);
    }
    if (variablesOf(spec).length > 1) out.push(`Weights between the variables: ${spec.weights === 'variance' ? 'each counts equally' : spec.weights === 'none' ? 'natural units' : 'custom'}`);
  }
  const metric = METRICS.find((x) => x.value === m.metric);
  out.push(`Distance: ${metric ? metric.label.toLowerCase() : m.metric}${m.metric === 'mahalanobis' ? ` (${m.components === 'auto' ? 'directions from the data' : `${m.components_n} directions`}, exponent ${m.alpha})` : ''}`);
  const size = check && check.map && check.map.rows ? `${check.map.rows} × ${check.map.cols} units` : (m.size === 'custom' ? `${m.rows} × ${m.cols} units` : `${m.size} size`);
  out.push(`Map: ${size}, ${m.lattice} lattice${m.shape === 'toroid' ? ', wrapping around' : ''}`);
  out.push(`Training: ${m.mode} algorithm, ${m.init === 'linear' ? 'linear' : 'random'} start, ${m.neighbourhood} neighbourhood, ${nTxt(m.epochs_rough)} rough and ${nTxt(m.epochs_fine)} fine epochs`);
  out.push(`Checking: ${check && check.validation ? String(check.validation.words).replace(/\.\s*$/, '') : `${m.folds}-fold validation`}; seed ${m.seed}`);
  out.push(`Groups: ${m.cluster.method === 'hierarchical' ? 'Ward clustering' : m.cluster.method === 'kmeans' ? 'k-means' : 'Gaussian mixture'} of the prototypes, ${m.cluster.k === 'auto' ? 'the number chosen from the data' : `${m.cluster.k_n} groups`}`);
  return out;
}

export const SomStepCompute = {
  name: 'SomStepCompute',
  components: { Icon, Field, NumField, Callout, Switch, JobBar, FpcaFrame, FpcaExplain },
  setup() {
    const m = computed(() => fp.spec.map);
    const A = computed(() => fp.som);
    const running = computed(() => somJobs.fit.state.running);
    const lines = computed(() => describeSom());
    const issues = computed(() => ctx.allIssues());
    const check = computed(() => A.value.check);
    const blocking = computed(() => {
      const a = check.value.ans;
      return a && check.value.key === JSON.stringify(requestBody()) ? (a.blocking || []) : [];
    });
    const nSamples = computed(() => (check.value.ans ? check.value.ans.n : sampleCount()));
    const few = computed(() => { const c = sampleCount(); return !m.value.other.on && c !== null && c < FEW && c >= 4; });
    const time = computed(() => (check.value.ans && check.value.ans.estimate ? check.value.ans.estimate.words : ''));
    const canRun = computed(() => !running.value && !A.value.preparing && !issues.value.length && !blocking.value.length);
    const why = computed(() => (running.value ? 'A map is being computed.' : A.value.preparing ? 'Asking the server for the advice …' : issues.value.length ? `Something needs to be fixed first: ${issues.value[0].message}` : blocking.value[0] || ''));
    const hasResult = computed(() => !!(fp.rid && A.value.res));
    const soon = debounce(() => { runMapCheck(); }, 700);
    onMounted(() => { loadDefaults(); soon(); });
    watch(() => JSON.stringify(requestBody()), () => soon());
    return { fp, m, A, running, lines, issues, blocking, nSamples, few, FEW, time, canRun, why, hasResult, run: () => runFit(), cancelHeavy, dropQueued, somJobs, nTxt,
      DOES: 'Computes the map. The distance between samples is fitted, the units are trained on the samples, the samples are placed on the map, the units are grouped, and the map is checked on samples it was not trained on (k-fold validation: the map is fitted again several times, each time without a part of the samples, and those samples are placed on it).',
      CHECK: 'That the facts on the right are what you expect: the number of samples, the size of the map and the time. Computing takes seconds to a few minutes; Cancel stops it and saves nothing.' };
  },
  template: `
  <FpcaFrame :n="6" next-label="See the results" :no-next="!hasResult" :next-disabled="!hasResult">
    <FpcaExplain :does="DOES" :check="CHECK"/>
    <div class="fp-two fp-two-ctl">
      <div class="col gap-12">
        <div class="card flat fp-summary" data-som-summary>
          <div class="card-head"><h3>What will be done</h3></div>
          <div class="card-body"><ul class="fp-list"><li v-for="l in lines" :key="l">{{ l }}</li></ul>
            <p class="small muted mt-8" v-if="nSamples !== null">{{ nTxt(nSamples) }} samples take part.<span v-if="time"> Time: {{ time }}.</span></p></div>
        </div>
        <details class="fp-more" open><summary>Settings of the check</summary>
          <div class="col gap-12" style="padding-top:8px">
            <div class="pair">
              <Field label="Folds" help="The map is fitted this many times, each time without one fold of the samples; the held-out samples are placed on that map to see how well it describes samples it has not seen."><NumField v-model="m.folds" :min="2" :max="20" small data-som-folds/></Field>
              <Field label="Random seed" help="Everything is reproducible: the same seed gives the same map."><NumField v-model="m.seed" :min="0" small data-som-seed/></Field>
            </div>
            <Field label="Limit for unusual samples" help="A new sample farther from its best unit than this limit is reported as poorly described. The limit comes from samples the map was not trained on."><select class="select sm" v-model.number="m.limit" data-som-limit><option :value="95">95 %</option><option :value="99">99 %</option></select></Field>
            <Field label="Title of the result"><input class="input sm" v-model="m.title" placeholder="(made from the settings)" data-som-title/></Field>
          </div>
        </details>
        <Switch v-if="few" v-model="m.allow_few" label="Compute on fewer than 8 samples anyway" data-som-allow-few/>
        <Callout kind="warning" v-if="few">Only {{ nSamples }} samples: a map of so few samples is a picture, not a model. Read the held-out numbers in step 8 before relying on it.</Callout>
      </div>
      <div class="col gap-12">
        <Callout kind="critical" v-for="b in blocking" :key="b">{{ b }}</Callout>
        <Callout kind="critical" v-if="A.fitError" data-som-fit-error>{{ A.fitError }}</Callout>
        <div class="row wrap">
          <button type="button" class="btn primary" data-som-run :disabled="!canRun" :title="why" @click="run"><Icon name="play" size="16"/>Compute the map</button>
          <button type="button" class="btn danger" v-if="running" data-som-cancel @click="cancelHeavy"><Icon name="stop-circle" size="16"/>Cancel</button>
          <span class="muted xs" v-if="!canRun && why" data-som-why>{{ why }}</span>
        </div>
        <JobBar :job="somJobs.fit" label="Computing the map"/>
        <p class="muted small" v-if="A.queued" data-som-queued>Another computation is running: "{{ A.queued }}" starts right after it (a newer request replaces this one). <button type="button" class="btn xs" @click="dropQueued">Forget it</button></p>
        <p class="muted small" v-if="A.preparing">Asking the server for the advice …</p>
      </div>
    </div>
  </FpcaFrame>`,
};
