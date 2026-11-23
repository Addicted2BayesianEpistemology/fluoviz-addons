// Step 8 of the Multimodal latent factors page: how far the model can be trusted, and how to use it elsewhere.
//   Convergence: the verdict, the R-hat and effective sample sizes of every rotation-invariant quantity, the log density of every chain, the divergent transitions.
//   Basis: what each basis leaves out, against measurement noise passed through the same projection (mblf's projection residuals).
//   Use: the variables the model made, saving it as a model for new samples, the downloads, the methods paragraph.
import { computed, onMounted, watch } from 'vue';
import { Icon } from '/static/js/icons.js';
import { Callout, Seg } from '/static/js/components/Common.js';
import { PlotlyChart, ChartCard } from '/static/js/plot.js';
import { FpcaFrame } from '/static/js/components/FpcaStepper.js';
import { FpcaExplain } from '/static/js/components/FpcaCommon.js';
import { SaveModel } from '/static/js/components/SaveModel.js';
import { go } from '/static/js/router.js';
import { toast } from '/static/js/store.js';
import { catalogue, loadCatalogue } from '/static/js/variables.js';
import { tokens, themeState } from '/static/js/theme.js';
import { fp, ctx } from './MblfState.js';
import { loadMethods, exportFile } from './MblfActions.js';
import { MblfVerdict, MblfSection, MblfModeView, nd, arr, modeOf, modeLabel, num, nTxt } from './MblfCommon.js';
import { MblfGate, MblfStaleNote } from './MblfResults.js';

const MblfConvergence = {
  name: 'MblfConvergence',
  components: { MblfVerdict, PlotlyChart, ChartCard, Callout },
  setup() {
    const p = computed(() => fp.mb.res.p);
    const d = computed(() => p.value.diagnostics);
    const lvl = (v) => (v === 'ok' ? 'good' : v === 'warn' ? 'warning' : 'critical');
    const lpTraces = computed(() => {
      themeState.version;
      const lp = nd(d.value.lp);
      if (!lp || lp.shape.length < 2) return [];
      const [C, n] = lp.shape, t = tokens();
      return Array.from({ length: C }, (_, c) => ({ y: Array.from(lp.data.subarray(c * n, (c + 1) * n)), type: 'scatter', mode: 'lines', name: `Chain ${c + 1}`, line: { width: 1, color: t.series[c % 8] } }));
    });
    const lpLayout = { xaxis: { title: 'Iteration (kept, thinned)' }, yaxis: { title: 'Log density' }, margin: { t: 10 }, legend: { orientation: 'h', y: -0.25 } };
    return { p, d, lvl, lpTraces, lpLayout, num, nTxt };
  },
  template: `
  <div class="col gap-12" data-mblf-convergence>
    <MblfVerdict :level="lvl(d.verdict.level)">{{ d.verdict.text }}</MblfVerdict>
    <ul class="fp-list small" v-if="d.verdict.items.length"><li v-for="t in d.verdict.items" :key="t">{{ t }}</li></ul>
    <template v-if="p.estimation === 'bayes'">
      <p class="fp-note">The factors are only defined up to a rotation, so convergence is checked on quantities that do not depend on it (as in the appendix of the method): the log density, the noise and residual variances, the coefficients of every sample, the shared variance of every coefficient and of every mode, the strengths of the factors, the cross-covariance between modes and the effects of the covariates. R-hat at most 1.01 and at least 400 effective draws are the usual thresholds.</p>
      <div class="table-wrap"><table class="table" data-mblf-rhat><thead><tr><th>Quantity</th><th>How many</th><th>Largest R-hat</th><th>Smallest bulk ESS</th><th>Smallest tail ESS</th></tr></thead>
        <tbody><tr v-for="r in d.table" :key="r.quantity"><td>{{ r.label }}</td><td>{{ nTxt(r.n) }}</td><td :class="{ bad: r.max_rhat > 1.01 }">{{ num(r.max_rhat, 4) }}</td><td :class="{ bad: r.min_ess_bulk < 400 }">{{ num(r.min_ess_bulk, 3) }}</td><td :class="{ bad: r.min_ess_tail < 400 }">{{ num(r.min_ess_tail, 3) }}</td></tr></tbody></table></div>
      <p class="small">{{ d.n_divergent }} divergent transition(s); {{ d.n_max_treedepth }} iteration(s) at the largest tree depth; E-BFMI of the chains {{ (d.ebfmi || []).map((x) => x.toFixed(2)).join(', ') }}; {{ d.n_noncentered }} coefficient(s) sampled in the non-centred form.</p>
      <ChartCard title="The log density of every chain" help="Well-mixed chains overlap and look like noise around the same level; a chain that stays apart did not find the same region." :data="lpTraces" flat><PlotlyChart :data="lpTraces" :layout="lpLayout" :height="260"/></ChartCard>
    </template>
    <template v-else>
      <p class="small">Maximum likelihood: log-likelihood {{ num(d.loglik, 6) }}, BIC {{ num(d.bic, 6) }}, {{ d.n_starts_at_optimum }} of {{ d.n_starts }} starting points reached the best value.</p>
    </template>
  </div>`,
};

const MblfBasisCheck = {
  name: 'MblfBasisCheck',
  components: { MblfVerdict, MblfModeView, Callout },
  setup() {
    const p = computed(() => fp.mb.res.p);
    const summary = (b) => ({ mean: b.ratio, quantiles: [b.ratio, b.ratio, b.ratio, b.ratio, b.ratio, b.ratio, b.ratio] });
    return { p, summary, modeOf: (n) => modeOf(fp.mb.res, n), modeLabel, num };
  },
  template: `
  <div class="col gap-12" data-mblf-basis>
    <p class="fp-note">What the basis cannot represent of a sample should be measurement noise. For every mode, the residual of every sample after its projection on the basis is compared with Gaussian noise at the noise level of the posterior, passed through the same projection: the residual energy of the samples, the profile over the domain (the picture: mean squared residual over its expectation, 1 = like noise) and the lag-1 autocorrelation along the wavelengths (structure left over shows as a positive correlation).</p>
    <div v-for="b in p.basis_check" :key="b.mode" class="col gap-8">
      <h3 class="fp-h3">{{ modeLabel(modeOf(b.mode)) }}</h3>
      <Callout kind="info" v-if="!b.available">{{ b.text }}</Callout>
      <template v-else>
        <MblfVerdict :level="b.level === 'ok' ? 'good' : 'warning'">{{ b.text }}</MblfVerdict>
        <div class="table-wrap"><table class="table"><thead><tr><th>Check</th><th>Observed</th><th>Expected under noise</th><th>90 % range of the noise</th><th>p-value</th></tr></thead>
          <tbody><tr v-for="r in b.rows" :key="r.statistic"><td>{{ r.statistic }}</td><td>{{ num(r.observed, 3) }}</td><td>{{ num(r.expected, 3) }}</td><td>{{ num(r.lower, 3) }} – {{ num(r.upper, 3) }}</td><td :class="{ bad: r.p_value < 0.01 }">{{ num(r.p_value, 2) }}</td></tr></tbody></table></div>
        <MblfModeView v-if="modeOf(b.mode).kind !== 'vector'" :mode="modeOf(b.mode)" :summary="summary(b)" :stats="['mean']" title="Mean squared residual over its expectation under noise (1 = like noise)" units="ratio" :simultaneous="false" compact/>
      </template>
    </div>
  </div>`,
};

const MblfUse = {
  name: 'MblfUse',
  components: { Icon, Callout, SaveModel, MblfSection },
  setup() {
    const rid = computed(() => fp.mb.resRid);
    onMounted(() => { loadCatalogue(true); loadMethods(); });
    watch(rid, () => loadMethods());
    const vars = computed(() => {
      const pre = `res:${rid.value}:`;
      const fromCat = catalogue.value.filter((v) => v.key.startsWith(pre));
      if (fromCat.length) return fromCat.map((v) => ({ key: v.key, name: v.key.slice(pre.length), label: v.label }));
      return ((fp.mb.res && fp.mb.res.result.variables) || []).map((n) => ({ key: pre + n, name: n, label: n }));
    });
    const methods = computed(() => fp.mb.extra[`methods|${rid.value}`] || null);
    const text = computed(() => (methods.value && methods.value.data ? methods.value.data.text : ''));
    const refs = computed(() => (methods.value && methods.value.data ? methods.value.data.references || [] : []));
    async function copy() {
      try { await navigator.clipboard.writeText(text.value); toast('The methods paragraph was copied.', 'ok', '', 2500); }
      catch { toast('Copying is blocked here: select the text and copy it yourself.', 'warn', '', 3500); }
    }
    const scoreKeys = computed(() => vars.value.filter((v) => /^mblf_f\d+$/.test(v.name)).map((v) => v.key));
    const toCluster = () => go('/cluster', { keys: scoreKeys.value.join(',') });
    return { rid, vars, methods, text, refs, copy, dl: exportFile, toCluster, go, goStep: ctx.goStep };
  },
  template: `
  <div class="col gap-12" data-mblf-use>
    <MblfSection title="Variables made by this model" sentence="They appear in the Plots page and in every selector of the program, under the name of this result.">
      <ul class="mblf-varlist" data-mblf-vars><li v-for="v in vars" :key="v.key"><span class="chip">{{ v.name }}</span> <span class="small">{{ v.label }}</span></li></ul>
      <div class="row wrap"><button type="button" class="btn" @click="toCluster" :disabled="!vars.length"><Icon name="bullseye" size="16"/>Cluster the samples on their factor scores</button>
        <button type="button" class="btn" @click="go('/compare')"><Icon name="chart-pie" size="16"/>Plot them</button></div>
    </MblfSection>
    <MblfSection title="Keep the model" sentence="A saved model keeps the posterior of the population (not your data). Applied to new samples it gives their factor scores with their uncertainty, predicts the values the model was fitted on with 95 % intervals from what the samples have, and says which samples are unusual for it.">
      <Callout kind="info">A new sample needs the EEM and the absorbance spectrum when the model was fitted on both (the data recipe of FluoViz reads every measurement a model was fitted on); its values and covariates are read from the metadata when they are there, and predicted or treated as unknown when they are not.</Callout>
      <div class="row wrap"><SaveModel :rid="rid"/><button type="button" class="btn" @click="go('/models')"><Icon name="files-landscapes-alt" size="16"/>Apply a saved model</button>
        <button type="button" class="btn quiet" @click="goStep(4)"><Icon name="arrow-left" size="15"/>Change the settings and fit again</button></div>
    </MblfSection>
    <MblfSection title="Downloads" sentence="The files are made from this model; the file names say what they contain.">
      <div class="fp-downloads" data-mblf-downloads>
        <div class="fp-dl"><b>Scores (CSV)</b><span class="muted xs">One row per sample: the posterior mean and standard deviation of every factor score, the held-out log density and the Pareto k.</span><button type="button" class="btn sm" @click="dl('scores')"><Icon name="download-alt" size="15"/>Download</button></div>
        <div class="fp-dl"><b>Loadings (CSV)</b><span class="muted xs">Every point of every mode: the posterior mean and 95 % interval of every loading function.</span><button type="button" class="btn sm" @click="dl('loadings')"><Icon name="download-alt" size="15"/>Download</button></div>
        <div class="fp-dl"><b>What tells what (CSV)</b><span class="muted xs">The predicted share of every mode for every choice of sources, with its interval.</span><button type="button" class="btn sm" @click="dl('matrix')"><Icon name="download-alt" size="15"/>Download</button></div>
        <div class="fp-dl"><b>Held-out comparison (CSV)</b><span class="muted xs">The comparison of step 7, when it was computed.</span><button type="button" class="btn sm" @click="dl('compare')"><Icon name="download-alt" size="15"/>Download</button></div>
      </div>
    </MblfSection>
    <MblfSection title="Methods paragraph" sentence="A text for the methods section of a report or paper. It can be pasted as it is.">
      <Callout kind="critical" v-if="methods && methods.error">{{ methods.error }}</Callout>
      <template v-if="text"><div class="fp-recipe" data-mblf-methods>{{ text }}</div><ul class="fp-list small"><li v-for="r in refs" :key="r">{{ r }}</li></ul>
        <div class="row"><button type="button" class="btn" @click="copy"><Icon name="copy-alt" size="15"/>Copy</button></div></template>
    </MblfSection>
  </div>`,
};

export const MblfChecks = {
  name: 'MblfChecks',
  components: { Seg, FpcaFrame, FpcaExplain, MblfGate, MblfStaleNote, MblfConvergence, MblfBasisCheck, MblfUse },
  setup() {
    const A = computed(() => fp.mb);
    const ready = computed(() => !!(A.value.res && A.value.resRid));
    const TABS = [{ value: 'convergence', label: 'Convergence' }, { value: 'basis', label: 'Basis' }, { value: 'use', label: 'Use the model' }];
    return { A, ready, TABS,
      DOES: 'Says how far the model can be trusted, and passes it on. Convergence: did the chains agree? Basis: is what each basis leaves out just noise? Then the factor scores and the held-out log densities become variables of FluoViz, and the model can be saved and applied to new samples.',
      CHECK: 'That every check is fine before you report anything: a model whose chains disagree, or whose basis leaves structure out, describes something, but not reliably what you think.' };
  },
  template: `
  <FpcaFrame :n="8" no-next>
    <FpcaExplain :does="DOES" :check="CHECK"/>
    <MblfGate/>
    <template v-if="ready">
      <MblfStaleNote/>
      <Seg v-model="A.tab8" :options="TABS" data-mblf-tabs8/>
      <MblfConvergence v-if="A.tab8 === 'convergence'"/>
      <MblfBasisCheck v-else-if="A.tab8 === 'basis'"/>
      <MblfUse v-else/>
    </template>
  </FpcaFrame>`,
};
