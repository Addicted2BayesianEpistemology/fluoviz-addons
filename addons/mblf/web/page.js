// Multimodal latent factors (add-on): a guided pipeline like the Functional PCA page, in eight steps — what to model, domain, basis (the Functional PCA's own components), model
// and sampler, fit, results, prediction and uncertainty, checks and use.  The page is loaded by FluoViz at run time from /addon-web/mblf/web/page.js (manifest `provides.nav`); the
// shared components of FluoViz are imported with absolute addresses, the add-on's own files with relative ones.  Everything the page knows lives in components/MblfState.js.
import { computed, watch, onMounted, onActivated } from 'vue';
import { Icon } from '/static/js/icons.js';
import { store, toast } from '/static/js/store.js';
import { go } from '/static/js/router.js';
import { debounce, loadCss } from '/static/js/util.js';
import { JobBar } from '/static/js/components/JobBar.js';
import { ResultBar } from '/static/js/components/ResultBar.js';
import { loadCatalogue } from '/static/js/variables.js';
import { refreshSamples } from '/static/js/actions.js';
import { provideFpca } from '/static/js/components/FpcaCtx.js';
import { FpcaStepper } from '/static/js/components/FpcaStepper.js';
import { ctx, fp, mbJobs } from './components/MblfState.js';
import { loadDefaults, loadEngine, loadResult, runFit } from './components/MblfActions.js';
import { MblfStepWhat, MblfStepDomain, MblfStepRepr, MblfStepModel, MblfStepFit } from './components/MblfSteps.js';
import { MblfResults } from './components/MblfResults.js';
import { MblfPredict } from './components/MblfPredict.js';
import { MblfChecks } from './components/MblfChecks.js';

loadCss('/static/css/page-decomp.css');
loadCss('/static/css/page-fpca.css');
loadCss(new URL('./css/page-mblf.css', import.meta.url).pathname);

export default {
  name: 'MblfPage',
  components: { Icon, JobBar, ResultBar, FpcaStepper, MblfStepWhat, MblfStepDomain, MblfStepRepr, MblfStepModel, MblfStepFit, MblfResults, MblfPredict, MblfChecks },
  setup() {
    provideFpca(ctx);
    ctx.loadChoices();
    const refresh = () => { ctx.loadData(); loadCatalogue(); refreshSamples(); loadDefaults(); loadEngine(); };
    onMounted(refresh);
    onActivated(refresh);
    watch(() => [fp.spec.fluorescence.min_coverage, fp.spec.absorbance.processed], debounce(() => ctx.loadData(), 350));
    watch(() => store.samplesRev, debounce(() => { refreshSamples(); ctx.loadData(true); loadDefaults(); if (fp.rid) loadResult(fp.rid, true); }, 250));
    watch(() => fp.rid, (r) => loadResult(r), { immediate: true });
    const rid = computed({ get: () => fp.rid, set: (v) => { const prev = fp.rid; fp.rid = v; if (prev && v && prev !== v && fp.step < 6) ctx.goStep(6); } });
    async function recompute(cur) {
      const res = await runFit(cur.params, { replace: cur.id, go: false });
      if (res) toast(`The model ${res.rid} was fitted again with the same settings, on the current data.`, 'ok', '', 5000);
    }
    const running = computed(() => mbJobs.fit.state.running);
    return { fp, rid, recompute, go, running, mbJobs };
  },
  template: `
  <div class="fp-page mblf-page" :data-mblf-page="fp.step" :data-busy="running ? '1' : '0'">
    <div class="page-head"><div><h1 class="h1">Multimodal latent factors (add-on) <span class="fp-exp" data-mblf-addon-pill>add-on</span></h1>
      <p class="lead">A few <b>shared factors</b> describe how your measurements — the fluorescence (EEMs), the absorbance spectra, other values — vary together from sample to sample, and covariates can shift them. The model is Bayesian, so every result carries its <b>uncertainty</b>: the factors have credible bands, and any measurement of a sample can be <b>predicted from the others</b> with the uncertainty split into what more samples, more measurements, nothing, or the instrument would change. Every choice of predictors is checked on held-out samples. Follow the eight steps.</p></div>
      <div class="row"><button type="button" class="btn quiet sm" data-mblf-howto @click="go('/help', { tab: 'addon:mblf:mblf' })"><Icon name="question-circle" size="16"/>How to read this</button></div></div>

    <ResultBar kind="mblf" v-model="rid" noun="latent factor model" recompute-label="Fit again with the same settings" @recompute="recompute" data-mblf-resultbar/>
    <JobBar v-if="running && fp.step !== 5" :job="mbJobs.fit" label="Fitting the model"/>
    <FpcaStepper/>
    <MblfStepWhat v-if="fp.step === 1"/>
    <MblfStepDomain v-else-if="fp.step === 2"/>
    <MblfStepRepr v-else-if="fp.step === 3"/>
    <MblfStepModel v-else-if="fp.step === 4"/>
    <MblfStepFit v-else-if="fp.step === 5"/>
    <MblfResults v-else-if="fp.step === 6"/>
    <MblfPredict v-else-if="fp.step === 7"/>
    <MblfChecks v-else-if="fp.step === 8"/>
  </div>`,
};
