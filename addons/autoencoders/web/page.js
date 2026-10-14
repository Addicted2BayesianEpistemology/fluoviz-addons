// Autoencoders (add-on): a guided pipeline like the Functional PCA page, in nine steps — what to model, domain, representation, smoothing (the Functional PCA's own components),
// network, train, results, checks, use.  The page is loaded by FluoViz at run time from /addon-web/autoencoders/web/page.js (manifest `provides.nav`); the shared components of FluoViz
// are imported with absolute addresses, the add-on's own files with relative ones.  Everything the page knows lives in components/AeState.js (module level, so it is found as it was left).
import { computed, watch, onMounted, onActivated } from 'vue';
import { Icon } from '/static/js/icons.js';
import { store, toast } from '/static/js/store.js';
import { go } from '/static/js/router.js';
import { debounce, loadCss } from '/static/js/util.js';
import { Callout } from '/static/js/components/Common.js';
import { JobBar } from '/static/js/components/JobBar.js';
import { ResultBar } from '/static/js/components/ResultBar.js';
import { loadCatalogue } from '/static/js/variables.js';
import { refreshSamples } from '/static/js/actions.js';
import { provideFpca } from '/static/js/components/FpcaCtx.js';
import { FpcaStepper } from '/static/js/components/FpcaStepper.js';
import { ctx, fp, aeJobs } from './components/AeState.js';
import { loadDefaults, loadResult, runFit, runRecommended, cancelHeavy } from './components/AeActions.js';
import { AeStepWhat, AeStepDomain, AeStepRepr, AeStepSmooth } from './components/AeStepsReuse.js';
import { AeStepNetwork } from './components/AeStepNetwork.js';
import { AeStepTrain } from './components/AeStepTrain.js';
import { AeResults } from './components/AeResults.js';
import { AeChecks } from './components/AeChecks.js';
import { AeUse } from './components/AeUse.js';

loadCss('/static/css/page-decomp.css');                 // the colour bars (DecompScale) and the shared decomposition pieces
loadCss('/static/css/page-fpca.css');                   // the stepper, the step panel, the cards and the explorer of steps 1 to 4
loadCss(new URL('./css/page-autoencoders.css', import.meta.url).pathname);

export default {
  name: 'AutoencodersPage',
  components: { Icon, Callout, JobBar, ResultBar, FpcaStepper, AeStepWhat, AeStepDomain, AeStepRepr, AeStepSmooth, AeStepNetwork, AeStepTrain, AeResults, AeChecks, AeUse },
  setup() {
    provideFpca(ctx);                                   // steps 1 to 4 find the state of THIS page through it
    ctx.loadChoices();
    const refresh = () => { ctx.loadData(); loadCatalogue(); refreshSamples(); loadDefaults(); };
    onMounted(refresh);
    onActivated(refresh);
    watch(() => [fp.spec.fluorescence.min_coverage, fp.spec.absorbance.processed], debounce(() => { ctx.loadData(); loadDefaults(); }, 350));
    watch(() => store.samplesRev, debounce(() => { refreshSamples(); ctx.loadData(true); loadDefaults(true); if (fp.rid) loadResult(fp.rid, true, { keep: true }); }, 250));
    watch(() => fp.rid, (r) => loadResult(r), { immediate: true });
    // the network shown, and which of its parts changed since (the result bar says it in words)
    const rid = computed({ get: () => fp.rid, set: (v) => { const prev = fp.rid; fp.rid = v; if (prev && v && prev !== v && fp.step < 7) ctx.goStep(7); } });
    async function recompute(cur) {
      const res = await runFit(cur.params, { replace: cur.id, go: false });
      if (res) toast(`The network ${res.rid} was trained again with the same settings, on the current data.`, 'ok', '', 5000);
    }
    const running = computed(() => aeJobs.fit.state.running);
    const scanning = computed(() => aeJobs.scan.state.running);
    return { fp, rid, recompute, go, running, scanning, aeJobs, runRecommended, cancelHeavy, store };
  },
  template: `
  <div class="fp-page ae-page" :data-ae-page="fp.step" :data-busy="running || scanning || fp.ae.preparing ? '1' : '0'">
    <div class="page-head"><div><h1 class="h1">Autoencoders (add-on) <span class="fp-exp" data-ae-addon-pill>add-on</span></h1>
      <p class="lead">A neural network learns a few numbers, the <b>code</b>, that describe each sample, and rebuilds the sample from them. It can work on the pixels or, like the functional PCA, on the coefficients of a smooth basis. Follow the nine steps: each one explains what it does and what to check, and the checks say honestly whether the network beats PCA. The codes become variables you can plot, cluster and classify.</p></div>
      <div class="row"><button type="button" class="btn quiet sm" data-ae-howto @click="go('/help', { tab: 'addon:autoencoders:autoencoders' })"><Icon name="question-circle" size="16"/>How to read this</button></div></div>

    <ResultBar kind="autoencoder" v-model="rid" noun="autoencoder" recompute-label="Train again with the same settings" @recompute="recompute" data-ae-resultbar/>
    <JobBar v-if="running && fp.step !== 6" :job="aeJobs.fit" label="Training the network"/>
    <FpcaStepper/>
    <AeStepWhat v-if="fp.step === 1" @run-recommended="runRecommended"/>
    <AeStepDomain v-else-if="fp.step === 2"/>
    <AeStepRepr v-else-if="fp.step === 3"/>
    <AeStepSmooth v-else-if="fp.step === 4"/>
    <AeStepNetwork v-else-if="fp.step === 5"/>
    <AeStepTrain v-else-if="fp.step === 6"/>
    <AeResults v-else-if="fp.step === 7"/>
    <AeChecks v-else-if="fp.step === 8"/>
    <AeUse v-else-if="fp.step === 9"/>
  </div>`,
};
