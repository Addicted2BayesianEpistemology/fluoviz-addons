// Self-organising map (add-on): a guided pipeline like the Functional PCA page, in nine steps — what to map, domain, representation, smoothing (the Functional PCA's own components),
// map and distance, compute, results, checks, use.  The page is loaded by FluoViz at run time from /addon-web/som/web/page.js (manifest `provides.nav`); the shared components of FluoViz
// are imported with absolute addresses, the add-on's own files with relative ones.  Everything the page knows lives in components/SomState.js (module level, so it is found as it was left).
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
import { ctx, fp, somJobs } from './components/SomState.js';
import { loadDefaults, loadResult, runFit, runRecommended, cancelHeavy } from './components/SomActions.js';
import { SomStepWhat, SomStepDomain, SomStepRepr, SomStepSmooth } from './components/SomStepsReuse.js';
import { SomStepMap } from './components/SomStepMap.js';
import { SomStepCompute } from './components/SomStepCompute.js';
import { SomResults } from './components/SomResults.js';
import { SomChecks } from './components/SomChecks.js';
import { SomUse } from './components/SomUse.js';

loadCss('/static/css/page-decomp.css');                 // the colour bars and the shared decomposition pieces
loadCss('/static/css/page-fpca.css');                   // the stepper, the step panel, the cards and the explorer of steps 1 to 4
loadCss(new URL('./css/page-som.css', import.meta.url).pathname);

export default {
  name: 'SomPage',
  components: { Icon, Callout, JobBar, ResultBar, FpcaStepper, SomStepWhat, SomStepDomain, SomStepRepr, SomStepSmooth, SomStepMap, SomStepCompute, SomResults, SomChecks, SomUse },
  setup() {
    provideFpca(ctx);                                   // steps 1 to 4 find the state of THIS page through it
    ctx.loadChoices();
    const refresh = () => { ctx.loadData(); loadCatalogue(); refreshSamples(); loadDefaults(); };
    onMounted(refresh);
    onActivated(refresh);
    watch(() => [fp.spec.fluorescence.min_coverage, fp.spec.absorbance.processed], debounce(() => { ctx.loadData(); loadDefaults(); }, 350));
    watch(() => store.samplesRev, debounce(() => { refreshSamples(); ctx.loadData(true); loadDefaults(true); if (fp.rid) loadResult(fp.rid, true, { keep: true }); }, 250));
    watch(() => fp.rid, (r) => loadResult(r), { immediate: true });
    // the map shown, and which of its parts changed since (the result bar says it in words)
    const rid = computed({ get: () => fp.rid, set: (v) => { const prev = fp.rid; fp.rid = v; if (prev && v && prev !== v && fp.step < 7) ctx.goStep(7); } });
    async function recompute(cur) {
      const res = await runFit(cur.params, { replace: cur.id, go: false });
      if (res) toast(`The map ${res.rid} was computed again with the same settings, on the current data.`, 'ok', '', 5000);
    }
    const running = computed(() => somJobs.fit.state.running);
    const scanning = computed(() => somJobs.scan.state.running);
    return { fp, rid, recompute, go, running, scanning, somJobs, runRecommended, cancelHeavy, store };
  },
  template: `
  <div class="fp-page som-page" :data-som-page="fp.step" :data-busy="running || scanning || fp.som.preparing ? '1' : '0'">
    <div class="page-head"><div><h1 class="h1">Self-organising map (add-on) <span class="fp-exp" data-som-addon-pill>add-on</span></h1>
      <p class="lead">A map lays your samples out on a small grid of units. Every unit holds a typical EEM or spectrum (its <b>prototype</b>), and neighbouring units hold similar ones, so similar samples sit together and gradients run across the map. It can work on the pixels or, like the functional PCA, on the coefficients of a smooth basis. Follow the nine steps: each one explains what it does and what to check, and the checks say honestly whether the map keeps neighbours together better than a plain scatter. The positions become variables you can plot, cluster and classify.</p></div>
      <div class="row"><button type="button" class="btn quiet sm" data-som-howto @click="go('/help', { tab: 'addon:som:som' })"><Icon name="question-circle" size="16"/>How to read this</button></div></div>

    <ResultBar kind="som" v-model="rid" noun="map" recompute-label="Compute again with the same settings" @recompute="recompute" data-som-resultbar/>
    <JobBar v-if="running && fp.step !== 6" :job="somJobs.fit" label="Computing the map"/>
    <FpcaStepper/>
    <SomStepWhat v-if="fp.step === 1" @run-recommended="runRecommended"/>
    <SomStepDomain v-else-if="fp.step === 2"/>
    <SomStepRepr v-else-if="fp.step === 3"/>
    <SomStepSmooth v-else-if="fp.step === 4"/>
    <SomStepMap v-else-if="fp.step === 5"/>
    <SomStepCompute v-else-if="fp.step === 6"/>
    <SomResults v-else-if="fp.step === 7"/>
    <SomChecks v-else-if="fp.step === 8"/>
    <SomUse v-else-if="fp.step === 9"/>
  </div>`,
};
