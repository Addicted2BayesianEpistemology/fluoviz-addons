// Steps 1 to 5 of the Multimodal latent factors page.  Steps 1 to 3 ARE the Functional PCA's step components (same controls, same routes /api/fpca/*), with the words of this page;
// step 1 adds the values to model, the covariates and the samples that lack a measurement; step 3 offers the Gaussian-process basis only (the prior of the method is built on it).
// Step 4 holds the model, the sampler and the Stan engine; step 5 says what will be done and fits.
import { computed, onMounted, watch } from 'vue';
import { Icon } from '/static/js/icons.js';
import { Seg, Field, NumField, Switch, Callout, Fold } from '/static/js/components/Common.js';
import { JobBar } from '/static/js/components/JobBar.js';
import { FpcaStepWhat } from '/static/js/components/FpcaStepWhat.js';
import { FpcaStepDomain } from '/static/js/components/FpcaStepDomain.js';
import { FpcaStepRepr } from '/static/js/components/FpcaStepRepr.js';
import { FpcaFrame } from '/static/js/components/FpcaStepper.js';
import { FpcaExplain } from '/static/js/components/FpcaCommon.js';
import { VarSelect } from '/static/js/components/VarSelect.js';
import { loadCatalogue, labelOf } from '/static/js/variables.js';
import { debounce } from '/static/js/util.js';
import { fp, ctx, mbJobs, requestBody, variablesOf, PRESET_WORDS, MODE_NICE } from './MblfState.js';
import { loadEngine, prepareEngine, removeEngine, runCheck, runFit, cancelFit, loadDefaults } from './MblfActions.js';
import { MblfSection, MblfVerdict, MblfStat, pct, nTxt, listWords } from './MblfCommon.js';

// ---------------------------------------------------------------- 1 · What to model
export const MblfStepWhat = {
  name: 'MblfStepWhat',
  components: { Icon, Switch, Callout, VarSelect, FpcaStepWhat, MblfSection },
  setup() {
    onMounted(() => loadCatalogue());
    const m = computed(() => fp.spec.mblf);
    const add = (list, key) => { if (key && !list.includes(key)) list.push(key); };
    const remove = (list, key) => { const i = list.indexOf(key); if (i >= 0) list.splice(i, 1); };
    const combo = computed(() => {
      const names = variablesOf(fp.spec).map((n) => MODE_NICE[n].toLowerCase());
      if (m.value.values.length) names.push(`${m.value.values.length} value${m.value.values.length === 1 ? '' : 's'}`);
      const cov = m.value.covariates.length ? `, with ${m.value.covariates.length} covariate${m.value.covariates.length === 1 ? '' : 's'} shifting the scores` : ', without covariates';
      const kind = names.length === 1 ? 'one mode: a Bayesian functional factor analysis of it' : `${names.length} modes sharing their factors`;
      return `${listWords(names)} — ${kind}${cov}.`;
    });
    return {
      m, add, remove, labelOf, combo,
      DOES: 'You choose what the model describes. Every kind of measurement is a mode: the fluorescence (EEMs), the absorbance spectra, and optionally other numbers you measured (a laboratory value, an index). A few shared factors describe how ALL the modes vary together from sample to sample; each sample also keeps a part of its own. Covariates (metadata) shift the factors from sample to sample: they explain, they are not explained.',
      CHECK: 'That the combination answers your question. One mode alone: the patterns of that measurement. Two or more: what they share, and how well one predicts another (step 7). Covariates: how the patterns depend on site, season, treatment … A value you want to PREDICT (for example the DOC) belongs to the values; a value that EXPLAINS belongs to the covariates.',
      WORDS: { both: 'Fluorescence and absorbance of the same samples: the factors are shared by both, so the model learns how one predicts the other.' },
    };
  },
  template: `
  <div>
    <FpcaStepWhat :explain-does="DOES" :explain-check="CHECK" :words="WORDS" :weights="false"/>
    <div class="card flat mblf-what" data-mblf-what>
      <div class="card-head"><h3>Other values and covariates</h3></div>
      <div class="card-body col gap-12">
        <p class="fp-note" data-mblf-combo><b>This model:</b> {{ combo }}</p>
        <MblfSection title="Values to model (optional)" sentence="Numbers measured on the samples that the factors should describe and predict: a laboratory value (DOC, nitrate), a fluorescence index, scores of another model. A sample may lack them: the model predicts them.">
          <div class="row wrap"><div class="field" style="min-width:260px"><label>Add a value</label><VarSelect :model-value="null" :kinds="['numeric']" :exclude="m.values.concat(m.covariates)" keep-open placeholder="Choose a variable …" small data-mblf-add-value @update:modelValue="(k) => add(m.values, k)"/></div></div>
          <div class="row wrap" v-if="m.values.length"><span class="chip" v-for="k in m.values" :key="k" data-mblf-value>{{ labelOf(k) }}<span class="x" role="button" tabindex="0" :aria-label="'Remove ' + labelOf(k)" @click="remove(m.values, k)" @keydown.enter="remove(m.values, k)">×</span></span></div>
        </MblfSection>
        <MblfSection title="Covariates (optional)" sentence="Metadata that shift the factor scores: numbers (temperature, depth) or categories (site, season, treatment). The model then shows the effect of each one on every wavelength and pixel, with its uncertainty. Samples without a covariate value are left out.">
          <div class="row wrap"><div class="field" style="min-width:260px"><label>Add a covariate</label><VarSelect :model-value="null" :kinds="['numeric', 'category']" :exclude="m.values.concat(m.covariates)" keep-open placeholder="Choose a variable …" small data-mblf-add-covariate @update:modelValue="(k) => add(m.covariates, k)"/></div></div>
          <div class="row wrap" v-if="m.covariates.length"><span class="chip" v-for="k in m.covariates" :key="k" data-mblf-covariate>{{ labelOf(k) }}<span class="x" role="button" tabindex="0" :aria-label="'Remove ' + labelOf(k)" @click="remove(m.covariates, k)" @keydown.enter="remove(m.covariates, k)">×</span></span></div>
        </MblfSection>
        <Switch v-model="m.partial" label="Use samples that lack a measurement (they contribute what they have, and the model predicts what they lack)" data-mblf-partial/>
        <Callout kind="info" v-if="!m.partial">Only the samples that have every mode are used. Switch it on to use all of them: a sample without an absorbance spectrum still informs the fluorescence part, and gets a predicted spectrum.</Callout>
      </div>
    </div>
  </div>`,
};

// ---------------------------------------------------------------- 2 · Domain, 3 · Basis
export const MblfStepDomain = {
  name: 'MblfStepDomain',
  components: { FpcaStepDomain },
  template: `<FpcaStepDomain method="the model"/>`,
};

export const MblfStepRepr = {
  name: 'MblfStepRepr',
  components: { FpcaStepRepr, Callout },
  setup() {
    return {
      DOES: 'Writes every EEM and spectrum with the eigenfunctions of a Gaussian process on its domain (a Karhunen–Loève basis), each multiplied by its prior standard deviation: smooth building blocks, the smoothest first. The model works on their coefficients. The share of the prior variance you keep has a meaning: what is left out is the expected relative error of truncating the basis (a theorem of the method), so 99 % means the basis can represent a typical smooth EEM up to about 1 % of its variance. What the basis cannot represent goes to the noise.',
      CHECK: 'That the number of functions stays moderate (the sampler explores every coefficient of every sample: tens of functions per mode, not hundreds) and that the length-scales look like the width of the real features. Step 8 checks afterwards that what the basis leaves out looks like noise.',
    };
  },
  template: `
  <FpcaStepRepr :kinds="['gp']" :explain-does="DOES" :explain-check="CHECK">
    <template #top><Callout kind="info" data-mblf-basis-note>The model needs the Gaussian-process basis (its prior is built on it). No smoothing is chosen here: the model itself separates the shared variation, each sample's own part and the measurement noise.</Callout></template>
  </FpcaStepRepr>`,
};

// ---------------------------------------------------------------- 4 · Model and sampler
export const MblfEngineCard = {
  name: 'MblfEngineCard',
  components: { Icon, JobBar, Callout },
  setup() {
    onMounted(() => loadEngine());
    const E = computed(() => fp.mb.engine);
    const running = computed(() => mbJobs.engine.state.running);
    return { E, running, mbJobs, prepare: prepareEngine, remove: removeEngine, A: computed(() => fp.mb) };
  },
  template: `
  <div class="card flat mblf-engine" data-mblf-engine :data-state="E ? E.state : 'unknown'">
    <div class="card-head"><h3><Icon name="cog" size="16"/> The Stan engine</h3></div>
    <div class="card-body col gap-8">
      <p class="small" v-if="!E">Asking the server …</p>
      <template v-else>
        <p class="small" :class="{ ok: E.state === 'ready' }">{{ E.message }}</p>
        <p class="muted xs">The posterior is sampled by Stan (CmdStan), which needs a C++ compiler. They live in a private environment of this add-on, outside FluoViz's own, because installing them there would replace compilers FluoViz uses. It is prepared once; the first fit then compiles the model (about a minute), later fits start at once.</p>
        <div class="row wrap">
          <button type="button" class="btn primary" v-if="E.state === 'missing'" :disabled="running" data-mblf-engine-prepare @click="prepare"><Icon name="download-alt" size="16"/>Prepare the Stan engine (about {{ E.download_mb }} MB, once)</button>
          <button type="button" class="btn danger" v-if="running" @click="mbJobs.engine.cancel()"><Icon name="stop-circle" size="16"/>Cancel</button>
          <button type="button" class="btn quiet sm" v-if="E.state === 'ready' && E.source === 'private'" data-mblf-engine-remove @click="remove">Remove the Stan engine</button>
        </div>
        <JobBar :job="mbJobs.engine" label="Preparing the Stan engine"/>
        <Callout kind="critical" v-if="E.state === 'unavailable'">{{ E.message }}</Callout>
      </template>
    </div>
  </div>`,
};

export const MblfStepModel = {
  name: 'MblfStepModel',
  components: { Icon, Seg, Field, NumField, Switch, Callout, Fold, FpcaFrame, FpcaExplain, MblfSection, MblfEngineCard },
  setup() {
    const m = computed(() => fp.spec.mblf);
    const priors = computed(() => (fp.mb.defaults && fp.mb.defaults.priors) || null);
    onMounted(() => loadDefaults());
    const ESTIMATIONS = [{ value: 'bayes', label: 'Full posterior (MCMC)', title: 'The posterior of every parameter: every band includes the uncertainty about the population' },
      { value: 'ml', label: 'Maximum likelihood (quick)', title: 'One set of parameters: faster, but the bands leave out the uncertainty about the population' }];
    const PRESETS = [{ value: 'quick', label: 'Quick look' }, { value: 'standard', label: 'Standard' }, { value: 'thorough', label: 'Thorough' }];
    const resetPriors = () => { if (priors.value) for (const [k, v] of Object.entries(priors.value)) m.value.priors[k] = v.default; };
    return { m, priors, ESTIMATIONS, PRESETS, PRESET_WORDS, resetPriors,
      DOES: 'Chooses the model and how its posterior is computed. The largest number of factors is a CAP: the prior (a multiplicative gamma process) shrinks every further factor more strongly, so the factors the data do not need fade away; the results say how many the data really use, with its probability. The full posterior is sampled by Markov chain Monte Carlo (the No-U-Turn sampler of Stan): every result then carries its uncertainty.',
      CHECK: 'That the cap is generous (5 to 8 is usual; the checks say when the data would like more) and that the length of the sampling fits your patience: start with the quick look to see whether the settings are sensible, then fit with the standard setting before you interpret anything.' };
  },
  template: `
  <FpcaFrame :n="4">
    <FpcaExplain :does="DOES" :check="CHECK"/>
    <div class="fp-two fp-two-ctl">
      <div class="col gap-12">
        <MblfSection title="Factors" sentence="The largest number of shared factors. More than the data need is harmless (they are shrunk away); too few forces different patterns into one.">
          <Field label="Largest number of factors" help="A cap, not a choice: the shrinkage prior switches off the factors the data do not need."><NumField v-model="m.K" :min="1" :max="12" small data-mblf-k/></Field>
        </MblfSection>
        <MblfSection title="Estimation" sentence="Both give the same factors and predictions; only the full posterior also says how sure they are about the population.">
          <Seg v-model="m.estimation" :options="ESTIMATIONS" data-mblf-estimation/>
          <Callout kind="warning" v-if="m.estimation === 'ml'">A maximum-likelihood fit treats the population parameters as known. Its prediction bands leave out that part of the uncertainty (in the uncertainty budget it is zero), and the leave-one-out checks become in-sample. Use it to explore; use the full posterior to conclude.</Callout>
        </MblfSection>
        <MblfSection v-if="m.estimation === 'bayes'" title="Length of the sampling" sentence="Several chains run in parallel from different starting points; their agreement is the check of convergence.">
          <Seg v-model="m.preset" :options="PRESETS" :disabled="m.custom" data-mblf-preset/>
          <p class="fp-note">{{ m.custom ? 'Custom: ' + m.chains + ' chains of ' + m.warmup + ' + ' + m.draws + ' iterations.' : PRESET_WORDS[m.preset] }}</p>
          <Fold title="Advanced sampler settings" data-mblf-advanced>
            <div class="col gap-12">
              <Switch v-model="m.custom" label="Choose the chains and iterations myself" data-mblf-custom/>
              <div class="pair" v-if="m.custom">
                <Field label="Chains"><NumField v-model="m.chains" :min="1" :max="16" small/></Field>
                <Field label="Warm-up iterations" help="Iterations that tune the sampler; they are thrown away."><NumField v-model="m.warmup" :min="50" :max="20000" small/></Field>
                <Field label="Kept iterations" help="Iterations kept, per chain."><NumField v-model="m.draws" :min="50" :max="20000" small/></Field>
              </div>
              <div class="pair">
                <Field label="Target acceptance rate" help="adapt_delta: raise it (0.9, 0.95, 0.99) when the checks report divergent transitions. Slower, but it explores difficult regions properly."><NumField v-model="m.adapt_delta" :min="0.5" :max="0.999" :step="0.01" small data-mblf-adapt/></Field>
                <Field label="Largest tree depth" help="Rarely changed: raise it when many iterations hit it."><NumField v-model="m.max_treedepth" :min="5" :max="15" small/></Field>
                <Field label="Draws kept in the result" help="The draws every summary of the page is computed from (evenly thinned across the chains)."><NumField v-model="m.keep" :min="100" :max="4000" :step="100" small/></Field>
              </div>
            </div>
          </Fold>
        </MblfSection>
        <MblfSection v-else title="Starting points" sentence="The likelihood can have several maxima; it is maximised from several starting points and the best is kept.">
          <Field label="Starting points"><NumField v-model="m.ml_starts" :min="1" :max="64" small/></Field>
        </MblfSection>
        <Fold title="Priors (advanced)" data-mblf-priors>
          <p class="fp-note">The data are standardised mode by mode, so these defaults of the method mean the same thing for every unit and every mode. Change them only with a reason, and say so when you report the results.</p>
          <div class="mblf-priors" v-if="priors">
            <Field v-for="(p, k) in priors" :key="k" :label="p.label"><NumField v-model="m.priors[k]" :min="p.min" :max="p.max" :step="0.1" small :data-mblf-prior="k"/></Field>
          </div>
          <button type="button" class="btn sm quiet" @click="resetPriors">Back to the defaults of the method</button>
        </Fold>
      </div>
      <div class="col gap-12"><MblfEngineCard/></div>
    </div>
  </FpcaFrame>`,
};

// ---------------------------------------------------------------- 5 · Fit
export const MblfStepFit = {
  name: 'MblfStepFit',
  components: { Icon, Field, NumField, Switch, Callout, JobBar, FpcaFrame, FpcaExplain, MblfStat, MblfVerdict, MblfEngineCard },
  setup() {
    const m = computed(() => fp.spec.mblf);
    const A = computed(() => fp.mb);
    const running = computed(() => mbJobs.fit.state.running);
    const check = computed(() => A.value.check);
    const ans = computed(() => (check.value.key === JSON.stringify(requestBody().input) ? check.value.ans : null));
    const issues = computed(() => ctx.allIssues());
    const engineReady = computed(() => A.value.engine && A.value.engine.state === 'ready');
    const canRun = computed(() => !running.value && !issues.value.length && engineReady.value && !(ans.value && ans.value.problem));
    const why = computed(() => (running.value ? 'A model is being fitted.' : issues.value.length ? `Something needs to be fixed first: ${issues.value[0].message}` : !engineReady.value ? 'The Stan engine is not ready (step 4).' : ans.value && ans.value.problem ? ans.value.problem : ''));
    const few = computed(() => !!(ans.value && ans.value.problem && /fewer than/.test(ans.value.problem)));
    const soon = debounce(() => { runCheck(); }, 800);
    onMounted(() => { loadEngine(); soon(); });
    watch(() => JSON.stringify(requestBody().input), () => soon());
    const hasResult = computed(() => !!(fp.rid && A.value.res));
    const est = computed(() => (m.value.estimation === 'bayes' ? (m.value.custom ? `${m.value.chains} chains of ${m.value.warmup} + ${m.value.draws} iterations` : PRESET_WORDS[m.value.preset]) : `maximum likelihood from ${m.value.ml_starts} starting points`));
    return { m, A, running, ans, issues, canRun, why, few, hasResult, est, run: () => runFit(), cancelFit, mbJobs, pct, nTxt, check, engineReady,
      DOES: 'Fits the model. The Stan engine samples the posterior of every parameter with several chains; the page then summarises it: the factors with their bands, how much of each mode the others predict, the effects of the covariates, the leave-one-out checks. Cancel stops the sampler and saves nothing.',
      CHECK: 'That the facts on the right are what you expect: the samples, the number of basis functions of each mode, the samples that lack a mode. A model with tens of thousands of unknowns takes minutes.' };
  },
  template: `
  <FpcaFrame :n="5" next-label="See the results" :no-next="!hasResult" :next-disabled="!hasResult">
    <FpcaExplain :does="DOES" :check="CHECK"/>
    <div class="fp-two fp-two-ctl">
      <div class="col gap-12">
        <div class="card flat" data-mblf-summary>
          <div class="card-head"><h3>What will be done</h3></div>
          <div class="card-body">
            <p class="muted small" v-if="check.busy && !ans">Building the bases …</p>
            <Callout kind="critical" v-if="check.error">{{ check.error }}</Callout>
            <template v-if="ans">
              <div class="fp-stats">
                <MblfStat label="Samples" :value="nTxt(ans.n)" :sub="nTxt(ans.complete) + ' with every mode'"/>
                <MblfStat v-for="md in ans.modes" :key="md.name" :label="md.label" :value="md.n_basis + ' functions'" :sub="md.kind === 'vector' ? md.n_samples + ' samples have them' : (md.captured !== null ? pct(md.captured, 1) + ' of the prior variance; ' : '') + md.n_samples + ' samples'"/>
              </div>
              <ul class="fp-list small">
                <li v-if="ans.covariates">Covariates: {{ ans.covariates.columns.map((c) => c.label).join(', ') }}</li>
                <li>Estimation: {{ est }}; at most {{ m.K }} factors</li>
                <li>{{ ans.words }}</li>
                <li v-for="nt in ans.notes" :key="nt">{{ nt }}</li>
              </ul>
            </template>
          </div>
        </div>
        <details class="fp-more"><summary>Seed and title</summary>
          <div class="col gap-12" style="padding-top:8px">
            <Field label="Random seed" help="The same seed and the same settings give the same chains."><NumField v-model="m.seed" :min="0" small data-mblf-seed/></Field>
            <Field label="Title of the result"><input class="input sm" v-model="m.title" placeholder="(made from the settings)" data-mblf-title/></Field>
          </div>
        </details>
        <Switch v-if="few || m.allow_few" v-model="m.allow_few" label="Fit on fewer than 10 samples anyway" data-mblf-allow-few/>
      </div>
      <div class="col gap-12">
        <MblfEngineCard v-if="!engineReady"/>
        <Callout kind="critical" v-if="A.fitError" data-mblf-fit-error>{{ A.fitError }}</Callout>
        <div class="row wrap">
          <button type="button" class="btn primary" data-mblf-run :disabled="!canRun" :title="why" @click="run"><Icon name="play" size="16"/>Fit the model</button>
          <button type="button" class="btn danger" v-if="running" data-mblf-cancel @click="cancelFit"><Icon name="stop-circle" size="16"/>Cancel</button>
          <span class="muted xs" v-if="!canRun && why" data-mblf-why>{{ why }}</span>
        </div>
        <JobBar :job="mbJobs.fit" label="Fitting the model"/>
      </div>
    </div>
  </FpcaFrame>`,
};
