// Step 7 of the Autoencoders page: four cards with the main numbers and the tabs Latent space, Traversals, Samples, Interpolation and Generate.
// Also `AeGate` (what steps 7 to 9 show while there is no network, while it is being opened, or when it cannot be opened) and `AeStaleNote`.
import { computed, watch } from 'vue';
import { Icon } from '/static/js/icons.js';
import { Callout } from '/static/js/components/Common.js';
import { FpcaFrame } from '/static/js/components/FpcaStepper.js';
import { FpcaExplain } from '/static/js/components/FpcaCommon.js';
import { fp, ctx, STALE_WORDS } from './AeState.js';
import { AeStat, AeTabs, AeVerdict, pct, nTxt } from './AeCommon.js';
import { AeLatentTab } from './AeLatentTab.js';
import { AeTraversalsTab } from './AeTraversalsTab.js';
import { AeNonlinearTab } from './AeNonlinearTab.js';
import { AeSamplesTab } from './AeSamplesTab.js';
import { AeInterpTab, AeGenerateTab } from './AeInterpGenTabs.js';

/** The comparison with the linear method, as the server judged it (the page applies no threshold of its own): the difference in points of variance explained, and, when the server says who is better
 *  (`baseline.better` = network | baseline | same), the colour of the card. */
export function baselineInfo(q) {
  const b = q && q.baseline; if (!b) return null;
  const diff = (q.r2_val ?? NaN) - (b.r2_val ?? NaN);
  if (!Number.isFinite(diff)) return null;
  const better = b.better || null;
  return { diff, name: b.name || 'PCA', k: b.k, better, level: better === 'network' ? 'ok' : better === 'baseline' ? 'bad' : '', text: `${diff >= 0 ? '+' : '−'}${(Math.abs(diff) * 100).toFixed(1)} points` };
}
/** The colour of a verdict from what the server says: the comparison (`better`) and the overfitting level of the training ('clear' | 'mild' | 'none'). */
export function verdictLevel(q, t) {
  const better = q && q.baseline ? q.baseline.better : null, over = t && t.overfit ? t.overfit.level : null;
  if (better === 'baseline' || over === 'clear') return 'warning';
  if (better === 'network' && over !== 'mild') return 'good';
  return 'info';
}

export const AeStaleNote = {
  name: 'AeStaleNote',
  components: { Icon },
  setup() {
    const stale = computed(() => (fp.ae.res ? fp.ae.res.result.stale || [] : []));
    const what = computed(() => stale.value.map((s) => STALE_WORDS[s] || s).join(', '));
    return { stale, what };
  },
  template: `<div class="callout warning" v-if="stale.length" data-ae-stale><Icon name="exclamation-triangle"/><div class="grow"><b>The data changed since this network was trained</b> ({{ what }}). What is stored in the network can still be looked at: codes, training curves, quality numbers, traversals, mixtures of codes, new samples. Pictures of single samples need the current data: press <b>Train again with the same settings</b> in the bar above to train the network on the current data. Nothing is trained by itself.</div></div>`,
};

export const AeGate = {
  name: 'AeGate',
  components: { Icon, Callout },
  setup() {
    const A = computed(() => fp.ae);
    const training = computed(() => A.value.live.status === 'running' && A.value.live.kind === 'fit');
    return { A, training, goStep: ctx.goStep };
  },
  template: `
  <div data-ae-gate>
    <div class="fp-empty" v-if="!A.resBusy && !A.resError && !(A.res && A.resRid)" data-ae-noresult>
      <p v-if="training"><b>The network is being trained.</b> The results appear here when it is done.</p>
      <p v-else>No network has been trained yet. Choose the settings and press <b>Train the network</b> in step 6, or press <b>Run with the recommended settings</b> in step 1.</p>
      <div class="row" style="justify-content:center"><button type="button" class="btn" @click="goStep(6)"><Icon name="arrow-left" size="15"/>Go to step 6 · Train</button><button type="button" class="btn" @click="goStep(1)">Go to step 1</button></div>
    </div>
    <div class="fp-wait muted" v-else-if="A.resBusy && !A.res">Opening the network …</div>
    <Callout kind="critical" v-else-if="A.resError" data-ae-result-error><b>This network could not be opened.</b> {{ A.resError }}</Callout>
  </div>`,
};

export const AeResults = {
  name: 'AeResults',
  components: { FpcaFrame, FpcaExplain, AeStat, AeTabs, AeVerdict, AeGate, AeStaleNote, AeLatentTab, AeTraversalsTab, AeNonlinearTab, AeSamplesTab, AeInterpTab, AeGenerateTab },
  setup() {
    const A = computed(() => fp.ae), R = computed(() => fp.ae.res);
    const q = computed(() => (R.value ? R.value.p.quality || {} : {}));
    const base = computed(() => baselineInfo(q.value));
    const T = computed(() => (R.value ? R.value.p.training || {} : {}));
    const level = computed(() => verdictLevel(q.value, T.value));
    const nFlagged = computed(() => (R.value ? R.value.flag.filter((f) => f && !/^inside/i.test(f)).length : 0));
    const isVae = computed(() => !!R.value && R.value.family === 'vae');
    const tabs = computed(() => [
      { id: 'latent', label: 'Latent space' }, { id: 'traversals', label: 'Traversals' }, { id: 'nonlinear', label: 'Nonlinear effects' }, { id: 'samples', label: 'Samples' }, { id: 'interpolation', label: 'Interpolation' },
      { id: 'generate', label: 'Generate', disabled: !isVae.value, why: 'Only variational autoencoders can draw new samples. Train a variational autoencoder (step 5).' },
    ]);
    // 'Generate' exists only for a variational autoencoder: another network (or an older choice of the tab) falls back to the first tab
    const tab7 = computed({ get: () => (A.value.tab7 === 'generate' && !isVae.value ? 'latent' : A.value.tab7), set: (v) => { A.value.tab7 = v; } });
    const ready = computed(() => !!(R.value && A.value.resRid));
    return { A, R, q, base, nFlagged, isVae, tabs, tab7, ready, level, T, pct, nTxt };
  },
  template: `
  <FpcaFrame :n="7" next-label="Check the network" :no-next="!ready" :next-disabled="!ready">
    <FpcaExplain does="Shows what the network has learned: where every sample lies in the space of the codes, what each code does to the EEM (and whether it does it in a straight line), how well each single sample is rebuilt, what lies between two samples, and (for a variational autoencoder) new samples drawn by the network."
                 check="That the held-out variance explained is high and not much lower than the training one (otherwise the network memorised the samples), that the network is at least as good as PCA with the same number of codes (step 8 says whether the nonlinearity is worth it), and that the codes mean something: colour the latent space by your own variables."/>
    <AeGate/>
    <template v-if="ready">
      <AeStaleNote/>
      <div class="fp-cards4" data-ae-cards>
        <AeStat label="Variance explained on held-out samples" :value="pct(q.r2_val)" :sub="'training samples: ' + pct(q.r2_train)" :level="T.overfit && T.overfit.level === 'clear' ? 'bad' : ''" data-ae-stat7="r2"
                help="The share of the variation of samples the network was NOT trained on that it rebuilds from the codes. Higher is better."/>
        <AeStat label="Codes" :value="R.d" :sub="R.family === 'vae' ? 'variational autoencoder' : 'autoencoder'" data-ae-stat7="codes"/>
        <AeStat v-if="base" :label="base.name === 'Functional PCA' ? 'Against functional PCA' : 'Against PCA'" :value="base.text" :sub="base.name + ' with ' + base.k + ' ' + (base.k === 1 ? 'component' : 'components') + ': ' + pct(q.baseline.r2_val) + ' (network: ' + pct(q.r2_val) + ')'" :level="base.level" data-ae-stat7="baseline"
                help="Is the nonlinear network better than the linear method with the same number of codes, on held-out samples?"/>
        <AeStat label="Flagged samples" :value="nFlagged + ' of ' + R.n" sub="rebuilt poorly or with an unusual code" data-ae-stat7="flagged"/>
      </div>
      <AeVerdict v-if="q.verdict" :level="level">{{ q.verdict }}</AeVerdict>
      <AeTabs v-model="tab7" :tabs="tabs" label="Views of the results"/>
      <div data-ae-panel="latent" v-if="tab7 === 'latent'"><AeLatentTab/></div>
      <div data-ae-panel="traversals" v-else-if="tab7 === 'traversals'"><AeTraversalsTab/></div>
      <div data-ae-panel="nonlinear" v-else-if="tab7 === 'nonlinear'"><AeNonlinearTab/></div>
      <div data-ae-panel="samples" v-else-if="tab7 === 'samples'"><AeSamplesTab/></div>
      <div data-ae-panel="interpolation" v-else-if="tab7 === 'interpolation'"><AeInterpTab/></div>
      <div data-ae-panel="generate" v-else-if="tab7 === 'generate'"><AeGenerateTab/></div>
    </template>
  </FpcaFrame>`,
};
