// Steps 1 to 4 of the Self-organising map page ARE the Functional PCA's step components (same controls, same routes `/api/fpca/*`); these thin wrappers only give them the words of
// this page, add the weights between the variables and the choice "Other variables instead" to step 1, and the functional switch to step 3.  Modelled on AeStepsReuse.js.
import { computed, onMounted } from 'vue';
import { Icon } from '/static/js/icons.js';
import { Seg, Callout, Switch } from '/static/js/components/Common.js';
import { FpcaStepWhat } from '/static/js/components/FpcaStepWhat.js';
import { FpcaStepDomain } from '/static/js/components/FpcaStepDomain.js';
import { FpcaStepRepr } from '/static/js/components/FpcaStepRepr.js';
import { FpcaStepSmooth } from '/static/js/components/FpcaStepSmooth.js';
import { FpcaFrame } from '/static/js/components/FpcaStepper.js';
import { FpcaExplain } from '/static/js/components/FpcaCommon.js';
import { VarSelect } from '/static/js/components/VarSelect.js';
import { loadCatalogue, labelOf } from '/static/js/variables.js';
import { fp, isFunctional, setFunctional } from './SomState.js';

export const SomStepWhat = {
  name: 'SomStepWhat',
  components: { Icon, Switch, Callout, VarSelect, FpcaStepWhat },
  emits: ['run-recommended'],
  setup() {
    onMounted(() => loadCatalogue());
    const other = computed(() => fp.spec.map.other);
    function add(key) { if (key && !other.value.keys.includes(key)) other.value.keys.push(key); }
    function remove(key) { const i = other.value.keys.indexOf(key); if (i >= 0) other.value.keys.splice(i, 1); }
    return {
      fp, other, add, remove, labelOf,
      DOES: 'You choose what the map looks at: the fluorescence surfaces (EEMs), the absorbance spectra, or both at the same time. Every sample is placed on a small grid of units, each unit holding a typical EEM or spectrum; neighbouring units hold similar ones, so similar samples sit together.',
      CHECK: "That the number of samples is what you expect (a map needs at least 8 samples and works best with dozens or more) and that the pre-treatment fits your question. Leave it at 'As measured' to describe intensities; choose a 'shape only' treatment (divide by the maximum, for example) to map the shape of the EEMs or spectra and ignore how much material there is.",
      WORDS: { both: 'One map of fluorescence and absorbance at the same time: every sample gets one place that describes both. Unless you change the weights below, each variable counts equally.' },
    };
  },
  template: `
  <div>
    <FpcaStepWhat :explain-does="DOES" :explain-check="CHECK" :words="WORDS" :weights="true" @run-recommended="$emit('run-recommended')"/>
    <div class="card flat som-other" data-som-other>
      <div class="card-head"><h3>Other variables instead</h3></div>
      <div class="card-body col gap-12">
        <p class="fp-note">Instead of whole EEMs or spectra you can map any numbers you have for each sample: scores of a PCA, PARAFAC or autoencoder, fluorescence indices, metadata. This is the way self-organising maps are used most often in the literature.</p>
        <Switch v-model="other.on" label="Map other variables instead of the EEMs and spectra" data-som-other-switch/>
        <template v-if="other.on">
          <div class="row wrap">
            <div class="field" style="min-width:260px"><label>Add a variable</label><VarSelect :model-value="null" :kinds="['numeric']" none-label="Choose a variable …" small @update:modelValue="add"/></div>
          </div>
          <div class="row wrap" v-if="other.keys.length"><span class="chip" v-for="k in other.keys" :key="k">{{ labelOf(k) }}<span class="x" role="button" tabindex="0" :aria-label="'Remove ' + labelOf(k)" @click="remove(k)" @keydown.enter="remove(k)">×</span></span></div>
          <Switch v-model="other.standardize" label="Standardise the variables (divide each by its standard deviation)" data-som-other-standardize/>
          <Callout kind="info">Steps 2 to 4 (domain, representation, smoothing) are not needed for other variables, and the map cannot be saved as a model that new samples are applied to. The prototype pictures are the mean EEMs of the samples of each unit and of its neighbours.</Callout>
        </template>
      </div>
    </div>
  </div>`,
};

const NOT_NEEDED = 'These choices are not needed while "Other variables instead" is switched on in step 1.';

export const SomStepDomain = {
  name: 'SomStepDomain',
  components: { FpcaStepDomain, FpcaFrame, Callout },
  setup() { return { other: computed(() => fp.spec.map.other), NOT_NEEDED }; },
  template: `<FpcaFrame v-if="other.on" :n="2"><Callout kind="info">{{ NOT_NEEDED }}</Callout></FpcaFrame><FpcaStepDomain v-else method="the map"/>`,
};

export const SomStepRepr = {
  name: 'SomStepRepr',
  components: { Icon, Seg, Callout, FpcaStepRepr, FpcaFrame },
  setup() {
    const other = computed(() => fp.spec.map.other);
    const functional = computed(() => isFunctional(fp.spec));
    const choice = computed({ get: () => (functional.value ? 'functional' : 'pixels'), set: (v) => setFunctional(fp.spec, v === 'functional') });
    return {
      other, choice, functional, NOT_NEEDED,
      DOES: 'Chooses what the map sees. In the functional version every EEM and every absorbance spectrum is first written as a smooth function (a mean plus a sum of a few building blocks) and the map works on the weights of the blocks; the distance between two samples is then the distance between their smooth functions. In the pixel version it works on the observed values themselves.',
      OPTIONS: [{ value: 'functional', label: 'On the coefficients of a basis (functional)', title: 'The map works on the weights of a smooth basis: fewer, noise-free coordinates' }, { value: 'pixels', label: 'On the pixels', title: 'The map works on the observed values themselves' }],
    };
  },
  template: `
  <FpcaFrame v-if="other.on" :n="3"><Callout kind="info">{{ NOT_NEEDED }}</Callout></FpcaFrame>
  <FpcaStepRepr v-else :kinds="['gp', 'bspline', 'cosine']" :explain-does="DOES">
    <template #top>
      <div class="card flat som-functional" data-som-functional-card>
        <div class="card-head"><h3>Functional version</h3></div>
        <div class="card-body col gap-12">
          <Seg v-model="choice" :options="OPTIONS" data-som-functional-seg/>
          <div class="fp-note" v-if="functional" data-som-functional="functional"><b>The functional version.</b> The map sees a few smooth, noise-free numbers per sample instead of thousands of noisy pixels, and the distance between samples is the distance between their smooth functions. This is the "functional SOM" of the literature. Choose how the functions are written down below, and how much to smooth in step 4.</div>
          <div class="fp-note" v-else data-som-functional="pixels"><b>The pixel version.</b> Nothing is assumed about smoothness: every observed pixel is one coordinate, as in the classic self-organising map of whole EEMs. Use it as a reference, or when the functional version hides something you want to see. Noise stays in the data, so weak pixels add their noise to the distance.</div>
        </div>
      </div>
    </template>
    <template #pixels>
      <Callout kind="info" data-som-pixels-note>There is nothing to choose for the pixel version: no basis and no smoothing are used, and pixels that are missing in some samples are left out of the distance between that sample and a prototype. Switch to the functional version above to choose a Gaussian-process basis, B-splines or cosines.</Callout>
    </template>
  </FpcaStepRepr>`,
};

export const SomStepSmooth = {
  name: 'SomStepSmooth',
  components: { FpcaStepSmooth, FpcaFrame, Callout },
  setup() {
    return { other: computed(() => fp.spec.map.other), NOT_NEEDED, DOES: 'Measured EEMs and spectra contain noise. In the functional version smoothing removes it from every sample before the map sees the coefficients. The amount of smoothing is not guessed: it is chosen from your data (by maximum likelihood, which also estimates the noise level, or by cross-validation). Too little smoothing leaves noise in the data; too much removes real detail.' };
  },
  template: `<FpcaFrame v-if="other.on" :n="4"><Callout kind="info">{{ NOT_NEEDED }}</Callout></FpcaFrame><FpcaStepSmooth v-else :explain-does="DOES"/>`,
};
