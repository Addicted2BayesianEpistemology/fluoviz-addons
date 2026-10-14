// Steps 1 to 4 of the Autoencoders page ARE the Functional PCA's step components (same controls, same routes `/api/fpca/*`); these thin wrappers only give them the words of
// this page and, in step 3, the functional switch.  The components find the state of this page through the context that page.js provides.
import { computed } from 'vue';
import { Icon } from '/static/js/icons.js';
import { Seg, Callout } from '/static/js/components/Common.js';
import { FpcaStepWhat } from '/static/js/components/FpcaStepWhat.js';
import { FpcaStepDomain } from '/static/js/components/FpcaStepDomain.js';
import { FpcaStepRepr } from '/static/js/components/FpcaStepRepr.js';
import { FpcaStepSmooth } from '/static/js/components/FpcaStepSmooth.js';
import { fp, isFunctional, setFunctional } from './AeState.js';

export const AeStepWhat = {
  name: 'AeStepWhat',
  components: { FpcaStepWhat },
  emits: ['run-recommended'],
  setup() {
    return {
      DOES: 'You choose what the network looks at: the fluorescence surfaces (EEMs), the absorbance spectra, or both at the same time. The network learns a code, a handful of numbers, that describes each sample, and rebuilds the sample from it.',
      CHECK: "That the number of samples is what you expect (an autoencoder needs at least 12 samples and works best with dozens or more) and that the pre-treatment fits your question. Leave it at 'As measured' to describe intensities; choose a 'shape only' treatment to describe the shape of the EEMs or spectra and ignore how much material there is.",
      WORDS: { both: 'One network for fluorescence and absorbance at the same time: every sample gets one code that describes both. Each variable counts equally.' },
    };
  },
  template: `<FpcaStepWhat :explain-does="DOES" :explain-check="CHECK" :words="WORDS" :weights="false" @run-recommended="$emit('run-recommended')"/>`,
};

export const AeStepDomain = {
  name: 'AeStepDomain',
  components: { FpcaStepDomain },
  template: `<FpcaStepDomain method="the autoencoder"/>`,
};

export const AeStepRepr = {
  name: 'AeStepRepr',
  components: { Icon, Seg, Callout, FpcaStepRepr },
  setup() {
    const functional = computed(() => isFunctional(fp.spec));
    const choice = computed({ get: () => (functional.value ? 'functional' : 'pixels'), set: (v) => setFunctional(fp.spec, v === 'functional') });
    return {
      choice, functional,
      DOES: 'Chooses what the network sees. In the functional version every EEM and every absorbance spectrum is first written as a smooth function (a mean plus a sum of a few building blocks) and the network works on the weights of the blocks. In the pixel version it works on the observed values themselves.',
      OPTIONS: [{ value: 'functional', label: 'On the coefficients of a basis (functional)', title: 'The network works on the weights of a smooth basis: fewer, noise-free inputs' }, { value: 'pixels', label: 'On the pixels', title: 'The network works on the observed values themselves' }],
    };
  },
  template: `
  <FpcaStepRepr :kinds="['gp', 'bspline', 'cosine']" :explain-does="DOES">
    <template #top>
      <div class="card flat ae-functional" data-ae-functional-card>
        <div class="card-head"><h3>Functional version</h3></div>
        <div class="card-body col gap-12">
          <Seg v-model="choice" :options="OPTIONS" data-ae-functional-seg/>
          <div class="fp-note" v-if="functional" data-ae-functional="functional"><b>The functional version.</b> The network sees a few smooth, noise-free numbers per sample instead of thousands of noisy pixels. This usually trains faster and gives smoother results, and the setting <b>smoothify</b> in step 5 can push the rebuilt samples further towards smooth functions. Choose how the functions are written down below, and how much to smooth in step 4.</div>
          <div class="fp-note" v-else data-ae-functional="pixels"><b>The pixel version.</b> Nothing is assumed about smoothness: every observed pixel is one input. Use it as a reference, or when the functional version hides something you want to see. Noise stays in the data, so the network may learn it.</div>
        </div>
      </div>
    </template>
    <template #pixels>
      <Callout kind="info" data-ae-pixels-note>There is nothing to choose for the pixel version: no basis and no smoothing are used, and pixels that are missing in some samples are simply left out of the error that the network tries to make small. Switch to the functional version above to choose a Gaussian-process basis, B-splines or cosines.</Callout>
    </template>
  </FpcaStepRepr>`,
};

export const AeStepSmooth = {
  name: 'AeStepSmooth',
  components: { FpcaStepSmooth },
  setup() {
    return { DOES: 'Measured EEMs and spectra contain noise. In the functional version smoothing removes it from every sample before the network sees the coefficients. The amount of smoothing is not guessed: it is chosen from your data (by maximum likelihood, which also estimates the noise level, or by cross-validation). Too little smoothing leaves noise in the data; too much removes real detail.' };
  },
  template: `<FpcaStepSmooth :explain-does="DOES"/>`,
};
