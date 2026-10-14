// Step 5 of the Autoencoders page: the network (type, number of codes, size, activation, regularisation, denoising, the options of the variational autoencoder, smoothify) and
// "The network in numbers" (computed by the server for the current settings, nothing is trained).
import { computed, watch, onMounted } from 'vue';
import { Icon } from '/static/js/icons.js';
import { Field, NumField, Seg, Callout, HelpTip, Switch } from '/static/js/components/Common.js';
import { FpcaFrame } from '/static/js/components/FpcaStepper.js';
import { FpcaExplain } from '/static/js/components/FpcaCommon.js';
import { debounce, fmt, clamp } from '/static/js/util.js';
import { fp, requestBody, latentOf, hiddenOf, parseHidden, isFunctional, SIZES, ACTIVATIONS, ACTIVATION_WORDS } from './AeState.js';
import { loadDefaults, runNetCheck } from './AeActions.js';
import { AeSection, AeStat, nTxt, plural } from './AeCommon.js';
import { AeScanBlock, AeLatentScanChart, AeLCurveChart } from './AeScans.js';

const SLIDER_MAX = 100;                                      // 0 = off, then 0.0001 … 10 on a logarithmic scale (the range of the smoothify scan)
export const sliderToValue = (p) => (p <= 0 ? 0 : Number((1e-4 * 10 ** (((p - 1) * 5) / (SLIDER_MAX - 1))).toPrecision(3)));
export const valueToSlider = (v) => (!(v > 0) ? 0 : clamp(Math.round(1 + (Math.log10(v / 1e-4) * (SLIDER_MAX - 1)) / 5), 1, SLIDER_MAX));
const SLIDER_MARKS = [{ p: 0, t: '0 (off)' }, { p: 1, t: '0.0001' }, { p: 21, t: '0.001' }, { p: 40, t: '0.01' }, { p: 60, t: '0.1' }, { p: 80, t: '1' }, { p: 100, t: '10' }];

const FAMILIES = [
  { id: 'ae', icon: 'brain', title: 'Autoencoder', text: 'Squeezes every sample into a few numbers, the code, and rebuilds the sample from them. Use it to describe and compare samples.' },
  { id: 'vae', icon: 'layers-alt', title: 'Variational autoencoder', text: 'Also learns what a typical code looks like. It can draw new, similar samples and tell when a sample is unusual.' },
];

export const AeStepNetwork = {
  name: 'AeStepNetwork',
  components: { Icon, Field, NumField, Seg, Callout, HelpTip, Switch, FpcaFrame, FpcaExplain, AeSection, AeStat, AeScanBlock, AeLatentScanChart, AeLCurveChart },
  setup() {
    const net = computed(() => fp.spec.net);
    const A = computed(() => fp.ae);
    const advice = computed(() => (A.value.defaults ? A.value.defaults.advice : null));
    const latent = computed(() => latentOf(net.value, A.value.defaults));
    const isVae = computed(() => net.value.family === 'vae');
    const pixelsOnly = computed(() => !isFunctional(fp.spec));
    const check = computed(() => A.value.check);
    const ans = computed(() => check.value.ans);
    const fresh = computed(() => !!(ans.value && check.value.key === JSON.stringify(requestBody())));

    onMounted(() => { loadDefaults(); soon(); });
    const soon = debounce(() => { runNetCheck(); }, 700);
    watch(() => JSON.stringify(requestBody()), () => soon());
    watch(() => [fp.spec.fluorescence.min_coverage, fp.spec.absorbance.processed, fp.spec.what], debounce(() => loadDefaults(), 500));

    const SIZE_OPTIONS = [
      { value: 'small', label: 'Small', title: '64, 32 neurons in the two hidden layers: the safe choice for tens to a few hundred samples' },
      { value: 'medium', label: 'Medium', title: '128, 64' },
      { value: 'large', label: 'Large', title: '256, 128, 64: only with hundreds of samples' },
      { value: 'custom', label: 'Custom', title: 'Type the widths of the layers yourself' },
    ];
    const hiddenOk = computed(() => net.value.size !== 'custom' || !!parseHidden(net.value.hidden));
    const sizeWords = computed(() => `The encoder has the layers ${hiddenOf(net.value).join(' → ')} and ends in ${latent.value} ${latent.value === 1 ? 'code' : 'codes'}; the decoder is a mirror image of it.`);

    // smoothify: a logarithmic slider and a number
    const sliderPos = computed({ get: () => valueToSlider(Number(net.value.smoothify)), set: (p) => { net.value.smoothify = sliderToValue(Number(p)); } });
    const smoothText = computed(() => (Number(net.value.smoothify) > 0 ? `Smoothify ${fmt(Number(net.value.smoothify), 3)}: the rebuilt samples are pushed towards smooth functions.` : 'Smoothify is off (0).'));
    const smoothWho = computed(() => (pixelsOnly.value
      ? 'With the pixels there is no smoothness to measure: smoothify then acts as a plain penalty on the size of the values.'
      : net.value.smoothify_metric === 'ridge' ? 'The penalty is the plain size (the sum of squares) of the coefficients.'
        : 'With the Gaussian-process basis this is exactly the penalty of the Gaussian process (large coefficients of rough components cost the most); with B-splines or cosines it is the roughness penalty of the representation.'));

    // the network in numbers
    const stats = computed(() => {
      const a = ans.value; if (!a) return [];
      const n = a.network || {};
      return [
        { k: 'inputs', l: 'Inputs', v: nTxt(a.input && a.input.n_inputs), s: a.input && a.input.blocks ? a.input.blocks.map((b) => `${b.name}: ${nTxt(b.n_inputs)} ${b.mode === 'functional' ? 'coefficients' : 'pixels'}`).join(' · ') : '' },
        { k: 'codes', l: 'Codes', v: nTxt(n.latent_dim), s: n.family === 'vae' ? 'variational' : 'deterministic' },
        { k: 'params', l: 'Parameters', v: nTxt(n.n_params), s: 'numbers the network learns' },
        { k: 'per-param', l: 'Samples per parameter', v: fmt(n.samples_per_param, 2), s: `${nTxt(a.n)} samples take part` },
        { k: 'per-code', l: 'Samples per code', v: fmt(n.samples_per_latent, 3), s: 'the more the better' },
        { k: 'start', l: 'Start', v: net.value.skip ? 'PCA (linear shortcut)' : 'random (classical)', s: net.value.skip ? 'learns what bends add on top' : 'a classical autoencoder' },
        { k: 'validation', l: 'Validation', v: a.validation ? (a.validation.mode === 'kfold' ? `${a.validation.folds} folds` : `${nTxt(a.validation.n_val)} held out`) : '–', s: a.validation ? a.validation.words : '' },
        { k: 'time', l: 'Time', v: a.estimate ? a.estimate.words : '–', s: '' },
      ];
    });
    const warnings = computed(() => {
      const a = ans.value, out = [];
      if (a) {
        for (const w of a.warnings || []) out.push(w);
      }
      const d = A.value.defaults;
      if (d && d.advice) for (const w of d.advice.warnings || []) if (!out.includes(w) && !/memoris|parameters/i.test(w)) out.push(w);          // (the advice describes the default network: the numbers of the network on the page come from the check)
      return out;
    });
    const blocking = computed(() => {
      const out = [], a = ans.value, d = A.value.defaults;
      if (a) for (const b of a.blocking || []) out.push(b);
      else if (d && d.advice) for (const b of d.advice.blocking || []) out.push(b);
      if (d && d.torch && d.torch.available === false) out.push(d.torch.words || 'Training needs PyTorch. Open Settings → Add-ons and press Install (or Repair) for Autoencoders.');
      return out;
    });
    const scanOptions = computed(() => ({ families: [net.value.family] }));
    const useLatent = (n) => { net.value.latent_dim = n; };
    const useSmooth = (s) => { net.value.smoothify = s; };
    function pickFamily(id) { net.value.family = id; }
    return { fp, net, A, advice, latent, isVae, pixelsOnly, check, ans, fresh, FAMILIES, SIZE_OPTIONS, SIZES, ACTIVATIONS, ACTIVATION_WORDS, hiddenOk, sizeWords, sliderPos, smoothText, smoothWho, stats, warnings, blocking,
      scanOptions, useLatent, useSmooth, pickFamily, plural, nTxt, fmt, SLIDER_MAX, SLIDER_MARKS };
  },
  template: `
  <FpcaFrame :n="5" next-label="Go to the training">
    <FpcaExplain does="Chooses the network. An autoencoder has two halves: the encoder squeezes a sample into a few numbers (the code) and the decoder rebuilds the sample from them. The network is trained so that the rebuilt sample is as close as possible to the real one. Here you choose how many codes there are, how big the network is, and how the codes are learned."
                 check="The number of codes and the notices on the right. With a few dozen samples the safe choice is a small network, a few codes and the default regularisation; the latent-dimension scan shows how many codes are worth having. Leave smoothify at 0 unless you want the rebuilt samples to be smoother."/>
    <div class="fp-two fp-two-ctl">
      <div class="col gap-12 ae-controls">
        <AeSection title="Type of network" sentence="A deterministic network, or one that also learns the distribution of the codes.">
          <div class="ae-cards2" role="radiogroup" aria-label="Type of network">
            <div v-for="c in FAMILIES" :key="c.id" class="fp-card" :class="{ on: net.family === c.id }" role="radio" :aria-checked="net.family === c.id ? 'true' : 'false'" tabindex="0" :data-ae-family="c.id"
                 @click="pickFamily(c.id)" @keydown.enter.prevent="pickFamily(c.id)" @keydown.space.prevent="pickFamily(c.id)">
              <span class="fp-card-head"><Icon :name="c.icon" size="22"/><b>{{ c.title }}</b><Icon v-if="net.family === c.id" name="check-circle" size="18" class="fp-tick"/></span>
              <span class="fp-card-text">{{ c.text }}</span>
            </div>
          </div>
        </AeSection>

        <AeSection title="Number of codes (latent dimension)" sentence="How many numbers describe one sample. Few codes give a simple, robust description; many codes rebuild the samples better but can start to memorise them.">
          <div class="row wrap ae-latent-row">
            <Field label="Number of codes" help="A whole number between 1 and 32. The advice below is computed from your data.">
              <NumField :model-value="latent" :min="1" :max="32" :step="1" data-ae-latent @update:modelValue="useLatent(Math.round($event))"/>
            </Field>
            <button type="button" class="btn quiet sm" v-if="advice && net.latent_dim !== null && net.latent_dim !== advice.latent_dim" @click="net.latent_dim = null"><Icon name="rocket" size="15"/>Use the advice ({{ advice.latent_dim }})</button>
          </div>
          <div class="fp-note" data-ae-latent-advice v-if="advice && advice.latent_reason"><b>Advice:</b> {{ advice.latent_reason }}</div>
          <div class="fp-note muted" v-else-if="A.defaultsBusy">Reading the data to give advice …</div>
          <AeScanBlock kind="latent" source="request" label="Latent-dimension scan" :options="scanOptions" sentence="Trains small networks for 1, 2, 3 … codes and compares them with PCA or functional PCA. It takes minutes.">
            <template #default="{ ans: s }">
              <AeLatentScanChart :ans="s"/>
              <p class="fp-note" v-if="s.words">{{ s.words }}</p>
              <div class="row wrap" v-if="s.recommended"><span class="fp-note"><b>Recommended: {{ s.recommended.latent_dim }} codes.</b> {{ s.recommended.reason }}.</span>
                <button type="button" class="btn sm" data-ae-use-latent :disabled="latent === s.recommended.latent_dim" @click="useLatent(s.recommended.latent_dim)">Use {{ s.recommended.latent_dim }} codes</button></div>
            </template>
          </AeScanBlock>
        </AeSection>

        <AeSection title="Size of the network" :sentence="sizeWords">
          <Seg v-model="net.size" :options="SIZE_OPTIONS" data-ae-size-seg/>
          <Field v-if="net.size === 'custom'" label="Layer widths" help="One to four whole numbers between 2 and 1024, separated by commas: the number of neurons of each hidden layer of the encoder, for example 64, 32." :hint="hiddenOk ? '' : 'Type one to four whole numbers between 2 and 1024, for example 64, 32.'">
            <input class="input" :class="{ invalid: !hiddenOk }" v-model="net.hidden" placeholder="64, 32" data-ae-hidden/>
          </Field>
        </AeSection>

        <AeSection title="Activation" sentence="The activation is the bend each neuron puts into its output. silu and tanh are smooth, so the rebuilt functions are smooth. linear makes the network equivalent to PCA: use it as a reference.">
          <Field label="Activation" help="The curve each neuron applies to its output. The default (silu) is smooth.">
            <select class="select" v-model="net.activation" data-ae-activation><option v-for="a in ACTIVATIONS" :key="a" :value="a">{{ a }} ({{ ACTIVATION_WORDS[a] }})</option></select>
          </Field>
        </AeSection>

        <AeSection title="Regularisation" sentence="Settings against memorising the samples. The defaults suit tens to a few hundred samples.">
          <div class="fp-grid2">
            <Field label="Dropout" help="The share of the neurons of the encoder that are switched off at random while training (never when the network is used). 0 to 0.5." hint="Default 0."><NumField v-model="net.dropout" :min="0" :max="0.5" :step="0.05" data-ae-dropout/></Field>
            <Field label="Weight decay" help="A small pull of all the weights towards zero, which keeps the network from over-fitting. 0 to 1." hint="Default 0.001."><NumField v-model="net.weight_decay" :min="0" :max="1" :step="0.001" data-ae-weight-decay/></Field>
          </div>
          <div class="fp-note"><b>Denoising.</b> Denoising trains the network to rebuild the clean sample from a damaged copy. The codes become more robust, and missing values are handled better.</div>
          <div class="fp-grid2">
            <Field label="Noise added to the input (in standard deviations)" help="Gaussian noise added to every input while training, in units of the standard deviation of the data. 0 to 2." hint="Default 0."><NumField v-model="net.noise" :min="0" :max="2" :step="0.05" data-ae-noise/></Field>
            <Field label="Share of inputs hidden" help="The share of the inputs that is set to zero at random while training. 0 to 0.9." hint="Default 0."><NumField v-model="net.mask" :min="0" :max="0.9" :step="0.05" data-ae-mask/></Field>
          </div>
        </AeSection>

        <details class="fp-more ae-advanced" data-ae-advanced>
          <summary>Advanced network settings</summary>
          <div class="col gap-12 mt-8">
            <Switch v-model="net.skip" label="Start from PCA (linear shortcut)" data-ae-skip/>
            <div class="fp-note" data-ae-skip-words>The network starts as the PCA and learns only what the bends add on top of it, so it is never worse than PCA on held-out samples; switch it off for a classical autoencoder.<template v-if="net.activation === 'linear' && net.skip"> With the <i>linear</i> activation and the shortcut the network is the PCA itself.</template></div>
          </div>
        </details>

        <details class="fp-more ae-vae" :open="isVae" data-ae-vae-options>
          <summary>Options of the variational autoencoder</summary>
          <div class="col gap-12 mt-8">
            <div class="fp-note" v-if="!isVae" data-ae-vae-why>Only for the variational autoencoder: choose it in <i>Type of network</i> to change these.</div>
            <div class="fp-grid2">
              <Field label="Weight of the typical-code term (β)" help="How strongly the codes are pulled towards a typical, regular cloud. Below 1: sharper rebuilt samples; above 1: more regular codes. Default 1."><NumField v-model="net.vae.beta" :min="0" :max="100" :step="0.1" :disabled="!isVae" data-ae-beta/></Field>
              <Field label="Warm-up (epochs)" help="The pull towards the typical cloud is increased slowly over this many epochs, which avoids codes that carry no information. Empty = 20 % of the epochs."><NumField v-model="net.vae.warmup" :min="0" :max="5000" :step="1" nullable placeholder="automatic" :disabled="!isVae" data-ae-warmup/></Field>
              <Field label="Free bits (nats per code)" help="A code is not pulled towards the typical cloud while it carries less than this much information. 0 to 5. Default 0."><NumField v-model="net.vae.free_bits" :min="0" :max="5" :step="0.05" :disabled="!isVae" data-ae-freebits/></Field>
            </div>
            <Field label="Noise level of the data (σ)" help="How much the rebuilt sample is allowed to differ from the real one. Learned: the network finds it. Fixed: you give it.">
              <div class="row wrap"><Seg v-model="net.vae.sigma_mode" :options="[{ value: 'learned', label: 'Learned' }, { value: 'fixed', label: 'Fixed' }]"/><NumField v-if="net.vae.sigma_mode === 'fixed'" v-model="net.vae.sigma" :min="0.000001" :step="0.01" :disabled="!isVae"/></div>
            </Field>
            <Field label="Prior" help="What a typical code looks like. One cloud: a normal distribution. A mixture of clouds lets the network group the samples by itself (the groups become published variables).">
              <div class="row wrap"><Seg v-model="net.vae.prior" :options="[{ value: 'normal', label: 'One cloud (normal)' }, { value: 'mixture', label: 'Mixture of clouds' }]" data-ae-prior/>
                <Field v-if="net.vae.prior === 'mixture'" label="Number of clouds"><NumField v-model="net.vae.n_components" :min="2" :max="20" :step="1" :disabled="!isVae"/></Field></div>
            </Field>
          </div>
        </details>

        <AeSection title="Smoothify" sentence="A setting that makes the rebuilt samples smoother.">
          <div class="ae-smooth">
            <div class="fp-slider">
              <input type="range" class="ae-range" min="0" :max="SLIDER_MAX" step="1" v-model.number="sliderPos" aria-label="Smoothify (logarithmic slider from 0 to 10)" data-ae-smoothify-slider/>
              <NumField v-model="net.smoothify" :min="0" :max="100" :step="0.1" data-ae-smoothify/>
            </div>
            <div class="ae-slider-ends muted xs" aria-hidden="true"><span v-for="m in SLIDER_MARKS" :key="m.t" :style="{ left: (m.p / SLIDER_MAX * 100) + '%' }" :class="{ first: m.p === 0, last: m.p === SLIDER_MAX }">{{ m.t }}</span></div>
          </div>
          <div class="fp-note" data-ae-smoothify-words><b>{{ smoothText }}</b> Smoothify penalises big coefficients of the rebuilt sample, above all in the rough directions. The larger it is, the more the rebuilt EEMs move towards smooth functions, at the price of a slightly larger rebuilding error. {{ smoothWho }}</div>
          <Field label="Smoothness is measured by" help="Which penalty smoothify uses.">
            <select class="select" v-model="net.smoothify_metric" :disabled="pixelsOnly" data-ae-smoothify-metric>
              <option value="smooth">The smoothness of the representation</option><option value="ridge">The plain size of the coefficients (ridge)</option></select>
          </Field>
          <AeScanBlock kind="smoothify" source="request" label="Smoothify scan" sentence="Trains the network for several smoothify values (0 to 10) and draws the error against the roughness. It takes minutes.">
            <template #default="{ ans: s }">
              <AeLCurveChart :ans="s"/>
              <p class="fp-note" v-if="s.words">{{ s.words }}</p>
              <div class="row wrap" v-if="Number.isFinite(s.suggested)"><span class="fp-note"><b>Suggested: {{ fmt(s.suggested, 3) }}.</b></span>
                <button type="button" class="btn sm" data-ae-use-suggested :disabled="Number(net.smoothify) === s.suggested" @click="useSmooth(s.suggested)">Use {{ fmt(s.suggested, 3) }}</button></div>
            </template>
          </AeScanBlock>
        </AeSection>
      </div>

      <div class="col gap-12 ae-side">
        <div class="card flat fp-summary" data-ae-numbers :class="{ 'fp-dim': ans && !fresh }">
          <div class="card-head"><h3>The network in numbers</h3></div>
          <div class="card-body">
            <p class="fp-note muted" v-if="!ans && check.busy">Counting …</p>
            <p class="fp-note muted" v-else-if="!ans && check.error" data-ae-check-error>{{ check.error }}</p>
            <p class="fp-note muted" v-else-if="!ans">The numbers appear when the settings of steps 1 to 5 are complete.</p>
            <dl class="fp-facts" v-else><template v-for="s in stats" :key="s.k"><dt>{{ s.l }}</dt><dd :data-ae-stat="s.k" :title="s.s">{{ s.v }}<span class="muted xs" v-if="s.s"> · {{ s.s }}</span></dd></template></dl>
          </div>
        </div>
        <Callout kind="critical" v-for="b in blocking" :key="b" data-ae-block>{{ b }}</Callout>
        <Callout kind="warning" v-for="w in warnings" :key="w" data-ae-warning>{{ w }}</Callout>
      </div>
    </div>
  </FpcaFrame>`,
};
