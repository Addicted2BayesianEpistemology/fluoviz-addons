// Step 6 of the Autoencoders page: what will be done (in words), the training settings, the button that trains the network, the live chart and the way to cancel.
import { computed, watch, onMounted } from 'vue';
import { Icon } from '/static/js/icons.js';
import { Field, NumField, Callout, Switch } from '/static/js/components/Common.js';
import { JobBar } from '/static/js/components/JobBar.js';
import { FpcaFrame } from '/static/js/components/FpcaStepper.js';
import { FpcaExplain } from '/static/js/components/FpcaCommon.js';
import { KERNEL_NAMES, dimOf } from '/static/js/components/FpcaState.js';
import { preName } from '/static/js/components/FpcaStepWhat.js';
import { debounce, fmt } from '/static/js/util.js';
import { fp, ctx, aeJobs, requestBody, latentOf, hiddenOf, isFunctional, variablesOf, sampleCount, FEW } from './AeState.js';
import { runFit, cancelHeavy, loadDefaults, runNetCheck, dropQueued } from './AeActions.js';
import { nTxt } from './AeCommon.js';
import { AeLive } from './AeScans.js';

const NICE = { fluorescence: 'Fluorescence', absorbance: 'Absorbance' };

/** The choices of steps 1 to 6 in plain words, one line each. */
export function describeAe(spec = fp.spec, data = fp.data, check = fp.ae.check.ans, defaults = fp.ae.defaults) {
  const n = spec.net, out = [];
  for (const name of variablesOf(spec)) {
    const v = spec[name], b = v.basis, d = dimOf(name);
    const px = data ? (name === 'fluorescence' ? data.fluorescence.n_pixels : data.absorbance.n_wl) : null;
    const parts = [px ? `${nTxt(px)} ${name === 'fluorescence' ? 'pixels' : 'wavelengths'}` : (name === 'fluorescence' ? 'pixels' : 'wavelengths')];
    if (v.transform !== 'none') parts.push(`pre-treatment: ${preName(name, v.transform).toLowerCase()}`);
    if (b.kind === 'pixels') parts.push('the observed values themselves (pixel version: no basis, no smoothing)');
    else {
      const ls = b.lmode === 'typed' ? ` ${b.typed.slice(0, d).map((x) => Number(x)).join(' × ')} nm` : (b.lmode === 'ml' ? ', length-scales by maximum likelihood' : ', length-scales from the variogram');
      parts.push(b.kind === 'gp' ? `Gaussian-process basis (${KERNEL_NAMES[b.kernel]}${d === 2 ? `, ${b.structure}` : ''}${ls})` : b.kind === 'bspline' ? `B-splines (${d === 2 ? `${b.bs[0]} × ${b.bs[1]} per axis` : b.bs[0]})` : `cosines (${d === 2 ? `${b.cos[0]} × ${b.cos[1]} per axis` : b.cos[0]})`);
      const s = v.smooth;
      parts.push(s.method === 'ml' ? 'maximum-likelihood smoothing' : s.method === 'gcv' ? 'smoothing by generalised cross-validation' : s.method === 'fixed' ? 'fixed smoothing' : 'no smoothing');
    }
    out.push(`${NICE[name]}: ${parts.join(', ')}`);
  }
  const code = latentOf(n, defaults);
  out.push(`Network: ${n.family === 'vae' ? 'variational autoencoder' : 'autoencoder'} with ${code} ${code === 1 ? 'code' : 'codes'}, layers ${hiddenOf(n).join(' → ')} → ${code}, activation ${n.activation}${n.skip ? ', starting from the PCA (linear shortcut)' : ', classical start (no shortcut)'}${n.family === 'vae' ? `, β = ${n.vae.beta}${n.vae.prior === 'mixture' ? `, mixture of ${n.vae.n_components} clouds` : ''}` : ''}`);
  const extras = [];
  if (Number(n.dropout) > 0) extras.push(`dropout ${n.dropout}`);
  if (Number(n.noise) > 0) extras.push(`noise ${n.noise} SD added to the input`);
  if (Number(n.mask) > 0) extras.push(`${Math.round(n.mask * 100)} % of the inputs hidden`);
  out.push(`Regularisation: weight decay ${n.weight_decay}${extras.length ? `, ${extras.join(', ')}` : ''}`);
  out.push(Number(n.smoothify) > 0 ? `Smoothify: ${fmt(Number(n.smoothify), 3)}${isFunctional(spec) ? (n.smoothify_metric === 'ridge' ? ' (plain size of the coefficients)' : ' (smoothness of the representation)') : ' (plain penalty on the size of the values)'}` : 'Smoothify: off');
  const val = check && check.validation ? String(check.validation.words || '').replace(/\.\s*$/, '') : `${Math.round(n.val_fraction * 100)} % of the samples are held out (fewer than 30 samples: ${n.folds}-fold validation)`;
  out.push(`Training: up to ${nTxt(n.epochs)} epochs, stopping after ${nTxt(n.patience)} without improvement; ${val}; ${n.n_restarts} random ${n.n_restarts === 1 ? 'start' : 'starts'}, seed ${n.seed}`);
  return out;
}

export const AeStepTrain = {
  name: 'AeStepTrain',
  components: { Icon, Field, NumField, Callout, Switch, JobBar, FpcaFrame, FpcaExplain, AeLive },
  setup() {
    const net = computed(() => fp.spec.net);
    const A = computed(() => fp.ae);
    const running = computed(() => aeJobs.fit.state.running);
    const anyRunning = computed(() => aeJobs.fit.state.running || aeJobs.scan.state.running);
    const lines = computed(() => describeAe());
    const issues = computed(() => ctx.allIssues());
    const check = computed(() => A.value.check);
    const blocking = computed(() => {
      const out = [], a = check.value.ans, d = A.value.defaults;
      if (a && check.value.key === JSON.stringify(requestBody())) for (const b of a.blocking || []) out.push(b);
      if (d && d.torch && d.torch.available === false) out.push(d.torch.words || 'Training needs PyTorch. Open Settings → Add-ons and press Install for Autoencoders (if it is already listed as installed, remove it and install it again).');
      return out;
    });
    const nSamples = computed(() => (check.value.ans ? check.value.ans.n : sampleCount()));
    const few = computed(() => { const c = sampleCount(); return c !== null && c < FEW && c >= 4; });
    const time = computed(() => (check.value.ans && check.value.ans.estimate ? check.value.ans.estimate.words : ''));
    const canTrain = computed(() => !running.value && !A.value.preparing && !issues.value.length && !blocking.value.length);
    const why = computed(() => (running.value ? 'A network is being trained.' : A.value.preparing ? 'Asking the server for the advice …' : issues.value.length ? `Something needs to be fixed first: ${issues.value[0].message}` : blocking.value[0] || ''));
    const hasResult = computed(() => !!(fp.rid && A.value.res));
    onMounted(() => { loadDefaults(); soon(); });
    const soon = debounce(() => { runNetCheck(); }, 700);
    watch(() => JSON.stringify(requestBody()), () => soon());
    function train() { return runFit(); }
    return { fp, net, A, running, anyRunning, lines, issues, blocking, nSamples, few, FEW, time, canTrain, why, hasResult, train, cancelHeavy, dropQueued, aeJobs, nTxt };
  },
  template: `
  <FpcaFrame :n="6" next-label="See the results" :no-next="!hasResult" :next-disabled="!hasResult">
    <FpcaExplain does="Trains the network on your samples. Some of the samples are kept back (held out) and never used for the training: the loss on them tells how well the network will do on samples it has not seen. Training stops by itself when the held-out loss stops falling, and the best moment is kept. The network is trained from several random starts and the best start is kept."
                 check="That the held-out loss (the held-out line of the chart) falls together with the training loss and then levels off. If the training loss keeps falling while the held-out loss rises, the network is memorising the samples: use fewer codes, a smaller network or more weight decay."/>
    <div class="fp-two fp-two-ctl">
      <div class="col gap-12">
        <div class="card flat fp-summary" data-ae-summary>
          <div class="card-head"><h3>What will be done</h3></div>
          <div class="card-body"><ul class="fp-list"><li v-for="l in lines" :key="l">{{ l }}</li></ul>
            <p class="small muted mt-8" v-if="nSamples !== null">{{ nTxt(nSamples) }} samples take part.<span v-if="time"> Time: {{ time }}.</span></p></div>
        </div>
        <details class="fp-more" data-ae-train-settings>
          <summary>Training settings</summary>
          <div class="col gap-12 mt-8">
            <div class="fp-grid2">
              <Field label="Maximum number of epochs" help="An epoch is one pass of the network over all the training samples. Training stops earlier when the held-out loss stops falling. 1 to 5000." hint="Default 600."><NumField v-model="net.epochs" :min="1" :max="5000" :step="10" data-ae-epochs/></Field>
              <Field label="Patience (epochs without improvement)" help="Training stops when the held-out loss has not improved for this many epochs, and the best network is restored. 1 to 1000." hint="Default 60."><NumField v-model="net.patience" :min="1" :max="1000" :step="5" data-ae-patience/></Field>
              <Field label="Share of samples held out" help="The share of the samples that is kept back to test the network (at least 5 samples). 0.1 to 0.5." hint="Default 0.2."><NumField v-model="net.val_fraction" :min="0.1" :max="0.5" :step="0.05" data-ae-val-fraction/></Field>
              <Field label="Folds (fewer than 30 samples)" help="With fewer than 30 samples the samples are split into this many groups and every group is held out once, so that all the samples are used for the testing. 2 to 10." hint="Default 5."><NumField v-model="net.folds" :min="2" :max="10" :step="1" data-ae-folds/></Field>
              <Field label="Learning rate" help="The size of the steps of the training. A smaller rate is slower and steadier. 0.00001 to 0.1." hint="Default 0.001."><NumField v-model="net.lr" :min="0.00001" :max="0.1" :step="0.0005" data-ae-lr/></Field>
              <Field label="Batch size" help="How many samples are used for one step of the training. Empty = automatic (32 or fewer). 2 to 1024."><NumField v-model="net.batch_size" :min="2" :max="1024" :step="1" nullable placeholder="automatic" data-ae-batch/></Field>
              <Field label="Number of random starts" help="The network is trained this many times from different random numbers and the best one is kept; the spread between them is reported. 1 to 10." hint="Default 3."><NumField v-model="net.n_restarts" :min="1" :max="10" :step="1" data-ae-restarts/></Field>
              <Field label="Random seed" help="Makes everything repeatable: the same seed gives the same network." hint="Default 0."><NumField v-model="net.seed" :min="0" :step="1" data-ae-seed/></Field>
            </div>
            <Field label="Title of the result" help="The name under which the network is saved. Leave it empty for an automatic name."><input class="input" v-model="net.title" placeholder="automatic" maxlength="120" data-ae-title/></Field>
          </div>
        </details>
        <div class="card flat" v-if="few" data-ae-few><div class="card-body col gap-8">
          <Switch v-model="net.allow_few" label="Train on fewer than 12 samples anyway" data-ae-allow-few/>
          <div class="fp-note">With so few samples the network is very likely to memorise them, and the numbers on held-out samples are very uncertain: take the codes as a rough sketch only. At least 4 samples are needed.</div></div></div>
        <Callout kind="critical" v-if="issues.length"><b>Something needs to be fixed first.</b><ul class="fp-list"><li v-for="i in issues" :key="i.message">Step {{ i.step }}: {{ i.message }}</li></ul></Callout>
        <Callout kind="critical" v-for="b in blocking" :key="b" data-ae-block>{{ b }}</Callout>
        <div class="row wrap">
          <button type="button" class="btn primary" data-ae-train :disabled="!canTrain" @click="train"><Icon name="play" size="16"/>{{ running ? 'Training …' : 'Train the network' }}</button>
          <button type="button" class="btn danger" v-if="running" data-ae-cancel @click="cancelHeavy"><Icon name="stop-circle" size="16"/>Cancel</button>
          <span class="muted xs" v-if="!canTrain && why" data-ae-train-why>{{ why }}</span>
          <span class="muted xs" v-else-if="!running">The network is saved as a numbered result you can come back to.</span>
        </div>
        <div class="callout info" v-if="A.queued" data-ae-queued><Icon name="info-circle"/><div class="grow">Another computation is running: <b>{{ A.queued }}</b> starts right after it (a newer request replaces this one).</div><button class="btn xs" @click="dropQueued">Forget it</button></div>
        <div class="callout info" v-if="A.preparing" data-ae-preparing><Icon name="info-circle"/><div class="grow">Asking the server for the advice for your data, then the network is trained with the recommended settings …</div></div>
        <JobBar v-if="running" :job="aeJobs.fit" label="Training the network"/>
        <Callout kind="critical" v-if="A.fitError" data-ae-fit-error><b>The network could not be trained.</b><br>{{ A.fitError }}</Callout>
      </div>
      <div class="col gap-12">
        <AeLive/>
      </div>
    </div>
  </FpcaFrame>`,
};
