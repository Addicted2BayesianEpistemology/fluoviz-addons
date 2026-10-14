// The state of the Autoencoders page: ONE context made with the Functional PCA's machinery (steps 1 to 4 are its components), plus the autoencoder's own choices (`spec.net`)
// and answers (`fp.ae`).  Module level on purpose: the choices survive leaving the page (and are remembered per project through PUT /api/state/ui, key `autoencoders`), and
// a computation that was started keeps running while the person looks at another page.  When another project becomes the open one the answers are dropped (`fp.ae` is made again).
import { markRaw } from 'vue';
import { createFpca } from '/static/js/components/FpcaCtx.js';
import { defaultSpec, variablesOf, NAMES } from '/static/js/components/FpcaState.js';
import { useJob } from '/static/js/jobs.js';

export const BASE = '/api/addon/autoencoders';

// ---------------------------------------------------------------- the steps (the Help guide quotes these names: do not rename)
export const AE_STEPS = [
  { n: 1, id: 'what', label: '1 · What to model' },
  { n: 2, id: 'domain', label: '2 · Domain' },
  { n: 3, id: 'repr', label: '3 · Representation' },
  { n: 4, id: 'smooth', label: '4 · Smoothing' },
  { n: 5, id: 'network', label: '5 · Network' },
  { n: 6, id: 'train', label: '6 · Train' },
  { n: 7, id: 'results', label: '7 · Results' },
  { n: 8, id: 'checks', label: '8 · Checks' },
  { n: 9, id: 'use', label: '9 · Use' },
];

export const SIZES = { small: [64, 32], medium: [128, 64], large: [256, 128, 64] };
export const SIZE_LABEL = { small: 'Small', medium: 'Medium', large: 'Large', custom: 'Custom' };
export const ACTIVATIONS = ['silu', 'tanh', 'gelu', 'relu', 'softplus', 'elu', 'linear'];
export const ACTIVATION_WORDS = {
  silu: 'smooth, the default', tanh: 'smooth, bounded', gelu: 'smooth', relu: 'piecewise linear, rough functions', softplus: 'smooth version of relu', elu: 'smooth for negative values',
  linear: 'no bend at all: the network is then like PCA (a reference)',
};
export const STALE_WORDS = { raw: 'the raw data', corrections: 'the corrections', masks: 'the scatter removal or excised regions', window: 'the focus window', samples: 'the sample selection' };

// ---------------------------------------------------------------- the choices
export function netDefault() {
  return {
    family: 'ae', latent_dim: null,                            // null = "as advised" (the advice arrives from the server)
    size: 'small', hidden: '64, 32',                           // the widths typed for the custom size
    activation: 'silu', dropout: 0, weight_decay: 0.001, noise: 0, mask: 0,
    skip: true,                                                // the linear shortcut: the network starts as the PCA of the training samples
    allow_few: false,                                          // the person agreed to train on fewer than 12 samples (never fewer than 4)
    vae: { beta: 1, warmup: null, free_bits: 0, sigma_mode: 'learned', sigma: 0.1, prior: 'normal', n_components: 3 },
    smoothify: 0, smoothify_metric: 'smooth',
    epochs: 600, patience: 60, val_fraction: 0.2, folds: 5, lr: 0.001, batch_size: null, n_restarts: 3, seed: 0, title: '',
    memory: { fluorescence: 'gp', absorbance: 'gp' },          // the functional representation chosen before "On the pixels" was switched on
  };
}
export function defaultAeSpec() {
  const s = defaultSpec();
  s.net = netDefault();
  return s;
}
export const isFunctional = (spec) => variablesOf(spec).every((n) => spec[n].basis.kind !== 'pixels');
/** The functional switch of step 3: on = the network works on the coefficients of a basis (the one chosen before, the Gaussian-process basis at first), off = on the pixels. */
export function setFunctional(spec, on) {
  for (const name of NAMES) {
    const b = spec[name].basis, s = spec[name].smooth;
    if (on) { if (b.kind === 'pixels') { b.kind = spec.net.memory[name] || 'gp'; s.method = b.kind === 'gp' ? 'ml' : 'gcv'; } }
    else if (b.kind !== 'pixels') { spec.net.memory[name] = b.kind; b.kind = 'pixels'; s.method = 'none'; }
  }
}

/** The widths of the typed custom layers: [64, 32] (null when the text is not a list of 1 to 4 whole numbers between 2 and 1024). */
export function parseHidden(text) {
  const parts = String(text ?? '').split(/[,;\s]+/).filter(Boolean).map(Number);
  if (!parts.length || parts.length > 4 || !parts.every((x) => Number.isInteger(x) && x >= 2 && x <= 1024)) return null;
  return parts;
}
export const hiddenOf = (net) => (net.size === 'custom' ? parseHidden(net.hidden) : SIZES[net.size]) || SIZES.small;

// ---------------------------------------------------------------- the answers (made again with every project)
function freshAe() {
  return {
    defaults: null, defaultsKey: '', defaultsError: '', defaultsBusy: false,
    check: { key: '', ans: null, error: '', busy: false },        // POST /check: the network in numbers
    res: null, resRid: null, resError: '', resBusy: false,        // the result shown (GET /{rid}), prepared
    fitError: '', queued: '', preparing: false,
    live: freshLive(),
    tab7: 'latent', tab8: 'training', checkRows: 'total',
    latent: { x: 'z1', y: 'z2', color: null, symbol: null, sd: false, relate: [] },
    travView: { range: 3, show: 'images', all: false },
    sampView: { filter: 'all', find: '' },
    anomView: { method: 'mad', c: 3, level: 0.99 },
    interpView: { a: null, b: null, steps: 9, pos: 0.5 },
    genView: { n: 6, seed: 1, temperature: 1 },
    lossView: { what: 'total' },
    stabView: { n_runs: 5, bootstrap: false },
    latScanView: { max_dim: 8, both: false },
    sample: null,                                                  // the sample looked at (a click in the latent plot, the table of samples)
    scans: {},                                                     // kind -> { key, source, ans, error, busy, rid }
    extra: {},                                                     // traversals, interpolation, generation, sensitivity, anomalies, relate, methods ...
    gallery: null,
  };
}
export function freshLive() {
  return { kind: '', jobId: null, rows: [], next: 0, phase: '', restart: 0, nRestarts: 0, fold: null, nFolds: null, message: '', best: null, status: '', seq: 0, error: '' };
}

// ---------------------------------------------------------------- the problems of a step (plain English; they block the run)
const inRange = (x, lo, hi) => Number.isFinite(Number(x)) && Number(x) >= lo && Number(x) <= hi;
export function netIssues(spec) {
  const n = spec.net, out = [];
  const d = n.latent_dim;
  if (d !== null && !(Number.isInteger(Number(d)) && inRange(d, 1, 32))) out.push('The number of codes must be a whole number between 1 and 32.');
  if (n.size === 'custom' && !parseHidden(n.hidden)) out.push('The layer widths must be one to four whole numbers between 2 and 1024, separated by commas (for example 64, 32).');
  if (!inRange(n.dropout, 0, 0.5)) out.push('Dropout must be between 0 and 0.5.');
  if (!inRange(n.weight_decay, 0, 1)) out.push('Weight decay must be between 0 and 1.');
  if (!inRange(n.noise, 0, 2)) out.push('The noise added to the input must be between 0 and 2 standard deviations.');
  if (!inRange(n.mask, 0, 0.9)) out.push('The share of inputs hidden must be between 0 and 0.9.');
  if (!inRange(n.smoothify, 0, 100)) out.push('Smoothify must be between 0 and 100.');
  if (n.family === 'vae') {
    const v = n.vae;
    if (!inRange(v.beta, 0, 100)) out.push('The weight of the typical-code term (β) must be between 0 and 100.');
    if (v.warmup !== null && v.warmup !== '' && !inRange(v.warmup, 0, 5000)) out.push('The warm-up must be between 0 and 5000 epochs.');
    if (!inRange(v.free_bits, 0, 5)) out.push('Free bits must be between 0 and 5.');
    if (v.sigma_mode === 'fixed' && !(Number(v.sigma) > 0)) out.push('The fixed noise level of the data must be larger than zero.');
    if (v.prior === 'mixture' && !(Number.isInteger(Number(v.n_components)) && inRange(v.n_components, 2, 20))) out.push('The number of clouds must be a whole number between 2 and 20.');
  }
  return out;
}
/** How many samples take part with the variables chosen (null while the data are not read). */
export function sampleCount(spec = fp.spec, data = fp.data) {
  if (!data) return null;
  const ab = data.absorbance;
  if (spec.what === 'both' && ab && ab.available) return ab.n_common;
  if (spec.what === 'absorbance' && ab && ab.available) return ab.n;
  return data.n_samples;
}
export const FEW = 12;                                         // the fewest samples that an autoencoder is trained on without asking the person
/** Fewer than 12 samples: refused unless the person agreed (the switch of step 6). */
export function fewIssue(spec, data) {
  const count = sampleCount(spec, data);
  return count !== null && count < FEW && !spec.net.allow_few
    ? [`Only ${count} samples are included: an autoencoder needs at least ${FEW} samples to be trained honestly. Add more samples, use the PCA or functional PCA pages for very small data sets, or switch on 'Train on fewer than ${FEW} samples anyway' in step 6.`] : [];
}
export function trainIssues(spec) {
  const n = spec.net, out = [];
  if (!inRange(n.epochs, 1, 5000)) out.push('The maximum number of epochs must be between 1 and 5000.');
  if (!inRange(n.patience, 1, 1000)) out.push('The patience must be between 1 and 1000 epochs.');
  if (!inRange(n.val_fraction, 0.1, 0.5)) out.push('The share of samples held out must be between 0.1 and 0.5.');
  if (!inRange(n.folds, 2, 10)) out.push('The number of folds must be between 2 and 10.');
  if (!inRange(n.lr, 0.00001, 0.1)) out.push('The learning rate must be between 0.00001 and 0.1.');
  if (n.batch_size !== null && n.batch_size !== '' && !inRange(n.batch_size, 2, 1024)) out.push('The batch size must be between 2 and 1024 (or empty for automatic).');
  if (!inRange(n.n_restarts, 1, 10)) out.push('The number of random starts must be between 1 and 10.');
  if (!inRange(n.seed, 0, 2147483647)) out.push('The random seed must be a whole number between 0 and 2 147 483 647.');
  return out;
}

// ---------------------------------------------------------------- the stepper: which steps are done / need attention
function aeStepStatus(n, ctx) {
  const { fp, stepIssues } = ctx;
  if (n <= 5) {
    const issues = stepIssues(n);
    if (issues.length && (fp.visited[n] || n <= fp.step)) return 'attention';
    return fp.visited[n] || n < fp.step ? 'done' : 'todo';
  }
  if (n === 6) {
    if (fp.ae.live.status === 'running') return 'todo';
    const issues = stepIssues(6);
    if (issues.length && (fp.visited[6] || fp.step >= 6)) return 'attention';
    return fp.rid && fp.ae.res ? 'done' : (fp.visited[6] ? 'done' : 'todo');
  }
  if (!fp.rid || !fp.ae.res) return 'todo';
  const stale = fp.ae.res.result.stale || [];
  if (stale.length) return 'attention';
  return n === 7 || fp.visited[n] || fp.step === n ? 'done' : 'todo';
}

// ---------------------------------------------------------------- the context
export const ctx = createFpca({
  steps: AE_STEPS, stateKey: 'autoencoders', defaultSpec: defaultAeSpec, minSamples: 4,
  extraState: () => ({ ae: freshAe() }),
  issueSteps: [1, 2, 3, 4, 5, 6],
  issues: (n, spec, data, base) => (n === 1 ? [...base(1), ...fewIssue(spec, data)] : n <= 4 ? base(n) : n === 5 ? netIssues(spec) : n === 6 ? trainIssues(spec) : []),
  stepStatus: aeStepStatus,
});
export const { fp } = ctx;
export const aeJobs = { fit: useJob(), scan: useJob(), check: useJob() };

/** The number of codes in force: the one the person typed, else the advice, else 3. */
export function latentOf(net = fp.spec.net, defaults = fp.ae.defaults) {
  if (net.latent_dim !== null && net.latent_dim !== undefined && net.latent_dim !== '') return Math.round(Number(net.latent_dim));
  return defaults && defaults.advice && defaults.advice.latent_dim ? defaults.advice.latent_dim : 3;
}

/** The request of a fit / check / scan, as docs/api.md section 1 words it. */
export function requestBody(spec = fp.spec, defaults = fp.ae.defaults, extra = {}) {
  const n = spec.net, names = variablesOf(spec);
  const intOrNull = (x) => (x === null || x === undefined || x === '' ? null : Math.round(Number(x)));
  const body = {
    family: n.family, latent_dim: latentOf(n, defaults), size: n.size, hidden: hiddenOf(n), activation: n.activation,
    dropout: Number(n.dropout), weight_decay: Number(n.weight_decay), corruption: { noise: Number(n.noise), mask: Number(n.mask) }, skip: !!n.skip, allow_few_samples: !!n.allow_few,
    vae: { beta: Number(n.vae.beta), warmup: intOrNull(n.vae.warmup), free_bits: Number(n.vae.free_bits), sigma: n.vae.sigma_mode === 'learned' ? 'learned' : Number(n.vae.sigma), prior: n.vae.prior, n_components: Math.round(Number(n.vae.n_components)) },
    smoothify: Number(n.smoothify), smoothify_metric: n.smoothify_metric,
    epochs: Math.round(Number(n.epochs)), patience: Math.round(Number(n.patience)), val_fraction: Number(n.val_fraction), folds: Math.round(Number(n.folds)),
    lr: Number(n.lr), batch_size: intOrNull(n.batch_size), n_restarts: Math.round(Number(n.n_restarts)), seed: Math.round(Number(n.seed)),
    input: { variables: names }, title: String(n.title || '').trim(),
  };
  for (const name of names) body.input[name] = ctx.varBody(name, spec);
  return { ...body, ...extra };
}
/** The settings of a result (its `params`) back to the choices of the page (to look at them again, or to change one). */
export function specFromRequest(p) {
  const s = defaultAeSpec(), n = s.net;
  const names = (p.input && p.input.variables) || ['fluorescence'];
  s.what = names.length > 1 ? 'both' : names[0];
  for (const name of names) {
    const v = p.input && p.input[name]; if (!v) continue;
    s[name].transform = v.transform || 'none';
    if (name === 'fluorescence') s[name].min_coverage = v.min_coverage ?? 0.9; else s[name].processed = v.processed !== false;
    const b = v.basis || {}, t = s[name].basis;
    t.kind = b.kind || 'gp';
    if (b.kind === 'gp') {
      t.kernel = b.kernel || t.kernel; t.structure = b.structure || t.structure;
      if (Array.isArray(b.lengths)) { t.lmode = 'typed'; t.typed = b.lengths.map(Number); } else t.lmode = b.lengths === 'ml' ? 'ml' : 'auto';
      t.share = Math.round((b.var_fraction ?? 0.999) * 10000) / 100; t.n_terms = b.n_basis || null; t.k_max = b.k_max || t.k_max;
    } else if (b.kind === 'bspline') t.bs = [].concat(b.n_basis ?? t.bs).map(Number);
    else if (b.kind === 'cosine') t.cos = [].concat(b.orders ?? t.cos).map(Number);
    const sm = v.smoothing || {};
    s[name].smooth.method = sm.method || (t.kind === 'gp' ? 'ml' : 'gcv');
    if (sm.tau !== undefined) s[name].smooth.tau = sm.tau;
    if (sm.lambda) s[name].smooth.lam = [].concat(sm.lambda).map(Number);
  }
  Object.assign(n, { family: p.family || 'ae', latent_dim: p.latent_dim ?? null, size: p.size || 'small', activation: p.activation || 'silu', dropout: p.dropout ?? 0, weight_decay: p.weight_decay ?? 0.001,
    smoothify: p.smoothify ?? 0, smoothify_metric: p.smoothify_metric || 'smooth', epochs: p.epochs ?? 600, patience: p.patience ?? 60, val_fraction: p.val_fraction ?? 0.2, folds: p.folds ?? 5,
    lr: p.lr ?? 0.001, batch_size: p.batch_size ?? null, n_restarts: p.n_restarts ?? 3, seed: p.seed ?? 0, title: p.title || '', skip: p.skip !== false, allow_few: !!p.allow_few_samples });
  if (Array.isArray(p.hidden)) n.hidden = p.hidden.join(', ');
  if (p.corruption) { n.noise = p.corruption.noise ?? 0; n.mask = p.corruption.mask ?? 0; }
  if (p.vae) {
    const v = p.vae;
    Object.assign(n.vae, { beta: v.beta ?? 1, warmup: v.warmup ?? null, free_bits: v.free_bits ?? 0, prior: v.prior || 'normal', n_components: v.n_components ?? 3 });
    if (typeof v.sigma === 'number') { n.vae.sigma_mode = 'fixed'; n.vae.sigma = v.sigma; }
  }
  return s;
}

export { markRaw, NAMES, variablesOf };
