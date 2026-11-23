// The state of the Multimodal latent factors page: ONE context made with the Functional PCA's machinery (steps 1 to 3 are its components), plus the model's own choices
// (`spec.mblf`) and answers (`fp.mb`).  Module level on purpose: the choices survive leaving the page (and are remembered per project through PUT /api/state/ui, key `mblf`),
// and a fit that was started keeps running while the person looks at another page.  Modelled on SomState.js of the self-organising map add-on.
import { createFpca } from '/static/js/components/FpcaCtx.js';
import { defaultSpec, variablesOf, NAMES } from '/static/js/components/FpcaState.js';
import { useJob } from '/static/js/jobs.js';

export const BASE = '/api/addon/mblf';

// ---------------------------------------------------------------- the steps (the Help guide quotes these names: do not rename)
export const MBLF_STEPS = [
  { n: 1, id: 'what', label: '1 · What to model' },
  { n: 2, id: 'domain', label: '2 · Domain' },
  { n: 3, id: 'repr', label: '3 · Basis' },
  { n: 4, id: 'model', label: '4 · Model and sampler' },
  { n: 5, id: 'fit', label: '5 · Fit' },
  { n: 6, id: 'results', label: '6 · Results' },
  { n: 7, id: 'predict', label: '7 · Prediction and uncertainty' },
  { n: 8, id: 'checks', label: '8 · Checks and use' },
];
export const MODE_WORDS = { fluorescence: 'the fluorescence (EEMs)', absorbance: 'the absorbance spectra', values: 'the values', covariates: 'the covariates' };
export const MODE_NICE = { fluorescence: 'Fluorescence', absorbance: 'Absorbance', values: 'Values', covariates: 'Covariates', nothing: 'Nothing' };
export const PARTS = ['parameters', 'scores', 'residual', 'noise'];
export const PART_LABELS = {
  parameters: 'Population parameters',
  scores: 'Factor scores of the sample',
  residual: 'The sample’s own residual',
  noise: 'Measurement noise',
};
export const PART_WORDS = {
  parameters: 'what the training samples could not pin down: more training samples shrink it',
  scores: 'how well the sources fix the factors of THIS sample: more, or more precise, measurements of the sample shrink it',
  residual: 'what is specific to this sample and its mode and no other measurement can tell',
  noise: 'the noise of a new measurement (only when a measurement, not the latent function, is predicted)',
};
export const PRESET_WORDS = {
  quick: 'Quick look: 2 chains of 400 + 400 iterations (to try settings; the checks will often ask for more)',
  standard: 'Standard: 4 chains of 1000 + 1000 iterations (what the checks of the method expect)',
  thorough: 'Thorough: 4 chains of 2000 + 2000 iterations (when the standard run does not converge)',
};

// ---------------------------------------------------------------- the choices
export function modelDefault() {
  return {
    values: [], covariates: [], partial: true,
    K: 5, estimation: 'bayes', preset: 'standard', custom: false, chains: 4, warmup: 1000, draws: 1000, adapt_delta: 0.8, max_treedepth: 12, keep: 1000, ml_starts: 8,
    priors: { a_psi: 2, b_psi: 2, a_tau: 2, b_tau: 2, a1: 2, a2: 2, b_lambdavar: 1, nu: 3, sigma_regr_coeffs: 10 },
    seed: 1, title: '', allow_few: false,
  };
}
export function defaultMblfSpec() {
  const s = defaultSpec();
  s.what = 'both';
  for (const n of NAMES) { s[n].basis.share = 99; s[n].basis.k_max = n === 'fluorescence' ? 40 : 25; s[n].smooth.method = 'none'; }
  s.mblf = modelDefault();
  return s;
}

// ---------------------------------------------------------------- the answers (made again with every project)
export function freshMb() {
  return {
    engine: null, engineBusy: false, engineError: '',
    defaults: null, defaultsError: '',
    check: { key: '', ans: null, error: '', busy: false },
    res: null, resRid: null, resError: '', resBusy: false, fitError: '',
    tab6: 'overview', tab8: 'convergence',
    view: { factor: 0, stat: 'mean', simultaneous: true, target: null, combo: null, corrA: null, corrB: null, corrX: null, corrY: null, corrJ: 0 },
    pred: { sample: null, target: null, sources: null, covariates: true, heldOut: true, noisy: false, stat: 'mean' },
    cmp: { target: null, row: null },
    extra: {},
  };
}

// ---------------------------------------------------------------- the problems of a step (plain English; they block the run)
const inRange = (x, lo, hi) => Number.isFinite(Number(x)) && Number(x) >= lo && Number(x) <= hi;
const isInt = (x) => Number.isInteger(Number(x));
export function whatIssues(spec) {
  const m = spec.mblf, out = [];
  const both = m.values.filter((k) => m.covariates.includes(k));
  if (both.length) out.push('A variable cannot be both a value to model and a covariate: remove it from one of the two lists.');
  return out;
}
export function modelIssues(spec) {
  const m = spec.mblf, out = [];
  if (!(isInt(m.K) && inRange(m.K, 1, 12))) out.push('The largest number of factors must be a whole number from 1 to 12.');
  if (m.custom) {
    if (!(isInt(m.chains) && inRange(m.chains, 1, 16))) out.push('The number of chains must be a whole number from 1 to 16.');
    if (!(isInt(m.warmup) && inRange(m.warmup, 50, 20000))) out.push('The warm-up iterations must be a whole number from 50 to 20 000.');
    if (!(isInt(m.draws) && inRange(m.draws, 50, 20000))) out.push('The kept iterations must be a whole number from 50 to 20 000.');
  }
  if (!inRange(m.adapt_delta, 0.5, 0.999)) out.push('The target acceptance rate must be between 0.5 and 0.999.');
  if (!(isInt(m.max_treedepth) && inRange(m.max_treedepth, 5, 15))) out.push('The largest tree depth must be a whole number from 5 to 15.');
  if (!(isInt(m.ml_starts) && inRange(m.ml_starts, 1, 64))) out.push('The number of starting points must be a whole number from 1 to 64.');
  if (!(isInt(m.seed) && inRange(m.seed, 0, 2147483647))) out.push('The random seed must be a whole number from 0 to 2 147 483 647.');
  for (const [k, v] of Object.entries(m.priors)) if (!(Number(v) > 0)) out.push(`The prior setting ${k} must be a positive number.`);
  return out;
}

// ---------------------------------------------------------------- the stepper
function stepStatus(n, c) {
  const { fp, stepIssues } = c;
  if (n <= 4) {
    const issues = stepIssues(n);
    if (issues.length && (fp.visited[n] || n <= fp.step)) return 'attention';
    return fp.visited[n] || n < fp.step ? 'done' : 'todo';
  }
  if (n === 5) {
    if (mbJobs.fit.state.running) return 'todo';
    return fp.rid && fp.mb.res ? 'done' : (fp.visited[5] ? 'done' : 'todo');
  }
  if (!fp.rid || !fp.mb.res) return 'todo';
  if ((fp.mb.res.result.stale || []).length) return 'attention';
  return n === 6 || fp.visited[n] || fp.step === n ? 'done' : 'todo';
}

export const ctx = createFpca({
  steps: MBLF_STEPS, stateKey: 'mblf', defaultSpec: defaultMblfSpec, minSamples: 5,
  extraState: () => ({ mb: freshMb() }),
  issueSteps: [1, 2, 3, 4],
  issues: (n, spec, data, base) => (n === 1 ? [...base(1), ...whatIssues(spec)] : n <= 3 ? base(n).filter((t) => !/smooth/i.test(t)) : n === 4 ? modelIssues(spec) : []),
  stepStatus,
});
export const { fp } = ctx;
export const mbJobs = { fit: useJob(), check: useJob(), engine: useJob(), compare: useJob() };

/** The request of a fit / check, as docs/api.md section 1 words it. */
export function requestBody(spec = fp.spec, extra = {}) {
  const m = spec.mblf, names = variablesOf(spec);
  const input = { variables: names, values: m.values.slice(), covariates: m.covariates.slice(), partial: !!m.partial };
  for (const name of names) {
    const v = ctx.varBody(name, spec);
    input[name] = { transform: v.transform, basis: v.basis, ...(name === 'fluorescence' ? { min_coverage: v.min_coverage } : { processed: v.processed }) };
  }
  const num = Number;
  const sampler = m.custom ? { chains: Math.round(num(m.chains)), warmup: Math.round(num(m.warmup)), draws: Math.round(num(m.draws)) } : { preset: m.preset };
  Object.assign(sampler, { adapt_delta: num(m.adapt_delta), max_treedepth: Math.round(num(m.max_treedepth)), keep: Math.round(num(m.keep)) });
  const body = {
    input, model: { K: Math.round(num(m.K)), estimation: m.estimation, priors: Object.fromEntries(Object.entries(m.priors).map(([k, v]) => [k, num(v)])) },
    sampler, ml: { starts: Math.round(num(m.ml_starts)) }, seed: Math.round(num(m.seed)), allow_few_samples: !!m.allow_few, title: String(m.title || '').trim(),
  };
  return { ...body, ...extra };
}

/** The settings of a result (its `params`) back to the choices of the page. */
export function specFromRequest(p) {
  const s = defaultMblfSpec(), m = s.mblf, inp = p.input || {};
  const names = inp.variables || ['fluorescence'];
  s.what = names.length > 1 ? 'both' : (names[0] || 'fluorescence');
  for (const name of names) {
    const v = inp[name]; if (!v) continue;
    s[name].transform = v.transform || 'none';
    if (name === 'fluorescence') s[name].min_coverage = v.min_coverage ?? 0.5; else s[name].processed = v.processed !== false;
    const b = v.basis || {}, t = s[name].basis;
    t.kernel = b.kernel || t.kernel; t.structure = b.structure || t.structure;
    if (Array.isArray(b.lengths)) { t.lmode = 'typed'; t.typed = b.lengths.map(Number); } else t.lmode = b.lengths === 'ml' ? 'ml' : 'auto';
    t.share = Math.round((b.var_fraction ?? 0.99) * 10000) / 100; t.n_terms = b.n_basis || null; t.k_max = b.k_max || t.k_max;
  }
  m.values = (inp.values || []).slice(); m.covariates = (inp.covariates || []).slice(); m.partial = inp.partial !== false;
  const mo = p.model || {}, sa = p.sampler || {};
  m.K = mo.K ?? 5; m.estimation = mo.estimation || 'bayes'; Object.assign(m.priors, mo.priors || {});
  const preset = Object.entries({ quick: [2, 400, 400], standard: [4, 1000, 1000], thorough: [4, 2000, 2000] }).find(([, v]) => v[0] === sa.chains && v[1] === sa.warmup && v[2] === sa.draws);
  if (preset) { m.preset = preset[0]; m.custom = false; } else { m.custom = true; }
  Object.assign(m, { chains: sa.chains ?? 4, warmup: sa.warmup ?? 1000, draws: sa.draws ?? 1000, adapt_delta: sa.adapt_delta ?? 0.8, max_treedepth: sa.max_treedepth ?? 12, keep: sa.keep ?? 1000,
    ml_starts: (p.ml || {}).starts ?? 8, seed: p.seed ?? 1, title: p.title || '', allow_few: !!p.allow_few_samples });
  return s;
}

export { NAMES, variablesOf };
