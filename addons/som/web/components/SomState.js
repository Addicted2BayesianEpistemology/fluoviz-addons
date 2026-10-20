// The state of the Self-organising map page: ONE context made with the Functional PCA's machinery (steps 1 to 4 are its components), plus the map's own choices (`spec.map`) and
// answers (`fp.som`).  Module level on purpose: the choices survive leaving the page (and are remembered per project through PUT /api/state/ui, key `som`), and a computation that was
// started keeps running while the person looks at another page.  Modelled on AeState.js of the autoencoders add-on.
import { markRaw } from 'vue';
import { createFpca } from '/static/js/components/FpcaCtx.js';
import { defaultSpec, variablesOf, NAMES } from '/static/js/components/FpcaState.js';
import { useJob } from '/static/js/jobs.js';

export const BASE = '/api/addon/som';

// ---------------------------------------------------------------- the steps (the Help guide quotes these names: do not rename)
export const SOM_STEPS = [
  { n: 1, id: 'what', label: '1 · What to map' },
  { n: 2, id: 'domain', label: '2 · Domain' },
  { n: 3, id: 'repr', label: '3 · Representation' },
  { n: 4, id: 'smooth', label: '4 · Smoothing' },
  { n: 5, id: 'map', label: '5 · Map and distance' },
  { n: 6, id: 'compute', label: '6 · Compute' },
  { n: 7, id: 'results', label: '7 · Results' },
  { n: 8, id: 'checks', label: '8 · Checks' },
  { n: 9, id: 'use', label: '9 · Use' },
];
export const METRICS = [
  { value: 'l2', label: 'Plain distance (L2)', text: 'The usual distance between EEMs or spectra: big differences count big. The map is organised by the directions in which the samples differ most.' },
  { value: 'mahalanobis', label: 'Standardised (Mahalanobis)', text: 'Every direction of variation counts the same, so subtle differences in shape matter as much as the overall intensity. The leading directions are standardised and everything beyond them counts as one unit in total, so noise cannot dominate.' },
  { value: 'prior', label: 'Gaussian-process prior', text: 'The distance in the standardised coefficients of the Gaussian-process basis: rough components, which the prior expects to be small, count most. Needs the Gaussian-process basis.' },
  { value: 'standardize', label: 'Every pixel or variable standardised', text: 'Every pixel (or variable) is divided by its standard deviation across the samples. A weak pixel then counts as much as a strong one, which also amplifies its noise.' },
];
export const SIZE_PRESETS = { compact: 'Compact', standard: 'Standard', fine: 'Fine' };
export const STALE_WORDS = { raw: 'the raw data', corrections: 'the corrections', masks: 'the scatter removal or excised regions', window: 'the focus window', samples: 'the sample selection' };
export const FEW = 8;

// ---------------------------------------------------------------- the choices
export function mapDefault() {
  return {
    metric: 'l2', alpha: 1, components: 'auto', components_n: 5,
    size: 'standard', rows: 6, cols: 8, lattice: 'hexagonal', shape: 'sheet',
    mode: 'batch', init: 'linear', neighbourhood: 'gaussian', epochs_rough: 50, epochs_fine: 150, radius_initial: 'auto', radius_initial_n: 3, radius_final: 1, alpha_initial: 0.05, alpha_final: 0.01,
    gap_weighting: 'observed', folds: 5, limit: 99, seed: 0, title: '', allow_few: false,
    cluster: { method: 'hierarchical', k: 'auto', k_n: 4, k_min: 2, k_max: 10, skip_empty: 'auto' },
    other: { on: false, keys: [], standardize: true },
    memory: { fluorescence: 'gp', absorbance: 'gp' },
  };
}
export function defaultSomSpec() {
  const s = defaultSpec();
  s.map = mapDefault();
  return s;
}
export const isFunctional = (spec) => variablesOf(spec).every((n) => spec[n].basis.kind !== 'pixels');
/** The functional switch of step 3: on = the map works on the coefficients of a basis (the one chosen before, the Gaussian-process basis at first), off = on the pixels. */
export function setFunctional(spec, on) {
  for (const name of NAMES) {
    const b = spec[name].basis, s = spec[name].smooth;
    if (on) { if (b.kind === 'pixels') { b.kind = spec.map.memory[name] || 'gp'; s.method = b.kind === 'gp' ? 'ml' : 'gcv'; } }
    else if (b.kind !== 'pixels') { spec.map.memory[name] = b.kind; b.kind = 'pixels'; s.method = 'none'; }
  }
}

// ---------------------------------------------------------------- the answers (made again with every project)
export function freshSom() {
  return {
    defaults: null, defaultsKey: '', defaultsError: '', defaultsBusy: false,
    check: { key: '', ans: null, error: '', busy: false },
    res: null, resRid: null, resError: '', resBusy: false,
    fitError: '', queued: '', preparing: false,
    tab7: 'map', tab8: 'quality',
    view: { plane: 'umatrix', place: 'position', color: null, symbol: null, borders: true, proto: 'prototype', protoScale: 'shared', unit: null, sample: null, find: '', filter: 'all' },
    planesAdded: [],
    planeCache: {},
    groupsView: { method: 'hierarchical', k: 'auto', k_n: 4, skip_empty: 'auto' },
    anomView: { method: 'mad', c: 3, q: 0.99 },
    trajView: { order: null, group: null },
    scanView: { sizeUnits: '', n_runs: 8, bootstrap: false, k: 4 },
    scans: {},
    extra: {},
    gallery: null,
  };
}

// ---------------------------------------------------------------- the problems of a step (plain English; they block the run)
const inRange = (x, lo, hi) => Number.isFinite(Number(x)) && Number(x) >= lo && Number(x) <= hi;
const isInt = (x) => Number.isInteger(Number(x));
export function mapIssues(spec) {
  const m = spec.map, out = [];
  if (m.metric === 'mahalanobis') {
    if (!inRange(m.alpha, 0, 1)) out.push('How far to standardise must be between 0 and 1.');
    if (m.components !== 'auto' && !(isInt(m.components_n) && inRange(m.components_n, 1, 200))) out.push('The number of directions to standardise must be a whole number from 1 to 200.');
  }
  if (m.metric === 'prior' && !m.other.on && (!isFunctional(spec) || variablesOf(spec).some((n) => spec[n].basis.kind !== 'gp'))) out.push('The Gaussian-process prior distance needs the Gaussian-process basis for every variable (step 3).');
  if (m.metric === 'prior' && m.other.on) out.push('The Gaussian-process prior distance needs functions; choose another distance for a map of other variables.');
  if (m.size === 'custom') {
    if (!(isInt(m.rows) && isInt(m.cols) && inRange(m.rows, 2, 200) && inRange(m.cols, 2, 200))) out.push('The rows and columns of the map must be whole numbers of at least 2.');
    else if (Number(m.rows) * Number(m.cols) > 400) out.push('A map of more than 400 units is too many.');
  }
  return out;
}
export function computeIssues(spec) {
  const m = spec.map, out = [];
  if (!(isInt(m.epochs_rough) && inRange(m.epochs_rough, 1, 2000))) out.push('The epochs of the rough phase must be a whole number from 1 to 2000.');
  if (!(isInt(m.epochs_fine) && inRange(m.epochs_fine, 1, 5000))) out.push('The epochs of the fine phase must be a whole number from 1 to 5000.');
  if (m.radius_initial !== 'auto' && !inRange(m.radius_initial_n, 0.5, 100)) out.push('The initial radius must be between 0.5 and 100.');
  if (!inRange(m.radius_final, 0.3, 5)) out.push('The final radius must be between 0.3 and 5.');
  if (!inRange(m.alpha_initial, 0.0001, 1)) out.push('The initial learning rate must be between 0.0001 and 1.');
  if (!inRange(m.alpha_final, 0.00001, 1)) out.push('The final learning rate must be between 0.00001 and 1.');
  else if (Number(m.alpha_final) > Number(m.alpha_initial)) out.push('The final learning rate cannot be larger than the initial one.');
  if (!(isInt(m.folds) && inRange(m.folds, 2, 20))) out.push('The number of folds must be a whole number from 2 to 20.');
  if (!(isInt(m.seed) && inRange(m.seed, 0, 2147483647))) out.push('The random seed must be a whole number from 0 to 2 147 483 647.');
  const c = m.cluster;
  if (c.k !== 'auto' && !(isInt(c.k_n) && inRange(c.k_n, 2, 50))) out.push('The number of groups must be a whole number from 2 to 50.');
  if (!(isInt(c.k_min) && isInt(c.k_max) && inRange(c.k_min, 2, 50) && inRange(c.k_max, 2, 50) && Number(c.k_min) <= Number(c.k_max))) out.push('The range of group numbers must be two whole numbers from 2 to 50, the smaller first.');
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
export function fewIssue(spec, data) {
  if (spec.map.other.on) return [];
  const count = sampleCount(spec, data);
  return count !== null && count < FEW && !spec.map.allow_few
    ? [`Only ${count} samples are included: a map of fewer than ${FEW} samples shows very little. Add more samples, or switch on 'Compute on fewer than ${FEW} samples anyway' in step 6 (never fewer than 4).`] : [];
}
export function otherIssue(spec) {
  const o = spec.map.other;
  return o.on && !o.keys.length ? ['Choose at least one variable to map, or switch "Other variables instead" off.'] : [];
}

// ---------------------------------------------------------------- the stepper
function somStepStatus(n, ctx) {
  const { fp, stepIssues } = ctx;
  if (n <= 5) {
    const issues = stepIssues(n);
    if (issues.length && (fp.visited[n] || n <= fp.step)) return 'attention';
    return fp.visited[n] || n < fp.step ? 'done' : 'todo';
  }
  if (n === 6) {
    if (somJobs.fit.state.running) return 'todo';
    const issues = stepIssues(6);
    if (issues.length && (fp.visited[6] || fp.step >= 6)) return 'attention';
    return fp.rid && fp.som.res ? 'done' : (fp.visited[6] ? 'done' : 'todo');
  }
  if (!fp.rid || !fp.som.res) return 'todo';
  if ((fp.som.res.result.stale || []).length) return 'attention';
  return n === 7 || fp.visited[n] || fp.step === n ? 'done' : 'todo';
}

// ---------------------------------------------------------------- the context
export const ctx = createFpca({
  steps: SOM_STEPS, stateKey: 'som', defaultSpec: defaultSomSpec, minSamples: 4,
  extraState: () => ({ som: freshSom() }),
  issueSteps: [1, 2, 3, 4, 5, 6],
  issues: (n, spec, data, base) => (n === 1 ? [...base(1), ...fewIssue(spec, data), ...otherIssue(spec)] : n <= 4 ? (spec.map.other.on ? [] : base(n)) : n === 5 ? mapIssues(spec) : n === 6 ? computeIssues(spec) : []),
  stepStatus: somStepStatus,
});
export const { fp } = ctx;
export const somJobs = { fit: useJob(), scan: useJob(), check: useJob() };

/** The request of a fit / check / scan, as docs/api.md section 1 words it. */
export function requestBody(spec = fp.spec, extra = {}) {
  const m = spec.map, names = variablesOf(spec);
  const num = (x) => Number(x);
  const input = {};
  if (m.other.on) { input.kind = 'variables'; input.keys = m.other.keys.slice(); input.standardize = !!m.other.standardize; }
  else {
    input.kind = isFunctional(spec) ? 'functional' : 'pixels';
    input.variables = names;
    for (const name of names) input[name] = ctx.varBody(name, spec);
    input.weights = spec.weights === 'custom' ? Object.fromEntries(names.map((n) => [n, num(spec.custom[n])])) : spec.weights;
  }
  const body = {
    input,
    metric: { kind: m.metric, alpha: num(m.alpha), components: m.components === 'auto' ? 'auto' : Math.round(num(m.components_n)) },
    map: { size: m.size === 'custom' ? [Math.round(num(m.rows)), Math.round(num(m.cols))] : m.size, lattice: m.lattice, shape: m.shape },
    train: { mode: m.mode, init: m.init, neighbourhood: m.neighbourhood, epochs_rough: Math.round(num(m.epochs_rough)), epochs_fine: Math.round(num(m.epochs_fine)),
      radius_initial: m.radius_initial === 'auto' ? 'auto' : num(m.radius_initial_n), radius_final: num(m.radius_final), alpha_initial: num(m.alpha_initial), alpha_final: num(m.alpha_final) },
    gap_weighting: m.gap_weighting === 'equal' ? 'equal' : 'observed',
    validation: { folds: Math.round(num(m.folds)), limit: Math.round(num(m.limit)) },
    cluster: { method: m.cluster.method, k: m.cluster.k === 'auto' ? 'auto' : Math.round(num(m.cluster.k_n)), k_range: [Math.round(num(m.cluster.k_min)), Math.round(num(m.cluster.k_max))], skip_empty: m.cluster.skip_empty },
    seed: Math.round(num(m.seed)), allow_few_samples: !!m.allow_few, title: String(m.title || '').trim(),
  };
  return { ...body, ...extra };
}

/** The settings of a result (its `params`) back to the choices of the page (to look at them again, or to change one). */
export function specFromRequest(p) {
  const s = defaultSomSpec(), m = s.map, inp = p.input || {};
  if (inp.kind === 'variables') { m.other.on = true; m.other.keys = (inp.keys || []).slice(); m.other.standardize = inp.standardize !== false; }
  else {
    const names = inp.variables || ['fluorescence'];
    s.what = names.length > 1 ? 'both' : names[0];
    for (const name of names) {
      const v = inp[name]; if (!v) continue;
      s[name].transform = v.transform || 'none';
      if (name === 'fluorescence') s[name].min_coverage = v.min_coverage ?? 0.5; else s[name].processed = v.processed !== false;
      s[name].gaps = v.gaps === 'plain' ? 'plain' : 'auto';
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
    if (inp.weights && typeof inp.weights === 'object') { s.weights = 'custom'; Object.assign(s.custom, inp.weights); } else s.weights = inp.weights || 'variance';
  }
  const me = p.metric || {}, mp = p.map || {}, tr = p.train || {}, va = p.validation || {}, cl = p.cluster || {};
  m.metric = me.kind || 'l2'; m.alpha = me.alpha ?? 1; m.components = me.components === 'auto' || me.components == null ? 'auto' : 'n'; if (typeof me.components === 'number') m.components_n = me.components;
  if (Array.isArray(mp.size)) { m.size = 'custom'; m.rows = mp.size[0]; m.cols = mp.size[1]; } else m.size = mp.size || 'standard';
  m.lattice = mp.lattice || 'hexagonal'; m.shape = mp.shape || 'sheet';
  Object.assign(m, { mode: tr.mode || 'batch', init: tr.init || 'linear', neighbourhood: tr.neighbourhood || 'gaussian', epochs_rough: tr.epochs_rough ?? 50, epochs_fine: tr.epochs_fine ?? 150,
    radius_final: tr.radius_final ?? 1, alpha_initial: tr.alpha_initial ?? 0.05, alpha_final: tr.alpha_final ?? 0.01 });
  if (tr.radius_initial && tr.radius_initial !== 'auto') { m.radius_initial = 'n'; m.radius_initial_n = tr.radius_initial; }
  m.gap_weighting = p.gap_weighting === 'equal' ? 'equal' : 'observed'; m.folds = va.folds ?? 5; m.limit = va.limit ?? 99; m.seed = p.seed ?? 0; m.title = p.title || ''; m.allow_few = !!p.allow_few_samples;
  Object.assign(m.cluster, { method: cl.method || 'hierarchical', k: cl.k === 'auto' || cl.k == null ? 'auto' : 'n', k_n: typeof cl.k === 'number' ? cl.k : 4, k_min: (cl.k_range || [2, 10])[0], k_max: (cl.k_range || [2, 10])[1], skip_empty: cl.skip_empty ?? 'auto' });
  return s;
}

export { markRaw, NAMES, variablesOf };
