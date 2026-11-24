// Everything the Multimodal latent factors page asks the server for (docs/api.md).  Module level, like the state: a fit that was started keeps running while the person looks at
// another page.  Every answer is checked against a sequence number (a quick second click is never overwritten by an older answer); nothing is ever computed silently.
import { markRaw } from 'vue';
import { get, post, del, download } from '/static/js/api.js';
import { store, toast } from '/static/js/store.js';
import { loadCatalogue } from '/static/js/variables.js';
import { ctx, fp, mbJobs, BASE, requestBody } from './MblfState.js';

const enc = encodeURIComponent;

// ---------------------------------------------------------------- the Stan engine
export async function loadEngine(size = false) {
  const A = fp.mb;
  A.engineBusy = true;
  try { A.engine = markRaw(await get(`${BASE}/engine`, size ? { size: 1 } : null, { quiet: true })); A.engineError = ''; }
  catch (e) { A.engineError = e.message || 'The state of the Stan engine could not be read.'; }
  finally { A.engineBusy = false; }
  return A.engine;
}
export async function prepareEngine() {
  const res = await mbJobs.engine.run(`${BASE}/engine/prepare`, {}, { quiet: true });
  if (!res) {
    if (mbJobs.engine.state.cancelled) toast('Cancelled: the Stan engine was not prepared, nothing was changed.', 'info', '', 4000);
    else toast(mbJobs.engine.state.error || 'The Stan engine could not be prepared.', 'warn', '', 9000);
  } else toast('The Stan engine is ready.', 'ok', '', 4000);
  return loadEngine();
}
export async function removeEngine() {
  try { await del(`${BASE}/engine`); toast('The Stan engine was removed. It is prepared again the next time you fit a model.', 'ok', '', 5000); }
  catch (e) { /* the API client already said why */ }
  return loadEngine();
}

// ---------------------------------------------------------------- the advice and the check
export async function loadDefaults() {
  if (!store.project || !store.project.has_data) return null;
  try { fp.mb.defaults = markRaw(await get(`${BASE}/defaults`, null, { quiet: true })); fp.mb.defaultsError = ''; if (fp.mb.defaults.engine) fp.mb.engine = markRaw(fp.mb.defaults.engine); }
  catch (e) { fp.mb.defaultsError = e.message || 'The facts about the data could not be read.'; }
  return fp.mb.defaults;
}
let checkSeq = 0;
export async function runCheck() {
  if (!store.project || !store.project.has_data || ctx.allIssues().length) return null;
  const body = requestBody(), key = JSON.stringify(body.input);
  const A = fp.mb;
  if (A.check.key === key && (A.check.ans || A.check.busy)) return A.check.ans;
  const my = ++checkSeq;
  A.check = { key, ans: A.check.ans, error: '', busy: true };
  if (mbJobs.check.state.running) mbJobs.check.cancel();
  const res = await mbJobs.check.run(`${BASE}/check`, body, { quiet: true });
  if (my !== checkSeq) return null;
  A.check.busy = false;
  if (!res) { if (!mbJobs.check.state.cancelled) { A.check.error = mbJobs.check.state.error || 'The settings could not be checked.'; A.check.ans = null; } return null; }
  A.check.ans = markRaw(res);
  return res;
}

// ---------------------------------------------------------------- fitting
export async function runFit(body = null, { replace = null, go = true } = {}) {
  const A = fp.mb;
  A.fitError = '';
  if (!body) {
    const issues = ctx.allIssues();
    if (issues.length) { A.fitError = issues[0].message; ctx.goStep(issues[0].step); return null; }
  }
  const res = await mbJobs.fit.run(`${BASE}/fit`, { ...(body || requestBody()), ...(replace ? { replace } : {}) }, { quiet: true });
  if (!res) {
    if (mbJobs.fit.state.cancelled) toast('Cancelled: the sampler was stopped and nothing was saved.', 'info', '', 3500);
    else { fp.mb.fitError = mbJobs.fit.state.error || 'The model could not be fitted.'; toast(fp.mb.fitError, 'warn', '', 9000); }
    return null;
  }
  fp.rid = res.rid; fp.params = res.result.params; fp.stale = [];
  await loadResult(res.rid, true);
  loadCatalogue(true);
  if (go) ctx.goStep(6);
  return res;
}
export function cancelFit() { if (mbJobs.fit.state.running) mbJobs.fit.cancel(); }

// ---------------------------------------------------------------- a fitted model
let resSeq = 0;
export async function loadResult(rid = fp.rid, force = false) {
  const A = fp.mb;
  if (!rid) { A.res = null; A.resRid = null; return null; }
  if (!force && A.resRid === rid && A.res) return A.res;
  const my = ++resSeq;
  A.resBusy = true;
  try {
    const r = await get(`${BASE}/${enc(rid)}`, null, { quiet: true });
    if (my !== resSeq) return null;
    const B = fp.mb;
    if (B.resRid !== rid) { B.extra = {}; B.pred.sample = null; B.view.combo = null; B.cmp.row = null; }
    B.res = markRaw({ result: r.result, p: r.payload });
    B.resRid = rid; B.resError = '';
    fp.stale = r.result.stale || []; fp.params = r.result.params;
    return B.res;
  } catch (e) {
    if (my === resSeq) { fp.mb.res = null; fp.mb.resRid = null; fp.mb.resError = e.message || 'This model could not be opened.'; }
    return null;
  } finally { if (my === resSeq) fp.mb.resBusy = false; }
}

/** Ask once and keep the answer (per key): `{data, error, busy}`. */
function ask(key, fn) {
  const A = fp.mb;
  const hit = A.extra[key];
  if (hit && (hit.data || hit.busy)) return hit;
  A.extra[key] = { data: null, error: '', busy: true };
  const cur = A.extra[key];
  fn().then((d) => { cur.data = markRaw(d); }).catch((e) => { cur.error = e.message || 'The answer could not be read.'; }).finally(() => { cur.busy = false; });
  return cur;
}
export function samplePrediction(o) {
  const rid = fp.mb.resRid;
  if (!rid || !o.sample) return null;
  const q = { target: o.target, sources: o.sources, covariates: o.covariates ? 1 : 0, noisy: o.noisy ? 1 : 0, held_out: o.heldOut ? 1 : 0 };
  return ask(`sample|${rid}|${o.sample}|${JSON.stringify(q)}`, () => get(`${BASE}/${enc(rid)}/sample/${enc(o.sample)}`, q, { quiet: true }));
}
export function predictable(target, sources, covariates) {
  const rid = fp.mb.resRid;
  if (!rid || !target) return null;
  const q = { target, sources, covariates: covariates ? 1 : 0 };
  return ask(`pred|${rid}|${JSON.stringify(q)}`, () => get(`${BASE}/${enc(rid)}/predictable`, q, { quiet: true }));
}
export function correlation(a, b, x, y, j) {
  const rid = fp.mb.resRid;
  if (!rid || !a || !b) return null;
  const q = { a, b, ...(x !== null && x !== undefined ? { x } : {}), ...(y !== null && y !== undefined ? { y } : {}), ...(j !== null && j !== undefined ? { j } : {}) };
  return ask(`corr|${rid}|${JSON.stringify(q)}`, () => get(`${BASE}/${enc(rid)}/correlation`, q, { quiet: true }));
}
export function contrast(key, a, b) {
  const rid = fp.mb.resRid;
  if (!rid || !key || !a || !b) return null;
  const q = { key, a, b };
  return ask(`contrast|${rid}|${JSON.stringify(q)}`, () => get(`${BASE}/${enc(rid)}/contrast`, q, { quiet: true }));
}
export function category(o) {
  const rid = fp.mb.resRid;
  if (!rid || !o.key || !o.level) return null;
  const q = { key: o.key, level: o.level, how: o.how || 'pure', target: o.target, noisy: o.noisy ? 1 : 0 };
  return ask(`category|${rid}|${JSON.stringify(q)}`, () => get(`${BASE}/${enc(rid)}/category`, q, { quiet: true }));
}
export async function runCompare(targets = null, maxPoints = 300) {
  const rid = fp.mb.resRid;
  if (!rid) return null;
  const res = await mbJobs.compare.run(`${BASE}/${enc(rid)}/compare`, { ...(targets ? { targets } : {}), max_points: maxPoints }, { quiet: true });
  if (!res) { if (!mbJobs.compare.state.cancelled) toast(mbJobs.compare.state.error || 'The comparison could not be made.', 'warn', '', 8000); return null; }
  await loadResult(rid, true);
  return res;
}
export function loadMethods() {
  const rid = fp.mb.resRid;
  if (!rid) return null;
  return ask(`methods|${rid}`, () => get(`${BASE}/${enc(rid)}/methods`, null, { quiet: true }));
}
export function exportFile(what) {
  const rid = fp.mb.resRid;
  if (rid) download(`${BASE}/${enc(rid)}/export?what=${enc(what)}`, `${rid}-${what}.csv`);
}
export { post };
