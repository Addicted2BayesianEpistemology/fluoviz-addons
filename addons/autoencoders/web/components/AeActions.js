// Everything the Autoencoders page asks the server for (docs/api.md).  Module level, like the state: a computation that was started keeps running while the person looks at
// another page.  Rules kept here: every answer is checked against a sequence number (a quick second click is never overwritten by an older answer); the long computations
// (training, the scans) run one at a time and a request that arrives meanwhile is QUEUED, the last one wins; nothing is ever trained silently.
import { markRaw, watch } from 'vue';
import { get, post, download } from '/static/js/api.js';
import { store, toast } from '/static/js/store.js';
import { sleep } from '/static/js/util.js';
import { loadCatalogue } from '/static/js/variables.js';
import { useLatest } from '/static/js/components/FpcaCommon.js';
import { vec, mat } from '/static/js/components/DecompCommon.js';
import { ctx, fp, aeJobs, BASE, requestBody, freshLive, variablesOf } from './AeState.js';

const enc = encodeURIComponent;

// ---------------------------------------------------------------- the advice of the server (facts about the data, the number of codes, whether PyTorch is there)
let defaultsSeq = 0;
export async function loadDefaults(force = false) {
  if (!store.project || !store.project.has_data) return null;
  const A = fp.ae, names = variablesOf(fp.spec);
  const key = [names.join(','), fp.spec.fluorescence.min_coverage, fp.spec.absorbance.processed, store.project ? JSON.stringify(store.project.signature) : ''].join('|');
  if (!force && A.defaults && A.defaultsKey === key) return A.defaults;
  const my = ++defaultsSeq;
  A.defaultsBusy = true;
  try {
    const d = await get(`${BASE}/defaults`, { variables: names.join(','), min_coverage: fp.spec.fluorescence.min_coverage, abs_processed: fp.spec.absorbance.processed, advise: 1 }, { quiet: true });
    if (my !== defaultsSeq) return fp.ae.defaults;
    A.defaults = markRaw(d); A.defaultsKey = key; A.defaultsError = '';
    return d;
  } catch (e) {
    if (my === defaultsSeq) A.defaultsError = e.message || 'The advice could not be read.';
    return null;
  } finally { if (my === defaultsSeq) A.defaultsBusy = false; }
}

// ---------------------------------------------------------------- heavy computations, one at a time (the last request that arrives while one runs is kept)
const heavy = { busy: false, pending: null };
function submitHeavy(label, fn) {
  return new Promise((resolve) => {
    const run = async () => {
      heavy.busy = true; fp.ae.queued = '';
      try { resolve(await fn()); } catch (e) { console.error(e); resolve(null); }
      finally {
        heavy.busy = false;
        const next = heavy.pending; heavy.pending = null;
        if (next) next.run(); else fp.ae.queued = '';
      }
    };
    if (!heavy.busy) { run(); return; }
    if (heavy.pending) heavy.pending.resolve(null);              // an older queued request is dropped: the last one wins
    heavy.pending = { run, resolve };
    fp.ae.queued = label;
  });
}
export const heavyBusy = () => heavy.busy;
/** Cancel what runs (training or a scan) and forget the queued request. */
export function cancelHeavy() {
  if (heavy.pending) { heavy.pending.resolve(null); heavy.pending = null; fp.ae.queued = ''; }
  if (aeJobs.fit.state.running) aeJobs.fit.cancel();
  if (aeJobs.scan.state.running) aeJobs.scan.cancel();
}
export function dropQueued() { if (heavy.pending) { heavy.pending.resolve(null); heavy.pending = null; } fp.ae.queued = ''; }

// ---------------------------------------------------------------- the live history of the network that is being trained (polled twice a second)
let pollSeq = 0;
async function poll(jobId, kind) {
  const my = ++pollSeq;
  const L = Object.assign(fp.ae.live, freshLive(), { kind, jobId, status: 'running' });
  let rows = [];
  for (;;) {
    if (my !== pollSeq) return;
    try {
      const h = await get(`${BASE}/fit/${enc(jobId)}/history`, { since: L.next }, { quiet: true });
      if (my !== pollSeq) return;
      if (h.epochs && h.epochs.length) { rows = rows.concat(h.epochs); L.rows = markRaw(rows); }
      L.next = h.next ?? L.next; L.phase = h.phase || ''; L.restart = h.restart ?? 0; L.nRestarts = h.n_restarts ?? 0; L.fold = h.fold ?? null; L.nFolds = h.n_folds ?? null;
      L.message = h.message || ''; L.best = h.best || null; L.seq++;
      if (['done', 'error', 'cancelled'].includes(h.status)) return;
    } catch { /* the job is not known yet, or the server is busy: try again */ }
    await sleep(500);
  }
}
watch(() => aeJobs.fit.state.jobId, (id) => { if (id) poll(id, 'fit'); else pollSeq++; });
watch(() => aeJobs.scan.state.jobId, (id) => { if (id) poll(id, 'scan'); else pollSeq++; });

// ---------------------------------------------------------------- the network in numbers (no training: what a request would do)
const latestCheck = useLatest(aeJobs.check);
export async function runNetCheck() {
  if (!store.project || !store.project.has_data) return null;
  const A = fp.ae;
  if (ctx.allIssues().length) return null;
  const body = requestBody(), key = JSON.stringify(body);
  if (A.check.key === key && (A.check.ans || A.check.busy)) return A.check.ans;
  const prev = A.check.ans;
  A.check = { key, ans: prev, error: '', busy: true };
  const slot = A.check;
  const res = await latestCheck.run(`${BASE}/check`, body);
  if (fp.ae.check !== slot) return null;
  slot.busy = false;
  if (!res) { if (!aeJobs.check.state.cancelled) { slot.error = aeJobs.check.state.error || 'The network could not be checked.'; slot.ans = null; } return null; }
  slot.ans = markRaw(res);
  return res;
}

// ---------------------------------------------------------------- training
/** Train the network (docs/api.md 2.3).  `body` = a complete request (a stored `params`: "Train again"), `replace` = the id of the result to replace in place. */
export function runFit(body = null, { replace = null, go = true } = {}) {
  const A = fp.ae;
  A.fitError = '';
  if (!body) {
    const issues = ctx.allIssues();
    if (issues.length) { A.fitError = issues[0].message; ctx.goStep(issues[0].step); return Promise.resolve(null); }
    const d = A.defaults;
    if (d && d.torch && d.torch.available === false) { A.fitError = d.torch.words || 'Training needs PyTorch. Open Settings → Add-ons and press Install for Autoencoders (if it is already listed as installed, remove it and install it again).'; return Promise.resolve(null); }
  }
  return submitHeavy('Train the network', async () => {
    const L = Object.assign(fp.ae.live, freshLive(), { kind: 'fit', status: 'running' });
    const res = await aeJobs.fit.run(`${BASE}/fit`, { ...(body || requestBody()), ...(replace ? { replace } : {}) }, { quiet: true });
    const B = fp.ae;
    if (!res) {
      L.status = aeJobs.fit.state.cancelled ? 'cancelled' : 'error';
      if (aeJobs.fit.state.cancelled) toast('Cancelled: nothing was saved.', 'info', '', 3200);
      else { B.fitError = aeJobs.fit.state.error || 'The network could not be trained.'; toast(B.fitError, 'warn', '', 9000); }
      return null;
    }
    L.status = 'done';
    fp.rid = res.rid; fp.params = res.result.params; fp.stale = [];
    B.scans = {};
    await loadResult(res.rid, true);
    loadCatalogue(true);
    if (go) ctx.goStep(7);
    return res;
  });
}
/** "Run with the recommended settings": the person's variables are kept, everything else comes from the advice of the server.  The page goes to step 6 at once (it says that the advice is being asked for). */
export async function runRecommended() {
  const A = fp.ae;
  A.fitError = '';
  const d0 = fp.data || await ctx.loadData();
  if (!d0) { A.fitError = fp.dataError || 'The data could not be read. Is the FluoViz server still running?'; ctx.goStep(6); return null; }
  const issues = ctx.allIssues().filter((i) => i.step <= 2);
  if (issues.length) { A.fitError = issues[0].message; ctx.goStep(issues[0].step); return null; }
  A.preparing = true;
  ctx.goStep(6);
  try {
    const def = await loadDefaults(true);
    if (!def) { fp.ae.fitError = fp.ae.defaultsError || 'The advice for the network could not be read.'; return null; }
    ctx.resetSpec(true);
    const n = fp.spec.net;
    n.latent_dim = def.advice && def.advice.latent_dim ? def.advice.latent_dim : null;
    n.size = def.advice && def.advice.size && ['small', 'medium', 'large'].includes(def.advice.size) ? def.advice.size : 'small';
    n.family = 'ae';
  } finally { fp.ae.preparing = false; }
  return runFit(requestBody(fp.spec, fp.ae.defaults));
}

// ---------------------------------------------------------------- a trained network
function prepareResult(r) {
  const p = r.payload, rec = p.recon || {};
  const o = { result: r.result, p, ids: p.ids || [], n: p.n, d: p.latent_dim, family: p.family };
  o.codes = mat(p.codes); o.codesSd = p.codes_sd ? mat(p.codes_sd) : null;
  o.codeNames = (p.code_names || []).filter((x) => !/_sd$/.test(x));
  o.codeMean = vec(p.code_mean); o.codeSd = vec(p.code_sd);
  o.err = vec(rec.error); o.r2 = vec(rec.r2); o.rmse = vec(rec.rmse); o.typ = rec.typicality ? vec(rec.typicality) : null;
  o.held = p.split && p.split.flag ? vec(p.split.flag) : null;
  o.missing = rec.missing ? vec(rec.missing) : null;                   // the share of the domain each sample lacked (a region cut out of it); null = none did
  o.flag = rec.flag || [];
  o.scansSaved = p.scans || {};
  return o;
}
let resSeq = 0;
export async function loadResult(rid = fp.rid, force = false, { keep = false } = {}) {
  const A = fp.ae;
  if (!rid) { A.res = null; A.resRid = null; return null; }
  if (!force && A.resRid === rid && A.res) return A.res;
  const my = ++resSeq;
  A.resBusy = true;
  try {
    const r = await get(`${BASE}/${enc(rid)}`, null, { quiet: true });
    if (my !== resSeq) return null;
    const B = fp.ae;
    if (!keep || B.resRid !== rid) { B.extra = {}; B.sample = null; }
    else if ((r.result.stale || []).join('|') !== (fp.stale || []).join('|') && B.extra.samples) B.extra.samples = {};         // the data changed (or are as they were): the single-sample views are asked for again
    B.res = markRaw(prepareResult(r)); B.resRid = rid; B.resError = '';
    fp.stale = r.result.stale || []; fp.params = r.result.params;
    return B.res;
  } catch (e) {
    if (my === resSeq) { const B = fp.ae; B.res = null; B.resRid = null; B.resError = e.message || 'This network could not be opened.'; }
    return null;
  } finally { if (my === resSeq) fp.ae.resBusy = false; }
}

// ---------------------------------------------------------------- one sample against the network
export async function loadSample(sid, thumb = false) {
  const A = fp.ae, rid = A.resRid;
  if (!rid || !sid) return null;
  if (!A.extra.samples) A.extra.samples = {};
  const map = A.extra.samples, key = `${rid}|${sid}|${thumb ? 't' : 'f'}`;
  const hit = map[key];
  if (hit && (hit.data || hit.busy)) return hit;
  map[key] = { data: null, error: '', busy: true };
  const cur = map[key];                                           // (read back: the reactive version, so that the page sees the answer arrive)
  try { cur.data = markRaw(await get(`${BASE}/${enc(rid)}/sample/${enc(sid)}`, thumb ? { thumb: 1 } : null, { quiet: true })); }
  catch (e) { cur.error = e.message || 'The sample could not be looked at.'; }
  cur.busy = false;
  return cur;
}

/** What every code does: the decoded samples along each code, in standard deviations of the codes of the training samples (docs/api.md 2.6). */
let travSeq = 0;
export async function loadTraverse(range = 3) {
  const A = fp.ae, rid = A.resRid;
  if (!rid) return null;
  const key = `${rid}|${range}`;
  if (A.extra.trav && A.extra.trav.key === key && (A.extra.trav.data || A.extra.trav.busy)) return A.extra.trav;
  const my = ++travSeq, prev = A.extra.trav;
  const values = [];
  for (let v = -range; v <= range; v += 1) values.push(v);
  A.extra.trav = { key, data: prev ? prev.data : null, error: '', busy: true, values };
  const slot = A.extra.trav;
  try { const r = await post(`${BASE}/${enc(rid)}/traverse`, { code: null, values, base: 'mean' }, { quiet: true }); if (my === travSeq) slot.data = markRaw(r); }
  catch (e) { if (my === travSeq) slot.error = e.message || 'The pictures could not be made.'; }
  if (my === travSeq) slot.busy = false;
  return slot;
}
let sensSeq = 0;
export async function loadSensitivity() {
  const A = fp.ae, rid = A.resRid;
  if (!rid) return null;
  if (A.extra.sens && A.extra.sens.rid === rid && (A.extra.sens.data || A.extra.sens.busy)) return A.extra.sens;
  const my = ++sensSeq;
  A.extra.sens = { rid, data: null, error: '', busy: true };
  const slot = A.extra.sens;
  try { const r = await get(`${BASE}/${enc(rid)}/sensitivity`, null, { quiet: true }); if (my === sensSeq) slot.data = markRaw(r); }
  catch (e) { if (my === sensSeq) slot.error = e.message || 'The sensitivity could not be computed.'; }
  if (my === sensSeq) slot.busy = false;
  return slot;
}
let interpSeq = 0;
export async function loadInterp(a, b, steps) {
  const A = fp.ae, rid = A.resRid;
  if (!rid || !a || !b) return null;
  const key = `${rid}|${a}|${b}|${steps}`;
  if (A.extra.interp && A.extra.interp.key === key && (A.extra.interp.data || A.extra.interp.busy)) return A.extra.interp;
  const my = ++interpSeq, prev = A.extra.interp;
  A.extra.interp = { key, data: prev ? prev.data : null, error: '', busy: true };
  const slot = A.extra.interp;
  try { const r = await post(`${BASE}/${enc(rid)}/interpolate`, { a, b, steps }, { quiet: true }); if (my === interpSeq) slot.data = markRaw(r); }
  catch (e) { if (my === interpSeq) { slot.error = e.message || 'The two samples could not be mixed.'; slot.data = null; } }
  if (my === interpSeq) slot.busy = false;
  return slot;
}
let genSeq = 0;
export async function runGenerate(opts) {
  const A = fp.ae, rid = A.resRid;
  if (!rid) return null;
  const my = ++genSeq, prev = A.extra.gen;
  A.extra.gen = { opts: { ...opts }, data: prev ? prev.data : null, error: '', busy: true };
  const slot = A.extra.gen;
  try { const r = await post(`${BASE}/${enc(rid)}/generate`, opts, { quiet: true }); if (my === genSeq) slot.data = markRaw(r); }
  catch (e) { if (my === genSeq) { slot.error = e.message || 'New samples could not be drawn.'; slot.data = null; } }
  if (my === genSeq) slot.busy = false;
  return slot;
}
let anSeq = 0;
export async function runAnomalies(opts) {
  const A = fp.ae, rid = A.resRid;
  if (!rid) return null;
  const my = ++anSeq, prev = A.extra.anom;
  A.extra.anom = { opts: { ...opts }, data: prev ? prev.data : null, error: '', busy: true };
  const slot = A.extra.anom;
  try { const r = await post(`${BASE}/${enc(rid)}/anomalies`, opts, { quiet: true }); if (my === anSeq) slot.data = markRaw(r); }
  catch (e) { if (my === anSeq) { slot.error = e.message || 'The unusual samples could not be found.'; slot.data = null; } }
  if (my === anSeq) slot.busy = false;
  return slot;
}
let relSeq = 0;
export async function runRelate(variables) {
  const A = fp.ae, rid = A.resRid;
  if (!rid || !variables.length) return null;
  const my = ++relSeq, prev = A.extra.relate;
  A.extra.relate = { vars: variables.slice(), data: prev ? prev.data : null, error: '', busy: true };
  const slot = A.extra.relate;
  try { const r = await post(`${BASE}/${enc(rid)}/relate`, { variables, folds: 5 }, { quiet: true }); if (my === relSeq) slot.data = markRaw(r); }
  catch (e) { if (my === relSeq) { slot.error = e.message || 'The codes could not be related to these variables.'; slot.data = null; } }
  if (my === relSeq) slot.busy = false;
  return slot;
}
let methSeq = 0;
export async function loadMethods() {
  const A = fp.ae, rid = A.resRid;
  if (!rid) return null;
  if (A.extra.methods && A.extra.methods.rid === rid && (A.extra.methods.data || A.extra.methods.busy)) return A.extra.methods;
  const my = ++methSeq;
  A.extra.methods = { rid, data: null, error: '', busy: true };
  const slot = A.extra.methods;
  try { const r = await get(`${BASE}/${enc(rid)}/methods`, null, { quiet: true }); if (my === methSeq) slot.data = r; }
  catch (e) { if (my === methSeq) slot.error = e.message || 'The text could not be made.'; }
  if (my === methSeq) slot.busy = false;
  return slot;
}

// ---------------------------------------------------------------- the scans (they train several small networks: minutes)
/** kind: 'latent' | 'smoothify' | 'stability' | 'baseline'.  source 'result' = the network that is shown (its stored settings; the table is kept in the result), 'request' = the settings
 *  on the page (before training).  `options`: the options of the scan (docs/api.md 2.11). */
export function runScan(kind, { source = 'request', options = {} } = {}) {
  const A = fp.ae;
  const rid = source === 'result' ? A.resRid : null;
  if (source === 'request') { const issues = ctx.allIssues(); if (issues.length) { A.scans[kind] = { key: '', source, ans: A.scans[kind] ? A.scans[kind].ans : null, error: issues[0].message, busy: false }; return Promise.resolve(null); } }
  const body = { scan: kind, ...options };
  if (rid) { body.rid = rid; body.save = true; } else body.request = requestBody();
  const key = JSON.stringify([kind, rid || body.request, options]);
  const label = { latent: 'Latent-dimension scan', smoothify: 'Smoothify scan', stability: 'Check the stability', baseline: 'Compare with PCA for every number of codes' }[kind] || 'The scan';
  const prev = A.scans[kind];
  A.scans[kind] = { key, source, ans: prev ? prev.ans : null, error: '', busy: true, rid, queued: heavy.busy };
  const slot = A.scans[kind];
  return submitHeavy(label, async () => {
    slot.queued = false;
    const L = Object.assign(fp.ae.live, freshLive(), { kind: 'scan', status: 'running' });
    const res = await aeJobs.scan.run(`${BASE}/scan`, body, { quiet: true });
    L.status = res ? 'done' : (aeJobs.scan.state.cancelled ? 'cancelled' : 'error');
    if (fp.ae.scans[kind] !== slot) return null;
    slot.busy = false;
    if (!res) { if (!aeJobs.scan.state.cancelled) slot.error = aeJobs.scan.state.error || 'The scan could not be run.'; return null; }
    slot.ans = markRaw(res);
    if (rid && fp.ae.resRid === rid) loadResult(rid, true, { keep: true });          // the table is now kept in the result too
    return res;
  });
}

// ---------------------------------------------------------------- downloads
export function exportFile(what, extra = {}) {
  const rid = fp.ae.resRid;
  if (!rid) return Promise.resolve(false);
  const q = new URLSearchParams({ what, ...extra }).toString();
  return download(`${BASE}/${enc(rid)}/export?${q}`, `${rid}_${what}`);
}

// ---------------------------------------------------------------- thumbnails of all the samples (hover cards): the gallery of the project, once
let galSeq = 0;
export async function loadGallery() {
  const A = fp.ae, stamp = `${store.samplesRev}|${store.project ? store.project.n_included : ''}`;
  if (A.gallery && A.gallery.stamp === stamp && (A.gallery.data || A.gallery.busy)) return A.gallery;
  const my = ++galSeq;
  A.gallery = { stamp, data: null, error: '', busy: true };
  const slot = A.gallery;
  try { const g = await get('/api/gallery', { max_em: 48, max_ex: 24 }, { quiet: true }); if (my === galSeq) slot.data = markRaw({ ...g, index: new Map(g.ids.map((id, i) => [id, i])) }); }
  catch (e) { if (my === galSeq) slot.error = e.message || 'The pictures of the samples could not be read.'; }
  if (my === galSeq) slot.busy = false;
  return slot;
}

