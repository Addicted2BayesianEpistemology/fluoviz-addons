// Everything the Self-organising map page asks the server for (docs/api.md).  Module level, like the state: a computation that was started keeps running while the person looks at
// another page.  Rules kept here: every answer is checked against a sequence number (a quick second click is never overwritten by an older answer); the long computations
// run one at a time and a request that arrives meanwhile is QUEUED, the last one wins; nothing is ever computed silently.
import { markRaw, reactive } from 'vue';
import { get, post, download } from '/static/js/api.js';
import { store, toast } from '/static/js/store.js';
import { loadCatalogue } from '/static/js/variables.js';
import { useLatest } from '/static/js/components/FpcaCommon.js';
import { vec, mat, stack } from '/static/js/components/DecompCommon.js';
import { ctx, fp, somJobs, BASE, requestBody, variablesOf } from './SomState.js';

const enc = encodeURIComponent;

// ---------------------------------------------------------------- the advice of the server
let defaultsSeq = 0;
export async function loadDefaults(force = false) {
  if (!store.project || !store.project.has_data) return null;
  const A = fp.som, names = variablesOf(fp.spec);
  const key = [names.join(','), fp.spec.fluorescence.min_coverage, fp.spec.absorbance.processed, store.project ? JSON.stringify(store.project.signature) : ''].join('|');
  if (!force && A.defaults && A.defaultsKey === key) return A.defaults;
  const my = ++defaultsSeq;
  A.defaultsBusy = true;
  try {
    const d = await get(`${BASE}/defaults`, { variables: names.join(','), min_coverage: fp.spec.fluorescence.min_coverage, abs_processed: fp.spec.absorbance.processed, advise: 1 }, { quiet: true });
    if (my !== defaultsSeq) return fp.som.defaults;
    A.defaults = markRaw(d); A.defaultsKey = key; A.defaultsError = '';
    return d;
  } catch (e) {
    if (my === defaultsSeq) A.defaultsError = e.message || 'The advice could not be read.';
    return null;
  } finally { if (my === defaultsSeq) A.defaultsBusy = false; }
}

// ---------------------------------------------------------------- long computations, one at a time (the last request that arrives while one runs is kept)
const heavy = reactive({ busy: false, pending: null });
function submitHeavy(label, fn) {
  return new Promise((resolve) => {
    const run = async () => {
      heavy.busy = true; fp.som.queued = '';
      try { resolve(await fn()); } catch (e) { console.error(e); resolve(null); }
      finally {
        heavy.busy = false;
        const next = heavy.pending; heavy.pending = null;
        if (next) next.run(); else fp.som.queued = '';
      }
    };
    if (!heavy.busy) { run(); return; }
    if (heavy.pending) heavy.pending.resolve(null);
    heavy.pending = { run, resolve };
    fp.som.queued = label;
  });
}
export const heavyBusy = () => heavy.busy;
export function cancelHeavy() {
  if (heavy.pending) { heavy.pending.resolve(null); heavy.pending = null; fp.som.queued = ''; }
  if (somJobs.fit.state.running) somJobs.fit.cancel();
  if (somJobs.scan.state.running) somJobs.scan.cancel();
}
export function dropQueued() { if (heavy.pending) { heavy.pending.resolve(null); heavy.pending = null; } fp.som.queued = ''; }

// ---------------------------------------------------------------- the map in numbers (no fit: what a request would do)
const latestCheck = useLatest(somJobs.check);
export async function runMapCheck() {
  if (!store.project || !store.project.has_data) return null;
  const A = fp.som;
  if (ctx.allIssues().length) return null;
  const body = requestBody(), key = JSON.stringify(body);
  if (A.check.key === key && (A.check.ans || A.check.busy)) return A.check.ans;
  const prev = A.check.ans;
  A.check = { key, ans: prev, error: '', busy: true };
  const slot = A.check;
  const res = await latestCheck.run(`${BASE}/check`, body);
  if (fp.som.check !== slot) return null;
  slot.busy = false;
  if (!res) { if (!somJobs.check.state.cancelled) { slot.error = somJobs.check.state.error || 'The map could not be checked.'; slot.ans = null; } return null; }
  slot.ans = markRaw(res);
  return res;
}

// ---------------------------------------------------------------- fitting
/** Compute the map (docs/api.md 2.3).  `body` = a complete request (a stored `params`: "Compute again"), `replace` = the id of the result to replace in place. */
export function runFit(body = null, { replace = null, go = true } = {}) {
  const A = fp.som;
  A.fitError = '';
  if (!body) {
    const issues = ctx.allIssues();
    if (issues.length) { A.fitError = issues[0].message; ctx.goStep(issues[0].step); return Promise.resolve(null); }
  }
  return submitHeavy('Compute the map', async () => {
    const res = await somJobs.fit.run(`${BASE}/fit`, { ...(body || requestBody()), ...(replace ? { replace } : {}) }, { quiet: true });
    const B = fp.som;
    if (!res) {
      if (somJobs.fit.state.cancelled) toast('Cancelled: nothing was saved.', 'info', '', 3200);
      else { B.fitError = somJobs.fit.state.error || 'The map could not be computed.'; toast(B.fitError, 'warn', '', 9000); }
      return null;
    }
    fp.rid = res.rid; fp.params = res.result.params; fp.stale = [];
    B.scans = {}; B.planeCache = {}; B.planesAdded = [];
    await loadResult(res.rid, true);
    loadCatalogue(true);
    if (go) ctx.goStep(7);
    return res;
  });
}
/** "Run with the recommended settings": the person's variables are kept, everything else comes from the advice of the server. */
export async function runRecommended() {
  const A = fp.som;
  A.fitError = '';
  const d0 = fp.data || await ctx.loadData();
  if (!d0) { A.fitError = fp.dataError || 'The data could not be read. Is the FluoViz server still running?'; ctx.goStep(6); return null; }
  const issues = ctx.allIssues().filter((i) => i.step <= 2);
  if (issues.length) { A.fitError = issues[0].message; ctx.goStep(issues[0].step); return null; }
  A.preparing = true;
  ctx.goStep(6);
  try {
    const def = await loadDefaults(true);
    if (!def) { fp.som.fitError = fp.som.defaultsError || 'The advice for the map could not be read.'; return null; }
    ctx.resetSpec(true);
    fp.spec.map.other.on = false;
  } finally { fp.som.preparing = false; }
  return runFit(requestBody(fp.spec));
}

// ---------------------------------------------------------------- a computed map
const grid = (a) => (a === null || a === undefined ? null : a);
function prepareResult(r) {
  const p = r.payload, M = p.map.units;
  const o = { result: r.result, p, ids: (p.ids || []).map(String), n: p.n, M, rows: p.map.rows, cols: p.map.cols, lattice: p.map.lattice, shape: p.map.shape, kind: p.kind };
  o.coords = mat(p.map.coords); o.umat = vec(p.map.umatrix); o.hits = vec(p.map.hits); o.unitNames = p.map.unit_names; o.edges = mat(p.map.edges);
  const S = p.samples;
  o.unit = vec(S.unit); o.pos = mat(S.pos); o.qe = vec(S.qe); o.qeRel = vec(S.qe_rel); o.held = vec(S.heldout_qe); o.heldRel = vec(S.heldout_qe_rel); o.fold = vec(S.fold);
  o.status = Array.from(S.status || []); o.flag = Array.from(S.flag || []); o.missing = S.missing ? vec(S.missing) : null; o.scores = S.scores ? mat(S.scores) : null;
  o.planes = {};
  for (const [k, pl] of Object.entries(p.planes || {})) o.planes[k] = preparePlane(pl);
  const c = p.clusters || {};
  o.clusters = { ...c, unitLabels: vec(c.unit_labels), sampleLabels: vec(c.sample_labels) };
  o.protos = p.prototypes ? Object.fromEntries(Object.entries(p.prototypes).map(([nm, a]) => [nm, stack(a)])) : null;
  o.protoCurves = p.prototype_curves ? Object.fromEntries(Object.entries(p.prototype_curves).map(([nm, a]) => [nm, mat(a)])) : null;
  o.protoValues = p.prototype_values ? mat(p.prototype_values) : null;
  o.quality = p.quality; o.verdicts = p.verdicts; o.history = p.training && p.training.history;
  o.variablesMode = p.kind === 'variables';
  return o;
}
export function preparePlane(pl) {
  const out = { ...pl, v: vec(pl.values) };
  if (pl.purity) out.purity = vec(pl.purity);
  return out;
}
let resSeq = 0;
export async function loadResult(rid = fp.rid, force = false, { keep = false } = {}) {
  const A = fp.som;
  if (!rid) { A.res = null; A.resRid = null; return null; }
  if (!force && A.resRid === rid && A.res) return A.res;
  const my = ++resSeq;
  A.resBusy = true;
  try {
    const r = await get(`${BASE}/${enc(rid)}`, null, { quiet: true });
    if (my !== resSeq) return null;
    const B = fp.som;
    if (!keep || B.resRid !== rid) { B.extra = {}; B.view.sample = null; B.view.unit = null; B.planeCache = {}; B.planesAdded = []; }
    B.res = markRaw(prepareResult(r)); B.resRid = rid; B.resError = '';
    fp.stale = r.result.stale || []; fp.params = r.result.params;
    return B.res;
  } catch (e) {
    if (my === resSeq) { const B = fp.som; B.res = null; B.resRid = null; B.resError = e.message || 'This map could not be opened.'; }
    return null;
  } finally { if (my === resSeq) fp.som.resBusy = false; }
}

// ---------------------------------------------------------------- small helpers for the "ask once, keep the answer" views
async function ask(store_, key, fn, prepare = (x) => markRaw(x)) {
  const hit = store_[key];
  if (hit && (hit.data || hit.busy)) return hit;
  store_[key] = { data: null, error: '', busy: true };
  const cur = store_[key];
  try { cur.data = prepare(await fn()); } catch (e) { cur.error = e.message || 'The answer could not be read.'; }
  cur.busy = false;
  return cur;
}

export function loadSample(sid, thumb = false) {
  const A = fp.som, rid = A.resRid;
  if (!rid || !sid) return null;
  if (!A.extra.samples) A.extra.samples = {};
  return ask(A.extra.samples, `${rid}|${sid}|${thumb ? 't' : 'f'}`, () => get(`${BASE}/${enc(rid)}/sample/${enc(sid)}`, thumb ? { thumb: 1 } : null, { quiet: true }));
}
export function loadUnit(index) {
  const A = fp.som, rid = A.resRid;
  if (!rid || index === null || index === undefined) return null;
  if (!A.extra.units) A.extra.units = {};
  return ask(A.extra.units, `${rid}|${index}`, () => get(`${BASE}/${enc(rid)}/unit/${index}`, null, { quiet: true }));
}
export function loadPrototypes(what = 'prototype') {
  const A = fp.som, rid = A.resRid;
  if (!rid) return null;
  if (!A.extra.protos) A.extra.protos = {};
  return ask(A.extra.protos, `${rid}|${what}`, () => get(`${BASE}/${enc(rid)}/prototypes`, { what }, { quiet: true }), (d) => markRaw({ ...d,
    stacks: Object.fromEntries(Object.entries(d.surfaces || {}).map(([nm, a]) => [nm, stack(a)])), curveMats: Object.fromEntries(Object.entries(d.curves || {}).map(([nm, a]) => [nm, mat(a)])) }));
}
/** A plane of the map: `{kind: 'stored', key}`, a feature, an index, or a variable.  The answer is kept under `key` in `fp.som.planeCache`. */
export async function loadPlane(body, keep = true) {
  const A = fp.som, rid = A.resRid;
  if (!rid) return null;
  const key = JSON.stringify(body);
  const hit = A.planeCache[key];
  if (hit && (hit.data || hit.busy)) return hit;
  A.planeCache[key] = { data: null, error: '', busy: true, body };
  const cur = A.planeCache[key];
  try { const d = await post(`${BASE}/${enc(rid)}/plane`, { ...body, keep }, { quiet: true }); cur.data = markRaw(preparePlane(d)); }
  catch (e) { cur.error = e.message || 'The plane could not be computed.'; }
  cur.busy = false;
  return cur;
}
export async function loadIndices() {
  const A = fp.som, rid = A.resRid;
  if (!rid) return null;
  return ask(A.extra, `indices|${rid}`, () => get(`${BASE}/${enc(rid)}/indices`, null, { quiet: true }));
}
export function loadGroups(force = false) {
  const A = fp.som, rid = A.resRid;
  if (!rid) return null;
  if (force) delete A.extra[`groups|${rid}`];
  return ask(A.extra, `groups|${rid}`, () => get(`${BASE}/${enc(rid)}/groups`, null, { quiet: true }), (d) => markRaw({ ...d, images: d.images ? { ...d.images, mean: stack(d.images.mean), diff: stack(d.images.difference), z: d.images.z ? stack(d.images.z) : null } : null }));
}
export async function runCluster(opts) {
  const A = fp.som, rid = A.resRid;
  if (!rid) return null;
  A.extra.clusterBusy = true; A.extra.clusterError = '';
  try {
    await post(`${BASE}/${enc(rid)}/cluster`, opts, { quiet: true });
    delete A.extra[`groups|${rid}`];
    await loadResult(rid, true, { keep: true });
    loadCatalogue(true);
  } catch (e) { A.extra.clusterError = e.message || 'The groups could not be made.'; }
  A.extra.clusterBusy = false;
  return A.res;
}
export function runAnomalies(opts) {
  const A = fp.som, rid = A.resRid;
  if (!rid) return null;
  const key = `anom|${rid}|${JSON.stringify(opts)}`;
  delete A.extra[key];
  return ask(A.extra, key, () => post(`${BASE}/${enc(rid)}/anomalies`, opts, { quiet: true })).then((s) => { A.extra.anomKey = key; return s; });
}
export function runTrajectories(opts) {
  const A = fp.som, rid = A.resRid;
  if (!rid) return null;
  const key = `traj|${rid}|${JSON.stringify(opts)}`;
  delete A.extra[key];
  return ask(A.extra, key, () => post(`${BASE}/${enc(rid)}/trajectories`, opts, { quiet: true })).then((s) => { A.extra.trajKey = key; return s; });
}
export function loadProjection() {
  const A = fp.som, rid = A.resRid;
  if (!rid) return null;
  return ask(A.extra, `proj|${rid}`, () => get(`${BASE}/${enc(rid)}/projection`, null, { quiet: true }));
}
export function loadMethods() {
  const A = fp.som, rid = A.resRid;
  if (!rid) return null;
  return ask(A.extra, `methods|${rid}`, () => get(`${BASE}/${enc(rid)}/methods`, null, { quiet: true }), (d) => d);
}

// ---------------------------------------------------------------- the scans (they fit several maps: seconds to minutes)
/** kind: 'size' | 'stability' | 'metric'.  source 'result' = the map that is shown (its stored settings), 'request' = the settings on the page. */
export function runScan(kind, { source = 'request', options = {} } = {}) {
  const A = fp.som;
  const rid = source === 'result' ? A.resRid : null;
  if (source === 'request') { const issues = ctx.allIssues(); if (issues.length) { A.scans[kind] = { key: '', source, ans: A.scans[kind] ? A.scans[kind].ans : null, error: issues[0].message, busy: false }; return Promise.resolve(null); } }
  const body = { kind, ...options };
  if (rid) body.result = rid; else body.request = requestBody();
  const key = JSON.stringify([kind, rid || body.request, options]);
  const label = { size: 'Map-size scan', stability: 'Check the stability', metric: 'Compare the distances' }[kind] || 'The scan';
  const prev = A.scans[kind];
  A.scans[kind] = { key, source, ans: prev ? prev.ans : null, error: '', busy: true, rid, queued: heavy.busy };
  const slot = A.scans[kind];
  return submitHeavy(label, async () => {
    slot.queued = false;
    const res = await somJobs.scan.run(`${BASE}/scan`, body, { quiet: true });
    if (fp.som.scans[kind] !== slot) return null;
    slot.busy = false;
    if (!res) { if (!somJobs.scan.state.cancelled) slot.error = somJobs.scan.state.error || 'The scan could not be run.'; return null; }
    slot.ans = markRaw(res);
    return res;
  });
}

export function exportFile(what) {
  const rid = fp.som.resRid;
  if (!rid) return Promise.resolve(false);
  return download(`${BASE}/${enc(rid)}/export?${new URLSearchParams({ what }).toString()}`, `${rid}_${what}`);
}

// ---------------------------------------------------------------- thumbnails of all the samples (hover cards): the gallery of the project, once
let galSeq = 0;
export async function loadGallery() {
  const A = fp.som, stamp = `${store.samplesRev}|${store.project ? store.project.n_included : ''}`;
  if (A.gallery && A.gallery.stamp === stamp && (A.gallery.data || A.gallery.busy)) return A.gallery;
  const my = ++galSeq;
  A.gallery = { stamp, data: null, error: '', busy: true };
  const slot = A.gallery;
  try { const g = await get('/api/gallery', { max_em: 48, max_ex: 24 }, { quiet: true }); if (my === galSeq) slot.data = markRaw({ ...g, index: new Map(g.ids.map((id, i) => [id, i])) }); }
  catch (e) { if (my === galSeq) slot.error = e.message || 'The pictures of the samples could not be read.'; }
  if (my === galSeq) slot.busy = false;
  return slot;
}
