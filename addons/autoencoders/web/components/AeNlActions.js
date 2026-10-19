// What the tab "Nonlinear effects" asks the server for (docs/api.md 2.15): the response curves, the maps of where the response bends, the local loadings, two codes together.  Each answer
// lives in `fp.ae.extra.nl*` = { key, data, error, busy } (the key says what it was computed for); a quick second request is never overwritten by an older answer (a sequence number
// per kind), the old answer stays on the screen (dimmed) while a new one is computed, and an error is the server's own plain sentence.  The choices are remembered with the project.
import { markRaw, watch } from 'vue';
import { post, put } from '/static/js/api.js';
import { store } from '/static/js/store.js';
import { debounce } from '/static/js/util.js';
import { vec, mat, image, stack } from '/static/js/components/DecompCommon.js';
import { fp, BASE } from './AeState.js';
import { responseBody, bendsBody, localBody, pairBody, savedNl, nlDefault, SHOWN } from './AeNlState.js';

const enc = encodeURIComponent;
const mapObj = (o, f) => (o ? Object.fromEntries(Object.entries(o).map(([k, v]) => [k, f(v)])) : o);

// ---------------------------------------------------------------- remembering the choices with the project
const saveNow = () => {
  const A = fp.ae, p = store.project;
  if (!A || !A.nl || !p || !p.has_data) return;
  const payload = savedNl(A.nl), ui = (p.state && p.state.ui) || {}, was = ui.autoencoders && ui.autoencoders.nl;
  if (!was && JSON.stringify(payload) === JSON.stringify(savedNl(nlDefault()))) return;              // nothing was chosen: leave the project alone
  if (was && JSON.stringify(was) === JSON.stringify(payload)) return;
  if (p.state) p.state.ui = { ...ui, autoencoders: { ...(ui.autoencoders || {}), nl: payload } };
  put('/api/state/ui', { autoencoders: { nl: payload } }, { quiet: true }).catch(() => {});
};
const save = debounce(saveNow, 900);
watch(() => (fp.ae && fp.ae.nl ? JSON.stringify(fp.ae.nl) : ''), (s) => { if (s) save(); });
if (typeof window !== 'undefined') window.addEventListener('pagehide', () => save.flush && save.flush());

// ---------------------------------------------------------------- one loader per kind of answer
function loader(name, route, bodyOf, prepare, failText) {
  let seq = 0;
  return async function load(force = false) {
    const A = fp.ae, rid = A.resRid;
    if (!rid || !A.res) return null;
    const body = bodyOf(A);
    if (!body) return null;
    const key = `${rid}|${JSON.stringify(body)}`;
    const cur = A.extra[name];
    if (!force && cur && cur.key === key && (cur.data || cur.busy)) return cur;
    const my = ++seq, prev = A.extra[name];
    A.extra[name] = { key, data: prev ? prev.data : null, error: '', busy: true };
    const slot = A.extra[name];                                   // (read back: the reactive version, so that the page sees the answer arrive)
    try { const r = await post(`${BASE}/${enc(rid)}/${route}`, body, { quiet: true }); if (my === seq) slot.data = markRaw(prepare(r)); }
    catch (e) { if (my === seq) { slot.error = e.message || failText; slot.data = null; } }
    if (my === seq) slot.busy = false;
    return slot;
  };
}

/** The columns of a matrix {data, n, k} as Float64Arrays, and its rows as views. */
const colsOf = (m) => (m ? Array.from({ length: m.k }, (_, j) => { const o = new Float64Array(m.n); for (let i = 0; i < m.n; i++) o[i] = m.data[i * m.k + j]; return o; }) : []);
const rowsOf = (v) => { const M = mat(v); return Array.from({ length: M.n }, (_, i) => M.data.subarray(i * M.k, (i + 1) * M.k)); };

function prepareResponse(r) {
  const o = { ...r, t: vec(r.t), mean: vec(r.mean), sd: vec(r.sd) };
  o.codes = r.codes.map((c) => ({
    ...c,
    curves: mapObj(c.curves, (cv) => ({ ...cv, yv: vec(cv.y) })),
    ice: c.ice ? { samples: c.ice.samples, x: vec(c.ice.x), y: mapObj(c.ice.y, rowsOf) } : null,
  }));
  if (r.overlay) {
    const per = new Set(r.features.filter((f) => f.per_code).map((f) => f.id));                  // the feature of a pixel that differs from code to code has one row of samples per code
    o.overlay = { ids: r.overlay.ids, x: colsOf(mat(r.overlay.x)), y: Object.fromEntries(Object.entries(r.overlay.y).map(([fid, v]) => [fid, per.has(fid) ? rowsOf(v) : vec(v)])) };
  }
  return o;
}

function prepareBends(r) {
  const o = { ...r, share: vec(r.share), effectShare: vec(r.effect_share), t: vec(r.t), sdv: vec(r.sd) };
  if (r.fluorescence) o.fl = { nl: stack(r.fluorescence.nl), amp: stack(r.fluorescence.amplitude), share: vec(r.fluorescence.share), nEffect: vec(r.fluorescence.n_effect), maxAmp: vec(r.fluorescence.max_amplitude), ex: vec(r.ex), em: vec(r.em) };
  if (r.absorbance) o.ab = { x: vec(r.absorbance.x), nl: mat(r.absorbance.nl), amp: mat(r.absorbance.amplitude), share: vec(r.absorbance.share), maxAmp: vec(r.absorbance.max_amplitude) };
  return o;
}

function prepareLocal(r) {
  const o = { ...r };
  if (r.fluorescence) o.fl = { atMean: image(r.fluorescence.at_mean), samples: r.fluorescence.samples ? stack(r.fluorescence.samples) : null, variability: image(r.fluorescence.variability) };
  if (r.absorbance) o.ab = { x: vec(r.absorbance.x), atMean: vec(r.absorbance.at_mean), samples: r.absorbance.samples ? mat(r.absorbance.samples) : null, variability: vec(r.absorbance.variability) };
  o.changeV = mapObj(r.change, vec);
  return o;
}

function preparePair(r) {
  const o = { ...r, u: vec(r.u) };
  o.fl = r.gallery.fluorescence ? { images: stack(r.gallery.fluorescence.images), interaction: image(r.interaction.fluorescence) } : null;
  o.ab = r.gallery.absorbance ? { x: vec(r.gallery.absorbance.x), series: mat(r.gallery.absorbance.series), interaction: vec(r.interaction.absorbance) } : null;
  o.heatV = { x: vec(r.heat.x), y: vec(r.heat.y), z: mat(r.heat.z) };
  o.dotsV = { ids: r.dots.ids, x: vec(r.dots.x), y: vec(r.dots.y), value: r.dots.value ? vec(r.dots.value) : null };
  return o;
}

export const codesShown = () => (fp.ae.res ? (fp.ae.nl.showAll ? fp.ae.res.d : Math.min(fp.ae.res.d, SHOWN)) : 0);

export const loadResponse = loader('nlResp', 'response', (A) => responseBody(A.nl, codesShown(), A.res.ids), prepareResponse, 'The curves could not be drawn.');
export const loadBends = loader('nlBends', 'bends', (A) => bendsBody(A.nl), prepareBends, 'The maps could not be made.');
export const loadLocal = loader('nlLocal', 'local', (A) => (A.nl.local.code <= A.res.d ? localBody(A.nl) : null), prepareLocal, 'The local effects could not be computed.');
export const loadPair = loader('nlPair', 'pair', (A) => (A.res.d >= 2 && !(A.nl.pair.i && A.nl.pair.i === A.nl.pair.j) ? pairBody(A.nl) : null), preparePair, 'The two codes could not be looked at together.');
