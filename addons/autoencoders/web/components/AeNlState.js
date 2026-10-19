// The choices of the tab "Nonlinear effects" (step 7): the section, the range and the number of points of the curves, the features, the switches, the codes of the local and the pair views.
// They are remembered with the project (PUT /api/state/ui, key autoencoders.nl, deep-merged next to the settings of the page) and read back when the page is made again for a project
// (`freshNl`); what the server answered lives in `fp.ae.extra.nl*` (AeNlActions.js).  Nothing here imports the page's state: AeState.js imports this file.
export const SECTIONS = [
  { id: 'response', label: 'Response curves' },
  { id: 'bends', label: 'Where it bends' },
  { id: 'local', label: 'Local effects' },
  { id: 'pair', label: 'Two codes' },
];
export const SHOWN = 8;                       // the codes drawn at first (the others on request)
export const MAX_ICE = 12;
export const LIMITS = { span: [0.5, 4], points: [5, 201], grid: [3, 5, 7], loadings: 6 };

export function nlDefault() {
  return {
    section: 'response', span: 3, n_points: 41,
    features: null,                           // null = the features the page starts with (the pixel each code moves most, the overall intensity); else the list the person built
    overlay: true, ice: false, iceNames: '', relative: true, showAll: false,
    bendsShow: 'nl',                          // nl = where it bends | amplitude = the size of the effect
    local: { code: 1, mode: 'percentiles', names: [] },
    pair: { i: null, j: null, grid: 5, span: 2, feature: null },     // null codes = the two with the largest nonlinear share (the server chooses)
  };
}

const num = (v, lo, hi, dflt) => (typeof v === 'number' && Number.isFinite(v) && v >= lo && v <= hi ? v : dflt);
const bool = (v, dflt) => (typeof v === 'boolean' ? v : dflt);
const spec = (s) => (s && typeof s === 'object' && !Array.isArray(s) && typeof s.kind === 'string' ? { ...s } : null);

/** The saved choices (anything may be wrong in a file written by another version) merged into the defaults: only values of the right kind and range are taken. */
export function restoreNl(saved) {
  const d = nlDefault();
  if (!saved || typeof saved !== 'object') return d;
  const s = saved;
  if (SECTIONS.some((x) => x.id === s.section)) d.section = s.section;
  d.span = num(s.span, LIMITS.span[0], LIMITS.span[1], d.span);
  d.n_points = Number.isInteger(s.n_points) && s.n_points >= LIMITS.points[0] && s.n_points <= LIMITS.points[1] ? s.n_points : d.n_points;
  if (Array.isArray(s.features) && s.features.length && s.features.length <= 6 && s.features.every(spec)) d.features = s.features.map(spec);
  d.overlay = bool(s.overlay, d.overlay); d.ice = bool(s.ice, d.ice); d.relative = bool(s.relative, d.relative); d.showAll = bool(s.showAll, d.showAll);
  if (typeof s.iceNames === 'string') d.iceNames = s.iceNames.slice(0, 400);
  if (s.bendsShow === 'nl' || s.bendsShow === 'amplitude') d.bendsShow = s.bendsShow;
  const l = s.local;
  if (l && typeof l === 'object') {
    if (Number.isInteger(l.code) && l.code >= 1 && l.code <= 32) d.local.code = l.code;
    if (l.mode === 'percentiles' || l.mode === 'chosen') d.local.mode = l.mode;
    if (Array.isArray(l.names)) d.local.names = l.names.filter((x) => typeof x === 'string').slice(0, LIMITS.loadings);
  }
  const p = s.pair;
  if (p && typeof p === 'object') {
    if (Number.isInteger(p.i) && Number.isInteger(p.j) && p.i >= 1 && p.j >= 1 && p.i <= 32 && p.j <= 32 && p.i !== p.j) { d.pair.i = p.i; d.pair.j = p.j; }
    if (LIMITS.grid.includes(p.grid)) d.pair.grid = p.grid;
    d.pair.span = num(p.span, LIMITS.span[0], LIMITS.span[1], d.pair.span);
    d.pair.feature = spec(p.feature);
  }
  return d;
}

/** The choices of the project that is open now (`store.project.state.ui.autoencoders.nl`). */
export function freshNl(project) {
  const ui = project && project.state && project.state.ui ? project.state.ui.autoencoders : null;
  return restoreNl(ui && ui.nl);
}

/** What is written to the project: a plain copy (no reactive proxies). */
export const savedNl = (nl) => JSON.parse(JSON.stringify(nl));

// ---------------------------------------------------------------- the requests
export function parseNames(text) {
  return String(text || '').split(/[,;\n]+/).map((x) => x.trim()).filter(Boolean);
}
/** The body of POST /{rid}/response.  `codesShown` = how many codes are drawn (they get the curves of single samples); `ids` = the names of the samples of the network. */
export function responseBody(nl, codesShown, ids) {
  const body = { span: nl.span, n_points: nl.n_points, overlay: nl.overlay };
  if (nl.features) body.features = nl.features;
  if (nl.ice) {
    const names = parseNames(nl.iceNames).filter((x) => ids.includes(x));
    const ice = { codes: Array.from({ length: codesShown }, (_, k) => k + 1) };
    if (names.length) ice.samples = names.slice(0, MAX_ICE); else ice.n = MAX_ICE;
    body.ice = ice;
  }
  return body;
}
export const bendsBody = (nl) => ({ span: nl.span, n_points: nl.n_points });
export function localBody(nl) {
  const body = { code: nl.local.code };
  if (nl.local.mode === 'chosen' && nl.local.names.length) body.samples = nl.local.names.slice(0, LIMITS.loadings);
  return body;
}
export function pairBody(nl) {
  const p = nl.pair, body = { grid: p.grid, span: p.span };
  if (p.i && p.j) body.codes = [p.i, p.j];
  if (p.feature) body.feature = p.feature;
  return body;
}
