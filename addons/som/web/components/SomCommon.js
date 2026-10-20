// Small pieces shared by the steps of the Self-organising map page: plain-wording helpers, a number card, a verdict notice, a titled block, and the colours of the map.
import { Icon } from '/static/js/icons.js';
import { pct, nTxt, plural, listWords, useNarrow } from '/addon-web/common/web/words.js';
import { makeParts } from '/addon-web/common/web/parts.js';
import { tokens, lutFor, sequentialScale, divergingScale, categoryStyle, registerCategories } from '/static/js/theme.js';
import { prefs } from '/static/js/prefs.js';
import { fmt } from '/static/js/util.js';

export { fmt, Icon };

export { pct, nTxt, plural, listWords, useNarrow };
export const STATUS_LABEL = { ok: 'Fits the map', uncertain: 'On a unit no sample reached', outside: 'Poorly described by the map', unusable: 'Cannot be used' };
export const num = (x, d = 3) => (x === null || x === undefined || Number.isNaN(x) ? '–' : fmt(x, d));

// ---------------------------------------------------------------- colours of the map
/** A function value -> 'rgb(r,g,b)' for a numeric plane (sequential, or diverging and symmetric around 0 for a signed one); NaN -> null.  Also the Plotly colour scale and the range. */
export function numericColours(values, signed = false, scaleName = null) {
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < values.length; i++) { const v = values[i]; if (v === v && Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; } }
  if (lo === Infinity) return { rgb: () => null, scale: [[0, '#ccc'], [1, '#ccc']], lo: 0, hi: 1 };
  if (signed) { const m = Math.max(Math.abs(lo), Math.abs(hi)) || 1; lo = -m; hi = m; }
  else if (hi === lo) { hi = lo + 1; }
  const scale = signed ? divergingScale(scaleName || prefs.div) : sequentialScale(scaleName || prefs.seq);
  const lut = lutFor(scale), span = hi - lo;
  const rgb = (v) => { if (!(v === v) || !Number.isFinite(v)) return null; const q = Math.round(Math.min(1, Math.max(0, (v - lo) / span)) * 255) * 3; return `rgb(${lut[q]},${lut[q + 1]},${lut[q + 2]})`; };
  return { rgb, scale, lo, hi };
}
/** The colour of category number `i` of a plane (stable: the series colours of FluoViz, the hue repeats after 8). */
export function categoryColour(key, cats, i) {
  registerCategories(key, cats);
  return categoryStyle(key, cats[i]).color;
}
export const groupColour = (g) => (g >= 0 ? tokens().series[g % 8] : tokens().other);

// ---------------------------------------------------------------- one number, a verdict, a titled block (the shared components, with this page's prefix)
const parts = makeParts({ name: 'Som', prefix: 'som' });
export const SomStat = parts.Stat;
export const SomVerdict = parts.Verdict;
export const SomSection = parts.Section;
