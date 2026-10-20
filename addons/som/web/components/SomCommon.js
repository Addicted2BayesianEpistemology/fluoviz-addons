// Small pieces shared by the steps of the Self-organising map page: plain-wording helpers, a number card, a verdict notice, a titled block, and the colours of the map.
import { ref, computed, onMounted, onBeforeUnmount } from 'vue';
import { Icon } from '/static/js/icons.js';
import { HelpTip } from '/static/js/components/Common.js';
import { tokens, lutFor, sequentialScale, divergingScale, categoryStyle, registerCategories } from '/static/js/theme.js';
import { prefs } from '/static/js/prefs.js';
import { fmt } from '/static/js/util.js';

export { fmt, Icon };

export const pct = (x, d = 1) => (Number.isFinite(x) ? `${(100 * x).toFixed(d)} %` : '–');
export const nTxt = (n) => (Number.isFinite(n) ? String(Math.round(n)).replace(/\B(?=(\d{3})+(?!\d))/g, ' ') : '–');
export const plural = (n, one, many) => `${nTxt(n)} ${n === 1 ? one : (many || `${one}s`)}`;
export const listWords = (a) => (a.length <= 1 ? a.join('') : `${a.slice(0, -1).join(', ')} and ${a[a.length - 1]}`);
export const STATUS_LABEL = { ok: 'Fits the map', uncertain: 'On a unit no sample reached', outside: 'Poorly described by the map', unusable: 'Cannot be used' };
export const num = (x, d = 3) => (x === null || x === undefined || Number.isNaN(x) ? '–' : fmt(x, d));

/** True while the window is narrow (phones, split screens). */
export function useNarrow(px = 760) {
  const narrow = ref(typeof window !== 'undefined' && window.matchMedia ? window.matchMedia(`(max-width: ${px}px)`).matches : false);
  let mq = null;
  const on = (e) => { narrow.value = e.matches; };
  onMounted(() => { if (window.matchMedia) { mq = window.matchMedia(`(max-width: ${px}px)`); narrow.value = mq.matches; mq.addEventListener('change', on); } });
  onBeforeUnmount(() => { if (mq) mq.removeEventListener('change', on); });
  return narrow;
}

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

// ---------------------------------------------------------------- one number
export const SomStat = {
  name: 'SomStat',
  props: { label: String, value: [String, Number], sub: { type: String, default: '' }, level: { type: String, default: '' }, help: { type: String, default: '' } },
  components: { HelpTip },
  template: `
  <div class="fp-stat som-stat" :data-level="level">
    <span class="l">{{ label }}<HelpTip v-if="help" :text="help"/></span>
    <span class="v" :class="level">{{ value }}</span>
    <span class="s" v-if="sub">{{ sub }}</span>
  </div>`,
};

/** A verdict in words: the server's sentence, with the colour of how good it is.  level: good | info | warn | warning | critical */
export const SomVerdict = {
  name: 'SomVerdict',
  components: { Icon },
  props: { level: { type: String, default: 'info' } },
  setup(props) { return { kind: computed(() => (props.level === 'good' ? 'good' : props.level === 'warn' || props.level === 'warning' ? 'warning' : props.level === 'critical' ? 'critical' : 'info')) }; },
  template: `<div class="callout" :class="kind" data-som-verdict :data-level="level"><Icon :name="kind === 'good' ? 'check-circle' : kind === 'info' ? 'info-circle' : 'exclamation-triangle'"/><div class="grow"><slot/></div></div>`,
};

/** A block of a step: a heading, one sentence about what it is, then the controls. */
export const SomSection = {
  name: 'SomSection',
  components: { HelpTip },
  props: { title: String, sentence: { type: String, default: '' }, help: { type: String, default: '' } },
  template: `<section class="som-section"><h3 class="fp-h3">{{ title }}<HelpTip v-if="help" :text="help"/></h3><p class="fp-note" v-if="sentence">{{ sentence }}</p><slot/></section>`,
};
