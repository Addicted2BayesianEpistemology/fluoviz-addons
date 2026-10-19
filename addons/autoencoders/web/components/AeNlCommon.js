// Pieces shared by the sections of the tab "Nonlinear effects": how a feature is drawn (colour + line + marker, never colour alone), the legend, and the editor in which the person
// chooses what to measure (a pixel, a box, the overall intensity, a wavelength of the absorbance, a coefficient of the network), by typing or by clicking the picture of the mean EEM.
import { ref, computed, watch } from 'vue';
import { Icon } from '/static/js/icons.js';
import { Field, NumField, Callout } from '/static/js/components/Common.js';
import { EEMView } from '/static/js/components/EEMView.js';
import { image, stack, sliceImage, vec, maskArr } from '/static/js/components/DecompCommon.js';
import { tokens } from '/static/js/theme.js';
import { fmt } from '/static/js/util.js';
import { fp } from './AeState.js';

// ---------------------------------------------------------------- how a feature looks
export const DASHES = ['solid', 'dash', 'dot', 'dashdot', 'longdash', 'longdashdot'];
export const SYMBOLS = ['circle', 'square', 'diamond', 'triangle-up', 'cross', 'x'];
const SVG_DASH = { solid: '', dash: '7 4', dot: '2 3', dashdot: '7 3 2 3', longdash: '11 4', longdashdot: '11 3 2 3' };
/** The colour, line and marker of the k-th feature (same everywhere: chips, legend, charts). */
export function featureStyle(k, t = tokens()) {
  return { color: t.series[k % t.series.length], dash: DASHES[k % DASHES.length], symbol: SYMBOLS[k % SYMBOLS.length] };
}
export const sdText = (v) => (Math.abs(v) < 0.05 ? '0 SD' : `${v > 0 ? '+' : '−'}${Math.abs(v).toFixed(1)} SD`);
/** A share as a percentage; 'under 1 %' for a small one that is not zero ("0 %" would read as exactly none). */
export const pctText = (x, d = 0) => (Number.isFinite(x) ? (d === 0 && x > 0 && x < 0.005 ? 'under 1 %' : `${(100 * x).toFixed(d)} %`) : '–');

/** A little drawing of the line and the marker of a feature (SVG, with the theme's colours). */
export const AeFeatureSwatch = {
  name: 'AeFeatureSwatch',
  props: { k: { type: Number, default: 0 } },
  setup(props) {
    const sw = computed(() => { const s = featureStyle(props.k); return { ...s, dasharray: SVG_DASH[s.dash] }; });
    return { sw };
  },
  template: `<svg class="ae-swatch" width="34" height="14" viewBox="0 0 34 14" aria-hidden="true"><line x1="1" x2="33" y1="7" y2="7" :stroke="sw.color" stroke-width="2.4" :stroke-dasharray="sw.dasharray"/>
    <g :fill="sw.color" :stroke="sw.color"><circle v-if="sw.symbol === 'circle'" cx="17" cy="7" r="3.6"/><rect v-else-if="sw.symbol === 'square'" x="13.4" y="3.4" width="7.2" height="7.2"/>
    <polygon v-else-if="sw.symbol === 'diamond'" points="17,2.6 21.4,7 17,11.4 12.6,7"/><polygon v-else-if="sw.symbol === 'triangle-up'" points="17,3 21,10.6 13,10.6"/>
    <path v-else-if="sw.symbol === 'cross'" d="M17 3 V11 M13 7 H21" fill="none" stroke-width="2"/><path v-else d="M13.6 3.6 L20.4 10.4 M20.4 3.6 L13.6 10.4" fill="none" stroke-width="2"/></g></svg>`,
};

/** The legend of the features of the charts below it. */
export const AeNlLegend = {
  name: 'AeNlLegend',
  components: { AeFeatureSwatch },
  props: { features: { type: Array, required: true }, extra: { type: Array, default: () => [] } },
  template: `
  <div class="ae-nl-legend" data-ae-nl-legend>
    <span class="ae-nl-legend-item" v-for="(f, k) in features" :key="f.id" :data-ae-nl-legend-item="f.id"><AeFeatureSwatch :k="k"/><span>{{ f.label }}<span class="muted" v-if="f.per_code"> (a different pixel for every code: see under each chart)</span></span></span>
    <span class="ae-nl-legend-item muted" v-for="e in extra" :key="e">{{ e }}</span>
  </div>`,
};

// ---------------------------------------------------------------- the network, as far as the choice of a feature is concerned
export function networkFacts(R) {
  const blocks = R && R.p && R.p.input && R.p.input.blocks ? R.p.input.blocks : [];
  const fl = blocks.find((b) => b.name === 'fluorescence') || null, ab = blocks.find((b) => b.name === 'absorbance') || null;
  return { blocks, fl, ab, functional: blocks.filter((b) => b.basis && b.basis !== 'pixels') };
}
/** Where the mean EEM peaks, in nm: a sensible place to start from. */
function peakOf(img, ex, em, mask) {
  if (!img || !img.shape) return null;
  const [nex, nem] = img.shape; let best = -Infinity, bi = 0, bj = 0;
  for (let i = 0; i < nex; i++) for (let j = 0; j < nem; j++) { const v = img[i * nem + j]; if (mask && mask[i * nem + j]) continue; if (v === v && v > best) { best = v; bi = i; bj = j; } }
  return best === -Infinity ? null : { ex: ex[bi], em: em[bj] };
}

export const KIND_LABEL = {
  moved: 'The pixel each code moves most', point: 'Intensity at a pixel', region_mean: 'Mean intensity in a box', region_max: 'Highest intensity in a box', total: 'Overall intensity',
  absorbance: 'Absorbance at a wavelength', coordinate: 'A coefficient of the network',
};
const HELP_KIND = {
  moved: 'The intensity at the pixel that this code changes most (a different pixel for every code).', point: 'The intensity at one pixel of the EEM: type its wavelengths or click it on the picture.',
  region_mean: 'The average intensity of the pixels inside a box of wavelengths.', region_max: 'The highest intensity among the pixels inside a box of wavelengths.',
  total: 'The average intensity over the whole EEM (or the average absorbance over all wavelengths).', absorbance: 'The absorbance at one wavelength.',
  coordinate: 'One of the numbers the network works with for a sample (the coefficient of one term of the basis of a functional network).',
};

/** The editor of what to measure.  `mode` 'list': the chips of the current features (removable) and a form that adds one; 'single': only the form (it emits the choice).
 *  Emits `add(spec)` and `remove(index)`. */
export const AeNlFeatureEditor = {
  name: 'AeNlFeatureEditor',
  components: { Icon, Field, NumField, Callout, EEMView, AeFeatureSwatch },
  props: { features: { type: Array, default: () => [] }, mode: { type: String, default: 'list' }, error: { type: String, default: '' }, current: { type: Object, default: null }, offerMoved: { type: Boolean, default: false },
           addLabel: { type: String, default: 'Add this feature' } },
  emits: ['add', 'remove'],
  setup(props, { emit }) {
    const R = computed(() => fp.ae.res);
    const facts = computed(() => networkFacts(R.value));
    const kinds = computed(() => {
      const f = facts.value, out = [];
      if (props.offerMoved) out.push('moved');
      if (f.fl) out.push('point', 'region_mean', 'region_max');
      out.push('total');
      if (f.ab) out.push('absorbance');
      if (f.functional.length) out.push('coordinate');
      return out;
    });
    const form = ref({ kind: 'total', block: null, ex: null, em: null, ex2: null, em2: null, wavelength: null, k: 1 });
    const picture = computed(() => {
      const r = R.value; if (!r || !r.p.images || !r.p.ex) return null;
      const mean = sliceImage(stack(r.p.images.mean), 0), ex = vec(r.p.ex), em = vec(r.p.em), mask = r.p.mask ? maskArr(r.p.mask) : null;
      return { z: mean, ex, em, mask, units: String(r.p.units || ''), peak: peakOf(mean, ex, em, mask) };
    });
    // sensible starting numbers
    watch([picture, () => facts.value.ab], () => {
      const p = picture.value, f = form.value;
      if (p && p.peak && f.ex === null) { f.ex = Math.round(p.peak.ex); f.em = Math.round(p.peak.em); f.ex2 = Math.round(p.peak.ex + 20); f.em2 = Math.round(p.peak.em + 30); }
      if (facts.value.ab && f.wavelength === null && R.value && R.value.p.curves) { const x = R.value.p.curves.x; f.wavelength = Math.round(x[Math.floor(x.length / 3)]); }
    }, { immediate: true });
    watch(kinds, (k) => { if (!k.includes(form.value.kind)) form.value.kind = k[0]; }, { immediate: true });
    const blockChoices = computed(() => {
      const f = form.value, fa = facts.value;
      if (f.kind === 'total') return fa.blocks.map((b) => b.name);
      if (f.kind === 'coordinate') return fa.functional.map((b) => b.name);
      return [];
    });
    watch([blockChoices, () => form.value.kind], () => { const c = blockChoices.value; if (c.length && !c.includes(form.value.block)) form.value.block = c[0]; }, { immediate: true });
    const coefCount = computed(() => { const b = facts.value.blocks.find((x) => x.name === (form.value.block || (facts.value.functional[0] || {}).name)); return b ? b.n_inputs : 1; });
    const spec = computed(() => {
      const f = form.value;
      if (f.kind === 'moved') return { kind: 'moved' };
      if (f.kind === 'point') return { kind: 'point', ex: f.ex, em: f.em };
      if (f.kind === 'region_mean' || f.kind === 'region_max') return { kind: f.kind, ex: [Math.min(f.ex, f.ex2), Math.max(f.ex, f.ex2)], em: [Math.min(f.em, f.em2), Math.max(f.em, f.em2)] };
      if (f.kind === 'total') return facts.value.blocks.length > 1 ? { kind: 'total', block: f.block } : { kind: 'total' };
      if (f.kind === 'absorbance') return { kind: 'absorbance', wavelength: f.wavelength };
      return { kind: 'coordinate', k: f.k, ...(facts.value.functional.length > 1 ? { block: f.block } : {}) };
    });
    const missing = computed(() => {
      const f = form.value;
      const nums = f.kind === 'point' ? [f.ex, f.em] : f.kind === 'region_mean' || f.kind === 'region_max' ? [f.ex, f.ex2, f.em, f.em2] : f.kind === 'absorbance' ? [f.wavelength] : f.kind === 'coordinate' ? [f.k] : [];
      return nums.some((v) => !Number.isFinite(v)) ? 'Type the numbers first.' : '';
    });
    function add() { if (!missing.value) emit('add', spec.value); }
    // the picture: a click fills the pixel, a dragged box fills the box
    const show = ref(false);
    function onClick(e) { if (!e || !Number.isFinite(e.ex)) return; const f = form.value; f.ex = Math.round(e.ex); f.em = Math.round(e.em); if (!['point', 'region_mean', 'region_max'].includes(f.kind)) f.kind = 'point'; }
    function onSelect(e) { if (!e) return; const f = form.value; f.ex = Math.round(e.ex[0]); f.ex2 = Math.round(e.ex[1]); f.em = Math.round(e.em[0]); f.em2 = Math.round(e.em[1]); if (f.kind !== 'region_mean' && f.kind !== 'region_max') f.kind = 'region_mean'; }
    const marks = computed(() => {
      const pts = [], regs = [];
      props.features.forEach((f, k) => {
        const w = f.where || {};
        if (f.kind === 'point') pts.push({ ex: w.ex, em: w.em, label: `f${k + 1}` });
        if (f.kind === 'region_mean' || f.kind === 'region_max') regs.push({ ex: w.ex, em: w.em, label: `f${k + 1}` });
      });
      const f = form.value;
      if (f.kind === 'point' && Number.isFinite(f.ex) && Number.isFinite(f.em)) pts.push({ ex: f.ex, em: f.em, label: 'new' });
      if ((f.kind === 'region_mean' || f.kind === 'region_max') && [f.ex, f.ex2, f.em, f.em2].every(Number.isFinite)) regs.push({ ex: [Math.min(f.ex, f.ex2), Math.max(f.ex, f.ex2)], em: [Math.min(f.em, f.em2), Math.max(f.em, f.em2)], label: 'new' });
      return { pts, regs };
    });
    const isBox = computed(() => form.value.kind === 'region_mean' || form.value.kind === 'region_max');
    return { R, facts, kinds, form, spec, missing, add, show, onClick, onSelect, picture, marks, blockChoices, coefCount, isBox, KIND_LABEL, HELP_KIND, fmt, fp };
  },
  template: `
  <div class="col gap-8" data-ae-nl-features>
    <div class="row wrap ae-nl-chips" v-if="mode === 'list'">
      <span class="chip ae-nl-chip" v-for="(f, k) in features" :key="f.id" :data-ae-nl-feature="f.id" :title="f.label"><AeFeatureSwatch :k="k"/>{{ f.label }}
        <span class="x" role="button" tabindex="0" :aria-label="'Remove ' + f.label" data-ae-nl-feature-remove :aria-disabled="features.length < 2 ? 'true' : null" @click="features.length > 1 && $emit('remove', k)" @keydown.enter="features.length > 1 && $emit('remove', k)">×</span></span>
    </div>
    <div class="row wrap" style="gap:10px 14px;align-items:flex-end">
      <Field :label="mode === 'list' ? 'Add a feature' : 'What to measure'" help="What is read from the decoded picture of each position: one number per picture, drawn against the code. You can draw several at once."><select class="select sm" v-model="form.kind" data-ae-nl-kind>
        <option v-for="k in kinds" :key="k" :value="k">{{ KIND_LABEL[k] }}</option></select></Field>
      <Field v-if="blockChoices.length > 1" :label="form.kind === 'total' ? 'Of the' : 'In the'" help="The network models both the fluorescence and the absorbance: choose which one."><select class="select sm" style="min-width:132px" v-model="form.block" data-ae-nl-block>
        <option v-for="b in blockChoices" :key="b" :value="b">{{ b }}</option></select></Field>
      <template v-if="form.kind === 'point' || isBox">
        <Field :label="isBox ? 'Excitation from (nm)' : 'Excitation (nm)'" help="An excitation wavelength of the EEM, in nm. The nearest pixel of the model is used."><NumField v-model="form.ex" :step="1" small data-ae-nl-ex/></Field>
        <Field v-if="isBox" label="to (nm)" help="The other side of the box, in nm."><NumField v-model="form.ex2" :step="1" small data-ae-nl-ex2/></Field>
        <Field :label="isBox ? 'Emission from (nm)' : 'Emission (nm)'" help="An emission wavelength of the EEM, in nm. The nearest pixel of the model is used."><NumField v-model="form.em" :step="1" small data-ae-nl-em/></Field>
        <Field v-if="isBox" label="to (nm)" help="The other side of the box, in nm."><NumField v-model="form.em2" :step="1" small data-ae-nl-em2/></Field>
      </template>
      <Field v-if="form.kind === 'absorbance'" label="Wavelength (nm)" help="A wavelength of the absorbance spectra, in nm."><NumField v-model="form.wavelength" :step="1" small data-ae-nl-wavelength/></Field>
      <Field v-if="form.kind === 'coordinate'" :label="'Number of the coefficient (1 to ' + coefCount + ')'" help="Which of the numbers of the basis: the first ones hold the largest part of the variation of the samples."><NumField v-model="form.k" :min="1" :max="coefCount" :step="1" small data-ae-nl-k/></Field>
      <button type="button" class="btn sm" data-ae-nl-feature-add :disabled="!!missing" :title="missing || HELP_KIND[form.kind]" @click="add"><Icon name="plus" size="15"/>{{ addLabel }}</button>
    </div>
    <p class="fp-note muted" v-if="HELP_KIND[form.kind]">{{ HELP_KIND[form.kind] }}</p>
    <Callout kind="warning" v-if="error" data-ae-nl-feature-error>{{ error }}</Callout>
    <details class="ae-nl-pick" v-if="picture && kinds.includes('point')" :open="show" @toggle="show = $event.target.open" data-ae-nl-pick>
      <summary>Pick a pixel or a box on the picture of the mean EEM</summary>
      <p class="fp-note muted">Click a pixel to read its wavelengths, or choose <b>Select region</b> and drag a box. The numbers above are filled in; then press <b>{{ addLabel }}</b>. Pixels already chosen are marked.</p>
      <div v-if="show" data-ae-nl-picture><EEMView :ex="picture.ex" :em="picture.em" :z="picture.z" :mask="picture.mask" :units="picture.units" marginals="none" :height="300" :points="marks.pts" :regions="marks.regs" @click="onClick" @select="onSelect"/></div>
    </details>
  </div>`,
};
