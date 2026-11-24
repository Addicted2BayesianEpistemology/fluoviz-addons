// Small building blocks of the Help guide of the add-on "Multimodal latent factors": the interface word, the links, and schematic pictures drawn with the theme's colours.
import { Callout } from '/static/js/components/Common.js';
import { helpParts } from '/addon-web/common/web/help.js';

/** The Help tab of this guide (the address is  #/help?tab=addon:mblf:mblf&topic=<page>). */
export const TAB = 'addon:mblf:mblf';

const { Ui, GLink, PLink } = helpParts(TAB);
export { Ui, GLink, PLink };

const T = 'font-family:var(--font);font-size:11px;fill:var(--text-3)';
const TB = 'font-family:var(--font);font-size:12px;font-weight:700;fill:var(--text)';

/** A figure frame: the picture and its caption. */
export const Fig = { name: 'Fig', template: '<figure class="g-fig"><slot/><figcaption><slot name="caption"/></figcaption></figure>' };

/** The structure of the model: the shared factors feed every mode; every mode adds its own part and noise; covariates shift the factors. */
export const FigModel = {
  name: 'FigModel',
  setup() { return { T, TB }; },
  template: `
  <svg viewBox="0 0 560 200" width="100%" style="max-width:560px" role="img" aria-label="The structure of the model">
    <rect x="20" y="78" width="96" height="44" rx="8" style="fill:var(--surface);stroke:var(--line-strong)"/><text x="68" y="98" text-anchor="middle" :style="TB">covariates</text><text x="68" y="113" text-anchor="middle" :style="T">site, season …</text>
    <line x1="116" y1="100" x2="170" y2="100" style="stroke:var(--text-3);stroke-width:1.5" marker-end="url(#mbA)"/>
    <circle cx="210" cy="100" r="36" style="fill:color-mix(in srgb, var(--accent) 18%, var(--surface));stroke:var(--accent);stroke-width:2"/><text x="210" y="97" text-anchor="middle" :style="TB">shared</text><text x="210" y="112" text-anchor="middle" :style="TB">factors</text>
    <g v-for="(m, i) in ['fluorescence (EEM)', 'absorbance', 'values']" :key="m">
      <line x1="246" :y1="100" x2="356" :y2="40 + 60 * i" style="stroke:var(--accent);stroke-width:1.5" marker-end="url(#mbA)"/>
      <rect x="360" :y="20 + 60 * i" width="150" height="40" rx="8" style="fill:var(--surface);stroke:var(--line-strong)"/><text x="435" :y="38 + 60 * i" text-anchor="middle" :style="TB">{{ m }}</text>
      <text x="435" :y="52 + 60 * i" text-anchor="middle" :style="T">+ its own part + noise</text>
    </g>
    <defs><marker id="mbA" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--text-3)"/></marker></defs>
  </svg>`,
};

/** The uncertainty budget: a stacked bar with the four parts. */
export const FigBudget = {
  name: 'FigBudget',
  setup() {
    const parts = [{ w: 0.12, c: 'var(--series-4)', t: 'parameters' }, { w: 0.38, c: 'var(--series-1)', t: 'scores' }, { w: 0.4, c: 'var(--series-2)', t: 'own variation' }, { w: 0.1, c: 'var(--series-3)', t: 'noise' }];
    let x = 20;
    const bars = parts.map((p) => { const o = { ...p, x, width: 520 * p.w }; x += 520 * p.w; return o; });
    return { bars, T, TB };
  },
  template: `
  <svg viewBox="0 0 560 90" width="100%" style="max-width:560px" role="img" aria-label="The four parts of the predictive variance">
    <rect v-for="b in bars" :key="b.t" :x="b.x" y="20" :width="b.width - 2" height="30" rx="4" :style="'fill:' + b.c"/>
    <text v-for="b in bars" :key="b.t + 't'" :x="b.x + b.width / 2" y="68" text-anchor="middle" :style="T">{{ b.t }}</text>
    <text x="20" y="12" :style="TB">predictive variance at one wavelength = </text>
  </svg>`,
};

/** A fan chart: the median, the 50 / 80 / 95 % bands and the dashed band for the whole curve. */
export const FigFan = {
  name: 'FigFan',
  setup() {
    const xs = Array.from({ length: 41 }, (_, i) => i);
    const f = (i) => 60 - 32 * Math.exp(-((i - 16) ** 2) / 40) - 14 * Math.exp(-((i - 30) ** 2) / 30);
    const s = (i) => 4 + 6 * Math.exp(-((i - 30) ** 2) / 60);
    const X = (i) => 20 + i * 12.5;
    const band = (k) => xs.map((i) => `${X(i)},${f(i) - k * s(i)}`).join(' ') + ' ' + xs.slice().reverse().map((i) => `${X(i)},${f(i) + k * s(i)}`).join(' ');
    const line = (k) => xs.map((i) => `${X(i)},${f(i) + k * s(i)}`).join(' ');
    return { band, line, T };
  },
  template: `
  <svg viewBox="0 0 560 110" width="100%" style="max-width:560px" role="img" aria-label="A fan chart">
    <polygon :points="band(2)" style="fill:color-mix(in srgb, var(--series-1) 16%, transparent)"/><polygon :points="band(1.3)" style="fill:color-mix(in srgb, var(--series-1) 22%, transparent)"/>
    <polygon :points="band(0.67)" style="fill:color-mix(in srgb, var(--series-1) 30%, transparent)"/><polyline :points="line(0)" style="fill:none;stroke:var(--series-1);stroke-width:2"/>
    <polyline :points="line(2.8)" style="fill:none;stroke:var(--series-1);stroke-dasharray:5 4"/><polyline :points="line(-2.8)" style="fill:none;stroke:var(--series-1);stroke-dasharray:5 4"/>
    <text x="24" y="104" :style="T">dark: 50 %, lighter: 80 % and 95 % at every point; dashed: 95 % for the whole curve at once</text>
  </svg>`,
};

export const parts = { Ui, GLink, PLink, Fig, FigModel, FigBudget, FigFan, Callout };
