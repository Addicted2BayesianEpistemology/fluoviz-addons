// Small building blocks of the Help guide of the add-on "Autoencoders": the way an interface word is written, the links, and the pieces the pictures are made of.
// Nothing here is imported from FluoViz except the shared components with absolute URLs (the icons, the Callout box, the router), which are the stable interface for add-ons.
// The pictures themselves are in figures.js and are drawn from numbers computed with the add-on's own code (data.js: tests/help_data.py writes them, tests/test_help_autoencoders.py recomputes them).
import { Icon } from '/static/js/icons.js';
import { Callout } from '/static/js/components/Common.js';
import { helpParts } from '/addon-web/common/web/help.js';
import { prefs } from '/static/js/prefs.js';

/** Are the EEM pictures of the program drawn with excitation across and emission up?  (Settings > Appearance > Axes of fluorescence maps; the maps of this guide follow it, like every other EEM picture.
 *  A FluoViz that does not have the setting has no `eemTranspose`: the usual layout, emission across.)  Reactive: read it inside a computed property or a template. */
export const transposed = () => !!prefs.eemTranspose;

/** The Help tab of this guide (the address is  #/help?tab=addon:autoencoders:autoencoders&topic=<page>). */
export const TAB = 'addon:autoencoders:autoencoders';

const { Ui, GLink, PLink } = helpParts(TAB);                     // (the shared components, with the address of this guide)
export { Ui, GLink, PLink };

// ------------------------------------------------------------------------------------------------------------------------------- styles of the pictures (theme variables only)
export const T = 'font-family:var(--font);font-size:11px;fill:var(--text-3)';
export const T2 = 'font-family:var(--font);font-size:11.5px;fill:var(--text-2)';
export const TB = 'font-family:var(--font);font-size:12px;font-weight:700;fill:var(--text)';
export const AXIS = 'stroke:var(--line-strong);stroke-width:1;fill:none';
export const GRID = 'stroke:var(--line);stroke-width:1;fill:none';
export const S = { TXT: T, TXT2: T2, T2, TB, AXIS, GRID };

/** "Nice" tick values between `lo` and `hi` (about `n` of them). */
export function niceTicks(lo, hi, n = 3) {
  const raw = (hi - lo) / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((f) => f * mag).find((s) => s >= raw) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9; v += step) out.push(Number(v.toPrecision(6)));
  return out;
}
/** A signed number with a true minus sign, and no sign for a number that rounds to zero. */
export const signed = (v, d = 2) => { const r = Number(Math.abs(v).toFixed(d)); return (v < 0 && r !== 0 ? '\u2212' : '') + r.toFixed(d); };
export const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? '–' : Number(v).toFixed(d));
export const pct = (v, d = 0) => (v === null || v === undefined ? '–' : `${(100 * v).toFixed(d)} %`);
/** An integer with a thin space between the thousands: 10432 -> "10 432". */
export const thousands = (n) => String(Math.round(n)).replace(/\B(?=(\d{3})+(?!\d))/g, ' ');

/** An EEM (or a difference between two) as a small map made of squares.  `grid` is `[row][column]` (rows = excitation from the top = the highest, columns = emission), `null` = a pixel that does not exist (grey).
 *  The map is drawn like every EEM picture of the program: emission across and excitation up, or, with the setting Axes of fluorescence maps, excitation across and emission up (the grid is turned). 
 *  `mode` 'seq': the more intense, the stronger the accent; 'div': positive and negative around 0 with the two series colours.  `max` = the value that is drawn at full strength. */
export const Heat = {
  name: 'Heat',
  props: { grid: { type: Array, required: true }, x: { type: [Number, String], default: 0 }, y: { type: [Number, String], default: 0 }, w: { type: [Number, String], default: 100 }, h: { type: [Number, String], default: 60 },
           mode: { type: String, default: 'seq' }, max: { type: [Number, String], default: 1 } },
  setup(props) {
    const cells = () => {
      const g0 = props.grid, nr0 = g0.length, nc0 = g0[0].length, tp = transposed();
      const g = tp ? Array.from({ length: nc0 }, (_, r) => Array.from({ length: nr0 }, (_, c) => g0[nr0 - 1 - c][nc0 - 1 - r])) : g0;           // excitation across (the lowest first), emission up (the highest on the top)
      const nr = g.length, nc = g[0].length, W = Number(props.w), H = Number(props.h), X = Number(props.x), Y = Number(props.y), cw = W / nc, ch = H / nr, out = [];
      for (let i = 0; i < nr; i++) for (let j = 0; j < nc; j++) {
        const v = g[i][j];
        const o = { x: X + j * cw, y: Y + i * ch, cw: cw + 0.3, ch: ch + 0.3, hole: v === null || v === undefined };
        if (!o.hole) {
          if (props.mode === 'div') { o.neg = v < 0; o.o = Math.min(1, Math.abs(v) / (Number(props.max) || 1)); } else o.o = Math.min(1, Math.pow(Math.max(v / (Number(props.max) || 1), 0), 0.8));
        }
        out.push(o);
      }
      return out;
    };
    return { cells };
  },
  template: `
  <g>
    <rect v-for="(c, k) in cells()" :key="k" :x="c.x" :y="c.y" :width="c.cw" :height="c.ch"
      :style="c.hole ? 'fill:var(--line-strong);fill-opacity:.45' : (c.neg ? 'fill:var(--series-6)' : (mode === 'div' ? 'fill:var(--series-1)' : 'fill:var(--accent)')) + ';fill-opacity:' + c.o.toFixed(3)"/>
    <rect :x="Number(x)" :y="Number(y)" :width="Number(w)" :height="Number(h)" style="fill:none;stroke:var(--line-strong);stroke-width:1"/>
  </g>`,
};

export const parts = { Icon, Callout, GLink, PLink, Ui, Heat };
