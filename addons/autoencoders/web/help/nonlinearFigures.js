// The two pictures of the guide page "Nonlinear effects of the codes" (Help).  Drawn in SVG with the colours of the theme and from numbers that the add-on's own code computed (data.js:
// tests/help_data.py writes them, tests/test_help_autoencoders.py recomputes them and compares).  What a colour means is also said by a line style, a marker or a word: never by colour alone.
import { computed } from 'vue';
import { Heat, S, GRID, fmt, pct, signed, transposed } from './parts.js';
import DATA from './data.js';

const MINUS = '−';
/** A text broken into lines of at most `n` characters (SVG does not wrap). */
export function wrap(text, n) {
  const out = []; let line = '';
  for (const w of String(text).split(' ')) { if (line && (line + ' ' + w).length > n) { out.push(line); line = w; } else line = line ? `${line} ${w}` : w; }
  if (line) out.push(line);
  return out;
}
const pathOf = (xs, ys, px, py) => xs.map((x, i) => `${i ? 'L' : 'M'}${px(x).toFixed(1)},${py(ys[i]).toFixed(1)}`).join(' ');

// ------------------------------------------------------------------------------------------------------------------------------- 8. the vocabulary: eight response curves in words
export const FigShapes = {
  name: 'FigShapes',
  setup() {
    const D = DATA.shapes, t = D.t;
    const W = 150, H = 92, X0 = 22, DX = 172, Y0 = 34, DY = 168;
    const panels = D.curves.map((c, i) => {
      const x0 = X0 + (i % 4) * DX, y0 = Y0 + Math.floor(i / 4) * DY;
      const px = (v) => x0 + ((v + 3) / 6) * W, py = (v) => y0 + H - 4 - v * (H - 8);
      return { ...c, x0, y0, px, py, curve: pathOf(t, c.y, px, py), line: pathOf(t, c.line, px, py), lines: wrap(c.label.replace(/\s*\(.*?\)/, ''), 25), mid: px(0) };
    });
    return { ...S, panels, W, H, fmt, pct, MINUS };
  },
  template: `
  <figure class="sheet-fig">
    <svg viewBox="0 0 720 372" style="width:100%;max-width:760px;height:auto;display:block" role="img" aria-label="Eight response curves of a code: a straight line, bends upwards and downwards, levelling off, an S, a peak, a valley and a wave, each with the words the page uses and what a straight line explains of it">
      <g v-for="(p, i) in panels" :key="i">
        <text :x="p.x0" :y="p.y0 - 10" :style="TB">{{ p.name }}</text>
        <rect :x="p.x0" :y="p.y0" :width="W" :height="H" :style="AXIS"/>
        <line :x1="p.mid" :x2="p.mid" :y1="p.y0" :y2="p.y0 + H" style="stroke:var(--line);stroke-width:1;stroke-dasharray:2 3"/>
        <clipPath :id="'shapes-clip-' + i"><rect :x="p.x0" :y="p.y0" :width="W" :height="H"/></clipPath>
        <g :clip-path="'url(#shapes-clip-' + i + ')'">
          <path :d="p.line" style="fill:none;stroke:var(--text-3);stroke-width:1.4;stroke-dasharray:5 3"/>
          <path :d="p.curve" style="fill:none;stroke:var(--series-1);stroke-width:2.4;stroke-linejoin:round"/>
        </g>
        <text v-for="(l, k) in p.lines" :key="k" :x="p.x0" :y="p.y0 + H + 16 + k * 13" :style="T2">{{ l }}</text>
        <text :x="p.x0" :y="p.y0 + H + 16 + p.lines.length * 13 + 3" :style="TXT">a straight line explains {{ pct(p.r2, 0) }}</text>
      </g>
      <g transform="translate(22,354)">
        <line x1="0" x2="26" y1="0" y2="0" style="stroke:var(--series-1);stroke-width:2.4"/><text x="34" y="4" :style="T2">the response of one measurement to one code</text>
        <line x1="330" x2="356" y1="0" y2="0" style="stroke:var(--text-3);stroke-width:1.4;stroke-dasharray:5 3"/><text x="364" y="4" :style="T2">the best straight line</text>
      </g>
    </svg>
    <figcaption><slot/></figcaption>
  </figure>`,
};

// ------------------------------------------------------------------------------------------------------------------------------- 9. two codes of one network: the response and where it bends
export const FigNonlinear = {
  name: 'FigNonlinear',
  components: { Heat },
  setup() {
    const D = DATA.nonlinear, t = D.t;
    const X0s = [58, 392], W = 252, H = 132, Y0 = 46;
    const cols = D.codes.map((c, j) => {
      const x0 = X0s[j];
      const all = [...c.curve, ...c.line, ...c.dots_y].filter((v) => v !== null);
      const lo = Math.min(...all), hi = Math.max(...all), pad = (hi - lo) * 0.06;
      const px = (v) => x0 + ((v + 3) / 6) * W, py = (v) => Y0 + H - ((v - (lo - pad)) / (hi - lo + 2 * pad)) * H;
      const dots = c.dots_x.map((x, i) => ({ x: px(x), y: py(c.dots_y[i]) })).filter((d) => d.x >= x0 && d.x <= x0 + W);
      const ticks = [lo, (lo + hi) / 2, hi].map((v) => ({ y: py(v), label: v.toFixed(1) }));
      return { ...c, x0, px, py, dots, ticks, curvePath: pathOf(t, c.curve, px, py), linePath: pathOf(t, c.line, px, py), words: wrap(c.label, 36), where: `${c.pixel.ex} / ${c.pixel.em} nm` };
    });
    const where = computed(() => (transposed() ? 'excitation to the right, emission up' : 'emission to the right, excitation up'));
    return { ...S, D, cols, W, H, Y0, t, fmt, pct, signed, MINUS, where, GRID };
  },
  template: `
  <figure class="sheet-fig">
    <svg viewBox="0 0 700 486" style="width:100%;max-width:740px;height:auto;display:block" role="img" aria-label="Two codes of one network: the response of the pixel each code moves most, with the samples around it and the best straight line, and the map of where each code bends">
      <g v-for="c in cols" :key="c.name">
        <text :x="c.x0 - 36" y="14" :style="TB">{{ c.name }}: the pixel it moves most</text>
        <text :x="c.x0 - 36" y="29" :style="TXT">excitation / emission {{ c.where }}; 1 SD of {{ c.name }} = {{ fmt(c.sd, 2) }}</text>
        <g v-for="(k, i) in c.ticks" :key="'g' + i"><line :x1="c.x0" :x2="c.x0 + W" :y1="k.y" :y2="k.y" :style="GRID"/><text :x="c.x0 - 6" :y="k.y + 4" text-anchor="end" :style="TXT">{{ k.label }}</text></g>
        <rect :x="c.x0" :y="Y0" :width="W" :height="H" :style="AXIS"/>
        <g v-for="s in [-3, -2, -1, 0, 1, 2, 3]" :key="'x' + s"><line :x1="c.px(s)" :x2="c.px(s)" :y1="Y0 + H" :y2="Y0 + H + 4" :style="AXIS"/><text :x="c.px(s)" :y="Y0 + H + 16" text-anchor="middle" :style="TXT">{{ s > 0 ? '+' + s : (s < 0 ? MINUS + Math.abs(s) : '0') }}</text></g>
        <text :x="c.x0 + W / 2" :y="Y0 + H + 32" text-anchor="middle" :style="T2">position of {{ c.name }} (SD from its mean)</text>
        <circle v-for="(d, i) in c.dots" :key="'d' + i" :cx="d.x" :cy="d.y" r="2.3" style="fill:var(--text-3);fill-opacity:.55"/>
        <path :d="c.linePath" style="fill:none;stroke:var(--text-2);stroke-width:1.4;stroke-dasharray:5 3"/>
        <path :d="c.curvePath" style="fill:none;stroke:var(--series-1);stroke-width:2.6;stroke-linejoin:round"/>
        <text v-for="(l, k) in c.words" :key="'w' + k" :x="c.x0 - 36" :y="Y0 + H + 54 + k * 14" :style="TB">{{ k === 0 ? c.name + ': ' : '' }}{{ l }}</text>
        <text :x="c.x0 - 36" :y="Y0 + H + 54 + c.words.length * 14" :style="T2">a straight line explains {{ pct(c.r2, 0) }} of it (R² = {{ fmt(c.r2, 2) }})</text>
        <Heat :grid="c.nl" :x="c.x0 + 36" :y="288" :w="180" :h="93" :max="1"/>
        <text :x="c.x0 + 126" y="400" text-anchor="middle" :style="TXT">{{ c.name }}: where it bends</text>
        <text :x="c.x0 + 126" y="415" text-anchor="middle" :style="TB">{{ pct(c.share, 0) }} of its effect is not a straight line</text>
      </g>
      <rect x="58" y="436" width="22" height="11" style="fill:var(--accent);fill-opacity:.9"/><text x="86" y="446" :style="T2">strong colour: bends a lot</text>
      <rect x="252" y="436" width="22" height="11" style="fill:var(--accent);fill-opacity:.12;stroke:var(--accent);stroke-opacity:.4"/><text x="280" y="446" :style="T2">pale: a straight line</text>
      <rect x="424" y="436" width="22" height="11" style="fill:var(--line-strong);fill-opacity:.45;stroke:var(--line-strong)"/><text x="452" y="446" :style="T2">grey: outside the model or no effect</text>
      <text x="58" y="464" :style="TXT">Maps: {{ where }}.</text>
      <text x="58" y="479" :style="TXT">Faint dots: the {{ D.n }} made-up samples; the dashed line is the best straight line.</text>
    </svg>
    <figcaption><slot/></figcaption>
  </figure>`,
};
