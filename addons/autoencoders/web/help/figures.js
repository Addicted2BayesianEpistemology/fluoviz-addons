// The pictures of the guide "Autoencoders" (Help).  They are drawn in SVG with the colours of the theme (they follow the light / dark switch), and from numbers that the add-on's own code computed
// (data.js: tests/help_data.py writes them, tests/test_help_autoencoders.py recomputes them and compares).  What a colour means is also said by a shape, a dash or a word: never by colour alone.
// SVG does not wrap text: the lines of every picture are short and placed by hand.
import { Heat, S, T, T2, TB, AXIS, GRID, fmt, pct, thousands, niceTicks, signed } from './parts.js';
import DATA from './data.js';

const MINUS = '−';
const sgn = signed;

/** Points of a line as an SVG path; a `null` value breaks the line. */
function path(xs, ys, px, py) {
  let d = '', pen = false;
  xs.forEach((x, i) => {
    const y = ys[i];
    if (y === null || y === undefined || Number.isNaN(y)) { pen = false; return; }
    d += `${pen ? 'L' : 'M'}${px(x).toFixed(1)},${py(y).toFixed(1)} `;
    pen = true;
  });
  return d.trim();
}

/** Marker shapes: a circle, a square and a diamond, so that the series differ by more than colour. */
export const Marker = {
  name: 'Marker',
  props: { shape: { type: String, default: 'circle' }, cx: Number, cy: Number, r: { type: Number, default: 4 }, colour: { type: String, default: 'var(--text)' } },
  setup(props) {
    const diamond = () => `${props.cx},${props.cy - props.r * 1.3} ${props.cx + props.r * 1.3},${props.cy} ${props.cx},${props.cy + props.r * 1.3} ${props.cx - props.r * 1.3},${props.cy}`;
    return { diamond };
  },
  template: `
  <g>
    <circle v-if="shape === 'circle'" :cx="cx" :cy="cy" :r="r" :style="'fill:' + colour + ';stroke:var(--surface);stroke-width:1.5'"/>
    <rect v-else-if="shape === 'square'" :x="cx - r" :y="cy - r" :width="2 * r" :height="2 * r" :style="'fill:' + colour + ';stroke:var(--surface);stroke-width:1.5'"/>
    <polygon v-else :points="diamond()" :style="'fill:' + colour + ';stroke:var(--surface);stroke-width:1.5'"/>
  </g>`,
};

// ------------------------------------------------------------------------------------------------------------------------------- 1. EEM -> code -> EEM
export const FigAutoencoder = {
  name: 'FigAutoencoder',
  components: { Heat },
  setup() {
    const D = DATA.one;
    const codeX = (i) => 296 + i * 36;
    const w = D.n_codes * 36 - 6;
    return { ...S, D, codeX, w, sgn, pct, thousands, fmt, em0: D.em[0], em1: D.em[D.em.length - 1], ex0: D.ex[D.ex.length - 1], ex1: D.ex[0] };
  },
  template: `
  <figure class="sheet-fig">
    <svg viewBox="0 0 700 252" style="width:100%;max-width:740px;height:auto;display:block" role="img" aria-label="An EEM goes into the encoder, which gives a short code; the decoder draws an EEM from the code">
      <defs><marker id="ae-arr1" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--text-3)"/></marker></defs>
      <text x="14" y="20" :style="TB">Measured EEM</text>
      <Heat :grid="D.measured" :x="18" :y="30" :w="172" :h="89" :max="1"/>
      <text x="102" y="136" text-anchor="middle" :style="TXT">Emission {{ em0 }}–{{ em1 }} nm</text>
      <text transform="translate(12,76) rotate(-90)" text-anchor="middle" :style="TXT">Excitation {{ ex0 }}–{{ ex1 }} nm</text>
      <text x="102" y="156" text-anchor="middle" :style="T2">{{ thousands(D.n_pixels) }} pixels that exist,</text>
      <text x="102" y="170" text-anchor="middle" :style="T2">smoothed into {{ D.n_inputs }} coefficients</text>

      <line x1="194" x2="206" y1="76" y2="76" style="stroke:var(--text-3);stroke-width:1.6" marker-end="url(#ae-arr1)"/>
      <polygon points="210,24 210,128 286,62 286,90" style="fill:var(--surface-2);stroke:var(--line-strong);stroke-width:1.2"/>
      <text x="236" y="80" text-anchor="middle" :style="TB">Encoder</text>
      <text x="248" y="156" text-anchor="middle" :style="T2">layers of 64 and</text>
      <text x="248" y="170" text-anchor="middle" :style="T2">32 neurons</text>

      <text :x="codeX(0)" y="50" :style="TB">Code</text>
      <g v-for="(v, i) in D.code" :key="i">
        <rect :x="codeX(i)" y="62" width="30" height="28" rx="4" style="fill:var(--accent-weak);stroke:var(--accent);stroke-width:1.4"/>
        <text :x="codeX(i) + 15" y="80" text-anchor="middle" style="font-family:var(--font);font-size:10.5px;font-weight:700;fill:var(--text)">{{ sgn(v, 1) }}</text>
        <text :x="codeX(i) + 15" y="105" text-anchor="middle" :style="TXT">z{{ i + 1 }}</text>
      </g>
      <text :x="codeX(0) + w / 2" y="156" text-anchor="middle" :style="T2">{{ D.n_codes }} numbers</text>
      <text :x="codeX(0) + w / 2" y="170" text-anchor="middle" :style="TXT">the code</text>

      <line x1="292" x2="292" y1="76" y2="76" style="stroke:none"/>
      <polygon points="410,62 410,90 486,128 486,24" style="fill:var(--surface-2);stroke:var(--line-strong);stroke-width:1.2"/>
      <line x1="392" x2="406" y1="76" y2="76" style="stroke:var(--text-3);stroke-width:1.6" marker-end="url(#ae-arr1)"/>
      <text x="460" y="80" text-anchor="middle" :style="TB">Decoder</text>
      <text x="448" y="156" text-anchor="middle" :style="T2">layers of 32 and</text>
      <text x="448" y="170" text-anchor="middle" :style="T2">64 neurons</text>

      <line x1="490" x2="504" y1="76" y2="76" style="stroke:var(--text-3);stroke-width:1.6" marker-end="url(#ae-arr1)"/>
      <text x="510" y="20" :style="TB">Rebuilt EEM</text>
      <Heat :grid="D.rebuilt" :x="510" :y="30" :w="176" :h="91" :max="1"/>
      <text x="598" y="156" text-anchor="middle" :style="T2">{{ pct(D.r2, 2) }} of the difference from</text>
      <text x="598" y="170" text-anchor="middle" :style="T2">the average EEM is rebuilt</text>

      <rect x="14" y="196" width="14" height="11" style="fill:var(--line-strong);fill-opacity:.45;stroke:var(--line-strong)"/><text x="34" y="206" :style="T2">grey: a pixel that does not exist (scatter band, below the Rayleigh line)</text>
      <rect x="14" y="214" width="14" height="11" style="fill:var(--accent);fill-opacity:.9"/><text x="34" y="224" :style="T2">blue: the more intense the EEM, the stronger the colour (the same scale on both sides)</text>
      <text x="14" y="244" :style="TXT">Sample {{ D.id }}, not used for training. The network has {{ thousands(D.n_parameters) }} weights.</text>
    </svg>
    <figcaption><slot/></figcaption>
  </figure>`,
};

// ------------------------------------------------------------------------------------------------------------------------------- 2. PCA against the network, on two data sets
export const FigPcaVsAe = {
  name: 'FigPcaVsAe',
  components: { Marker },
  setup() {
    const C = DATA.compare, dims = C.dims;
    const W = 262, H = 168, Y0 = 44, LO = Math.log10(0.04), HI = Math.log10(40);
    const panels = [['linear', 'Three fluorophores that add up', 54], ['bending', 'A band that moves, and its amount', 392]].map(([key, title, x0]) => {
      const D = C[key];
      const px = (k) => x0 + ((k - 1) / (dims.length - 1)) * W;
      const py = (v) => Y0 + H - ((Math.log10(Math.max(v * 100, 0.01)) - LO) / (HI - LO)) * H;
      const line = (vals) => vals.map((v, i) => `${i ? 'L' : 'M'}${px(dims[i]).toFixed(1)},${py(v).toFixed(1)}`).join(' ');
      const kb = D.recommended, target = D.ae[kb - 1];
      const kp = D.pca.findIndex((v) => v <= target * 1.3) + 1;
      return { key, title, x0, D, px, py, pca: line(D.pca), ae: line(D.ae), lin: line(D.linear), kb, kp,
               bars: D.ae.map((v, i) => ({ x: px(dims[i]), y1: py(Math.max(v - D.ae_se[i], 1e-5)), y2: py(v + D.ae_se[i]) })) };
    });
    const ticksY = [0.1, 1, 10].map((p) => ({ p, label: p + ' %' }));
    return { ...S, C, dims, panels, W, H, Y0, ticksY, LO, HI };
  },
  template: `
  <figure class="sheet-fig">
    <svg viewBox="0 0 700 316" style="width:100%;max-width:740px;height:auto;display:block" role="img" aria-label="Error on held-out samples against the number of codes for PCA and for the network, on two data sets">
      <g v-for="p in panels" :key="p.key">
        <text :x="p.x0" y="16" :style="TB">{{ p.title }}</text>
        <text :x="p.x0" y="30" :style="TXT">{{ p.D.n }} made-up EEMs, held-out samples</text>
        <g v-for="t in ticksY" :key="t.p">
          <line :x1="p.x0" :x2="p.x0 + W" :y1="p.py(t.p / 100)" :y2="p.py(t.p / 100)" :style="GRID"/>
          <text :x="p.x0 - 8" :y="p.py(t.p / 100) + 4" text-anchor="end" :style="TXT">{{ t.label }}</text>
        </g>
        <rect :x="p.x0" :y="Y0" :width="W" :height="H" :style="AXIS"/>
        <g v-for="k in dims" :key="'k' + k"><line :x1="p.px(k)" :x2="p.px(k)" :y1="Y0 + H" :y2="Y0 + H + 4" :style="AXIS"/><text :x="p.px(k)" :y="Y0 + H + 17" text-anchor="middle" :style="TXT">{{ k }}</text></g>
        <text :x="p.x0 + W / 2" :y="Y0 + H + 34" text-anchor="middle" :style="T2">Number of codes (components)</text>
        <path :d="p.pca" style="fill:none;stroke:var(--text-3);stroke-width:2;stroke-linejoin:round"/>
        <path :d="p.lin" style="fill:none;stroke:var(--series-5);stroke-width:1.8;stroke-dasharray:2 3;stroke-linejoin:round"/>
        <path :d="p.ae" style="fill:none;stroke:var(--series-1);stroke-width:2.2;stroke-linejoin:round"/>
        <line v-for="(b, i) in p.bars" :key="'b' + i" :x1="b.x" :x2="b.x" :y1="b.y1" :y2="b.y2" style="stroke:var(--series-1);stroke-width:1.2"/>
        <Marker v-for="(v, i) in p.D.pca" :key="'p' + i" shape="square" :cx="p.px(dims[i])" :cy="p.py(v)" :r="3.6" colour="var(--text-3)"/>
        <Marker v-for="(v, i) in p.D.linear" :key="'l' + i" shape="diamond" :cx="p.px(dims[i])" :cy="p.py(v)" :r="3" colour="var(--series-5)"/>
        <Marker v-for="(v, i) in p.D.ae" :key="'a' + i" shape="circle" :cx="p.px(dims[i])" :cy="p.py(v)" :r="3.8" colour="var(--series-1)"/>
        <text :x="p.x0 + W - 4" :y="Y0 + 15" text-anchor="end" :style="T2">Same quality: network {{ p.kb }} codes,</text>
        <text :x="p.x0 + W - 4" :y="Y0 + 29" text-anchor="end" :style="TB">PCA {{ p.kp }} components</text>
      </g>
      <g transform="translate(54,286)">
        <line x1="0" x2="22" y1="0" y2="0" style="stroke:var(--text-3);stroke-width:2"/><Marker shape="square" :cx="11" :cy="0" :r="3.6" colour="var(--text-3)"/><text x="30" y="4" :style="T2">Functional PCA</text>
        <line x1="160" x2="182" y1="0" y2="0" style="stroke:var(--series-5);stroke-width:1.8;stroke-dasharray:2 3"/><Marker shape="diamond" :cx="171" :cy="0" :r="3" colour="var(--series-5)"/><text x="190" y="4" :style="T2">Network, activation linear (no bends)</text>
        <line x1="420" x2="442" y1="0" y2="0" style="stroke:var(--series-1);stroke-width:2.2"/><Marker shape="circle" :cx="431" :cy="0" :r="3.8" colour="var(--series-1)"/><text x="450" y="4" :style="T2">Network with bends (silu), ± 1 SE</text>
      </g>
    </svg>
    <figcaption><slot/></figcaption>
  </figure>`,
};

// ------------------------------------------------------------------------------------------------------------------------------- 3. a traversal strip
export const FigTraversal = {
  name: 'FigTraversal',
  components: { Heat },
  setup() {
    const D = DATA.traversal;
    const cw = 84, gap = 9, x0 = 22;
    const cols = D.values.map((v, i) => ({ v, x: x0 + i * (cw + gap), label: v === 0 ? '0 SD' : (v > 0 ? '+' : MINUS) + Math.abs(v) + ' SD' }));
    return { ...S, D, cols, cw, MINUS, sgn };
  },
  template: `
  <figure class="sheet-fig">
    <svg viewBox="0 0 700 232" style="width:100%;max-width:740px;height:auto;display:block" role="img" aria-label="The EEMs decoded when one code goes from minus three to plus three standard deviations, and their difference to the middle one">
      <text x="22" y="14" :style="TB">Code z{{ D.code }} from {{ MINUS }}3 to +3 SD, the other code at its mean: the decoded EEM</text>
      <g v-for="(c, i) in cols" :key="'t' + i">
        <text :x="c.x + cw / 2" y="30" text-anchor="middle" :style="TXT">{{ c.label }}</text>
        <Heat :grid="D.images[i]" :x="c.x" :y="36" :w="cw" :h="44" :max="1"/>
      </g>
      <text x="22" y="104" :style="TB">The difference to the EEM at 0 SD</text>
      <g v-for="(c, i) in cols" :key="'d' + i">
        <Heat :grid="D.differences[i]" :x="c.x" :y="112" :w="cw" :h="44" mode="div" :max="1"/>
      </g>
      <rect x="22" y="174" width="14" height="11" style="fill:var(--series-1);fill-opacity:.85"/><text x="42" y="184" :style="T2">blue: stronger than at 0 SD</text>
      <rect x="226" y="174" width="14" height="11" style="fill:var(--series-6);fill-opacity:.85"/><text x="246" y="184" :style="T2">orange: weaker</text>
      <rect x="352" y="174" width="14" height="11" style="fill:var(--line-strong);fill-opacity:.45;stroke:var(--line-strong)"/><text x="372" y="184" :style="T2">grey: no pixel</text>
      <text x="22" y="206" :style="TXT">1 SD of this code = {{ D.sd }}, the standard deviation of the codes of the training samples.</text>
      <text x="22" y="220" :style="TXT">The strongest difference is {{ Math.round(D.difference_scale * 100) }} % of the highest intensity. Maps: emission to the right, excitation up.</text>
    </svg>
    <figcaption><slot/></figcaption>
  </figure>`,
};

// ------------------------------------------------------------------------------------------------------------------------------- 4. smoothify on one rebuilt function
export const FigSmoothify = {
  name: 'FigSmoothify',
  setup() {
    const C = DATA.smoothify.cut;
    const X0 = 56, Y0 = 20, W = 390, H = 196;
    const em = C.em;
    const all = [...C.measured, ...C.off, ...C.on, ...C.smoothed].filter((v) => v !== null);
    const lo = Math.min(0, Math.min(...all)), hi = Math.max(...all) * 1.08;
    const px = (v) => X0 + ((v - em[0]) / (em[em.length - 1] - em[0])) * W;
    const py = (v) => Y0 + H - ((v - lo) / (hi - lo)) * H;
    const ticksX = [300, 400, 500].map((v) => ({ v, x: px(v) }));
    const ticksY = niceTicks(lo, hi, 3).map((v) => ({ v, y: py(v), label: String(v) }));
    const dots = em.map((x, i) => ({ x: px(x), y: C.measured[i] === null ? null : py(C.measured[i]) })).filter((d) => d.y !== null);
    const R = C.roughness, V = C.evr_val;
    const sOn = Number(C.smoothify_on.toPrecision(2));
    return { ...S, C, sOn, X0, Y0, W, H, px, py, ticksX, ticksY, dots, off: path(em, C.off, px, py), on: path(em, C.on, px, py), smooth: path(em, C.smoothed, px, py), R, V, pct, fmt, sgn };
  },
  template: `
  <figure class="sheet-fig">
    <svg viewBox="0 0 700 270" style="width:100%;max-width:740px;height:auto;display:block" role="img" aria-label="One rebuilt function with and without smoothify: with it the ripples are gone">
      <g v-for="t in ticksY" :key="'y' + t.v"><line :x1="X0" :x2="X0 + W" :y1="t.y" :y2="t.y" :style="GRID"/><text :x="X0 - 8" :y="t.y + 4" text-anchor="end" :style="TXT">{{ t.label }}</text></g>
      <rect :x="X0" :y="Y0" :width="W" :height="H" :style="AXIS"/>
      <g v-for="t in ticksX" :key="'x' + t.v"><line :x1="t.x" :x2="t.x" :y1="Y0 + H" :y2="Y0 + H + 4" :style="AXIS"/><text :x="t.x" :y="Y0 + H + 17" text-anchor="middle" :style="TXT">{{ t.v }}</text></g>
      <text :x="X0 + W / 2" :y="Y0 + H + 36" text-anchor="middle" :style="T2">Emission (nm), at excitation {{ C.ex_nm }} nm</text>
      <text :transform="'translate(14,' + (Y0 + H / 2) + ') rotate(-90)'" text-anchor="middle" :style="T2">Intensity (arbitrary units)</text>
      <path :d="smooth" style="fill:none;stroke:var(--text-2);stroke-width:1.2;stroke-dasharray:4 3"/>
      <path :d="off" style="fill:none;stroke:var(--series-6);stroke-width:2;stroke-linejoin:round"/>
      <path :d="on" style="fill:none;stroke:var(--series-1);stroke-width:2.4;stroke-linejoin:round"/>
      <circle v-for="(d, i) in dots" :key="'m' + i" :cx="d.x" :cy="d.y" r="2.6" style="fill:var(--text-3);fill-opacity:.8"/>
      <g transform="translate(462,34)">
        <text x="0" y="0" :style="TB">One held-out sample ({{ C.sample }})</text>
        <circle cx="8" cy="20" r="2.8" style="fill:var(--text-3)"/><text x="24" y="24" :style="T2">measured values</text>
        <line x1="0" x2="16" y1="42" y2="42" style="stroke:var(--text-2);stroke-width:1.2;stroke-dasharray:4 3"/><text x="24" y="46" :style="T2">smoothed (step 4)</text>
        <line x1="0" x2="16" y1="64" y2="64" style="stroke:var(--series-6);stroke-width:2"/><text x="24" y="68" :style="T2">rebuilt, smoothify 0</text>
        <line x1="0" x2="16" y1="86" y2="86" style="stroke:var(--series-1);stroke-width:2.4"/><text x="24" y="90" :style="T2">rebuilt, smoothify {{ sOn }}</text>
        <text x="0" y="122" :style="TB">Roughness of the whole EEM</text>
        <text x="0" y="140" :style="T2">smoothed measured: {{ fmt(R.measured_smoothed, 2) }}</text>
        <text x="0" y="156" :style="T2">rebuilt, smoothify 0: {{ fmt(R.off, 2) }}</text>
        <text x="0" y="172" :style="T2">rebuilt, smoothify {{ sOn }}: {{ fmt(R.on, 2) }}</text>
        <text x="0" y="198" :style="TB">The price (all held-out samples)</text>
        <text x="0" y="216" :style="T2">explained: {{ pct(V.off, 1) }} without,</text>
        <text x="0" y="232" :style="T2">{{ pct(V.on, 1) }} with smoothify {{ sOn }}</text>
      </g>
    </svg>
    <figcaption><slot/></figcaption>
  </figure>`,
};

// ------------------------------------------------------------------------------------------------------------------------------- 6. the L-curve of the smoothify scan
export const FigLcurve = {
  name: 'FigLcurve',
  components: { Marker },
  setup() {
    const D = DATA.smoothify.scan;
    const X0 = 62, Y0 = 20, W = 380, H = 190;
    const rough = D.roughness.map((v) => Math.max(v, 0.003)), err = D.error.map((v) => Math.max(v, 0.003));
    const xLo = Math.log10(0.02), xHi = Math.log10(Math.max(...rough) * 1.4), yLo = Math.log10(0.005), yHi = Math.log10(1.4);
    const px = (v) => X0 + ((Math.log10(Math.max(v, 0.02)) - xLo) / (xHi - xLo)) * W;
    const py = (v) => Y0 + H - ((Math.log10(v) - yLo) / (yHi - yLo)) * H;
    const pts = D.values.map((s, i) => ({ s, x: px(rough[i]), y: py(err[i]), rough: D.roughness[i], err: D.error[i], sug: Math.abs(s - D.suggested) <= 1e-3 * Math.max(Math.abs(D.suggested), 1e-3) }));
    const line = pts.map((p, i) => `${i ? 'L' : 'M'}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ');
    const ticksY = [0.01, 0.1, 1].map((v) => ({ v, y: py(v), label: v * 100 + ' %' }));
    const ticksX = [0.1, 1, 10].filter((v) => v <= Math.max(...rough) * 1.4).map((v) => ({ v, x: px(v), label: String(v) }));
    const sug = pts.find((p) => p.sug) || null;
    // labels: smoothify 0 and the values in turn, placed to avoid each other
    const order = pts.map((p, i) => i).sort((a, b) => (pts[b].sug ? 1 : 0) - (pts[a].sug ? 1 : 0) || (pts[a].s === 0 ? -1 : 0) - (pts[b].s === 0 ? -1 : 0));
    const placed = [];
    for (const i of order) {
      const p = pts[i];
      if (placed.every((q) => Math.hypot(q.x - p.x, q.y - p.y) > 34)) placed.push({ x: p.x, y: p.y, text: p.s === 0 ? '0' : String(Number(p.s.toPrecision(2))), dy: p.y > Y0 + 24 ? -10 : 18 });
    }
    return { ...S, D, X0, Y0, W, H, pts, line, ticksX, ticksY, sug, labels: placed, inputX: px(D.roughness_input), pct, fmt, sugText: String(Number(D.suggested.toPrecision(2))) };
  },
  template: `
  <figure class="sheet-fig">
    <svg viewBox="0 0 700 270" style="width:100%;max-width:740px;height:auto;display:block" role="img" aria-label="The smoothify scan: the rebuilding error against the roughness for several values of smoothify">
      <g v-for="t in ticksY" :key="'y' + t.v"><line :x1="X0" :x2="X0 + W" :y1="t.y" :y2="t.y" :style="GRID"/><text :x="X0 - 8" :y="t.y + 4" text-anchor="end" :style="TXT">{{ t.label }}</text></g>
      <g v-for="t in ticksX" :key="'x' + t.v"><line :x1="t.x" :x2="t.x" :y1="Y0" :y2="Y0 + H" :style="GRID"/><line :x1="t.x" :x2="t.x" :y1="Y0 + H" :y2="Y0 + H + 4" :style="AXIS"/><text :x="t.x" :y="Y0 + H + 17" text-anchor="middle" :style="TXT">{{ t.label }}</text></g>
      <rect :x="X0" :y="Y0" :width="W" :height="H" :style="AXIS"/>
      <line :x1="inputX" :x2="inputX" :y1="Y0" :y2="Y0 + H" style="stroke:var(--text-3);stroke-width:1.2;stroke-dasharray:4 3"/>
      <text :x="inputX - 4" :y="Y0 + 12" text-anchor="end" :style="TXT">roughness of the</text><text :x="inputX - 4" :y="Y0 + 25" text-anchor="end" :style="TXT">measured EEMs</text>
      <path :d="line" style="fill:none;stroke:var(--series-1);stroke-width:2;stroke-linejoin:round"/>
      <Marker v-for="(p, i) in pts" :key="'p' + i" shape="circle" :cx="p.x" :cy="p.y" :r="4" colour="var(--series-1)"/>
      <circle v-if="sug" :cx="sug.x" :cy="sug.y" r="9.5" style="fill:none;stroke:var(--text);stroke-width:1.8"/>
      <text v-for="(l, i) in labels" :key="'l' + i" :x="l.x" :y="l.y + l.dy" text-anchor="middle" :style="T2">{{ l.text }}</text>
      <text :x="X0 + W / 2" :y="Y0 + H + 36" text-anchor="middle" :style="T2">Roughness of the rebuilt EEMs (log scale)</text>
      <text :transform="'translate(14,' + (Y0 + H / 2) + ') rotate(-90)'" text-anchor="middle" :style="T2">Error on held-out samples (log scale)</text>
      <g transform="translate(470,40)">
        <text x="0" y="0" :style="TB">Smoothify scan</text>
        <text x="0" y="18" :style="T2">{{ D.values.length }} networks, one for each value;</text>
        <text x="0" y="32" :style="T2">some points carry their smoothify.</text>
        <circle cx="8" cy="64" r="9.5" style="fill:none;stroke:var(--text);stroke-width:1.8"/>
        <text x="26" y="62" :style="T2">the value the page suggests:</text>
        <text x="26" y="76" :style="TB">{{ sugText }}</text>
        <text x="0" y="108" :style="T2">Right of the corner: much rougher</text>
        <text x="0" y="122" :style="T2">for no gain in error.</text>
        <text x="0" y="146" :style="T2">Left of it: smoother, but the error</text>
        <text x="0" y="160" :style="T2">rises, and at the end every EEM</text>
        <text x="0" y="174" :style="T2">is rebuilt as the mean.</text>
      </g>
    </svg>
    <figcaption><slot/></figcaption>
  </figure>`,
};

// ------------------------------------------------------------------------------------------------------------------------------- 5. the KL of every code of a VAE
export const FigCollapse = {
  name: 'FigCollapse',
  setup() {
    const D = DATA.collapse;
    const X0 = 62, Y0 = 24, W = 400, H = 176, LO = Math.log10(0.004), HI = Math.log10(8);
    const n = D.kl.length, bw = W / n;
    const py = (v) => Y0 + H - ((Math.log10(Math.max(v, 0.004)) - LO) / (HI - LO)) * H;
    const bars = D.kl.map((v, i) => ({ i, v, x: X0 + i * bw + bw * 0.2, w: bw * 0.6, y: py(v), off: D.inactive[i], name: 'z' + (i + 1), share: D.effect_share[i],
      state: !D.inactive[i] ? 'active' : v < D.limit ? 'collapsed' : 'unused' }));
    const lastActive = D.effect_share.filter((s, i) => !D.inactive[i]).slice(-1)[0];
    const ticks = [0.01, 0.1, 1].map((v) => ({ v, y: py(v), label: String(v) }));
    return { ...S, D, X0, Y0, W, H, bars, bw, ticks, yLimit: py(D.limit), fmt, pct, lastActive, effText: (s) => (s < 0.0005 ? '0 %' : s < 0.01 ? '<1 %' : Math.round(s * 100) + ' %') };
  },
  template: `
  <figure class="sheet-fig">
    <svg viewBox="0 0 700 276" style="width:100%;max-width:740px;height:auto;display:block" role="img" aria-label="The KL divergence of each of six codes of a variational autoencoder, and how much each code does to the rebuilt EEM: some codes are inactive">
      <g v-for="t in ticks" :key="t.v"><line :x1="X0" :x2="X0 + W" :y1="t.y" :y2="t.y" :style="GRID"/><text :x="X0 - 8" :y="t.y + 4" text-anchor="end" :style="TXT">{{ t.label }}</text></g>
      <rect :x="X0" :y="Y0" :width="W" :height="H" :style="AXIS"/>
      <g v-for="b in bars" :key="b.i">
        <rect :x="b.x" :y="b.y" :width="b.w" :height="Y0 + H - b.y" :style="b.off ? 'fill:none;stroke:var(--text-3);stroke-width:1.6;stroke-dasharray:4 3' : 'fill:var(--accent);fill-opacity:.9'"/>
        <text :x="b.x + b.w / 2" :y="(b.v < D.limit ? yLimit : b.y) - 6" text-anchor="middle" :style="T2">{{ fmt(b.v, b.v < 1 ? 3 : 2) }}</text>
        <text :x="b.x + b.w / 2" :y="Y0 + H + 17" text-anchor="middle" :style="TB">{{ b.name }}</text>
        <text :x="b.x + b.w / 2" :y="Y0 + H + 31" text-anchor="middle" :style="TXT">{{ b.state }}</text>
        <text :x="b.x + b.w / 2" :y="Y0 + H + 45" text-anchor="middle" :style="TXT">effect {{ effText(b.share) }}</text>
      </g>
      <line :x1="X0" :x2="X0 + W" :y1="yLimit" :y2="yLimit" style="stroke:var(--text);stroke-width:1.2;stroke-dasharray:6 3"/>
      <text :transform="'translate(14,' + (Y0 + H / 2) + ') rotate(-90)'" text-anchor="middle" :style="T2">KL of the code (nats, log scale)</text>
      <g transform="translate(490,52)">
        <text x="0" y="0" :style="TB">{{ D.n_active }} of {{ D.kl.length }} codes are used</text>
        <rect x="0" y="16" width="22" height="12" style="fill:var(--accent);fill-opacity:.9"/><text x="30" y="26" :style="T2">active: carries information</text>
        <rect x="0" y="38" width="22" height="12" style="fill:none;stroke:var(--text-3);stroke-width:1.6;stroke-dasharray:4 3"/><text x="30" y="48" :style="T2">inactive: collapsed or unused</text>
        <line x1="0" x2="22" y1="64" y2="64" style="stroke:var(--text);stroke-width:1.2;stroke-dasharray:6 3"/><text x="30" y="68" :style="T2">{{ D.limit }} nats: below it, collapsed</text>
        <text x="0" y="98" :style="T2">The made-up data have two hidden</text>
        <text x="0" y="112" :style="T2">factors; the last active code holds</text>
        <text x="0" y="126" :style="T2">{{ effText(lastActive) }} of the effect on the EEM.</text>
        <text x="0" y="150" :style="T2">Variance explained on held-out</text>
        <text x="0" y="164" :style="T2">samples: {{ pct(D.evr_val, 1) }}.</text>
      </g>
    </svg>
    <figcaption><slot/></figcaption>
  </figure>`,
};

// ------------------------------------------------------------------------------------------------------------------------------- 7. loss curves: healthy and over-fitted
export const FigLoss = {
  name: 'FigLoss',
  setup() {
    const L = DATA.loss;
    const W = 262, H = 168, Y0 = 52;
    const every = [...L.healthy.train, ...L.healthy.val, ...L.overfit.train, ...L.overfit.val].filter((v) => v !== null && v > 0);
    const LO = Math.floor(Math.log10(Math.min(...every) * 0.8) * 2) / 2, HI = Math.ceil(Math.log10(Math.max(...every) * 1.25) * 2) / 2;
    const panels = [['healthy', 'Healthy', 'small network, weight decay, early stopping', 54], ['overfit', 'Over-fitting', 'large network, no weight decay, no stop', 392]].map(([key, title, sub, x0]) => {
      const D = L[key], xmax = Math.max(...D.epoch);
      const px = (e) => x0 + (e / xmax) * W;
      const py = (v) => Y0 + H - ((Math.log10(Math.max(v, 1e-6)) - LO) / (HI - LO)) * H;
      const lastTrain = D.train[D.train.length - 1], lastVal = D.val[D.val.length - 1];
      return { key, title, sub, x0, D, px, py, tr: path(D.epoch, D.train, px, py), va: path(D.epoch, D.val, px, py), best: px(D.best_epoch), xmax,
               ticksX: [0, 0.5, 1].map((f) => ({ e: Math.round(f * xmax), x: px(f * xmax) })), lastTrain, lastVal, yTr: py(lastTrain), yVa: py(lastVal) };
    });
    const ticksY = [];
    for (let k = Math.floor(LO) - 1; k <= Math.ceil(HI); k++) for (const m of [1, 3]) { const v = m * Math.pow(10, k); if (Math.log10(v) >= LO - 1e-9 && Math.log10(v) <= HI + 1e-9) ticksY.push({ v, label: Number((v * 100).toPrecision(2)) + ' %' }); }
    return { ...S, L, panels, W, H, Y0, ticksY, pct };
  },
  template: `
  <figure class="sheet-fig">
    <svg viewBox="0 0 700 308" style="width:100%;max-width:740px;height:auto;display:block" role="img" aria-label="Loss curves of two trainings: in the healthy one the held-out loss flattens; in the over-fitted one it rises while the training loss keeps falling">
      <g v-for="p in panels" :key="p.key">
        <text :x="p.x0" y="16" :style="TB">{{ p.title }}</text>
        <text :x="p.x0" y="30" :style="TXT">{{ p.sub }}</text>
        <text :x="p.x0" y="44" :style="TXT">{{ p.D.n_train }} training, {{ p.D.n_val }} held-out samples; {{ p.D.n_parameters.toLocaleString('en') }} weights</text>
        <g v-for="t in ticksY" :key="t.v"><line :x1="p.x0" :x2="p.x0 + W" :y1="p.py(t.v)" :y2="p.py(t.v)" :style="GRID"/><text :x="p.x0 - 8" :y="p.py(t.v) + 4" text-anchor="end" :style="TXT">{{ t.label }}</text></g>
        <rect :x="p.x0" :y="Y0" :width="W" :height="H" :style="AXIS"/>
        <g v-for="t in p.ticksX" :key="'x' + t.e"><line :x1="t.x" :x2="t.x" :y1="Y0 + H" :y2="Y0 + H + 4" :style="AXIS"/><text :x="t.x" :y="Y0 + H + 17" text-anchor="middle" :style="TXT">{{ t.e }}</text></g>
        <text :x="p.x0 + W / 2" :y="Y0 + H + 34" text-anchor="middle" :style="T2">Epoch</text>
        <line :x1="p.best" :x2="p.best" :y1="Y0" :y2="Y0 + H" style="stroke:var(--text);stroke-width:1.2;stroke-dasharray:5 3"/>
        <text :x="p.best + (p.best > p.x0 + W * 0.6 ? -4 : 4)" :y="Y0 + H - 6" :text-anchor="p.best > p.x0 + W * 0.6 ? 'end' : 'start'" :style="TXT">best epoch {{ p.D.best_epoch }}</text>
        <path :d="p.tr" style="fill:none;stroke:var(--series-1);stroke-width:2;stroke-linejoin:round"/>
        <path :d="p.va" style="fill:none;stroke:var(--series-6);stroke-width:2;stroke-dasharray:6 3;stroke-linejoin:round"/>
        <text :x="p.x0 + W - 4" :y="Y0 + 14" text-anchor="end" :style="T2">at the end: held-out {{ pct(p.lastVal, 1) }}, training {{ pct(p.lastTrain, 1) }}</text>
      </g>
      <g transform="translate(54,284)">
        <line x1="0" x2="26" y1="0" y2="0" style="stroke:var(--series-1);stroke-width:2"/><text x="34" y="4" :style="T2">loss on the training samples (error: the share of the variation not rebuilt)</text>
        <line x1="440" x2="466" y1="0" y2="0" style="stroke:var(--series-6);stroke-width:2;stroke-dasharray:6 3"/><text x="474" y="4" :style="T2">on the held-out samples</text>
      </g>
    </svg>
    <figcaption><slot/></figcaption>
  </figure>`,
};
