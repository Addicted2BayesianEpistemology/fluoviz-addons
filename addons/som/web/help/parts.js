// Small building blocks of the Help guide of the add-on "Self-organising map": the way an interface word is written, the links, and two schematic pictures drawn with the theme's colours.
// Nothing here is imported from FluoViz except the shared components with absolute URLs (the icons, the Callout box, the router), which are the stable interface for add-ons.
import { Icon } from '/static/js/icons.js';
import { Callout } from '/static/js/components/Common.js';
import { go } from '/static/js/router.js';

/** The Help tab of this guide (the address is  #/help?tab=addon:som:som&topic=<page>). */
export const TAB = 'addon:som:som';

/** A word of the interface (a button, a field, a step, a tab).  The test of the guide looks for every one of them in docs/ui.md. */
export const Ui = { name: 'Ui', template: `<i class="ui"><slot/></i>` };

export const GLink = {
  name: 'GLink',
  props: { to: { type: String, required: true }, tab: { type: String, default: TAB } },
  setup(props) { return { open: () => go('/help', { tab: props.tab, topic: props.to }) }; },
  template: `<a class="g-link" href="#" @click.prevent="open"><slot/></a>`,
};

export const PLink = {
  name: 'PLink',
  props: { path: { type: String, required: true }, query: { type: Object, default: null } },
  setup(props) { return { open: () => go(props.path, props.query || undefined) }; },
  template: `<a class="g-link" href="#" @click.prevent="open" :data-plink="path"><slot/></a>`,
};

const T = 'font-family:var(--font);font-size:11px;fill:var(--text-3)';
const TB = 'font-family:var(--font);font-size:12px;font-weight:700;fill:var(--text)';

/** A figure frame: the picture and its caption. */
export const Fig = { name: 'Fig', template: `<figure class="g-fig"><slot/><figcaption><slot name="caption"/></figcaption></figure>` };

const hex = (cx, cy, r) => Array.from({ length: 6 }, (_, k) => { const a = Math.PI / 6 + (k * Math.PI) / 3; return `${(cx + r * Math.cos(a)).toFixed(1)},${(cy + r * Math.sin(a)).toFixed(1)}`; }).join(' ');

/** A small map: hexagonal units whose colour changes smoothly across the map (similar prototypes next to each other), with three samples and the neighbourhood of the winner. */
export const FigMap = {
  name: 'FigMap',
  setup() {
    const rows = 4, cols = 6, step = 46, r = step / Math.sqrt(3), cells = [];
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
      const cx = 30 + j * step + (i % 2 ? step / 2 : 0), cy = 30 + i * step * 0.866, t = (j + 0.5 * (i % 2)) / (cols - 0.5);
      cells.push({ pts: hex(cx, cy, r - 1), cx, cy, fill: `color-mix(in srgb, var(--accent) ${Math.round(12 + 78 * t)}%, var(--surface))` });
    }
    const win = cells[2 * cols + 3];
    return { cells, win, T, TB, W: 30 + cols * step + 10, H: 30 + rows * step * 0.866 + 14 };
  },
  template: `
  <svg :viewBox="'0 0 ' + W + ' ' + H" role="img" aria-label="A map of hexagonal units whose colours change smoothly" style="width:100%;max-width:440px;height:auto">
    <polygon v-for="(c, i) in cells" :key="i" :points="c.pts" :style="{ fill: c.fill, stroke: 'var(--surface)', strokeWidth: 1.5 }"/>
    <circle :cx="win.cx" :cy="win.cy" r="52" style="fill:none;stroke:var(--text);stroke-width:1.6;stroke-dasharray:4 3"/>
    <circle :cx="win.cx" :cy="win.cy" r="6" style="fill:var(--text);stroke:var(--surface);stroke-width:1.5"/>
    <text :x="win.cx + 9" :y="win.cy - 8" :style="TB">winner</text>
    <text :x="W / 2 - 80" :y="H - 2" :style="T">similar units are neighbours, unlike ones far apart</text>
  </svg>`,
};

/** The plain distance (a circle) against the standardised one (an ellipse that follows the cloud): the same two samples, a different verdict about which is nearer. */
export const FigMetric = {
  name: 'FigMetric',
  setup() {
    const pts = [], rng = (k) => { const x = Math.sin(k * 9301 + 49297) * 233280; return x - Math.floor(x); };
    for (let i = 0; i < 46; i++) { const u = (rng(i) + rng(i + 99) - 1) * 70, v = (rng(i + 500) + rng(i + 700) - 1) * 14; pts.push([120 + u * 0.94 - v * 0.34, 90 - u * 0.34 - v * 0.94]); }
    return { pts, T, TB };
  },
  template: `
  <svg viewBox="0 0 440 180" role="img" aria-label="Two samples at the same plain distance from the centre but not the same standardised distance" style="width:100%;max-width:440px;height:auto">
    <g transform="translate(0,0)"><circle v-for="(p, i) in pts" :key="i" :cx="p[0]" :cy="p[1]" r="2.3" style="fill:var(--text-3);opacity:.75"/>
      <circle cx="120" cy="90" r="62" style="fill:none;stroke:var(--accent);stroke-width:1.6"/>
      <circle cx="120" cy="90" r="2.6" style="fill:var(--text)"/>
      <text x="22" y="170" :style="T">plain distance: a circle</text></g>
    <g transform="translate(220,0)"><circle v-for="(p, i) in pts" :key="i" :cx="p[0]" :cy="p[1]" r="2.3" style="fill:var(--text-3);opacity:.75"/>
      <ellipse cx="120" cy="90" rx="74" ry="16" transform="rotate(-20 120 90)" style="fill:none;stroke:var(--accent);stroke-width:1.6"/>
      <circle cx="120" cy="90" r="2.6" style="fill:var(--text)"/>
      <text x="22" y="170" :style="T">standardised: an ellipse</text></g>
  </svg>`,
};

export const parts = { Ui, GLink, PLink, Callout, Icon, Fig, FigMap, FigMetric };
