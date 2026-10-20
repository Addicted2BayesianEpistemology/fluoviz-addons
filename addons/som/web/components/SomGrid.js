// The map drawn as a grid of hexagons (or squares), one per unit, coloured by a plane; the samples as dots on it; the borders between groups; trajectories.  One Plotly chart (so it
// has the PNG / table / .rds / .fig buttons of every chart of FluoViz), light and dark through the theme tokens.  A click on a unit or on a dot is an event.
import { computed } from 'vue';
import { PlotlyChart } from '/static/js/plot.js';
import { tokens, themeState } from '/static/js/theme.js';
import { numericColours, categoryColour, groupColour } from './SomCommon.js';

const R_HEX = 1 / Math.sqrt(3);                      // pointy-top hexagon whose flat-to-flat width is one lattice step
const hexVerts = (cx, cy) => Array.from({ length: 6 }, (_, k) => { const a = Math.PI / 6 + (k * Math.PI) / 3; return [cx + R_HEX * Math.cos(a), cy + R_HEX * Math.sin(a)]; });
const rectVerts = (cx, cy) => [[cx - 0.5, cy - 0.5], [cx + 0.5, cy - 0.5], [cx + 0.5, cy + 0.5], [cx - 0.5, cy + 0.5]];

/** The corners of every unit: [[x, y] × 4 or 6]. */
export function unitPolygons(coords, M, lattice) {
  const out = new Array(M);
  for (let j = 0; j < M; j++) out[j] = (lattice === 'hexagonal' ? hexVerts : rectVerts)(coords.data[j * 2], coords.data[j * 2 + 1]);
  return out;
}
const close = (a, b) => Math.abs(a[0] - b[0]) < 1e-6 && Math.abs(a[1] - b[1]) < 1e-6;
/** The segments between neighbouring units of different groups (the shared side of their polygons). */
export function borderSegments(polys, edges, labels) {
  const segs = [];
  for (let e = 0; e < edges.n; e++) {
    const j = edges.data[e * 3], k = edges.data[e * 3 + 1];
    if (labels[j] === labels[k]) continue;
    const shared = polys[j].filter((p) => polys[k].some((q) => close(p, q)));
    if (shared.length >= 2) segs.push([shared[0], shared[1]]);
  }
  return segs;
}

export const SomGrid = {
  name: 'SomGrid',
  components: { PlotlyChart },
  props: {
    map: { type: Object, required: true },                 // { M, rows, cols, lattice, shape, coords (mat), edges (mat) }
    plane: { type: Object, default: null },                // a prepared plane: { v, kind, signed, label, units, categories?, key }
    dots: { type: Object, default: null },                 // { x: [], y: [], ids: [], colors: [], symbols: [], text: [], name }
    groups: { type: Object, default: null },               // Float64Array per unit: the group number (borders are drawn between different groups)
    lines: { type: Array, default: () => [] },             // trajectories: [{ name, x: [], y: [], color }]
    selected: { type: Number, default: null },
    height: { type: Number, default: 460 },
    hoverTexts: { type: Array, default: null },
    showColourbar: { type: Boolean, default: true },
  },
  emits: ['unit', 'dot', 'hover-dot', 'hover-unit'],
  setup(props, { emit }) {
    const polys = computed(() => unitPolygons(props.map.coords, props.map.M, props.map.lattice));
    const bounds = computed(() => {
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      for (const poly of polys.value) for (const [x, y] of poly) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
      return { x0, x1, y0, y1 };
    });
    const traces = computed(() => {
      themeState.version;
      const t = tokens(), pl = props.plane, M = props.map.M, out = [];
      const bg = t.surface;
      const bucketOf = new Array(M).fill('none');
      const buckets = new Map();
      let cinfo = null;
      if (pl && pl.kind === 'numeric') {
        cinfo = numericColours(pl.v, !!pl.signed);
        for (let j = 0; j < M; j++) { const c = cinfo.rgb(pl.v[j]); bucketOf[j] = c || 'none'; }
      } else if (pl && pl.kind === 'category') {
        const cats = pl.categories || [];
        for (let j = 0; j < M; j++) { const c = pl.v[j]; bucketOf[j] = c === c && cats[c] !== undefined ? categoryColour(pl.key || pl.label, cats, c) : 'none'; }
      }
      for (let j = 0; j < M; j++) { const k = bucketOf[j]; if (!buckets.has(k)) buckets.set(k, []); buckets.get(k).push(j); }
      for (const [colour, units] of buckets) {
        const x = [], y = [];
        for (const j of units) { for (const [px, py] of polys.value[j]) { x.push(px); y.push(py); } x.push(polys.value[j][0][0], null); y.push(polys.value[j][0][1], null); }
        out.push({ type: 'scatter', mode: 'lines', x, y, fill: 'toself', fillcolor: colour === 'none' ? t.line : colour, line: { color: bg, width: 1.2 }, hoverinfo: 'skip', showlegend: false });
      }
      if (props.groups && props.map.edges) {
        const segs = borderSegments(polys.value, props.map.edges, props.groups);
        const x = [], y = [];
        for (const [a, b] of segs) { x.push(a[0], b[0], null); y.push(a[1], b[1], null); }
        if (x.length) out.push({ type: 'scatter', mode: 'lines', x, y, line: { color: t.text, width: 3 }, hoverinfo: 'skip', showlegend: false, name: 'Group borders' });
      }
      if (props.selected !== null && props.selected !== undefined && polys.value[props.selected]) {
        const poly = polys.value[props.selected];
        out.push({ type: 'scatter', mode: 'lines', x: [...poly.map((p) => p[0]), poly[0][0]], y: [...poly.map((p) => p[1]), poly[0][1]], line: { color: t.accent, width: 4 }, hoverinfo: 'skip', showlegend: false });
      }
      for (const ln of props.lines) {
        out.push({ type: 'scatter', mode: 'lines+markers', x: ln.x, y: ln.y, name: ln.name, line: { color: ln.color, width: 2.2 }, marker: { size: 6, color: ln.color, line: { color: bg, width: 1 } }, hoverinfo: 'text', text: ln.text || ln.x.map(() => ln.name) });
        if (ln.x.length) {
          out.push({ type: 'scatter', mode: 'markers', x: [ln.x[0]], y: [ln.y[0]], marker: { size: 12, symbol: 'circle-open', color: ln.color, line: { width: 3, color: ln.color } }, hoverinfo: 'skip', showlegend: false });
          out.push({ type: 'scatter', mode: 'markers', x: [ln.x[ln.x.length - 1]], y: [ln.y[ln.y.length - 1]], marker: { size: 12, symbol: 'square', color: ln.color }, hoverinfo: 'skip', showlegend: false });
        }
      }
      if (props.dots && props.dots.x.length) {
        const d = props.dots;
        out.push({ type: 'scatter', mode: 'markers', x: d.x, y: d.y, customdata: d.ids, text: d.text, hoverinfo: 'text', name: d.name || 'Samples', showlegend: false,
          marker: { size: 8, color: d.colors || t.text, symbol: d.symbols || 'circle', line: { color: bg, width: 1.2 } } });
      }
      // the centres: hover and click on a unit
      const cx = [], cy = [], txt = [], idx = [];
      for (let j = 0; j < M; j++) {
        cx.push(props.map.coords.data[j * 2]); cy.push(props.map.coords.data[j * 2 + 1]); idx.push(j);
        txt.push(props.hoverTexts ? props.hoverTexts[j] : `Unit ${j + 1}`);
      }
      out.push({ type: 'scatter', mode: 'markers', x: cx, y: cy, customdata: idx, text: txt, hoverinfo: 'text', showlegend: false, name: 'Units', marker: { size: 22, opacity: 0.001, color: t.text } });
      if (cinfo && props.showColourbar) {
        out.push({ type: 'scatter', mode: 'markers', x: [cx[0]], y: [cy[0]], hoverinfo: 'skip', showlegend: false,
          marker: { size: 0.001, opacity: 0, color: [cinfo.lo, cinfo.hi], cmin: cinfo.lo, cmax: cinfo.hi, colorscale: cinfo.scale, showscale: true,
            colorbar: { thickness: 10, len: 0.8, outlinewidth: 0, title: { text: (pl && (pl.units || '')) || '', side: 'right', font: { size: 11 } }, tickfont: { size: 10 } } } });
      }
      return out;
    });
    const layout = computed(() => {
      const b = bounds.value, pad = 0.6;
      return { height: props.height, margin: { l: 6, r: 6, t: 6, b: 6 }, showlegend: false, hovermode: 'closest',
        xaxis: { visible: false, range: [b.x0 - pad, b.x1 + pad], constrain: 'domain' }, yaxis: { visible: false, range: [b.y1 + pad, b.y0 - pad], scaleanchor: 'x', scaleratio: 1 } };
    });
    function click(e) {
      const p = e && e.points && e.points[0]; if (!p) return;
      if (p.data.name === 'Units') emit('unit', p.customdata);
      else if (p.customdata !== undefined && typeof p.customdata === 'string') emit('dot', p.customdata);
      else if (p.data.name === (props.dots && props.dots.name || 'Samples') && p.customdata !== undefined) emit('dot', p.customdata);
    }
    function hover(e) {
      const p = e && e.points && e.points[0]; if (!p) return;
      if (p.data.name === 'Units') emit('hover-unit', p.customdata); else if (typeof p.customdata === 'string') emit('hover-dot', p.customdata);
    }
    function unhover() { emit('hover-dot', null); emit('hover-unit', null); }
    const table = computed(() => {
      const pl = props.plane, M = props.map.M, rows = [];
      for (let j = 0; j < M; j++) rows.push([j + 1, Math.floor(j / props.map.cols) + 1, (j % props.map.cols) + 1, pl ? (pl.kind === 'category' ? ((pl.categories || [])[pl.v[j]] ?? '') : (Number.isFinite(pl.v[j]) ? pl.v[j] : null)) : null]);
      return { columns: ['unit', 'row', 'column', pl ? pl.label : 'value'], rows };
    });
    return { traces, layout, click, hover, unhover, table };
  },
  template: `<div class="som-grid" data-som-grid><PlotlyChart :data="traces" :layout="layout" :height="height" @click="click" @hover="hover" @unhover="unhover"/></div>`,
};
