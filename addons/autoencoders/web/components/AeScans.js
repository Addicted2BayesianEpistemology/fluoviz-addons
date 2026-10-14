// The live training chart, the block that runs a scan (button, progress, queue, answer dimmed when it was made for other settings), and the charts of the scans:
// error against the number of codes, error against roughness (the smoothify L-curve), agreement of the codes between runs, the linear baselines.
import { computed } from 'vue';
import { Icon } from '/static/js/icons.js';
import { PlotlyChart, ChartCard } from '/static/js/plot.js';
import { Callout } from '/static/js/components/Common.js';
import { JobBar } from '/static/js/components/JobBar.js';
import { tokens, themeState, withAlpha, sequentialScale } from '/static/js/theme.js';
import { fmt } from '/static/js/util.js';
import { fp, aeJobs, ctx, requestBody } from './AeState.js';
import { runScan, cancelHeavy } from './AeActions.js';
import { AeVerdict } from './AeCommon.js';

// ---------------------------------------------------------------- the network that is being trained, epoch by epoch
export const AeLive = {
  name: 'AeLive',
  components: { PlotlyChart, ChartCard },
  props: { height: { type: Number, default: 300 }, title: { type: String, default: 'Training so far' }, hook: { type: String, default: 'data-ae-live' } },
  setup(props) {
    const L = computed(() => fp.ae.live);
    const units = computed(() => {
      L.value.seq;
      const m = new Map();
      for (const r of L.value.rows) { const k = `${r.restart}|${r.fold ?? ''}`; if (!m.has(k)) m.set(k, []); m.get(k).push(r); }
      return [...m.values()];
    });
    const traces = computed(() => {
      themeState.version;
      const t = tokens(), us = units.value, out = [];
      us.forEach((rows, i) => {
        const cur = i === us.length - 1 && L.value.status === 'running';
        const x = rows.map((r) => r.epoch + 1);
        if (!cur) {
          out.push({ type: 'scatter', mode: 'lines', x, y: rows.map((r) => r.val), line: { color: withAlpha(t.text3, 0.5), width: 1.2 }, name: 'earlier starts (held-out)', legendgroup: 'earlier', showlegend: i === 0 && us.length > 1,
            hovertemplate: 'epoch %{x}: %{y:.4g}<extra>earlier start</extra>' });
        }
      });
      const rows = us.length ? us[us.length - 1] : [];
      if (rows.length) {
        const x = rows.map((r) => r.epoch + 1);
        out.push({ type: 'scatter', mode: 'lines', x, y: rows.map((r) => r.train), line: { color: t.series[0], width: 2 }, name: 'Training samples', hovertemplate: 'epoch %{x}: %{y:.4g}<extra>training samples</extra>' });
        out.push({ type: 'scatter', mode: 'lines', x, y: rows.map((r) => r.val), line: { color: t.series[5], width: 2 }, name: 'Held-out samples', hovertemplate: 'epoch %{x}: %{y:.4g}<extra>held-out samples</extra>' });
        let b = -1; for (let i = 0; i < rows.length; i++) if (rows[i].best) b = i;
        if (b >= 0) out.push({ type: 'scatter', mode: 'markers', x: [rows[b].epoch + 1], y: [rows[b].val], name: 'Best so far', marker: { symbol: 'star', size: 14, color: t.series[5], line: { color: t.surface, width: 1.5 } }, hovertemplate: 'best so far: epoch %{x}<extra></extra>' });
      }
      return out;
    });
    const logY = computed(() => { let lo = Infinity; for (const r of L.value.rows) { if (r.val > 0 && r.val < lo) lo = r.val; if (r.train > 0 && r.train < lo) lo = r.train; } return lo < Infinity && L.value.rows.every((r) => r.train > 0 && r.val > 0); });
    const layout = computed(() => ({ height: props.height, margin: { l: 62, r: 14, t: 10, b: 66 }, showlegend: true, legend: { orientation: 'h', y: -0.28, x: 0, font: { size: 11 } }, hovermode: 'x unified',
      xaxis: { title: { text: 'Epoch (one pass over the training samples)' } }, yaxis: { title: { text: 'Loss (lower is better)' }, type: logY.value ? 'log' : 'linear' } }));
    const line = computed(() => {
      const l = L.value;
      if (!l.rows.length && l.status !== 'running') return '';
      const last = l.rows.length ? l.rows[l.rows.length - 1] : null;
      const where = l.nRestarts ? `Start ${Math.min(l.restart + 1, l.nRestarts)} of ${l.nRestarts}${l.nFolds ? `, fold ${(l.fold ?? 0) + 1} of ${l.nFolds}` : ''}` : '';
      return [where, last ? `epoch ${last.epoch + 1}` : '', l.best ? (l.best.epoch < 0 ? 'nothing has beaten the PCA start yet' : `best held-out loss ${fmt(l.best.val, 3)} at epoch ${l.best.epoch + 1}`) : ''].filter(Boolean).join(', ');
    });
    const table = computed(() => ({ columns: ['start', 'fold', 'epoch', 'training loss', 'held-out loss'], rows: L.value.rows.map((r) => [r.restart + 1, r.fold === null || r.fold === undefined ? '' : r.fold + 1, r.epoch + 1, r.train, r.val]) }));
    return { L, traces, layout, line, table };
  },
  template: `
  <ChartCard :title="title" :table="table" filename="autoencoder-training-live" help="The loss is the error the network is trying to make small: lower is better. The line for the held-out samples is the honest one: it tells how well the network does on samples it was not trained on. When it stops falling, training stops and the best epoch is kept.">
    <div data-ae-live :data-status="L.status" :data-rows="L.rows.length">
      <PlotlyChart v-if="L.rows.length" :data="traces" :layout="layout" :height="height"/>
      <div class="fp-wait muted" v-else>{{ L.status === 'running' ? 'Preparing the data …' : 'The loss of every epoch appears here while the network is trained.' }}</div>
      <p class="fp-note ae-live-line" data-ae-live-line v-if="line">{{ line }}</p>
    </div>
  </ChartCard>`,
};

// ---------------------------------------------------------------- one scan: the button, the progress, the answer
/** kind: latent | smoothify | stability | baseline.  source: 'request' (the settings on the page) | 'result' (the trained network that is shown; its table is kept in the result).
 *  Scoped slot `default` gets { ans, stale }: the answer (a fresh one, or the one kept in the result) and whether it was made for other settings. */
export const AeScanBlock = {
  name: 'AeScanBlock',
  components: { Icon, Callout, JobBar, AeVerdict },
  props: { kind: { type: String, required: true }, source: { type: String, default: 'request' }, label: { type: String, required: true }, sentence: { type: String, default: '' }, options: { type: Object, default: () => ({}) }, disabled: Boolean, why: { type: String, default: '' } },
  setup(props) {
    const slot = computed(() => fp.ae.scans[props.kind] || null);
    const saved = computed(() => (props.source === 'result' && fp.ae.res ? (fp.ae.res.scansSaved || {})[props.kind] || null : null));
    const ans = computed(() => (slot.value && slot.value.source === props.source && slot.value.ans ? slot.value.ans : saved.value));
    const busy = computed(() => !!(slot.value && slot.value.busy && slot.value.source === props.source));
    const running = computed(() => aeJobs.scan.state.running && busy.value);
    const stale = computed(() => {
      if (props.source === 'result') return false;
      const s = slot.value; if (!s || !s.ans || s.source !== 'request') return false;
      return s.key !== JSON.stringify([props.kind, requestBody(), props.options]);
    });
    const err = computed(() => (slot.value && slot.value.source === props.source ? slot.value.error : ''));
    const issues = computed(() => (props.source === 'request' ? ctx.allIssues() : []));
    const blocked = computed(() => props.disabled || (props.source === 'request' && issues.value.length > 0));
    const reason = computed(() => props.why || (issues.value.length ? issues.value[0].message : ''));
    function run() { runScan(props.kind, { source: props.source, options: props.options }); }
    return { slot, ans, busy, running, stale, err, blocked, reason, run, aeJobs, fp, cancelHeavy };
  },
  template: `
  <div class="ae-scan" :data-ae-scan="kind" :data-state="busy ? 'busy' : (ans ? 'ready' : 'empty')">
    <div class="row wrap">
      <button type="button" class="btn" :data-ae-scan-run="kind" :disabled="busy || blocked" @click="run"><Icon name="play" size="16"/>{{ label }}</button>
      <button type="button" class="btn danger sm" v-if="busy" data-ae-scan-cancel @click="cancelHeavy"><Icon name="stop-circle" size="14"/>Cancel</button>
      <span class="muted xs" v-if="sentence && !busy">{{ sentence }}</span>
      <span class="muted xs" v-if="blocked && reason" data-ae-scan-why>{{ reason }}</span>
    </div>
    <div class="callout info" v-if="busy && slot && slot.queued && !running" data-ae-queued><Icon name="info-circle"/><div class="grow">Another computation is running: this one starts right after it (a newer request replaces this one).</div></div>
    <JobBar v-if="running" :job="aeJobs.scan" :label="label"/>
    <Callout kind="critical" v-if="err" data-ae-scan-error>{{ err }}</Callout>
    <div v-if="ans" :class="{ 'fp-dim': stale || busy }">
      <Callout kind="info" v-if="stale" data-ae-scan-stale>These numbers were computed for other settings: run the scan again.</Callout>
      <slot :ans="ans" :stale="stale"/>
    </div>
  </div>`,
};

// ---------------------------------------------------------------- error against the number of codes
export const AeLatentScanChart = {
  name: 'AeLatentScanChart',
  components: { PlotlyChart, ChartCard },
  props: { ans: { type: Object, required: true }, height: { type: Number, default: 320 } },
  setup(props) {
    const FAM = { ae: 'Autoencoder', vae: 'Variational autoencoder' };
    const traces = computed(() => {
      themeState.version;
      const t = tokens(), a = props.ans, out = [], dims = a.dims || [];
      Object.keys(a.families || {}).forEach((f, i) => {
        const fam = a.families[f], col = t.series[i % 2 === 0 ? 0 : 1];
        out.push({ type: 'scatter', mode: 'lines+markers', x: dims, y: fam.val, error_y: fam.val_se ? { type: 'data', array: fam.val_se, visible: true, color: col, thickness: 1.2, width: 3 } : undefined,
          line: { color: col, width: 2 }, marker: { size: 8, color: col, line: { color: t.surface, width: 2 } }, name: `${FAM[f] || f} (held-out)`, hovertemplate: `${FAM[f] || f}<br>%{x} codes: %{y:.4f}<extra>held-out</extra>` });
        if (fam.train) out.push({ type: 'scatter', mode: 'lines', x: dims, y: fam.train, line: { color: withAlpha(col, 0.55), width: 1.4, dash: 'dot' }, name: `${FAM[f] || f} (training)`, hovertemplate: `${FAM[f] || f}<br>%{x} codes: %{y:.4f}<extra>training</extra>` });
      });
      const b = a.baseline;
      if (b && b.val) out.push({ type: 'scatter', mode: 'lines+markers', x: b.k || dims, y: b.val, line: { color: t.text2, width: 2, dash: 'dash' }, marker: { size: 6, symbol: 'square', color: t.text2, line: { color: t.surface, width: 1.5 } }, name: `${b.name} (held-out)`, hovertemplate: `${b.name}<br>%{x} components: %{y:.4f}<extra>held-out</extra>` });
      return out;
    });
    const layout = computed(() => {
      themeState.version;
      const t = tokens(), rec = props.ans.recommended && props.ans.recommended.latent_dim;
      const L = { height: props.height, margin: { l: 66, r: 14, t: 14, b: 76 }, showlegend: true, legend: { orientation: 'h', y: -0.3, x: 0, font: { size: 11 } }, hovermode: 'x unified',
        xaxis: { title: { text: 'Number of codes' }, dtick: 1 }, yaxis: { title: { text: 'Share of the variation not rebuilt' }, rangemode: 'tozero', tickformat: '.0%' }, shapes: [], annotations: [] };
      if (rec) {
        L.shapes.push({ type: 'line', xref: 'x', yref: 'paper', x0: rec, x1: rec, y0: 0, y1: 1, line: { color: t.series[5], width: 1.6, dash: 'dash' } });
        L.annotations.push({ xref: 'x', yref: 'paper', x: rec, y: 1, xanchor: 'left', yanchor: 'top', text: ` recommended: ${rec}`, showarrow: false, font: { size: 11, color: t.text2, family: t.font } });
      }
      return L;
    });
    const table = computed(() => {
      const a = props.ans, cols = ['codes'], fams = Object.keys(a.families || {});
      for (const f of fams) cols.push(`${FAM[f] || f}: held-out`, `${FAM[f] || f}: held-out SE`, `${FAM[f] || f}: training`);
      if (a.baseline) cols.push(`${a.baseline.name}: held-out`, `${a.baseline.name}: training`);
      return { columns: cols, rows: (a.dims || []).map((d, i) => [d, ...fams.flatMap((f) => [a.families[f].val[i], a.families[f].val_se ? a.families[f].val_se[i] : null, a.families[f].train ? a.families[f].train[i] : null]), ...(a.baseline ? [a.baseline.val ? a.baseline.val[i] : null, a.baseline.train ? a.baseline.train[i] : null] : [])]) };
    });
    return { traces, layout, table };
  },
  template: `<ChartCard title="Error on held-out samples against the number of codes" :table="table" filename="autoencoder-latent-scan"
      help="One small network was trained for each number of codes, and compared with PCA (or functional PCA) with the same number of components. The vertical bars are one standard error. The recommended number is the smallest one within one standard error of the best."><PlotlyChart :data="traces" :layout="layout" :height="height" data-ae-latent-scan-chart/></ChartCard>`,
};

// ---------------------------------------------------------------- the smoothify L-curve
export const AeLCurveChart = {
  name: 'AeLCurveChart',
  components: { PlotlyChart, ChartCard },
  props: { ans: { type: Object, required: true }, height: { type: Number, default: 320 } },
  setup(props) {
    const lab = (v) => (v === 0 ? '0' : fmt(v, 2));
    const traces = computed(() => {
      themeState.version;
      const t = tokens(), a = props.ans, vals = a.values || [], out = [];
      out.push({ type: 'scatter', mode: 'lines+markers+text', x: a.roughness, y: a.error, text: vals.map((v, i) => (i === 0 || i === vals.length - 1 || i % 3 === 0 ? lab(v) : '')), textposition: 'top center', textfont: { size: 10, color: t.text3 },
        error_y: a.error_se ? { type: 'data', array: a.error_se, visible: true, color: withAlpha(t.series[0], 0.6), thickness: 1, width: 2 } : undefined,
        line: { color: t.series[0], width: 2 }, marker: { size: 8, color: t.series[0], line: { color: t.surface, width: 2 } }, name: 'smoothify value', customdata: vals,
        hovertemplate: 'smoothify %{customdata}<br>roughness %{x:.4g}<br>error %{y:.4f}<extra></extra>' });
      if (Number.isFinite(a.roughness_input)) out.push({ type: 'scatter', mode: 'lines', x: [a.roughness_input, a.roughness_input], y: [Math.min(...a.error), Math.max(...a.error)], name: 'roughness of the measured samples', line: { color: t.text3, width: 1.4, dash: 'dot' }, hovertemplate: 'the measured samples: roughness %{x:.3g}<extra></extra>' });
      const k = vals.findIndex((v) => Number.isFinite(a.suggested) && Math.abs(v - a.suggested) <= 1e-9 * Math.max(1, Math.abs(a.suggested)));
      if (k >= 0) out.push({ type: 'scatter', mode: 'markers', x: [a.roughness[k]], y: [a.error[k]], name: 'suggested', marker: { symbol: 'star', size: 16, color: t.series[5], line: { color: t.surface, width: 1.5 } }, hovertemplate: 'suggested: smoothify %{customdata}<extra></extra>', customdata: [a.suggested] });
      return out;
    });
    const layout = computed(() => ({ height: props.height, margin: { l: 66, r: 20, t: 16, b: 70 }, showlegend: true, legend: { orientation: 'h', y: -0.28, x: 0, font: { size: 11 } },
      xaxis: { title: { text: 'Roughness of the rebuilt samples (lower is smoother)' } }, yaxis: { title: { text: 'Share of the variation not rebuilt' }, tickformat: '.1%' } }));
    const table = computed(() => ({ columns: ['smoothify', 'error', 'error SE', 'roughness'], rows: (props.ans.values || []).map((v, i) => [v, props.ans.error[i], props.ans.error_se ? props.ans.error_se[i] : null, props.ans.roughness[i]]) }));
    return { traces, layout, table };
  },
  template: `<ChartCard title="Error against roughness" :table="table" filename="autoencoder-smoothify-scan"
      help="A larger smoothify makes the rebuilt samples smoother (to the left) and the rebuilding error a little larger (up). The numbers on some of the points are the smoothify values (hover over the others). Look for the bend of the curve: the star marks the suggested value, where more smoothing starts to cost more than it gains."><PlotlyChart :data="traces" :layout="layout" :height="height" data-ae-lcurve/></ChartCard>`,
};

// ---------------------------------------------------------------- stability of the codes between runs
export const AeStabilityChart = {
  name: 'AeStabilityChart',
  components: { PlotlyChart, ChartCard },
  props: { ans: { type: Object, required: true }, names: { type: Array, default: () => [] } },
  setup(props) {
    const per = computed(() => (props.ans.alignment && props.ans.alignment.per_code) || []);
    const labels = computed(() => per.value.map((_, i) => props.names[i] || `z${i + 1}`));
    const traces = computed(() => {
      themeState.version;
      const t = tokens();
      return [{ type: 'bar', x: labels.value, y: per.value, marker: { color: t.series[0], line: { color: t.surface, width: 1 } }, hovertemplate: '%{x}: agreement %{y:.2f}<extra></extra>', name: 'agreement' }];
    });
    const layout = computed(() => ({ height: 260, margin: { l: 56, r: 14, t: 12, b: 48 }, xaxis: { title: { text: 'Code' } }, yaxis: { title: { text: 'Agreement between runs (1 = identical)' }, range: [0, 1.02] } }));
    const table = computed(() => ({ columns: ['code', 'agreement'], rows: per.value.map((v, i) => [labels.value[i], v]) }));
    const matTraces = computed(() => {
      themeState.version;
      const m = props.ans.matrix; if (!m) return [];
      const rows = Array.from({ length: m.length ?? 0 }, (_, i) => Array.from(m[i]));
      return [{ type: 'heatmap', z: rows, x: rows.map((_, i) => `run ${i + 1}`), y: rows.map((_, i) => `run ${i + 1}`), zmin: 0, zmax: 1, colorscale: sequentialScale('polimi'), showscale: true, colorbar: { thickness: 10, len: 0.9, outlinewidth: 0, tickfont: { size: 10 } }, hovertemplate: '%{y} and %{x}: %{z:.2f}<extra></extra>' }];
    });
    const matLayout = computed(() => ({ height: 280, margin: { l: 60, r: 14, t: 12, b: 48 }, xaxis: { title: { text: '' } }, yaxis: { autorange: 'reversed' } }));
    const matTable = computed(() => { const m = props.ans.matrix; return m ? { columns: ['run', ...m.map((_, i) => `run ${i + 1}`)], rows: m.map((r, i) => [`run ${i + 1}`, ...Array.from(r)]) } : { columns: [], rows: [] }; });
    return { traces, layout, table, matTraces, matLayout, matTable };
  },
  template: `
  <div class="fp-two">
    <ChartCard title="Agreement of the codes between runs" :table="table" filename="autoencoder-stability" help="The network was trained again from other random starts. For each code of the first run the bar shows how well the best-matching code of the other runs agrees with it (1 = the same numbers up to sign, scale and mixing with other codes).">
      <PlotlyChart :data="traces" :layout="layout" :height="260" data-ae-stability-chart/></ChartCard>
    <ChartCard v-if="matTraces.length" title="Agreement between every two runs" :table="matTable" filename="autoencoder-stability-runs" help="Every cell compares two runs of the training: dark = the two runs found the same codes.">
      <PlotlyChart :data="matTraces" :layout="matLayout" :height="280"/></ChartCard>
  </div>`,
};

// ---------------------------------------------------------------- the linear baselines for every number of components
/** `ans` = { name: 'PCA' | 'Functional PCA', dims, val, val_se, train } (the share of the variation NOT rebuilt on held-out / training samples); `network` = { k, val } puts this network on the chart. */
export const AeBaselineChart = {
  name: 'AeBaselineChart',
  components: { PlotlyChart, ChartCard },
  props: { ans: { type: Object, required: true }, network: { type: Object, default: null } },
  setup(props) {
    const traces = computed(() => {
      themeState.version;
      const t = tokens(), a = props.ans, dims = a.dims || [], out = [], name = a.name || 'PCA';
      out.push({ type: 'scatter', mode: 'lines+markers', x: dims, y: a.val, error_y: a.val_se ? { type: 'data', array: a.val_se, visible: true, color: t.text2, thickness: 1, width: 3 } : undefined,
        line: { color: t.text2, width: 2, dash: 'dash' }, marker: { size: 6, symbol: 'square', color: t.text2, line: { color: t.surface, width: 1.5 } }, name: `${name} (held-out)`, hovertemplate: `${name}<br>%{x} components: %{y:.4f}<extra>held-out</extra>` });
      if (a.train) out.push({ type: 'scatter', mode: 'lines', x: dims, y: a.train, line: { color: withAlpha(t.text2, 0.5), width: 1.4, dash: 'dot' }, name: `${name} (training)`, hovertemplate: `${name}<br>%{x} components: %{y:.4f}<extra>training</extra>` });
      if (props.network && Number.isFinite(props.network.val)) out.push({ type: 'scatter', mode: 'markers', x: [props.network.k], y: [props.network.val], marker: { symbol: 'star', size: 16, color: t.series[5], line: { color: t.surface, width: 1.5 } }, name: 'This network (held-out)', hovertemplate: 'this network, %{x} codes: %{y:.4f}<extra></extra>' });
      return out;
    });
    const layout = computed(() => ({ height: 300, margin: { l: 66, r: 14, t: 12, b: 70 }, showlegend: true, legend: { orientation: 'h', y: -0.28, x: 0, font: { size: 11 } }, hovermode: 'x unified',
      xaxis: { title: { text: 'Number of components (or codes)' }, dtick: 1 }, yaxis: { title: { text: 'Share of the variation not rebuilt' }, rangemode: 'tozero', tickformat: '.0%' } }));
    const table = computed(() => { const a = props.ans; return { columns: ['components', `${a.name || 'PCA'}: held-out`, 'standard error', `${a.name || 'PCA'}: training`], rows: (a.dims || []).map((d, i) => [d, a.val[i], a.val_se ? a.val_se[i] : null, a.train ? a.train[i] : null]) }; });
    return { traces, layout, table };
  },
  template: `<ChartCard :title="'Error of ' + (ans.name || 'PCA') + ' for every number of components'" :table="table" filename="autoencoder-baseline"
      help="The linear method, without any training: the share of the variation of held-out samples that it does not rebuild with 1, 2, 3 … components. The star is this network. A network that is not below the line is not better than the linear method."><PlotlyChart :data="traces" :layout="layout" :height="300" data-ae-baseline-chart/></ChartCard>`,
};
