// Step 8 of the Autoencoders page: the checks, each with a verdict in words: Training, Quality (against PCA / functional PCA), Latent dimensions, Stability, Smoothness.
import { computed } from 'vue';
import { Icon } from '/static/js/icons.js';
import { Field, NumField, Switch, Seg, Callout } from '/static/js/components/Common.js';
import { PlotlyChart, ChartCard } from '/static/js/plot.js';
import { FpcaFrame } from '/static/js/components/FpcaStepper.js';
import { FpcaExplain, FpcaPicture } from '/static/js/components/FpcaCommon.js';
import { vec, image, mat, maskArr } from '/static/js/components/DecompCommon.js';
import { tokens, themeState, withAlpha, divergingScale } from '/static/js/theme.js';
import { prefs } from '/static/js/prefs.js';
import { fmt } from '/static/js/util.js';
import { fp, ctx } from './AeState.js';
import { AeTabs, AeVerdict, AeStat, AeHistogram, AeSection, pct, nTxt, shortUnits, unitsOf } from './AeCommon.js';
import { AeScanBlock, AeLatentScanChart, AeLCurveChart, AeStabilityChart, AeBaselineChart } from './AeScans.js';
import { AeGate, AeStaleNote, baselineInfo, verdictLevel } from './AeResults.js';
import { histogram } from './AeSamplesTab.js';

const entries = (o) => (o && typeof o === 'object' && !Array.isArray(o) ? Object.entries(o) : o === null || o === undefined ? [] : [['', o]]);

// ---------------------------------------------------------------- Training
export const AeCheckTraining = {
  name: 'AeCheckTraining',
  components: { PlotlyChart, ChartCard, Seg, AeVerdict, AeStat },
  setup() {
    const A = computed(() => fp.ae), R = computed(() => fp.ae.res), V = computed(() => fp.ae.lossView);
    const T = computed(() => (R.value ? R.value.p.training || {} : {}));
    const H = computed(() => T.value.history || null);
    const chosen = computed(() => { const rs = T.value.restarts || []; const k = rs.findIndex((r) => r.chosen); return k >= 0 ? k : 0; });
    const groups = computed(() => {
      const h = H.value; if (!h || !h.epoch) return [];
      const n = h.epoch.length, m = new Map();
      for (let i = 0; i < n; i++) { const key = `${h.restart[i]}|${h.fold && h.fold[i] !== null && h.fold[i] !== undefined && h.fold[i] >= 0 ? h.fold[i] : ''}`; if (!m.has(key)) m.set(key, { restart: h.restart[i], fold: h.fold ? h.fold[i] : null, idx: [] }); m.get(key).idx.push(i); }
      return [...m.values()];
    });
    const OPTIONS = [{ value: 'total', label: 'Total' }, { value: 'rec', label: 'Reconstruction' }, { value: 'kl', label: 'KL' }, { value: 'smooth', label: 'Smoothify' }, { value: 'beta', label: 'β and σ²' }];
    const hasField = (f) => !!(H.value && H.value[f] && H.value[f].some((v) => v !== null && v !== undefined && Number.isFinite(v)));
    const options = computed(() => OPTIONS.filter((o) => o.value === 'total' || o.value === 'rec' || (o.value === 'kl' && hasField('kl')) || (o.value === 'smooth' && hasField('smooth') && H.value.smooth.some((v) => v > 0)) || (o.value === 'beta' && (hasField('beta') || hasField('sigma2')))));
    const traces = computed(() => {
      themeState.version;
      const h = H.value, t = tokens(), out = [];
      if (!h) return out;
      const what = V.value.what, g = groups.value, ch = chosen.value;
      const ep = (idx) => idx.map((i) => h.epoch[i] + 1);
      const line = (name, field, idx, color, width, dash, extra = {}) => out.push({ type: 'scatter', mode: 'lines', x: ep(idx), y: idx.map((i) => h[field][i]), name, line: { color, width, dash }, hovertemplate: `epoch %{x}: %{y:.4g}<extra>${name}</extra>`, ...extra });
      g.forEach((u) => {
        const main = u.restart === ch, faint = withAlpha(t.text3, 0.4);
        if (what === 'total') {
          if (!main) { line('earlier / other starts (held-out)', 'val', u.idx, faint, 1, 'solid', { legendgroup: 'other', showlegend: u === g.find((x) => x.restart !== ch) }); return; }
          line('Training samples', 'train', u.idx, t.series[0], 2, 'solid', { legendgroup: 'train', showlegend: u === g.find((x) => x.restart === ch) });
          line('Held-out samples', 'val', u.idx, t.series[5], 2, 'solid', { legendgroup: 'val', showlegend: u === g.find((x) => x.restart === ch) });
        } else if (main && what === 'rec') {
          line('Training samples', 'rec', u.idx, t.series[0], 2, 'solid', { showlegend: true });
          if (h.rec_val) line('Held-out samples', 'rec_val', u.idx, t.series[5], 2, 'solid', { showlegend: true });
        } else if (main) {
          const f = what;
          if (what === 'beta') { line('β (weight of the KL term)', 'beta', u.idx, t.series[0], 2, 'solid'); }
          else line({ rec: 'Reconstruction', kl: 'KL', smooth: 'Smoothify term' }[what], f, u.idx, t.series[0], 2, 'solid');
        }
      });
      if (T.value.best_epoch !== undefined && T.value.best_epoch !== null && T.value.best_epoch >= 0) out.push({ type: 'scatter', mode: 'lines', x: [T.value.best_epoch + 1, T.value.best_epoch + 1], y: [null, null], name: 'Best epoch', line: { color: t.text2, width: 1.4, dash: 'dash' }, hoverinfo: 'skip' });
      return out;
    });
    const sigmaTraces = computed(() => {
      themeState.version;
      const h = H.value, t = tokens(); if (!h || !hasField('sigma2')) return [];
      const idx = groups.value.filter((u) => u.restart === chosen.value).flatMap((u) => u.idx);
      return [{ type: 'scatter', mode: 'lines', x: idx.map((i) => h.epoch[i] + 1), y: idx.map((i) => h.sigma2[i]), name: 'σ² (noise level of the data)', line: { color: t.series[1], width: 2 }, hovertemplate: 'epoch %{x}: %{y:.4g}<extra>σ²</extra>' }];
    });
    const layout = computed(() => {
      themeState.version;
      const t = tokens(), what = V.value.what;
      const L = { height: 320, margin: { l: 66, r: 14, t: 12, b: 70 }, showlegend: true, legend: { orientation: 'h', y: -0.26, x: 0, font: { size: 11 } }, hovermode: 'x unified', xaxis: { title: { text: 'Epoch' } },
        yaxis: { title: { text: { total: 'Loss (lower is better)', rec: 'Share of the variation not rebuilt', kl: 'KL part of the loss (nats)', smooth: 'Smoothify part of the loss', beta: 'β' }[what] }, type: what === 'total' ? 'log' : 'linear', rangemode: what === 'total' ? undefined : 'tozero' }, shapes: [] };
      if (T.value.best_epoch !== undefined && T.value.best_epoch !== null && T.value.best_epoch >= 0) L.shapes.push({ type: 'line', xref: 'x', yref: 'paper', x0: T.value.best_epoch + 1, x1: T.value.best_epoch + 1, y0: 0, y1: 1, line: { color: t.text2, width: 1.4, dash: 'dash' } });
      return L;
    });
    const sigmaLayout = computed(() => ({ height: 240, margin: { l: 66, r: 14, t: 12, b: 50 }, xaxis: { title: { text: 'Epoch' } }, yaxis: { title: { text: 'σ²' }, type: 'log' }, showlegend: false }));
    const table = computed(() => { const h = H.value; return h ? { columns: ['start', 'fold', 'epoch', 'training', 'held-out', 'reconstruction', 'KL', 'smoothify', 'β', 'σ²'], rows: h.epoch.map((e, i) => [h.restart[i] + 1, h.fold && h.fold[i] !== null && h.fold[i] >= 0 ? h.fold[i] + 1 : '', e + 1, h.train[i], h.val[i], h.rec ? h.rec[i] : null, h.kl ? h.kl[i] : null, h.smooth ? h.smooth[i] : null, h.beta ? h.beta[i] : null, h.sigma2 ? h.sigma2[i] : null]) } : { columns: [], rows: [] }; });
    const level = computed(() => { const l = T.value.overfit ? T.value.overfit.level : null; return l === 'clear' ? 'warning' : l === 'none' ? 'good' : 'info'; });
    return { A, R, V, T, H, options, traces, layout, sigmaTraces, sigmaLayout, table, level, fmt, nTxt, pct };
  },
  template: `
  <div class="col gap-12" data-ae-check-training>
    <AeVerdict :level="level" v-if="T.verdict || (T.overfit && T.overfit.words)"><span data-ae-training-verdict>{{ T.verdict || T.overfit.words }}</span> <span v-if="T.verdict && T.overfit && T.overfit.words && !T.verdict.includes(T.overfit.words)">{{ T.overfit.words }}</span></AeVerdict>
    <div class="fp-cards4"><AeStat label="Best epoch" :value="T.best_epoch === -1 ? 'the PCA start' : (T.best_epoch !== undefined && T.best_epoch !== null ? T.best_epoch + 1 : '–')" :sub="'after ' + nTxt(T.epochs_run) + ' epochs' + (T.stopped === 'early' ? ' (stopped early)' : T.stopped === 'time' ? ' (time limit)' : T.stopped === 'max' ? ' (the maximum)' : '')"/>
      <AeStat label="Random starts" :value="(T.restarts || []).length" :sub="T.spread && T.spread.words ? T.spread.words : 'the best one was kept'"/></div>
    <ChartCard title="Loss during training" :table="table" filename="autoencoder-loss" help="The loss is the error the network made small while it was trained. The held-out line is the honest one. The dashed vertical line is the epoch whose network was kept.">
      <template #actions><Seg small v-model="V.what" :options="options" data-ae-loss-what/></template>
      <PlotlyChart v-if="traces.length" :data="traces" :layout="layout" :height="320" data-ae-loss-chart/>
      <div class="fp-wait muted" v-else>The history of the training is not kept in this result.</div>
    </ChartCard>
    <ChartCard v-if="V.what === 'beta' && sigmaTraces.length" title="Noise level of the data (σ²) during training" filename="autoencoder-sigma"><PlotlyChart :data="sigmaTraces" :layout="sigmaLayout" :height="240"/></ChartCard>
    <div class="card" data-ae-restarts-table v-if="T.restarts && T.restarts.length"><div class="card-head"><h3>Random starts</h3></div><div class="card-body">
      <div class="table-wrap"><table class="table"><thead><tr><th>Start</th><th>Seed</th><th class="num">Variance explained (held-out)</th><th class="num">Best epoch</th><th>Kept</th><th class="num">Epochs trained</th></tr></thead>
        <tbody><tr v-for="(r, i) in T.restarts" :key="i"><td>{{ i + 1 }}</td><td>{{ r.seed }}</td><td class="num">{{ pct(r.val) }}</td><td class="num">{{ r.best_epoch === -1 ? 'PCA start' : (r.best_epoch !== undefined && r.best_epoch !== null ? r.best_epoch + 1 : '–') }}</td><td>{{ r.chosen ? 'kept' : '' }}</td><td class="num">{{ r.epochs_run !== undefined && r.epochs_run !== null ? nTxt(r.epochs_run) : '–' }}{{ r.stopped === 'early' ? ' (stopped early)' : r.stopped === 'time' ? ' (time limit)' : '' }}</td></tr></tbody></table></div>
      <p class="fp-note muted mt-8">The network was trained from several random starts and the one that rebuilt the held-out samples best was kept. Starts that end far apart say that the result depends on luck: look at the stability check.</p></div></div>
  </div>`,
};

// ---------------------------------------------------------------- Quality
export const AeCheckQuality = {
  name: 'AeCheckQuality',
  components: { Icon, Callout, PlotlyChart, ChartCard, FpcaPicture, AeVerdict, AeStat, AeHistogram, AeScanBlock, AeBaselineChart },
  setup() {
    const A = computed(() => fp.ae), R = computed(() => fp.ae.res);
    const Q = computed(() => (R.value ? R.value.p.quality || {} : {}));
    const base = computed(() => baselineInfo(Q.value));
    const T = computed(() => (R.value ? R.value.p.training || {} : {}));
    const blockRows = computed(() => [['Overall', Q.value.r2_val, Q.value.r2_train, Q.value.rmse_val, ''], ...entries(Q.value.r2_by_block).map(([k, v]) => [k === 'fluorescence' ? 'Fluorescence' : k === 'absorbance' ? 'Absorbance' : k, v && v.val, v && v.train, (Q.value.rmse_by_block || {})[k], R.value ? shortUnits(unitsOf(R.value.p.units, k)) : ''])]);
    const hist = computed(() => (R.value ? histogram(R.value.err, R.value.held, 24, true) : null));
    const limit = computed(() => (R.value && R.value.p.recon ? R.value.p.recon.limit : null));
    const level = computed(() => verdictLevel(Q.value, T.value));
    // maps of where the network rebuilds well, and the typical size of the difference
    const surf = computed(() => {
      const r = R.value, q = Q.value; if (!r || !r.p.ex || !q.local_r2 || !q.local_r2.fluorescence) return null;
      const axes = { ex: vec(r.p.ex), em: vec(r.p.em) }, mask = r.p.mask ? maskArr(r.p.mask) : null;
      const rr = image(q.local_r2.fluorescence), rms = q.residual_rms && q.residual_rms.fluorescence ? image(q.residual_rms.fluorescence) : null;
      let hi = 0; if (rms) for (let i = 0; i < rms.length; i++) if (rms[i] > hi) hi = rms[i];
      return { axes, mask, r2: rr, rms, hi: hi || 1, units: unitsOf(r.p.units) };
    });
    const curve = computed(() => {
      const r = R.value, q = Q.value; if (!r || !r.p.curves || !q.local_r2 || !q.local_r2.absorbance) return null;
      return { x: vec(r.p.curves.x), r2: vec(q.local_r2.absorbance), rms: q.residual_rms && q.residual_rms.absorbance ? vec(q.residual_rms.absorbance) : null };
    });
    const netPoint = computed(() => (R.value && Number.isFinite(Q.value.r2_val) ? { k: R.value.d, val: 1 - Q.value.r2_val } : null));
    function look(id) { A.value.sample = id; A.value.tab7 = 'samples'; ctx.goStep(7); }
    return { A, R, Q, base, blockRows, hist, limit, level, surf, curve, netPoint, look, pct, fmt };
  },
  template: `
  <div class="col gap-12" data-ae-check-quality>
    <AeVerdict :level="level" v-if="Q.verdict"><span data-ae-quality-verdict>{{ Q.verdict }}</span></AeVerdict>
    <div class="fp-cards4">
      <AeStat label="Variance explained on held-out samples" :value="pct(Q.r2_val)" :sub="'training samples: ' + pct(Q.r2_train)"/>
      <AeStat label="Error (RMSE) on held-out samples" :value="fmt(Q.rmse_val, 3)" sub="in the units of the data, after the pre-treatment"/>
    </div>
    <div class="card" data-ae-baseline>
      <div class="card-head"><h3>Is the nonlinearity worth it?</h3></div>
      <div class="card-body col gap-12" v-if="base">
        <AeVerdict :level="level"><span data-ae-baseline-verdict>{{ Q.baseline.verdict }}</span></AeVerdict>
        <div class="table-wrap"><table class="table"><thead><tr><th></th><th class="num">Held-out samples</th><th class="num">Training samples</th></tr></thead>
          <tbody><tr><td><b>This network ({{ R.d }} {{ R.d === 1 ? 'code' : 'codes' }})</b></td><td class="num">{{ pct(Q.r2_val) }}</td><td class="num">{{ pct(Q.r2_train) }}</td></tr>
            <tr><td>{{ base.name }} ({{ Q.baseline.k }} {{ Q.baseline.k === 1 ? 'component' : 'components' }})</td><td class="num">{{ pct(Q.baseline.r2_val) }}</td><td class="num">{{ pct(Q.baseline.r2_train) }}</td></tr></tbody></table></div>
        <p class="fp-note muted">A linear autoencoder (activation <i>linear</i>) should come out about equal to PCA: that is the sanity check of the whole pipeline.</p>
      </div>
      <div class="card-body" v-else><p class="fp-note muted">No comparison with PCA is stored in this result.</p></div>
    </div>
    <AeScanBlock kind="baseline" source="result" label="Compare with PCA for every number of codes" sentence="The linear methods for 1, 2, 3 … components, no training needed.">
      <template #default="{ ans }"><AeBaselineChart :ans="ans" :network="netPoint"/><p class="fp-note" v-if="ans.words">{{ ans.words }}</p></template>
    </AeScanBlock>
    <div class="card" data-ae-worst v-if="Q.worst && Q.worst.length"><div class="card-head"><h3>Worst rebuilt samples</h3></div><div class="card-body">
      <div class="table-wrap"><table class="table"><thead><tr><th>Sample</th><th class="num">Not rebuilt</th><th>Used for</th><th></th></tr></thead>
        <tbody><tr v-for="w in Q.worst" :key="w.id"><td><b>{{ w.id }}</b></td><td class="num">{{ pct(w.error, 1) }}</td><td>{{ w.heldout ? 'held out' : 'training' }}</td><td><button type="button" class="btn xs" @click="look(w.id)">Look at it</button></td></tr></tbody></table></div></div></div>
    <AeHistogram v-if="hist" title="Rebuilding error of the samples" :edges="hist.edges" :counts="hist.counts" :held="hist.held" :limit="limit" limit-label="limit" x-title="Share of the variation not rebuilt (logarithmic scale)" log-x filename="autoencoder-error-distribution"
                 help="One bar per range of errors. The dashed line is the limit above which a sample is called poorly described (the 99 % point of the held-out errors)."/>
    <div class="fp-triple" v-if="surf">
      <div class="fp-tri-item" data-ae-localr2><div class="fp-gal-t">Where the network rebuilds well (R² at each pixel)</div><FpcaPicture surface :axes="surf.axes" :z="surf.r2" :mask="surf.mask" :zrange="[0, 1]" units="R²" :height="260" scale-bar title="Where the network rebuilds well"/></div>
      <div class="fp-tri-item" v-if="surf.rms"><div class="fp-gal-t">Typical size of the difference</div><FpcaPicture surface :axes="surf.axes" :z="surf.rms" :mask="surf.mask" :zrange="[0, surf.hi]" :units="surf.units" :height="260" scale-bar title="Typical size of the difference"/></div>
    </div>
    <div class="fp-triple" v-if="curve">
      <FpcaPicture :axes="{ wl: curve.x }" :z="curve.r2" :height="220" curve-label="R² at each wavelength" title="Where the network rebuilds the absorbance well" :compact="false"/>
      <FpcaPicture v-if="curve.rms" :axes="{ wl: curve.x }" :z="curve.rms" :height="220" curve-label="Typical size of the difference" title="Typical size of the difference (absorbance)" :compact="false"/>
    </div>
  </div>`,
};

// ---------------------------------------------------------------- Latent dimensions
export const AeCheckLatent = {
  name: 'AeCheckLatent',
  components: { Icon, Field, NumField, Switch, Callout, PlotlyChart, ChartCard, AeVerdict, AeStat, AeSection, AeScanBlock, AeLatentScanChart },
  setup() {
    const A = computed(() => fp.ae), R = computed(() => fp.ae.res), V = computed(() => fp.ae.latScanView);
    const Lt = computed(() => (R.value ? R.value.p.latent || {} : {}));
    const H = computed(() => Lt.value.health || Lt.value);          // (the fields of the health are directly in `latent`)
    const names = computed(() => (R.value ? R.value.codeNames : []));
    const klTraces = computed(() => {
      themeState.version;
      const h = H.value, t = tokens(); if (!h.kl_per_dim) return [];
      const inactive = h.inactive || [];
      return [{ type: 'bar', x: names.value, y: Array.from(h.kl_per_dim), marker: { color: h.kl_per_dim.map((_, j) => (inactive[j] ? t.muted : t.series[0])), line: { color: t.surface, width: 1 } }, hovertemplate: '%{x}: KL %{y:.3f} nats<extra></extra>', name: 'KL' }];
    });
    const klLayout = computed(() => ({ height: 240, margin: { l: 60, r: 14, t: 12, b: 48 }, xaxis: { title: { text: 'Code' } }, yaxis: { title: { text: 'KL (nats): information the code carries' }, rangemode: 'tozero' } }));
    const klTable = computed(() => ({ columns: ['code', 'KL (nats)', 'inactive'], rows: names.value.map((n, j) => [n, H.value.kl_per_dim ? H.value.kl_per_dim[j] : null, (H.value.inactive || [])[j] ? 'yes' : 'no']) }));
    const corrTraces = computed(() => {
      themeState.version;
      const C = Lt.value.correlation; if (!C) return [];
      const M = mat(C);
      const rows = names.value.map((_, i) => names.value.map((__, j) => M.data[i * M.k + j]));
      return [{ type: 'heatmap', x: names.value, y: names.value, z: rows, zmin: -1, zmax: 1, colorscale: divergingScale(prefs.div), xgap: 2, ygap: 2, colorbar: { thickness: 10, len: 0.9, outlinewidth: 0, tickfont: { size: 10 } }, hovertemplate: '%{y} and %{x}: correlation %{z:.2f}<extra></extra>' }];
    });
    const corrLayout = computed(() => ({ height: Math.max(260, 60 + 34 * names.value.length), margin: { l: 56, r: 14, t: 10, b: 48 }, yaxis: { autorange: 'reversed' }, xaxis: { side: 'bottom' } }));
    const corrTable = computed(() => { const C = Lt.value.correlation; if (!C) return { columns: [], rows: [] }; const M = mat(C); return { columns: ['code', ...names.value], rows: names.value.map((n, i) => [n, ...names.value.map((_, j) => M.data[i * M.k + j])]) }; });
    const typMid = computed(() => { const t = Lt.value.typicality; return t ? (t.median ?? t.mean) : NaN; });
    const level = computed(() => ((H.value.inactive || []).some(Boolean) || (H.value.unused || []).some(Boolean) ? 'warning' : 'info'));
    const options = computed(() => { const v = V.value, d = R.value ? R.value.d : 4; const max = Math.max(2, Math.min(32, Math.round(Number(v.max_dim) || d + 3))); return { dims: Array.from({ length: max }, (_, i) => i + 1), families: v.both ? ['ae', 'vae'] : [R.value ? R.value.family : 'ae'] }; });
    return { A, R, V, Lt, H, names, klTraces, klLayout, klTable, corrTraces, corrLayout, corrTable, level, options, typMid, nTxt, fmt, pct };
  },
  template: `
  <div class="col gap-12" data-ae-check-latent>
    <AeVerdict :level="level" v-if="H.words"><span data-ae-latent-verdict>{{ H.words }}</span></AeVerdict>
    <div class="fp-cards4">
      <AeStat label="Effective number of codes" :value="Number.isFinite(H.effective_dim) ? fmt(H.effective_dim, 3) : '–'" :sub="'of ' + (R ? R.d : '') + ' codes'" help="How many codes the samples really use (the participation ratio of their variances). Much smaller than the number of codes: some codes are not needed."/>
      <AeStat v-if="Lt.typicality" label="Typical typicality of the codes" :value="Number.isFinite(typMid) ? fmt(typMid, 2) : '–'" :sub="Lt.typicality.words || ''" help="The typicality of a code is its p-value among the codes of the training samples: 1 = in the middle, near 0 = far out."/>
    </div>
    <ChartCard v-if="klTraces.length" title="KL of each code" :table="klTable" filename="autoencoder-kl-per-code" help="How much information each code of the variational autoencoder carries. A code with almost none (grey) is inactive: the network does not use it, so the number of codes could be smaller.">
      <PlotlyChart :data="klTraces" :layout="klLayout" :height="240" data-ae-kl-chart/></ChartCard>
    <ChartCard v-if="corrTraces.length" title="Correlation between the codes" :table="corrTable" filename="autoencoder-code-correlation" help="Codes that are strongly correlated carry the same information twice. A good set of codes is mostly independent: pale off-diagonal cells.">
      <PlotlyChart :data="corrTraces" :layout="corrLayout" :height="corrLayout.height" data-ae-corr-chart/></ChartCard>
    <p class="fp-note" v-if="Lt.coverage && Lt.coverage.words">{{ Lt.coverage.words }}</p>
    <AeSection title="Latent-dimension scan" sentence="Trains small networks for 1, 2, 3 … codes and compares them with PCA or functional PCA, to see how many codes are worth having. It takes minutes and does not change this network.">
      <div class="row wrap" style="gap:12px 22px;align-items:flex-end">
        <Field label="Largest number of codes" help="The scan tries 1 up to this number of codes. 2 to 32."><NumField v-model="V.max_dim" :min="2" :max="32" :step="1"/></Field>
        <Switch v-model="V.both" label="Also try the other type of network"/>
      </div>
      <AeScanBlock kind="latent" source="result" label="Latent-dimension scan" :options="options">
        <template #default="{ ans }"><AeLatentScanChart :ans="ans"/><p class="fp-note" v-if="ans.words">{{ ans.words }}</p></template>
      </AeScanBlock>
    </AeSection>
  </div>`,
};

// ---------------------------------------------------------------- Stability
export const AeCheckStability = {
  name: 'AeCheckStability',
  components: { Field, NumField, Switch, AeVerdict, AeScanBlock, AeStabilityChart, AeStat },
  setup() {
    const R = computed(() => fp.ae.res), V = computed(() => fp.ae.stabView);
    const options = computed(() => ({ n_runs: Math.round(Number(V.value.n_runs) || 5), bootstrap: !!V.value.bootstrap }));
    return { R, V, options, fmt };
  },
  template: `
  <div class="col gap-12" data-ae-check-stability>
    <p class="fp-note">Are the codes a property of your data, or of the random numbers the training started from? The network is trained again from other starts (and, if you ask, on resampled samples) and the codes of the runs are compared: they agree when they describe the same directions of variation, whatever their order, sign or scale.</p>
    <div class="row wrap" style="gap:12px 22px;align-items:flex-end">
      <Field label="Number of runs" help="How many times the network is trained again. 2 to 10. More runs take longer."><NumField v-model="V.n_runs" :min="2" :max="10" :step="1" data-ae-stab-runs/></Field>
      <Switch v-model="V.bootstrap" label="Resample the samples (bootstrap)"/>
    </div>
    <AeScanBlock kind="stability" source="result" label="Check the stability" :options="options" :sentence="'Trains the network ' + options.n_runs + ' more times: minutes.'">
      <template #default="{ ans }">
        <AeVerdict :level="ans.reproducible === false || (ans.reproducible === undefined && /are not reproducible/i.test(ans.verdict || '')) ? 'warning' : 'info'"><span data-ae-stability-verdict>{{ ans.verdict }}</span></AeVerdict>
        <div class="fp-cards4" v-if="ans.alignment"><AeStat label="Mean agreement" :value="fmt(ans.alignment.mean, 2)" sub="1 = identical codes"/><AeStat label="Worst code" :value="fmt(ans.alignment.min, 2)" sub="the least reproducible code"/>
          <AeStat v-if="Number.isFinite(ans.recon_spread)" label="Spread of the rebuilt samples" :value="fmt(ans.recon_spread, 3)" sub="between the runs"/></div>
        <AeStabilityChart :ans="ans" :names="R ? R.codeNames : []"/>
        <p class="fp-note" v-if="ans.words">{{ ans.words }}</p>
      </template>
    </AeScanBlock>
  </div>`,
};

// ---------------------------------------------------------------- Smoothness
export const AeCheckSmooth = {
  name: 'AeCheckSmooth',
  components: { Icon, Callout, PlotlyChart, ChartCard, AeVerdict, AeStat, AeSection, AeScanBlock, AeLCurveChart },
  setup() {
    const A = computed(() => fp.ae), R = computed(() => fp.ae.res);
    const S = computed(() => (R.value ? R.value.p.smoothness || null : null));
    const NICE_NAME = (k) => (k === 'fluorescence' ? 'Fluorescence' : k === 'absorbance' ? 'Absorbance' : (k || 'Samples'));
    const rows = computed(() => {
      const s = S.value; if (!s) return [];
      if (Array.isArray(s.blocks)) return s.blocks.map((b) => ({ k: b.name, label: NICE_NAME(b.name), measured: b.input, rebuilt: b.reconstruction, ratio: b.ratio }));
      const inp = entries(s.roughness_input);
      return entries(s.roughness_recon).map(([k, v]) => ({ k, label: NICE_NAME(k), measured: (inp.find(([a]) => a === k) || [null, null])[1], rebuilt: v }));          // (the older shape of the answer)
    });
    const traces = computed(() => {
      themeState.version;
      const t = tokens(), r = rows.value; if (!r.length) return [];
      return [{ type: 'bar', x: r.map((x) => x.label), y: r.map((x) => x.measured), name: 'Measured samples', marker: { color: t.series[1], line: { color: t.surface, width: 1 } }, hovertemplate: '%{x}, measured: %{y:.4g}<extra></extra>' },
        { type: 'bar', x: r.map((x) => x.label), y: r.map((x) => x.rebuilt), name: 'Rebuilt samples', marker: { color: t.series[0], line: { color: t.surface, width: 1 } }, hovertemplate: '%{x}, rebuilt: %{y:.4g}<extra></extra>' }];
    });
    const layout = computed(() => ({ height: 260, margin: { l: 62, r: 14, t: 12, b: 70 }, barmode: 'group', showlegend: true, legend: { orientation: 'h', y: -0.28, x: 0 }, yaxis: { title: { text: 'Roughness (lower is smoother)' }, rangemode: 'tozero' } }));
    const table = computed(() => ({ columns: ['variable', 'measured samples', 'rebuilt samples'], rows: rows.value.map((x) => [x.label, x.measured, x.rebuilt]) }));
    function useSuggested(s) { fp.spec.net.smoothify = s; ctx.goStep(6); }
    return { A, R, S, rows, traces, layout, table, useSuggested, fmt };
  },
  template: `
  <div class="col gap-12" data-ae-check-smoothness>
    <AeVerdict level="info" v-if="S && S.words"><span data-ae-smoothness-verdict>{{ S.words }}</span></AeVerdict>
    <Callout kind="info" v-else-if="!S">The roughness of the samples is measured for the functional version (a basis with a smoothness measure). For the pixel version smoothify only acts as a plain penalty on the size of the values.</Callout>
    <div class="fp-cards4" v-if="S"><AeStat label="Smoothify of this network" :value="S.smoothify ? fmt(S.smoothify, 3) : '0 (off)'" :sub="S.metric === 'ridge' ? 'plain size of the coefficients' : 'smoothness of the representation'"/></div>
    <ChartCard v-if="traces.length" title="Roughness of the samples and of the rebuilt samples" :table="table" filename="autoencoder-roughness" help="How rough the measured (smoothed) samples are and how rough the network's rebuilt samples are. Rebuilt samples that are much rougher than the measured ones are a sign of noise being learned; a larger smoothify lowers it.">
      <PlotlyChart :data="traces" :layout="layout" :height="260"/></ChartCard>
    <AeSection title="Smoothify scan" sentence="Trains the network for several smoothify values (0 to 10) and draws the rebuilding error against the roughness. It takes minutes and does not change this network.">
      <AeScanBlock kind="smoothify" source="result" label="Smoothify scan">
        <template #default="{ ans }">
          <AeLCurveChart :ans="ans"/><p class="fp-note" v-if="ans.words">{{ ans.words }}</p>
          <div class="row wrap" v-if="Number.isFinite(ans.suggested)"><span class="fp-note"><b>Suggested: {{ fmt(ans.suggested, 3) }}.</b></span>
            <button type="button" class="btn sm" data-ae-use-suggested @click="useSuggested(ans.suggested)">Use the suggested value</button><span class="muted xs">Sets smoothify in step 5 and goes to step 6, where you train again.</span></div>
        </template>
      </AeScanBlock>
    </AeSection>
  </div>`,
};

// ---------------------------------------------------------------- the step
export const AeChecks = {
  name: 'AeChecks',
  components: { FpcaFrame, FpcaExplain, AeTabs, AeGate, AeStaleNote, AeCheckTraining, AeCheckQuality, AeCheckLatent, AeCheckStability, AeCheckSmooth },
  setup() {
    const A = computed(() => fp.ae), R = computed(() => fp.ae.res);
    const ready = computed(() => !!(R.value && A.value.resRid));
    const tabs = [{ id: 'training', label: 'Training' }, { id: 'quality', label: 'Quality' }, { id: 'latent', label: 'Latent dimensions' }, { id: 'stability', label: 'Stability' }, { id: 'smoothness', label: 'Smoothness' }];
    return { A, ready, tabs };
  },
  template: `
  <FpcaFrame :n="8" next-label="Use the network" :no-next="!ready" :next-disabled="!ready">
    <FpcaExplain does="Asks whether the network can be trusted. Training: did it learn without memorising? Quality: how much of the variation does it rebuild on samples it has not seen, and is it better than PCA? Latent dimensions: are all the codes used, and how many are needed? Stability: do other random starts give the same codes? Smoothness: are the rebuilt samples smooth?"
                 check="Every number here is computed on held-out samples, never on the samples the network was trained on. Read the sentence at the top of each tab first: it says in words whether the check is passed. With only a few dozen samples keep an eye on the spread between random starts."/>
    <AeGate/>
    <template v-if="ready">
      <AeStaleNote/>
      <AeTabs v-model="A.tab8" :tabs="tabs" label="Checks"/>
      <div data-ae-check="training" v-if="A.tab8 === 'training'"><AeCheckTraining/></div>
      <div data-ae-check="quality" v-else-if="A.tab8 === 'quality'"><AeCheckQuality/></div>
      <div data-ae-check="latent" v-else-if="A.tab8 === 'latent'"><AeCheckLatent/></div>
      <div data-ae-check="stability" v-else-if="A.tab8 === 'stability'"><AeCheckStability/></div>
      <div data-ae-check="smoothness" v-else-if="A.tab8 === 'smoothness'"><AeCheckSmooth/></div>
    </template>
  </FpcaFrame>`,
};
