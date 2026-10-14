// Step 7, tab "Samples": the table of the samples (worst rebuilt first), and for the chosen sample the picture the network rebuilt next to the measured one, its code, its error and its status;
// the card "Unusual samples" with the limits and the list of the flagged samples.
import { computed, watch, onMounted } from 'vue';
import { Icon } from '/static/js/icons.js';
import { Field, NumField, Seg, Callout, Status } from '/static/js/components/Common.js';
import { PlotlyChart, ChartCard } from '/static/js/plot.js';
import { FpcaPicture, FpcaCurves, FpcaAxesNote, finiteRange, symRange } from '/static/js/components/FpcaCommon.js';
import { vec, image, maskArr } from '/static/js/components/DecompCommon.js';
import { tokens, themeState } from '/static/js/theme.js';
import { fmt } from '/static/js/util.js';
import { fp } from './AeState.js';
import { loadSample, runAnomalies } from './AeActions.js';
import { AeHistogram, AeVerdict, pct, STATUS_LABEL, shortUnits } from './AeCommon.js';

/** counts of `values` in `bins` equal bins (and of the subset `held`, a 0 / 1 array, in the same bins) */
export function histogram(values, held = null, bins = 20, logScale = false) {
  const v = Array.from(values).filter((x) => Number.isFinite(x));
  if (!v.length) return null;
  const f = logScale ? (x) => Math.log10(Math.max(x, 1e-12)) : (x) => x;
  let lo = Infinity, hi = -Infinity; for (const x of v) { const y = f(x); if (y < lo) lo = y; if (y > hi) hi = y; }
  if (hi === lo) { lo -= 0.5; hi += 0.5; }
  const w = (hi - lo) / bins, counts = new Array(bins).fill(0), hcounts = new Array(bins).fill(0);
  Array.from(values).forEach((x, i) => { if (!Number.isFinite(x)) return; const k = Math.min(bins - 1, Math.floor((f(x) - lo) / w)); if (held && held[i]) hcounts[k]++; else counts[k]++; });
  const g = logScale ? (y) => 10 ** y : (y) => y;
  return { edges: Array.from({ length: bins + 1 }, (_, i) => g(lo + i * w)), counts, held: held ? hcounts : null };
}

export const AeSampleView = {
  name: 'AeSampleView',
  components: { Icon, Callout, Status, PlotlyChart, ChartCard, FpcaPicture, FpcaCurves, FpcaAxesNote, AeVerdict },
  props: { sid: { type: String, required: true } },
  setup(props) {
    const A = computed(() => fp.ae), R = computed(() => fp.ae.res);
    const slot = computed(() => (A.value.extra.samples ? A.value.extra.samples[`${A.value.resRid}|${props.sid}|f`] || null : null));
    const d = computed(() => (slot.value ? slot.value.data : null));
    onMounted(() => loadSample(props.sid));
    watch(() => [props.sid, A.value.resRid], () => loadSample(props.sid));
    const level = computed(() => (d.value ? (d.value.status === 'ok' ? 'good' : d.value.status === 'outside' ? 'warning' : 'critical') : 'info'));
    const blocks = computed(() => {
      const x = d.value; if (!x) return [];
      return x.blocks.map((b) => {
        const surface = b.kind === 'surface';
        const f = (a) => (surface ? image(a) : vec(a));
        const o = { name: b.name, surface, units: b.units, rmse: b.rmse, error: b.error, measured: f(b.measured), rebuilt: f(b.reconstruction), diff: f(b.residual), mean: f(b.mean) };
        if (surface) { o.axes = { ex: vec(b.ex), em: vec(b.em) }; o.mask = b.mask ? maskArr(b.mask) : null; }
        else o.axes = { wl: vec(b.x) };
        const r = finiteRange([o.measured, o.rebuilt]); o.range = r ? [Math.min(0, r[0]), r[1]] : null;
        o.drange = symRange([o.diff]);
        return o;
      });
    });
    const codeTraces = computed(() => {
      themeState.version;
      const x = d.value, r = R.value; if (!x || !r) return [];
      const t = tokens(), names = x.code_names || r.codeNames, mean = r.codeMean, sd = r.codeSd;
      const z = x.code.map((c, j) => (sd[j] > 0 ? (c - mean[j]) / sd[j] : 0));
      const err = x.code_sd ? x.code_sd.map((s, j) => (sd[j] > 0 ? s / sd[j] : 0)) : null;
      return [{ type: 'bar', x: names, y: z, error_y: err ? { type: 'data', array: err, visible: true, thickness: 1.2, width: 3 } : undefined, marker: { color: t.series[0], line: { color: t.surface, width: 1 } },
        customdata: x.code, hovertemplate: '%{x} = %{customdata:.4g} (%{y:.2f} SD from the mean code)<extra></extra>', name: 'code' }];
    });
    const codeLayout = computed(() => ({ height: 220, margin: { l: 56, r: 14, t: 10, b: 44 }, xaxis: { title: { text: 'Code' } }, yaxis: { title: { text: 'Distance from the mean code (SD)' }, zeroline: true } }));
    const codeTable = computed(() => (d.value ? { columns: ['code', 'value'], rows: d.value.code.map((c, j) => [(d.value.code_names || [])[j] || `z${j + 1}`, c]) } : { columns: [], rows: [] }));
    return { A, R, slot, d, level, blocks, codeTraces, codeLayout, codeTable, pct, fmt, STATUS_LABEL, shortUnits };
  },
  template: `
  <div class="col gap-12" data-ae-sample-view :data-id="sid">
    <Callout kind="critical" v-if="slot && slot.error" data-ae-sample-error>{{ slot.error }}</Callout>
    <div class="fp-wait muted" v-if="slot && slot.busy && !d">Looking at {{ sid }} …</div>
    <template v-if="d">
      <AeVerdict :level="level" :title="d.id + ' · ' + (STATUS_LABEL[d.status] || d.status) + '.'"><span data-ae-sample-words v-if="!(d.reasons && d.reasons.length)">{{ d.text || d.words }}</span>
        <ul class="fp-list" v-if="d.reasons && d.reasons.length"><li v-for="r in d.reasons" :key="r">{{ r }}</li></ul></AeVerdict>
      <div class="fp-cards4">
        <div class="fp-stat"><span class="l">Used for</span><span class="v" style="font-size:20px">{{ d.split === 'validation' ? 'held out' : d.split }}</span></div>
        <div class="fp-stat"><span class="l">Not rebuilt</span><span class="v">{{ pct(d.error, 1) }}</span><span class="s">of the typical variation</span></div>
        <div class="fp-stat"><span class="l">Rebuilt (R²)</span><span class="v">{{ fmt(d.r2, 3) }}</span></div>
        <div class="fp-stat" v-if="d.typicality !== null && d.typicality !== undefined"><span class="l">Typicality of the code</span><span class="v">{{ fmt(d.typicality, 2) }}</span><span class="s">low = an unusual code</span></div>
        <div class="fp-stat" v-else><span class="l">Error (RMSE)</span><span class="v">{{ fmt(d.rmse, 3) }}</span></div>
      </div>
      <template v-for="b in blocks" :key="b.name">
        <h3 class="fp-h3">{{ b.name === 'fluorescence' ? 'Fluorescence' : 'Absorbance' }}<span class="muted xs" v-if="Number.isFinite(b.rmse)"> · RMSE {{ fmt(b.rmse, 3) }} {{ shortUnits(b.units) }}</span></h3>
        <FpcaAxesNote v-if="b.surface"/>
        <div class="fp-triple" v-if="b.surface" data-ae-images>
          <div class="fp-tri-item" data-ae-image="measured"><div class="fp-gal-t">Measured</div><FpcaPicture surface :axes="b.axes" :z="b.measured" :mask="b.mask" :zrange="b.range" :units="b.units" :height="230" scale-bar title="Measured"/></div>
          <div class="fp-tri-item" data-ae-image="rebuilt"><div class="fp-gal-t">Rebuilt</div><FpcaPicture surface :axes="b.axes" :z="b.rebuilt" :mask="b.mask" :zrange="b.range" :units="b.units" :height="230" scale-bar title="Rebuilt"/></div>
          <div class="fp-tri-item" data-ae-image="difference"><div class="fp-gal-t">Difference</div><FpcaPicture surface :axes="b.axes" :z="b.diff" :mask="b.mask" zkind="diverging" :zrange="b.drange" :units="b.units" :height="230" scale-bar title="Difference"/></div>
        </div>
        <div class="fp-two" v-else data-ae-curves>
          <FpcaCurves :x="b.axes.wl" :lines="[{ name: 'Measured', y: b.measured }, { name: 'Rebuilt', y: b.rebuilt, dash: 'dash' }, { name: 'Mean', y: b.mean, width: 1.4 }]" y-title="Absorbance" :height="230" title="Absorbance: measured and rebuilt"/>
          <FpcaCurves :x="b.axes.wl" :lines="[{ name: 'Difference', y: b.diff }]" y-title="Absorbance" :height="230" zero-line title="Absorbance: difference"/>
        </div>
      </template>
      <ChartCard title="The code of this sample" :table="codeTable" filename="autoencoder-sample-code" help="The bars show how far each code of this sample lies from the mean code of the training samples, in standard deviations. A bar far from zero means that the sample is unusual in that code.">
        <PlotlyChart :data="codeTraces" :layout="codeLayout" :height="220"/></ChartCard>
    </template>
  </div>`,
};

export const AeAnomalies = {
  name: 'AeAnomalies',
  components: { Icon, Field, NumField, Callout, AeHistogram },
  setup() {
    const A = computed(() => fp.ae), R = computed(() => fp.ae.res), V = computed(() => fp.ae.anomView);
    const slot = computed(() => A.value.extra.anom || null);
    const ans = computed(() => (slot.value ? slot.value.data : null));
    const body = () => ({ method: V.value.method, c: Number(V.value.c), level: Number(V.value.level), top: 12 });
    const run = () => runAnomalies(body());
    function open(id) { A.value.sample = id; }
    const hist = computed(() => (R.value ? histogram(R.value.err, R.value.held, 24, true) : null));          // (the errors span several powers of ten: logarithmic bins)
    return { A, R, V, slot, ans, run, open, hist, pct, fmt };
  },
  template: `
  <div class="card" data-ae-anomalies>
    <div class="card-head"><h3>Unusual samples</h3></div>
    <div class="card-body col gap-12">
      <p class="fp-note">A sample is unusual when the network rebuilds it much worse than the others (and, for a variational autoencoder, when its code is atypical). Choose how the limit is set, then look at the samples above it.</p>
      <div class="row wrap" style="align-items:flex-end;gap:12px 18px">
        <Field label="Method" help="Median plus c × MAD: the limit is the median error plus c times the median absolute deviation (robust to a few bad samples). Quantile: the limit is that quantile of the errors of the held-out samples. Limit of the saved model: the limit a saved version of this network would use."><select class="select sm" v-model="V.method" data-ae-anom-method>
          <option value="mad">Median plus c × MAD</option><option value="quantile">Quantile of the held-out errors</option><option value="saved">Limit of the saved model</option></select></Field>
        <Field v-if="V.method === 'mad'" label="c" help="How many MADs above the median. 3 is the usual choice."><NumField v-model="V.c" :min="0.5" :max="20" :step="0.5"/></Field>
        <Field v-if="V.method === 'quantile'" label="Quantile" help="For example 0.99: the error that 99 % of the held-out samples stay below."><NumField v-model="V.level" :min="0.5" :max="0.9999" :step="0.01"/></Field>
        <button type="button" class="btn" data-ae-anom-go :disabled="slot && slot.busy" @click="run"><Icon name="play" size="16"/>Find them</button>
      </div>
      <Callout kind="critical" v-if="slot && slot.error">{{ slot.error }}</Callout>
      <div v-if="ans" :class="{ 'fp-dim': slot && slot.busy }" class="col gap-12">
        <p class="fp-note" data-ae-anom-words>{{ ans.words }}</p>
        <AeHistogram v-if="hist" title="Rebuilding error of the samples" :edges="hist.edges" :counts="hist.counts" :held="hist.held" :limit="ans.limit" limit-label="limit" x-title="Share of the variation not rebuilt (logarithmic scale)" log-x filename="autoencoder-anomalies" :height="240"/>
        <div class="table-wrap" v-if="ans.flagged && ans.flagged.length"><table class="table"><thead><tr><th>Sample</th><th>Not rebuilt</th><th>Why</th><th></th></tr></thead>
          <tbody><tr v-for="f in ans.flagged" :key="f.id" data-ae-flagged-row><td><b>{{ f.id }}</b></td><td class="num">{{ pct(f.error, 1) }}</td><td>{{ (f.reasons || []).join(' ') }}</td><td><button type="button" class="btn xs" @click="open(f.id)">Look at it</button></td></tr></tbody></table></div>
        <p class="fp-note" v-else>No sample is above the limit.</p>
      </div>
    </div>
  </div>`,
};

export const AeSamplesTab = {
  name: 'AeSamplesTab',
  components: { Icon, Seg, Status, AeSampleView, AeAnomalies },
  setup() {
    const A = computed(() => fp.ae), R = computed(() => fp.ae.res), V = computed(() => fp.ae.sampView);
    const rows = computed(() => {
      const r = R.value; if (!r) return [];
      return r.ids.map((id, i) => ({ id, i, held: !!(r.held && r.held[i]), err: r.err[i], r2: r.r2[i], flag: r.flag[i] || '' }));
    });
    const isFlagged = (x) => !!x.flag && !/^inside/i.test(x.flag);
    const shown = computed(() => {
      let l = rows.value;
      const f = V.value.filter;
      if (f === 'training') l = l.filter((x) => !x.held); else if (f === 'held') l = l.filter((x) => x.held); else if (f === 'flagged') l = l.filter(isFlagged);
      const q = (V.value.find || '').trim().toLowerCase();
      if (q) l = l.filter((x) => x.id.toLowerCase().includes(q));
      return l.slice().sort((a, b) => (b.err ?? -1) - (a.err ?? -1));
    });
    const sid = computed(() => (A.value.sample && R.value && R.value.ids.includes(A.value.sample) ? A.value.sample : (shown.value[0] ? shown.value[0].id : (R.value ? R.value.ids[0] : null))));
    function pick(id) { A.value.sample = id; }
    function step(d) { const l = shown.value; if (!l.length) return; const k = Math.max(0, l.findIndex((x) => x.id === sid.value)); A.value.sample = l[(k + d + l.length) % l.length].id; }
    const counts = computed(() => ({ all: rows.value.length, training: rows.value.filter((x) => !x.held).length, held: rows.value.filter((x) => x.held).length, flagged: rows.value.filter(isFlagged).length }));
    const opts = computed(() => [{ value: 'all', label: `All (${counts.value.all})` }, { value: 'training', label: `Training (${counts.value.training})` }, { value: 'held', label: `Held out (${counts.value.held})` }, { value: 'flagged', label: `Flagged (${counts.value.flagged})` }]);
    return { A, R, V, shown, sid, pick, step, opts, pct, isFlagged };
  },
  template: `
  <div class="col gap-12" data-ae-samples-tab>
    <div class="ae-samples-grid">
      <div class="card ae-sample-list" data-ae-sample-table>
        <div class="card-head"><h3>Samples</h3><span class="muted xs">worst rebuilt first</span></div>
        <div class="card-body col gap-8">
          <Seg small v-model="V.filter" :options="opts" data-ae-sample-filter/>
          <input class="input sm" v-model="V.find" placeholder="Find a sample" aria-label="Find a sample" data-ae-sample-find/>
          <div class="table-wrap" style="max-height:420px"><table class="table"><thead><tr><th>Sample</th><th>Used for</th><th class="num">Not rebuilt</th><th>Status</th></tr></thead>
            <tbody><tr v-for="x in shown" :key="x.id" :class="{ sel: x.id === sid }" data-ae-sample-row :data-id="x.id" tabindex="0" @click="pick(x.id)" @keydown.enter="pick(x.id)" @keydown.space.prevent="pick(x.id)">
              <td><b>{{ x.id }}</b></td><td>{{ x.held ? 'held out' : 'training' }}</td><td class="num">{{ pct(x.err, 1) }}</td><td><span class="status" :class="isFlagged(x) ? 'warning' : 'good'">{{ isFlagged(x) ? 'unusual' : 'fits' }}</span></td></tr>
              <tr v-if="!shown.length"><td colspan="4" class="muted">No sample matches.</td></tr></tbody></table></div>
        </div>
      </div>
      <div class="col gap-12" style="min-width:0">
        <div class="row" v-if="sid"><button type="button" class="btn icon sm quiet" aria-label="Previous sample" title="Previous sample" @click="step(-1)"><Icon name="angle-left"/></button>
          <button type="button" class="btn icon sm quiet" aria-label="Next sample" title="Next sample" @click="step(1)"><Icon name="angle-right"/></button><span class="muted xs">in the order of the table</span></div>
        <AeSampleView v-if="sid" :sid="sid" :key="sid"/>
      </div>
    </div>
    <AeAnomalies/>
  </div>`,
};
