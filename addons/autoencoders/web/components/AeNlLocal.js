// Tab "Nonlinear effects", section "Local effects": what one SD of a code does to the EEM (the derivative of the decoder) is NOT the same everywhere when the network bends.  The loading at the
// mean code next to the loadings at samples chosen along the code, where the loading differs between all the samples, and for every code how much the shape of its effect changes.
import { ref, computed, watch, onMounted } from 'vue';
import { Icon } from '/static/js/icons.js';
import { Field, Callout } from '/static/js/components/Common.js';
import { PlotlyChart, ChartCard } from '/static/js/plot.js';
import { sliceImage, vec, maskArr } from '/static/js/components/DecompCommon.js';
import { tokens, themeState, withAlpha } from '/static/js/theme.js';
import { debounce, fmt } from '/static/js/util.js';
import { fp } from './AeState.js';
import { loadLocal } from './AeNlActions.js';
import { LIMITS } from './AeNlState.js';
import { sdText, pctText } from './AeNlCommon.js';
import { AeStripCard, useNarrow, shortUnits, unitsOf } from './AeCommon.js';

const finiteMax = (arrs) => { let m = 0; for (const a of arrs) for (let i = 0; i < a.length; i++) { const v = Math.abs(a[i]); if (v === v && v > m) m = v; } return m || 1; };

export const AeNlLocal = {
  name: 'AeNlLocal',
  components: { Icon, Field, Callout, PlotlyChart, ChartCard, AeStripCard },
  setup() {
    const A = computed(() => fp.ae), R = computed(() => fp.ae.res), N = computed(() => fp.ae.nl), L = computed(() => fp.ae.nl.local);
    const narrow = useNarrow();
    const slot = computed(() => A.value.extra.nlLocal || null);
    const P = computed(() => (slot.value ? slot.value.data : null));
    const go = debounce(() => loadLocal(), 300);
    onMounted(() => { if (R.value && L.value.code > R.value.d) L.value.code = 1; loadLocal(); });
    watch(() => [L.value.code, L.value.mode, L.value.names.slice(), A.value.resRid], () => { if (R.value && L.value.code > R.value.d) L.value.code = 1; go(); }, { deep: true });
    const names = computed(() => (R.value ? R.value.codeNames : []));
    const pick = ref('');
    const candidates = computed(() => (R.value ? R.value.ids.filter((x) => !L.value.names.includes(x)) : []));
    function addSample() { if (pick.value && L.value.names.length < LIMITS.loadings && !L.value.names.includes(pick.value)) L.value.names = [...L.value.names, pick.value]; pick.value = ''; }
    function removeSample(id) { L.value.names = L.value.names.filter((x) => x !== id); }
    const strip = computed(() => {
      const p = P.value; if (!p || !p.fl) return null;
      const fl = p.fl, imgs = [fl.atMean], labels = ['At the mean code'];
      p.samples.forEach((s, k) => { imgs.push(sliceImage(fl.samples, k)); labels.push(`${s.id} · ${sdText(s.position)}`); });
      const m = finiteMax(imgs);
      return { imgs, labels, ex: vec(p.ex), em: vec(p.em), mask: p.mask ? maskArr(p.mask) : null, zrange: [-m, m], units: shortUnits(unitsOf(p.units)) };
    });
    const vstrip = computed(() => {
      const p = P.value; if (!p || !p.fl) return null;
      const v = p.fl.variability, m = finiteMax([v]);
      return { imgs: [v], labels: [`${p.name}: standard deviation of its loading over all the samples`], zrange: [0, m] };
    });
    const absTraces = computed(() => {
      themeState.version;
      const p = P.value; if (!p || !p.ab) return [];
      const t = tokens(), x = Array.from(p.ab.x), ab = p.ab, out = [];
      const arr = (a) => Array.from(a, (v) => (Number.isFinite(v) ? v : null));
      p.samples.forEach((s, k) => out.push({ type: 'scatter', mode: 'lines', x, y: arr(ab.samples.data.subarray(k * ab.samples.k, (k + 1) * ab.samples.k)), name: `${s.id} (${sdText(s.position)})`, line: { color: t.series[(k + 1) % 8], width: 1.8, dash: ['dash', 'dot', 'dashdot', 'longdash', 'solid', 'longdashdot'][k % 6] } }));
      out.push({ type: 'scatter', mode: 'lines', x, y: arr(ab.atMean), name: 'At the mean code', line: { color: t.text, width: 2.8 } });
      return out;
    });
    const absLayout = computed(() => ({ height: 260, margin: { l: 62, r: 14, t: 8, b: 90 }, showlegend: true, legend: { orientation: 'h', y: -0.34, x: 0, font: { size: 10.5 } }, hovermode: 'x unified', xaxis: { title: { text: 'Wavelength (nm)' } }, yaxis: { title: { text: 'Change of the absorbance per SD of the code' }, zeroline: true } }));
    const absVar = computed(() => {
      themeState.version;
      const p = P.value; if (!p || !p.ab) return [];
      return [{ type: 'scatter', mode: 'lines', x: Array.from(p.ab.x), y: Array.from(p.ab.variability, (v) => (Number.isFinite(v) ? v : null)), line: { color: tokens().series[0], width: 2.2 }, name: 'Standard deviation of the loading', hovertemplate: '%{x:.0f} nm: %{y:.4g}<extra></extra>' }];
    });
    const absVarLayout = computed(() => ({ height: 200, margin: { l: 62, r: 14, t: 8, b: 46 }, xaxis: { title: { text: 'Wavelength (nm)' } }, yaxis: { title: { text: 'SD of the loading' }, rangemode: 'tozero' } }));
    // for every code: how much the shape of its effect changes
    const bars = computed(() => {
      themeState.version;
      const p = P.value; if (!p) return [];
      const t = tokens(), ch = p.changeV, ok = (v) => Number.isFinite(v);
      return [{ type: 'bar', x: p.names, y: Array.from(ch.mean, (v) => (ok(v) ? v : 0)), marker: { color: Array.from(ch.mean, (v, i) => (!ok(v) ? withAlpha(t.text3, 0.35) : i === p.code - 1 ? t.series[1] : t.series[0])), line: { color: t.surface, width: 1 } },
        text: Array.from(ch.mean, (v) => (!ok(v) ? 'no effect' : v < 0.0005 ? '0' : fmt(v, 3))), textposition: 'outside', cliponaxis: false, textfont: { size: 11, color: t.text2 },
        customdata: Array.from(ch.mean, (v, i) => [ok(ch.max[i]) ? fmt(ch.max[i], 3) : '–', ok(ch.size_sd[i]) ? pctText(ch.size_sd[i]) : '–']),
        hovertemplate: '%{x}: the shape of its effect changes by %{text} on average (at most %{customdata[0]}); its strength varies by %{customdata[1]}<extra></extra>', name: 'change' }];
    });
    const barLayout = computed(() => { const p = P.value, top = p ? Math.max(0, ...Array.from(p.changeV.mean, (v) => (Number.isFinite(v) ? v : 0))) : 0; return { height: 250, margin: { l: 60, r: 14, t: 24, b: 46 }, xaxis: { title: { text: 'Code' } }, yaxis: { title: { text: 'How much the shape of its effect changes' }, range: [0, Math.max(0.05, 1.3 * top)], zeroline: false } }; });
    const barTable = computed(() => { const p = P.value; return p ? { columns: ['code', 'change of shape (mean of 1 − cosine)', 'largest change of shape', 'strength: mean (1 = at the mean code)', 'strength: standard deviation'],
      rows: p.names.map((n, i) => [n, p.changeV.mean[i], p.changeV.max[i], p.changeV.size_mean[i], p.changeV.size_sd[i]]) } : { columns: [], rows: [] }; });
    const sampleLine = computed(() => (P.value ? P.value.samples.map((s) => `${s.id} (${s.label}, ${sdText(s.position)})`).join(', ') : ''));
    const ncols = computed(() => Math.max(1, Math.min(4, strip.value ? strip.value.imgs.length : 1)));
    return { A, R, N, L, P, slot, names, pick, candidates, addSample, removeSample, strip, vstrip, absTraces, absLayout, absVar, absVarLayout, bars, barLayout, barTable, narrow, ncols, sampleLine, LIMITS };
  },
  template: `
  <div class="col gap-12" data-ae-nl-panel="local">
    <p class="fp-note" data-ae-nl-explain>The loading of a code is what one standard deviation of the code does to the EEM; where the network bends this is not the same for every sample, so the pictures below compare the loading at the average code with the loadings at samples spread along the code, show where the loading differs between samples, and measure how much its shape changes for every code.</p>
    <div class="card"><div class="card-body col gap-12" style="padding:12px 16px"><div class="row wrap" style="gap:12px 26px;align-items:flex-end">
      <Field label="Code" help="The code whose loading is looked at."><select class="select sm" v-model.number="L.code" data-ae-nl-local-code><option v-for="(n, i) in names" :key="n" :value="i + 1">{{ n }}</option></select></Field>
      <div class="col" style="gap:4px"><span class="lbl">Samples</span>
        <div class="seg ae-seg-wrap" role="group" data-ae-nl-local-mode><button type="button" :class="{ on: L.mode === 'percentiles' }" title="The samples at the 10th, 50th and 90th percentile of the code" @click="L.mode = 'percentiles'">10th, 50th and 90th percentile</button>
          <button type="button" :class="{ on: L.mode === 'chosen' }" title="Choose the samples yourself" @click="L.mode = 'chosen'">Samples I choose</button></div></div>
      <Field v-if="L.mode === 'chosen'" label="Add a sample" :help="'Up to ' + LIMITS.loadings + ' samples. Their loading is drawn next to the one at the mean code.'"><div class="row" style="gap:6px;flex-wrap:nowrap"><select class="select sm" v-model="pick" data-ae-nl-local-pick>
        <option value="">Choose a sample …</option><option v-for="id in candidates" :key="id" :value="id">{{ id }}</option></select>
        <button type="button" class="btn sm" :disabled="!pick || L.names.length >= LIMITS.loadings" data-ae-nl-local-add @click="addSample"><Icon name="plus" size="15"/>Add</button></div></Field>
    </div>
    <div class="row wrap ae-nl-chips" v-if="L.mode === 'chosen'">
      <span class="chip" v-for="id in L.names" :key="id" data-ae-nl-local-chip>{{ id }}<span class="x" role="button" tabindex="0" :aria-label="'Remove ' + id" @click="removeSample(id)" @keydown.enter="removeSample(id)">×</span></span>
      <span class="muted xs" v-if="!L.names.length">No sample chosen yet: the loading at the 10th, 50th and 90th percentile of the code is shown meanwhile.</span>
    </div></div></div>
    <Callout kind="critical" v-if="slot && slot.error" data-ae-nl-error>{{ slot.error }}</Callout>
    <div class="fp-wait muted" v-if="slot && slot.busy && !P">Taking the derivative of the decoder at every sample …</div>
    <template v-if="P">
      <p class="fp-note" data-ae-nl-words>{{ P.words }}</p>
      <div :class="{ 'fp-dim': slot && slot.busy }" class="col gap-12">
        <template v-if="strip">
          <div data-ae-nl-loadings><AeStripCard :title="'Loading of ' + P.name + ': the change of the EEM per SD of the code'" :images="strip.imgs" :labels="strip.labels" :ex="strip.ex" :em="strip.em" :mask="strip.mask" kind="diverging"
              :zrange="strip.zrange" :units="strip.units" :cols="ncols" :row-height="narrow ? 120 : 150" filename="autoencoder-local-loadings"
              help="The derivative of the rebuilt EEM with respect to the code, per standard deviation of the code: at the average code and at the samples named above each picture (with their position along the code). One colour scale for all of them."/>
            <p class="fp-note muted" data-ae-nl-sample-line>Samples: {{ sampleLine }}. The same colour scale is used for all the pictures.</p></div>
          <div data-ae-nl-variability><AeStripCard :title="'Where the loading of ' + P.name + ' differs between samples'" :images="vstrip.imgs" :labels="[]" :ex="strip.ex" :em="strip.em" :mask="strip.mask" :zrange="vstrip.zrange" :units="strip.units" :cols="1" :row-height="200"
              filename="autoencoder-local-variability" help="The standard deviation, over all the training samples, of the loading at each pixel. Low on the colour bar = the loading is the same for every sample; high = it changes from sample to sample."/></div>
        </template>
        <template v-if="P.ab">
          <ChartCard :title="'Loading of ' + P.name + ' on the absorbance'" :data="absTraces" filename="autoencoder-local-absorbance"><PlotlyChart :data="absTraces" :layout="absLayout" :height="260"/></ChartCard>
          <ChartCard :title="'Where the loading of ' + P.name + ' differs between samples (absorbance)'" :data="absVar" filename="autoencoder-local-absorbance-variability"><PlotlyChart :data="absVar" :layout="absVarLayout" :height="200"/></ChartCard>
        </template>
        <ChartCard title="How much the shape of the effect of each code changes from sample to sample" :table="barTable" filename="autoencoder-local-change"
                   help="For every code: the average over the samples of 1 minus the cosine between the loading at the sample and the loading at the average code, measured the way the network was trained. 0 = the same picture everywhere."><PlotlyChart :data="bars" :layout="barLayout" :height="250" data-ae-nl-change-chart/></ChartCard>
        <p class="fp-note muted" data-ae-nl-change-note>0 means the picture of the effect has the same shape at every sample, larger numbers mean it changes shape. A code that only gets stronger or weaker (it saturates) keeps its shape and shows 0 here: the map above and the section <b>Where it bends</b> show that.</p>
      </div>
    </template>
  </div>`,
};
