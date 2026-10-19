// Tab "Nonlinear effects", section "Response curves": what a measurement made on the decoded EEM does when ONE code is moved along its range (the others at their average): one small chart per
// code, the training samples as faint dots, the curves of single samples on request, and the shape of every curve in words.
import { ref, computed, watch, onMounted } from 'vue';
import { Icon } from '/static/js/icons.js';
import { Field, NumField, Switch, Callout } from '/static/js/components/Common.js';
import { PlotlyChart, ChartCard } from '/static/js/plot.js';
import { tokens, themeState, withAlpha } from '/static/js/theme.js';
import { debounce, fmt } from '/static/js/util.js';
import { fp } from './AeState.js';
import { loadResponse, codesShown } from './AeNlActions.js';
import { parseNames, MAX_ICE, LIMITS, SHOWN } from './AeNlState.js';
import { AeNlLegend, AeNlFeatureEditor, AeFeatureSwatch, featureStyle } from './AeNlCommon.js';

const finite = (a) => { let lo = Infinity, hi = -Infinity; for (let i = 0; i < a.length; i++) { const v = a[i]; if (v === v && v !== null) { if (v < lo) lo = v; if (v > hi) hi = v; } } return lo === Infinity ? null : [lo, hi]; };
const arr = (a, f) => Array.from(a, (v) => (Number.isFinite(v) ? f(v) : null));

/** One chart: the response of every chosen feature to one code. */
export const AeNlCurve = {
  name: 'AeNlCurve',
  components: { ChartCard, PlotlyChart, AeFeatureSwatch },
  props: { answer: { type: Object, required: true }, j: { type: Number, required: true }, relative: Boolean, dots: Boolean, busy: Boolean },
  setup(props) {
    const c = computed(() => props.answer.codes[props.j]);
    const feats = computed(() => props.answer.features);
    const unitsOf = (f) => (f.units ? ` ${f.units}` : '');
    const scaler = (y) => { const r = finite(y); return (v) => (props.relative && r ? (r[1] > r[0] ? (v - r[0]) / (r[1] - r[0]) : 0.5) : v); };
    const traces = computed(() => {
      themeState.version;
      const t = tokens(), P = props.answer, cc = c.value, x = Array.from(P.t), out = [], name = cc.name;
      feats.value.forEach((f, k) => {
        const st = featureStyle(k, t), cv = cc.curves[f.id], sc = scaler(cv.yv);
        if (cc.ice && cc.ice.y[f.id]) cc.ice.y[f.id].forEach((row, s) => out.push({ type: 'scatter', mode: 'lines', x, y: arr(row, sc), line: { color: withAlpha(st.color, 0.38), width: 1.1, dash: st.dash }, name: `${cc.ice.samples[s]} (${f.short})`,
          hovertemplate: `${cc.ice.samples[s]}: ${name} = %{x:+.1f} SD<br>%{customdata:.4g}${unitsOf(f)}<extra>${f.short}</extra>`, customdata: Array.from(row), showlegend: false }));
        if (props.dots && P.overlay) {
          const xs = P.overlay.x[props.j], ys = f.per_code ? P.overlay.y[f.id][props.j] : P.overlay.y[f.id];
          out.push({ type: 'scatter', mode: 'markers', x: Array.from(xs), y: arr(ys, sc), marker: { size: 5, color: withAlpha(st.color, 0.42), symbol: st.symbol, line: { width: 0 } }, name: `Samples (${f.short})`, showlegend: false,
            customdata: P.overlay.ids.map((id, i) => [id, ys[i]]), hovertemplate: `%{customdata[0]}: ${name} = %{x:+.2f} SD<br>%{customdata[1]:.4g}${unitsOf(f)}<extra>${f.short}</extra>` });
        }
        out.push({ type: 'scatter', mode: 'lines', x, y: arr(cv.yv, sc), line: { color: st.color, width: 2.8, dash: st.dash }, name: f.short, customdata: Array.from(cv.yv),
          hovertemplate: `${name} = %{x:+.1f} SD<br>${f.short}: %{customdata:.4g}${unitsOf(f)}<extra></extra>` });
        const every = Math.max(1, Math.round(x.length / 6)), idx = x.map((_, i) => i).filter((i) => i % every === Math.floor(every / 2));
        out.push({ type: 'scatter', mode: 'markers', x: idx.map((i) => x[i]), y: idx.map((i) => (Number.isFinite(cv.yv[i]) ? sc(cv.yv[i]) : null)), marker: { size: 8, color: st.color, symbol: st.symbol, line: { color: t.surface, width: 1.5 } }, hoverinfo: 'skip', showlegend: false });
      });
      return out;
    });
    const layout = computed(() => {
      themeState.version;
      const t = tokens(), span = props.answer.span, same = new Set(feats.value.map((f) => f.units)).size === 1;
      return { height: 240, margin: { l: 58, r: 12, t: 8, b: 48 }, showlegend: false, hovermode: 'closest',
        xaxis: { title: { text: `Position of ${c.value.name} (SD from its mean)` }, range: [-span, span], zeroline: false, dtick: span > 2.5 ? 1 : 0.5 },
        yaxis: { title: { text: props.relative ? 'Scaled value (curve: 0 to 1)' : (same && feats.value[0].units ? feats.value[0].units : 'Value of the feature') }, zeroline: false },
        shapes: [{ type: 'line', xref: 'x', yref: 'paper', x0: 0, x1: 0, y0: 0, y1: 1, line: { color: t.lineStrong, width: 1, dash: 'dot' } }] };
    });
    const table = computed(() => ({ columns: ['position (SD)', ...feats.value.map((f) => f.short + (f.units ? ` (${f.units})` : ''))], rows: Array.from(props.answer.t, (v, i) => [v, ...feats.value.map((f) => c.value.curves[f.id].yv[i])]) }));
    const subtitle = computed(() => `1 SD = ${fmt(props.answer.sd[props.j], 3)}`);
    return { c, feats, traces, layout, table, subtitle, fmt };
  },
  template: `
  <div :data-ae-nl-curve="c.code" :class="{ 'fp-dim': busy }">
    <ChartCard :title="c.name" :subtitle="subtitle" :table="table" :filename="'autoencoder-response-' + c.name"
               help="The response of the chosen features to this code. The code moves from its lowest to its highest position while all the other codes stay at their average; the dotted vertical line is the average code.">
      <PlotlyChart :data="traces" :layout="layout" :height="240"/>
      <ul class="ae-nl-words" data-ae-nl-words>
        <li v-for="(f, k) in feats" :key="f.id" :data-ae-nl-kind-of="c.curves[f.id].kind"><AeFeatureSwatch :k="k"/><span>{{ c.curves[f.id].words }}</span></li>
      </ul>
    </ChartCard>
  </div>`,
};

export const AeNlResponse = {
  name: 'AeNlResponse',
  components: { Icon, Field, NumField, Switch, Callout, AeNlCurve, AeNlLegend, AeNlFeatureEditor },
  setup() {
    const A = computed(() => fp.ae), R = computed(() => fp.ae.res), N = computed(() => fp.ae.nl);
    const slot = computed(() => A.value.extra.nlResp || null);
    const P = computed(() => (slot.value ? slot.value.data : null));
    const featureError = ref('');
    const go = debounce(() => loadResponse(), 350);
    onMounted(() => loadResponse());
    watch(() => [N.value.span, N.value.n_points, N.value.features, N.value.overlay, N.value.ice, N.value.iceNames, N.value.showAll, A.value.resRid], () => go(), { deep: true });
    const d = computed(() => (R.value ? R.value.d : 0));
    const shown = computed(() => (P.value ? P.value.codes.slice(0, N.value.showAll ? d.value : SHOWN) : []));
    const base = () => N.value.features || (P.value ? P.value.specs : null);
    async function addFeature(spec) {
      const cur = base(); if (!cur) return;
      if (cur.length >= 6) { featureError.value = 'At most 6 features can be drawn at once: remove one first.'; return; }
      featureError.value = '';
      N.value.features = [...cur, spec];
      await loadResponse();
      const s = A.value.extra.nlResp;
      if (s && s.error) { featureError.value = s.error; N.value.features = cur.length ? cur : null; await loadResponse(true); }       // a feature that cannot be drawn is not kept
    }
    function removeFeature(k) { const cur = base(); if (!cur || cur.length < 2) return; featureError.value = ''; N.value.features = cur.filter((_, i) => i !== k); }
    function resetFeatures() { featureError.value = ''; N.value.features = null; }
    const unknown = computed(() => (R.value ? parseNames(N.value.iceNames).filter((x) => !R.value.ids.includes(x)) : []));
    const mixedUnits = computed(() => (P.value ? new Set(P.value.features.map((f) => f.units)).size > 1 : false));
    const relative = computed({ get: () => N.value.relative || mixedUnits.value, set: (v) => { N.value.relative = v; } });
    const hidden = computed(() => (P.value ? Math.max(0, P.value.codes.length - shown.value.length) : 0));
    return { A, R, N, slot, P, d, shown, addFeature, removeFeature, resetFeatures, featureError, unknown, mixedUnits, relative, hidden, LIMITS, MAX_ICE, SHOWN };
  },
  template: `
  <div class="col gap-12" data-ae-nl-panel="response">
    <p class="fp-note" data-ae-nl-explain>Each chart moves one code from the low to the high end of its range while all the other codes stay at their average, and draws what the chosen measurements of the decoded EEM do: a straight line means the code acts like a PCA loading, a curve is a nonlinear trend, and the faint dots are your own samples.</p>
    <div class="card"><div class="card-body col gap-12" style="padding:12px 16px">
      <div class="row wrap" style="gap:12px 22px;align-items:flex-end">
        <Field label="Range (± SD)" help="How far each code is moved from its average, in standard deviations (SD) of the code over the training samples. 0.5 to 4."><NumField v-model="N.span" :min="LIMITS.span[0]" :max="LIMITS.span[1]" :step="0.5" small data-ae-nl-span/></Field>
        <Field label="Points per curve" help="How many positions are decoded along each code. 5 to 201."><NumField v-model="N.n_points" :min="LIMITS.points[0]" :max="LIMITS.points[1]" :step="2" small data-ae-nl-points/></Field>
        <div class="col" style="gap:6px;padding-bottom:4px"><Switch v-model="N.overlay" label="Show the training samples" title="Draw your own samples as faint dots: the horizontal position is the sample's own code, the vertical position is the measurement made on the sample (its smoothed EEM)." data-ae-nl-overlay/></div>
        <div class="col" style="gap:6px;padding-bottom:4px"><Switch v-model="relative" label="Compare shapes (scale each feature from 0 to 1)" :disabled="mixedUnits" title="Scale every feature from 0 (its lowest value on the curve) to 1 (its highest) so that the shapes can be compared; switch it off to see the real values." data-ae-nl-relative/></div>
      </div>
      <div class="col gap-8">
        <div class="col" style="gap:6px"><Switch v-model="N.ice" label="Also draw single samples' curves (the other codes at that sample's own values)" title="For each chosen sample the code is moved along its range while the sample keeps its own values of the other codes." data-ae-nl-ice/>
          <span class="muted xs">If these curves are parallel to the thick one, the code acts independently of the others; if their shape differs, its effect depends on the other codes.</span></div>
        <Field v-if="N.ice" label="Samples for these curves" :hint="unknown.length ? 'Not a sample of this network: ' + unknown.join(', ') + '.' : 'Names separated by commas (at most ' + MAX_ICE + '). Empty: ' + MAX_ICE + ' samples chosen automatically over the range of each code.'"
               help="The curves of single samples: the sample keeps the values of its other codes and only this code is moved."><input class="input sm" v-model="N.iceNames" placeholder="for example S01, S07" aria-label="Samples for the curves of single samples" data-ae-nl-ice-names/></Field>
      </div>
      <AeNlFeatureEditor :features="P ? P.features : []" :error="featureError" offer-moved @add="addFeature" @remove="removeFeature"/>
      <p class="fp-note muted" v-if="mixedUnits">The features have different units, so they are drawn scaled from 0 to 1.</p>
    </div></div>
    <Callout kind="critical" v-if="slot && slot.error" data-ae-nl-error>{{ slot.error }}
      <button type="button" class="btn xs" v-if="N.features" style="margin-left:8px" @click="resetFeatures">Use the usual features</button></Callout>
    <div class="fp-wait muted" v-if="slot && slot.busy && !P">Decoding the codes …</div>
    <template v-if="P">
      <Callout kind="info" v-for="n in P.notes" :key="n" data-ae-nl-note>{{ n }}</Callout>
      <AeNlLegend :features="P.features" :extra="N.ice ? ['Thin lines: single samples (the other codes at their own values)'] : []"/>
      <div class="ae-nl-grid" :class="{ 'fp-dim': slot && slot.busy }" data-ae-nl-curves>
        <AeNlCurve v-for="(c, j) in shown" :key="c.code" :answer="P" :j="j" :relative="relative" :dots="N.overlay"/>
      </div>
      <div class="row wrap" v-if="hidden"><button type="button" class="btn sm" data-ae-nl-all title="Draw the charts of the remaining codes too." @click="N.showAll = true">Show the other {{ hidden }} codes</button></div>
      <div class="row wrap" v-else-if="N.showAll && d > SHOWN"><button type="button" class="btn sm quiet" title="Draw only the charts of the first codes." @click="N.showAll = false">Show only the first {{ SHOWN }} codes</button></div>
      <p class="fp-note muted">The horizontal axis is the position of the code in standard deviations of the codes of the training samples; samples beyond the range are not shown.</p>
    </template>
  </div>`,
};
