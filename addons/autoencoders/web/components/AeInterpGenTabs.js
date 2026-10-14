// Step 7, tabs "Interpolation" (mix two samples in code space, next to the plain mixture of the two measured samples) and "Generate" (new samples drawn by a variational
// autoencoder; always labelled synthetic).
import { computed, watch, onMounted } from 'vue';
import { Icon } from '/static/js/icons.js';
import { Field, NumField, Callout } from '/static/js/components/Common.js';
import { FpcaPicture } from '/static/js/components/FpcaCommon.js';
import { DecompScale } from '/static/js/components/DecompGallery.js';
import { PlotlyChart, ChartCard } from '/static/js/plot.js';
import { vec, stack, sliceImage, maskArr, image } from '/static/js/components/DecompCommon.js';
import { debounce, fmt } from '/static/js/util.js';
import { fp } from './AeState.js';
import { loadInterp, runGenerate, exportFile } from './AeActions.js';
import { AeStripCard, useNarrow, shortUnits, unitsOf } from './AeCommon.js';
import { AeCurveSet } from './AeTraversalsTab.js';

const rangeOf = (arrs) => { let lo = Infinity, hi = -Infinity; for (const a of arrs) { if (!a) continue; for (let i = 0; i < a.length; i++) { const v = a[i]; if (v === v) { if (v < lo) lo = v; if (v > hi) hi = v; } } } return lo === Infinity ? null : [lo, hi]; };

export const AeInterpTab = {
  name: 'AeInterpTab',
  components: { Icon, Field, NumField, Callout, AeStripCard, AeCurveSet, FpcaPicture, DecompScale },
  setup() {
    const A = computed(() => fp.ae), R = computed(() => fp.ae.res), V = computed(() => fp.ae.interpView);
    const ids = computed(() => (R.value ? R.value.ids : []));
    const narrow = useNarrow();
    // two different samples to start with: the first and the last one
    watch(ids, (l) => { if (l.length > 1 && (!V.value.a || !l.includes(V.value.a))) V.value.a = l[0]; if (l.length > 1 && (!V.value.b || !l.includes(V.value.b))) V.value.b = l[l.length - 1]; }, { immediate: true });
    const go = debounce(() => { const v = V.value; if (v.a && v.b && Number.isInteger(Number(v.steps)) && v.steps >= 3 && v.steps <= 25) loadInterp(v.a, v.b, Math.round(v.steps)); }, 350);
    onMounted(() => go.flush());
    watch(() => [V.value.a, V.value.b, V.value.steps, A.value.resRid], () => go());
    const slot = computed(() => A.value.extra.interp || null);
    const d = computed(() => (slot.value ? slot.value.data : null));
    const ex = computed(() => (d.value ? vec(d.value.ex) : null)), em = computed(() => (d.value ? vec(d.value.em) : null));
    const mask = computed(() => (d.value && d.value.mask ? maskArr(d.value.mask) : null));
    const k = computed(() => (d.value ? d.value.t.length : 0));
    const labels = computed(() => (d.value ? d.value.t.map((t, i) => (i === 0 ? d.value.a.id : i === k.value - 1 ? d.value.b.id : fmt(t, 2))) : []));
    const imgs = (blk) => { const f = blk && blk.fluorescence; if (!f) return null; const st = stack(f.images); return d.value.t.map((_, i) => sliceImage(st, i)); };
    const latent = computed(() => (d.value ? imgs(d.value.latent) : null));
    const linear = computed(() => (d.value && d.value.linear ? imgs(d.value.linear) : null));
    const diff = computed(() => (d.value && d.value.difference ? imgs(d.value.difference) : null));
    const range = computed(() => rangeOf([...(latent.value || []), ...(linear.value || [])]));
    const drange = computed(() => { const r = rangeOf(diff.value || []); if (!r) return null; const m = Math.max(Math.abs(r[0]), Math.abs(r[1])) || 1; return [-m, m]; });
    const pos = computed({ get: () => Math.round(V.value.pos * (k.value - 1)), set: (i) => { V.value.pos = k.value > 1 ? i / (k.value - 1) : 0; } });
    const cols = computed(() => (narrow.value ? Math.min(4, k.value) : Math.min(9, k.value)));
    const absLines = computed(() => {
      const x = d.value; if (!x || !x.latent || !x.latent.absorbance) return null;
      return { x: vec(x.latent.absorbance.x), latent: x.latent.absorbance.series.map((s) => vec(s)), linear: x.linear && x.linear.absorbance ? x.linear.absorbance.series.map((s) => vec(s)) : null };
    });
    const codeLine = computed(() => { const x = d.value, r = R.value; if (!x || !r || !x.code_path) return ''; const row = Array.from(x.code_path[pos.value] || []); return row.map((v, j) => `${r.codeNames[j]} = ${fmt(v, 3)}`).join(', '); });
    return { A, R, V, ids, slot, d, ex, em, mask, k, labels, latent, linear, diff, range, drange, pos, cols, absLines, codeLine, narrow, shortUnits, unitsOf, image, fmt };
  },
  template: `
  <div class="col gap-12" data-ae-interp-tab>
    <div class="card"><div class="card-body" style="padding:12px 16px"><div class="row wrap" style="gap:12px 22px;align-items:flex-end">
      <Field label="Sample A" style="width:200px"><select class="select sm" v-model="V.a" data-ae-interp-a><option v-for="id in ids" :key="id" :value="id">{{ id }}</option></select></Field>
      <Field label="Sample B" style="width:200px"><select class="select sm" v-model="V.b" data-ae-interp-b><option v-for="id in ids" :key="id" :value="id">{{ id }}</option></select></Field>
      <Field label="Number of steps" help="How many pictures between A and B, including A and B. 3 to 25."><NumField v-model="V.steps" :min="3" :max="25" :step="1" data-ae-interp-steps/></Field>
    </div></div></div>
    <p class="fp-note">The codes of the two samples are mixed in steps from A (0) to B (1) and the network decodes every step: this is the path through the code. Next to it is the plain mixture of the two measured samples. Where the two rows differ, the network knows something about the samples that a simple mixture does not.</p>
    <Callout kind="critical" v-if="slot && slot.error" data-ae-interp-error>{{ slot.error }}</Callout>
    <Callout kind="warning" v-if="V.a && V.a === V.b">Sample A and sample B are the same sample: choose two different ones.</Callout>
    <div class="fp-wait muted" v-if="slot && slot.busy && !d">Mixing the two samples …</div>
    <div v-if="d" :class="{ 'fp-dim': slot.busy }" class="col gap-12">
      <Callout kind="info" v-for="n in d.notes || []" :key="n">{{ n }}</Callout>
      <template v-if="latent">
        <div data-ae-interp-strip="latent"><AeStripCard title="Through the code" :images="latent" :labels="labels" :ex="ex" :em="em" :mask="mask" :zrange="range" :units="shortUnits(d.units)" :cols="cols" :row-height="narrow ? 120 : 150" :highlight="pos" filename="autoencoder-interpolation-code"
                      help="The codes of the two samples are mixed and the network decodes each mixture."/></div>
        <div data-ae-interp-strip="linear" v-if="linear"><AeStripCard title="Plain mixture" :images="linear" :labels="labels" :ex="ex" :em="em" :mask="mask" :zrange="range" :units="shortUnits(d.units)" :cols="cols" :row-height="narrow ? 120 : 150" :highlight="pos" filename="autoencoder-interpolation-linear"
                      help="The two measured samples mixed directly, pixel by pixel."/></div>
        <p class="fp-note muted" v-else>The plain mixture of the measured samples needs the current data, and the data changed since this network was trained.</p>
        <div data-ae-interp-strip="difference" v-if="diff"><AeStripCard title="Difference" :images="diff" :labels="labels" :ex="ex" :em="em" :mask="mask" kind="diverging" :zrange="drange" :units="shortUnits(d.units)" :cols="cols" :row-height="narrow ? 120 : 150" :highlight="pos" filename="autoencoder-interpolation-difference"
                      help="Through the code minus the plain mixture."/></div>
        <div class="card"><div class="card-body col gap-8"><div class="fp-slider"><label class="lbl" for="ae-interp-pos">Position</label>
          <input id="ae-interp-pos" type="range" class="ae-range" min="0" :max="k - 1" step="1" v-model.number="pos" data-ae-interp-pos/><span class="fp-slider-v">{{ labels[pos] }}</span></div>
          <p class="fp-note muted" v-if="codeLine">Code at this position: {{ codeLine }}</p>
          <div class="fp-triple">
            <div class="fp-tri-item"><div class="fp-gal-t">Through the code</div><FpcaPicture surface :axes="{ ex, em }" :z="latent[pos]" :mask="mask" :zrange="range" :units="unitsOf(d.units)" :height="230" scale-bar title="Through the code"/></div>
            <div class="fp-tri-item" v-if="linear"><div class="fp-gal-t">Plain mixture</div><FpcaPicture surface :axes="{ ex, em }" :z="linear[pos]" :mask="mask" :zrange="range" :units="unitsOf(d.units)" :height="230" scale-bar title="Plain mixture"/></div>
            <div class="fp-tri-item" v-if="diff"><div class="fp-gal-t">Difference</div><FpcaPicture surface :axes="{ ex, em }" :z="diff[pos]" :mask="mask" zkind="diverging" :zrange="drange" :units="unitsOf(d.units)" :height="230" scale-bar title="Difference"/></div>
          </div></div></div>
      </template>
      <div class="fp-two" v-if="absLines">
        <AeCurveSet title="Absorbance through the code" :x="absLines.x" :series="absLines.latent" :labels="labels" y-title="Absorbance" filename="autoencoder-interpolation-absorbance"/>
        <AeCurveSet v-if="absLines.linear" title="Absorbance, plain mixture" :x="absLines.x" :series="absLines.linear" :labels="labels" y-title="Absorbance" filename="autoencoder-interpolation-absorbance-linear"/>
      </div>
    </div>
  </div>`,
};

export const AeGenerateTab = {
  name: 'AeGenerateTab',
  components: { Icon, Field, NumField, Callout, AeStripCard, AeCurveSet },
  setup() {
    const A = computed(() => fp.ae), R = computed(() => fp.ae.res), V = computed(() => fp.ae.genView);
    const narrow = useNarrow();
    const slot = computed(() => A.value.extra.gen || null);
    const d = computed(() => (slot.value ? slot.value.data : null));
    const body = () => ({ n: Math.round(V.value.n), seed: Math.round(V.value.seed), temperature: Number(V.value.temperature) });
    const run = () => runGenerate(body());
    const ex = computed(() => (d.value ? vec(d.value.ex) : null)), em = computed(() => (d.value ? vec(d.value.em) : null));
    const mask = computed(() => (d.value && d.value.mask ? maskArr(d.value.mask) : null));
    const imgs = computed(() => { const f = d.value && d.value.fluorescence; if (!f) return null; const st = stack(f.images); return d.value.ids.map((_, i) => sliceImage(st, i)); });
    const labels = computed(() => (d.value ? d.value.ids.map((id) => id.replace('synthetic-', 'synthetic ')) : []));
    const range = computed(() => rangeOf(imgs.value || []));
    const abs = computed(() => { const x = d.value; return x && x.absorbance ? { x: vec(x.absorbance.x), series: x.absorbance.series.map((s) => vec(s)) } : null; });
    const table = computed(() => (d.value ? d.value.ids.map((id, i) => ({ id, near: d.value.nearest ? d.value.nearest[i] : null })) : []));
    const cols = computed(() => (narrow.value ? 3 : Math.min(6, d.value ? d.value.ids.length : 6)));
    const dl = (format) => exportFile('synthetic', { ...body(), format });
    const issues = computed(() => { const v = V.value, o = []; if (!(v.n >= 1 && v.n <= 24)) o.push('The number of samples must be between 1 and 24.'); if (!(v.temperature >= 0 && v.temperature <= 2)) o.push('The temperature must be between 0 and 2.'); return o; });
    return { A, R, V, slot, d, run, ex, em, mask, imgs, labels, range, abs, table, cols, dl, issues, narrow, shortUnits, unitsOf, fmt };
  },
  template: `
  <div class="col gap-12" data-ae-generate-tab>
    <div class="card"><div class="card-body col gap-12" style="padding:12px 16px">
      <p class="fp-note">A variational autoencoder knows what a typical code looks like, so it can draw new codes and decode them into new, similar samples. <b>They are made up by the network and were never measured.</b> Use them to see what the network has learned, never as data.</p>
      <div class="row wrap" style="gap:12px 22px;align-items:flex-end">
        <Field label="Number of samples" help="1 to 24."><NumField v-model="V.n" :min="1" :max="24" :step="1" data-ae-gen-n/></Field>
        <Field label="Seed" help="The same seed gives the same new samples."><NumField v-model="V.seed" :min="0" :step="1" data-ae-gen-seed/></Field>
        <Field label="Temperature" help="0 to 2. 1 draws codes as typical as the training samples; below 1 the new samples stay close to the average sample; above 1 they stray further away."><NumField v-model="V.temperature" :min="0" :max="2" :step="0.1" data-ae-gen-temperature/></Field>
        <button type="button" class="btn primary" data-ae-generate-go :disabled="(slot && slot.busy) || issues.length" @click="run"><Icon name="play" size="16"/>Draw new samples</button>
      </div>
      <Callout kind="warning" v-for="m in issues" :key="m">{{ m }}</Callout>
    </div></div>
    <Callout kind="critical" v-if="slot && slot.error" data-ae-gen-error>{{ slot.error }}</Callout>
    <div class="fp-wait muted" v-if="slot && slot.busy && !d">Drawing …</div>
    <div v-if="d" :class="{ 'fp-dim': slot.busy }" class="col gap-12" data-ae-generated>
      <div class="callout warning ae-synthetic" data-ae-synthetic-banner><Icon name="exclamation-triangle"/><div class="grow"><b>{{ d.label }}</b></div></div>
      <Callout kind="info" v-for="n in d.notes || []" :key="n">{{ n }}</Callout>
      <AeStripCard v-if="imgs" title="Synthetic samples" :images="imgs" :labels="labels" :ex="ex" :em="em" :mask="mask" :zrange="range" :units="shortUnits(d.units)" :cols="cols" :row-height="narrow ? 120 : 150" filename="autoencoder-synthetic"
                   help="New EEMs drawn from the network. Every picture is synthetic."/>
      <AeCurveSet v-if="abs" title="Synthetic absorbance spectra" :x="abs.x" :series="abs.series" :labels="labels" y-title="Absorbance (synthetic)" filename="autoencoder-synthetic-absorbance"/>
      <div class="table-wrap"><table class="table"><thead><tr><th>Sample</th><th>Nearest real sample</th><th class="num">Distance (SD)</th></tr></thead>
        <tbody><tr v-for="r in table" :key="r.id" data-ae-synthetic><td><b>{{ r.id }}</b> <span class="chip accent">synthetic</span></td><td>{{ r.near ? r.near.id : '–' }}</td><td class="num">{{ r.near ? fmt(r.near.distance, 3) : '–' }}</td></tr></tbody></table></div>
      <div class="row wrap"><button type="button" class="btn" data-ae-download="synthetic" @click="dl('npz')"><Icon name="download-alt" size="16"/>Download as .npz</button>
        <button type="button" class="btn" data-ae-download="synthetic-csv" @click="dl('csv')"><Icon name="download-alt" size="16"/>Download as CSV</button><span class="muted xs">The files say "synthetic" in their name and inside.</span></div>
    </div>
  </div>`,
};
