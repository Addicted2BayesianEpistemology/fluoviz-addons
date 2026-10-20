// Step 9 of the Self-organising map page: use the map everywhere else. The positions and the groups are variables (plot, cluster, classify), the map can be saved and applied to new
// samples, the numbers can be downloaded, and the methods paragraph is ready to copy.
import { computed, onMounted, watch } from 'vue';
import { Icon } from '/static/js/icons.js';
import { Callout } from '/static/js/components/Common.js';
import { FpcaFrame } from '/static/js/components/FpcaStepper.js';
import { FpcaExplain } from '/static/js/components/FpcaCommon.js';
import { SaveModel } from '/static/js/components/SaveModel.js';
import { go } from '/static/js/router.js';
import { toast } from '/static/js/store.js';
import { catalogue, loadCatalogue } from '/static/js/variables.js';
import { fp, ctx } from './SomState.js';
import { loadMethods, exportFile } from './SomActions.js';
import { SomSection } from './SomCommon.js';
import { SomGate, SomStaleNote } from './SomResults.js';

export const SomUse = {
  name: 'SomUse',
  components: { Icon, Callout, FpcaFrame, FpcaExplain, SaveModel, SomSection, SomGate, SomStaleNote },
  setup() {
    const A = computed(() => fp.som), R = computed(() => fp.som.res);
    const ready = computed(() => !!(R.value && A.value.resRid));
    const rid = computed(() => A.value.resRid);
    onMounted(() => { loadCatalogue(true); loadMethods(); });
    watch(rid, () => loadMethods());
    const vars = computed(() => {
      const r = rid.value; if (!r) return [];
      const pre = `res:${r}:`;
      const fromCat = catalogue.value.filter((v) => v.key.startsWith(pre));
      if (fromCat.length) return fromCat.map((v) => ({ key: v.key, name: v.key.slice(pre.length), label: v.label }));
      return ((A.value.res && A.value.res.result.variables) || []).map((n) => ({ key: pre + n, name: n, label: n }));
    });
    const posKeys = computed(() => (rid.value ? [`res:${rid.value}:som_x`, `res:${rid.value}:som_y`] : []));
    const toCluster = () => go('/cluster', { keys: posKeys.value.join(',') });
    const toPlots = () => go('/compare');
    const toModels = () => go('/models');
    const methods = computed(() => (A.value.extra[`methods|${A.value.resRid}`] || null));
    const text = computed(() => (methods.value && methods.value.data ? methods.value.data.text : ''));
    const refs = computed(() => (methods.value && methods.value.data ? methods.value.data.references || [] : []));
    async function copy() {
      const t = text.value; if (!t) return;
      try { await navigator.clipboard.writeText(t); toast('The methods paragraph was copied.', 'ok', '', 2500); }
      catch {
        const ta = document.createElement('textarea'); ta.value = t; ta.style.position = 'fixed'; ta.style.opacity = '0'; document.body.appendChild(ta); ta.select();
        let ok = false; try { ok = document.execCommand('copy'); } catch { ok = false; }
        ta.remove(); toast(ok ? 'The methods paragraph was copied.' : 'Copying is blocked here: select the text and copy it yourself.', ok ? 'ok' : 'warn', '', 3500);
      }
    }
    const savable = computed(() => !!R.value && !R.value.variablesMode);
    const dl = (what) => exportFile(what);
    return { A, R, ready, rid, vars, toCluster, toPlots, toModels, methods, text, refs, copy, savable, dl, goStep: ctx.goStep };
  },
  template: `
  <FpcaFrame :n="9" no-next>
    <FpcaExplain does="Takes the map out of this page. The position of each sample on the map, its unit, its distance to the unit and its group become ordinary variables of FluoViz: you can plot them, colour other plots by them, cluster the samples on them and classify with them. The map can be saved and applied to new samples, and everything can be downloaded."
                 check="That the map you pass on is the one you checked in step 8: a map that was not better than the scatter of the first two components, or whose neighbourhoods are not stable, is a poor basis for a classification. Save the map only when it is up to date."/>
    <SomGate/>
    <template v-if="ready">
      <SomStaleNote/>
      <SomSection title="Variables made by this map" sentence="They appear in the Plots page and in every selector of the program, under the name of this map.">
        <ul class="som-varlist" data-som-vars><li v-for="v in vars" :key="v.key" :data-name="v.name"><span class="chip">{{ v.name }}</span> <span class="small">{{ v.label }}</span></li></ul>
      </SomSection>
      <SomSection title="Use the positions" sentence="The two positions on the map are numbers like any other: send them to the clustering and classification page, or plot them.">
        <div class="row wrap">
          <button type="button" class="btn primary" data-som-to-cluster @click="toCluster"><Icon name="bullseye" size="16"/>Cluster the map positions</button>
          <button type="button" class="btn" data-som-to-plots @click="toPlots"><Icon name="chart-pie" size="16"/>Plot the map positions</button>
        </div>
      </SomSection>
      <SomSection title="Keep the map" sentence="A saved model keeps the map (not your data). You can apply it to new samples, in this project or another, and use it as the first link of a classification.">
        <Callout kind="info" v-if="!savable" data-som-nosave>A map of other variables cannot be saved as a model: it has no way to read new samples. Map the EEMs or the spectra themselves to get a map that can be saved.</Callout>
        <div class="row wrap">
          <SaveModel v-if="savable" :rid="rid"/>
          <button type="button" class="btn" data-som-to-models @click="toModels"><Icon name="files-landscapes-alt" size="16"/>Apply a saved model</button>
          <button type="button" class="btn quiet" data-som-change @click="goStep(5)"><Icon name="arrow-left" size="15"/>Change the settings and compute again</button>
        </div>
      </SomSection>
      <SomSection title="Downloads" sentence="The files are made from this map; the file names say what they contain.">
        <div class="fp-downloads" data-som-downloads>
          <div class="fp-dl"><b>Positions (CSV)</b><span class="muted xs">One row per sample: its unit, row and column, its position, its distance to the unit, the distance when it was held out, its group and its status.</span><button type="button" class="btn sm" data-som-download="positions" @click="dl('positions')"><Icon name="download-alt" size="15"/>Download</button></div>
          <div class="fp-dl"><b>Prototypes (.npz)</b><span class="muted xs">The prototype EEMs (and spectra) of the units, the hits and the coordinates of the units, for use in other programs.</span><button type="button" class="btn sm" data-som-download="prototypes" @click="dl('prototypes')"><Icon name="download-alt" size="15"/>Download</button></div>
          <div class="fp-dl"><b>Planes (CSV)</b><span class="muted xs">One row per unit: every plane of the map.</span><button type="button" class="btn sm" data-som-download="planes" @click="dl('planes')"><Icon name="download-alt" size="15"/>Download</button></div>
          <div class="fp-dl"><b>Groups (CSV)</b><span class="muted xs">The group of every unit and the number of samples on it.</span><button type="button" class="btn sm" data-som-download="clusters" @click="dl('clusters')"><Icon name="download-alt" size="15"/>Download</button></div>
          <div class="fp-dl"><b>Training history (CSV)</b><span class="muted xs">The energy and the mean distance of every epoch.</span><button type="button" class="btn sm" data-som-download="history" @click="dl('history')"><Icon name="download-alt" size="15"/>Download</button></div>
        </div>
      </SomSection>
      <SomSection title="Methods paragraph" sentence="A text for the methods section of a report or paper: the data, the representation, the distance, the map, the training and the checks. It can be pasted as it is.">
        <Callout kind="critical" v-if="methods && methods.error">{{ methods.error }}</Callout>
        <div class="fp-wait muted" v-if="methods && methods.busy && !text">Writing the text …</div>
        <template v-if="text">
          <div class="fp-recipe" data-som-methods>{{ text }}</div>
          <ul class="fp-list small" v-if="refs.length"><li v-for="r in refs" :key="r">{{ r }}</li></ul>
          <div class="row"><button type="button" class="btn" data-som-copy @click="copy"><Icon name="copy-alt" size="15"/>Copy</button></div>
        </template>
      </SomSection>
    </template>
  </FpcaFrame>`,
};
