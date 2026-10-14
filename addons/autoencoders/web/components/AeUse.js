// Step 9 of the Autoencoders page: use the network everywhere else.  The codes are variables (cluster them, classify with them, plot them), the network can be saved and applied to
// new samples, the numbers can be downloaded, and the methods paragraph is ready to copy.
import { computed, onMounted, watch } from 'vue';
import { Icon } from '/static/js/icons.js';
import { Callout } from '/static/js/components/Common.js';
import { FpcaFrame } from '/static/js/components/FpcaStepper.js';
import { FpcaExplain } from '/static/js/components/FpcaCommon.js';
import { SaveModel } from '/static/js/components/SaveModel.js';
import { go } from '/static/js/router.js';
import { toast } from '/static/js/store.js';
import { catalogue, loadCatalogue } from '/static/js/variables.js';
import { fp, ctx } from './AeState.js';
import { loadMethods, exportFile } from './AeActions.js';
import { AeSection } from './AeCommon.js';
import { AeGate, AeStaleNote } from './AeResults.js';

export const AeUse = {
  name: 'AeUse',
  components: { Icon, Callout, FpcaFrame, FpcaExplain, SaveModel, AeSection, AeGate, AeStaleNote },
  setup() {
    const A = computed(() => fp.ae), R = computed(() => fp.ae.res);
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
    const codeKeys = computed(() => (R.value ? R.value.codeNames.map((n) => `res:${rid.value}:${n}`) : []));
    const toCluster = () => go('/cluster', { keys: codeKeys.value.join(',') });
    const toClassify = () => go('/cluster', { keys: codeKeys.value.join(','), tab: 'separation' });
    const toPlots = () => go('/compare');
    const toModels = () => go('/models');
    const methods = computed(() => A.value.extra.methods || null);
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
    const isVae = computed(() => !!R.value && R.value.family === 'vae');
    const dl = (what) => exportFile(what);
    return { A, R, ready, rid, vars, toCluster, toClassify, toPlots, toModels, methods, text, refs, copy, isVae, dl, goStep: ctx.goStep };
  },
  template: `
  <FpcaFrame :n="9" no-next>
    <FpcaExplain does="Takes the network out of this page. The codes become ordinary variables of FluoViz: you can plot them, colour other plots by them, cluster the samples on them and classify with them. The network can be saved and applied to new samples, and everything can be downloaded or described in a methods paragraph."
                 check="That the codes you pass on are the ones you checked in step 8: a network that was not better than PCA, or whose codes are not stable, is a poor basis for a classification. Save the network only when it is up to date."/>
    <AeGate/>
    <template v-if="ready">
      <AeStaleNote/>
      <AeSection title="Variables made by this network" sentence="They appear in the Plots page and in every selector of the program, under the name of this network.">
        <ul class="ae-varlist" data-ae-vars><li v-for="v in vars" :key="v.key" :data-name="v.name"><span class="chip">{{ v.name }}</span> <span class="small">{{ v.label }}</span></li></ul>
      </AeSection>
      <AeSection title="Use the codes" sentence="The codes are numbers like any other: send them to the clustering and classification page, or plot them.">
        <div class="row wrap">
          <button type="button" class="btn primary" data-ae-to-cluster @click="toCluster"><Icon name="bullseye" size="16"/>Cluster the latent space</button>
          <button type="button" class="btn" data-ae-to-classify @click="toClassify"><Icon name="analysis" size="16"/>Classify with the latent space</button>
          <button type="button" class="btn" data-ae-to-plots @click="toPlots"><Icon name="chart-pie" size="16"/>Plot the codes</button>
        </div>
      </AeSection>
      <AeSection title="Keep the network" sentence="A saved model keeps the network (not your data). You can apply it to new samples, in this project or another, and use it as the first link of a classification.">
        <div class="row wrap">
          <SaveModel :rid="rid"/>
          <button type="button" class="btn" data-ae-to-models @click="toModels"><Icon name="files-landscapes-alt" size="16"/>Apply a saved model</button>
          <button type="button" class="btn quiet" @click="goStep(5)"><Icon name="arrow-left" size="15"/>Change the settings and train again</button>
        </div>
      </AeSection>
      <AeSection title="Downloads" sentence="The files are made from this network; the file names say what they contain.">
        <div class="fp-downloads" data-ae-downloads>
          <div class="fp-dl"><b>Codes (CSV)</b><span class="muted xs">One row per sample: the codes, the rebuilding error, the status, and for a variational autoencoder the uncertainty and the KL.</span><button type="button" class="btn sm" data-ae-download="codes" @click="dl('codes')"><Icon name="download-alt" size="15"/>Download</button></div>
          <div class="fp-dl"><b>Rebuilt samples (.npz)</b><span class="muted xs">The rebuilt EEMs and spectra (and, while the data are unchanged, the measured ones and the differences) for use in other programs.</span><button type="button" class="btn sm" data-ae-download="reconstruction" @click="dl('reconstruction')"><Icon name="download-alt" size="15"/>Download</button></div>
          <div class="fp-dl"><b>Training history (CSV)</b><span class="muted xs">The loss of every epoch of every start.</span><button type="button" class="btn sm" data-ae-download="history" @click="dl('history')"><Icon name="download-alt" size="15"/>Download</button></div>
          <div class="fp-dl" v-if="isVae"><b>Synthetic samples (.npz)</b><span class="muted xs">New samples drawn by the variational autoencoder: made up, not measured.</span><button type="button" class="btn sm" data-ae-download="synthetic" @click="dl('synthetic')"><Icon name="download-alt" size="15"/>Download</button></div>
        </div>
      </AeSection>
      <AeSection title="Methods paragraph" sentence="A text for the methods section of a report or paper: the data, the representation, the network, the training and the versions. It can be pasted as it is.">
        <Callout kind="critical" v-if="methods && methods.error">{{ methods.error }}</Callout>
        <div class="fp-wait muted" v-if="methods && methods.busy && !text">Writing the text …</div>
        <template v-if="text">
          <div class="fp-recipe" data-ae-methods>{{ text }}</div>
          <ul class="fp-list small" v-if="refs.length"><li v-for="r in refs" :key="r">{{ r }}</li></ul>
          <div class="row"><button type="button" class="btn" data-ae-copy @click="copy"><Icon name="copy-alt" size="15"/>Copy</button></div>
        </template>
      </AeSection>
    </template>
  </FpcaFrame>`,
};
