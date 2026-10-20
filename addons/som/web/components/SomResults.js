// Step 7 of the Self-organising map page: four cards with the main numbers and the tabs Map, Prototypes, Planes, Groups, Trajectories, Samples and Projection.
// Also `SomGate` (what steps 7 to 9 show while there is no map, while it is being opened, or when it cannot be opened) and `SomStaleNote`.
import { computed } from 'vue';
import { Icon } from '/static/js/icons.js';
import { Callout, Tabs } from '/static/js/components/Common.js';
import { FpcaFrame } from '/static/js/components/FpcaStepper.js';
import { FpcaExplain } from '/static/js/components/FpcaCommon.js';
import { fp, ctx, STALE_WORDS, somJobs } from './SomState.js';
import { SomStat, SomVerdict, pct, nTxt, fmt } from './SomCommon.js';
import { SomMapTab } from './SomMapTab.js';
import { SomPrototypesTab } from './SomPrototypes.js';
import { SomPlanesTab } from './SomPlanes.js';
import { SomGroupsTab } from './SomGroups.js';
import { SomTrajectoriesTab } from './SomTrajectories.js';
import { SomSamplesTab } from './SomSamples.js';
import { SomProjectionTab } from './SomProjection.js';

export const SomStaleNote = {
  name: 'SomStaleNote',
  components: { Icon },
  setup() {
    const stale = computed(() => (fp.som.res ? fp.som.res.result.stale || [] : []));
    const what = computed(() => stale.value.map((s) => STALE_WORDS[s] || s).join(', '));
    return { stale, what };
  },
  template: `<div class="callout warning" v-if="stale.length" data-som-stale><Icon name="exclamation-triangle"/><div class="grow"><b>The data changed since this map was computed</b> ({{ what }}). What is stored in the map (the positions, the planes, the prototypes, the checks) can still be looked at; the pictures that need the measured data cannot, and a saved model cannot be made from it. Use <b>Compute again with the same settings</b> in the bar above to compute it on the current data.</div></div>`,
};

export const SomGate = {
  name: 'SomGate',
  components: { Icon, Callout },
  setup() {
    const A = computed(() => fp.som);
    const computing = computed(() => somJobs.fit.state.running);
    return { A, computing, goStep: ctx.goStep };
  },
  template: `
  <div data-som-gate>
    <div class="fp-empty" v-if="!A.resBusy && !A.resError && !(A.res && A.resRid)" data-som-noresult>
      <p v-if="computing"><b>The map is being computed.</b> The results appear here when it is done.</p>
      <p v-else>No map has been computed yet. Choose the settings and press <b>Compute the map</b> in step 6, or press <b>Run with the recommended settings</b> in step 1.</p>
      <div class="row" style="justify-content:center"><button type="button" class="btn" @click="goStep(6)"><Icon name="arrow-left" size="15"/>Go to step 6 · Compute</button><button type="button" class="btn" @click="goStep(1)">Go to step 1</button></div>
    </div>
    <div class="fp-wait muted" v-else-if="A.resBusy && !A.res">Opening the map …</div>
    <Callout kind="critical" v-else-if="A.resError" data-som-result-error><b>This map could not be opened.</b> {{ A.resError }}</Callout>
  </div>`,
};

export const SomResults = {
  name: 'SomResults',
  components: { Icon, FpcaFrame, FpcaExplain, Tabs, SomGate, SomStaleNote, SomStat, SomVerdict, SomMapTab, SomPrototypesTab, SomPlanesTab, SomGroupsTab, SomTrajectoriesTab, SomSamplesTab, SomProjectionTab },
  setup() {
    const A = computed(() => fp.som), R = computed(() => fp.som.res);
    const ready = computed(() => !!(R.value && A.value.resRid));
    const q = computed(() => (R.value ? R.value.quality : null));
    const tabs = [{ id: 'map', label: 'Map' }, { id: 'prototypes', label: 'Prototypes' }, { id: 'planes', label: 'Planes' }, { id: 'groups', label: 'Groups' }, { id: 'trajectories', label: 'Trajectories' },
      { id: 'samples', label: 'Samples' }, { id: 'projection', label: 'Projection' }];
    const vsPca = computed(() => (R.value && R.value.verdicts ? R.value.verdicts.vs_pca : null));
    return { A, R, ready, q, tabs, vsPca, pct, nTxt, fmt,
      DOES: 'Shows the map. The grid is the map itself: every hexagon is a unit holding a prototype, and the dots are your samples, placed between the units they resemble. Colour the units by a plane to see how a quantity changes across the map; open a unit to see its prototype; look at the groups of units and at the samples one by one.',
      CHECK: 'That samples you know to be alike sit close together and samples you know to differ sit apart (colour the dots by a metadata column), and that no region of the map is a tangle of folded ridges. Step 8 puts numbers on how far the map can be trusted.' };
  },
  template: `
  <FpcaFrame :n="7" next-label="Checks" :no-next="!ready" :next-disabled="!ready">
    <FpcaExplain :does="DOES" :check="CHECK"/>
    <SomGate/>
    <template v-if="ready">
      <SomStaleNote/>
      <div class="fp-stats" data-som-cards>
        <SomStat label="Captured on held-out samples" :value="q.heldout_captured !== undefined ? pct(q.heldout_captured, 1) : '–'" :sub="'On the samples it was trained on: ' + pct(q.captured, 1)" help="The share of the variation between samples that the prototypes capture, measured on samples the map was not trained on."/>
        <SomStat label="Units" :value="q.n_units" :sub="q.n_empty + ' without a sample'"/>
        <SomStat label="Against the scatter of the first two components" :value="vsPca ? (vsPca.difference >= 0 ? '+' : '−') + Math.abs(vsPca.difference).toFixed(3) : '–'" :level="vsPca ? (vsPca.level === 'good' ? 'ok' : vsPca.level === 'warn' ? 'bad' : '') : ''" sub="neighbours kept, map minus scatter" help="Trustworthiness and continuity at five neighbours, averaged: how well each picture keeps the neighbours of the samples. Positive: the map keeps them better."/>
        <SomStat label="Groups" :value="R.clusters.k" :sub="R.p.n_outside + ' sample' + (R.p.n_outside === 1 ? '' : 's') + ' poorly described'"/>
      </div>
      <SomVerdict :level="vsPca ? vsPca.level : 'info'" v-if="vsPca" data-som-verdict-main>{{ vsPca.verdict }}</SomVerdict>
      <Tabs v-model="A.tab7" :tabs="tabs" data-som-tabs7/>
      <SomMapTab v-if="A.tab7 === 'map'"/>
      <SomPrototypesTab v-else-if="A.tab7 === 'prototypes'"/>
      <SomPlanesTab v-else-if="A.tab7 === 'planes'"/>
      <SomGroupsTab v-else-if="A.tab7 === 'groups'"/>
      <SomTrajectoriesTab v-else-if="A.tab7 === 'trajectories'"/>
      <SomSamplesTab v-else-if="A.tab7 === 'samples'"/>
      <SomProjectionTab v-else-if="A.tab7 === 'projection'"/>
    </template>
  </FpcaFrame>`,
};
