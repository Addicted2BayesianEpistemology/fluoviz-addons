// Step 7, tab "Nonlinear effects": which nonlinear trend does each code fit?  Four sections (Response curves, Where it bends, Local effects, Two codes); each starts with one sentence that
// says what it shows and how to read it.  The choices are remembered with the project (AeNlState.js) and the answers come from POST /{rid}/response | bends | local | pair (AeNlActions.js).
import { computed } from 'vue';
import { fp } from './AeState.js';
import { SECTIONS } from './AeNlState.js';
import { AeNlResponse } from './AeNlResponse.js';
import { AeNlBends } from './AeNlBends.js';
import { AeNlLocal } from './AeNlLocal.js';
import { AeNlPair } from './AeNlPair.js';

const TITLES = {
  response: 'Draw a measurement of the decoded EEM against the position of each code',
  bends: 'Show, pixel by pixel, how far the response to each code is from a straight line',
  local: 'Compare what one SD of a code does at different samples',
  pair: 'Move two codes together and see whether their effects add up',
};

export const AeNonlinearTab = {
  name: 'AeNonlinearTab',
  components: { AeNlResponse, AeNlBends, AeNlLocal, AeNlPair },
  setup() {
    const N = computed(() => fp.ae.nl);
    const section = computed({ get: () => (SECTIONS.some((s) => s.id === N.value.section) ? N.value.section : 'response'), set: (v) => { N.value.section = v; } });
    function key(e, i) {
      const step = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0; if (!step) return;
      e.preventDefault();
      const next = SECTIONS[(i + step + SECTIONS.length) % SECTIONS.length];
      section.value = next.id;
      const el = e.currentTarget.parentElement.querySelector(`[data-ae-nl-section="${next.id}"]`); if (el) el.focus();
    }
    return { SECTIONS, TITLES, section, key };
  },
  template: `
  <div class="col gap-12" data-ae-nonlinear-tab>
    <div class="card"><div class="card-body" style="padding:12px 16px"><div class="row wrap" style="gap:10px 22px;align-items:center">
      <div class="seg ae-seg-wrap" role="group" aria-label="Sections of the nonlinear effects" data-ae-nl-sections>
        <button v-for="(s, i) in SECTIONS" :key="s.id" type="button" :class="{ on: section === s.id }" :aria-pressed="section === s.id ? 'true' : 'false'" :title="TITLES[s.id]" :data-ae-nl-section="s.id" @click="section = s.id" @keydown="key($event, i)">{{ s.label }}</button>
      </div>
      <span class="muted xs ae-grow">A code is nonlinear when what it does to the EEM is not a simple proportion: these four views show where and how much.</span>
    </div></div></div>
    <AeNlResponse v-if="section === 'response'"/>
    <AeNlBends v-else-if="section === 'bends'"/>
    <AeNlLocal v-else-if="section === 'local'"/>
    <AeNlPair v-else/>
  </div>`,
};
