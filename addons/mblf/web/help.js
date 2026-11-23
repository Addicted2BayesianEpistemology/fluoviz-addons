// The Help guide of the add-on "Multimodal latent factors": a list of pages on the left, the page on the right, previous / next at the bottom (the look of the guides of FluoViz).
// FluoViz loads this module from /addon-web/mblf/web/help.js (manifest: provides.help_tabs) and shows it as a tab of the Help page.  The page shown is in the address
// (#/help?tab=addon:mblf:mblf&topic=<id>).  The pages are in help/pages.js.
import { computed, watch, nextTick } from 'vue';
import { Icon } from '/static/js/icons.js';
import { route, go } from '/static/js/router.js';
import { MBLF_PAGES, MBLF_IDS } from './help/pages.js';
import { TAB } from './help/parts.js';

export default {
  name: 'MblfGuide',
  components: { Icon },
  setup() {
    const topic = computed(() => (MBLF_IDS.includes(route.query.topic) ? route.query.topic : MBLF_IDS[0]));
    const idx = computed(() => MBLF_IDS.indexOf(topic.value));
    const page = computed(() => MBLF_PAGES[idx.value]);
    const prev = computed(() => MBLF_PAGES[idx.value - 1] || null);
    const next = computed(() => MBLF_PAGES[idx.value + 1] || null);
    const open = (id) => go('/help', { tab: TAB, topic: id });
    watch(topic, () => nextTick(() => window.scrollTo({ top: 0 })));
    return { MBLF_PAGES, topic, idx, page, prev, next, open };
  },
  template: `
  <div class="guide" data-mblf-guide>
    <nav class="guide-nav" aria-label="Pages of the guide to the multimodal latent factors">
      <div class="eyebrow">Multimodal latent factors (add-on)</div>
      <button v-for="(p, i) in MBLF_PAGES" :key="p.id" class="guide-link" :class="{ on: p.id === topic }" :data-topic="p.id" :aria-current="p.id === topic ? 'page' : null" @click="open(p.id)">
        <span class="n">{{ i + 1 }}</span><span class="grow"><b>{{ p.title }}</b><small>{{ p.summary }}</small></span></button>
    </nav>
    <article class="guide-body">
      <div class="guide-head"><Icon :name="page.icon" size="22"/><div><div class="eyebrow">Page {{ idx + 1 }} of {{ MBLF_PAGES.length }}</div><h2 class="h2" data-mblf-guide-title>{{ page.title }}</h2></div></div>
      <component :is="page.component" :key="page.id"/>
      <div class="guide-pager" style="flex-wrap:wrap;gap:8px">
        <button class="btn" v-if="prev" style="white-space:normal;max-width:100%;height:auto;text-align:left" @click="open(prev.id)"><Icon name="angle-left" size="15"/>{{ prev.title }}</button><span class="grow"></span>
        <button class="btn primary" v-if="next" style="white-space:normal;max-width:100%;height:auto;text-align:left" @click="open(next.id)">{{ next.title }}<Icon name="angle-right" size="15"/></button>
      </div>
    </article>
  </div>`,
};
