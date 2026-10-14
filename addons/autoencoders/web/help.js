// The Help guide of the add-on "Autoencoders": a list of pages on the left, the page on the right, previous / next at the bottom (the same look as the guides "Experimental methods" and "Saved models").
// FluoViz loads this module from /addon-web/autoencoders/web/help.js when the add-on is installed (manifest: provides.help_tabs) and shows it as a tab of the Help page; its default export is a Vue component.
// The page shown is in the address (#/help?tab=addon:autoencoders:autoencoders&topic=<id>).  The pages are in help/pages.js.
import { computed, watch, nextTick } from 'vue';
import { Icon } from '/static/js/icons.js';
import { route, go } from '/static/js/router.js';
import { AE_PAGES, AE_IDS } from './help/pages.js';
import { TAB } from './help/parts.js';

export default {
  name: 'AutoencodersGuide',
  components: { Icon },
  setup() {
    const topic = computed(() => (AE_IDS.includes(route.query.topic) ? route.query.topic : AE_IDS[0]));
    const idx = computed(() => AE_IDS.indexOf(topic.value));
    const page = computed(() => AE_PAGES[idx.value]);
    const prev = computed(() => AE_PAGES[idx.value - 1] || null);
    const next = computed(() => AE_PAGES[idx.value + 1] || null);
    const open = (id) => go('/help', { tab: TAB, topic: id });
    watch(topic, () => nextTick(() => window.scrollTo({ top: 0 })));
    return { AE_PAGES, topic, idx, page, prev, next, open };
  },
  template: `
  <div class="guide" data-ae-guide>
    <nav class="guide-nav" aria-label="Pages of the guide to the autoencoders">
      <div class="eyebrow">Autoencoders (add-on)</div>
      <button v-for="(p, i) in AE_PAGES" :key="p.id" class="guide-link" :class="{ on: p.id === topic }" :data-topic="p.id" :aria-current="p.id === topic ? 'page' : null" @click="open(p.id)">
        <span class="n">{{ i + 1 }}</span><span class="grow"><b>{{ p.title }}</b><small>{{ p.summary }}</small></span></button>
    </nav>
    <article class="guide-body">
      <div class="guide-head"><Icon :name="page.icon" size="22"/><div><div class="eyebrow">Page {{ idx + 1 }} of {{ AE_PAGES.length }}</div><h2 class="h2" data-ae-guide-title>{{ page.title }}</h2></div></div>
      <component :is="page.component" :key="page.id"/>
      <div class="guide-pager">
        <button class="btn" v-if="prev" @click="open(prev.id)"><Icon name="angle-left" size="15"/>{{ prev.title }}</button><span class="grow"></span>
        <button class="btn primary" v-if="next" @click="open(next.id)">{{ next.title }}<Icon name="angle-right" size="15"/></button>
      </div>
    </article>
  </div>`,
};
