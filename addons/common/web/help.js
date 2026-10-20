// The small components every Help guide of a model add-on uses: the way an interface word is written and the two kinds of link.  Shared: each guide makes its own set with the address of its
// Help tab, so that a link without a `tab` opens a page of THAT guide:   const { Ui, GLink, PLink } = helpParts('addon:som:som');
import { go } from '/static/js/router.js';

export function helpParts(TAB) {
  /** A word of the interface (a button, a field, a step, a tab).  The test of each guide looks for every one of them in its docs/ui.md. */
  const Ui = { name: 'Ui', template: `<i class="ui"><slot/></i>` };

  /** A link to another page of the guide.  Without `tab` it is a page of this guide; tab="experimental" is the guide "Experimental methods", tab="saved" the guide "Saved models" (both of FluoViz). */
  const GLink = {
    name: 'GLink',
    props: { to: { type: String, required: true }, tab: { type: String, default: TAB } },
    setup(props) { return { open: () => go('/help', { tab: props.tab, topic: props.to }) }; },
    template: `<a class="g-link" href="#" @click.prevent="open"><slot/></a>`,
  };

  /** A link to a page of the program (the page of the add-on, Settings, Plots ...). */
  const PLink = {
    name: 'PLink',
    props: { path: { type: String, required: true }, query: { type: Object, default: null } },
    setup(props) { return { open: () => go(props.path, props.query || undefined) }; },
    template: `<a class="g-link" href="#" @click.prevent="open" :data-plink="path"><slot/></a>`,
  };
  return { Ui, GLink, PLink };
}
