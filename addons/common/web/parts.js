// The three small components every step of a model page uses: a number card, a verdict notice and a titled block.  Shared: every add-on makes its own set with its own prefix, so that its
// classes and test hooks keep their names:   const { Stat, Verdict, Section } = makeParts({ name: 'Som', prefix: 'som' });   // classes som-stat, som-section; hook data-som-verdict
import { computed } from 'vue';
import { Icon } from '/static/js/icons.js';
import { HelpTip } from '/static/js/components/Common.js';

export function makeParts({ name, prefix }) {
  /** One number: a label (with an optional help tip), the value, and a line under it.  level: '' | good | warning | critical (the colour of the value). */
  const Stat = {
    name: `${name}Stat`,
    props: { label: String, value: [String, Number], sub: { type: String, default: '' }, level: { type: String, default: '' }, help: { type: String, default: '' } },
    components: { HelpTip },
    template: `
    <div class="fp-stat ${prefix}-stat" :data-level="level">
      <span class="l">{{ label }}<HelpTip v-if="help" :text="help"/></span>
      <span class="v" :class="level">{{ value }}</span>
      <span class="s" v-if="sub">{{ sub }}</span>
    </div>`,
  };

  /** A verdict in words: the server's sentence (or the page's), with the colour of how good it is.  level: good | info | warn | warning | critical */
  const Verdict = {
    name: `${name}Verdict`,
    components: { Icon },
    props: { level: { type: String, default: 'info' }, title: { type: String, default: '' } },
    setup(props) { return { kind: computed(() => (props.level === 'good' ? 'good' : props.level === 'warn' || props.level === 'warning' ? 'warning' : props.level === 'critical' ? 'critical' : 'info')) }; },
    template: `<div class="callout" :class="kind" data-${prefix}-verdict :data-level="level"><Icon :name="kind === 'good' ? 'check-circle' : kind === 'info' ? 'info-circle' : 'exclamation-triangle'"/><div class="grow"><b v-if="title">{{ title }} </b><slot/></div></div>`,
  };

  /** A block of a step: a heading, one sentence about what it is, then the controls. */
  const Section = {
    name: `${name}Section`,
    components: { HelpTip },
    props: { title: String, sentence: { type: String, default: '' }, help: { type: String, default: '' } },
    template: `<section class="${prefix}-section"><h3 class="fp-h3">{{ title }}<HelpTip v-if="help" :text="help"/></h3><p class="fp-note" v-if="sentence">{{ sentence }}</p><slot/></section>`,
  };
  return { Stat, Verdict, Section };
}
