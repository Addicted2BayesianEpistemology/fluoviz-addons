// Plain-wording helpers of the pages of the model add-ons (shared: imported from /addon-web/common/web/words.js).
import { ref, onMounted, onBeforeUnmount } from 'vue';

/** 0.1234 -> '12.3 %' ('–' when it is not a number). */
export const pct = (x, d = 1) => (Number.isFinite(x) ? `${(100 * x).toFixed(d)} %` : '–');
/** 10432 -> '10 432' (a thin space between the thousands; '–' when it is not a number). */
export const nTxt = (n) => (Number.isFinite(n) ? String(Math.round(n)).replace(/\B(?=(\d{3})+(?!\d))/g, ' ') : '–');
/** plural(3, 'sample') -> '3 samples'; plural(1, 'sample') -> '1 sample'; plural(2, 'child', 'children') -> '2 children'. */
export const plural = (n, one, many) => `${nTxt(n)} ${n === 1 ? one : (many || `${one}s`)}`;
/** ['a', 'b', 'c'] -> 'a, b and c' */
export const listWords = (a) => (a.length <= 1 ? a.join('') : `${a.slice(0, -1).join(', ')} and ${a[a.length - 1]}`);

/** True while the window is narrow (phones, split screens): the strips use fewer pictures per row. */
export function useNarrow(px = 760) {
  const narrow = ref(typeof window !== 'undefined' && window.matchMedia ? window.matchMedia(`(max-width: ${px}px)`).matches : false);
  let mq = null;
  const on = (e) => { narrow.value = e.matches; };
  onMounted(() => { if (window.matchMedia) { mq = window.matchMedia(`(max-width: ${px}px)`); narrow.value = mq.matches; mq.addEventListener('change', on); } });
  onBeforeUnmount(() => { if (mq) mq.removeEventListener('change', on); });
  return narrow;
}
