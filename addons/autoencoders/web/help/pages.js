// The pages of the guide "Autoencoders" (Help), in reading order.  The guide itself is ../help.js.
import { whatPage, pipelinePage } from './introPages.js';
import { functionalPage, trainingPage } from './functionalPages.js';
import { checksPage, toolsPage } from './readPages.js';
import { usePage, limitsPage } from './usePages.js';

export const AE_PAGES = [whatPage, pipelinePage, functionalPage, trainingPage, checksPage, toolsPage, usePage, limitsPage];
export const AE_IDS = AE_PAGES.map((p) => p.id);
