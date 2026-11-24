// The pages of the guide "Multimodal latent factors" (Help), in reading order.  The guide itself is ../help.js.
import { parts } from './parts.js';

const page = (id, icon, title, summary, template) => ({ id, icon, title, summary, component: { components: parts, template: `<div>${template}</div>` } });

export const whatPage = page('what', 'lightbulb-alt', 'What the model is', 'Shared factors, each sample’s own part, noise; Bayesian, so everything has its uncertainty',
`<p class="lead">The <b>multimodal latent factor model</b> (Ursino, Epifani and Trovò; the software is <i>mblf-python</i>) describes several kinds of measurement of the same samples at once — here the fluorescence (EEMs), the absorbance spectra and, if you wish, other numbers you measured. Each kind is a <b>mode</b>. A few <b>shared factors</b> say how all the modes vary together from sample to sample; every sample and mode also keeps a smooth part of <b>its own</b>; and every measurement has <b>noise</b>.</p>
<Fig><FigModel/><template #caption>The structure of the model. Covariates (metadata) shift the factors; the factors feed every mode through its <b>loading functions</b>; each mode adds a part of its own and noise.</template></Fig>
<h2>Why it is useful for EEMs and absorbance</h2>
<ul>
<li><b>What the measurements share.</b> A factor is a pattern that appears together in the EEM and in the absorbance spectrum (and in the values): the loading functions show where.</li>
<li><b>Predicting one measurement from the others.</b> Absorbance is quick and cheap; an EEM is slower; a laboratory value (DOC) is expensive. The model says how well one predicts another, for the population and for every sample, with honest intervals.</li>
<li><b>Gaps are no problem.</b> A sample without an absorbance spectrum, without a laboratory value, or with a region of its EEM cut out, contributes what it has; the model predicts what it lacks.</li>
<li><b>Covariates.</b> The effect of site, season or treatment on every wavelength and pixel, with its uncertainty.</li>
</ul>
<h2>Why Bayesian</h2>
<p>The posterior is sampled with Stan: every result is a <b>distribution</b>, not one number. The page never shows a curve without its band or a share without its interval, and the predictions carry every source of uncertainty there is (page 5). How to read the bands is page 4.</p>
<Callout kind="warning"><b>An add-on, and a new method.</b> The paper that describes the model is in preparation. The page checks itself: convergence of the sampler, adequacy of the bases, and held-out calibration of the predictions (pages 6 and 7). Read them before you rely on a result.</Callout>`);

export const combosPage = page('combos', 'layer-group', 'The combinations', 'One mode or several, values, covariates: what each combination answers',
`<p class="lead">In <Ui>1 · What to model</Ui> you choose the modes (the fluorescence, the absorbance, or both), optional <Ui>Values to model (optional)</Ui> and optional <Ui>Covariates (optional)</Ui>. Every combination is a different question.</p>
<table class="table">
<thead><tr><th>You choose</th><th>The model answers</th></tr></thead>
<tbody>
<tr><td>Only the fluorescence (or only the absorbance)</td><td>A Bayesian functional factor analysis: the patterns of that measurement with credible bands, how many the data really use, and the reconstruction of regions a sample lacks.</td></tr>
<tr><td>Fluorescence and absorbance</td><td>What the two share; how much of every pixel the absorbance predicts (and of every wavelength the EEM predicts); the predicted EEM of a sample from its spectrum, with its uncertainty.</td></tr>
<tr><td>… and values (a laboratory value, an index)</td><td>Predicting the value from the spectra (function-to-value regression) with a predictive interval; which wavelengths carry the information.</td></tr>
<tr><td>… with covariates</td><td>The effect of every covariate on every wavelength and pixel; how much of each mode they explain; predictions that use them, or not.</td></tr>
</tbody></table>
<h2>A value or a covariate?</h2>
<p>A <b>value</b> is something the factors should <b>describe and predict</b>: it has noise, it may be missing, and the model predicts it. A <b>covariate</b> <b>explains</b>: it shifts the scores of the factors, it must be known for every sample used, and the model never predicts it. The same variable cannot be both.</p>
<h2>One fit, many combinations of predictors</h2>
<p>You do not need one model per combination of predictors. A model of the fluorescence, the absorbance and the values can predict any of them from any of the others, with or without the covariates: <Ui>What tells what</Ui> (step 6) and <Ui>Every choice of sources, held out</Ui> (step 7) compare all the combinations from the same fit. Fit separate models only to compare different modes or covariates as a whole.</p>`);

export const stepsPage = page('steps', 'list-ul', 'The eight steps', 'What to model, domain, basis, model and sampler, fit, results, prediction, checks',
`<p class="lead">The page <PLink path="/mblf"><Ui>Multimodal latent factors (add-on)</Ui></PLink> is one guided pipeline of eight steps. Each step has a box that says what it does and what to check.</p>
<p><b><Ui>1 · What to model</Ui></b>: the modes, their pre-treatment, the values and the covariates, and <Ui>Use samples that lack a measurement (they contribute what they have, and the model predicts what they lack)</Ui>.</p>
<p><b><Ui>2 · Domain</Ui></b> and <b><Ui>3 · Basis</Ui></b> are the steps of the Functional PCA page: which pixels take part, and the Gaussian-process basis. The share of the prior variance the basis keeps (99 % by default) has a meaning: what it leaves out is the expected relative truncation error. Under <Ui>Where the basis functions come from</Ui> you can choose <Ui>Functional PCA of your samples</Ui> instead: the principal components of your smoothed EEMs and spectra become the building blocks (fewer functions, a faster model, but learned from the same samples). <Ui>A functional PCA you fitted</Ui> uses the eigenfunctions of an analysis of the Functional PCA page, chosen in <Ui>Fitted functional PCA</Ui>: the model then takes that analysis' domain and pre-treatment, so it reads every sample exactly as that analysis did. The Gaussian-process basis can also be a tensor product of one-dimensional bases (its <i>Structure</i>): every building block is then an emission profile times an excitation profile, the structure of a fluorophore. A basis of more than 150 functions makes the fit extremely slow: it needs <Ui>Allow more than 150 basis functions</Ui>. No smoothing is chosen: the model separates the signal from the noise itself.</p>
<p>The card <Ui>What the model sees</Ui> (step 3) says how every EEM and spectrum is sent to the model. <Ui>Log scale: log10(fluorescence + offset)</Ui> makes a change by a factor (a fluorophore twice as strong) the same step everywhere, so the brightest peaks no longer dominate; the <Ui>Offset</Ui> (empty = 1 % of the 99th percentile of the values) keeps the noise near zero from blowing up, and a value at or below minus the offset counts as not measured. Everything is then read in log10 units. <Ui>Centred on</Ui>: <Ui>Mean</Ui> (default), <Ui>Median</Ui> (less moved by a few unusual samples) or <Ui>None</Ui> (only for data already centred). <Ui>Scaled by</Ui>: <Ui>One number</Ui> (default: keeps the shape of the variation), <Ui>Pareto</Ui>, <Ui>Every point</Ui> (every point counts the same, noise included) or <Ui>None</Ui> (the data units: the default priors then may not fit). In step 1 the <b>log</b> box of a value or a numeric covariate models it as its log10, and <Ui>How they enter the model</Ui> sets, for every value and covariate, how it is centred (<Ui>Mean</Ui>, <Ui>Median</Ui>, <Ui>Not centred</Ui>; a category: <Ui>Centred indicators</Ui> or <Ui>0 / 1 indicators</Ui>) and scaled (<Ui>Standard deviation</Ui>, <Ui>Robust (MAD)</Ui>, <Ui>Not scaled</Ui>: a covariate's effect then reads per unit). Keep covariates centred on their mean unless you mean otherwise: the data are centred on the mean of the samples, and a covariate centred elsewhere ties that mean to the samples at its centre (a category with 0 / 1 indicators: to the reference). With two values or more, <Ui>Basis of the values</Ui> offers <Ui>Principal components</Ui>: fewer coefficients when the values move together.</p>
<p><b><Ui>4 · Model and sampler</Ui></b>: the <Ui>Largest number of factors</Ui> (a cap: unneeded factors are shrunk away), the estimation (<Ui>Full posterior (MCMC)</Ui> or <Ui>Maximum likelihood (quick)</Ui>), the length of the sampling, the priors, and <Ui>The Stan engine</Ui> (page 8).</p>
<p><b><Ui>5 · Fit</Ui></b> says what will be done and fits (<Ui>Fit the model</Ui>; <Ui>Cancel</Ui> stops the sampler and saves nothing). A fitted model is stored with the project as <i>mblf-1</i>, <i>mblf-2</i> …</p>
<p><b><Ui>6 · Results</Ui></b>: <Ui>Overview</Ui>, <Ui>Factors</Ui>, <Ui>What tells what</Ui>, <Ui>How the variation splits</Ui>, <Ui>Covariates</Ui>, <Ui>Categories</Ui>, <Ui>Samples</Ui>. <b><Ui>7 · Prediction and uncertainty</Ui></b>: <Ui>One sample</Ui> and <Ui>Every choice of sources, held out</Ui>. <b><Ui>8 · Checks and use</Ui></b>: <Ui>Convergence</Ui>, <Ui>Basis</Ui>, <Ui>Use the model</Ui>.</p>`);

export const bandsPage = page('bands', 'chart-line', 'Reading the uncertainty', 'Fans of bands, the band for the whole curve, the sign of a pattern, intervals of shares',
`<p class="lead">Every function the page draws comes with its posterior (or predictive) distribution. Four ways of showing it are used everywhere.</p>
<h2>The fan: bands at every point</h2>
<Fig><FigFan/><template #caption>The line is the median (or the mean); the darkest band holds the value with 50 % probability at that point, the lighter ones 80 % and 95 %. The dashed lines hold the <b>whole curve</b> with 95 % probability.</template></Fig>
<p>A pointwise band answers "where is the value at this wavelength?". It does <b>not</b> say that the whole curve stays inside: with 100 wavelengths, a curve will leave a 95 % pointwise band somewhere. The <b>band for the whole curve</b> (switch <Ui>Band for the whole curve</Ui>; a simultaneous band, from the largest standardised deviation of whole draws) holds the entire curve: use it to say "the pattern differs from zero over this range".</p>
<h2>Pictures of an EEM</h2>
<p>An EEM cannot carry a band, so the picture shows one statistic at a time: <Ui>Mean</Ui>, <Ui>2.5 %</Ui> and <Ui>97.5 %</Ui> (the limits of the central 95 % interval), <Ui>Width</Ui> (of that interval: where it is uncertain), <Ui>P(> 0)</Ui> (the posterior probability of a positive value: near 1 or 0 = the sign is sure, near 0.5 = it is not), and <Ui>Sure sign</Ui> (the mean, only where the probability of the opposite sign — the local false sign rate — is below 5 %). The last two answer "where is this pattern credibly present?".</p>
<h2>Is a factor one fluorophore?</h2>
<p>PARAFAC assumes that every component of an EEM is one emission spectrum times one excitation spectrum (trilinearity). The latent factor model does not impose it, but the <Ui>Factors</Ui> tab checks it: the card <Ui>Is it one fluorophore? The trilinear share of the EEM loading</Ui> gives, for the EEM loading of the factor, the share that the best product of one emission and one excitation profile holds (fitted on the pixels that exist), computed at every posterior draw, so it comes with its interval, and draws the two profiles. Near 100 %: the factor behaves like one fluorophore, or several that always change together; clearly lower: it mixes patterns with different spectra (for example two fluorophores that trade places).</p>
<h2>Values of very different size</h2>
<p>A chart of values offers <Ui>In SDs of each value</Ui> (default: every value in standard deviations of that value over the training samples, so a total fluorescence and a pH can be read side by side; hovering shows the value in its own units) and <Ui>Own units, one panel each</Ui>.</p>
<h2>Categories</h2>
<p>For a covariate with categories, every category can be chosen, the reference (the most frequent) included; the effect is first shown against the reference, and <Ui>Compared with</Ui> shows the difference between ANY two categories, with its bands. The tab <Ui>Categories</Ui> shows each category as a sample. <Ui>The category alone</Ui>: its indicator is 1, the others 0, every other covariate at the average of all the training samples — the pattern of the category itself, like an eigenface. <Ui>Its typical member</Ui>: the covariates take their average over the training samples of that category, so the other covariates come along as they are in it. For each, the expected EEM or spectrum (only the uncertainty about the population), <Ui>Difference from the average sample</Ui>, and what a new member would look like, with its uncertainty budget.</p>
<h2>Numbers with intervals</h2>
<p>A share (of variance, of predicted variation) is drawn as a bar with its 90 % credible interval and its posterior mean. The number of factors the data use has a whole posterior distribution (<Ui>How many factors the data use</Ui>): a spread over 2 and 3 is honest uncertainty, not a failure.</p>
<h2>Why the factors are aligned</h2>
<p>Factors are defined only up to a rotation: two posterior draws can describe the same patterns in a different order or sign. Every draw is rotated (varimax in the L2 geometry of the loading functions of all the modes at once) and matched in sign and order to a reference (Varimax-RSP), then the factors are ordered by strength. The bands of a factor are bands under this convention. Predictions, shares, covariate effects and correlations do not depend on it.</p>`);

export const budgetPage = page('budget', 'adjust-circle', 'The four parts of the predictive uncertainty', 'Parameters, scores, the sample’s own part, noise: what would shrink each one',
`<p class="lead">When the page predicts a mode of a sample from what is known about it (<Ui>7 · Prediction and uncertainty</Ui>), the predictive variance at every point is <b>exactly</b> the sum of four parts. The page shows their shares: the <b>uncertainty budget</b>.</p>
<Fig><FigBudget/><template #caption>The budget of one prediction: each part has a different cause and a different cure.</template></Fig>
<table class="table"><thead><tr><th>Part</th><th>What it is</th><th>What would shrink it</th></tr></thead><tbody>
<tr><td><b>Population parameters</b></td><td>The posterior uncertainty of the loadings, variances and covariate effects: the conditional mean changes from draw to draw.</td><td>More training samples. It is zero in a maximum-likelihood fit, which ignores it.</td></tr>
<tr><td><b>Factor scores of the sample</b></td><td>The sources do not fix the factors of THIS sample exactly.</td><td>More, or more precise, measurements of the sample (another mode, a covariate, a complete spectrum).</td></tr>
<tr><td><b>The sample's own variation</b></td><td>The smooth part of the target that is specific to the sample and the mode.</td><td>Nothing: no other measurement can tell it. Only measuring the target itself.</td></tr>
<tr><td><b>Measurement noise</b></td><td>The noise of the instrument.</td><td>Only for <Ui>A new measurement (include the noise of the instrument)</Ui>; the latent function has none.</td></tr>
</tbody></table>
<h2>How it is computed</h2>
<p>At every posterior draw the prediction is Gaussian, given the sample's sources: the scores of the sample have a Gaussian posterior, and the target is a linear function of them plus its own part. The draws are averaged with equal weights (the "modular" rule of the method: a new sample informs its own scores, not the population). The bands are quantiles of this mixture, computed exactly.</p>
<h2>Held out or not</h2>
<p>For a training sample, <Ui>Held out (as if the model had never seen this sample)</Ui> reweights the draws as if the sample had been left out of the fit (Pareto-smoothed importance sampling). Compare the prediction with what was measured only when it is held out: otherwise the sample has helped to make its own prediction.</p>
<h2>Is it worth measuring?</h2>
<p>Switch a source on and off: the part <b>scores</b> grows when an informative source is removed. When the part <b>the sample's own variation</b> dominates, no other measurement will help: the target has to be measured.</p>
<h2>What the prediction missed</h2>
<p>The budget says how large the error is expected to be; when the sample has the target, <Ui>What the prediction missed</Ui> shows the actual error: the measured minus the predicted (<Ui>Measured − predicted</Ui>), or the same divided by the predictive standard deviation (<Ui>In predictive SDs</Ui>: beyond ±2 at about 5 % of the points if the uncertainty is honest). Structure in it — a peak, a slope — is what the sources could not tell about this sample.</p>`);

export const heldoutPage = page('heldout', 'bullseye', 'Honest checks of the predictions', 'Leave-one-out from one fit, coverage, calibration, the PIT, the one-standard-error choice',
`<p class="lead">A predictive interval is only useful if it contains the measured value as often as it claims. <Ui>Every choice of sources, held out</Ui> checks it for every combination of sources, on every training sample that has the target, each time as if that sample had been left out.</p>
<h2>Leaving samples out without fitting again</h2>
<p>Pareto-smoothed importance sampling (Vehtari et al.) reweights the posterior draws so that they describe the fit without one sample; its <b>Pareto k</b> says when this works (above 0.7 the sample is so influential that the estimate is rough: the sample deserves a look in <Ui>Samples</Ui>).</p>
<h2>The numbers of the table</h2>
<ul>
<li><b>Log density per sample</b> and <b>gain over nothing</b>: the held-out log predictive density of the measured values, the score that rewards both accuracy and honest uncertainty. The gain is paired with its standard error.</li>
<li><b>Inside 50 / 80 / 95 %</b>: the share of measured values inside the intervals. Near the nominal value: honest. Below: over-confident. Above: too cautious.</li>
<li><b>Error (RMS)</b> and <b>width of the 80 % interval</b>, in the units of the data.</li>
</ul>
<p>The <b>calibration</b> chart puts the three coverages against their nominal values (the dotted diagonal is perfect); the <b>PIT</b> histogram shows where every measured value falls in its own predictive distribution: flat = calibrated; U-shaped = intervals too narrow; a hump in the middle = too wide; tilted = biased.</p>
<h2>The simplest good choice</h2>
<p>The best combination of sources has the highest log density. A simpler combination whose loss against the best is within one paired standard error is marked <i>simplest within 1 SE</i>: it is the one-standard-error rule of the method, and the cheaper set of measurements to use in practice.</p>
<h2>And for the population</h2>
<p><Ui>What tells what</Ui> (step 6) gives the share of the variation of every mode that the other modes and the covariates predict, as a property of the population with its posterior. The held-out table checks it on real samples.</p>`);

export const checksPage = page('checks', 'shield-check', 'Convergence and the basis', 'R-hat, effective sample size, divergences; is what the basis leaves out only noise?',
`<p class="lead"><Ui>8 · Checks and use</Ui> says how far the fit can be trusted.</p>
<h2>Convergence</h2>
<p>Several chains start from different points; if they describe the same posterior, R-hat is close to 1 (at most 1.01) and the effective sample size (ESS) is large (at least 400). Because factors can rotate, these checks use rotation-invariant quantities only: the log density, the variances, the coefficients of every sample, shared variances, the strengths of the factors, the cross-covariance between modes, the covariate effects. <b>Divergent transitions</b> mean the sampler met a region it could not explore: raise <Ui>Target acceptance rate</Ui> (0.95). Not converged? Run <i>Thorough</i>, or lower the cap of factors.</p>
<h2>The basis</h2>
<p>What a Gaussian-process basis cannot represent of a sample goes to the noise. The check compares, for every mode, what the basis leaves out with noise passed through the same projection: its energy, its profile over the wavelengths, and its lag-1 autocorrelation (a structure left over correlates neighbouring wavelengths). If it does not look like noise, keep a larger share of the prior variance, or choose shorter length-scales, in step 3.</p>`);

export const enginePage = page('engine', 'cog', 'The Stan engine', 'Why a private environment, what is downloaded, how to remove it',
`<p class="lead">The posterior is sampled by <b>Stan</b> (CmdStan), which compiles every model to a program with a C++ compiler. In <Ui>4 · Model and sampler</Ui>, <Ui>The Stan engine</Ui> says whether it is ready.</p>
<ul>
<li><b>Prepared once.</b> <Ui>Prepare the Stan engine (about 450 MB, once)</Ui> downloads CmdStan, its compiler and a few Python packages from conda-forge into the add-on's own folder (inside FluoViz's <i>runtime</i> folder). FluoViz itself is not changed: its own environment uses other compilers, which installing CmdStan there would replace.</li>
<li><b>The first fit compiles the model</b> (about a minute); later fits start at once.</li>
<li><b>Remove it</b> with <Ui>Remove the Stan engine</Ui>; it is prepared again when you need it. Removing the add-on keeps it (it is in the add-on's data folder), so remove it first if you want the space back.</li>
<li>CmdStan cannot compile in a folder whose path contains a space: install FluoViz in such a folder and the page says so.</li>
</ul>`);

export const savedPage = page('saved', 'files-landscapes-alt', 'Saved models and new samples', 'What a saved model does with new samples, and its limits',
`<p class="lead"><Ui>Use the model</Ui> in step 8 saves the model (the posterior of the population, not your data). In <PLink path="/models">Saved models</PLink> it can be applied to new samples, in this project or another.</p>
<ul>
<li>Every new sample is read exactly like the training samples (the same pre-treatment, mean and scale; nothing is estimated from the new samples).</li>
<li>It gets its <b>factor scores</b> with their standard deviations, and, for every <b>value</b> the model was fitted on, a <b>predicted value with its 95 % predictive interval</b> from what the sample has (every part of the uncertainty budget included).</li>
<li><b>How typical</b> the sample is: the log predictive density of its measurements per value, against the training samples with the same measurements, scored held out; below the 99 % (or 95 %) limit it is <i>Poorly described by the model</i>.</li>
<li>Looking at one sample shows its predicted EEM (or spectrum) with its limits, its uncertainty and its budget.</li>
</ul>
<Callout kind="info">The data recipe of FluoViz needs every measurement a saved model was fitted on: a model of the fluorescence and the absorbance needs both for every new sample. Its values and covariates are read from the metadata when they are there, and predicted (values) or treated as unknown (covariates) when they are not. Within the page of a result, samples that lack a measurement are fully supported.</Callout>`);

export const refsPage = page('refs', 'document-info', 'References', 'The method, the sampler, the checks',
`<ul>
<li>B. Ursino, I. Epifani and F. Trovò. <i>A Bayesian latent factor approach for multimodal functional data</i> (manuscript). Software: mblf-python 1.0.0, MIT licence (carried by this add-on).</li>
<li>A. Bhattacharya and D. B. Dunson (2011). Sparse Bayesian infinite factor models. <i>Biometrika</i> 98, 291–306.</li>
<li>B. Carpenter et al. (2017). Stan: a probabilistic programming language. <i>Journal of Statistical Software</i> 76.</li>
<li>P. Papastamoulis and I. Ntzoufras (2022). On the identifiability of Bayesian factor analytic models. <i>Statistics and Computing</i> 32, 23.</li>
<li>A. Vehtari, A. Gelman and J. Gabry (2017). Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. <i>Statistics and Computing</i> 27, 1413–1432.</li>
<li>A. Vehtari et al. (2021). Rank-normalization, folding, and localization: an improved R-hat. <i>Bayesian Analysis</i> 16, 667–718.</li>
<li>M. Stephens (2017). False discovery rates: a new deal. <i>Biostatistics</i> 18, 275–294 (the local false sign rate).</li>
</ul>`);

export const MBLF_PAGES = [whatPage, combosPage, stepsPage, bandsPage, budgetPage, heldoutPage, checksPage, enginePage, savedPage, refsPage];
export const MBLF_IDS = MBLF_PAGES.map((p) => p.id);
