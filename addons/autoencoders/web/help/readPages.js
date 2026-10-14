// Help > Autoencoders, pages 5-6: reading the checks (step 8) and the exploration tools (step 7).
// tests/test_help_autoencoders.py checks every word of the interface (<Ui>...</Ui>) against the page and docs/ui.md, and every number against the add-on's code.
import { parts } from './parts.js';
import { FigCollapse, FigTraversal } from './figures.js';

const comps = { ...parts, FigCollapse, FigTraversal };

export const checksPage = {
  id: 'checks',
  icon: 'check-circle',
  title: 'Reading the checks',
  summary: 'Variance explained on held-out samples, the comparison with PCA, how many codes, collapsed codes, stability, smoothness',
  component: {
    components: comps,
    template: `
    <div>
      <p class="lead">Step <Ui>8 · Checks</Ui> has five tabs, and each starts with a verdict in words that the program writes from the numbers. This page explains the numbers behind the verdicts and what to do about them. The order matters: first <b>did the network learn something that carries over to new samples</b>, then <b>is it better than the simple method</b>, then <b>how many codes</b>, and only then the finer questions.</p>

      <h2>Variance explained, on samples the network did not learn from</h2>
      <p>The tab <Ui>Quality</Ui> opens with <Ui>Variance explained</Ui>: the share of the variation between your samples that the rebuilt EEMs hold. 1 (100 %) would be a perfect rebuilding; 0 means no better than drawing the average EEM every time. It is given for the <b>held-out</b> samples (the honest number) and for the training samples, and for each variable when there are two, together with the <b>RMSE</b> in the units of your data (the typical size of the difference between a measured and a rebuilt pixel). The <b>rebuilding error of one sample</b> is the other side of the same number: the share of the typical sample-to-sample variation that the network does not rebuild for that sample. When there are fewer than 30 samples every sample is rebuilt by a network that did not see it (page 4), and the number is for all of them. The verdict puts words on the held-out number: <b>excellent</b> (99 % or more), <b>good</b> (95 % or more), <b>fair</b> (90 % or more; "More codes may help") and <b>poor</b> (below 90 %; "Try more codes, a larger network or more training epochs").</p>
      <ul class="g-list">
        <li><b>The gap.</b> The training value is always the higher. The tab <Ui>Training</Ui> compares the two errors: the verdict is <b>clear overfitting</b> when the held-out error is more than three times the training error and more than one point of the variation above it, <b>mild overfitting</b> when it is more than 1.6 times and more than 0.3 point above, and otherwise no sign of overfitting (a five-fold ratio between two tiny errors is not overfitting). Overfitting is memorising (page 4). The verdict also says which epoch gave the best weights and why the training stopped.</li>
        <li><b>The worst samples.</b> <Ui>Worst rebuilt samples</Ui> lists them; a click opens the sample. Look at the picture of the difference: noise-like is fine; a <b>structured</b> difference (a band, a peak) is something the network does not know: an unusual sample, or a limit of the model.</li>
        <li><b>The maps.</b> <Ui>Where the network rebuilds well</Ui> is the variance explained pixel by pixel (the local R²); <Ui>Typical size of the difference</Ui> is the size of the rebuilding error pixel by pixel. A bad region at the edge of a removed band means the basis or the smoothing is cutting there (steps 3 and 4); a bad region on a peak means the network lacks a code for what varies there.</li>
        <li><b>The histogram.</b> <Ui>Rebuilding error of the samples</Ui> shows the error of every sample with the limit that flags the unusual ones.</li>
      </ul>

      <h2>Is the nonlinearity worth it?</h2>
      <p>The card <Ui>Is the nonlinearity worth it?</Ui> compares the network with PCA, or with functional PCA when the network works on a basis, <b>on the same held-out samples and with the same number of codes</b>. The card <Ui>Against PCA</Ui> (or <Ui>Against functional PCA</Ui>) of step 7 shows the difference in points of variance explained, network minus linear method, and the sentence of the verdict says in words which of three cases it is. The network is <b>better</b> only when all three hold: its error is more than 15 % lower than the linear method's on the same held-out samples, the comparison sample by sample is clear (a paired test), and the two shares of variance explained differ by at least 0.3 point. <b>Worse</b> is the mirror image. Otherwise it is <b>about the same</b>, and the verdict says that the linear method is the safer choice. With fewer than 5 held-out samples there is no verdict. The button <Ui>Compare with PCA for every number of codes</Ui> adds the linear method's curve for 1, 2, 3 … codes: it needs no training and is instant.</p>
      <div class="table-wrap"><table class="table g-table"><thead><tr><th>Verdict</th><th>What it means</th><th>What to do</th></tr></thead><tbody>
        <tr><td>Better</td><td>The bends pay: the data are curved, or the network is using a code better than a straight line can.</td><td>Check the gap between training and held-out numbers, and the tab <Ui>Stability</Ui>: a gain that is not stable is chance.</td></tr>
        <tr><td>About as good</td><td>The data are essentially linear at this size. Nothing is gained in rebuilding.</td><td>Use PCA or functional PCA, which are exact and reproducible. Keep the network only for what PCA cannot do (a VAE, the anomaly view, the saved model).</td></tr>
        <tr><td>Worse</td><td>Rare with the shortcut on (the network starts as PCA and keeps that start when nothing beats it on the held-out samples); more likely with the shortcut off: too few samples, too many codes, a network too large, or too much smoothify.</td><td>Page 4: fewer codes, smaller network, more weight decay; or use PCA.</td></tr>
      </tbody></table></div>
      <Callout kind="info">A network can only appear better than PCA by <b>memorising</b> if the two are compared on samples it learned from. Here both are judged on held-out samples, so a gain is a real one, up to the small optimism of choosing the best of several starts. When the network starts from PCA and no epoch beats the start on the held-out samples, the page says "The network could not improve on its PCA starting point on held-out samples": the straight-line part is then the answer, and the network is the PCA.</Callout>

      <h2>How many codes?</h2>
      <p>The <Ui>Latent-dimension scan</Ui> (in step 5, and again in the tab <Ui>Latent dimensions</Ui>) trains small networks for 1, 2, 3 … codes (with <Ui>Largest number of codes</Ui> as the limit, and optionally <Ui>Also try the other type of network</Ui>) and draws <Ui>Error on held-out samples against the number of codes</Ui>, with the linear method beside it. The page recommends <b>the smallest number of codes whose held-out error is within one standard error (or 1 % of the variation) of the best</b>: the simplest network that is, as far as the samples can tell, as good as the best. The button <Ui>Use N codes</Ui> sets it. It takes minutes, because many networks are trained (each from one start).</p>
      <ul class="g-list">
        <li>A curve that falls and then <b>flattens</b>: the bend is the number of codes the data support.</li>
        <li>A curve that keeps falling: the information is spread over many directions; take as many codes as the samples allow (one tenth of the number of samples), or use another representation.</li>
        <li>The network's curve <b>below</b> PCA's by one or two codes: that is the saving the bends give.</li>
      </ul>

      <h2>The health of the codes</h2>
      <p>The tab <Ui>Latent dimensions</Ui> opens with a sentence on the health of the codes (are they all used? are some inactive?) and the cards <Ui>Effective number of codes</Ui> and <Ui>Typical typicality of the codes</Ui>.</p>
      <ul class="g-list">
        <li><b>For an autoencoder:</b> how many codes are <b>unused</b>: a code is unused when it accounts for less than 1 % of what the codes do to the rebuilt EEM. The <b>effective dimension</b> says how many codes are really used. If you asked for six codes and the effective dimension is 2.3, the network uses about two directions: ask for fewer. The page also names two codes that are strongly correlated (above 0.5): the network spends two codes on one direction.</li>
        <li><b>For a variational autoencoder:</b> <Ui>KL of each code</Ui>, the amount by which each code departs from the prior cloud (page 1). A code whose KL is below 0.05 nats (the unit of KL) has <b>collapsed</b>: its blobs are the same for every sample and equal to the prior, so it carries nothing. A code is <b>inactive</b> when it is unused (less than 1 % of the effect on the rebuilt EEM) or collapsed, and the page marks the inactive codes in grey. The page also says whether the codes of the training samples fill the prior cloud (fewer than 85 % of the samples inside its 95 % region means they do not): if they do not, samples drawn from it may not look like real ones.</li>
      </ul>
      <FigCollapse>The KL of each of six codes of a variational autoencoder, and the share of the effect on the rebuilt EEM that each code has (real numbers; made-up EEMs in which only two things vary). The codes that carry information have tall bars and almost all of the effect. A code whose KL is below the dashed line has collapsed onto the prior. A code can also keep some KL and still have almost no effect on the rebuilt EEM: it is unused. The page marks every inactive code and says how many codes would be as good.</FigCollapse>
      <p>A collapsed code is not necessarily a failure: when the data need two codes and you asked for six, four collapsing is the right answer, and the number of active codes is the number to use. It is a problem when <b>all</b> codes collapse (the network ignores the codes and rebuilds the mean), or when you expected more structure. Remedies: fewer codes; a <b>smaller β</b> (<Ui>Weight of the typical-code term (β)</Ui>, below 1 gives sharper rebuilt samples); a longer <Ui>Warm-up (epochs)</Ui>, which lets the network learn to use the codes before the pull towards the prior grows; <Ui>Free bits (nats per code)</Ui>, which lets every code have a little information for free (Bowman et al. 2016; Kingma et al. 2016).</p>
      <p><Ui>Correlation between the codes</Ui> shows whether two codes carry the same information. A deterministic autoencoder does not make its codes independent, so correlations are normal; a correlation near 1 between two codes means one of them is redundant.</p>

      <h2>Stability: would another run give the same codes?</h2>
      <p>The tab <Ui>Stability</Ui> has the button <Ui>Check the stability</Ui>. It trains the network again <Ui>Number of runs</Ui> times from other random starts (and, with the switch <Ui>Resample the samples (bootstrap)</Ui>, on resampled sets of samples, a <b>bootstrap</b>: new sets made by drawing samples from yours at random, with replacement, which mimics having measured another set of samples) and compares the codes of every pair of runs <b>after allowing for the sign, the order and the mixing that are arbitrary</b>. The chart <Ui>Agreement of the codes between runs</Ui> shows, for every pair of runs, the mean canonical correlation of their codes: 1 means that the codes of one run can be rebuilt exactly from the codes of the other by a straight-line recipe. A code counts as <b>found again</b> when the codes of the other runs can rebuild it with a multiple correlation of 0.8 or more. The verdict has three forms: "The codes are reproducible" (every code found again and a mean of 0.95 or more), "The codes are only partly reproducible" (at least half of the codes found again; the others are named) and "The codes are NOT reproducible" (fewer).</p>
      <p>If the codes are not reproducible, <b>do not interpret an individual code</b>: another run would give another code. The space the codes span may still be good (look at whether the rebuilding error is the same in every run); it is the individual directions that are arbitrary. Fewer codes, more samples or more weight decay usually help.</p>

      <h2>Smoothness</h2>
      <p>The tab <Ui>Smoothness</Ui> compares <Ui>Roughness of the samples and of the rebuilt samples</Ui>: how wiggly the (smoothed) measured EEMs are, and how wiggly the rebuilt ones. The roughness is the quantity <b>smoothify</b> penalises: for the Gaussian-process basis the size of the standardised coefficients (about 1 for a typical sample), for B-splines and cosines, and for the pixel version, how much the function bends from pixel to pixel (its second differences on the grid); for the bases the numbers are scaled so that it is about 1 for a typical sample. The words say whether the rebuilt EEMs are <i>smoother than</i> (the ratio of the roughness of the rebuilt samples to that of the samples is below 0.9), <i>about as rough as</i> (0.9 to 1.1) or <i>rougher than</i> (above 1.1: the page then suggests the smoothify scan) the smoothed samples. Rougher: the network adds wiggles that are not in the smoothed data, so look at the <Ui>Smoothify scan</Ui> on the same tab and take <Ui>Use the suggested value</Ui>, which sets smoothify and goes back to step 6. Much smoother: the network may be flattening the samples (too much smoothify, or too few codes).</p>

      <h2>Symptoms and what to do</h2>
      <div class="table-wrap"><table class="table g-table"><thead><tr><th>You see</th><th>Likely reason</th><th>Try</th></tr></thead><tbody>
        <tr><td>Held-out variance explained far below the training value</td><td>Memorising.</td><td>Fewer codes, <Ui>Small</Ui>, more <Ui>Weight decay</Ui>, <Ui>Denoising</Ui>, more samples.</td></tr>
        <tr><td>About as good as PCA</td><td>Linear data, or too few samples for the bends.</td><td>Use PCA or functional PCA.</td></tr>
        <tr><td>The loss jumps about, or "The training diverged"</td><td>The learning rate is too large.</td><td>Lower <Ui>Learning rate</Ui>.</td></tr>
        <tr><td>Both loss curves high and still falling at the end</td><td>Under-trained, or too few codes.</td><td>More epochs, or one more code.</td></tr>
        <tr><td>Codes not reproducible</td><td>Too many codes, too few samples.</td><td>Fewer codes; read <Ui>Stability</Ui> again.</td></tr>
        <tr><td>Rebuilt EEMs with ripples</td><td>Rough components drawn freely.</td><td>The <Ui>Smoothify scan</Ui>.</td></tr>
        <tr><td>Rebuilt EEMs flat, peaks too low</td><td>Too much smoothify, too few terms in the basis, too few codes.</td><td>Lower smoothify; check step 3; one more code.</td></tr>
        <tr><td>VAE: all codes inactive</td><td>β too large, or the warm-up too short.</td><td>Lower β, lengthen <Ui>Warm-up (epochs)</Ui>, use <Ui>Free bits (nats per code)</Ui>.</td></tr>
        <tr><td>VAE: rebuilt EEMs blurry</td><td>The pull towards the prior is strong.</td><td>β below 1, or the plain autoencoder.</td></tr>
      </tbody></table></div>

      <h2>What to check</h2>
      <ul class="g-list">
        <li>That the <b>held-out variance explained</b> is high enough for your purpose and close to the training value.</li>
        <li>That the verdict of <Ui>Is the nonlinearity worth it?</Ui> says <b>better</b> before you choose a network over PCA.</li>
        <li>That the <b>number of codes</b> comes from the scan, not from a wish.</li>
        <li>For a VAE: <b>how many codes are active</b>; use that number.</li>
        <li>That the codes are <b>reproducible</b> before you read one of them, and that the rebuilt EEMs are about as <b>smooth</b> as the measured ones.</li>
      </ul>
    </div>`,
  },
};

export const toolsPage = {
  id: 'tools',
  icon: 'search-alt',
  title: 'The exploration tools',
  summary: 'The latent map, traversals, sensitivity images, single samples, interpolation and generation (and why generated EEMs are not data)',
  component: {
    components: comps,
    template: `
    <div>
      <p class="lead">Step <Ui>7 · Results</Ui> is where you look at what the network learned. A network is a black box only if you do not open it; these tools open it from outside, by asking it questions: where does each sample sit, what does each code do to an EEM, what lies between two samples, and what does it think a new sample looks like. Use them <b>after</b> the checks of page 5: pictures of a network that does not generalise are pictures of the wrong thing.</p>

      <h2>The map of the samples</h2>
      <p>The tab <Ui>Latent space</Ui> draws the chart <Ui>Codes of the samples</Ui>: one point per sample, placed by two of its codes (<Ui>Horizontal axis</Ui>, <Ui>Vertical axis</Ui>; with more than two codes the first view shows the two main directions of the codes, like a PCA of the codes). <Ui>Colour by</Ui> and <Ui>Symbol by</Ui> take any variable of the project: metadata, quality numbers, results of other pages. Samples that were <b>held out</b> carry a ring: they are the ones that tell you how the map treats new samples. For a variational autoencoder the switch <Ui>Show the uncertainty of each code</Ui> draws the blob of each sample: a large one means the network is unsure where that sample belongs. Hovering a point shows its EEM, its split, its status and its error; a click selects it, and <Ui>Open in Samples</Ui> goes to the sample.</p>
      <ul class="g-list">
        <li><b>Reading it.</b> Points that are close are similar samples. A smooth change of colour along a direction means that the codes follow that variable. Groups of points are groups of samples.</li>
        <li><b>Do not over-read it.</b> The axes have no fixed meaning or sign, and a nonlinear map is only reliable near the samples: do not measure distances between far-away points as if the map were flat.</li>
        <li><b>Codes against other variables.</b> The card <Ui>Codes against other variables</Ui> takes variables you add with <Ui>Add a variable</Ui>, and <Ui>Relate the codes</Ui> gives the correlation of each code with each variable and the cross-validated R² of predicting the variable <b>from the codes</b>, next to the same with the same number of PCA scores. That is the fair test of whether the network's codes hold more about this variable than the PCA scores do.</li>
      </ul>

      <h2>Traversals: what does each code do?</h2>
      <p>A code is a number with no unit; to know what it means, change it and look. The tab <Ui>Traversals</Ui> takes the <b>average code</b> (every code at the mean of the training samples), changes <b>one</b> code from −3 to +3 standard deviations of the codes of the training samples (<Ui>Range</Ui>: <Ui>±1 SD</Ui>, <Ui>±2 SD</Ui> or <Ui>±3 SD</Ui>), and decodes each value into an EEM, one row for each code. With <Ui>Show</Ui> = <Ui>Images</Ui> you see the EEMs; with <Ui>Difference to the mean</Ui> you see how each differs from the EEM of the average code, which is what the eye needs.</p>
      <FigTraversal>The traversal of one code (real numbers; made-up EEMs in which one band moves). Top: the EEMs decoded for the code at −3, −2, −1, 0, +1, +2 and +3 SD. Bottom: the difference to the middle one; blue where the EEM is stronger, orange where it is weaker. Here the code moves a band along the emission axis: the difference is a blue patch on one side and an orange one on the other, and the pair moves with the code.</FigTraversal>
      <ul class="g-list">
        <li><b>Where it changes.</b> A patch that gets stronger as the code rises is blue in the difference; one that gets weaker is orange. The sign of a code is arbitrary: if you flip it, blue and orange swap.</li>
        <li><b>A straight or a bent code.</b> If the −3 and +3 differences are mirror images, the code acts like a PCA component. If they differ in shape, the code bends: that is what the network gains over PCA.</li>
        <li><b>Stay near the data.</b> Look at ±1 and ±2 SD first. With few samples, ±3 SD is already beyond what the network has seen, and what it draws there is a guess.</li>
      </ul>
      <p>Under the rows, the card <Ui>Sensitivity of the output to each code</Ui> gives a second view. The <b>sensitivity</b> of the output to a code is what a <i>small</i> change of that code does to the EEM at the average code (mathematically its derivative); the card shows it for every code, with the <b>share</b> of each code in the total and one sentence. It is the nonlinear counterpart of a loading image. It is <b>local</b>: valid near the average code, which is why the traversal is also useful, since it shows the whole path.</p>

      <h2>One sample at a time</h2>
      <p>The tab <Ui>Samples</Ui> lists the samples (the worst rebuilt first; filters <Ui>All</Ui>, <Ui>Training</Ui>, <Ui>Held out</Ui>, <Ui>Flagged</Ui>, each with its count, and the field <Ui>Find a sample</Ui>). For the chosen sample you see its status in words, its code, and the pictures <Ui>Measured</Ui>, <Ui>Rebuilt</Ui> and <Ui>Difference</Ui> (and the absorbance curves). A good rebuilt sample has a difference that looks like noise. The card <Ui>Unusual samples</Ui> finds the ones that do not: choose the <Ui>Method</Ui>, <Ui>Median plus c × MAD</Ui> (a robust limit: the median of the logarithms of the errors plus c times their typical spread, the MAD; c is 3 unless you change it), <Ui>Quantile of the held-out errors</Ui> or, for a network that was saved, <Ui>Limit of the saved model</Ui>, and press <Ui>Find them</Ui>. You get the histogram of the errors with the limit and the list of flagged samples, each with its residual map. <b>A flag is a reason to look at a sample, not to delete it.</b></p>

      <h2>Between two samples</h2>
      <p>The tab <Ui>Interpolation</Ui> takes <Ui>Sample A</Ui> and <Ui>Sample B</Ui> and builds <Ui>Number of steps</Ui> in-between EEMs in two ways. <Ui>Through the code</Ui>: the two codes are mixed step by step and each mixture is decoded. <Ui>Plain mixture</Ui>: the two measured EEMs are mixed directly. <Ui>Difference</Ui> shows the second against the first, and the slider <Ui>Position</Ui> looks at one step large.</p>
      <ul class="g-list">
        <li>If the two strips are the same, the change from A to B is a straight line, and the network gains nothing there.</li>
        <li>If <b>through the code</b> every step looks like a real EEM, with a band that slides, while the plain mixture is two overlapping bands (one fading out, one fading in), the network learned a <b>curved</b> path.</li>
        <li>Between samples that are far apart, in a part of the map where there are no samples, what is decoded is a guess.</li>
      </ul>

      <h2>Drawing new samples (variational autoencoders)</h2>
      <p>The tab <Ui>Generate</Ui> exists only for a variational autoencoder (for the plain kind it is disabled with the reason "Only variational autoencoders can draw new samples"). Choose <Ui>Number of samples</Ui>, <Ui>Seed</Ui> and <Ui>Temperature</Ui> (0 to 2) and press <Ui>Draw new samples</Ui>: codes are drawn from the prior cloud, with its width multiplied by the temperature, and decoded. At 0 every draw is the typical sample; at 1 the variety of your samples; above 1 the draws go beyond what is typical. Each picture is tagged <b>synthetic</b>, with the nearest real sample, under the banner "Synthetic samples: made up by the network, not measured". <Ui>Download as .npz</Ui> and <Ui>Download as CSV</Ui> write files that say <i>synthetic</i> in their name and in their content.</p>
      <Callout kind="warning"><b>Synthetic samples are not data.</b> They contain nothing the network has not already seen from your samples, so they add no information. They carry the network's own errors and its blur, pulled towards the typical. Training another model on them, or computing a statistic from them, repeats the network's opinion and makes it look like evidence. And anyone who sees one without its label may take it for a measurement. Fair uses: looking at what the network finds typical; checking that the network understood the data (do the draws look like EEMs of your waters, is the nearest real sample close?); teaching.</Callout>

      <h2>What to check</h2>
      <ul class="g-list">
        <li>That the <b>held-out samples</b> (the rings) sit among the others on the map and are rebuilt about as well as the training ones.</li>
        <li>That you know what each code does: <b>look at its traversal</b> before you name it.</li>
        <li>That the <b>worst samples</b> have been looked at one by one, and that a flag was followed by a look, not a deletion.</li>
        <li>That between two samples you compared <b>through the code</b> with the <b>plain mixture</b>.</li>
        <li>That nothing that was <b>drawn</b> leaves the page without the word synthetic.</li>
      </ul>
    </div>`,
  },
};
