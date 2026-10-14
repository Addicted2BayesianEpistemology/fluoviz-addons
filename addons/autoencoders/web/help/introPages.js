// Help > Autoencoders, pages 1-2: what an autoencoder is, and the nine steps of the page "Autoencoders (add-on)".
// tests/test_help_autoencoders.py checks every word of the interface (<Ui>...</Ui>) against the page and docs/ui.md, every default and number against the add-on's code, and the pictures against a recomputation.
import { parts } from './parts.js';
import { FigAutoencoder, FigPcaVsAe } from './figures.js';

const comps = { ...parts, FigAutoencoder, FigPcaVsAe };

export const whatPage = {
  id: 'what',
  icon: 'lightbulb-alt',
  title: 'What an autoencoder is',
  summary: 'A code that describes every EEM; PCA as the straight-line case; when the bends help; the variational kind',
  component: {
    components: comps,
    template: `
    <div>
      <p class="lead">An <b>autoencoder</b> is a small computer program that learns to describe each of your EEMs with a handful of numbers, and to draw the EEM back from those numbers. That is the whole job. It is useful because the handful of numbers is a compact, comparable description of every sample, and because the way the program fails to rebuild a sample tells you something about it. The page is <b>Autoencoders (add-on)</b>, in the sidebar group <b>Experimental methods</b>. You do not need to know anything about neural networks: every word is explained where it first appears, and again in the glossary on the last page.</p>
      <Callout kind="warning"><b>Experimental, and an add-on.</b> The mathematics is standard in machine learning, but its use on EEMs is new, and <b>a network is far more flexible than a few dozen EEMs can support</b>. That flexibility is a risk before it is a gift. The page therefore always tests the network on samples it did not learn from, and always compares it with PCA (or with functional PCA). <b>Read those two numbers before anything else.</b> The add-on is not installed with FluoViz: you press <i>Install</i> in <PLink path="/settings" :query="{ section: 'addons' }">Settings, Add-ons</PLink>, and it downloads PyTorch (the machine-learning library) into FluoViz's own folder.</Callout>

      <h2>The idea: describe, then rebuild</h2>
      <p>A measured EEM is thousands of numbers, one per pixel. Two EEMs from the same river differ in only a few ways: a little more humic-like fluorescence in one, a little less protein-like in the other. An autoencoder looks for a short list of numbers that holds those differences. The short list is the <b>code</b>, also called the <b>latent code</b> (<i>latent</i> only means hidden: nobody measured it, the program found it). You choose its length with <Ui>Number of codes (latent dimension)</Ui>; two to eight is usual.</p>
      <p>The network has two halves. The <b>encoder</b> reads an EEM and produces its code. The <b>decoder</b> reads a code and draws an EEM. The program is <b>trained</b> to make the drawn EEM as close as possible to the one that went in. Because the code is much shorter than the EEM, the network cannot simply copy: to rebuild the sample from a few numbers it has to learn what matters in your samples.</p>
      <FigAutoencoder>One EEM through an autoencoder with real numbers (the data set of this picture is made up for the guide: three fluorophores and a removed scatter band). On the left the measured EEM; in the middle the code, the few numbers that describe it; on the right the EEM the decoder draws from the code. Grey: pixels that do not exist.</FigAutoencoder>

      <h2>What is inside the network</h2>
      <p>A <b>neural network</b> is a chain of small calculation steps called <b>layers</b>. A layer takes a list of numbers and makes a new list, each new number being a <i>weighted sum</i> of the old ones. The <b>weights</b> are the numbers the network learns. Each new number is called a <b>neuron</b>. After the weighted sum every neuron bends its result with a fixed curve, the <b>activation</b>, and the list goes on to the next layer. The default network is small: the encoder has a layer of 64 neurons and one of 32, then comes the code, then the decoder mirrors the encoder (32, then 64) and ends with one number for every input.</p>
      <p>Without the bends, a chain of weighted sums would still be one weighted sum, a straight-line description. The bends are what allow a curved one. The smooth ones (<i>silu</i> and <i>tanh</i>) are the default because they give smooth results.</p>
      <p>The network of this add-on has one more piece, a <b>linear shortcut</b>: a plain weighted sum from the EEM to the code, and another from the code back to the EEM, beside the layers with bends. The shortcut goes through the code, so the code stays short. Training <b>starts from PCA</b>: the shortcut is set to the PCA of your training samples, so the network begins as PCA and learns only what the bends add on top of it, and the starting weights are one of the candidates when the best weights are chosen with the held-out samples, so the network you get is never worse than PCA on those samples. The switch is <Ui>Start from PCA (linear shortcut)</Ui>, in the fold <Ui>Advanced network settings</Ui> of step 5, and it is on by default; switched off you get a classical autoencoder that starts from random weights.</p>
      <p><b>Training</b> is the search for good weights. They start as random numbers. The program looks at the samples, measures how far each rebuilt EEM is from the real one (the <b>loss</b>: a number that is small when the rebuilding is good), and changes every weight a little in the direction that lowers the loss. One pass through all the samples it learns from is an <b>epoch</b>; training takes tens to a few hundred epochs, from seconds to a few minutes here. The next pages say what a healthy training looks like.</p>

      <h2>Compared with PCA</h2>
      <p>PCA is an autoencoder without the bends: the code is a weighted sum of the pixels and the rebuilding is another weighted sum. This is not a manner of speaking. A network with the activation <Ui>linear</Ui> learns the same space as PCA (Baldi &amp; Hornik 1989); its codes may be mixtures of the PCA scores in another order, but the EEMs it rebuilds are the same, and with the shortcut on it is the PCA itself. The add-on uses this as a check: the network with <Ui>linear</Ui> should land on PCA, and the page compares your network with PCA (or with functional PCA, when it works on a basis) <b>with the same number of codes</b>.</p>
      <div class="table-wrap"><table class="table g-table"><thead><tr><th></th><th>PCA</th><th>Autoencoder</th></tr></thead><tbody>
        <tr><td><b>Description</b></td><td>Weighted sums only: straight lines.</td><td>Weighted sums with bends: it can follow curved relationships between samples.</td></tr>
        <tr><td><b>Answer</b></td><td>One exact answer, the same every time.</td><td>Depends on the random start and the settings. The add-on trains from three random starts and keeps the best, and reports how much they differ.</td></tr>
        <tr><td><b>Components</b></td><td>Ordered by importance and perpendicular; each has a loading image.</td><td>The codes share the information, in no order. A code has no fixed sign and no single meaning; the page shows what each one does (traversals and sensitivity images, page 6).</td></tr>
        <tr><td><b>Samples needed</b></td><td>A dozen is enough to try.</td><td>More, and a part of them must be held back for testing (page 4).</td></tr>
        <tr><td><b>Time</b></td><td>Instant.</td><td>Seconds to minutes.</td></tr>
      </tbody></table></div>

      <h2>When the bends help, and when they do not</h2>
      <p>The picture below is the whole argument. On the left, EEMs that are sums of three fluorophores whose amounts vary: the case PCA and PARAFAC are made for. PCA, the network with <Ui>linear</Ui> and the network with bends all need the same number of codes. On the right, one fluorophore whose emission band <i>moves</i> as a hidden condition changes, and whose amount varies: two hidden factors. A moving band is not a sum of fixed shapes, so PCA needs several more components than there are factors, while a network with bends needs only as many codes as there are factors.</p>
      <FigPcaVsAe>The share of the variation that is <b>not</b> rebuilt, on samples the network did not learn from, against the number of codes (lower is better; real numbers from the add-on's own code, two made-up data sets of 120 EEMs). Left: the network and PCA are equally good. Right: the network with bends is much better with the same number of codes.</FigPcaVsAe>
      <ul class="g-list">
        <li><b>The bends help</b> when a band changes position or width with the conditions (pH, ionic strength, temperature, the source of the water), when the fluorescence is not proportional to the amount (quenching, an inner-filter effect that was not fully corrected), when the samples come from very different waters, and when there are many samples.</li>
        <li><b>The bends do not help</b> when every fluorophore simply adds up, and when there are few samples. Then the page says so: the card <Ui>Against PCA</Ui> shows a difference of about zero (or below), the verdict says that the network is about as good or worse, and <b>the simpler method wins</b>: PCA is exact, instant and the same every time.</li>
      </ul>

      <h2>The variational autoencoder: codes that have a shape</h2>
      <p>A plain autoencoder may put its codes anywhere. Between two samples there can be empty regions where the decoder has never been asked to draw anything, and nothing tells you whether a new code is ordinary or strange. The <b>variational autoencoder</b> (VAE; Kingma &amp; Welling 2014) adds a second aim: the codes of all samples should together look like a simple, bell-shaped cloud (a normal distribution; the <b>prior</b>). To allow that, its encoder gives not one point but a small blob for every sample: a centre and a spread. The extra term of the loss is the <b>KL divergence</b>: a number that measures how far the blobs are from the prior cloud, and that is zero when they agree. Its weight is the <b>β</b> (beta) setting.</p>
      <ul class="g-list">
        <li><b>New samples can be drawn.</b> Pick a code from the cloud and decode it: you get an EEM that looks like your samples but was never measured. The page labels such EEMs <i>synthetic</i> and says why they are not data (page 6).</li>
        <li><b>Unusual samples are noticed.</b> A sample whose code falls far from the cloud, or that the decoder rebuilds badly, is flagged.</li>
        <li><b>The codes are more regular.</b> Nearby codes give similar EEMs, so moving from one sample to another through the code gives sensible in-between EEMs.</li>
        <li><b>The price.</b> The rebuilt EEMs are a little blurrier than with the plain kind, and codes the data do not need can switch themselves off, which is called <b>collapse</b> (page 5).</li>
      </ul>
      <p>Choose <Ui>Autoencoder</Ui> to describe and compare samples. Choose <Ui>Variational autoencoder</Ui> when you want to see what the network finds typical, to draw samples, or to judge how unusual a sample is. With a few dozen samples the second aim is not reliable: the cloud is only as good as the samples that shaped it.</p>

      <h2>The functional versions, in one paragraph</h2>
      <p>The network can work on the observed pixels, or on the coefficients of a smooth basis, exactly as the functional PCA does. In the second case the EEM is first written as a smooth function (noise smoothed away, removed scatter bands left out of the domain), and the network only sees the few coefficients. A setting called <b>smoothify</b> then pushes the rebuilt EEMs further towards smooth functions. Page 3 explains both.</p>

      <h2>When it is the right tool, and when another is better</h2>
      <ul class="g-list">
        <li>You want a <b>compact description</b> of each EEM that follows curved changes, to plot, cluster or classify.</li>
        <li>You want to <b>find samples the rest of the set does not explain</b>, with the residual map of each.</li>
        <li>You want to see <b>how a hidden condition changes the EEM</b>, with the traversals (page 6).</li>
        <li><b>You need the spectra of the fluorophores:</b> use <b>PARAFAC</b> (or NMF). A code of an autoencoder is not the amount of a substance, and the images of the sensitivity are not spectra of substances.</li>
        <li><b>You need a standard method for a paper:</b> PCA is known to every reviewer. Use the autoencoder when you can say what it adds, and report PCA beside it.</li>
        <li><b>You have fewer than a few dozen samples:</b> try functional PCA first. The page refuses to train on fewer than 12 samples unless you insist.</li>
      </ul>

      <h2>What to check</h2>
      <ul class="g-list">
        <li>The <b>variance explained on held-out samples</b> and the card <Ui>Against PCA</Ui> (or <Ui>Against functional PCA</Ui>) come first. If the network is not better than PCA with the same number of codes, use PCA.</li>
        <li>That the <b>code is short</b>: a few numbers, far fewer than the number of samples divided by ten.</li>
        <li>That you do not read a single code as the amount of a substance: <b>the codes have no fixed meaning</b> until you have looked at what each one does.</li>
        <li>That anything the network <b>draws</b> (VAE) is called synthetic and never used as data.</li>
      </ul>
    </div>`,
  },
};

export const pipelinePage = {
  id: 'pipeline',
  icon: 'process',
  title: 'The pipeline, step by step',
  summary: 'The nine steps of the page: what each does, what to decide, what to look at, the usual mistakes',
  component: {
    components: comps,
    template: `
    <div>
      <p class="lead">The page <b>Autoencoders (add-on)</b> is a list of nine steps. The first four are the steps of the functional PCA (what to model, the domain, the representation, the smoothing); the last five are the network's own (the network, the training, the results, the checks, the use). Go through them in order, or press <Ui>Run with the recommended settings</Ui> and come back to look at what it did. Whatever you set is kept when you leave the page. Each step has a box, <Ui>What this step does and what to check</Ui>, with the short version of this page, and the button <Ui>How to read this</Ui> in the head of the page opens this guide.</p>

      <h2>The quickest way: the recommended settings</h2>
      <p><Ui>Run with the recommended settings</Ui> (at the bottom of the first step) keeps what you chose in step 1 (variables, pre-treatment, weights), asks FluoViz for advice about your data, <b>replaces all the other settings by the ones below</b>, trains, and opens the results.</p>
      <div class="table-wrap"><table class="table g-table"><thead><tr><th>Setting</th><th>Recommended</th><th>Why</th></tr></thead><tbody>
        <tr><td><Ui>Functional version</Ui></td><td><Ui>On the coefficients of a basis (functional)</Ui></td><td>Smoother, noise-free and far fewer inputs than pixels (page 3).</td></tr>
        <tr><td>Representation</td><td><Ui>Gaussian process (Karhunen–Loève)</Ui>, Matérn 5/2 kernel, smoothing by maximum likelihood</td><td>The functional PCA's own recommendation, built on the pixels that exist.</td></tr>
        <tr><td><Ui>Type of network</Ui></td><td><Ui>Autoencoder</Ui></td><td>The simplest; compare with the VAE afterwards.</td></tr>
        <tr><td><Ui>Number of codes (latent dimension)</Ui></td><td>The smallest of: the number of principal components that hold 99 % of the variation (a PCA of your samples), one tenth of the number of samples, and 8; never fewer than 2</td><td>Never more codes than the data support; at least ten samples per code.</td></tr>
        <tr><td><Ui>Size of the network</Ui></td><td><Ui>Small</Ui> (layers of 64 and 32 neurons); <Ui>Medium</Ui> only above 300 samples</td><td>Tens to a few hundred samples do not need more.</td></tr>
        <tr><td><Ui>Activation</Ui></td><td>silu</td><td>Smooth, so the rebuilt functions are smooth.</td></tr>
        <tr><td><Ui>Smoothify</Ui></td><td>0 (off)</td><td>The functional version is already smooth; turn it on after you have looked at the checks.</td></tr>
        <tr><td>Training</td><td>Starting from PCA (<Ui>Start from PCA (linear shortcut)</Ui> on), up to 600 epochs, patience 60, 20 % of the samples held out (5-fold validation under 30 samples), 3 random starts, seed 0, weight decay 0.001, learning rate 0.001</td><td>Chosen for tens to a few hundred samples (page 4).</td></tr>
      </tbody></table></div>
      <Callout kind="info">These are a <b>starting point</b>, not an answer. Look at the held-out numbers, the comparison with PCA and the loss curves (steps 7 and 8) before you rely on the codes.</Callout>

      <h2>A first run in five steps</h2>
      <ol class="g-steps">
        <li>Open the page and press <Ui>Run with the recommended settings</Ui> in step 1. The live chart <Ui>Training so far</Ui> shows the loss falling.</li>
        <li>When the page moves to <Ui>7 · Results</Ui>, read the four cards: <Ui>Variance explained on held-out samples</Ui>, <Ui>Codes</Ui>, <Ui>Against PCA</Ui>, <Ui>Flagged samples</Ui>.</li>
        <li>Open <Ui>8 · Checks</Ui>, tab <Ui>Training</Ui>: the verdict says whether the network over-fitted.</li>
        <li>Tab <Ui>Quality</Ui>: the card <Ui>Is the nonlinearity worth it?</Ui> says whether the network beats PCA.</li>
        <li>Only then look at the codes (<Ui>Latent space</Ui>, <Ui>Traversals</Ui>) and use them (<Ui>9 · Use</Ui>).</li>
      </ol>

      <h2>1 · What to model</h2>
      <p>You choose the <b>variables</b>: the <Ui>Fluorescence</Ui> EEMs, the <Ui>Absorbance</Ui> spectra, or <Ui>Both together (multivariate)</Ui>, and how each is prepared before the network (<Ui>Pre-treatment</Ui>). With two variables there are no weights to choose: each variable counts equally (the card <Ui>Both together (multivariate)</Ui> says so), so the result does not depend on the units. This step is the one of the functional PCA, with the same words.</p>
      <ul class="g-list">
        <li><b>Decide.</b> Fluorescence alone is the usual case. Both together needs an absorbance spectrum for every sample; samples without one are left out and listed under <Ui>Samples that are left out</Ui>.</li>
        <li><b>Look at.</b> The number of samples on each card.</li>
        <li><b>Usual mistakes.</b> Expecting the page to correct the data: everything you set on the other pages applies (the Corrections, the scatter removal, the focus window, the excluded samples), so correct first.</li>
      </ul>

      <h2>2 · Domain</h2>
      <p>The map <Ui>Observed pixels</Ui> shows where the EEMs exist (blue: used; grey: not used), and <Ui>Minimum coverage of a pixel</Ui> decides which pixels stay in the domain: a pixel observed in fewer samples than that is dropped for everybody. Nothing is filled in. The step has the same links as the functional PCA to the scatter removal and the focus window.</p>
      <ul class="g-list">
        <li><b>Decide.</b> Whether this is the region you want the network to learn.</li>
        <li><b>Look at.</b> The map. If scatter remains in the blue area, the network will learn the scatter.</li>
        <li><b>Usual mistakes.</b> Leaving scatter in; a minimum coverage so high that one noisy sample empties a whole band.</li>
      </ul>

      <h2>3 · Representation</h2>
      <p>First the choice that is new, <Ui>Functional version</Ui>: <Ui>On the coefficients of a basis (functional)</Ui> (the default) or <Ui>On the pixels</Ui>. In the first, the cards <Ui>Gaussian process (Karhunen–Loève)</Ui> (recommended), <Ui>B-splines</Ui> and <Ui>Cosines</Ui> and the explorer <Ui>How many terms do I need?</Ui> are those of the functional PCA; the card <Ui>Pixels (no basis)</Ui> is not offered, because <Ui>On the pixels</Ui> is that choice. With <Ui>On the pixels</Ui> the cards say that there is nothing to choose here and in step 4.</p>
      <ul class="g-list">
        <li><b>Decide.</b> The functional version with the Gaussian-process basis, unless you want the pixels as a reference. Page 3 compares the two; the guide <GLink tab="experimental" to="gp">Experimental methods, the Gaussian-process basis</GLink> explains the basis.</li>
        <li><b>Look at.</b> The number of terms against what they hold.</li>
        <li><b>Usual mistakes.</b> Too few terms (it cuts the peaks), a length-scale shorter than two grid steps.</li>
      </ul>

      <h2>4 · Smoothing</h2>
      <p>Every EEM is the true smooth function plus noise; this step estimates the function from the noisy pixels, with the <Ui>Method</Ui> chosen from the data, and <Ui>Check the smoothing</Ui> shows measured, smoothed and residual for one sample. It is the step of the functional PCA; read <GLink tab="experimental" to="other">how to read the smoothing check</GLink> there. It is not <b>smoothify</b> (step 5), which acts on what the network draws, not on the data it sees.</p>
      <ul class="g-list">
        <li><b>Decide.</b> Almost always: keep the proposed method.</li>
        <li><b>Look at.</b> The residual: it should look like noise.</li>
        <li><b>Usual mistakes.</b> Not looking. Accepting a smooth that cuts a peak.</li>
      </ul>

      <h2>5 · Network</h2>
      <p>Everything about the network, in this order. <Ui>Type of network</Ui>: <Ui>Autoencoder</Ui> or <Ui>Variational autoencoder</Ui>. <Ui>Number of codes (latent dimension)</Ui>: 1 to 32, with the advice sentence under it, and the <Ui>Latent-dimension scan</Ui> that trains small networks for 1, 2, 3 … codes and draws <Ui>Error on held-out samples against the number of codes</Ui> next to PCA, with the button <Ui>Use N codes</Ui>. <Ui>Size of the network</Ui>: <Ui>Small</Ui>, <Ui>Medium</Ui>, <Ui>Large</Ui> or <Ui>Custom</Ui> (<Ui>Layer widths</Ui> such as 64, 32). <Ui>Activation</Ui>. <Ui>Regularisation</Ui>, the settings that keep the network from memorising the samples: <Ui>Dropout</Ui>, <Ui>Weight decay</Ui>, and the <Ui>Denoising</Ui> pair <Ui>Noise added to the input (in standard deviations)</Ui> and <Ui>Share of inputs hidden</Ui>. The fold of the variational autoencoder (page 5 and the glossary): <Ui>Weight of the typical-code term (β)</Ui>, <Ui>Warm-up (epochs)</Ui>, <Ui>Free bits (nats per code)</Ui>, <Ui>Noise level of the data (σ)</Ui> and <Ui>Prior</Ui>. And <Ui>Smoothify</Ui> with <Ui>Smoothness is measured by</Ui> and the <Ui>Smoothify scan</Ui> (page 3). The fold <Ui>Advanced network settings</Ui> has the switch <Ui>Start from PCA (linear shortcut)</Ui> (page 1). On the right the card <Ui>The network in numbers</Ui> shows <Ui>Inputs</Ui>, <Ui>Codes</Ui>, <Ui>Parameters</Ui>, <Ui>Samples per parameter</Ui>, <Ui>Samples per code</Ui>, <Ui>Start</Ui>, <Ui>Validation</Ui> and <Ui>Time</Ui>, and the warnings.</p>
      <ul class="g-list">
        <li><b>Decide.</b> The number of codes first; leave the rest as proposed for the first run. Change one thing at a time.</li>
        <li><b>Look at.</b> The card <Ui>The network in numbers</Ui>: <Ui>Samples per code</Ui> should be at least 10; the yellow notices say what is wrong.</li>
        <li><b>Usual mistakes.</b> Many codes and a large network because more seems better (it memorises); changing five settings between two runs and not knowing which mattered.</li>
      </ul>

      <h2>6 · Train</h2>
      <p>The card <Ui>What will be done</Ui> sums up your choices in words and gives the time estimate. The fold <Ui>Training settings</Ui> has <Ui>Maximum number of epochs</Ui> (600), <Ui>Patience (epochs without improvement)</Ui> (60), <Ui>Share of samples held out</Ui> (0.2), <Ui>Folds (fewer than 30 samples)</Ui> (5), <Ui>Learning rate</Ui> (0.001), <Ui>Batch size</Ui> (empty: automatic, a sixth of the training samples, but at least 8 and at most 32), <Ui>Number of random starts</Ui> (3), <Ui>Random seed</Ui> (0) and <Ui>Title of the result</Ui>. <Ui>Train the network</Ui> starts the training in a separate process, so FluoViz stays responsive; <Ui>Cancel</Ui> stops it and <b>nothing is saved</b> (the page says <Ui>Cancelled: nothing was saved.</Ui>). The chart <Ui>Training so far</Ui> shows the loss on the training samples and on the held-out samples, twice a second; the line under it says where it is, for example "Start 2 of 3, epoch 34".</p>
      <ul class="g-list">
        <li><b>Decide.</b> Almost nothing: the defaults are chosen for tens to a few hundred samples. Page 4 explains each.</li>
        <li><b>Look at.</b> The two lines of the chart: they should both fall, and the held-out one should flatten.</li>
        <li><b>Usual mistakes.</b> Pressing <Ui>Cancel</Ui> because it "seems slow" in the first seconds; a refusal in red (too few samples, PyTorch missing) taken for a bug: it is the reason, in words.</li>
      </ul>

      <h2>7 · Results</h2>
      <p>Four cards and five tabs: <Ui>Latent space</Ui>, <Ui>Traversals</Ui>, <Ui>Samples</Ui>, <Ui>Interpolation</Ui> and <Ui>Generate</Ui> (variational autoencoders only). The result is saved with a number, like the other results (<i>autoencoder-1</i>), and is flagged outdated when the data change: the notice "Outdated: … changed since this autoencoder was computed." comes with the button <Ui>Train again with the same settings</Ui>. <b>The network is never trained again silently.</b> Page 6 describes the tabs.</p>
      <ul class="g-list">
        <li><b>Decide.</b> Which samples to look at; which codes to use.</li>
        <li><b>Usual mistakes.</b> Reading the codes before the checks; using the codes of an outdated result.</li>
      </ul>

      <h2>8 · Checks</h2>
      <p>Five tabs, each with a verdict in words on top: <Ui>Training</Ui>, <Ui>Quality</Ui>, <Ui>Latent dimensions</Ui>, <Ui>Stability</Ui> and <Ui>Smoothness</Ui>. Page 5 explains how to read them.</p>
      <ul class="g-list">
        <li><b>Decide.</b> Whether the network is good enough to use. If the checks disagree with each other, go back to step 5.</li>
        <li><b>Usual mistakes.</b> Skipping the step because the results look nice.</li>
      </ul>

      <h2>9 · Use</h2>
      <p>The variables the network made (<Ui>Variables made by this network</Ui>), the buttons <Ui>Cluster the latent space</Ui>, <Ui>Classify with the latent space</Ui> and <Ui>Plot the codes</Ui> under <Ui>Use the codes</Ui>, <Ui>Save this model</Ui> and <Ui>Apply a saved model</Ui> under <Ui>Keep the network</Ui>, the <Ui>Downloads</Ui> and the <Ui>Methods paragraph</Ui> with <Ui>Copy</Ui>. Page 7 describes them.</p>

      <h2>What to check</h2>
      <ul class="g-list">
        <li>That step 2 shows <b>the region you mean</b> and nothing else.</li>
        <li>That the card <Ui>The network in numbers</Ui> shows <b>at least 10 samples per code</b> and no warning you do not understand.</li>
        <li>That you <b>changed one thing at a time</b> between runs, and kept the numbers of each.</li>
        <li>That you looked at <b>the held-out numbers and the comparison with PCA</b> before the pictures.</li>
      </ul>
    </div>`,
  },
};
