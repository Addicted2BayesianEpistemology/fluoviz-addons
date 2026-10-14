// Help > Autoencoders, pages 7-8: using the codes everywhere else (Plots, Clustering & LDA, saved models, chains), then the limits, the good practice, the glossary and the references.
// tests/test_help_autoencoders.py checks every word of the interface (<Ui>...</Ui>) against the page and docs/ui.md, and the names of the variables against the API.
import { parts } from './parts.js';

const comps = { ...parts };

export const usePage = {
  id: 'use',
  icon: 'share-alt',
  title: 'Using the codes elsewhere',
  summary: 'Variables in Plots, clustering and classification of the codes, the anomaly status, saved models, chains, downloads, the methods text',
  component: {
    components: comps,
    template: `
    <div>
      <p class="lead">A trained network is not an end in itself: its codes are numbers you can plot, cluster and classify like any others, its rebuilding error is a ready-made check of whether a sample is ordinary, and the network itself can be saved and used on samples you measure later. Step <Ui>9 · Use</Ui> collects all of this.</p>

      <h2>The codes are variables</h2>
      <p>As soon as a network is trained, FluoViz publishes numbers for every sample under the title of the result (<i>autoencoder-1</i> and so on). They are listed in step 9 under <Ui>Variables made by this network</Ui>, and they appear in the <PLink path="/compare">Plots</PLink> page, in the links between metadata and results, and in <b>every selector of a variable</b> (colour by, axis, group by).</p>
      <div class="table-wrap"><table class="table g-table"><thead><tr><th>Variable</th><th>What it is</th></tr></thead><tbody>
        <tr><td><span class="mono">z1</span> … <span class="mono">zd</span></td><td>The codes. For an autoencoder the code itself; for a variational autoencoder the <b>centre</b> of the blob of each sample (the posterior mean).</td></tr>
        <tr><td><span class="mono">recon_rmse</span>, <span class="mono">recon_r2</span></td><td>How well the sample is rebuilt: the typical size of the difference in the units of the data, and the share of its variation that the network rebuilds.</td></tr>
        <tr><td><span class="mono">flag</span></td><td>The anomaly status, in words: inside the limits, reconstruction error above the limit, unusual code, or both.</td></tr>
        <tr><td><span class="mono">split</span></td><td>Whether the sample was used for training or held out. Use it to colour plots: the held-out samples are the honest ones.</td></tr>
        <tr><td><span class="mono">z1_sd</span> … (VAE)</td><td>The uncertainty of each code (the spread of the blob), for up to 12 codes.</td></tr>
        <tr><td><span class="mono">kl</span>, <span class="mono">elbo</span> (VAE)</td><td>How unusual the code of the sample is (the KL divergence), and a number that is higher when the network explains the sample better (the evidence lower bound).</td></tr>
        <tr><td><span class="mono">cluster</span>, <span class="mono">membership</span> (mixture prior)</td><td>With <Ui>Mixture of clouds</Ui> as the <Ui>Prior</Ui> the variational autoencoder groups the samples itself: the group of each sample and how firmly it belongs.</td></tr>
      </tbody></table></div>
      <p>The button <Ui>Plot the codes</Ui> opens the Plots page with the codes at hand. Colour a plot of two codes by a variable of your project, and by <span class="mono">split</span>, and you have the map of the previous page with all the tools of Plots.</p>

      <h2>Clustering and classifying the latent space</h2>
      <p>The codes are a short, tidy description of every EEM, which is what clustering and classification like. Two buttons open the page <b>Clustering &amp; LDA</b> with the codes already chosen: <Ui>Cluster the latent space</Ui>, and <Ui>Classify with the latent space</Ui>, which opens the tab <Ui>Separation (LDA)</Ui> to see how well the classes of your own metadata are separated.</p>
      <ul class="g-list">
        <li><b>Scale.</b> The codes of a plain autoencoder have no unit and an arbitrary scale, and a clustering that measures distances will favour the code with the largest spread. Switch on <Ui>Standardise the columns (z-scores)</Ui>. The codes of a variational autoencoder are already on the scale of a unit cloud, but standardising does no harm.</li>
        <li><b>The network never saw your classes.</b> It learned from the EEMs only, so a classifier that separates your sites on the codes has found something the EEMs contain, not something the network was told. The classifier's own accuracy is estimated by that page, with its cross-validation.</li>
        <li><b>The training samples are the network's own.</b> The network was fitted to them, so their codes are a little too tidy; the held-out samples are not. For an honest judgement of a classifier on the codes, look at how it does on the samples marked <Ui>Held out</Ui> in <span class="mono">split</span>, or, better, apply the saved model to new samples (below).</li>
        <li><b>Few samples.</b> A classifier on four codes of 30 samples is a toy; read its cross-validated accuracy, not its fit.</li>
      </ul>

      <h2>The anomaly status</h2>
      <p>The variable <span class="mono">flag</span> marks the samples the network does not describe well: the rebuilding error is above a <b>limit</b> (by default one that is estimated from the errors of the held-out samples, on a logarithmic scale, so that about 1 ordinary sample in 100 lies above it), and, for a variational autoencoder, also those whose code is unusual under the prior cloud. The card <Ui>Flagged samples</Ui> in step 7 counts them and the card <Ui>Unusual samples</Ui> (page 6) lists them with their residual maps. <b>A flag is a reason to look.</b> It can be a contaminated sample, a new kind of water, a measurement problem, or an ordinary sample at the edge: about one in a hundred ordinary samples is flagged at the 99 % limit, by construction.</p>

      <h2>Keeping the network and using it on new samples</h2>
      <p>The result bar has the button <Ui>Save this model</Ui> (only for an up-to-date network; an outdated one says "Recompute the model first"). A saved model is the network's numbers together with the exact recipe that turns a sample into the network's input (the corrections, the scatter treatment, the focus window and the grid; for the functional versions the same basis and the same smoothing, <b>never estimated again</b>), the limits, and a reference to the samples it was trained on. It is kept by FluoViz, like the saved models of the other pages, and can be exported as a file for a colleague. Applying it does not train anything and the network never changes.</p>
      <ol class="g-steps">
        <li>Press <Ui>Save this model</Ui> in the result bar of an up-to-date network, give it a name and a note.</li>
        <li>Later, in this or another project with the new samples, open the page <b>Saved models</b> (group <b>New samples</b>), or follow <Ui>Apply a saved model</Ui> in step 9.</li>
        <li>Choose the model and the samples. FluoViz prepares each sample as the training samples were prepared and checks what the model needs from the project (a blank, an absorbance spectrum) before it applies anything.</li>
        <li>Read the result: the codes, the rebuilding error, and the <b>status</b> of every sample: <i>Fits the model</i>, <i>Poorly described by the model</i> or <i>Cannot be used</i>. The thresholds are options at the time of the application: <b>Limit for the reconstruction error</b> (the 95 % or the 99 % point of the errors of samples the network did not learn from; 99 % by default) and <b>Smallest typicality of the code</b> (a sample whose code is less typical than this is flagged: 0.01 by default for a variational autoencoder; 0 for a plain one, which switches the check off, and you can raise it).</li>
        <li>Put some training samples among the new ones as controls: they should come back with their old codes and the status <i>Fits the model</i> (<GLink tab="saved" to="limits">Saved models, limits and good practice</GLink>).</li>
      </ol>
      <p>The guide <GLink tab="saved" to="what">Saved models</GLink> explains the page that applies a model, the statuses and what the model needs. A model of this add-on can only be opened while the add-on is installed: without it FluoViz says "This model needs the add-on '…' (Settings > Add-ons)", with the name of the add-on, instead of failing.</p>

      <h2>Chains: a classifier on top of the network</h2>
      <p>If you cluster or classify the codes (<span class="mono">res:autoencoder-1:z1</span> and so on) and save <b>that</b> model, FluoViz saves the network with it. A new EEM then goes through the network first, and its codes through the classifier. This is the <b>chain</b> of <GLink tab="saved" to="classes">Saved models, classes and clusters</GLink>. What the network says counts: if it cannot use a new sample, the sample <i>Cannot be used</i>; if it describes it poorly, the sample is <i>Poorly described by the model</i>: the classifier still gives a class, but the codes it was given are less reliable.</p>

      <h2>Downloads and the methods paragraph</h2>
      <ul class="g-list">
        <li><Ui>Codes (CSV)</Ui>: one row per sample with its split, the codes, the rebuilding error and the flag.</li>
        <li><Ui>Rebuilt samples (.npz)</Ui>: the rebuilt EEMs and curves, the average one, the codes and the axes, with the observed values and the residuals when the data are still those of the training.</li>
        <li><Ui>Training history (CSV)</Ui>: the loss of every epoch.</li>
        <li><Ui>Synthetic samples (.npz)</Ui> (variational autoencoders): the drawn samples, which say <i>synthetic</i> in the file.</li>
        <li>The <Ui>Methods paragraph</Ui> with <Ui>Copy</Ui>: a paragraph for the methods section of a paper, written from what was really done (data, representation, smoothing, network, loss, training, validation, the versions and the seed), with the references.</li>
      </ul>
      <p>Every chart of the page also has its own buttons for a table, a CSV, a PNG, and <i>.rds</i> and <i>.fig</i> files to restyle it in R or MATLAB.</p>

      <h2>What to check</h2>
      <ul class="g-list">
        <li>That the codes you plot or cluster come from a network that <b>passed the checks</b> and is <b>up to date</b>.</li>
        <li>That a clustering on the codes is <b>standardised</b>, and that you remember the codes have no fixed meaning.</li>
        <li>That a saved model is applied to samples that are <b>of the same kind</b> as the training samples, with <b>controls</b>.</li>
        <li>That the methods paragraph is in your paper, with the <b>seed and the versions</b>.</li>
      </ul>
    </div>`,
  },
};

const GLOSSARY = [
  ['Activation', 'The bend a neuron puts into its output before passing it on. Without bends a network is only weighted sums, like PCA. silu and tanh are smooth, so the rebuilt functions are smooth; linear means no bend.'],
  ['Autoencoder (AE)', 'A network that rebuilds its input from a short code. The encoder makes the code, the decoder rebuilds the sample from it. It is trained on the samples alone, without classes.'],
  ['Basis, coefficient', 'A basis is a set of smooth building-block functions; the coefficients are the weights that say how much of each block a sample has. The functional version of the network works on the coefficients.'],
  ['Batch size', 'How many samples the network looks at before it adjusts its weights once.'],
  ['β (beta)', 'The weight of the KL term of a variational autoencoder. Below 1: sharper rebuilt samples; above 1: more regular codes.'],
  ['Bootstrap', 'Making a new set of samples by drawing samples from yours at random, with replacement. Used to ask whether a result depends on which samples you happened to measure.'],
  ['Code (latent code)', 'The short list of numbers that describes a sample; its size is the number of codes. Latent means hidden: it was learned, not measured. A code has no fixed sign, order or meaning.'],
  ['Collapse (inactive code)', 'In a variational autoencoder, a code that carries no information: its KL is near zero and the decoder ignores it. A sign that fewer codes are enough.'],
  ['Decoder, encoder', 'The two halves of the network. The encoder turns a sample into its code; the decoder turns a code into a rebuilt sample.'],
  ['Denoising', 'Training the network to rebuild the clean sample from a damaged copy (noise added, some inputs hidden). Makes the codes robust and helps with missing values.'],
  ['Dropout', 'Switching off a share of the neurons at random during training, so that no neuron can be relied on alone. Not used when the network is applied.'],
  ['Early stopping, patience', 'Stopping the training when the held-out loss has not improved for a number of epochs (the patience) and keeping the weights of the best epoch.'],
  ['ELBO', 'The evidence lower bound: the quantity a variational autoencoder maximises, the rebuilding quality minus the KL term. Higher means the network explains the sample better.'],
  ['Epoch', 'One pass of the network over all the training samples. Training takes hundreds of epochs.'],
  ['Free bits', 'A small amount of KL per code that costs nothing, so that no code collapses at the start. In nats, the unit of KL.'],
  ['Held-out (validation) samples', 'Samples that are never used to change a weight and are only used to measure how well the network rebuilds samples it has not seen. Every quality number on the page is a held-out number.'],
  ['KL divergence', 'A number that measures how far the blobs of the samples are from the prior cloud: zero when they agree. In a variational autoencoder it is the price of codes that have a shape.'],
  ['L-curve', 'A chart of the rebuilding error against the roughness for several values of smoothify. The corner is the best compromise.'],
  ['Layer, neuron', 'A layer is one step of the network: it turns a list of numbers into a new list. Each new number is a neuron: a weighted sum of the old list, then bent by the activation.'],
  ['Learning rate', 'The size of each adjustment of the weights. Too large and the loss jumps about; too small and training is slow.'],
  ['Linear shortcut (start from PCA)', 'A plain weighted sum from the EEM to the code and from the code back to the EEM, beside the layers with bends. With the switch Start from PCA it is set to the PCA of the training samples, so the network begins as PCA and learns only what the bends add. It goes through the code, so the code stays short.'],
  ['Loss', 'The number the network tries to make small: the rebuilding error (plus the KL term of a variational autoencoder, plus smoothify times the roughness). It is shown for the training and the held-out samples.'],
  ['Mixture prior', 'A prior made of several clouds instead of one. The variational autoencoder then groups the samples by itself.'],
  ['Neural network', 'A chain of layers whose weights are learned from the samples.'],
  ['Over-fitting (memorising)', 'The network learns the particular training samples instead of what they have in common: the training loss keeps falling while the held-out loss rises.'],
  ['Prior', 'The simple cloud (a normal distribution) that a variational autoencoder wants the codes of all samples to look like.'],
  ['Random start', 'The weights begin as random numbers. The network is trained from several starts and the one with the best held-out loss is kept.'],
  ['Regularisation', 'Everything that keeps a network from memorising: weight decay, dropout, denoising, early stopping, a small size.'],
  ['Seed', 'The number from which everything random is derived. The same seed gives the same split, the same start and the same network on the same computer.'],
  ['Sensitivity', 'What a small change of one code does to the rebuilt EEM at the average code (its derivative). The nonlinear counterpart of a loading image; it is local.'],
  ['σ (noise level of the data)', 'In a variational autoencoder, how large the unexplained part of a sample is taken to be. Learned from the data by default.'],
  ['Smoothify', 'A setting of the functional versions: a penalty on the size of the coefficients of the rebuilt sample, heaviest on the rough building blocks, so that rebuilt samples move towards smooth functions.'],
  ['Synthetic samples', 'Samples drawn from a variational autoencoder. They are made up by the network and are not measurements; never use them as data.'],
  ['Training, training samples', 'The search for good weights, and the samples it learns from.'],
  ['Variance explained', 'The share of the variation between samples that the rebuilt samples hold: 1 is perfect, 0 is no better than the average sample. The same number as in PCA.'],
  ['Variational autoencoder (VAE)', 'An autoencoder whose codes are made to look like a prior cloud. It can draw new samples and judge how unusual a sample is.'],
  ['Warm-up', 'Letting the weight of the KL term rise slowly from 0 to β over the first epochs of a variational autoencoder, so that the codes are learned before they are pulled towards the prior. 20 % of the epochs by default.'],
  ['Weight', 'A number the network learns: how much one input counts in one neuron.'],
  ['Weight decay', 'A small pull of every weight towards zero during training, so that a weight only grows when the data need it.'],
];

export const limitsPage = {
  id: 'limits',
  icon: 'shield-check',
  title: 'Limits, good practice, glossary',
  summary: 'What a network cannot tell you, rules for using it, what to report, the glossary and the references',
  component: {
    components: comps,
    setup() { return { GLOSSARY }; },
    template: `
    <div>
      <p class="lead">An autoencoder is a flexible description of the samples you gave it, and a description is only good for what it was made from. This page lists what it cannot tell you, the rules that keep you honest, what to put in a paper, and the words and references of the whole guide.</p>

      <h2>What a network cannot do</h2>
      <ul class="g-list">
        <li><b>It does not know what it has not seen.</b> A new lake, a new season, another instrument or another preparation give EEMs the network was not made for. Codes outside the range of the training codes are extrapolations: numbers, but not measurements of anything the network understands. The status <i>Poorly described by the model</i> is there to warn you.</li>
        <li><b>Its codes are not substances.</b> A code is not the amount of a fluorophore and a sensitivity image is not a spectrum. For spectra of substances use PARAFAC; for amounts use the components of a model that is built for it.</li>
        <li><b>Its codes have no fixed meaning.</b> The sign, the order and the mixing of the codes depend on the random start. Give a code a name only after its traversal, and only if the tab <Ui>Stability</Ui> says the codes are reproducible.</li>
        <li><b>It needs samples.</b> With a few dozen it is a rough sketch, and PCA or functional PCA is usually as good and always more stable.</li>
        <li><b>It follows the majority.</b> A handful of strange samples among the training samples can shape the network. Look at the flagged samples, and train again without them if you decide they do not belong.</li>
        <li><b>A variational autoencoder blurs.</b> Its rebuilt EEMs are a little smoother than the measured ones, and what it draws is a typical sample, not a real one.</li>
        <li><b>It is not exactly reproducible across computers.</b> The same seed gives the same result on the same computer; another computer or another version of PyTorch can differ.</li>
      </ul>

      <h2>Good practice, in ten rules</h2>
      <ol class="g-rules">
        <li><b>Correct the data first.</b> The network learns whatever is there, scatter and uncorrected inner-filter effects included.</li>
        <li><b>Look at the held-out numbers</b>, not the training ones.</li>
        <li><b>Compare with PCA</b> or functional PCA with the same number of codes, always, and use the simpler method when it is as good.</li>
        <li><b>Keep the codes few:</b> at most one tenth of the number of samples.</li>
        <li><b>Start from the recommended settings</b> and change one thing at a time.</li>
        <li><b>Run the pixel version once</b> as a reference for the functional one.</li>
        <li><b>Choose smoothify with the scan</b>, and check that the variance explained did not fall.</li>
        <li><b>Check the stability</b> before you interpret a code, and the traversal before you name it.</li>
        <li><b>Never use synthetic samples as data</b>, and never show one without its label.</li>
        <li><b>Say what you did:</b> the <Ui>Methods paragraph</Ui> writes it down.</li>
      </ol>

      <h2>What to report in a paper</h2>
      <ul class="g-list">
        <li>The <b>samples</b>: how many, from where, and how they were corrected.</li>
        <li>The <b>representation</b>: pixels, or the basis, its kernel and number of terms, and the smoothing; the value of smoothify and the way it was chosen.</li>
        <li>The <b>network</b>: the type, the number of codes, the layers, the activation, and the regularisation.</li>
        <li>The <b>training</b>: the split or the folds, early stopping, the number of random starts, the <b>seed</b>, and the versions of PyTorch, FluoViz and the add-on.</li>
        <li>The <b>numbers</b>: variance explained on held-out samples, the comparison with PCA or functional PCA, the stability of the codes.</li>
      </ul>

      <h2>Glossary</h2>
      <div class="table-wrap"><table class="table g-table"><thead><tr><th style="width:26%">Word</th><th>Meaning</th></tr></thead><tbody>
        <tr v-for="g in GLOSSARY" :key="g[0]"><td><b>{{ g[0] }}</b></td><td>{{ g[1] }}</td></tr>
      </tbody></table></div>

      <h2>References</h2>
      <ul class="g-list">
        <li>Hinton, G. E. &amp; Salakhutdinov, R. R. (2006) Reducing the dimensionality of data with neural networks. <i>Science</i> 313:504–507. Autoencoders as a nonlinear PCA.</li>
        <li>Baldi, P. &amp; Hornik, K. (1989) Neural networks and principal component analysis: learning from examples without local minima. <i>Neural Networks</i> 2:53–58. A linear autoencoder spans the principal subspace.</li>
        <li>Kingma, D. P. &amp; Welling, M. (2014) Auto-encoding variational Bayes. <i>International Conference on Learning Representations</i> (arXiv:1312.6114). The variational autoencoder.</li>
        <li>Higgins, I. et al. (2017) beta-VAE: learning basic visual concepts with a constrained variational framework. <i>International Conference on Learning Representations</i>. The weight β.</li>
        <li>Bowman, S. R., Vilnis, L., Vinyals, O., Dai, A. M., Jozefowicz, R. &amp; Bengio, S. (2016) Generating sentences from a continuous space. <i>Proceedings of the 20th SIGNLL Conference on Computational Natural Language Learning</i>, 10–21. Warm-up of the KL term and collapse.</li>
        <li>Kingma, D. P., Salimans, T., Jozefowicz, R., Chen, X., Sutskever, I. &amp; Welling, M. (2016) Improving variational inference with inverse autoregressive flow. <i>Advances in Neural Information Processing Systems</i> 29. Free bits.</li>
        <li>Vincent, P., Larochelle, H., Bengio, Y. &amp; Manzagol, P.-A. (2008) Extracting and composing robust features with denoising autoencoders. <i>Proceedings of the 25th International Conference on Machine Learning</i>, 1096–1103. Denoising.</li>
        <li>Loshchilov, I. &amp; Hutter, F. (2019) Decoupled weight decay regularization. <i>International Conference on Learning Representations</i>. AdamW, the way the weights are adjusted, and weight decay.</li>
        <li>Ramsay, J. O. &amp; Silverman, B. W. (2005) <i>Functional Data Analysis</i>, 2nd ed. Springer. Functions as data, smoothing and roughness penalties.</li>
        <li>Rasmussen, C. E. &amp; Williams, C. K. I. (2006) <i>Gaussian Processes for Machine Learning</i>. MIT Press. Kernels, the Karhunen–Loève basis, and the penalty of a Gaussian process.</li>
        <li>Stedmon, C. A. &amp; Bro, R. (2008) Characterizing dissolved organic matter fluorescence with parallel factor analysis: a tutorial. <i>Limnol. Oceanogr. Methods</i> 6:572–579. PARAFAC of EEMs, for the spectra of substances.</li>
      </ul>
      <p class="muted small">The functional PCA that the functional versions build on is described in the guide <GLink tab="experimental" to="what">Experimental methods</GLink> with its references; the saved models in <GLink tab="saved" to="what">Saved models</GLink>.</p>

      <h2>What to check</h2>
      <ul class="g-list">
        <li>That every claim about <b>what a code means</b> rests on a traversal and on a stability check.</li>
        <li>That the new samples are of the <b>same kind</b> as the training samples, and that the status was read.</li>
        <li>That the paper gives the <b>seed, the versions</b> and the held-out numbers.</li>
        <li>That PCA, or functional PCA, <b>was tried</b> and is reported beside the network.</li>
      </ul>
    </div>`,
  },
};
