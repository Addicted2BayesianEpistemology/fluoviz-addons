// Help > Autoencoders, pages 3-4: pixels or coefficients of a basis, and smoothify; training and small data sets.
// The numbers of the pictures are in data.js (tests/help_data.py computes them with the add-on's own code); tests/test_help_autoencoders.py recomputes and compares them.
import { parts } from './parts.js';
import { FigSmoothify, FigLcurve, FigLoss } from './figures.js';

const comps = { ...parts, FigSmoothify, FigLcurve, FigLoss };

export const functionalPage = {
  id: 'functional',
  icon: 'layer-group',
  title: 'Pixels, coefficients and smoothify',
  summary: 'The network on observed values or on the coefficients of a basis; why coefficients help; what smoothify does and how to choose it',
  component: {
    components: comps,
    template: `
    <div>
      <p class="lead">A network can be given an EEM in two ways: <b>as it was measured</b>, one number per observed pixel, or <b>as a smooth function</b>, written with the few coefficients of a basis. The second is what the functional PCA does, and it is the default here. This page says what each gives, why the second usually works better with removed scatter bands, noise and few samples, and what <b>smoothify</b> adds to it.</p>

      <h2>Two ways to give an EEM to a network</h2>
      <p>The choice is the first block of step 3, <Ui>Functional version</Ui>.</p>
      <div class="table-wrap"><table class="table g-table"><thead><tr><th></th><th><Ui>On the pixels</Ui></th><th><Ui>On the coefficients of a basis (functional)</Ui></th></tr></thead><tbody>
        <tr><td><b>What the network sees</b></td><td>Every observed pixel, minus the average EEM and divided by one number that makes the variables comparable.</td><td>A few coefficients per EEM: the weights of smooth building blocks that, added up, give the smoothed EEM.</td></tr>
        <tr><td><b>How many numbers</b></td><td>Thousands: one per pixel of the domain.</td><td>Tens: the terms of the basis (steps 3 and 4 decide how many).</td></tr>
        <tr><td><b>Scatter bands and gaps</b></td><td>Absent pixels are given as zero and ignored by the loss, whether a scatter band or a rectangle cut out of some samples. The network must discover for itself that neighbouring pixels go together.</td><td>The domain is the pixels that exist; the basis is built on them. A scatter band is not data and is left out. A rectangle cut out of only some samples is estimated for those samples (see below).</td></tr>
        <tr><td><b>Noise</b></td><td>Stays in the data. The network may learn it.</td><td>Estimated and smoothed away before the network sees it (step 4).</td></tr>
        <tr><td><b>Smoothness of the result</b></td><td>Nothing is assumed: the rebuilt EEM may have ripples.</td><td>Always a combination of smooth blocks; <b>smoothify</b> can push further.</td></tr>
        <tr><td><b>Use it for</b></td><td>A reference; or when the functional version might hide something you want to see.</td><td>The usual choice.</td></tr>
      </tbody></table></div>
      <p>In both versions the network is graded with the <b>same measure</b>: how far the rebuilt EEM is from the real one, as a share of the typical difference between samples. One minus that share is the <b>variance explained</b>, the number of PCA, so the comparison with PCA (or with functional PCA) is fair. In the functional version the coefficients are also divided by how large the Gaussian process says each one should be, so that every number the network sees is on the same footing.</p>

      <h2>Why coefficients help</h2>
      <ul class="g-list">
        <li><b>Removed scatter bands.</b> After scatter removal the EEM is an irregular part of the grid. The functional versions treat that part as the domain and write the EEM as a smooth function on it, so a band that is gone is not a hole the network has to learn to step over.</li>
        <li><b>Noise.</b> Every pixel carries noise. A network on the pixels spends part of its capacity on noise it cannot predict; on coefficients the noise has been smoothed away first, with the smoothing you checked in step 4.</li>
        <li><b>Few samples.</b> The number of weights grows with the number of inputs. With the default <Ui>Small</Ui> network and 4 codes, 10 000 pixels give more than a million weights, and 90 coefficients give fewer than 20 000 (the card <Ui>The network in numbers</Ui> shows the exact count for your data). With 40 samples, a network of 20 000 weights can still memorise; one of a million certainly can.</li>
        <li><b>Smooth results.</b> The rebuilt EEM is a sum of smooth blocks, so it is smooth whatever the network does.</li>
      </ul>
      <p>What the functional version can hide: a basis with too few terms cuts the narrow peaks, and a smoothing that is too strong blurs them. That is why steps 3 and 4 have their checks, and why the pixel version stays available as a reference: if both give the same variance explained and the same codes, the functional version is not hiding anything.</p>

      <h2>A region cut out of some samples only</h2>
      <p>In the viewer or on the Corrections page you can cut a rectangle out of one EEM, or out of a few. For those samples the pixels are <b>missing</b>: not measured, and not zero. A pixel stays in the domain as long as at least the share <Ui>Minimum coverage of a pixel</Ui> of step 2 (0.5 by default) of the samples have it; what happens to the samples that lack it depends on the version.</p>
      <ul class="g-list">
        <li><b>On the pixels</b> nothing is made up. The missing pixels are left out of the grade, and the sample is graded on the rest.</li>
        <li><b>On the coefficients of a basis</b> the coefficients of such a sample have to be <b>estimated</b>. The choice <Ui>Pixels missing in some samples</Ui> in step 2 decides how. <i>Estimate them from the other samples</i> (the default) uses the pixels the sample has together with what the other samples show about how the parts of an EEM go together. <i>Smooth each sample on its own pixels</i> uses only the pixels it has, and the part it lacks drifts towards the average EEM. A saved model reads a new sample with a gap in the same way.</li>
      </ul>
      <p>An estimated coefficient is less certain than a measured one, and the more a sample lacks, the more of it is a guess. So, in the advanced network settings of step 5, the switch <Ui>Weigh samples that lack part of the domain</Ui> (on by default) lets a sample count in proportion to the share of its EEM that was measured: a sample that lacks a fifth counts for four fifths, one that lacks half counts for half, and none counts for less than 5 %. Switch it off to count every sample alike. It changes nothing when no sample lacks anything. On the tab <Ui>Samples</Ui> a sample that lacks a part says how much: the measured picture is blank there and the rebuilt one is filled in by the network. The variable <i>Autoencoder: share of the domain not measured</i> can colour a plot.</p>
      <p>What no setting can do is to replace a large and important region. If a region that carries the signal is cut out of many samples, the codes of those samples describe what was measured plus a guess, and a code that lives mostly in that region is less reliable for them. Colour a plot of the codes by the variable above to see whether those samples sit apart, or cut the region out of every sample.</p>

      <h2>What smoothify penalises</h2>
      <p>While it learns, the network is graded on one thing: how well it rebuilds. <Ui>Smoothify</Ui> adds a second thing. The grade becomes</p>
      <p style="text-align:center"><b>rebuilding error + smoothify × size of the coefficients of the rebuilt EEM</b></p>
      <p>The second term adds up the squares of the coefficients of what the decoder draws, each divided by how large the Gaussian process says it should be. In a Gaussian-process basis the building blocks run from smooth to rough, and a rough block is expected to have a small coefficient, because smooth surfaces contain little of it. So <b>the same amount of a rough block costs far more than the same amount of a smooth one</b>. Big coefficients are discouraged, above all in the rough directions, and the rebuilt EEMs move towards smooth functions, at the price of a slightly larger rebuilding error.</p>
      <p>This is not an arbitrary penalty. It is exactly the one a Gaussian process puts on a function: the rebuilt EEM that makes the grade smallest is the most probable function under the process that fits the data, which is also what a smoothing spline does (Rasmussen &amp; Williams 2006; Ramsay &amp; Silverman 2005). With B-splines or cosines it is the roughness penalty of the representation, in the same spirit.</p>
      <p>It is <b>not the same as the smoothing of step 4</b>. Step 4 smooths each <i>measured</i> EEM before the network sees it. Smoothify limits what the network can <i>draw</i>. Both can be on.</p>
      <FigSmoothify>One rebuilt function, with and without smoothify (a cut through a rebuilt EEM along emission at one excitation wavelength; real numbers from the add-on's own code, 40 noisy made-up EEMs, the held-out sample and wavelength where the effect is clearest). Grey dots: the measured values. Without smoothify the rebuilt function carries wiggles that the smoothed measured curve does not have; with it they are much smaller, and the peak is where it was. The roughness numbers are those of the whole rebuilt EEM; the last lines are the price, over all held-out samples.</FigSmoothify>
      <p>The strength is on a scale where <b>1</b> weighs the roughness of an ordinary sample as much as the rebuilding error of the mean EEM, the rebuilding that does nothing; the scale is the same for every basis. The page offers a logarithmic slider from 0 to 10, and the scan starts at 0.01. What is right depends on how well the network already rebuilds: a network that rebuilds 99 % of the variation has an error of only 0.01, so against it even a smoothify of 0.01 is a heavy weight. The better the fit, the smaller the value that matters, so let the scan say; if it suggests its smallest value, try smaller ones too.</p>

      <h2>How to choose it</h2>
      <p>Start with <b>0</b>, which is off. The functional version is already smooth, and smoothify costs some rebuilding error. Think of turning it on when the tab <Ui>Smoothness</Ui> of step 8 says that the rebuilt EEMs are rougher than the measured ones, when held-out samples are rebuilt worse than training samples, or when you have few samples. Then do not guess: use the scan.</p>
      <ol class="g-steps">
        <li>In step 5 press <Ui>Smoothify scan</Ui> (or the same button on the tab <Ui>Smoothness</Ui> of step 8). It trains the network again for smoothify 0 and for eleven values between 0.0001 and 10 (equal steps on a logarithmic scale) and takes some minutes.</li>
        <li>Look at the chart <Ui>Error against roughness</Ui> (an L-curve, below): one point per value, the rebuilding error up, the roughness to the right.</li>
        <li>Find the <b>corner</b>: the point after which the error rises quickly while the roughness falls less and less. The page suggests the <b>largest value whose held-out error is still within one standard error (or 2 %) of the error without smoothify</b>, and marks it.</li>
        <li>Press <Ui>Use N</Ui> (or <Ui>Use the suggested value</Ui> on the tab <Ui>Smoothness</Ui>) and train the network with it (<Ui>Train the network</Ui>).</li>
        <li>Check again: the variance explained on held-out samples should be about the same as before, and the roughness lower. If the held-out variance explained fell by more than a point or two, the value is too large.</li>
      </ol>
      <FigLcurve>The smoothify scan with real numbers (the data set of the previous picture): the rebuilding error (up) against the roughness of the rebuilt EEMs (right), one point per value of smoothify. From the right (smoothify 0) towards the left the roughness falls while the error hardly moves; further left the roughness falls less and the error rises, and at the far end every EEM is rebuilt as the mean. The ring marks the value the page suggests.</FigLcurve>

      <h2>What too much does</h2>
      <p>With a large value the cheapest way to avoid the penalty is to draw small coefficients everywhere: every rebuilt EEM shrinks towards the mean EEM. Peaks lose height, the differences between samples fade, the variance explained falls, and in the end every sample is rebuilt as the mean. On the L-curve this is the far end: the roughness is almost zero and the error approaches that of the mean. The codes then carry little, and <b>a low roughness reached that way is not a better result</b>.</p>

      <h2>Which smoothness is measured</h2>
      <p><Ui>Smoothness is measured by</Ui> has two choices. <Ui>The smoothness of the representation</Ui> (the default) is what was described above: with the Gaussian-process basis the standardised coefficients, with B-splines and cosines how much the function bends from pixel to pixel (second differences on the grid), scaled so that an ordinary sample counts about 1. <Ui>The plain size of the coefficients (ridge)</Ui> adds up the squares of the coefficients as they are, without regard for which are rough; use it as a reference. With <Ui>On the pixels</Ui> there is no smoothness to measure, so smoothify can only act as a plain penalty on the size of the values, and the page says so.</p>
      <p>The variational autoencoder uses it in the same way: the penalty is divided by the noise level of the data, as the rebuilding error is, so the same value means the same.</p>

      <h2>What to check</h2>
      <ul class="g-list">
        <li>That <b>both versions agree</b> when it matters: run the pixel version once as a reference and compare the variance explained and the codes.</li>
        <li>That the <b>basis and the smoothing</b> of steps 3 and 4 were checked: the functional version is only as good as they are.</li>
        <li>That <b>smoothify is chosen with the scan</b>, not by eye, and that the variance explained on held-out samples did not fall.</li>
        <li>That the rebuilt EEMs <b>still have the peaks</b>: look at the traversals and at a few samples (step 7).</li>
        <li>That the samples with a <b>region cut out</b> do not sit apart from the others in the plots of the codes; if they do, the codes are partly a guess.</li>
      </ul>
    </div>`,
  },
};

export const trainingPage = {
  id: 'training',
  icon: 'sliders-v-alt',
  title: 'Training and small data sets',
  summary: 'The split into training and held-out samples, early stopping, random starts, the loss curves, over-fitting, how many samples are enough',
  component: {
    components: comps,
    template: `
    <div>
      <p class="lead">Training is where an autoencoder can go wrong without telling you: a network with tens of thousands of weights can <b>memorise</b> a few dozen EEMs perfectly and know nothing about the next one. Everything on this page is a safeguard against that, and a way to see whether it happened. The defaults are chosen for tens to a few hundred samples, which is the normal case.</p>

      <h2>Training and held-out samples</h2>
      <p>The samples are split in two. The network learns from the <b>training samples</b>. The <b>held-out samples</b> (also called the validation samples) are never used to change a weight; they are only used to ask, after every epoch, how well the network rebuilds samples it has not seen. That number is the honest one. <b>Every quality number on the page is a held-out number</b>, shown with the training value next to it so that you can see the gap.</p>
      <ul class="g-list">
        <li><b>30 samples or more:</b> a random 20 % are held out (at least 5), set by <Ui>Share of samples held out</Ui>. The split is made from the random seed, so it is the same every time.</li>
        <li><b>Fewer than 30 samples:</b> there are too few to give up a fifth of them, so the page uses <b>5-fold validation</b> (<Ui>Folds (fewer than 30 samples)</Ui>). The samples are divided into five groups; five networks are trained, each rebuilding one group that it never saw; the numbers are those of samples rebuilt by a network that had not seen them. The final network is then trained on <b>all</b> samples, for the number of epochs the folds found (their median best epoch, never fewer than 20); with no held-out samples left, the best of its random starts is the one with the lowest training loss.</li>
        <li><b>Fewer than 12 samples:</b> the page refuses, with the reason in words ("Only 9 samples are included: …"). Use PCA or functional PCA. The page has a switch in step 6 for people who insist ("Train on fewer than 12 samples anyway"); the numbers are then rough sketches and not to be trusted.</li>
      </ul>

      <h2>Early stopping and the loss curves</h2>
      <p>The <b>loss</b> is the number the network tries to make small. The chart <Ui>Training so far</Ui> (and, afterwards, <Ui>Loss during training</Ui> in step 8) shows it twice: on the training samples and on the held-out samples. A healthy training looks like this:</p>
      <ul class="g-list">
        <li>Both lines fall at the start, until the bends have learned what they can add. (With the default start from PCA the first epoch is already at the level of PCA, so the lines begin lower than they would from random weights.)</li>
        <li>The training line keeps falling slowly; the held-out line <b>flattens</b> and stays above it (some gap is normal, and larger with noisy pixels).</li>
        <li>The best epoch, marked on the chart, comes well before the end.</li>
        <li>If the bends do not help for your data, the held-out line does not fall below its first value: the best epoch is the start, and the verdict says that the network could not improve on its PCA starting point.</li>
      </ul>
      <p><b>Over-fitting</b> looks different: the training line keeps falling, but the held-out line stops falling and rises. The network is learning the particular samples, not what they have in common.</p>
      <FigLoss>Loss curves of two real trainings (the same 36 noisy made-up EEMs, on the pixels; the lower the better). Left: the default small network with weight decay and early stopping. Right: a large network without weight decay and without a stop. The vertical dashed line marks the epoch of the best held-out loss, and the weights of that epoch are the ones the page keeps.</FigLoss>
      <p><b>Early stopping</b> is the safeguard: when the held-out loss has not improved for <Ui>Patience (epochs without improvement)</Ui> epochs (60 by default), training stops, and the weights of the <b>best</b> epoch are restored, not those of the last. When the network starts from PCA, the starting weights are a candidate too: if no epoch does better on the held-out samples than PCA does, the verdict says so ("The network could not improve on its PCA starting point on held-out samples") and the bends did not help for these data. <Ui>Maximum number of epochs</Ui> (600) is only a ceiling. A training that runs to the ceiling with the held-out loss still falling needs more epochs; one that stops after 100 is normal.</p>
      <p><b>If training stops too early.</b> The verdict in step 8 says why a training ended. A noisy held-out loss (a few held-out samples) can use up the patience before the bends have helped. The fold <Ui>Advanced training settings</Ui> in step 6 lets you change that without touching the defaults: <Ui>Never stop before this many epochs</Ui> forces a minimum, <Ui>Patience (epochs without improvement)</Ui> and <Ui>Smallest improvement that counts (%)</Ui> make the stop more or less patient, <Ui>Average the loss over this many epochs</Ui> steadies a jumpy loss, and <Ui>Stop training when</Ui> can use the training loss or switch the stop off altogether (every epoch is trained and the best one is still kept). <Ui>Longest time for one network (minutes)</Ui> is the safety net for the other direction. With fewer than 30 samples the final networks are trained for the number of epochs the folds found, but at least 20: <Ui>Epochs of the final networks</Ui> sets that number yourself.</p>

      <h2>Random starts</h2>
      <p>The weights start as random numbers, which is called a <b>random start</b>, and different random numbers lead to different networks, some better than others. <Ui>Number of random starts</Ui> (3 by default) trains the network from that many starts and <b>keeps the one with the best held-out loss</b>. The table <Ui>Random starts</Ui> in step 8 shows all of them. When the best and the worst of the starts are less than 2 points of variance explained apart the verdict says that they agree; when they are further apart it says "the result depends on the start": the network is fragile, and the codes of another start would not be the same. One more reason to read the tab <Ui>Stability</Ui>. Because the best of three is chosen with the held-out samples, the held-out numbers are very slightly optimistic; with very few samples treat them as a guide, not as a promise.</p>

      <h2>The settings that keep a network from memorising</h2>
      <ul class="g-list">
        <li><b>Size.</b> The smaller the network, the less it can memorise. <Ui>Small</Ui> has layers of 64 and 32 neurons; <Ui>Medium</Ui> (128, 64) and <Ui>Large</Ui> (256, 128, 64) are for hundreds of samples.</li>
        <li><b>Weight decay</b> (<Ui>Weight decay</Ui>, 0.001): a small pull of every weight towards zero, so that a weight only grows when the data need it.</li>
        <li><b>Dropout</b> (<Ui>Dropout</Ui>, 0 by default): during training a share of the neurons of the encoder is switched off at random, so that no neuron can be relied on alone. Switched off when the network is used.</li>
        <li><b>Denoising</b> (<Ui>Noise added to the input (in standard deviations)</Ui> and <Ui>Share of inputs hidden</Ui>, both 0 by default): the network is shown a damaged copy of every sample (some noise added, some inputs hidden) and must rebuild the <i>clean</i> one (Vincent et al. 2008). The codes become more robust, and missing values are handled better.</li>
        <li><b>Learning rate</b> (<Ui>Learning rate</Ui>, 0.001): the size of each adjustment of the weights. Too large and the loss jumps about, or the page says "The training diverged": lower it. <b>Batch size</b> (<Ui>Batch size</Ui>): how many samples are looked at before each adjustment; automatic by default: a sixth of the training samples, but at least 8 and at most 32.</li>
      </ul>

      <h2>How many samples are enough?</h2>
      <p>There is no single number, but the page gives its own rules, and you can follow them.</p>
      <div class="table-wrap"><table class="table g-table"><thead><tr><th>Samples</th><th>What the page does</th><th>What to expect</th></tr></thead><tbody>
        <tr><td>Fewer than 12</td><td>Refuses.</td><td>Use PCA or functional PCA.</td></tr>
        <tr><td>12 to 29</td><td>5-fold validation; advises a few codes.</td><td>A rough sketch. With two or three codes the network is unlikely to beat PCA; look at the comparison.</td></tr>
        <tr><td>30 to 100</td><td>20 % held out; the advice keeps the codes at one tenth of the samples (40 samples, at most 4 codes).</td><td>The functional version with the default small network works for smooth data. The VAE's cloud is poorly known.</td></tr>
        <tr><td>More than 100</td><td>The same, with more freedom.</td><td>The bends begin to pay if the data bend. A medium network is possible.</td></tr>
      </tbody></table></div>
      <p>The card <Ui>The network in numbers</Ui> keeps the score: <Ui>Samples per code</Ui> and <Ui>Samples per parameter</Ui>. The page warns when there are fewer than 10 training samples per code ("Only … training samples per code") and when the network has far more weights than samples ("it can memorise them"). Take the warnings seriously; the held-out numbers are the judge.</p>

      <h2>Reproducibility</h2>
      <p>Everything random is set from the <Ui>Random seed</Ui> (0 by default): the split into samples, the starting weights, the order of the batches. The same data, settings, seed, computer and program versions give the same network. Another computer or another version of PyTorch can differ in the last digits and, after hundreds of epochs, a little more, so the result stores the seed and the versions, and the methods paragraph prints them. To find out how much your conclusions depend on the seed, train with two or three others and compare the codes (tab <Ui>Stability</Ui>).</p>

      <h2>Training carefully, in six steps</h2>
      <ol class="g-steps">
        <li>Read <Ui>The network in numbers</Ui> in step 5: samples per code, the parameters and the warnings.</li>
        <li>Train with the recommended settings. Do not change anything else yet.</li>
        <li>Look at <Ui>Loss during training</Ui>: both curves fall, the held-out one flattens, the best epoch is not the last.</li>
        <li>Look at the verdict of the tab <Ui>Training</Ui> and at the table <Ui>Random starts</Ui>.</li>
        <li>If the network over-fits: fewer codes, the <Ui>Small</Ui> size, more <Ui>Weight decay</Ui>, some <Ui>Denoising</Ui>. If it under-fits (both curves high, still falling at the end): more epochs, or one more code.</li>
        <li>Change <b>one</b> setting, train again, and compare the held-out variance explained with the previous run.</li>
      </ol>

      <h2>What to check</h2>
      <ul class="g-list">
        <li>That the <b>held-out loss flattens</b> and the best epoch is not the last one.</li>
        <li>That the <b>gap</b> between training and held-out numbers is small; a large gap is memorising.</li>
        <li>That the <b>random starts agree</b>, and that the codes do not change much with another seed.</li>
        <li>That there are <b>at least 10 samples per code</b>, and that you read every yellow warning.</li>
        <li>That you did not <b>tune the settings on the held-out samples until they looked good</b>: after many rounds they are no longer held out. With enough samples keep a last part untouched.</li>
      </ul>
    </div>`,
  },
};
