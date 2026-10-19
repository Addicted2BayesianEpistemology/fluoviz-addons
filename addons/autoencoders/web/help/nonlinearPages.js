// Help > Autoencoders, page 7: the tab "Nonlinear effects" of step 7 (which nonlinear trend each code fits).
// tests/test_help_autoencoders.py checks every word of the interface (<Ui>...</Ui>) against the page and docs/ui.md, every rule and number against the add-on's code, and the pictures against a recomputation.
import { parts } from './parts.js';
import { FigShapes, FigNonlinear } from './nonlinearFigures.js';

const comps = { ...parts, FigShapes, FigNonlinear };

export const nonlinearPage = {
  id: 'nonlinear',
  icon: 'arrow-growth',
  title: 'Nonlinear effects of the codes',
  summary: 'Which trend each code fits: response curves, where it bends, local effects, and two codes together',
  component: {
    components: comps,
    template: `
    <div>
      <p class="lead">Page 6 showed what each code does to the EEM. This page asks the next question: <b>does it do it in proportion?</b> Twice as far along a code, is the change of the EEM twice as large (a straight line, as with a PCA loading), or does the network bend: does the effect level off, turn round, or depend on another code? The bends are the reason to use an autoencoder instead of PCA, so it is worth seeing <b>which</b> bend each code fits, <b>where</b> in the EEM it happens and whether two codes <b>interact</b>. The tab <Ui>Nonlinear effects</Ui> of step <Ui>7 · Results</Ui> has four sections for it: <Ui>Response curves</Ui>, <Ui>Where it bends</Ui>, <Ui>Local effects</Ui> and <Ui>Two codes</Ui>. They need no new training and no new data: they only ask the trained network questions.</p>
      <Callout kind="info">Like the traversals, everything here describes <b>this trained network</b>. Another network (another seed, other settings) can bend in other places, and the codes have no fixed meaning. Trust a bend where there are samples, and look for it again in a second network before you build on it.</Callout>

      <h2>How a code is moved</h2>
      <p>All four sections move a code the way the traversals do: from −3 to +3 <b>standard deviations</b> (SD) of that code over your training samples, with every other code at its average. The standard deviation of a code is the usual size of its variation over your samples, and using it as the unit lets codes of different sizes be compared. The page starts with 41 positions and a range of ±3 SD; <Ui>Range (± SD)</Ui> (0.5 to 4) and <Ui>Points per curve</Ui> (5 to 201) change them. A variational autoencoder is moved through the centre of each blob (the code the program uses for a sample). At each position the network draws an EEM, and the sections read different things from these EEMs.</p>

      <h2>Response curves: one measurement against one code</h2>
      <p>A <b>feature</b> is a single number read from a decoded EEM: the intensity at one pixel, the average over a box, the overall intensity, the absorbance at a wavelength, or one of the numbers the network works with. Drawn against the position of a code it is the <b>response curve</b> of that code. Without any choice by you the page draws two features: <Ui>Intensity at the pixel this code moves most</Ui> (a different pixel for every code: the one whose intensity changes most along the code, so that a code is never judged on a pixel it does not touch) and <Ui>Overall intensity (the average over the whole EEM)</Ui>. To measure something else press <Ui>Add a feature</Ui>, choose its kind (<Ui>Intensity at a pixel</Ui>, <Ui>Mean intensity in a box</Ui>, <Ui>Highest intensity in a box</Ui>, <Ui>Overall intensity</Ui>, <Ui>Absorbance at a wavelength</Ui> or <Ui>A coefficient of the network</Ui>), type the wavelengths and press <Ui>Add this feature</Ui>. Or open <Ui>Pick a pixel or a box on the picture of the mean EEM</Ui> and click a pixel, or choose <Ui>Select region</Ui> and drag a box. A pixel must be one the model uses; the page says so when it is not. Up to six features can be drawn at once, and each chip can be removed.</p>
      <p>Each chart is one code. The thick line is the response, with the other codes at their average. Every feature has its own colour, line style and marker, named in the legend above the charts, so no feature is told apart by colour alone. <Ui>Compare shapes (scale each feature from 0 to 1)</Ui> is on at first: it scales each feature from its lowest to its highest value along the code, so that a single pixel and an overall intensity can be compared by shape; switch it off to see the real values. The page draws the first 8 codes; <Ui>Show the other N codes</Ui> draws the rest.</p>
      <ul class="g-list">
        <li><b>The dots.</b> The faint dots are your own samples (<Ui>Show the training samples</Ui>): the horizontal position of a dot is the sample's own code, its vertical position is the same measurement made on the sample (on the smoothed EEM the network was trained to rebuild). Dots close to the line say that the code explains the measurement by itself; the scatter around the line is what the <b>other codes</b> do.</li>
        <li><b>Single samples.</b> <Ui>Also draw single samples' curves (the other codes at that sample's own values)</Ui> adds, for each code, the curves of up to 12 samples chosen over the range of the code (or the samples you name in <Ui>Samples for these curves</Ui>). A sample keeps its own values of the other codes and only the code of the chart moves. If the thin curves run <b>parallel</b> to the thick one, the code acts independently of the others and only the level changes; if their <b>shapes</b> differ, what the code does depends on the other codes.</li>
      </ul>

      <h2>The words under a curve</h2>
      <p>Under each chart the page says in words what the curve looks like and how far a straight line goes. It fits the best straight line to the curve, giving the positions near the average more weight (the weights follow the bell curve of the codes, because most samples lie there), and quotes <b>what a straight line explains</b>: the R² of the line, 100 % being a perfect line, and the largest distance of the curve from the line as a share of the range of the curve. The words come from fixed rules, applied in this order:</p>
      <div class="table-wrap"><table class="table g-table"><thead><tr><th>The page says</th><th>When</th></tr></thead><tbody>
        <tr><td>has almost no effect on this feature</td><td>the curve moves by less than 2 % of the largest movement among the codes for the same feature.</td></tr>
        <tr><td>changes direction several times</td><td>it turns round twice or more.</td></tr>
        <tr><td>rises then falls (peak at about …), falls then rises (lowest at about …)</td><td>it turns round once.</td></tr>
        <tr><td>nearly linear</td><td>it does not turn round and stays within 5 % of its range of the straight line.</td></tr>
        <tr><td>S-shaped</td><td>it is flat at both ends, or it crosses its best straight line three times or more.</td></tr>
        <tr><td>levels off (saturates) above about …, or below about …</td><td>it is flat at one end.</td></tr>
        <tr><td>has a bend (convex or concave)</td><td>otherwise. Convex: it bends upwards, the slope grows with the code. Concave: it bends downwards, the slope shrinks.</td></tr>
      </tbody></table></div>
      <p>A curve <b>turns round</b> when it moves back from its highest or lowest point by at least 10 % of its range (a smaller dip is ignored). It is <b>flat at an end</b> when, over the last quarter of the positions or more, it stays within 5 % of its range of its value at that end. A curve that has both positive and negative values also says where it switches sign; for an intensity this is a warning, since the network then draws a negative intensity: it is beyond what it has seen.</p>
      <FigShapes>Eight response curves of known shape (real numbers from the add-on's own code), each with the words the page uses and what the best straight line (dashed) explains of it. Bends are common and are not all alike: a curve that levels off or an S is not the same thing as a peak, and a wave shows that a code has several regimes.</FigShapes>

      <h2>Where it bends: the whole EEM at once</h2>
      <p>One pixel can mislead. <Ui>Where it bends</Ui> does the straight-line test at <b>every pixel</b> of the EEM, for every code, and reports <b>NL</b>, which is 1 minus the R² of the best straight line: 0 means that the pixel responds in a straight line, 1 means that nothing about its response is a straight line. The strip <Ui>Where each code bends the picture (NL = 1 − R² of the best straight line, 0 to 1)</Ui> has one EEM per code on one colour scale, and the colour bar says which colour is which: the bottom is a pixel that responds in a straight line, the top a pixel that bends a lot. A <b>grey</b> pixel is outside the model or is changed by this code by less than 2 % of its largest effect, so that the index would mean nothing there. <Ui>Show</Ui> has a second choice, <Ui>Size of the effect</Ui>, that draws how much each pixel changes along the code: read the two together, because a bend where nothing changes does not matter.</p>
      <p>The bars <Ui>Share of the effect of each code that is not a straight line</Ui> give one number per code: the part of everything the code does to the EEM that no straight line can describe. It is measured the way the network was trained, so every pixel counts by the area it covers and the absorbance counts as much as the fluorescence. 0 % is a code that acts like a PCA loading and 100 % a code that is entirely nonlinear. The dotted line is 5 %: below it the code acts almost like a straight line. A code the network ignores has no share and is marked no effect. A network that reads the absorbance also gets one chart for every code with the same index against the wavelength.</p>
      <FigNonlinear>Two codes of the network of 120 made-up EEMs in which one band moves (real numbers). Top: the response of the pixel each code moves most, with the samples as faint dots and the best straight line dashed; the code on the left bends and the code on the right acts almost like a PCA loading. Bottom: where each code bends the picture, and the share of its effect that is not a straight line.</FigNonlinear>
      <ul class="g-list">
        <li><b>Share and size go together.</b> A code with a large share but a small effect matters little: the text that appears when the pointer is over a bar also says how much of the total effect of all the codes that code carries.</li>
        <li><b>Where the bend is.</b> A region of high values (the top of the colour bar) that sits on a band of the EEM says that the band moves, grows or fades nonlinearly with the code. A patch of high values at the edge of the model is more often noise.</li>
      </ul>

      <h2>Local effects: the loading is not the same everywhere</h2>
      <p>A PCA loading is the same for every sample. The <b>local loading</b> of a code is what one SD of the code does to the EEM <b>at a particular sample</b> (mathematically the derivative of the decoder there, which the program works out exactly, not by trial and error). Where the network bends it differs from sample to sample. <Ui>Local effects</Ui> takes the code chosen in <Ui>Code</Ui> and draws its loading at the average code (<Ui>At the mean code</Ui>) next to the loadings at samples spread along the code: at first the samples at the 10th, 50th and 90th percentile (<Ui>10th, 50th and 90th percentile</Ui>), or up to six that you choose with <Ui>Samples I choose</Ui> and <Ui>Add a sample</Ui>. One signed colour scale is used for all the pictures. If they look alike, the code acts the same way everywhere; a band that appears in one of them but not in the others is a bend.</p>
      <p>The map under them shows the <b>variability</b> of the loading: at each pixel, the standard deviation of the loading over <b>all</b> your samples (low on the colour bar: the same everywhere). The bars <Ui>How much the shape of the effect of each code changes from sample to sample</Ui> give, for every code, the average over the samples of 1 minus the <b>cosine</b> between the loading at the sample and the loading at the average code. The cosine of two pictures is 1 when they have the same shape, whatever their size, and falls as the shapes differ; so 0 means the same shape everywhere. It measures changes of <b>shape</b> only: a code that just gets stronger or weaker keeps its shape and shows 0, and the variability map and the section <Ui>Where it bends</Ui> show that instead.</p>

      <h2>Two codes together</h2>
      <p>Codes can interact: what one does depends on where the other is. <Ui>Two codes</Ui> moves two codes together on a grid of 3 × 3, 5 × 5 or 7 × 7 pictures (<Ui>Pictures per code</Ui>), at first 5 × 5 over ±2 SD, with all the other codes at their average. <Ui>First code (across)</Ui> and <Ui>Second code (up)</Ui> are at first the two codes with the largest share of bending. The gallery is the decoded EEM at every grid point, with the position of each code written above it; the middle picture is the average code. If the two codes acted independently, the picture at a grid point would be the average picture plus what the first code does alone plus what the second does alone. The <b>interaction image</b> shows, pixel by pixel, the largest difference over the grid between the real picture and that sum, and <Ui>Interaction index</Ui> is the squared size of all those differences divided by the squared size of all the changes over the grid: 0 means that the two codes act independently (their effects simply add up), and larger numbers mean that they interact. The words say that the codes <i>act almost independently</i> below 0.05, <i>interact a little</i> below 0.25 and <i>interact strongly</i> above.</p>
      <p>The heat map shows one feature over the plane of the two codes (the pixel the two codes move most, unless you choose another in <Ui>What to measure</Ui>), with your samples as dots at the positions of their codes. When the dots are coloured by the value of the feature measured on each sample, a dot whose colour matches the colour behind it says that the map agrees with the data.</p>

      <h2>Reading the answers honestly</h2>
      <ul class="g-list">
        <li><b>Look at the dots.</b> A bend in a region without samples is the network's guess, not a finding. Beyond about ±2 SD there are usually few samples.</li>
        <li><b>A bend is not always chemistry.</b> A band that moves with a condition, quenching or an inner-filter effect that was not fully corrected are real nonlinear effects (page 1); but a network trained on few samples can also bend to follow noise. Step 8 helps: the card <Ui>Is the nonlinearity worth it?</Ui> says whether the bends improve the rebuilding of samples the network has not seen.</li>
        <li><b>A straight code is not a failure.</b> A code that is nearly linear in all these views is one that PCA would also find; the interest lies in the codes that are not.</li>
        <li><b>These are views of one network.</b> Train again with another seed and see whether the same codes bend in the same way, as the stability check of step 8 invites you to.</li>
      </ul>

      <h2>What to check</h2>
      <ul class="g-list">
        <li>That the <b>dots</b> follow the curve in the region you want to talk about.</li>
        <li>For a bend you want to report: that it shows in <b>more than one feature</b>, in the <b>map</b> of <Ui>Where it bends</Ui> and in a <b>second network</b>.</li>
        <li>That a code with a large <b>share</b> is also a code with a large <b>effect</b>.</li>
        <li>Whether two codes <b>interact</b> before you give either of them a meaning of its own.</li>
        <li>That the sentences under the charts are read with the <b>numbers</b>: a straight line that explains 95 % of a curve may be good enough for your purpose.</li>
      </ul>
    </div>`,
  },
};
