# Multimodal latent factors add-on: the interface (contract for the Help guide and for the tests)

*Words in bold are labels the page shows exactly; the Help guide (`web/help.js`) quotes them with `<Ui>`, and a test looks for every bold label in the files of `web/` (or in the Functional PCA components of FluoViz for steps 1 to 3) and for every `<Ui>` word of the guide in this file.  Words in `code font` are not shown (test hooks, routes).  What the numbers mean: `SPEC.md`, `docs/api.md`.*

## Where things are

* Sidebar: group **Experimental methods**, item **Multimodal latent factors** with the badge **add-on**.  Route `/mblf`, page title **Multimodal latent factors (add-on)**, `needsData`.
* **Help**: the guide tab **Multimodal latent factors** (`#/help?tab=addon:mblf:mblf`); the button **How to read this** in the page head opens it.
* EEM pictures are drawn with FluoViz's `EEMView` and follow the setting *Axes of fluorescence maps*; the server sends images `(n_ex, n_em)`.
* One **stepper** of eight steps above ONE panel, like Functional PCA.  Steps 1 to 3 are the Functional PCA components (routes `/api/fpca/*`).

## Page head

* Title **Multimodal latent factors (add-on)**, pill **add-on**, a lead paragraph, **How to read this**.
* Result bar (`ResultBar kind="mblf"`): the fitted models (`mblf-N`), **Save this model**, and for an outdated model **Fit again with the same settings**.

## The stepper

**1 · What to model** · **2 · Domain** · **3 · Basis** · **4 · Model and sampler** · **5 · Fit** · **6 · Results** · **7 · Prediction and uncertainty** · **8 · Checks and use**.  Every step has the box **What this step does and what to check**, its problems and **Back** / **Next**.

## Steps 1 to 3

* **1 · What to model**: the Functional PCA cards **Fluorescence**, **Absorbance**, **Both together (multivariate)** and **Pre-treatment**; then the card **Other values and covariates**: **Values to model (optional)** (**Add a value**), **Covariates (optional)** (**Add a covariate** (a searchable picker that stays open after each choice, so several variables are added in one go; Esc closes it)), beside every number a small **log** box (model it as its log10; a category has none), the section **How they enter the model** (a table: **Variable**, **Role**, **Log**, **Centred on** (**Mean**, **Median**, **Not centred**; a category **Centred indicators** / **0 / 1 indicators**), **Scaled by** (**Standard deviation**, **Robust (MAD)**, **Not scaled**)), the switch **Use samples that lack a measurement (they contribute what they have, and the model predicts what they lack)**, and the sentence **This model:** that says what the combination answers.
* **2 · Domain**: the Functional PCA step.  **3 · Basis**: the Functional PCA representation step with the Gaussian-process basis only, and on top the card **What the model sees** (per function the switch **Log scale: log10(fluorescence + offset)** / **Log scale: log10(absorbance + offset)** with its **Offset** (empty = from the data), **Centred on** (**Mean**, **Median**, **None**) and **Scaled by** (**One number**, **Pareto**, **Every point**, **None**)), with two values or more the card **Basis of the values** (**One per value** / **Principal components**, **Share of their variance to keep (%)**, **Or a number of components**), the card **Where the basis functions come from** (**Gaussian-process prior (the method’s own)** / **Functional PCA of your samples** / **A functional PCA you fitted**, with **Fitted functional PCA**: the analyses of the Functional PCA page) with the switch **Allow more than 150 basis functions** (a step-3 problem blocks the fit while a basis has more than 150 terms and the switch is off).

## Step 4 · Model and sampler

* **Factors**: **Largest number of factors**.  **Estimation**: **Full posterior (MCMC)** / **Maximum likelihood (quick)**.
* **Length of the sampling**: **Quick look**, **Standard**, **Thorough**; the fold **Advanced sampler settings** (**Choose the chains and iterations myself**, **Chains**, **Warm-up iterations**, **Kept iterations**, **Target acceptance rate**, **Largest tree depth**, **Draws kept in the result**).  For maximum likelihood **Starting points**.
* The fold **Priors (advanced)** with **Back to the defaults of the method**.
* The card **The Stan engine**: its state, **Prepare the Stan engine (about N MB, once)**, **Cancel**, **Remove the Stan engine**.

## Step 5 · Fit

* The card **What will be done** (the samples, the basis of every mode, the covariates, the estimation, the size of the problem); the fold **Seed and title** (**Random seed**, **Title of the result**); **Fit the model**, **Cancel**, the progress of the chains.

## Step 6 · Results

Tabs **Overview** · **Factors** · **What tells what** · **How the variation splits** · **Covariates** · **Categories** (only with a covariate that has categories) · **Samples**.

Every chart of values (not shares, correlations or probabilities) has the choice **In SDs of each value** (default: every value in standard deviations of that value, so values of very different size can be read side by side) / **Own units, one panel each**.

* **Overview**: the verdict on convergence, cards **Samples**, **Posterior draws**, **Held-out fit (elpd)**, **Factors used (95 % of the shared variation)**; the charts **How many factors the data use** and **How much of each mode the factors carry**.
* **Factors**: under the loadings of a factor, for the EEM, the card **Is it one fluorophore? The trilinear share of the EEM loading** (the posterior of the share and the two profiles); one button per factor with its share; the switch **Band for the whole curve**; per mode the loading function (a fan for a spectrum; for an EEM the choice **Mean**, **2.5 %**, **97.5 %**, **Width**, **P(> 0)**, **Sure sign**; points with intervals for values).
* **What tells what**: **Predict** (one button per mode), the table of every choice of sources with the predicted share and its interval; for the chosen row the chart **Posterior of the predicted share** and the map of the share point by point; **Correlation of one point with a whole mode** (**Point of**, **Against**).
* **How the variation splits**: the table of shares (**Shared factors**, **Each sample's own**, **Noise**) and the maps.
* **Covariates**: the shares the covariates explain and the effect of each covariate on every mode; one button per number and one per category of every category covariate, the reference included (marked **(reference)**); for a category **Compared with** (any other category of the same covariate; the reference by default, or for the reference the next category).
* **Categories**: **Covariate**, **Category**, **As** (**The category alone** / **Its typical member**), **Show** (one button per mode), the switch **A new measurement (include the noise of the instrument)**; the text, the other covariates of the typical member, the expected function of the category, **Difference from the average sample**, a new member of the category and its uncertainty budget.
* **Samples**: the chart **Scores of the samples, with their uncertainty** and the table **The samples the model finds most surprising first**.

## Step 7 · Prediction and uncertainty

* **One sample**: **Sample**, **Predict**, **From** (the other modes, **Covariates**), the switches **Held out (as if the model had never seen this sample)** and **A new measurement (include the noise of the instrument)**; the prediction (fan / EEM statistic / intervals), the measured values, the map of the probability of being above the training mean, the uncertainty budget (**Population parameters**, **Factor scores of the sample**, **The sample’s own variation**, **Measurement noise**), when the sample has the target the card **What the prediction missed** (measured minus predicted: **Measured − predicted** / **In predictive SDs** for an EEM; the residual with the predictive intervals around zero for a spectrum or values) and **Factor scores of this sample, given what is known about it**.
* **Every choice of sources, held out**: **Target**, **Compare every choice of sources** / **Compare again**, the verdict, the table, **Calibration of the intervals (held out)** and the PIT histogram.

## Step 8 · Checks and use

* **Convergence**: the verdict, the table of the rotation-invariant quantities, **The log density of every chain**.
* **Basis**: per mode the verdict, the table and the map of the residual profile.
* **Use the model**: **Variables made by this model**, **Cluster the samples on their factor scores**, **Keep the model**, **Apply a saved model**, **Downloads**, **Methods paragraph**.

## Fixed phrases

Plain English from the server is shown as it comes (verdicts, notes, the text of a prediction, errors).
