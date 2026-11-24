# Multimodal latent factors (add-on) — a short note for users

**What it is.** A Bayesian model that describes several measurements of the same samples at once — fluorescence EEMs, absorbance spectra, and other numbers you measured — with
a few shared factors, optionally shifted by metadata (covariates).  Method: B. Ursino, I. Epifani, F. Trovò, *A Bayesian latent factor approach for multimodal functional data*
(software mblf-python).  Everything it shows carries its uncertainty.

**Install.** Settings → Add-ons → *Multimodal latent factors* → Install (nothing extra is downloaded).  The first time you fit a model, step 4 asks you to **prepare the Stan
engine**: about 450 MB (CmdStan and a C++ compiler) downloaded once into the add-on's own folder; FluoViz itself is not changed.  The first fit then compiles the model (about a
minute).

**What you can ask.**

| choose | to learn |
|---|---|
| one mode (only EEMs, or only spectra) | its patterns with credible bands, how many the data use |
| EEMs + absorbance | what they share; how much of every pixel the spectrum predicts; a sample's EEM predicted from its spectrum |
| + values (e.g. DOC) | the value predicted from the spectra with an honest interval; which wavelengths matter |
| + covariates (site, season …) | their effect on every wavelength and pixel, with bands |

**Reading the uncertainty.** Spectra are drawn as fans (50, 80, 95 % at every point; dashed = the whole curve with 95 %).  EEMs show one statistic at a time: the mean, the
limits, the width, `P(> 0)` or the mean where the sign is sure.  A prediction's uncertainty is split into four parts: the population parameters (more training samples shrink
it), the factor scores of the sample (more measurements of it shrink it), the sample's own variation (nothing shrinks it) and the measurement noise.

**The basis (step 3).** By default the building blocks are the Gaussian-process eigenfunctions of the method; **Functional PCA of your samples** uses the principal
components of your smoothed EEMs and spectra instead (fewer functions, a faster model, learned from the same samples).  Above 150 functions per mode the fit becomes
extremely slow: the page asks you to switch on **Allow more than 150 basis functions** first.

**What the model sees (step 3).** Every EEM and spectrum can be put on a **log scale** (log10 of the value plus a small offset: changes by a factor become equal steps,
the brightest peaks stop dominating), then **centred** (mean, median or not at all) and **scaled** (one number for the whole mode, Pareto, every point by its own spread, or not
at all).  In step 1 a value or a numeric covariate can be put on the log scale with its **log** box; a predicted value of a saved model is given back in its own units.  With two
values or more, **Principal components** describes them with fewer coefficients when they move together.

**Reading the results.** Values of very different size (a total fluorescence next to a pH) are drawn in standard deviations of each value, or each in its own units on its own
panel.  With a category among the covariates (site, season), **Compared with** shows the difference between ANY two categories, and the tab **Categories** shows each category as a
sample: **The category alone** (the other covariates at the average of all the samples: the pattern of the category itself) or **Its typical member** (the other covariates as
they are in that category), with the expected EEM or spectrum and what a new member would look like.  A prediction of a sample that has the target also shows **What the
prediction missed**: the measured minus the predicted, also in predictive standard deviations.

**Check before you trust.** Step 8: the chains must converge (R-hat ≤ 1.01, ESS ≥ 400, no divergences); the basis must leave only noise out.  Step 7: the held-out intervals must
contain about as many measured values as they claim (50 / 80 / 95 %).

**Gaps.** Samples without an absorbance spectrum or a laboratory value, or with regions cut out of their EEM, are used as they are; the model predicts what they lack.  A
*saved* model applied to new samples needs the EEM and the spectrum when it was fitted on both (values and covariates are optional).

The Help tab *Multimodal latent factors* explains every step in detail.

**Versions.** Multimodal latent factors 1.1.0 (1.1.0: a basis from a functional PCA of your samples; more than 150 basis functions on request; log scale, centring and scaling of what the model sees; principal
components of the values; values drawn per standard deviation; any two categories compared and categories as samples; the residual of a prediction); needs FluoViz 1.2.0 or newer (its searchable pickers that stay open while you add values and covariates).
