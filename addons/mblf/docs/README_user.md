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
it), the factor scores of the sample (more measurements of it shrink it), the sample's own residual (nothing shrinks it) and the measurement noise.

**Check before you trust.** Step 8: the chains must converge (R-hat ≤ 1.01, ESS ≥ 400, no divergences); the basis must leave only noise out.  Step 7: the held-out intervals must
contain about as many measured values as they claim (50 / 80 / 95 %).

**Gaps.** Samples without an absorbance spectrum or a laboratory value, or with regions cut out of their EEM, are used as they are; the model predicts what they lack.  A
*saved* model applied to new samples needs the EEM and the spectrum when it was fitted on both (values and covariates are optional).

The Help tab *Multimodal latent factors* explains every step in detail.
