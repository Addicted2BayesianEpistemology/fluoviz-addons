# Add-on "Multimodal latent factors" (`mblf`) — specification

*The third model add-on.  It brings the Bayesian multimodal latent factor model of Ursino, Epifani and Trovò (mblf-python 1.0.0, vendored in `fluoviz_addon_mblf/vendor/`, MIT) into
FluoViz.  Framework: FluoViz `docs/modules/addons.md`.  Model of the code: the self-organising map add-on; shared code: the `common` add-on.  User note: `docs/README_user.md`;
routes: `docs/api.md`; interface contract: `docs/ui.md`.*

## 1. What the person gets

A page **Multimodal latent factors (add-on)** in *Experimental methods*: an eight-step pipeline (what to model, domain, basis, model and sampler, fit, results, prediction and
uncertainty, checks and use).  It fits the model to any combination of **modes** — the fluorescence (EEM, a surface), the absorbance spectrum (a curve), a vector of other
**values** (laboratory values, indices, scores of other results) — with optional **covariates** (metadata, numbers or categories) on the means of the factor scores.  Samples that
lack part of a mode, or a whole mode, are used as they are (the likelihood is exact on the observed points).  Every result carries its posterior uncertainty; any mode of any
sample can be predicted from any choice of sources with the predictive variance split into four parts; every choice of sources is compared held out (PSIS-LOO) with the coverage
of its intervals.  Results publish per-sample variables and can be saved as a model (`mblf` kind) for new samples.

## 2. Modes (`core/modes.py`)

* Functions: FluoViz's `fluorescence_variable` / `absorbance_variable` (pre-treatment, domain, min coverage, cleaned absorbance).  Standardised: minus the mean of the training
  samples at every point, divided by the root mean square of the centred samples over the domain (quadrature weights; each sample averaged over the part it has).
* Basis: FluoViz's Gaussian-process Karhunen–Loève basis (`resolve_basis`, `build_representation`: variogram / ML / given length-scales, separable kernels, Lanczos / Nyström),
  columns multiplied by `sqrt(lam_k / trace)` (the paper's `B = W^-1/2 V Xi^1/2`), truncated at the share of the prior variance asked (default **99 %**; the discarded share is the
  expected relative truncation error), at most 40 (surface) / 25 (curve) functions by default.
* **Functional PCA basis** (1.1.0, `basis_from: "fpca"`, per function; the page sets both): the samples that have the mode are smoothed with that GP basis (ML smoothing,
  0.999 of the prior variance, gaps estimated: FluoViz's `prepare_element`), `fit_fpca` gives the eigenfunctions of the smoothed functions; the basis keeps the fewest that hold
  the share asked (or exactly `n_basis`; at most `k_max`, at most N-1), each times `sqrt(eigenvalue) / scale`.  Data-driven: fewer functions for the same variation, but estimated
  from the training samples (bands slightly optimistic; the paper's truncation theorem does not apply).  The saved model stores the matrix, as for the GP basis.
* **A fitted functional PCA** (1.2.0, `basis_from: "fitted"`, `input.fpca_result`): the function is the Variable of that analysis (its domain, pre-treatment and absorbance
  clean-up: the page's settings of steps 2-3 are not used for it), the basis its eigenfunctions (the univariate analysis of the variable when the fit was multivariate), each
  times `sqrt(eigenvalue)` divided by the scale of the data; the share / number of terms / `k_max` choose how many; no log (refused).  The saved model reads new samples with the
  analysis' pre-treatment (its `FrozenElement`).
* **Tensor-product bases** (FluoViz 1.3.0): the Gaussian-process `structure: "tensor"` (products of one-dimensional KL bases on the whole rectangle) is accepted like the others.
* **Trilinearity, checked not imposed** (`structure.trilinear_loadings`): for every factor's EEM loading, FluoViz's `rank_one` share at up to 200 draws (posterior mean and 90 %
  interval) and the emission / excitation profiles of the posterior-mean loading (payload `factors[k].modes[...].trilinear`).
* **Basis size.**  FluoViz's check limits `k_max` / `n_basis` to 400.  Above 150 the request is refused (the field, the mode and the reason in words) unless `input.large_basis`
  is true; the page shows it as a step-3 problem with the switch *Allow more than 150 basis functions* and a warning once it is on.
* **What the model sees** (per function, `input.<function>`): `log` (`log10(value + c)` after FluoViz's pre-treatment; `c` given, or 1 % of the 99th percentile of the
  absolute values; a value `<= -c` counts as not measured), `centre` (`mean` / `median` of the training samples at every point, or `none`), `scaling` (`global`: one number, the
  RMS of the centred samples over the domain; `pointwise`: every point's SD, floored at 1e-3 of the global RMS; `pareto`: its square root; both rescaled so the standardised data
  have RMS one; `none`: the data units).  `Mode.mean` / `Mode.scale` (M,) hold the centre and the scale, `Mode.log` the offset: `Mode.forward` applies the log to new samples
  (saved models), `Mode.backward` undoes it for values and quantiles.  The basis length-scales are estimated on the logged data (divided by the pointwise scale when there is
  one); a functional PCA basis of data not centred on their mean also holds the mean of the samples.
* Values: each variable (its `log10` when it is in `input.log_keys`; zero or below = missing) centred and divided by its SD; basis the identity, or (`values_basis.kind = "pca"`)
  the leading eigenvectors of their pairwise-complete correlation matrix times the square roots of the eigenvalues (the share asked, or `n_basis`).  Covariates: numbers
  standardised (their `log10` when in `log_keys`; zero or below drops the sample); categories as centred indicators of every level but the most frequent (the reference); at most
  12 levels; samples with a missing covariate are left out (noted).
* **How values and covariates enter** (`input.var_prep`, per key): `centre` mean (default) / median / none and `scaling` sd (default) / mad (1.4826 × the median absolute
  deviation) / none; a category's indicators centred (`mean`) or 0 / 1 (`none`).  The covariate columns keep the centre and the scale (`mean`, `sd`), so new samples are read the
  same way.  The model has no intercept of its own (the data are centred on their mean), so a covariate not centred on its mean ties the data mean to the samples at the centre
  chosen; the algebra stays exact for any design: `Covariates.cov()` is the covariance about the mean design `xbar`, and with the covariates unknown the scores have the prior
  `N(beta xbar, I + beta Cov(x) beta')` (`predict.score_posterior`, `loo.loglik`; tested against the dense Gaussian density).  "The category alone" puts the other covariates at
  `xbar`.  A values PCA basis uses the covariance of the values as the model sees them (their correlations by default) and, when a value is not centred on its mean, also
  holds their mean.
* `partial` (default true): the samples are the union of those that have any mode; otherwise the intersection.

## 3. The Stan engine (`core/engine.py`, `engine_job.py`)

**Decision.** conda-forge's `cmdstan` pins its compiler (gcc 15 on Linux, clang 21 on macOS, gxx 15 on Windows); installing it into FluoViz's environment would downgrade the
gcc 16 toolchain FluoViz has (a dry run changes 10 packages), which the framework forbids.  So the manifest asks for **no conda package** and the add-on keeps a **private
environment** `<data_dir>/engine/env` (`python=3.12 cmdstan>=2.36 cmdstanpy>=1.2 xarray numpy scipy pandas`), created once from the page with FluoViz's micromamba
(`fluoviz.addons.conda.mamba_env`: everything stays in `runtime/`).  A fit is a job in that environment: `micromamba run -p <env> python -u engine_job.py <job dir>` (activation
sets `CMDSTAN` and the compiler); input/output are `.json` + `.npz`; progress lines `@@ f text` and CmdStan's `Chain [c] Iteration: i / n`; Cancel kills the process tree (psutil).
The compiled programs are cached in `<engine>/stan-cache` (`MBLF_CACHE_DIR`).  CmdStan cannot compile in a path with a space: the engine then says so (`FLUOVIZ_MBLF_ENGINE_DIR`
moves it).  `FLUOVIZ_MBLF_ENGINE=<prefix>` uses an existing environment (development, tests).  The FluoViz server never imports cmdstanpy or xarray (tested).

## 4. The posterior (`core/posterior.py`, `core/structure.py`)

* Draws kept: at most `keep` (default 1000), thinned evenly **within** every chain (chain structure kept for R-hat and PSIS); the page uses at most 400 (`VIEW_DRAWS`).
* Alignment: per draw, Varimax-RSP (`mblf.align.varimax_rsp`) of the stacked loadings in the **L2 frame of the loading functions** (`G_d^1/2 Lambda_d`, `G_d = B_d' W_d B_d`,
  modes standardised so each counts equally); factors ordered by mean strength; sign: largest entry of the mean aligned loading positive.  `T` (`S, K, K`) is stored; aligned
  scores are `T' eta`.  Rotation-invariant quantities never use it.
* Function summaries: posterior mean, the quantiles 2.5 / 10 / 25 / 50 / 75 / 90 / 97.5 % (the fan), a **simultaneous** 95 % band (sup-t: the 95 % quantile of the largest
  standardised deviation of whole draws), `p_pos = P(> 0)` and the local false sign rate `lfsr = min(P(>0), P(<0))`.
* Variance split at every point: factors (`B Lambda Psi Lambda' B'`, `Psi = I + beta Cov(x) beta'`), of which covariates, each sample's own (`B diag(sig2) B'`), noise
  (`psi2`); whole-mode shares weighted by the quadrature weights.
* **Predictable fraction** `R2` of a target from fully measured sources (and / or the covariates): `[B_T Lambda_T (Psi - V) Lambda_T' B_T']_mm / total(m)`, `V` the posterior
  covariance of the scores given the sources (Gaussian: independent of the values); per point and for the whole mode, latent and measured, at every draw (posterior histogram).
  The matrix of every target × every subset of the other modes × with / without covariates is stored with the result.
* Correlation of one point with every point of a mode (population, noise excluded) per draw.  Covariate effects `scale * B Lambda beta_l` per design column and mode.
* Effective dimension: `mblf.secondary.effective_dimension` (point weights = quadrature weights), with the posterior probability of every number of directions.

## 5. Predictions and the uncertainty budget (`core/predict.py`)

At every draw: `eta | sources ~ N(eta_hat, V)` (precision `Psi^-1 + sum_d G_d' W_d^-1 G_d`, `G_d = R_d Lambda_d`, `W_d = R_d diag(sig2_d) R_d' + psi2_d I`, `B_d,obs = U_d R_d`,
samples with the same observed points share the algebra); covariates known: prior `N(beta x, I)`; unknown or absent: `N(0, Psi)`.  The target: mean `B Lambda eta_hat`, variance
`B Lambda V Lambda' B'` (scores) `+ B diag(sig2) B'` (residual) `[+ psi2]` (noise, for a new measurement).  Mixture over the draws (modular rule; optional weights = PSIS weights
of a held-out training sample).  **Budget** at every point = parameters (weighted variance of the conditional means) + scores + residual + noise, exactly the predictive
variance; also its domain-weighted shares.  Pointwise quantiles of the mixture by bisection of its CDF inside mean ± 7 SD (Chebyshev); a simultaneous band from draws of whole
functions; `P(above the training mean)`; aligned scores with mixture intervals.  Checked against brute-force Gaussian conditioning in `tests/test_core_numerics.py`.

**The residual** of a prediction of a sample that has the target: measured minus the predicted mean, the same divided by the predictive SD (`sqrt` of the budget), the
predictive quantiles minus the mean (where the residual should lie), domain-weighted RMS and the share beyond ±2 SD.

**Categories** (`views.contrast_route`, `views.category_route`; `Covariates.levels / level_rows / pure / typical`): the difference between any two levels `a`, `b` of a category is
`scale B Lambda beta (x_a - x_b)` per draw (`x_l` the design of level `l` with the other covariates fixed; the reference is a level like the others).  A level as a sample: `pure`
= its indicator 1, the other levels 0, every other column 0 (the average of the training samples); `typical` = the mean design row of its training samples (read back from the
centred indicators).  For the design `x`: the expected function `mean + scale B Lambda beta x` per draw (population uncertainty only), its shift from the training mean, and the
prediction of a new member (`predict` with no sources and the covariates known: scores, own part, noise).

## 6. Held-out checks (`core/loo.py`, `core/checks.py`)

* Marginal likelihood of any subset of modes of a sample (scores and residual coefficients integrated out), covariates known or not (`Lambda chol(Psi)` trick); equal to mblf's
  `marginal_loglik` and to the dense Gaussian density (tests).
* PSIS weights leaving out each sample (all its modes), `r_eff` from the chains; PSIS-LOO of the model; for a target and a source set the held-out conditional log density
  `log sum_s w_s(n) p(y_T | y_S, draw s)` with `p(y_T | y_S) = p(y_T, y_S) / p(y_S)`; paired gains vs "nothing" and vs the best with SEs; the one-SE choice (the simplest set
  within one paired SE of the best).
* Calibration of a source set: held-out mixture predictive of every measured target value (at most `max_points` per sample): PIT, coverage of 50 / 80 / 95 %, RMSE, mean
  80 % width (mixture quantiles on at most 40 points per sample), PIT histogram.
* Convergence (engine): per rotation-invariant quantity of mblf (`identified_quantities`) the largest R-hat and smallest bulk / tail ESS; divergences, tree depth, E-BFMI; verdict
  ok / warn / bad with what to do (thresholds 1.01, 400).  ML: starts at the optimum, variances at the floor.
* Basis adequacy: `mblf.secondary.projection_diagnostics` + `projection_reference` per functional mode (simulations capped so they stay below ~160 MB).

## 7. Result, variables, saved model

* Result kind `mblf` (`mblf-N`); payload = everything of sections 4–6 in browser form (`server/payload.py`), plus `frozen` (common format).
* Variables: `mblf_f<k>`, `mblf_f<k>_sd` (aligned posterior mean / SD of the scores from everything the sample has, Rao-Blackwellised), `mblf_loo` (held-out log density per
  value), `mblf_pareto_k`.
* Saved kind `mblf` (`kinds/`): the tree holds the modes (with their `FrozenElement` for reading new samples), covariates, draws, the training data (float32) and the reference
  (training scores; per set of modes the 5 % / 1 % / median of the held-out log density per value).  Applying: reads what a sample has, covariates and values from the metadata
  when present (`meta:` keys; listed as optional, not as recipe needs), gives scores ± SD, predicted values with 95 % intervals (`mblf_pred_<value>`, `_lo`, `_hi`), `mblf_surprise`,
  status `outside` below the 1 % (or 5 %) point.  `inspect` shows the predicted EEM / spectrum with limits, SD and budget.  Freezing re-reads training samples through the recipe:
  scores must agree to 1e-5 (also checked at fit time).
* **Known limit (framework):** FluoViz's data recipe requires every function a model reads for every applied sample, so a saved model of EEM + absorbance cannot be applied to
  samples without a spectrum (the page of a result handles them fully).

## 8. Decisions where a choice was made

* Smoothing is never chosen: the model separates shared variation, each sample's own part and noise.
* "Values" are modes (explained and predicted); covariates explain only.  A variable cannot be both.
* The training data are stored in the result (float32), so every view and the comparison job work after the project is reopened, without the project data.
* The fit is a FluoViz job (heavy pool); the comparison of sources is a separate job (it costs seconds to a minute) whose result is stored in the payload.
* The worker of the autoencoders add-on was not reused: this add-on runs a different interpreter (the engine's), so it has its own small process runner (`core/engine.py`).

## 9. Tests

`tests/` (FluoViz's `scripts/run_tests.sh`): numerics against brute-force algebra (no engine), the vendored copy, the engine's words, the saved-model side, the Help contract, the
manifest and build, and — with `FLUOVIZ_MBLF_ENGINE` — the API end to end with a real Stan fit.  `tests/e2e/flow_mblf.py`: the whole page in Chrome (steps, every tab, a fit,
a prediction, the comparison, the guide, 600 px, no console errors).
