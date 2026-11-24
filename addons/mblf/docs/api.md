# Multimodal latent factors add-on — the API (`/api/addon/mblf`)

Errors are plain-English 400 / 404 (`{"error": {"message"}}`).  Jobs return `{job_id}` (FluoViz's job runner; kinds `mblf-engine`, `mblf-check`, `mblf-fit`, `mblf-compare`).

## 1. The request of a fit / check

```json
{"input": {"variables": ["fluorescence", "absorbance"],
           "fluorescence": {"transform": "none", "min_coverage": 0.5, "basis": {"kind": "gp", "kernel": "matern52", "structure": "separable", "lengths": "auto", "var_fraction": 0.99, "n_basis": null, "k_max": 40}},
           "absorbance":   {"transform": "none", "processed": true, "basis": {"kind": "gp", "lengths": "auto", "var_fraction": 0.99, "k_max": 25}},
           "values": ["meta:DOC"], "covariates": ["meta:site"], "partial": true, "large_basis": false},
 "model": {"K": 5, "estimation": "bayes", "priors": {"a_psi": 2, "b_psi": 2, "a_tau": 2, "b_tau": 2, "a1": 2, "a2": 2, "b_lambdavar": 1, "nu": 3, "sigma_regr_coeffs": 10}},
 "sampler": {"preset": "standard" | "chains": 4, "warmup": 1000, "draws": 1000, "adapt_delta": 0.8, "max_treedepth": 12, "keep": 1000},
 "ml": {"starts": 8}, "seed": 1, "allow_few_samples": false, "title": "", "replace": null}
```
Per function also `"log": false, "log_offset": null` (the model sees `log10(value + offset)`; null = 1 % of the 99th percentile of the values; a value at or below `-offset` counts as not measured), `"centre": "mean" | "median" | "none"`, `"scaling": "global" | "pointwise" | "pareto" | "none"`.  At the top of `input`: `"log_keys": [...]` (values and numeric covariates that enter as their `log10`; a value of zero or below is then missing, a covariate drops the sample) and `"values_basis": {"kind": "identity" | "pca", "var_fraction": 0.95, "n_basis": null}` (the principal components of the correlations of the standardised values), and `"var_prep": {key: {"centre": "mean" | "median" | "none", "scaling": "sd" | "mad" | "none"}}` (how a value or a covariate is centred and scaled; a category takes `centre` only: `mean` = centred indicators, `none` = 0 / 1; default the mean and the SD).  Per function also `"basis_from": "gp" | "fpca" | "fitted"` (`fitted`: the eigenfunctions of the functional PCA named by `input.fpca_result`, with its domain, pre-treatment and absorbance clean-up; no log); the Gaussian-process `structure` may be `"tensor"` (FluoViz 1.3: products of one-dimensional bases).  Per function also `"basis_from": "gp" | "fpca"` (the Gaussian-process prior, default, or a functional PCA of the training samples).  The basis sizes (`k_max`, `n_basis`) are limited to 400 by FluoViz's check; above 150 the request is refused (plain words naming the field and the mode) unless `large_basis` is true.  Presets: quick 2 × (400 + 400), standard 4 × (1000 + 1000), thorough 4 × (2000 + 2000).  The basis must be `gp`; `var_fraction` 0.5–0.9999; `k_max` 2–400 (above 150 only with `large_basis`); `K` 1–12; at least 10
samples (5 with `allow_few_samples`).

## 2. Routes

| route | what |
|---|---|
| `GET /engine[?size=1]` | `{state: ready \| missing \| unavailable, message, source: private \| given, size_mb, download_mb, prefix}` |
| `POST /engine/prepare` (job) | prepares the private engine; result = the engine |
| `DELETE /engine` | removes it |
| `GET /defaults` | `{n_samples, fluorescence: {available, n}, absorbance: {…}, meta: [{key, label, kind, n, levels}], engine, presets, priors, request}` |
| `POST /check` (job) | `Prepared.summary()` + `n_parameters`, `words`, `problem` (nothing fitted) |
| `POST /fit` (job) | `{result, rid}` |
| `GET /{rid}` | `{result, payload}` (section 3) |
| `GET /{rid}/sample/{sid}?target=&sources=auto\|none\|a,b&covariates=1&noisy=0&held_out=1` | the prediction (section 4) |
| `GET /{rid}/predictable?target=&sources=&covariates=` | the predictable fraction per point (latent, measured: mean, lower, upper), whole-mode intervals, posterior histogram |
| `GET /{rid}/contrast?key=&a=&b=` | the difference of every mode between categories `a` and `b` of the covariate `key` (any two, the reference included), everything else equal: `{key, a, b, n_a, n_b, modes: [{mode, …function summary}]}` |
| `GET /{rid}/category?key=&level=&how=pure\|typical&target=&noisy=0` | a category as a sample: `pure` = its indicator 1, the others 0, every other covariate at the average of the training samples; `typical` = the average design of its training samples.  `{label, level, how, target, n, noisy, text, design [{label, kind, value}] (numbers in their own units, categories as shares), expected (function summary of the mean function: population uncertainty only), shift (expected minus the training mean), member {mean, quantiles, probs, band} (a new member), budget, budget_total, n_draws}` |
| `GET /{rid}/correlation?a=&b=&x=&y=&j=` | correlation of the point nearest (x = excitation or wavelength, y = emission; j = variable) of mode a with every point of b: mean, 90 % limits, `p_pos`, `lfsr` |
| `POST /{rid}/compare` (job) `{targets?, max_points?}` | per target: rows (one per source set: `key, sources, covariates, elpd, se, gain, gain_se, vs_best, vs_best_se, n, coverage {0.5, 0.8, 0.95}, pit_hist, rmse, mean_width80, mean_sd, pointwise`), `best`, `one_se`; stored in `payload.compare` |
| `GET /{rid}/samples` | ids, presence, scores, their SDs, held-out log density per value, Pareto k |
| `GET /{rid}/methods` | `{text, references}` |
| `GET /{rid}/export?what=scores\|matrix\|compare\|loadings` | CSV |

## 3. The payload of a result

`estimation, K, n_draws, n_view_draws, chains`; `modes` (per mode: name, label, kind, n_points, n_basis, captured, basis, units, n_samples, `mean`; surfaces `ex, em, mask`
(uint8, 1 = outside); curves `x`; values `labels` (with `log10 ` for a value on the log scale), `columns, sd, log_cols`; every mode `log` (`{offset, auto}` or null) and `prep` (`{centre, scaling}`)); `data` (the prepared summary); `diagnostics` (engine numbers, `table` per quantity, `verdict {level, text,
items}`, `lp` per chain); `factors` (per factor: `share {mean, lower, upper}`, per mode a function summary); `dimension`; `variance` (per mode: `variance` maps, `share` maps,
`whole` intervals); `matrix` (every target × source set: latent / measured whole-mode intervals, `hist`); `predictable` (per target, all other modes + covariates, per point);
`covariates` (`columns`, `effects`, `shares`) or null; `loo` (`elpd, se, p_loo, pointwise, pareto_k, per_value, verdict`); `samples` (`ids, present, scores, scores_sd, n_obs`);
`basis_check` (per mode: `level, text, rows, ratio, outside`); `notes`; `compare`.  For an EEM every factor's mode entry also has `trilinear` `{mean, lower, upper,
share_of_mean, em, ex, em_axis, ex_axis}`: the posterior (90 % interval) of the trilinear share of its loading and the profiles of the posterior-mean loading.

A **function summary**: `mean`, `quantiles` (7 × …; probs 2.5 … 97.5 %), `probs`, `sd`, `p_pos`, `lfsr`, `band {level, multiplier, lower, upper}`.  Arrays are in the browser's
form: surfaces `(n_ex, n_em)` float32 with NaN outside the domain, curves on the full wavelength grid, values as they are.

## 4. A prediction

`{id, target, sources, covariates, noisy, held_out, pareto_k, text, observed, inside95, mean, quantiles, probs, band, budget {parameters, scores, residual, noise} (variances per
point, data units²), budget_total (domain-weighted shares), residual (null when the sample does not have the target; else `{value, z, quantiles, probs, rms, rms_z, beyond2}`: measured minus predicted, divided by the predictive SD, the predictive quantiles minus the mean, the domain-weighted root mean squares and the share beyond ±2 SD), p_above (P(above the training mean) per point), scores {mean, sd, lower, upper} (aligned), sources_data, n_draws}`.
