# Multimodal latent factors add-on — the API (`/api/addon/mblf`)

Errors are plain-English 400 / 404 (`{"error": {"message"}}`).  Jobs return `{job_id}` (FluoViz's job runner; kinds `mblf-engine`, `mblf-check`, `mblf-fit`, `mblf-compare`).

## 1. The request of a fit / check

```json
{"input": {"variables": ["fluorescence", "absorbance"],
           "fluorescence": {"transform": "none", "min_coverage": 0.5, "basis": {"kind": "gp", "kernel": "matern52", "structure": "separable", "lengths": "auto", "var_fraction": 0.99, "n_basis": null, "k_max": 40}},
           "absorbance":   {"transform": "none", "processed": true, "basis": {"kind": "gp", "lengths": "auto", "var_fraction": 0.99, "k_max": 25}},
           "values": ["meta:DOC"], "covariates": ["meta:site"], "partial": true},
 "model": {"K": 5, "estimation": "bayes", "priors": {"a_psi": 2, "b_psi": 2, "a_tau": 2, "b_tau": 2, "a1": 2, "a2": 2, "b_lambdavar": 1, "nu": 3, "sigma_regr_coeffs": 10}},
 "sampler": {"preset": "standard" | "chains": 4, "warmup": 1000, "draws": 1000, "adapt_delta": 0.8, "max_treedepth": 12, "keep": 1000},
 "ml": {"starts": 8}, "seed": 1, "allow_few_samples": false, "title": "", "replace": null}
```
Presets: quick 2 × (400 + 400), standard 4 × (1000 + 1000), thorough 4 × (2000 + 2000).  The basis must be `gp`; `var_fraction` 0.5–0.9999; `k_max` 3–150; `K` 1–12; at least 10
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
| `GET /{rid}/correlation?a=&b=&x=&y=&j=` | correlation of the point nearest (x = excitation or wavelength, y = emission; j = variable) of mode a with every point of b: mean, 90 % limits, `p_pos`, `lfsr` |
| `POST /{rid}/compare` (job) `{targets?, max_points?}` | per target: rows (one per source set: `key, sources, covariates, elpd, se, gain, gain_se, vs_best, vs_best_se, n, coverage {0.5, 0.8, 0.95}, pit_hist, rmse, mean_width80, mean_sd, pointwise`), `best`, `one_se`; stored in `payload.compare` |
| `GET /{rid}/samples` | ids, presence, scores, their SDs, held-out log density per value, Pareto k |
| `GET /{rid}/methods` | `{text, references}` |
| `GET /{rid}/export?what=scores\|matrix\|compare\|loadings` | CSV |

## 3. The payload of a result

`estimation, K, n_draws, n_view_draws, chains`; `modes` (per mode: name, label, kind, n_points, n_basis, captured, basis, units, n_samples, `mean`; surfaces `ex, em, mask`
(uint8, 1 = outside); curves `x`; values `labels, columns, sd`); `data` (the prepared summary); `diagnostics` (engine numbers, `table` per quantity, `verdict {level, text,
items}`, `lp` per chain); `factors` (per factor: `share {mean, lower, upper}`, per mode a function summary); `dimension`; `variance` (per mode: `variance` maps, `share` maps,
`whole` intervals); `matrix` (every target × source set: latent / measured whole-mode intervals, `hist`); `predictable` (per target, all other modes + covariates, per point);
`covariates` (`columns`, `effects`, `shares`) or null; `loo` (`elpd, se, p_loo, pointwise, pareto_k, per_value, verdict`); `samples` (`ids, present, scores, scores_sd, n_obs`);
`basis_check` (per mode: `level, text, rows, ratio, outside`); `notes`; `compare`.

A **function summary**: `mean`, `quantiles` (7 × …; probs 2.5 … 97.5 %), `probs`, `sd`, `p_pos`, `lfsr`, `band {level, multiplier, lower, upper}`.  Arrays are in the browser's
form: surfaces `(n_ex, n_em)` float32 with NaN outside the domain, curves on the full wavelength grid, values as they are.

## 4. A prediction

`{id, target, sources, covariates, noisy, held_out, pareto_k, text, observed, inside95, mean, quantiles, probs, band, budget {parameters, scores, residual, noise} (variances per
point, data units²), budget_total (domain-weighted shares), p_above (P(above the training mean) per point), scores {mean, sd, lower, upper} (aligned), sources_data, n_draws}`.
