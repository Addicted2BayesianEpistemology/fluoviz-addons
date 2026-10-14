# The saved-model kind `autoencoder` and the router

*Written by A2.  The routes are in `api.md`; the numerics of training are `core/README.md`; the contract of saved models is FluoViz's `docs/modules/models.md`.  This note says what is frozen, what comes out, the thresholds and their defaults, and the caveats.*

```
fluoviz_addon_autoencoders/
    __init__.py            register(api): router, saved kind, result kind (and the job pool for the long jobs)
    kinds/                 numpy only (no PyTorch, no server): what a saved model is and does
        model.py             AEModel: the frozen network + how a sample becomes input (core.data.AEFrame, core.forward.Network); Evaluation
        autoencoder.py       the applier (apply / inspect / describe / options), judge() (status + reasons), publish() (variables), reference_for()
        freeze.py            freeze_autoencoder(p, result, title, note), pack_frozen / unpack_frozen (the `payload["frozen"]` format)
        stats.py             error limits, typicality of a code, KL, ELBO
        tools.py             traversals, interpolation, generation, sensitivity (decoder only)
    server/                the router (thin): request, prep, advice, fitjob, payload, views, scans, exports, methods, history, results, trainer
tests/                     (the API tests start the add-on with FluoViz's own dev-mode loader, `FLUOVIZ_ADDONS_DEV`; a hand-wired `AddonAPI` stand-in is only the fallback) test_kind_autoencoder.py, test_api_autoencoders.py, test_api_models.py, test_api_scans.py, test_api_training.py (+ kind_support.py, api_support.py)
```

## 1. What is frozen (`SavedModel.meta`)

| key | content |
|---|---|
| `kind_label` | "Autoencoder", "Variational autoencoder", with "Functional ..." in front when a basis is used |
| `model` | a plain tree (no class of the add-on, so a model file opens without the add-on; applying it needs the add-on): `spec` (the `ModelSpec`: family, sizes, activation, `vae` {prior, n_components, beta, sigma, n_eff}, blocks), `arrays` (`enc.W0 … enc.mu.W/b [enc.logvar.W/b] dec.W0 … dec.out.W/b [the linear shortcuts of `spec.skip`] [prior.pi/mu/logvar] in.mean in.scale out.weights`, float64), `frame` = for every variable `{name, kind (gp \| bspline \| cosine \| pixels), K, scale (K,), element}` |
| `model.frame[i].element` | the Functional PCA kind's own `FrozenElement`: the domain (axes, observed-pixel mask, quadrature weights), the **pre-treatment** (and the constant of `asinh`), the clean-up of the absorbance spectra, the **representation** (the Gaussian-process basis itself and its prior variances; B-splines / cosines by their orders), **the smoothing parameters the fit chose** (never estimated again for a new sample), the mean function.  `scale` is `d` of SPEC §3.2 (the coordinates are `xi = Z / d`) |
| `reference` | the training samples: `ids`, `heldout` (bool: the network did not train on them), `scores` (their codes), `scores_sd` (VAE), `error`, `rmse`, `typicality`, `d2`; `limits` (below), `code_stats` (mean / covariance of the codes: the distribution a new code is compared with), `n_eff` |
| `validation` | `n_samples`, `family`, `latent_dim`, `r2_val`, `r2_train`, `baseline` + `baseline_r2_val`, `n_heldout`, `mode`, `error_limit_95/99`, `n_eff`, `n_flagged_in_training` |
| `options`, `outputs`, `description`, `needs` | thresholds with help, the published variables, a plain-English paragraph, what the samples must provide (EEM and / or absorbance) |
| `recipe`, `source`, `addon` | the data recipe of the project (`capture_recipe`: corrections, scatter, regions, window, grid, mask), provenance, and `{"id": "autoencoders", "version": ...}` |

Nothing is frozen from the new samples: no centring, smoothing parameter, scale, loading or limit is estimated from them (rule 1 of the contract).  The frozen block of a result lives in the result itself (`result.payload["frozen"]`, arrays split out when the project is saved, no pickles, **no model object**): freezing needs no training and no recomputation of the basis, also after the project was closed and opened again.  The Gaussian-process basis is stored in the result (about 3 MB for the 64-sample example with GP bases of both variables; up to ~20 MB for a 150-term basis on a big grid); B-spline and cosine bases are rebuilt from their orders and cost nothing.

**Freezing refuses** (plain English, `FrozenError`): an outdated result ("... Use 'Train again with the same settings' first"), a result from another version of the add-on or a damaged one ("Use 'Train again with the same settings', then save it"), a recipe that cannot be captured (no absorbance loaded ...), and a network that does not reproduce its own codes: six training samples are read through the **recipe** (the corrections, scatter treatment, window and grid of the project) and encoded with the frozen numbers; the largest difference to the codes of the result must be below 1e-6 of the largest code (measured: ~1e-13).  The fit job makes the same check on all training samples before it saves anything (the coordinates read through the frozen description must equal those the network was trained on).

## 2. Applying a model to new samples

1. The recipe puts the samples in the state of the training samples (the router of FluoViz does this; `ApplyData`).
2. For each variable: values at the pixels of the domain → clean-up of the absorbance (as in training) → pre-treatment → **functional**: smoothed on the pixels the sample has with the stored smoothing, centred, moved to the L2 frame, divided by `scale`; **pixels**: `(y - mean) / s`, unmeasured pixels = 0 and ignored by the error.
3. The network standardises (`in.mean`, `in.scale`), encodes (a VAE's code is the posterior mean) and decodes.
4. **Reconstruction error** = the squared L2 distance between the sample's coordinates and their reconstruction, divided by the total variance of the variable, summed over the variables with equal weight (`core.forward.recon_error`): 0 = perfect, about 1 = as bad as the mean, `1 - error` = the share of the typical variation that is rebuilt.  A block with unmeasured pixels is scaled up to the whole block.

**Status.**  `unusable` (all outputs NaN, with the reason): a variable is missing (a model of fluorescence and absorbance needs both; a spectrum with fewer than 3 values is none), a pre-treatment is not defined, or fewer than `min_coverage` (default 0.8, by area, per variable) of a domain is measured.  `outside` ("Poorly described by the model"): reconstruction error above the limit, or (when switched on) an unusual code; the reasons are written out, naming the worse variable for two variables.  A sample that is `ok` but had gaps carries an informational reason.  Other statuses (`uncertain`, `unknown`) do not occur.

**Thresholds (options of the application) and their defaults**

| option | default | meaning |
|---|---|---|
| `limit` | `99` | which point of the errors of samples the network was NOT trained on is the limit: `95` (about 1 ordinary sample in 20 above it) or `99` (1 in 100) |
| `typicality` | `0.01` for a VAE, `0` (off) for a deterministic AE | a code whose typicality (p-value) is below this is `outside`.  Typicality = Hotelling's T² of the code against the training codes (mean and covariance frozen; for a VAE the covariance of the aggregate posterior = covariance of the posterior means + mean posterior variance), as an F-test for a NEW observation; with a mixture prior: the distance to the nearest cluster, chi-square |
| `min_coverage` | `0.8` | the router's: a sample with a smaller share of the domain measured is `unusable` |

**The limit** (`kinds/stats.error_limits`) is estimated from the errors of the held-out samples (hold-out: the validation samples; few samples: the out-of-fold errors of the k-fold networks), on the log scale: `exp(median + z x spread)`, spread = 1.4826 x MAD, never below 0.4; `z` = 1.645 / 2.326.  With fewer than 3 held-out errors the training errors are used and the limit is doubled (and the limit text says so).  Measured on synthetic data (eight networks, each judged on 420 fresh ordinary samples, limits from 20 held-out samples): 2.2 % of fresh samples above the 99 % limit (range 0 – 9 %) and 8.3 % above the 95 % limit (range 0 – 27 %): a limit from a handful of samples is rough, which the description of the model says.

**Published variables** (`ApplyOutput.variables`; the result of `/fit` publishes the same, plus `split`): `z1 … zd`, `recon_rmse` (RMSE to the observed pre-treated values, data units), `recon_r2` (`1 - error`), `typicality`, `flag` (words), `status`; a VAE: `z1_sd … ` (at most 12), `kl`, `elbo` (= `-[(N/2)(error/sigma² + log sigma²) + KL]`, the training loss, up to a constant; higher = explained better; the KL of a mixture prior is that to the nearest cluster plus `-log` of its weight), a mixture prior: `cluster`, `membership`.

**Payload** (generic blocks of `models.md`): `plot` (the first 8 codes, the training codes as the reference cloud), `diagnostics` (reconstruction error against the distance of the code from the training codes, with the limits), `limits`, `images` (`mean` = the decoding of the mean code, `directions` = change of the output per 1 SD of each code) / `curves` for the absorbance, `ex`, `em`, `mask`, `notes`.  `inspect(index)`: `measured`, `reconstruction`, `residual`, `mean` images (and curves), a sentence, the facts of the sample.

**As the first link of a chain.**  A classifier or clustering built on `res:autoencoder-N:z1…` embeds this model as a part (named `autoencoder-N`); applying it encodes the new samples first, a sample the network describes poorly makes the classifier's status `outside` ("The Functional autoencoder model does not describe this sample well ...").  Tested through the real models router with LDA: the discriminant scores of the page are reproduced from the file, in another project under other sample names the same classes come out, a planted sample is flagged.

## 3. Caveats (also in `describe()["caveats"]`)

* The limit comes from few samples (the same few that are held out): rough, see above.  The page's own numbers (`recon.limit`) are the same as the saved model's at default thresholds, so a result and its saved model agree.
* A nonlinear network rebuilds well what looks like its training samples: a sample measured under other conditions is flagged too (the reason says "or it was measured differently").
* A coarser measurement grid is regridded by the recipe (bilinear); codes stay close (tests: correlation > 0.9 with the page's codes for a grid with half the wavelengths) but are not identical.
* Gaps: a functional model fills them with its smooth shape (less certain codes), a pixel model ignores them; both say so.
* The saved model is the *final* network.  With fewer than 30 samples the honest numbers come from the k-fold networks, the final network is trained on all samples (as `core/README.md` describes), so the training samples are in-sample for it.

## 4. The router (what is not obvious)

* **One truth for numbers and words.**  The anomaly limit and the typicality of a code (new samples) are `kinds/stats.py`, used by the saved kind, by the page's flags and by `core/diagnostics` alike.  Every other threshold and verdict of the result's checks (overfitting, baseline comparison, health of the codes, smoothness, calibration, stability, scans) is `core/diagnostics.py`: `server/payload.py` only calls those functions and keeps no threshold of its own.

* **The server never imports PyTorch.**  `fit` and the scans run in `core.worker.run_in_worker` (a `spawn` process; cancel = terminate; the result comes back through a file).  `kinds/` and `server/` are numpy; `core.train` is only imported lazily (its module has no top-level torch import).  A test starts a fresh server process, fits, looks at results, saves and applies a model, and checks `sys.modules`.
* **Job pool.**  `register(api)` adds `autoencoder` and `autoencoder-scan` to `fluoviz.server.jobs.HEAVY_KINDS` (a plain set) so fits do not take the places of interactive jobs.
* **History.**  The worker appends JSON lines to `runtime/tmp/autoencoders-<token>.jsonl`; `GET /fit/{job}/history` reads the file (`server/history.py`), after the job the rows stay in memory (12 jobs) and the file is deleted.
* **The prepared data is cached** (the last request on the current data), so `defaults?advise=1`, `check` and `fit` in a row prepare the basis and smoothing once.
* **Results.**  `save_result(kind "autoencoder", params = the replayable request, data_sig = the signature when the fit started)`; if the data changed while training, nothing is saved.  `replace` trains in place (same id, same title unless one is sent).  `r.obj` caches the numpy network after the first use; it is never saved.
* **Stale results** answer from what they store; endpoints that need the measured data refuse in plain words (see `api.md`).  The network is never trained silently.
* **Where the numbers come from.**  The page's codes are computed by the same code a saved model uses (`AEModel.evaluate` on the project's data), not taken from the worker; the fit checks they equal the worker's (1e-6) before saving.
