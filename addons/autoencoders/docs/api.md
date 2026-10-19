# Autoencoders add-on: HTTP API

*Written by A2 (kind + router) for A3 (the page) and for the Help guide.  Binding with `SPEC.md` §8.  It is kept accurate while the code is built; "TBC" marks a detail that depends on a part another agent is still writing.*

All routes live under **`/api/addon/autoencoders`** (the framework refuses any other prefix).  The add-on also uses, unchanged, the generic FluoViz routes: `/api/fpca/*` (steps 1–4: `data`, `gp/spectrum`, `gp/scan`, `gp/estimate`, `basis/check`), `/api/jobs/{id}` (poll a job), `/api/variables` (the published variables), `/api/models*` (save / apply / chain the model), `/api/results*` (list, rename, delete).

## 0. Conventions

* **JSON encoding** is FluoViz's: `NaN` / `±inf` → `null`; a numpy array with more than 1500 numbers → `{"$nd": "f4" | "i4" | "u1", "shape": [...], "b64": "..."}` (little endian, C order; turn it into a typed array), smaller arrays are nested lists.
* **Orientation.**  Every image is in *browser orientation*: one EEM is `(n_ex, n_em)` (rows = excitation, columns = emission), a stack is `(k, n_ex, n_em)`.  `ex` and `em` are the wavelength axes (nm) of the model's grid; `mask` is `uint8 (n_ex, n_em)`, **1 = this pixel is not in the model** (scatter, regions, below the Rayleigh line).  Absorbance and other curves are plain arrays on `abs_wl` (or the `x` of the block), `null` outside the model's domain.  Payload keys documented below as "`(…)` module orientation" exist only inside the saved result; the `GET` answers always turn them.  This is the orientation of the DATA: how the pictures are drawn (emission across and excitation up, or the other way round) is the person's setting in FluoViz (Settings → Appearance → Axes of fluorescence maps) and is decided in the browser; downloads are never turned.
* **Errors**: `{"error": {"message": "<plain English>", "kind": "user" | "bug", "detail": ""}}`.  400 = something the person can fix (the message says what to do), 404 = unknown result / sample, 500 = a bug.  A job that fails ends with `status: "error"` and the same message in `error`.
* **Jobs** (`fit`, `check`, `scan`): the POST answers `{"job_id": "..."}` (after the request was validated: a bad setting is an immediate 400, never a failed job); poll `GET /api/jobs/{id}` (`status` queued | running | done | error | cancelled, `progress` 0–1, `message`, `result` when done); cancel with `POST /api/jobs/{id}/cancel`.  A cancelled fit saves nothing.
* **Staleness.**  Like every result, `autoencoder-N` records the data signature it was trained on.  When the data, corrections, scatter treatment, focus window or sample selection change, `result.stale` lists the changed parts (`raw`, `corrections`, `masks`, `window`, `samples`).  *The network is never trained again silently.*  A stale result can still be looked at (everything stored in it: codes, training curves, quality, traversals, interpolation between codes, generation, sensitivity, exports of codes); endpoints that need the measured data (`sample`, `anomalies`, the observed part of `export?what=reconstruction`) answer 400 *"The data or the corrections changed since this network was trained (the focus window changed). Use 'Train again with the same settings' to train it on the current data."*  **Train again** = `POST /fit` with the stored `params` plus `"replace": "<rid>"` (the same id and title, the new network).  An outdated result cannot be saved as a model (the generic `/api/models` says so).
* **Plain English** in every `message`, `words`, `help`, `reasons`, `notes`, `warnings` and `text` field; keys are `snake_case`.
* **Torch**: the server never imports PyTorch.  Training runs in a separate process.  When the library is not installed, `defaults` says so (`torch.available: false`) and `fit` / `scan` answer 400 *"PyTorch is not installed: open Settings → Add-ons and press Install for Autoencoders (if it is already listed as installed, remove it and install it again)."*; everything else (looking at results, saved models, exports) works without it.

## 1. The request of a fit (`params` of the result; also the body of `check` and `fit`)

```jsonc
{
  "family": "ae",                       // "ae" | "vae"
  "latent_dim": 4,                      // 1..32 (default from the advice)
  "size": "small",                      // "small" [64,32] | "medium" [128,64] | "large" [256,128,64] | "custom" (then "hidden" must be the list); a convenience: "hidden": "small" says the same
  "hidden": [64, 32],                   // widths of the encoder (the decoder mirrors it): a preset name or a list of at most 6 widths (1..4096)
  "activation": "silu",                 // silu | tanh | gelu | relu | softplus | elu | linear
  "dropout": 0.0,                       // 0..0.9 (the page offers 0..0.5), on the hidden layers of the encoder
  "weight_decay": 0.001,                // 0..1 (AdamW)
  "corruption": {"noise": 0.0, "mask": 0.0},     // denoising: noise 0..2 (in SD of the data), mask 0..0.9 (share of inputs dropped)
  "vae": {"beta": 1.0, "warmup": null, "free_bits": 0.0, "sigma": "learned", "prior": "normal", "n_components": 3},
                                        // beta 0..100; warmup = epochs of KL annealing (null = 20 % of the epochs); free_bits 0..5 nats; sigma "learned" | a number > 0; prior "normal" | "mixture" (n_components 2..20)
  "smoothify": 0.0,                     // >= 0 (the page offers 0..10), functional versions only (ignored for pixels, with a note)
  "smoothify_metric": "smooth",         // "smooth" | "ridge"
  "gap_weighting": "observed",          // functional versions: "observed" = a sample that lacks part of the domain (a region excised from it) counts in proportion to the share of the domain that was measured (at least 5 %, the weights average 1) | "equal" = like any other sample.  Without such samples nothing changes
  "epochs": 600, "patience": 60,        // 1..5000, 1..1000
  "val_fraction": 0.2,                  // 0.05..0.6 (at least 5 samples are held out)
  "folds": 5,                           // 2..20: used instead of the hold-out when there are fewer than 30 samples (`kfold_below`, default 30)
  "lr": 0.001, "batch_size": null,      // lr 1e-6..1; batch_size null = automatic (min(32, max(8, n_train / 6))) | 2..100000
  "n_restarts": 3, "seed": 0,           // 1..50 random starts; seed 0..2^31-1
  "grad_clip": 5.0, "kfold_below": 30,  // gradient clipping 0 (off)..1e6; k-fold validation below this many samples (0..100000)
  // ---- advanced training settings: every default is what the training did before they existed (the history of every epoch is the same)
  "stop_rule": "validation",            // "validation": stop when the held-out loss has not improved for `patience` epochs | "training": the same on the training loss | "never": train all `epochs`
  "min_epochs": 0,                      // 0..100000: training never stops before this many epochs (the patience applies from then on)
  "min_delta": 0.001,                   // 0..0.5: the smallest RELATIVE improvement of the monitored loss that counts (0.001 = 0.1 %)
  "val_window": 1,                      // 1..100: the monitored loss is the average of the last `val_window` epochs (the history keeps the loss of each epoch)
  "max_minutes": null,                  // null (no limit) | 0.05..100000: a time limit for ONE network; the best weights so far are kept (`stopped` = "time")
  "restore_best": true,                 // true: the weights of the best epoch are kept | false: the weights of the last epoch
  "lr_schedule": "cosine",              // "cosine" (over `epochs`) | "cosine_run" (over the epochs this network is planned to train) | "constant" | "plateau" (halved after max(3, patience/4) epochs without improvement)
  "lr_final": 0.01,                     // 1e-4..1: the learning rate at the end of the cosine, and the floor of "plateau", as a share of `lr`
  "final_epochs": "auto",               // k-fold validation: the epochs of the final networks: "auto" (the median best epoch of the folds, at least `final_epochs_min`) | "max" (= `epochs`) | a number 1..100000
  "final_epochs_min": 20,               // 1..100000
  "skip": true,                         // a linear shortcut through the code beside the encoder and decoder; training starts from the PCA solution of the training samples.  false = the classical pure autoencoder.  Stored in `params` and in the saved ModelSpec (`spec.skip`)
  "allow_few_samples": false,           // true only after the person was asked and agreed to train on fewer than 12 samples (never fewer than 4)
  "input": {                            // the same words and the same checks as the Functional PCA request (/api/fpca/fit)
    "variables": ["fluorescence", "absorbance"],        // "fluorescence", "absorbance" or both
    "fluorescence": {"transform": "none", "min_coverage": 0.5, "gaps": "auto",
                     "basis": {"kind": "gp", "kernel": "matern52", "structure": "separable", "lengths": "auto", "var_fraction": 0.999, "k_max": 150},
                     "smoothing": {"method": "ml"}},
    "absorbance":   {"transform": "none", "processed": true, "gaps": "auto", "basis": {"kind": "gp", "lengths": "auto"}, "smoothing": {"method": "ml"}}
  },                                    // basis.kind "pixels" = the observed values themselves (no smoothing, gaps allowed); "gp" | "bspline" | "cosine" = the functional versions.
                                        // min_coverage 0.05..1: a pixel is part of the domain when this share of the samples have it.  gaps "auto" | "plain": a pixel only some samples have
                                        // (a region excised from some samples) is ESTIMATED from the other samples (default) | every sample is smoothed on its own pixels
  "title": ""                           // name of the result ("" = made from the settings)
}
```
Every key is optional (the defaults above, `input.variables` default `["fluorescence"]`).  The checks and their wording are those of `core.spec.Config` (network settings: *"The number of codes (latent dimension) must be between 1 and 32."*, *"Unknown setting 'foo'."*, ...) and of the Functional PCA request (`input`).  The weights between variables are fixed: each variable counts equally (SPEC §3.2), so there is no `weights` key (*"Unknown data setting 'weights'"*).  Other `Config` keys (`kfold_below`, `grad_clip`, `threads`, `vae.n_eff`) are accepted too.  `params` (the stored request) always holds the resolved `hidden` list, `size` and the full `input`.

**Samples that lack part of the domain** (a region excised from some samples, `docs/modules/models_core.md` of FluoViz).  A pixel that only some samples have stays in the domain when at least `min_coverage` of the samples have it.  *Pixel versions*: the loss leaves the missing pixels out (the error of a partly observed sample is scaled up to the whole block), nothing else changes.  *Functional versions*: with `gaps: "auto"` the coefficients of such a sample are estimated from the pixels it has **and from what the other samples show about the part it lacks** (the empirical-Bayes estimate of the Functional PCA, `docs/modules/fpca.md` §2b of FluoViz; `gaps: "plain"` smooths every sample on its own pixels and leaves the lacking part to the mean function), the mean function and the basis follow the same rule, and the saved model reads a new sample with a gap the same way.  An estimated sample is less certain than a measured one, so with `gap_weighting: "observed"` (default) its weight in the loss is the share of the domain that was measured (the mean over the functional variables, at least 5 %, rescaled so that the weights average 1); the PCA the training starts from and the linear baseline it is compared with use the same weights.  Nothing is weighed when no sample lacks 0.5 % of a domain.  The result says so in `notes`, publishes `missing_part` (below) and the sample view (2.5) says what a sample lacks.

Plain-English refusals: a bad setting is an immediate **400** for `check` and `fit` (*"The activation must be one of: silu, tanh, ..."*, `input` errors worded as `/api/fpca/fit` does: *"The basis must be one of: ..."*, *"Maximum likelihood is only available for the Gaussian-process basis ..."*).  Problems found while the data are read are errors of the **job**: absorbance asked for but not loaded (*"No absorbance spectra are loaded yet ..."*), fewer than 12 samples (*"Only 9 samples are included: an autoencoder needs at least 12 samples to be trained honestly. Add more samples, or use the PCA / functional PCA pages for very small data sets."*), more codes than inputs.  `vae` options with `family: "ae"` are ignored (not refused).

## 2. Endpoints

### 2.1 `GET /defaults`  — facts and advice (fast; synchronous)

Query: `variables` (comma list, default `fluorescence`; `absorbance` alone and `fluorescence,absorbance` are allowed: the recommended request models exactly these), `min_coverage` (0.05–1, default 0.5), `abs_processed` (default true), `advise` (0 | 1, default 0: with 1 the fPCA-style advice and the number of components of the functional PCA are computed too: a few seconds).

```jsonc
{
  "n_samples": 64, "ids": [...],
  "fluorescence": {"available": true, "n": 63, "n_pixels": 10432, "grid": [n_em, n_ex], "masked_fraction": 0.31, "units": "Raman units (RU)", "mask": uint8 (n_ex, n_em), "notes": [...]},
  "absorbance":   {"available": true, "n": 62, "n_wl": 301, "n_common": 61, ...} | {"available": false},
  "n_common": 61,                                  // samples with every chosen variable
  "components_99": {"pca": 6, "fpca": 5 | null},   // components that explain 99 % of the variation (PCA on the observed pixels; fpca only with advise=1)
  "advice": {
    "latent_dim": 5, "latent_reason": "...",       // min(components for 99 %, n/10, 8), at least 2
    "size": "small", "hidden": [64, 32], "family": "ae",
    "estimate": {"seconds": 40, "words": "about 40 seconds on this computer ..."},      // a guess before the basis exists (about 80 coordinates per variable): `check` gives the exact numbers
    "representation": {...the /api/fpca/data advice, with advise=1} | null,
    "warnings": ["Only 24 samples: ...", "..."],   // n small, samples per code, strong masking, samples without absorbance, ...
    "blocking": []                                   // reasons training is refused ("Only 9 samples ...")
  },
  "request": { ...the complete recommended request (section 1), the person's variables kept...},   // "Run with the recommended settings"
  "torch": {"available": true, "version": "2.14.1"|null, "words": ""}            // found with importlib.util.find_spec: torch is NOT imported
  "activations": [...], "sizes": {"small": [64,32], "medium": [128,64], "large": [256,128,64]}, "versions": {...}
}
```

### 2.2 `POST /check` (job) — what a request would do; nothing is trained

Body: a request (section 1).  Validated immediately (400).  Job result:

```jsonc
{
  "ok": true,                                       // false when `blocking` is not empty
  "n": 61, "dropped": [["S07", "no absorbance spectrum"], ...],
  "input": {"variables": [...], "mode": "functional" | "pixels" | "mixed", "n_inputs": 143, "blocks": [ <block>, ... ]},
  "network": {"family": "ae", "latent_dim": 4, "hidden": [64, 32], "activation": "silu", "n_params": 18932, "samples_per_param": 0.003, "samples_per_latent": 15.2},
  "validation": {"mode": "holdout" | "kfold", "n_train": 49, "n_val": 12, "folds": null | 5, "words": "12 samples are held out ..."},
  "estimate": {"seconds": 40, "words": "..."},
  "warnings": ["The network has 18,932 parameters for 61 samples: it can memorise the samples. ..."],
  "blocking": [], "torch": {"available": true, ...}, "notes": [...], "advice": {"components_99": 4, "latent_dim": 4, "latent_reason": "..."}   // the advice of the prepared data
}
```
`<block>` = `{"name": "fluorescence", "mode": "functional" | "pixels", "basis": "gp" | "bspline" | "cosine" | "pixels", "n_inputs": 87, "n_pixels": 10432, "n_grid": 26250, "units": "RU", "transform": "none", "variance": 1.7e-3, "smoothing": {"criterion": "ml", "theta": {...}, "sigma2": .., "sigma_f2": ..}, "missing_fraction": 0.0, "notes": [...]}`.  The prepared data is cached (the last request), so `fit` right after `check` or `defaults?advise=1` does not redo the basis and smoothing (a few seconds for a Gaussian-process basis).  `blocking` also holds *"PyTorch is not installed ..."*.

### 2.3 `POST /fit` (job) — train the network, save `autoencoder-N`

Body: a request (section 1) plus optional `"replace": "autoencoder-3"` (train again in place: same id, same title unless `title` is sent).  Refused up front (400): invalid settings, torch missing, blocking reasons (too few samples), `replace` of something that is not an autoencoder result.

The job message says what it is doing (*"Preparing the data"*, *"Training, start 2 of 3, epoch 120"*, *"Checking the result"*).  Cancel = nothing saved.  Job `result`:

```jsonc
{"result": {id, kind: "autoencoder", title, created, params, summary, stale: [], sample_ids, variables: [...names]},   // no payload
 "rid": "autoencoder-1"}
```
`summary` (listing facts): `{n, family, latent_dim, hidden, variables, mode, n_params, r2_val, r2_train, rmse_val, best_epoch, epochs_run, baseline: {name, r2_val}, verdict, n_flagged, seed, seconds}`.

**History while it runs:** `GET /fit/{job_id}/history?since=0`

```jsonc
{"job": "a1b2c3", "status": "running" | "done" | "error" | "cancelled", "phase": "preparing" | "training" | "checking" | "done",
 "restart": 1, "n_restarts": 3, "fold": null | 2, "n_folds": null | 5,     // what is being trained now
 "epochs": [ {"restart": 0, "fold": null, "epoch": 0, "train": 1.02, "val": 1.05, "rec": 0.9, "rec_val": 0.95, "kl": null, "kl_val": null, "smooth": null, "smooth_val": null, "beta": null, "sigma2": null, "lr": 0.001, "train_batch": 1.1, "best": true}, ... ],
                     // the epoch rows after record `since`: `train` / `val` = the objective (comparable between epochs), `rec` / `rec_val` = share of variation NOT rebuilt (0 = perfect) on the training / held-out samples, `best` = best so far of this run
 "events": [ {"event": "start" | "run_start" | "run_end" | "done" | "cancelled" | "error", ...} ],   // the other records of the file (what the worker started / finished)
 "next": 133,                                 // pass as `since` next time
 "best": {"restart": 0, "fold": null, "epoch": 98, "val": 0.12} | null,         // where the best weights of the run in progress (or the last run) are; epoch -1 = the starting weights (the PCA start), which no epoch improved on
 "message": "Training, start 2 of 3, epoch 34"}
```
`since` counts the records of the file (epoch rows and events); pass back `next`.  The rows come from the worker's progress file (`runtime/tmp/`); after the job ends the rows stay in memory for this server run (the last 12 jobs; the file is removed) and the kept run's history is always inside the result (`payload.training.history`).  The same route serves the live training of a **scan**.  Unknown job (or too old) → 404; `since < 0` → 400.  Polling twice a second is fine.  `phase` is `preparing` (before the first epoch), `training`, `checking` (the worker is done, the server checks and saves), `done`, `error` or `cancelled`.

### 2.4 `GET /{rid}` — the result for the page

`{"result": {...no payload...}, "payload": <payload>}`.  `payload` (images and masks already in browser orientation; the frozen weights are NOT included, they stay in the project):

| key | content |
|---|---|
| `ids`, `n`, `family`, `latent_dim`, `title` | the samples (those with every chosen variable), the settings |
| `input` | as `check.input` (blocks with their basis, `n_inputs`, `n_pixels`, `units`, `smoothing`) + `mode` ("functional" / "pixels") + `weights` |
| `network` | `{"spec": ModelSpec (SPEC §7.1, no arrays), "n_params", "hidden", "activation", "samples_per_param", "samples_per_latent"}` |
| `split` | `{"mode": "holdout"|"kfold", "n_train", "n_val", "folds", "flag": uint8 (n,) 1 = held out / out-of-fold, "fold": int (n,) | null, "words"}` |
| `code_names` | `["z1", ..., "zd"]` (also `z1_sd…` for a VAE, at most 12) |
| `codes` | `(n, d)` the codes (AE) / posterior means (VAE) of every sample, `codes_sd` `(n, d)` | null (VAE posterior SDs), `code_mean` `(d,)`, `code_sd` `(d,)` (of the training codes: the unit of the traversals) |
| `recon` | `{"error": (n,), "r2": (n,), "rmse": (n,), "rmse_by_block": {name: (n,)}, "error_by_block": {name: (n,)}, "heldout": (n,) bool (true = the network did not train on it), "limit": float (default level), "limit_level": "99", "limits": {"95": .., "99": ..}, "limit_words", "flag": [word per sample], "typicality": (n,) p-value, "kl": (n,) \| null, "elbo": (n,) \| null, "missing": (n,) (only when some sample lacked part of the domain)}` — `missing` is the share of the domain (by area, the mean over the variables) each sample lacked when the network was trained, e.g. a region cut out of it (the Samples tab writes it under the name of the sample, "9 % missing"); `error` is the share of the typical sample-to-sample variation that the network does **not** rebuild for that sample (0 = perfect, 1 = as bad as the mean), `r2 = 1 - error`, `rmse` in the units of the data (after the pre-treatment) |
| `training` | `{"history": {"restart": [...], "fold": [...], "epoch": [...], "train": [...], "val": [...], "rec": [...], "rec_val": [...], "kl": [...], "smooth": [...], "beta": [...], "sigma2": [...], "lr": [...]}` (the kept run; `null` where not defined; fold −1 = none), `"best_epoch": 98 (-1 = the starting PCA solution, which no epoch improved on), "epochs_run": 158, "stopped": "early" \| "max", "restarts": [{"seed", "val", "best_epoch", "epochs_run", "chosen": bool}, ...], "spread": {n, kept, mean, sd, min, max}, "overfit": {"level": "none" \| "mild" \| "clear", "gap": 0.04, "words": "..."}, "stopping": "words", "restarts_words": "words", "verdict": "plain English (quality, overfitting, stopping, random starts)", "fold_info": [...], "seconds"}` — every threshold and wording of these verdicts is `core/diagnostics.py` (`training_verdict`) |
| `quality` | `{"mode": "holdout"\|"kfold", "r2_val", "r2_train" (share of the variation rebuilt on held-out / training samples; k-fold: out-of-fold), "r2_by_block": {name: {"val", "train"}}, "rmse_val" (hold-out only, else null), "rmse_by_block", "rmse_all" (all samples, final network), "worst": [{"id", "error", "r2", "heldout"}...] (10, held-out value when there is one), "baseline": {"name": "PCA" \| "Functional PCA", "k": 4, "r2_val" (of the baseline), "r2_train", "difference" (network minus baseline), "better": "network" \| "baseline" \| "same", "relative_improvement", "z" (paired comparison of the per-sample errors), "verdict": "plain English: is the nonlinearity worth it?"} (`diagnostics.compare_baseline`), "local_r2": {"fluorescence": image (n_ex, n_em), "absorbance": curve (n_wl)} (share of the variance across samples that the network rebuilds at each pixel, on `local_on`), "residual_rms": {…same…}, "local_on": "held-out samples" \| "all samples", "verdict"}` |
| `latent` | `diagnostics.latent_health` + two extras: `{"kind": "ae"\|"vae", "names", "mean": (d,), "sd": (d,), "correlation": (d, d), "effect_share": (d,) (share of each code in what the decoder does), "inactive": [bool x d], "n_active", "effective_dim", "kl_per_dim": (d,) \| null, "coverage": {"mean", "variance", "inside95"} \| null (VAE, normal prior), "words", "typicality": {"covariance_ok", "median", "words"}, "pca": {"scores": (n, 2), "explained": [..]} (the codes' own PCA, only when d > 2)}` |
| `calibration` | VAE with a noise level only, else `null`: `diagnostics.calibration` `{"coverage68", "coverage95", "mean_z2", "per_code": [..], "verdict"}` — are the posterior SDs believable? |
| `smoothness` | `{"smoothify", "metric", "blocks": [{"name", "measure", "input", "reconstruction", "ratio", "verdict"}], "words"}` — `diagnostics.smoothness_report` (ratio < 0.9 smoother, > 1.1 rougher than the samples) over `diagnostics.roughness`: x'Qx / K in the smoothness metric of the representation (about 1 for a typical sample) for the functional blocks, the root mean square curvature for pixels (`measure` says which) |
| `scans` | `{"latent": {...}, "smoothify": {...}, ...}` the scans kept in the result (section 2.11) |
| `limits` | the entries of the saved kind (`[{label, value, meaning}]`; the reconstruction-error limit at the default level) |
| `images`, `ex`, `em`, `mask`, `units` | fluorescence in the model: `images.mean` `(1, n_ex, n_em)` (decoding of the mean code), `images.sensitivity` `(d, n_ex, n_em)` (change of the output per 1 SD of each code at the mean code; signed; NaN outside the domain), `image_labels`; `units` is the units of the data after the pre-treatment |
| `curves` | absorbance in the model: `{"x": wl, "label", "series": [{"name": "Decoding of the mean code", "values": [...]}, {"name": "z1: change for one SD of the code", "values": [...]}, ...]}` (null outside the domain) |
| `component_images` | the contract with Clustering & LDA, **module orientation** `images (d, n_em, n_ex)` as in every other result: `{images, score_vars: ["z1", ...], names, mean_image, kind: "signed", ex, em, description}` — what a unit change of each code does at the mean code (a linearisation; the description says so) |
| `notes`, `warnings` | plain English (what the data side noticed; what to be careful about) |
| `n_outside` | number of samples above the error limit or with an unusual code (with the default options) |
| `text` | the methods paragraph (also `GET /{rid}/methods`) |
| `versions` | `{"torch", "fluoviz", "addon", "numpy", "seed"}` |

### 2.5 `GET /{rid}/sample/{sid}` — one sample against the network

Needs the data (stale → 400).  Query: `thumb=1` leaves out the `residual` and `mean` images (hover cards).  `status` and `reasons` are those of the saved model at its default thresholds.  Response:

```jsonc
{"id": "S07", "title": "...", "split": "training" | "validation" | "held out" | "held out in fold 2",
 "status": "ok" | "outside", "reasons": ["Its reconstruction error (0.31) is above the 99 % limit (0.12): the network does not rebuild this sample well ..."], "words": "Poorly described by the model", "flag": "inside the limits",
 "code": [..d..], "code_sd": [..d..] | null, "code_names": [...], "limit": 0.12,
 "error": 0.03, "r2": 0.97, "rmse": 0.004, "typicality": 0.41, "kl": null | 3.1, "elbo": null | -12.2,
 "missing": {"fluorescence": 0.19}, "missing_words": "The fluorescence of S07 lacks 19 % of the domain (not measured, or cut out): ...",     // only when the sample lacks at least 0.5 % of the domain of a variable
 "blocks": [{"name": "fluorescence", "kind": "surface",
             "measured": image (n_ex, n_em), "reconstruction": image, "residual": image, "mean": image,
             "mask": uint8, "ex": [...], "em": [...], "units": "RU", "rmse": 0.004, "error": 0.02},
            {"name": "absorbance", "kind": "curve", "x": [...], "measured": [...], "reconstruction": [...], "residual": [...], "mean": [...], "units": "absorbance", "rmse": ..., "error": ...}],
 "text": "plain English summary"}
```
`missing` is the share of the domain (by area) the sample lacks, per variable; `missing_words` says what the network was shown instead (pixel versions: the pixels are ignored and the error is judged on the rest; functional versions: an estimate made from the pixels the sample has and, when the training samples had gaps, from what the other samples show).  In the pictures `measured` is blank (NaN) where the sample lacks the values, `reconstruction` is filled in by the network and `residual` is NaN there.
Unknown sample: 404 "'X' is not one of the samples of this network."

### 2.6 `POST /{rid}/traverse` — decode along a code

Body `{"code": 2 | null, "values": [-3, -2, -1, 0, 1, 2, 3], "base": "mean"}`: `code` 1-based (null = every code; the answer then has `d` entries), `values` in SD of the training codes (1..13 values, |value| ≤ 6), `base` `"mean"` (all other codes at their training mean).  Works without the data (also for a stale result).

```jsonc
{"values": [...], "units": "RU", "ex": [...], "em": [...], "mask": uint8,
 "codes": [{"code": 2, "name": "z2", "sd": 0.93, "mean": 0.01, "z": [..actual code values..],
            "fluorescence": {"images": (k, n_ex, n_em), "difference": (k, n_ex, n_em), "min": .., "max": .., "difference_abs_max": ..},
            "absorbance":   {"x": [...], "series": [[...] x k], "difference": [[...] x k]}}],
 "mean": {"fluorescence": image, "absorbance": [...]}}
```
`images[i]` is the decoded function at code value `z`; `difference[i]` = `images[i] - mean` (the decoding of the training mean code).

### 2.7 `POST /{rid}/interpolate`

Body `{"a": "S01", "b": "S09", "steps": 9}` (steps 3..25): the codes of the two samples are mixed `(1-t) z_a + t z_b`, `t = 0 … 1`, and decoded; next to it the plain linear mixture of the two measured functions.  The linear mixture needs the data: for a stale result it is `null` with a note.

```jsonc
{"t": [0, 0.125, ..., 1], "a": {"id", "code"}, "b": {...}, "code_path": (steps, d),
 "latent": {"fluorescence": {"images": (steps, n_ex, n_em)}, "absorbance": {"x", "series": [...]}},
 "linear": {…same shape…} | null, "difference": {…latent minus linear…} | null, "ex", "em", "mask", "units", "notes": []}
```

### 2.8 `POST /{rid}/generate` — new samples from a variational autoencoder

Body `{"n": 6, "seed": 1, "temperature": 1.0}` (n 1..24, temperature 0..2: the codes are drawn from the prior with its SD multiplied by the temperature).  A deterministic autoencoder answers 400 *"Only variational autoencoders can draw new samples (this network is a deterministic autoencoder). Train a variational autoencoder."*  (For a mixture prior each draw first picks a cluster.)

```jsonc
{"synthetic": true, "label": "Synthetic samples: made up by the network, not measured",
 "ids": ["synthetic-1", ...], "codes": (n, d), "nearest": [{"id": "S12", "distance": 0.8}, ...],     // nearest real sample in code space (SDs)
 "fluorescence": {"images": (n, n_ex, n_em)}, "absorbance": {"x", "series": [...]}, "ex", "em", "mask", "units", "notes": []}
```

### 2.9 `GET /{rid}/sensitivity`

What each code does: ∂(output)/∂(code j) at the mean code and its average magnitude over the samples, per 1 SD of the code (so the codes are comparable).  Works for a stale result.

```jsonc
{"names": ["z1", ...], "sd": [...], "units": "RU",
 "fluorescence": {"at_mean": (d, n_ex, n_em) (signed), "magnitude": (d, n_ex, n_em) (mean |.| over the training samples), "share": [..d..] (share of the total magnitude)},
 "absorbance": {"x": [...], "at_mean": [[...]xd], "magnitude": [[...]xd]},
 "ex", "em", "mask", "words": "Code z2 changes the fluorescence mostly around excitation 270 nm / emission 340 nm."}
```

### 2.10 `POST /{rid}/anomalies` — anomaly view with thresholds

Body `{"method": "mad" | "quantile" | "saved", "c": 3.0, "level": 0.99, "top": 12}`: `mad` = median + c·MAD of the (log) errors, `quantile` = that quantile of the held-out errors, `saved` = the limit of the saved model.  Works for a stale result too (the residual maps are then left out, with a `notes` entry).  Response: `{"limit": .., "method", "words", "n_flagged", "flagged": [{"id", "error", "typicality", "reasons": [...], "residual": {"fluorescence": image, "absorbance": curve}}...] (at most `top`, worst first), "histogram": {"edges", "counts", "limit"}, "notes": [...]?}`.  Errors: *"'method' must be mad, quantile or saved."*, *"'c' must be a number from 0.5 to 20."*, *"'level' must be a number from 0.5 to 0.999."*, *"'top' must be a whole number from 1 to 50."*

### 2.11 `POST /scan` (job)

Body: `{"scan": "latent" | "smoothify" | "stability" | "baseline", "rid": "autoencoder-1" | null, "request": {...} | null, "save": true, ...options}`: the settings come from `request`, or from the `params` of `rid` (when both are given, `request` overrides single keys of the stored `params`; one of the two is needed).  Scans train several small networks in ONE worker process (restarts reduced), so they take minutes; progress, cancel and the live training (`GET /fit/{job}/history`) work as for `fit`.  With `rid` and `save` (default true) the table is also kept in the result (`payload.scans.<kind>`), so it survives saving the project and is returned by `GET /{rid}`; a stale result keeps nothing.  Nothing else of the result changes.  `baseline` needs no PyTorch; the other scans refuse (400) without it.  Every scan result also carries `settings` (`latent_dim`, `family`, `smoothify`, `n`, `seed` of the request: the page dims a scan computed for other settings) and `saved_in` (the result it was kept in, or null).

| scan | options | job result |
|---|---|---|
| `latent` | `dims` (default 1…min(12, inputs − 1, n/3); at most 24 whole numbers 1..32), `families` (default the family of the request; `["ae", "vae"]` allowed), `n_restarts` (1..5, default 1) | `{"scan": "latent", "dims": [...], "families": {"ae": {"val": [...], "val_se": [...], "train": [...]}, "vae": {...}}, "baseline": {"name": "PCA" \| "Functional PCA", "k": [...], "val": [...], "val_se": [...], "train": [...]}, "recommended": {"latent_dim": 4, "reason": "..."}, "words"}` — errors are the share of the variation **not** rebuilt on held-out samples (out-of-fold for few samples), `val_se` their standard error; the recommendation is the smallest number of codes within one standard error (or 1 % of the variation) of the best |
| `smoothify` | `values` (default 0 and 11 log-spaced values 0.0001 … 10; at most 16 numbers ≥ 0), `n_restarts` | `{"scan": "smoothify", "values": [...], "error": [...], "error_se": [...], "roughness": [...], "roughness_input": .., "suggested": 0.3, "words"}` — the L-curve (held-out error against roughness of the reconstructions); `suggested` = the largest value whose error is within one standard error (or 2 %) of the error without smoothing |
| `stability` | `n_runs` (2..10, default 5), `bootstrap` (default false) | `{"scan": "stability", "n_runs", "alignment": {"mean", "min", "per_code": [...]}, "recon_spread", "matrix": (n_runs, n_runs), "procrustes": (n_runs, n_runs), "verdict": "The codes are / are not reproducible: ...", "words"}` |
| `baseline` | `dims` (default as for `latent`) | `{"scan": "baseline", "name": "PCA" \| "Functional PCA", "dims": [...], "val": [...], "val_se": [...], "train": [...], "mode": "holdout" \| "kfold", "network": {"k": 4, "val": 0.012} \| null (the network of `rid`, on the same scale: its held-out error), "words"}` — the linear baselines for every number of components on the split of the network, no training |

Errors (400): *"'scan' must be one of: latent, smoothify, stability, baseline."*, *"Say what to scan: give 'rid' ... or 'request' ..."*, *"'dims' must hold whole numbers from 1 to 32."*, *"'families' must be a list holding ae and / or vae."*, *"'n_restarts' must be a whole number from 1 to 5."*, *"'values' must be a list of 1 to 16 numbers, each 0 or more."*, *"'n_runs' must be a whole number from 2 to 10."*, and every refusal of the request itself.

### 2.12 `GET /{rid}/export?what=…`

| `what` | file |
|---|---|
| `codes` | `<rid>_codes.csv`: `sample, split, z1…zd, recon_rmse, recon_r2, typicality, flag, [z1_sd…, kl, elbo, cluster, membership] (VAE), status` (`split` = training / validation / held out in fold k) |
| `reconstruction` | `<rid>_reconstruction.npz`: `ids`, `<variable>_reconstruction` `(n, n_em, n_ex)` / `(n, n_wl)` (NaN outside the domain), `<variable>_mean`, the axes (`fluorescence_em_nm`, `fluorescence_ex_nm`, `absorbance_wavelength_nm`), `codes`, `code_names`; with current data also `<variable>_observed` and `<variable>_residual` |
| `history` | `<rid>_history.csv`: one row per epoch of the kept run (`restart, fold, epoch, train, val, rec, rec_val, kl, kl_val, smooth, smooth_val, beta, sigma2, lr`; empty where not defined) |
| `synthetic` | VAE only; query `n` (1..1000, default 24), `seed`, `temperature`, `format` = `npz` (default; `synthetic` = true, `codes`, the generated functions, the axes, `README` text) or `csv` (the codes and the nearest real sample); every file says "synthetic" in its name (`<rid>_synthetic.npz`) and content |

Unknown `what`: 400 *"what must be codes, reconstruction, history or synthetic."*

### 2.13 `GET /{rid}/methods`

`{"text": "<paragraph for a methods section: data, representation, smoothing, network, loss, training, validation, versions, seed>", "references": ["Kingma & Welling 2014 ...", ...]}`.  Works for a stale result (it describes the network that was trained).

### 2.14 Extras

`POST /{rid}/relate` — codes against other variables: body `{"variables": ["meta:DOC", "qc:total", "res:pca-1:PC1", ...], "folds": 5}` (keys as in `/api/variables`) → `{"names": ["z1", ...], "variables": [{"key", "label", "correlation": [..d..], "r2_codes", "r2_pca", "n"}], "skipped": [keys that are not numbers], "words", "note"}`: for every numeric variable the correlation with each code, the cross-validated R² of "predict the variable from the codes" and the same from as many PCA scores of the network coordinates (`r2_pca` is null for a stale result, with a `note`).  Errors (400): *"Choose the variables to compare the codes with ..."*, an unknown key (the project's own words), *"None of these variables is a number ..."*, *"'folds' must be a whole number from 2 to 20."*

### 2.15 The nonlinear effects of the codes: `POST /{rid}/response`, `/bends`, `/local`, `/pair`

*Which nonlinear trend does each code fit?*  Four views of the decoder (numerics: `core/curves.py`, `core/features.py`, `core/nonlinear.py`, pure numpy; the answers: `server/nonlinear.py`; the requests are checked in `server/nlrequest.py`).  They need no measured data, except for the dots of the training samples (an outdated network can still be asked: the dots are then left out and `notes` says why).  The routes are synchronous (well under a second for the usual network; 500 samples, 32 codes and 17 000 pixels take about a second for a functional network and a few seconds for a pixel network).  A wrong request is a **400** with a sentence (**404** for an unknown sample), and a key that the route does not know is refused: *"Unknown setting 'x': the settings of this request are …"*.  Images are in browser orientation `(n_ex, n_em)` like everywhere (the page turns what it draws when the person asks for excitation across).

**Definitions, the same in the four routes**

* **Position of a code**: `t` in standard deviations (SD) of that code over the training samples (`sd` = `code_sd`), the other codes at their training mean (`mean` = `code_mean`): exactly the position of the traversals (2.6).  A variational autoencoder decodes its posterior mean.  `n_points` positions from −`span` to +`span` (defaults 41 and 3; `span` 0.5 to 4, `n_points` 5 to 201; the position 0 is among them when `n_points` is odd).
* **Weights**: `w_i` = the standard normal density at the position `t_i`, divided by the sum of them (a position near the mean code counts more than one far out, like the samples).
* **Straight line**: the weighted least-squares line of `y` against `t`.  `r2_linear = 1 − SSE / SST` with `SSE = Σ w_i (y_i − line_i)²`, `SST = Σ w_i (y_i − ȳ)²`, `ȳ = Σ w_i y_i` (null when the curve is constant up to rounding: its range below 1e-9 of its largest absolute value); `max_deviation = max_i |y_i − line_i| / (max y − min y)`.
* **Units** are those of the data after the pre-treatment (`units`, as the traversals say); a coefficient of the network has none.
* **The metric of the loss**: the squared distance of two decodings is `Σ_blocks β_b Σ_j w_j (f_j − g_j)²`, with `w_j` = `domain.w` (the quadrature weights of the observed pixels of the block, adding up to 1) and `β_b` the weight of the block in the training loss (each variable counts equally, divided by its total variance).  The server computes it in the network coordinates with `out.weights` (the same number: the map from coordinates to functions is an isometry), so a network with fluorescence and absorbance gets ONE number per code.

**Features** (a feature is one number read from a decoded function; `features` of `response` and `feature` of `pair`).  A request is a group `{"kind": …}`:

| `kind` | settings | value |
|---|---|---|
| `moved` | `block` (default: the first block with an EEM, else the first block) | the value at the pixel (the wavelength of the absorbance) that the code moves most: the pixel with the largest range `max − min` of the decoded response along the traversal; it is resolved **for every code** (`pixel` of an entry of `codes`); for `pair` it is the pixel with the largest range over the grid |
| `point` | `ex`, `em` (nm) | the intensity at the grid pixel nearest to the wavelengths; that pixel must be part of the model |
| `region_mean`, `region_max` | `ex: [lo, hi]`, `em: [lo, hi]` | the mean / the largest value over the pixels of the model inside the box (at least one) |
| `total` | `block` | the overall intensity: `Σ_j w_j f_j` with `domain.w` (the average over the whole EEM, or over all wavelengths of the absorbance) |
| `absorbance` | `wavelength` (nm) | the absorbance at the nearest wavelength of the model (only when the network has an absorbance block) |
| `coordinate` | `k` (1 … K), `block` | the k-th network coordinate of a block: for a functional network the standardised coefficient of the k-th term of its basis (for the Gaussian-process basis a Karhunen–Loève score divided by its prior SD); not for a pixel network |

At most 6 features.  Without `features` the page's defaults are used: `moved` and `total` of the first block with an EEM (else the first block).  A feature gets the id `f1, f2, …` in order.  The answer describes each one: `{"id", "kind", "block", "label", "short", "units", "per_code", "where"}` (`where`: the pixel `{ex, em}`, the box `{ex: [lo, hi], em: [lo, hi], n_pixels}`, the wavelength, or the coefficient `{k, n}` that was actually used).  Errors: *"Choose what to measure: the kind of a feature must be one of: …"*, *"The emission wavelength must be a number (in nm)."*, *"The excitation wavelength 5000 nm is outside the range of the data the network uses (240–450 nm)."*, *"The pixel at excitation X nm and emission Y nm is not part of the model (it is scatter, or below the Rayleigh line, or was cut out). Choose a pixel in the coloured part of the picture."*, *"No pixel of the model lies inside the box …"*, *"This network does not use the absorbance, so there is no absorbance to look at."*, *"The coefficients of a basis exist only for a functional network; this network works on the pixels. Choose another feature."*, *"The number of the coefficient must be a whole number from 1 to K."*, *"At most 6 features can be drawn at once."*

**The words of a curve** (`describe_curve`; the first rule that fits wins; the thresholds are constants of `core/curves.py`)

| `kind` | when | `label` |
|---|---|---|
| `flat` | the range of the curve is below **2 %** of the largest range among the codes for the same feature (or the curve is constant up to rounding) | *has almost no effect on this feature* |
| `wiggly` | two or more changes of direction | *changes direction several times* |
| `peak` / `valley` | one change of direction | *rises then falls (peak at about +x SD)* / *falls then rises (lowest at about −x SD)* |
| `linear` | `max_deviation` ≤ **5 %** | *nearly linear* |
| `s_shaped` | flat at both ends (below), or no plateau and the residuals from the line change sign three times or more | *S-shaped (levels off below about −x SD and above about +x SD)* / *S-shaped* |
| `saturating` | flat at one end (below) | *levels off (saturates) above about +x SD* / *… below about −x SD* |
| `convex` / `concave` | otherwise: the curve lies below / above its chord (the straight line between its two ends) in the middle | *has a bend (convex: it bends upwards, the slope grows with the code)* / *has a bend (concave: it bends downwards, the slope shrinks with the code)* |

*A change of direction* counts when the curve moves back from an extreme by at least **10 %** of its range (the zigzag rule; a smaller dip is not one); an extreme at an end of the positions is not one.  *Flat at an end* means: from some position to that end the curve stays within **5 %** of its range of the value at the end, over at least **25 %** of the positions (`plateau` gives where it starts); a residual counts for the sign pattern when it is above **2 %** of the range.  A position in a label is rounded to 0.1 SD and written with a true minus sign.  Whatever its shape, a curve that has both signs (each beyond 2 % of its range) also gets `sign_change` (the position of the first zero) and `sign_label` = *switches sign near +x SD* (the page shows it for intensities, where a negative value says that the network is beyond what it has seen, and not for coefficients, which are centred).  The curve entry also carries `range` `[min, max]`, `slope` (of the line, per SD of the code), `direction` (`up` | `down` | null), `position` (the peak or valley), `turns` `[{position, kind: max | min}]`, `plateau` `{above, below}`.

**`POST /{rid}/response`** — the feature against the position of every code

```jsonc
{"span": 3, "n_points": 41,
 "features": [{"kind": "moved"}, {"kind": "total"}],        // default: these two
 "overlay": true,                                           // the training samples around the curves (default true)
 "ice": {"codes": [1, 2, 3], "n": 12} | {"codes": [1], "samples": ["S01", "S07"]} | true | false}   // curves of single samples (default false; true = the first 8 codes, 12 samples)
```
`ice.codes` (1-based; default the first 8) are the codes that get curves of single samples, `ice.n` (1–12) how many samples are chosen over the distribution of each code (the samples at the ranks `(k + 0.5) / n` of the sorted codes, each once), `ice.samples` (at most 12 names) the same samples for every code.  Answer:

```jsonc
{"span", "n_points", "t": [...], "weights": [...],           // the positions and their weights
 "specs": [...],                                             // the features as asked (the defaults filled in)
 "features": [{"id": "f1", "kind", "block", "label", "short", "units", "per_code", "where"}, ...],
 "names": ["z1", ...], "mean": [...], "sd": [...], "units": {"fluorescence": "RU"}, "ex", "em", "mask", "abs_wl",
 "codes": [{"code": 1, "name": "z1",                         // one entry for EVERY code (the page draws the first 8 and the others on request)
            "curves": {"f1": {"y": [...n_points...], "r2_linear", "max_deviation", "kind", "label", "words", "range", "slope", "direction", "position", "turns", "plateau", "sign_change", "sign_label"}, "f2": {...}},
            "pixel": {"f1": {"ex": 350.0, "em": 450.0}},     // only for a feature resolved per code ({"wavelength"} for the absorbance)
            "ice": {"samples": ["S07", ...], "x": [..SD..], "y": {"f1": [[...n_points...] × k], ...}} | null}, ...],
 "overlay": {"ids": [...], "x": (n, d), "y": {"f1": (d, n), "f2": (n,)}} | null,    // x = the codes of the samples in SD from the mean; y = the feature of the function the network was trained to rebuild (smoothed sample; the measured values for a pixel network), read only where the sample was measured (a region cut out of a sample is left out of its feature: NaN where nothing is left); (d, n) for a feature resolved per code
 "notes": []}
```
`words` is the sentence under a curve: *"Most moved pixel (excitation 320 nm, emission 410 nm): has a bend (convex …). A straight line explains 99 % of it (R² = 0.99); at its worst the curve is 13 % of its range away from the line."* (*"… nearly linear. A straight line explains 100 % of it."* for a linear curve).  A curve of a single sample is the feature of the decoding of the sample's own codes with this code moved: if it is parallel to the mean curve the code acts independently of the others.  The sample dots and the `ice` samples are the training samples (`sample_ids`); outdated network: `overlay` null and `notes` = *"The data changed since this network was trained, so the training samples are not drawn …"*.

**`POST /{rid}/bends`** — where the response bends, pixel by pixel

Body `{"span": 3, "n_points": 41}`.  For every code, along the same traversal, at EVERY pixel of every block, the weighted straight line of the decoded value against `t` is fitted and `NL = SSE / SST = 1 − R²` is worked out (0 = a straight line, 1 = nothing of the response is linear; clipped to [0, 1]).  The **amplitude** of a pixel is its range `max − min` along the traversal.  A pixel whose amplitude is below **2 %** (`min_effect`) of the largest amplitude of that code (in that block) has *no effect*: its `NL` is `null`, not 0 or 1.  The **nonlinear share** of a code is `Σ_j w_j SSE_j / Σ_j w_j SST_j` with `w_j` = `domain.w`: the share of the effect of the code, in the metric of the loss, that no straight line describes (0 = the code acts like a PCA loading, 1 = entirely nonlinear); `share` is that number in the metric of the loss over all blocks, and `fluorescence.share` / `absorbance.share` the same for one block (for a network with one block they are equal).  A code that the decoder ignores (its effect below 1e-10 of the largest) is `dead`: `share` is null.  Answer:

```jsonc
{"span", "n_points", "t", "min_effect": 0.02, "nearly_straight": 0.05, "names", "mean", "sd", "units", "ex", "em", "mask", "abs_wl",
 "share": [..d..],  "effect_share": [..d..],  "dead": [bool × d],      // effect = Σ_j w_j SST_j in the metric of the loss; effect_share = its share among the codes (0 for a dead code)
 "fluorescence": {"nl": (d, n_ex, n_em), "amplitude": (d, n_ex, n_em), "share": [..d..], "n_effect": [..d..], "max_amplitude": [..d..]},     // NaN outside the model and where there is no effect (NL)
 "absorbance":   {"x": wl, "nl": (d, n_wl), "amplitude": (d, n_wl), "share": [...], "n_effect": [...], "max_amplitude": [...]},
 "words": "z2 bends the most: 11 % of its effect on the picture cannot be described by a straight line. z1 acts almost like a straight line (below 5 %)."}
```
The words call a code *almost like a straight line* below **5 %** (`nearly_straight`, which the page also draws as the dotted line of the bar chart), and say *"Every code acts almost like a straight line (the largest share is x %): this network is nearly linear."* when the largest share is below that; a dead code *does not change the output at all*.

**`POST /{rid}/local`** — local loadings and how they vary

Body `{"code": 2, "samples": ["S01", "S07"] | null}` (`code` 1-based; at most 6 samples; without `samples` the samples nearest to the 10th, 50th and 90th percentile of the code).  The **local loading** of code *j* at a sample is the Jacobian of the decoder (analytic through the layers, no finite differences) with respect to the code, **per 1 SD of the code**, evaluated at the codes of the sample; at the mean code it is the loading that `GET /sensitivity` shows.  The training samples go through in batches (at most 4 million numbers of Jacobian at a time), so memory stays small.  **Variability map**: at each pixel the standard deviation (n − 1) over ALL training samples of the loading of this code.  **Change of the shape**, for EVERY code: the mean over the training samples of `1 − cos`, where `cos` is the cosine similarity (the metric of the loss: weights `domain.w` and the weights of the blocks) between the loading at the sample and the loading at the mean code (0 = the same picture everywhere; up to 2); `size_mean` / `size_sd` are the mean and standard deviation over the samples of the length of the loading relative to the one at the mean code (1 = as strong), because the cosine does not see a loading that only gets stronger or weaker.  Answer:

```jsonc
{"code": 2, "name": "z2", "sample_sd": 1.2, "norm_at_mean": 0.43, "names", "mean", "sd", "units", "ex", "em", "mask", "abs_wl",
 "samples": [{"id": "S07", "label": "10th percentile of z2" | "median of z2" | "90th percentile of z2" | "chosen by you", "position": -1.3 (SD of the code), "cosine": 0.97}, ...],
 "fluorescence": {"at_mean": (n_ex, n_em), "samples": (k, n_ex, n_em), "variability": (n_ex, n_em)},       // signed loadings; NaN outside the model
 "absorbance":   {"x": wl, "at_mean": [...], "samples": (k, n_wl), "variability": [...]},
 "change": {"mean": [..d..], "max": [..d..], "size_mean": [..d..], "size_sd": [..d..]},                      // null for a code the decoder ignores
 "words": "The picture of its effect changes shape from sample to sample and its strength varies by about 33 % (one standard deviation)."}
```
The words say *"has the same shape for every sample"* when `change.mean` is below **0.005**, *"changes a little from sample to sample"* below **0.05** and *"changes shape from sample to sample"* above; then *"and about the same strength."* when `size_sd` is below **0.05**, else *"and its strength varies by about x % (one standard deviation)."*; for a code the decoder ignores: *"z2 does not change the output, so it has no loading to compare."*  Errors (400): *"'code' must be a whole number from 1 to d …"*, *"At most 6 samples can be chosen."*, *"A sample can be chosen only once."*; 404 for a sample that is not one of the network's.

**`POST /{rid}/pair`** — two codes together

Body `{"codes": [1, 3] | null, "grid": 5, "span": 2, "feature": {…} | null, "heat_points": 31}`: `codes` 1-based and different (null = the two codes with the largest `nonlinear share`, the lower number first; `auto` says so in the answer), `grid` 3, 5 or 7 positions per code from −`span` to +`span` (default 2) SD (the middle one is the mean), `feature` a feature as above (default `moved`: the pixel with the largest range over the grid), `heat_points` 11 to 61 (default 31).  The two codes are moved together with all the others at their mean.  With `F(a, b)` the decoded function with the first code at position `a` and the second at `b`, and `c` the position 0: the **interaction** is `I(a, b) = F(a, b) − F(a, c) − F(c, b) + F(c, c)` (what is left when the effect of each code alone is taken off); the **interaction image** is, per pixel, `max_{a,b} |I(a, b)|` (values below 1e-10 of the largest `|F|` are set to 0); the **interaction index** is `Σ_{a,b,j} I² / Σ_{a,b,j} (F − F̄)²` over the grid and the pixels (in the metric of the loss), `F̄` = the mean of `F` over the grid: 0 = the two codes act independently (additively), larger = the effect of one depends on where the other is (null when the picture does not change over the grid).  `index_by_block` is the same for each block with its own `domain.w`.  Answer:

```jsonc
{"codes": [1, 3], "auto": false, "grid": 5, "span": 2, "u": [-2, -1, 0, 1, 2], "index": 0.04, "interaction_ss", "variation_ss", "index_by_block": {"fluorescence": 0.04}, "index_limits": [0.05, 0.25],
 "words": "z1 and z2 act almost independently: the change of the picture is the sum of what each does alone (4 % of the change is not).",
 "names", "mean", "sd", "units", "ex", "em", "mask", "abs_wl", "notes": [],
 "feature": {…info of the feature…, "where": {"ex": 320, "em": 430}},                            // for `moved`: the pixel found
 "gallery": {"fluorescence": {"images": (grid², n_ex, n_em)}, "absorbance": {"x": wl, "series": (grid², n_wl)}},      // index = position of the second code × grid + position of the first
 "interaction": {"fluorescence": (n_ex, n_em), "absorbance": (n_wl,)},
 "heat": {"x": [...], "y": [...], "z": (heat_points, heat_points)},                              // the feature over the plane; z[row = second code][column = first code]
 "dots": {"ids": [...], "x": [..SD of the first code..], "y": [..SD of the second..], "value": [..the feature of each measured sample..] | null}}
```
The words say the two codes *act almost independently* below an index of **0.05**, *interact a little* below **0.25** and *interact strongly* above (`index_limits`: the two numbers, which the page uses for the colour of the card).  Errors (400): *"Two codes are needed to look at two codes together: this network has only one."*, *"'codes' must be two code numbers, for example [1, 2]."*, *"Choose two different codes."*, *"'grid' (pictures per code) must be 3, 5 or 7."*, *"'span' (how far each code is moved, in SD of the code) must be a number from 0.5 to 4."*, *"'heat_points' (points of the heat map per code) must be a whole number from 11 to 61."*; for `response`, `bends` and `local`: *"'n_points' (positions per curve) must be a whole number from 5 to 201."*, *"'code' must be a whole number from 1 to d (the network has d codes)."*, *"'overlay' must be true or false."*, *"'ice' must be true, false or a group of settings (n, samples, codes)."*, *"'ice.codes' must be a list of code numbers from 1 to d."*, *"At most 12 samples can be chosen."* / *"At most 6 samples can be chosen."*, *"'x' is not one of the samples of this network."* (404), *"A sample can be chosen only once."*.

## 3. What the result publishes

`result.variables` (all aligned with `result.sample_ids`; they appear under the result's title in Plots, metadata links and every variable selector, so the codes can be **clustered and classified on the Clustering & LDA page**: `#/cluster?keys=res:autoencoder-1:z1,res:autoencoder-1:z2`):

| name | kind | label (the `var_info`) |
|---|---|---|
| `z1 … zd` | numeric | "z1 (code 1)" — AE: the code; VAE: the posterior mean.  Description: "Code 1 of the autoencoder: a number the network learned to describe this sample with ..." |
| `recon_rmse` | numeric | "Autoencoder reconstruction error (RMSE, data units)" |
| `recon_r2` | numeric | "Autoencoder reconstruction R² (share of the variation rebuilt)" |
| `typicality` | numeric | "Typicality of the code (p-value)": 1 = in the middle of the training codes, near 0 = far out |
| `flag` | category | "Autoencoder anomaly status": `inside the limits` / `reconstruction error above the limit` / `unusual code` / `reconstruction error above the limit and unusual code` |
| `split` | category | "Autoencoder: used for training or held out": `training` / `held out` (only in the result, not in the saved model) |
| `missing_part` | numeric | "Autoencoder: share of the domain not measured": 0 to 1, the mean over the variables of the share of the domain (by area) the sample lacked when the network was trained (a region excised from it); only when some sample lacked at least 0.5 % (only in the result, not in the saved model) |
| `status` | category | "Status": Fits the model / Poorly described by the model (the words of the models page: `ok` / `outside`) |
| VAE: `z1_sd … zd_sd` (at most 12) | numeric | "Uncertainty of z1 (posterior SD)" |
| VAE: `kl`, `elbo` | numeric | "KL divergence (how unusual the code is)", "Evidence lower bound (higher = the network explains the sample better)" |
| mixture prior: `cluster`, `membership` | category / numeric | "Cluster of the mixture prior", "Membership of its cluster" |

The result is `kind: "autoencoder"`, `saveable: true`: **Save this model** is `POST /api/models` `{"result": "autoencoder-1", "title": "..."}` (job) → the saved kind `autoencoder`.

## 4. The saved model (kind `autoencoder`)

See `kind_and_router.md` for what is frozen.  Summary for the page:

* **Network**: `meta.model.spec` is the `ModelSpec` of the trained network, including `skip` (the linear shortcut through the code, `Config.skip`, default true; arrays `enc.lin.W`, `dec.lin.W`; false = the classical pure autoencoder), which the numpy forward pass of the applier handles.
* **Applies to**: EEMs and / or absorbance exactly as the training samples were prepared (the data recipe of FluoViz does the corrections, scatter treatment, window and grid).
* **Outputs** (`meta.outputs`): `z1…zd`, `recon_rmse`, `recon_r2`, `flag`, `status` (+ VAE `z1_sd…`, `kl`, `elbo`).  `status`: `ok` "Fits the model", `outside` "Poorly described by the model" (reconstruction error above the limit; for a VAE also an atypical code), `unusable` "Cannot be used" (too few observed pixels: `min_coverage`, a variable missing, a pre-treatment that is not defined).
* **Options** (thresholds, each with a plain-English `help`): `limit` "Limit for the reconstruction error" `95` / `99` (default `99`: the 95 % / 99 % point of the errors of samples the network was not trained on), `typicality` "Smallest typicality of the code" a number 0–0.5 (default **0.01 for a variational autoencoder, 0 = off for a deterministic one**).
* **Chain**: a classifier / clustering built on `res:autoencoder-N:z1…` embeds this model as its first link (applying it projects the new samples with the network first).
* **Payload of an application** (generic blocks of `docs/modules/models.md`): `plot` (the codes against the training codes), `diagnostics` (reconstruction error against typicality, with limits), `limits`, `images` / `curves` (mean, code directions), `notes`; `inspect` gives measured / reconstruction / residual of one sample.
