# Autoencoders add-on: HTTP API

*Written by A2 (kind + router) for A3 (the page) and for the Help guide.  Binding with `SPEC.md` §8.  It is kept accurate while the code is built; "TBC" marks a detail that depends on a part another agent is still writing.*

All routes live under **`/api/addon/autoencoders`** (the framework refuses any other prefix).  The add-on also uses, unchanged, the generic FluoViz routes: `/api/fpca/*` (steps 1–4: `data`, `gp/spectrum`, `gp/scan`, `gp/estimate`, `basis/check`), `/api/jobs/{id}` (poll a job), `/api/variables` (the published variables), `/api/models*` (save / apply / chain the model), `/api/results*` (list, rename, delete).

## 0. Conventions

* **JSON encoding** is FluoViz's: `NaN` / `±inf` → `null`; a numpy array with more than 1500 numbers → `{"$nd": "f4" | "i4" | "u1", "shape": [...], "b64": "..."}` (little endian, C order; turn it into a typed array), smaller arrays are nested lists.
* **Orientation.**  Every image is in *browser orientation*: one EEM is `(n_ex, n_em)` (rows = excitation, columns = emission), a stack is `(k, n_ex, n_em)`.  `ex` and `em` are the wavelength axes (nm) of the model's grid; `mask` is `uint8 (n_ex, n_em)`, **1 = this pixel is not in the model** (scatter, regions, below the Rayleigh line).  Absorbance and other curves are plain arrays on `abs_wl` (or the `x` of the block), `null` outside the model's domain.  Payload keys documented below as "`(…)` module orientation" exist only inside the saved result; the `GET` answers always turn them.
* **Errors**: `{"error": {"message": "<plain English>", "kind": "user" | "bug", "detail": ""}}`.  400 = something the person can fix (the message says what to do), 404 = unknown result / sample, 500 = a bug.  A job that fails ends with `status: "error"` and the same message in `error`.
* **Jobs** (`fit`, `check`, `scan`): the POST answers `{"job_id": "..."}` (after the request was validated: a bad setting is an immediate 400, never a failed job); poll `GET /api/jobs/{id}` (`status` queued | running | done | error | cancelled, `progress` 0–1, `message`, `result` when done); cancel with `POST /api/jobs/{id}/cancel`.  A cancelled fit saves nothing.
* **Staleness.**  Like every result, `autoencoder-N` records the data signature it was trained on.  When the data, corrections, scatter treatment, focus window or sample selection change, `result.stale` lists the changed parts (`raw`, `corrections`, `masks`, `window`, `samples`).  *The network is never trained again silently.*  A stale result can still be looked at (everything stored in it: codes, training curves, quality, traversals, interpolation between codes, generation, sensitivity, exports of codes); endpoints that need the measured data (`sample`, `anomalies`, the observed part of `export?what=reconstruction`) answer 400 *"The data or the corrections changed since this network was trained (the focus window changed). Use 'Train again with the same settings' to train it on the current data."*  **Train again** = `POST /fit` with the stored `params` plus `"replace": "<rid>"` (the same id and title, the new network).  An outdated result cannot be saved as a model (the generic `/api/models` says so).
* **Plain English** in every `message`, `words`, `help`, `reasons`, `notes`, `warnings` and `text` field; keys are `snake_case`.
* **Torch**: the server never imports PyTorch.  Training runs in a separate process.  When the library is not installed, `defaults` says so (`torch.available: false`) and `fit` / `scan` answer 400 *"PyTorch is not installed: open Settings → Add-ons and press Install (or Repair) for Autoencoders."*; everything else (looking at results, saved models, exports) works without it.

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
  "epochs": 600, "patience": 60,        // 1..5000, 1..1000
  "val_fraction": 0.2,                  // 0.05..0.6 (at least 5 samples are held out)
  "folds": 5,                           // 2..20: used instead of the hold-out when there are fewer than 30 samples (`kfold_below`, default 30)
  "lr": 0.001, "batch_size": null,      // lr 1e-6..1; batch_size null = automatic (min(32, max(8, n_train / 6))) | 2..100000
  "n_restarts": 3, "seed": 0,           // 1..50 random starts; seed 0..2^31-1
  "skip": true,                         // a linear shortcut through the code beside the encoder and decoder; training starts from the PCA solution of the training samples.  false = the classical pure autoencoder.  Stored in `params` and in the saved ModelSpec (`spec.skip`)
  "allow_few_samples": false,           // true only after the person was asked and agreed to train on fewer than 12 samples (never fewer than 4)
  "input": {                            // the same words and the same checks as the Functional PCA request (/api/fpca/fit)
    "variables": ["fluorescence", "absorbance"],        // "fluorescence", "absorbance" or both
    "fluorescence": {"transform": "none", "min_coverage": 0.9,
                     "basis": {"kind": "gp", "kernel": "matern52", "structure": "separable", "lengths": "auto", "var_fraction": 0.999, "k_max": 150},
                     "smoothing": {"method": "ml"}},
    "absorbance":   {"transform": "none", "processed": true, "basis": {"kind": "gp", "lengths": "auto"}, "smoothing": {"method": "ml"}}
  },                                    // basis.kind "pixels" = the observed values themselves (no smoothing, gaps allowed); "gp" | "bspline" | "cosine" = the functional versions
  "title": ""                           // name of the result ("" = made from the settings)
}
```
Every key is optional (the defaults above, `input.variables` default `["fluorescence"]`).  The checks and their wording are those of `core.spec.Config` (network settings: *"The number of codes (latent dimension) must be between 1 and 32."*, *"Unknown setting 'foo'."*, ...) and of the Functional PCA request (`input`).  The weights between variables are fixed: each variable counts equally (SPEC §3.2), so there is no `weights` key (*"Unknown data setting 'weights'"*).  Other `Config` keys (`kfold_below`, `grad_clip`, `threads`, `vae.n_eff`) are accepted too.  `params` (the stored request) always holds the resolved `hidden` list, `size` and the full `input`.

Plain-English refusals: a bad setting is an immediate **400** for `check` and `fit` (*"The activation must be one of: silu, tanh, ..."*, `input` errors worded as `/api/fpca/fit` does: *"The basis must be one of: ..."*, *"Maximum likelihood is only available for the Gaussian-process basis ..."*).  Problems found while the data are read are errors of the **job**: absorbance asked for but not loaded (*"No absorbance spectra are loaded yet ..."*), fewer than 12 samples (*"Only 9 samples are included: an autoencoder needs at least 12 samples to be trained honestly. Add more samples, or use the PCA / functional PCA pages for very small data sets."*), more codes than inputs.  `vae` options with `family: "ae"` are ignored (not refused).

## 2. Endpoints

### 2.1 `GET /defaults`  — facts and advice (fast; synchronous)

Query: `variables` (comma list, default `fluorescence`; `absorbance` alone and `fluorescence,absorbance` are allowed: the recommended request models exactly these), `min_coverage` (0.05–1, default 0.9), `abs_processed` (default true), `advise` (0 | 1, default 0: with 1 the fPCA-style advice and the number of components of the functional PCA are computed too: a few seconds).

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
| `recon` | `{"error": (n,), "r2": (n,), "rmse": (n,), "rmse_by_block": {name: (n,)}, "error_by_block": {name: (n,)}, "heldout": (n,) bool (true = the network did not train on it), "limit": float (default level), "limit_level": "99", "limits": {"95": .., "99": ..}, "limit_words", "flag": [word per sample], "typicality": (n,) p-value, "kl": (n,) \| null, "elbo": (n,) \| null}` — `error` is the share of the typical sample-to-sample variation that the network does **not** rebuild for that sample (0 = perfect, 1 = as bad as the mean), `r2 = 1 - error`, `rmse` in the units of the data (after the pre-treatment) |
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
 "blocks": [{"name": "fluorescence", "kind": "surface",
             "measured": image (n_ex, n_em), "reconstruction": image, "residual": image, "mean": image,
             "mask": uint8, "ex": [...], "em": [...], "units": "RU", "rmse": 0.004, "error": 0.02},
            {"name": "absorbance", "kind": "curve", "x": [...], "measured": [...], "reconstruction": [...], "residual": [...], "mean": [...], "units": "absorbance", "rmse": ..., "error": ...}],
 "text": "plain English summary"}
```
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
