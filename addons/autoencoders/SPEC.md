# Add-on "Autoencoders" — specification

*The first FluoViz add-on.  It is NOT part of the FluoViz repository and is NOT installed with FluoViz: the person presses **Install** in Settings → Add-ons, which downloads this add-on and the extra libraries (PyTorch, CPU build) into FluoViz's private environment.  The framework it plugs into is `fluoviz/docs/modules/addons.md` (read it first: manifest, `register(api)`, hooks, the browser contract).  Binding for everyone who works on this add-on; change it only by telling the lead.*

Folder layout (this repository, `addons/autoencoders/`):

```
manifest.json  SPEC.md
fluoviz_addon_autoencoders/
    __init__.py            register(api): routes, saved kind, result kind          (A2)
    core/                  torch numerics: data -> coordinates, models, training, worker, diagnostics     (A1)
    kinds/                 numpy inference + the saved-model kind `autoencoder`     (A2)
    server/                the router /api/addon/autoencoders/...                    (A2)
web/                       page.js, components/*.js, help.js, css                    (A3)
docs/                      api.md (A2), ui.md (A3), user-facing notes
tests/
```

## 1. What the person gets

A page **Autoencoders (add-on)** in *Experimental methods*, a guided pipeline like the Functional PCA page, to:

1. learn a **latent code** (a handful of numbers per sample) from EEMs, absorbance, or both, with a **deterministic autoencoder (AE)** or a **variational autoencoder (VAE)**;
2. do it on the **observed values** (pixels) or — the **functional versions** — on the **coefficients of a basis** chosen with the same machinery as the Functional PCA (Gaussian-process / B-spline / cosine basis, the same smoothing, the same masked domain), with a **smoothify** setting that makes the reconstructions smoother;
3. **check** the result honestly (does it beat PCA / fPCA with the same number of codes? how many codes are needed? is it stable? is it overfitting a handful of samples?);
4. **explore** it (latent maps, what each code does to the EEM, mixing two samples, drawing new samples from a VAE, anomaly scores);
5. **use** it everywhere else: the codes are variables (Plots, metadata links, **Clustering & LDA**), the network can be **saved** and applied to new samples (also inside classification chains), exported, and described in a methods paragraph.

Everything in plain English for scientists who do not program.  The page is honest about small data sets (a few dozen EEMs are the normal case).

## 2. Model families and options

| option | values (defaults in bold) | meaning |
|---|---|---|
| `family` | **`ae`**, `vae` | deterministic / variational |
| `latent_dim` | integer, default from the advice (§6), 1–32 | size of the code |
| `hidden` | list of widths, preset **"small"** `[64, 32]`, "medium" `[128, 64]`, "large" `[256, 128, 64]`, or custom | encoder = Linear stack to the code; decoder mirrors it |
| `activation` | `silu`, `tanh`, `gelu`, `relu`, `softplus`, `elu`, `linear` (= PCA-like, the baseline) | smooth activations (`silu`, `tanh`) are the default: the functions are smooth |
| `dropout` | **0.0**–0.5 | on the hidden layers of the encoder (never at inference) |
| `weight_decay` | **1e-3** (AdamW) | |
| `corruption` | `{noise: 0.0, mask: 0.0}` — denoising: Gaussian noise (in units of the data SD) and random dropping of inputs | makes the code robust and handles missing pixels |
| `vae.beta` | **1.0**, ≥ 0 | weight of the KL term (β-VAE); β < 1 = sharper reconstructions, β > 1 = more regular codes |
| `vae.warmup` | **epochs of KL annealing** (default 20 % of the epochs) | avoids posterior collapse |
| `vae.free_bits` | **0.0** nats per latent dimension | |
| `vae.sigma` | `"learned"` (**default**, one scalar noise SD) or a number | Gaussian likelihood of the data given the code |
| `vae.prior` | **`normal`**, `mixture` with `n_components` K | `mixture` = VaDE-style Gaussian-mixture prior: the VAE then **clusters** the samples (labels + memberships published as variables, assignment of new samples) — second priority |
| `smoothify` | **0.0**, ≥ 0 (functional versions) | §3.3 |
| `smoothify_metric` | **`smooth`** (the representation's own smoothness measure), `ridge` (plain L2 of the coefficients) | §3.3 |
| `gap_weighting` | **`observed`** (a sample that lacks part of the domain counts in proportion to the share that was measured), `equal` | functional versions; §4 "Missing values" |
| `epochs` | **600** with early stopping | |
| `patience` | **60** epochs without improvement of the validation loss; the best weights are restored | |
| `stop_rule`, `min_epochs`, `min_delta`, `val_window`, `max_minutes`, `restore_best` | **`validation`**, **0**, **0.001** (relative), **1**, **none**, **true** | the advanced controls of *when training stops* (step 6): the held-out loss (default), the training loss or nothing (all epochs) decides; training never stops before `min_epochs`; an improvement must be at least `min_delta` of the best loss; the loss may be averaged over `val_window` epochs; a time limit per network; the best (or the last) weights are kept.  The defaults are the training of release 1.0 |
| `lr_schedule`, `lr_final` | **`cosine`** over `epochs`, **0.01** | `cosine_run` plans the decay over the epochs each network is going to train, `constant`, `plateau` halves the rate when the loss stalls (floor `lr_final`) |
| `final_epochs`, `final_epochs_min` | **`auto`**, **20** | k-fold validation: the final networks train for what the folds found (at least `final_epochs_min`), for `epochs` (`max`) or a chosen number |
| `val_fraction` | **0.2** (at least 5 samples); for fewer than 30 samples k-fold validation (`folds` 5) is used for the diagnostics and the final network is fitted on all samples with the number of epochs found | |
| `lr`, `batch_size` | **1e-3** (Adam, cosine decay), **min(32, n)** | |
| `n_restarts` | **3**: the network is trained from 3 random starts, the best validation loss is kept, the spread is reported | |
| `seed` | **0** | everything reproducible (torch seeds, deterministic algorithms, the data split) |
| `input` | `{"variables": ["fluorescence", "absorbance"], "transform": {...}, "representation": {...}}` | same vocabulary and the same validation as the Functional PCA request (`fluoviz/docs/modules/fpca.md`) |

## 3. Coordinates, losses and `smoothify`

### 3.1 The data (shared with the Functional PCA)

Use `fluoviz.analysis.fpca` as it is: `variables.fluorescence_variable` / `absorbance_variable` (pre-treatments, coverage, the masked domain), `domain.Domain` (quadrature weights `w`, summing to 1, so that sums over the domain are L2 integrals), `pipeline.resolve_basis` / `bases.build_representation` (GP-KL / B-spline / cosine / pixels), `fit.prepare_element` (centred smoothed coefficients `C`, frame coordinates `Z`, the fixed smoothing parameters).  **Do not copy this code; import it** (it is a stable interface of FluoViz ≥ 1.0).  The functional steps of the page call the existing `/api/fpca/*` endpoints (`basis/check`, `gp/spectrum`, `gp/estimate`, `basis/smooth`...) unchanged.

### 3.2 Network coordinates

* **Pixels** (`representation.kind == "pixels"`): per variable `v` the observed values `y` (NaN = not observed) minus the training mean function `ȳ_v`, divided by one scalar `s_v` (the RMS of the L2 norm of the centred samples, so that variables count equally, as the "variance" weights of the multivariate fPCA); NaN → 0 at the input, the loss ignores unobserved pixels.
* **Functional** (GP / B-spline / cosine): per variable the frame coordinates `Z` (`Z = C·R`, Euclidean = L2 of the functions) of the centred smoothed coefficients, **standardised** to `ξ = Z / d`, where
  * GP basis: `d_k = sqrt(σ_f² · λ_k)` = the prior standard deviation of coefficient k (`rep.prior_lam`, amplitude `σ_f²` from the fit): every coordinate is N(0,1) under the Gaussian-process prior;
  * other bases: `d_k = s_v` (one scalar, as for pixels).
  Blocks of several variables are concatenated; each block has weight 1 (variables count equally).

### 3.3 Losses (the same L2 metric in both modes)

* **Reconstruction** `L_rec = Σ_blocks ‖z − ẑ‖² / V_v`, i.e. the squared L2 distance between the function and its reconstruction, `V_v` the total variance of the block: so `1 − L_rec` is the **fraction of variance explained** and is directly comparable with the explained variance of the PCA / fPCA with the same number of components.  In functional mode with a GP basis `‖z − ẑ‖² = Σ d_k² (ξ_k − ξ̂_k)²` (a weighted MSE in the standardised coordinates); in pixel mode the pixel weights `w_j` are the domain quadrature weights.
* **VAE**: `L = L_rec / (2σ²) + log σ·(number of coordinates)… + β · KL(q(z|x) ‖ p(z))`, with σ² learned (scalar) by default; free bits and warm-up as above.  The exact normalisation is the author's, but it must be documented and tested (KL ≥ 0, the ELBO improves on average when β is lowered…).
* **smoothify** `s ≥ 0` (default **0**, off): `L += s · P(ξ̂)`, `P = mean over the batch of ξ̂ᵀ Q ξ̂ / K` — **an L2 penalty on the coefficients the decoder produces**, so that big coefficients, above all in the high-frequency directions, are discouraged and the reconstructions move towards smooth functions.  `Q` is the representation's **smoothness metric** in the standardised coordinates:
  * **GP basis**: `Q = I` — because the coordinates are standardised by the prior standard deviations, `ξ̂ᵀξ̂ = Σ ẑ_k²/(σ_f²λ_k)` is the squared RKHS norm of the reconstruction, i.e. the Tikhonov / MAP penalty of the Gaussian process: rough components (small λ_k) cost the most;
  * **B-splines**: the P-spline roughness `Q ∝ Dᵀ(Σ_j τ_j P_j)D` mapped to the standardised coordinates (the penalties `rep.penalties` and the fitted `τ`), normalised so that `tr Q = K`;
  * **cosines**: `Q = diag((1 + k²)^q)` normalised likewise (the shrinkage exponent `q` of the representation);
  * **pixels** (no smoothness defined): `Q = I` (a plain ridge on the pixel values) and the page says so.
  `smoothify_metric = "ridge"` forces `Q = I` in the frame coordinates `Z` for every basis (the literal "L2 of the coefficients").  The scale is chosen so that `s = 1` makes the penalty of an average sample comparable to the reconstruction loss of an average sample; the page offers a log slider 0 … 10 and the **smoothify scan** (§5).
  The penalty is on the decoder OUTPUT (the reconstruction side) and is therefore also part of the VAE loss.

## 4. Training: process, defaults, safeguards

* **Training runs in a separate process** (`multiprocessing`, `spawn` start method: works on Windows and macOS): crashes, memory and cancellation are isolated, and **the FluoViz server never imports PyTorch** (importing it takes seconds and ~300 MB).  The worker reports every epoch (train / validation loss, reconstruction, KL, smoothify term, β, σ², learning rate) to a small file in `runtime/tmp/` that the page polls (a live chart), and returns the weights as numpy arrays.  Cancel = terminate the process, nothing half-saved.
* `torch.set_num_threads(min(4, cores))` (the environment variable `FLUOVIZ_BLAS_THREADS` wins); deterministic algorithms on; float32 training, float64 arrays saved.
* **Small data**: the page shows `n` samples, number of parameters, `n / params`; warns below 10 samples per latent dimension and when parameters ≫ n ("the network can memorise the samples"); the defaults (small network, weight decay, early stopping, validation split, 3 restarts) are chosen for tens to a few hundred samples.  Never silently train on fewer than 12 samples (message).
* **Missing values**: pixel mode — masked loss + input filled with 0 (the error of a partly observed sample is scaled up to the whole block).  Functional mode — the coefficients of a sample that lacks part of the domain (a rectangle excised from it, per sample: FluoViz `docs/modules/models_core.md`) are estimated from the pixels it has **and from what the other samples show about the part it lacks** (`input[name].gaps` `auto`, the gap-aware estimate of the Functional PCA, `docs/modules/fpca.md` §2b; `plain` smooths every sample on its own pixels), and the saved model reads a new sample with a gap the same way (the frozen element carries the estimator).  An estimated sample is less certain, so with `gap_weighting` `observed` (default) it counts in the loss in proportion to the share of the domain that was measured (the mean over the functional variables, at least 5 %, weights averaging 1; the PCA start and the linear baseline use the same weights); `equal` counts every sample alike.  Nothing changes when no sample lacks 0.5 % of a domain.  A large, important region still biases the codes of the samples that lack it, whatever is done; the result and the sample view say which samples lack what (`missing_part`).
* **Honest numbers**: every quality number is reported on **validation** samples (never the training ones), together with the training value.
* **Reproducibility**: seed, versions of torch / FluoViz / the add-on, the full request, the split, the epoch of the best weights are stored in the result and printed in the **methods paragraph**.

## 5. Diagnostics and tools (priority: **M** must · **S** should · **C** could)

**Checks (step 8)**
* **M** Training curves (train / validation total, reconstruction, KL, smoothify, β, σ²), the epoch of the best weights, overfitting verdict in words.
* **M** Reconstruction quality: fraction of variance explained on validation samples (overall and per variable), RMSE in the units of the data, the worst-reconstructed samples, **local R² map** (like the fPCA), residual maps.
* **M** **Baseline comparison with PCA (pixels) / fPCA (functional) with the same number of components** on the same validation samples: "Is the nonlinearity worth it?" — table + plain-English verdict (a linear AE ≈ PCA is the sanity check).
* **M** **Latent-dimension scan**: validation error for d = 1…D for AE (and VAE) and PCA/fPCA (a job; restarts reduced), the recommended d with the reason (smallest d within one SE / 1 % of the best).
* **M** **Latent-space health**: VAE — KL per dimension, **inactive (collapsed) dimensions**, correlation between codes, aggregate posterior vs prior (coverage); AE — spread, effective dimension (participation ratio), unused (constant) code dimensions.
* **S** **Stability** across restarts / bootstrap: alignment of the codes between runs (Procrustes / canonical correlations), variation of the reconstruction; verdict "the codes are / are not reproducible".
* **S** **Smoothness**: roughness of reconstructions vs inputs (RKHS norm for the GP basis, curvature otherwise); the **smoothify scan** (reconstruction error against roughness for s = 0 … 10: an L-curve) with a suggested value.
* **S** Calibration of the VAE (posterior SDs vs reconstruction error of the codes' means: "are the uncertainties believable?").

**Exploration (step 7)**
* **M** Latent scatter (any two codes; PCA of the codes when d > 2), coloured by any variable (metadata, QC, other results), hover with the EEM thumbnail; VAE: posterior SD ellipses.
* **M** **Latent traversals**: decode along each code (mean ± 1, 2, 3 SD of the training codes, others at their mean) → EEM images / curves, the difference to the mean EEM.
* **M** **Sensitivity (Jacobian) images**: ∂(output)/∂(code j) at the mean code and the average magnitude over samples ("what this code does", the nonlinear analogue of a loading).
* **M** Per-sample view: original, reconstruction, residual (EEM images, absorbance curves), code, error, anomaly status.
* **S** **Nonlinear effects** (tab "Nonlinear effects"; `api.md` 2.15): *which nonlinear trend does each code fit?*  Four tools on the decoder, all along the traversal of a code (SD of the training codes, the others at their mean): **response curves** (a feature of the decoded EEM — the pixel the code moves most, the overall intensity, a typed or clicked pixel / box, an absorbance wavelength, a network coefficient — against the code, with the training samples as dots, the curves of single samples with the other codes at their own values, and the shape of every curve in words with the weighted R² of a straight line); **where it bends** (the nonlinearity index `NL = 1 − R²` of a weighted straight line at every pixel, the amplitude image, and per code the `nonlinear_share` = Σ w SSE / Σ w SST in the metric of the loss); **local effects** (the analytic Jacobian of the decoder at chosen samples, its pixel-wise standard deviation over all the training samples, and per code the mean of `1 − cosine` between the loading at a sample and at the mean code); **two codes** (a 3 × 3, 5 × 5 or 7 × 7 grid, the interaction image `F(a,b) − F(a,0) − F(0,b) + F(0,0)`, the interaction index and a heat map of a feature with the samples as dots).  Pure numpy (the server never imports PyTorch); works for AE and VAE (posterior means), pixel and functional networks, fluorescence and / or absorbance, and follows the setting Axes of fluorescence maps.
* **S** **Interpolation** between two samples in code space (slider) next to the plain linear mixture of the two EEMs.
* **S** **Generation** (VAE): draw new samples from the prior (n, seed, temperature), nearest real sample, download as `.npz` / CSV, **always labelled "synthetic"**.
* **S** Codes against metadata: correlation heat map codes × numeric variables; "predict the variable from the codes": cross-validated R² against the same from PCA scores.
* **S** Anomaly view: per-sample reconstruction error (+ ELBO / latent typicality for the VAE), robust thresholds (median + c·MAD, quantile), list of flagged samples with the residual maps.
* **C** Mixture prior (VaDE): clusters, memberships, cluster means decoded to EEMs.
* **C** Conditional VAE (condition on a categorical variable such as site) — not in 1.0.

**Use (step 9)**
* **M** Published variables of the result (they appear in Plots, Metadata links and every selector, so the codes can be **clustered and classified on the Clustering & LDA page**): `z1…zd` (AE: the code; VAE: the posterior mean), `recon_rmse`, `recon_r2`, `flag` (anomaly status), VAE also `z1_sd…` (≤ 12 dims), `kl`, `elbo`; mixture prior also `cluster`, `membership`.
* **M** Buttons: **Cluster the latent space** (`#/cluster?keys=res:<id>:z1,...`), **Classify with the latent space** (same, opens the separation tab), **Save this model**, **Apply a saved model** (link), downloads (codes CSV, reconstructions `.npz`, training history CSV, synthetic samples), **methods paragraph** (copy).
* **M** **Saved model** `autoencoder` (§7) — applying it gives codes, reconstruction error and statuses for new samples; it can be the first link of a classification chain.

## 6. Advice (the defaults the page starts from)

`GET /defaults` returns facts and advice from the data: n samples, pixels / coefficients, the fPCA-style recommended representation (GP-KL, Matérn 5/2, ML smoothing), the number of components that explain 99 % by PCA / fPCA, a starting `latent_dim` = min(that number, `n/10`, 8) (at least 2), the size preset (`small` unless n > 300), the time estimate, and warnings in words (n small, strong missing data, a mask that changes between samples).  **Run with the recommended settings** keeps the person's choice of variables and fills everything else (functional on, GP basis, AE, small, smoothify 0).

## 7. Network specification and the saved model

### 7.1 `ModelSpec` and arrays (what the worker returns, what the saved model stores)

```json
{"family": "ae" | "vae", "input_dim": P, "latent_dim": d,
 "encoder": {"hidden": [64, 32], "activation": "silu"},
 "decoder": {"hidden": [32, 64], "activation": "silu"},
 "output_activation": "linear",
 "vae": {"prior": "normal" | "mixture", "n_components": K, "beta": 1.0, "sigma": 0.12},
 "blocks": [{"name": "fluorescence", "slice": [0, P1], "kind": "gp" | "bspline" | "cosine" | "pixels", "weight": 1.0}, ...]}
```
Arrays (float64, PyTorch `nn.Linear` convention: `W` has shape `(out, in)`): `enc.W0, enc.b0, …`, `enc.mu.W, enc.mu.b` (AE: the code layer), `enc.logvar.W, enc.logvar.b` (VAE only), `dec.W0, dec.b0, …, dec.out.W, dec.out.b`; mixture prior: `prior.pi (K)`, `prior.mu (K,d)`, `prior.logvar (K,d)`; input standardisation: `in.mean (P)`, `in.scale (P)`; `out.weights (P)` (the L2 metric weights).  **Inference is pure numpy** (`kinds/forward.py`): activations `silu, tanh, gelu (exact, erf), relu, softplus, elu, linear` with tested parity to torch (1e-5 relative); dropout and noise are off at inference.  A VAE's code = the posterior mean `μ`; `σ` = `exp(logvar/2)`.

### 7.2 Kind `autoencoder`

Registered with `api.add_saved_kind("autoencoder", "Autoencoder", AutoencoderApplier, result_kinds=("autoencoder",), freezer=freeze_autoencoder)`.  `meta["addon"] = {"id": "autoencoders", "version": ...}`.  Contents: the `ModelSpec` + arrays; **how a sample becomes network input** — for pixels the domain (observed-pixel mask, weights), the training mean function and scale; for the functional versions the frozen representation(s) exactly as the Functional PCA kind freezes them (reuse `fluoviz.analysis.frozen.kinds.fpca`: `FrozenElement`, `Scorer`, with the fixed smoothing parameters, never re-estimated) followed by the standardisation `ξ = Z/d`; the recipe (`capture_recipe`); `reference`: training ids, codes, reconstruction errors, the error distribution, the limits; `validation`: the validation metrics; `options` (thresholds); `outputs`; `description`.  **Apply**: new samples → coordinates (fixed pre-treatment and smoothing) → encoder → codes (+ decoder → reconstruction error in the L2 metric).  Published variables as §5 (`z1…`, `recon_rmse`, `flag`…) plus `status`: `outside` when the reconstruction error is above the limit (default: the 99 % point of the validation errors, option) or, for the VAE, the code is atypical (Mahalanobis distance of μ under the aggregate posterior, χ² p < 0.01) — with the reasons in words; `unusable` below `min_coverage`.  The generic payload (`plot`, `diagnostics`, `limits`, `images`/`curves`, `notes`) as in `docs/modules/models.md`.  A classifier built on `z1…` embeds this model as a chain part like any other kind.

## 8. API (`/api/addon/autoencoders/…`; exact shapes in `docs/api.md`, written first by A2)

`GET /defaults` · `POST /check` · `POST /fit` (job; the numbered result `autoencoder-N`; progress file) · `GET /fit/{job}/history` · `GET /{rid}` (payload for the page) · `GET /{rid}/sample/{sid}` · `POST /{rid}/traverse` · `POST /{rid}/interpolate` · `POST /{rid}/generate` · `GET /{rid}/sensitivity` · `POST /{rid}/response` · `POST /{rid}/bends` · `POST /{rid}/local` · `POST /{rid}/pair` (the nonlinear effects of the codes) · `POST /scan` (job: `latent` | `smoothify` | `stability` | `baseline`) · `GET /{rid}/export?what=codes|reconstruction|history|synthetic` · `GET /{rid}/methods` — plus the generic `/api/models/...` (saved models) and the existing `/api/fpca/*` for the representation steps.  Errors in plain English.  Staleness, recompute-in-place and `ensure_run` like the Functional PCA result (the network is NOT re-trained silently: a stale result says what changed and offers **Train again with the same settings**).

## 9. The page (`web/page.js`) — labels the Help guide quotes (`docs/ui.md` by A3 lists them exactly)

Title **Autoencoders (add-on)**; the stepper **1 · What to model**, **2 · Domain**, **3 · Representation**, **4 · Smoothing**, **5 · Network**, **6 · Train**, **7 · Results**, **8 · Checks**, **9 · Use**; each step has the "what this step does and what to check" box; buttons **Run with the recommended settings**, **Train the network**, **Cancel**, **Train again with the same settings**, **Cluster the latent space**, **Classify with the latent space**, **Save this model**.  Steps 1–4 reuse the Functional PCA components (parameterised, backward compatible edits to `FpcaStep*.js` are allowed); step 5 offers the families as cards (Autoencoder / Variational autoencoder), sizes, the denoising and VAE options, **smoothify** with its explanation and the **smoothify scan**; step 6 shows the estimate, the live training chart and the cancel button; step 7 has the tabs **Latent space**, **Traversals**, **Nonlinear effects** (the sections **Response curves**, **Where it bends**, **Local effects**, **Two codes**), **Samples**, **Interpolation**, **Generate** (VAE only); step 8: **Training**, **Quality**, **Latent dimensions**, **Stability**, **Smoothness**; step 9 the buttons above, downloads and the methods text.  Follows `docs/FRONTEND.md` (tokens, light / dark, 600 px, `.rds` / `.fig` export buttons on every chart, robustness).

## 10. Tests (add-on repository `tests/`; run with `scripts/run_tests.sh` of FluoViz with `FLUOVIZ_ADDONS_DEV` pointing here; skipped when PyTorch is missing)

Numerics (A1): linear AE ≈ PCA; the functional AE on a GP basis ≈ fPCA explained variance; `smoothify` lowers roughness monotonically; VAE KL ≥ 0 and collapse detection; determinism with a seed; masks / missing pixels; cancellation; torch ↔ numpy parity.  Kind and API (A2): freeze → file → thaw → apply reproduces the codes of the page; another project / coarser grid; planted anomaly → `outside`; chains (LDA on `z1…`); every endpoint with plain-English errors; the base program never imports torch (framework test).  Page (A3): an e2e flow `autoencoders` (installs the add-on in dev mode, runs the pipeline on the example data, light / dark, 600 px, no console errors).

## 11. Distribution

`scripts/build_addon.py` makes `dist/autoencoders-<version>.zip` (one top-level folder, no `__pycache__`), its sha256 and `dist/index.json` (§ index of `addons.md`); the zip and the index are the assets of a release of the add-ons repository; the FluoViz **Install** button downloads them.  CI of this repository: install FluoViz, install the add-on from the built zip through the framework on Linux, Windows and macOS, run the tests.
