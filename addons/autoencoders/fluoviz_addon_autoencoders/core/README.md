# `core/` — the numerical core of the Autoencoders add-on (A1)

API for the saved kind / router (A2), the page (A3) and the Help guide (A4).  Binding together with `SPEC.md`; the deviations from the SPEC are listed in section 7.
**Rule: the FluoViz server never imports PyTorch.**  Which module may be imported where:

| module | imports | who uses it |
|---|---|---|
| `spec.py`, `forward.py`, `linear.py`, `tools.py`, `diagnostics.py` (everything except the three scan *jobs*), `advice.py`, `curves.py`, `features.py`, `nonlinear.py` (the response curves, features and effects of the tab Nonlinear effects: `docs/api.md` 2.15) | numpy (+ scipy) — **no torch** | server, saved kind, tests |
| `data.py` | numpy + FluoViz (`fluoviz.analysis.fpca`, `fluoviz.analysis.frozen.kinds.fpca`) — **no torch** | server (`prepare`), saved kind (`AEFrame`) |
| `train.py` | torch, **imported lazily inside functions** (the module itself imports without torch: `train.make_split`, `train.read_history` are fine in the server) | the worker process, tests |
| `worker.py` | multiprocessing (spawn); the child imports `train` | the server calls `run_in_worker(...)` |

Everything a person can read (errors, verdicts) is plain English; errors the person can fix are `AutoencoderError` (a `ValueError`).

---

## 1. `spec.py` — what is chosen, what a network is

```python
from fluoviz_addon_autoencoders.core.spec import Config, ModelSpec, AutoencoderError, pack_model, unpack_model, ACTIVATIONS, SIZES, MIN_SAMPLES

cfg = Config.from_dict(body)              # unknown key -> AutoencoderError("Unknown setting 'foo'."); every value checked ("The number of codes (latent dimension) must be between 1 and 32.")
cfg.to_dict() / cfg.to_json() / Config.from_json(s)     # JSON round trip (the stored `params`)
cfg.replace(vae={"beta": 4.0}, latent_dim=5)             # copy with changes (nested vae / corruption dictionaries are merged)
cfg.hidden_widths() / cfg.decoder_widths()               # [64, 32] / [32, 64] ("small" | "medium" | "large" | a list)
cfg.warmup_epochs()                                      # KL warm-up (default 20 % of the epochs; 0 for an AE)
cfg.batch_for(n_train)                                   # the batch size chosen, else min(32, max(8, n_train // 6))
cfg.check_samples(n)                                     # AutoencoderError below 12 samples unless cfg.allow_few_samples  ("... needs at least 12 samples to be trained honestly ...")
```
`Config` fields (SPEC §2): `family latent_dim hidden activation dropout weight_decay corruption{noise,mask} vae{beta,warmup,free_bits,sigma,prior,n_components,n_eff} smoothify smoothify_metric epochs patience val_fraction folds lr batch_size n_restarts seed input`
+ `skip=True` (the linear shortcut, §4), `kfold_below=30`, `allow_few_samples=False`, `grad_clip=5.0`, `threads=None`.  `input` is free-form (the data vocabulary of `data.prepare`).

```python
spec = ModelSpec.from_config(cfg, input_dim, blocks, sigma=None, n_eff=None) / ModelSpec.from_dict(d) / .to_dict() / .to_json()
spec.array_shapes()           # {"enc.W0": (64, P), "enc.b0": ..., "enc.mu.W", "enc.logvar.W" (VAE), "dec.W0" ..., "dec.out.W", "enc.lin.W" (d, P) + "dec.lin.W" (P, d) (skip), "prior.pi|mu|logvar" (mixture), "in.mean", "in.scale", "out.weights"}
spec.check_arrays(arrays)     # -> float64 dict; AutoencoderError in plain English for a missing / unexpected / mis-shaped / non-finite array
spec.n_parameters(); spec.is_vae(); spec.has_mixture(); spec.block_slices(); spec.skip
pack_model(spec, arrays, extra) -> dict for np.savez (spec_json, extra_json, "a.<name>" arrays; no pickles);  unpack_model(npz) -> (spec, arrays, extra)
```
`ModelSpec` JSON = SPEC §7.1 plus two optional keys: `"skip": true|false` (absent = false: older models) and `vae.n_eff` (N of the likelihood, needed for the ELBO of a sample).
Blocks: `[{"name": "fluorescence", "slice": [0, P1], "kind": "gp"|"bspline"|"cosine"|"pixels", "weight": 1.0}, ...]`.

## 2. `forward.py` — the network in pure numpy (deterministic; parity with torch tested to 1e-5, with and without `skip`)

```python
from fluoviz_addon_autoencoders.core import forward as fw
mu, logvar = fw.encode(spec, arrays, X, M=None)       # X (n, P) NETWORK COORDINATES; NaN or M False = not observed (set to the mean). AE: logvar is None. VAE: the code is mu.
Xhat       = fw.decode(spec, arrays, Z)               # Z (n, d) -> (n, P) in network coordinates
Xhat, mu, logvar = fw.reconstruct(spec, arrays, X, M=None)     # decode(mu): deterministic
fw.sample_codes(mu, logvar, rng, n_draws)             # draws from the posterior
err, per_block = fw.recon_error(spec, arrays, X, Xhat, M=None)   # per sample, in the metric of the loss: 0 = perfect, ~1 = as bad as the mean (1 - err = share rebuilt)
null, per0     = fw.null_error(spec, arrays, X, M=None)           # the same for "reconstruction = the mean"
fw.weighted_error(X, Xhat, omega, blocks, M)          # the same with the weights and blocks given directly
fw.memberships(spec, arrays, mu, logvar=None)         # mixture prior: (n, K) cluster probabilities;  fw.prior_log_density(spec, arrays, z)
fw.Network(spec, arrays)                              # the same, arrays checked once: .encode .decode .decode_with_trace .standardise
fw.activation(name) / fw.activation_grad(name)        # silu tanh gelu(erf) relu softplus elu linear
```
Network input `u = (x - in.mean) / in.scale`; code `mu = MLP_enc(u) + enc.lin.W u` (shortcut only with `skip`); output `xhat = (MLP_dec(z) + dec.lin.W z) * in.scale + in.mean`.
In trained networks `in.scale_j = 1 / (c sqrt(omega_j))` (the coordinates of the L2 frame, `c^2` = components for 99 % of the variance), `in.mean` = the training mean, `out.weights = omega`.
**The shortcut passes THROUGH the code**: input -> `enc.lin.W` (d numbers) -> code -> `dec.lin.W` -> output.  There is no path from the input to the output that avoids the d-dimensional code, so the bottleneck stays honest (the network cannot copy a sample); `skip=False` is the classical pure autoencoder.

## 3. `data.py` — from the project's functional variables to network coordinates and back

```python
from fluoviz_addon_autoencoders.core import data as D
prep = D.prepare(variables, input, progress=None, seed=0, ife=None)    # -> Prepared
```
* `variables`: list of `fluoviz.analysis.fpca.variables.Variable` (fluorescence and/or absorbance) built as the Functional PCA router builds them (`make_variables`); **for a `pixels` block build the fluorescence variable with `complete=False`** (gaps inside the domain stay NaN).  `D.variables_from_arrays(X, ex, em, ids, A, wl, abs_ids, input)` builds them from arrays (tests, scripts).
* `input`: the Functional PCA vocabulary, any of `{"fluorescence": {"basis": {...}, "smoothing": {...}}}`, `{"basis": {"fluorescence": {...}}, "smoothing": {...}}`, `{"representation": {"kind": "pixels"}}`, `{"functional": False}` (= pixels).  `basis.kind`: `gp` (default; `lengths` "auto" | "ml" | [..]), `bspline`, `cosine`, `pixels`.  A pre-treatment named in `input[name]["transform"]` must equal the one the variable was built with.  Blocks may be of different kinds.
* `ife`: the absorbance clean-up dict the data used (kept for the saved model).

`Prepared`: `ids, n, P, mode ("pixels"|"functional"|"mixed"), names, blocks[Block], X (n,P), M (n,P)|None, frame (AEFrame), input (resolved), sd, notes, dropped`.
`Block`: `name kind sl K domain units transform scale omega variance q_smooth q_ridge Y ybar element rep smoothing notes`.

```python
prep.train_data(smoothify_metric="smooth") -> TrainData     # numpy only: X, M, omega, blocks, q, sd, ids, info (grid), q_rough — this travels to the worker;  TrainData.from_array(X) for plain arrays
prep.to_functions(Xhat) -> {name: (n, m)}                    # functions at the observed pixels, in the units AFTER the pre-treatment (embed with prep.embed(name, F) -> grid, NaN outside the domain)
prep.observed() / prep.smoothed() / prep.mean_functions()    # measured (NaN = not observed) / the smoothed functions the network rebuilds (pixels: = measured) / mean functions
prep.summary() -> dict                                       # facts for the "check" step (per variable: mode, basis, n_inputs, n_pixels, variance, smoothing, missing_fraction, notes)
prep.frame                                                   # AEFrame (below)
D.curvature_operator(domain)                                 # sparse second-difference operator of a domain (the smoothness of the function itself)
```
**Network coordinates** (SPEC §3.2): pixels `x = (y - ybar) / s` (NaN -> 0, `M` False); functional `xi = Z / d` (`Z = C R`, the L2 frame; GP: `d_k = sqrt(sigma_f^2 lambda_k)`, others: one scalar).  **Loss weights** `omega` (per coordinate): `L_rec = sum_j omega_j m_j (x_j - xhat_j)^2` is the squared L2 distance between the function and its reconstruction divided by the total variance of the variable, **averaged over the variables** (blocks count equally, weights add up to 1): 0 = perfect, 1 = as bad as the mean, **`1 - L_rec` = share of the variation rebuilt**, and the weighted PCA of the coordinates has exactly the explained variances of the fPCA / multivariate fPCA of FluoViz (tested).  A block with unobserved coordinates is scaled up to the whole block (coverage).
**Smoothness metric `Q`** (for `smoothify`, per block, in the standardised coordinates): GP `Q ∝ I` (the RKHS norm / Tikhonov penalty of the Gaussian process); B-splines and cosines `Q = T' A'A T` (the second-difference roughness of the function itself on the pixel grid, `T` the unit-frame functions at the pixels, `A = curvature_operator`); pixels `Q ∝ I`; `smoothify_metric = "ridge"`: `diag(d^2)` for GP (a ridge on the frame coordinates), `I` otherwise.  Every `Q` is scaled so that **the penalty `x'Qx/K` of the average sample of the data is 1** (so `smoothify = 1` weighs it like the loss of the mean, the same for every basis, and roughness numbers read "1 = as rough as a typical sample").

**`AEFrame`** (what the saved model keeps; `@serial.allow`-ed when FluoViz is importable, class names are unique): `blocks: [AEFrameBlock(name, kind, K, scale (K,), element: FrozenElement)]`, built with `fluoviz.analysis.frozen.kinds.fpca._element_freeze` (same representation / fixed smoothing / mean function / pre-treatment as the Functional PCA kind).
```python
frame.to_functions(X) -> {name: (n, m)};  frame.mean_functions();  frame.slices();  frame.names();  frame.block(name).domain / .to_function(Xb) / .mean_function() / .treat(Y) / .coordinates(Yt)
frame.coordinates({name: raw values (n, m_domain)}, min_coverage=0.8) -> {"X", "M", "usable" (n,), "why": [[str]], "coverage" (n, n_blocks), "treated": {name: (n, m)}}     # NEW samples: pre-treat, smooth with the FIXED parameters, centre, to the L2 frame, / scale
```

## 4. `train.py` + `worker.py` — training (torch, in a spawn process)

```python
from fluoviz_addon_autoencoders.core import worker as W, train as T
result = W.run_in_worker("fit", {"config": cfg, "data": prep.train_data(cfg.smoothify_metric)}, progress=progress, history_path=path)    # -> TrainResult
```
* `run_in_worker(name, kwargs, progress=None, history_path=None, workdir=None, poll=0.05, timeout=None)`: fresh `spawn` process; `progress(fraction, message)` is called from the waiting thread, **if it raises (Cancelled) the process is terminated and nothing is returned or saved** (the history file gets a last `cancelled` record).  Raises `AutoencoderError` (plain-English problem), `WorkerError(message, detail)` (the process failed / died / timed out: "stopped unexpectedly ... out of memory"), `Cancelled` (`fluoviz.analysis.common.Cancelled`).  `W.torch_available()` looks PyTorch up without importing it.  Jobs (`W.JOBS`): `fit`, `latent_scan`, `smoothify_scan`, `stability_scan`, `selftest`.
* `T.fit(config, data, progress=None, history_path=None)` in-process (tests): `data` = `Prepared` or `TrainData`.
* History file (JSON lines, one line per write; read it while it grows): `T.read_history(path, since=0) -> (records, next_index)`.  Records: `{"event": "start"|"run_start"|"epoch"|"run_end"|"done"|"cancelled"|"error", ...}`; `epoch` records: `restart, fold, epoch, train, val, rec, rec_val, kl, kl_val, smooth, smooth_val, beta, sigma2, lr, train_batch, best` (`train`/`val` = the objective, comparable between epochs; `rec` = share of variation NOT rebuilt (0 = perfect) on the training / validation samples).
* **Protocol**: `n >= kfold_below (30)`: random hold-out split (`val_fraction`, at least 5 samples), `n_restarts` random starts, the start with the best validation loss is kept, early stopping on the validation loss (better by 0.1 %, `patience`), best weights restored.  Fewer samples: k-fold validation (every sample predicted by a network that never saw it = the honest numbers), then `n_restarts` networks on ALL samples for the epochs the folds found.  Everything derives from `cfg.seed` (same seed = same arrays and history, tested).
* **The linear shortcut (`skip=True`, default)**: `enc.lin.W` / `dec.lin.W` run beside the encoder and decoder THROUGH the code.  The **first random start begins at the PCA solution** (shortcuts = the principal directions of the training samples that carry signal, the code has unit variance, the nonlinear branches output 0, a code without signal starts at 0 / at the prior) and the **start itself is a candidate for the best weights** (`best_epoch == -1`: no epoch beat the linear start, the kept network is a PCA / functional PCA), so on held-out samples a network is never worse than the PCA with the same number of codes.  The other starts begin at random weights, so a nonlinearity that really helps is still found.  A linear path also extrapolates (a sample with five times the concentration of the others).  `skip=False`: the classical autoencoder, random start for every restart.
* Loss, `n_eff`, warm-up, free bits, the mixture prior and `smoothify`: documented at the top of `train.py`.  VAE: `L = (N/2)[rec/sigma^2 + log sigma^2] + (N/2) s pen/sigma^2 + beta_t KL`, `sigma^2` = the current reconstruction error when `"learned"`, `N = n_eff` = `min(P, max(15, 3 d))` by default.

`TrainResult` (a dataclass; pickle-able; `result.meta()` is the JSON-able part):

| field | content |
|---|---|
| `spec`, `arrays` | the `ModelSpec` and the float64 arrays of the kept network (save these) |
| `history` | per epoch of the kept run (dicts as above, without `event`); `all_histories`: every run `{"restart", "fold", "history"}` |
| `validation` | `mode` ("holdout"/"kfold"), `n_val`, `evr` (share of variation rebuilt on held-out samples), `evr_train`, `error`, `error_train`, `evr_blocks {name: {val, train}}`, `loss_val`, `loss_train`, `baseline {name: "PCA"/"Functional PCA", k, evr, evr_train}`, `kl` (VAE), `n_eff` |
| `samples` | per sample (all `n`): `codes (n,d)`, `codes_sd (n,d)|None`, `error (n,)` (final network; in-sample for training samples), `error_blocks (n, n_blocks)`, `null (n,)`, `heldout (n,)` bool, `error_heldout (n,)` (NaN where never held out; k-fold: out-of-fold for every sample), `error_baseline (n,)` (the PCA's error on the same held-out samples), `fold (n,)|None`, `kl (n,)|None` |
| `restarts` | `{"n", "runs": [{restart, seed, best_epoch, epochs_run, stopped, chosen, train_loss, evr_train, val_loss, evr_val}], "kept", "mean", "sd", "min", "max"}` |
| `best_epoch` (0-based; **-1 = the PCA start**), `epochs_run`, `split`, `config`, `n_parameters`, `sigma`, `seconds`, `versions`, `notes`, `fold_info`, `restart_codes (R, n, d)` | |

## 5. `diagnostics.py`, `tools.py`, `advice.py`, `linear.py` — final signatures (numpy)

### `diagnostics.py` — checks and ONE set of thresholds and words (the page and the saved kind use these functions; A2's `kinds/stats.py` stays the source of the anomaly limits)

```python
from fluoviz_addon_autoencoders.core import diagnostics as G
G.assess(prep, result, level=0.99, p_min=0.01) -> {"reconstruction", "baseline", "health", "training", "limit", "typicality", "flag", "flag_words", "roughness", ("kl", "elbo", "calibration" for a VAE)}     # one call for the page
G.reconstruction(prep, spec, arrays, rows=None) -> {"X_hat", "codes", "logvar", "error" (n,), "error_blocks", "null", "r2" = 1 - error, "evr", "evr_blocks", "rmse" (first variable), "rmse_blocks" {name: (n,)} (L2 distance to the function the network rebuilds, data units after the pre-treatment), "rmse_observed" {name: (n,)} (to the measured values), "functions" {name: (n, m)}, "local_r2" {name: (m,)}, "residual_rms" {name: (m,)}}
G.worst_samples(ids, error, heldout=None, top=10) -> [{"id", "error", "r2", "heldout"}]
G.evr(err, null) / G.error_se(err, null) / G.pct(x)
G.pca_curve(td, split, kmax) -> {"k", "error_val", "error_val_se", "error_train", "mode"}          # the PCA / functional PCA baseline for k = 0..kmax on the same split (errors = share NOT rebuilt)
G.compare_baseline(result) -> {"name", "k", "evr_network", "evr_baseline", "relative_improvement", "z", "better": "network"|"baseline"|"same", "verdict", "n"}
G.latent_health(spec, arrays, X, M=None, codes=None, logvar=None) -> {"kind", "names", "mean", "sd", "correlation", "effect_share", "inactive", "unused", "collapsed", "kl_per_dim", "effective_dim", "n_active", "coverage", "words"}
G.stability_from_codes(codes (R, n, d)) -> {"n_runs", "matrix", "procrustes", "per_code", "mean", "min", "verdict"}
G.roughness(td, X) -> {"mean": {name: float}, "per_sample": {name: (n,)}};  G.curvature(values, grid)
G.smoothness_report(td, X_input, X_recon, smoothify=0.0, metric="smooth") -> {"smoothify", "metric", "blocks": [{"name", "measure", "input", "reconstruction", "ratio", "verdict"}], "words"}
G.calibration(spec, arrays, X, M=None, n_rep=5, seed=0) -> {"coverage68", "coverage95", "mean_z2", "per_code", "verdict"}      # VAE
G.elbo(spec, arrays, X, M=None) -> {"kl", "error", "elbo"}                                                                   # VAE, with kinds.stats' KL and ELBO
G.default_limit(errors_heldout, level=0.99, fallback=None) -> {"limit", "method", "n", "level", "source", "words", "all": {"95", "99"}}      # = kinds.stats.error_limits
G.typicality(codes_ref, codes, logvar_ref=None) -> p (n,)                                                                      # = kinds.stats.code_stats + typicality
G.robust_limit(errors, method="mad"|"quantile", c=3.0, level=0.99) -> {"limit", ...};   G.flag_samples(error, limit, p_typical=None, p_min=0.01) -> (codes 0..3, words)
G.training_verdict(result) -> {"overfit": "none"|"mild"|"clear", "gap", "overfit_words", "stopping_words", "restarts_words", "quality_words", "words"};   G.quality_verdict(evr_val, n_val, k) -> str
# jobs for the worker (train several networks; run_in_worker("latent_scan", {"config": cfg, "data": td, ...})):
G.latent_scan(config, data, dims=None, families=None, n_restarts=1, progress=None, history_path=None) -> {"scan": "latent", "dims", "families": {"ae": {"val", "val_se", "train"}}, "baseline": {"name", "k", "val", "val_se", "train"}, "recommended": {"latent_dim", "reason"}, "words"}
G.smoothify_scan(config, data, values=None, n_restarts=1, ...) -> {"scan": "smoothify", "values", "error", "error_se", "roughness", "roughness_input", "suggested", "words"}
G.stability_scan(config, data, n_runs=5, bootstrap=False, ...) -> {"scan": "stability", "n_runs", "alignment": {"mean", "min", "per_code"}, "recon_spread", "matrix", "procrustes", "verdict", "words"}
```

**The rules (every threshold and verdict wording of the add-on lives here):**

| what | rule |
|---|---|
| quality words (`quality_verdict`) | held-out share rebuilt `>= 99 %` excellent, `>= 95 %` good, `>= 90 %` fair (+ "More codes may help"), below poor (+ "Try more codes, a larger network or more training epochs") |
| overfitting (`training_verdict`) | held-out error / training error `> 3` **and** difference `> 0.01` of the variation = **clear**; `> 1.6` **and** difference `> 0.003` = **mild**; else none (a five-fold ratio of a tiny error is still none) |
| stopping words | `best_epoch == -1`: "The network could not improve on its PCA starting point on held-out samples: the straight-line (linear) part is the answer ..."; early stop: "stopped at epoch E because the validation loss had not improved for P epochs; the weights of epoch B were restored"; all epochs used and best epoch `>= 90 %` of them: "more epochs may help"; k-fold: epochs taken from the folds |
| random starts | spread of the held-out share rebuilt `< 2` points = "agree", else "differ ... depends on the start"; one start: "may depend on luck" |
| baseline (`compare_baseline`) | paired per-sample errors on the same held-out samples: **network better** = relative error reduction `> 15 %` and paired `z < -2` and a difference of the shares rebuilt of at least 0.3 points; **baseline better** = the mirror image (`z > 2`); otherwise **same** ("a linear model is enough ... the {PCA / Functional PCA} is the safer choice"); fewer than 5 held-out samples: no verdict |
| latent health | a code is **unused** when its share of the effect of the codes on the output (analytic Jacobian of the decoder, in the metric of the loss, times the code SD) is `< 1 %`; for a VAE it is **collapsed** when its KL is `< 0.05` nats; **inactive** = unused or collapsed (or the variance of its posterior means `< 0.01`); effective dimension = participation ratio of the effect shares; two codes with `|r| > 0.5` are named; VAE: `< 85 %` of the samples inside the 95 % region of the prior = "do not fill the prior" |
| smoothness (`smoothness_report`) | ratio roughness(reconstructions) / roughness(samples): `< 0.9` smoother, `0.9-1.1` about as rough, `> 1.1` rougher (suggests the smoothify scan) |
| stability (`stability_from_codes`) | a code is *found again* when the multiple correlation of the reference code with the codes of the other runs is `>= 0.8`; **reproducible** = every code found again and mean canonical correlation `>= 0.95`; **partly** = at least half of the codes found again (the others are named); else **NOT reproducible** |
| calibration (VAE) | 95 % bands hold `85-99 %` of the code errors = believable; below 85 % = overconfident; above 99 % = underconfident |
| anomaly limit | `kinds.stats.error_limits` of the held-out errors (log-median + robust spread; 95 % / 99 %); typicality = `kinds.stats` F-calibrated Mahalanobis p-value; `flag_samples`: 0 inside, 1 error above the limit, 2 unusual code (p < 0.01, VAE), 3 both |
| latent scan | recommended `d` = the smallest whose held-out error is within one standard error (or 1 % of the variation, or 1e-4) of the best |
| smoothify scan | suggested `s` = the largest whose held-out error is within one standard error (or 2 % relative) of the error without smoothing |
| the effective N of the VAE | `n_eff = min(P, max(15, 3 d))` |

### `tools.py` — exploration (numpy; `frame` = `prep.frame` adds the functions, else only network coordinates)

```python
from fluoviz_addon_autoencoders.core import tools
tools.code_stats(codes) -> (mean, sd)                                                       # the unit of a code = SD of the training codes
tools.traverse(spec, arrays, codes_ref, dims=None, values=(-3..3), frame=None) -> {"values", "dims", "mean_code", "sd", "z" (D,k,d), "X" (D,k,P), "X_mean", "functions" {name: (D,k,m)}, "difference" (function minus the decoding of the mean code), "mean"}
tools.jacobian(spec, arrays, Z) -> (n, P, d)                                                # analytic through the decoder (shortcut and output activation included)
tools.sensitivity(spec, arrays, codes_ref, frame=None) -> {"names", "mean_code", "sd", "share" (d,), "effect" (d,), "at_mean", "magnitude"}   # per 1 SD of each code; (d, P) or {name: (d, m)}
tools.interpolate(spec, arrays, za, zb, steps=9, frame=None, Fa=None, Fb=None) -> {"t", "z", "X", "functions", "linear" (the plain mixture of Fa, Fb), "difference"}
tools.generate(spec, arrays, n, seed=0, temperature=1.0, codes_ref=None, frame=None) -> {"synthetic": True, "z", "X", "functions", "cluster", "nearest", "distance"}     # VAE only ("Only variational autoencoders can draw new samples ...")
tools.relate(codes, {name: values}, pca_scores=None, folds=5, seed=0, degree=1) -> {name: {"correlation" (d,), "r2_codes", "r2_pca", "n"}};  tools.correlations(codes, targets);  tools.cv_r2(F, y, ...)
tools.clusters(spec, arrays, mu, logvar=None) -> {"membership", "label", "confidence"};  tools.cluster_means(spec, arrays, frame=None) -> {"z", "pi", "X", "functions"}      # mixture prior
```

### `advice.py` and `linear.py`

```python
from fluoviz_addon_autoencoders.core import advice, linear
advice.advise(data, cfg=None) -> {"n", "P", "components_99", "latent_dim", "latent_reason", "size", "hidden", "family", "n_params", "samples_per_param", "samples_per_latent", "validation": {...words}, "estimate": {"seconds", "words"}, "warnings": [...], "blocking": [...]}
advice.count_parameters(P, d, hidden, family="ae", skip=False);  advice.estimate_seconds(cfg, n, P);  advice.components_for_share(td, 0.99)
linear.fit_pca(X, M, omega, k_max) -> PCAModel (.project .reconstruct .components .lam .total);  linear.pca_errors(model, X, M, omega, blocks, ks) -> {k: (n,)}      # the weighted PCA of the network coordinates = fPCA / MFPCA in the L2 frame
```

## 6. Numbers on the example data (64 samples, GP basis, fluorescence + absorbance, defaults: 600 epochs / 3 starts / early stopping)

Share of the variation rebuilt on HELD-OUT samples (validation split), autoencoder against the functional PCA with the same number of components (measured with `tests/` helpers on a shared computer, load about 1 per core):

| data | d | autoencoder | fPCA | time |
|---|---|---|---|---|
| example data with its extreme sample S04_conc (5x concentration) | 4 | 99.21 % (the PCA start was kept) | 99.21 % | 3.5 s |
| the same | 2 | 92.1 % | 86.0 % | 5.0 s |
| no outliers, bilinear | 4 | 99.68 % | 99.56 % | 5.2 s |
| no outliers, bilinear | 2 | 97.35 % | 96.96 % | 5.1 s |
| no outliers, inner-filter effect (a nonlinear manifold) | 4 | 99.59 % | 99.44 % | 5.3 s |
| the same | 2 | 96.68 % | 96.36 % | 3.0 s |

(Before the linear shortcut the default network rebuilt 90.8 % where the fPCA rebuilt 99.4 %.)  Speed on the 64 example samples of the full grid (156 x 43 EEM, 300 epochs, one start, no early stop; `test_speed_on_the_example_data`, quiet computer): **functional (GP basis, 55 coordinates) 1.4 s, pixels (4985 coordinates) 10.6 s**; the defaults (3 starts, early stopping) take 3-5 s functional and about 25 s in pixel mode (on a computer with a load of 25-30).  The threads: one for a network with fewer than 100 000 weights, two above (the environment variable `FLUOVIZ_BLAS_THREADS` wins); with four threads on a busy computer the pixel version was five times slower than with two.  Starting the worker process costs about a second (PyTorch import).

## 7. Deviations from `SPEC.md` (and why)

1. **`skip` option (default on)**: linear shortcuts through the code, PCA-initialised; not in the SPEC.  Reason: with the plain default the AE rebuilt 91 % of the held-out variation where the fPCA rebuilt 99 % (slow convergence, no extrapolation to a sample of five times the concentration).  `skip=False` = SPEC.
2. **Batch size default** `min(32, max(8, n_train/6))` instead of `min(32, n)`: with tens of samples a batch of 32 is only one or two steps per epoch and the network does not converge in 600 epochs.
3. **Network input** is scaled `u = x / (c sqrt(omega))` (`in.scale`), i.e. the L2-frame coordinates; the loss is a plain squared error there.  (The SPEC leaves `in.scale` free.)
4. **`L_rec` averages over the variables** (blocks count equally, weights sum to 1) instead of summing: `1 - L_rec` stays the share rebuilt with two variables.
5. **Smoothness metric**: B-splines / cosines use the second-difference roughness of the function on the pixel grid instead of the penalty matrix on the coefficients (that one is hopelessly ill-conditioned on a masked domain: the invisible directions get a weight of 1e8); every `Q` is scaled to unit average penalty of the data instead of `tr Q = K` (which made `smoothify = 1` do nothing for B-splines); GP `Q = I` as in the SPEC.
6. **VAE likelihood**: profiled (optimal) `sigma^2` at every step, `N = n_eff = min(P, max(15, 3d))`; the penalty divided by the same `sigma^2`.  Documented in `train.py`.
7. `ModelSpec` gets the optional keys `skip` and `vae.n_eff`.
8. **Threads**: one for small networks, two above 100 000 weights (not `min(4, cores)`): oversubscribed thread pools made the pixel version five times slower on a busy computer.
9. k-fold validation below 30 samples (as the SPEC), but with ONE start per fold and `n_restarts` starts for the final network (chosen by the training loss, there is no validation set left).
