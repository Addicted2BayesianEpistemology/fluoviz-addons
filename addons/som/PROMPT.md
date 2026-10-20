# Brief: implement the FluoViz add-on "Self-organising map" (SOM)

*This is the implementation prompt for the second FluoViz add-on.  You (the implementer) build it in this repository under `addons/som/`, following the framework and the conventions of the first add-on, `addons/autoencoders/`.  Read this whole brief first, then the reading list of §0, then start with the order of work of §14.  The first thing you write is `addons/som/SPEC.md`: a condensed, binding version of this brief with the decisions you took where the brief leaves you a choice; keep it accurate while you build.*

---

## 0. Context, reading list, rules

**What FluoViz is.**  A FastAPI + Vue application (repository `80_fluoViz`, a sibling folder of this one) that explores fluorescence excitation-emission matrices (EEMs: a surface over emission × excitation, with scatter bands and excised regions masked as NaN) and absorbance spectra (a curve over wavelength) of a few dozen to a few hundred samples, with metadata per sample.  It has PCA / NMF / PARAFAC, a **Functional PCA** page (`fluoviz.analysis.fpca`: each sample is a smooth function written in a basis chosen from the data: a Gaussian-process Karhunen-Loève basis, tensor B-splines, cosines, or the raw pixels), a **Clustering & LDA** page that clusters *any* set of per-sample variables, a feature module of fluorescence indices, **saved models** that are applied to new samples (also chained into classifiers), and an **add-on** framework: optional pages downloaded on request.

**What an add-on is.**  A zip with `manifest.json`, a Python package whose `register(api)` adds a router under `/api/addon/<id>`, a saved-model kind and a result kind, and a `web/` folder with the page, its components and a Help guide tab (ES modules the browser loads from `/addon-web/<id>/web/...`).  The first add-on, **Autoencoders**, is the model for everything you do: same folder layout, same stepper page that reuses the Functional PCA's step components for the data part, same result / staleness / saved-model conventions, same tests and packaging.

**Read, in this order, before writing code** (paths relative to the two repositories):

1. `80_fluoViz/docs/modules/addons.md` — the contract (manifest, `register(api)`, hooks, the browser contract).  Binding.
2. `81_fluoviz-addons/README.md`, `scripts/build_addon.py`, `.github/workflows/release.yml` — how add-ons are built and released.
3. `addons/autoencoders/SPEC.md`, `docs/api.md`, `docs/ui.md`, `docs/kind_and_router.md`, `fluoviz_addon_autoencoders/core/README.md` — how the first add-on is specified and documented.  Your docs mirror these.
4. The code of the first add-on: `fluoviz_addon_autoencoders/__init__.py`, `core/data.py` (functions → coordinates and back, the `AEFrame`), `server/prep.py`, `server/results.py`, `server/fitjob.py`, `server/payload.py`, `server/views.py`, `kinds/autoencoder.py`, `kinds/freeze.py`, `kinds/stats.py`, `web/page.js`, `web/components/AeState.js`, `AeStepsReuse.js`, `web/help.js` + `help/*.js`, `tests/api_support.py`, `tests/conftest.py`, `tests/e2e/flow_autoencoders.py`.
5. FluoViz: `docs/modules/fpca.md` (bases, the L2 frame `Z = C·G^½`, gaps, frozen elements), `docs/modules/clustering.md` (`cluster_scores`, `choose_k`, `cluster_mean_images`, `metadata_associations`, `crosstab_against`, `directions_to_images`), `docs/modules/features.md` (`compute_index`, `list_indices`, Coble peaks, FI, HIX, BIX), `docs/modules/models.md`, `models_core.md`, `models_functional.md`, `models_classify.md` (the Applier contract, statuses, chains), `docs/FRONTEND.md` (tokens, light / dark, 600 px, export buttons, robustness, the section *Add-ons*, the section *Functional PCA — the guided pipeline*, the section *The layout of EEM pictures*).
6. The interfaces you will call (verify the signatures in the code; the names below were checked):
   * `fluoviz.analysis.fpca`: `fluorescence_variable(X, ex, em, ids, *, transform, min_coverage, units, complete)`, `absorbance_variable(wl, A, ids, *, transform, units)`, `common_samples`, `Domain` (`w` quadrature weights summing to 1, `take`, `embed`), `pipeline.resolve_basis(v, spec, progress)`, `bases.build_representation(domain, spec, progress)` → `Representation` (`B`, `G`, `frame()`, `to_frame`, `from_frame`, `evaluate`, `prior_lam`), `fit.prepare_element(name, domain, Y, basis, smoothing, units, progress, seed, gaps)` → `Element` (`rep`, `fit` with `sigma_f2`, `ybar`, `cbar`, `C`, `Z`).
   * `fluoviz.analysis.frozen.kinds.fpca`: `FrozenElement`, `element_freeze`, `ElementRuntime` (`dom`, `rep`, `centred_coefficients`), `pretreat`, `clean_absorbance`, `basis_words`, `smoothing_words`.  `fluoviz.core.recipe.capture_recipe(p, uses_eem, uses_abs, ids)`.
   * `fluoviz.analysis.frozen`: `Applier`, `ApplyData`, `ApplyOutput`, `FrozenError`, `SavedModel`, `Statuses`, `STATUS_WORDS`, `serial` (`serial.allow`, `encode`, `decode`); `fluoviz.analysis.frozen.kinds._decomp` helpers (`plot_block`, `diagnostic`, `limit_entry`, `choice_option`, `number_option`, `settings`, ...).
   * `fluoviz.server.analysis`: `project()`, `current_view()`, `start_job(kind, title, fn)`, `save_result(p, kind, title, params, payload, summary, *, sample_ids, variables, var_info, obj, data_sig, rid)`, `result_public`, `plot_image(s)`, `plot_mask`, `component_images_payload`; `fluoviz.server.encoding.JsonRouter`; `fluoviz.server.jobs.HEAVY_KINDS`; `fluoviz.analysis.common.Progress` (`progress(frac, msg)`, `.sub(lo, hi)`, cancellation by `Cancelled`).
   * Project: `p.view(only=ids)`, `p.abs_view(include_excluded)`, `p.meta` (DataFrame), `p.meta_types`, `p.variable_catalogue(qc)`, `p.variable_values(keys, qc)`, `p.signature()`, `p.stale_parts(r)`, `PART_LABELS`.
   * Browser: `/static/js/{store,api,jobs,router,util,prefs,theme,plot,variables,eem,actions,icons}.js`, `/static/js/components/{Common,JobBar,ResultBar,DecompCommon,FpcaCtx,FpcaStepper,FpcaState,FpcaCommon,FpcaStepWhat,FpcaStepDomain,FpcaStepRepr,FpcaStepSmooth,EEMView,EEMThumb}.js`, `ChartCard`, `FigureHost`, `tokens()`, `categoryStyle`, `sequentialScale`, `divergingScale`.

**Rules that bind you** (they are the add-on framework's and the first add-on's; do not relax them):

* **Nothing of the autoencoders add-on is imported** by this add-on: each add-on must work alone.  Where you need the same logic (the coordinate construction of `core/data.py`, the step wrappers of `AeStepsReuse.js`, the freeze / unpack of a result, the error-limit statistics), write your own module modelled on it (read it, keep the same FluoViz interfaces, keep the same frozen structure) and say in `SPEC.md` which modules were modelled on which.  Tell the lead in your final report that these pieces are now duplicated in two add-ons and would deserve a home in FluoViz (`fluoviz.analysis.fpca` or a shared helper) in a later release.
* **No extra libraries.**  Everything is numpy / scipy (both are in FluoViz's environment).  `manifest.json`: `"conda": {"all": []}`, `"extra_download_mb": 0`, `"libraries_in_server": true` (nothing heavy is imported).  A batch SOM on a few hundred samples takes well under a second for coefficients and a few seconds for pixels, so **no separate process**: the fit is an ordinary job of the FluoViz job runner (`start_job`), cancellable through `Progress`.  Add `som-fit` and `som-scan` to `HEAVY_KINDS` so scans do not take the places of interactive jobs.  Do not vendor MiniSom / SOMPY: the algorithm is ~300 lines and must handle NaN, sample weights and a quadratic metric (§3), which they do not.
* **Plain English** everywhere a person reads (labels, help, verdicts, errors that say what to do); no jargon without an explanation; honest about small data sets.  British spelling as in the rest of the program ("self-organising", "standardised", "colour").
* **Reproducible**: a `seed` in every request; batch training with linear initialisation is deterministic, and says so.
* **Everything the page shows is computed on the server** from the result payload; the page applies no threshold of its own.  Images travel in browser orientation `(n_ex, n_em)` and the page follows the setting *Axes of fluorescence maps* (`prefs.eemTranspose`) like every EEM picture.
* **Results are never recomputed silently**: staleness, `ensure_run`, "Compute again with the same settings" (`replace`), exactly as the autoencoders result (`docs/api.md` §0 of that add-on).
* **The honest numbers come from samples the map was not trained on** (k-fold) wherever a number is used to judge or to set a limit.

---

## 1. What the person gets

A page **Self-organising map (add-on)** in *Experimental methods* (badge *add-on*), a guided pipeline like Functional PCA and Autoencoders, that

1. lays the samples out on a small two-dimensional grid of **units** (a map), each unit holding a **prototype**: a typical EEM / absorbance spectrum / set of variables.  Neighbouring units hold similar prototypes, so the map is a picture of the whole data set in which similar samples sit together and gradients (humic → protein-like, upstream → downstream, raw → treated) run across the map;
2. works on **fluorescence alone, absorbance alone, or both together** — in every mode of §2 — on the **observed values** (pixels; the classic SOM of whole EEMs of the DOM literature) or, the **functional versions**, on the **coefficients of the basis** chosen with the same machinery as the Functional PCA (Gaussian-process / B-spline / cosine basis, same smoothing, same masked domain, same gap handling), with distances computed in the coefficient space under a metric the person chooses and understands (§3);
3. also, as a secondary mode, on **any set of per-sample variables** (PARAFAC scores, PCA / fPCA scores, autoencoder codes, fluorescence indices, metadata), the way the literature mostly uses SOMs (§2.3);
4. **shows** the map: U-matrix, hit map, the prototypes drawn as EEM thumbnails on the grid, component planes (fPCA scores, chosen pixels, fluorescence indices of the prototypes, metadata means), label maps, trajectories of samples over time, a Sammon projection of the prototypes;
5. **checks** it honestly: quantisation and topographic error, held-out quantisation error, a comparison with the plain 2-D PCA / fPCA plot ("does the map keep neighbours together better than a scatter of the first two components?"), map-size scan, stability over random starts / bootstrap;
6. **groups** the units (two-level clustering of the prototypes, Vesanto & Alhoniemi 2000) and so the samples, with the usual cluster interpretation (mean EEMs, metadata associations);
7. **uses** it elsewhere: the map position, the unit, the quantisation error and the cluster are published variables (Plots, Clustering & LDA, metadata links); the map can be **saved** and applied to new samples (position, unit, cluster, "poorly described" status), also as the first link of a classification chain; exports; a methods paragraph with references.

---

## 2. Inputs and coordinates

Three input modes.  The first two are the heart of the add-on (**must**); the third is **should**.

### 2.1 Variables and blocks (fluorescence, absorbance, both)

Exactly as the autoencoders add-on (`core/data.py`, `server/prep.py`): the request's `input` has the Functional PCA vocabulary (`variables`: `["fluorescence"]`, `["absorbance"]` or both; per variable `transform`, `min_coverage`, `gaps`, `basis`, `smoothing`, `processed` for absorbance); the samples are those that have every chosen variable; each variable is one **block** of coordinates; blocks are concatenated.  **All three choices of variables must work in the pixel path and in the functional path**, and in both the map can be looked at per variable (prototype EEMs *and* prototype absorbance curves on the grid; planes per variable).

* **Pixels** (`basis.kind == "pixels"`): per variable the observed values `y` at the pixels of the domain minus the training mean function `ȳ_v`, with NaN where a sample lacks a pixel (a region excised from it); the coordinates keep the NaN (the SOM handles them, §4.5).  Domain quadrature weights `w_j` (summing to 1) are the pixel weights of the L2 metric.
* **Functional** (`gp` / `bspline` / `cosine`): per variable the centred smoothed coefficients moved to the **L2 frame**, `Z = C·R` (`rep.to_frame`), so that the Euclidean distance between rows is the L2 distance between the smooth functions.  Gap-aware coefficients (`gaps: "auto"`) for samples that lack part of the domain, as the Functional PCA and the autoencoders do.  The frozen element (`element_freeze`) is kept per block so a new sample is smoothed with the fixed parameters.
* **Block weights** `w_v`: `"variance"` (default: `1 / V_v`, `V_v` the total variance of the block = mean squared L2 norm of the centred samples, so each variable counts equally whatever its units), `"none"`, or a custom dict as the multivariate fPCA (`fpca.md` §4).  Unlike the autoencoders, **offer the weights** here (the SOM is a distance method; the person may legitimately want absorbance to count half).
* Everything in a `SomFrame` (modelled on `AEFrame`): per block `name, kind, K, element, scale/weights`, with `coordinates(values, min_coverage)` for new samples and `to_function(X)` to turn prototypes back into functions (functional: `rep.evaluate(rep.from_frame(Z) + cbar)`; pixels: `ȳ + y`).  **Prototypes are always decodable to EEMs and curves** in modes 2.1–2.2.

### 2.2 Pre-treatments that the literature applies before a SOM

They already exist as the Functional PCA pre-treatments; the Help guide must say which is which: *divide each EEM by its maximum* (Ejarque-Gonzalez & Butturini 2014) = transform `max`; *unit area / unit L2 norm* = `area` / `l2` (shape-only maps); *square root / asinh* to tame the peaks.  Bieroza et al. (2009) also **standardised each pixel across samples** (variance scaling): offer it as the metric `standardize` of §3.3 (pixels), never as a hidden default.

### 2.3 Variables mode (should)

`input.kind == "variables"` with `keys` = variable-catalogue keys (`res:<rid>:<name>`, `meta:<column>`, `qc:<key>`; the same keys and the same column picker as the Clustering page, prop `request` / `#/cluster?keys=`).  Coordinates = the columns, z-scored by default (`standardize: true`, the Clustering page's rule, constant columns only centred), NaN kept (partial distances).  No functional representation, no EEM decoding of prototypes from the coordinates — but the page still shows EEMs per unit: the **NaN-aware mean EEM of the samples of the unit and of its neighbours** (neighbourhood-weighted, `cluster_mean_images`-like), and, when *all* the keys are the score variables of one decomposition result that has `component_images` in its payload (PCA / NMF / PARAFAC / fPCA / autoencoder results do), the **back-projected prototype** via `directions_to_images` / `class_mean_images_via_scores`.  The saved model of this mode is a **chain** like the classifier kinds (`models_classify.md` §2: `plan_inputs`, `apply_chain`): keys of results are embedded as parts, metadata columns become `needs["meta_columns"]`.  If the chain machinery turns out to be too entangled with the classifier kinds, deliver the variables mode *without* saved model and say so (the result and its published variables still work).

---

## 3. The distance: which metric, and why (think this through, then implement what is decided here)

The SOM needs a distance `d²(x, m)` between a sample's coordinates and a prototype.  For the functional versions the natural question is which metric on the **coefficient space** is right.  Decision and reasoning:

### 3.1 Four candidates on the frame coordinates `z` (per block, then blocks weighted by `w_v`)

| kind | `A` in `d² = (z − m)ᵀ A (z − m)` | what it means | verdict |
|---|---|---|---|
| **`l2`** (default) | `I` (block weights `w_v` outside) | the L2 distance between the smooth functions = the "functional SOM" of Rossi, Conan-Guez & El Golli (2004): they Cholesky-factor the Gram matrix and run a plain SOM on `U·α`; our `Z = C·R` *is* that.  In pixel mode it is the quadrature-weighted distance between EEMs.  Units of the data; the quantisation error is an RMS misfit in data units; dominated by the directions of largest variation (the first fPCs), which is what a person expects a map of EEMs to be organised by | **must; the default** |
| **`mahalanobis`** | `Σ⁻¹` with `Σ` the covariance of the training coordinates | affine-invariant: every direction of variation counts the same, so the map also organises along the *minor* modes (a subtle shape difference counts as much as the overall humic intensity).  `Σ` is `P × P` with rank `≤ n − 1 < P`: it **must be regularised**, otherwise the distance is dominated by noise directions.  With the regularisation of §3.2 it is exactly the Euclidean distance on **standardised fPCA scores** (the first `r` components divided by their SD) plus one unit for everything beyond them: a clear thing to explain | **must, as an option, with the partial-whitening exponent of §3.2** |
| **`prior`** (GP basis only) | `diag(1 / (σ_f² λ_k))` (the autoencoders' standardisation `ξ = Z / d`) | the RKHS norm of the difference: every coefficient is N(0, 1) under the Gaussian-process prior, so rough components are up-weighted by `1 / λ_k`.  After the Wiener smoothing the rough coefficients are shrunk, so in practice it emphasises medium-scale shape over gross intensity.  It is not the sample covariance, so it is not "Mahalanobis"; it is an expert option that is cheap because the numbers exist | **should** (one line of code once §3.4 exists; label it "Gaussian-process prior (shape over intensity)") |
| `standardize` (pixels / variables) | `diag(1 / s_j²)`, `s_j` the SD of coordinate `j` across samples | Bieroza et al. 2009; the Clustering page's `standardize` | **must** for `variables` mode (its default) and for pixels (option, not default: it amplifies the noise of weak pixels; the help says so) |

### 3.2 The regularised Mahalanobis metric (the one to implement)

Let `Σ = UΛUᵀ` be the eigendecomposition of the covariance of the stacked, block-weighted training coordinates (`√w_v · Z_v`; that is the multivariate fPCA of `fpca.md` §4), `U` `(P × r)` the leading `r` directions, `λ_1 ≥ … ≥ λ_r`, and `ρ = tr(Σ) − Σ_k≤r λ_k` the total variance beyond them.  Define, with the **partial-whitening exponent** `α ∈ [0, 1]`:

```
A(α) = U diag(λ_k^−α) Uᵀ  +  (I − UUᵀ) · ρ^−α
```

* `α = 0` → `A = I`: the `l2` metric.  `α = 1` → full Mahalanobis on the first `r` directions, and **the whole complement counts like one standardised component** (its total variance `ρ` is scaled to 1): the noise beyond the kept components can never dominate, whatever `P` is.  `α = 0.5` is the half-way house.  Expose `metric.alpha` (default 1 for `mahalanobis`) and `metric.components` `r` (default: the parsimonious number of the fPCA's cross-validation on the same coordinates (`fpca.validation.cross_validate`, `parsimonious`), at least 2, at most `min(n − 2, 20)`; the page says which `r` was used and lets the person change it).
* **Implementation: whiten, run a Euclidean SOM, un-whiten.**  `A = LᵀL` with the symmetric root `L = U diag(λ_k^−α/2) Uᵀ + (I − UUᵀ) ρ^−α/2` and `L⁻¹ = U diag(λ_k^α/2) Uᵀ + (I − UUᵀ) ρ^α/2` (never form `P × P` matrices when `P` is large: apply them as `U (…) + (x − U Uᵀ x) (…)`).  Train on `x̃ = L x`; prototypes `m = L⁻¹ m̃`.  **This is exact**, not an approximation: for any positive-definite `A` the batch-SOM update `m_j = Σ_i h_{c(i) j} w_i x_i / Σ_i h_{c(i) j} w_i` is the minimiser of the neighbourhood-weighted energy whatever the quadratic metric (a weighted mean), so only the best-matching-unit search depends on `A`, and that is Euclidean in `x̃`.  Decoding prototypes to EEMs uses `m` in the frame coordinates, so the pictures do not depend on which metric organised the map, only the assignments do.
* **Report distances twice**: in the training metric (unitless under `mahalanobis` / `prior`) and as the plain L2 RMS in data units (`A = I`), so the person always has a number with units.  The U-matrix is drawn in the training metric (option: in L2).
* **Pixels with missing pixels**: a rotated metric cannot drop a NaN coordinate, so `mahalanobis` is only offered for pixels when every sample has every pixel of the domain (or with `gaps: "auto"`, which fills them by `impute_pixels` first); otherwise the page says so and offers `l2` / `standardize` (diagonal metrics work with partial distances, §4.5).
* **Several variables**: under `mahalanobis` the joint covariance already equalises the blocks, so `w_v` only matters for `α < 1`; the page explains that in one sentence.
* **Map shape under Mahalanobis**: the side-ratio heuristic (§4.2) uses the eigenvalues *in the training metric*, which are all 1 under `α = 1`: the map is then square.  Say so in the advice.
* **New samples / saved model**: `U, λ, ρ, α` are frozen with the model and never re-estimated.

### 3.3 What is NOT done, and why (write it into the Help guide's "Choices" page in plain words)

* No adaptive / learned metric (GRLVQ-style relevance learning): with dozens of samples it overfits, and it would make the map unreadable ("organised by a metric the program invented").
* No dot-product (cosine) SOM as a separate kind: the shape-only pre-treatments (`l2`, `area`, `max`) give the same effect with the same L2 code path.
* Full (unregularised) Mahalanobis: singular below `n = P + 1`, and dominated by noise directions just above it.

### 3.4 Metric module

`core/metric.py`: `Metric` dataclass (`kind`, `alpha`, `components`, `U`, `lam`, `rho`, per-block weights and SDs, `whiten(X)`, `unwhiten(M)`, `d2(X, M)` with partial distances for NaN, `d2_l2(X, M)`), `fit_metric(X, M_obs, w_blocks, spec)`, `@serial.allow` so it freezes.  Tested: `whiten → unwhiten` is the identity; `d2` under `mahalanobis` at `α = 1, r = P` on complete data equals scipy's `mahalanobis`; `α = 0` equals `l2`; prototypes of a map trained under `mahalanobis` are the same weighted means as under `l2` for the same assignments.

---

## 4. The SOM itself (`core/som.py`, pure numpy)

### 4.1 Lattice and shape

* `lattice`: **`hexagonal`** (default; Kohonen's recommendation and every EEM paper) | `rectangular`.  Unit coordinates on the plane (hexagonal rows offset by ½, vertical spacing `√3/2`); neighbourhood distances are Euclidean on these coordinates (so 6 neighbours at distance 1 on a hexagonal grid, 4 on a rectangular one).
* `shape`: **`sheet`** (default) | `toroid` (no border: distances wrap in both directions; removes the border effect for data without a natural boundary; the drawings say "the map wraps around" and draw the wrap).  `cylinder` is **could**.
* `rows × cols`: from `size` = `"auto"` | `[rows, cols]`.

### 4.2 Map size (`core/advice.py`)

The SOM Toolbox rule, which is what the field uses: `units ≈ 5·√n` (Bieroza: 120 units for 625 EEMs; Zhang et al. 2015: 30 units for 36 EEMs); side ratio `rows / cols = √(λ₁ / λ₂)` of the two largest eigenvalues of the training covariance **in the training metric** (hexagonal rows multiplied by `0.866` to make the physical map that shape), ratio capped at 4; at least 2 × 2; the number of units rounded to the nearest `rows × cols`.  Offer three presets in the advice and the page: **Compact** (`≈ n/4` units, at least 6: roughly four samples per unit; a map to *group* samples), **Standard** (`5√n`: the usual map, many empty "interpolating" units: a map to *see* the data), **Fine** (`2 × 5√n`, capped at 400 units).  Default **Standard**, with the honest sentence under it: "With N samples on M units, most units will hold no sample: they interpolate between their neighbours.  That is normal for a map; use *Compact* if you want every unit to stand for real samples."  Refuse `units > 4 n` and `n < 8` (plain words; `allow_few_samples` as the autoencoders, never below 4).  The **map-size scan** (§6) is the data-driven answer.

### 4.3 Initialisation

* **`linear`** (default, Kohonen 2001 §3.7): prototypes spread on the plane of the first two principal directions of the training coordinates (in the training metric), between `±2` SD — deterministic, no seed needed; "every unit starts where a PCA plot would put it".
* `random`: `units` random training samples (seeded), `sample` initialisation; used by the stability check.

### 4.4 Training

* **`batch`** (default; Kohonen 2013, SOM Toolbox `som_batchtrain`): per epoch, best-matching unit `c(i)` of every sample in the current metric, then `m_j = Σ_i h_{c(i) j} w_i x_i / Σ_i h_{c(i) j} w_i` over the observed coordinates (§4.5), with the neighbourhood `h` evaluated at the epoch's radius.  Vectorised: `H (units × units)`, `assignments (n)`, one matrix product `(H[:, c]) · (w ⊙ X)` per epoch.  Two phases as the SOM Toolbox: **rough** (radius from `max(1, max(rows, cols) / 4)` down to a quarter of that, `epochs_rough` default `max(10, …)`) and **fine** (down to 1 (sheet) — the SOM Toolbox's final radius; offer `radius_final` 0.5…2), linear decay, `epochs` default **50 + 150** (cheap; the energy settles well before).  A unit with no sample and no neighbour mass (possible only with tiny radii) keeps its prototype.
* `sequential` (**should**; the kohonen R package's mode, used by Ejarque-Gonzalez & Butturini): one sample at a time, `m_j += α(t) h_{c j}(t) (x − m_j)`, `α` from `0.05` to `0.01` linearly, the same radius schedule, samples shuffled per epoch (seeded).  Offered for people who want to reproduce a published analysis; the page says batch is deterministic and sequential is not.
* **Neighbourhood** `h(d) `: **`gaussian`** `exp(−d² / 2σ²)` (default) | `cutgauss` (gaussian truncated at `σ`) | `bubble` (1 for `d ≤ σ`) | `epanechnikov` (`max(0, 1 − d²/σ²)`), `d` the lattice distance, `σ` the radius.
* **Energy curve**: the distortion `Σ_i w_i Σ_j h_{c(i) j} d²(x_i, m_j)` and the quantisation error per epoch, returned as the training history (a small chart, like the autoencoders' live chart; here the job is seconds, so one chart after the fit is enough: no polling route, unless a pixel fit on thousands of pixels and hundreds of units turns out to take more than ~5 s on the example data — then add the same `/fit/{job}/history` mechanism the autoencoders use, as a file written by the job).
* **Sample weights** `w_i`: `gap_weighting` `observed` (default) | `equal`, exactly as the autoencoders (a functional sample that lacks part of the domain counts in proportion to the share measured, at least 5 %, weights averaging 1).  Also used by the quality numbers.
* Threads: numpy's; nothing to set.

### 4.5 Missing coordinates (partial distances; SOM Toolbox `som_bmus` rule)

Pixel mode with excised regions, and variables mode with missing values: a NaN coordinate is left out of the distance, and the distance is scaled up by `(Σ_j a_j) / (Σ_{j observed} a_j)` (the share of the metric's weight that was observed), so partly observed samples are comparable with complete ones.  The batch update of coordinate `j` averages only the samples that have it (per-coordinate weights `h w · M`); a coordinate no sample in the neighbourhood has keeps its old value.  The result says how many samples had missing coordinates and the sample view says what a sample lacks (`missing_part`, as the autoencoders).  In functional mode there are no NaN (the gap estimate fills them), as in the autoencoders.

### 4.6 Numerics to test (`tests/test_core_som.py`)

Batch SOM with radius → 0 and one unit per sample reaches quantisation error 0; with one unit the prototype is the (weighted) mean; a 1 × M map on data on a curve orders the samples along the curve (the lattice order correlates with the arc length, |Spearman| > 0.95); linear init is reproducible run to run; hexagonal neighbour distances are 1 for the six neighbours; toroid wrapping; partial distances equal full distances when nothing is missing and are unbiased in expectation when 30 % of coordinates are removed at random; sample weights 0 remove a sample; the energy decreases (non-increasing) over the fine phase with a fixed radius; the metric tests of §3.4; the quality measures of §6 against `somperf` formulas on a tiny hand-computed map.

---

## 5. What the result holds, and what the page shows

### 5.1 Result `som-N` (`save_result(kind "som")`), payload (module orientation inside, turned by the GET routes)

* `spec` (the resolved request), `map` (`rows, cols, lattice, shape, units`, unit coordinates `(M, 2)`, neighbour lists), `metric` (kind, α, r, words), `prototypes` `(M, P)` in frame coordinates (and `whitened` if needed), `frame` (the `SomFrame`, frozen: this is what the saved model needs, kept in `payload["frozen"]` like the autoencoders so freezing needs no recomputation), `blocks`.
* Per sample: `bmu` (unit index), `bmu2` (second best), `row`, `col`, `x`, `y` (continuous position, §5.2), `qe` (distance to the BMU in the training metric), `qe_l2` (RMS in data units), `qe_rel` (`qe_l2² / V`, the share of the typical variation not captured: `1 − qe_rel` is comparable to a variance explained), `missing_part`, `heldout_qe` (the out-of-fold QE of §6), `cluster` (§7).
* Per unit: `hits` (count and the weight sum), `members`, `umatrix` (mean distance to the lattice neighbours in the training metric; also the interstitial U-matrix of the SOM Toolbox `som_umat` for drawing: `(2 rows − 1) × (2 cols − 1)`), `dmean` (distance of the prototype to the data mean), `qe_unit` (mean QE of its members), the **planes** (§5.3), the **images** of the prototypes per surface block `(M, n_em, n_ex)` and `curves` per curve block `(M, n_wl)` (functional: decoded; pixels: `ȳ + m`; variables mode: the neighbourhood mean EEM and, when available, the back-projection), stored as float32.
* `quality` (§6), `clusters` (§7), `training` (history: energy and QE per epoch, phases, seconds), `notes`, `warnings`, `recipe` (the methods paragraph), `versions`.

### 5.2 Continuous position `(x, y)` (one definition, frozen with the model)

The weighted mean of the lattice coordinates of **all** units with weights `exp(−d²_j / 2τ)` where `d²_j` is the sample's distance to prototype `j` in the training metric and `τ` the mean over the training samples of their squared BMU distance (the squared quantisation error); on a toroid, computed on the unwrapped neighbourhood of the BMU.  It is what `som_x`, `som_y` publish, what the "scatter on the map" draws, and what the trustworthiness comparison (§6) uses.  Document it in `SPEC.md` and the Help guide ("a sample sits between the units it resembles").

### 5.3 Planes (one number per unit, drawn as a coloured hexagon map)

* **U-matrix**, **Hits**, **Distance to the mean**, **Quantisation error of the unit**.
* **Scores**: the projection of each prototype on the first `k` principal directions of the training coordinates (in L2: the multivariate fPCA / PCA of the data) — the "component planes" that make sense in functional mode (`k` ≤ 8).
* **Pixels and boxes**: the intensity of the prototype EEM at a chosen pixel, the mean / maximum in a box, the overall intensity, the absorbance at a wavelength — the same feature kinds as the autoencoders' `core/features.py` (`point`, `region_mean`, `region_max`, `total`, `absorbance`), picked by clicking on the mean EEM.
* **Fluorescence and absorbance indices of the prototypes**: `fluoviz.analysis.features.compute_index(key, inputs)` on the prototype EEMs embedded on the grid (NaN outside the domain; an index whose points fall outside the domain is NaN, with a note): Coble peaks A, C, M, T, B, the ratios, FI, HIX, BIX, β/α, total fluorescence, FRI regions; absorbance indices (`a254`, slopes, `S_R`, `E2:E3`) when the absorbance is a block.  These are the "component planes" a DOM scientist expects.  **Note the caveat in the plane's help**: an index of a *prototype* (an average, possibly after a shape-only pre-treatment or a smoothing) is not the index of a sample; a prototype under `max` / `l2` / `area` has no intensity units.
* **Metadata planes**: for a numeric variable (metadata, QC, any result variable of the catalogue) the mean over the members of the unit (NaN for empty units; option *smooth over neighbours* = the neighbourhood-weighted mean, so empty units get a value); for a categorical variable the **label map** (majority label per unit, its purity, the counts; drawn as colour + the label text) — "which site / season / treatment stage lands where".
* **Variables mode**: one plane per input variable (the prototype's value), the classic component planes.

Planes are computed on request (`POST /{rid}/plane`), cached in the result, with a list of the available ones.

### 5.4 Tabs of step 7 (Map)

* **Map**: the main drawing: the hexagonal / rectangular grid, any plane as colour (select **Colour the units by**), the samples as dots at their continuous position (or at the unit, with jitter; **Place the samples** at *their position* / *on their unit*), coloured / shaped by any variable (**Colour by**, **Symbol by**), hover = a card with the EEM thumbnail (`<EEMThumb>`), the unit, the QE, the status; click = select the sample (**Open in Samples**).  Cluster borders (§7) as thick edges between units of different clusters.  Held-out samples of the k-fold (§6) are not a thing here (every sample is on the final map), but the **out-of-fold QE** is a plane and a sample colour.
* **Prototypes**: the grid of prototype EEM thumbnails (one `<EEMThumb>` per unit, a shared colour scale or per-unit; the hit count in the corner; empty units dimmed), and for an absorbance block the grid of curves; **Show** (*the prototype* / *the difference to the mean EEM* / *the mean of the samples of the unit*); click a unit = the unit card: its prototype at full size, its members, the mean of its members next to the prototype (and their difference: "how well the prototype stands for its samples"), its planes' values, its cluster.
* **Planes**: a gallery of small maps: the chosen planes side by side (U-matrix, hits, the scores, the indices, the metadata) with **Add a plane** (the kinds of §5.3) and a shared / own colour scale switch; the classic "component planes" picture.
* **Trajectories** (**should**): **Order by** (a numeric or date metadata column), **One line per** (a categorical column; e.g. site), draw the path of each group's samples over the map in time order (arrows, start and end marked), with the list of the samples; the classic "monitoring over time" picture (Yang et al. 2019).
* **Samples**: the table (worst quantised first; filters *All* / *Poorly described* / *Lacks part of the domain*; search), the sample's measured EEM / curve, its prototype, the difference, its position and unit, its QE in words, the status; **Unusual samples** (threshold methods as the autoencoders: median + c·MAD / quantile of the out-of-fold QE / the saved model's limit).
* **Projection** (**should**): Sammon mapping (`scipy.optimize` on the prototype distances; seeded) of the prototypes with the lattice edges drawn: a folded map shows as crossing edges; the sentence says whether the map is folded.

---

## 6. Checks (step 8) — priorities **M** must · **S** should · **C** could

* **M Quality numbers** with plain-English verdicts: **quantisation error** (mean QE in the training metric and the L2 RMS in data units; `1 − mean(qe_rel)` as "share of the variation captured", compared with the fPCA / PCA explained variance for the same number of components as the map has principal "degrees of freedom" (2) — "a map with M units captures X %, the first two components Y %"); **topographic error** (Kiviluoto 1996: share of samples whose first and second BMU are not neighbours on the lattice; on a toroid with the wrapped neighbourhood); **topographic product** (Bauer & Pawelzik 1992; the sign says folded vs too-large); **trustworthiness and continuity** at `k = 5` neighbours (Venna & Kaski 2001) of the continuous positions `(x, y)` against the distances in the training metric — **compared with the same numbers for the 2-D PCA / fPCA score plot of the same coordinates** ("Is the map worth it?": a verdict in words; this is the SOM analogue of the autoencoders' "is the nonlinearity worth it?").
* **M Held-out quantisation error** by k-fold (default 5 folds; `folds` 2…20): a map trained on the other folds, the QE of the held-out samples (in the training metric, and relative), per sample (`heldout_qe`, published) and overall, against the in-sample QE: the gap says whether the map memorises ("the held-out error is 2.3 × the training error: the map is too fine for this many samples").  **The limits of the saved model come from these out-of-fold errors** (log-scale median + z·MAD as `kinds/stats.py` of the autoencoders; doubled with fewer than 3 values).
* **M Map-size scan** (job `som-scan`, kind `size`): the quality numbers and the held-out QE for a list of unit counts (`n/8 … 4√n … 10√n`, ≤ 8 sizes), a chart with the suggested size (the smallest map whose held-out QE is within one SE / 2 % of the best, with TE < 0.1 preferred) and **Use this size**.
* **M Training curve**: energy and QE per epoch with the two phases marked; the verdict "settled" / "still moving (train longer)".
* **S Stability**: `n_runs` random initialisations (and optionally bootstrap resamples): for every pair of samples, the share of runs in which they land on the same or adjacent units (co-location), summarised as the mean over pairs that are co-located on the kept map, and the adjusted Rand index between the second-level clusterings of the runs; the verdict "the neighbourhoods are / are not reproducible".  Job `som-scan` kind `stability`.
* **S Metric comparison** (job `som-scan` kind `metric`): the same map under `l2` / `mahalanobis` (α = 1, 0.5) / `prior` / `standardize`: trustworthiness, TE, held-out QE and the agreement of the resulting second-level clusters, so a person can see what the metric changes before choosing it.
* **C Statistical reliability of the neighbourhoods** (de Bodt, Cottrell & Verleysen 2002): the bootstrap distribution of the neighbourhood relation vs a random map.

Every verdict is a server sentence (`core/diagnostics.py`); the page shows it and applies no rule of its own.

---

## 7. Two-level clustering (step 7 "Groups" and the published `som_cluster`)

Vesanto & Alhoniemi (2000): cluster the **prototypes** (in the training metric, i.e. the whitened coordinates) with `fluoviz.analysis.clustering.cluster_scores` — `hierarchical` (Ward) by default, `kmeans` as the SOM Toolbox does (Davies-Bouldin), `gmm` — and `choose_k` over `2 … min(10, units/2)` with its vote and reason; option **leave out empty units** (default on when more than half of the units are empty: clustering 100 interpolating prototypes finds structure that no sample has).  Every sample takes the cluster of its unit; clusters are renumbered by decreasing number of *samples* (not units); a cluster that is not contiguous on the lattice is said so in words (a warning, not an error).  Then the usual interpretation, reusing the clustering module as it is: **cluster mean EEMs and z-maps** (`cluster_mean_images` on the samples), **profiles** of the planes per cluster, **metadata associations** (`metadata_associations`, `crosstab_against`), typical samples (the unit nearest the cluster centre and its most central sample).  Buttons **Cluster the map** (recompute with other settings; `POST /{rid}/cluster`, stored in the result, publishes `som_cluster`), **Open in Clustering & LDA** (`go('/cluster', {keys: 'res:som-N:som_x,res:som-N:som_y'})` — the map positions as the score space, so every method of that page and the LDA / separation tab work on the map) and **Separate the groups (LDA)** (`tab=separation` with `som_cluster` suggested as the grouping, if the page supports a preset grouping; otherwise just the tab).

---

## 8. Published variables, the saved model `som`, the chain

**Published by the result and by the saved model (same names):** `som_unit` (category: `"r3c5"` labels, 1-based), `som_row`, `som_col` (numeric), `som_x`, `som_y` (the continuous position; these are the ones to plot and to cluster), `som_qe` (training metric), `som_qe_rmse` (data units), `som_qe_rel`, `som_cluster` (category, once §7 ran; `"none"` before), `som_umatrix` (U-matrix value at the unit: "sits on a ridge"), `missing_part`, `heldout_qe` (result only), `flag` (words), `status`.

**Saved kind `som`** (`api.add_saved_kind("som", "Self-organising map", SomApplier, result_kinds=("som",), freezer=freeze_som)`, `meta["addon"] = {"id": "som", "version"}`).  Frozen (modelled on `kinds/freeze.py` of the autoencoders; everything from `payload["frozen"]`, no recomputation): the map (`rows, cols, lattice, shape`, prototypes `(M, P)`), the `Metric` (`U, λ, ρ, α`, block weights, per-coordinate SDs), the `SomFrame` (per block the `FrozenElement` of the Functional PCA kind with the fixed smoothing, the mean function; pixels: domain, mean, mask), `τ` of §5.2, the clusters (unit → cluster), the reference (training ids, positions, QEs, out-of-fold QEs, the limits at 95 / 99 %), the validation summary, `options`, `outputs`, `description`, `needs`, the recipe (`capture_recipe`).  Freezing refuses a stale result and re-reads six training samples through the recipe to check they land on their own units with their own QE (1e-6), as the autoencoders do.

**Apply**: new samples → coordinates (fixed pre-treatment, smoothing, gap estimate, centring; `min_coverage` 0.8 as the router's) → whitening → BMU, second BMU, position, QE (both), unit, cluster.  **Status**: `unusable` (a variable missing, pre-treatment undefined, coverage below `min_coverage`: all NaN); `outside` ("Poorly described by the map") when the QE is above the limit (option `limit` 95 / 99, default 99); `uncertain` when the BMU is an **empty unit** (no training sample landed there: the sample sits between what the map knows) — the reasons in words.  The generic payload of `models.md`: `plot` (the map positions, the training samples as the reference cloud), `diagnostics` (QE against the distance of the position from the nearest occupied unit, with the limits), `limits`, `images` (`mean` and the prototype of the BMU per sample is too much: give the mean EEM and the `directions` = the first two principal directions of the prototypes), `notes`; `inspect(index)`: measured, prototype of its unit, difference, position on a small map.

**Chain**: a classifier or clustering built on `res:som-N:som_x, som_y` (or `som_cluster` as the grouping) embeds this model as a part; an upstream `outside` / `uncertain` makes the classifier's status `outside`.  Test it through the real `/api/models` router with an LDA on `som_x, som_y` as the autoencoders tests do (`test_api_models.py`).

---

## 9. API (`/api/addon/som/…`; write `docs/api.md` first, as A2 did, with every shape)

* `GET /defaults?variables=&min_coverage=&abs_processed=&advise=` — facts and advice: `n`, pixels / coefficients, the fPCA-style representation advice (with `advise=1`), the **map-size presets** with the numbers (`compact`, `standard`, `fine`: rows, cols, units, the eigenvalue side ratio), the recommended request (functional on, GP basis, `l2`, hexagonal sheet, standard size, batch, gaussian, 5 folds, seed 0), warnings and blocking reasons in words, the time estimate, `versions`.
* `POST /check` (job) — what a request would do (blocks, `n_inputs`, map, metric with `r`, the validation plan, warnings).
* `POST /fit` (job) — the map, result `som-N`; `replace` for "Compute again with the same settings"; the quality numbers and the k-fold are part of the fit (seconds), so no second step is needed for the checks.
* `POST /scan` (job: `size` | `stability` | `metric`).
* `GET /{rid}` — the payload for the page (turned to browser orientation; images float32).  `GET /{rid}/sample/{sid}?thumb=`.  `GET /{rid}/unit/{index}` (the unit card).  `POST /{rid}/plane` (compute / fetch a plane: `{"kind": "index", "key": "HIX"}`, `{"kind": "point", "ex": 350, "em": 450}`, `{"kind": "meta", "key": "meta:site", "smooth": true}`, …).  `POST /{rid}/cluster` (two-level clustering; stores and publishes).  `POST /{rid}/trajectories`.  `GET /{rid}/projection` (Sammon).  `POST /{rid}/anomalies`.  `GET /{rid}/export?what=positions|prototypes|planes|clusters|history` (CSV / `.npz`).  `GET /{rid}/methods` (the paragraph with references).
* Conventions: those of the autoencoders `docs/api.md` §0 (JSON encoding, orientation, errors `{"error": {"message", "kind"}}`, jobs, staleness with the message "… Use 'Compute again with the same settings' …", plain English).  Every bad setting is an immediate 400 with the sentence that says what to do (`core/spec.py`: `Config` with the same style of checks as the autoencoders').

### The request (`params` of the result; the body of `check` and `fit`)

```jsonc
{
  "input": { "kind": "functional" | "pixels" | "variables",              // "functional" and "pixels" are the fpca vocabulary below; "pixels" = basis.kind "pixels" for every variable (a convenience)
             "variables": ["fluorescence", "absorbance"], "weights": "variance" | "none" | {"fluorescence": 1, "absorbance": 0.5},
             "fluorescence": {"transform": "none", "min_coverage": 0.5, "gaps": "auto", "basis": {...}, "smoothing": {...}},
             "absorbance":   {"transform": "none", "processed": true, "gaps": "auto", "basis": {...}, "smoothing": {...}},
             "keys": ["res:parafac-1:C1_Fmax", ...], "standardize": true },                 // variables mode only
  "metric": {"kind": "l2" | "mahalanobis" | "prior" | "standardize", "alpha": 1.0, "components": "auto" | 5},
  "map":    {"size": "standard" | "compact" | "fine" | [rows, cols], "lattice": "hexagonal" | "rectangular", "shape": "sheet" | "toroid"},
  "train":  {"mode": "batch" | "sequential", "init": "linear" | "random", "neighbourhood": "gaussian" | "cutgauss" | "bubble" | "epanechnikov",
             "epochs_rough": 50, "epochs_fine": 150, "radius_initial": "auto" | number, "radius_final": 1.0, "alpha_initial": 0.05, "alpha_final": 0.01},
  "gap_weighting": "observed" | "equal",
  "validation": {"folds": 5, "limit": 99},
  "cluster":  {"method": "hierarchical", "k": "auto" | 4, "k_range": [2, 10], "skip_empty": "auto" | true | false},   // run at the end of the fit with these settings (k "auto" = choose_k's suggestion)
  "seed": 0, "title": "", "allow_few_samples": false
}
```

---

## 10. The page (`web/page.js`; write `docs/ui.md` with every label exactly, as A3 did; the Help guide quotes it and a test checks the labels exist)

Title **Self-organising map (add-on)**, pill *add-on*, lead paragraph, **How to read this**; `ResultBar kind="som"` with `recompute-label="Compute again with the same settings"`; `JobBar` while a fit or scan runs.  Stepper of nine steps, reusing the Functional PCA step components through thin wrappers (your `SomStepsReuse.js`, modelled on `AeStepsReuse.js`; state in `SomState.js` via `createFpca({steps, stateKey: 'som', ...})` + `provideFpca`; the choices persist per project through the same `PUT /api/state/ui` mechanism):

1. **What to map** — the three cards **Fluorescence**, **Absorbance**, **Both together (multivariate)** with pre-treatments and the weights (offered here, unlike the autoencoders), the samples left out; below them the fold **Other variables instead** (variables mode: the column picker of the Clustering page; choosing it dims steps 2–4 with "not needed for variables"); **Run with the recommended settings**.
2. **Domain** — the Functional PCA's step (`method="the map"`).
3. **Representation** — the functional switch (**On the coefficients of a basis (functional)** / **On the pixels**) and the basis cards, as the autoencoders page.
4. **Smoothing** — the Functional PCA's step.
5. **Map** — **Distance between samples** (cards: **Plain distance (L2)** "the usual distance between EEMs: big differences count big", **Standardised (Mahalanobis)** "every direction of variation counts the same, so subtle shape differences matter as much as the overall intensity" with **Components** (`r`, the advice's value) and **How far to standardise** (α slider 0…1), **Gaussian-process prior** (GP basis only), **Standardise every pixel / variable**; the metric comparison button **Compare the distances** → the chart); **Size of the map** (the three presets with their numbers and the honest sentence, **Custom** rows × cols, the **Map-size scan** button and its chart with **Use this size**); **Shape** (**Hexagonal** / **Rectangular**; **Flat** / **Wraps around (toroid)**); a fold **Advanced training settings** (batch / sequential, initialisation, neighbourhood, epochs of each phase, radii, learning rates); right: the card **The map in numbers** (`POST /check` 700 ms after a change: inputs, units, samples per unit, empty units expected, metric, validation, time).
6. **Compute** — **What will be done**, **Folds**, **Random seed**, **Title**, **Compute the map**, **Cancel**; the training curve when done; then step 7.
7. **Map** — the tabs **Map**, **Prototypes**, **Planes**, **Groups** (§7: the clustering controls, the cluster map, mean EEMs and z-maps, metadata associations, **Cluster the map**, **Open in Clustering & LDA**), **Trajectories**, **Samples**, **Projection**.
8. **Checks** — the tabs **Quality** (the numbers, the verdicts, the comparison with the PCA / fPCA plot), **Held-out error** (the k-fold chart, per-sample out-of-fold QE on a map), **Map size** (the scan), **Stability**, **Training**.
9. **Use** — the published variables, **Plot the map positions**, **Cluster the map positions** (Clustering & LDA), **Save this model**, **Apply a saved model**, **Change the settings and compute again**, downloads (positions CSV, prototypes `.npz` (EEMs on the grid + curves), planes CSV, clusters CSV, history CSV), the **Methods paragraph** with **Copy**.

Drawing the map: one component `SomGrid.js` (SVG hexagons / squares through Plotly's shapes or a plain SVG inside `<FigureHost>` so the export buttons exist; a hexagon per unit coloured by the plane with `sequentialScale` / `divergingScale` (diverging for signed planes, centred at 0), the dots of the samples, the cluster borders, hover and click; keyboard-reachable; light and dark; no sideways scroll at 600 px: the grid scales to the width).  Prototype thumbnails: `<EEMThumb>` (they follow the axes setting by themselves).  Every chart: `.rds` / `.fig` / PNG / CSV buttons.  Colours only from `tokens()`.

**Help guide** (`web/help.js`, `help/*.js`, the `guide` CSS classes; 7–9 pages): *What a self-organising map is* (with a small drawn example of 20 points on a 4 × 3 map, numbers generated by `tests/help_data.py` as the autoencoders do); *Pixels or coefficients, and the distance* (§3 in plain words, with the Mahalanobis explanation and when to use it); *How big a map* (the presets, empty units, the scan); *Reading the map* (U-matrix, hits, prototypes, planes, label maps, trajectories; what a ridge is; what an empty unit is); *Groups*; *Is the map any good?* (QE, TE, held-out error, trustworthiness vs the PCA plot, stability, what the verdicts mean); *New samples and the saved model* (statuses); *What the literature does* (Bieroza 2009, Ejarque-Gonzalez & Butturini 2014, Cuss & Guéguen 2016, Cuss et al. 2019, Zhang et al. 2015, Yang et al. 2019 — which input each used, map sizes, what they drew; the four input modes of Fluor_SOmap map onto this page's modes); *Glossary*.

---

## 11. Methods paragraph (`server/methods.py`) and references

Generated from the resolved request: variables and pre-treatments; representation and smoothing (reuse `basis_words`, `smoothing_words`); the metric (with `r`, `α`); the map (rows × cols, lattice, shape, units); initialisation and training (mode, neighbourhood, phases, epochs, radii); the quality numbers (QE, TE, held-out QE, trustworthiness vs PCA); the two-level clustering (method, k, how chosen); the seed and versions.  References, printed with it:

* Kohonen T. (2013) Essentials of the self-organizing map. *Neural Networks* 37:52–65.  ·  Kohonen T. (2001) *Self-Organizing Maps*, 3rd ed., Springer.
* Vesanto J., Himberg J., Alhoniemi E., Parhankangas J. (2000) SOM Toolbox for Matlab 5. Report A57, Helsinki University of Technology (map size, phases, U-matrix, missing values).
* Vesanto J., Alhoniemi E. (2000) Clustering of the self-organizing map. *IEEE Trans. Neural Networks* 11(3):586–600.
* Ultsch A., Siemon H.P. (1990) Kohonen's self organizing feature maps for exploratory data analysis. *INNC'90* (U-matrix).
* Kiviluoto K. (1996) Topology preservation in self-organizing maps. *Proc. ICNN* 294–299 (topographic error).  ·  Bauer H.-U., Pawelzik K. (1992) *IEEE Trans. Neural Networks* 3(4):570–579 (topographic product).  ·  Venna J., Kaski S. (2001) Neighborhood preservation in nonlinear projection methods. *ICANN 2001*, LNCS 2130:485–491 (trustworthiness, continuity).  ·  Forest F., Lebbah M., Azzag H., Lacaille J. (2020) A survey and implementation of performance metrics for self-organized maps. arXiv:2011.05847 (`somperf`: cross-check your formulas against it).
* Rossi F., Conan-Guez B., El Golli A. (2004) Clustering functional data with the SOM algorithm. *ESANN 2004*:305–312 (the functional SOM: SOM on `U·α` with the Cholesky factor of the Gram matrix = our L2 frame).
* Sammon J.W. (1969) *IEEE Trans. Computers* C-18:401–409.  ·  de Bodt E., Cottrell M., Verleysen M. (2002) Statistical tools to assess the reliability of self-organizing maps. *Neural Networks* 15:967–978.
* Fluorescence: Bieroza M., Baker A., Bridgeman J. (2009) *J. Geophys. Res. Biogeosci.* 114:G00F07 (SOM of 625 whole EEMs, 15 × 8 map, k-means on prototypes); Bieroza, Baker, Bridgeman (2011) *Environmetrics* 22:256–270; (2012) *Educ. Chem. Eng.* 7:e22–e31 (tutorial); Ejarque-Gonzalez E., Butturini A. (2014) *PLoS ONE* 9(6):e99618 (270 EEMs each divided by its maximum, kohonen R, stability by de Bodt et al., a second SOM of the component planes); Cuss C.W., Guéguen C. (2016) Analysis of dissolved organic matter fluorescence using self-organizing maps: mini-review and tutorial. *Anal. Methods* 8:716–725 (Fluor_SOmap: whole EEMs / chosen pixels / PARAFAC scores / composition); Cuss C.W. et al. (2019) *Environ. Chem.* 16:360–374 (EEM-PARAFAC-SOM); Zhang H. et al. (2015) *Sci. Rep.* 5:16079 (36 EEMs, 10 × 3 map); Yang W. et al. (2019) *Sci. Total Environ.* 654:1187–1195 (trajectories over seasons); Wheeler K.I., Levia D.F., Hudson J.E. (2017) *J. Geophys. Res. Biogeosci.* 122:2233–2250.
* Alternatives named in the Help guide but not implemented: GTM (Bishop, Svensén, Williams 1998 *Neural Computation* 10:215–234), growing hierarchical SOM (Rauber, Merkl, Dittenbach 2002), relational SOM (Olteanu, Villa-Vialaneix 2015), supervised XY-fused SOM (Melssen, Wehrens, Buydens 2006 *Chemom. Intell. Lab. Syst.* 83:99–113).

---

## 12. Tests (`addons/som/tests/`; run as the README says with `FLUOVIZ_ADDONS_DEV` pointing at `addons/som`)

* **Numerics** (`test_core_som.py`, `test_core_metric.py`, `test_core_quality.py`): §4.6 and §3.4; the quality measures on a hand-made 2 × 2 map; trustworthiness of a perfect 2-D embedding = 1; the Sammon stress decreases; k-fold held-out QE ≥ in-sample QE on average; the planes (an index of a prototype equals `compute_index` on the embedded image).
* **Data** (`test_core_data.py`): fluorescence alone, absorbance alone, both, in pixels and functional mode; block weights; gaps (a rectangle cut from a quarter of the samples: the functional coordinates come from the gap-aware estimate, the pixel coordinates keep NaN and the partial distances work; a sample that lacks 30 % lands next to its complete twin); `SomFrame.coordinates` of a training sample equals the training coordinates (1e-10); `to_function` of a prototype is a smooth EEM with NaN outside the domain.
* **Kind and API** (`test_kind_som.py`, `test_api_som.py`, `test_api_models.py`, `test_api_scans.py`, with `api_support.make_app` modelled on the autoencoders'): every endpoint with plain-English errors; fit → result → positions published and visible in `/api/variables`; `replace`; staleness; planes of every kind; two-level clustering publishes `som_cluster` and renumbers by sample count; freeze → file → thaw → apply reproduces the positions of the page (1e-6); another project with other sample names; a planted outlier → `outside`; a sample on an empty unit → `uncertain`; a coarser grid still lands the training samples on their units; LDA chain on `som_x, som_y`; variables mode on PARAFAC scores (fixture: a saved PARAFAC result) with the back-projected prototypes; `allow_few_samples`; cancellation leaves nothing.
* **Help** (`test_help_som.py`): every bold label of `docs/ui.md` exists in `web/`; every page of the guide renders (through FluoViz's help-browser test helper as the autoencoders do).
* **E2E** (`tests/e2e/flow_som.py`, Playwright through FluoViz's `tests/e2e/browse.py`): dev-mode install, the pipeline on the example data (fluorescence; then both variables), every tab, light / dark, 600 px, no console errors, the Help tab, Save this model and apply it on the Saved models page.
* **Framework**: starting FluoViz with the add-on in dev mode imports nothing beyond numpy / scipy (no sentinel is needed; assert `sys.modules` has no `torch` and the start-up time is unchanged).

---

## 13. Distribution

`manifest.json`: `"id": "som"`, `"name": "Self-organising map (SOM)"`, `"version": "1.0.0"`, `"summary"` (one sentence), `"description"` (a paragraph in plain English that also says **no extra libraries are downloaded**), `"requires_fluoviz": ">=1.0.0,<2"`, all five platforms, `"python_package": "fluoviz_addon_som"`, `"entry": "fluoviz_addon_som:register"`, `"conda": {"all": []}`, `"extra_download_mb": 0`, `"libraries_in_server": true`, `provides`: nav `{group: "Experimental methods", id: "som", path: "/som", label: "Self-organising map", badge: "add-on", icon: "grid" (or `apps`), title: "Self-organising map (add-on)", page: "web/page.js", needsData: true}`, help tab `{key: "som", label: "Self-organising map", module: "web/help.js"}`, result kind `{kind: "som", noun: "map", saveable: true}`, saved kinds `["som"]`.  `python scripts/build_addon.py som` must produce `dist/som-1.0.0.zip` and the index; add the add-on to the table of the repository `README.md` and write `docs/README_user.md` (model: the autoencoders' note, without the PyTorch paragraphs).  Bump nothing in the autoencoders add-on.

---

## 14. Order of work, and what to deliver

Work in this order, and keep every step runnable (tests green) before the next:

1. `SPEC.md` (condensed from this brief, with your decisions), `manifest.json`, `fluoviz_addon_som/__init__.py` (`register`), the empty router — the add-on loads in dev mode and shows an empty page.
2. `core/`: `spec.py` (`Config`, the checks and their sentences), `data.py` (blocks, `SomFrame`, both paths, all three variable choices), `metric.py`, `som.py` (lattice, init, batch, sequential, partial distances, weights), `quality.py` (QE, TE, topographic product, trustworthiness / continuity, Sammon), `validation.py` (k-fold, limits), `planes.py` (planes of §5.3 incl. the indices), `advice.py` (size presets, side ratio, warnings), `diagnostics.py` (verdict sentences), `README.md` of the core.  Tests of §12 first group.
3. `kinds/`: `model.py` (`SomModel`: frozen map + frame + metric; `evaluate`), `som.py` (the applier, `judge`, `publish`), `freeze.py`, `stats.py`.  `server/`: `request.py`, `prep.py`, `fitjob.py`, `payload.py`, `results.py`, `views.py`, `scans.py`, `cluster.py`, `exports.py`, `methods.py`, `router.py`; `docs/api.md`, `docs/kind_and_router.md`.  Tests of §12 third group.
4. `web/`: `page.js`, `components/` (`SomState.js`, `SomActions.js`, `SomStepsReuse.js`, `SomStepMap.js`, `SomStepCompute.js`, `SomGrid.js`, `SomMapTab.js`, `SomPrototypes.js`, `SomPlanes.js`, `SomGroups.js`, `SomTrajectories.js`, `SomSamples.js`, `SomProjection.js`, `SomChecks.js`, `SomUse.js`), `css/page-som.css`; `docs/ui.md`.  Then the Help guide and its generated pictures.  E2E flow.
5. `docs/README_user.md`, the repository `README.md` row, the build, the final report.

**Deliver**: the add-on folder complete as above, all tests passing (say exactly which were run and how, with the output of the final run), the built zip, and a final report that lists (a) every place where you departed from this brief and why, (b) the duplicated pieces (§0, rule 1) with their line counts, (c) what was left out with its priority letter, (d) open questions for the lead.  Priorities: everything marked **must** is in 1.0.0; **should** items go in if time allows, in this order: variables mode, sequential training, `prior` metric, stability, trajectories, projection, metric comparison; **could** items are not for 1.0.0.

**Where the brief leaves you a choice, choose the option that is simplest to explain to a scientist who does not program, say so in `SPEC.md`, and do not ask.**  Where you find that FluoViz does not expose something this brief assumed (a function, a component prop, a page hook), do not patch FluoViz: work around it inside the add-on and list it in the final report under (d).
