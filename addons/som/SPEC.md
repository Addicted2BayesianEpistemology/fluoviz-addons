# Add-on "Self-organising map" — specification

*The second FluoViz add-on.  The long brief it implements is `PROMPT.md` (same folder); this file is the condensed, binding version plus the decisions taken where the brief left a choice.  Framework: FluoViz `docs/modules/addons.md`.  Model: `addons/autoencoders/` (nothing is imported from it; the pieces modelled on it are listed in section 12).*

## 1. What the person gets
A page **Self-organising map (add-on)** in *Experimental methods*: a nine-step pipeline that lays the samples on a grid of units, each holding a prototype EEM / spectrum.  Works on fluorescence, absorbance or both, on the **pixels** or on the **coefficients of a functional basis** (Functional-PCA machinery: GP / B-spline / cosine, same smoothing, same domain, same gap handling); a secondary **variables** mode clusters any per-sample variables.  Shows the map, checks it on held-out samples, groups the units, publishes the positions as variables, and saves the map as a model that new samples can be applied to.

## 2. Coordinates
* **Raw coordinates** `R` per block.  Functional: frame coordinates `Z = C·R_frame` (Euclidean = L2 of the functions).  Pixels: `sqrt(q_j)·(y_j − ȳ_j)`, `q_j` the domain quadrature weight; NaN where a sample lacks the pixel.
* **Block weights** `w_v`: `"variance"` (default, `1 / V_v`, `V_v` = mean squared norm of the centred block), `"none"` (1), or a dict of multipliers in natural units.
* Everything is held in a `SomFrame` (one `SomFrameBlock` per variable; the Functional PCA `FrozenElement` is frozen inside, so a new sample is smoothed with the fixed parameters).  Prototypes decode to functions in every mode but `variables`.

## 3. Metric  (`core/metric.py`)
`l2` (default) · `mahalanobis` (regularised, partial whitening exponent `alpha`, `components` r) · `prior` (GP basis only) · `standardize`.  Every metric is a **linear map of the raw coordinates followed by the Euclidean distance**: diagonal gain `g` (block weights, SDs, prior SDs) and, for `mahalanobis`, the whitening `L = U diag(λ^-α/2) Uᵀ + (I − UUᵀ) ρ^-α/2`.  Training runs on the transformed coordinates; prototypes are un-transformed (`L⁻¹`, `1/g`) before they are stored and decoded.  Exact, because the batch update is a weighted mean under any quadratic metric.  `mahalanobis` and `prior` need complete coordinates (pixels with missing pixels: use `l2` or `standardize`); partial distances (scaled by the observed share of the metric weight) serve the diagonal metrics.
* Default `components` r = the smallest number whose cross-validated reconstruction error is within one standard error (or 0.5 % of the total variance) of the best, at least 2, at most `min(n−2, 20)`.
* **Decision:** `standardize` divides every coordinate by its SD and then gives each variable the same total weight (`1/K_v`, times a custom weight if given).

## 4. The SOM  (`core/som.py`)
Hexagonal (default) or rectangular lattice, sheet (default) or toroid (hexagonal toroids need an even number of rows: rounded up).  Linear initialisation (default, deterministic) on the two leading principal directions of the transformed coordinates, or random.  **Batch** training (default; 50 rough + 150 fine epochs, linear radius decay, gaussian neighbourhood) or **sequential**; neighbourhoods gaussian / cutgauss / bubble / epanechnikov.  Sample weights `w_i` (gap weighting).  Missing coordinates: partial distances, updates average over the samples that have the coordinate.
* **Map size:** presets *compact* (`≈ n/4` units, at least 6), *standard* (`5√n`), *fine* (`10√n`, at most 400 units); side ratio `√(λ₁/λ₂)` of the transformed coordinates (capped at 4; square when the metric is fully whitened).  Never more than `4n` units, never fewer than 8 samples (4 with `allow_few_samples`).
* **Continuous position** `(x, y)` = mean of the lattice coordinates weighted by `exp(−d²_j / 2τ)`, `d²_j` the squared distance to prototype j, `τ` the mean squared quantisation error of the training samples (on a toroid: around the best unit, unwrapped).

## 5. Checks  (`core/quality.py`, `core/validation.py`)
Quantisation error (training metric and RMS in data units; `qe_rel` = share of the variation not captured), topographic error, topographic product, trustworthiness and continuity (k = 5) of the positions against the same numbers for the 2-D PCA of the same coordinates, **k-fold held-out QE** (the saved model's limits come from these), map-size scan, stability (co-location over random starts), metric comparison.  Each metric is re-fitted inside every fold.  Verdicts are server sentences.

## 6. Groups
Two-level clustering of the prototypes (`cluster_scores` / `choose_k` of FluoViz; hierarchical Ward by default), empty units left out when more than half are empty; clusters renumbered by number of samples; cluster mean EEMs / z-maps, profiles, metadata associations from FluoViz.

## 7. Published variables and the saved model
`som_unit, som_row, som_col, som_x, som_y, som_qe, som_qe_rel, som_qe_rmse (one block) | som_qe_rmse_<block>, som_cluster, som_umatrix, missing_part, heldout_qe (result only), flag, status`.  Saved kind `som`: frozen map, metric, frame, clusters, reference (training positions, QEs, out-of-fold QEs, limits).  Status `unusable` (a variable missing / coverage below 0.8), `outside` (QE above the 95 / 99 % limit), `uncertain` (the best unit is empty).

## 8. API, page, tests
API: `/api/addon/som/` — see `docs/api.md`.  Page: nine steps (`docs/ui.md`).  Tests: `tests/` (run with FLUOVIZ_ADDONS_DEV pointing at this folder and at the shared add-on `addons/common`; the tests do it themselves).

## 9. Decisions where the brief left a choice
* The fit is an ordinary FluoViz job (seconds); no history route and no separate process.
* `impute_pixels` is not used: a pixel map with missing pixels offers `l2` / `standardize` only (partial distances).
* The interstitial U-matrix is not drawn: the U-matrix is per unit (mean distance to the lattice neighbours), with the edge distances in the payload.
* Variables mode: the result, its published variables, the planes, the groups and the pictures (neighbourhood-weighted mean EEMs) are there; **a map of other variables cannot be saved as a model** (the chain machinery of the classifier kinds was not used; the brief allows this).  The page says so.
* Step 5 is named **5 · Map and distance** and step 7 **7 · Results** (the brief named both "Map"); the page needed distinct names.

## 10–11. (Layout, distribution) as in the brief, sections 13–14.

## 12. Shared with the autoencoders add-on (the `common` add-on)
The map was modelled on the autoencoders add-on, and what the two do the same way lives once, in the add-on `common` (`addons/common`, a library add-on that FluoViz installs with either of them; this add-on names it in `requires_addons`):

* `core/data.py`: the request's basis / smoothing / gaps settings, `variables_from_arrays`, the pixel and Gaussian-process pieces of the blocks (`fluoviz_addon_common.functional`), `serial.allow`;
* `server/prep.py`, `server/results.py`, `server/router.py`: the project's data as functional variables, the cache of the prepared data, finding a result and its model, `num_query` (`project_data`, `results`, `routes`);
* `kinds/freeze.py`, `kinds/stats.py`, `kinds/__init__.py`, `kinds/model.py`: freezing and reading back a model, the log-scale limit, registering the kind, the grid check (`freeze`, `limits`, `kinds`);
* `web/components/SomCommon.js`, `web/help/parts.js`: the number card, the verdict, the titled block, the wording helpers and the guide's links (`/addon-web/common/web/*.js`).

What stayed in the add-on is what differs: the blocks and the frame of a map (`SomFrame`, `Metric`: classes that saved models record by module, so they cannot move), the request, the model, the steps of the page and the guide.  `web/components/SomStepsReuse.js` is still a sibling of `AeStepsReuse.js` (the same wrappers with different words).  `docs/README.md` of `common` lists every shared module.
