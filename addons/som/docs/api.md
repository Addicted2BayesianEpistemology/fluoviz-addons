# Self-organising map add-on: HTTP API

*Binding with `SPEC.md`.  All routes live under **`/api/addon/som`** (the framework refuses any other prefix).  The add-on also uses, unchanged, the generic FluoViz routes: `/api/fpca/*` (steps 1–4), `/api/jobs/{id}` (poll a job), `/api/variables` (the published variables), `/api/models*` (save / apply / chain the model), `/api/results*` (list, rename, delete).*

## 0. Conventions

* **JSON encoding** is FluoViz's: `NaN` / `±inf` → `null`; a numpy array with more than 1500 numbers → `{"$nd": "f4" | "i4" | "u1", "shape": [...], "b64": "..."}`; smaller arrays are nested lists.
* **Orientation.**  Every image is in *browser orientation*: one EEM is `(n_ex, n_em)`, a stack `(k, n_ex, n_em)`; `ex`, `em` are the wavelength axes and `mask` is `uint8 (n_ex, n_em)`, 1 = not in the model.  Curves are arrays on `abs_wl`, `null` outside the domain.
* **Errors**: `{"error": {"message": "<plain English>", "kind": "user" | "bug"}}`.  400 = something the person can fix, 404 = unknown result / sample / unit, 500 = a bug.  A job that fails ends with `status: "error"` and the same message in `error`.
* **Jobs** (`check`, `fit`, `scan`): the POST answers `{"job_id": ...}` after the request was validated (a bad setting is an immediate 400); poll `GET /api/jobs/{id}`; cancel with `POST /api/jobs/{id}/cancel`.  A cancelled fit saves nothing.
* **Staleness.**  Like every result, `som-N` records the data signature it was computed on; `result.stale` lists the changed parts.  The map is never computed again silently.  A stale result can still be looked at (everything stored in it); endpoints that need the measured data (`sample`, the mean of the samples of a unit, the pictures of the groups) answer 400 *"The data or the corrections changed since this map was computed (…). Use 'Compute again with the same settings' …"*.  **Compute again** = `POST /fit` with the stored `params` plus `"replace": "<rid>"`.
* **Plain English** in every `message`, `words`, `verdict`, `reasons`, `notes`, `warnings` and `text` field; keys are `snake_case`.
* Category variables published by the result are plain string arrays (`""` = no value); numeric ones are float arrays (`NaN` = no value).

## 1. The request of a fit (`params` of the result; also the body of `check` and `fit`)

```jsonc
{
  "input": {
    "kind": "functional" | "pixels" | "variables",       // functional: a basis (default); pixels: the observed values (every variable's basis is "pixels"); variables: arbitrary per-sample variables
    "variables": ["fluorescence", "absorbance"],          // functional / pixels: fluorescence, absorbance or both
    "weights": "variance" | "none" | {"fluorescence": 1, "absorbance": 0.5},   // variance: each variable counts equally (1 / its total variance); none: natural units; a dict: multipliers in natural units
    "fluorescence": {"transform": "none", "min_coverage": 0.5, "gaps": "auto", "basis": {"kind": "gp", ...}, "smoothing": {"method": "ml"}},
    "absorbance":   {"transform": "none", "processed": true, "gaps": "auto", "basis": {...}, "smoothing": {...}},   // the Functional PCA vocabulary, checked as /api/fpca/fit
    "keys": ["res:parafac-1:C1_Fmax", "meta:DOC"], "standardize": true    // variables mode only: keys of the variable catalogue (numeric); divide each by its SD
  },
  "metric": {"kind": "l2" | "mahalanobis" | "prior" | "standardize", "alpha": 1.0, "components": "auto" | 5},
  "map":    {"size": "standard" | "compact" | "fine" | [rows, cols], "lattice": "hexagonal" | "rectangular", "shape": "sheet" | "toroid"},
  "train":  {"mode": "batch" | "sequential", "init": "linear" | "random", "neighbourhood": "gaussian" | "cutgauss" | "bubble" | "epanechnikov",
             "epochs_rough": 50, "epochs_fine": 150, "radius_initial": "auto" | 3.0, "radius_final": 1.0, "alpha_initial": 0.05, "alpha_final": 0.01},
  "gap_weighting": "observed" | "equal",
  "validation": {"folds": 5, "limit": 99},                 // folds 2..20; limit 95 | 99 (the level of the saved model's limit)
  "cluster":  {"method": "hierarchical" | "kmeans" | "gmm", "k": "auto" | 4, "k_range": [2, 10], "skip_empty": "auto" | true | false},
  "seed": 0, "title": "", "allow_few_samples": false
}
```
Every key is optional.  Refusals (immediate 400): *"The distance must be one of: l2, mahalanobis, prior, standardize."*, *"A map needs at least 2 rows and 2 columns and at most 400 units."*, *"Unknown setting 'foo'."*, *"The Gaussian-process prior distance needs functions …"* (variables mode), *"Choose the variables to map: 'keys' must be a list …"*.  Problems found while the data are read are errors of the **job**: fewer than 8 samples (*"Only 5 samples are included: a map of fewer than 8 samples shows very little. …"*; `allow_few_samples` lowers the floor to 4), a missing absorbance, *"The standardised (Mahalanobis) distance needs every coordinate of every sample: with missing pixels use the plain distance or standardise every pixel."*, *"The variable 'meta:site' is not a number …"*.

## 2. Endpoints

### 2.1 `GET /defaults` — facts and advice (synchronous)

Query: `variables` (`fluorescence`, `absorbance` or both separated by a comma), `min_coverage` (0.05–1), `abs_processed`, `advise` (0 | 1; with 1 the coordinates are prepared and the side ratio of the map is computed: a few seconds).

```jsonc
{"n_samples": 36, "ids": [...], "n_common": 36,
 "fluorescence": {"available": true, "n": 36, "grid": [n_em, n_ex], "units": "RU", "masked_fraction": 0.26},
 "absorbance": {"available": true, "n": 36, "n_wl": 40} | {"available": false},
 "advice": {"presets": {"compact": {"rows", "cols", "units", "samples_per_unit", "words"}, "standard": {...}, "fine": {...}}, "side_ratio": 1.4 | null, "warnings": [...], "blocking": [...]},
 "request": { ...the complete recommended request... }, "metrics": [{"kind", "words"}, ...], "sizes": {"compact": "...", ...}}
```

### 2.2 `POST /check` (job) — what a request would do; nothing is fitted

Job result: `{"ok", "n", "dropped", "input": {"variables", "mode": "functional" | "pixels" | "mixed" | "variables", "n_inputs", "blocks": [...]}, "metric": {"kind", "alpha", "components", "words", "name"}, "map": {"rows", "cols", "units", "lattice", "shape", "samples_per_unit"}, "validation": {"folds", "words"}, "estimate": {"seconds", "words"}, "warnings", "blocking"}`.  The prepared data are cached (the last request), so `fit` right after `check` does not redo the basis.

### 2.3 `POST /fit` (job) — fit the map, save `som-N`

Body: a request plus an optional `"replace": "som-3"`.  Job result `{"result": {id, kind: "som", title, created, params, summary, stale, sample_ids, variables}, "rid": "som-1"}` with `summary`: `{n, mode, variables, rows, cols, units, metric, captured, heldout_captured, topographic_error, trust, trust_pca, verdict, n_flagged, groups, seed, seconds}`.

**Published variables** (also those of a saved model, see `kind_and_router.md`): `som_unit` (category, `r3c5`), `som_row`, `som_col`, `som_x`, `som_y` (position in lattice units), `som_qe`, `som_qe_rel`, `som_qe_rmse` (one variable) or `som_qe_rmse_<variable>`, `som_umatrix`, `som_cluster` (category, `group 1` …), `flag`, `status`; the result also `heldout_qe` and, when a sample lacks part of the domain, `missing_part`.

### 2.4 `GET /{rid}` — the result for the page

`{"result": {...}, "payload": {...}}`.  `payload`: `ids, n, mode, variables, kind ("functions" | "variables"), input {blocks, weights, dropped}`, `map {rows, cols, lattice, shape, units, coords (M,2), edges (E,3: unit, unit, distance), umatrix, hits, hits_weight, unit_names, period, proto_dist (M,M)}`, `metric {kind, alpha, components, words, name}`, `samples {unit, unit2, pos (n,2), qe, qe_blocks, qe_rel, heldout_qe, heldout_qe_rel, fold, flag, status, scores (n,4), missing, weight}`, `planes {key: plane}` (`umatrix`, `hits`, `dmean`, `qe_unit`, `score1…score8`, and every plane added later), `quality`, `verdicts`, `training {history {energy, qe, radius, phase}, energy, epochs}`, `kfold`, `clusters {k, unit_labels, sample_labels, sizes, reason, k_values, silhouette, skipped_empty, warnings, method, settings}`, `limits`, `notes`, `warnings`, `versions`, `n_outside`; for maps of functions also `prototypes {block: (M, n_ex, n_em)}`, `prototype_curves {block: (M, n_wl)}`, `mean_image`, `mean_curve`, `ex`, `em`, `mask`, `units`, `abs_wl`; for maps of other variables `prototype_values (M, P)`, `columns`.  The frozen weights are never sent.  A **plane** is `{key, label, values (M,), kind: "numeric" | "category", signed, units, words}`; a category plane also `categories`, `unit_labels`, `purity`, `counts`.

`quality`: `qe_mean, qe_median, qe_rel_mean, captured, qe_l2_mean, topographic_error, topographic_product, trust_map {k, trustworthiness, continuity}, trust_pca {…}, n_units, n_empty, samples_per_unit, pca2_captured` and, with held-out samples, `heldout_qe_mean, heldout_qe_rel_mean, heldout_over_train, heldout_captured, heldout_empty_share`.  `verdicts`: `vs_pca`, `topographic`, `heldout`, `training` (each `{verdict, level}`; levels `good`, `info`, `warn`) and `empty` (a sentence).

### 2.5 Looking at a result

* `GET /{rid}/sample/{sid}?thumb=` → `{id, title, unit, unit_name, row, col, pos, qe, qe_rel, heldout_qe, status, words, flag, cluster, text, reasons, limit, limit_level, blocks: [{name, kind: "surface" | "curve", units, qe, measured, prototype, difference, mean, ex, em, mask | x}], missing, missing_words}` (`thumb=1` leaves out `difference` and `mean`); a map of other variables: `{…, variables: [{name, value, prototype}]}`.
* `GET /{rid}/unit/{index}` → `{unit, name, row, col, hits, members, qe_members, cluster, planes {key: value}, plane_labels, blocks: [{name, kind, prototype, members_mean, difference, x}], ex, em, mask, units}`.
* `GET /{rid}/prototypes?what=prototype|members|difference` → `{what, units, hits, surfaces {block: (M, n_ex, n_em)}, curves {block: (M, n_wl)}, ex, em, mask, units}` (`members` and `difference` need the current data).  A map of other variables answers with the neighbourhood-weighted mean EEM of the samples of each unit.
* `POST /{rid}/plane` — body `{"kind": "stored", "key": "umatrix"}`, a feature (`{"kind": "point", "ex": 330, "em": 450}`, `{"kind": "region_mean" | "region_max", "ex": [a, b], "em": [a, b]}`, `{"kind": "total"}`, `{"kind": "absorbance", "wl": 300}`), an index (`{"kind": "index", "key": "HIX"}`) or a variable (`{"kind": "var", "key": "meta:site", "smooth": true}`); a plane (cached in the result under its `key`).  `GET /{rid}/planes` lists the stored ones, `GET /{rid}/indices` the indices that can be computed.
* `GET /{rid}/groups` → the groups with their interpretation on the current data: `{k, sizes, labels, unit_labels, reason, warnings, k_values, silhouette, profiles, associations, images {mean, difference, z (k, n_ex, n_em), ex, em, n_samples}, images_note}`.  `POST /{rid}/cluster` (body: the `cluster` settings) groups again, stores the result and publishes `som_cluster`; the groups travel with the saved model.
* `POST /{rid}/trajectories` — body `{"order_by": "meta:day", "group_by": "meta:site" | null}` → `{lines: [{group, ids, pos (k,2), order}], order_by, group_by}`.
* `POST /{rid}/anomalies` — body `{"method": "mad", "c": 3}` | `{"method": "quantile", "q": 0.99}` | `{"method": "limit", "level": 99}` → `{method, limit, words, flagged: [{id, qe, unit}], histogram {counts, edges}, uses_heldout}`.
* `GET /{rid}/projection` → `{xy (M,2), stress, crossings, edges, words, hits}` (Sammon mapping of the prototypes).
* `GET /{rid}/methods` → `{text, references}`.  `GET /{rid}/export?what=positions|planes|clusters|history|prototypes` → CSV, or `.npz` for the prototypes.

### 2.6 `POST /scan` (job)

Body: `{"kind": "size" | "stability" | "metric", "request": {...} | "result": "som-1", "units": [8, 16, 30], "n_runs": 10, "bootstrap": false, "k": 4}`.  Results: `size`: `{rows: [{units, rows, cols, qe_rel, heldout_qe_rel, heldout_se, topographic_error, trustworthiness, n_empty}], suggested, verdict}`; `stability`: `{value, baseline, ari, ari_runs, n_runs, bootstrap, k, verdict {verdict, level}}`; `metric`: `{rows: [{kind, alpha, trustworthiness, continuity, topographic_error, heldout_qe_rel, agreement, components} | {kind, alpha, error}], k, reference}`.
