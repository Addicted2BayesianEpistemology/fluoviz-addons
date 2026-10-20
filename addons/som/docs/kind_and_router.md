# The saved-model kind `som` and the router

*The routes are in `api.md`; the numerics are `core/README.md`; the contract of saved models is FluoViz's `docs/modules/models.md`.*

```
fluoviz_addon_som/
    __init__.py            register(api): router, saved kind, result kind (and the job pool for the long jobs)
    core/                  numpy / scipy: data -> coordinates, metric, the map, quality, planes, groups, scans
    kinds/                 numpy only: what a saved model is and does
        model.py             SomModel: the frozen map + how a sample becomes coordinates; MapView (what publishing needs); Evaluation
        som.py               the applier, judge() (status + reasons), publish() (variables), reference_for()
        freeze.py            freeze_som(p, result, title, note), pack_frozen / unpack_frozen (the `payload["frozen"]` format)
        stats.py             the limit of the distance to the best unit (from held-out distances)
    server/                the router (thin): request, prep, advice, fitjob, payload, results, views, scans, exports, methods
```

## What is frozen (`SavedModel.meta`)

| key | content |
|---|---|
| `kind_label` | "Self-organising map", "Functional self-organising map" when a basis is used |
| `model` | a plain tree: `map` {rows, cols, lattice, shape}, `arrays` {`protos_t` (transformed coordinates), `protos_raw` (frame coordinates), `hits`, `unit_cluster`}, `tau` (the scale of the continuous position), `metric` (`Metric`: gain, weights, whitening `U, lam, rho, alpha, mean`), `frame` (`SomFrame`: per variable `{name, kind, K, element (FrozenElement), variance, prior_scale}`), `layout` |
| `reference` | the training samples: `ids`, `scores` (positions), `unit`, `qe`, `heldout_qe`, `limits` (`95`, `99`, words) |
| `validation`, `options`, `outputs`, `description`, `needs` | the checks, the `limit` option, the published variables, a paragraph, what the samples must provide |
| `recipe`, `source`, `addon` | the data recipe of the project (`capture_recipe`), provenance, `{"id": "som", "version": ...}` |

Freezing refuses an outdated result, a map of other variables (*"This map cannot be saved as a model …"*), a damaged one, and a map that does not put six of its own training samples, read through the recipe, on the units and at the distances of the result (1e-6).

## Applying a model

New samples → values at the pixels of each domain → clean-up of the absorbance, pre-treatment → functional: smoothed on the pixels the sample has with the stored parameters (a gap estimated as in training); pixels: `sqrt(q)(y − mean)` with NaN kept → the stored metric → best unit, second unit, position, distances.  **Status**: `unusable` (all outputs NaN: a variable is missing, a pre-treatment is undefined, or less than `min_coverage`, default 0.8, of a domain is measured); `outside` ("Poorly described by the model": the distance to the best unit is above the 95 % or 99 % limit, the option `limit`, default 99); `uncertain` (the best unit is a unit no training sample reached).  `ok` samples with gaps carry an informational reason.  The limit is `distance_limits` of the out-of-fold distances: on the log scale of the squared distance, median + z × max(1.4826 MAD, 0.4), widened by 2 and said to be rough when fewer than 3 held-out samples exist.

**As the first link of a chain.**  A classifier built on `res:som-N:som_x, som_y` (or `som_cluster`) embeds the map as a part (named `som-N`); an upstream `unusable` makes the sample `unusable`, an upstream `outside` / `uncertain` makes it `outside`.  Tested through the real models router with LDA.

## Router: what is not obvious

* **Jobs** `som-fit` and `som-scan` are in `HEAVY_KINDS`; the server runs the fit in a job thread (no separate process: the whole fit takes seconds and needs nothing but numpy).
* **Results.** `save_result(kind "som", params = the replayable request, data_sig = the signature when the fit started)`; if the data changed while computing, nothing is saved.  `replace` computes in place.  `r.obj` caches the numpy model after the first use; it is never saved.  Category variables are stored as plain string arrays (object arrays cannot be reloaded without pickling).
* **One truth for numbers and words.**  The limit and the statuses are `kinds/`; every verdict is `core/diagnostics.py`; `server/payload.py` and the page keep no threshold.
* **The numbers of the page are those of the saved model**: the training samples are read back through `SomModel.evaluate` on the project data and must land on the units of the fit (1e-6) before anything is saved.
