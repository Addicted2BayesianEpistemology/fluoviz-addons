# core: the numerics of the Self-organising map add-on

Pure numpy / scipy (the Functional PCA of FluoViz supplies the bases and the smoothing; FluoViz's clustering module supplies the grouping of the units).

| module | what it does |
|---|---|
| `spec.py` | `Config` (the settings, checked in plain English), `SomError` |
| `data.py` | variables → raw coordinates (`Prepared`, `SomFrame`): functional (frame coordinates of the smoothed coefficients), pixels (`sqrt(q)(y − mean)`, NaN kept), or a table of other variables; block weights; samples that lack part of the domain (weights, `missing_part`) |
| `metric.py` | the four distances as a linear map of the raw coordinates followed by the Euclidean distance; the regularised Mahalanobis whitening `L = U diag(λ^-α/2) Uᵀ + (I − UUᵀ) ρ^-α/2`; the number of directions by cross-validation |
| `som.py` | the lattice (hexagonal / rectangular, sheet / toroid), linear and random initialisation, batch and sequential training with partial distances and sample weights, best units, continuous positions, hits, U-matrix |
| `quality.py` | topographic error, topographic product, trustworthiness and continuity, Sammon mapping, crossing edges |
| `fitting.py` | `fit_map`: metric → map → assignment → k-fold held-out distances → quality; `assign` (the one function a new sample goes through) |
| `advice.py` | the three sizes, the side ratio, warnings |
| `planes.py` | basic planes, principal-direction planes, features and indices of the prototypes, planes of a variable of the samples |
| `groups.py` | two-level clustering of the prototypes |
| `scans.py` | map-size scan, stability, comparison of the distances |
| `diagnostics.py` | the verdict sentences (every threshold of the checks lives here) |

**Why training on transformed coordinates is exact.**  The batch update `m_j = Σ_i h_{c(i)j} w_i x_i / Σ_i h_{c(i)j} w_i` is the minimiser of the neighbourhood-weighted energy under *any* quadratic metric; only the search for the best unit depends on the metric.  So the metric is applied as a linear map `x → Lx`, the map is trained in the new coordinates with the Euclidean distance, and the prototypes are mapped back with `L⁻¹` before they are decoded to EEMs.  (Test: `tests/test_core_metric.py`.)

**The batch map is not exactly a descent method** (Rynkiewicz 2008): the energy of one epoch may rise a little when the best units change.  The tests ask for a fall over a phase, not for monotone steps.

**Continuous position** `(x, y)`: the mean of the lattice coordinates of all units weighted by `exp(−d² / 2τ)`, `d²` the squared distance to the prototype and `τ` the mean squared quantisation error of the training samples (on a toroid: around the best unit, the shortest way round).
