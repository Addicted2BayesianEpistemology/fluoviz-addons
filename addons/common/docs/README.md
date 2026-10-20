# The shared add-on (`common`)

Code that the add-ons **Autoencoders** and **Self-organising map** both need, so that it exists once.  It is a *library add-on*: `"library": true` in `manifest.json`, no page, no guide, no route, no saved model.  FluoViz installs it together with the first add-on that names it in `requires_addons`, starts it before them (so they can `import fluoviz_addon_common`), updates it when an add-on needs a newer one, and removes it with the last add-on that uses it.  The person never sees an Install button for it: Settings → Add-ons lists it under *Installed* as "Shared code", with the add-ons that use it.

The rules of the framework are in FluoViz `docs/modules/addons.md` ("Add-ons that need other add-ons").

## Python (`fluoviz_addon_common`)

| module | what it holds | used by |
|---|---|---|
| `serial` | `allow(cls)`: lets a dataclass be stored in a saved model | the data classes of both add-ons |
| `limits` | `log_limits(values, fallback, squared, words)`: the 95 % and 99 % limit of an error or a distance, on the log scale, from samples the model was not trained on | `kinds/stats.py` of both (`error_limits`, `distance_limits`) |
| `functional` | what a request says about the basis, smoothing and gaps; `variables_from_arrays`; the pixel and Gaussian-process pieces of the data preparation (`centred_pixels`, `pixel_element`, `total_variance`, `gp_prior_scale`) | `core/data.py` of both |
| `project_data` | the project's data as functional variables (`make_variables`, `abs_matrix`, `ife_of`), the cache of the prepared data (`PreparedCache`), the samples of a result as an `ApplyData` | `server/prep.py` of both |
| `results` | `Results`: finding a result, its model, whether it is outdated; images in the orientation of the browser | `server/results.py` of both |
| `freeze` | `FreezeKit`, `pack_frozen`, `unpack_frozen`, `saved_from_frozen`, `freeze_result`: freezing a result as a saved model and reading it back, with the check that the model reproduces its own training samples | `kinds/freeze.py` of both |
| `kinds` | `register_standalone`, `declare_heavy`, `check_grid` | `kinds/__init__.py`, `__init__.py`, `kinds/model.py` of both |
| `routes` | `num_query`: a number of a query string inside its limits, or a plain 400 | `server/router.py` of both |

Everything is numpy only except the `server` side (`project_data`, `results`, `routes`: FastAPI and FluoViz's server) and what imports FluoViz's saved-model machinery (`freeze`, `kinds`).  The modules import those lazily or at the top as the code they replaced did; importing an add-on still loads no heavy library.

The words of a message belong to the add-on that says it ("This map ...", "Train again with the same settings"): the shared code receives them (`FreezeKit`, `Results`, `LimitWords`, the `nothing` sentence of `centred_pixels`) and never contains a sentence that names one add-on.

## Browser (`/addon-web/common/web/...`)

| file | what it holds |
|---|---|
| `words.js` | `pct`, `nTxt`, `plural`, `listWords`, `useNarrow` |
| `parts.js` | `makeParts({ name, prefix })` → `{ Stat, Verdict, Section }`: the number card, the verdict notice and the titled block of a step, with the add-on's own CSS class and test hook (`som-stat`, `data-som-verdict`, `som-section`) |
| `help.js` | `helpParts(TAB)` → `{ Ui, GLink, PLink }`: the interface word and the two kinds of link of a Help guide |

A page imports them by absolute address, like the components of FluoViz: `import { makeParts } from '/addon-web/common/web/parts.js';`.

## What does NOT belong here

* Anything specific to one model: networks, maps, their settings, their pages, their guides.
* **Classes that are written into saved model files** (`AEFrame`, `SomFrame`, `Metric`, ...).  A saved file records the module of each class it holds, so a class that moved would no longer open: such classes stay in their add-on (`serial.allow` only registers them).
* The training processes.  A worker process of an add-on imports this package like any other module (a spawned process gets the parent's `sys.path`).

## Versions

The shared add-on keeps its Python and browser interface stable within a major version; the add-ons ask for `>=1.0.0,<2`.  A change that would break an add-on is a new major version and is released together with the new versions of the add-ons that need it (an add-on that asks for `<2` is then not offered the new one until it is updated: FluoViz says so in plain words).

## Tests

`tests/` (run like those of the other add-ons, see the README of the repository): the limits and the functional helpers against independent formulas, the freezing steps with their refusals, the manifests of all add-ons against the framework's validator, the built zip, and a real installation: installing the map from a built source installs this add-on first, removing the map takes it away.
