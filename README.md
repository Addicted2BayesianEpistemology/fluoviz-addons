# FluoViz add-ons

Optional tools for [FluoViz](https://github.com/Addicted2BayesianEpistemology/fluoviz) that are **not part of its installation**: they need large libraries that most people never use.  A person installs one with a single button, **Settings → Add-ons → Install**; FluoViz downloads the add-on and the extra libraries it needs into its private environment, and the new page appears in the menu.  Nothing changes for people who never install one.

| add-on | what it adds | extra libraries |
|---|---|---|
| `autoencoders` | autoencoders (AE) and variational autoencoders (VAE) for EEMs and absorbance, also **functional** versions on the coefficients of the Functional-PCA bases with a `smoothify` penalty; latent-space tools, anomaly detection, saved models | PyTorch (CPU), about 130 MB |

Specification of the first add-on: [`addons/autoencoders/SPEC.md`](addons/autoencoders/SPEC.md); user note: [`docs/README_user.md`](addons/autoencoders/docs/README_user.md).  The framework that loads add-ons (manifest, index, the `register(api)` hook) is documented in FluoViz: `docs/modules/addons.md` and `docs/modules/addons_core.md`.

## Layout

```
addons/<id>/manifest.json            what it is, what it needs, what it adds to the menu
addons/<id>/<python_package>/        the Python code (register(api) is the entry point)
addons/<id>/web/                     the pages and the Help guide (ES modules loaded by the browser)
addons/<id>/tests/                   the tests of the add-on
scripts/build_addon.py               builds dist/<id>-<version>.zip and dist/index.json
```

## Build the files that the Install button downloads

```bash
python scripts/build_addon.py            # every add-on  ->  dist/
```

`dist/index.json` lists the add-ons (sha256, size, manifest); FluoViz reads it from the **source** set in *Settings → Add-ons → Where add-ons come from*: a web address (for example the assets of a release of this repository) **or a folder** (copy `dist/` to a USB stick or a shared drive and choose that folder).  *Install from a file* installs a single `.zip` without any index.

## Develop an add-on against a FluoViz checkout

```bash
cd <FluoViz folder>
FLUOVIZ_ADDONS_DEV=<this repository>/addons/autoencoders bash start.sh
# tests (PyTorch must be installed in FluoViz's private environment, as the Install button does):
FLUOVIZ_ADDONS_DEV=<this repository>/addons/autoencoders bash scripts/run_tests.sh <this repository>/addons/autoencoders/tests -q -p no:cacheprovider
```

`FLUOVIZ_ADDONS_DEV` loads the working copy in place (no download, no copy, no libraries installed).
