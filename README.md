# FluoViz add-ons

Optional tools for FluoViz (an explorer for fluorescence excitation-emission matrices) that are **not part of its installation**: they need large libraries that most people never use.  A person installs one with a single button, **Settings → Add-ons → Install**; FluoViz downloads the add-on and the extra libraries it needs into its private environment, and the new page appears in the menu.  Nothing changes for people who never install one.

| add-on | what it adds | extra libraries |
|---|---|---|
| `autoencoders` | autoencoders (AE) and variational autoencoders (VAE) for EEMs and absorbance, also **functional** versions on the coefficients of the Functional-PCA bases with a `smoothify` penalty; latent-space tools, anomaly detection, saved models | PyTorch (CPU), about 130 MB |

Specification of the first add-on: [`addons/autoencoders/SPEC.md`](addons/autoencoders/SPEC.md); user note: [`docs/README_user.md`](addons/autoencoders/docs/README_user.md).  The framework that loads add-ons (manifest, index, the `register(api)` hook) is documented in FluoViz itself (`docs/modules/addons.md`, `docs/modules/addons_core.md`).

## Layout

```
addons/<id>/manifest.json            what it is, what it needs, what it adds to the menu
addons/<id>/<python_package>/        the Python code (register(api) is the entry point)
addons/<id>/web/                     the pages and the Help guide (ES modules loaded by the browser)
addons/<id>/tests/                   the tests of the add-on
scripts/build_addon.py               builds dist/<id>-<version>.zip and dist/index.json
```

## Install

In FluoViz open **Settings → Add-ons** and press **Install** next to the add-on.  FluoViz downloads the files of the latest release of this repository (`index.json` and one zip per add-on, checked against their sha256) and the extra libraries the add-on needs, into its own private folder.  **Remove** takes it away again.  Add-ons are programs that run on your computer: install them only from sources you trust.

## Publish a release

```bash
git tag v1.0.0 && git push --tags         # a tag starts .github/workflows/release.yml: it builds dist/ and attaches it to a GitHub release
```

The Install button always reads `https://github.com/<owner>/fluoviz-addons/releases/latest/download/index.json`, so the **latest** release must contain the index and the zips of EVERY add-on that should stay installable.

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
