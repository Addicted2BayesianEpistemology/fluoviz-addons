# Self-organising map add-on: the interface (contract for the Help guide and for the tests)

*Words in bold are labels the page shows exactly; the Help guide (`web/help.js`) quotes them with `<Ui>`, and a test looks for every bold label in the files of `web/` and for every `<Ui>` word of the guide in this file.  Words in `code font` are not shown (test hooks, routes).  What the numbers mean and how the server computes them: `SPEC.md`, `docs/api.md`.  Everything the server says (verdicts, notes, warnings, error messages) is shown as it comes, in plain English.*

## Where things are

* Sidebar: group **Experimental methods**, item **Self-organising map** with the small badge **add-on** (it exists only while the add-on is installed and running).  Route `/som`, page title **Self-organising map (add-on)**, `needsData`.
* **Help**: the guide tab **Self-organising map** of the Help page (`#/help?tab=addon:som:som`); the button **How to read this** in the page head opens it.
* Pictures of EEMs (the prototypes, the cards of units and samples, the pictures of the groups) are drawn like every EEM picture of FluoViz and follow the setting *Axes of fluorescence maps* (`prefs.eemTranspose`); the images the server sends stay `(n_ex, n_em)`.
* The page is one **stepper** of nine steps above ONE panel, like Functional PCA.  Steps 1 to 4 are the Functional PCA components themselves (same controls, same routes `/api/fpca/*`); steps 5 to 9 are the add-on's own.

## Page head

* Title **Self-organising map (add-on)** with the pill **add-on**; a lead paragraph; the button **How to read this**.
* Result bar (`ResultBar kind="som"`): the list of computed maps (`som-N`, with "(outdated)" when the data changed), rename, delete, **Save this model** (only for an up-to-date map of EEMs or spectra) and, for an outdated map, the notice "Outdated: … changed since this map was computed." with the button **Compute again with the same settings** (same id, same title, a new map: nothing is computed silently).
* While a map is being computed a bar with the progress, the elapsed time and **Cancel** is shown above the stepper.

## The stepper

**1 · What to map** · **2 · Domain** · **3 · Representation** · **4 · Smoothing** · **5 · Map and distance** · **6 · Compute** · **7 · Results** · **8 · Checks** · **9 · Use**.  Every step has a box **What this step does and what to check**, the plain-English problems of the step (yellow notices, they block **Compute the map**), and **Back** / **Next**.

## Steps 1 to 4 (the Functional PCA components)

* **1 · What to map**: three cards **Fluorescence**, **Absorbance**, **Both together (multivariate)**; per variable **Pre-treatment**; with two variables the card **Weight of each variable** (**Each variable counts equally**, **Natural units**, **Custom**); **Samples that are left out**; the button **Run with the recommended settings**.  Below the panel the card **Other variables instead** with the switch **Map other variables instead of the EEMs and spectra**, **Add a variable**, **Standardise the variables (divide each by its standard deviation)**.
* **2 · Domain**, **3 · Representation**, **4 · Smoothing**: the Functional PCA steps.  Step 3 starts with the card **Functional version** with the choice **On the coefficients of a basis (functional)** / **On the pixels**.  While **Other variables instead** is on these steps say they are not needed.

## Step 5 · Map and distance

* **Distance between samples**: four cards **Plain distance (L2)**, **Standardised (Mahalanobis)**, **Gaussian-process prior**, **Every pixel or variable standardised**; for the standardised distance **Directions to standardise** (**From the data** / **I choose**) and **How far to standardise**; the button **Compare the distances** and its table.
* **Size of the map**: cards **Compact**, **Standard**, **Fine**, **Custom** (fields **Rows**, **Columns**); the sentence under them; the button **Map-size scan**, its verdict, the chart **Error against the size of the map** and the button **Use N units**.
* **Shape of the map**: **Lattice** (**Hexagonal** / **Rectangular**) and **Edges** (**Flat** / **Wraps around (toroid)**).
* A fold **Advanced training settings**: **Training mode** (**Batch** / **Sequential**), **Starting point** (**Linear** / **Random**), **Neighbourhood function**, **Epochs, rough phase**, **Epochs, fine phase**, **Starting radius**, **Final radius**, **Learning rate, start**, **Learning rate, end**, **Samples that lack part of the domain** (**Weigh by what was measured** / **Count every sample alike**).
* Right: the card **The map in numbers** (the answer of `POST /check`, 700 ms after the last change).

## Step 6 · Compute

* The card **What will be done**; a fold **Settings of the check**: **Folds**, **Random seed**, **Limit for unusual samples**, **Title of the result**; the switch **Compute on fewer than 8 samples anyway** (only with 4 to 7 samples).
* **Compute the map** (primary), **Cancel**; the progress bar **Computing the map**; "Another computation is running: … starts right after it".  When it finishes the page goes to step 7.

## Step 7 · Results

Four cards (**Captured on held-out samples**, **Units**, **Against the scatter of the first two components**, **Groups**), the verdict sentence, and seven tabs:

* **Map**: **Colour the units by**, **Place the samples** (**At their position** / **On their unit**), **Colour by**, **Symbol by**, **Draw the borders between groups**; the map; the card of the sample under the pointer (**Hover over a dot**, **Open in Samples**); the card of a unit (**Prototype**, **Mean of the samples of the unit**, **Mean of the samples minus prototype**, **Samples on the unit**).
* **Prototypes**: **Show** (**The prototype** / **The mean of the samples** / **Samples minus prototype**), **Colour scale** (**The same for all units** / **Own for each unit**).
* **Planes**: **Planes to show**, **Add a plane** with the kinds **An index of the prototypes**, **Intensity at a pixel**, **Intensity in a box**, **Overall intensity**, **Absorbance at a wavelength**, **A variable of the samples**; **Add this plane**.
* **Groups**: **Cluster the map** (**Method**, **Number of groups**, **Empty units**), **Groups on the map**, **Sizes**, **Mean EEM of each group**, **What characterises each group**, **Links to the metadata**, **Open in Clustering & LDA**, **Separate the groups (LDA)**.
* **Trajectories**: **Order by**, **One line per**, **Draw the paths**, **Paths over the map**.
* **Samples**: **The samples** (filters **All**, **Poorly described**, **Lack part of the domain**; **Find a sample**), the card of the sample (**Measured**, **Prototype of the unit**, **Difference**), the card **Unusual samples** (**Method**: **Median plus c × MAD** / **Quantile of the distances** / **Limit of the saved model**; **Find them**).
* **Projection**: **The prototypes by their true distances**.

## Step 8 · Checks

Six tabs: **Quality** (the numbers **Captured (trained on)**, **Captured (left out)**, **Topographic error**, **Topographic product**, **Trustworthiness**, **Continuity** and the verdicts), **Held-out error** (the chart **Distance to the best unit: trained on and left out**), **Map size** (**Map-size scan**), **Stability** (**Number of runs**, **Groups compared**, **Resample the samples (bootstrap)**, **Check the stability**), **Distances** (**Compare the distances**), **Training** (the chart **The energy of the map during training**).

## Step 9 · Use

* **Variables made by this map**; **Use the positions**: **Cluster the map positions**, **Plot the map positions**.
* **Keep the map**: **Save this model**, **Apply a saved model**, **Change the settings and compute again**; for a map of other variables the note that it cannot be saved.
* **Downloads**: **Positions (CSV)**, **Prototypes (.npz)**, **Planes (CSV)**, **Groups (CSV)**, **Training history (CSV)**.
* **Methods paragraph** with **Copy** and the references.

## Fixed phrases of the page (not from the server)

* Few samples: "Only N samples are included: a map of fewer than 8 samples shows very little. Add more samples, or switch on 'Compute on fewer than 8 samples anyway' in step 6 (never fewer than 4)."
* Outdated: "The data changed since this map was computed" followed by what can still be looked at and the pointer to **Compute again with the same settings**.
* Queue: "Another computation is running: … starts right after it (a newer request replaces this one)."
* Empty state: "No map has been computed yet. Choose the settings and press **Compute the map** in step 6, or press **Run with the recommended settings** in step 1."

## Test hooks

`data-som-page` (the step), `data-som-resultbar`, `data-som-metric=<kind>`, `data-som-size=<preset>`, `data-som-run`, `data-som-cancel`, `data-som-cards`, `data-som-tabs7`, `data-som-tabs8`, `data-som-grid`, `data-som-protos`, `data-som-unit-card`, `data-som-plane*`, `data-som-group*`, `data-som-traj*`, `data-som-sample*`, `data-som-anom*`, `data-som-quality`, `data-som-vars`, `data-som-download=<what>`, `data-som-methods`, `data-som-guide`.
