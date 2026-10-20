# Self-organising maps for FluoViz: a note for the person who installs it

This add-on is **optional and experimental**. FluoViz works fully without it, and people who never install it see no change at all.

## What it adds

* A page **Self-organising map** in the sidebar group **Experimental methods** (it carries the small badge *add-on*). A map lays your samples out on a small grid of units; every unit holds a typical EEM or absorbance spectrum, and neighbouring units hold similar ones, so similar samples sit together and gradients run across the map.
* It works on the **fluorescence**, on the **absorbance** or on **both together**, on the **observed pixels** (the classic map of whole EEMs) or on the **coefficients of a smooth basis** (the functional version, like the functional PCA), with a choice of distance: the plain one, or a standardised one that gives subtle differences of shape as much weight as the overall intensity. It can also map any other numbers you have for each sample (PCA or PARAFAC scores, fluorescence indices, metadata).
* The map shows the prototypes as small EEMs on the grid, the U-matrix, the hits, planes (pixels, fluorescence indices, metadata) coloured on the map, label maps, paths of the samples over time and groups of units; it checks itself on samples it was not trained on and says in words whether it keeps neighbours together better than a plain scatter of two components.
* A guide, **Help, Self-organising map** (the button *How to read this* on the page opens it): nine short pages for people who have never met a self-organising map.
* The map positions and groups as **variables**: they appear in the Plots page and in every selector, and the button *Cluster the map positions* opens Clustering & LDA with them.
* The button **Save this model**: a computed map can be saved and applied to samples you measure later (page *Saved models*, group *New samples*), also inside a classification chain.

## What it downloads

**Nothing.** The add-on needs only numpy and scipy, which FluoViz already has, so installing it downloads the add-on itself (a few hundred kilobytes) and no extra libraries. Everything you do with it stays on your computer. Computing a map takes seconds for the usual few dozen to a few hundred EEMs; *Cancel* stops it and saves nothing.

## How to install it

1. Open **Settings** and find **Add-ons**.
2. Find **Self-organising map (SOM)** and press **Install**.
3. The page and the guide appear in the menu without restarting FluoViz.

Only the **computer that runs FluoViz** can install, update or remove add-ons. Add-ons are programs: install them only from sources you trust.

## How to remove it

**Settings**, **Add-ons**, press **Remove** on *Self-organising map (SOM)*. Your **projects are not touched**: the maps kept in a project stay in the project file, and you look at them again once the add-on is installed.

## Where things are kept

| What | Where |
|---|---|
| The add-on's files | `runtime/addons/som/<version>/` inside the FluoViz folder |
| Your maps | in your project, with the other results |
