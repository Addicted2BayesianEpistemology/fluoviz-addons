# Autoencoders for FluoViz: a note for the person who installs it

This add-on is **optional and experimental**. FluoViz works fully without it, and people who never install it see no change at all.

## What it adds

* A page **Autoencoders** in the sidebar group **Experimental methods** (it carries the small badge *add-on*), where a neural network learns a few numbers (the **code**) that describe each EEM and rebuilds the EEM from them. It comes in two kinds: the plain **autoencoder**, and the **variational autoencoder**, which can also draw new, similar samples and tell when a sample is unusual. It can work on the pixels or, like the functional PCA, on the coefficients of a smooth basis, with a setting called **smoothify** that makes the rebuilt EEMs smoother.
* A guide, **Help, Autoencoders** (the button *How to read this* on the page opens it): eight short pages for people who have never met a neural network, from "what an autoencoder is" to the glossary.
* The codes as **variables**: they appear in the Plots page and in every selector, and the buttons *Cluster the latent space* and *Classify with the latent space* open Clustering & LDA with them.
* The button **Save this model**: a trained network can be saved and applied to samples you measure later (page *Saved models*, group *New samples*), also inside a classification chain.

The page checks the network honestly (numbers on samples it did not learn from, a comparison with PCA or functional PCA) and says in words when the simpler method is as good. Start with the first two pages of the guide.

## What it downloads

* About **150 MB of extra libraries** (PyTorch, the processor-only version), plus the add-on itself, which is small. They go into FluoViz's own private folder (`runtime/`), **nothing anywhere else on your computer**, and nothing is installed system-wide.
* The download needs an internet connection **once**, at the install. After that the add-on works offline.
* Nothing you do with the add-on leaves your computer. Training uses the processor (no graphics card is needed), in a separate process, so FluoViz stays responsive; a few seconds to a few minutes for the usual few dozen to few hundred EEMs. *Cancel* stops it and saves nothing.

## How to install it

1. Open **Settings** and find **Add-ons**.
2. Find **Autoencoders (AE, VAE)** and press **Install**. The progress is shown in words. It can take several minutes, depending on your connection.
3. The page and the guide appear in the menu without restarting FluoViz (if a restart is ever needed, the page says so).

Only the **computer that runs FluoViz** can install, update or remove add-ons, not another computer that uses FluoViz through the network. Add-ons are programs: install them only from sources you trust.

## How to remove it

1. **Settings**, **Add-ons**, press **Remove** on *Autoencoders (AE, VAE)*.
2. In the window that opens, tick **Also remove the extra libraries it downloaded** if you want the 150 MB back (they are downloaded again if you install the add-on once more). Press **Remove**.

Your **projects are not touched**. The page and the guide disappear from the menu. The autoencoder results kept in a project stay in the project file; you look at them again once the add-on is installed.

## Where things are kept

| What | Where |
|---|---|
| The add-on's files | `runtime/addons/autoencoders/<version>/` inside the FluoViz folder |
| The extra libraries (PyTorch) | the private environment `runtime/env/` of FluoViz |
| A trained network (*autoencoder-1*, *autoencoder-2*, ...) | in the project, like every result; it is kept when you save the project |
| **Saved models** (the button *Save this model*) | in FluoViz's workspace folder, in `models/` (one small file pair for each model: they belong to this installation, not to a project). *Export* on the page *Saved models* makes a single `.fluoviz-model` file you can send to a colleague |
| Downloads (codes as CSV, rebuilt samples, training history, the methods paragraph) | where your browser saves downloads |

**Saved models stay when you remove the add-on.** Without the add-on FluoViz still lists them and says "This model needs the add-on 'Autoencoders (AE, VAE)' (Settings > Add-ons)" if you try to use one; after you install the add-on again they work as before. A colleague who gets your `.fluoviz-model` file needs the add-on too.

## If something goes wrong

* **"It could not start"** in Settings, Add-ons: the reason is written there. Remove the add-on and install it again.
* **"Training needs PyTorch"** on the page: the extra libraries are missing. Remove the add-on with the box *Also remove the extra libraries* ticked and install it again.
* A refusal in red on the page (for example "Only 9 samples are included: ...") is a reason in words, not a bug: the page does not train on fewer than 12 samples.
* The network gives a different result on another computer or after an update: the same seed gives the same result on the same computer and the same versions; the methods paragraph records the seed and the versions.

## Good to know before you rely on it

An autoencoder is far more flexible than a few dozen EEMs can support. Look at the numbers for held-out samples and at the comparison with PCA before anything else, never use the *synthetic* samples a variational autoencoder draws as if they were measurements, and give a code a name only after you have looked at what it does. The guide says how.

Version 1.1.0.
