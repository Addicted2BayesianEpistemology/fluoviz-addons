"""The numbers behind the pictures of the Help guide "Autoencoders" (``web/help/data.js``), computed with the add-on's own code.

Not a test file: ``tests/test_help_autoencoders.py`` imports ``figure_data()`` (recomputes the numbers and compares them with the stored ones) and writes them with ``--write-data``::

    cd <FluoViz folder> && bash scripts/run_tests.sh -q   (the environment)  /  PYTHONPATH=. runtime/env/bin/python <add-on>/tests/test_help_autoencoders.py --write-data

PyTorch is needed (the pictures show networks that were really trained); the test that uses this module skips cleanly without it.

Everything is made-up EEMs (nothing of a person's project): a small grid (29 emission x 15 excitation wavelengths) with the scatter bands removed, and fluorophores that are Gaussian bumps.
Two data sets, both of 120 samples:

* ``linear``: three fluorophores whose amounts vary independently: the sum of three fixed shapes (the case PCA and PARAFAC are made for);
* ``bending``: one fluorophore whose emission band MOVES with a hidden condition ``t`` and whose amount ``a`` varies, plus a weak second one that follows ``a``: two hidden factors, not a sum of
  fixed shapes (the case where the bends of a network pay).

Nine pictures, each from a real computation:

* ``one``      the EEM of one held-out sample of ``linear``, its code and its rebuilding by a network with 3 codes (default settings);
* ``compare``  the error on held-out samples against the number of codes for PCA (functional PCA), the network with ``silu`` and the network with ``linear`` activation (``latent_scan`` of the add-on);
* ``traversal`` the code of a 2-code network on ``bending`` that follows the hidden condition: the decoded EEMs for -3 ... +3 SD and their difference to the middle one;
* ``smoothify`` one held-out sample of a noisy data set: a cut along emission for smoothify = 0 and for a larger value, and the roughness numbers;
* ``lcurve``   the smoothify scan of the same data (``smoothify_scan`` of the add-on);
* ``collapse`` the KL of every code of a variational autoencoder with more codes than the data need;
* ``loss``     the loss curves of a healthy training and of an over-fitted one;
* ``shapes``   eight response curves of known shape and the words ``core.curves.describe_curve`` puts to them (no training);
* ``nonlinear`` the 2-code network of ``bending`` looked at with the tools of the tab Nonlinear effects: the response of the pixel each code moves most with the samples around it, where each code bends, its nonlinear share.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ADDON = Path(__file__).resolve().parents[1]
if str(ADDON) not in sys.path:
    sys.path.insert(0, str(ADDON))

EM = np.arange(290.0, 571.0, 10.0)                        # emission (nm): 29 values
EX = np.arange(240.0, 451.0, 15.0)                        # excitation (nm): 15 values
E, X = np.meshgrid(EM, EX, indexing="ij")
MASK = ~((np.abs(E - X) < 16) | (np.abs(E - 2 * X) < 16) | (E < X - 10))        # Rayleigh, its second order, and below the Rayleigh line: pixels that do not exist
N_SAMPLES = 120
GP_INPUT = {"fluorescence": {"basis": {"kind": "gp"}, "smoothing": {"method": "ml"}}}


def bump(ce, cx, se, sx):
    return np.exp(-0.5 * (((E - ce) / se) ** 2 + ((X - cx) / sx) ** 2))


# ---------------------------------------------------------------------------------------------------------------- the made-up EEMs
def eems_linear(n: int = N_SAMPLES, seed: int = 1, noise: float = 0.01) -> np.ndarray:
    """``(n, 29, 15)``: three fluorophores of fixed shape with independent log-normal amounts, plus noise; NaN where the pixel does not exist."""
    rng = np.random.default_rng(seed)
    shapes = np.array([bump(450, 340, 45, 35), bump(405, 285, 32, 24), bump(340, 275, 26, 14)])
    amounts = np.exp(0.35 * rng.standard_normal((n, 3))) * np.array([1.0, 0.8, 0.6])
    Y = np.einsum("nf,fij->nij", amounts, shapes) + noise * rng.standard_normal((n,) + E.shape)
    Y[:, ~MASK] = np.nan
    return Y


def eems_bending(n: int = N_SAMPLES, seed: int = 1, noise: float = 0.01):
    """``(Y, t, a)``: a band whose emission maximum moves from 375 to 460 nm with the hidden condition ``t`` and whose amount is ``a``, and a weak second fluorophore that follows ``a``."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 1, n)
    a = rng.uniform(0.5, 1.5, n)
    Y = np.array([a_ * bump(375 + 85 * t_, 320, 30, 30) + 0.7 * (1 - a_ / 2) * bump(345, 275, 24, 14) for t_, a_ in zip(t, a)])
    Y = Y + noise * rng.standard_normal(Y.shape)
    Y[:, ~MASK] = np.nan
    return Y, t, a


def eems_noisy(n: int = 40, seed: int = 3, noise: float = 0.08) -> np.ndarray:
    """Noisy EEMs of the ``bending`` kind (8 % noise) and few samples: where a network can draw ripples that are not in the data."""
    return eems_bending(n, seed, noise)[0]


def prepare(Y: np.ndarray, input: dict | None = None, ids=None):
    from fluoviz_addon_autoencoders.core.data import prepare as _prepare
    from fluoviz_addon_autoencoders.core.data import variables_from_arrays
    ids = ids or [f"S{i + 1:03d}" for i in range(Y.shape[0])]
    return _prepare(variables_from_arrays(Y, EX, EM, ids=ids), input or GP_INPUT)


# ---------------------------------------------------------------------------------------------------------------- small helpers
def r(x, d=4):
    """Rounded floats for the JSON (NaN -> None)."""
    a = np.asarray(x, float)
    if a.ndim == 0:
        return None if not np.isfinite(a) else round(float(a), d)
    return [r(v, d) for v in a]


def rows(img: np.ndarray, d: int = 3):
    """An image ``(n_em, n_ex)`` as rows of the picture: excitation from the top (the highest) down, emission to the right; ``None`` where the pixel does not exist."""
    g = np.asarray(img, float).T[::-1]
    return [[None if not np.isfinite(v) else round(float(v), d) for v in row] for row in g]


def grid_of(prep, F: np.ndarray, name: str = "fluorescence") -> np.ndarray:
    """Functions at the observed pixels ``(m,)`` or ``(k, m)`` -> grids ``(n_em, n_ex)`` / ``(k, n_em, n_ex)`` with NaN outside the domain."""
    return prep.embed(name, F)


def rebuilt(prep, result, rows_idx=None) -> np.ndarray:
    """The rebuilt functions of ``result`` for the samples ``rows_idx`` (default all) as grids ``(k, n_em, n_ex)``."""
    from fluoviz_addon_autoencoders.core import forward as fw
    Xh = fw.reconstruct(result.spec, result.arrays, prep.X, prep.M)[0]
    F = prep.to_functions(Xh)["fluorescence"]
    G = grid_of(prep, F)
    return G if rows_idx is None else G[rows_idx]


def config(**kw):
    from fluoviz_addon_autoencoders.core.spec import Config
    return Config(**kw)


# ---------------------------------------------------------------------------------------------------------------- pictures 1 and 2: the autoencoder, and PCA against it
def compare_data(prep_lin, prep_bend) -> dict:
    """The held-out error against the number of codes: the network (``silu``), the network with ``linear`` activation, and the functional PCA, for both data sets."""
    from fluoviz_addon_autoencoders.core import diagnostics as D
    out = {"dims": [1, 2, 3, 4, 5]}
    for key, prep in (("linear", prep_lin), ("bending", prep_bend)):
        ae = D.latent_scan(config(activation="silu"), prep, dims=out["dims"], n_restarts=2)
        li = D.latent_scan(config(activation="linear"), prep, dims=out["dims"], n_restarts=2)
        fam = ae["families"]["ae"]
        out[key] = {"pca": r(ae["baseline"]["val"], 5), "ae": r(fam["val"], 5), "ae_se": r(fam["val_se"], 5), "linear": r(li["families"]["ae"]["val"], 5),
                    "baseline_name": ae["baseline"]["name"], "recommended": ae["recommended"]["latent_dim"], "n": int(prep.n), "n_inputs": int(prep.P)}
    return out


def one_data(prep, result) -> dict:
    """One held-out sample of ``linear`` through the network of 3 codes: the measured EEM, the code, the rebuilt EEM."""
    held = np.flatnonzero(result.samples["heldout"])
    err = result.samples["error"][held]
    k = int(held[np.argsort(err)[len(err) // 2]])                 # the held-out sample with the median error: an ordinary one
    obs = grid_of(prep, prep.observed()["fluorescence"][k])
    reb = rebuilt(prep, result, [k])[0]
    top = float(np.nanmax(obs))
    return {"id": prep.ids[k], "measured": rows(obs / top), "rebuilt": rows(reb / top), "code": r(result.samples["codes"][k], 2), "n_codes": int(result.spec.latent_dim),
            "n_pixels": int(prep.block("fluorescence").domain.m), "n_grid": int(MASK.size), "n_inputs": int(prep.P), "n_parameters": int(result.spec.n_parameters()),
            "r2": r(1.0 - result.samples["error"][k] / max(result.samples["null"][k], 1e-300), 4), "peak": r(top, 3), "ex": [float(v) for v in EX[::-1]], "em": [float(v) for v in EM]}


# ---------------------------------------------------------------------------------------------------------------- picture 3: a traversal
def traversal_data(prep, result, t: np.ndarray, a: np.ndarray) -> dict:
    """The code of the 2-code network that follows the hidden condition ``t`` (the band position), decoded from -3 to +3 SD with the other code at its mean."""
    from fluoviz_addon_autoencoders.core import tools
    codes = result.samples["codes"]
    corr_t = [abs(float(np.corrcoef(codes[:, j], t)[0, 1])) for j in range(codes.shape[1])]
    corr_a = [abs(float(np.corrcoef(codes[:, j], a)[0, 1])) for j in range(codes.shape[1])]
    j = int(np.argmax(corr_t))
    tr = tools.traverse(result.spec, result.arrays, codes, dims=[j], values=(-3, -2, -1, 0, 1, 2, 3), frame=prep.frame)
    F = tr["functions"]["fluorescence"][0]                         # (7, m)
    D = tr["difference"]["fluorescence"][0]
    G = grid_of(prep, F)
    GD = grid_of(prep, D)
    top = float(np.nanmax(G))
    dmax = float(np.nanmax(np.abs(GD)))
    return {"code": j + 1, "values": [-3, -2, -1, 0, 1, 2, 3], "images": [rows(g / top) for g in G], "differences": [rows(g / dmax) for g in GD], "difference_scale": r(dmax / top, 3),
            "corr_with_band_position": r(corr_t, 3), "corr_with_amount": r(corr_a, 3), "sd": r(tr["sd"][j], 3)}


# ---------------------------------------------------------------------------------------------------------------- pictures 4 and 6: smoothify
def smoothify_data(prep) -> dict:
    """A noisy set of EEMs with a network that has more freedom than it needs: the held-out sample whose rebuilding at smoothify 0 is the roughest, as a cut along emission, and the
    scan (the L-curve) of the add-on."""
    from fluoviz_addon_autoencoders.core import diagnostics as D
    cfg = config(latent_dim=4, n_restarts=1)
    scan = D.smoothify_scan(cfg, prep, n_restarts=1)
    suggested = round(float(scan["suggested"]), 4)
    return {"scan": {"values": r(scan["values"], 4), "error": r(scan["error"], 5), "error_se": r(scan["error_se"], 5), "roughness": r(scan["roughness"], 4), "roughness_input": r(scan["roughness_input"], 4),
                     "suggested": suggested, "words": scan["words"]}, "n": int(prep.n), "n_inputs": int(prep.P), "noise": 0.08}


def cut_data(prep, s_big: float) -> dict:
    """The cut: smoothify 0 against ``s_big`` on the same held-out sample."""
    from fluoviz_addon_autoencoders.core import diagnostics as D
    from fluoviz_addon_autoencoders.core import forward as fw
    from fluoviz_addon_autoencoders.core import train
    td = prep.train_data()
    res = {}
    for key, s in (("off", 0.0), ("on", float(s_big))):
        res[key] = train.fit(config(latent_dim=4, n_restarts=1, smoothify=s), prep)
    held = np.flatnonzero(res["off"].samples["heldout"])
    G = {k: rebuilt(prep, v) for k, v in res.items()}
    smooth = grid_of(prep, prep.smoothed()["fluorescence"])
    meas = grid_of(prep, prep.observed()["fluorescence"])
    # the held-out sample and the excitation wavelength where smoothify 0 is roughest relative to the smoothed measured curve
    best = None
    for k in held:
        for j in range(len(EX)):
            m = np.isfinite(smooth[k][:, j])
            if m.sum() < 14 or np.nanmax(smooth[k][:, j]) < 0.35 * np.nanmax(smooth[k]):
                continue
            d2 = lambda v: float(np.sqrt(np.mean(np.diff(v[m], 2) ** 2)))
            score = d2(G["off"][k][:, j]) / max(d2(smooth[k][:, j]), 1e-9)
            if best is None or score > best[0]:
                best = (score, int(k), int(j))
    _, k, j = best
    rough = {key: D.roughness(td, fw.reconstruct(v.spec, v.arrays, td.X, td.M)[0])["per_sample"]["fluorescence"] for key, v in res.items()}
    rin = D.roughness(td, td.X)["per_sample"]["fluorescence"]
    out = {"sample": prep.ids[k], "ex_nm": float(EX[j]), "em": [float(v) for v in EM], "measured": r(meas[k][:, j], 4), "smoothed": r(smooth[k][:, j], 4), "off": r(G["off"][k][:, j], 4), "on": r(G["on"][k][:, j], 4),
           "smoothify_on": round(float(s_big), 4), "roughness": {"measured_smoothed": r(rin[k], 3), "off": r(rough["off"][k], 3), "on": r(rough["on"][k], 3)},
           "roughness_heldout_mean": {"measured_smoothed": r(rin[held].mean(), 3), "off": r(rough["off"][held].mean(), 3), "on": r(rough["on"][held].mean(), 3)},
           "evr_val": {"off": r(res["off"].validation["evr"], 4), "on": r(res["on"].validation["evr"], 4)}}
    return out


# ---------------------------------------------------------------------------------------------------------------- picture 5: a collapsed code
def collapse_data(prep_bend) -> dict:
    """A variational autoencoder with six codes on ``bending`` (two hidden factors): the KL of every code and the codes the page calls inactive."""
    from fluoviz_addon_autoencoders.core import diagnostics as D
    from fluoviz_addon_autoencoders.core import forward as fw
    from fluoviz_addon_autoencoders.core import train
    cfg = config(family="vae", latent_dim=6, n_restarts=2)
    res = train.fit(cfg, prep_bend)
    mu, lv = fw.encode(res.spec, res.arrays, prep_bend.X, prep_bend.M)
    h = D.latent_health(res.spec, res.arrays, prep_bend.X, prep_bend.M, mu, lv)
    return {"kl": r(h["kl_per_dim"], 4), "inactive": [bool(v) for v in h["inactive"]], "n_active": int(h["n_active"]), "limit": 0.05, "effect_share": r(h["effect_share"], 4),
            "evr_val": r(res.validation["evr"], 4), "words": h["words"], "n": int(prep_bend.n)}


# ---------------------------------------------------------------------------------------------------------------- picture 7: loss curves
def loss_data(prep) -> dict:
    """Two real trainings of the same 36 noisy EEMs on the pixels (the case where memorising is easiest): the default small network with weight decay and early stopping, and a large one without weight decay that is not stopped."""
    from fluoviz_addon_autoencoders.core import train
    out = {}
    for key, kw in (("healthy", dict(hidden="small", weight_decay=1e-3, epochs=400, patience=60)), ("overfit", dict(hidden="large", weight_decay=0.0, epochs=400, patience=400))):
        res = train.fit(config(latent_dim=2, n_restarts=1, **kw), prep)
        h = res.history
        keep = list(range(0, len(h), max(1, len(h) // 160)))
        if keep[-1] != len(h) - 1:
            keep.append(len(h) - 1)
        out[key] = {"epoch": [int(h[i]["epoch"]) + 1 for i in keep], "train": [r(h[i]["train"], 5) for i in keep], "val": [r(h[i]["val"], 5) for i in keep], "best_epoch": int(res.best_epoch) + 1,
                    "epochs_run": int(res.epochs_run), "evr_val": r(res.validation["evr"], 4), "evr_train": r(res.validation["evr_train"], 4), "n_parameters": int(res.n_parameters),
                    "n_train": int(res.validation["n_train"]), "n_val": int(res.validation["n_val"])}
    return out


# ---------------------------------------------------------------------------------------------------------------- pictures 8 and 9: the nonlinear effects of the codes
SHAPES = [("A straight line", lambda t: 0.4 + 0.15 * t), ("Bends upwards", lambda t: ((t + 3.0) / 6.0) ** 2), ("Bends downwards", lambda t: 1.0 - (1.0 - (t + 3.0) / 6.0) ** 2),
          ("Levels off", lambda t: 1.0 - np.exp(-4.5 * (t + 3.0) / 6.0)), ("An S", lambda t: np.tanh(1.1 * t)), ("A peak", lambda t: np.exp(-0.5 * ((t - 0.7) / 1.1) ** 2)),
          ("A valley", lambda t: ((t + 0.8) / 2.2) ** 2), ("A wave", lambda t: np.sin(1.9 * t + 0.4))]


def shapes_data() -> dict:
    """Six response curves of known shape, each put into words by ``core.curves.describe_curve`` (the numbers the page and the guide quote): what a straight line explains of each."""
    from fluoviz_addon_autoencoders.core.curves import describe_curve, line_fit, positions
    t = positions(3, 41)
    out = []
    for name, fn in SHAPES:
        y = np.asarray(fn(t), float)
        y = (y - y.min()) / (y.max() - y.min())
        d = describe_curve(t, y)
        fit = line_fit(t, y)
        out.append({"name": name, "y": r(y, 4), "line": r(fit["intercept"] + fit["slope"] * t, 4), "kind": d["kind"], "label": d["label"], "r2": r(d["r2_linear"], 4), "max_dev": r(d["max_deviation"], 4)})
    return {"t": r(t, 3), "curves": out}


def nonlinear_data(prep, result) -> dict:
    """The 2-code network of ``bending`` seen with the tools of the tab Nonlinear effects: for each code the response of the pixel it moves most with the samples around it, its words and
    its straight-line numbers, the map of where it bends (``NL``) and its nonlinear share; and how much the two codes interact."""
    from fluoviz_addon_autoencoders.core.curves import line_fit
    from fluoviz_addon_autoencoders.core.features import compile_features
    from fluoviz_addon_autoencoders.core.nonlinear import Decoder, bend_maps, pair_effects, response_curves
    codes = np.asarray(result.samples["codes"], float)
    dec = Decoder(result.spec, result.arrays, prep.frame, codes)
    feats = compile_features([{"kind": "moved"}, {"kind": "total"}], prep.frame)
    smooth = prep.smoothed()
    rc = response_curves(dec, feats, 3.0, 41, sample_functions=smooth, sample_coords=prep.X)
    bm = bend_maps(dec, 3.0, 41)
    t = rc["t"]
    out = {"t": r(t, 3), "n": int(prep.n), "n_codes": int(dec.d), "codes": [], "pixels": int(prep.block("fluorescence").domain.m)}
    for j in range(dec.d):
        d = rc["description"]["f1"][j]
        fit = line_fit(t, rc["curves"]["f1"][j])
        top = float(np.max(np.abs(rc["curves"]["f1"][j]))) or 1.0
        nl = bm["blocks"]["fluorescence"]["nl"][j]
        out["codes"].append({"name": f"z{j + 1}", "sd": r(dec.sd[j], 3), "curve": r(rc["curves"]["f1"][j], 4), "line": r(fit["intercept"] + fit["slope"] * t, 4), "kind": d["kind"], "label": d["label"], "r2": r(d["r2_linear"], 4),
                             "max_dev": r(d["max_deviation"], 4), "pixel": {k: float(v) for k, v in rc["pixel"]["f1"][j].items()}, "dots_x": r(rc["overlay"]["x"][:, j], 3), "dots_y": r(rc["overlay"]["y"]["f1"][j], 4),
                             "total_label": rc["description"]["f2"][j]["label"], "total_r2": r(rc["description"]["f2"][j]["r2_linear"], 4), "share": r(bm["share"][j], 4), "effect_share": r(bm["effect_share"][j], 4),
                             "nl": rows(grid_of(prep, nl)), "n_effect": int(bm["blocks"]["fluorescence"]["n_effect"][j]), "top": r(top, 4)})
    out["codes"].sort(key=lambda c: -(c["share"] or 0.0))                                           # the code that bends most first: the picture reads left to right from the bent one to the straight one
    pe = pair_effects(dec, 0, 1, 5, 2.0)
    out["pair"] = {"index": r(pe["index"], 4), "span": 2.0, "grid": 5}
    return out


# ---------------------------------------------------------------------------------------------------------------- everything
def figure_data(parts: list[str] | None = None) -> dict:
    """The numbers of all the pictures (``parts``: a list of keys to compute only some of them)."""
    want = set(parts or ["one", "compare", "traversal", "smoothify", "collapse", "loss", "shapes", "nonlinear"])
    from fluoviz_addon_autoencoders.core import train
    out: dict = {}
    Y_lin = eems_linear()
    Y_bend, t, a = eems_bending()
    prep_lin, prep_bend = prepare(Y_lin), prepare(Y_bend)
    if "compare" in want:
        out["compare"] = compare_data(prep_lin, prep_bend)
    if "one" in want:
        res = train.fit(config(latent_dim=3), prep_lin)
        out["one"] = one_data(prep_lin, res)
    if "traversal" in want or "nonlinear" in want:
        res = train.fit(config(latent_dim=2), prep_bend)
        if "traversal" in want:
            out["traversal"] = traversal_data(prep_bend, res, t, a)
        if "nonlinear" in want:
            out["nonlinear"] = nonlinear_data(prep_bend, res)
    if "shapes" in want:
        out["shapes"] = shapes_data()
    if "smoothify" in want:
        prep_noisy = prepare(eems_noisy(), {"fluorescence": {"basis": {"kind": "gp", "var_fraction": 0.9999}, "smoothing": {"method": "ml"}}})
        sm = smoothify_data(prep_noisy)
        sm["cut"] = cut_data(prep_noisy, sm["scan"]["suggested"])
        out["smoothify"] = sm
    if "collapse" in want:
        out["collapse"] = collapse_data(prep_bend)
    if "loss" in want:
        prep36 = prepare(eems_bending(36, 5, 0.05)[0], {"fluorescence": {"basis": {"kind": "pixels"}}})
        out["loss"] = loss_data(prep36)
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(figure_data(sys.argv[1:] or None), separators=(",", ":"))[:3000])
