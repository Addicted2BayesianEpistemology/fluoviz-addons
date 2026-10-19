"""Which nonlinear trend does each code fit?  Four tools that answer it (pure numpy: the FluoViz server never imports PyTorch).

The *position of a code* is the one of the traversals: ``t`` standard deviations (SD) of that code over the training samples from its mean, the other codes at their mean
(``tools.code_stats``).  Everything is worked out in the units of the data after the pre-treatment, and in the metric of the training loss (the L2 metric of the functions with
the quadrature weights ``domain.w`` of every block and the weights of the blocks: in the network coordinates it is ``arrays["out.weights"]``).

* :func:`response_curves`  a feature of the decoded function (the intensity at a pixel, the overall intensity ...) against the position of each code, with the training samples
  around it, curves of single samples (the other codes at THEIR values: do the codes act independently?), and the words (``curves.describe_curve``);
* :func:`bend_maps`        at every pixel a straight line is fitted against the position, with the weights of the normal density; ``NL = 1 - R^2`` of that line shows where the
  response bends, and ``nonlinear_share`` is the share of the whole effect of the code (in the metric of the loss) that no straight line describes;
* :func:`local_effects`    the Jacobian of the decoder (analytic) at the codes of samples: the loading of a code is different from sample to sample when the network bends;
* :func:`pair_effects`     two codes together on a grid: how much of the picture is NOT the sum of the two separate effects (the interaction image and index).
"""
from __future__ import annotations

import numpy as np

from . import forward as fw
from .curves import describe_curve, line_fit, normal_weights, positions
from .features import Feature, moved_index, pixel_where
from .spec import AutoencoderError, ModelSpec
from .tools import code_stats, jacobian_net

MIN_EFFECT = 0.02                # a pixel whose range along the traversal is below this share of the largest range of the code has "no effect"
DEAD = 1e-10                     # a code whose effect (in the metric of the loss) is below this share of the largest effect among the codes has no effect at all
LOADINGS = (0.1, 0.5, 0.9)       # the samples whose local loading is shown: at these quantiles of the code
FLOATS_JACOBIAN = 4_000_000      # a batch of Jacobians (samples x coordinates x codes) never holds more numbers than this (32 MB)
FLOATS_MAPPING = 4_000_000       # nor a batch of loadings mapped back to the pixels (samples x pixels)
MAX_ICE = 12


class Decoder:
    """A trained network (``spec``, ``arrays``) with the description of how its coordinates become functions (``frame``: ``core.data.AEFrame``) and the training codes that give the
    centre (``mean``) and the unit (``sd``, standard deviation) of every code (``mean`` / ``sd`` given: the ones stored in a result).  ``from_model`` takes the ``AEModel`` of the server."""

    def __init__(self, spec: ModelSpec, arrays: dict, frame, codes_ref, mean=None, sd=None):
        self.spec = spec
        self.net = fw.Network(spec, arrays)
        self.frame = frame
        self.d = int(spec.latent_dim)
        self.codes = np.asarray(codes_ref, float)
        if self.codes.ndim != 2 or self.codes.shape[1] != self.d:
            raise AutoencoderError(f"The training codes must be a table with {self.d} columns.")
        self.mean, self.sd = code_stats(self.codes)
        if mean is not None and sd is not None:                          # the centre and the unit the result stored (the traversals use them)
            self.mean, self.sd = np.asarray(mean, float), np.maximum(np.asarray(sd, float), 1e-12)
        self.names = frame.names()
        self.slices = frame.slices()
        self.omega = np.asarray(self.net.a["out.weights"], float)
        self.P = int(spec.input_dim)

    @classmethod
    def from_model(cls, model, codes_ref, mean=None, sd=None) -> "Decoder":
        return cls(model.spec, model.arrays, model.frame, codes_ref, mean, sd)

    def block(self, name: str):
        return self.frame.block(name)

    def along(self, j: int, t) -> np.ndarray:
        """Codes ``(len(t), d)``: every code at its mean except code ``j`` (0-based) at ``mean_j + t * sd_j``."""
        t = np.asarray(t, float)
        Z = np.tile(self.mean, (t.size, 1))
        Z[:, j] = self.mean[j] + t * self.sd[j]
        return Z

    def decode(self, Z) -> np.ndarray:
        return self.net.decode(Z)

    def functions(self, X) -> dict:
        return self.frame.to_functions(X)

    def decode_functions(self, Z) -> tuple[np.ndarray, dict]:
        X = self.decode(Z)
        return X, self.functions(X)

    def jacobian(self, Z) -> np.ndarray:
        return jacobian_net(self.net, Z)

    def position(self, j: int) -> np.ndarray:
        """The codes of the training samples in SD from the mean: ``(n,)``."""
        return (self.codes[:, j] - self.mean[j]) / self.sd[j]


def _chunks(n: int, size: int):
    size = max(1, int(size))
    for a in range(0, n, size):
        yield a, min(n, a + size)


# ----------------------------------------------------------------------------- 1. response curves
def ice_samples(position_sd: np.ndarray, n: int) -> np.ndarray:
    """Up to ``n`` samples spread over the distribution of a code: the samples at the ranks ``(k + 0.5) / n`` of the sorted codes, ``k = 0 ... n - 1`` (each sample once, since
    the ranks are at least one apart).  ``position_sd``: the codes of all the samples ``(N,)``.  Returned in the order of the code."""
    x = np.asarray(position_sd, float)
    N = x.size
    n = int(min(n, N))
    if n <= 0:
        return np.zeros(0, int)
    order = np.argsort(x, kind="stable")
    return order[np.minimum(((np.arange(n) + 0.5) * N / n).astype(int), N - 1)]


def response_curves(dec: Decoder, features: list[Feature], span: float = 3.0, n_points: int = 41, ice: dict | None = None, sample_functions: dict | None = None,
                    sample_coords: np.ndarray | None = None) -> dict:
    """A feature against the position of every code.

    ``features``: compiled features (``features.compile_features``).  ``ice``: ``None`` (no curves of single samples), ``{"codes": [0-based codes], "n": 12}`` (``n`` samples chosen
    at the quantiles of each code) or ``{"codes": [...], "samples": [indices]}`` (the same samples for every code).  ``sample_functions`` ``{block: (n, m)}`` and ``sample_coords``
    ``(n, P)``: the functions the network was trained to rebuild (smoothed samples; the measured values for a pixel network) and their network coordinates, for the training
    samples drawn around the curves; ``None`` = not drawn.

    Returns ``{"t", "weights", "features": [info], "curves": {fid: (d, n_points)}, "pixel": {fid: [where per code]}, "description": {fid: [describe_curve per code]},
    "ice": {code: {"samples": (k,), "x": (k,), "y": {fid: (k, n_points)}}} | None, "overlay": {"x": (n, d), "y": {fid: (d, n) | (n,)}} | None}``; ``pixel`` only for
    the features that are resolved per code."""
    t = positions(span, n_points)
    d = dec.d
    curves = {f.id: np.zeros((d, t.size)) for f in features}
    index: dict[str, list] = {f.id: [None] * d for f in features if f.per_code}
    for j in range(d):
        X, F = dec.decode_functions(dec.along(j, t))
        for f in features:
            idx = None
            if f.per_code:
                idx = moved_index(F[f.block])
                index[f.id][j] = idx
            curves[f.id][j] = f.evaluate(F, X, idx)
    ref = {f.id: float(np.ptp(curves[f.id], axis=1).max()) for f in features}
    w = normal_weights(t)
    description = {f.id: [describe_curve(t, curves[f.id][j], w, ref[f.id]) for j in range(d)] for f in features}
    pixel = {f.id: [pixel_where(dec.block(f.block), ix) for ix in index[f.id]] for f in features if f.per_code}
    out: dict = {"t": t, "weights": w, "features": [f.info() for f in features], "curves": curves, "description": description, "pixel": pixel, "ice": None, "overlay": None}
    if ice and ice.get("codes"):
        out["ice"] = {}
        for j in ice["codes"]:
            idx_s = np.asarray(ice["samples"], int) if "samples" in ice else ice_samples(dec.position(j), int(ice.get("n", MAX_ICE)))
            ys = {f.id: np.zeros((idx_s.size, t.size)) for f in features}
            for a, s in enumerate(idx_s):
                Z = np.tile(dec.codes[s], (t.size, 1))
                Z[:, j] = dec.mean[j] + t * dec.sd[j]
                X, F = dec.decode_functions(Z)
                for f in features:
                    ys[f.id][a] = f.evaluate(F, X, index[f.id][j] if f.per_code else None)
            out["ice"][int(j)] = {"samples": idx_s, "x": dec.position(j)[idx_s] if idx_s.size else np.zeros(0), "y": ys}
    if sample_functions is not None:
        n = dec.codes.shape[0]
        over = {}
        for f in features:
            if f.per_code:
                over[f.id] = np.stack([f.evaluate(sample_functions, sample_coords, index[f.id][j]) for j in range(d)])
            else:
                over[f.id] = f.evaluate(sample_functions, sample_coords)
        out["overlay"] = {"x": (dec.codes - dec.mean) / dec.sd, "y": over, "n": n}
    return out


# ----------------------------------------------------------------------------- 2. where the response bends
def bend_maps(dec: Decoder, span: float = 3.0, n_points: int = 41, min_effect: float = MIN_EFFECT) -> dict:
    """At every pixel of every block, along the traversal of each code, a straight line is fitted against the position (the weights are the normal density at the positions) and
    ``NL = SSE / SST = 1 - R^2`` of that line is worked out: 0 = a straight line, 1 = nothing of the response is linear.  Pixels whose range along the traversal is below
    ``min_effect`` of the largest range of the code (in that block) have no effect: their ``NL`` is NaN.

    ``nonlinear_share`` of a code = ``sum_j w_j SSE_j / sum_j w_j SST_j`` over the pixels with the quadrature weights ``domain.w`` (per block: ``share_by_block``) and, over all the
    blocks in the metric of the loss, ``share`` (computed from the network coordinates with ``out.weights``: for one block the same number).  0 = the code acts like a PCA
    loading, 1 = its effect is entirely nonlinear.  ``effect`` = ``sum_j w_j SST_j`` in the metric of the loss: how much the code moves the picture (a share is only worth reading
    next to it).

    A code that the decoder ignores (``dead``) has NaN for its shares and its NL.

    Returns ``{"t", "share" (d,), "effect" (d,), "effect_share" (d,), "dead" (d,), "blocks": {name: {"nl" (d, m), "amplitude" (d, m), "share" (d,), "max_amplitude" (d,), "n_effect" (d,)}}}``."""
    t = positions(span, n_points)
    w = normal_weights(t)
    d = dec.d
    blocks = {nm: {"nl": np.full((d, dec.block(nm).domain.m), np.nan), "amplitude": np.zeros((d, dec.block(nm).domain.m)), "share": np.full(d, np.nan), "max_amplitude": np.zeros(d),
                   "n_effect": np.zeros(d, int)} for nm in dec.names}
    share, effect = np.full(d, np.nan), np.zeros(d)
    for j in range(d):
        X, F = dec.decode_functions(dec.along(j, t))
        fx = line_fit(t, X, w)
        sst_w = float(fx["sst"] @ dec.omega)
        effect[j] = sst_w
        share[j] = float(np.clip((fx["sse"] @ dec.omega) / sst_w, 0.0, 1.0)) if sst_w > 0 else np.nan
        for nm in dec.names:
            blk = dec.block(nm)
            fit = line_fit(t, F[nm], w)
            amp = F[nm].max(axis=0) - F[nm].min(axis=0)
            top = float(amp.max())
            sst = fit["sst"]
            with np.errstate(invalid="ignore", divide="ignore"):
                nl = np.where(sst > 0, fit["sse"] / np.where(sst > 0, sst, 1.0), np.nan)
            nl = np.clip(nl, 0.0, 1.0)
            quiet = ~(amp >= min_effect * top) if top > 0 else np.ones_like(amp, bool)
            nl[quiet] = np.nan
            b = blocks[nm]
            b["nl"][j], b["amplitude"][j], b["max_amplitude"][j], b["n_effect"][j] = nl, amp, top, int((~quiet).sum())
            den = float(fit["sst"] @ blk.domain.w)
            b["share"][j] = float(np.clip((fit["sse"] @ blk.domain.w) / den, 0.0, 1.0)) if den > 0 else np.nan
    dead = ~(effect > DEAD * float(effect.max())) if effect.max() > 0 else np.ones(d, bool)
    share[dead] = np.nan
    for b in blocks.values():                                                 # (a code the decoder ignores: what is left is rounding dust, not a result)
        b["nl"][dead], b["share"][dead], b["n_effect"][dead] = np.nan, np.nan, 0
    tot = float(effect[~dead].sum())
    return {"t": t, "share": share, "effect": effect, "effect_share": np.where(dead, 0.0, effect / tot) if tot > 0 else np.zeros(d), "blocks": blocks, "min_effect": float(min_effect),
            "dead": dead}


def default_pair(share: np.ndarray) -> tuple[int, int]:
    """The two codes with the largest nonlinear share (0-based, the lower number first); the codes that have no effect (NaN) come last."""
    s = np.where(np.isfinite(share), share, -1.0)
    top = np.argsort(-s, kind="stable")[:2]
    return int(min(top)), int(max(top))


# ----------------------------------------------------------------------------- 3. local effects
def local_effects(dec: Decoder, code: int, sample_index, floats: int = FLOATS_JACOBIAN) -> dict:
    """The Jacobian of the decoder with respect to code ``code`` (0-based; analytic), per 1 SD of the code, at the mean code and at the codes of chosen samples, and how it varies.

    * ``at_mean`` ``{block: (m,)}`` and ``samples`` ``{block: (k, m)}``: the loading as a function (a change of the picture per SD of the code; signed), with ``cosine`` ``(k,)``
      = the cosine similarity of each chosen sample's loading with the one at the mean code (the L2 metric of the loss);
    * ``variability`` ``{block: (m,)}``: the standard deviation over ALL the training samples of the loading at each pixel (0 = the loading is the same everywhere);
    * ``change`` ``{"mean" (d,), "max" (d,), "size_mean" (d,), "size_sd" (d,)}``: for every code, ``1 - cosine`` between the loading at a training sample and at the mean code,
      averaged (and the largest) over the samples: 0 = the loading has the same SHAPE everywhere, larger = its effect changes shape.  The cosine does not see a loading that only
      gets stronger or weaker (a code that saturates): ``size_mean`` and ``size_sd`` are the mean and the standard deviation over the samples of the length of the loading relative
      to the one at the mean code (1 = as strong; the metric of the loss).

    The training samples go through in batches (the Jacobian of a batch is at most ``floats`` numbers), so 500 samples, 32 codes and 17 000 pixels need little memory."""
    d, P = dec.d, dec.P
    j = int(code)
    om = dec.omega
    J0 = dec.jacobian(dec.mean[None])[0] * dec.sd[None, :]                         # (P, d)
    norm0 = np.sqrt(om @ (J0 * J0))                                                # (d,)
    idx = np.asarray(sample_index, int).ravel()
    mean_fun = {nm: dec.block(nm).mean_function() for nm in dec.names}

    def mapped(D: np.ndarray, nm: str, sl: slice) -> np.ndarray:
        """Coordinates ``(k, K)`` of a loading -> the loading as a function ``(k, m)`` (the map is affine: the mean function is taken off)."""
        return dec.block(nm).to_function(D[:, sl]) - mean_fun[nm][None, :]

    out: dict = {"code": j, "sd": float(dec.sd[j]), "norm_at_mean": float(norm0[j])}
    out["at_mean"] = {nm: mapped(J0[:, j][None], nm, sl)[0] for nm, sl in zip(dec.names, dec.slices)}
    nrm_j = norm0[j]
    if idx.size:
        Jc = dec.jacobian(dec.codes[idx]) * dec.sd[None, None, :]
        Lj = Jc[:, :, j]                                                           # (k, P)
        out["samples"] = {nm: mapped(Lj, nm, sl) for nm, sl in zip(dec.names, dec.slices)}
        nrm = np.sqrt((Lj * Lj) @ om)
        with np.errstate(invalid="ignore", divide="ignore"):
            out["cosine"] = np.where((nrm > 0) & (nrm_j > 0), (Lj * J0[:, j][None, :]) @ om / (nrm * nrm_j), np.nan)
    else:
        out["samples"] = {nm: np.zeros((0, dec.block(nm).domain.m)) for nm in dec.names}
        out["cosine"] = np.zeros(0)
    # all the training samples, in batches
    n = dec.codes.shape[0]
    run = {nm: {"n": 0, "mean": np.zeros(dec.block(nm).domain.m), "m2": np.zeros(dec.block(nm).domain.m)} for nm in dec.names}
    change_sum, change_max, change_n = np.zeros(d), np.full(d, -np.inf), np.zeros(d)
    size_sum, size_sq = np.zeros(d), np.zeros(d)
    for a, b in _chunks(n, floats // max(P * d, 1)):
        J = dec.jacobian(dec.codes[a:b]) * dec.sd[None, None, :]                  # (k, P, d)
        Jw = J * om[None, :, None]
        dots = np.einsum("npd,pd->nd", Jw, J0)                                    # <J_i, J_0> per code, in the metric of the loss
        nr = np.sqrt(np.einsum("npd,npd->nd", Jw, J))
        del Jw
        with np.errstate(invalid="ignore", divide="ignore"):
            cos = np.where((nr > 0) & (norm0[None, :] > 0), dots / (nr * norm0[None, :]), np.nan)
        loss = np.clip(1.0 - cos, 0.0, 2.0)                                       # (a cosine a rounding error above 1 is not a negative change)
        ok = np.isfinite(loss)
        dev = np.where(ok, nr / np.where(norm0[None, :] > 0, norm0[None, :], 1.0) - 1.0, 0.0)           # the strength of the loading at the sample relative to the mean code, minus 1
        size_sum += dev.sum(axis=0)
        size_sq += (dev * dev).sum(axis=0)
        change_sum += np.where(ok, loss, 0.0).sum(axis=0)
        change_n += ok.sum(axis=0)
        change_max = np.maximum(change_max, np.where(ok, loss, -np.inf).max(axis=0))
        D = J[:, :, j] - J0[:, j][None, :]                                         # (k, P): around the loading at the mean code (no cancellation for a straight network)
        for nm, sl in zip(dec.names, dec.slices):
            m = dec.block(nm).domain.m
            r = run[nm]
            for c, e in _chunks(D.shape[0], FLOATS_MAPPING // max(m, 1)):
                L = mapped(D[c:e], nm, sl)
                nb = L.shape[0]
                mb = L.mean(axis=0)
                m2b = ((L - mb) ** 2).sum(axis=0)
                tot = r["n"] + nb
                delta = mb - r["mean"]
                r["mean"] = r["mean"] + delta * nb / tot
                r["m2"] = r["m2"] + m2b + delta * delta * r["n"] * nb / tot
                r["n"] = tot
    out["variability"] = {nm: np.sqrt(r["m2"] / max(r["n"] - 1, 1)) for nm, r in run.items()}
    has = change_n > 0
    cn = np.where(has, change_n, 1.0)
    dmean = size_sum / cn                                                      # (around 1, the strength at the mean code: no cancellation when every loading is as strong)
    out["change"] = {"mean": np.where(has, change_sum / cn, np.nan), "max": np.where(has, change_max, np.nan), "size_mean": np.where(has, 1.0 + dmean, np.nan),
                     "size_sd": np.where(has, np.sqrt(np.maximum(size_sq / cn - dmean * dmean, 0.0)), np.nan)}
    return out


def loading_samples(dec: Decoder, code: int, quantiles=LOADINGS) -> np.ndarray:
    """The samples at the given quantiles of a code (0-based): the sample nearest to each quantile, once each."""
    x = dec.position(code)
    order = np.argsort(x, kind="stable")
    out: list[int] = []
    for q in quantiles:
        target = np.quantile(x, q)
        for i in np.argsort(np.abs(x - target), kind="stable"):
            if int(i) not in out:
                out.append(int(i))
                break
    return np.asarray(out, int)


# ----------------------------------------------------------------------------- 4. two codes together
def pair_effects(dec: Decoder, i: int, j: int, grid: int = 5, span: float = 2.0, feature: Feature | None = None, heat_points: int = 31, sample_functions: dict | None = None,
                 sample_coords: np.ndarray | None = None) -> dict:
    """Codes ``i`` and ``j`` (0-based, different) on a ``grid`` x ``grid`` square (3, 5 or 7 positions from ``-span`` to ``+span`` SD; the others at their mean).

    ``F(a, b)`` is the decoded function with code ``i`` at position ``u_a`` and code ``j`` at ``u_b``; ``c`` is the position 0 (the mean).  The *interaction* is
    ``F(a, b) - F(a, c) - F(c, b) + F(c, c)``: what is left when the effect of each code alone (the other at its mean) is taken off.  Returns

    * ``u`` the positions, ``F`` ``{block: (grid, grid, m)}`` (index ``[position of j, position of i]``), ``interaction`` ``{block: (m,)}`` = the largest absolute interaction over
      the grid at each pixel;
    * ``index`` = ``sum over the grid and pixels of interaction^2 / sum of (F - mean of F over the grid)^2`` in the metric of the loss (0 = the two codes act independently,
      additively; NaN when the picture does not change at all), with the two sums (``interaction_ss``, ``variation_ss``) and ``index_by_block`` (the L2 metric of that block);
    * with a ``feature``: ``heat`` ``(heat_points, heat_points)`` the feature over the plane (rows: code ``j``), positions ``v``; for ``moved`` the pixel that varies most over the
      gallery is used (``pixel``);
    * ``dots``: the training samples in SD ``(n, 2)`` and, given ``sample_functions`` / ``sample_coords``, the feature of each measured sample."""
    if i == j:
        raise AutoencoderError("Choose two different codes.")
    if grid not in (3, 5, 7):
        raise AutoencoderError("The grid must be 3, 5 or 7 positions per code.")
    u = positions(span, grid)
    c = grid // 2
    Z = np.tile(dec.mean, (grid * grid, 1))
    bb, aa = np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij")
    Z[:, i] = dec.mean[i] + u[aa.ravel()] * dec.sd[i]
    Z[:, j] = dec.mean[j] + u[bb.ravel()] * dec.sd[j]
    X, F0 = dec.decode_functions(Z)
    Xg = X.reshape(grid, grid, -1)
    inter_x = Xg - Xg[:, c:c + 1, :] - Xg[c:c + 1, :, :] + Xg[c, c][None, None, :]
    var_x = Xg - Xg.mean(axis=(0, 1), keepdims=True)
    num, den = float(((inter_x ** 2) @ dec.omega).sum()), float(((var_x ** 2) @ dec.omega).sum())
    F = {nm: F0[nm].reshape(grid, grid, -1) for nm in dec.names}
    inter, by_block = {}, {}
    for nm in dec.names:
        G = F[nm]
        I = G - G[:, c:c + 1, :] - G[c:c + 1, :, :] + G[c, c][None, None, :]
        top = np.abs(I).max(axis=(0, 1))
        inter[nm] = np.where(top < 1e-10 * max(float(np.abs(G).max()), 1e-300), 0.0, top)                 # (rounding dust is not an interaction)
        wb = dec.block(nm).domain.w
        nb, db = float(((I ** 2) @ wb).sum()), float((((G - G.mean(axis=(0, 1), keepdims=True)) ** 2) @ wb).sum())
        by_block[nm] = nb / db if db > 0 else float("nan")
    index = float("nan") if not den > 0 else (0.0 if num / den < 1e-14 else num / den)                     # (additive up to rounding: exactly 0)
    out: dict = {"u": u, "F": F, "interaction": inter, "interaction_ss": num, "variation_ss": den, "index": index, "index_by_block": by_block}
    n = dec.codes.shape[0]
    pos = (dec.codes - dec.mean) / dec.sd
    out["dots"] = {"x": pos[:, i], "y": pos[:, j], "value": None}
    if feature is not None:
        idx = moved_index(F0[feature.block]) if feature.per_code else None
        if idx is not None:
            out["pixel"] = pixel_where(dec.block(feature.block), idx)
        hp = int(heat_points)
        v = positions(span, hp)
        heat = np.zeros((hp, hp))
        bj, ai = np.meshgrid(np.arange(hp), np.arange(hp), indexing="ij")
        ZZ = np.tile(dec.mean, (hp * hp, 1))
        ZZ[:, i] = dec.mean[i] + v[ai.ravel()] * dec.sd[i]
        ZZ[:, j] = dec.mean[j] + v[bj.ravel()] * dec.sd[j]
        flat = np.zeros(hp * hp)
        rows = max(1, FLOATS_MAPPING // max(dec.block(feature.block).domain.m, 1))
        for a, b in _chunks(hp * hp, rows):
            Xc, Fc = dec.decode_functions(ZZ[a:b])
            flat[a:b] = feature.evaluate(Fc, Xc, idx)
        out["v"], out["heat"] = v, flat.reshape(hp, hp)
        if sample_functions is not None:
            out["dots"]["value"] = feature.evaluate(sample_functions, sample_coords, idx)
    return out
