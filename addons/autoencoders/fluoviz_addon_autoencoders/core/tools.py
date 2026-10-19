"""The exploration tools (SPEC section 5): traversals, sensitivity, interpolation, generation, codes against metadata, clusters.  **Pure numpy** (the server calls them; no PyTorch).

Every tool works in the network coordinates with ``spec`` / ``arrays`` (``core.forward``) and, when a ``frame`` (``core.data.AEFrame``) is given, also returns the functions at the
observed pixels of each variable (``{name: (..., m)}``, units after the pre-treatment; embed them on the grid with ``frame.block(name).domain.embed``).  The *unit of a code* is the
standard deviation of the training codes (``code_stats``), so that codes of different scale are comparable.
"""
from __future__ import annotations

import numpy as np

from . import forward as fw
from .spec import AutoencoderError, ModelSpec


def code_stats(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation of the codes ``(n, d)`` of the training samples (the SD is floored so that a constant code does not divide by zero)."""
    codes = np.asarray(codes, float)
    sd = codes.std(axis=0, ddof=1) if codes.shape[0] > 1 else np.ones(codes.shape[1])
    return codes.mean(axis=0), np.maximum(sd, 1e-12)


def _functions(frame, X: np.ndarray):
    if frame is None:
        return None
    lead = X.shape[:-1]
    F = frame.to_functions(X.reshape(-1, X.shape[-1]))
    return {k: v.reshape(*lead, v.shape[-1]) for k, v in F.items()}


# ----------------------------------------------------------------------------- latent traversals
def traverse(spec: ModelSpec, arrays: dict, codes_ref: np.ndarray, dims=None, values=(-3, -2, -1, 0, 1, 2, 3), frame=None) -> dict:
    """Decode along each code: code ``j`` set to ``mean_j + v * sd_j`` for every ``v`` in ``values`` (SD of the training codes), the others at their mean.

    ``dims``: 0-based codes (default all).  Returns ``{"values", "dims", "mean_code", "sd", "z": (D, k, d), "X": (D, k, P), "X_mean": (P,), "functions": {name: (D, k, m)} | None,
    "difference": {name: (D, k, m)} | None, "mean": {name: (m,)} | None}``; ``difference`` is the function minus the decoding of the mean code."""
    mean, sd = code_stats(codes_ref)
    d = spec.latent_dim
    dims = list(range(d)) if dims is None else [int(j) for j in dims]
    if any(not 0 <= j < d for j in dims):
        raise AutoencoderError(f"The network has {d} codes: choose codes between 1 and {d}.")
    vals = np.asarray(values, float)
    net = fw.Network(spec, arrays)
    Z = np.tile(mean, (len(dims), len(vals), 1))
    for a, j in enumerate(dims):
        Z[a, :, j] = mean[j] + vals * sd[j]
    X = net.decode(Z.reshape(-1, d)).reshape(len(dims), len(vals), -1)
    X0 = net.decode(mean[None])[0]
    F, F0 = _functions(frame, X), _functions(frame, X0[None])
    out = {"values": vals, "dims": dims, "mean_code": mean, "sd": sd, "z": Z, "X": X, "X_mean": X0, "functions": F, "difference": None, "mean": None if F0 is None else {k: v[0] for k, v in F0.items()}}
    if F is not None:
        out["difference"] = {k: F[k] - F0[k][0][None, None] for k in F}
    return out


# ----------------------------------------------------------------------------- Jacobian and sensitivity
def jacobian(spec: ModelSpec, arrays: dict, Z: np.ndarray) -> np.ndarray:
    """The Jacobian of the decoder, analytic through the layers: ``(n, P, d)`` = d(reconstruction in the network coordinates) / d(code) at every row of ``Z`` ``(n, d)``."""
    return jacobian_net(fw.Network(spec, arrays), Z)


def jacobian_net(net: "fw.Network", Z: np.ndarray) -> np.ndarray:
    """:func:`jacobian` for a network that is already built (``core.nonlinear`` calls it many times)."""
    spec = net.spec
    Z = np.atleast_2d(np.asarray(Z, float))
    _, pre = net.decode_with_trace(Z)
    dact = net.dact_dec
    J = None                                                            # (n, h, d)
    for i, (W, _) in enumerate(net.dec):
        g = dact(pre[i])                                                # (n, h)
        J = g[:, :, None] * (W[None] if J is None else W @ J)
    Wo, bo = net.out
    if J is None:
        Jo = np.broadcast_to(Wo[None], (Z.shape[0],) + Wo.shape).copy()
    else:
        Jo = Wo @ J                                                       # (P, h) @ (n, h, d) -> (n, P, d): one matrix product per sample
    if net.lin_dec is not None:
        Jo = Jo + net.lin_dec[None]
    if spec.output_activation != "linear":
        h = Z if not net.dec else net.act_dec(pre[-1])
        o_pre = h @ Wo.T + bo + (0.0 if net.lin_dec is None else Z @ net.lin_dec.T)
        Jo = Jo * fw.activation_grad(spec.output_activation)(o_pre)[:, :, None]
    return Jo * net.scale[None, :, None]


def sensitivity(spec: ModelSpec, arrays: dict, codes_ref: np.ndarray, frame=None, batch: int = 32) -> dict:
    """What each code does to the output: the derivative of the output with respect to code ``j`` **per 1 SD of the code**, at the mean code (signed) and its average magnitude
    over the training samples (mean of ``|.|`` over the codes of the samples).  ``frame`` given: in functions (``{name: (d, m)}``), else in network coordinates ``(d, P)``.

    Returns ``{"names", "mean_code", "sd", "at_mean", "magnitude", "share" (d,), "effect" (d,)}``: ``share`` = share of the total squared effect (in the metric of the loss) of every
    code on the output (a code with a share near 0 is not used)."""
    codes_ref = np.asarray(codes_ref, float)
    mean, sd = code_stats(codes_ref)
    d = spec.latent_dim
    w = np.asarray(arrays["out.weights"], float)
    J0 = jacobian(spec, arrays, mean[None])[0] * sd[None, :]                    # (P, d)
    mag = np.zeros((spec.input_dim, d))
    eff = np.zeros(d)
    func_mag = None
    if frame is not None:
        func_mag = {nm: np.zeros((d, frame.block(nm).domain.m)) for nm in frame.names()}
    n = codes_ref.shape[0]
    for a in range(0, n, batch):
        J = jacobian(spec, arrays, codes_ref[a:a + batch]) * sd[None, None, :]       # (b, P, d)
        mag += np.abs(J).sum(axis=0)
        eff += (w[None, :, None] * J ** 2).sum(axis=(0, 1))
        if frame is not None:
            for nm, sl in zip(frame.names(), frame.slices()):
                blk = frame.block(nm)
                for j in range(d):
                    Fj = blk.to_function(J[:, sl, j]) - blk.mean_function()[None, :]
                    func_mag[nm][j] += np.abs(Fj).sum(axis=0)
    mag /= n
    eff /= n
    share = eff / max(float(eff.sum()), 1e-300)
    out = {"names": [f"z{j + 1}" for j in range(d)], "mean_code": mean, "sd": sd, "share": share, "effect": eff}
    if frame is None:
        out.update({"at_mean": J0.T, "magnitude": mag.T})
    else:
        at = {}
        for nm, sl in zip(frame.names(), frame.slices()):
            blk = frame.block(nm)
            at[nm] = np.stack([blk.to_function(J0[sl, j][None])[0] - blk.mean_function() for j in range(d)])
        out.update({"at_mean": at, "magnitude": {nm: v / n for nm, v in func_mag.items()}})
    return out


# ----------------------------------------------------------------------------- interpolation
def interpolate(spec: ModelSpec, arrays: dict, za: np.ndarray, zb: np.ndarray, steps: int = 9, frame=None, Fa: dict | None = None, Fb: dict | None = None) -> dict:
    """Between the codes ``za`` and ``zb`` of two samples: ``steps`` codes ``(1-t) za + t zb`` (``t`` from 0 to 1), decoded; next to it the plain linear mixture ``(1-t) Fa + t Fb`` of
    the two functions when ``Fa`` / ``Fb`` (``{name: (m,)}``, e.g. the smoothed or measured functions of the two samples) are given.

    Returns ``{"t", "z": (steps, d), "X": (steps, P), "functions": {name: (steps, m)} | None, "linear": {name: (steps, m)} | None, "difference": latent minus linear | None}``."""
    if steps < 2:
        raise AutoencoderError("An interpolation needs at least 2 steps.")
    t = np.linspace(0.0, 1.0, int(steps))
    za, zb = np.asarray(za, float).ravel(), np.asarray(zb, float).ravel()
    Z = (1.0 - t)[:, None] * za[None] + t[:, None] * zb[None]
    X = fw.decode(spec, arrays, Z)
    F = _functions(frame, X)
    lin = None
    if Fa is not None and Fb is not None:
        lin = {k: (1.0 - t)[:, None] * np.asarray(Fa[k], float)[None] + t[:, None] * np.asarray(Fb[k], float)[None] for k in Fa}
    diff = None if F is None or lin is None else {k: F[k] - lin[k] for k in F}
    return {"t": t, "z": Z, "X": X, "functions": F, "linear": lin, "difference": diff}


# ----------------------------------------------------------------------------- generation (VAE)
def generate(spec: ModelSpec, arrays: dict, n: int, seed: int = 0, temperature: float = 1.0, codes_ref: np.ndarray | None = None, frame=None) -> dict:
    """Draw ``n`` new samples from the prior of a variational autoencoder: ``z ~ N(0, temperature^2 I)`` (a mixture prior first picks a cluster and then draws around its mean with
    the cluster's SD times the temperature), decoded.  ``codes_ref`` (the training codes) gives the nearest real sample of every draw, by distance in units of the code SD.

    Returns ``{"synthetic": True, "z", "X", "functions", "cluster" (mixture), "nearest" (index), "distance"}`` - always synthetic: made up by the network, not measured."""
    if not spec.is_vae():
        raise AutoencoderError("Only variational autoencoders can draw new samples (this network is a deterministic autoencoder). Train a variational autoencoder.")
    if n < 1:
        raise AutoencoderError("Ask for at least one sample.")
    if temperature < 0:
        raise AutoencoderError("The temperature must be 0 or more.")
    rng = np.random.default_rng(int(seed))
    d = spec.latent_dim
    cluster = None
    if spec.has_mixture():
        pi = np.asarray(arrays["prior.pi"], float)
        cluster = rng.choice(len(pi), size=int(n), p=pi / pi.sum())
        mu, lv = np.asarray(arrays["prior.mu"], float)[cluster], np.asarray(arrays["prior.logvar"], float)[cluster]
        Z = mu + float(temperature) * np.exp(0.5 * lv) * rng.standard_normal((int(n), d))
    else:
        Z = float(temperature) * rng.standard_normal((int(n), d))
    X = fw.decode(spec, arrays, Z)
    out = {"synthetic": True, "z": Z, "X": X, "functions": _functions(frame, X), "cluster": cluster, "nearest": None, "distance": None}
    if codes_ref is not None:
        mean, sd = code_stats(codes_ref)
        D = np.sqrt((((Z[:, None, :] - np.asarray(codes_ref, float)[None]) / sd) ** 2).sum(axis=2))
        out["nearest"], out["distance"] = D.argmin(axis=1), D.min(axis=1)
    return out


# ----------------------------------------------------------------------------- codes against other variables
def correlations(codes: np.ndarray, targets: dict) -> dict:
    """Pearson correlation of every code with every numeric variable (NaN pairs left out): ``{name: (d,)}`` (NaN where it cannot be computed)."""
    codes = np.asarray(codes, float)
    out = {}
    for nm, y in targets.items():
        y = np.asarray(y, float)
        ok = np.isfinite(y)
        r = np.full(codes.shape[1], np.nan)
        if ok.sum() >= 3 and np.nanstd(y[ok]) > 0:
            for j in range(codes.shape[1]):
                c = codes[ok, j]
                if np.isfinite(c).all() and c.std() > 0:
                    r[j] = float(np.corrcoef(c, y[ok])[0, 1])
        out[nm] = r
    return out


def cv_r2(F: np.ndarray, y: np.ndarray, folds: int = 5, seed: int = 0, ridge: float = 1e-3, degree: int = 1) -> float:
    """Cross-validated R^2 of predicting ``y`` from the features ``F`` ``(n, k)`` (ridge regression on standardised features, optionally with squares for ``degree`` 2); samples with
    a missing ``y`` are left out.  Predictive R^2 = ``1 - PRESS / SS_total`` (0 = no better than the mean of the training samples)."""
    F, y = np.asarray(F, float), np.asarray(y, float)
    ok = np.isfinite(y) & np.isfinite(F).all(axis=1)
    F, y = F[ok], y[ok]
    n = y.size
    if n < 6 or np.std(y) == 0:
        return float("nan")
    V = int(max(2, min(folds, n)))
    perm = np.random.default_rng(seed).permutation(n)
    pred = np.zeros(n)
    for f in range(V):
        te = perm[f::V]
        tr = np.setdiff1d(np.arange(n), te)
        mu, sd = F[tr].mean(axis=0), np.maximum(F[tr].std(axis=0), 1e-12)
        A = (F[tr] - mu) / sd
        B = (F[te] - mu) / sd
        if degree >= 2:
            A, B = np.hstack([A, A ** 2]), np.hstack([B, B ** 2])
        A1, B1 = np.hstack([np.ones((A.shape[0], 1)), A]), np.hstack([np.ones((B.shape[0], 1)), B])
        R = ridge * A1.shape[0] * np.eye(A1.shape[1])
        R[0, 0] = 0.0
        beta = np.linalg.solve(A1.T @ A1 + R, A1.T @ y[tr])
        pred[te] = B1 @ beta
    ss = float(((y - y.mean()) ** 2).sum())
    return float(1.0 - ((y - pred) ** 2).sum() / ss)


def relate(codes: np.ndarray, targets: dict, pca_scores: np.ndarray | None = None, folds: int = 5, seed: int = 0, degree: int = 1) -> dict:
    """Codes against numeric variables: for each variable the correlation with every code, the cross-validated R^2 of predicting it from the codes and (``pca_scores`` given: the
    same number of PCA / functional PCA scores) from the PCA scores.  Returns ``{name: {"correlation": (d,), "r2_codes", "r2_pca", "n"}}``."""
    cor = correlations(codes, targets)
    out = {}
    for nm, y in targets.items():
        y = np.asarray(y, float)
        out[nm] = {"correlation": cor[nm], "r2_codes": cv_r2(codes, y, folds, seed, degree=degree), "r2_pca": None if pca_scores is None else cv_r2(pca_scores, y, folds, seed, degree=degree),
                   "n": int(np.isfinite(y).sum())}
    return out


# ----------------------------------------------------------------------------- mixture prior: clusters
def clusters(spec: ModelSpec, arrays: dict, mu: np.ndarray, logvar: np.ndarray | None = None) -> dict:
    """The clusters of a mixture prior: ``{"membership": (n, K), "label": (n,), "confidence": (n,)}`` for codes ``mu`` (the label is the most probable cluster)."""
    g = fw.memberships(spec, arrays, mu, logvar)
    return {"membership": g, "label": g.argmax(axis=1), "confidence": g.max(axis=1)}


def cluster_means(spec: ModelSpec, arrays: dict, frame=None) -> dict:
    """The mean code of every cluster decoded: ``{"z": (K, d), "pi": (K,), "X": (K, P), "functions"}``."""
    if not spec.has_mixture():
        raise AutoencoderError("This network has no mixture prior, so it has no clusters.")
    Z = np.asarray(arrays["prior.mu"], float)
    X = fw.decode(spec, arrays, Z)
    return {"z": Z, "pi": np.asarray(arrays["prior.pi"], float), "X": X, "functions": _functions(frame, X)}
