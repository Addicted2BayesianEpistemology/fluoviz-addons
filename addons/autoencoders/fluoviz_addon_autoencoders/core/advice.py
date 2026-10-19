"""Facts and advice from the data, in words (SPEC section 6): how many codes to start with, which size, how long it will take, and what to be careful about.  Numpy only.

``advise(data, cfg)`` is what ``GET /defaults`` and ``POST /check`` need beyond the facts of the data side (``Prepared.summary()``).
"""
from __future__ import annotations

import numpy as np

from . import linear as lin
from .spec import MIN_SAMPLES, SIZES, Config

FEEDBACK_SECONDS_PER_STEP = 4.0e-4          # fixed cost of one optimiser step of a small network on this kind of computer (measured: 0.4 ms)
SECONDS_PER_FLOP = 1.0 / 15e9               # matrix products of the (small) layers
SECONDS_PER_PARAMETER = 2.0e-9              # the optimiser update of every weight


def count_parameters(P: int, d: int, hidden: list[int], family: str = "ae", skip: bool = False) -> int:
    """The trainable weights of an autoencoder with an encoder of ``hidden`` widths and the mirrored decoder (``skip``: and the two linear shortcuts)."""
    enc = [P] + list(hidden)
    n = sum(a * b + b for a, b in zip(enc[:-1], enc[1:]))
    n += enc[-1] * d + d + (enc[-1] * d + d if family == "vae" else 0)
    dec = [d] + list(hidden[::-1])
    n += sum(a * b + b for a, b in zip(dec[:-1], dec[1:]))
    n += dec[-1] * P + P
    if skip:
        n += 2 * P * d
    return int(n)


def estimate_seconds(cfg: Config, n: int, P: int) -> float:
    """A rough time for the whole training (all random starts, and the folds when there are few samples) on a laptop CPU: every run is assumed to use about 70 % of the epochs
    (early stopping; all of them when early stopping is off), at most the time limit of one network when there is one.  Within a factor of two of what is measured on a 4-core computer."""
    hidden = cfg.hidden_widths()
    n_par = count_parameters(P, cfg.latent_dim, hidden, cfg.family, cfg.skip)
    kfold = n < cfg.kfold_below
    n_train = int(n * (1 - 1 / max(cfg.folds, 2))) if kfold else int(n - max(5, round(cfg.val_fraction * n)))
    B = cfg.batch_for(max(n_train, 2))
    steps = int(np.ceil(max(n_train, 1) / B))
    flops_step = 6.0 * B * n_par                                         # forward + backward of the dense layers (2 flops per weight, 3 for the backward pass)
    t_step = FEEDBACK_SECONDS_PER_STEP + flops_step * SECONDS_PER_FLOP + n_par * SECONDS_PER_PARAMETER
    t_epoch = steps * t_step + 2.0 * (n * 2.0 * n_par * SECONDS_PER_FLOP) + 2.0e-4          # + the evaluation of the training and validation sets
    runs = cfg.n_restarts + (cfg.folds if kfold else 0)
    share = 0.7 if cfg.uses_early_stopping() else 1.0                    # (without early stopping every network trains all its epochs)
    one = share * cfg.epochs * t_epoch
    if cfg.max_minutes is not None:
        one = min(one, 60.0 * cfg.max_minutes)
    return float(runs * one)


def words_for_seconds(s: float) -> str:
    if s < 10:
        return "a few seconds"
    if s < 90:
        return f"about {int(round(s / 5.0) * 5)} seconds"
    if s < 3600:
        return f"about {int(round(s / 60.0))} minutes"
    return f"about {s / 3600.0:.1f} hours"


def components_for_share(td, share: float = 0.99, k_max: int = 40) -> int:
    """The number of principal components (of the weighted coordinates: the PCA of the pixels, or the functional PCA in the L2 frame) that hold ``share`` of the variance."""
    n = td.n
    k = int(max(1, min(k_max, n - 1, td.P)))
    m = lin.fit_pca(td.X, td.M, td.omega, k)
    c = np.cumsum(m.lam) / max(float(m.total), 1e-300)
    hit = np.flatnonzero(c >= share - 1e-12)
    return int(hit[0]) + 1 if hit.size else k


def advise(data, cfg: Config | None = None) -> dict:
    """Advice for ``data`` (a ``Prepared`` or ``TrainData``) and the settings ``cfg`` (default: the defaults).

    Returns ``{"n", "P", "components_99", "latent_dim", "latent_reason", "size", "hidden", "family", "n_params", "samples_per_param", "samples_per_latent", "validation": {"mode", "n_train", "n_val",
    "folds", "words"}, "estimate": {"seconds", "words"}, "warnings": [...], "blocking": [...]}``.  The starting number of codes is ``min(components for 99 %, n / 10, 8)``, at least 2."""
    td = data.train_data("smooth") if hasattr(data, "train_data") else data
    cfg = cfg or Config()
    n, P = td.n, td.P
    k99 = components_for_share(td, 0.99)
    d0 = int(max(2, min(k99, n // 10, 8)))
    size = "small" if n <= 300 else "medium"
    hidden = list(SIZES[size])
    blocking: list[str] = []
    warnings: list[str] = []
    if n < MIN_SAMPLES:
        blocking.append(f"Only {n} samples are included: an autoencoder needs at least {MIN_SAMPLES} samples to be trained honestly. Add more samples, or use the PCA / functional PCA pages for very small data sets.")
    d = cfg.latent_dim
    if d >= P:
        blocking.append(f"The number of codes ({d}) must be smaller than the number of inputs ({P}).")
    n_par = count_parameters(P, d, cfg.hidden_widths(), cfg.family, cfg.skip)
    if n < cfg.kfold_below:
        folds = int(max(2, min(cfg.folds, n // 2)))
        val = {"mode": "kfold", "n_train": n, "n_val": n, "folds": folds,
               "words": f"With only {n} samples the network is checked by {folds}-fold validation: every sample is rebuilt by a network that never saw it, then the final network is trained on all samples."}
    else:
        nv = int(min(max(5, round(cfg.val_fraction * n)), n - 4))
        val = {"mode": "holdout", "n_train": n - nv, "n_val": nv, "folds": None, "words": f"{nv} samples are held out to check the network; it is trained on the other {n - nv}."}
    if n >= MIN_SAMPLES and n < 30:
        warnings.append(f"Only {n} samples: the numbers are uncertain and the network can easily memorise them. Treat this as an exploration, and compare with the PCA / functional PCA.")
    if n >= MIN_SAMPLES and n / d < 10:
        warnings.append(f"There are {n / d:.1f} samples per code: fewer than 10. The codes will be noisy; try {max(1, n // 10)} code{'s' if n // 10 != 1 else ''} or fewer.")
    if n_par > 5 * n and n >= MIN_SAMPLES:
        warnings.append(f"The network has {n_par:,} parameters for {n} samples: it can memorise the samples. The small size, weight decay and early stopping are on by default, but trust the numbers on held-out samples only.")
    if td.M is not None:
        miss = float(1.0 - td.M.mean())
        if miss > 0.2:
            warnings.append(f"{100 * miss:.0f} % of the values are not observed: the network ignores them in the loss, but strong missing data make the codes less reliable.")
        varies = float(np.mean((td.M.mean(axis=0) > 0) & (td.M.mean(axis=0) < 1)))
        if varies > 0.02:
            warnings.append("The set of unobserved points differs between samples: each sample is judged on the points it has; consider the functional version, which smooths over gaps.")
    est = estimate_seconds(cfg, n, P)
    return {"n": n, "P": P, "components_99": k99, "latent_dim": d0,
            "latent_reason": f"the smallest of: the components that explain 99 % of the variation ({k99}), one tenth of the samples ({n // 10}) and 8, but at least 2",
            "size": size, "hidden": hidden, "family": "ae", "n_params": n_par, "samples_per_param": n / max(n_par, 1), "samples_per_latent": n / max(d, 1), "validation": val,
            "estimate": {"seconds": est, "words": words_for_seconds(est)}, "warnings": warnings, "blocking": blocking}
