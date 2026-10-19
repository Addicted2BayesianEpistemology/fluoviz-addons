"""Training of the autoencoders (PyTorch, CPU).  **Imported only by the worker process and the tests** - never by the FluoViz server (PyTorch is imported lazily, inside functions).

The loss (SPEC section 3.3), per sample, with ``omega`` the weights of the L2 metric (``TrainData.omega``) and ``Q`` the smoothness metric of every block:

    rec   = sum_blocks sum_j omega_j m_j (x_j - xhat_j)^2 / (share of the block observed)        0 = perfect, 1 = as bad as the mean;  1 - rec = share of the variation rebuilt
    pen   = (1 / n_blocks) sum_blocks  xhat_b' Q_b xhat_b / K_b                                  the ``smoothify`` penalty on what the decoder produces (tr Q_b = K_b)

* autoencoder:            ``L = rec + s * pen``
* variational autoencoder (Gaussian likelihood with one noise variance ``sigma^2`` for all coordinates, ``N`` = the number of effective coordinates, see ``n_eff``):

      ``L = (N / 2) * [ rec / sigma^2 + log sigma^2 ]  +  (N / 2) * s * pen / sigma^2  +  beta_t * KL``

  ``KL`` is the Kullback-Leibler divergence of the posterior ``N(mu, sigma_z^2)`` from the prior, summed over the codes and averaged over the samples of the batch, with ``free_bits`` nats
  per code as a floor (below it a code costs nothing).  With ``sigma = "learned"`` the optimal ``sigma^2`` for the current reconstruction is used (``sigma^2 = rec``: the noise
  variance that maximises the likelihood, exact at every step); a number fixes it.  ``beta_t`` rises linearly from 0 to ``beta`` over the warm-up epochs.  ``N`` is a count of the
  independent numbers the data contain: ``n_eff = "auto"`` is ``min(P, max(15, 3 d))`` (see :func:`auto_n_eff`).  The penalty is divided by the same ``sigma^2`` as the misfit, so ``smoothify`` means the same in both families.  The ELBO is ``-L`` up to a constant.
* the linear shortcut (``skip``, default on): ``code = MLP(u) + L_e u`` and ``output = MLP(z) + L_d z`` - both linear maps go THROUGH the d-dimensional code (there is no path from the input to
  the output that avoids it).  The first random start begins at the PCA of the training samples (``L_e`` / ``L_d`` = the principal directions that carry signal, whitened; the MLP branches output
  zero; a code without signal starts at zero and, for a VAE, at the prior), and that start is itself a candidate for the best weights (``best_epoch == -1``): on held-out samples the kept
  network is never worse than the PCA with the same number of codes.  The other starts begin at random weights.  ``skip = False`` is the classical autoencoder.
* when training stops (``Config.stop_rule``): ``validation`` (default) the loss on the held-out samples has not improved by ``min_delta`` (0.1 %) for ``patience`` epochs and at least ``min_epochs`` have run (the
  loss can be averaged over ``val_window`` epochs first); ``training``: the same on the training loss (no held-out check); ``never``: all ``epochs`` are trained.  Either way the weights of the best epoch are
  restored (``restore_best``), a time limit per network (``max_minutes``) can stop it, and the learning rate follows ``lr_schedule`` (cosine over ``epochs`` | cosine over the epochs this network is
  planned to train | constant | halved when the loss stalls).  With k-fold validation the final networks train for the epochs the folds found (``final_epochs``: "auto", "max" or a number).  The defaults are
  what the training did before these settings existed.
* the mixture prior (VaDE-like) replaces the KL by ``E_q[log q(z|x) - log p(z)]`` with ``p(z) = sum_c pi_c N(mu_c, sigma_c^2)`` (one-sample estimate); the mixture is initialised by a
  Gaussian mixture fitted to the codes after a first stage with the normal prior.
"""
from __future__ import annotations

import copy
import json
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np

from . import forward as fw
from . import linear as lin
from .data import TrainData
from .spec import AutoencoderError, Config, ModelSpec

_NET = None
LOGVAR_BIAS0 = -3.0          # the starting log variance of the posterior of every code


# ----------------------------------------------------------------------------- torch, lazily
def _torch():
    import torch
    return torch


def configure_torch(cfg: Config | None = None, n_params: int | None = None) -> int:
    """Threads and deterministic algorithms; returns the number of threads.  ``FLUOVIZ_BLAS_THREADS`` wins, then ``cfg.threads``; otherwise the size of the network decides: one thread
    for a small network (the work per step is so small that threads only wait for each other), two above 100 000 weights (the pixel versions), never more than the cores and four.
    (Measured on the example data: with four threads on a busy computer the pixel version took five times longer than with two.)"""
    torch = _torch()
    env = os.environ.get("FLUOVIZ_BLAS_THREADS", "").strip()
    cores = os.cpu_count() or 1
    if env.isdigit() and int(env) > 0:
        n = int(env)
    elif cfg is not None and cfg.threads:
        n = int(cfg.threads)
    else:
        n = 1 if (n_params is not None and n_params < 100_000) else 2
    n = max(1, min(n, cores, 4)) if not (env.isdigit() and int(env) > 0) else max(1, n)
    torch.set_num_threads(n)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:                                                   # noqa: BLE001
        pass
    return int(n)


def _net_class():
    """The ``torch.nn.Module`` of the autoencoder (defined on first use so that this module imports without PyTorch)."""
    global _NET
    if _NET is not None:
        return _NET
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    acts = {"silu": F.silu, "tanh": torch.tanh, "gelu": lambda x: F.gelu(x, approximate="none"), "relu": F.relu, "softplus": F.softplus, "elu": F.elu, "linear": lambda x: x}

    class Net(nn.Module):
        def __init__(self, P: int, d: int, enc_hidden, dec_hidden, act: str, vae: bool, dropout: float, n_mix: int = 0, skip: bool = False):
            super().__init__()
            self.act = acts[act]
            self.vae = vae
            self.skip = skip
            if skip:
                self.lin_enc = nn.Linear(P, d, bias=False)
                self.lin_dec = nn.Linear(d, P, bias=False)
            dims = [P] + list(enc_hidden)
            self.enc = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
            self.mu = nn.Linear(dims[-1], d)
            self.logvar = nn.Linear(dims[-1], d) if vae else None
            if vae:
                with torch.no_grad():
                    self.logvar.weight.mul_(0.1)
                    self.logvar.bias.fill_(LOGVAR_BIAS0)
            dims = [d] + list(dec_hidden)
            self.dec = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
            self.out = nn.Linear(dims[-1], P)
            self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            if n_mix:
                self.mix_logit = nn.Parameter(torch.zeros(n_mix))
                self.mix_mu = nn.Parameter(torch.zeros(n_mix, d))
                self.mix_logvar = nn.Parameter(torch.zeros(n_mix, d))

        def encode(self, u):
            h = u
            for lay in self.enc:
                h = self.drop(self.act(lay(h)))
            mu = self.mu(h)
            if self.skip:
                mu = mu + self.lin_enc(u)
            return mu, (self.logvar(h) if self.vae else None)

        def decode(self, z):
            h = z
            for lay in self.dec:
                h = self.act(lay(h))
            o = self.out(h)
            return o + self.lin_dec(z) if self.skip else o

    _NET = Net
    return Net


# ----------------------------------------------------------------------------- the data of one run as tensors
class _Data:
    """Tensors of a set of samples: ``X`` (0 where not observed), the observed mask ``M`` and ``F`` (the factor that scales the error of a partly observed block up to the block)."""

    def __init__(self, td: TrainData, idx):
        torch = _torch()
        idx = np.asarray(idx)
        self.idx = idx
        self.n = int(idx.size)
        X = td.X[idx]
        self.X = torch.as_tensor(np.where(np.isfinite(X), X, 0.0), dtype=torch.float32)
        self.M = None
        self.F = None
        self.W = None if td.sw is None else torch.as_tensor(np.asarray(td.sw, float)[idx], dtype=torch.float32)
        if td.M is not None:
            M = td.M[idx] & np.isfinite(X)
            if not M.all():
                self.M = torch.as_tensor(M.astype(np.float32))
                F = np.ones((self.n, len(td.blocks)))
                for j, b in enumerate(td.blocks):
                    a, c = b["slice"]
                    den = (M[:, a:c] * td.omega[a:c]).sum(axis=1)
                    F[:, j] = td.omega[a:c].sum() / np.maximum(den, 1e-300)
                self.F = torch.as_tensor(F, dtype=torch.float32)


class _Loss:
    """The pieces of the loss as torch functions (see the module docstring)."""

    def __init__(self, td: TrainData, cfg: Config, n_eff: int, scale: np.ndarray):
        torch = _torch()
        self.cfg, self.n_eff = cfg, int(n_eff)
        self.scale = torch.as_tensor(scale, dtype=torch.float32)
        self.sd_u = torch.as_tensor(td.sd / scale, dtype=torch.float32)
        self.omega = torch.as_tensor(td.omega, dtype=torch.float32)
        self.slices = [slice(int(b["slice"][0]), int(b["slice"][1])) for b in td.blocks]
        self.nb = len(self.slices)
        self.q = []
        for sl, q in zip(self.slices, td.q):
            K = sl.stop - sl.start
            if q["kind"] == "identity":
                self.q.append(("identity", None, K))
            elif q["kind"] == "diag":
                self.q.append(("diag", torch.as_tensor(q["d"], dtype=torch.float32), K))
            else:
                self.q.append(("dense", torch.as_tensor(q["Q"], dtype=torch.float32), K))

    def rec(self, x, xh, m, f, w=None):
        """Per-sample reconstruction error ``(B,)``; ``w`` ``(B,)``: the weight of every sample (a sample that lacks part of the domain counts in proportion to what was measured)."""
        e2 = (x - xh) ** 2 * self.omega
        if m is not None:
            e2 = e2 * m
        out = e2.sum(dim=1) if f is None else sum(e2[:, sl].sum(dim=1) * f[:, j] for j, sl in enumerate(self.slices))
        return out if w is None else out * w

    def pen(self, xh):
        """Per-sample ``smoothify`` penalty ``(B,)`` (average over the blocks of ``xhat' Q xhat / K``)."""
        tot = 0.0
        for sl, (kind, Q, K) in zip(self.slices, self.q):
            v = xh[:, sl]
            if kind == "identity":
                tot = tot + (v * v).sum(dim=1) / K
            elif kind == "diag":
                tot = tot + ((v * v) * Q).sum(dim=1) / K
            else:
                tot = tot + ((v @ Q) * v).sum(dim=1) / K
        return tot / self.nb


# ----------------------------------------------------------------------------- the result
@dataclass
class TrainResult:
    spec: ModelSpec
    arrays: dict                       # float64 arrays of the network (spec.array_shapes())
    history: list                      # per epoch of the network that was kept: dicts (train, val, rec, kl, smooth, beta, sigma2, lr, best ...)
    validation: dict                   # honest numbers on held-out samples (and the training values), the PCA baseline, VAE facts
    samples: dict                      # per sample: codes, codes_sd, error, error_blocks, heldout, error_heldout, fold
    restarts: dict                     # summary of the random starts: spread, which one was kept
    best_epoch: int
    epochs_run: int
    split: dict                        # {"mode": "holdout" | "kfold", "train", "val", "folds": [[...]], "seed"}
    config: dict
    n_parameters: int
    sigma: float | None
    seconds: float
    versions: dict
    notes: list = field(default_factory=list)
    restart_codes: np.ndarray | None = None          # (R, n, d) codes of every sample from each random start (for the stability check)
    fold_info: list = field(default_factory=list)    # k-fold mode: per fold {fold, best_epoch, val_error, ...}
    all_histories: list = field(default_factory=list)  # the histories of every run (restart / fold), light dicts

    def meta(self) -> dict:
        """Everything except the weights, as plain JSON-able data."""
        return {"spec": self.spec.to_dict(), "validation": _jsonable(self.validation), "restarts": _jsonable(self.restarts), "best_epoch": int(self.best_epoch),
                "epochs_run": int(self.epochs_run), "split": _jsonable(self.split), "config": _jsonable(self.config), "n_parameters": int(self.n_parameters),
                "sigma": self.sigma, "seconds": float(self.seconds), "versions": dict(self.versions), "notes": list(self.notes), "fold_info": _jsonable(self.fold_info)}


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# ----------------------------------------------------------------------------- the history file (JSON lines, appended one line at a time)
class HistoryWriter:
    """One JSON object per line, appended and flushed with a single ``write`` (the page can read the file while it grows: a line that is not complete yet is ignored by
    :func:`read_history`).  ``None`` as a path does nothing."""

    def __init__(self, path):
        self.path = None if path is None else str(path)
        self._fd = None
        if self.path:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            self._fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)

    def write(self, rec: dict) -> None:
        if self._fd is None:
            return
        line = (json.dumps(_jsonable(rec), allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
        os.write(self._fd, line)

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


def read_history(path, since: int = 0) -> tuple[list[dict], int]:
    """The records of a history file after the first ``since`` records, and the index to pass next time (an incomplete last line is left for the next call).
    A missing file gives ``([], since)``."""
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError:
        return [], since
    lines = raw.split(b"\n")
    complete = lines[:-1]                                                # the last piece has no newline yet (empty, or being written)
    out = []
    for ln in complete[since:]:
        try:
            out.append(json.loads(ln.decode("utf-8")))
        except ValueError:
            continue
    return out, len(complete)


# ----------------------------------------------------------------------------- helpers
def make_split(n: int, cfg: Config) -> dict:
    """The validation scheme: a hold-out split (``n >= kfold_below``) or k folds (fewer samples).  Deterministic from the seed."""
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, 1]))
    perm = rng.permutation(n)
    if n < cfg.kfold_below:
        F = int(max(2, min(cfg.folds, n // 2)))
        folds = [np.sort(f).tolist() for f in np.array_split(perm, F)]
        return {"mode": "kfold", "folds": folds, "train": list(range(n)), "val": [], "seed": cfg.seed}
    n_val = int(min(max(5, round(cfg.val_fraction * n)), n - 4))
    val = np.sort(perm[:n_val])
    train = np.sort(perm[n_val:])
    return {"mode": "holdout", "folds": [], "train": train.tolist(), "val": val.tolist(), "seed": cfg.seed}


def _seeds(cfg: Config, n_runs: int, tag: int) -> list[int]:
    ss = np.random.SeedSequence([cfg.seed, 2, tag])
    return [int(c.generate_state(1)[0] & 0x7FFFFFFF) for c in ss.spawn(n_runs)]


def auto_n_eff(td: TrainData, d: int | None = None) -> int:
    """The effective number of coordinates ``N`` of the VAE likelihood when ``vae.n_eff`` is ``"auto"``: ``min(P, max(15, 3 d))``.  It is how many independent numbers the network is
    asked to reproduce: it sets the exchange rate between reconstruction (``N/2`` per unit of ``log`` error) and the KL cost of a code.  Measured: below about 10 a variational autoencoder
    collapses codes it needs, the larger it is the more small details (noise) the codes are used for, the more their KL stays above zero and the worse they fill the prior.  With 15-20,
    two real factors and six codes leave four codes at the prior (KL below 0.05 nats) and the reconstructions are as good as with 50."""
    return int(min(td.P, max(15, 3 * (d or 0))))


def _train_mean(td: TrainData, idx) -> np.ndarray:
    X = td.X[idx]
    if td.M is None:
        return X.mean(axis=0)
    M = td.M[idx]
    return np.where(M, X, 0.0).sum(axis=0) / np.maximum(M.sum(axis=0), 1)


def input_scale(td: TrainData, idx, share: float = 0.99) -> np.ndarray:
    """The input standardisation ``in.scale`` ``(P,)``: the network works on ``u = (x - mean) / scale`` with ``scale_j = 1 / (c sqrt(omega_j))``, i.e. on the coordinates of the L2 frame
    (the metric of the loss), so the loss is a plain squared error in ``u`` and the many coordinates that carry almost no variance (the rough, noisy ones) are small inputs instead of
    unit-variance noise the encoder could use to memorise the samples.  ``c^2`` is the number of principal components that hold ``share`` of the variance (so that every
    effective direction has about unit variance)."""
    idx = np.asarray(idx)
    X, mean = td.X[idx], _train_mean(td, idx)
    Xc = X - mean
    if td.M is not None:
        Xc = np.where(td.M[idx], Xc, 0.0)
    sw = np.sqrt(td.omega)
    s = np.linalg.svd(Xc * sw, compute_uv=False)
    c2 = np.cumsum(s ** 2) / max(float((s ** 2).sum()), 1e-300)
    k = int(np.searchsorted(c2, share - 1e-12) + 1)
    return 1.0 / (np.sqrt(float(min(max(k, 1), 64))) * sw)


# ----------------------------------------------------------------------------- the objective, evaluated on a set of samples
def _objective(net, loss: _Loss, cfg: Config, mean, data: _Data, rows, *, train: bool, beta_t: float, sigma_fixed: float | None, mix_active: bool, eps=None, corrupt_gen=None):
    scale = loss.scale
    """Loss pieces on the rows ``rows`` of ``data`` (a torch index tensor or slice).  Returns a dict of torch scalars (``loss`` carries the gradient in training)."""
    torch = _torch()
    x = data.X[rows]
    m = None if data.M is None else data.M[rows]
    f = None if data.F is None else data.F[rows]
    w = None if data.W is None else data.W[rows]
    u = (x - mean) / scale
    if m is not None:
        u = u * m
    if train:
        c = cfg.corruption
        if c["noise"] > 0:
            u = u + c["noise"] * loss.sd_u * torch.randn(u.shape, generator=corrupt_gen)
        if c["mask"] > 0:
            u = u * (torch.rand(u.shape, generator=corrupt_gen) >= c["mask"]).to(u.dtype)
    mu, lv = net.encode(u)
    out = {}
    s = cfg.smoothify
    if not net.vae:
        xh = net.decode(mu) * scale + mean
        rec_i = loss.rec(x, xh, m, f, w)
        pen_i = loss.pen(xh) if s > 0 else None
        rec = rec_i.mean()
        pen = pen_i.mean() if pen_i is not None else torch.zeros(())
        out.update(loss=rec + s * pen, rec=rec, pen=pen, kl=torch.zeros(()), sigma2=torch.zeros(()), rec_det=rec)
        return out
    lv = torch.clamp(lv, -20.0, 10.0)
    if eps is None:
        eps = torch.randn(mu.shape, generator=corrupt_gen)
    z = mu + torch.exp(0.5 * lv) * eps
    xh = net.decode(z) * scale + mean
    rec_i = loss.rec(x, xh, m, f, w)
    pen_i = loss.pen(xh) if s > 0 else None
    rec = rec_i.mean()
    pen = pen_i.mean() if pen_i is not None else torch.zeros(())
    if sigma_fixed is None:
        sig2 = torch.clamp(rec.detach(), min=1e-8)
    else:
        sig2 = torch.as_tensor(sigma_fixed ** 2, dtype=rec.dtype)
    if mix_active:
        logq = -0.5 * (math.log(2 * math.pi) + lv + eps ** 2).sum(dim=1)
        lp = torch.log_softmax(net.mix_logit, dim=0)[None, :] - 0.5 * (math.log(2 * math.pi) + net.mix_logvar[None] + (z[:, None, :] - net.mix_mu[None]) ** 2 / torch.exp(net.mix_logvar)[None]).sum(dim=2)
        logp = torch.logsumexp(lp, dim=1)
        kl = (logq - logp).mean()
        kl_loss = kl
    else:
        kl_dim = (0.5 * (mu ** 2 + torch.exp(lv) - 1.0 - lv)).mean(dim=0)
        kl = kl_dim.sum()
        kl_loss = torch.clamp(kl_dim, min=cfg.vae["free_bits"]).sum() if cfg.vae["free_bits"] > 0 else kl
    half_n = 0.5 * loss.n_eff
    obj = half_n * (rec / sig2 + torch.log(sig2)) + half_n * s * pen / sig2 + beta_t * kl_loss
    out.update(loss=obj, rec=rec, pen=pen, kl=kl.detach(), sigma2=sig2.detach())
    if not train:
        with torch.no_grad():
            out["rec_det"] = loss.rec(x, net.decode(mu) * scale + mean, m, f, w).mean()
    return out


def _evaluate(net, loss, cfg, mean, data: _Data, beta, sigma_fixed, mix_active, chunk: int = 512):
    """Objective and parts on all of ``data`` in evaluation mode (no dropout, no corruption); the posterior sample uses a fixed draw so that epochs are comparable."""
    torch = _torch()
    net.eval()
    tot = {"loss": 0.0, "rec": 0.0, "pen": 0.0, "kl": 0.0, "rec_det": 0.0, "sigma2": 0.0}
    if net.vae:                                                        # the objective is not additive over chunks (sigma^2 depends on the whole set): one pass
        chunk = max(chunk, data.n)
    with torch.no_grad():
        g = torch.Generator().manual_seed(12345)
        for a in range(0, data.n, chunk):
            rows = slice(a, min(a + chunk, data.n))
            nrow = rows.stop - rows.start
            eps = torch.randn((nrow, net.mu.out_features), generator=g) if net.vae else None
            o = _objective(net, loss, cfg, mean, data, rows, train=False, beta_t=beta, sigma_fixed=sigma_fixed, mix_active=mix_active, eps=eps)
            for k in tot:
                tot[k] += float(o[k]) * nrow
    return {k: v / data.n for k, v in tot.items()}


def _lr_at(epoch: int, epochs: int, lr: float, final: float = 0.01) -> float:
    """Cosine decay from ``lr`` to ``final * lr`` (default 0.01) over the planned epochs."""
    lo = final * lr
    return lo + 0.5 * (lr - lo) * (1.0 + math.cos(math.pi * min(epoch, epochs) / max(epochs, 1)))


def _gmm_init(Z: np.ndarray, K: int, seed: int, n_iter: int = 60, n_init: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Diagonal Gaussian mixture of the codes ``Z`` ``(n, d)``: k-means++ starts (``n_init`` of them), a few k-means steps, then EM; the start with the best log likelihood is kept.
    Returns ``(pi, mu, logvar)``."""
    rng = np.random.default_rng(seed)
    n, d = Z.shape
    K = int(min(K, n))
    floor = 1e-3 * float(Z.var(axis=0).mean() + 1e-12)
    best = None
    for _ in range(int(n_init)):
        c = [Z[rng.integers(n)]]
        for _ in range(1, K):
            D = np.min([((Z - ci) ** 2).sum(axis=1) for ci in c], axis=0)
            p = D / D.sum() if D.sum() > 0 else np.full(n, 1.0 / n)
            c.append(Z[rng.choice(n, p=p)])
        mu = np.array(c)
        for _ in range(10):                                                  # Lloyd's k-means
            lab = ((Z[:, None, :] - mu[None]) ** 2).sum(axis=2).argmin(axis=1)
            mu = np.array([Z[lab == k].mean(axis=0) if (lab == k).any() else mu[k] for k in range(K)])
        lab = ((Z[:, None, :] - mu[None]) ** 2).sum(axis=2).argmin(axis=1)
        var = np.array([Z[lab == k].var(axis=0) + floor if (lab == k).sum() > 1 else Z.var(axis=0) + floor for k in range(K)])
        pi = np.maximum(np.bincount(lab, minlength=K) / n, 1e-3)
        pi = pi / pi.sum()
        ll = -np.inf
        for _ in range(n_iter):
            lp = np.log(pi)[None] - 0.5 * (np.log(2 * np.pi * var)[None] + (Z[:, None, :] - mu[None]) ** 2 / var[None]).sum(axis=2)
            m = lp.max(axis=1, keepdims=True)
            ll_new = float((m[:, 0] + np.log(np.exp(lp - m).sum(axis=1))).sum())
            r = np.exp(lp - m)
            r /= r.sum(axis=1, keepdims=True)
            nk = r.sum(axis=0) + 1e-9
            pi = nk / n
            mu = (r.T @ Z) / nk[:, None]
            var = np.maximum((r.T @ (Z ** 2)) / nk[:, None] - mu ** 2, 0.0) + floor
            if abs(ll_new - ll) < 1e-8 * max(abs(ll_new), 1.0):
                ll = ll_new
                break
            ll = ll_new
        if best is None or ll > best[0]:
            best = (ll, pi.copy(), mu.copy(), np.log(var))
    return best[1], best[2], best[3]


# ----------------------------------------------------------------------------- one training run
@dataclass
class _Run:
    net: object
    state: dict
    history: list
    best_epoch: int
    epochs_run: int
    val_loss: float
    train_eval: dict
    val_eval: dict | None
    stopped: str
    seed: int
    mean: np.ndarray
    sigma2: float | None
    scale: np.ndarray


def _train_run(td: TrainData, tr_idx, va_idx, cfg: Config, n_eff: int, seed: int, stop_at: int, hist_cb=None, tag: dict | None = None, pca_start: bool = True) -> _Run:
    """One network trained from one random start on the samples ``tr_idx`` (early stopping on ``va_idx`` when given; otherwise it runs ``stop_at`` epochs)."""
    torch = _torch()
    Net = _net_class()
    tag = tag or {}
    P, d = td.P, cfg.latent_dim
    vae = cfg.family == "vae"
    mixture = vae and cfg.vae["prior"] == "mixture"
    trd, vad = _Data(td, tr_idx), (None if va_idx is None or len(va_idx) == 0 else _Data(td, va_idx))
    scale_np = input_scale(td, tr_idx)
    loss = _Loss(td, cfg, n_eff, scale_np)
    mean_np = _train_mean(td, tr_idx)
    mean = torch.as_tensor(mean_np, dtype=torch.float32)
    torch.manual_seed(seed)
    net = Net(P, d, cfg.hidden_widths(), cfg.decoder_widths(), cfg.activation, vae, cfg.dropout, int(cfg.vae["n_components"]) if mixture else 0, cfg.skip)
    pca_start = bool(cfg.skip and pca_start)                                     # the first random start begins at the PCA solution, the others from random weights
    if pca_start:
        _pca_init(net, td, tr_idx, mean_np, scale_np, d)
    main = [p for nme, p in net.named_parameters() if not nme.startswith("mix_")]
    groups = [{"params": main, "weight_decay": cfg.weight_decay}]
    if mixture:
        groups.append({"params": [net.mix_logit, net.mix_mu, net.mix_logvar], "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, lr=cfg.lr)
    B = cfg.batch_for(trd.n)
    warm = cfg.warmup_epochs()
    beta_final = float(cfg.vae["beta"]) if vae else 0.0
    sigma_fixed = None if (not vae or cfg.vae["sigma"] == "learned") else float(cfg.vae["sigma"])
    stage1 = int(max(warm, round(0.25 * cfg.epochs))) if mixture else 0            # the mixture prior takes over after this many epochs
    first_eligible = max(warm, stage1)                                              # the best epoch is searched after the warm-up (the loss changes while beta rises)
    scale = 0.5 * loss.n_eff if vae else 1.0
    gen = torch.Generator().manual_seed(seed + 7919)
    cgen = torch.Generator().manual_seed(seed + 104729)
    best_val, best_state, best_epoch, bad = math.inf, None, -1, 0
    rule = cfg.stop_rule
    watch = "train" if rule == "training" else ("val" if vad is not None else None)  # which loss decides what the best epoch is and when training stops (None: it does not)
    can_stop = rule != "never"
    recent: list = []                                                                # the last monitored losses (``cfg.val_window`` of them are averaged)
    if pca_start and watch == "val" and not vae:
        # the start (the PCA of the training samples, a purely linear network) is the first candidate: the network that is kept is never worse than the linear one on the held-out samples
        v0 = _evaluate(net, loss, cfg, mean, vad, beta_final, sigma_fixed, False)
        best_val, best_epoch = v0["loss"], -1
        best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    history: list = []
    mix_on = False
    stopped = "max"
    t_start = time.time()
    lr_scale, bad_lr, best_lr = 1.0, 0, math.inf                                     # the 'plateau' schedule: halve the learning rate when the loss stalls
    for epoch in range(int(stop_at)):
        if cfg.lr_schedule == "cosine":
            lr = _lr_at(epoch, cfg.epochs, cfg.lr, cfg.lr_final)
        elif cfg.lr_schedule == "cosine_run":
            lr = _lr_at(epoch, int(stop_at), cfg.lr, cfg.lr_final)
        elif cfg.lr_schedule == "constant":
            lr = cfg.lr
        else:
            lr = cfg.lr * lr_scale
        for gp in opt.param_groups:
            gp["lr"] = lr
        beta_t = beta_final * (min(1.0, (epoch + 1) / warm) if warm > 0 else 1.0)
        if mixture and not mix_on and epoch >= stage1:                              # switch to the mixture prior, initialised from the codes
            net.eval()
            with torch.no_grad():
                u = (trd.X - mean) / loss.scale * (1.0 if trd.M is None else trd.M)
                Z = net.encode(u)[0].numpy().astype(float)
            pi, mu_c, lv_c = _gmm_init(Z, int(cfg.vae["n_components"]), seed)
            with torch.no_grad():
                net.mix_logit.copy_(torch.as_tensor(np.log(np.maximum(pi, 1e-9)), dtype=torch.float32))
                net.mix_mu.copy_(torch.as_tensor(mu_c, dtype=torch.float32))
                net.mix_logvar.copy_(torch.as_tensor(lv_c, dtype=torch.float32))
            mix_on = True
        net.train()
        perm = torch.randperm(trd.n, generator=gen)
        batch_loss, nb = 0.0, 0
        for a in range(0, trd.n, B):
            rows = perm[a:a + B]
            o = _objective(net, loss, cfg, mean, trd, rows, train=True, beta_t=beta_t, sigma_fixed=sigma_fixed, mix_active=mix_on, corrupt_gen=cgen)
            opt.zero_grad(set_to_none=True)
            o["loss"].backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
            opt.step()
            batch_loss += float(o["loss"].detach())
            nb += 1
        tr = _evaluate(net, loss, cfg, mean, trd, beta_final, sigma_fixed, mix_on)
        va = _evaluate(net, loss, cfg, mean, vad, beta_final, sigma_fixed, mix_on) if vad is not None else None
        rec = {"event": "epoch", **tag, "epoch": epoch, "train": tr["loss"], "val": None if va is None else va["loss"], "rec": tr["rec_det"], "rec_val": None if va is None else va["rec_det"],
               "kl": tr["kl"] if vae else None, "kl_val": (None if va is None else va["kl"]) if vae else None, "smooth": tr["pen"] if cfg.smoothify > 0 else None,
               "smooth_val": (None if va is None else va["pen"]) if cfg.smoothify > 0 else None, "beta": beta_t if vae else None, "sigma2": tr["sigma2"] if vae else None,
               "lr": lr, "train_batch": batch_loss / max(nb, 1), "best": False}
        if not math.isfinite(tr["loss"]):
            raise AutoencoderError("The training diverged (the loss is not a number). Lower the learning rate, or use a smaller network or more weight decay.")
        stop = False
        mon = tr["loss"] if watch == "train" else (va["loss"] if (watch == "val" and va is not None) else None)
        if mon is not None:
            recent.append(mon)
            smooth = float(np.mean(recent[-cfg.val_window:])) if cfg.val_window > 1 else mon          # the loss the decisions are made on
            if epoch >= first_eligible:
                if not math.isfinite(best_val) or smooth < best_val - cfg.min_delta * (abs(best_val) + 1e-3 * scale):        # better by at least ``min_delta`` (relative; 0.1 % by default)
                    best_val, best_epoch, bad = smooth, epoch, 0
                    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                    rec["best"] = True
                else:
                    bad += 1
                    if can_stop and bad >= cfg.patience and epoch + 1 >= cfg.min_epochs:
                        stop, stopped = True, "early"
            if cfg.lr_schedule == "plateau":
                if not math.isfinite(best_lr) or smooth < best_lr - cfg.min_delta * (abs(best_lr) + 1e-3 * scale):
                    best_lr, bad_lr = smooth, 0
                else:
                    bad_lr += 1
                    if bad_lr >= max(3, cfg.patience // 4):
                        lr_scale, bad_lr = max(cfg.lr_final, lr_scale * 0.5), 0
        elif cfg.lr_schedule == "plateau":                                          # nothing is monitored (the final networks of the k-fold validation): the training loss
            if not math.isfinite(best_lr) or tr["loss"] < best_lr - cfg.min_delta * (abs(best_lr) + 1e-3 * scale):
                best_lr, bad_lr = tr["loss"], 0
            else:
                bad_lr += 1
                if bad_lr >= max(3, cfg.patience // 4):
                    lr_scale, bad_lr = max(cfg.lr_final, lr_scale * 0.5), 0
        if not stop and cfg.max_minutes is not None and time.time() - t_start >= 60.0 * cfg.max_minutes:
            stop, stopped = True, "time"
        history.append(rec)
        if hist_cb is not None:
            hist_cb(rec)
        if stop:
            break
    epochs_run = len(history)
    if cfg.restore_best and best_state is not None and watch is not None:
        net.load_state_dict(best_state)
    else:                                                                            # nothing was monitored, or the weights of the last epoch are wanted
        best_epoch = epochs_run - 1
    tr = _evaluate(net, loss, cfg, mean, trd, beta_final, sigma_fixed, mix_on)
    va = _evaluate(net, loss, cfg, mean, vad, beta_final, sigma_fixed, mix_on) if vad is not None else None
    if vae:
        sig2 = float(sigma_fixed ** 2) if sigma_fixed is not None else float(max((va or tr)["rec"], 1e-8))
    else:
        sig2 = None
    net.eval()
    return _Run(net, {k: v.detach().clone() for k, v in net.state_dict().items()}, history, int(best_epoch), epochs_run, float(va["loss"] if va is not None else tr["loss"]), tr, va, stopped, seed, mean_np, sig2, scale_np)


PCA_FLOOR = 1e-3        # principal directions with less than this share of the variance of the first one are not part of the linear start (see _pca_init)


def _pca_init(net, td: TrainData, tr_idx, mean_np: np.ndarray, scale_np: np.ndarray, d: int) -> None:
    """Start from the PCA of the training samples: the linear shortcuts become the first ``d`` principal directions that carry signal (the code has unit variance; a direction with less
    than ``PCA_FLOOR`` of the variance of the first one is noise and its code starts at zero, so that a code the decoder does not need stays at the prior), and the nonlinear branches start at
    zero output.  The network then begins with the quality of the PCA / functional PCA with the same number of components, and training can only add what the nonlinearity helps with;
    on top of that a linear path extrapolates (a sample with five times the concentration of the others is still described)."""
    torch = _torch()
    X = td.X[np.asarray(tr_idx)]
    obs = np.ones_like(X, bool) if td.M is None else (td.M[np.asarray(tr_idx)] & np.isfinite(X))
    u = np.where(obs, (X - mean_np) / scale_np, 0.0)
    m = lin.fit_pca(u, None if obs.all() else obs, np.ones(td.P), d, sample_w=td.sw_of(tr_idx))
    k = m.Vw.shape[0]
    lam = np.maximum(m.lam[:k], 1e-300)
    use = lam >= PCA_FLOOR * lam[0]                                        # the directions that carry signal; the others (noise) are left at zero: their codes start at the prior
    Le = np.zeros((d, td.P))
    Ld = np.zeros((td.P, d))
    Le[:k][use] = m.Vw[use] / np.sqrt(lam[use])[:, None]
    Ld[:, :k][:, use] = (m.Vw[use] * np.sqrt(lam[use])[:, None]).T
    with torch.no_grad():
        net.lin_enc.weight.copy_(torch.as_tensor(Le, dtype=torch.float32))
        net.lin_dec.weight.copy_(torch.as_tensor(Ld, dtype=torch.float32))
        net.mu.weight.zero_()
        net.mu.bias.zero_()
        net.out.weight.zero_()
        net.out.bias.zero_()
        if net.vae:                                                        # a code without signal starts AT the prior (mean 0, variance 1): nothing to un-learn if the decoder never needs it
            free = np.ones(d, bool)
            free[:k] = ~use
            net.logvar.weight[torch.as_tensor(free)] = 0.0
            net.logvar.bias[torch.as_tensor(free)] = 0.0


def _export_arrays(run: _Run, td: TrainData, cfg: Config) -> dict[str, np.ndarray]:
    """The network of a run as the float64 arrays of ``ModelSpec.array_shapes()``."""
    net = run.net
    a: dict[str, np.ndarray] = {}

    def lin_(name, m):
        a[f"{name}.W"] = m.weight.detach().numpy().astype(np.float64)
        a[f"{name}.b"] = m.bias.detach().numpy().astype(np.float64)

    for i, m in enumerate(net.enc):
        a[f"enc.W{i}"], a[f"enc.b{i}"] = m.weight.detach().numpy().astype(np.float64), m.bias.detach().numpy().astype(np.float64)
    lin_("enc.mu", net.mu)
    if net.vae:
        lin_("enc.logvar", net.logvar)
    for i, m in enumerate(net.dec):
        a[f"dec.W{i}"], a[f"dec.b{i}"] = m.weight.detach().numpy().astype(np.float64), m.bias.detach().numpy().astype(np.float64)
    lin_("dec.out", net.out)
    if net.skip:
        a["enc.lin.W"] = net.lin_enc.weight.detach().numpy().astype(np.float64)
        a["dec.lin.W"] = net.lin_dec.weight.detach().numpy().astype(np.float64)
    if net.vae and cfg.vae["prior"] == "mixture":
        import torch
        a["prior.pi"] = torch.softmax(net.mix_logit.detach(), 0).numpy().astype(np.float64)
        a["prior.mu"] = net.mix_mu.detach().numpy().astype(np.float64)
        a["prior.logvar"] = net.mix_logvar.detach().numpy().astype(np.float64)
    a["in.mean"] = np.asarray(run.mean, float).copy()
    a["in.scale"] = np.asarray(run.scale, float).copy()
    a["out.weights"] = np.asarray(td.omega, float).copy()
    return a


# ----------------------------------------------------------------------------- numbers from the numpy forward pass (what the saved model will do)
def _numbers(spec: ModelSpec, arrays: dict, td: TrainData, idx) -> dict:
    """Error of the samples ``idx`` with the numpy network: per sample total and per block, codes, posterior SDs, KL per sample (normal prior; a mixture: one-sample estimate)."""
    idx = np.asarray(idx)
    X = td.X[idx]
    M = None if td.M is None else td.M[idx]
    Xh, mu, lv = fw.reconstruct(spec, arrays, X, M)
    tot, per = fw.recon_error(spec, arrays, X, Xh, M)
    null, _ = fw.null_error(spec, arrays, X, M)
    out = {"error": tot, "error_blocks": per, "null": null, "codes": mu, "logvar": lv}
    if lv is not None:                                                   # the same KL as the saved model's (kinds.stats)
        from ..kinds import stats as kstats
        out["kl"] = kstats.kl_normal(mu, lv) if not spec.has_mixture() else kstats.kl_mixture(mu, lv, arrays["prior.pi"], arrays["prior.mu"], arrays["prior.logvar"])
    return out


def evr(err: np.ndarray, null: np.ndarray) -> float:
    """Share of the variation rebuilt: ``1 - sum(error) / sum(error of the mean)``."""
    err, null = np.asarray(err, float), np.asarray(null, float)
    ok = np.isfinite(err) & np.isfinite(null)
    s0 = float(null[ok].sum())
    return float(1.0 - err[ok].sum() / s0) if s0 > 0 else float("nan")


def _evr_blocks(per: np.ndarray, per0: np.ndarray, names: list[str]) -> dict:
    out = {}
    for j, nme in enumerate(names):
        ok = np.isfinite(per[:, j]) & np.isfinite(per0[:, j])
        out[nme] = float(1.0 - per[ok, j].sum() / max(per0[ok, j].sum(), 1e-300)) if ok.any() else float("nan")
    return out


# ----------------------------------------------------------------------------- the whole fit
def fit(config, data, progress=None, history_path=None) -> TrainResult:
    """Train an autoencoder: ``config`` (a :class:`Config` or its dict), ``data`` (a ``core.data.Prepared`` or a :class:`TrainData`), ``progress(fraction, message)``
    (may raise to cancel), ``history_path`` (a JSON-lines file with one record per epoch, see :func:`read_history`).  Returns a :class:`TrainResult`.

    Protocol (SPEC section 4): at least 30 samples: a random hold-out split, ``n_restarts`` random starts, the start with the best validation loss is kept (early stopping, the best weights
    restored).  Fewer: k-fold validation (every sample is predicted by a network that did not see it: the honest numbers), then ``n_restarts`` networks on ALL samples for the number of
    epochs the folds found; the one with the lowest training loss is kept."""
    cfg = config if isinstance(config, Config) else Config.from_dict(config)
    td = data.train_data(cfg.smoothify_metric, cfg.gap_weighting) if hasattr(data, "train_data") else data
    cfg.check_samples(td.n)
    if cfg.latent_dim >= td.P:
        raise AutoencoderError(f"The number of codes ({cfg.latent_dim}) must be smaller than the number of inputs ({td.P}).")
    hist = HistoryWriter(history_path)
    try:
        return _fit(cfg, td, progress, hist)
    finally:
        hist.close()


def _fit(cfg: Config, td: TrainData, progress, hist: HistoryWriter) -> TrainResult:
    t0 = time.time()
    torch = _torch()
    from .advice import count_parameters
    nthreads = configure_torch(cfg, count_parameters(td.P, cfg.latent_dim, cfg.hidden_widths(), cfg.family, cfg.skip))
    n = td.n
    split = make_split(n, cfg)
    n_eff = int(cfg.vae["n_eff"]) if (cfg.family == "vae" and cfg.vae["n_eff"] != "auto") else (auto_n_eff(td, cfg.latent_dim) if cfg.family == "vae" else 1)
    R = cfg.n_restarts
    kfold = split["mode"] == "kfold"
    F = len(split["folds"])
    n_runs = (F + R) if kfold else R
    done_runs = [0]

    def prog(msg: str, frac_in_run: float = 0.0):
        if progress is not None:
            progress(min((done_runs[0] + frac_in_run) / n_runs, 0.999), msg)

    def make_cb(tag: dict, label: str, epochs: int):
        def cb(rec):
            hist.write(rec)
            if progress is not None and (rec["epoch"] % 5 == 0 or rec["epoch"] + 1 == epochs):
                prog(f"{label}, epoch {rec['epoch'] + 1}", min((rec["epoch"] + 1) / max(epochs, 1), 1.0))
        return cb

    hist.write({"event": "start", "mode": split["mode"], "n": n, "n_train": len(split["train"]), "n_val": len(split["val"]), "folds": F, "restarts": R, "epochs": cfg.epochs,
                "family": cfg.family, "latent_dim": cfg.latent_dim, "threads": nthreads})
    notes: list[str] = []
    if td.sw is not None:
        k = int((np.asarray(td.sw) < 0.995 * float(np.max(td.sw))).sum())
        notes.append(f"{k} of the {n} samples lack part of the domain: their coefficients were estimated, not measured, so they count in proportion to the share that was measured "
                     "(the setting 'Weigh samples that lack part of the domain' in the advanced network settings changes that).")
    spec_blocks = [dict(b) for b in td.blocks]
    all_hist: list = []
    fold_info: list = []
    pca_k = cfg.latent_dim
    err_oof = np.full(n, np.nan)
    null_all = np.full(n, np.nan)
    fold_of = np.full(n, -1, dtype=int)
    pca_err_oof = np.full(n, np.nan)
    per_oof = np.full((n, len(td.blocks)), np.nan)
    per0_oof = np.full((n, len(td.blocks)), np.nan)
    kl_oof = np.full(n, np.nan)

    # ------------------------------------------------------------ k-fold (few samples): the honest numbers, and the number of epochs
    if kfold:
        seeds_f = _seeds(cfg, F, 10)
        best_epochs = []
        for f, vi in enumerate(split["folds"]):
            tr_i = np.setdiff1d(np.arange(n), vi)
            tag = {"restart": 0, "fold": f}
            hist.write({"event": "run_start", **tag, "n_train": int(tr_i.size), "n_val": len(vi)})
            label = f"Checking on held-out samples, fold {f + 1} of {F}"
            prog(label, 0.0)
            run = _train_run(td, tr_i, vi, cfg, n_eff, seeds_f[f], cfg.epochs, make_cb(tag, label, cfg.epochs), tag)
            arrays = _export_arrays(run, td, cfg)
            spec = ModelSpec.from_config(cfg, td.P, spec_blocks, None if run.sigma2 is None else math.sqrt(run.sigma2), n_eff if cfg.family == "vae" else None)
            nm = _numbers(spec, arrays, td, vi)
            err_oof[vi], per_oof[vi], per0_oof[vi], null_all[vi], fold_of[vi] = nm["error"], nm["error_blocks"], _null_blocks(spec, arrays, td, vi), nm["null"], f
            if "kl" in nm:
                kl_oof[vi] = nm["kl"]
            pm = lin.fit_pca(td.X[tr_i], None if td.M is None else td.M[tr_i], td.omega, pca_k, sample_w=td.sw_of(tr_i))
            pca_err_oof[vi] = lin.pca_errors(pm, td.X[vi], None if td.M is None else td.M[vi], td.omega, td.blocks, [pca_k])[pca_k]
            best_epochs.append(run.best_epoch + 1)
            fold_info.append({"fold": f, "best_epoch": run.best_epoch, "epochs_run": run.epochs_run, "val_error": float(nm["error"].mean()), "evr": evr(nm["error"], nm["null"]), "stopped": run.stopped})
            all_hist.append({"restart": 0, "fold": f, "history": [_light(r) for r in run.history]})
            hist.write({"event": "run_end", **tag, "best_epoch": run.best_epoch, "epochs_run": run.epochs_run, "val": run.val_loss})
            done_runs[0] += 1
        E = int(max(cfg.final_epochs_min, round(float(np.median(best_epochs)))))
        if cfg.family == "vae":
            E = int(max(E, cfg.warmup_epochs() + 10))
        if cfg.final_epochs == "max":
            E, why = int(cfg.epochs), "the maximum number of epochs, as chosen"
        elif cfg.final_epochs != "auto":
            E, why = int(cfg.final_epochs), "the number of epochs chosen"
        else:
            why = "the number the folds found"
        notes.append(f"With {n} samples the network was checked by {F}-fold validation: every sample was rebuilt by a network that had not seen it. The final network was trained on all samples for {E} epochs, {why}.")
    else:
        E = cfg.epochs
    # ------------------------------------------------------------ the random starts
    seeds_r = _seeds(cfg, R, 20)
    runs: list[_Run] = []
    tr_idx = np.arange(n) if kfold else np.asarray(split["train"])
    va_idx = None if kfold else np.asarray(split["val"])
    for r in range(R):
        tag = {"restart": r, "fold": None}
        hist.write({"event": "run_start", **tag, "n_train": int(tr_idx.size), "n_val": 0 if va_idx is None else int(va_idx.size)})
        label = f"Training, start {r + 1} of {R}"
        prog(label, 0.0)
        run = _train_run(td, tr_idx, va_idx, cfg, n_eff, seeds_r[r], E, make_cb(tag, label, E), tag, pca_start=(r == 0))
        runs.append(run)
        all_hist.append({"restart": r, "fold": None, "history": [_light(h) for h in run.history]})
        hist.write({"event": "run_end", **tag, "best_epoch": run.best_epoch, "epochs_run": run.epochs_run, "val": run.val_loss if not kfold else None, "train": run.train_eval["loss"]})
        done_runs[0] += 1
    key = (lambda ru: ru.train_eval["loss"]) if kfold else (lambda ru: ru.val_loss)
    kept = int(np.argmin([key(ru) for ru in runs]))
    best = runs[kept]
    arrays = _export_arrays(best, td, cfg)
    spec = ModelSpec.from_config(cfg, td.P, spec_blocks, None if best.sigma2 is None else math.sqrt(best.sigma2), n_eff if cfg.family == "vae" else None)
    arrays = spec.check_arrays(arrays)
    names = [b["name"] for b in td.blocks]
    allnum = _numbers(spec, arrays, td, np.arange(n))
    per0_all = _null_blocks(spec, arrays, td, np.arange(n))
    # restart summary and codes
    rcodes = []
    rsum = []
    for r, ru in enumerate(runs):
        a_r = _export_arrays(ru, td, cfg)
        s_r = ModelSpec.from_config(cfg, td.P, spec_blocks, None if ru.sigma2 is None else math.sqrt(ru.sigma2), n_eff if cfg.family == "vae" else None)
        nm = _numbers(s_r, a_r, td, np.arange(n))
        rcodes.append(nm["codes"])
        vi = np.asarray(split["val"], int)
        row = {"restart": r, "seed": int(ru.seed), "best_epoch": int(ru.best_epoch), "epochs_run": int(ru.epochs_run), "stopped": ru.stopped, "chosen": r == kept,
               "train_loss": float(ru.train_eval["loss"]), "evr_train": evr(nm["error"][tr_idx] if not kfold else nm["error"], nm["null"][tr_idx] if not kfold else nm["null"])}
        if not kfold:
            row["val_loss"] = float(ru.val_loss)
            row["evr_val"] = evr(nm["error"][vi], nm["null"][vi])
        rsum.append(row)
    key_name = "evr_val" if not kfold else "evr_train"
    vals = np.array([r_[key_name] for r_ in rsum], float)
    restarts = {"n": R, "runs": rsum, "kept": kept, "metric": key_name, "mean": float(vals.mean()), "sd": float(vals.std(ddof=1)) if R > 1 else 0.0, "min": float(vals.min()), "max": float(vals.max())}
    # validation numbers
    vae = cfg.family == "vae"
    if kfold:
        held = np.ones(n, bool)
        err_held = err_oof
        e_val, n_val = err_oof, null_all
        evr_val = evr(e_val, n_val)
        evr_blocks_val = _evr_blocks(per_oof, per0_oof, names)
        pca_val = evr(pca_err_oof, n_val)
        tr_sel = np.arange(n)
        kl_val = float(np.nanmean(kl_oof)) if vae else None
        val_idx = np.arange(n)
        fold_arr = fold_of
    else:
        vi = np.asarray(split["val"], int)
        held = np.zeros(n, bool)
        held[vi] = True
        err_held = np.full(n, np.nan)
        err_held[vi] = allnum["error"][vi]
        e_val, n_val = allnum["error"][vi], allnum["null"][vi]
        evr_val = evr(e_val, n_val)
        evr_blocks_val = _evr_blocks(allnum["error_blocks"][vi], per0_all[vi], names)
        tr_sel = np.asarray(split["train"], int)
        pm = lin.fit_pca(td.X[tr_sel], None if td.M is None else td.M[tr_sel], td.omega, pca_k, sample_w=td.sw_of(tr_sel))
        pe = lin.pca_errors(pm, td.X, td.M, td.omega, td.blocks, [pca_k])[pca_k]
        pca_val = evr(pe[vi], allnum["null"][vi])
        pca_err_oof[vi] = pe[vi]
        kl_val = float(allnum["kl"][vi].mean()) if vae else None
        val_idx = vi
        fold_arr = None
        pca_err_all = pe
    evr_train = evr(allnum["error"][tr_sel], allnum["null"][tr_sel])
    if kfold:
        pm = lin.fit_pca(td.X, td.M, td.omega, pca_k, sample_w=td.sw_of())
        pe_tr = lin.pca_errors(pm, td.X, td.M, td.omega, td.blocks, [pca_k])[pca_k]
        pca_train = evr(pe_tr, allnum["null"])
    else:
        pca_train = evr(pca_err_all[tr_sel], allnum["null"][tr_sel])
    evr_blocks_train = _evr_blocks(allnum["error_blocks"][tr_sel], per0_all[tr_sel], names)
    validation = {"mode": split["mode"], "n_val": int(len(val_idx)), "n_train": int(len(tr_sel)), "evr": evr_val, "evr_train": evr_train, "error": float(np.mean(e_val)), "error_train": float(np.mean(allnum["error"][tr_sel])),
                  "evr_blocks": {nme: {"val": evr_blocks_val[nme], "train": evr_blocks_train[nme]} for nme in names},
                  "loss_val": None if kfold else float(best.val_loss), "loss_train": float(best.train_eval["loss"]),
                  "baseline": {"name": "PCA" if all(b["kind"] == "pixels" for b in td.blocks) else "Functional PCA", "k": int(pca_k), "evr": pca_val, "evr_train": pca_train},
                  "kl": kl_val, "n_eff": int(n_eff) if vae else None, "heldout_is_out_of_fold": kfold}
    samples = {"codes": allnum["codes"], "codes_sd": None if allnum["logvar"] is None else np.exp(0.5 * allnum["logvar"]), "error": allnum["error"], "error_blocks": allnum["error_blocks"],
               "null": allnum["null"], "heldout": held, "error_heldout": err_held, "error_baseline": pca_err_oof, "fold": fold_arr, "kl": allnum.get("kl")}
    sigma = None if best.sigma2 is None else math.sqrt(best.sigma2)
    versions = {"torch": str(torch.__version__), "numpy": np.__version__, "threads": nthreads}
    try:
        import fluoviz
        versions["fluoviz"] = fluoviz.__version__
    except Exception:                                                   # noqa: BLE001
        pass
    try:
        from .. import __version__ as addon_version
        versions["addon"] = addon_version
    except Exception:                                                   # noqa: BLE001
        pass
    if cfg.n_restarts > 1 and restarts["sd"] > 0.02:
        notes.append("The random starts gave rather different results; the best was kept. See the spread of the restarts.")
    if spec.n_parameters() > 2 * n:
        notes.append(f"The network has {spec.n_parameters():,} parameters for {n} samples: it can memorise the samples, so only the validation numbers are to be trusted.")
    result = TrainResult(spec, arrays, [_light(h) for h in best.history], validation, samples, restarts, int(best.best_epoch), int(best.epochs_run), split, cfg.to_dict(), spec.n_parameters(), sigma,
                         time.time() - t0, versions, notes, np.stack(rcodes), fold_info, all_hist)
    hist.write({"event": "done", "best_epoch": result.best_epoch, "evr": evr_val, "seconds": result.seconds})
    if progress is not None:
        progress(1.0, "Done")
    return result


def _light(rec: dict) -> dict:
    return {k: (None if (isinstance(v, float) and not math.isfinite(v)) else v) for k, v in rec.items() if k != "event"}


def _null_blocks(spec: ModelSpec, arrays: dict, td: TrainData, idx) -> np.ndarray:
    idx = np.asarray(idx)
    return fw.null_error(spec, arrays, td.X[idx], None if td.M is None else td.M[idx])[1]
