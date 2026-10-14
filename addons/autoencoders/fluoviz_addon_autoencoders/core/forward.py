"""The trained network in pure numpy: encode, decode, reconstruct, derivatives.  **This module never imports PyTorch** (the FluoViz server uses it for everything it does with a network).

Conventions (SPEC section 7.1; ``W`` has shape ``(out, in)`` as in ``torch.nn.Linear``)::

    u    = (x - in.mean) / in.scale                  the network input; coordinates that are not observed (NaN, or ``M`` False) are set to ``in.mean`` (u = 0)
    h    = act(W_i h + b_i)                          hidden layers of the encoder (``enc.W0, enc.b0, ...``)
    mu   = enc.mu.W h + enc.mu.b                     the code (AE) / the posterior mean (VAE);  logvar = enc.logvar.W h + enc.logvar.b   (VAE)
    h    = act(W_i z + b_i)                          hidden layers of the decoder (``dec.W0, ...``)
    o    = dec.out.W h + dec.out.b ;  x_hat = out_act(o) * in.scale + in.mean       the reconstruction, in the network coordinates

Everything is deterministic: dropout and the denoising corruption exist only in training, and a variational autoencoder's code is the posterior mean ``mu``.
"""
from __future__ import annotations

import numpy as np

from .spec import ACTIVATIONS, AutoencoderError, ModelSpec


# ----------------------------------------------------------------------------- activations (the same functions as torch.nn.functional, tested to 1e-5)
def _erf(x: np.ndarray) -> np.ndarray:
    from scipy.special import erf                                        # scipy is a dependency of FluoViz; imported once, here
    return erf(x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def _silu(x):
    return x * _sigmoid(x)


def _gelu(x):
    return 0.5 * x * (1.0 + _erf(x / np.sqrt(2.0)))


def _relu(x):
    return np.maximum(x, 0.0)


def _softplus(x):
    return np.logaddexp(0.0, x)                                          # torch switches to the identity above 20: the difference there is below 2e-9


def _elu(x):
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0.0)))


def _tanh(x):
    return np.tanh(x)


def _linear(x):
    return x


_ACT = {"silu": _silu, "tanh": _tanh, "gelu": _gelu, "relu": _relu, "softplus": _softplus, "elu": _elu, "linear": _linear}


def _d_silu(x):
    s = _sigmoid(x)
    return s * (1.0 + x * (1.0 - s))


def _d_tanh(x):
    return 1.0 - np.tanh(x) ** 2


def _d_gelu(x):
    return 0.5 * (1.0 + _erf(x / np.sqrt(2.0))) + x * np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def _d_relu(x):
    return (x > 0).astype(float)


def _d_softplus(x):
    return _sigmoid(x)


def _d_elu(x):
    return np.where(x > 0, 1.0, np.exp(np.minimum(x, 0.0)))


def _d_linear(x):
    return np.ones_like(x)


_DACT = {"silu": _d_silu, "tanh": _d_tanh, "gelu": _d_gelu, "relu": _d_relu, "softplus": _d_softplus, "elu": _d_elu, "linear": _d_linear}
assert set(_ACT) == set(ACTIVATIONS) == set(_DACT)


def activation(name: str):
    """The activation function ``name`` (``silu tanh gelu relu softplus elu linear``) as a numpy function."""
    try:
        return _ACT[name]
    except KeyError:
        raise AutoencoderError(f"Unknown activation '{name}': choose one of {', '.join(ACTIVATIONS)}.") from None


def activation_grad(name: str):
    """The derivative of :func:`activation` (analytic)."""
    try:
        return _DACT[name]
    except KeyError:
        raise AutoencoderError(f"Unknown activation '{name}': choose one of {', '.join(ACTIVATIONS)}.") from None


# ----------------------------------------------------------------------------- the network
class Network:
    """A :class:`ModelSpec` with its arrays, ready to be called many times (the arrays are checked once)."""

    def __init__(self, spec: ModelSpec, arrays: dict):
        self.spec = spec
        self.a = spec.check_arrays(arrays)
        a = self.a
        self.n_enc = len(spec.encoder["hidden"])
        self.n_dec = len(spec.decoder["hidden"])
        self.enc = [(a[f"enc.W{i}"], a[f"enc.b{i}"]) for i in range(self.n_enc)]
        self.dec = [(a[f"dec.W{i}"], a[f"dec.b{i}"]) for i in range(self.n_dec)]
        self.mu = (a["enc.mu.W"], a["enc.mu.b"])
        self.logvar = (a["enc.logvar.W"], a["enc.logvar.b"]) if spec.is_vae() else None
        self.out = (a["dec.out.W"], a["dec.out.b"])
        self.lin_enc = a["enc.lin.W"] if spec.skip else None             # linear shortcuts (SPEC deviation, see ModelSpec.skip)
        self.lin_dec = a["dec.lin.W"] if spec.skip else None
        self.act_enc, self.act_dec = activation(spec.encoder["activation"]), activation(spec.decoder["activation"])
        self.dact_dec = activation_grad(spec.decoder["activation"])
        self.act_out = activation(spec.output_activation)
        self.mean, self.scale = a["in.mean"], a["in.scale"]

    # ------------------------------------------------------------------ input
    def standardise(self, X, M=None) -> np.ndarray:
        """The network input ``u`` of coordinates ``X`` ``(n, P)`` (or ``(P,)``); not-observed values (NaN, or ``M`` False) become 0."""
        X = np.atleast_2d(np.asarray(X, float))
        if X.ndim != 2 or X.shape[1] != self.spec.input_dim:
            raise AutoencoderError(f"The network expects {self.spec.input_dim} input numbers per sample but got {X.shape[-1] if X.ndim else 0}.")
        U = (X - self.mean) / self.scale
        bad = ~np.isfinite(U)
        if M is not None:
            bad |= ~np.asarray(M, bool)
        if bad.any():
            U = np.where(bad, 0.0, U)
        return U

    # ------------------------------------------------------------------ encoder / decoder
    def encode_u(self, U: np.ndarray):
        h = U
        for W, b in self.enc:
            h = self.act_enc(h @ W.T + b)
        mu = h @ self.mu[0].T + self.mu[1]
        if self.lin_enc is not None:
            mu = mu + U @ self.lin_enc.T
        logvar = None if self.logvar is None else h @ self.logvar[0].T + self.logvar[1]
        return mu, logvar

    def encode(self, X, M=None):
        """``(mu, logvar | None)`` for coordinates ``X`` ``(n, P)``."""
        return self.encode_u(self.standardise(X, M))

    def decode(self, Z) -> np.ndarray:
        """Reconstructions (network coordinates ``(n, P)``) of codes ``Z`` ``(n, d)`` (or ``(d,)``)."""
        Z = np.atleast_2d(np.asarray(Z, float))
        if Z.shape[1] != self.spec.latent_dim:
            raise AutoencoderError(f"The network has {self.spec.latent_dim} codes but {Z.shape[1]} were given.")
        h = Z
        for W, b in self.dec:
            h = self.act_dec(h @ W.T + b)
        pre = h @ self.out[0].T + self.out[1]
        if self.lin_dec is not None:
            pre = pre + Z @ self.lin_dec.T
        return self.act_out(pre) * self.scale + self.mean

    def decode_with_trace(self, Z) -> tuple[np.ndarray, list]:
        """``decode`` that also returns the pre-activations of the decoder layers (for the analytic Jacobian of :mod:`.tools`)."""
        Z = np.atleast_2d(np.asarray(Z, float))
        h, pre = Z, []
        for W, b in self.dec:
            p = h @ W.T + b
            pre.append(p)
            h = self.act_dec(p)
        o_pre = h @ self.out[0].T + self.out[1]
        if self.lin_dec is not None:
            o_pre = o_pre + Z @ self.lin_dec.T
        return self.act_out(o_pre) * self.scale + self.mean, pre


# ----------------------------------------------------------------------------- the module-level functions (what the other parts call)
def encode(spec: ModelSpec, arrays: dict, X, M=None):
    """``(mu, logvar | None)``: the codes of coordinates ``X`` ``(n, P)`` (AE: ``mu`` is the code and ``logvar`` is None; VAE: posterior mean and log variance)."""
    return Network(spec, arrays).encode(X, M)


def decode(spec: ModelSpec, arrays: dict, Z) -> np.ndarray:
    """Reconstructions ``(n, P)`` of codes ``Z`` ``(n, d)``, in the network coordinates."""
    return Network(spec, arrays).decode(Z)


def reconstruct(spec: ModelSpec, arrays: dict, X, M=None):
    """``(X_hat, mu, logvar | None)``: encode and decode (deterministically: a VAE decodes its posterior mean)."""
    net = Network(spec, arrays)
    mu, logvar = net.encode(X, M)
    return net.decode(mu), mu, logvar


def sample_codes(mu: np.ndarray, logvar: np.ndarray, rng: np.random.Generator, n_draws: int = 1) -> np.ndarray:
    """Draws from the posterior ``N(mu, exp(logvar))``: ``(n_draws, n, d)``."""
    mu, sd = np.asarray(mu, float), np.exp(0.5 * np.asarray(logvar, float))
    return mu[None] + sd[None] * rng.standard_normal((n_draws, *mu.shape))


# ----------------------------------------------------------------------------- the error in the metric of the loss
def recon_error(spec: ModelSpec, arrays: dict, X, Xhat, M=None) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruction error of every sample in the metric of the training loss (``out.weights``): ``(total (n,), per_block (n, n_blocks))``.

    ``per_block`` is the squared L2 distance between the sample and its reconstruction divided by the total variance of the block, weighted by the block
    (``out.weights`` already carry the block weights, which add up to 1): ``total = sum(per_block)`` is 0 for a perfect reconstruction and about 1 for a sample
    as far from the reconstruction as the sample is from the mean (``1 - total`` = the share of the variation that was rebuilt).  Coordinates that are not
    observed (NaN, or ``M`` False) are left out and the error of that block is scaled up to the whole block (the share of the domain that was observed)."""
    return weighted_error(X, Xhat, np.asarray(arrays["out.weights"], float), spec.blocks or [{"slice": [0, spec.input_dim]}], M)


def weighted_error(X, Xhat, omega, blocks, M=None) -> tuple[np.ndarray, np.ndarray]:
    """:func:`recon_error` with the weights ``omega`` ``(P,)`` and the ``blocks`` (``[{"slice": [a, b]}, ...]``) given directly."""
    X, Xhat = np.atleast_2d(np.asarray(X, float)), np.atleast_2d(np.asarray(Xhat, float))
    w = np.asarray(omega, float)
    obs = np.isfinite(X)
    if M is not None:
        obs &= np.asarray(M, bool)
    e2 = np.where(obs, (np.where(obs, X, 0.0) - Xhat) ** 2, 0.0) * w
    per = np.empty((X.shape[0], len(blocks)))
    for j, b in enumerate(blocks):
        a, c = b["slice"]
        num = e2[:, a:c].sum(axis=1)
        if obs[:, a:c].all():
            per[:, j] = num
        else:
            den = (obs[:, a:c] * w[a:c]).sum(axis=1)
            per[:, j] = np.where(den > 0, num * (w[a:c].sum() / np.maximum(den, 1e-300)), np.nan)
    return per.sum(axis=1), per


def null_error(spec: ModelSpec, arrays: dict, X, M=None) -> tuple[np.ndarray, np.ndarray]:
    """The same error for the 'reconstruction' that is just the mean (``in.mean``): what the variation of every sample is, in the metric of the loss."""
    X = np.atleast_2d(np.asarray(X, float))
    return recon_error(spec, arrays, X, np.broadcast_to(np.asarray(arrays["in.mean"], float), X.shape), M)


# ----------------------------------------------------------------------------- the mixture prior
def log_normal_diag(z: np.ndarray, mu: np.ndarray, logvar: np.ndarray) -> np.ndarray:
    """log N(z; mu, diag(exp(logvar))) summed over the last axis."""
    return -0.5 * (np.log(2.0 * np.pi) + logvar + (z - mu) ** 2 / np.exp(logvar)).sum(axis=-1)


def memberships(spec: ModelSpec, arrays: dict, mu: np.ndarray, logvar: np.ndarray | None = None) -> np.ndarray:
    """Probability of every cluster of a mixture prior for codes ``mu`` ``(n, d)``: ``(n, K)`` (rows add up to 1).  The posterior variance of the sample is added to the
    variance of each cluster, so an uncertain sample is shared between neighbouring clusters."""
    if not spec.has_mixture():
        raise AutoencoderError("This network has no mixture prior, so it has no clusters.")
    mu = np.atleast_2d(np.asarray(mu, float))
    pi, cm, clv = np.asarray(arrays["prior.pi"], float), np.asarray(arrays["prior.mu"], float), np.asarray(arrays["prior.logvar"], float)
    var_c = np.exp(clv)[None]                                                                        # (1, K, d)
    var = var_c + (0.0 if logvar is None else np.exp(np.asarray(logvar, float))[:, None, :])
    lp = np.log(np.maximum(pi, 1e-300))[None] - 0.5 * (np.log(2.0 * np.pi * var) + (mu[:, None, :] - cm[None]) ** 2 / var).sum(axis=-1)
    lp -= lp.max(axis=1, keepdims=True)
    p = np.exp(lp)
    return p / p.sum(axis=1, keepdims=True)


def prior_log_density(spec: ModelSpec, arrays: dict, z: np.ndarray) -> np.ndarray:
    """log p(z) of codes ``z`` ``(n, d)`` under the prior (standard normal, or the mixture)."""
    z = np.atleast_2d(np.asarray(z, float))
    if not spec.has_mixture():
        return log_normal_diag(z, 0.0, 0.0)
    pi, cm, clv = np.asarray(arrays["prior.pi"], float), np.asarray(arrays["prior.mu"], float), np.asarray(arrays["prior.logvar"], float)
    lp = np.log(np.maximum(pi, 1e-300))[None] + log_normal_diag(z[:, None, :], cm[None], clv[None])
    m = lp.max(axis=1, keepdims=True)
    return (m + np.log(np.exp(lp - m).sum(axis=1, keepdims=True)))[:, 0]
