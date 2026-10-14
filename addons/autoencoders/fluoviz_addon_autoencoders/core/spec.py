"""The two small objects every other part of the add-on shares: :class:`Config` (what the person chooses) and :class:`ModelSpec` (what a trained network is).

Both are plain data, with checks that say in plain English what is wrong, and a JSON round trip.  **No PyTorch here, and no import of FluoViz**: the FluoViz server and the
saved model read these objects, and the server never imports PyTorch.

``ModelSpec`` (SPEC section 7.1)::

    {"family": "ae" | "vae", "input_dim": P, "latent_dim": d,
     "encoder": {"hidden": [64, 32], "activation": "silu"}, "decoder": {"hidden": [32, 64], "activation": "silu"}, "output_activation": "linear",
     "vae": {"prior": "normal" | "mixture", "n_components": K, "beta": 1.0, "sigma": 0.12},
     "blocks": [{"name": "fluorescence", "slice": [0, P1], "kind": "gp" | "bspline" | "cosine" | "pixels", "weight": 1.0}, ...]}

and the weights (float64, PyTorch ``nn.Linear`` convention: ``W`` is ``(out, in)``) live in a flat dict of arrays named as :meth:`ModelSpec.array_shapes` says.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np

ACTIVATIONS = ("silu", "tanh", "gelu", "relu", "softplus", "elu", "linear")
SIZES = {"small": [64, 32], "medium": [128, 64], "large": [256, 128, 64]}
FAMILIES = ("ae", "vae")
BLOCK_KINDS = ("gp", "bspline", "cosine", "pixels")
PRIORS = ("normal", "mixture")
SMOOTHIFY_METRICS = ("smooth", "ridge")
MAX_LATENT = 32
MIN_SAMPLES = 12


class AutoencoderError(ValueError):
    """Something a person can fix; the message is plain English."""


# ----------------------------------------------------------------------------- small checks
def _num(v, what: str, lo=None, hi=None, integer: bool = False, allow_none: bool = False):
    if v is None and allow_none:
        return None
    if isinstance(v, bool):
        raise AutoencoderError(f"{what} must be a number.")
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise AutoencoderError(f"{what} must be a number.") from None
    if not np.isfinite(x):
        raise AutoencoderError(f"{what} must be a finite number.")
    if integer and abs(x - round(x)) > 1e-9:
        raise AutoencoderError(f"{what} must be a whole number.")
    if (lo is not None and x < lo) or (hi is not None and x > hi):
        if lo is not None and hi is not None:
            raise AutoencoderError(f"{what} must be between {lo:g} and {hi:g}.")
        raise AutoencoderError(f"{what} must be {'at least ' + format(lo, 'g') if lo is not None else 'at most ' + format(hi, 'g')}.")
    return int(round(x)) if integer else x


def _choice(v, what: str, options) -> str:
    if v not in options:
        raise AutoencoderError(f"{what} must be one of: {', '.join(options)}.")
    return v


# ----------------------------------------------------------------------------- the person's choices
_VAE_DEFAULTS = {"beta": 1.0, "warmup": None, "free_bits": 0.0, "sigma": "learned", "prior": "normal", "n_components": 3, "n_eff": "auto"}
_CORRUPTION_DEFAULTS = {"noise": 0.0, "mask": 0.0}


@dataclass
class Config:
    """The options of SPEC section 2 (defaults in bold there).  ``Config.from_dict`` refuses unknown keys; every value is checked in plain English."""
    family: str = "ae"
    latent_dim: int = 4
    hidden: Any = "small"                     # a preset name or a list of widths (the decoder mirrors the encoder)
    activation: str = "silu"
    dropout: float = 0.0                      # on the hidden layers of the encoder, in training only
    weight_decay: float = 1e-3                # AdamW
    corruption: dict = field(default_factory=lambda: dict(_CORRUPTION_DEFAULTS))     # denoising: Gaussian noise (in SD of each coordinate), share of inputs dropped
    vae: dict = field(default_factory=lambda: dict(_VAE_DEFAULTS))                    # beta, warmup (epochs; None = 20 %), free_bits, sigma, prior, n_components, n_eff
    smoothify: float = 0.0
    smoothify_metric: str = "smooth"
    epochs: int = 600
    patience: int = 60
    val_fraction: float = 0.2
    folds: int = 5
    lr: float = 1e-3
    batch_size: int | None = None             # None = min(32, max(8, n_train / 6))
    n_restarts: int = 3
    seed: int = 0
    input: dict = field(default_factory=dict)             # the data side: the Functional PCA vocabulary (see ``core.data.prepare``)
    skip: bool = True                         # a linear shortcut beside the encoder and decoder (initialised with the PCA of the training samples): see ModelSpec.skip
    kfold_below: int = 30                     # fewer samples than this: k-fold validation instead of one hold-out split
    allow_few_samples: bool = False           # train even below MIN_SAMPLES (the page asks the person first)
    grad_clip: float = 5.0
    threads: int | None = None                # PyTorch threads (None: min(4, cores); the environment variable FLUOVIZ_BLAS_THREADS wins)

    def __post_init__(self):
        self.validate()

    # ------------------------------------------------------------------ checks
    def validate(self) -> "Config":
        self.family = _choice(self.family, "The family", FAMILIES)
        self.latent_dim = _num(self.latent_dim, "The number of codes (latent dimension)", 1, MAX_LATENT, True)
        if isinstance(self.hidden, str):
            if self.hidden not in SIZES:
                raise AutoencoderError(f"The network size must be one of: {', '.join(SIZES)}, or a list of layer widths.")
        else:
            try:
                w = list(self.hidden)
            except TypeError:
                raise AutoencoderError("The hidden layers must be a size (small, medium, large) or a list of widths.") from None
            if len(w) > 6:
                raise AutoencoderError("At most 6 hidden layers are allowed.")
            self.hidden = [_num(x, "A hidden layer width", 1, 4096, True) for x in w]
        self.activation = _choice(self.activation, "The activation", ACTIVATIONS)
        self.dropout = _num(self.dropout, "The dropout", 0.0, 0.9)
        self.weight_decay = _num(self.weight_decay, "The weight decay", 0.0, 1.0)
        c = dict(self.corruption or {})
        bad = set(c) - set(_CORRUPTION_DEFAULTS)
        if bad:
            raise AutoencoderError(f"Unknown denoising setting '{sorted(bad)[0]}': use noise and mask.")
        c = {**_CORRUPTION_DEFAULTS, **c}
        self.corruption = {"noise": _num(c["noise"], "The denoising noise", 0.0, 5.0), "mask": _num(c["mask"], "The share of inputs dropped", 0.0, 0.95)}
        v = dict(self.vae or {})
        bad = set(v) - set(_VAE_DEFAULTS)
        if bad:
            raise AutoencoderError(f"Unknown variational autoencoder setting '{sorted(bad)[0]}'.")
        v = {**_VAE_DEFAULTS, **v}
        v["beta"] = _num(v["beta"], "beta (the weight of the KL term)", 0.0, 1000.0)
        v["warmup"] = _num(v["warmup"], "The KL warm-up (epochs)", 0, 100000, True, allow_none=True)
        v["free_bits"] = _num(v["free_bits"], "The free bits", 0.0, 10.0)
        if v["sigma"] != "learned":
            v["sigma"] = _num(v["sigma"], "The noise level sigma ('learned' or a number)", 1e-4, 100.0)
        v["prior"] = _choice(v["prior"], "The prior", PRIORS)
        v["n_components"] = _num(v["n_components"], "The number of clusters of the mixture prior", 2, 50, True)
        if v["n_eff"] != "auto":
            v["n_eff"] = _num(v["n_eff"], "The effective number of coordinates ('auto' or a number)", 1, 1e7, True)
        self.vae = v
        self.smoothify = _num(self.smoothify, "smoothify", 0.0, 1e6)
        self.smoothify_metric = _choice(self.smoothify_metric, "smoothify_metric", SMOOTHIFY_METRICS)
        self.epochs = _num(self.epochs, "The number of epochs", 1, 100000, True)
        self.patience = _num(self.patience, "The patience (epochs without improvement)", 1, 100000, True)
        self.val_fraction = _num(self.val_fraction, "The share of samples held out for validation", 0.05, 0.6)
        self.folds = _num(self.folds, "The number of folds", 2, 20, True)
        self.lr = _num(self.lr, "The learning rate", 1e-6, 1.0)
        self.batch_size = _num(self.batch_size, "The batch size", 2, 100000, True, allow_none=True)
        self.n_restarts = _num(self.n_restarts, "The number of random starts", 1, 50, True)
        self.seed = _num(self.seed, "The seed", 0, 2 ** 31 - 1, True)
        self.kfold_below = _num(self.kfold_below, "The sample size below which k-fold validation is used", 0, 100000, True)
        self.grad_clip = _num(self.grad_clip, "The gradient clipping", 0.0, 1e6)
        self.threads = _num(self.threads, "The number of threads", 1, 256, True, allow_none=True)
        self.allow_few_samples = bool(self.allow_few_samples)
        self.skip = bool(self.skip)
        if not isinstance(self.input, dict):
            raise AutoencoderError("The input settings must be a dictionary (variables, transform, representation, smoothing).")
        return self

    # ------------------------------------------------------------------ derived values
    def hidden_widths(self) -> list[int]:
        return list(SIZES[self.hidden]) if isinstance(self.hidden, str) else [int(w) for w in self.hidden]

    def decoder_widths(self) -> list[int]:
        return self.hidden_widths()[::-1]

    def warmup_epochs(self) -> int:
        """Epochs over which the KL weight rises from 0 to beta (default: 20 % of the epochs; 0 for a plain autoencoder)."""
        if self.family != "vae":
            return 0
        w = self.vae["warmup"]
        return int(round(0.2 * self.epochs)) if w is None else int(min(w, self.epochs))

    def batch_for(self, n_train: int) -> int:
        """The batch size: the one chosen, or by default ``min(32, max(8, n_train / 6))`` (at least about six optimiser steps per epoch: with tens of samples a batch of 32 is only one or two
        steps and the network is slow to learn); never more than the samples."""
        b = self.batch_size or int(min(32, max(8, n_train // 6)))
        return int(min(b, max(n_train, 2)))

    def check_samples(self, n: int) -> None:
        """Refuse to train on too few samples (never silently)."""
        if n < MIN_SAMPLES and not self.allow_few_samples:
            raise AutoencoderError(f"Only {n} samples are included: an autoencoder needs at least {MIN_SAMPLES} samples to be trained honestly. "
                                   "Add more samples, or use the PCA / functional PCA pages for very small data sets.")
        if n < 4:
            raise AutoencoderError(f"Only {n} samples are included: at least 4 are needed even to try.")

    # ------------------------------------------------------------------ copies and JSON
    def replace(self, **changes) -> "Config":
        """A copy with some settings changed (nested ``vae`` / ``corruption`` dictionaries are merged, not replaced)."""
        d = self.to_dict()
        for k, v in changes.items():
            if k in ("vae", "corruption") and isinstance(v, dict):
                d[k] = {**d[k], **v}
            else:
                d[k] = v
        return Config.from_dict(d)

    def to_dict(self) -> dict:
        d = {f.name: copy.deepcopy(getattr(self, f.name)) for f in fields(self)}
        return d

    @classmethod
    def from_dict(cls, d: dict | None) -> "Config":
        d = dict(d or {})
        known = {f.name for f in fields(cls)}
        bad = set(d) - known
        if bad:
            raise AutoencoderError(f"Unknown setting '{sorted(bad)[0]}'.")
        return cls(**d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), allow_nan=False)

    @classmethod
    def from_json(cls, s: str) -> "Config":
        return cls.from_dict(json.loads(s))


# ----------------------------------------------------------------------------- the network
@dataclass
class ModelSpec:
    """What a trained network is (see the module docstring); the weights are held separately as a flat dict of float64 arrays."""
    family: str
    input_dim: int
    latent_dim: int
    encoder: dict                              # {"hidden": [...], "activation": "silu"}
    decoder: dict
    output_activation: str = "linear"
    vae: dict = field(default_factory=lambda: {"prior": "normal", "n_components": 1, "beta": 1.0, "sigma": None})
    blocks: list = field(default_factory=list)          # [{"name", "slice": [a, b], "kind", "weight"}]
    skip: bool = False                                   # linear shortcuts: code = MLP(u) + enc.lin.W u  and  output = MLP(z) + dec.lin.W z (training starts from the PCA solution)

    def __post_init__(self):
        self.validate()

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_config(cls, config: Config, input_dim: int, blocks: list[dict], sigma: float | None = None, n_eff: int | None = None) -> "ModelSpec":
        enc = config.hidden_widths()
        vae = {"prior": config.vae["prior"] if config.family == "vae" else "normal", "n_components": int(config.vae["n_components"]) if config.family == "vae" and config.vae["prior"] == "mixture" else 1,
               "beta": float(config.vae["beta"]) if config.family == "vae" else 1.0, "sigma": sigma}
        if n_eff is not None:
            vae["n_eff"] = int(n_eff)                                    # N of the Gaussian likelihood (train.py): needed for the ELBO of a sample
        return cls(config.family, int(input_dim), int(config.latent_dim), {"hidden": enc, "activation": config.activation},
                   {"hidden": enc[::-1], "activation": config.activation}, "linear", vae, [dict(b) for b in blocks], bool(config.skip))

    # ------------------------------------------------------------------ checks
    def validate(self) -> "ModelSpec":
        self.family = _choice(self.family, "The family", FAMILIES)
        self.input_dim = _num(self.input_dim, "The number of inputs", 1, 10 ** 8, True)
        self.latent_dim = _num(self.latent_dim, "The number of codes", 1, 1024, True)
        for side in ("encoder", "decoder"):
            e = getattr(self, side)
            if not isinstance(e, dict) or "hidden" not in e:
                raise AutoencoderError(f"The {side} description is incomplete.")
            e["hidden"] = [_num(w, "A hidden layer width", 1, 100000, True) for w in e["hidden"]]
            e["activation"] = _choice(e.get("activation", "silu"), "The activation", ACTIVATIONS)
        self.output_activation = _choice(self.output_activation, "The output activation", ACTIVATIONS)
        self.skip = bool(self.skip)
        v = dict(self.vae or {})
        v["prior"] = _choice(v.get("prior", "normal"), "The prior", PRIORS)
        v["n_components"] = _num(v.get("n_components", 1), "The number of mixture components", 1, 1000, True)
        v["beta"] = _num(v.get("beta", 1.0), "beta", 0.0, 1e6)
        v["sigma"] = _num(v.get("sigma"), "sigma", 0.0, 1e6, allow_none=True)
        if v.get("n_eff") is not None:
            v["n_eff"] = _num(v["n_eff"], "n_eff", 1, 1e9, True)
        if v["prior"] == "mixture" and v["n_components"] < 2:
            raise AutoencoderError("A mixture prior needs at least 2 components.")
        self.vae = v
        end = 0
        for b in self.blocks:
            if set(b) - {"name", "slice", "kind", "weight"} or not {"name", "slice", "kind"} <= set(b):
                raise AutoencoderError("Every block needs a name, a slice and a kind.")
            a, c = (int(x) for x in b["slice"])
            if a != end or c <= a:
                raise AutoencoderError("The blocks must follow each other without gaps.")
            _choice(b["kind"], "A block kind", BLOCK_KINDS)
            b["slice"] = [a, c]
            b["weight"] = _num(b.get("weight", 1.0), "A block weight", 0.0, 1e9)
            end = c
        if self.blocks and end != self.input_dim:
            raise AutoencoderError(f"The blocks cover {end} inputs but the network has {self.input_dim}.")
        return self

    # ------------------------------------------------------------------ arrays
    def is_vae(self) -> bool:
        return self.family == "vae"

    def has_mixture(self) -> bool:
        return self.is_vae() and self.vae["prior"] == "mixture"

    def array_shapes(self) -> dict[str, tuple]:
        """Name -> shape of every array of the network (see SPEC 7.1)."""
        P, d = self.input_dim, self.latent_dim
        s: dict[str, tuple] = {}
        dims = [P] + list(self.encoder["hidden"])
        for i in range(len(dims) - 1):
            s[f"enc.W{i}"], s[f"enc.b{i}"] = (dims[i + 1], dims[i]), (dims[i + 1],)
        s["enc.mu.W"], s["enc.mu.b"] = (d, dims[-1]), (d,)
        if self.is_vae():
            s["enc.logvar.W"], s["enc.logvar.b"] = (d, dims[-1]), (d,)
        dims = [d] + list(self.decoder["hidden"])
        for i in range(len(dims) - 1):
            s[f"dec.W{i}"], s[f"dec.b{i}"] = (dims[i + 1], dims[i]), (dims[i + 1],)
        s["dec.out.W"], s["dec.out.b"] = (P, dims[-1]), (P,)
        if self.skip:
            s["enc.lin.W"], s["dec.lin.W"] = (d, P), (P, d)
        if self.has_mixture():
            K = self.vae["n_components"]
            s["prior.pi"], s["prior.mu"], s["prior.logvar"] = (K,), (K, d), (K, d)
        s["in.mean"], s["in.scale"], s["out.weights"] = (P,), (P,), (P,)
        return s

    def n_parameters(self) -> int:
        """Trainable weights (everything except the fixed ``in.*`` and ``out.weights``; the mixture prior counts)."""
        return int(sum(int(np.prod(v)) for k, v in self.array_shapes().items() if not (k.startswith("in.") or k == "out.weights")))

    def check_arrays(self, arrays: dict) -> dict[str, np.ndarray]:
        """The arrays as float64, after checking names and shapes (``AutoencoderError`` in plain English)."""
        want = self.array_shapes()
        missing = [k for k in want if k not in arrays]
        if missing:
            raise AutoencoderError(f"The network is incomplete: the array '{missing[0]}' is missing.")
        extra = [k for k in arrays if k not in want]
        if extra:
            raise AutoencoderError(f"The network has an unexpected array '{extra[0]}'.")
        out = {}
        for k, shp in want.items():
            a = np.asarray(arrays[k], dtype=float)
            if a.shape != shp:
                raise AutoencoderError(f"The array '{k}' has shape {a.shape} but the network needs {shp}.")
            if not np.isfinite(a).all():
                raise AutoencoderError(f"The array '{k}' contains values that are not numbers.")
            out[k] = a
        return out

    # ------------------------------------------------------------------ JSON
    def to_dict(self) -> dict:
        return {"family": self.family, "input_dim": int(self.input_dim), "latent_dim": int(self.latent_dim),
                "encoder": {"hidden": [int(w) for w in self.encoder["hidden"]], "activation": self.encoder["activation"]},
                "decoder": {"hidden": [int(w) for w in self.decoder["hidden"]], "activation": self.decoder["activation"]},
                "output_activation": self.output_activation, "vae": dict(self.vae), "skip": bool(self.skip),
                "blocks": [{"name": b["name"], "slice": [int(b["slice"][0]), int(b["slice"][1])], "kind": b["kind"], "weight": float(b.get("weight", 1.0))} for b in self.blocks]}

    @classmethod
    def from_dict(cls, d: dict) -> "ModelSpec":
        d = dict(d)
        try:
            return cls(d["family"], d["input_dim"], d["latent_dim"], dict(d["encoder"]), dict(d["decoder"]), d.get("output_activation", "linear"), dict(d.get("vae") or {}),
                       [dict(b) for b in d.get("blocks", [])], bool(d.get("skip", False)))
        except KeyError as exc:
            raise AutoencoderError(f"The network description lacks '{exc.args[0]}'.") from None

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), allow_nan=False)

    @classmethod
    def from_json(cls, s: str) -> "ModelSpec":
        return cls.from_dict(json.loads(s))

    def block(self, name: str) -> dict:
        for b in self.blocks:
            if b["name"] == name:
                return b
        raise KeyError(name)

    def block_slices(self) -> dict[str, slice]:
        return {b["name"]: slice(int(b["slice"][0]), int(b["slice"][1])) for b in self.blocks}


# ----------------------------------------------------------------------------- packing (one .npz with the JSON inside)
def pack_model(spec: ModelSpec, arrays: dict, extra: dict | None = None) -> dict[str, np.ndarray]:
    """``{"spec_json": ..., "extra_json": ..., <arrays>}`` ready for ``np.savez`` (no pickles: the loader uses ``allow_pickle=False``)."""
    packed: dict[str, np.ndarray] = {"spec_json": np.array(spec.to_json()), "extra_json": np.array(json.dumps(extra or {}, allow_nan=False, default=_json_default))}
    packed.update({f"a.{k}": np.asarray(v, float) for k, v in spec.check_arrays(arrays).items()})
    return packed


def unpack_model(npz) -> tuple[ModelSpec, dict[str, np.ndarray], dict]:
    spec = ModelSpec.from_json(str(npz["spec_json"]))
    arrays = {k[2:]: np.asarray(npz[k], float) for k in npz.files if k.startswith("a.")}
    return spec, spec.check_arrays(arrays), json.loads(str(npz["extra_json"]))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"cannot store {type(o).__name__}")
