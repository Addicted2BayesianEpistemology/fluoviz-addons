"""The settings of a multimodal latent factor model, checked in plain English (``Config``), and the words the numerics share.  Pure Python.

The data settings (``input``: which modes, their pre-treatment, domain and basis, the values to model and the covariates) are checked by ``server/request.py`` with the function the
Functional PCA page uses; everything about the model, the priors and the sampler is here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

ESTIMATIONS = ("bayes", "ml")
ESTIMATION_WORDS = {"bayes": "full Bayesian posterior (MCMC, the No-U-Turn sampler of Stan)", "ml": "maximum likelihood (a point estimate: no uncertainty about the population parameters)"}
PRESETS = {
    "quick": {"chains": 2, "warmup": 400, "draws": 400},
    "standard": {"chains": 4, "warmup": 1000, "draws": 1000},
    "thorough": {"chains": 4, "warmup": 2000, "draws": 2000},
}
# the priors of mblf.MvtBLF, with the words of the page (``docs/README_user.md`` explains each one)
PRIORS = {
    "a_psi": (2.0, 0.01, 1000.0, "Shape of the prior of the measurement-noise variance"),
    "b_psi": (2.0, 1e-6, 1e6, "Scale of the prior of the measurement-noise variance"),
    "a_tau": (2.0, 0.01, 1000.0, "Shape of the prior of the precisions of the sample-specific residual"),
    "b_tau": (2.0, 1e-6, 1e6, "Rate of the prior of the precisions of the sample-specific residual"),
    "a1": (2.0, 0.01, 100.0, "Shape of the shrinkage of the first factor"),
    "a2": (2.0, 0.01, 100.0, "Shape of the extra shrinkage of every further factor"),
    "b_lambdavar": (1.0, 1e-6, 1e6, "Scale of the loadings"),
    "nu": (3.0, 1.0, 1000.0, "Degrees of freedom of the Student-t prior of the loadings"),
    "sigma_regr_coeffs": (10.0, 1e-3, 1e4, "Prior standard deviation of the covariate effects on the scores"),
}
MIN_SAMPLES = 10
ABSOLUTE_MIN = 5
MAX_FACTORS = 12
MAX_KEEP = 4000


class MblfError(ValueError):
    """A problem the person can fix; the message says what to do."""


def _num(d: dict, key: str, default, lo: float, hi: float, what: str, integer: bool = False):
    v = d.get(key, default)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise MblfError(f"{what} must be a number.")
    if integer and float(v) != int(v):
        raise MblfError(f"{what} must be a whole number.")
    if not lo <= float(v) <= hi:
        raise MblfError(f"{what} must be between {lo:g} and {hi:g}.")
    return int(v) if integer else float(v)


def _group(raw, name: str, allowed: tuple) -> dict:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise MblfError(f"'{name}' must be a group of settings.")
    bad = sorted(set(raw) - set(allowed))
    if bad:
        raise MblfError(f"Unknown setting '{bad[0]}' in '{name}'. Known settings: {', '.join(allowed)}.")
    return raw


@dataclass
class Config:
    K: int = 5                                   # the cap of the number of factors (the shrinkage prior switches off what the data do not need)
    estimation: str = "bayes"
    priors: dict = field(default_factory=lambda: {k: v[0] for k, v in PRIORS.items()})
    chains: int = 4
    warmup: int = 1000
    draws: int = 1000
    adapt_delta: float = 0.8
    max_treedepth: int = 12
    keep: int = 1000                             # posterior draws kept in the result (evenly thinned): every summary is computed from them
    ml_starts: int = 8
    seed: int = 1
    allow_few_samples: bool = False
    input: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, b: dict) -> "Config":
        if not isinstance(b, dict):
            raise MblfError("The settings must be sent as a group of named values.")
        top = {"input", "model", "sampler", "ml", "seed", "allow_few_samples", "title", "replace"}
        bad = sorted(set(b) - top)
        if bad:
            raise MblfError(f"Unknown setting '{bad[0]}'. Known settings: {', '.join(sorted(top - {'title', 'replace'}))}.")
        m = _group(b.get("model"), "model", ("K", "estimation", "priors"))
        s = _group(b.get("sampler"), "sampler", ("preset", "chains", "warmup", "draws", "adapt_delta", "max_treedepth", "keep"))
        ml = _group(b.get("ml"), "ml", ("starts",))
        c = cls()
        c.K = _num(m, "K", 5, 1, MAX_FACTORS, "The largest number of factors", integer=True)
        est = m.get("estimation", "bayes")
        if est not in ESTIMATIONS:
            raise MblfError("The estimation must be 'bayes' (full posterior, MCMC) or 'ml' (maximum likelihood).")
        c.estimation = est
        pr = _group(m.get("priors"), "priors", tuple(PRIORS))
        c.priors = {k: _num(pr, k, v[0], v[1], v[2], v[3]) for k, v in PRIORS.items()}
        preset = s.get("preset")
        if preset is not None and preset not in PRESETS:
            raise MblfError(f"The length of the sampling must be one of: {', '.join(PRESETS)}.")
        base = PRESETS[preset] if preset else PRESETS["standard"]
        c.chains = _num(s, "chains", base["chains"], 1, 16, "The number of chains", integer=True)
        c.warmup = _num(s, "warmup", base["warmup"], 50, 20000, "The warm-up iterations of every chain", integer=True)
        c.draws = _num(s, "draws", base["draws"], 50, 20000, "The kept iterations of every chain", integer=True)
        c.adapt_delta = _num(s, "adapt_delta", 0.8, 0.5, 0.999, "The target acceptance rate (adapt_delta)")
        c.max_treedepth = _num(s, "max_treedepth", 12, 5, 15, "The largest tree depth", integer=True)
        c.keep = _num(s, "keep", 1000, 100, MAX_KEEP, "The number of posterior draws kept", integer=True)
        c.ml_starts = _num(ml, "starts", 8, 1, 64, "The number of starting points of the maximum-likelihood fit", integer=True)
        c.seed = _num(b, "seed", 1, 0, 2 ** 31 - 1, "The seed", integer=True)
        afs = b.get("allow_few_samples", False)
        if not isinstance(afs, bool):
            raise MblfError("'allow_few_samples' must be true or false.")
        c.allow_few_samples = afs
        c.input = dict(b.get("input") or {})
        return c

    def to_dict(self) -> dict:
        return {"model": {"K": self.K, "estimation": self.estimation, "priors": dict(self.priors)},
                "sampler": {"chains": self.chains, "warmup": self.warmup, "draws": self.draws, "adapt_delta": self.adapt_delta, "max_treedepth": self.max_treedepth, "keep": self.keep},
                "ml": {"starts": self.ml_starts}, "seed": self.seed, "allow_few_samples": self.allow_few_samples, "input": self.input}

    def check_samples(self, n: int) -> None:
        if n < ABSOLUTE_MIN:
            raise MblfError(f"Only {n} samples can be used: a latent factor model needs at least {ABSOLUTE_MIN}.")
        if n < MIN_SAMPLES and not self.allow_few_samples:
            raise MblfError(f"Only {n} samples can be used: with fewer than {MIN_SAMPLES} the factors are hardly identified and the uncertainty is huge. Add samples, or switch on "
                            f"'Fit on fewer than {MIN_SAMPLES} samples anyway' (never fewer than {ABSOLUTE_MIN}).")
