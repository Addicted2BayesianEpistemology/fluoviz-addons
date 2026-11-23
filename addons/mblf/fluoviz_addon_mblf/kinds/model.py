"""A fitted multimodal latent factor model as a numpy object that reads NEW samples (SPEC section 7).

``meta["model"]`` of the saved kind (and ``payload["frozen"]["model"]`` of a result) is a plain tree::

    {"version": 1, "modes": [Mode], "cov": Covariates | None, "draws": Draws, "Y": [(M, N) float32], "ids": [...], "reference": {...}}

A new sample may have ANY of the modes (an absorbance spectrum without an EEM, an EEM without the lab values ...): what it has is read exactly like the training samples (the
pre-treatment, the cleaning of the absorbance, the mean and the scale of the training samples: nothing is estimated from the new samples), and what it lacks is predicted.  Its
covariates are read from the metadata when the model has covariates of the metadata (``meta:...``); a sample without them is predicted with the covariates unknown (the scores then
have the spread of the whole population), and the description says so.  Pure numpy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fluoviz.analysis.frozen import FrozenError
from fluoviz.analysis.frozen.kinds import _decomp as D
from fluoviz.analysis.frozen.kinds.fpca import ElementRuntime, clean_absorbance, pretreat
from fluoviz_addon_common.kinds import check_grid as _check_grid

from ..core import loo as LOO
from ..core import predict as P
from ..core.posterior import Fitted, thin

APPLY_DRAWS = 400
FORMAT = 1


def model_tree(f: Fitted, reference: dict) -> dict:
    """``meta["model"]`` from a fit (the training data are stored as float32: they serve the views of the result, never the predictions of new samples)."""
    return {"version": FORMAT, "modes": list(f.modes), "cov": f.cov, "draws": f.draws, "Y": [np.asarray(y, np.float32) for y in f.Y], "ids": [str(i) for i in f.ids],
            "reference": reference}


@dataclass
class Evaluation:
    n: int
    usable: np.ndarray                  # (n,) at least one mode could be read
    why: list
    Y: dict                             # {mode index: (M, n) standardised, NaN = not measured}
    present: np.ndarray                 # (n, D) the mode was measured
    coverage: np.ndarray                # (n, D) share of the domain measured (0 when absent)
    X: np.ndarray | None                # (n, L) design, NaN rows = covariates unknown; None without covariates
    scores: np.ndarray                  # (n, K) aligned posterior mean of the scores
    scores_sd: np.ndarray               # (n, K)
    surprise: np.ndarray                # (n,) log predictive density per measured value (higher = more typical)
    notes: list = field(default_factory=list)


class MblfModel:
    def __init__(self, tree: dict):
        if not isinstance(tree, dict) or not {"modes", "draws"} <= set(tree):
            raise FrozenError("This model file is damaged: its model is missing.")
        try:
            self.modes = list(tree["modes"])
            self.cov = tree.get("cov")
            self.draws = tree["draws"]
            self.Y = [np.asarray(y, float) for y in (tree.get("Y") or [])]
            self.ids = list(tree.get("ids") or [])
            self.reference = dict(tree.get("reference") or {})
        except (KeyError, TypeError, ValueError):
            raise FrozenError("This model file is damaged: the description of the model is incomplete.") from None
        if self.draws.lam.shape[1] != sum(m.p for m in self.modes):
            raise FrozenError("This model file is damaged: the loadings do not fit the modes.")
        self.f = Fitted(self.modes, self.cov, self.draws, self.Y, self.ids)
        self.idx = thin(self.draws.S, APPLY_DRAWS, self.draws.chains)

    @property
    def names(self) -> list[str]:
        """The modes a data recipe must deliver (fluorescence, absorbance)."""
        return [m.name for m in self.modes if m.element is not None]

    @property
    def K(self) -> int:
        return self.draws.K

    def meta_keys(self) -> list[str]:
        keys = [] if self.cov is None else [k for k in self.cov.keys if k.startswith("meta:")]
        for m in self.modes:
            if m.kind == "vector":
                keys += [k for k in m.columns if k.startswith("meta:")]
        return list(dict.fromkeys(keys))

    def to_tree(self) -> dict:
        return {"version": FORMAT, "modes": self.modes, "cov": self.cov, "draws": self.draws, "Y": [np.asarray(y, np.float32) for y in self.Y], "ids": self.ids, "reference": self.reference}

    # ------------------------------------------------------------------ reading new samples
    def read(self, data, min_coverage: float = 0.5) -> tuple[dict, np.ndarray, np.ndarray, list]:
        n = data.n
        Y, present, cover = {}, np.zeros((n, len(self.modes)), bool), np.zeros((n, len(self.modes)))
        why: list[list[str]] = [[] for _ in range(n)]
        for d, m in enumerate(self.modes):
            raw = None
            if m.name == "fluorescence" and data.X is not None:
                fe = m.element
                X = D.eem_array(data, tuple(int(np.asarray(a).size) for a in fe.axes), "latent factor")
                _check_grid(fe, (data.em, data.ex), "fluorescence")
                raw = ElementRuntime(fe).dom.take(X)
            elif m.name == "absorbance" and data.A is not None:
                fe = m.element
                A = np.asarray(data.A, float)
                if A.ndim != 2 or A.shape[1] != np.asarray(fe.axes[0]).size:
                    raise FrozenError("The absorbance of the samples is not on the wavelength grid of the model. Apply the model through its own data recipe, which puts the samples on "
                                      "the model's grid.")
                _check_grid(fe, (data.abs_wl,), "absorbance")
                raw = ElementRuntime(fe).dom.take(clean_absorbance(fe, np.where(np.isinf(A), np.nan, A)))
            elif m.kind == "vector":
                cols = []
                for k in m.columns:
                    col = k[5:] if k.startswith("meta:") else None
                    if col is not None and col in getattr(data.meta, "columns", []):
                        cols.append(np.asarray(data.meta.loc[list(data.ids), col], dtype=float))
                    else:
                        cols.append(np.full(n, np.nan))
                raw = np.column_stack(cols)
            if raw is None:
                Y[d] = np.full((m.M, n), np.nan)
                continue
            if m.element is not None:
                none = np.isfinite(raw).sum(axis=1) < (3 if m.name == "absorbance" else 1)
                Yt, ok = pretreat(m.element, ElementRuntime(m.element).dom, np.where(none[:, None], np.nan, raw))
                for i in np.flatnonzero(~ok & ~none):
                    why[i].append(f"The {m.name} of the sample cannot be pre-treated as the model requires; it was left out.")
                Yt[~ok] = np.nan
            else:
                Yt = raw
            cov = (np.isfinite(Yt) * m.w).sum(axis=1)
            low = (cov > 0) & (cov < min_coverage - 1e-12)
            for i in np.flatnonzero(low):
                why[i].append(f"Only {100 * cov[i]:.0f} % of the {m.label} points of the model were measured (at least {100 * min_coverage:.0f} % are needed); it was left out.")
            Yt[low] = np.nan
            Y[d] = m.standardise(Yt)
            present[:, d] = np.isfinite(Yt).any(axis=1)
            cover[:, d] = np.where(present[:, d], cov, 0.0)
        return Y, present, cover, why

    def design(self, data) -> np.ndarray | None:
        if self.cov is None:
            return None
        vals = {}
        for k in self.cov.keys:
            col = k[5:] if k.startswith("meta:") else None
            if col is not None and col in getattr(data.meta, "columns", []):
                vals[k] = list(data.meta.loc[list(data.ids), col])
            else:
                vals[k] = [None] * data.n
        return self.cov.design(vals)

    def evaluate(self, data, min_coverage: float = 0.5) -> Evaluation:
        Y, present, cover, why = self.read(data, min_coverage)
        n, K = data.n, self.K
        usable = present.any(axis=1)
        for i in np.flatnonzero(~usable):
            if not why[i]:
                why[i].append("The sample has none of the measurements the model reads.")
        X = self.design(data)
        scores = np.full((n, K), np.nan)
        sd = np.full((n, K), np.nan)
        surprise = np.full(n, np.nan)
        for known in (True, False):
            rows = np.flatnonzero(usable & ((np.isfinite(X).all(axis=1) if X is not None else np.ones(n, bool)) == known))
            if rows.size == 0 or (X is None and not known):
                continue
            src = {d: Y[d][:, rows] for d in range(len(self.modes))}
            Xr = X[rows] if (X is not None and known) else None
            eta, V = P.score_posterior(self.f, src, Xr, self.idx, n=rows.size)
            ea, Va = P.aligned_scores(self.f, eta, V, self.idx)
            scores[rows] = ea.mean(axis=0)
            sd[rows] = np.sqrt(np.mean(np.diagonal(Va, axis1=2, axis2=3), axis=0) + ea.var(axis=0))
            from ..vendor.mblf.data import mode_summaries
            summ = [mode_summaries(m.B, src[d]) for d, m in enumerate(self.modes)]
            ll = LOO.loglik(self.f, summ, list(range(len(self.modes))), self.idx, Xr, Xr is not None)
            nobs = sum(s[2] for s in summ)
            from scipy.special import logsumexp
            surprise[rows] = (logsumexp(ll, axis=0) - np.log(ll.shape[0])) / np.maximum(nobs, 1)
        return Evaluation(n, usable, why, Y, present, cover, X, scores, sd, surprise)

    # ------------------------------------------------------------------ predictions
    def predict(self, ev: Evaluation, i: int, t: int, noisy: bool = False) -> dict:
        """The prediction of mode ``t`` of sample ``i`` from everything else it has (and its covariates when they are known)."""
        src = {d: ev.Y[d][:, [i]] for d in range(len(self.modes)) if d != t and ev.present[i, d]}
        X = None
        if ev.X is not None and np.isfinite(ev.X[i]).all():
            X = ev.X[[i]]
        return P.predict(self.f, t, src, X, self.idx, noisy=noisy)

    def pattern_key(self, present_row) -> str:
        return "+".join(m.name for m, p in zip(self.modes, present_row) if p)
