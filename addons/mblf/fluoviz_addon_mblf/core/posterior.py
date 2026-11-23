"""A fitted model as numpy objects: the modes, the covariates and the posterior draws of the population parameters, with the alignment of the factors (SPEC section 4).

The model (paper, mblf ``MvtBLF``), for sample ``n`` and mode ``d`` (standardised units):

    eta_n ~ N_K(beta x_n, I),   theta_dn = Lambda_d eta_n + v_dn,  v_dn ~ N(0, diag(sig2_d)),   y_dn = B_d theta_dn + e_dn,  e_dn ~ N(0, psi2_d I).

``Draws`` holds ``S`` draws of ``(Lambda, sig2 = 1/tau, psi2, beta)`` (``S = 1`` for a maximum-likelihood fit: everything below then holds with the parameters known) and the
alignment ``T`` (``Lambda T`` are the aligned loadings, ``T' eta`` the aligned scores): the factors are only defined up to a rotation, so every draw is rotated by a varimax in the L2
geometry of the loading FUNCTIONS of all modes at once (each mode standardised, so each counts the same) and matched in sign and order to a common reference (Varimax-RSP,
Papastamoulis & Ntzoufras 2022, as in mblf); the factors are then ordered by their mean strength.  Quantities that do not depend on the rotation (predictions, shares of variance,
covariate effects, correlations) never use ``T``.

The data of the training samples (standardised, NaN = not observed) are kept with the fit (``Fitted.Y``), so every view of a result can be computed again without the project.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from fluoviz_addon_common.serial import allow as _serial_allow

from ..vendor.mblf.align import varimax_rsp


@_serial_allow
@dataclass
class Draws:
    lam: np.ndarray                  # (S, P, K) loadings, the modes stacked
    sig2: np.ndarray                 # (S, P) variances of the sample-specific residual coefficients
    psi2: np.ndarray                 # (S, D) measurement-noise variances
    beta: np.ndarray                 # (S, K, L) covariate effects on the scores
    T: np.ndarray                    # (S, K, K) the alignment
    chains: int = 1                  # the draws are chain by chain (S = chains * per chain), for R-hat and the relative efficiency of PSIS
    estimation: str = "bayes"

    @property
    def S(self) -> int:
        return int(self.lam.shape[0])

    @property
    def K(self) -> int:
        return int(self.lam.shape[2])

    @property
    def L(self) -> int:
        return int(self.beta.shape[2])

    def subset(self, idx) -> "Draws":
        idx = np.asarray(idx, int)
        per = max(self.S // max(self.chains, 1), 1)
        chains = len(set((idx // per).tolist())) if self.chains > 1 else 1
        return Draws(self.lam[idx], self.sig2[idx], self.psi2[idx], self.beta[idx], self.T[idx], chains if idx.size % max(chains, 1) == 0 else 1, self.estimation)


def thin(S: int, at_most: int | None, chains: int = 1) -> np.ndarray:
    """Indices of at most ``at_most`` draws, evenly spaced WITHIN every chain (so the chain structure is kept)."""
    if at_most is None or S <= at_most:
        return np.arange(S)
    per = S // max(chains, 1)
    k = max(1, at_most // max(chains, 1))
    within = np.unique(np.round(np.linspace(0, per - 1, k)).astype(int))
    return np.concatenate([c * per + within for c in range(max(chains, 1))])


@dataclass
class Fitted:
    """The modes, covariates, draws and the training data of a fit; the slices of every mode in the stacked parameters."""
    modes: list                      # [Mode]
    cov: object | None               # Covariates | None
    draws: Draws
    Y: list = field(default_factory=list)          # per mode (M, N) standardised training data
    ids: list = field(default_factory=list)

    @property
    def D(self) -> int:
        return len(self.modes)

    @property
    def K(self) -> int:
        return self.draws.K

    @property
    def L(self) -> int:
        return 0 if self.cov is None else self.cov.L

    @property
    def N(self) -> int:
        return int(self.Y[0].shape[1]) if self.Y else 0

    def offsets(self) -> list[slice]:
        out, a = [], 0
        for m in self.modes:
            out.append(slice(a, a + m.p))
            a += m.p
        return out

    def index(self, name: str) -> int:
        for d, m in enumerate(self.modes):
            if m.name == name:
                return d
        raise KeyError(name)

    def lam(self, d: int, aligned: bool = False, idx=None) -> np.ndarray:
        """``(S, p_d, K)`` loadings of mode ``d`` (0-based), aligned or not."""
        dr = self.draws
        L = dr.lam[:, self.offsets()[d], :] if idx is None else dr.lam[idx][:, self.offsets()[d], :]
        if aligned:
            L = L @ (dr.T if idx is None else dr.T[idx])
        return L

    def sig2(self, d: int, idx=None) -> np.ndarray:
        s = self.draws.sig2[:, self.offsets()[d]]
        return s if idx is None else s[idx]

    def psi2(self, d: int, idx=None) -> np.ndarray:
        s = self.draws.psi2[:, d]
        return s if idx is None else s[idx]

    def beta(self, idx=None, aligned: bool = False) -> np.ndarray:
        b = self.draws.beta if idx is None else self.draws.beta[idx]
        if aligned and b.shape[2]:
            T = self.draws.T if idx is None else self.draws.T[idx]
            b = np.swapaxes(T, 1, 2) @ b
        return b

    def loading_functions(self, d: int, aligned: bool = True, data_units: bool = True, idx=None) -> np.ndarray:
        """``(S, K, M)`` the loading functions of mode ``d`` (the change of the mode for one unit of a score), in data units (the scale of the mode) or standardised."""
        F = np.einsum("mp,spk->skm", self.modes[d].B, self.lam(d, aligned, idx))
        return F * self.modes[d].scale[None, None, :] if data_units else F

    def psi_cov(self, idx=None) -> np.ndarray | None:
        """``(S, K, K)`` the covariance of the scores over the population when the covariates are NOT known: ``I + beta Cov(x) beta'`` (None without covariates)."""
        if self.L == 0:
            return None
        b = self.beta(idx)
        return np.eye(self.K)[None] + b @ self.cov.cov()[None] @ np.swapaxes(b, 1, 2)


# ----------------------------------------------------------------------------- the alignment
def alignment(modes: list, lam: np.ndarray) -> tuple[np.ndarray, dict]:
    """The rotation ``T`` ``(S, K, K)`` of every draw of the stacked loadings ``lam`` ``(S, P, K)`` (Varimax-RSP of the loadings in the L2 frame of the functions; see the module text),
    with the factors ordered by their mean strength and the sign of each fixed so that its mean loading function has a positive largest entry.  Also the strengths."""
    S, P, K = lam.shape
    roots, a = [], 0
    blocks = []
    for m in modes:
        R = m.frame_root()
        blocks.append(slice(a, a + m.p))
        roots.append(R)
        a += m.p
    Z = np.concatenate([np.einsum("ij,sjk->sik", R, lam[:, sl, :]) for R, sl in zip(roots, blocks)], axis=1)
    if K == 1:
        T = np.ones((S, 1, 1))
        ref = Z.mean(axis=0)
    else:
        res = varimax_rsp(Z[None])
        T = (res["rotations"][0] @ res["signed_permutations"][0])
    Za = Z @ T
    strength = np.sum(Za ** 2, axis=1)                                       # (S, K) the variance each factor carries, summed over the (standardised) modes
    order = np.argsort(-strength.mean(axis=0), kind="stable")
    T = T[:, :, order]
    Za = Za[:, :, order]
    ref = Za.mean(axis=0)
    big = np.argmax(np.abs(ref), axis=0)
    sign = np.where(ref[big, np.arange(K)] < 0, -1.0, 1.0)
    T = T * sign[None, None, :]
    strength = strength[:, order]
    share = strength / np.where(strength.sum(axis=1, keepdims=True) > 0, strength.sum(axis=1, keepdims=True), 1.0)
    return T, {"strength": strength, "share": share}


# ----------------------------------------------------------------------------- what mblf's diagnostics expect of a fit
class _FD:
    def __init__(self, coords, data, basis):
        self.coords, self.data, self.basis, self.basis_info = coords, data, basis, None

    @property
    def M(self):
        return self.data.shape[0]

    @property
    def N(self):
        return self.data.shape[1]


class MblfView:
    """The few attributes of ``mblf.MvtBLF`` that ``mblf.secondary`` (effective dimension, projection residuals) reads, on top of a :class:`Fitted` (draws as ONE chain)."""

    def __init__(self, f: Fitted):
        self.f = f
        self.K, self.D = f.K, f.D
        self.p = [m.p for m in f.modes]
        self.X = f.cov.X if f.cov is not None else np.zeros((f.N, 0))
        self.modes = [_FD(m.coords, Y, m.B) for m, Y in zip(f.modes, f.Y)]
        dr = f.draws
        self.draws = {"Lambda": dr.lam[None], "tau": 1.0 / dr.sig2[None], "psi2": dr.psi2[None], "regr_coeffs": dr.beta[None]}

    def loadings(self, d):
        return self.f.lam(d - 1)[None]

    def coefficient_variances(self, d):
        return self.f.sig2(d - 1)[None]
