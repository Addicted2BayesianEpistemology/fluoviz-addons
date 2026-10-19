"""The linear baseline of the autoencoder, in exactly the metric of its loss: a **weighted PCA** of the network coordinates.

With the loss weights ``omega`` the squared error ``sum_j omega_j (x_j - xhat_j)^2`` is the squared L2 distance between functions divided by the total variance of the variable.
The best linear code of size ``k`` for this metric is the PCA of ``sqrt(omega) * (x - mean)``: for coordinates of the pixel mode it is the discretised functional PCA (the PCA of
the pixels weighted by the quadrature weights), for the coordinates of a functional mode it is the functional PCA / multivariate functional PCA of ``fluoviz.analysis.fpca`` (``Z = xi d``,
``omega`` proportional to ``d^2``).  Samples with unobserved coordinates (pixel mode) are handled by EM (the components) and least squares on the observed coordinates (the scores).

Pure numpy (no PyTorch, no FluoViz).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .forward import weighted_error


@dataclass
class PCAModel:
    mean: np.ndarray                   # (P,) mean of the observed coordinates of the training samples
    Vw: np.ndarray                     # (k_max, P) orthonormal directions in the weighted coordinates sqrt(omega) * x
    sw: np.ndarray                     # (P,) sqrt(omega)
    lam: np.ndarray                    # (k_max,) variances of the components (mean squared score)
    total: float                       # total weighted variance of the training samples

    def project(self, X: np.ndarray, M: np.ndarray | None, k: int) -> np.ndarray:
        """Scores ``(n, k)`` of samples ``X`` ``(n, P)`` on the first ``k`` components (least squares on the observed coordinates)."""
        X = np.atleast_2d(np.asarray(X, float))
        k = int(min(k, self.Vw.shape[0]))
        obs = np.isfinite(X) if M is None else (np.isfinite(X) & np.asarray(M, bool))
        Yw = np.where(obs, X - self.mean, 0.0) * self.sw
        V = self.Vw[:k]
        if obs.all():
            return Yw @ V.T
        S = np.zeros((X.shape[0], k))
        pats, inv = np.unique(obs, axis=0, return_inverse=True)
        inv = np.asarray(inv).ravel()
        for g in range(pats.shape[0]):
            rows = np.flatnonzero(inv == g)
            o = pats[g]
            Vo = V[:, o]
            A = Vo @ Vo.T + 1e-10 * np.eye(k)
            S[rows] = np.linalg.solve(A, (Yw[rows][:, o] @ Vo.T).T).T
        return S

    def reconstruct(self, X: np.ndarray, M: np.ndarray | None, k: int) -> np.ndarray:
        """Reconstructions ``(n, P)`` (network coordinates) with ``k`` components."""
        k = int(min(k, self.Vw.shape[0]))
        if k <= 0:
            return np.broadcast_to(self.mean, (np.atleast_2d(X).shape[0], self.mean.size)).copy()
        S = self.project(X, M, k)
        return self.mean + (S @ self.Vw[:k]) / self.sw

    def components(self, k: int) -> np.ndarray:
        """The first ``k`` directions in the network coordinates ``(k, P)`` (what a unit score adds to the mean)."""
        return self.Vw[:k] / self.sw


def fit_pca(X: np.ndarray, M: np.ndarray | None, omega: np.ndarray, k_max: int, n_iter: int = 40, sample_w: np.ndarray | None = None) -> PCAModel:
    """Weighted PCA of the training coordinates ``X`` ``(n, P)`` (``M``: observed mask or None) with weights ``omega`` ``(P,)``; ``k_max`` components.  ``sample_w`` ``(n,)``: the weight of every
    SAMPLE (None = all alike; the network's loss gives a sample that lacks part of the domain less weight): the mean and the components are the weighted ones."""
    X = np.asarray(X, float)
    n, P = X.shape
    omega = np.maximum(np.asarray(omega, float), 1e-300)
    sw = np.sqrt(omega)
    obs = np.ones_like(X, bool) if M is None else (np.isfinite(X) & np.asarray(M, bool))
    if sample_w is None:
        rw = None
        cnt = obs.sum(axis=0)
        mean = np.where(cnt > 0, np.where(obs, X, 0.0).sum(axis=0) / np.maximum(cnt, 1), 0.0)
    else:
        w = np.maximum(np.asarray(sample_w, float), 0.0)
        w = w / max(float(w.mean()), 1e-300)
        rw = np.sqrt(w)[:, None]
        cnt = (obs * w[:, None]).sum(axis=0)
        mean = np.where(cnt > 0, (np.where(obs, X, 0.0) * w[:, None]).sum(axis=0) / np.maximum(cnt, 1e-300), 0.0)
    k_max = int(max(1, min(k_max, n - 1, P)))
    Yw = np.where(obs, X - mean, 0.0) * sw
    if rw is not None:
        Yw = Yw * rw                                                  # the rows weighted: the covariance of the rows is the weighted covariance of the samples
    fill = Yw.copy()
    if not obs.all():                                                 # EM: the unobserved coordinates are filled with what the first k_max components say
        for _ in range(n_iter):
            U, s, Vt = np.linalg.svd(fill, full_matrices=False)
            new = np.where(obs, Yw, (U[:, :k_max] * s[:k_max]) @ Vt[:k_max])
            done = np.abs(new - fill).max() < 1e-9 * max(float(np.abs(Yw).max()), 1e-300)
            fill = new
            if done:
                break
    U, s, Vt = np.linalg.svd(fill, full_matrices=False)
    lam = s[:k_max] ** 2 / n
    # the total variance: the mean (over samples) squared weighted norm of the centred samples, the part that was observed scaled up to the whole block is not needed here
    total = float(np.mean((fill ** 2).sum(axis=1)))
    return PCAModel(mean, Vt[:k_max], sw, lam, total)


def pca_errors(model: PCAModel, X: np.ndarray, M: np.ndarray | None, omega: np.ndarray, blocks: list, ks) -> dict[int, np.ndarray]:
    """Per-sample errors (in the metric of the loss) of the PCA reconstruction with ``k`` components for every ``k`` in ``ks``: ``{k: (n,)}`` (``k = 0``: the mean)."""
    out = {}
    for k in ks:
        Xh = model.reconstruct(X, M, int(k))
        out[int(k)] = weighted_error(X, Xh, omega, blocks, M)[0]
    return out
