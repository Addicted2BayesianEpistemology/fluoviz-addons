"""Functional data of one mode, and the per-subject summaries the likelihood needs."""

import numpy as np

# Singular values of the observed basis rows below this fraction of the largest are treated as zero (their data
# energy goes to the residual sum of squares).
SINGULAR_RTOL = float(np.sqrt(np.finfo(float).eps))


class FunctionalData:
    """Curves of one mode observed on a common grid.

    Parameters
    ----------
    coords : array, shape (M,) or (M, q)
        Locations of the grid (q-dimensional domains are allowed).
    data : array, shape (M, N)
        One column per subject; NaN marks a missing value.
    """

    def __init__(self, coords, data):
        coords = np.asarray(coords, dtype=float)
        self.coords = coords[:, None] if coords.ndim == 1 else coords
        self.data = np.asarray(data, dtype=float)
        if self.data.ndim != 2 or self.data.shape[0] != self.coords.shape[0]:
            raise ValueError("data must have shape (M, N), one row per location")
        self.basis = None          # basis matrix (M, p)
        self.basis_info = None     # the GPBasis object when the basis is a GP eigenbasis

    @property
    def M(self):
        return self.data.shape[0]

    @property
    def N(self):
        return self.data.shape[1]

    def set_basis(self, B):
        """Use a given basis matrix B of shape (M, p)."""
        B = np.asarray(B, dtype=float)
        if B.ndim != 2 or B.shape[0] != self.M:
            raise ValueError(f"the basis must have {self.M} rows")
        self.basis, self.basis_info = B, None
        return self

    def set_gp_basis(self, basis):
        """Fit a `mblf.bases.GPBasis` to these curves (on this grid) and use it."""
        basis.fit(self.coords, self.data)
        self.basis, self.basis_info = basis.evaluate(self.coords), basis
        return self

    def subset(self, columns):
        """The same mode restricted to some subjects (the basis is kept)."""
        out = FunctionalData(self.coords, self.data[:, columns])
        out.basis, out.basis_info = self.basis, self.basis_info
        return out

    def semivariogram(self, n_lags=15, max_range=None, min_pairs=30):
        """Pooled empirical (Matheron) semivariogram of the curves.

        Returns the bin centres, gamma (NaN for bins with fewer than ``min_pairs`` pairs) and the pair counts.
        """
        from scipy.spatial.distance import pdist

        h = pdist(self.coords)
        if max_range is None:
            max_range = 0.5 * float(h.max())
        edges = np.linspace(0.0, max_range, n_lags + 1)
        centres = 0.5 * (edges[:-1] + edges[1:])
        b = np.digitize(h, edges) - 1
        b[h == edges[-1]] = n_lags - 1
        in_range = (b >= 0) & (b < n_lags)
        iu, ju = np.triu_indices(self.M, k=1)
        total, count = np.zeros(n_lags), np.zeros(n_lags, dtype=int)
        for y in self.data.T:
            ok = ~np.isnan(y[iu]) & ~np.isnan(y[ju]) & in_range
            total += np.bincount(b[ok], weights=0.5 * (y[ju] - y[iu])[ok] ** 2, minlength=n_lags)
            count += np.bincount(b[ok], minlength=n_lags)
        gamma = np.where(count >= min_pairs, total / np.maximum(count, 1), np.nan)
        return centres, gamma, count


def mode_summaries(B, Y, rtol=SINGULAR_RTOL):
    """Least-squares summaries of every subject of one mode (Theorem 2 of the paper).

    For subject n with observed basis rows B_n (SVD U S V'), keeping the singular values above ``rtol`` times the largest:
    T_n = V S^{-1} U' y_n, rss_n = ||y_n - U U' y_n||^2 and G_n = S V' (zero-padded to p x p rows), so that
    ||y_n - B_n theta||^2 = rss_n + ||G_n (T_n - theta)||^2 for every theta. Subjects with the same missing values
    share one SVD.

    Returns
    -------
    T : (p, N), rss : (N,), n_obs : (N,), G : (N, p, p)
    """
    B = np.asarray(B, dtype=float)
    Y = np.asarray(Y, dtype=float)
    M, p = B.shape
    N = Y.shape[1]
    T, rss, G = np.zeros((p, N)), np.zeros(N), np.zeros((N, p, p))
    observed = ~np.isnan(Y)
    n_obs = observed.sum(axis=0)
    patterns, which = np.unique(observed.T, axis=0, return_inverse=True)
    which = np.asarray(which).reshape(-1)
    for g, pattern in enumerate(patterns):
        rows, cols = np.flatnonzero(pattern), np.flatnonzero(which == g)
        if rows.size == 0:
            continue
        U, s, Vt = np.linalg.svd(B[rows], full_matrices=False)
        keep = s > rtol * s[0] if s[0] > 0 else np.zeros(s.size, dtype=bool)
        U, s, Vt = U[:, keep], s[keep], Vt[keep]
        y = Y[np.ix_(rows, cols)]
        z = U.T @ y
        T[:, cols] = Vt.T @ (z / s[:, None])
        rss[cols] = np.sum((y - U @ z) ** 2, axis=0)
        G[cols, : s.size, :] = s[:, None] * Vt
    return T, rss, n_obs, G
