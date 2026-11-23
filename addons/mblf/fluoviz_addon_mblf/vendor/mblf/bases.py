"""Gaussian-process (kernel) eigenbases: kernels, hyperparameter estimation and the basis itself."""

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist, pdist

KERNELS = ("squared_exponential", "matern52")


def kernel_matrix(kernel, x1, x2=None, lengthscale=1.0, variance=1.0):
    """Covariance matrix k(x1, x2).

    squared_exponential: k = variance * exp(-r^2 / (2 l^2));
    matern52:            k = variance * (1 + s + s^2/3) exp(-s), s = sqrt(5) r / l.
    """
    x1 = np.asarray(x1, dtype=float).reshape(len(x1), -1)
    x2 = x1 if x2 is None else np.asarray(x2, dtype=float).reshape(len(x2), -1)
    r = cdist(x1, x2) / lengthscale
    if kernel == "squared_exponential":
        return variance * np.exp(-0.5 * r**2)
    if kernel == "matern52":
        s = np.sqrt(5.0) * r
        return variance * (1.0 + s + s**2 / 3.0) * np.exp(-s)
    raise ValueError(f"kernel must be one of {KERNELS}")


def _kernel_and_gradients(kernel, x, lengthscale, variance):
    """k(x, x) and its derivatives with respect to log(variance) and log(lengthscale)."""
    r = cdist(x, x) / lengthscale
    if kernel == "squared_exponential":
        K = variance * np.exp(-0.5 * r**2)
        return K, [K, K * r**2]
    s = np.sqrt(5.0) * r
    e = np.exp(-s)
    K = variance * (1.0 + s + s**2 / 3.0) * e
    return K, [K, variance * e * s**2 * (1.0 + s) / 3.0]


def trapezoid_weights(coords):
    """Trapezoidal-rule weights of a one-dimensional grid, normalised to mean one."""
    x = np.asarray(coords, dtype=float).reshape(len(coords), -1)
    if x.shape[1] != 1:
        raise ValueError("trapezoidal weights need one-dimensional coordinates")
    u, inv, counts = np.unique(x[:, 0], return_inverse=True, return_counts=True)
    gaps = np.diff(u)
    w = np.zeros(u.size)
    w[:-1] += 0.5 * gaps
    w[1:] += 0.5 * gaps
    inv = np.asarray(inv).reshape(-1)
    w = w[inv] / counts[inv]
    return w * (w.size / w.sum())


def gp_loglik(kernel, coords, Y, lengthscale, variance, noise, gradient=False):
    """Log-likelihood of independent curves y_n ~ N(0, K + noise I) on their observed rows (and its gradient in the
    log of lengthscale, variance and noise)."""
    from scipy.linalg import cho_factor, cho_solve

    if gradient:
        K, (dvar, dls) = _kernel_and_gradients(kernel, coords, lengthscale, variance)
    else:
        K = kernel_matrix(kernel, coords, lengthscale=lengthscale, variance=variance)
    jitter = 1e-10 * (variance + noise)
    observed = ~np.isnan(Y)
    patterns, which = np.unique(observed.T, axis=0, return_inverse=True)
    which = np.asarray(which).reshape(-1)
    ll, grad = 0.0, np.zeros(3)  # order: log lengthscale, log variance, log noise
    for g, pattern in enumerate(patterns):
        rows, cols = np.flatnonzero(pattern), np.flatnonzero(which == g)
        if rows.size == 0:
            continue
        V = K[np.ix_(rows, rows)] + (noise + jitter) * np.eye(rows.size)
        cf = cho_factor(V, lower=True, check_finite=False)
        R = Y[np.ix_(rows, cols)]
        A = cho_solve(cf, R, check_finite=False)
        logdet = 2.0 * np.sum(np.log(np.diag(cf[0])))
        ll += -0.5 * (np.sum(R * A) + cols.size * (logdet + rows.size * np.log(2 * np.pi)))
        if gradient:
            Q = A @ A.T - cols.size * cho_solve(cf, np.eye(rows.size), check_finite=False)
            grad[0] += 0.5 * np.sum(Q * dls[np.ix_(rows, rows)])
            grad[1] += 0.5 * np.sum(Q * dvar[np.ix_(rows, rows)])
            grad[2] += 0.5 * noise * np.trace(Q)
    return (ll, grad) if gradient else ll


def fit_gp(coords, Y, kernel="squared_exponential", n_starts=4, fixed_lengthscale=None):
    """Maximum marginal likelihood estimate of (lengthscale, variance, noise) from the curves of one mode.

    The curves are independent zero-mean replicates of a GP with the given kernel plus a nugget. L-BFGS-B on the
    log-hyperparameters from ``n_starts`` lengthscales spread geometrically between twice the smallest and half the
    largest distance of the grid. ``fixed_lengthscale`` holds the lengthscale fixed (variance and nugget estimated).
    Returns a dict with lengthscale, variance, noise_variance and loglik.
    """
    coords = np.asarray(coords, dtype=float).reshape(len(coords), -1)
    Y = np.asarray(Y, dtype=float)
    Y = Y[:, np.any(~np.isnan(Y), axis=0)]
    scale = float(np.nanmean(Y**2)) or 1.0
    u = np.unique(coords, axis=0)
    if u.shape[0] > 400:
        u = u[np.random.default_rng(0).choice(u.shape[0], 400, replace=False)]
    d = pdist(u)
    d_min, d_max = float(d[d > 0].min()), float(d.max())
    n_obs = int(np.sum(~np.isnan(Y)))
    bounds = [(np.log(d_min / 10), np.log(50 * d_max)), (np.log(1e-6 * scale), np.log(1e4 * scale)),
              (np.log(1e-10 * scale), np.log(10 * scale))]
    free = [fixed_lengthscale is None, True, True]
    full = np.zeros(3)
    if fixed_lengthscale is not None:
        full[0] = np.log(fixed_lengthscale)

    def unpack(x):
        theta = full.copy()
        theta[np.array(free)] = x
        return np.exp(theta)

    def objective(x):
        ls, var, noise = unpack(x)
        try:
            ll, g = gp_loglik(kernel, coords, Y, ls, var, noise, gradient=True)
        except np.linalg.LinAlgError:
            return 1e10, np.zeros_like(x)
        return -ll / n_obs, -g[np.array(free)] / n_obs

    fb = [b for b, f in zip(bounds, free) if f]
    best = None
    for ls in np.geomspace(2 * d_min, 0.5 * d_max, max(int(n_starts), 1)):
        x0 = np.log([ls, 0.9 * scale, 0.1 * scale])[np.array(free)]
        x0 = np.clip(x0, [b[0] for b in fb], [b[1] for b in fb])
        opt = minimize(objective, x0, jac=True, method="L-BFGS-B", bounds=fb, options={"maxiter": 500})
        if best is None or opt.fun < best.fun:
            best = opt
    ls, var, noise = unpack(best.x)
    return {"kernel": kernel, "lengthscale": float(ls), "variance": float(var), "noise_variance": float(noise),
            "loglik": float(-best.fun * n_obs), "converged": bool(best.success)}


class GPBasis:
    """Eigenbasis of a kernel on the grid (truncated Karhunen-Loeve expansion; Section "Basis functions" of the paper).

    With quadrature weights W, C_W = W^{1/2} C W^{1/2} = V Xi V' and the basis is B = W^{-1/2} V_p Xi_p^{1/2}; at other
    locations it is extended by Nystrom, b(t)' = k(t, grid) W^{1/2} V_p Xi_p^{-1/2}.

    Parameters
    ----------
    kernel : "squared_exponential" or "matern52"
    n_basis : int, optional
        Number of functions. With ``energy`` too, the smaller of the two.
    energy : float, optional
        Smallest p whose eigenvalues reach this fraction of the trace (default 0.99 when n_basis is None).
    weights : None, "trapezoid" or array (M,)
        Quadrature weights of the grid points (None: equal weights).
    lengthscale, variance : float
        Kernel hyperparameters (starting/fixed values).
    estimate : bool
        Estimate lengthscale, variance and nugget from the curves the basis is fitted to (`fit_gp`).
    fix_lengthscale : bool
        With ``estimate``: keep ``lengthscale`` fixed, estimate variance and nugget only.
    n_starts : int
        Starts of `fit_gp`.

    Column signs are fixed so that the entry of largest absolute value of every function is positive.
    """

    def __init__(self, kernel="matern52", n_basis=None, energy=None, weights=None, lengthscale=1.0, variance=1.0,
                 estimate=False, fix_lengthscale=False, n_starts=4, eigen_tol=1e-10):
        if kernel not in KERNELS:
            raise ValueError(f"kernel must be one of {KERNELS}")
        self.kernel, self.n_requested, self.energy, self.weights = kernel, n_basis, energy, weights
        self.lengthscale, self.variance, self.noise_variance = float(lengthscale), float(variance), None
        self.estimate, self.fix_lengthscale, self.n_starts, self.eigen_tol = estimate, fix_lengthscale, n_starts, eigen_tol
        self.n_basis = self.captured_energy = None

    def fit(self, coords, Y=None):
        coords = np.asarray(coords, dtype=float).reshape(len(coords), -1)
        if self.estimate:
            est = fit_gp(coords, Y, self.kernel, self.n_starts,
                         fixed_lengthscale=self.lengthscale if self.fix_lengthscale else None)
            self.lengthscale, self.variance, self.noise_variance = est["lengthscale"], est["variance"], est["noise_variance"]
        C = kernel_matrix(self.kernel, coords, lengthscale=self.lengthscale, variance=self.variance)
        if self.weights is None:
            w = None
        elif isinstance(self.weights, str):
            w = trapezoid_weights(coords)
        else:
            w = np.asarray(self.weights, dtype=float)
            w = w * (w.size / w.sum())
        root = None if w is None else np.sqrt(w)
        A = C if root is None else root[:, None] * C * root[None, :]
        evals, evecs = np.linalg.eigh(0.5 * (A + A.T))
        order = np.argsort(evals)[::-1]
        evals, evecs = np.clip(evals[order], 0.0, None), evecs[:, order]
        usable = int(np.sum(evals > self.eigen_tol * evals[0]))
        energy = self.energy if (self.energy is not None or self.n_requested is not None) else 0.99
        p = self.n_requested
        if energy is not None:
            p_energy = int(np.searchsorted(np.cumsum(evals) / evals.sum(), energy - 1e-12) + 1)
            p = p_energy if p is None else min(p, p_energy)
        p = min(p, usable)
        U = evecs[:, :p] if root is None else evecs[:, :p] / root[:, None]
        mag = np.abs(U)
        first = np.argmax(mag >= (1 - 1e-6) * mag.max(axis=0), axis=0)
        signs = np.sign(U[first, np.arange(p)])
        U = U * np.where(signs == 0, 1.0, signs)
        self.grid, self.grid_weights, self.U, self.S = coords, w, U, evals[:p]
        self.eigenvalues, self.n_basis = evals, p
        self.captured_energy = float(evals[:p].sum() / evals.sum())
        return self

    def evaluate(self, coords):
        """Basis at any locations: the stored values on the grid, the Nystrom extension elsewhere."""
        coords = np.asarray(coords, dtype=float).reshape(len(coords), -1)
        on_grid = self.U * np.sqrt(self.S)
        d = cdist(coords, self.grid)
        hit = d.min(axis=1) <= 1e-12
        out = np.empty((coords.shape[0], self.n_basis))
        out[hit] = on_grid[d[hit].argmin(axis=1)]
        if not hit.all():
            Uw = self.U if self.grid_weights is None else self.U * self.grid_weights[:, None]
            k = kernel_matrix(self.kernel, coords[~hit], self.grid, self.lengthscale, self.variance)
            out[~hit] = k @ (Uw / np.sqrt(self.S))
        return out

    @property
    def hyperparameters(self):
        return {"lengthscale": self.lengthscale, "variance": self.variance, "noise_variance": self.noise_variance}
