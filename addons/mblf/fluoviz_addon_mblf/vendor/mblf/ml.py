"""Maximum-likelihood estimation of the model (scores and basis coefficients integrated out).

The marginal log-likelihood is the Stan program ``mblf_ml.stan``, maximised with CmdStan's L-BFGS from several
starting points (a moment-based principal-axis start plus perturbed and random starts), then refined from the best
start with tighter tolerances. Every mode is divided by the root mean square of its observed values first (an
exact reparameterisation; the reported log-likelihood is that of the original data). Variances are bounded below by
``variance_floor`` times the squared scale of their mode.
"""

import json
import time

import numpy as np

from .data import mode_summaries
from .likelihood import marginal_loglik
from .model import stan_model


class MLResult:
    """Result of `fit_ml`. Parameters are on the scale of the data: ``Lambda``, ``sigma2`` (lists per mode),
    ``psi2`` (D,), ``beta`` (K, L)."""

    def __init__(self, **fields):
        self.__dict__.update(fields)

    def to_dict(self):
        def conv(v):
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, (list, tuple)):
                return [conv(x) for x in v]
            if isinstance(v, dict):
                return {k: conv(x) for k, x in v.items()}
            if isinstance(v, (np.floating, np.integer, np.bool_)):
                return v.item()
            return v
        return {k: conv(v) for k, v in self.__dict__.items()}

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        for k in ("Lambda", "sigma2"):
            d[k] = [np.asarray(x) for x in d[k]]
        for k in ("psi2", "beta", "factor_scores", "factor_scores_cov"):
            d[k] = np.asarray(d[k])
        return cls(**d)

    def __repr__(self):
        return (f"MLResult(K={self.K}, loglik={self.loglik:.3f}, bic={self.bic:.3f}, converged={self.converged}, "
                f"starts at optimum {self.n_starts_at_optimum}/{len(self.starts)})")


def _moment_start(summ, X, K, floor=1e-6):
    """Principal-axis factor analysis of the covariance of the least-squares coefficients (standardised scale)."""
    N = X.shape[0]
    resid, usable, gammas, psi0, noise = [], [], [], [], []
    for T, rss, n_obs, G in summ:
        p = T.shape[0]
        rank = np.sum(np.any(G != 0, axis=2), axis=1)
        use = rank == p
        dof = np.sum(np.maximum(n_obs - rank, 0))
        u = np.einsum("npq,qn->np", G, T)
        psi = rss.sum() / dof if dof > 0 else np.nan
        if not np.isfinite(psi) or psi <= 0:
            psi = 0.1 * np.mean(u**2) if np.any(u) else 0.1
        psi0.append(psi)
        noise.append(psi * np.mean(np.linalg.pinv(np.swapaxes(G[use], 1, 2) @ G[use]), axis=0) if use.any()
                     else np.zeros((p, p)))
        if not use.any():
            use = np.ones(N, dtype=bool)
        Tn = T.T
        gamma = np.linalg.lstsq(X[use], Tn[use], rcond=None)[0] if X.shape[1] else np.zeros((0, p))
        resid.append(Tn - X @ gamma)
        usable.append(use)
        gammas.append(gamma)
    p = [r.shape[1] for r in resid]
    off = np.cumsum([0] + p)
    S = np.zeros((off[-1], off[-1]))
    for a in range(len(p)):
        for b in range(len(p)):
            both = usable[a] & usable[b]
            if both.any():
                S[off[a]:off[a + 1], off[b]:off[b + 1]] = resid[a][both].T @ resid[b][both] / both.sum()
        S[off[a]:off[a + 1], off[a]:off[a + 1]] -= noise[a]
    S = 0.5 * (S + S.T)
    diag = np.maximum(np.diag(S), floor)
    ev, vec = np.linalg.eigh(S)
    ev, vec = ev[::-1], vec[:, ::-1]
    P = off[-1]
    k = min(K, P)
    rest = float(np.mean(ev[k:])) if P > k else 0.0
    Lam = np.zeros((P, K))
    Lam[:, :k] = vec[:, :k] * np.sqrt(np.maximum(ev[:k] - max(rest, 0.0), 1e-3 * max(ev[0], floor)))
    if k < K:
        Lam[:, k:] = 1e-2 * np.sqrt(diag.mean())
    sigma2 = np.maximum(diag - np.sum(Lam**2, axis=1), np.maximum(0.05 * diag, floor))
    beta = (np.linalg.lstsq(Lam, np.vstack([g.T for g in gammas]), rcond=None)[0] if X.shape[1]
            else np.zeros((K, 0)))
    return {"Lambda": Lam, "sigma2": sigma2, "psi2": np.array(psi0), "beta": beta}


def _perturbed(start, p, n, rng, random_fraction=0.6):
    """n further starts: the first round(random_fraction n) with loadings drawn afresh, the rest jittered."""
    off = np.cumsum([0] + list(p))
    starts = []
    n_random = int(round(n * random_fraction))
    for i in range(n):
        fresh = i < n_random
        Lam = start["Lambda"].copy()
        for a, b in zip(off[:-1], off[1:]):
            s = max(float(np.sqrt(np.mean(start["Lambda"][a:b] ** 2))), 1e-3)
            noise = rng.normal(size=Lam[a:b].shape) * s
            Lam[a:b] = noise if fresh else Lam[a:b] + 0.5 * noise
        sigma2 = start["sigma2"] * np.exp(rng.normal(size=start["sigma2"].shape) * 0.5)
        psi2 = start["psi2"] * np.exp(rng.normal(size=start["psi2"].shape) * 0.25)
        b0 = start["beta"]
        s = max(float(np.sqrt(np.mean(b0**2))), 1e-2) if b0.size else 1.0
        beta = (0.0 if fresh else b0) + 0.5 * s * rng.normal(size=b0.shape)
        starts.append({"Lambda": Lam, "sigma2": sigma2, "psi2": psi2, "beta": beta})
    return starts


def _n_redundant(Lam, beta, n_sigma, n_psi):
    """Number of independent directions of the rotation invariance Lambda -> Lambda O, beta -> O' beta."""
    P, K = Lam.shape
    dirs = []
    for a in range(K):
        for b in range(a + 1, K):
            A = np.zeros((K, K))
            A[a, b], A[b, a] = 1.0, -1.0
            dirs.append(np.concatenate([(Lam @ A).ravel(), np.zeros(n_sigma + n_psi), (-A @ beta).ravel()]))
    return int(np.linalg.matrix_rank(np.array(dirs))) if dirs else 0


def fit_ml(modes, X=None, n_components=1, n_starts=16, seed=0, variance_floor=1e-8, max_iter=10000,
           optimum_tolerance=1e-4, verbose=False, history_size=20):
    """Maximum-likelihood estimate of the factor model for a list of `FunctionalData` modes (each with a basis).

    Returns an `MLResult` with the parameters, ``loglik``, ``n_params``, ``n_effective_params`` (parameters minus the
    rotation invariance), ``aic``, ``bic`` (log N penalty), the record of every start, ``n_starts_at_optimum``,
    the variances at their lower bound, and the conditional mean and covariance of the factor scores.
    """
    t0 = time.perf_counter()
    K = int(n_components)
    D, N = len(modes), modes[0].N
    X = np.zeros((N, 0)) if X is None else np.asarray(X, dtype=float).reshape(N, -1)
    scales = []
    for m in modes:
        s = float(np.sqrt(np.nanmean(m.data**2)))
        scales.append(s if np.isfinite(s) and s > 0 else 1.0)
    summ = [mode_summaries(m.basis, m.data / s) for m, s in zip(modes, scales)]
    p = [m.basis.shape[1] for m in modes]
    P = sum(p)
    off = np.cumsum([0] + p)
    R = np.zeros((N, P, P))
    for d, (T, rss, n_obs, G) in enumerate(summ):
        R[:, off[d]:off[d + 1], off[d]:off[d + 1]] = G
    U = np.vstack([np.einsum("npq,qn->pn", G, T) for T, rss, n_obs, G in summ])
    data = dict(N=N, D=D, K=K, L=X.shape[1], X=X.T, P=P, p=p, R=R, U=U, rss=np.vstack([s[1] for s in summ]),
                n_obs=np.vstack([s[2] for s in summ]).astype(float), variance_floor=variance_floor)
    inside = variance_floor * np.exp(1e-6)
    start = _moment_start(summ, X, K)
    rng = np.random.default_rng(seed)
    starts = [("moment", start)] + [("random" if i < int(round(0.6 * (n_starts - 1))) else "jitter", s)
                                    for i, s in enumerate(_perturbed(start, p, n_starts - 1, rng))]
    program = stan_model("mblf_ml")

    # CmdStan's default L-BFGS tolerances stop flat factor-model optima early (starts then differ by ~0.05 in the
    # log-likelihood): every start runs with tight relative tolerances. A history of 20 curvature pairs (CmdStan's
    # default is 5) needed about 40% fewer iterations on the air-quality data.
    tight = dict(tol_rel_obj=1.0, tol_rel_grad=1e2, tol_obj=1e-14, tol_grad=1e-10, tol_param=1e-12)

    def run(init, **tol):
        init = {k: np.maximum(v, inside) if k in ("sigma2", "psi2") else v for k, v in init.items()}
        if X.shape[1] == 0:
            init = {k: v for k, v in init.items() if k != "beta"}
        fit = program.optimize(data=data, inits=init, algorithm="lbfgs", iter=max_iter, jacobian=False, seed=seed,
                               sig_figs=18, require_converged=False, show_console=False, history_size=history_size,
                               **{**tight, **tol})
        par = {k: np.asarray(fit.stan_variable(k), dtype=float) for k in ("Lambda", "sigma2", "psi2")}
        par["beta"] = (np.asarray(fit.stan_variable("beta"), dtype=float).reshape(K, X.shape[1]) if X.shape[1]
                       else np.zeros((K, 0)))
        ll = float(fit.optimized_params_dict["lp__"])
        # A variance whose optimum is the lower bound approaches it only slowly (the gradient in the log scale
        # vanishes there): try every small variance exactly at the bound and keep it there if the likelihood is higher.
        small_s, small_p = par["sigma2"] < 1e-4, par["psi2"] < 1e-4
        if small_s.any() or small_p.any():
            snapped = dict(par, sigma2=np.where(small_s, variance_floor, par["sigma2"]),
                           psi2=np.where(small_p, variance_floor, par["psi2"]))
            ll_snapped = std_loglik(snapped)
            if ll_snapped > ll:
                par, ll = snapped, ll_snapped
        return ll, par, bool(fit.converged)

    def std_loglik(par):
        """Log-likelihood of the standardised data (the Stan target), computed with NumPy."""
        lam = [par["Lambda"][None, off[d]:off[d + 1]] for d in range(D)]
        sig = [par["sigma2"][None, off[d]:off[d + 1]] for d in range(D)]
        return float(marginal_loglik(summ, lam, sig, par["psi2"][None], par["beta"][None], X).sum())

    records, best = [], None
    for i, (kind, init) in enumerate(starts):
        try:
            ll, par, conv = run(init)
        except Exception as exc:  # a failing start must not stop the fit
            records.append({"start": i, "kind": kind, "loglik": None, "converged": False, "status": str(exc)[:200]})
            continue
        records.append({"start": i, "kind": kind, "loglik": ll, "converged": conv})
        if verbose:
            print(f"start {i:2d} ({kind:6s}) loglik {ll:.6f} converged {conv}")
        if best is None or ll > best[0]:
            best = (ll, par, conv)
    if best is None:
        raise RuntimeError("every start failed: " + "; ".join(r["status"] for r in records))
    try:  # restart from the best optimum (L-BFGS restarts with a fresh curvature estimate)
        ll, par, conv = run(best[1])
        if ll >= best[0] - 1e-9:
            best = (ll, par, conv)
    except Exception:
        pass
    ll_std, par, converged = best
    correction = -sum(float(s[2].sum()) * np.log(sc) for s, sc in zip(summ, scales))
    loglik = ll_std + correction
    for r in records:
        if r["loglik"] is not None:
            r["loglik"] += correction
    floor_names = [f"sigma2[{d + 1}][{j + 1}]" for d in range(D) for j in range(p[d])
                   if par["sigma2"][off[d] + j] <= variance_floor * np.exp(1e-6)]
    floor_names += [f"psi2[{d + 1}]" for d in range(D) if par["psi2"][d] <= variance_floor * np.exp(1e-6)]
    n_params = P * K + P + D + K * X.shape[1]
    n_eff = n_params - _n_redundant(par["Lambda"], par["beta"], P, D)
    # back to the scale of the data
    Lam = [par["Lambda"][off[d]:off[d + 1]] * scales[d] for d in range(D)]
    sigma2 = [par["sigma2"][off[d]:off[d + 1]] * scales[d] ** 2 for d in range(D)]
    psi2 = par["psi2"] * np.array(scales) ** 2
    summ_orig = [mode_summaries(m.basis, m.data) for m in modes]
    _, eta_mean, eta_cov = marginal_loglik(summ_orig, [l[None] for l in Lam], [s[None] for s in sigma2], psi2[None],
                                           par["beta"][None], X, scores=True)
    best_ll = max(r["loglik"] for r in records if r["loglik"] is not None)
    return MLResult(K=K, D=D, Lambda=Lam, sigma2=sigma2, psi2=psi2, beta=par["beta"], loglik=loglik, n_params=n_params,
                    n_effective_params=n_eff, n_subjects=N, aic=-2 * loglik + 2 * n_eff, bic=-2 * loglik + n_eff * np.log(N),
                    converged=converged, starts=records,
                    n_starts_at_optimum=int(sum(r["loglik"] is not None and r["loglik"] >= max(best_ll, loglik) - optimum_tolerance
                                                for r in records)),
                    at_variance_floor=floor_names, factor_scores=eta_mean[0], factor_scores_cov=eta_cov[0],
                    scales=scales, seconds=time.perf_counter() - t0)
