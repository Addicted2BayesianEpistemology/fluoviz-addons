"""The multivariate Bayesian latent factor model (`MvtBLF`): Stan fit, posterior summaries and predictions.

Model (paper, Section "Likelihood" and "Prior"), for subjects n = 1..N and modes d = 1..D:

    eta_n ~ N_K(beta x_n, I),  theta_dn ~ N(Lambda_d eta_n, diag(1/tau_d)),  y_dn ~ N(B_d theta_dn, psi2_d I).

Mode indices are 1-based in every public method, as in the paper.
"""

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from . import diagnostics as dg
from .align import varimax_rsp
from .data import FunctionalData, mode_summaries
from .likelihood import marginal_loglik

STAN_DIR = Path(__file__).parent / "stan"


def stan_model(name):
    """Compiled CmdStan program ``name`` ("mblf" or "mblf_ml"), cached in $MBLF_CACHE_DIR (default ~/.cache/mblf).

    The program is copied to the cache first: CmdStan's makefiles fail on paths with spaces. A file lock makes
    parallel processes compile it once.
    """
    import cmdstanpy

    code = (STAN_DIR / f"{name}.stan").read_text()
    cache = Path(os.environ.get("MBLF_CACHE_DIR", Path.home() / ".cache" / "mblf"))
    cache.mkdir(parents=True, exist_ok=True)
    stan_file = cache / f"{name}_{hashlib.sha1(code.encode()).hexdigest()[:10]}.stan"
    try:
        import fcntl
    except ImportError:  # not on Unix: no lock
        fcntl = None
    with open(cache / f"{stan_file.stem}.lock", "w") as lock:
        if fcntl:
            fcntl.flock(lock, fcntl.LOCK_EX)
        if not stan_file.exists():
            stan_file.write_text(code)
        return cmdstanpy.CmdStanModel(stan_file=str(stan_file))


def thin_index(n_chains, n_draws, max_draws):
    """Draw indices (per chain) that keep at most ``max_draws`` draws in total, evenly spaced."""
    if max_draws is None or n_chains * n_draws <= max_draws:
        return np.arange(n_draws)
    return np.arange(0, n_draws, int(np.ceil(n_chains * n_draws / max_draws)))


class MvtBLF:
    """Multivariate Bayesian latent factor model.

    Parameters
    ----------
    n_components : int
        Number of factors K (a cap: the MGPS prior shrinks unneeded columns).
    a_psi, b_psi : inverse-gamma prior of psi2_d (scalar or one value per mode).
    a_tau, b_tau : gamma (shape, rate) prior of the coefficient precisions tau_dj.
    a1, a2 : MGPS shapes of delta_1 and delta_k (k >= 2).
    b_lambdavar : multiplier of the loading variance of each mode.
    nu : degrees of freedom of the Student-t prior of the loadings.
    sigma_regr_coeffs : prior standard deviation of beta.
    noncentering_threshold : a coefficient is sampled non-centred when its prior/data precision ratio
        kappa_dj = (a_tau/b_tau) psi2_hat_d / g_dj exceeds this value (psi2_hat_d: pooled residual variance of the
        least-squares fits; g_dj: average squared basis value over the observed rows).
    """

    def __init__(self, n_components, a_psi=2.0, b_psi=2.0, a_tau=2.0, b_tau=2.0, a1=2.0, a2=2.0, b_lambdavar=1.0,
                 nu=3.0, sigma_regr_coeffs=10.0, noncentering_threshold=1.0):
        self.K = int(n_components)
        self.priors = dict(a_psi=a_psi, b_psi=b_psi, a_tau=a_tau, b_tau=b_tau, a1=a1, a2=a2, b_lambdavar=b_lambdavar,
                           nu=nu, sigma_regr_coeffs=sigma_regr_coeffs)
        self.noncentering_threshold = float(noncentering_threshold)
        self.draws = self.sample_stats = self.modes = self.X = None
        self.info = {}
        self._rsp = None

    # ------------------------------------------------------------------------------------------------ data
    def _per_mode(self, name, D):
        v = np.asarray(self.priors[name], dtype=float)
        return np.full(D, float(v)) if v.ndim == 0 else v

    def noncentering(self, modes):
        """Non-centring flags of every stacked coefficient, and the precision ratios kappa that define them."""
        D = len(modes)
        a_tau, b_tau = self._per_mode("a_tau", D), self._per_mode("b_tau", D)
        a_psi, b_psi = self._per_mode("a_psi", D), self._per_mode("b_psi", D)
        flags, kappas = [], []
        for d, m in enumerate(modes):
            B, Y = m.basis, m.data
            _, rss, n_obs, _ = mode_summaries(B, Y)
            df = int(np.maximum(n_obs - B.shape[1], 0).sum())
            psi2_hat = rss.sum() / df if df else (b_psi[d] / (a_psi[d] - 1) if a_psi[d] > 1 else np.inf)
            g = (B * B).T @ (~np.isnan(Y)).astype(float)
            g = g.sum(axis=1) / Y.shape[1]
            with np.errstate(divide="ignore", invalid="ignore"):
                kappa = np.where(g > 0, (a_tau[d] / b_tau[d]) * psi2_hat / np.where(g > 0, g, 1), np.inf)
            flags.append(kappa > self.noncentering_threshold)
            kappas.append(kappa)
        return np.concatenate(flags), kappas

    def stan_data(self, modes, X=None):
        """The data of the Stan program for a list of `FunctionalData` modes and covariates X (N, L)."""
        from scipy.sparse import coo_matrix

        D = len(modes)
        if any(m.basis is None for m in modes):
            raise ValueError("every mode needs a basis (set_basis or set_gp_basis)")
        N = modes[0].N
        if any(m.N != N for m in modes):
            raise ValueError("all modes must have the same subjects (columns)")
        X = np.zeros((N, 0)) if X is None else np.asarray(X, dtype=float).reshape(N, -1)
        summ = [mode_summaries(m.basis, m.data) for m in modes]
        p = [m.basis.shape[1] for m in modes]
        # A: rows grouped by mode (subject by subject within a mode), columns in the order of to_vector(T - theta)
        P, start, rows, cols, vals = sum(p), np.cumsum([0] + p), [], [], []
        for d, (_, _, _, G) in enumerate(summ):
            i, j = np.meshgrid(np.arange(p[d]), np.arange(p[d]), indexing="ij")
            for n in range(N):
                rows.append(start[d] * N + n * p[d] + i.ravel())
                cols.append(n * P + start[d] + j.ravel())
                vals.append(G[n].ravel())
        csr = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(P * N, P * N)).tocsr()
        csr.eliminate_zeros()
        csr.sort_indices()
        flags, _ = self.noncentering(modes)
        data = dict(N=N, D=D, K=self.K, L=X.shape[1], X=X.T, P=sum(p), p=p, T=np.vstack([s[0] for s in summ]),
                    sum_rss=[float(s[1].sum()) for s in summ], sum_M=[int(s[2].sum()) for s in summ],
                    n_gram=int(csr.nnz), gram_w=csr.data, gram_v=csr.indices + 1, gram_u=csr.indptr + 1,
                    noncentered=flags.astype(int))
        for name in ("a_psi", "b_psi", "a_tau", "b_tau", "b_lambdavar"):
            data[name] = self._per_mode(name, D)
        for name in ("a1", "a2", "nu", "sigma_regr_coeffs"):
            data[name] = float(self.priors[name])
        return data

    # ------------------------------------------------------------------------------------------------- fit
    def fit(self, modes, X=None, chains=4, iter_warmup=1000, iter_sampling=1000, thin=1, max_treedepth=12,
            adapt_delta=0.8, seed=12345, parallel_chains=None, output_dir=None, show_progress=False, **sample_kwargs):
        """Sample the posterior with CmdStan NUTS.

        Parameters
        ----------
        modes : list of FunctionalData, each with a basis; all with the same N subjects.
        X : array (N, L), optional
            Covariates of the factor scores (no intercept is added).
        Other arguments are those of `cmdstanpy.CmdStanModel.sample`.
        """
        t0 = time.perf_counter()
        data = self.stan_data(modes, X)
        model = stan_model("mblf")
        t1 = time.perf_counter()
        fit = model.sample(data=data, chains=chains, parallel_chains=parallel_chains or chains, iter_warmup=iter_warmup,
                           iter_sampling=iter_sampling, thin=thin, max_treedepth=max_treedepth, adapt_delta=adapt_delta,
                           seed=seed, output_dir=output_dir, show_progress=show_progress, **sample_kwargs)
        t2 = time.perf_counter()
        ds = fit.draws_xr()

        def get(name):
            da = ds[name]
            return np.asarray(da.transpose("chain", "draw", *[x for x in da.dims if x not in ("chain", "draw")]).values)

        draws = {name: get(name) for name in ("Lambda", "tau", "eta", "psi2", "delta", "omega")}
        # without covariates (L = 0) CmdStan reports no regr_coeffs: keep an empty (chain, draw, K, 0) array
        draws["regr_coeffs"] = (get("regr_coeffs") if data["L"] > 0
                                else np.zeros(draws["eta"].shape[:2] + (self.K, 0)))
        theta = get("theta_tilde")
        nc = np.flatnonzero(data["noncentered"])
        if nc.size:
            mean = draws["Lambda"][:, :, nc, :] @ draws["eta"]
            theta[:, :, nc, :] = mean + theta[:, :, nc, :] / np.sqrt(draws["tau"][:, :, nc, None])
        draws["theta"] = theta
        mv = fit.method_variables()
        draws["lp__"] = mv["lp__"].T
        self.sample_stats = {k: mv[k + "__"].T for k in ("divergent", "treedepth", "energy", "stepsize", "n_leapfrog")}
        self.draws, self.modes = draws, list(modes)
        self.X = np.zeros((data["N"], 0)) if X is None else np.asarray(X, dtype=float).reshape(data["N"], -1)
        self.noncentered = data["noncentered"].astype(bool)
        self.info = dict(chains=chains, iter_warmup=iter_warmup, iter_sampling=iter_sampling, thin=thin,
                         max_treedepth=max_treedepth, adapt_delta=adapt_delta, seed=seed,
                         seconds_prepare_compile=t1 - t0, seconds_sampling=t2 - t1, seconds_total=time.perf_counter() - t0)
        self._rsp = None
        return self

    # ---------------------------------------------------------------------------------------------- access
    @property
    def D(self):
        return len(self.modes)

    @property
    def p(self):
        return [m.basis.shape[1] for m in self.modes]

    def rows(self, d):
        """Rows of mode d (1-based) in the stacked coefficients."""
        start = int(np.sum(self.p[: d - 1]))
        return slice(start, start + self.p[d - 1])

    def loadings(self, d):
        """Loadings Lambda_d on the basis-coefficient scale, (chain, draw, p_d, K)."""
        return self.draws["Lambda"][:, :, self.rows(d), :]

    def functional_loadings(self, d, coords=None):
        """Loading functions B_d Lambda_d at the grid of mode d (or at ``coords`` for a GP basis), (chain, draw, M, K)."""
        B = self.modes[d - 1].basis if coords is None else self.modes[d - 1].basis_info.evaluate(coords)
        return np.einsum("mp,cspk->csmk", B, self.loadings(d))

    def coefficient_variances(self, d):
        """Variances 1/tau_d of the basis coefficients of mode d, (chain, draw, p_d)."""
        return 1.0 / self.draws["tau"][:, :, self.rows(d)]

    def compute_varimax_rsp(self):
        """Varimax-RSP alignment of the stacked loading draws (cached); see `mblf.align.varimax_rsp`."""
        if self._rsp is None:
            self._rsp = varimax_rsp(self.draws["Lambda"])
        return self._rsp

    def aligned_loadings(self, d):
        """Varimax-RSP aligned loadings of mode d, (chain, draw, p_d, K)."""
        return self.compute_varimax_rsp()["aligned"][:, :, self.rows(d), :]

    def _flat(self, name, index=None):
        a = self.draws[name] if index is None else self.draws[name][:, index]
        return a.reshape((-1,) + a.shape[2:])

    # ----------------------------------------------------------------------------------------- diagnostics
    def identified_quantities(self):
        """Rotation-invariant quantities whose convergence is checked (paper, Appendix "Simulation details"):
        log density, psi2, tau, theta, shared variance diag(Lambda_d Lambda_d') and its total, the norm of the
        cross-mode covariance ||Lambda_d Lambda_e'||, the factor strengths (eigenvalues of Lambda'Lambda) and the
        covariate effects Lambda_d beta. Each value is an array (chain, draw, ...)."""
        lam = [self.loadings(d) for d in range(1, self.D + 1)]
        gram = [np.einsum("csjk,csjl->cskl", l, l) for l in lam]
        q = {"lp__": self.draws["lp__"], "psi2": self.draws["psi2"], "tau": self.draws["tau"], "theta": self.draws["theta"],
             "shared_variance": np.sum(self.draws["Lambda"] ** 2, axis=-1),
             "shared_variance_total": np.stack([np.trace(g, axis1=-2, axis2=-1) for g in gram], axis=-1),
             "factor_strengths": np.linalg.eigvalsh(np.sum(gram, axis=0))[..., ::-1]}
        pairs = [(d, e) for d in range(self.D) for e in range(d + 1, self.D)]
        if pairs:
            q["cross_covariance_norm"] = np.stack([np.sqrt(np.maximum(np.sum(gram[d] * gram[e], axis=(-2, -1)), 0))
                                                   for d, e in pairs], axis=-1)
        if self.X.shape[1] > 0:
            q["covariate_effects"] = self.draws["Lambda"] @ self.draws["regr_coeffs"]
        return q

    def diagnostics(self, max_rhat=1.01, min_ess=400, min_ebfmi=0.3):
        """One row of sampler and convergence diagnostics over the identified quantities."""
        rh, eb, et = [], [], []
        for v in self.identified_quantities().values():
            rh.append(dg.rhat(v))
            eb.append(dg.ess_bulk(v))
            et.append(dg.ess_tail(v))
        rh, eb, et = (np.concatenate(x) for x in (rh, eb, et))
        e = dg.ebfmi(self.sample_stats["energy"])
        row = {"n_chains": self.info["chains"], "n_warmup": self.info["iter_warmup"],
               "n_retained": self.draws["lp__"].shape[1], "max_rhat": float(np.nanmax(rh)),
               "min_ess_bulk": float(np.nanmin(eb)), "min_ess_tail": float(np.nanmin(et)),
               "n_divergent": int(self.sample_stats["divergent"].sum()),
               "n_max_treedepth": int((self.sample_stats["treedepth"] >= self.info["max_treedepth"]).sum()),
               "min_ebfmi": float(e.min()), "sampling_seconds": self.info["seconds_sampling"],
               "total_seconds": self.info["seconds_total"]}
        row["converged"] = bool(row["max_rhat"] <= max_rhat and row["min_ess_bulk"] >= min_ess
                                and row["min_ess_tail"] >= min_ess and row["n_divergent"] == 0 and row["min_ebfmi"] >= min_ebfmi)
        return row

    # ------------------------------------------------------------------------------------------ likelihood
    def marginal_log_likelihood(self, modes=None, X=None, max_draws=None):
        """log p(y_n | parameters of a draw) with scores and coefficients integrated out, (draws, N).

        Uses the training data unless ``modes`` (and ``X``) are given; the draws are thinned to ``max_draws``.
        """
        if modes is None:
            modes, X = self.modes, self.X
        elif X is not None:
            X = np.asarray(X, dtype=float).reshape(modes[0].N, -1)
        if self.X.shape[1] and (X is None or X.shape[1] != self.X.shape[1]):
            raise ValueError("the model has covariates: give X of the subjects")
        idx = thin_index(*self.draws["lp__"].shape, max_draws)
        lam = [self.loadings(d)[:, idx].reshape(-1, self.p[d - 1], self.K) for d in range(1, self.D + 1)]
        sig2 = [self.coefficient_variances(d)[:, idx].reshape(-1, self.p[d - 1]) for d in range(1, self.D + 1)]
        summ = [mode_summaries(m.basis, m.data) for m in modes]
        beta = self._flat("regr_coeffs", idx) if X is not None and X.shape[1] else None
        return marginal_loglik(summ, lam, sig2, self._flat("psi2", idx), beta, X)

    def loo(self, max_draws=None):
        """PSIS-LOO over subjects (whole subjects held out; scores and coefficients integrated out)."""
        C, S = self.draws["lp__"].shape
        ll = self.marginal_log_likelihood(max_draws=max_draws)
        return dg.loo(ll.reshape(C, -1, ll.shape[1]))

    # ------------------------------------------------------------------------------------------ prediction
    def predict(self, target, sources, X=None, level=0.8, latent=True, seed=None, max_draws=None):
        """Conditional prediction of mode ``target`` for new subjects from their observed source modes.

        For every posterior draw, the scores of a new subject given its observed source values are Gaussian
        (precision I + sum_d G_d' W_d^{-1} G_d with G_d = R_d Lambda_d, W_d = R_d diag(1/tau_d) R_d' + psi2_d I, where
        B_d,obs = U_d R_d compresses the observed rows); the target curve follows from the scores (paper, Theorem 3
        and the "modular" rule: an equal-weight mixture over the training draws).

        Parameters
        ----------
        target : int (1-based mode)
        sources : dict {mode: array (M_d, n)} of source curves on the grid of the fit (NaN = not observed)
        X : array (n, L), covariates of the new subjects (required when the model has covariates)
        level : central credible level of the bands
        latent : True for the latent curve B theta, False for a new noisy observation (adds psi2)
        seed : seed of the predictive draws
        max_draws : thin the posterior draws to at most this many

        Returns
        -------
        dict with ``mean`` (M_T, n): the average over draws of the conditional means; ``lower``, ``upper``
        (M_T, n): pointwise quantiles of one predictive draw per posterior draw; ``n_draws``.
        """
        rng = np.random.default_rng(seed)
        idx = thin_index(*self.draws["lp__"].shape, max_draws)
        lam = {d: self.loadings(d)[:, idx].reshape(-1, self.p[d - 1], self.K) for d in range(1, self.D + 1)}
        sig2 = {d: self.coefficient_variances(d)[:, idx].reshape(-1, self.p[d - 1]) for d in range(1, self.D + 1)}
        psi2 = self._flat("psi2", idx)
        S, K = psi2.shape[0], self.K
        src = sorted(sources)
        if target in src:
            raise ValueError("the target cannot be a source")
        Y = {d: np.asarray(sources[d], dtype=float).reshape(self.modes[d - 1].M, -1) for d in src}
        n = Y[src[0]].shape[1]
        L = self.X.shape[1]
        if L and X is None:
            raise ValueError("the model has covariates: give X of the new subjects")
        mu = np.zeros((S, n, K)) if not L else np.einsum("skl,nl->snk", self._flat("regr_coeffs", idx),
                                                         np.asarray(X, dtype=float).reshape(n, L))
        B_T = self.modes[target - 1].basis
        lam_T, sig_T, psi_T = lam[target], sig2[target], psi2[:, target - 1]
        BL_T = np.einsum("mp,spk->smk", B_T, lam_T)
        probs = [(1 - level) / 2, (1 + level) / 2]
        mean, lower, upper = (np.zeros((B_T.shape[0], n)) for _ in range(3))
        masks = np.vstack([np.isnan(Y[d]) for d in src])
        patterns, which = np.unique(masks.T, axis=0, return_inverse=True)
        which = np.asarray(which).reshape(-1)
        for g in range(patterns.shape[0]):
            cols = np.flatnonzero(which == g)
            prec = np.broadcast_to(np.eye(K), (S, K, K)).copy()
            parts = []
            for d in src:
                rows = np.flatnonzero(~np.isnan(Y[d][:, cols[0]]))
                if rows.size == 0:
                    continue
                U, s, Vt = np.linalg.svd(self.modes[d - 1].basis[rows], full_matrices=False)
                R = s[:, None] * Vt
                G = R @ lam[d]                                                  # (S, q, K)
                W = (R[None] * sig2[d][:, None, :]) @ R.T + psi2[:, d - 1, None, None] * np.eye(R.shape[0])
                WG = np.linalg.solve(W, G)
                prec += np.swapaxes(G, 1, 2) @ WG
                parts.append((G, WG, U.T @ Y[d][np.ix_(rows, cols)]))          # z = U' y_obs, (q, n_g)
            V = np.linalg.inv(prec)
            LV = np.linalg.cholesky(0.5 * (V + np.swapaxes(V, 1, 2)))
            eta = mu[:, cols].copy()                                            # (S, n_g, K)
            for G, WG, z in parts:
                resid = z[None] - G @ np.swapaxes(mu[:, cols], 1, 2)            # (S, q, n_g)
                eta += np.swapaxes(V @ np.swapaxes(WG, 1, 2) @ resid, 1, 2)
            for j, i in enumerate(cols):
                m_T = np.einsum("smk,sk->sm", BL_T, eta[:, j])
                z1 = np.einsum("skl,sl->sk", LV, rng.standard_normal((S, K)))
                coef = np.einsum("spk,sk->sp", lam_T, z1) + np.sqrt(sig_T) * rng.standard_normal(sig_T.shape)
                draw = m_T + coef @ B_T.T
                if not latent:
                    draw += np.sqrt(psi_T)[:, None] * rng.standard_normal(draw.shape)
                mean[:, i] = m_T.mean(axis=0)
                lower[:, i], upper[:, i] = np.quantile(draw, probs, axis=0)
        return {"mean": mean, "lower": lower, "upper": upper, "n_draws": S}

    # ------------------------------------------------------------------------------------------ covariance
    def covariance(self, component="latent", coords=None, max_draws=1000):
        """Posterior mean of the covariance of the latent curves of all modes (stacked), given the covariates.

        ``component="shared"``: B Lambda Lambda' B' (the factor part); ``"latent"``: plus the residual basis
        variation blkdiag(B_d diag(1/tau_d) B_d'). Measurement noise is never included. ``coords`` is an optional
        list of locations per mode (GP bases only). Returns a dict with ``covariance``, ``shared``, ``private`` and the
        per-mode ``bases`` used.
        """
        idx = thin_index(*self.draws["lp__"].shape, max_draws)
        lam = self._flat("Lambda", idx)
        shared_coef = np.einsum("sik,sjk->ij", lam, lam) / lam.shape[0]
        bases = [m.basis if coords is None else m.basis_info.evaluate(c)
                 for m, c in zip(self.modes, coords if coords is not None else [None] * self.D)]
        from scipy.linalg import block_diag

        Bb = block_diag(*bases)
        shared = Bb @ shared_coef @ Bb.T
        out = {"shared": shared, "bases": bases, "covariance": shared, "private": None}
        if component == "latent":
            var = 1.0 / self._flat("tau", idx)
            private = block_diag(*[B @ (var.mean(axis=0)[self.rows(d)][:, None] * B.T)
                                   for d, B in enumerate(bases, start=1)])
            out.update(private=private, covariance=shared + private)
        return out

    # -------------------------------------------------------------------------------------------- save/load
    def save(self, path):
        """Save the fit to ``path`` (a folder): posterior draws (npz), settings (json) and the data with its bases."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        arrays = {f"draws_{k}": v for k, v in self.draws.items()}
        arrays.update({f"stats_{k}": v for k, v in self.sample_stats.items()})
        for d, m in enumerate(self.modes, start=1):
            arrays.update({f"coords{d}": m.coords, f"data{d}": m.data, f"basis{d}": m.basis})
        np.savez_compressed(path / "fit.npz", X=self.X, noncentered=self.noncentered, **arrays)
        meta = {"K": self.K, "D": self.D, "priors": self.priors, "noncentering_threshold": self.noncentering_threshold,
                "info": self.info,
                "gp_bases": [None if m.basis_info is None else {k: v for k, v in vars(m.basis_info).items()
                                                                 if isinstance(v, (int, float, str, bool, type(None)))}
                             for m in self.modes]}
        (path / "fit.json").write_text(json.dumps(meta, indent=1, default=float))

    @classmethod
    def load(cls, path):
        """Load a fit saved by `save` (the basis is restored as a matrix; GP settings are kept in ``info``)."""
        path = Path(path)
        meta = json.loads((path / "fit.json").read_text())
        model = cls(meta["K"], noncentering_threshold=meta["noncentering_threshold"], **meta["priors"])
        z = np.load(path / "fit.npz")
        model.draws = {k[6:]: z[k] for k in z.files if k.startswith("draws_")}
        model.sample_stats = {k[6:]: z[k] for k in z.files if k.startswith("stats_")}
        model.modes = [FunctionalData(z[f"coords{d}"], z[f"data{d}"]).set_basis(z[f"basis{d}"])
                       for d in range(1, meta["D"] + 1)]
        model.X, model.noncentered, model.info = z["X"], z["noncentered"], meta["info"]
        model.info["gp_bases"] = meta["gp_bases"]
        return model


def copy_stan_files(destination):
    """Copy the Stan programs of the package to a folder (to inspect or reuse them)."""
    for f in STAN_DIR.glob("*.stan"):
        shutil.copy(f, destination)
