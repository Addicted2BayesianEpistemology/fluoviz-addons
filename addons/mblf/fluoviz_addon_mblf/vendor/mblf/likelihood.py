"""Marginal likelihood of the model with the scores and the basis coefficients integrated out (Theorem 1 of the
paper), evaluated for many parameter draws at once.

For subject n and mode d, with the summaries of `mblf.data.mode_summaries` (R = G_n, u = R T_n, rss, m observed values),
E_d = psi2_d I + R diag(sigma2_d) R' and the score mean mu_n = beta x_n:

    log p(y_dn | eta_n) = -m/2 log(2 pi psi2_d) + p/2 log psi2_d - rss/(2 psi2_d) - 1/2 log|E_d|
                          - 1/2 (u - R Lambda_d eta_n)' E_d^{-1} (u - R Lambda_d eta_n).

Integrating eta_n ~ N(mu_n, I) gives, with H = I + sum_d Lambda_d'R' E_d^{-1} R Lambda_d and
h = sum_d Lambda_d'R' E_d^{-1} (u - R Lambda_d mu_n),

    log p(y_n) = sum_d c_d - 1/2 log|H| - 1/2 (sum_d r_d' E_d^{-1} r_d - h' H^{-1} h),

and eta_n | y_n ~ N(mu_n + H^{-1} h, H^{-1}).
"""

import numpy as np

LOG_2PI = np.log(2 * np.pi)


def marginal_loglik(summaries, lam, sig2, psi2, beta=None, X=None, scores=False):
    """Per-draw, per-subject marginal log-likelihood.

    Parameters
    ----------
    summaries : list over modes of (T (p, N), rss (N,), n_obs (N,), G (N, p, p))
    lam : list over modes of arrays (S, p_d, K)
    sig2 : list over modes of arrays (S, p_d): variances 1/tau of the basis coefficients
    psi2 : array (S, D)
    beta : array (S, K, L) or None
    X : array (N, L) or None
    scores : bool
        Also return the conditional mean (S, N, K) and covariance (S, N, K, K) of the scores.
    """
    S, _, K = lam[0].shape
    N = summaries[0][0].shape[1]
    out = np.zeros((S, N))
    eta_mean = np.zeros((S, N, K)) if scores else None
    eta_cov = np.zeros((S, N, K, K)) if scores else None
    eye = np.eye(K)
    for n in range(N):
        mu = np.zeros((S, K)) if beta is None or X is None or X.shape[1] == 0 else beta @ X[n]
        H = np.broadcast_to(eye, (S, K, K)).copy()
        h = np.zeros((S, K))
        quad = np.zeros(S)
        const = np.zeros(S)
        for d, (T, rss, n_obs, G) in enumerate(summaries):
            R = G[n]
            p = R.shape[0]
            psi = psi2[:, d]
            E = (R[None] * sig2[d][:, None, :]) @ R.T + psi[:, None, None] * np.eye(p)
            RL = R @ lam[d]                                                    # (S, p, K)
            r = R @ T[:, n] - (RL @ mu[..., None])[..., 0]                    # (S, p)
            sol = np.linalg.solve(E, np.concatenate([RL, r[..., None]], axis=-1))
            quad += np.einsum("sp,sp->s", r, sol[..., K])
            h += np.einsum("spk,sp->sk", RL, sol[..., K])
            H += np.swapaxes(RL, 1, 2) @ sol[..., :K]
            logdet_E = 2 * np.sum(np.log(np.diagonal(np.linalg.cholesky(E), axis1=1, axis2=2)), axis=1)
            const += -0.5 * n_obs[n] * (LOG_2PI + np.log(psi)) + 0.5 * p * np.log(psi) - 0.5 * rss[n] / psi - 0.5 * logdet_E
        LH = np.linalg.cholesky(H)
        Hinv_h = np.linalg.solve(H, h[..., None])[..., 0]
        out[:, n] = const - np.sum(np.log(np.diagonal(LH, axis1=1, axis2=2)), axis=1) - 0.5 * (quad - np.einsum("sk,sk->s", h, Hinv_h))
        if scores:
            eta_mean[:, n] = mu + Hinv_h
            eta_cov[:, n] = np.linalg.inv(H)
    return (out, eta_mean, eta_cov) if scores else out
