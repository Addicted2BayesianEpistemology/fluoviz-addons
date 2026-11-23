"""The numerics against brute-force Gaussian algebra: the likelihood of any subset of modes (covariates known or not), the posterior of the scores, the parts of the predictive
variance, the predictable fraction, the mixture quantiles, the alignment, the leave-one-out comparison."""
import numpy as np
import pytest
from scipy.stats import multivariate_normal

import mblf_support as T
from fluoviz_addon_mblf.core import loo as LOO
from fluoviz_addon_mblf.core import predict as P
from fluoviz_addon_mblf.core import structure as ST


@pytest.fixture(scope="module")
def world():
    modes = T.make_modes(values=2)
    tr = T.truth(modes, K=2, L=1)
    Y, X = T.simulate(modes, tr, N=30)
    f = T.fitted(modes, tr, Y, X, S=40)
    return f


def _obs(f, n, mods):
    ys, rows = [], []
    a = 0
    for d, m in enumerate(f.modes):
        if d in mods:
            y = f.Y[d][:, n]
            ok = np.flatnonzero(np.isfinite(y))
            ys.append(y[ok])
            rows.append(a + ok)
        a += m.M
    return np.concatenate(ys), np.concatenate(rows)


@pytest.mark.parametrize("mods,cov", [([0, 1, 2], True), ([0, 1, 2], False), ([1], True), ([0, 2], False)])
def test_the_likelihood_of_any_subset_of_modes_is_the_gaussian_density(world, mods, cov):
    f = world
    idx = np.arange(3)
    summ = LOO.summaries(f)
    ll = LOO.loglik(f, summ, mods, idx, f.cov.X, cov)
    for s in idx:
        BL, Psi, R, noise, C = T.joint_cov(f, s, cov)
        for n in (0, 5, 17):
            y, rows = _obs(f, n, mods)
            mu = BL @ (f.draws.beta[s] @ f.cov.X[n]) if cov else np.zeros(BL.shape[0])
            want = multivariate_normal(mu[rows], C[np.ix_(rows, rows)]).logpdf(y)
            assert ll[s, n] == pytest.approx(want, rel=1e-8, abs=1e-6)


def test_the_likelihood_matches_mblf_with_the_covariates(world):
    from fluoviz_addon_mblf.vendor.mblf.likelihood import marginal_loglik
    f = world
    idx = np.arange(5)
    summ = LOO.summaries(f)
    ours = LOO.loglik(f, summ, list(range(f.D)), idx, f.cov.X, True)
    theirs = marginal_loglik(summ, [f.lam(d, idx=idx) for d in range(f.D)], [f.sig2(d, idx) for d in range(f.D)], f.draws.psi2[idx], f.draws.beta[idx], f.cov.X)
    assert np.allclose(ours, theirs, rtol=1e-9, atol=1e-8)


@pytest.mark.parametrize("cov", [True, False])
def test_the_prediction_and_its_budget_are_gaussian_conditioning(world, cov):
    f = world
    s, n, t = 2, 7, 1
    idx = np.array([s])
    src = {0: f.Y[0][:, [n]], 2: f.Y[2][:, [n]]}
    eta, V = P.score_posterior(f, src, f.cov.X[[n]] if cov else None, idx)
    mom = P.target_moments(f, t, eta, V, idx)
    BL, Psi, R, noise, C = T.joint_cov(f, s, cov)
    y, rows = _obs(f, n, [0, 2])
    off = f.modes[0].M + f.modes[1].M
    trows = np.arange(f.modes[0].M, off)
    mu = BL @ (f.draws.beta[s] @ f.cov.X[n]) if cov else np.zeros(BL.shape[0])
    Cl = C - np.diag(noise)                                                    # the target's latent function (no noise), the sources keep theirs
    Css = C[np.ix_(rows, rows)]
    Cts = Cl[np.ix_(trows, rows)]
    want_mean = mu[trows] + Cts @ np.linalg.solve(Css, y - mu[rows])
    want_var = np.diag(Cl[np.ix_(trows, trows)] - Cts @ np.linalg.solve(Css, Cts.T))
    assert np.allclose(mom["mean"][0, 0], want_mean, atol=1e-8)
    assert np.allclose(mom["scores"][0, 0] + mom["residual"][0], want_var, atol=1e-8)
    b = P.budget(mom, noisy=True)
    assert b["parameters"] == pytest.approx(np.zeros_like(b["parameters"])) and np.allclose(b["noise"], f.draws.psi2[s, t])


def test_the_parameter_part_is_the_spread_of_the_conditional_means(world):
    f = world
    idx = np.arange(f.draws.S)
    out = P.predict(f, 0, {1: f.Y[1][:, [3]]}, f.cov.X[[3]], idx, noisy=False)
    eta, V = P.score_posterior(f, {1: f.Y[1][:, [3]]}, f.cov.X[[3]], idx)
    mom = P.target_moments(f, 0, eta, V, idx)
    assert np.allclose(out["budget"]["parameters"], mom["mean"][:, 0].var(axis=0) * f.modes[0].scale ** 2)
    assert sum(out["budget_total"].values()) == pytest.approx(1.0) and out["budget_total"]["noise"] == 0.0
    q = out["quantiles"]
    assert np.all(np.diff(q, axis=0) >= -1e-9) and np.all(out["band"]["lower"] <= q[3] + 1e-9) and np.all(out["band"]["upper"] >= q[3] - 1e-9)


def test_mixture_quantiles_against_sampling():
    rng = np.random.default_rng(0)
    m = rng.normal(size=(50, 4)) * np.array([0.1, 1, 3, 0.5])
    sd = rng.uniform(0.2, 1.0, size=(50, 4))
    q = P.mixture_quantiles(m, sd, (0.05, 0.5, 0.95))
    draws = m[None] + sd[None] * rng.normal(size=(4000, 50, 4))
    want = np.quantile(draws.reshape(-1, 4), [0.05, 0.5, 0.95], axis=0)
    assert np.allclose(q, want, atol=0.06)
    assert np.allclose(P.mixture_cdf(q[1], m, sd), 0.5, atol=1e-5)


def test_the_predictable_fraction_is_one_minus_the_expected_conditional_variance(world):
    f = world
    s = 4
    idx = np.array([s])
    r = ST.predictable(f, 1, [0], True, idx)
    BL, Psi, R, noise, C = T.joint_cov(f, s, True)
    M0, M1 = f.modes[0].M, f.modes[1].M
    src = np.arange(M0)
    tg = np.arange(M0, M0 + M1)
    Cl = C - np.diag(noise)
    bS = BL @ f.draws.beta[s]                                                 # the covariates' part of the mean (known: it is explained)
    tot = np.diag(Cl[np.ix_(tg, tg)]) + (bS[tg] @ f.cov.cov() @ bS[tg].T).diagonal()
    cond = np.diag(Cl[np.ix_(tg, tg)] - Cl[np.ix_(tg, src)] @ np.linalg.solve(C[np.ix_(src, src)], Cl[np.ix_(src, tg)]))
    assert np.allclose(r["latent"]["mean"], 1 - cond / tot, atol=1e-8)
    none = ST.predictable(f, 1, [], False, idx)
    assert np.allclose(none["latent"]["mean"], 0.0, atol=1e-12)
    more = ST.predictable(f, 1, [0, 2], True, idx)
    assert np.all(more["latent"]["mean"] >= r["latent"]["mean"] - 1e-9)


def test_the_alignment_undoes_a_rotation_of_the_draws(world):
    f = world
    from fluoviz_addon_mblf.core.posterior import alignment
    rng = np.random.default_rng(5)
    lam = f.draws.lam.copy()
    rot = []
    for s in range(lam.shape[0]):
        Q, _ = np.linalg.qr(rng.normal(size=(2, 2)))
        lam[s] = lam[s] @ Q
        rot.append(Q)
    T1, info = alignment(f.modes, lam)
    a1 = lam @ T1
    a0 = f.draws.lam @ f.draws.T
    assert np.allclose(a1.std(axis=0).mean(), a0.std(axis=0).mean(), rtol=0.2)
    assert np.allclose(np.abs(a1.mean(axis=0)), np.abs(a0.mean(axis=0)), atol=0.05)
    assert np.all(np.diff(info["strength"].mean(axis=0)) <= 1e-12)


def test_leave_one_out_comparison_of_the_sources(world):
    f = world
    idx = np.arange(f.draws.S)
    base = LOO.loo_table(f, idx)
    assert np.isfinite(base["elpd"]) and base["pointwise"].shape == (f.N,)
    sets = LOO.source_sets(f, 1)
    assert sets[0]["key"] == "nothing" and {s["key"] for s in sets} >= {"absorbance", "absorbance+values+covariates", "covariates"}
    rows = LOO.conditional_elpd(f, 1, sets, idx, base)
    by = {r["key"]: r for r in rows}
    assert by["absorbance+values+covariates"]["elpd"] > by["nothing"]["elpd"]
    assert by["nothing"]["gain"] == 0.0
    cal = LOO.calibration(f, 1, by["absorbance"], idx, base)
    assert cal["n_samples"] == f.N and 0.5 < cal["coverage"]["0.95"] <= 1.0 and cal["rmse"] > 0
