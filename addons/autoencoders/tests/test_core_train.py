"""Training: the linear autoencoder against the PCA, the functional autoencoder against the functional PCA, the nonlinear network never worse than the linear one, smoothify, the
variational autoencoder (KL, collapse, mixture prior), determinism, missing pixels, k-fold validation, the history file, torch <-> numpy parity, and the speed on the example data.
Skipped cleanly when PyTorch is missing (the base installation has none)."""
import os
import time

import numpy as np
import pytest

pytest.importorskip("torch")

from core_support import GP, PIXELS, cached, example_eems, load_factor, planted, prepared                       # noqa: E402
from fluoviz.analysis.fpca.fit import fit_fpca                                                                  # noqa: E402
from fluoviz_addon_autoencoders.core import data as D, diagnostics as G, forward as fw, linear as lin, tools, train as T    # noqa: E402
from fluoviz_addon_autoencoders.core.spec import AutoencoderError, Config                                      # noqa: E402
from test_core_forward import torch_modules                                                                      # noqa: E402

FAST = dict(epochs=250, patience=60, n_restarts=1)


def eem(ife=True, outliers=False, n=40, basis=GP):
    return cached(("eem", ife, outliers, n, str(basis)), lambda: prepared({"fluorescence": {"basis": basis}}, n=n, ife=ife, outliers=outliers))


def fitted(key, cfg, data):
    return cached(("fit", key), lambda: T.fit(cfg, data))


def subspace_cosines(A: np.ndarray, B: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Cosines of the principal angles between the row spaces of ``A`` and ``B`` in the metric ``w``."""
    Qa, _ = np.linalg.qr((A * np.sqrt(w)).T)
    Qb, _ = np.linalg.qr((B * np.sqrt(w)).T)
    return np.linalg.svd(Qa.T @ Qb, compute_uv=False)


# ----------------------------------------------------------------------------- linear network = PCA
def test_a_linear_autoencoder_finds_the_pca_subspace_and_its_validation_error():
    A, _ = planted(n=60, P=30, k=3, noise=0.05)
    td = D.TrainData.from_array(A)
    res = T.fit(Config(latent_dim=3, activation="linear", hidden=[16], skip=False, epochs=1500, patience=1500, lr=3e-3, n_restarts=1), td)
    base = res.validation["baseline"]
    assert abs(res.validation["evr"] - base["evr"]) < 0.005 and res.validation["evr"] > 0.97
    J = tools.jacobian(res.spec, res.arrays, np.zeros((1, 3)))[0].T                                  # the decoder is linear: its directions in the network coordinates
    tr = np.asarray(res.split["train"])
    pca = lin.fit_pca(td.X[tr], None, td.omega, 3)
    assert subspace_cosines(J, pca.components(3), td.omega).min() > 0.98
    # with the linear shortcut the PCA is the starting point: the network is at least as good as the PCA on held-out samples, to rounding
    res2 = T.fit(Config(latent_dim=3, activation="linear", skip=True, **FAST), td)
    assert res2.validation["evr"] >= res2.validation["baseline"]["evr"] - 1e-4


def test_the_functional_autoencoder_on_a_gp_basis_reaches_the_functional_pca():
    prep = eem(ife=False)
    d = 4
    res = fitted("gp-ae", Config(latent_dim=d, n_restarts=2, epochs=300), prep)
    fpca = fit_fpca([prep.blocks[0].element], "none", d)
    s = res.samples
    evr_all = G.evr(s["error"], s["null"])                                                               # over all samples, the quantity the fPCA's explained variance describes
    assert evr_all >= fpca.cumulative[d - 1] - 0.003 and evr_all <= 1.0
    assert res.validation["evr"] >= res.validation["baseline"]["evr"] - 1e-3                              # held out: not worse than the functional PCA with the same d
    assert res.validation["baseline"]["name"] == "Functional PCA" and res.validation["baseline"]["k"] == d
    assert np.isfinite(res.samples["codes"]).all() and res.samples["codes"].shape == (prep.n, d)


@pytest.mark.parametrize("which", [dict(ife=True), dict(ife=False, outliers=True)])
def test_a_nonlinear_network_is_never_worse_than_the_linear_one(which):
    """The linear start (PCA) is a candidate of the early stopping, so the kept network is never worse than the PCA on held-out samples, and it is not worse than a network with linear
    activations either (also on data with a sample of five times the concentration of the others, which a nonlinear network alone cannot extrapolate)."""
    prep = eem(**which)
    for d in (2, 3):
        non = fitted(("nl", str(which), d), Config(latent_dim=d, epochs=300), prep)
        linear = fitted(("lin", str(which), d), Config(latent_dim=d, activation="linear", epochs=300), prep)
        base = non.validation["baseline"]["evr"]
        assert non.validation["evr"] >= base - 1e-3, (d, non.validation["evr"], base)
        assert non.validation["evr"] >= linear.validation["evr"] - 0.002, (d, non.validation["evr"], linear.validation["evr"])


def test_the_default_training_converges_on_the_example_data_and_is_not_bad_with_a_far_sample():
    """What a person sees first: the example data with its extreme sample (S04_conc has five times the concentration), 'Run with the recommended settings'."""
    prep = eem(ife=False, outliers=True)
    res = fitted("default-outliers", Config(latent_dim=4), prep)
    assert res.validation["evr"] > 0.97 and res.validation["evr"] >= res.validation["baseline"]["evr"] - 0.01
    assert res.best_epoch < res.config["epochs"] * 0.9 or res.best_epoch < 0                                # not still improving when it stops


# ----------------------------------------------------------------------------- smoothify
def roughness_of(res, prep):
    td = prep.train_data("smooth")
    Xh = fw.reconstruct(res.spec, res.arrays, prep.X, prep.M)[0]
    return G.roughness(td, Xh)["mean"]["fluorescence"]


def test_smoothify_lowers_the_roughness_monotonically_and_zero_is_off():
    prep = eem(ife=True)
    rough, evr = [], []
    for s in (0.0, 0.1, 1.0, 10.0):
        r = fitted(("sm", "gp", s), Config(latent_dim=4, smoothify=s, **FAST), prep)
        rough.append(roughness_of(r, prep))
        evr.append(r.validation["evr"])
    assert all(rough[i + 1] < rough[i] for i in range(3)), rough
    assert evr[-1] < evr[0] - 0.05 and evr[1] > evr[-1]                                                  # smoothing costs accuracy, the more the larger it is
    off, plain = fitted(("sm", "gp", 0.0), None, None), fitted(("sm0", "gp"), Config(latent_dim=4, **FAST), prep)
    assert all(np.array_equal(off.arrays[k], plain.arrays[k]) for k in off.arrays)                       # smoothify 0 is exactly the plain autoencoder
    assert all(h["smooth"] is None for h in off.history)
    on = fitted(("sm", "gp", 1.0), None, None)
    assert on.history[-1]["smooth"] is not None and on.history[-1]["smooth"] > 0


@pytest.mark.parametrize("basis", [{"kind": "bspline", "n_basis": [10, 8]}, {"kind": "cosine", "orders": [8, 6]}])
def test_smoothify_works_for_the_other_bases(basis):
    prep = eem(ife=True, basis=basis)
    r0 = fitted(("smb", str(basis), 0), Config(latent_dim=4, **FAST), prep)
    r1 = fitted(("smb", str(basis), 1), Config(latent_dim=4, smoothify=1.0, **FAST), prep)
    assert roughness_of(r1, prep) < 0.5 * roughness_of(r0, prep) and r1.validation["evr"] > 0.8
    ridge = fitted(("smb", str(basis), "ridge"), Config(latent_dim=4, smoothify=1.0, smoothify_metric="ridge", **FAST), prep)
    assert np.isfinite(ridge.validation["evr"])


# ----------------------------------------------------------------------------- the variational autoencoder
def planted_vae(beta=1.0, k=2, d=6, seed=0, **kw):
    A, _ = planted(n=60, P=30, k=k, noise=0.05, seed=seed)
    td = D.TrainData.from_array(A)
    res = T.fit(Config(family="vae", latent_dim=d, vae={"beta": beta}, n_restarts=1, **kw), td)
    return td, res


def test_vae_kl_is_nonnegative_and_codes_the_data_do_not_need_fall_to_the_prior():
    td, res = cached(("vae", "planted"), planted_vae)
    out = G.elbo(res.spec, res.arrays, td.X)
    assert (out["kl"] >= 0).all() and np.isfinite(out["elbo"]).all() and res.validation["kl"] >= 0
    mu, lv = fw.encode(res.spec, res.arrays, td.X)
    kl_dim = 0.5 * (mu ** 2 + np.exp(lv) - 1 - lv).mean(axis=0)
    assert (kl_dim >= -1e-12).all()
    h = G.latent_health(res.spec, res.arrays, td.X)
    assert (kl_dim < 0.05).sum() >= 3 and h["collapsed"].sum() >= 3 and h["unused"].sum() >= 3          # 2 real factors, 6 codes: at least 3 codes at the prior, ignored by the decoder
    assert (h["effect_share"][kl_dim < 0.05] < 0.01).all() and h["n_active"] <= 3 and res.validation["evr"] > 0.97
    assert "almost no effect" in h["words"] and "collapsed" in h["words"]
    assert h["coverage"]["inside95"] > 0.85                                                              # the used codes fill the prior


def test_lowering_beta_gives_more_information_in_the_codes_and_better_reconstructions():
    _, hi = cached(("vae", "planted"), planted_vae)
    _, lo = cached(("vae", "planted-beta0.1"), lambda: planted_vae(beta=0.1, k=3, d=4))
    _, hi3 = cached(("vae", "planted-beta1-k3"), lambda: planted_vae(beta=1.0, k=3, d=4))
    assert lo.validation["kl"] >= hi3.validation["kl"] - 1e-6 and lo.validation["error"] <= hi3.validation["error"] * 1.2 + 1e-4
    assert hi.sigma > 0 and hi.spec.vae["n_eff"] == T.auto_n_eff(D.TrainData.from_array(np.zeros((12, 30))), 6)


def test_vae_free_bits_warmup_and_a_fixed_sigma_run():
    A, _ = planted(n=40, P=20, k=2)
    td = D.TrainData.from_array(A)
    r = T.fit(Config(family="vae", latent_dim=3, n_restarts=1, epochs=120, vae={"free_bits": 0.5, "warmup": 30, "sigma": 0.2}), td)
    assert r.sigma == pytest.approx(0.2) and r.history[0]["beta"] < 0.1 and r.history[40]["beta"] == pytest.approx(1.0) and r.history[0]["sigma2"] == pytest.approx(0.04)


def test_the_mixture_prior_finds_the_clusters():
    rng = np.random.default_rng(0)
    centres = np.array([[-6.0, 0.0], [6.0, 0.0], [0.0, 8.0]])
    lab = np.repeat([0, 1, 2], 30)
    Z = centres[lab] + rng.standard_normal((90, 2))
    A = Z @ rng.standard_normal((2, 20)) + 0.05 * rng.standard_normal((90, 20))
    td = D.TrainData.from_array(A)
    res = T.fit(Config(family="vae", latent_dim=2, n_restarts=2, epochs=500, patience=150, vae={"prior": "mixture", "n_components": 3}), td)
    cl = tools.clusters(res.spec, res.arrays, res.samples["codes"], np.log(res.samples["codes_sd"] ** 2))
    assert cl["membership"].shape == (90, 3) and np.allclose(cl["membership"].sum(axis=1), 1)
    purity = sum(np.bincount(lab[cl["label"] == c], minlength=3).max() for c in range(3) if (cl["label"] == c).any()) / 90.0
    assert purity > 0.8, purity
    assert res.arrays["prior.pi"].shape == (3,) and res.spec.has_mixture()


# ----------------------------------------------------------------------------- determinism, missing values, k-fold, few samples
def test_the_same_seed_gives_the_same_network_and_another_seed_another_one():
    A, _ = planted(n=40, P=20, k=2)
    td = D.TrainData.from_array(A)
    a = T.fit(Config(latent_dim=2, epochs=60, n_restarts=2, seed=3), td)
    b = T.fit(Config(latent_dim=2, epochs=60, n_restarts=2, seed=3), td)
    c = T.fit(Config(latent_dim=2, epochs=60, n_restarts=2, seed=4), td)
    assert all(np.array_equal(a.arrays[k], b.arrays[k]) for k in a.arrays) and a.history == b.history and a.split == b.split
    assert any(not np.array_equal(a.arrays[k], c.arrays[k]) for k in a.arrays) and a.split != c.split


def test_missing_values_are_ignored_by_the_loss_and_filled_by_the_network():
    A, _ = planted(n=60, P=30, k=2, noise=0.02)
    rng = np.random.default_rng(1)
    holes = rng.random(A.shape) < 0.2
    Am = np.where(holes, np.nan, A)
    td = D.TrainData.from_array(Am)
    assert td.M is not None and abs((~td.M).mean() - 0.2) < 0.03
    cfg = Config(latent_dim=2, epochs=300, n_restarts=1, skip=True)
    res = T.fit(cfg, td)
    assert res.validation["evr"] > 0.9 and np.isfinite(res.samples["error"]).all()
    Xh = fw.reconstruct(res.spec, res.arrays, td.X, td.M)[0]
    truth = A - np.nanmean(Am, axis=0)
    err_missing = np.sqrt(np.mean((Xh[holes] - truth[holes]) ** 2))
    assert err_missing < 0.5 * np.sqrt(np.mean(truth[holes] ** 2))                                       # the unobserved values are predicted much better than by the mean
    # what stands at an unobserved coordinate is never read
    td2 = D.TrainData(np.where(td.M, td.X, 1.0e6), td.M, td.omega, td.blocks, td.q, td.sd, td.ids, td.info, td.q_rough)
    res2 = T.fit(cfg, td2)
    assert all(np.array_equal(res.arrays[k], res2.arrays[k]) for k in res.arrays)


def test_few_samples_use_k_fold_validation_and_every_sample_gets_an_out_of_fold_error():
    A, _ = planted(n=20, P=20, k=2)
    td = D.TrainData.from_array(A)
    res = T.fit(Config(latent_dim=2, epochs=120, n_restarts=2), td)
    assert res.split["mode"] == "kfold" and len(res.split["folds"]) == 5 and res.validation["mode"] == "kfold" and len(res.fold_info) == 5
    s = res.samples
    assert s["heldout"].all() and np.isfinite(s["error_heldout"]).all() and set(s["fold"]) == {0, 1, 2, 3, 4} and np.isfinite(s["error_baseline"]).all()
    assert any("5-fold" in n for n in res.notes) and res.validation["evr"] > 0.5
    assert sorted(i for f in res.split["folds"] for i in f) == list(range(20))


def test_too_few_samples_are_refused_in_plain_english():
    td = D.TrainData.from_array(planted(n=9, P=10, k=2)[0])
    with pytest.raises(AutoencoderError, match="at least 12 samples"):
        T.fit(Config(latent_dim=2), td)
    assert T.fit(Config(latent_dim=2, allow_few_samples=True, epochs=30, n_restarts=1), td).split["mode"] == "kfold"
    with pytest.raises(AutoencoderError, match="smaller than the number of inputs"):
        T.fit(Config(latent_dim=12), D.TrainData.from_array(planted(n=20, P=10, k=2)[0]))


# ----------------------------------------------------------------------------- the history file
def test_the_history_file_grows_one_line_at_a_time_and_an_incomplete_line_is_ignored(tmp_path):
    A, _ = planted(n=40, P=20, k=2)
    td = D.TrainData.from_array(A)
    path = tmp_path / "h" / "hist.jsonl"
    res = T.fit(Config(latent_dim=2, epochs=40, n_restarts=2), td, history_path=path)
    recs, nxt = T.read_history(path)
    ev = [r["event"] for r in recs]
    assert ev[0] == "start" and ev[-1] == "done" and ev.count("run_start") == 2 and ev.count("run_end") == 2 and ev.count("epoch") == sum(len(h["history"]) for h in res.all_histories)
    ep = [r for r in recs if r["event"] == "epoch"]
    assert {"restart", "fold", "epoch", "train", "val", "rec", "kl", "smooth", "beta", "sigma2", "lr", "best"} <= set(ep[0]) and ep[0]["restart"] == 0
    more, nxt2 = T.read_history(path, since=nxt)
    assert more == [] and nxt2 == nxt
    part, n3 = T.read_history(path, since=5)
    assert len(part) == nxt - 5
    with open(path, "ab") as fh:
        fh.write(b'{"event":"epoch","epo')                                                              # a line that is being written
    recs2, n4 = T.read_history(path)
    assert len(recs2) == len(recs) and n4 == nxt
    assert T.read_history(tmp_path / "missing.jsonl", since=3) == ([], 3)


# ----------------------------------------------------------------------------- torch <-> numpy
@pytest.mark.parametrize("skip", [False, True])
def test_the_numpy_network_is_the_trained_torch_network(skip):
    A, _ = planted(n=50, P=24, k=2)
    td = D.TrainData.from_array(A)
    res = T.fit(Config(latent_dim=2, skip=skip, family="ae", epochs=100, n_restarts=1), td)
    import torch
    enc, dec = torch_modules(res.spec, res.arrays)                                                       # the same arrays as float64 torch modules
    with torch.no_grad():
        mu_t, _ = enc(td.X)
        rec_t = dec(mu_t).numpy()
    xh, mu, _ = fw.reconstruct(res.spec, res.arrays, td.X)
    assert np.allclose(mu, mu_t.numpy(), rtol=1e-9, atol=1e-10) and np.allclose(xh, rec_t, rtol=1e-9, atol=1e-10)
    if res.best_epoch >= 0:                                                                              # the float32 training numbers agree with the numpy network (the saved model)
        vi = np.asarray(res.split["val"])
        assert res.validation["error"] == pytest.approx(res.history[res.best_epoch]["rec_val"], rel=2e-4)
        assert float(res.samples["error"][vi].mean()) == pytest.approx(res.history[res.best_epoch]["rec_val"], rel=2e-4)


# ----------------------------------------------------------------------------- speed (SPEC section 4) on the example data of 64 samples
def test_speed_on_the_example_data():
    """Functional mode under about 15 s and pixel mode under about 60 s for 300 epochs (one start, no early stop) on a laptop CPU.  Other programs share the machine, so the limit is
    met when EITHER the wall time is within the limit stretched by the load of the machine OR the CPU time of the process is within the limit itself."""
    X, ex, em, ids, wl, Aabs, _ = example_eems(n=64, coarse=False, ife=True)
    out, limits = {}, {"functional": 15.0, "pixels": 60.0}
    for name, inp in (("functional", {"fluorescence": {"basis": GP}}), ("pixels", {"fluorescence": {"basis": PIXELS}})):
        prep = D.prepare(D.variables_from_arrays(X=X, ex=ex, em=em, ids=ids, input=inp), inp)
        t0, c0 = time.time(), time.process_time()
        res = T.fit(Config(latent_dim=4, epochs=300, patience=300, n_restarts=1), prep)
        out[name] = (time.time() - t0, time.process_time() - c0, prep.P, res.validation["evr"], res.validation["baseline"]["evr"])
    f = load_factor()
    print("\nspeed (load factor %.1f):" % f, {k: "%.1f s wall, %.1f s cpu, P=%d, AE %.4f vs PCA %.4f" % v for k, v in out.items()})
    for name, (wall, cpu, P, evr, base) in out.items():
        assert wall < limits[name] * f or cpu < limits[name], (name, wall, cpu, f)
        assert evr >= base - 1e-3
