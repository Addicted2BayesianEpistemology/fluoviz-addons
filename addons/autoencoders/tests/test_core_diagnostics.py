"""The diagnostics and their words: thresholds, verdicts, health of the code space, stability, anomaly limits (one answer with the saved model), scans.  The parts that do not train need no PyTorch."""
import types

import numpy as np
import pytest

from core_support import GP, cached, planted, prepared
from fluoviz_addon_autoencoders.core import data as D, diagnostics as G, forward as fw, linear as lin
from fluoviz_addon_autoencoders.core.spec import AutoencoderError, Config
from test_core_forward import random_net


def fake_result(evr_train=0.985, evr_val=0.98, error_train=0.015, error_val=0.02, best_epoch=100, epochs_run=160, epochs=600, restarts=(0.98, 0.979, 0.981), mode="holdout", d=4, patience=60):
    spec, _ = random_net("ae", P=20, d=d)
    runs = [{"restart": i, "evr_val": v, "evr_train": 0.99, "chosen": i == 0} for i, v in enumerate(restarts)]
    return types.SimpleNamespace(validation={"evr": evr_val, "evr_train": evr_train, "error": error_val, "error_train": error_train, "n_val": 12}, restarts={"n": len(runs), "runs": runs},
                                 config={"epochs": epochs, "patience": patience}, split={"mode": mode}, epochs_run=epochs_run, best_epoch=best_epoch, spec=spec)


# ----------------------------------------------------------------------------- the words and their thresholds
def test_training_verdict_rules():
    v = G.training_verdict(fake_result())
    assert v["overfit"] == "none" and "No sign of overfitting" in v["overfit_words"] and "stopped at epoch 160" in v["stopping_words"] and "epoch 101" in v["stopping_words"] and "agree" in v["restarts_words"]
    assert "good" in v["quality_words"] and "98.0 %" in v["quality_words"] and v["words"].count(".") >= 4
    mild = G.training_verdict(fake_result(evr_train=0.995, evr_val=0.985, error_train=0.005, error_val=0.015))
    assert mild["overfit"] == "mild" and "Mild overfitting" in mild["overfit_words"]
    clear = G.training_verdict(fake_result(evr_train=0.99, evr_val=0.90, error_train=0.01, error_val=0.10))
    assert clear["overfit"] == "clear" and "memorises" in clear["overfit_words"] and "fewer codes" in clear["overfit_words"]
    tiny = G.training_verdict(fake_result(evr_train=0.9999, evr_val=0.9995, error_train=0.0001, error_val=0.0005))
    assert tiny["overfit"] == "none"                                                                      # five times a tiny error is still tiny
    assert "still near the end" in G.training_verdict(fake_result(best_epoch=590, epochs_run=600))["stopping_words"]
    assert "all 600 epochs" in G.training_verdict(fake_result(best_epoch=300, epochs_run=600))["stopping_words"]
    start = G.training_verdict(fake_result(best_epoch=-1, epochs_run=61))
    assert "could not improve on its PCA starting point" in start["stopping_words"] and "straight-line" in start["stopping_words"] and "epoch 0" not in start["stopping_words"]
    assert "k-fold validation" in G.training_verdict(fake_result(mode="kfold", epochs_run=90))["stopping_words"]
    assert "differ" in G.training_verdict(fake_result(restarts=(0.98, 0.90, 0.95)))["restarts_words"]
    assert "Only one random start" in G.training_verdict(fake_result(restarts=(0.98,)))["restarts_words"]


def test_quality_verdict_words():
    assert "excellent" in G.quality_verdict(0.995, 12, 4) and "good" in G.quality_verdict(0.96, 12, 4) and "fair" in G.quality_verdict(0.92, 12, 4) and "More codes" in G.quality_verdict(0.92, 12, 4)
    assert "poor" in G.quality_verdict(0.5, 12, 1) and "1 code)" in G.quality_verdict(0.5, 12, 1) and "could not be judged" in G.quality_verdict(float("nan"), 0, 1)


def baseline_result(e_net, e_pca, evr_net, evr_pca):
    n = len(e_net)
    return types.SimpleNamespace(samples={"error_heldout": np.asarray(e_net, float), "error_baseline": np.asarray(e_pca, float), "null": np.ones(n)},
                                 validation={"evr": evr_net, "baseline": {"name": "Functional PCA", "k": 4, "evr": evr_pca}})


def test_baseline_comparison_rules():
    rng = np.random.default_rng(0)
    base = rng.uniform(0.01, 0.03, 40)
    same = G.compare_baseline(baseline_result(base * (1 + 0.01 * rng.standard_normal(40)), base, 0.98, 0.98))
    assert same["better"] == "same" and "linear model" in same["verdict"] and "Functional PCA" in same["verdict"]
    better = G.compare_baseline(baseline_result(base * 0.5, base, 0.99, 0.98))
    assert better["better"] == "network" and better["relative_improvement"] == pytest.approx(0.5) and "nonlinearity is worth it" in better["verdict"]
    worse = G.compare_baseline(baseline_result(base * 2.0, base, 0.96, 0.98))
    assert worse["better"] == "baseline" and "does worse" in worse["verdict"] and "Use the Functional PCA" in worse["verdict"]
    immaterial = G.compare_baseline(baseline_result(base * 1.5, base, 0.9985, 0.999))              # 50 % more error, but of a 0.1 % error: not worth a word
    assert immaterial["better"] == "same"
    few = G.compare_baseline(baseline_result(base[:3], base[:3], 0.98, 0.98))
    assert few["better"] == "same" and "too few" in few["verdict"]


def test_smoothness_report_rules():
    A, _ = planted(n=30, P=20, k=2)
    td = D.TrainData.from_array(A)
    rough = np.random.default_rng(0).standard_normal(A.shape) * td.X.std()
    rep = G.smoothness_report(td, td.X, td.X * 0.3)
    assert rep["blocks"][0]["verdict"] == "smoother than the samples" and rep["blocks"][0]["ratio"] == pytest.approx(0.09) and "smoothify is off" in rep["words"]
    assert G.smoothness_report(td, td.X, td.X * 1.02)["blocks"][0]["verdict"] == "about as rough as the samples"
    r = G.smoothness_report(td, td.X, rough * 2, 0.0)
    assert r["blocks"][0]["verdict"] == "rougher than the samples" and "smoothify scan" in r["words"]
    assert "smoothify is 0.5" in G.smoothness_report(td, td.X, td.X, 0.5)["words"]


def test_flags_limits_and_typicality_are_the_saved_models_functions():
    from fluoviz_addon_autoencoders.kinds import stats as ks
    rng = np.random.default_rng(1)
    err = np.exp(rng.normal(np.log(0.01), 0.5, 40))
    lim = G.default_limit(err, 0.99)
    assert lim["limit"] == ks.error_limits(err)["99"] and G.default_limit(err, 0.95)["limit"] == ks.error_limits(err)["95"] < lim["limit"] and lim["source"] == "held-out" and "99 %" in lim["words"]
    assert G.default_limit(np.array([np.nan, np.nan]), 0.99, fallback=err)["source"] == "training"
    codes = rng.standard_normal((50, 3))
    new = np.array([[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]])
    st = ks.code_stats(codes)
    assert np.allclose(G.typicality(codes, new), ks.typicality(new, st)) and G.typicality(codes, new)[0] > 0.5 and G.typicality(codes, new)[1] < 1e-4
    flag, words = G.flag_samples(np.array([0.005, 0.5, 0.005, 0.5]), 0.05, np.array([0.5, 0.5, 0.001, 0.001]))
    assert flag.tolist() == [0, 1, 2, 3] and words == ["inside the limits", "reconstruction error above the limit", "unusual code", "both"]
    assert G.robust_limit(err, "mad", 3.0)["limit"] > np.median(err) and G.robust_limit(err, "quantile", level=0.9)["limit"] == pytest.approx(np.quantile(err, 0.9))
    with pytest.raises(AutoencoderError, match="mad or quantile"):
        G.robust_limit(err, "zscore")


# ----------------------------------------------------------------------------- the linear baseline and the stability
def test_pca_curve_is_monotone_and_k_zero_is_the_mean():
    from fluoviz_addon_autoencoders.core import train
    A, _ = planted(n=50, P=20, k=3, noise=0.05)
    td = D.TrainData.from_array(A)
    cfg = Config(latent_dim=3)
    sp = train.make_split(td.n, cfg)
    pc = G.pca_curve(td, sp, 6)
    e = np.asarray(pc["error_val"])
    assert pc["k"] == list(range(7)) and e[0] == pytest.approx(1.0) and (np.diff(e) <= 1e-9).all() and e[3] < 0.02 and e[2] > 5 * e[3]               # three factors: the error falls sharply at k = 3
    assert np.asarray(pc["error_train"])[3] < e[3] + 1e-9 and np.isfinite(pc["error_val_se"]).all() and pc["mode"] == "holdout"
    kf = G.pca_curve(td.subset(range(20)), train.make_split(20, cfg), 4)
    assert kf["mode"] == "kfold" and kf["error_val"][0] == pytest.approx(1.0) and kf["error_val"][3] < 0.1


def test_stability_of_the_codes_across_runs():
    rng = np.random.default_rng(0)
    base = rng.standard_normal((60, 3))
    R = np.linalg.qr(rng.standard_normal((3, 3)))[0]
    same = np.stack([base, base @ R, (base @ R) * 2 + 1])
    st = G.stability_from_codes(same)
    assert st["mean"] > 0.999 and st["per_code"].min() > 0.999 and "reproducible" in st["verdict"] and "NOT" not in st["verdict"] and np.allclose(np.diag(st["matrix"]), 1)
    assert st["procrustes"][0, 1] > 0.999
    curved = np.stack([base, np.hstack([np.sin(base[:, :1] * 3), base[:, 1:] * 0 + rng.standard_normal((60, 2))])])
    bad = G.stability_from_codes(curved)
    assert bad["mean"] < 0.8 and "NOT reproducible" in bad["verdict"] and bad["per_code"].min() < 0.5
    part = G.stability_from_codes(np.stack([base, np.hstack([base[:, :2], rng.standard_normal((60, 1))])]))
    assert "partly" in part["verdict"] and "z3" in part["verdict"]
    assert "Only one training run" in G.stability_from_codes(base[None])["verdict"]


# ----------------------------------------------------------------------------- the health of the code space (numpy: hand-made networks)
def test_latent_health_names_unused_and_collapsed_codes():
    spec, arrays = random_net("vae", P=16, d=4, enc=(10,), seed=2)
    arrays["dec.W0"] = arrays["dec.W0"].copy()
    arrays["dec.W0"][:, 2:] = 0.0                                                                        # the decoder reads only codes 1 and 2
    arrays["out.weights"] = np.full(16, 1.0 / 16)
    rng = np.random.default_rng(1)
    X = rng.standard_normal((50, 16))
    mu, lv = fw.encode(spec, arrays, X)
    mu[:, 2:] *= 0.02                                                                                    # codes 3 and 4 sit at the prior
    lv[:, 2:] = 0.0
    h = G.latent_health(spec, arrays, X, codes=mu, logvar=lv)
    assert h["unused"].tolist() == [False, False, True, True] and h["n_active"] == 2 and h["kl_per_dim"][2:].max() < 0.05
    assert h["collapsed"].tolist() == [False, False, True, True] and "2 of the 4 codes are used" in h["words"] and "z3, z4" in h["words"] and "collapsed" in h["words"]
    ae, ae_arrays = random_net("ae", P=16, d=4, enc=(10,), seed=2)
    ae_arrays["dec.W0"] = ae_arrays["dec.W0"].copy()
    ae_arrays["dec.W0"][:, 3] = 0.0
    ae_arrays["out.weights"] = np.full(16, 1.0 / 16)
    ha = G.latent_health(ae, ae_arrays, X)
    assert ha["unused"].tolist() == [False, False, False, True] and ha["kl_per_dim"] is None and ha["kind"] == "ae" and "z4" in ha["words"] and "collapsed" not in ha["words"]
    assert np.allclose(np.diag(ha["correlation"]), 1) and ha["effective_dim"] <= 3.0 + 1e-9


def test_calibration_and_elbo_need_a_vae():
    spec, arrays = random_net("ae")
    with pytest.raises(AutoencoderError, match="variational"):
        G.calibration(spec, arrays, np.zeros((3, 20)))
    with pytest.raises(AutoencoderError, match="variational"):
        G.elbo(spec, arrays, np.zeros((3, 20)))


# ----------------------------------------------------------------------------- with trained networks (PyTorch)
def trained(key, **kw):
    torch = pytest.importorskip("torch")
    from fluoviz_addon_autoencoders.core import train as T

    def build():
        prep = cached(("diag", "eem"), lambda: prepared({"fluorescence": {"basis": GP}}, n=40, ife=True))
        return prep, T.fit(Config(latent_dim=3, epochs=200, patience=60, n_restarts=2, **kw), prep)
    return cached(("diag-fit", key), build)


def test_assess_a_trained_autoencoder_end_to_end():
    prep, res = trained("ae")
    a = G.assess(prep, res)
    rec = a["reconstruction"]
    assert rec["error"].shape == (prep.n,) and np.allclose(rec["error"], res.samples["error"], rtol=1e-6, atol=1e-9) and rec["evr"] > 0.97
    assert np.allclose(rec["r2"], 1 - rec["error"]) and rec["rmse"].shape == (prep.n,) and rec["local_r2"]["fluorescence"].shape == (prep.blocks[0].domain.m,) and np.nanmax(rec["local_r2"]["fluorescence"]) <= 1 + 1e-9
    assert rec["rmse_observed"]["fluorescence"].mean() > rec["rmse"].mean()                              # the measured values include the noise the network is not asked to rebuild
    assert a["baseline"]["name"] == "Functional PCA" and a["baseline"]["better"] in ("network", "same", "baseline") and a["training"]["words"]
    assert a["limit"]["limit"] > 0 and a["limit"]["source"] in ("held-out", "training") and a["flag"].shape == (prep.n,) and set(a["flag_words"]) <= {"inside the limits", "reconstruction error above the limit", "unusual code", "both"}
    assert a["roughness"]["input"]["fluorescence"] == pytest.approx(1.0) and a["roughness"]["reconstruction"]["fluorescence"] < 1.0
    assert 0 <= a["typicality"].min() and a["typicality"].max() <= 1 and a["health"]["n_active"] >= 2 and "kl" not in a
    w = G.worst_samples(prep.ids, rec["error"], res.samples["heldout"], 3)
    assert len(w) == 3 and w[0]["error"] >= w[1]["error"] >= w[2]["error"] and w[0]["r2"] == pytest.approx(1 - w[0]["error"])
    sm = G.smoothness_report(prep.train_data("smooth"), prep.X, rec["X_hat"])
    assert sm["blocks"][0]["name"] == "fluorescence" and sm["words"]


def test_assess_a_trained_vae_has_kl_elbo_and_calibration():
    prep, res = trained("vae", family="vae")
    a = G.assess(prep, res)
    assert (a["kl"] >= 0).all() and np.isfinite(a["elbo"]).all() and set(a["calibration"]) >= {"coverage68", "coverage95", "mean_z2", "per_code", "verdict"} and 0 <= a["calibration"]["coverage95"] <= 1
    assert a["calibration"]["verdict"] and a["health"]["kind"] == "vae" and a["health"]["kl_per_dim"].shape == (3,)


@pytest.mark.parametrize("name", ["latent_scan", "smoothify_scan", "stability_scan"])
def test_the_scans_run_in_the_worker_and_say_what_they_found(name, tmp_path):
    pytest.importorskip("torch")
    from fluoviz_addon_autoencoders.core import worker as W
    prep = cached(("diag", "eem"), lambda: prepared({"fluorescence": {"basis": GP}}, n=40, ife=True))
    cfg = Config(latent_dim=3, epochs=60, patience=30, n_restarts=1)
    msgs = []
    kw = {"latent_scan": {"dims": [1, 2, 3, 4]}, "smoothify_scan": {"values": [0.0, 0.3, 3.0]}, "stability_scan": {"n_runs": 3}}[name]
    out = W.run_in_worker(name, {"config": cfg, "data": prep.train_data(cfg.smoothify_metric), **kw}, progress=lambda f, m: msgs.append(f), history_path=tmp_path / "h.jsonl")
    assert msgs and msgs[-1] >= 0.9 and out["scan"] == name.replace("_scan", "") and out["words"]
    if name == "latent_scan":
        v = np.asarray(out["families"]["ae"]["val"])
        assert out["dims"] == [1, 2, 3, 4] and len(out["baseline"]["val"]) == 4 and v[0] > v[2] and out["recommended"]["latent_dim"] in (1, 2, 3, 4) and out["baseline"]["name"] == "Functional PCA"
        assert out["recommended"]["latent_dim"] >= 2 and "Recommended" in out["words"] and np.asarray(out["baseline"]["val"])[0] > np.asarray(out["baseline"]["val"])[3]
    elif name == "smoothify_scan":
        assert out["values"] == [0.0, 0.3, 3.0] and out["roughness"][0] > out["roughness"][1] > out["roughness"][2] and out["error"][2] > out["error"][0] and out["suggested"] in (0.0, 0.3, 3.0)
    else:
        assert out["n_runs"] == 3 and 0 <= out["alignment"]["mean"] <= 1 and np.asarray(out["matrix"]).shape == (3, 3) and out["verdict"] in out["words"] and out["recon_spread"] >= 0
