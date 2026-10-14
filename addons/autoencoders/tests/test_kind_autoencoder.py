"""The saved kind ``autoencoder``: freeze -> file -> thaw -> apply reproduces the codes of the result, new samples and other grids, planted anomalies, gaps and ``min_coverage``,
thresholds as options, the variational terms, two variables, the payload and ``inspect``.  Networks need no PyTorch (the exact linear autoencoder, or random weights)."""
import numpy as np
import pytest
from fluoviz.analysis.common import Progress
from fluoviz.analysis.frozen import FrozenError, resolve_options, thaw

import kind_support as K


@pytest.fixture(scope="module", params=["pixels", "bspline", "cosine", "gp"])
def trained(request):
    X, A = K.small_eem(n=30, seed=3)
    prep, inp = K.prepare(X, kind=request.param)
    spec, arrays = K.linear_network(prep, d=3)
    frozen, model, ev = K.make_frozen(prep, spec, arrays)
    saved = K.roundtrip(K.make_saved(prep, spec, arrays))
    return {"X": X, "A": A, "prep": prep, "model": model, "ev": ev, "saved": saved, "kind": request.param, "spec": spec, "arrays": arrays}


def test_round_trip_reproduces_the_codes_of_the_result(trained):
    t = trained
    out = K.run(t["saved"], K.apply_data(t["X"], ids=t["prep"].ids), min_coverage=0.0)
    codes = np.column_stack([K.column(out, f"z{j + 1}") for j in range(3)])
    assert np.allclose(codes, t["ev"].codes, rtol=1e-9, atol=1e-10)                         # the codes the page showed, to numpy precision
    # and they are the network coordinates the training used (the training side and the apply side cannot drift apart)
    assert np.allclose(t["ev"].X, t["prep"].X, atol=1e-9)
    assert np.allclose(K.column(out, "recon_r2"), 1 - t["ev"].error)
    assert set(out.variables) == {"z1", "z2", "z3", "recon_rmse", "recon_r2", "typicality", "flag", "status"} and set(out.var_info) == set(out.variables)
    assert out.status.shape == (30,) and set(out.status) <= {"ok", "outside"}


def test_linear_autoencoder_rebuilds_three_fluorophores(trained):
    ev = trained["ev"]
    assert float(np.median(1 - ev.error)) > 0.9                                              # 3 codes for 3 fluorophores


def test_another_project_with_other_names_and_order_gives_the_same_numbers(trained):
    t = trained
    order = np.random.default_rng(0).permutation(30)
    out = K.run(t["saved"], K.apply_data(t["X"][order], ids=[f"other_{i}" for i in order]), min_coverage=0.0)
    codes = np.column_stack([K.column(out, f"z{j + 1}") for j in range(3)])
    assert np.allclose(codes, t["ev"].codes[order], rtol=1e-9, atol=1e-10)
    one = K.run(t["saved"], K.apply_data(t["X"][5:6], ids=["only"]), min_coverage=0.0)         # one sample alone = the same sample in a batch
    assert np.allclose([K.column(one, f"z{j + 1}")[0] for j in range(3)], t["ev"].codes[5], rtol=1e-9, atol=1e-10)


def test_a_planted_extra_band_is_outside_and_ordinary_samples_are_not(trained):
    t = trained
    X2, _ = K.small_eem(n=40, seed=11)                                                        # fresh samples from the same process
    out = K.run(t["saved"], K.apply_data(X2), min_coverage=0.0)
    assert (out.status == "outside").mean() < 0.15
    bad = K.run(t["saved"], K.apply_data(K.extra_band(X2[:6], size=1.0)), min_coverage=0.0)
    assert (bad.status == "outside").all()
    assert all("reconstruction error" in r[0] and "above the 99 % limit" in r[0] for r in bad.reasons)
    assert np.all(K.column(bad, "recon_r2") < K.column(out, "recon_r2").min() + 0.5)
    assert all(f == "reconstruction error above the limit" for f in bad.variables["flag"])


def test_gaps_and_min_coverage(trained):
    t = trained
    X = t["X"][:6].copy()
    rng = np.random.default_rng(1)
    obs = np.isfinite(X)
    X2 = X.copy()
    for i in range(6):
        idx = np.flatnonzero(obs[i].ravel())
        drop = rng.choice(idx, size=int(0.1 * idx.size), replace=False)
        flat = X2[i].ravel()
        flat[drop] = np.nan
    out = K.run(t["saved"], K.apply_data(X2), min_coverage=0.8)
    assert (out.status != "unusable").all()
    full = K.run(t["saved"], K.apply_data(X), min_coverage=0.8)
    assert np.abs(K.column(out, "z1") - K.column(full, "z1")).max() < 0.3 * np.abs(K.column(full, "z1")).max() + 1e-9      # a sample with gaps is read on its own pixels: close
    # too little measured: unusable, with the reason, and NaN outputs
    X3 = X.copy()
    X3[0][:, :9] = np.nan
    X3[1] = np.nan
    bad = K.run(t["saved"], K.apply_data(X3), min_coverage=0.8)
    assert list(bad.status[:2]) == ["unusable", "unusable"] and (bad.status[2:] != "unusable").all()
    assert "at least 80 %" in bad.reasons[0][0] and "measured" in bad.reasons[0][0]
    assert "no fluorescence measured" in bad.reasons[1][0]
    assert np.isnan(K.column(bad, "z1")[:2]).all() and np.isnan(K.column(bad, "recon_rmse")[:2]).all()
    low = K.run(t["saved"], K.apply_data(X3), min_coverage=0.1)
    assert low.status[0] != "unusable"                                                         # the share is an option of the application


def test_thresholds_are_options(trained):
    t = trained
    X2, _ = K.small_eem(n=40, seed=12)
    ap = thaw(t["saved"])
    names = [o["name"] for o in ap.options()]
    assert names == ["limit", "typicality"] and all(o["help"] for o in ap.options())
    out99 = K.run(t["saved"], K.apply_data(X2), min_coverage=0.0, limit="99")
    out95 = K.run(t["saved"], K.apply_data(X2), min_coverage=0.0, limit="95")
    assert (out95.status == "outside").sum() >= (out99.status == "outside").sum()
    assert out95.payload["limits"][0]["value"] < out99.payload["limits"][0]["value"]
    # the code check of a deterministic autoencoder is off by default and can be switched on
    off = K.run(t["saved"], K.apply_data(X2), min_coverage=0.0)
    on = K.run(t["saved"], K.apply_data(X2), min_coverage=0.0, typicality=0.2)
    assert (on.status == "outside").sum() >= (off.status == "outside").sum()
    assert any("code is unusual" in " ".join(r) for r in on.reasons) or (on.status == "outside").sum() == (off.status == "outside").sum()
    with pytest.raises(FrozenError, match="Unknown setting"):
        K.run(t["saved"], K.apply_data(X2), nonsense=1)
    with pytest.raises(FrozenError, match="choose one of"):
        K.run(t["saved"], K.apply_data(X2), limit="90")
    with pytest.raises(FrozenError, match="at most"):
        K.run(t["saved"], K.apply_data(X2), typicality=2.0)


def test_payload_describe_and_inspect(trained):
    t = trained
    X2, _ = K.small_eem(n=8, seed=13)
    out = K.run(t["saved"], K.apply_data(X2), min_coverage=0.0)
    pl = out.payload
    assert [a["name"] for a in pl["plot"]["axes"]] == ["z1", "z2", "z3"] and len(pl["plot"]["axes"][0]["ref"]) == 30 and len(pl["plot"]["axes"][0]["new"]) == 8
    d0 = pl["diagnostics"][0]
    assert d0["x"]["limit"] > 0 and len(d0["x"]["new"]) == 8 and len(d0["y"]["ref"]) == 30 and d0["y"]["limit"] is None                 # no typicality limit for a plain AE
    assert pl["images"]["mean"].shape == (1, 24, 16) and pl["images"]["directions"].shape == (3, 24, 16) and pl["mask"].shape == (24, 16)
    assert np.isnan(pl["images"]["mean"][0][K.below_rayleigh()]).all() and np.isfinite(pl["images"]["mean"][0][~K.below_rayleigh()]).any()
    assert "autoencoder" in out.text and "8 new samples" in out.text and "Method" in out.text
    ap = thaw(t["saved"])
    desc = ap.describe()
    assert desc["kind_label"].lower().endswith("autoencoder") and desc["training"]["Samples"] == 30 and desc["limits"] and desc["caveats"] and desc["inputs"]
    assert "autoencoder" in desc["what"]
    info = ap.inspect(K.apply_data(X2), {**resolve_options(ap.options(), None), "min_coverage": 0.0}, 2)
    assert set(info["images"]) == {"measured", "reconstruction", "residual", "mean"} and info["images"]["measured"].shape == (24, 16) and "residual" in info["signed"]
    assert info["facts"]["code"].shape == (3,) and "N02" in info["text"] and info["mask"].shape == (24, 16)
    with pytest.raises(FrozenError, match="no such sample"):
        ap.inspect(K.apply_data(X2), {"limit": "99", "typicality": 0.0, "min_coverage": 0.0}, 99)
    bad = np.full_like(X2[:1], np.nan)
    with pytest.raises(FrozenError, match="no fluorescence measured"):
        ap.inspect(K.apply_data(bad), {"limit": "99", "typicality": 0.0, "min_coverage": 0.0}, 0)


def test_the_file_holds_no_class_of_the_addon_and_wrong_grids_are_refused(trained):
    t = trained
    doc, arrays = t["saved"].to_document()
    text = str(doc["meta"]["model"])
    assert "FrozenElement" in text and "FrameBlock" not in text and "'Frame'" not in text         # only classes of FluoViz itself: the file opens without the add-on
    X = t["X"][:3]
    other = K.apply_data(X)
    other.em = other.em + 1.0
    with pytest.raises(FrozenError, match="wavelength grid"):
        K.run(t["saved"], other)
    wrong = K.apply_data(X[:, :-1, :], ids=["a", "b", "c"])
    with pytest.raises(FrozenError, match="different wavelength grid"):
        K.run(t["saved"], wrong)


def test_a_damaged_model_is_refused_politely(trained):
    from fluoviz.analysis.frozen import SavedModel
    saved = trained["saved"]
    bad = SavedModel(kind=saved.kind, title="x", meta={**saved.meta, "model": {"spec": saved.meta["model"]["spec"], "arrays": {}, "frame": saved.meta["model"]["frame"]}})
    with pytest.raises(FrozenError, match="damaged"):
        thaw(bad)
    bad2 = SavedModel(kind=saved.kind, title="x", meta={k: v for k, v in saved.meta.items() if k != "reference"})
    with pytest.raises(FrozenError, match="damaged"):
        thaw(bad2)


# ----------------------------------------------------------------------------- variational, mixture, two variables
def test_a_variational_autoencoder_publishes_its_extra_variables():
    X, _ = K.small_eem(n=24, seed=5)
    prep, _ = K.prepare(X, kind="bspline")
    spec, arrays = K.linear_network(prep, d=3, family="vae")
    saved = K.roundtrip(K.make_saved(prep, spec, arrays))
    out = K.run(saved, K.apply_data(X), min_coverage=0.0)
    assert {"z1_sd", "z2_sd", "z3_sd", "kl", "elbo"} <= set(out.variables) and np.isfinite(K.column(out, "kl")).all()
    assert np.allclose(K.column(out, "z1_sd"), np.exp(-2.0), rtol=1e-6)                       # logvar bias -4
    assert [o["name"] for o in thaw(saved).options()] == ["limit", "typicality"] and thaw(saved).options()[1]["default"] == 0.01
    # a much higher elbo for a sample the network rebuilds well than for a planted one
    bad = K.run(saved, K.apply_data(K.extra_band(X[:4], 1.0)), min_coverage=0.0)
    assert K.column(bad, "elbo").max() < np.median(K.column(out, "elbo"))
    # an atypical code is outside even when the reconstruction is fine: put a far code through the typicality check
    ap = thaw(saved)
    far = K.apply_data(X[:3] * 1.0)
    res = ap.apply(far, {**resolve_options(ap.options(), {"typicality": 0.5}), "min_coverage": 0.0}, Progress())
    assert (res.status == "outside").any()                                                    # at 0.5 a good share of ordinary samples counts as unusual


def test_a_mixture_prior_publishes_clusters():
    X, _ = K.small_eem(n=24, seed=6)
    prep, _ = K.prepare(X, kind="cosine")
    spec, arrays = K.random_network(prep, d=2, family="vae", prior="mixture", K=3)
    saved = K.roundtrip(K.make_saved(prep, spec, arrays))
    out = K.run(saved, K.apply_data(X), min_coverage=0.0)
    assert {"cluster", "membership"} <= set(out.variables)
    assert set(np.asarray(out.variables["cluster"]).tolist()) <= {"cluster 1", "cluster 2", "cluster 3"}
    m = K.column(out, "membership")
    assert (m > 0).all() and (m <= 1 + 1e-12).all()


def test_two_variables_fluorescence_and_absorbance():
    X, A = K.small_eem(n=30, seed=7)
    ab = K.small_abs(A)
    prep, _ = K.prepare(X, ab, kind="bspline")
    spec, arrays = K.linear_network(prep, d=3)
    saved = K.roundtrip(K.make_saved(prep, spec, arrays))
    out = K.run(saved, K.apply_data(X, ab), min_coverage=0.0)
    ev = AEmodel(saved).evaluate(K.apply_data(X, ab), 0.0)
    assert np.allclose(np.column_stack([K.column(out, f"z{j + 1}") for j in range(3)]), ev.codes)
    assert out.payload["images"]["mean"].shape == (1, 24, 16) and out.payload["curves"]["x"].shape == (40,) and len(out.payload["curves"]["series"]) == 4
    # a sample without an absorbance spectrum cannot be used by a model of both
    ab2 = ab.copy()
    ab2[0] = np.nan
    r = K.run(saved, K.apply_data(X, ab2), min_coverage=0.0)
    assert r.status[0] == "unusable" and "no absorbance measured" in r.reasons[0][0] and (r.status[1:] != "unusable").all()
    with pytest.raises(FrozenError, match="no absorbance was given"):
        K.run(saved, K.apply_data(X))
    # a planted feature in the absorbance only is found and named
    ab3 = ab.copy()
    ab3[:5] += 0.4 * np.exp(-0.5 * ((K.ABS_WL - 450.0) / 15.0) ** 2)
    r3 = K.run(saved, K.apply_data(X[:5], ab3[:5]), min_coverage=0.0)
    assert (r3.status == "outside").all() and "absorbance" in r3.reasons[0][0]
    info = thaw(saved).inspect(K.apply_data(X, ab), {**resolve_options(thaw(saved).options(), None), "min_coverage": 0.0}, 1)
    assert [s["name"] for s in info["curves"]["series"]] == ["Measured", "Rebuilt by the network", "Residual", "Mean of the training samples"]


def AEmodel(saved):
    from fluoviz_addon_autoencoders.kinds.model import AEModel
    return AEModel(saved.meta["model"])


def test_statistics_helpers():
    from fluoviz_addon_autoencoders.kinds import stats as S
    rng = np.random.default_rng(0)
    e = np.exp(rng.normal(-3, 0.5, 400))
    lim = S.error_limits(e)
    assert 0.97 < (e < lim["99"]).mean() <= 1.0 and (e < lim["95"]).mean() > 0.9 and lim["source"] == "held-out" and "99 %" in lim["words"]
    few = S.error_limits([0.01], fallback=[0.01, 0.02, 0.015, 0.012])
    assert few["source"] == "training" and few["99"] > 0.02
    none = S.error_limits([], fallback=[])
    assert np.isnan(none["99"]) and none["source"] == "none"
    codes = rng.normal(size=(200, 3))
    st = S.code_stats(codes)
    p = S.typicality(rng.normal(size=(2000, 3)), st)
    assert abs((p < 0.01).mean() - 0.01) < 0.01 and abs((p < 0.2).mean() - 0.2) < 0.04            # calibrated for NEW samples
    assert S.typicality(np.array([[8.0, 8.0, 8.0]]), st)[0] < 1e-6
    assert S.kl_normal(np.zeros((1, 2)), np.zeros((1, 2)))[0] == pytest.approx(0.0) and S.kl_normal(np.ones((1, 2)), np.zeros((1, 2)))[0] == pytest.approx(1.0)
    cut = S.typicality_cut(0.01, st)
    unit = {**st, "cov": np.eye(3), "mean": np.zeros(3)}
    assert S.typicality(np.array([[np.sqrt(cut), 0.0, 0.0]]), unit)[0] == pytest.approx(0.01, rel=1e-6)


def test_the_limits_are_calibrated_for_new_samples():
    """The limit comes from samples the network did not train on: fresh ordinary samples are flagged at about the promised share (checked on eight networks of 40 training samples
    with 20 held-out ones each, 420 fresh samples every time)."""
    from fluoviz_addon_autoencoders.kinds import autoencoder as AE
    from fluoviz_addon_autoencoders.kinds.model import AEModel, model_tree
    share99, share95 = [], []
    for seed in range(8):
        Xall, _ = K.small_eem(n=60, seed=100 + seed)
        prep, _ = K.prepare(Xall[:40], kind="bspline")
        spec, arrays = K.linear_network(prep, d=3)
        model = AEModel(model_tree(spec, arrays, prep.frame))
        ev = model.evaluate(K.apply_data(Xall), 0.0)
        held = np.zeros(60, bool)
        held[40:] = True
        ref = AE.reference_for(model, [f"s{i}" for i in range(60)], held, ev)
        assert ref["limits"]["source"] == "held-out" and ref["limits"]["n"] == 20 and ref["limits"]["95"] < ref["limits"]["99"]
        # (the noise of the synthetic data is a share of the largest value of a set, so fresh samples come in sets of 60 like the training set)
        fresh = np.concatenate([model.evaluate(K.apply_data(K.small_eem(n=60, seed=1000 + 10 * seed + b)[0]), 0.0).error for b in range(7)])
        share99.append(float((fresh > ref["limits"]["99"]).mean()))
        share95.append(float((fresh > ref["limits"]["95"]).mean()))
    assert np.mean(share99) < 0.04 and np.mean(share95) < 0.12 and np.mean(share95) > 0.01          # about 1 % and 5 % (a limit from 20 samples is rough: the average is what is promised)


def test_the_linear_shortcut_of_the_network_is_frozen_and_applied():
    """``skip`` (a linear shortcut through the code, started from the PCA) is part of the saved ModelSpec and of the numpy forward pass: the PCA carried by the shortcut alone gives the codes of
    the same PCA carried by the layers, through the file."""
    from fluoviz_addon_autoencoders.core import linear
    from fluoviz_addon_autoencoders.core.spec import Config, ModelSpec
    X, _ = K.small_eem(n=30, seed=9)
    prep, _ = K.prepare(X, kind="bspline")
    spec0, arrays0 = K.linear_network(prep, d=3)                                              # the PCA in the layers (hidden = [], skip arrays zero)
    td = prep.train_data()
    pca = linear.fit_pca(prep.X, prep.M, td.omega, 3)
    cfg = Config(family="ae", latent_dim=3, hidden=[8], activation="tanh", skip=True)
    spec = ModelSpec.from_config(cfg, prep.P, td.blocks)
    assert spec.skip and "enc.lin.W" in spec.array_shapes() and "dec.lin.W" in spec.array_shapes()
    arrays = {k: np.zeros(shp) for k, shp in spec.array_shapes().items()}
    arrays["enc.lin.W"] = pca.Vw * pca.sw[None, :]
    arrays["dec.lin.W"] = (pca.Vw / pca.sw[None, :]).T
    arrays["in.mean"], arrays["in.scale"], arrays["out.weights"] = pca.mean.copy(), np.ones(prep.P), td.omega.copy()
    saved = K.roundtrip(K.make_saved(prep, spec, arrays))
    plain = K.roundtrip(K.make_saved(prep, spec0, arrays0))
    assert saved.meta["model"]["spec"]["skip"] is True and "enc.lin.W" in saved.meta["model"]["arrays"] and plain.meta["model"]["spec"]["skip"] is True
    a = K.run(saved, K.apply_data(X), min_coverage=0.0)
    b = K.run(plain, K.apply_data(X), min_coverage=0.0)
    for j in (1, 2, 3):
        assert np.allclose(K.column(a, f"z{j}"), K.column(b, f"z{j}"), rtol=1e-9, atol=1e-10)
    assert np.allclose(K.column(a, "recon_r2"), K.column(b, "recon_r2"))
    classical = ModelSpec.from_config(Config(family="ae", latent_dim=3, hidden=[8], skip=False), prep.P, td.blocks)
    assert not classical.skip and "enc.lin.W" not in classical.array_shapes()                  # the classical pure autoencoder has no shortcut arrays
