"""The data side of the core: network coordinates, the L2 metric (against the functional PCA of FluoViz), functions <-> coordinates, new samples through the frame, missing pixels,
the smoothness metric, the request vocabulary, the advice; and the rule that the server side never imports PyTorch.  No PyTorch needed."""
import os
import pathlib
import pickle
import subprocess
import sys

import numpy as np
import pytest

from core_support import GP, PIXELS, example_eems, planted, prepared
from fluoviz.analysis.fpca.fit import fit_fpca
from fluoviz_addon_autoencoders.core import advice, data as D, forward as fw, linear as lin
from fluoviz_addon_autoencoders.core.spec import AutoencoderError, Config, ModelSpec


def fl(basis, **kw):
    return {"fluorescence": {"basis": basis, **kw}}


BASES = {"gp": GP, "bspline": {"kind": "bspline", "n_basis": [10, 8]}, "cosine": {"kind": "cosine", "orders": [8, 6]}}


@pytest.fixture(scope="module")
def gp():
    return prepared(fl(GP))


@pytest.fixture(scope="module")
def both():
    return prepared({"fluorescence": {"basis": GP}, "absorbance": {"basis": {"kind": "gp", "lengths": "auto"}}}, absorbance=True)


# ----------------------------------------------------------------------------- the metric
def test_the_loss_metric_is_the_share_of_the_variation_and_matches_the_functional_pca(gp, both):
    """``1 - L`` is the share of the variation rebuilt: the weighted PCA of the coordinates has exactly the explained variances of the fPCA / multivariate fPCA of FluoViz."""
    for prep, names in ((gp, ["fluorescence"]), (both, ["fluorescence", "absorbance"])):
        td = prep.train_data()
        assert td.X.shape == (prep.n, prep.P) and td.M is None and np.all(td.omega > 0)
        assert float((td.omega * (td.X ** 2).mean(axis=0)).sum()) == pytest.approx(1.0, rel=1e-9)                      # the mean squared L2 norm of the samples = 1 (blocks counted equally)
        for b in prep.blocks:
            assert float((b.omega * (prep.X[:, b.sl] ** 2).mean(axis=0)).sum()) == pytest.approx(1.0 / len(prep.blocks), rel=1e-9)
        m = lin.fit_pca(td.X, td.M, td.omega, 8)
        share = m.lam / m.total
        fit = fit_fpca([b.element for b in prep.blocks], "variance" if len(names) > 1 else "none", 8)
        assert np.allclose(share, fit.explained[:8], atol=1e-9), (share, fit.explained[:8])
        # the error of the PCA with k components in the metric of the loss is 1 - cumulative share
        e = lin.pca_errors(m, td.X, td.M, td.omega, td.blocks, [0, 3])
        assert e[0].mean() == pytest.approx(1.0, rel=1e-9) and e[3].mean() == pytest.approx(1.0 - fit.cumulative[2], abs=1e-9)


def test_gp_coordinates_are_standardised_by_the_prior_and_the_other_bases_by_one_scalar(gp):
    b = gp.blocks[0]
    lam = b.rep.prior_lam
    assert np.allclose(b.scale ** 2 / lam, b.scale[0] ** 2 / lam[0])                       # d_k^2 = sigma_f^2 lambda_k
    assert np.allclose(b.omega, b.scale ** 2 / b.variance)
    bs = prepared(fl(BASES["bspline"]))
    assert np.allclose(bs.blocks[0].scale, bs.blocks[0].scale[0]) and np.allclose(bs.blocks[0].omega, 1.0)


# ----------------------------------------------------------------------------- functions <-> coordinates
@pytest.mark.parametrize("name", ["gp", "bspline", "cosine"])
def test_coordinates_back_to_the_smoothed_functions_and_new_samples_reproduce_them(name):
    prep = prepared(fl(BASES[name]))
    F, S = prep.to_functions(prep.X)["fluorescence"], prep.smoothed()["fluorescence"]
    assert np.abs(F - S).max() < 1e-10
    out = prep.frame.coordinates({"fluorescence": prep.observed()["fluorescence"]}, min_coverage=0.5)           # the raw values of the training samples, as a new sample would come
    assert out["usable"].all() and np.abs(out["X"] - prep.X).max() < 1e-8 and out["M"] is None
    assert np.allclose(prep.frame.mean_functions()["fluorescence"], prep.blocks[0].element.mean_function())
    # a sample with a hole of missing pixels is smoothed on the pixels it has: it still gives coordinates close to the full sample's
    Y = prep.observed()["fluorescence"].copy()
    Y[0, 10:40] = np.nan
    o2 = prep.frame.coordinates({"fluorescence": Y}, min_coverage=0.5)
    assert o2["usable"].all() and np.abs(o2["X"][0] - prep.X[0]).max() < 0.5 * np.abs(prep.X[0]).max() and np.allclose(o2["X"][1:], out["X"][1:], atol=1e-9)
    Y[1, :] = np.nan
    o3 = prep.frame.coordinates({"fluorescence": Y}, min_coverage=0.5)
    assert not o3["usable"][1] and "no fluorescence" in o3["why"][1][0] and np.isnan(o3["X"][1]).all()


def test_new_samples_are_pretreated_like_the_training_samples():
    X, ex, em, ids, wl, A, _ = example_eems(n=30)
    inp = fl(GP, transform="sqrt")
    prep = D.prepare(D.variables_from_arrays(X=X, ex=ex, em=em, ids=ids, input=inp), inp)
    dom = prep.blocks[0].domain
    out = prep.frame.coordinates({"fluorescence": dom.take(X)}, min_coverage=0.5)           # RAW values: the frame applies the square root
    assert np.abs(out["X"] - prep.X).max() < 1e-8
    with pytest.raises(AutoencoderError, match="pre-treatment"):
        D.prepare(D.variables_from_arrays(X=X, ex=ex, em=em, ids=ids), {"fluorescence": {"transform": "sqrt", "basis": GP}})


def test_pixel_mode_coordinates_missing_values_and_the_masked_error():
    X, ex, em, ids, wl, A, _ = example_eems(n=30)
    rng = np.random.default_rng(0)
    holes = rng.random(X.shape) < 0.03
    X = np.where(holes & np.isfinite(X), np.nan, X)
    inp = fl(PIXELS)
    prep = D.prepare(D.variables_from_arrays(X=X, ex=ex, em=em, ids=ids, input=inp), inp)
    b = prep.blocks[0]
    assert prep.M is not None and 0.01 < 1 - prep.M.mean() < 0.06 and np.all(prep.X[~prep.M] == 0.0)
    assert np.allclose(b.omega, b.domain.w) and prep.mode == "pixels" and any("not observed" in n for n in prep.notes)
    F = prep.to_functions(prep.X)["fluorescence"]
    Y = b.Y
    ok = np.isfinite(Y)
    assert np.abs(F[ok] - Y[ok]).max() < 1e-10 and np.isfinite(F).all()                    # the observed pixels come back exactly, the unobserved ones are the mean function
    # the error of a perfect reconstruction of the observed pixels is 0; the error of the mean is the variation (with the partly observed samples scaled up)
    spec = ModelSpec("ae", prep.P, 3, {"hidden": [4], "activation": "silu"}, {"hidden": [4], "activation": "silu"}, blocks=prep.model_blocks())
    arrays = {k: np.zeros(v) for k, v in spec.array_shapes().items()}
    arrays["in.scale"] = np.ones(prep.P)
    arrays["out.weights"] = prep.train_data().omega
    e0, _ = fw.recon_error(spec, arrays, prep.X, prep.X, prep.M)
    assert np.allclose(e0, 0.0)
    null, _ = fw.null_error(spec, arrays, prep.X, prep.M)
    assert null.mean() == pytest.approx(1.0, rel=0.05)
    # a variable whose gaps are filled with garbage after the fact gives the same coordinates (the unobserved coordinates are never read)
    td = prep.train_data()
    td2 = D.TrainData(np.where(td.M, td.X, 12345.0), td.M, td.omega, td.blocks, td.q, td.sd, td.ids, td.info, td.q_rough)
    assert td.subset([0, 1, 2]).n == 3 and td2.X.shape == td.X.shape


# ----------------------------------------------------------------------------- the smoothness metric
def test_curvature_operator_is_the_second_derivative():
    from fluoviz.analysis.fpca.domain import surface_domain
    em, ex = np.arange(300.0, 340.0, 2.0), np.arange(250.0, 290.0, 5.0)
    obs = np.ones((em.size, ex.size), bool)
    obs[3:6, 2:4] = False
    dom = surface_domain(em, ex, obs)
    A = D.curvature_operator(dom)
    E, X = np.meshgrid(em, ex, indexing="ij")
    f = (0.5 * (E - 300) ** 2 + 2.0 * (X - 250) ** 2)[obs]                                # f_em_em = 1, f_ex_ex = 4
    r = A @ f
    assert A.shape[1] == dom.m and np.allclose(np.sort(np.unique(np.round(r, 9))), [1.0, 4.0])
    assert np.allclose(A @ np.ones(dom.m), 0.0) and np.allclose(A @ (E + 3 * X)[obs], 0.0)           # constants and planes have no curvature


@pytest.mark.parametrize("name", ["bspline", "cosine"])
def test_smoothness_penalty_is_one_for_the_data_and_larger_for_rough_functions(name):
    prep = prepared(fl(BASES[name]))
    td = prep.train_data("smooth")
    pen = D._penalty(td.q_rough[0], prep.X)
    assert pen.mean() == pytest.approx(1.0, rel=1e-9)
    rough = np.random.default_rng(0).standard_normal(prep.X.shape) * prep.X.std()          # coordinates of the same size but without smooth structure
    assert D._penalty(td.q_rough[0], rough).mean() > 20 * pen.mean()
    from fluoviz_addon_autoencoders.core import diagnostics as G
    assert G.roughness(td, prep.X)["mean"]["fluorescence"] == pytest.approx(1.0, rel=1e-9) and G.roughness(td, rough)["mean"]["fluorescence"] > 20


def test_gp_smoothness_metric_is_the_prior_norm_and_ridge_is_the_frame_norm(gp):
    q = gp.blocks[0].q_smooth
    assert q["kind"] == "diag" and np.allclose(q["d"], q["d"][0])                         # Q proportional to the identity in the standardised coordinates
    r = gp.blocks[0].q_ridge
    assert np.allclose(r["d"] / r["d"][0], gp.blocks[0].scale ** 2 / gp.blocks[0].scale[0] ** 2)       # ridge on the frame coordinates z = xi d: Q = diag(d^2)
    assert float(np.mean(D._penalty(r, gp.X))) == pytest.approx(1.0)
    assert gp.train_data("ridge").q[0] is gp.blocks[0].q_ridge and gp.train_data("smooth").q[0] is gp.blocks[0].q_smooth


# ----------------------------------------------------------------------------- the request and the facts
def test_two_variables_blocks_summary_and_pickling(both):
    assert both.mode == "functional" and both.names == ["fluorescence", "absorbance"] and both.P == sum(b.K for b in both.blocks) and both.blocks[1].domain.kind == "curve"
    sm = both.summary()
    assert [b["name"] for b in sm["blocks"]] == both.names and sm["blocks"][0]["smoothing"]["criterion"] == "ml" and sm["n"] == both.n
    td = both.train_data()
    td2 = pickle.loads(pickle.dumps(td))
    assert np.array_equal(td2.X, td.X) and td2.blocks == td.blocks and [b["kind"] for b in td2.blocks] == ["gp", "gp"]
    assert both.frame.slices()[1] == slice(both.blocks[0].K, both.P) and set(both.frame.to_functions(both.X)) == {"fluorescence", "absorbance"}
    with pytest.raises(AutoencoderError, match="lack the absorbance"):
        both.frame.coordinates({"fluorescence": both.observed()["fluorescence"]})


def test_the_request_vocabulary_has_several_spellings_and_plain_english_errors():
    X, ex, em, ids, wl, A, _ = example_eems(n=24)
    variables = D.variables_from_arrays(X=X, ex=ex, em=em, ids=ids)
    spell = [{"fluorescence": {"basis": PIXELS}}, {"basis": {"fluorescence": PIXELS}}, {"representation": PIXELS}, {"representation": {"fluorescence": PIXELS}}, {"functional": False}]
    base = D.prepare(variables, spell[0])
    for s in spell[1:]:
        p = D.prepare(variables, s)
        assert p.mode == "pixels" and np.array_equal(p.X, base.X)
    assert D.prepare(variables, {"fluorescence": {"basis": BASES["cosine"], "smoothing": {"method": "fixed", "lambda": [1.0]}}}).blocks[0].kind == "cosine"
    assert D.prepare(variables, {}).blocks[0].kind == "gp"                                  # the recommended default
    with pytest.raises(AutoencoderError, match="Choose what to analyse"):
        D.prepare([], {})
    with pytest.raises(ValueError, match="unknown basis"):
        D.prepare(variables, {"fluorescence": {"basis": {"kind": "wavelets"}}})


def test_traindata_from_an_array_and_the_advice():
    A, _ = planted(n=50, P=20, k=3)
    td = D.TrainData.from_array(A)
    assert float((td.omega * (td.X ** 2).mean(axis=0)).sum()) == pytest.approx(1.0) and td.blocks[0]["slice"] == [0, 20]
    a = advice.advise(td, Config(latent_dim=4))
    assert a["components_99"] == 3 and a["latent_dim"] == 3 and a["size"] == "small" and a["validation"]["mode"] == "holdout" and not a["blocking"]
    assert a["estimate"]["seconds"] > 0 and "second" in a["estimate"]["words"]
    small = advice.advise(D.TrainData.from_array(A[:9]), Config())
    assert small["blocking"] and "at least 12" in small["blocking"][0]
    few = advice.advise(D.TrainData.from_array(A[:20]), Config(latent_dim=5))
    assert few["validation"]["mode"] == "kfold" and any("samples per code" in w for w in few["warnings"]) and any("Only 20 samples" in w for w in few["warnings"])
    assert advice.advise(td, Config(latent_dim=25))["blocking"][0].startswith("The number of codes (25)")
    # the parameter count is the one of the specification, with and without the shortcuts
    for skip in (False, True):
        for fam in ("ae", "vae"):
            cfg = Config(family=fam, latent_dim=4, hidden=[16, 8], skip=skip)
            spec = ModelSpec.from_config(cfg, 20, [{"name": "x", "slice": [0, 20], "kind": "gp"}])
            assert advice.count_parameters(20, 4, [16, 8], fam, skip) == spec.n_parameters()
    cfg = Config()
    assert advice.estimate_seconds(cfg.replace(epochs=1200), 64, 100) > advice.estimate_seconds(cfg, 64, 100) > advice.estimate_seconds(cfg.replace(epochs=100), 64, 100)
    assert advice.estimate_seconds(cfg, 64, 6000) > advice.estimate_seconds(cfg, 64, 100)


def test_the_server_side_never_imports_torch():
    """``spec``, ``forward``, ``data``, ``tools``, ``diagnostics`` (without the scans), ``advice`` and ``worker`` (the parent) must run without PyTorch in the process."""
    root = pathlib.Path(__file__).resolve().parents[1]
    fluoviz_root = pathlib.Path(D.__file__).resolve()
    import fluoviz
    code = (
        "import sys, numpy as np\n"
        "from core_support import prepared\n"
        "from fluoviz_addon_autoencoders.core import spec, forward, data, tools, diagnostics, advice, worker, linear\n"
        "from fluoviz_addon_autoencoders.core.spec import Config\n"
        "prep = prepared({'fluorescence': {'basis': {'kind': 'gp', 'lengths': 'auto'}}}, n=24)\n"
        "td = prep.train_data()\n"
        "advice.advise(td, Config())\n"
        "diagnostics.pca_curve(td, {'mode': 'holdout', 'train': list(range(18)), 'val': list(range(18, 24))}, 3)\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
        "print('ok')\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(root), str(root.parent / "common"), str(pathlib.Path(fluoviz.__file__).resolve().parents[1]), str(pathlib.Path(__file__).resolve().parent)])
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-2000:]
