"""The exploration tools (numpy only): traversals, the analytic Jacobian and the sensitivity, interpolation, generation, codes against other variables, mixture clusters."""
import numpy as np
import pytest

from core_support import GP, cached, planted, prepared
from fluoviz_addon_autoencoders.core import forward as fw, tools
from fluoviz_addon_autoencoders.core.spec import ACTIVATIONS, AutoencoderError
from test_core_forward import random_net


def net(family="ae", act="silu", skip=False, **kw):
    spec, arrays = random_net(family, P=16, d=3, enc=(12, 6), act=act, skip=skip, **kw)
    arrays["in.mean"] = np.random.default_rng(1).normal(0, 0.2, 16)
    arrays["out.weights"] = np.random.default_rng(2).uniform(0.01, 0.1, 16)
    return spec, arrays


# ----------------------------------------------------------------------------- the Jacobian
@pytest.mark.parametrize("act,skip", [(a, s) for a in ACTIVATIONS for s in (False, True)])
def test_the_jacobian_is_the_derivative_of_the_decoder(act, skip):
    spec, arrays = net(act=act, skip=skip)
    Z = np.random.default_rng(3).normal(0, 1, (4, 3))
    J = tools.jacobian(spec, arrays, Z)
    assert J.shape == (4, 16, 3)
    eps = 1e-6
    for j in range(3):
        dz = np.zeros(3)
        dz[j] = eps
        fd = (fw.decode(spec, arrays, Z + dz) - fw.decode(spec, arrays, Z - dz)) / (2 * eps)
        assert np.allclose(J[:, :, j], fd, rtol=1e-5, atol=1e-7), (act, j)


def test_the_jacobian_matches_autograd():
    torch = pytest.importorskip("torch")
    from test_core_forward import torch_modules
    spec, arrays = net(act="gelu", skip=True)
    _, dec = torch_modules(spec, arrays)
    Z = np.random.default_rng(4).normal(0, 1, (3, 3))
    J = tools.jacobian(spec, arrays, Z)
    for i in range(3):
        Jt = torch.autograd.functional.jacobian(lambda z: dec(z[None])[0], torch.as_tensor(Z[i])).numpy()
        assert np.allclose(J[i], Jt, rtol=1e-8, atol=1e-10)


def test_the_output_activation_enters_the_jacobian():
    spec, arrays = net(act="tanh")
    spec.output_activation = "tanh"
    Z = np.random.default_rng(5).normal(0, 1, (3, 3))
    J = tools.jacobian(spec, arrays, Z)
    eps = 1e-6
    fd = (fw.decode(spec, arrays, Z + np.array([eps, 0, 0])) - fw.decode(spec, arrays, Z - np.array([eps, 0, 0]))) / (2 * eps)
    assert np.allclose(J[:, :, 0], fd, rtol=1e-5, atol=1e-8)


# ----------------------------------------------------------------------------- traversals and sensitivity
def test_traversal_decodes_along_one_code_with_the_others_at_their_mean():
    spec, arrays = net()
    codes = np.random.default_rng(6).normal(1.0, 2.0, (40, 3))
    tr = tools.traverse(spec, arrays, codes, dims=[1], values=(-2, 0, 2))
    mean, sd = tools.code_stats(codes)
    assert tr["z"].shape == (1, 3, 3) and np.allclose(tr["z"][0, :, 1], mean[1] + np.array([-2, 0, 2]) * sd[1]) and np.allclose(tr["z"][0, :, 0], mean[0])
    assert np.allclose(tr["X"][0, 1], tr["X_mean"]) and np.allclose(tr["X"][0, 0], fw.decode(spec, arrays, tr["z"][0, 0]))
    assert tools.traverse(spec, arrays, codes)["X"].shape == (3, 7, 16)
    with pytest.raises(AutoencoderError, match="3 codes"):
        tools.traverse(spec, arrays, codes, dims=[5])


def test_traversal_and_sensitivity_in_functions_of_a_real_frame():
    prep = cached(("tools", "eem"), lambda: prepared({"fluorescence": {"basis": GP}}, n=24))
    spec, arrays = random_net("ae", P=prep.P, d=3, enc=(8,), blocks=prep.model_blocks())
    arrays["in.mean"], arrays["out.weights"] = np.zeros(prep.P), prep.train_data().omega
    arrays["in.scale"] = 1.0 / (3.0 * np.sqrt(arrays["out.weights"]))
    codes = fw.encode(spec, arrays, prep.X)[0]
    tr = tools.traverse(spec, arrays, codes, values=(-1, 0, 1), frame=prep.frame)
    F, D0 = tr["functions"]["fluorescence"], tr["difference"]["fluorescence"]
    assert F.shape == (3, 3, prep.blocks[0].domain.m) and np.allclose(D0[:, 1], 0.0, atol=1e-10) and np.allclose(F[0, 1], tr["mean"]["fluorescence"])
    sn = tools.sensitivity(spec, arrays, codes, frame=prep.frame)
    assert np.isclose(sn["share"].sum(), 1.0) and sn["at_mean"]["fluorescence"].shape == (3, prep.blocks[0].domain.m) and sn["magnitude"]["fluorescence"].shape == (3, prep.blocks[0].domain.m)
    # the function-space sensitivity is the coordinate-space one pushed through the (affine) map back to the functions
    sc = tools.sensitivity(spec, arrays, codes)
    j = 1
    back = prep.frame.block("fluorescence").to_function(sc["at_mean"][j][None])[0] - prep.frame.block("fluorescence").mean_function()
    assert np.allclose(sn["at_mean"]["fluorescence"][j], back)
    assert (sn["magnitude"]["fluorescence"] >= 0).all()


def test_a_code_the_decoder_ignores_has_no_sensitivity():
    spec, arrays = net()
    arrays["dec.W0"] = arrays["dec.W0"].copy()
    arrays["dec.W0"][:, 2] = 0.0                                                                         # the first decoder layer never reads code 3
    codes = np.random.default_rng(7).normal(0, 1, (30, 3))
    sn = tools.sensitivity(spec, arrays, codes)
    assert sn["share"][2] < 1e-12 and sn["share"][:2].sum() == pytest.approx(1.0) and np.allclose(sn["magnitude"][2], 0.0)


# ----------------------------------------------------------------------------- interpolation and generation
def test_interpolation_has_the_codes_of_the_two_samples_at_its_ends_and_a_linear_mixture_beside_it():
    prep = cached(("tools", "eem"), lambda: prepared({"fluorescence": {"basis": GP}}, n=24))
    spec, arrays = random_net("ae", P=prep.P, d=3, enc=(8,), blocks=prep.model_blocks())
    za, zb = np.array([1.0, 0.0, -1.0]), np.array([-1.0, 2.0, 0.5])
    S = prep.smoothed()
    ip = tools.interpolate(spec, arrays, za, zb, 5, prep.frame, {k: v[0] for k, v in S.items()}, {k: v[3] for k, v in S.items()})
    assert np.allclose(ip["t"], [0, 0.25, 0.5, 0.75, 1.0]) and np.allclose(ip["z"][0], za) and np.allclose(ip["z"][-1], zb) and np.allclose(ip["z"][2], (za + zb) / 2)
    assert np.allclose(ip["X"][0], fw.decode(spec, arrays, za[None])[0]) and np.allclose(ip["X"][-1], fw.decode(spec, arrays, zb[None])[0])
    assert np.allclose(ip["linear"]["fluorescence"][0], S["fluorescence"][0]) and np.allclose(ip["linear"]["fluorescence"][-1], S["fluorescence"][3])
    assert np.allclose(ip["linear"]["fluorescence"][2], 0.5 * (S["fluorescence"][0] + S["fluorescence"][3])) and ip["difference"]["fluorescence"].shape == ip["functions"]["fluorescence"].shape
    with pytest.raises(AutoencoderError, match="at least 2"):
        tools.interpolate(spec, arrays, za, zb, 1)


def test_generation_is_a_vae_tool_seeded_tempered_and_always_synthetic():
    spec, arrays = net("vae")
    codes = np.random.default_rng(8).normal(0, 1, (30, 3))
    g = tools.generate(spec, arrays, 6, seed=2, temperature=1.0, codes_ref=codes)
    g2 = tools.generate(spec, arrays, 6, seed=2, temperature=1.0, codes_ref=codes)
    g3 = tools.generate(spec, arrays, 6, seed=3, temperature=1.0)
    assert g["synthetic"] is True and np.array_equal(g["z"], g2["z"]) and not np.array_equal(g["z"], g3["z"]) and g["X"].shape == (6, 16)
    assert g["nearest"].shape == (6,) and (g["distance"] >= 0).all() and np.allclose(g["X"], fw.decode(spec, arrays, g["z"]))
    cold = tools.generate(spec, arrays, 4, seed=1, temperature=0.0)
    assert np.allclose(cold["z"], 0.0)
    big = tools.generate(spec, arrays, 4000, seed=1, temperature=2.0)
    assert 1.8 < big["z"].std() < 2.2
    ae, ae_arrays = net("ae")
    with pytest.raises(AutoencoderError, match="Only variational autoencoders"):
        tools.generate(ae, ae_arrays, 3)
    mix, mix_arrays = net("vae", prior="mixture", K=3)
    mix_arrays["prior.mu"] = np.array([[-5.0, 0, 0], [0, 0, 0], [5.0, 0, 0]])
    mix_arrays["prior.logvar"] = np.full((3, 3), -4.0)
    gm = tools.generate(mix, mix_arrays, 300, seed=0)
    assert set(np.unique(gm["cluster"])) == {0, 1, 2} and np.abs(gm["z"][gm["cluster"] == 0, 0] + 5.0).max() < 1.5
    cm = tools.cluster_means(mix, mix_arrays)
    assert cm["z"].shape == (3, 3) and cm["X"].shape == (3, 16) and np.isclose(cm["pi"].sum(), 1)
    cl = tools.clusters(mix, mix_arrays, gm["z"])
    assert (cl["label"] == gm["cluster"]).mean() > 0.95 and np.allclose(cl["membership"].sum(axis=1), 1)
    with pytest.raises(AutoencoderError, match="no mixture"):
        tools.cluster_means(spec, arrays)


# ----------------------------------------------------------------------------- codes against other variables
def test_codes_against_variables_correlation_and_cross_validated_prediction():
    rng = np.random.default_rng(9)
    codes = rng.standard_normal((80, 3))
    y_lin = 2.0 * codes[:, 0] - 0.5 * codes[:, 2] + 0.05 * rng.standard_normal(80)
    y_noise = rng.standard_normal(80)
    y_sq = codes[:, 1] ** 2 + 0.01 * rng.standard_normal(80)
    y_gap = y_lin.copy()
    y_gap[:10] = np.nan
    pca = codes[:, :2]
    out = tools.relate(codes, {"lin": y_lin, "noise": y_noise, "sq": y_sq, "gap": y_gap}, pca_scores=pca, degree=1)
    assert out["lin"]["r2_codes"] > 0.99 and out["lin"]["correlation"][0] > 0.9 and out["lin"]["r2_pca"] < out["lin"]["r2_codes"]               # the PCA scores lack code 3
    assert out["noise"]["r2_codes"] < 0.1 and abs(out["noise"]["correlation"]).max() < 0.35
    assert out["sq"]["r2_codes"] < 0.2
    assert tools.relate(codes, {"sq": y_sq}, degree=2)["sq"]["r2_codes"] > 0.95                      # squares of the codes as extra features
    assert out["gap"]["n"] == 70 and out["gap"]["r2_codes"] > 0.99 and np.isnan(tools.cv_r2(codes[:4], y_lin[:4])) and np.isnan(tools.cv_r2(codes, np.ones(80)))
    assert np.allclose(tools.correlations(codes, {"c0": codes[:, 0]})["c0"], [1.0, *np.corrcoef(codes[:, 0], codes[:, 1:].T)[0, 1:]])
