"""Specification objects and the numpy forward pass: JSON round trips, plain-English errors, and parity of every activation and of whole networks with PyTorch (1e-5 relative)."""
import json

import numpy as np
import pytest

from fluoviz_addon_autoencoders.core import forward as fw
from fluoviz_addon_autoencoders.core.spec import ACTIVATIONS, AutoencoderError, Config, ModelSpec, pack_model, unpack_model


def random_net(family="ae", P=20, d=3, enc=(16, 8), act="silu", seed=0, prior="normal", K=3, blocks=None, skip=False):
    rng = np.random.default_rng(seed)
    cfg = Config(family=family, latent_dim=d, hidden=list(enc), activation=act, vae={"prior": prior, "n_components": K}, skip=skip)
    blocks = blocks or [{"name": "fluorescence", "slice": [0, P], "kind": "pixels", "weight": 1.0}]
    spec = ModelSpec.from_config(cfg, P, blocks, sigma=0.1 if family == "vae" else None)
    arrays = {}
    for k, shp in spec.array_shapes().items():
        arrays[k] = rng.normal(0, 0.5 if k.endswith("W") or ".W" in k else 0.2, shp)
    arrays["in.scale"] = rng.uniform(0.5, 2.0, P)
    arrays["out.weights"] = np.full(P, 1.0 / P)
    if prior == "mixture" and family == "vae":
        arrays["prior.pi"] = np.full(K, 1.0 / K)
    return spec, arrays


def torch_modules(spec, arrays):
    """The same network as float64 torch modules, built from the arrays."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    acts = {"silu": F.silu, "tanh": torch.tanh, "gelu": F.gelu, "relu": F.relu, "softplus": F.softplus, "elu": F.elu, "linear": lambda x: x}

    def lin(name):
        W, b = arrays[f"{name}.W"], arrays[f"{name}.b"]
        m = nn.Linear(W.shape[1], W.shape[0]).double()
        with torch.no_grad():
            m.weight.copy_(torch.as_tensor(W))
            m.bias.copy_(torch.as_tensor(b))
        return m

    enc = []
    for i in range(len(spec.encoder["hidden"])):
        W, b = arrays[f"enc.W{i}"], arrays[f"enc.b{i}"]
        m = nn.Linear(W.shape[1], W.shape[0]).double()
        with torch.no_grad():
            m.weight.copy_(torch.as_tensor(W))
            m.bias.copy_(torch.as_tensor(b))
        enc.append(m)
    dec = []
    for i in range(len(spec.decoder["hidden"])):
        W, b = arrays[f"dec.W{i}"], arrays[f"dec.b{i}"]
        m = nn.Linear(W.shape[1], W.shape[0]).double()
        with torch.no_grad():
            m.weight.copy_(torch.as_tensor(W))
            m.bias.copy_(torch.as_tensor(b))
        dec.append(m)
    mu, out = lin("enc.mu"), lin("dec.out")
    lv = lin("enc.logvar") if spec.is_vae() else None
    Le = torch.as_tensor(arrays["enc.lin.W"]) if spec.skip else None
    Ld = torch.as_tensor(arrays["dec.lin.W"]) if spec.skip else None
    a_enc, a_dec = acts[spec.encoder["activation"]], acts[spec.decoder["activation"]]
    mean, scale = torch.as_tensor(arrays["in.mean"]), torch.as_tensor(arrays["in.scale"])

    def encode(x):
        h = (torch.as_tensor(x) - mean) / scale
        u = h
        for m in enc:
            h = a_enc(m(h))
        code = mu(h) if Le is None else mu(h) + u @ Le.T
        return code, (None if lv is None else lv(h))

    def decode(z):
        z = torch.as_tensor(z)
        h = z
        for m in dec:
            h = a_dec(m(h))
        o = out(h) if Ld is None else out(h) + z @ Ld.T
        return o * scale + mean

    return encode, decode


# ----------------------------------------------------------------------------- configuration and model description
def test_config_defaults_roundtrip_and_plain_english_errors():
    c = Config()
    assert c.family == "ae" and c.hidden_widths() == [64, 32] and c.decoder_widths() == [32, 64] and c.epochs == 600 and c.patience == 60 and c.n_restarts == 3 and c.seed == 0
    assert c.warmup_epochs() == 0 and Config(family="vae").warmup_epochs() == 120 and Config(family="vae", epochs=100, vae={"warmup": 7}).warmup_epochs() == 7
    d = json.loads(json.dumps(c.to_dict()))
    assert Config.from_dict(d).to_dict() == c.to_dict() and Config.from_json(c.to_json()).to_dict() == c.to_dict()
    c2 = c.replace(vae={"beta": 4.0}, latent_dim=5, hidden=[32])
    assert c2.vae["beta"] == 4.0 and c2.vae["prior"] == "normal" and c2.hidden_widths() == [32] and c.vae["beta"] == 1.0
    for bad, word in (({"family": "gan"}, "family"), ({"latent_dim": 0}, "codes"), ({"latent_dim": 40}, "codes"), ({"hidden": "huge"}, "size"), ({"activation": "swish"}, "activation"),
                      ({"dropout": 1.5}, "dropout"), ({"vae": {"beta": -1}}, "beta"), ({"vae": {"foo": 1}}, "Unknown"), ({"vae": {"prior": "uniform"}}, "prior"),
                      ({"smoothify": -1}, "smoothify"), ({"smoothify_metric": "x"}, "smoothify_metric"), ({"epochs": 0}, "epochs"), ({"foo": 1}, "Unknown setting"),
                      ({"val_fraction": 0.99}, "held out"), ({"corruption": {"noise": -1}}, "noise"), ({"n_restarts": 0}, "random starts"), ({"latent_dim": 2.5}, "whole number")):
        with pytest.raises(AutoencoderError, match=word):
            Config.from_dict(bad)
    with pytest.raises(AutoencoderError, match="at least 12"):
        Config().check_samples(9)
    Config(allow_few_samples=True).check_samples(9)


def test_model_spec_json_roundtrip_shapes_and_checks():
    spec, arrays = random_net("vae", prior="mixture", K=4, blocks=[{"name": "fluorescence", "slice": [0, 12], "kind": "gp", "weight": 1.0}, {"name": "absorbance", "slice": [12, 20], "kind": "bspline"}])
    again = ModelSpec.from_json(spec.to_json())
    assert again.to_dict() == spec.to_dict() and again.blocks[1]["weight"] == 1.0
    shp = spec.array_shapes()
    assert shp["enc.W0"] == (16, 20) and shp["enc.W1"] == (8, 16) and shp["enc.mu.W"] == (3, 8) and shp["enc.logvar.W"] == (3, 8) and shp["dec.W0"] == (8, 3) and shp["dec.W1"] == (16, 8)
    assert shp["dec.out.W"] == (20, 16) and shp["prior.mu"] == (4, 3) and shp["in.mean"] == (20,) and shp["out.weights"] == (20,)
    assert spec.n_parameters() == sum(int(np.prod(v)) for k, v in shp.items() if not (k.startswith("in.") or k == "out.weights"))
    spec.check_arrays(arrays)
    bad = dict(arrays)
    bad.pop("dec.b0")
    with pytest.raises(AutoencoderError, match="dec.b0"):
        spec.check_arrays(bad)
    bad = {**arrays, "enc.W0": np.zeros((3, 3))}
    with pytest.raises(AutoencoderError, match="shape"):
        spec.check_arrays(bad)
    with pytest.raises(AutoencoderError, match="unexpected"):
        spec.check_arrays({**arrays, "junk": np.zeros(2)})
    with pytest.raises(AutoencoderError, match="not numbers"):
        spec.check_arrays({**arrays, "in.mean": np.full(20, np.nan)})
    with pytest.raises(AutoencoderError, match="cover"):
        ModelSpec("ae", 20, 3, {"hidden": [4], "activation": "silu"}, {"hidden": [4], "activation": "silu"}, blocks=[{"name": "a", "slice": [0, 10], "kind": "gp"}])
    with pytest.raises(AutoencoderError, match="gaps"):
        ModelSpec("ae", 20, 3, {"hidden": [4], "activation": "silu"}, {"hidden": [4], "activation": "silu"}, blocks=[{"name": "a", "slice": [2, 20], "kind": "gp"}])
    ae, _ = random_net("ae")
    assert "enc.logvar.W" not in ae.array_shapes() and "prior.pi" not in ae.array_shapes() and "enc.lin.W" not in ae.array_shapes()
    sk, _ = random_net("ae", skip=True)
    assert sk.array_shapes()["enc.lin.W"] == (3, 20) and sk.array_shapes()["dec.lin.W"] == (20, 3) and ModelSpec.from_json(sk.to_json()).skip


def test_pack_and_unpack_without_pickles(tmp_path):
    spec, arrays = random_net("vae")
    p = tmp_path / "m.npz"
    np.savez(p, **pack_model(spec, arrays, {"best_epoch": 12, "x": np.float64(0.5)}))
    with np.load(p, allow_pickle=False) as z:
        s2, a2, extra = unpack_model(z)
    assert s2.to_dict() == spec.to_dict() and extra == {"best_epoch": 12, "x": 0.5} and all(np.array_equal(a2[k], arrays[k]) for k in a2)


# ----------------------------------------------------------------------------- activations against torch
@pytest.mark.parametrize("name", ACTIVATIONS)
def test_activations_and_their_derivatives_match_torch(name):
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F
    f = {"silu": F.silu, "tanh": torch.tanh, "gelu": F.gelu, "relu": F.relu, "softplus": F.softplus, "elu": F.elu, "linear": lambda x: x}[name]
    x = np.concatenate([np.linspace(-30, 30, 4001), np.random.default_rng(1).normal(0, 3, 2000)])
    x = x[np.abs(x) > 1e-9]                                  # the derivative of relu at exactly 0 is a convention
    t = torch.as_tensor(x, dtype=torch.float64).requires_grad_(True)
    y = f(t)
    (g,) = torch.autograd.grad(y.sum(), t)
    ours = fw.activation(name)(x)
    assert np.allclose(ours, y.detach().numpy(), rtol=1e-5, atol=1e-8)
    assert np.allclose(fw.activation_grad(name)(x), g.numpy(), rtol=1e-5, atol=1e-8)
    assert np.isfinite(ours).all()
    with pytest.raises(AutoencoderError, match="Unknown activation"):
        fw.activation("swish")


# ----------------------------------------------------------------------------- whole networks against torch
@pytest.mark.parametrize("family,act,enc,skip", [("ae", a, (16, 8), False) for a in ACTIVATIONS] + [("vae", "silu", (16, 8), False), ("vae", "tanh", (8,), False), ("ae", "linear", (), False),
                                                 ("vae", "gelu", (12, 6, 4), False), ("ae", "silu", (16, 8), True), ("vae", "tanh", (8,), True), ("ae", "linear", (), True)])
def test_encode_decode_match_torch_to_1e5(family, act, enc, skip):
    torch = pytest.importorskip("torch")
    spec, arrays = random_net(family, enc=enc, act=act, seed=3, skip=skip)
    arrays["in.mean"] = np.random.default_rng(5).normal(0, 0.3, 20)
    X = np.random.default_rng(2).normal(0, 1.5, (11, 20))
    t_enc, t_dec = torch_modules(spec, arrays)
    with torch.no_grad():
        mu_t, lv_t = t_enc(X)
        rec_t = t_dec(mu_t).numpy()
    mu, lv = fw.encode(spec, arrays, X)
    assert np.allclose(mu, mu_t.numpy(), rtol=1e-5, atol=1e-9)
    if family == "vae":
        assert np.allclose(lv, lv_t.numpy(), rtol=1e-5, atol=1e-9)
    else:
        assert lv is None
    assert np.allclose(fw.decode(spec, arrays, mu), rec_t, rtol=1e-5, atol=1e-9)
    xh, mu2, _ = fw.reconstruct(spec, arrays, X)
    assert np.allclose(xh, rec_t, rtol=1e-5, atol=1e-9) and np.array_equal(mu2, mu)


def test_forward_is_deterministic_and_handles_missing_inputs():
    spec, arrays = random_net("vae", seed=1)
    X = np.random.default_rng(0).normal(size=(5, 20))
    a = fw.reconstruct(spec, arrays, X)
    b = fw.reconstruct(spec, arrays, X)
    assert all(np.array_equal(u, v) for u, v in zip(a, b))
    Xn = X.copy()
    Xn[1, 4:9] = np.nan
    M = np.isfinite(Xn)
    mu_nan, _ = fw.encode(spec, arrays, Xn)
    X0 = X.copy()
    X0[1, 4:9] = arrays["in.mean"][4:9]                                # a missing coordinate is replaced by the mean: the same as the network input 0
    mu_0, _ = fw.encode(spec, arrays, X0)
    mu_M, _ = fw.encode(spec, arrays, X, M)
    assert np.allclose(mu_nan, mu_0) and np.allclose(mu_M[1], mu_0[1]) and np.allclose(mu_M[[0, 2, 3, 4]], mu_nan[[0, 2, 3, 4]])
    one, _ = fw.encode(spec, arrays, X[0])
    assert one.shape == (1, 3)
    with pytest.raises(AutoencoderError, match="expects 20"):
        fw.encode(spec, arrays, X[:, :5])
    with pytest.raises(AutoencoderError, match="3 codes"):
        fw.decode(spec, arrays, np.zeros((2, 4)))


def test_recon_error_is_the_weighted_squared_distance_with_missing_coordinates_scaled_up():
    P = 12
    blocks = [{"name": "a", "slice": [0, 8], "kind": "pixels"}, {"name": "b", "slice": [8, 12], "kind": "pixels"}]
    spec, arrays = random_net("ae", P=P, d=2, enc=(6,), blocks=blocks)
    w = np.r_[np.full(8, 0.5 / 8), np.full(4, 0.5 / 4)]
    arrays["out.weights"] = w
    rng = np.random.default_rng(0)
    X, Xh = rng.normal(size=(6, P)), rng.normal(size=(6, P))
    tot, per = fw.recon_error(spec, arrays, X, Xh)
    assert np.allclose(per[:, 0], ((X - Xh)[:, :8] ** 2 * w[:8]).sum(axis=1)) and np.allclose(per[:, 1], ((X - Xh)[:, 8:] ** 2 * w[8:]).sum(axis=1)) and np.allclose(tot, per.sum(axis=1))
    Xm = X.copy()
    Xm[2, 0:4] = np.nan                                              # half of block a missing in sample 2: the error of the block is that of the observed half, scaled up
    tot_m, per_m = fw.recon_error(spec, arrays, Xm, Xh)
    half = ((X - Xh)[2, 4:8] ** 2 * w[4:8]).sum()
    assert np.isclose(per_m[2, 0], half * 2.0) and np.allclose(per_m[[0, 1, 3, 4, 5]], per[[0, 1, 3, 4, 5]])
    arrays["in.mean"] = np.zeros(P)
    t0, _ = fw.null_error(spec, arrays, X)
    assert np.allclose(t0, (X ** 2 * w).sum(axis=1))


def test_mixture_memberships_and_prior_density():
    spec, arrays = random_net("vae", prior="mixture", K=3, seed=4)
    arrays["prior.mu"] = np.array([[-4.0, 0, 0], [0, 0, 0], [4.0, 0, 0]])
    arrays["prior.logvar"] = np.zeros((3, 3))
    mu = np.array([[-4.0, 0, 0], [0.1, 0, 0], [4.2, 0.1, 0]])
    g = fw.memberships(spec, arrays, mu)
    assert np.allclose(g.sum(axis=1), 1) and g.argmax(axis=1).tolist() == [0, 1, 2] and g.min() >= 0
    g2 = fw.memberships(spec, arrays, mu, np.full((3, 3), 2.0))
    assert np.allclose(g2.sum(axis=1), 1) and g2[1].argmax() == 1
    lp = fw.prior_log_density(spec, arrays, np.array([[0.0, 0, 0], [20.0, 0, 0]]))
    assert lp[0] > lp[1] and np.isfinite(lp).all()
    with pytest.raises(AutoencoderError, match="no mixture"):
        fw.memberships(random_net("ae")[0], random_net("ae")[1], mu)
