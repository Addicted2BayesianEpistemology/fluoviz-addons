"""Helpers of the tests of the nonlinear-effects tools (``core/nonlinear.py``, ``core/curves.py``, ``core/features.py``, ``server/nonlinear.py``): networks whose decoder is
known exactly (a few hidden units with chosen weights, so that every response can be written down by hand) and independent reference calculations with plain numpy.

Nothing here needs PyTorch."""
from __future__ import annotations

import numpy as np

from core_support import GP, PIXELS, cached, prepared
from test_core_forward import random_net

from fluoviz_addon_autoencoders.core.curves import normal_weights, positions
from fluoviz_addon_autoencoders.core.nonlinear import Decoder


def standard_codes(n: int, d: int, seed: int = 1) -> np.ndarray:
    """Codes of ``n`` training samples with mean 0 and sample SD exactly 1 in every column: a position in SD is then the value of the code itself."""
    z = np.random.default_rng(seed).standard_normal((n, d))
    return (z - z.mean(axis=0)) / z.std(axis=0, ddof=1)


def prep_gp(absorbance: bool = False):
    inp = {"fluorescence": {"basis": GP}} | ({"absorbance": {"basis": GP}} if absorbance else {})
    return cached(("nl", "gp", absorbance), lambda: prepared(inp, n=24, absorbance=absorbance))


def prep_pixels():
    return cached(("nl", "pixels"), lambda: prepared({"fluorescence": {"basis": PIXELS}}, n=24))


def build(prep, d: int, act: str, W0, b0=None, patterns=None, lin=None, mean=None, seed: int = 0, codes=None):
    """A decoder ``x = sum_u pattern_u * act(W0[u] . z + b0[u]) (+ lin z)`` on the frame of ``prep`` (no output activation; ``in.mean`` 0, ``in.scale`` 1).

    ``W0`` ``(U, d)``: what every hidden unit reads; ``patterns`` ``(U, P)`` (random when omitted): the picture every unit draws; ``lin`` ``(P, d)``: a linear part (needs the skip).
    Returns ``(Decoder, spec, arrays, patterns)``; the training codes default to standard ones (mean 0, SD 1), so the position of a code in SD is its value."""
    W0 = np.atleast_2d(np.asarray(W0, float))
    U = W0.shape[0]
    P = prep.P
    spec, arrays = random_net("ae", P=P, d=d, enc=(U,), act=act, seed=seed, blocks=prep.model_blocks(), skip=lin is not None)
    rng = np.random.default_rng(seed + 10)
    pat = rng.normal(size=(U, P)) if patterns is None else np.asarray(patterns, float)
    arrays["dec.W0"] = W0.copy()
    arrays["dec.b0"] = np.zeros(U) if b0 is None else np.asarray(b0, float)
    arrays["dec.out.W"] = pat.T.copy()
    arrays["dec.out.b"] = np.zeros(P)
    if lin is not None:
        arrays["dec.lin.W"] = np.asarray(lin, float)
        arrays["enc.lin.W"] = np.zeros((d, P))
    arrays["in.mean"] = np.zeros(P) if mean is None else np.asarray(mean, float)
    arrays["in.scale"] = np.ones(P)
    arrays["out.weights"] = prep.train_data().omega
    codes = standard_codes(40, d, seed + 1) if codes is None else codes
    return Decoder(spec, arrays, prep.frame, codes), spec, arrays, pat


def linear_decoder(prep, d: int = 3, seed: int = 0):
    """A decoder that is a straight line in every code (linear activation, hidden layer and linear shortcut)."""
    rng = np.random.default_rng(seed)
    pos = lambda *shape: 0.5 + np.abs(rng.normal(size=shape))                                  # (positive pictures: every code moves the overall intensity by a similar amount)
    dec, spec, arrays, pat = build(prep, d, "linear", rng.normal(size=(4, d)), b0=rng.normal(size=4), patterns=pos(4, prep.P), lin=pos(prep.P, d), seed=seed)
    return dec


# ----------------------------------------------------------------------------- independent reference calculations
def weighted_nl(t, y) -> float:
    """``1 - R^2`` of the straight line through ``y(t)`` with the normal-density weights, from ``numpy.polyfit`` (not the code under test)."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    w = normal_weights(t)
    c = np.polyfit(t, y, 1, w=np.sqrt(w))
    res = y - np.polyval(c, t)
    ybar = (w * y).sum()
    return float((w * res ** 2).sum() / (w * (y - ybar) ** 2).sum())


def grid_interaction_index(G: np.ndarray) -> float:
    """The interaction index of a scalar function on a square grid ``G[b, a]`` (centre = the middle position): sum of the squared
    ``G[b,a] - G[b,c] - G[c,a] + G[c,c]`` over the sum of the squared deviations of ``G`` from its mean."""
    g = G.shape[0]
    c = g // 2
    I = G - G[:, c:c + 1] - G[c:c + 1, :] + G[c, c]
    return float((I ** 2).sum() / ((G - G.mean()) ** 2).sum())


# ----------------------------------------------------------------------------- a fast stand-in for the training that gives a network that BENDS (no PyTorch)
def nonlinear_network(prep, d: int = 3, family: str = "ae", sigma: float = 0.1, seed: int = 1):
    """What ``kind_support.linear_network`` returns, but a network with a tanh hidden layer and random weights (so its decoder bends); used in place of the linear one by ``use_bending_training``."""
    import kind_support as K
    from fluoviz_addon_autoencoders.core import forward as fw
    spec, arrays = K.random_network(prep, d=d, family=family, hidden=(8,), activation="tanh", seed=seed)
    sd = fw.encode(spec, arrays, prep.X, prep.M)[0].std(axis=0, ddof=1)
    arrays["dec.W0"] = arrays["dec.W0"] * (1.5 / np.maximum(sd, 1e-6))[None, :]                 # the decoder reads the codes in SD: +-3 SD drives the tanh units into saturation
    return spec, arrays


def use_bending_training(monkeypatch) -> None:
    """The API tests' fast training (``api_support.fake_train``) with a network whose decoder bends instead of the exact linear autoencoder."""
    import api_support as A
    import kind_support as K
    monkeypatch.setattr(K, "linear_network", lambda prep, d=3, family="ae", seed=0, sigma=0.1: nonlinear_network(prep, d, family, sigma))
    A.use_fake_training(monkeypatch)


# ----------------------------------------------------------------------------- helpers of the API tests of the four routes (the stand-in and the real training alike)
RID = "autoencoder-1"


def post(c, route: str, body: dict, rid: str = RID, ok: bool = True):
    """``POST /{rid}/{route}``: the decoded answer (typed arrays as numpy), or the raw response when ``ok`` is False."""
    import api_support as A
    r = c.post(f"{A.BASE}/{rid}/{route}", json=body)
    if ok:
        assert r.status_code == 200, r.text
        return A.decode(r.json())
    return r


def msg(r) -> str:
    return r.json()["error"]["message"]


def arr(x):
    return np.asarray(x, float)


def decoder(c, rid: str = RID):
    """The numerical decoder of a result of the open project (the one the server builds), and the result."""
    from fluoviz_addon_autoencoders.core.nonlinear import Decoder
    from fluoviz_addon_autoencoders.server.results import model_of
    r = c.S.project.get_result(rid)
    return Decoder.from_model(model_of(r), np.asarray(r.payload["codes"], float), r.payload["code_mean"], r.payload["code_sd"]), r
