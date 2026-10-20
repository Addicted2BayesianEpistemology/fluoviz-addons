"""The distances (SPEC section 3.4)."""
import numpy as np
import pytest
from scipy.spatial.distance import mahalanobis

from fluoviz_addon_som.core import metric as Mt
from fluoviz_addon_som.core import som as S
from fluoviz_addon_som.core.spec import SomError


def data(n=60, P=12, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((P, P))
    R = rng.standard_normal((n, P)) @ A * np.linspace(2.0, 0.1, P)
    return R - R.mean(axis=0)


def mk(R, kind="l2", **kw):
    P = R.shape[1]
    return Mt.fit_metric(R, kind, np.ones(P), np.ones(P), np.ones(P), kw.pop("prior", None), **kw)


def test_whiten_then_unwhiten_is_the_identity():
    R = data()
    m = mk(R, "mahalanobis", alpha=1.0, components=4)
    assert np.allclose(m.inverse(m.transform(R)), R, atol=1e-9)
    m5 = mk(R, "mahalanobis", alpha=0.5, components=3)
    assert np.allclose(m5.inverse(m5.transform(R)), R, atol=1e-9)


def test_alpha_zero_is_the_plain_distance():
    R = data()
    m = mk(R, "mahalanobis", alpha=0.0, components=4)
    X = m.transform(R)
    D_t = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
    D_l = ((R[:, None, :] - R[None, :, :]) ** 2).sum(axis=2)
    assert np.allclose(D_t, D_l)


def test_full_mahalanobis_equals_scipys_when_every_direction_is_kept():
    rng = np.random.default_rng(1)
    R = rng.standard_normal((400, 4)) @ rng.standard_normal((4, 4))
    R = R - R.mean(axis=0)
    m = mk(R, "mahalanobis", alpha=1.0, components=4)
    cov = np.cov(R, rowvar=False)
    X = m.transform(R[:6])
    for i in range(1, 6):
        d_ours = np.sqrt(((X[0] - X[i]) ** 2).sum())
        d_ref = mahalanobis(R[0], R[i], np.linalg.inv(cov))
        assert np.isclose(d_ours, d_ref, rtol=1e-6)


def test_the_remainder_counts_as_one_unit_whatever_the_dimension():
    rng = np.random.default_rng(2)
    n, P = 40, 300
    R = rng.standard_normal((n, 3)) @ rng.standard_normal((3, P)) * 5 + 0.1 * rng.standard_normal((n, P))
    R = R - R.mean(axis=0)
    m = mk(R, "mahalanobis", alpha=1.0, components=3)
    X = m.transform(R)
    rem = X - (X @ m.U) @ m.U.T
    assert 0.5 < (rem ** 2).sum(axis=1).mean() < 1.5            # the total variance of everything beyond the kept directions is scaled to about 1 (not 300)
    assert np.allclose((X @ m.U).var(axis=0, ddof=1), 1.0, atol=0.3)


def test_prototypes_under_a_whitened_metric_are_the_same_weighted_means():
    R = data(40, 6, 3)
    m = mk(R, "mahalanobis", alpha=1.0, components=3)
    c = np.arange(40) % 3
    X = m.transform(R)
    proto_t = np.array([X[c == j].mean(axis=0) for j in range(3)])
    proto_raw_via_metric = m.inverse(proto_t)
    direct = np.array([R[c == j].mean(axis=0) for j in range(3)])
    assert np.allclose(proto_raw_via_metric, direct, atol=1e-9)


def test_standardise_gives_every_coordinate_unit_variance_and_allows_missing_values():
    R = data(50, 5, 4)
    m = mk(R, "standardize")
    X = m.transform(R)
    assert np.allclose(X.std(axis=0, ddof=1), 1.0)       # (the gain of a variable's weight is 1 here)
    Rm = R.copy()
    Rm[0, 1] = np.nan
    assert np.isnan(m.transform(Rm)[0, 1])


def test_mahalanobis_refuses_missing_coordinates_in_plain_words():
    R = data(30, 5)
    R[0, 0] = np.nan
    with pytest.raises(SomError, match="every coordinate"):
        mk(R, "mahalanobis")


def test_prior_needs_the_gaussian_process_scale():
    R = data(30, 5)
    with pytest.raises(SomError, match="Gaussian-process"):
        mk(R, "prior")
    m = mk(R, "prior", prior=np.linspace(2.0, 0.5, 5))
    assert np.allclose(m.g, 1.0 / np.linspace(2.0, 0.5, 5))


def test_the_number_of_components_is_found_by_cross_validation():
    rng = np.random.default_rng(5)
    n = 80
    R = rng.standard_normal((n, 3)) @ rng.standard_normal((3, 40)) * 4 + 0.2 * rng.standard_normal((n, 40))
    R = R - R.mean(axis=0)
    cv = Mt.parsimonious_components(R)
    assert 2 <= cv["components"] <= 5
    m = mk(R, "mahalanobis")
    assert m.components == cv["components"]


def test_a_metric_is_storable_in_a_saved_model():
    from fluoviz.analysis.frozen import serial
    m = mk(data(30, 6), "mahalanobis", components=3)
    tree, arrays = serial.encode({"metric": m})
    back = serial.decode(tree, arrays)["metric"]
    assert np.allclose(back.transform(data(30, 6)), m.transform(data(30, 6)))
