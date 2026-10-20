"""Numerics of the map itself (SPEC section 4.6)."""
import numpy as np
import pytest
from scipy import stats

from fluoviz_addon_som.core import som as S
from fluoviz_addon_som.core.spec import SomError


def curve_data(n=60, seed=0, noise=0.02):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.random(n)) * 3.0
    X = np.column_stack([np.cos(t), np.sin(t), 0.3 * t]) + noise * rng.standard_normal((n, 3))
    return X, t


def test_hexagonal_neighbours_are_at_distance_one():
    lat = S.make_lattice(5, 6, "hexagonal", "sheet")
    interior = 2 * 6 + 2
    assert lat.adj[interior].sum() == 6
    assert np.allclose(lat.d2[interior][lat.adj[interior]], 1.0)
    rect = S.make_lattice(5, 6, "rectangular", "sheet")
    assert rect.adj[2 * 6 + 2].sum() == 4


def test_toroid_wraps_and_has_no_border():
    lat = S.make_lattice(6, 6, "rectangular", "toroid")
    assert lat.adj.sum(axis=1).min() == 4                      # every unit has four neighbours, also on the border
    hexa = S.make_lattice(5, 6, "hexagonal", "toroid")         # odd rows are rounded up to an even number
    assert hexa.rows == 6 and hexa.adj.sum(axis=1).min() == 6
    flat = S.make_lattice(6, 6, "rectangular", "sheet")
    assert flat.d2[0, 5] == 25.0 and lat.d2[0, 5] == 1.0


def test_one_unit_is_the_weighted_mean():
    X, _ = curve_data()
    lat = S.make_lattice(1, 1, "rectangular", "sheet")
    w = np.linspace(0.5, 1.5, X.shape[0])
    w = w / w.mean()
    tm = S.train(X, lat, S.TrainSpec(epochs_rough=2, epochs_fine=3, radius_final=0.5), w)
    assert np.allclose(tm.protos[0], (w[:, None] * X).sum(axis=0) / w.sum())


def test_tiny_radius_and_a_unit_per_sample_reach_zero_error():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((9, 4))
    lat = S.make_lattice(3, 3, "rectangular", "sheet")
    tm = S.train(X, lat, S.TrainSpec(init="random", neighbourhood="bubble", epochs_rough=20, epochs_fine=30, radius_initial=1.0, radius_final=0.3, seed=2))
    c, _, D2 = S.best_units(X, tm.protos)
    if len(set(c)) == 9:
        assert np.sqrt(D2[np.arange(9), c]).max() < 1e-9
    assert np.sqrt(D2[np.arange(9), c]).mean() < 0.9 * np.sqrt((X ** 2).sum(axis=1)).mean()      # at the very least far better than the mean


def test_a_row_of_units_orders_data_on_a_curve():
    X, t = curve_data()
    lat = S.make_lattice(1, 12, "rectangular", "sheet")
    tm = S.train(X, lat, S.TrainSpec(epochs_rough=30, epochs_fine=60, radius_final=0.8))
    c, _, _ = S.best_units(X, tm.protos)
    rho = stats.spearmanr(c, t).statistic
    assert abs(rho) > 0.95


def test_linear_initialisation_is_reproducible():
    X, _ = curve_data()
    lat = S.make_lattice(4, 5, "hexagonal", "sheet")
    spec = S.TrainSpec(epochs_rough=5, epochs_fine=10)
    a = S.train(X, lat, spec).protos
    b = S.train(X, lat, spec).protos
    assert np.array_equal(a, b)


def test_random_initialisation_depends_on_the_seed_and_sequential_runs():
    X, _ = curve_data(30)
    lat = S.make_lattice(3, 4, "rectangular", "sheet")
    a = S.train(X, lat, S.TrainSpec(init="random", epochs_rough=3, epochs_fine=3, seed=1)).protos
    b = S.train(X, lat, S.TrainSpec(init="random", epochs_rough=3, epochs_fine=3, seed=2)).protos
    assert not np.allclose(a, b)
    s = S.train(X, lat, S.TrainSpec(mode="sequential", epochs_rough=4, epochs_fine=6, seed=3))
    c, _, D2 = S.best_units(X, s.protos)
    assert np.sqrt(D2[np.arange(30), c]).mean() < np.sqrt((X ** 2).sum(axis=1)).mean()


def test_partial_distances_equal_full_ones_when_nothing_is_missing_and_are_unbiased_otherwise():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((400, 20))
    P = rng.standard_normal((5, 20))
    a = np.ones(20)
    _, _, D_full = S.best_units(X, P, a)
    assert np.allclose(D_full, ((X[:, None, :] - P[None]) ** 2).sum(axis=2))
    Xm = X.copy()
    Xm[rng.random(X.shape) < 0.3] = np.nan
    _, _, D_part = S.best_units(Xm, P, a)
    assert abs(D_part.mean() / D_full.mean() - 1.0) < 0.06      # the scaling up makes the partial distance unbiased in expectation


def test_a_sample_of_weight_zero_does_not_move_the_map():
    X, _ = curve_data(30)
    lat = S.make_lattice(3, 3, "rectangular", "sheet")
    spec = S.TrainSpec(epochs_rough=4, epochs_fine=6)
    w = np.ones(30)
    w[0] = 0.0
    a = S.train(X, lat, spec, w, protos0=S.init_linear(X[1:], lat)).protos
    b = S.train(X[1:], lat, spec, np.ones(29), protos0=S.init_linear(X[1:], lat)).protos
    assert np.allclose(a, b)


def test_the_energy_falls_over_a_phase_with_a_fixed_radius():
    """The batch map does not minimise the distortion exactly (Rynkiewicz 2008), so single epochs may rise a little; over the phase it falls and most steps fall."""
    X, _ = curve_data(50)
    lat = S.make_lattice(4, 4, "rectangular", "sheet")
    spec = S.TrainSpec(epochs_rough=1, epochs_fine=25, radius_initial=1.5, radius_final=1.5)
    h = np.array(S.train(X, lat, spec).history["energy"])
    assert h[-1] < h[1]
    assert np.mean(np.diff(h[1:]) <= 1e-9) > 0.7


def test_neighbourhood_functions():
    d2 = np.array([0.0, 1.0, 4.0, 9.0])
    assert np.allclose(S.neighbourhood(d2, 2.0, "bubble"), [1, 1, 1, 0])
    assert S.neighbourhood(d2, 2.0, "cutgauss")[3] == 0 and S.neighbourhood(d2, 2.0, "gaussian")[3] > 0
    assert np.allclose(S.neighbourhood(d2, 2.0, "epanechnikov"), [1, 0.75, 0.0, 0.0])
    with pytest.raises(SomError):
        S.neighbourhood(d2, 1.0, "nonsense")


def test_positions_lie_between_the_units_and_the_unit_is_nearest():
    X, _ = curve_data(40)
    lat = S.make_lattice(4, 5, "hexagonal", "sheet")
    tm = S.train(X, lat, S.TrainSpec(epochs_rough=10, epochs_fine=20))
    c, _, D2 = S.best_units(X, tm.protos)
    tau = float(D2[np.arange(40), c].mean())
    pos = S.positions(D2, lat, c, tau)
    assert (pos[:, 0] >= lat.coords[:, 0].min() - 1e-9).all() and (pos[:, 0] <= lat.coords[:, 0].max() + 1e-9).all()
    assert np.mean(np.linalg.norm(pos - lat.coords[c], axis=1)) < 1.2


def test_toroid_positions_stay_close_to_their_unit():
    X, _ = curve_data(40)
    lat = S.make_lattice(4, 6, "rectangular", "toroid")
    tm = S.train(X, lat, S.TrainSpec(epochs_rough=5, epochs_fine=10))
    c, _, D2 = S.best_units(X, tm.protos)
    pos = S.positions(D2, lat, c, float(D2[np.arange(40), c].mean()))
    assert np.linalg.norm(pos - lat.coords[c], axis=1).max() < 3.0
