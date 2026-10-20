"""The quality measures on tiny hand-made cases (SPEC section 5)."""
import numpy as np

from fluoviz_addon_som.core import quality as Q
from fluoviz_addon_som.core import som as S


def test_topographic_error_counts_second_units_that_are_not_neighbours():
    lat = S.make_lattice(1, 4, "rectangular", "sheet")
    c1 = np.array([0, 1, 2, 3])
    assert Q.topographic_error(c1, np.array([1, 0, 1, 2]), lat.adj) == 0.0
    assert Q.topographic_error(c1, np.array([2, 3, 0, 1]), lat.adj) == 1.0
    assert Q.topographic_error(c1, np.array([1, 3, 1, 2]), lat.adj) == 0.25


def test_topographic_product_is_near_zero_when_the_map_fits_and_has_the_sign_of_the_mismatch():
    plane = S.make_lattice(6, 6, "rectangular", "sheet")
    assert abs(Q.topographic_product(plane.coords + 0.0, np.sqrt(plane.d2))) < 1e-6
    # a chain of units laid over two-dimensional data: the map is too small in dimension (negative)
    chain = S.make_lattice(1, 36, "rectangular", "sheet")
    assert Q.topographic_product(plane.coords[np.argsort(np.arange(36))], np.sqrt(chain.d2)) < -0.01
    # a sheet of units laid over data on a line: the map is larger than the data (positive)
    line = np.column_stack([np.arange(36) * 1.0, np.zeros(36)])
    assert Q.topographic_product(line, np.sqrt(plane.d2)) > 0.01


def test_trustworthiness_of_a_perfect_embedding_is_one_and_of_noise_is_low():
    rng = np.random.default_rng(0)
    Y = rng.standard_normal((60, 2))
    D = Q.pairwise(Y)
    tc = Q.trust_continuity(D, D, 5)
    assert tc["trustworthiness"] == 1.0 and tc["continuity"] == 1.0
    noise = Q.pairwise(rng.standard_normal((60, 2)))
    assert Q.trust_continuity(D, noise, 5)["trustworthiness"] < 0.7


def test_pca_scatter_of_a_plane_has_perfect_trustworthiness():
    rng = np.random.default_rng(1)
    X = np.column_stack([rng.standard_normal((50, 2)) * [3.0, 1.0], np.zeros((50, 3))])
    tc = Q.trust_continuity(Q.pairwise(X), Q.pairwise(Q.pca_scatter(X)), 5)
    assert tc["trustworthiness"] > 0.999


def test_sammon_recovers_a_planar_configuration_and_crossings_detect_a_fold():
    lat = S.make_lattice(4, 4, "rectangular", "sheet")
    D = Q.pairwise(lat.coords)
    xy, stress = Q.sammon(D, lat.coords + 0.01 * np.random.default_rng(0).standard_normal((16, 2)))
    assert stress < 1e-4
    u, edges, _ = S.umatrix(lat.coords, lat)
    assert Q.edge_crossings(xy, edges) == 0
    folded = lat.coords.copy()
    folded[[5, 6]] = folded[[6, 5]]
    assert Q.edge_crossings(folded, edges) > 0


def test_umatrix_of_a_regular_grid_is_constant_inside():
    lat = S.make_lattice(4, 4, "rectangular", "sheet")
    u, edges, _ = S.umatrix(lat.coords, lat)
    assert np.allclose(u, 1.0) and edges.shape[0] == 24
