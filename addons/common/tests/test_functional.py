"""The functional-variable helpers the two add-ons share (``fluoviz_addon_common.functional``)."""
import numpy as np
import pytest

from fluoviz_addon_common import functional as F


class Boom(Exception):
    pass


# ----------------------------------------------------------------------------- what a request says
def test_the_basis_of_a_variable_is_found_in_every_way_a_request_can_say_it():
    assert F.lookup_basis({"fluorescence": {"basis": {"kind": "bspline"}}}, "fluorescence") == {"kind": "bspline"}
    assert F.lookup_basis({"basis": {"fluorescence": {"kind": "cosine"}}}, "fluorescence") == {"kind": "cosine"}
    assert F.lookup_basis({"basis": {"kind": "gp"}}, "absorbance") == {"kind": "gp"}
    assert F.lookup_basis({}, "fluorescence") is None and F.lookup_basis({"basis": {"absorbance": {"kind": "gp"}}}, "fluorescence") is None
    # the name of the key is the add-on's choice: the autoencoders also read ``representation``
    inp = {"representation": {"fluorescence": {"kind": "pixels"}}}
    assert F.lookup_basis(inp, "fluorescence") is None and F.lookup_basis(inp, "fluorescence", ("basis", "representation")) == {"kind": "pixels"}


def test_a_returned_basis_is_a_copy():
    inp = {"fluorescence": {"basis": {"kind": "gp"}}}
    F.lookup_basis(inp, "fluorescence")["kind"] = "changed"
    assert inp["fluorescence"]["basis"]["kind"] == "gp"


def test_gaps_are_estimated_unless_the_request_says_plain():
    assert F.gaps_spec({}, "fluorescence") == "auto" and F.gaps_spec({"fluorescence": {"gaps": "plain"}}, "fluorescence") == "plain"
    assert F.gaps_spec({"fluorescence": {"gaps": "auto"}}, "fluorescence") == "auto" and F.gaps_spec({"fluorescence": 3}, "fluorescence") == "auto"


def test_smoothing_is_per_variable_or_for_all_when_allowed():
    own = {"fluorescence": {"smoothing": {"method": "ml"}}, "smoothing": {"method": "cv"}}
    assert F.smoothing_spec(own, "fluorescence") == {"method": "ml"}
    assert F.smoothing_spec({"smoothing": {"method": "cv"}}, "fluorescence") == {"method": "cv"}
    assert F.smoothing_spec({"smoothing": {"fluorescence": {"method": "cv"}}}, "fluorescence") == {"method": "cv"}
    assert F.smoothing_spec({"smoothing": {"method": "cv"}}, "fluorescence", general=False) == {}
    assert F.smoothing_spec({}, "absorbance") == {}


# ----------------------------------------------------------------------------- variables from arrays
def eem(n=6, ne=5, nm=8, seed=0):
    rng = np.random.default_rng(seed)
    ex, em = np.linspace(250, 330, ne), np.linspace(300, 450, nm)
    X = 1 + np.abs(rng.normal(size=(n, nm, ne)))
    return X, ex, em


def test_variables_from_arrays_builds_the_fluorescence_and_the_absorbance_variable():
    X, ex, em = eem()
    wl = np.linspace(250, 500, 11)
    A = np.abs(np.random.default_rng(1).normal(size=(6, 11))) + 0.1
    both = F.variables_from_arrays(X, ex, em, A=A, wl=wl, input={"absorbance": {"transform": "none"}})
    assert [v.name for v in both] == ["fluorescence", "absorbance"] and [v.Y.shape[0] for v in both] == [6, 6]
    assert [v.name for v in F.variables_from_arrays(X, ex, em)] == ["fluorescence"] and [v.name for v in F.variables_from_arrays(A=A, wl=wl)] == ["absorbance"]
    assert F.variables_from_arrays() == []


# ----------------------------------------------------------------------------- pixels
class Dom:
    def __init__(self, w):
        self.w = np.asarray(w, float)


class Var:
    def __init__(self, Y, w=None, name="fluorescence"):
        self.Y = np.asarray(Y, float)
        self.name = name
        m = self.Y.shape[1]
        self.domain = Dom(np.full(m, 1.0 / m) if w is None else w)


def test_centred_pixels_returns_the_variance_over_the_part_of_the_domain_each_sample_has():
    Y = np.array([[1.0, 2.0, 3.0, np.nan], [2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0]])
    Yo, ybar, R, obs, w, msq = F.centred_pixels(Var(Y), Boom, "nothing here.")
    assert np.allclose(ybar, [2.0, 3.0, 4.0, 5.5]) and obs.sum() == 11 and w.sum() == pytest.approx(1.0) and np.allclose(Yo[np.isfinite(Yo)], Y[np.isfinite(Y)])
    cov = (obs * w).sum(axis=1)
    ss = (np.where(obs, R, 0.0) ** 2 @ w) / cov
    assert msq == pytest.approx(float(ss.mean()))


@pytest.mark.parametrize("Y, text", [
    (np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]]), "not measured in any sample"),
    (np.array([[np.nan, np.nan], [1.0, 2.0], [2.0, 3.0]]), "has no measured fluorescence value"),
    (np.ones((4, 3)), "all the same: there is nothing for a map to show."),
])
def test_centred_pixels_says_in_plain_words_what_is_wrong(Y, text):
    with pytest.raises(Boom, match=text):
        F.centred_pixels(Var(Y), Boom, "there is nothing for a map to show.")


# ----------------------------------------------------------------------------- functional coordinates
def test_the_total_variance_is_the_mean_squared_norm_and_all_alike_is_refused():
    Z = np.array([[3.0, 4.0], [0.0, 0.0], [6.0, 8.0]])
    assert F.total_variance(Z, "fluorescence", Boom, "nothing.") == pytest.approx((25 + 0 + 100) / 3)
    with pytest.raises(Boom, match="The absorbance functions are all the same: nothing."):
        F.total_variance(np.zeros((3, 2)), "absorbance", Boom, "nothing.")


class Rep:
    prior_lam = np.array([4.0, 1.0, 0.25])


class Fit:
    def __init__(self, sf2):
        self.sigma_f2 = sf2


class El:
    def __init__(self, sf2):
        self.rep, self.fit = Rep(), Fit(sf2)


def test_the_gaussian_process_scale_is_the_square_root_of_amplitude_times_eigenvalue():
    notes = []
    d = F.gp_prior_scale(El(9.0), np.zeros((4, 3)), notes)
    assert np.allclose(d, 3.0 * np.sqrt([4.0, 1.0, 0.25])) and notes == []


def test_an_amplitude_that_was_not_estimated_comes_from_the_data_and_the_note_says_so():
    Z = np.random.default_rng(2).normal(size=(10, 3))
    notes = []
    d = F.gp_prior_scale(El(float("nan")), Z, notes)
    sf2 = float(np.var(Z, axis=0, ddof=1).sum() / Rep.prior_lam.sum())
    assert np.allclose(d, np.sqrt(sf2 * Rep.prior_lam)) and len(notes) == 1 and "taken from the variance of the data" in notes[0]
