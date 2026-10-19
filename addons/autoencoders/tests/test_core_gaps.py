"""Samples that lack part of the domain (a rectangle excised from SOME samples only): the data side of the core.  The coefficients of such a sample are estimated from the pixels it has and
from what the other samples show (the Functional PCA's gap-aware estimate); how much each sample lacks; the weights that make an estimated sample count in proportion to what was measured; the
saved frame that scores a new sample like the training sample; the settings and their plain-English errors.  No PyTorch needed."""
import numpy as np
import pytest

from core_support import GP, PIXELS, example_eems
from fluoviz_addon_autoencoders.core import data as D
from fluoviz_addon_autoencoders.core.spec import AutoencoderError, Config
from fluoviz_addon_autoencoders.server.request import parse_request

N = 40
GAP_ROWS = list(range(0, N, 4))                                                       # these ten samples lack the rectangle


def cut(X, ex, em, rows=GAP_ROWS, em_range=(400, 500), ex_range=(280, 380)):
    """``X`` with a rectangle taken out of the samples ``rows`` only."""
    E, XX = np.meshgrid(em, ex, indexing="ij")
    rect = (E > em_range[0]) & (E < em_range[1]) & (XX > ex_range[0]) & (XX < ex_range[1])
    Xg = X.copy()
    for i in rows:
        Xg[i][rect] = np.nan
    return Xg


def fl(basis=GP, **kw):
    return {"fluorescence": {"basis": basis, **kw}}


@pytest.fixture(scope="module")
def world():
    X, ex, em, ids, wl, A, _ = example_eems(n=N)
    Xg = cut(X, ex, em)

    def prep(data, **kw):
        inp = fl(**kw)
        return D.prepare(D.variables_from_arrays(X=data, ex=ex, em=em, ids=ids, input=inp), inp, seed=0)

    return {"X": X, "Xg": Xg, "ex": ex, "em": em, "ids": ids, "prep": prep, "complete": prep(X), "auto": prep(Xg), "plain": prep(Xg, gaps="plain")}


def functions(prep):
    return prep.to_functions(prep.X)["fluorescence"]


def lacking(world):
    """The pixels of the domain that the samples of ``GAP_ROWS`` lack."""
    idx = np.asarray(world["complete"].blocks[0].domain.idx)
    return ~np.isfinite(world["Xg"][GAP_ROWS[0]].ravel()[idx])


# ----------------------------------------------------------------------------- the coordinates of a sample with a gap
def test_a_sample_that_lacks_a_rectangle_gets_it_estimated_from_the_other_samples(world):
    F0 = functions(world["complete"])
    gone = lacking(world)
    assert 0.1 < gone.mean() < 0.3                                                          # a real rectangle: about a fifth of the domain
    signal = np.sqrt(np.mean(F0[GAP_ROWS][:, gone] ** 2))
    err = {m: np.sqrt(np.mean((functions(world[m])[GAP_ROWS][:, gone] - F0[GAP_ROWS][:, gone]) ** 2)) for m in ("auto", "plain")}
    assert err["auto"] < 0.12 * signal                                                       # the missing part is found again to within a few per cent of what is there
    assert err["auto"] < 0.8 * err["plain"]                                                  # and better than smoothing every sample on its own pixels
    own = [i for i in range(N) if i not in GAP_ROWS]
    assert np.abs(functions(world["auto"])[own] - F0[own]).max() < 0.05 * np.abs(F0).max()  # the complete samples are what they were


def test_the_notes_say_that_the_missing_parts_were_estimated(world):
    auto = " ".join(world["auto"].notes)
    assert "10 sample(s) miss part of the domain" in auto and "estimated from the pixels they have" in auto
    assert not any("miss part of the domain" in n for n in world["complete"].notes)
    assert any("missing in some samples" in n for n in world["plain"].notes)                  # the plain smoothing says what it did instead


# ----------------------------------------------------------------------------- how much each sample lacks, and the weights
def test_the_share_every_sample_lacks(world):
    b = world["auto"].blocks[0]
    w = b.domain.w
    expected = float(w[lacking(world)].sum() / w.sum())
    assert np.allclose(b.missing[GAP_ROWS], expected, atol=1e-6) and 0.1 < expected < 0.3
    assert np.all(b.missing[[i for i in range(N) if i not in GAP_ROWS]] == 0.0)
    assert world["complete"].blocks[0].missing.max() == 0.0


def test_samples_that_lack_part_of_the_domain_weigh_in_proportion_to_what_was_measured(world):
    p = world["auto"]
    sw = p.sample_weights()
    m = p.blocks[0].missing
    assert sw.shape == (N,) and sw.mean() == pytest.approx(1.0, abs=1e-12)
    ok = [i for i in range(N) if i not in GAP_ROWS]
    assert np.allclose(sw[GAP_ROWS] / sw[ok[0]], 1.0 - m[GAP_ROWS]) and sw[GAP_ROWS].max() < sw[ok].min()
    assert p.sample_weights("equal") is None
    td = p.train_data()
    assert np.array_equal(td.sw, sw) and p.train_data(gap_weighting="equal").sw is None
    sub = td.subset([3, 4, 8])
    assert np.array_equal(sub.sw, sw[[3, 4, 8]]) and sub.n == 3
    assert world["complete"].sample_weights() is None and world["complete"].train_data().sw is None        # nothing lacks anything: nothing changes


def test_the_weight_of_a_sample_never_falls_below_five_per_cent_and_a_tiny_gap_changes_nothing(world):
    p = world["prep"](world["Xg"])
    p.blocks[0].missing = np.zeros(N)
    p.blocks[0].missing[0] = 0.999
    sw = p.sample_weights()
    assert sw[0] / sw[1] == pytest.approx(0.05)
    p.blocks[0].missing = np.full(N, 0.004)
    assert p.sample_weights() is None                                                        # less than half a per cent of the domain: nobody is weighed differently


def test_two_variables_average_what_they_lack():
    X, ex, em, ids, wl, A, _ = example_eems(n=N)
    inp = {"fluorescence": {"basis": GP}, "absorbance": {"basis": {"kind": "gp", "lengths": "auto"}}}
    vs = D.variables_from_arrays(X=cut(X, ex, em), ex=ex, em=em, ids=ids, A=A, wl=wl, abs_ids=ids, input=inp)
    p = D.prepare(vs, inp, seed=0)
    m_fl = p.block("fluorescence").missing
    m_ab = p.block("absorbance").missing
    assert m_fl[GAP_ROWS].min() > 0.1 and m_ab.max() == 0.0
    sw = p.sample_weights()
    assert sw[GAP_ROWS].max() < 1.0 < sw[1]
    assert sw[GAP_ROWS[0]] / sw[1] == pytest.approx((1 - m_fl[0] / 2) / 1.0)                  # the mean over the two variables: the absorbance lacks nothing


def test_the_pixel_version_needs_no_weights_its_loss_leaves_the_missing_pixels_out(world):
    p = world["prep"](world["Xg"], basis=PIXELS)
    assert p.mode == "pixels" and p.sample_weights() is None and p.train_data().sw is None
    assert p.M is not None and (~p.M[GAP_ROWS]).sum(axis=1).min() > 100 and p.M[1].all()    # exactly the pixels the ten samples lack are left out of their loss
    assert p.blocks[0].missing[GAP_ROWS].min() > 0.1                                         # (what each lacks is still known: the page can say so)


# ----------------------------------------------------------------------------- the saved frame
def test_a_new_sample_with_the_same_gap_gets_the_coordinates_of_the_training_sample(world):
    p = world["auto"]
    out = p.frame.coordinates({"fluorescence": p.observed()["fluorescence"]}, min_coverage=0.5)
    assert out["usable"].all() and np.abs(out["X"] - p.X).max() < 1e-6                       # training and scoring read a gap the same way
    assert (out["coverage"][GAP_ROWS] < 0.9).all() and any("estimated" in " ".join(w) or w == [] for w in out["why"])


def test_a_complete_sample_that_is_measured_with_a_gap_is_rebuilt_better_by_the_gap_aware_frame(world):
    """New samples that lack the rectangle: the function the frame gives back is judged where it was not measured, against what the complete sample looks like through the model of the complete data."""
    comp = world["complete"]
    dom = comp.blocks[0].domain
    rows = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13]
    new = dom.take(world["X"][rows]).copy()                                                  # complete samples, as they were measured
    E, XX = np.meshgrid(world["em"], world["ex"], indexing="ij")
    rect = ((E > 400) & (E < 500) & (XX > 280) & (XX < 380)).ravel()[np.asarray(dom.idx)]
    holed = new.copy()
    holed[:, rect] = np.nan
    truth = comp.frame.to_functions(comp.frame.coordinates({"fluorescence": new}, min_coverage=0.5)["X"])["fluorescence"][:, rect]
    err = {}
    for m in ("auto", "plain"):
        fr = world[m].frame
        err[m] = np.sqrt(np.mean((fr.to_functions(fr.coordinates({"fluorescence": holed}, min_coverage=0.5)["X"])["fluorescence"][:, rect] - truth) ** 2))
    signal = np.sqrt(np.mean(truth ** 2))
    assert err["auto"] < 0.07 * signal and err["auto"] < 0.8 * err["plain"]                  # the part the rectangle took away is found again to within a few per cent


# ----------------------------------------------------------------------------- the settings
def test_the_weighting_is_a_setting_with_a_plain_english_error():
    assert Config().gap_weighting == "observed" and Config(gap_weighting="equal").gap_weighting == "equal"
    assert Config.from_json(Config(gap_weighting="equal").to_json()).gap_weighting == "equal"
    assert Config.from_dict({k: v for k, v in Config().to_dict().items() if k != "gap_weighting"}).gap_weighting == "observed"      # a result made before it existed
    with pytest.raises(AutoencoderError, match="weighting of samples that lack part of the domain"):
        Config(gap_weighting="sometimes")


def test_the_request_carries_the_weighting_and_the_estimate_of_the_gaps():
    r = parse_request({"latent_dim": 3, "gap_weighting": "equal", "input": {"variables": ["fluorescence"], "fluorescence": {"gaps": "plain"}}})
    assert r.cfg.gap_weighting == "equal" and r.input["fluorescence"]["gaps"] == "plain"
    p = r.params()
    assert p["gap_weighting"] == "equal" and p["input"]["fluorescence"]["gaps"] == "plain"
    again = parse_request(p)
    assert again.cfg.to_dict() == r.cfg.to_dict() and again.input == r.input                  # the result's parameters replay to the same settings
    d = parse_request({"latent_dim": 3, "input": {"variables": ["fluorescence"]}})
    assert d.cfg.gap_weighting == "observed" and d.input["fluorescence"]["gaps"] == "auto" and d.input["fluorescence"]["min_coverage"] == 0.5
    with pytest.raises(Exception, match="missing in some samples"):
        parse_request({"input": {"variables": ["fluorescence"], "fluorescence": {"gaps": "sometimes"}}})
