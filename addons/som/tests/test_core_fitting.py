"""The whole fit: quality numbers, held-out error, the metrics, groups, planes, scans (SPEC sections 5 to 7)."""
import numpy as np
import pytest

import som_support as K
from fluoviz_addon_som.core import advice, diagnostics, fitting, groups, planes, scans
from fluoviz_addon_som.core.spec import Config, SomError

X0, A0 = K.small_eem(n=40, seed=3)
AB = K.small_abs(A0)


def fast(inp, **kw):
    d = {"input": inp, "train": {"epochs_rough": 8, "epochs_fine": 20}, "validation": {"folds": 4}}
    d.update(kw)
    return Config.from_dict(d)


@pytest.fixture(scope="module")
def fit_gp():
    prep, inp = K.prepare(X0, AB, kind="gp")
    return prep, fitting.fit_map(prep, fast(inp, **{"map": {"size": [4, 5]}}))


def test_quality_numbers_are_sensible(fit_gp):
    prep, fit = fit_gp
    q = fit.quality
    assert 0.5 < q["captured"] <= 1.0
    assert 0.0 <= q["topographic_error"] <= 0.4
    assert q["trust_map"]["trustworthiness"] > 0.85
    assert q["heldout_qe_mean"] >= 0.8 * q["qe_mean"]
    assert q["n_units"] == 20 and 0 <= q["n_empty"] < 20
    assert np.isfinite(fit.kfold["qe"]).all() and (fit.kfold["fold"] >= 0).all()


def test_neighbours_on_the_map_have_similar_total_concentration():
    prep, inp = K.prepare(X0, kind="bspline")
    fit = fitting.fit_map(prep, fast(inp, **{"map": {"size": [4, 5]}}), with_kfold=False)
    share = np.log(A0.sum(axis=1))
    D = np.linalg.norm(fit.pos[:, None] - fit.pos[None], axis=2)
    np.fill_diagonal(D, np.inf)
    nn = np.argsort(D, axis=1)[:, :3]
    near = np.abs(share[:, None] - share[nn]).mean()
    rnd = np.abs(share[:, None] - share[None, :]).mean()
    assert near < 0.7 * rnd


def test_each_metric_fits_and_keeps_neighbours():
    prep, inp = K.prepare(X0, AB, kind="gp")
    for kind in ("l2", "mahalanobis", "prior", "standardize"):
        cfg = fast(inp, metric={"kind": kind, "alpha": 1.0}, **{"map": {"size": [4, 5]}})
        fit = fitting.fit_map(prep, cfg, with_kfold=False)
        assert fit.quality["trust_map"]["trustworthiness"] > 0.8, kind
        assert fit.metric.kind == kind
        if kind == "mahalanobis":
            assert fit.metric.components >= 2 and fit.metric.U.shape[1] == fit.metric.components


def test_pixel_maps_refuse_the_whitened_metric_when_pixels_are_missing():
    X = X0.copy()
    X[0, 8:14, 6:10] = np.nan
    prep, inp = K.prepare(X, kind="pixels")
    with pytest.raises(SomError, match="every coordinate"):
        fitting.fit_map(prep, fast(inp, metric={"kind": "mahalanobis"}, **{"map": {"size": [3, 3]}}), with_kfold=False)
    fitting.fit_map(prep, fast(inp, metric={"kind": "standardize"}, **{"map": {"size": [3, 3]}}), with_kfold=False)


def test_assign_reproduces_the_training_assignments(fit_gp):
    prep, fit = fit_gp
    a = fitting.assign(prep.R, fit.metric, fit.protos_t, fit.protos_raw, fit.layout)
    assert np.array_equal(a["c1"], fit.asg["c1"]) and np.allclose(a["qe"], fit.asg["qe"])


def test_size_presets_and_checks():
    prep, inp = K.prepare(X0, kind="bspline")
    X = prep.R * prep.l2_gain()
    p = advice.presets(X)
    assert p["compact"]["units"] < p["standard"]["units"] < p["fine"]["units"]
    assert abs(p["standard"]["units"] - 5 * np.sqrt(40)) < 8
    with pytest.raises(SomError, match="too many"):
        advice.resolve_size([20, 20], X, None, "hexagonal", "sheet")
    r, c = advice.resolve_size([5, 5], X, None, "hexagonal", "toroid")
    assert r % 2 == 0


def test_groups_are_numbered_by_size_and_assigned_to_samples(fit_gp):
    prep, fit = fit_gp
    cfg = fast(prep.input, cluster={"k": 3})
    g = groups.cluster_units(fit.protos_t, fit.asg["c1"], fit.lattice.adj, cfg)
    assert g["k"] == 3 and len(g["sample_labels"]) == 40
    assert g["sizes"] == sorted(g["sizes"], reverse=True) and sum(g["sizes"]) == 40
    auto = groups.cluster_units(fit.protos_t, fit.asg["c1"], fit.lattice.adj, fast(prep.input))
    assert auto["k"] >= 2 and auto["reason"]


def test_planes(fit_gp):
    prep, fit = fit_gp
    M = fit.lattice.M
    basic = planes.basic_planes(fit.protos_t, np.zeros(prep.P), fit.hits, fit.umat, fit.asg["qe"], fit.asg["c1"])
    assert set(basic) == {"umatrix", "hits", "dmean", "qe_unit"} and np.isnan(basic["qe_unit"]["values"][fit.hits == 0]).all()
    sc = planes.score_planes(prep.R, fit.protos_raw, prep.l2_gain(), None, 4)
    assert len(sc) == 4 and sc["score1"]["values"].shape == (M,)
    funcs = prep.frame.to_functions(fit.protos_raw)
    tot = planes.feature_plane(prep.frame, funcs, {"kind": "total"})
    assert tot["values"].shape == (M,) and np.isfinite(tot["values"]).all()
    pt = planes.feature_plane(prep.frame, funcs, {"kind": "point", "ex": 330, "em": 450})
    assert np.isfinite(pt["values"]).all()
    with pytest.raises(SomError, match="not part of the map's domain"):
        planes.feature_plane(prep.frame, funcs, {"kind": "point", "ex": 400, "em": 300})
    ab = planes.feature_plane(prep.frame, funcs, {"kind": "absorbance", "wl": 300})
    assert ab["values"].shape == (M,)
    cat = [c["key"] for c in planes.index_catalogue(prep.frame)]
    assert "HIX" in cat and "a254" in cat or "E2_E3" in cat
    ix = planes.index_plane(prep.frame, funcs, "total_fluorescence", [f"u{j}" for j in range(M)])
    assert ix["values"].shape == (M,)
    meta = np.where(np.arange(40) % 2 == 0, "a", "b")
    vp = planes.variable_plane(meta, fit.asg["c1"], M, fit.lattice.d2, "site")
    assert vp["kind"] == "category" and vp["categories"] == ["a", "b"]
    num = planes.variable_plane(A0[:, 0], fit.asg["c1"], M, fit.lattice.d2, "c1", smooth=True)
    assert np.isfinite(num["values"]).all()


def test_verdicts_are_sentences(fit_gp):
    _, fit = fit_gp
    v = diagnostics.all_verdicts(fit.quality, fit.history)
    for k in ("vs_pca", "topographic", "heldout", "training"):
        assert isinstance(v[k]["verdict"], str) and len(v[k]["verdict"]) > 30


def test_size_scan_stability_and_metric_comparison():
    prep, inp = K.prepare(X0, kind="bspline")
    cfg = fast(inp)
    sc = scans.size_scan(prep, cfg, units=[8, 16, 30])
    assert [r["units"] for r in sc["rows"]] == sorted(r["units"] for r in sc["rows"]) and sc["suggested"] in {r["units"] for r in sc["rows"]}
    st = scans.stability(prep, fast(inp, **{"map": {"size": [3, 4]}}), n_runs=3)
    assert 0.0 <= st["value"] <= 1.0 and st["verdict"]["verdict"]
    mc = scans.metric_compare(prep, fast(inp, **{"map": {"size": [3, 4]}}))
    kinds = [r["kind"] for r in mc["rows"]]
    assert kinds[0] == "l2" and "mahalanobis" in kinds and all("error" not in r for r in mc["rows"])


def test_the_settings_are_checked_in_plain_words():
    for bad, text in [({"metric": {"kind": "euclid"}}, "distance must be one of"), ({"map": {"size": [1, 5]}}, "at least 2 rows"), ({"train": {"epochs_fine": 0}}, "between"),
                      ({"validation": {"limit": 80}}, "95 or 99"), ({"nonsense": 1}, "Unknown setting"), ({"map": {"lattice": "round"}}, "lattice must be one of"),
                      ({"train": {"alpha_initial": 0.01, "alpha_final": 0.05}}, "cannot be larger")]:
        with pytest.raises(SomError, match=text):
            Config.from_dict(bad)
    c = Config.from_dict({})
    assert Config.from_dict(c.to_dict()).to_dict() == c.to_dict()
