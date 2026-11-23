"""The list of a release may hold several versions of an add-on (scripts/build_addon.py --keep-from / --keep-release): FluoViz must offer each FluoViz the newest version it can
run, and resolve the shared add-on the same way.  Checked against FluoViz's own code (``fluoviz.addons.sources.Index.find`` and ``fluoviz.addons.deps.find_fitting``)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from fluoviz.addons import deps
from fluoviz.addons.sources import parse_index

REPO = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("build_addon", REPO / "scripts" / "build_addon.py")
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)
IDS = ("autoencoders", "common", "mblf", "som")
OLD = {"autoencoders": "1.1.0", "common": "1.0.0", "som": "1.0.0"}             # what release v1.1.0 offered, for FluoViz >= 1.1.0


def _old_release(tmp: Path) -> Path:
    """A stand-in for release v1.1.0: today's add-ons (minus mblf) renumbered to the old versions and asking for FluoViz >= 1.1.0."""
    src, out = tmp / "old-src", tmp / "old"
    entries = []
    for aid, ver in OLD.items():
        shutil.copytree(REPO / "addons" / aid, src / aid, ignore=shutil.ignore_patterns("tests", "__pycache__"))
        m = json.loads((src / aid / "manifest.json").read_text(encoding="utf-8"))
        m.update(version=ver, requires_fluoviz=">=1.1.0,<2")
        (src / aid / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
        entries.append(build.build_one(src / aid, out))
    (out / "index.json").write_text(json.dumps({"schema": 1, "addons": entries}), encoding="utf-8")
    return out


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rel")
    old = _old_release(tmp)
    new = tmp / "new"
    assert build.main(["--out", str(new), "--keep-from", str(old)]) == 0
    return new


def test_the_new_list_keeps_the_old_versions_with_their_zips(release):
    idx = json.loads((release / "index.json").read_text(encoding="utf-8"))
    have = {(e["id"], e["version"]): e for e in idx["addons"]}
    for aid in IDS:
        cur = json.loads((REPO / "addons" / aid / "manifest.json").read_text(encoding="utf-8"))
        assert (aid, cur["version"]) in have and have[(aid, cur["version"])]["requires_fluoviz"] == ">=1.2.0,<2"
    for aid, ver in OLD.items():
        assert have[(aid, ver)]["requires_fluoviz"] == ">=1.1.0,<2"
    assert len(idx["addons"]) == len(IDS) + len(OLD)
    for e in idx["addons"]:
        assert hashlib.sha256((release / e["url"]).read_bytes()).hexdigest() == e["sha256"], e["url"]


@pytest.mark.parametrize("fluoviz, want", [("1.1.0", {"autoencoders": "1.1.0", "common": "1.0.0", "som": "1.0.0"}),
                                          ("1.2.0", {"autoencoders": "1.1.2", "common": "1.0.1", "som": "1.0.2", "mblf": "1.0.0"})])
def test_each_fluoviz_is_offered_the_newest_version_it_can_run(release, fluoviz, want):
    idx = parse_index(json.loads((release / "index.json").read_text(encoding="utf-8")), None)
    offered = {e.id: e for e in idx.latest(fluoviz)}
    for aid, ver in want.items():
        e = offered[aid]
        assert e.version == ver and all(e.availability(fluoviz)[:2]), (fluoviz, aid, e.version)
    if fluoviz == "1.1.0":                                                  # mblf exists only for 1.2.0: listed, but with the reason in words
        ok, fits, why = offered["mblf"].availability(fluoviz)
        assert not fits and "needs FluoViz >=1.2.0,<2" in why and "Update FluoViz" in why
    # the shared add-on that an add-on needs is resolved for the same FluoViz
    for aid in ("autoencoders", "som"):
        need = offered[aid].manifest.requires_addons if hasattr(offered[aid].manifest, "requires_addons") else offered[aid].manifest["requires_addons"]
        rng = need["common"] if isinstance(need, dict) else ""
        assert deps.find_fitting(idx, "common", rng, fluoviz).version == want["common"]


def test_a_rebuilt_version_wins_and_an_unreachable_old_one_is_dropped(tmp_path):
    new = [{"id": "x", "version": "1.0.1", "requires_fluoviz": ">=1.2.0,<2"}, {"id": "y", "version": "2.0.0", "requires_fluoviz": ">=1.1.0,<2"}]
    kept = [{"id": "x", "version": "1.0.1", "requires_fluoviz": "old"}, {"id": "x", "version": "1.0.0", "requires_fluoviz": ">=1.1.0,<2"},
            {"id": "y", "version": "1.0.0", "requires_fluoviz": ">=1.1.0,<2"}]
    got = [(e["id"], e["version"], e["requires_fluoviz"]) for e in build.merge(new, kept)]
    assert got == [("x", "1.0.1", ">=1.2.0,<2"), ("x", "1.0.0", ">=1.1.0,<2"), ("y", "2.0.0", ">=1.1.0,<2")]


def test_a_kept_zip_that_does_not_match_its_sha256_is_refused(tmp_path):
    old = tmp_path / "old"
    old.mkdir()
    (old / "a-1.0.0.zip").write_bytes(b"not the right bytes")
    (old / "index.json").write_text(json.dumps({"schema": 1, "addons": [{"id": "a", "version": "1.0.0", "url": "a-1.0.0.zip", "sha256": "0" * 64}]}), encoding="utf-8")
    with pytest.raises(SystemExit, match="sha256"):
        build.kept_entries(old, tmp_path / "out")
