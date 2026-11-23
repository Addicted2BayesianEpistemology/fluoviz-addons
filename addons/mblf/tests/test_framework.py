"""The add-on and the framework: the manifest is valid, nothing heavy is imported (no Stan, no xarray in the server), the build makes the zip (with the vendored mblf and its
Stan programs) and the index."""
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ADDON = Path(__file__).resolve().parents[1]
ROOT = ADDON.parents[1]
MAN = json.loads((ADDON / "manifest.json").read_text(encoding="utf-8"))


def test_the_manifest_is_valid_and_downloads_nothing_at_install():
    from fluoviz.addons import manifest as mf
    m = mf.validate_manifest(MAN)
    assert m.id == "mblf" and m.conda == {"all": []} and m.extra_download_mb == 0 and m.requires_addons == {"common": ">=1.0.0,<2"}
    assert MAN["provides"]["result_kinds"] == [{"kind": "mblf", "noun": "latent factor model", "saveable": True}] and MAN["provides"]["saved_kinds"] == ["mblf"]
    nav = MAN["provides"]["nav"][0]
    assert nav["group"] == "Experimental methods" and nav["path"] == "/mblf" and nav["badge"] == "add-on" and nav["needsData"] is True


def test_the_package_version_is_the_manifest_version():
    import fluoviz_addon_mblf
    assert fluoviz_addon_mblf.__version__ == MAN["version"]


def test_importing_the_add_on_loads_no_heavy_library():
    code = (f"import sys; sys.path[:0] = [{str(ADDON)!r}, {str(ADDON.parent / 'common')!r}, {str(ROOT.parent / '80_fluoViz')!r}]; import fluoviz_addon_mblf; "
            "from fluoviz_addon_mblf.server import router; from fluoviz_addon_mblf.kinds import MblfApplier; from fluoviz_addon_mblf.core import predict, loo, structure; "
            "bad = [m for m in ('cmdstanpy', 'xarray', 'torch', 'stanio') if m in sys.modules]; assert not bad, bad; print('ok')")
    r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, cwd=str(ADDON))
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-800:]


def test_the_build_makes_the_zip_and_the_index(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_addon.py"), "mblf", "--out", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    entry = next(a for a in json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))["addons"] if a["id"] == "mblf")
    z = tmp_path / entry["url"]
    assert z.name == f"mblf-{MAN['version']}.zip" and hashlib.sha256(z.read_bytes()).hexdigest() == entry["sha256"]
    names = zipfile.ZipFile(z).namelist()
    top = f"mblf-{MAN['version']}/"
    assert all(n.startswith(top) for n in names) and not any("/tests/" in n or "__pycache__" in n for n in names)
    for must in ("manifest.json", "fluoviz_addon_mblf/__init__.py", "fluoviz_addon_mblf/engine_job.py", "fluoviz_addon_mblf/vendor/mblf/stan/mblf.stan",
                 "fluoviz_addon_mblf/vendor/mblf/stan/mblf_ml.stan", "fluoviz_addon_mblf/vendor/mblf/LICENSE", "web/page.js", "web/help.js", "docs/ui.md"):
        assert top + must in names, must
