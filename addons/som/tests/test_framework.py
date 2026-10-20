"""The add-on and the framework: nothing heavy is imported, the manifest is valid and says that no library is downloaded, the build makes the zip and the index."""
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ADDON = Path(__file__).resolve().parents[1]
ROOT = ADDON.parents[1]
MAN = json.loads((ADDON / "manifest.json").read_text(encoding="utf-8"))


def test_the_manifest_needs_no_extra_library():
    assert MAN["id"] == "som" and MAN["conda"] == {"all": []} and MAN["extra_download_mb"] == 0
    assert MAN["libraries_in_server"] is True and MAN["python_package"] == "fluoviz_addon_som" and MAN["entry"] == "fluoviz_addon_som:register"
    assert MAN["provides"]["result_kinds"] == [{"kind": "som", "noun": "map", "saveable": True}] and MAN["provides"]["saved_kinds"] == ["som"]
    nav = MAN["provides"]["nav"][0]
    assert nav["group"] == "Experimental methods" and nav["path"] == "/som" and nav["badge"] == "add-on" and nav["needsData"] is True


def test_importing_the_add_on_loads_no_heavy_library():
    code = (f"import sys; sys.path[:0] = [{str(ADDON)!r}, {str(ADDON.parent / 'common')!r}, {str(ROOT.parent / '80_fluoViz')!r}]; import fluoviz_addon_som; from fluoviz_addon_som.server import router; from fluoviz_addon_som.kinds import SomApplier; "
            "bad = [m for m in ('torch', 'tensorflow', 'sklearn.neural_network') if m in sys.modules]; assert not bad, bad; print('ok')")
    r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, cwd=str(ADDON))
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-800:]


def test_the_build_makes_the_zip_and_the_index(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_addon.py"), "som", "--out", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    index = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    entry = next(a for a in index["addons"] if a["id"] == "som")
    z = tmp_path / entry["url"]
    assert z.name == f"som-{MAN['version']}.zip" and hashlib.sha256(z.read_bytes()).hexdigest() == entry["sha256"]
    names = zipfile.ZipFile(z).namelist()
    top = f"som-{MAN['version']}/"
    assert all(n.startswith(top) for n in names)
    for must in ("manifest.json", "fluoviz_addon_som/__init__.py", "web/page.js", "web/help.js", "web/components/SomState.js", "docs/ui.md"):
        assert top + must in names, must
    assert not any("/tests/" in n or "__pycache__" in n for n in names) and top + "PROMPT.md" not in names
