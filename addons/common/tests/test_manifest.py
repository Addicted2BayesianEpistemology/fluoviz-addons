"""The shared add-on's manifest, what the other add-ons ask of it, and the zip that is built (``scripts/build_addon.py``)."""
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from fluoviz.addons import manifest as mf
from fluoviz.addons.versions import satisfies

ADDON = Path(__file__).resolve().parents[1]
ADDONS = ADDON.parent
ROOT = ADDONS.parent
MAN = json.loads((ADDON / "manifest.json").read_text(encoding="utf-8"))
OTHERS = sorted(p.name for p in ADDONS.iterdir() if (p / "manifest.json").is_file() and p.name != "common")


def test_the_manifest_is_a_valid_library_without_anything_to_show():
    m = mf.validate_manifest(MAN)
    assert m.id == "common" and m.library is True and m.requires_addons == {} and m.conda == {"all": []} and m.extra_download_mb == 0
    assert m.provides == {"nav": [], "help_tabs": [], "result_kinds": [], "saved_kinds": []} and m.python_package == "fluoviz_addon_common" and m.entry == "fluoviz_addon_common:register"


def test_the_package_version_is_the_manifest_version():
    import fluoviz_addon_common
    assert fluoviz_addon_common.__version__ == MAN["version"]


@pytest.mark.parametrize("other", OTHERS)
def test_every_other_add_on_asks_for_a_version_of_it_that_this_one_has(other):
    m = mf.validate_manifest(json.loads((ADDONS / other / "manifest.json").read_text(encoding="utf-8")))
    assert "common" in m.requires_addons and satisfies(MAN["version"], m.requires_addons["common"]) and m.library is False
    assert m.requires_fluoviz == MAN["requires_fluoviz"]                          # (the same FluoViz versions: nobody is installed that the other cannot start with)


def test_the_other_add_ons_are_the_two_that_share_it():
    assert OTHERS == ["autoencoders", "som"]


def test_the_build_makes_a_zip_with_the_python_package_and_the_browser_files(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_addon.py"), "common", "--out", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    entry = next(a for a in json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))["addons"] if a["id"] == "common")
    z = tmp_path / entry["url"]
    assert z.name == f"common-{MAN['version']}.zip" and hashlib.sha256(z.read_bytes()).hexdigest() == entry["sha256"] and entry["manifest"]["library"] is True
    names = zipfile.ZipFile(z).namelist()
    top = f"common-{MAN['version']}/"
    assert all(n.startswith(top) for n in names) and not any("/tests/" in n or "__pycache__" in n for n in names)
    for must in ("manifest.json", "fluoviz_addon_common/__init__.py", "fluoviz_addon_common/freeze.py", "fluoviz_addon_common/functional.py", "fluoviz_addon_common/limits.py"):
        assert top + must in names, must
