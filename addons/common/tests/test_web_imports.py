"""The browser files of the add-ons import the shared ones by address (``/addon-web/common/web/<file>.js``): every such import must name a file that exists and names that it exports.  A typo here
only shows in the browser, so it is looked for here (there is no JavaScript engine in the test environment)."""
import re
from pathlib import Path

import pytest

ADDONS = Path(__file__).resolve().parents[2]
SHARED = ADDONS / "common" / "web"
IMPORT = re.compile(r"import\s*\{([^}]*)\}\s*from\s*'/addon-web/common/web/([\w.\-]+)'")


def exports_of(text: str) -> set[str]:
    names = set(re.findall(r"export\s+(?:const|let|var|function|class|async function)\s+(\w+)", text))
    for block in re.findall(r"export\s*\{([^}]*)\}", text):
        for part in block.split(","):
            part = part.strip()
            if part:
                names.add(part.split(" as ")[-1].strip())
    return names


def imports():
    out = []
    for f in sorted(ADDONS.glob("*/web/**/*.js")):
        if f.is_relative_to(SHARED):
            continue
        for names, target in IMPORT.findall(f.read_text(encoding="utf-8")):
            out.append((f.relative_to(ADDONS).as_posix(), target, [n.strip().split(" as ")[0].strip() for n in names.split(",") if n.strip()]))
    return out


def test_the_add_ons_do_import_shared_files():
    assert {t for _, t, _ in imports()} >= {"words.js", "parts.js", "help.js"}


@pytest.mark.parametrize("where, target, names", imports())
def test_every_shared_import_names_a_file_and_exports_that_exist(where, target, names):
    f = SHARED / target
    assert f.is_file(), f"{where} imports /addon-web/common/web/{target}, which does not exist"
    missing = [n for n in names if n not in exports_of(f.read_text(encoding="utf-8"))]
    assert not missing, f"{where} imports {missing} from {target}, which does not export them"


def test_the_shared_files_only_import_what_a_page_can_load():
    """They run in the browser of every page of every add-on: only vue and FluoViz's own absolute addresses, never a relative file of one add-on."""
    for f in SHARED.glob("*.js"):
        for target in re.findall(r"from\s+'([^']+)'", f.read_text(encoding="utf-8")):
            assert target == "vue" or target.startswith("/static/js/"), f"{f.name} imports {target}"
