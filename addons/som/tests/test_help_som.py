"""The Help guide and the interface contract (``docs/ui.md``): every bold label of the contract is on the page, every interface word of the guide is in the contract, the links of the guide
lead to pages that exist, and the numbers the guide states agree with the code.  (The browser check of every page is ``tests/e2e/flow_som.py``.)"""
import json
import os
import re
from pathlib import Path

import pytest

ADDON = Path(__file__).resolve().parents[1]
WEB = ADDON / "web"
UI = (ADDON / "docs" / "ui.md").read_text(encoding="utf-8")
PAGES = (WEB / "help" / "pages.js").read_text(encoding="utf-8")
FLUOVIZ = Path(os.environ.get("FLUOVIZ_DIR") or ADDON.parents[2] / "80_fluoViz")

# labels that the Functional PCA step components of FluoViz draw (steps 1 to 4); they are looked up there
FPCA_LABELS = {"Fluorescence", "Absorbance", "Both together (multivariate)", "Pre-treatment", "Weight of each variable", "Each variable counts equally", "Natural units", "Custom", "Samples that are left out",
               "Run with the recommended settings", "What this step does and what to check", "Back", "Next", "2 · Domain", "3 · Representation", "4 · Smoothing", "Samples"}


FLUOVIZ_PAGES = {"Plots", "Saved models"}                     # pages of FluoViz that the guide names


def web_text() -> str:
    out = []
    for p in sorted(WEB.rglob("*.js")):
        out.append(p.read_text(encoding="utf-8"))
    return "\n".join(out)


def fluoviz_text() -> str:
    d = FLUOVIZ / "fluoviz" / "web" / "js" / "components"
    return "\n".join(p.read_text(encoding="utf-8") for p in sorted(d.glob("Fpca*.js"))) if d.exists() else ""


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("&amp;", "&").replace("&eacute;", "é")).strip()


def labels() -> list:
    body = UI.split("## Fixed phrases")[0]
    out = []
    for m in re.finditer(r"\*\*([^*]+)\*\*", body):
        t = norm(m.group(1))
        if t and len(t) > 1:
            out.append(t)
    return sorted(set(out))


def present(label: str, text: str) -> bool:
    t = norm(text.replace("&amp;", "&"))
    if label in t:
        return True
    pat = re.escape(label).replace(r"\ N\ ", r" [^<'`]{0,16} ")
    return re.search(pat, t) is not None


def test_every_bold_label_of_the_contract_is_on_the_page():
    text, fpca = web_text(), fluoviz_text()
    missing = []
    for lab in labels():
        if lab in FPCA_LABELS:
            if fpca and not present(lab, fpca + text):
                missing.append(lab + " (Functional PCA)")
            continue
        if not present(lab, text):
            missing.append(lab)
    assert not missing, "labels of docs/ui.md that are not on the page: " + "; ".join(missing)


def test_every_interface_word_of_the_guide_is_in_the_contract():
    words = {norm(w) for w in re.findall(r"<Ui>(.*?)</Ui>", PAGES, flags=re.S)}
    assert len(words) > 40
    ui = norm(UI)
    missing = [w for w in sorted(words) if w not in FLUOVIZ_PAGES and w not in ui and w.lower() not in ui.lower() and w.replace("&amp;", "&") not in ui]
    assert not missing, "interface words of the guide that are not in docs/ui.md: " + "; ".join(missing)


def test_the_links_of_the_guide_lead_to_pages_that_exist():
    ids = set(re.findall(r"^export const \w+Page = page\('(\w+)'", PAGES, flags=re.M))
    assert len(ids) == 9
    for target in re.findall(r'<GLink to="(\w+)"', PAGES):
        assert target in ids, target
    for path in re.findall(r'<PLink path="([^"]+)"', PAGES):
        assert path in ("/som", "/compare", "/models", "/cluster", "/settings"), path


def test_the_pages_of_the_guide_are_in_the_manifest_order_and_the_tab_is_declared():
    man = json.loads((ADDON / "manifest.json").read_text(encoding="utf-8"))
    tab = man["provides"]["help_tabs"][0]
    assert tab["key"] == "som" and (ADDON / tab["module"]).is_file()
    parts = (WEB / "help" / "parts.js").read_text(encoding="utf-8")
    assert f"addon:{man['id']}:{tab['key']}" in parts
    assert (ADDON / man["provides"]["nav"][0]["page"]).is_file()


def test_the_numbers_of_the_guide_agree_with_the_code():
    from fluoviz_addon_som.core import advice, spec
    assert advice.units_for(36, "standard") == 30 and "five times the square root" in PAGES
    assert advice.units_for(40, "compact") == 10 and advice.units_for(400, "fine") <= spec.MAX_UNITS
    assert spec.MIN_SAMPLES == 8 and "fewer than 8" in UI and "FEW = 8" in (WEB / "components" / "SomState.js").read_text(encoding="utf-8")
    c = spec.Config.from_dict({})
    assert (c.epochs_rough, c.epochs_fine, c.radius_final, c.folds) == (50, 150, 1.0, 5)
    assert c.metric_kind == "l2" and c.lattice == "hexagonal" and c.shape == "sheet" and c.mode == "batch" and c.init == "linear"
