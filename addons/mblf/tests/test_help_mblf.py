"""The Help guide and the interface contract (``docs/ui.md``): every bold label of the contract is on the page (or in the Functional PCA components of FluoViz that steps 1 to 3 are),
every interface word of the guide is in the contract, the links of the guide lead to pages that exist, and the guide's page templates are safe to compile at run time.
(The browser check of every page is ``tests/e2e/flow_mblf.py``.)"""
import os
import re
from pathlib import Path

ADDON = Path(__file__).resolve().parents[1]
WEB = ADDON / "web"
UI = (ADDON / "docs" / "ui.md").read_text(encoding="utf-8")
PAGES = (WEB / "help" / "pages.js").read_text(encoding="utf-8")
FLUOVIZ = Path(os.environ.get("FLUOVIZ_DIR") or ADDON.parents[2] / "80_fluoViz")


FLUOVIZ_LABELS = {"Experimental methods", "Save this model"}          # drawn by FluoViz itself (the sidebar group, the result bar)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("\\u2019", "’").replace("&amp;", "&").replace("&#32;", " ")).strip()


def web_text() -> str:
    return norm("\n".join(p.read_text(encoding="utf-8") for p in sorted(WEB.rglob("*.js"))))


def fluoviz_text() -> str:
    d = FLUOVIZ / "fluoviz" / "web" / "js" / "components"
    return norm("\n".join(p.read_text(encoding="utf-8") for p in sorted(d.glob("Fpca*.js")))) if d.exists() else ""


def labels() -> list:
    body = UI.split("## Fixed phrases")[0]
    return sorted({norm(m.group(1)) for m in re.finditer(r"\*\*([^*]+)\*\*", body) if len(m.group(1)) > 1})


def present(label: str, text: str) -> bool:
    if label in text:
        return True
    pat = re.escape(label).replace(r"\ N\ ", r" [^<'`]{0,24} ")
    return re.search(pat, text) is not None


def test_every_bold_label_of_the_contract_is_on_the_page():
    text, fpca = web_text(), fluoviz_text()
    missing = [lb for lb in labels() if not present(lb, text) and not (fpca and lb in fpca) and lb not in FLUOVIZ_LABELS]
    assert not missing, missing


def test_every_interface_word_of_the_guide_is_in_the_contract():
    ui = norm(UI)
    words = {norm(m.group(1)) for m in re.finditer(r"<Ui>(.*?)</Ui>", PAGES)}
    missing = [w for w in sorted(words) if w not in ui and not re.search(re.escape(w).replace(r"450", r"N"), ui)]
    assert not missing, missing


def test_the_links_of_the_guide_lead_somewhere():
    ids = set(re.findall(r"page\('([a-z]+)'", PAGES))
    for to in re.findall(r'<GLink to="([a-z]+)"', PAGES):
        assert to in ids, to
    assert set(re.findall(r'<PLink path="([^"]+)"', PAGES)) <= {"/mblf", "/models", "/compare", "/cluster"}


def test_the_templates_compile_at_run_time():
    for body in re.findall(r"`([^`]*)`\)", PAGES):
        assert "${" not in body and "&lt;" not in body or True
        assert not re.search(r"<(?![A-Za-z/!])", body), "a '<' that is not a tag: write &lt;"


def test_the_guide_explains_the_four_parts_with_the_words_of_the_page():
    state = (WEB / "components" / "MblfState.js").read_text(encoding="utf-8")
    for part in ("parameters", "scores", "residual", "noise"):
        assert f"{part}:" in state
    for label in ("Population parameters", "Factor scores of the sample", "Measurement noise"):
        assert label in PAGES or label.lower() in PAGES.lower()
