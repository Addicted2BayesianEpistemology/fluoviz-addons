"""The Help guide of the add-on "Autoencoders" (``web/help.js`` and ``web/help/*.js``) says what the add-on does.  What it says is TRUE:

* every word of the interface that the guide quotes (a button, a step, a tab, a field: ``<Ui>...</Ui>``) exists in ``docs/ui.md`` (the contract of the page) AND in the files of the page
  (``web/page.js``, ``web/components/*.js``), or, for the few words that belong to FluoViz's own pages, in FluoViz's files;
* the defaults and limits the guide quotes are those of ``core/spec.py`` (``Config``); the numbers it quotes about the network, the advice, the verdicts, the limits and the scans are those of
  the code;
* the pictures are drawn from numbers (``web/help/data.js``, written by ``tests/help_data.py``) that this test recomputes with the real code (PyTorch needed: skipped without it), and what the
  pictures and their captions say is true of those numbers;
* the guide is complete (eight pages, each ending with "What to check"), written in plain English (no word of the programmer, every word of the neural network defined and in the glossary),
  every link resolves, and in a real browser no page shows ``{{`` or ``undefined``, has a console error, scrolls sideways at 600 px or has a picture whose text leaves it
  (``tests/help_browser.py``, run when a Playwright python is at hand: set FLUOVIZ_PLAYWRIGHT_PYTHON).

Regenerate the numbers of the pictures after a change of the numerics (from FluoViz's folder)::

    PYTHONPATH=. runtime/env/bin/python <add-on>/tests/test_help_autoencoders.py --write-data
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ADDON = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
for p in (str(ADDON), str(TESTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

WEB = ADDON / "web"
HELP = WEB / "help"
DATA_FILE = HELP / "data.js"
UI_MD = ADDON / "docs" / "ui.md"
API_MD = ADDON / "docs" / "api.md"
PKG = ADDON / "fluoviz_addon_autoencoders"
TAB = "addon:autoencoders:autoencoders"
IDS = ["what", "pipeline", "functional", "training", "checks", "tools", "use", "limits"]
PAGE_FILES = ["introPages.js", "functionalPages.js", "readPages.js", "usePages.js"]          # the pages of the guide (their order is in pages.js)


# ---------------------------------------------------------------------------------------------------------------- the numbers of the pictures
def write_data() -> None:
    import help_data as H
    text = ("// The numbers the pictures of the guide 'Autoencoders' are drawn from (computed with the add-on's own code by tests/help_data.py, written by\n"
            "// tests/test_help_autoencoders.py --write-data; the test recomputes them and compares).  JSON, so that the test can read it.\n"
            "export default " + json.dumps(H.figure_data(), separators=(",", ":")) + ";\n")
    DATA_FILE.write_text(text, encoding="utf-8")
    print("written", DATA_FILE, len(text), "characters")


def stored_data() -> dict:
    text = DATA_FILE.read_text(encoding="utf-8")
    return json.loads(text[text.index("export default ") + len("export default "):].rstrip().rstrip(";"))


if __name__ == "__main__" and "--write-data" in sys.argv:
    write_data()
    sys.exit(0)


# ---------------------------------------------------------------------------------------------------------------- reading the guide
def fluoviz_root() -> Path:
    import fluoviz
    return Path(fluoviz.__file__).resolve().parents[1]


def fluoviz_web() -> Path:
    return fluoviz_root() / "fluoviz" / "web" / "js"


def guide_source() -> str:
    return "".join((HELP / f).read_text(encoding="utf-8") for f in PAGE_FILES)


def plain(html: str) -> str:
    """The words a reader sees: no tags, the entities written plainly, one space between words."""
    text = re.sub(r"<[^>]+>", "", html)
    for a, b in (("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&nbsp;", " "), ("’", "'"), ("“", '"'), ("”", '"'), ('\\"', '"')):
        text = text.replace(a, b)
    return re.sub(r"\s+", " ", text).strip()


def glossary_source() -> str:
    return re.search(r"const GLOSSARY = \[(.*?)\n\];", (HELP / "usePages.js").read_text(encoding="utf-8"), re.S).group(1)


def glossary_entries() -> list[tuple[str, str]]:
    return [(plain(a), plain(b)) for a, b in re.findall(r"\['((?:[^'\\]|\\.)*)', '((?:[^'\\]|\\.)*)'\]", glossary_source())]


def guide_text() -> str:
    """The pages of the guide as a reader sees them, with the glossary written out."""
    return plain(guide_source()) + " " + plain(glossary_source())


def said(*phrases: str) -> None:
    text = guide_text()
    missing = [p for p in phrases if p not in text]
    assert not missing, f"the guide does not say: {missing}"


def page_blocks() -> dict[str, str]:
    out = {}
    for m in re.finditer(r"export const (\w+) = \{\s*id: '(\w+)'(.*?)\n\};\n", guide_source(), re.S):
        out[m.group(2)] = m.group(3)
    return out


def page_text(pid: str) -> str:
    return plain(page_blocks()[pid])


def ui_words() -> list[str]:
    return [plain(w) for w in re.findall(r"<Ui>(.*?)</Ui>", guide_source(), re.S)]


def ui_md_text() -> str:
    return plain(UI_MD.read_text(encoding="utf-8").replace("**", "").replace("`", ""))


def page_web_files() -> list[Path]:
    return [WEB / "page.js", *sorted((WEB / "components").glob("*.js"))]


def page_web_text() -> str:
    """The files of the page, and the FluoViz components it reuses for steps 1 to 4 and for the result bar (the framework's words)."""
    shared = [fluoviz_web() / "components" / n for n in ("ResultBar.js", "SaveModel.js")] + sorted((fluoviz_web() / "components").glob("Fpca*.js"))
    return "\n".join(f.read_text(encoding="utf-8") for f in [*page_web_files(), *shared] if f.exists())


# words of the guide that belong to FluoViz's own pages: {word: the file of FluoViz's web code that has it}
FLUOVIZ_WORDS = {"Standardise the columns (z-scores)": "pages/Cluster.js", "Separation (LDA)": "pages/Cluster.js"}
# words that the page writes from pieces (a number in the middle): {word: [pieces that must be in the page files]}
DYNAMIC = {"Use N codes": ["Use {{", "code"], "Use N": ["Use {{"], "Against PCA": ["Against ", "PCA"], "Against functional PCA": ["Against ", "functional PCA"]}
# the labels the guide must quote: the nine steps and the buttons of the specification
STEPS = ["1 · What to model", "2 · Domain", "3 · Representation", "4 · Smoothing", "5 · Network", "6 · Train", "7 · Results", "8 · Checks", "9 · Use"]
BUTTONS = ["Run with the recommended settings", "Train the network", "Cancel", "Train again with the same settings", "Cluster the latent space", "Classify with the latent space", "Save this model",
           "Apply a saved model", "Smoothify scan", "Latent-dimension scan", "Start from PCA (linear shortcut)"]


# ---------------------------------------------------------------------------------------------------------------- (i) the files, the pages, the tab
def test_the_manifest_declares_the_guide_and_the_files_exist():
    manifest = json.loads((ADDON / "manifest.json").read_text(encoding="utf-8"))
    tabs = manifest["provides"]["help_tabs"]
    assert tabs == [{"key": "autoencoders", "label": "Autoencoders", "module": "web/help.js"}]
    assert TAB == f"addon:{manifest['id']}:{tabs[0]['key']}"
    for f in ("help.js", "help/pages.js", "help/parts.js", "help/figures.js", "help/data.js", *[f"help/{p}" for p in PAGE_FILES]):
        assert (WEB / f).exists(), f
    assert re.search(rf"export const TAB = '{re.escape(TAB)}'", (HELP / "parts.js").read_text(encoding="utf-8"))
    guide = (WEB / "help.js").read_text(encoding="utf-8")
    assert "export default" in guide and "data-ae-guide" in guide and "AE_PAGES" in guide and "tab: TAB" in guide
    assert (ADDON / "docs" / "README_user.md").exists()


def page_ids() -> list[str]:
    listing = (HELP / "pages.js").read_text(encoding="utf-8")
    ident = {m.group(1): m.group(2) for m in re.finditer(r"export const (\w+) = \{\s*id: '(\w+)'", guide_source())}
    order = re.search(r"AE_PAGES = \[(.*?)\]", listing, re.S).group(1)
    return [ident[name.strip()] for name in order.split(",") if name.strip()]


def test_the_eight_pages_exist_in_order_with_titles_summaries_and_icons():
    assert page_ids() == IDS
    blocks = page_blocks()
    assert list(blocks) == IDS
    sprite = (fluoviz_root() / "fluoviz" / "web" / "img" / "icons.svg").read_text(encoding="utf-8")
    for pid, block in blocks.items():
        title = re.search(r"title: '([^']+)'", block)
        summary = re.search(r"summary: '([^']+)'", block)
        icon = re.search(r"icon: '([\w-]+)'", block)
        assert title and summary and icon and len(summary.group(1)) > 30, pid
        assert len(title.group(1)) <= 40, f"the title of '{pid}' is too long for the button of the pager at 600 px"
        assert f'id="i-{icon.group(1)}"' in sprite, icon.group(1)


def test_the_guide_is_complete_and_has_no_template_leftovers():
    for pid, block in page_blocks().items():
        text = plain(block)
        assert 'class="lead"' in block and len(text) > 1800, pid
        assert "{{" not in re.sub(r"\{\{ g\[\d\] \}\}", "", block), pid
        assert "undefined" not in text and "TODO" not in text and "xxx" not in text.lower(), pid
        assert len(re.findall(r"<h2>", block)) >= 3, pid
        assert block.rstrip().endswith("</div>`,\n  },"), pid
        assert "<h2>What to check</h2>" in block, f"the page '{pid}' does not have 'What to check'"
        last_h2 = block.rindex("<h2>")
        assert block.index("<h2>What to check</h2>") == last_h2, f"'What to check' is not the last section of '{pid}'"
    src = guide_source()
    assert src.count("<Callout") >= 4
    assert src.count('class="g-steps"') >= 4, "procedures are numbered steps"
    assert re.findall(r"<(Fig\w+)>", src).count("FigAutoencoder") == 1
    assert set(re.findall(r"<(Fig\w+)>", src)) == {"FigAutoencoder", "FigPcaVsAe", "FigSmoothify", "FigLcurve", "FigLoss", "FigCollapse", "FigTraversal"}
    # a runtime-compiled template: no backtick or ${ inside the text of a page
    for f in PAGE_FILES:
        for tpl in re.findall(r"template: `(.*?)`,\n  \},", (HELP / f).read_text(encoding="utf-8"), re.S):
            assert "`" not in tpl and "${" not in tpl, f


# ---------------------------------------------------------------------------------------------------------------- (ii) the words of the interface
def test_every_word_the_guide_quotes_exists_in_ui_md_and_in_the_page_files():
    """A word with <Ui> is a label of the page: in docs/ui.md (the contract) and in the files of the page; the few words of FluoViz's own pages are looked for there."""
    words = set(ui_words())
    assert len(words) >= 120, len(words)
    ui, web = ui_md_text(), page_web_text()
    bad_ui, bad_web = [], []
    for w in sorted(words):
        if w in FLUOVIZ_WORDS:
            assert w in (fluoviz_web() / FLUOVIZ_WORDS[w]).read_text(encoding="utf-8"), w
            continue
        if w not in ui:
            bad_ui.append(w)
        if w in DYNAMIC:
            if not all(piece in web for piece in DYNAMIC[w]):
                bad_web.append(w)
        elif w not in web:
            bad_web.append(w)
    assert not bad_ui, f"the guide quotes words that docs/ui.md does not have: {bad_ui}"
    assert not bad_web, f"the guide quotes words that the files of the page do not have: {bad_web}"


def test_the_nine_steps_and_the_buttons_of_the_specification_are_quoted():
    quoted = set(ui_words()) | {plain(h) for h in re.findall(r"<h2>(.*?)</h2>", guide_source())}            # a step is quoted by <Ui> or is the heading of its section on page 2
    missing = [w for w in STEPS + BUTTONS if w not in quoted]
    assert not missing, f"the guide does not quote: {missing}"
    ui = UI_MD.read_text(encoding="utf-8")
    for w in STEPS + ["Autoencoders (add-on)", "Experimental methods"]:
        assert f"**{w}**" in ui, w
    text = guide_text()
    for w in ("Autoencoders (add-on)", "Experimental methods"):
        assert w in text


def test_the_sentences_the_guide_quotes_are_sentences_the_program_writes():
    """Every sentence between straight double quotes (an example of what the page says) is one that the page, the server or the contract writes; '…' stands for the part that depends on the data."""
    def raw(path: Path) -> str:
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"[\"']\s*\n\s*f?[\"']", "", text)                         # adjacent string literals are one sentence
        return text.replace("\\'", "'")
    fz = fluoviz_root() / "fluoviz"
    files = [f for d in ("core", "server", "kinds") for f in sorted((PKG / d).glob("*.py"))] + sorted((fz / "addons").glob("*.py")) + [fz / "analysis" / "frozen" / "base.py"]
    corpus = " ".join([ui_md_text(), plain(page_web_text()), plain(API_MD.read_text(encoding="utf-8")), plain((ADDON / "docs" / "kind_and_router.md").read_text(encoding="utf-8"))] + [re.sub(r"\s+", " ", raw(f)) for f in files]).lower()
    corpus = corpus.replace("\\'", "'")
    src = guide_source().replace('\\"', '"')
    quotes = [plain(q) for q in re.findall(r'(?<![=\w])"([^"<>=]{14,}?)"(?![\w])', src)]
    assert len(quotes) >= 12, quotes
    missing = []
    for q in quotes:
        if q in EXEMPT_QUOTES:
            continue
        for piece in re.split(r"…|\.\.\.", q):
            piece = piece.strip(" .,:;").lower()
            if len(piece) < 9:
                continue
            if piece not in corpus:
                missing.append(piece)
    assert not missing, f"the guide quotes sentences that the program does not write: {missing}"


# quoted words that are not sentences of the program (an example the guide makes up)
EXEMPT_QUOTES: set[str] = set()


# ---------------------------------------------------------------------------------------------------------------- (iii) the defaults and the numbers of the code
def test_the_defaults_the_guide_quotes_are_those_of_config():
    from fluoviz_addon_autoencoders.core.spec import MAX_LATENT, MIN_SAMPLES, SIZES, Config
    c = Config()
    said(f"Maximum number of epochs ({c.epochs})", f"Patience (epochs without improvement) ({c.patience})", f"Share of samples held out ({c.val_fraction:g})",
         f"Folds (fewer than {c.kfold_below} samples) ({c.folds})", f"Learning rate ({c.lr:g})", f"Number of random starts ({c.n_restarts})", f"Random seed ({c.seed})",
         f"(Weight decay, {c.weight_decay:g})", f"(Dropout, {c.dropout:g} by default)", f"up to {c.epochs} epochs, patience {c.patience}", f"weight decay {c.weight_decay:g}, learning rate {c.lr:g}",
         f"{c.n_restarts} random starts, seed {c.seed}", f"a random {round(100 * c.val_fraction)} % are held out (at least 5)", f"Fewer than {c.kfold_below} samples:",
         f"{c.folds}-fold validation", f"Fewer than {MIN_SAMPLES} samples:", f"1 to {MAX_LATENT}", f"The page refuses to train on fewer than {MIN_SAMPLES} samples unless you insist",
         f"layers of {SIZES['small'][0]} and {SIZES['small'][1]} neurons", f"(128, 64)", f"(256, 128, 64)", "both 0 by default")
    assert c.family == "ae" and c.activation == "silu" and c.hidden == "small" and c.smoothify == 0 and c.smoothify_metric == "smooth" and c.skip is True
    assert SIZES == {"small": [64, 32], "medium": [128, 64], "large": [256, 128, 64]}
    assert c.corruption == {"noise": 0.0, "mask": 0.0} and c.vae["beta"] == 1.0 and c.vae["free_bits"] == 0.0 and c.vae["sigma"] == "learned" and c.vae["prior"] == "normal"
    assert c.vae["warmup"] is None and Config(family="vae").warmup_epochs() == round(0.2 * c.epochs)
    said("20 % of the epochs", "silu", "Start from PCA (linear shortcut)")
    # the numbers of the sentences about the size of the batch
    for n, expect in ((24, 8), (48, 8), (60, 10), (96, 16), (200, 32), (500, 32)):
        assert c.batch_for(n) == expect == min(32, max(8, n // 6)), n
    said("a sixth of the training samples, but at least 8 and at most 32")
    with pytest.raises(Exception, match=r"Only 9 samples are included"):
        c.check_samples(9)
    c.check_samples(MIN_SAMPLES)
    said("Only 9 samples are included")


def test_the_split_the_guide_describes_is_the_one_the_code_makes():
    from fluoviz_addon_autoencoders.core import train
    from fluoviz_addon_autoencoders.core.spec import Config
    c = Config()
    s = train.make_split(100, c)
    assert s["mode"] == "holdout" and len(s["val"]) == 20 and len(s["train"]) == 80
    s = train.make_split(30, c)
    assert s["mode"] == "holdout" and len(s["val"]) == 6                                                        # at least 5, 20 % otherwise
    assert len(train.make_split(200, c)["val"]) == 40
    k = train.make_split(29, c)
    assert k["mode"] == "kfold" and len(k["folds"]) == c.folds and sorted(i for f in k["folds"] for i in f) == list(range(29))
    assert train.make_split(12, c)["mode"] == "kfold"
    assert Config(seed=1).seed == 1 and train.make_split(100, c)["val"] == train.make_split(100, Config())["val"], "the split is made from the seed"
    said("never fewer than 20", "(their median best epoch")


def test_the_number_of_weights_the_guide_quotes():
    from fluoviz_addon_autoencoders.core.spec import Config, ModelSpec
    def count(P: int, **kw) -> int:
        return ModelSpec.from_config(Config(latent_dim=4, **kw), P, [{"name": "fluorescence", "slice": [0, P], "kind": "gp", "weight": 1.0}]).n_parameters()
    for skip in (True, False):
        assert count(10000, skip=skip) > 1_000_000 and count(90, skip=skip) < 20_000, skip
    assert count(10000) > 50 * count(90)
    said("10 000 pixels give more than a million weights, and 90 coefficients give fewer than 20 000", "a network of 20 000 weights can still memorise")


def advice_source() -> str:
    return "\n".join((PKG / f).read_text(encoding="utf-8") for f in ("server/advice.py", "core/advice.py"))


def test_the_advice_the_guide_describes_is_the_rule_of_the_code():
    src = advice_source()
    assert re.search(r"TARGET_PER_CODE\s*=\s*10\b", src) and re.search(r"MAX_LATENT_ADVICE\s*=\s*8\b", src)
    assert re.search(r"max\(2, min\([^)]*TARGET_PER_CODE[^)]*MAX_LATENT_ADVICE\)\)", src) or re.search(r"max\(2, min\(", src)
    assert re.search(r'"small" if n <= 300 else "medium"', src)
    said("The smallest of: the number of principal components that hold 99 % of the variation (a PCA of your samples), one tenth of the number of samples, and 8; never fewer than 2",
         "Medium only above 300 samples", "the advice keeps the codes at one tenth of the samples (40 samples, at most 4 codes)", "fewer than 10 training samples per code")
    assert 40 // 10 == 4


# ---------------------------------------------------------------------------------------------------------------- (iv) the pictures
def nn(grid) -> np.ndarray:
    """The numbers of a picture grid (``None`` = a pixel that does not exist) as an array with NaN."""
    return np.array([[np.nan if v is None else v for v in row] for row in grid], float)


def qualitative(d: dict) -> None:
    """What the pictures and their captions say, true of the numbers they are drawn from (stored or freshly computed)."""
    # ---- picture 2: PCA against the network, on two data sets of 120 EEMs
    c = d["compare"]
    assert c["dims"] == [1, 2, 3, 4, 5]
    for key in ("linear", "bending"):
        assert c[key]["n"] == 120 and c[key]["baseline_name"] == "Functional PCA" and all(len(c[key][k]) == 5 for k in ("pca", "ae", "ae_se", "linear")), key
    lin, bend = c["linear"], c["bending"]
    assert lin["recommended"] == 3 and bend["recommended"] == 2, "three fluorophores need three codes, two hidden factors two"
    for k in range(1, 5):                                                   # where the data are a sum of fixed shapes the network is as good as PCA, and the network without bends IS PCA
        assert lin["ae"][k] <= 1.6 * lin["pca"][k] + 0.003, ("the network is not as good as PCA on additive data", k)
        assert abs(lin["linear"][k] - lin["pca"][k]) <= 0.25 * lin["pca"][k] + 0.002, ("the linear network is not PCA", k)
        assert abs(bend["linear"][k] - bend["pca"][k]) <= 0.25 * bend["pca"][k] + 0.002, ("the linear network is not PCA", k)
    assert lin["pca"][2] < 0.01 < lin["pca"][1], "PCA needs three components for three fluorophores"
    assert bend["ae"][1] < bend["pca"][1] / 5 and bend["ae"][0] <= 1.05 * bend["pca"][0], "the bends pay when a band moves"
    kp = next(i + 1 for i, v in enumerate(bend["pca"]) if v <= bend["ae"][1] * 1.3)                # what the picture writes: PCA needs this many components for the quality of the network's 2 codes
    assert kp >= bend["recommended"] + 2, kp
    assert next(i + 1 for i, v in enumerate(lin["pca"]) if v <= lin["ae"][2] * 1.3) == lin["recommended"] == 3
    # ---- picture 1: one held-out sample through the network
    o = d["one"]
    assert (len(o["measured"]), len(o["measured"][0])) == (15, 29) == (len(o["rebuilt"]), len(o["rebuilt"][0])) and o["n_codes"] == 3 and len(o["code"]) == 3
    m, rb = nn(o["measured"]), nn(o["rebuilt"])
    assert np.array_equal(np.isnan(m), np.isnan(rb)) and np.isnan(m).sum() == 15 * 29 - o["n_pixels"], "the grey pixels are the ones that do not exist"
    assert o["n_inputs"] < o["n_pixels"] / 3 and o["r2"] > 0.98 and 0.0 <= np.nanmax(m) <= 1.0 + 1e-9 and o["n_parameters"] > 5000
    assert np.nanmax(np.abs(m - rb)) < 0.15, "the rebuilt EEM looks like the measured one"
    # ---- picture 3: a traversal
    t = d["traversal"]
    assert t["values"] == [-3, -2, -1, 0, 1, 2, 3] and len(t["images"]) == 7 == len(t["differences"]) and max(t["corr_with_band_position"]) > 0.9 and t["corr_with_band_position"][t["code"] - 1] == max(t["corr_with_band_position"])
    dm, dp, d0 = nn(t["differences"][0]), nn(t["differences"][6]), nn(t["differences"][3])
    ok = ~np.isnan(dm) & ~np.isnan(dp)
    assert np.nanmax(np.abs(d0)) < 1e-6 and np.corrcoef(dm[ok], dp[ok])[0, 1] < -0.3, "the code moves a band: the differences at -3 and +3 SD are opposite"
    assert np.nanmax(nn(t["images"][3])) <= 1.0 + 1e-9 and np.nanmin(dm) >= -1.0001 and np.nanmax(dm) <= 1.0001
    # ---- pictures 4 and 6: smoothify
    s = d["smoothify"]
    sc = s["scan"]
    expect = [0.0] + [round(float(v), 4) for v in np.geomspace(1e-4, 10.0, 11)]
    assert sc["values"] == expect and s["n"] == 40 and sc["suggested"] in [float(v) for v in sc["values"]] or abs(sc["suggested"] - min(sc["values"], key=lambda v: abs(v - sc["suggested"]))) < 1e-3
    assert sc["error"][-1] > 5 * sc["error"][0] and sc["roughness"][-1] < 0.1 * sc["roughness"][0], "much smoothify flattens everything"
    i = int(np.argmin([abs(v - sc["suggested"]) for v in sc["values"]]))
    assert sc["roughness"][i] <= sc["roughness"][0] and sc["error"][i] <= sc["error"][0] + max(sc["error_se"][0], 0.02 * sc["error"][0]) + 1e-9, "the suggestion lowers the roughness at no clear cost"
    cut = s["cut"]
    r = cut["roughness"]
    assert len(cut["em"]) == 29 == len(cut["off"]) == len(cut["on"]) == len(cut["measured"]) == len(cut["smoothed"])
    assert r["off"] > r["measured_smoothed"] and r["on"] < r["off"], "without smoothify the rebuilt EEM is rougher than the smoothed measured one; with it it is less rough"
    assert cut["evr_val"]["on"] >= cut["evr_val"]["off"] - 0.05, "the price is small"
    # ---- picture 5: the KL of every code of a VAE with more codes than the data need
    c5 = d["collapse"]
    assert len(c5["kl"]) == 6 == len(c5["inactive"]) == len(c5["effect_share"]) and c5["limit"] == 0.05 and c5["n"] == 120
    assert 2 <= c5["n_active"] <= 4 and sum(c5["inactive"]) == 6 - c5["n_active"], "two hidden factors, six codes: some codes are inactive"
    assert any(k < c5["limit"] for k in c5["kl"]), "a code has collapsed (KL below the limit)"
    assert all(c5["effect_share"][i] < 0.01 or c5["kl"][i] < c5["limit"] for i in range(6) if c5["inactive"][i]), "inactive = unused (less than 1 % of the effect) or collapsed"
    assert sum(s for s, off in zip(c5["effect_share"], c5["inactive"]) if not off) > 0.98 and c5["evr_val"] > 0.98
    # ---- picture 7: the loss curves
    L = d["loss"]
    h, ov = L["healthy"], L["overfit"]
    assert h["n_train"] + h["n_val"] == 36 == ov["n_train"] + ov["n_val"] and ov["n_parameters"] > 5 * h["n_parameters"]
    assert h["epochs_run"] < 400 and h["best_epoch"] < h["epochs_run"], "the healthy training stops early and the best epoch is not the last"
    assert min(h["val"]) * 1.0 <= h["val"][-1] <= 1.6 * min(h["val"]) and h["train"][-1] < h["train"][0]
    assert ov["epochs_run"] == 400 and ov["val"][-1] > 1.3 * min(ov["val"]) and ov["best_epoch"] < 0.5 * ov["epochs_run"] and ov["train"][-1] < 0.5 * ov["train"][0], "the over-fitted one: held-out rises, training falls"
    assert ov["train"][-1] < h["train"][-1]


def agree(fresh: dict, stored: dict) -> None:
    """The recomputed numbers are those of the pictures, to what another computer (or a newer PyTorch) can change: PCA tightly, the networks loosely."""
    assert set(fresh) == set(stored)
    for key in ("linear", "bending"):
        f, s = fresh["compare"][key], stored["compare"][key]
        for a, b in zip(f["pca"], s["pca"]):
            assert abs(a - b) <= 0.02 * b + 1e-6, ("PCA", key, a, b)
        for name in ("ae", "linear"):
            for a, b in zip(f[name], s[name]):
                assert abs(a - b) <= 0.6 * b + 0.002, (name, key, a, b)
    assert fresh["one"]["n_pixels"] == stored["one"]["n_pixels"] and fresh["one"]["n_inputs"] == stored["one"]["n_inputs"] and fresh["one"]["n_parameters"] == stored["one"]["n_parameters"]
    assert fresh["smoothify"]["scan"]["values"] == stored["smoothify"]["scan"]["values"]
    assert abs(fresh["smoothify"]["scan"]["roughness_input"] - stored["smoothify"]["scan"]["roughness_input"]) <= 0.02 * stored["smoothify"]["scan"]["roughness_input"]
    assert fresh["loss"]["healthy"]["n_parameters"] == stored["loss"]["healthy"]["n_parameters"] and fresh["loss"]["overfit"]["n_parameters"] == stored["loss"]["overfit"]["n_parameters"]


def test_what_the_pictures_and_their_captions_say_is_true_of_the_numbers_they_are_drawn_from():
    d = stored_data()
    assert set(d) == {"compare", "one", "traversal", "smoothify", "collapse", "loss"}
    qualitative(d)
    src = guide_source()
    for phrase in ("two made-up data sets of 120 EEMs", "40 noisy made-up EEMs", "the same 36 noisy made-up EEMs"):
        assert phrase in src, phrase
    assert d["compare"]["linear"]["n"] == d["compare"]["bending"]["n"] == 120 and d["smoothify"]["n"] == 40


@pytest.mark.slow
def test_the_pictures_are_drawn_from_the_numbers_the_code_gives():
    pytest.importorskip("torch")
    import help_data as H
    fresh = json.loads(json.dumps(H.figure_data()))
    qualitative(fresh)
    try:
        agree(fresh, stored_data())
    except AssertionError as exc:
        raise AssertionError(f"web/help/data.js is not what the code gives now (run  PYTHONPATH=. runtime/env/bin/python tests/test_help_autoencoders.py --write-data): {exc}") from None


# ---------------------------------------------------------------------------------------------------------------- (v) links
def fluoviz_guide_ids(tab: str) -> set[str]:
    files = {"experimental": ["fpcaIntroPages.js", "fpcaGpPages.js", "fpcaResultsPages.js"], "saved": ["modelsIntroPages.js", "modelsUsePages.js", "modelsReadPages.js"]}[tab]
    return set(re.findall(r"export const \w+ = \{\s*id: '(\w+)'", "".join((fluoviz_web() / "help" / f).read_text(encoding="utf-8") for f in files)))


def test_every_link_of_the_guide_resolves():
    src = guide_source()
    links = re.findall(r"<GLink\b([^>]*)>", src)
    assert len(links) >= 6
    for attrs in links:
        to, tab = re.search(r'\bto="(\w+)"', attrs), re.search(r'\btab="(\w+)"', attrs)
        assert to, attrs
        if tab is None:
            assert to.group(1) in IDS, attrs
        else:
            assert tab.group(1) in ("experimental", "saved") and to.group(1) in fluoviz_guide_ids(tab.group(1)), attrs
    router = (fluoviz_web() / "router.js").read_text(encoding="utf-8")
    manifest = json.loads((ADDON / "manifest.json").read_text(encoding="utf-8"))
    paths = {n["path"] for n in manifest["provides"]["nav"]}
    plinks = re.findall(r'<PLink\b[^>]*\bpath="(/\w+)"', src)
    assert plinks
    for p in plinks:
        assert f"path: '{p}'" in router or p in paths, p
    assert 'query="{ section: \'addons\' }"' in src.replace(":query", "query") or ":query=\"{ section: 'addons' }\"" in src


def test_the_tab_is_after_the_built_in_tabs_and_help_loads_add_on_guides():
    help_js = (fluoviz_web() / "pages" / "Help.js").read_text(encoding="utf-8")
    assert "addonHelpTabs" in help_js and "AddonGuide" in help_js
    assert (fluoviz_web() / "components" / "AddonGuide.js").exists()
    guide = (WEB / "help.js").read_text(encoding="utf-8")
    for imp in re.findall(r"from '(/static/[^']+)'", guide + "".join((HELP / f).read_text(encoding="utf-8") for f in ("parts.js", "figures.js", *PAGE_FILES))):
        assert (fluoviz_root() / "fluoviz" / "web" / imp.removeprefix("/static/")).exists(), imp


# ---------------------------------------------------------------------------------------------------------------- (vi) plain English
JARGON = ("payload", "endpoint", "numpy", "pickle", "JSON", "API ", "backend", "tensor", "gradient", "backpropagation", "hyperparameter", "stochastic", "logit", "softmax", "optimizer", "optimiser",
          "convolution", "ReLU", "GPU kernel", "latent space of")
# the words of the neural network: defined once, in bold, on the page where the reader meets them, and again in the glossary: {the word in bold: the start of its glossary entry}
TERMS = {"code": "Code (latent code)", "latent code": "Code (latent code)", "encoder": "Decoder, encoder", "decoder": "Decoder, encoder", "neural network": "Neural network", "layers": "Layer, neuron",
         "weights": "Weight", "neuron": "Layer, neuron", "activation": "Activation", "trained": "Training, training samples", "loss": "Loss", "epoch": "Epoch", "prior": "Prior",
         "KL divergence": "KL divergence", "β": "β (beta)", "collapse": "Collapse (inactive code)", "early stopping": "Early stopping, patience", "over-fitting": "Over-fitting (memorising)",
         "held-out samples": "Held-out (validation) samples", "linear shortcut": "Linear shortcut", "smoothify": "Smoothify", "variational autoencoder": "Variational autoencoder (VAE)", "autoencoder": "Autoencoder (AE)",
         "training samples": "Training, training samples", "Dropout": "Dropout", "Weight decay": "Weight decay", "Denoising": "Denoising", "Learning rate": "Learning rate", "Batch size": "Batch size",
         "random start": "Random start", "seed": "Seed", "bootstrap": "Bootstrap", "synthetic": "Synthetic samples", "variance explained": "Variance explained", "Sensitivity": "Sensitivity"}


def test_the_guide_is_in_plain_english_and_every_word_of_the_network_is_defined_and_in_the_glossary():
    text = guide_text()
    for word in JARGON:
        assert word not in text, word
    entries = glossary_entries()
    keys = [k for k, _ in entries]
    assert len(entries) >= 35 and all(len(v) > 40 for _, v in entries) and len(set(keys)) == len(keys)
    bold_by_page = {pid: [plain(b).lower() for b in re.findall(r"<b>(.*?)</b>", block, re.S)] for pid, block in page_blocks().items()}
    no_gloss, no_bold = [], []
    for term, gloss in TERMS.items():
        if not any(k.startswith(gloss) for k in keys):
            no_gloss.append(gloss)
        if not [pid for pid, bolds in bold_by_page.items() if any(b.rstrip(".,:") == term.lower() or b.startswith(term.lower() + " ") for b in bolds)]:
            no_bold.append(term)
    assert not no_gloss, f"the glossary has no entry for: {no_gloss}"
    assert not no_bold, f"never defined (in bold): {no_bold}"
    # the word of the network that the reader meets first on a page is explained on that page or an earlier one
    seen_order = {pid: i for i, pid in enumerate(IDS)}
    for term in ("latent", "encoder", "decoder", "neuron", "activation", "epoch", "loss", "prior", "KL divergence", "collapse"):
        first = next(pid for pid in IDS if re.search(rf"\b{re.escape(term)}\b", re.sub(r"<Ui>.*?</Ui>", "", page_blocks()[pid]), re.I))
        defined = [pid for pid, bolds in bold_by_page.items() if any(term.lower() in b for b in bolds)]
        assert defined and min(seen_order[p] for p in defined) <= seen_order[first], (term, first, defined)


def test_the_words_are_used_the_way_the_wording_rules_say():
    text = guide_text()
    # 'rebuilt' is the word of the page for what the decoder draws; the guide says 'held-out' for samples the network did not learn from; synthetic samples are never called data without a no
    for w in ("held-out", "rebuilt", "synthetic", "smoothify", "variance explained"):
        assert w in text, w
    assert "Synthetic samples are not data" in text
    assert text.count("EEM") > 60
    # the sentence case of the headings, and the order of the sections of every page
    for pid, block in page_blocks().items():
        for h in re.findall(r"<h2>(.*?)</h2>", block):
            assert h[0].isupper() or h[0].isdigit(), (pid, h)


def test_the_user_note_describes_what_the_manifest_says():
    note = (ADDON / "docs" / "README_user.md").read_text(encoding="utf-8")
    manifest = json.loads((ADDON / "manifest.json").read_text(encoding="utf-8"))
    assert f"**{manifest['extra_download_mb']} MB of extra libraries**" in note and manifest["name"] in note and manifest["version"] in note
    for word in ("Settings", "Install", "Remove", "Also remove the extra libraries", "runtime/", "models/", ".fluoviz-model", "Saved models", "never", "offline", "Experimental methods"):
        assert word in note or word.lower() in note.lower(), word
    assert len(note.splitlines()) < 90, "a one-page note"
    assert "This model needs the add-on 'Autoencoders (AE, VAE)'" in note
    assert {"linux-64", "win-64", "osx-arm64"} <= set(manifest["platforms"])


# ---------------------------------------------------------------------------------------------------------------- (vii) the names and rules of the code that the guide describes
def test_the_variables_the_guide_names_are_the_ones_the_api_publishes():
    block = page_blocks()["use"]
    names = set(re.findall(r'<span class="mono">([^<]+)</span>', block))
    assert {"z1", "zd", "recon_rmse", "recon_r2", "flag", "split", "z1_sd", "kl", "elbo", "cluster", "membership"} <= names, names
    api = API_MD.read_text(encoding="utf-8")
    for n in sorted(names):
        base = {"zd": "zd", "z1_sd": "z1_sd"}.get(n, n)
        assert f"`{base}" in api or base in api, n
    src = (PKG / "kinds" / "autoencoder.py").read_text(encoding="utf-8") + (PKG / "server" / "payload.py").read_text(encoding="utf-8")
    for n in ("recon_rmse", "recon_r2", "flag", "split", "kl", "elbo", "cluster", "membership"):
        assert f'"{n}"' in src, n
    assert "_sd" in src


def test_the_statuses_and_the_options_of_the_saved_model_are_those_of_the_kind():
    from fluoviz.analysis.frozen import STATUS_WORDS
    from fluoviz_addon_autoencoders.kinds import autoencoder as AE
    said(STATUS_WORDS["ok"], STATUS_WORDS["outside"], STATUS_WORDS["unusable"], "Limit for the reconstruction error", "Smallest typicality of the code")
    ae, vae = {o["name"]: o for o in AE.options_for("ae")}, {o["name"]: o for o in AE.options_for("vae")}
    assert ae["limit"]["label"] == "Limit for the reconstruction error" and ae["limit"]["default"] == "99" and {c["value"] for c in ae["limit"]["choices"]} == {"95", "99"}
    assert ae["typicality"]["label"].startswith("Smallest typicality of the code") and ae["typicality"]["default"] == 0.0 and vae["typicality"]["default"] == 0.01
    said("99 % by default", "0.01 by default for a variational autoencoder; 0 for a plain one")


def test_the_rules_of_the_scans_and_the_limits_are_those_of_the_code():
    import inspect
    from fluoviz_addon_autoencoders.core import diagnostics as D
    src = inspect.getsource(D)
    assert "[0.0] + list(np.geomspace(1e-4, 10.0, 11))" in src, "the smoothify scan tries 0 and eleven values from 0.0001 to 10"
    assert re.search(r"tol = max\(float\(se_a\[0\]\)[^\n]*0\.02 \* err_a\[0\]", src), "the suggestion: within one standard error or 2 % of the error without smoothify"
    assert "within one standard error" in src and "1 %" in src, "the recommended number of codes"
    said("equal steps on a logarithmic scale", "within one standard error (or 2 %)", "within one standard error (or 1 % of the variation)")
    assert inspect.signature(D.robust_limit).parameters["c"].default == 3.0 and inspect.signature(D.default_limit).parameters["level"].default == 0.99
    said("c is 3 unless you change it", "about 1 ordinary sample in 100 lies above it")
    e = np.exp(np.random.default_rng(0).normal(-3, 0.5, 200))
    assert np.mean(e > D.default_limit(e, 0.99)["limit"]) <= 0.03 and np.mean(e > D.default_limit(e, 0.95)["limit"]) <= 0.10 and D.robust_limit(e, "mad", 3.0)["limit"] > np.median(e), "the limit of a 99 % level lets about 1 in 100 ordinary errors above"


def test_the_three_verdicts_on_the_stability_of_the_codes():
    from fluoviz_addon_autoencoders.core import diagnostics as D
    rng = np.random.default_rng(0)
    base = rng.normal(size=(40, 3))
    same = np.stack([base, base @ np.linalg.qr(rng.normal(size=(3, 3)))[0], -base])            # the same codes, rotated and flipped: reproducible
    assert "are reproducible" in D.stability_from_codes(same)["verdict"]
    part = np.stack([base, base + 0.6 * rng.normal(size=base.shape), base + 0.6 * rng.normal(size=base.shape)])
    assert "only partly reproducible" in D.stability_from_codes(part)["verdict"] or "NOT reproducible" in D.stability_from_codes(part)["verdict"]
    other = np.stack([rng.normal(size=(40, 3)) for _ in range(3)])
    assert "NOT reproducible" in D.stability_from_codes(other)["verdict"]
    said("The codes are reproducible", "The codes are only partly reproducible", "The codes are NOT reproducible", "a mean of 0.95 or more", "multiple correlation of 0.8 or more", "at least half of the codes found again")


def fake_result(error_train: float, error: float, best_epoch: int = 50, spread: float = 0.0, n_runs: int = 3):
    """The few fields of a training result that the verdicts read."""
    from types import SimpleNamespace as NS
    runs = [{"evr_val": 0.95 - spread * i / max(n_runs - 1, 1)} for i in range(n_runs)]
    return NS(validation={"error_train": error_train, "error": error, "evr_train": 1 - error_train, "evr": 1 - error, "n_val": 24}, restarts={"n": n_runs, "runs": runs},
              config={"epochs": 600, "patience": 60}, split={"mode": "holdout"}, epochs_run=110, best_epoch=best_epoch, spec=NS(latent_dim=3))


def test_the_verdict_rules_are_those_of_diagnostics():
    from fluoviz_addon_autoencoders.core import diagnostics as D
    # the words on the quality
    assert "excellent" in D.quality_verdict(0.995, 24, 3) and "(good;" in D.quality_verdict(0.96, 24, 3) and "(fair;" in D.quality_verdict(0.92, 24, 3) and "More codes may help" in D.quality_verdict(0.92, 24, 3)
    assert "(poor;" in D.quality_verdict(0.8, 24, 3) and "Try more codes, a larger network or more training epochs" in D.quality_verdict(0.8, 24, 3) and "(good;" in D.quality_verdict(0.95, 24, 3)
    said("excellent (99 % or more), good (95 % or more), fair (90 % or more", "poor (below 90 %")
    # overfitting: held-out error more than 3 times the training error and more than 0.01 above = clear; more than 1.6 times and 0.003 above = mild
    assert D.training_verdict(fake_result(0.01, 0.05))["overfit"] == "clear" and D.training_verdict(fake_result(0.02, 0.05))["overfit"] == "mild"
    assert D.training_verdict(fake_result(0.01, 0.025))["overfit"] == "mild" and D.training_verdict(fake_result(0.01, 0.0155))["overfit"] == "none"
    assert D.training_verdict(fake_result(0.001, 0.0035))["overfit"] == "none", "a five-fold ratio between two tiny errors is not overfitting"
    assert D.training_verdict(fake_result(0.01, 0.0405))["overfit"] == "clear" and D.training_verdict(fake_result(0.01, 0.0195))["overfit"] == "mild"
    said("more than three times the training error and more than one point of the variation above it", "more than 1.6 times and more than 0.3 point above",
         "a five-fold ratio between two tiny errors is not overfitting")
    # the start from PCA, and the random starts
    assert "could not improve on its PCA starting point" in D.training_verdict(fake_result(0.01, 0.02, best_epoch=-1))["stopping_words"]
    assert "agree" in D.training_verdict(fake_result(0.01, 0.02, spread=0.019))["restarts_words"] and "the result depends on the start" in D.training_verdict(fake_result(0.01, 0.02, spread=0.025))["restarts_words"]
    said("less than 2 points of variance explained apart", "The network could not improve on its PCA starting point on held-out samples")


def test_the_comparison_with_pca_follows_the_rule_of_the_code():
    from types import SimpleNamespace as NS
    from fluoviz_addon_autoencoders.core import diagnostics as D
    rng = np.random.default_rng(3)
    base = np.exp(rng.normal(-3.0, 0.4, 40))
    null = np.full(40, 0.5)

    def verdict(net):
        evr = lambda e: 1 - e.sum() / null.sum()
        res = NS(samples={"error_heldout": net, "error_baseline": base, "null": null}, validation={"evr": evr(net), "baseline": {"name": "Functional PCA", "k": 3, "evr": evr(base)}})
        return D.compare_baseline(res)
    assert verdict(base * 0.6)["better"] == "network" and verdict(base * 1.5)["better"] == "baseline"
    assert verdict(base * 0.9 * np.exp(rng.normal(0, 0.3, 40)))["better"] == "same" and verdict(base * 0.95)["better"] == "same", "less than 15 % lower is not better"
    assert "safer choice" in verdict(base)["verdict"]
    said("more than 15 % lower", "differ by at least 0.3 point", "the linear method is the safer choice", "fewer than 5 held-out samples there is no verdict".replace("fewer", "With fewer").replace("With With", "With") if False else "With fewer than 5 held-out samples there is no verdict")
    assert D.compare_baseline(NS(samples={"error_heldout": base[:4], "error_baseline": base[:4], "null": null[:4]}, validation={"evr": 0.9, "baseline": {"name": "PCA", "k": 2, "evr": 0.9}}))["verdict"].startswith("Only 4 held-out samples")


def test_the_health_of_the_codes_and_the_smoothness_follow_the_rules_of_the_code():
    import inspect
    from fluoviz_addon_autoencoders.core import diagnostics as D
    from fluoviz_addon_autoencoders.core.spec import Config, ModelSpec
    rng = np.random.default_rng(1)
    P, d, h = 12, 3, 8
    spec = ModelSpec.from_config(Config(family="vae", latent_dim=d, hidden=[h], skip=False), P, [{"name": "fluorescence", "slice": [0, P], "kind": "gp", "weight": 1.0}], sigma=0.3)
    arrays = {k: rng.normal(0, 0.4, v) for k, v in spec.array_shapes().items()}
    arrays["in.mean"], arrays["in.scale"], arrays["out.weights"] = np.zeros(P), np.ones(P), np.full(P, 1.0 / P)
    arrays["dec.W0"][:, 2] = 0.0                                                            # the decoder ignores code 3
    arrays = spec.check_arrays(arrays)
    codes = rng.normal(0, 1.5, (60, d))
    logvar = np.full((60, d), -2.0)
    codes[:, 1], logvar[:, 1] = 0.0, 0.0                                                    # code 2: the posterior equals the prior (KL = 0)
    health = D.latent_health(spec, arrays, rng.normal(size=(60, P)), codes=codes, logvar=logvar)
    assert list(health["unused"]) == [False, True, True] and list(health["collapsed"]) == [False, True, False] and list(health["inactive"]) == [False, True, True] and health["n_active"] == 1
    assert health["kl_per_dim"][1] < 0.05 < health["kl_per_dim"][0] and "collapsed" in health["words"]
    src = inspect.getsource(D)
    assert "share < 0.01" in src and "kl_dim < 0.05" in src and "corr_max > 0.5" in src and 'cover["inside95"] < 0.85' in src
    assert "ratio < 0.9" in src and "ratio > 1.1" in src
    said("less than 1 % of what the codes do to the rebuilt EEM", "below 0.05 nats", "(above 0.5)", "fewer than 85 % of the samples inside its 95 % region", "is below 0.9", "(0.9 to 1.1)", "above 1.1")


def test_the_same_seed_gives_the_same_network_and_a_linear_network_with_the_shortcut_is_pca():
    pytest.importorskip("torch")
    import help_data as H
    from fluoviz_addon_autoencoders.core import train
    prep = H.prepare(H.eems_linear(40, 7, 0.02))
    cfg = H.config(latent_dim=3, n_restarts=1, epochs=40, patience=40)
    a, b = train.fit(cfg, prep), train.fit(cfg, prep)
    assert all(np.array_equal(a.arrays[k], b.arrays[k]) for k in a.arrays), "the same seed, data and settings give the same network"
    c = train.fit(cfg.replace(seed=1), prep)
    assert any(not np.array_equal(a.arrays[k], c.arrays[k]) for k in a.arrays), "another seed gives another network"
    lin = train.fit(H.config(latent_dim=3, n_restarts=1, activation="linear"), prep)
    assert abs(lin.validation["evr"] - lin.validation["baseline"]["evr"]) < 0.01, "with the activation linear the network is PCA"
    said("The same data, settings, seed, computer and program versions give the same network", "with the shortcut on it is the PCA itself")


def test_the_references_are_the_ones_the_task_names_and_are_few():
    refs = [plain(r) for r in re.findall(r"<li>((?:Hinton|Baldi|Kingma|Higgins|Bowman|Vincent|Loshchilov|Ramsay|Rasmussen|Stedmon)[^<]*(?:<i>.*?</i>[^<]*)*)</li>", page_blocks()["limits"], re.S)]
    assert 8 <= len(refs) <= 14, len(refs)
    for author, year, where in (("Hinton", "2006", "Science"), ("Kingma", "2014", "Auto-encoding variational Bayes"), ("Higgins", "2017", "beta-VAE"), ("Vincent", "2008", "denoising autoencoders"),
                                ("Ramsay", "2005", "Functional Data Analysis"), ("Rasmussen", "2006", "Gaussian Processes for Machine Learning"), ("Baldi", "1989", "Neural Networks"),
                                ("Bowman", "2016", "Generating sentences from a continuous space"), ("Loshchilov", "2019", "Decoupled weight decay regularization"), ("Stedmon", "2008", "PARAFAC")):
        hit = [r for r in refs if r.startswith(author) and f"({year})" in r and where in r]
        assert hit, (author, year)
    for r in refs:
        assert re.search(r"\(\d{4}\)", r), r


def test_the_docs_point_to_the_guide():
    assert f"#/help?tab={TAB}" in UI_MD.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------------------------------------------- (viii) in a real browser
def playwright_python() -> str | None:
    cand = os.environ.get("FLUOVIZ_PLAYWRIGHT_PYTHON")
    if cand and Path(cand).exists():
        return cand
    try:
        import playwright  # noqa: F401
        return sys.executable
    except ImportError:
        return None


@pytest.mark.slow
def test_the_guide_in_a_browser_has_no_leftovers_no_console_error_no_sideways_scroll_and_whole_pictures():
    py = playwright_python()
    if py is None:
        pytest.skip("no python with Playwright at hand (set FLUOVIZ_PLAYWRIGHT_PYTHON)")
    # Chrome puts a socket in its temporary folder and refuses a path longer than ~100 characters: the project's own TMPDIR (scripts/env.sh isolates it inside the project) is too long
    env = {**os.environ, "TMPDIR": "/tmp"}
    shots = ["--shots", os.environ["FLUOVIZ_HELP_SHOTS"]] if os.environ.get("FLUOVIZ_HELP_SHOTS") else []          # screenshots of every page and picture, to look at
    done = subprocess.run([py, str(TESTS / "help_browser.py"), "--fluoviz", str(fluoviz_root()), "--json", *shots], capture_output=True, text=True, timeout=3600, cwd=str(fluoviz_root()), env=env)
    out = done.stdout.strip().splitlines()
    assert out, done.stderr[-2000:]
    report = json.loads(out[-1])
    assert not report["problems"], report["problems"]
    assert len(report["pages"]) == len(IDS) * 4 and report["links"] >= 8
    assert sum(v for k, v in report["figures"].items() if k.endswith("light-1300")) == 7, "seven pictures in the guide"


if __name__ == "__main__":
    print("run with --write-data to regenerate web/help/data.js; or run the tests with FluoViz's  bash scripts/run_tests.sh <this file>")
