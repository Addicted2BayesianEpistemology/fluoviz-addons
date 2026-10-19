"""The documents of the Nonlinear effects tab say what the code does (no PyTorch needed).

``docs/api.md`` (section 2.15), ``docs/ui.md`` (the tab, its labels, its fixed sentences and its test hooks) and ``SPEC.md`` are the contract; these tests read them next to the code and fail when one of the
two changes without the other: the thresholds of the words, the limits of the controls, the four routes, the sentences of the page, every ``data-ae-nl-*`` hook.  (The Help page has its own checks in
``test_help_autoencoders.py``.)"""
import re
from pathlib import Path

import pytest

from fluoviz_addon_autoencoders.core import curves as C
from fluoviz_addon_autoencoders.core import nonlinear as NL
from fluoviz_addon_autoencoders.server import nlrequest as Q
from fluoviz_addon_autoencoders.server import nonlinear as V

ADDON = Path(__file__).resolve().parents[1]
PKG = ADDON / "fluoviz_addon_autoencoders"
WEB = ADDON / "web" / "components"
PAGE_FILES = ["AeNonlinearTab.js", "AeNlState.js", "AeNlActions.js", "AeNlCommon.js", "AeNlResponse.js", "AeNlBends.js", "AeNlLocal.js", "AeNlPair.js"]
PY_FILES = ["core/curves.py", "core/features.py", "core/nonlinear.py", "server/nlrequest.py", "server/nonlinear.py"]


def squash(text: str) -> str:
    """Text as a reader sees it in a document: no bold marks or code ticks, one space between words, a real minus."""
    return re.sub(r"\s+", " ", text.replace("**", "").replace("`", "")).strip()


def section_of_api() -> str:
    text = (ADDON / "docs" / "api.md").read_text(encoding="utf-8")
    start = text.index("### 2.15 ")
    end = text.index("\n## 3. ", start)
    return text[start:end]


API = squash(section_of_api())
API_RAW = section_of_api()
UI_RAW = (ADDON / "docs" / "ui.md").read_text(encoding="utf-8")
UI = squash(UI_RAW)
SPEC = squash((ADDON / "SPEC.md").read_text(encoding="utf-8"))
PAGE = "\n".join((WEB / f).read_text(encoding="utf-8") for f in PAGE_FILES)
PAGE_SQ = squash(PAGE.replace("{{ pctText(P.nearly_straight) }}", "5 %"))                 # the page writes the number the server sends (V.NEARLY_STRAIGHT, checked below)
CODE = squash("\n".join((PKG / f).read_text(encoding="utf-8") for f in PY_FILES))


def pct(x: float) -> str:
    return f"{x * 100:g} %"


# ---------------------------------------------------------------------------------------------------------------- api.md
def test_the_four_routes_are_documented_and_served():
    router = (PKG / "server" / "router.py").read_text(encoding="utf-8")
    for name in ("response", "bends", "local", "pair"):
        assert f"POST /{{rid}}/{name}" in API_RAW, name
        assert re.search(rf'"/{{rid}}/{name}"', router), f"the route /{{rid}}/{name} is not registered"
    assert "### 2.15 The nonlinear effects of the codes: `POST /{rid}/response`, `/bends`, `/local`, `/pair`" in API_RAW


def test_the_thresholds_of_the_words_are_the_ones_the_document_states():
    # the words of a curve (core/curves.py)
    assert (C.FLAT, C.LINEAR_TOL, C.TURN, C.PLATEAU, C.PLATEAU_SHARE, C.SIGN_TOL) == (0.02, 0.05, 0.10, 0.05, 0.25, 0.02)
    for what, value in (("below **2 %** of the largest range among the codes", C.FLAT), ("`max_deviation` ≤ **5 %**", C.LINEAR_TOL), ("by at least **10 %** of its range", C.TURN),
                        ("stays within **5 %** of its range of the value at the end", C.PLATEAU), ("over at least **25 %** of the positions", C.PLATEAU_SHARE), ("a residual counts for the sign pattern when it is above **2 %** of the range", C.SIGN_TOL)):
        assert squash(what) in API, what
        assert pct(value) in what.replace("**", ""), (what, value)
    # the maps (core/nonlinear.py) and the words of the four answers (server/nonlinear.py)
    assert NL.MIN_EFFECT == 0.02 and "below **2 %** (`min_effect`) of the largest amplitude" in API_RAW.replace("\n", " ")
    assert (V.NEARLY_STRAIGHT, V.SMALL_INTERACTION, V.LARGE_INTERACTION, V.SAME_SHAPE, V.LITTLE_SHAPE, V.SAME_STRENGTH) == (0.05, 0.05, 0.25, 0.005, 0.05, 0.05)
    for what in ("almost like a straight line* below **5 %**", "*act almost independently* below an index of **0.05**, *interact a little* below **0.25**", "when `change.mean` is below **0.005**, *\"changes a little from sample to sample\"* below **0.05**",
                 "when `size_sd` is below **0.05**"):
        assert what in API_RAW.replace("\n", " "), what


def test_the_limits_and_the_defaults_of_the_controls_are_the_ones_the_document_states():
    assert Q.SPAN == (0.5, 4.0) and Q.POINTS == (5, 201) and Q.GRIDS == (3, 5, 7) and Q.HEAT == (11, 61) and Q.MAX_FEATURES == 6 and Q.MAX_LOADINGS == 6 and NL.MAX_ICE == 12 and Q.SHOWN == 8
    for what in ("`span` 0.5 to 4", "`n_points` 5 to 201", "defaults 41 and 3", "`grid` 3, 5 or 7 positions per code", "(default 2) SD", "`heat_points` 11 to 61 (default 31)", "At most 6 features.", "`ice.n` (1–12)",
                 "at most 6 samples", "`ice.samples` (at most 12 names)"):
        assert squash(what) in API, what
    assert "the first 8 codes, 12 samples" in API
    # the defaults really are those (an empty request)
    from core_support import GP, cached, prepared
    frame = cached(("nl", "gp", False), lambda: prepared({"fluorescence": {"basis": GP}}, n=24, absorbance=False)).frame
    r = Q.parse_response({}, [], 3, frame)
    assert (r["span"], r["n_points"], r["overlay"]) == (3.0, 41, True)
    b = Q.parse_bends({})
    assert (b["span"], b["n_points"]) == (3.0, 41)
    p = Q.parse_pair({}, 3, frame)
    assert (p["grid"], p["span"], p["heat_points"]) == (5, 2.0, 31)
    assert NL.LOADINGS == (0.1, 0.5, 0.9) and "the 10th, 50th and 90th percentile of the code" in API


def test_the_sentences_of_the_document_are_those_of_the_code():
    sentences = [
        # the errors that tell the person what to change
        "Two codes are needed to look at two codes together: this network has only one.", "'codes' must be two code numbers, for example [1, 2].", "Choose two different codes.",
        "'grid' (pictures per code) must be 3, 5 or 7.", "must be true or false.", "'ice' must be true, false or a group of settings (n, samples, codes).", "A sample can be chosen only once.",
        "At most 6 features can be drawn at once.", "This network does not use the absorbance, so there is no absorbance to look at.",
        "The coefficients of a basis exist only for a functional network; this network works on the pixels. Choose another feature.",
        "is not part of the model (it is scatter, or below the Rayleigh line, or was cut out).", "Choose a pixel in the coloured part of the picture.",
        "Choose what to measure: the kind of a feature must be one of:", "Unknown setting '", "The data changed since this network was trained, so the training samples are not drawn",
        # the words of the answers
        "bends the most", "cannot be described by a straight line", "Every code acts almost like a straight line (the largest share is", "this network is nearly linear.", "not change the output at all",
        "act almost independently: the change of the picture is the sum of what each does alone", "interact a little", "interact strongly",
        "has the same shape for every sample", "changes a little from sample to sample", "changes shape from sample to sample", "and about the same strength.", "(one standard deviation)",
        "does not change the output, so it has no loading to compare.", "A straight line explains", "at its worst the curve is",
    ]
    for piece in ("how far each code is moved, in SD of the code", "points of the heat map per code", "positions per curve"):                  # "'span' (how far each code ...) must be a number from 0.5 to 4."
        assert piece in CODE and piece in API, piece
    doc = API.replace("*", "").replace("\\", "")
    not_in_code = [x for x in sentences if x not in CODE]
    not_in_doc = [x for x in sentences if squash(x) not in doc and squash(x).replace("'", "") not in doc.replace("'", "")]
    assert not not_in_code, f"the code does not have: {not_in_code}"
    assert not not_in_doc, f"api.md does not have: {not_in_doc}"
    # the labels of the curves (describe_curve): all of them are in the table of the document
    for label in ("has almost no effect on this feature", "changes direction several times", "rises then falls (peak at about", "falls then rises (lowest at about", "nearly linear", "S-shaped (levels off below about",
                  "levels off (saturates) above about", "has a bend (convex: it bends upwards, the slope grows with the code)", "has a bend (concave: it bends downwards, the slope shrinks with the code)", "switches sign near"):
        assert label in CODE, label
        assert label in API.replace("*", ""), label


# ---------------------------------------------------------------------------------------------------------------- ui.md
def nonlinear_hooks_in_the_document() -> set[str]:
    line = next(l for l in UI_RAW.splitlines() if l.startswith("* **Step 7**"))
    part = line[line.index("Nonlinear effects: "): line.index("  Samples: ")]
    return set(re.findall(r"data-ae-[a-z0-9]+(?:-[a-z0-9]+)*", part))


def hooks_in_the_code() -> set[str]:
    return set(re.findall(r"data-ae-(?:nl-[a-z0-9]+(?:-[a-z0-9]+)*|nonlinear-tab)", PAGE))


def test_every_test_hook_of_the_tab_is_in_the_document_and_every_documented_hook_is_in_the_page():
    doc, code = nonlinear_hooks_in_the_document(), hooks_in_the_code()
    assert len(code) >= 60, len(code)
    assert code - doc == set(), f"hooks of the page that docs/ui.md does not list: {sorted(code - doc)}"
    assert doc - code == set(), f"hooks that docs/ui.md lists and the page does not have: {sorted(doc - code)}"
    # the values of the switch of sections and the tab itself
    assert "data-ae-nl-section=response|bends|local|pair" in UI_RAW
    state = (WEB / "AeNlState.js").read_text(encoding="utf-8")
    for sec in ("response", "bends", "local", "pair"):
        assert f"'{sec}'" in state or f"id: '{sec}'" in PAGE, sec
    assert "data-ae-tab=latent|traversals|nonlinear|samples|interpolation|generate" in UI_RAW
    results = (WEB / "AeResults.js").read_text(encoding="utf-8")
    assert "{ id: 'nonlinear', label: 'Nonlinear effects' }" in results and results.index("id: 'traversals'") < results.index("id: 'nonlinear'") < results.index("id: 'samples'")


def phrases_of(prefix: str) -> list[str]:
    """The sentences in double quotes of the bullet of docs/ui.md that starts with ``prefix`` (a sentence ends with a full stop or a bracket before its closing quote)."""
    line = next(l for l in UI_RAW.splitlines() if l.startswith(prefix))
    return [squash(m) for m in re.findall(r'"([A-Z].*?[.)])"(?=;|\.|$)', line)]


def test_the_fixed_sentences_of_the_document_are_on_the_page_or_come_from_the_server():
    each = phrases_of("* Nonlinear effects, the sentence that starts each section:")
    other = phrases_of("* Nonlinear effects, other fixed phrases:")
    assert len(each) == 4 and len(other) >= 8, (len(each), len(other))
    seen = PAGE_SQ + " " + CODE
    missing = [p for p in each + other if p not in seen and p.replace("'", "’") not in seen and p.replace("\\'", "'") not in seen]
    assert not missing, f"docs/ui.md quotes sentences that neither the page nor the server has: {missing}"
    # every section starts with its sentence, in the order of the switch
    sections = (WEB / "AeNonlinearTab.js").read_text(encoding="utf-8")
    for s in each:
        assert s in squash(sections) or s in PAGE_SQ, s


def test_every_label_of_the_document_is_on_the_page():
    """Every word in bold in the four bullets of the tab (the labels) is in the files of the page, or written by the server."""
    lines = UI_RAW.splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith("* **Nonlinear effects**"))
    j = next(k for k, l in enumerate(lines) if l.startswith("* **Samples**"))
    labels = {squash(m) for m in re.findall(r"\*\*(.+?)\*\*", "\n".join(lines[i:j]))} - {"Traversals"}                  # (the tab it is placed after)
    assert len(labels) >= 50, len(labels)
    dynamic = {"Show the other N codes": "Show the other ", "z1: absorbance": "absorbance", "Loading of z1: the change of the EEM per SD of the code": "Loading of ", "Where the loading of z1 differs between samples": "Where the loading of ",
               "z1 across, z2 up (positions in SD of each code)": "across", "Intensity at the pixel this code moves most": "Intensity at the pixel {who} most", "Overall intensity (the average over the whole EEM)": "Overall intensity (the average over the whole EEM)"}
    seen = PAGE_SQ + " " + CODE
    missing = []
    for lab in sorted(labels):
        want = dynamic.get(lab, lab)
        if want not in seen and want.replace("’", "'") not in seen:
            missing.append(lab)
    assert not missing, f"docs/ui.md has labels that the page does not: {missing}"


def test_every_control_of_the_tab_has_a_tooltip_sentence():
    """Each control (a button, a field, a switch, a select) in the page files has a ``title`` or a ``help`` of one sentence (docs/ui.md: every control has a one-sentence tooltip)."""
    bad, n = [], 0
    for f in [PAGE_FILES[0], *PAGE_FILES[3:]]:
        for line in (WEB / f).read_text(encoding="utf-8").splitlines():
            for tag in re.findall(r"<(?:button|select|input|textarea|NumField)\b[^>]*>", line):
                if "data-ae-" not in tag:
                    continue
                n += 1
                # the sentence is on the control itself, or on the <Field help="..."> that holds it (the same line), or the control is a switch of the framework with its own help
                if not (re.search(r"\btitle=|:title=|\bhelp=|:help=|aria-label=", tag) or re.search(r"<Field\b.*?\bhelp=", line)):
                    bad.append((f, tag[:100]))
    assert n >= 20, n
    assert not bad, bad


def test_spec_names_the_tab_the_sections_and_the_document_of_the_routes():
    row = next(squash(l) for l in (ADDON / "SPEC.md").read_text(encoding="utf-8").splitlines() if "Nonlinear effects" in l and l.lstrip().startswith("*"))
    for word in ("Nonlinear effects", "api.md", "response curves", "where it bends", "local effects", "two codes"):
        assert word.lower() in row.lower(), word
    assert "2.15" in row
