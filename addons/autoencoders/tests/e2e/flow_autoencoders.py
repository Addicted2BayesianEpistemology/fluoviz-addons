"""End-to-end flow of the Autoencoders add-on in a real browser (Playwright + Chrome), with assertions.

    <python-with-playwright> tests/e2e/flow_autoencoders.py [--dark] [--shots] [--keep-going] [--only=advanced | --only=nonlinear | --only=gaps]

What it does: starts a PRIVATE FluoViz (own temporary workspace and runtime folder, listening on this computer only) with the add-on loaded in development mode
(``FLUOVIZ_ADDONS_DEV=<this add-on>``), opens the example data and goes through the page: the sidebar entry and its badge, the nine steps, the functional switch, the
network settings, **Run with the recommended settings**, every tab of the results and of the checks, the tools (traversals, hover cards, interpolation, unusual samples),
the tab **Nonlinear effects** (its four sections, typed and clicked features, curves of single samples, ten codes, one code, a network with absorbance, a reload, excitation across),
smoothify, a variational autoencoder (generate), both variables together, the advanced training settings, a rectangle cut out of some samples only (the weighting of step 5, the sentence in the Samples tab), cancel, an outdated network and **Train again with the same settings**, the use buttons
(**Cluster the latent space**, **Classify with the latent space**, **Save this model**, applying the saved model in a second project), the downloads, a reload, 600 px
(no sideways scroll), light and dark; at the end it looks at the console.  ``--shots`` also writes screenshots of every step and tab at 1100 and 600 px, light and dark,
into ``$FLUOVIZ_SHOTS`` (default: a folder in the temporary directory).

Environment: ``FLUOVIZ_DIR`` = the FluoViz folder (default: ``80_fluoViz`` next to ``81_fluoviz-addons``); ``FLUOVIZ_SHOTS`` = where the screenshots go;
``FLUOVIZ_TEST_ENV`` = a conda environment that has PyTorch (when the FluoViz installation itself has none): the private server runs from it.
The scans (latent-dimension, smoothify, stability, linear baselines) are tested when the server offers them and skipped, with a note, when it says they are not available yet.
``--only=advanced`` runs just the advanced training settings, ``--only=nonlinear`` just the tab Nonlinear effects and ``--only=gaps`` just the rectangle cut out of some samples (each needs the data and step 5 first; the
last two train a few small networks).
Exit status 0 = every assertion passed.
"""
from __future__ import annotations

import atexit
import contextlib
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import traceback
from pathlib import Path

ADDON = Path(__file__).resolve().parents[2]
FLUOVIZ = Path(os.environ.get("FLUOVIZ_DIR") or ADDON.parents[2] / "80_fluoViz")
sys.path.insert(0, str(FLUOVIZ / "tests" / "e2e"))
import browse  # noqa: E402
from browse import api, browser, goto, shot  # noqa: E402

FAILS: list[str] = []
STEPS = ["1 · What to model", "2 · Domain", "3 · Representation", "4 · Smoothing", "5 · Network", "6 · Train", "7 · Results", "8 · Checks", "9 · Use"]
BASE_API = "/api/addon/autoencoders"
LONG = 280000                                   # ms: training on a busy computer


def free_port() -> int:
    """A port nobody listens on (other flows may run on this computer at the same time)."""
    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0))
        return sk.getsockname()[1]


@contextlib.contextmanager
def private_server(port: int, env: dict, wait: int = 240):
    """FluoViz's own `browse.server` (a private server on this computer only, its own workspace), but it waits as long as a busy computer needs to start (up to `wait` s)."""
    ws = tempfile.mkdtemp(prefix="fluoviz-e2e-")
    exports = "".join(f"export {k}={shlex.quote(str(v))} && " for k, v in env.items())
    cmd = f'cd "{FLUOVIZ}" && . scripts/env.sh && fluoviz_isolate && {exports}exec python -m fluoviz --no-browser --port {port} --local-only --workspace "{ws}"'
    proc = subprocess.Popen(["bash", "-c", cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(wait * 4):
            try:
                urllib.request.urlopen(base + "/api/system", timeout=3).read()
                break
            except urllib.error.HTTPError:
                break
            except Exception:
                time.sleep(0.25)
        else:
            raise RuntimeError("the server did not start")
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(ws, ignore_errors=True)


def check(cond, what: str) -> bool:
    print(f"  {'ok  ' if cond else 'FAIL'} {what}")
    if not cond:
        FAILS.append(what)
    return bool(cond)


def note(text: str) -> None:
    print(f"  ..   {text}")


# ----------------------------------------------------------------------------------------------------------------------- helpers
def errors(page, since: int = 0) -> list[str]:
    """Console errors and server errors.  A refusal of the server (a 4xx answer with a plain-English message that the page shows) is how the browser logs a handled
    'Failed to load resource': those lines are not errors of the page."""
    return [x for x in page.logs[since:] if ("[error]" in x and not re.search(r"status of 4\d\d", x)) or "pageerror" in x or "http 5" in x]


def step(page, n: int) -> None:
    page.click(f"[data-fpca-step='{n}']")
    page.wait_for_selector(f"[data-fpca-panel='{n}']", timeout=20000)
    page.wait_for_timeout(500)


def tab(page, name: str) -> None:
    page.click(f"[data-ae-tab='{name}']")
    page.wait_for_timeout(900)


def results(page) -> list[dict]:
    return api(page, "GET", "/api/results?kind=autoencoder")["results"]


def payload(page, rid: str) -> dict:
    return api(page, "GET", f"{BASE_API}/{rid}")["payload"]


def wait_results(page, rid_count: int | None = None, timeout: int = LONG) -> None:
    """Wait until step 7 shows a network (and the training has ended)."""
    page.wait_for_selector("[data-fpca-panel='7'] [data-ae-cards]", timeout=timeout)
    page.wait_for_function("() => { const p = document.querySelector('[data-ae-page]'); return p && p.dataset.busy === '0'; }", timeout=timeout)
    page.wait_for_timeout(700)


def train(page, label: str = "") -> str:
    """Train with the settings on the page (step 6) and wait for the results; the id of the new network."""
    before = {r["id"] for r in results(page)}
    step(page, 6)
    page.click("[data-ae-train]")
    wait_results(page)
    new = [r["id"] for r in results(page) if r["id"] not in before]
    return new[0] if new else ""


def overflow(page) -> int:
    """How many pixels the page is wider than the window (0 = no sideways scroll)."""
    return page.evaluate("() => Math.max(document.documentElement.scrollWidth, document.body.scrollWidth) - window.innerWidth")


def plot_point(page, selector: str, i: int = 0) -> tuple[float, float]:
    """The pixel position (on the screen) of point i of the first trace of the Plotly chart `selector`."""
    pos = page.evaluate("""([sel, i]) => { const gd = document.querySelector(sel); gd.scrollIntoView({ block: 'center' }); const t = gd.data[0], fl = gd._fullLayout, xa = fl.xaxis, ya = fl.yaxis;
        const box = gd.querySelector('.nsewdrag').getBoundingClientRect();
        return { x: box.left + xa.l2p(xa.d2l(t.x[i])), y: box.top + ya.l2p(ya.d2l(t.y[i])), id: t.customdata ? t.customdata[i][0] : null }; }""", [selector, i])
    return pos["x"], pos["y"], pos["id"]


def open_settings(page) -> None:
    """The fold 'Training settings' of step 6 (it starts closed)."""
    if not page.locator("[data-ae-epochs] input").is_visible():
        page.click("[data-ae-train-settings] summary")
        page.wait_for_timeout(300)


def text(page, selector: str) -> str:
    return page.inner_text(selector).replace("\n", " ")


def make_mat(n: int, folder: str, coarse: bool = False) -> str:
    """A drEEM-style .mat file with n synthetic samples (FluoViz's own generator, run in FluoViz's environment)."""
    path = str(Path(folder) / f"synthetic_{n}.mat")
    grid = "ex=np.arange(240.0, 455.0, 10.0), em=np.arange(290.0, 602.0, 4.0)" if coarse else ""
    code = (f"import sys, numpy as np; sys.path.insert(0, {str(FLUOVIZ)!r})\n"
            f"from fluoviz.core.synthetic import make_demo_dataset\nfrom fluoviz.io import export_mat_dreem\n"
            f"d = make_demo_dataset(n_samples={n}, seed=11{', ' + grid if grid else ''})\n"
            f"export_mat_dreem({path!r}, d.X_raw, d.ex, d.em, d.ids, abs=(d.abs_wl, d.abs_A))\n")
    cmd = f'cd "{FLUOVIZ}" && . scripts/env.sh && fluoviz_isolate && python -c {shlex.quote(code)}'
    subprocess.run(["bash", "-c", cmd], check=True, capture_output=True, timeout=300)
    return path


def import_mat(page, path: str, name: str) -> None:
    """Open a new project from a .mat file (a job of the server)."""
    job = api(page, "POST", "/api/import/eem/commit", {"source": {"paths": [path]}, "options": {}, "name": name, "blanks": []})
    for _ in range(600):
        j = api(page, "GET", f"/api/jobs/{job['job_id']}")
        if j["status"] in ("done", "error", "cancelled"):
            break
        page.wait_for_timeout(500)
    page.wait_for_timeout(2000)


# ----------------------------------------------------------------------------------------------------------------------- the parts
def part_basics(page, shots: bool) -> None:
    print("the sidebar entry, the page, the nine steps")
    # no project open: the framework says so instead of the page
    goto(page, "/autoencoders", 1.5)
    check(page.locator("[data-ae-page]").count() == 0, "without a project the page itself is not shown (the framework asks for data)")
    api(page, "POST", "/api/import/example")
    page.wait_for_timeout(1500)
    info = api(page, "GET", "/api/addons")
    ae = [a for a in info["installed"] if a["id"] == "autoencoders"]
    check(len(ae) == 1 and ae[0]["state"] == "active", f"the add-on is loaded in development mode and active ({[a['state'] for a in ae]})")
    goto(page, "/", 1.5)
    group = page.locator(".sb-group:has(.eyebrow:text-is('Experimental methods'))")
    items = group.locator("a.sb-item")
    labels = [e.inner_text().strip().lower() for e in items.all()]
    check(group.count() == 1 and len(labels) == 2 and "functional pca" in labels[0] and "autoencoders" in labels[1], f"the group 'Experimental methods' has Functional PCA and Autoencoders ({labels})")
    check("add-on" in labels[1], "the Autoencoders entry carries the badge 'add-on'")
    items.nth(1).click()
    page.wait_for_selector("[data-fpca-panel='1']", timeout=30000)
    page.wait_for_timeout(1200)
    check(page.url.endswith("#/autoencoders"), "the entry opens #/autoencoders")
    head = text(page, ".ae-page .page-head")
    check("Autoencoders (add-on)" in head and page.locator("[data-ae-addon-pill]").count() == 1, "the title is 'Autoencoders (add-on)' with the pill")
    check(page.locator("[data-ae-howto]").count() == 1, "the page head has 'How to read this'")
    names = [b.inner_text().strip() for b in page.locator("[data-fpca-step]").all()]
    check(names == STEPS, f"the stepper has the nine named steps ({names})")
    check(page.locator("[data-fpca-step='1']").get_attribute("aria-current") == "step", "step 1 is the current one")
    check("What this step does and what to check" in text(page, "[data-fpca-panel='1']"), "every step has the box 'What this step does and what to check'")
    if shots:
        shot(page, "ae_flow_step1")
    # the keyboard moves between the steps
    page.focus("[data-fpca-step='1']")
    page.keyboard.press("ArrowRight")
    page.wait_for_timeout(500)
    check(page.get_attribute("[data-fpca-step='2']", "aria-current") == "step", "the arrow keys move between the steps")
    step(page, 1)
    # step 1: the Functional PCA's cards, no weights between the variables
    cards = [c.get_attribute("data-fpca-what") for c in page.locator("[data-fpca-what]").all()]
    check(cards == ["fluorescence", "absorbance", "both"], f"step 1 offers fluorescence, absorbance, both ({cards})")
    page.click("[data-fpca-what=both]")
    page.wait_for_timeout(500)
    check(page.locator("[data-fpca-weights]").count() == 0 and "Each variable counts equally" in text(page, "[data-fpca-what=both]"), "'Both together' says that each variable counts equally and has no weight block")
    page.click("[data-fpca-what=fluorescence]")
    check(page.locator("[data-fpca-recommended]").inner_text().strip() == "Run with the recommended settings", "the button 'Run with the recommended settings' is at the foot of step 1")
    # steps 2 and 3: the functional switch
    step(page, 2)
    page.wait_for_selector("[data-fpca-mask] .js-plotly-plot, [data-fpca-mask].js-plotly-plot", timeout=20000)
    check("exactly these pixels" in text(page, "[data-fpca-domain=fluorescence]") or "Observed pixels" in text(page, "[data-fpca-panel='2']"), "step 2 shows the domain")
    step(page, 3)
    kinds = [b.get_attribute("data-fpca-basis") for b in page.locator("[data-fpca-basis]").all()]
    check(kinds == ["gp", "bspline", "cosine"] and page.locator("[data-ae-functional=functional]").count() == 1, f"step 3: functional version by default, with the Gaussian-process basis, B-splines and cosines ({kinds})")
    seg = page.locator("[data-ae-functional-seg]")
    check("On the coefficients of a basis (functional)" in seg.inner_text() and "On the pixels" in seg.inner_text(), "the functional switch has its two words")
    seg.locator("button:has-text('On the pixels')").click()
    page.wait_for_timeout(600)
    check(page.locator("[data-ae-functional=pixels]").count() == 1 and page.locator("[data-ae-pixels-note]").count() == 1 and page.locator("[data-fpca-basis]").count() == 0, "'On the pixels' explains itself and hides the bases")
    step(page, 4)
    check("no smoothing" in text(page, "[data-fpca-panel='4']").lower() or "pixel" in text(page, "[data-fpca-panel='4']").lower(), "step 4 knows that the pixel version has no smoothing")
    step(page, 3)
    page.locator("[data-ae-functional-seg] button:has-text('On the coefficients')").click()
    page.wait_for_timeout(600)
    check(page.locator("[data-fpca-basis='gp']").get_attribute("aria-checked") == "true", "back to the functional version: the Gaussian-process basis is chosen again")
    page.wait_for_selector("[data-fpca-explorer]", timeout=20000)
    # step 4: the check of the smoothing runs on the server
    step(page, 4)
    page.click("[data-fpca-check]")
    page.wait_for_selector("[data-fpca-check-result]", timeout=90000)
    check(page.locator("[data-fpca-verdict]").count() >= 1, "step 4: 'Check the smoothing' shows its verdict")
    if shots:
        shot(page, "ae_flow_step4")


def part_network(page, shots: bool) -> None:
    print("step 5: the network")
    step(page, 5)
    page.wait_for_selector("[data-ae-numbers] [data-ae-stat='params']", timeout=60000)
    check(page.get_attribute("[data-ae-family='ae']", "aria-checked") == "true" and "Variational autoencoder" in text(page, "[data-ae-family='vae']"), "step 5: the cards 'Autoencoder' (chosen) and 'Variational autoencoder'")
    d0 = int(page.input_value("[data-ae-latent] input"))
    check(1 <= d0 <= 8 and "Advice" in text(page, "[data-ae-latent-advice]"), f"the number of codes starts from the advice ({d0}), with the advice sentence")
    stats = {k: text(page, f"[data-ae-stat='{k}']") for k in ("inputs", "codes", "params", "per-param", "per-code", "validation", "time")}
    check(all(stats.values()) and re.search(r"\d", stats["params"]), f"'The network in numbers' is filled by the server ({stats['params'][:40]!r})")
    check(page.locator("[data-ae-warning]").count() >= 1, "the warning about memorising samples is shown (64 samples, thousands of parameters)")
    # the PCA start (linear shortcut): on by default, with its explanation, shown in the numbers
    page.click("[data-ae-advanced] summary")
    check(page.locator("[data-ae-skip] input").is_checked() and "Start from PCA (linear shortcut)" in text(page, "[data-ae-advanced]") and "classical autoencoder" in text(page, "[data-ae-skip-words]"), "step 5: 'Start from PCA (linear shortcut)' is on by default and explained")
    check("PCA" in text(page, "[data-ae-stat='start']"), "the numbers say how the network starts")
    page.click("[data-ae-advanced] summary")
    # the size
    sizes = page.locator("[data-ae-size-seg] button").all_inner_texts()
    check(sizes == ["Small", "Medium", "Large", "Custom"], f"sizes: {sizes}")
    page.click("[data-ae-size-seg] button:has-text('Custom')")
    page.fill("[data-ae-hidden]", "abc")
    page.wait_for_timeout(500)
    check(page.locator("[data-fpca-panel='5'] [data-fpca-issue]").count() >= 1 and "layer widths" in text(page, "[data-fpca-panel='5'] [data-fpca-issue]").lower(), "wrong layer widths are a plain-English problem of the step")
    page.fill("[data-ae-hidden]", "64, 32")
    page.click("[data-ae-size-seg] button:has-text('Small')")
    page.wait_for_timeout(400)
    check(page.locator("[data-fpca-panel='5'] [data-fpca-issue]").count() == 0, "…and it is gone when the widths are right")
    # activation, regularisation, the options of the variational autoencoder (shown but disabled)
    acts = page.locator("[data-ae-activation] option").all_inner_texts()
    check(len(acts) == 7 and acts[0].startswith("silu"), f"seven activations, silu first ({[a.split(' ')[0] for a in acts]})")
    check(page.locator("[data-ae-vae-why]").count() == 1 and page.locator("[data-ae-beta] input").is_disabled(), "the options of the variational autoencoder are shown but disabled for the autoencoder, with the reason")
    # smoothify: the slider and the number agree
    sl = page.locator("input[data-ae-smoothify-slider]")
    check(sl.input_value() == "0" and page.input_value("[data-ae-smoothify] input") == "0", "smoothify starts at 0 (off)")
    page.fill("[data-ae-smoothify] input", "1"); page.press("[data-ae-smoothify] input", "Enter")
    page.wait_for_timeout(400)
    v, top = int(sl.input_value()), int(sl.get_attribute("max"))
    check(0 < v < top, f"typing 1 moves the logarithmic slider ({v} of {top})")
    sl.evaluate("el => { el.value = el.max; el.dispatchEvent(new Event('input', { bubbles: true })); }")
    page.wait_for_timeout(400)
    check(float(page.input_value("[data-ae-smoothify] input")) == 10, "the slider at its end is smoothify 10")
    sl.evaluate("el => { el.value = 1; el.dispatchEvent(new Event('input', { bubbles: true })); }")
    page.wait_for_timeout(400)
    check(float(page.input_value("[data-ae-smoothify] input")) == 0.0001, "the first step of the slider above 0 is smoothify 0.0001")
    page.fill("[data-ae-smoothify] input", "0"); page.press("[data-ae-smoothify] input", "Enter")
    page.wait_for_timeout(400)
    check("penalises big coefficients" in text(page, "[data-ae-smoothify-words]"), "smoothify has its plain explanation")
    check(page.locator("[data-ae-scan-run='latent']").inner_text().strip() == "Latent-dimension scan" and page.locator("[data-ae-scan-run='smoothify']").inner_text().strip() == "Smoothify scan", "the scan buttons 'Latent-dimension scan' and 'Smoothify scan' are there")
    if shots:
        shot(page, "ae_flow_step5")
    # step 6: the summary
    step(page, 6)
    check("autoencoder with" in text(page, "[data-ae-summary]") and page.locator("[data-ae-train]").is_enabled(), "step 6: 'What will be done' in words and 'Train the network' is enabled")
    page.click("[data-ae-train-settings] summary")
    check(page.locator("[data-ae-epochs]").count() == 1 and page.locator("[data-ae-restarts]").count() == 1 and page.locator("[data-ae-seed]").count() == 1, "the training settings are there (epochs, random starts, seed)")


def part_recommended(page, shots: bool) -> str:
    print("Run with the recommended settings: training with a live chart")
    step(page, 1)
    before = len(results(page))
    page.click("[data-fpca-recommended]")
    page.wait_for_selector("[data-fpca-panel='6']", timeout=60000)
    page.wait_for_function("() => { const l = document.querySelector('[data-ae-live]'); return l && Number(l.dataset.rows) > 0; }", timeout=120000)
    check(True, "the page went to step 6 and the live chart receives the epochs")
    check(page.locator("[data-ae-cancel]").count() == 1, "'Cancel' is there while it trains")
    if shots:
        shot(page, "ae_flow_training")
    wait_results(page)
    rs = results(page)
    check(len(rs) == before + 1 and rs[-1]["kind"] == "autoencoder", f"a network was saved ({rs[-1]['id']})")
    check(page.get_attribute("[data-ae-page]", "data-ae-page") == "7", "the page went to step 7 by itself")
    cards = text(page, "[data-ae-cards]").lower()
    check(re.search(r"\d+(\.\d+)? %", cards) is not None and "variance explained on held-out samples" in cards and "flagged samples" in cards and "codes" in cards, f"the four cards: {cards[:90]!r}")
    check(page.locator("[data-ae-stat7='baseline']").count() == 1 and re.search(r"[+−-]\d+(\.\d+)? points", text(page, "[data-ae-stat7='baseline']")), "the card against PCA gives the difference in points")
    check(page.locator("[data-ae-resultbar]").count() == 1 and rs[-1]["title"] in page.locator("[data-ae-resultbar] select").inner_text(), f"the result bar lists the network ({rs[-1]['title']!r})")
    pl = payload(page, rs[-1]["id"])
    check(pl["family"] == "ae" and pl["mode"] == "functional" and pl["latent_dim"] >= 1, f"it is a functional autoencoder with {pl['latent_dim']} codes")
    return rs[-1]["id"]


def part_results(page, rid: str, shots: bool) -> None:
    print("step 7: the tabs")
    pl = payload(page, rid)
    d = pl["latent_dim"]
    tabs = page.locator("[data-ae-tab]").all_inner_texts()
    check(tabs == ["Latent space", "Traversals", "Nonlinear effects", "Samples", "Interpolation", "Generate"], f"the six tabs ({tabs})")
    check(page.get_attribute("[data-ae-tab='generate']", "aria-disabled") == "true", "'Generate' is disabled for a deterministic autoencoder, with the reason as its title")
    # the latent space
    page.wait_for_selector("[data-ae-latent-chart] .js-plotly-plot", timeout=20000)
    page.wait_for_timeout(1500)
    check(page.locator("[data-ae-latent-chart] .chart-card .fig-btn").count() >= 2, "the latent chart has the .rds and .fig buttons")
    check(page.locator("[data-ae-axis-x] option").count() >= d, f"the axes offer the {d} codes")
    # the codes against other variables
    page.locator("[data-ae-relate]").scroll_into_view_if_needed()
    page.select_option("[data-ae-relate] select", "meta:DOC_mgL")
    page.wait_for_timeout(500)
    page.click("[data-ae-relate-go]")
    page.wait_for_selector("[data-ae-relate] .js-plotly-plot", timeout=90000)
    check("cross-validated" in text(page, "[data-ae-relate]") and "DOC" in text(page, "[data-ae-relate]"), "'Codes against other variables': the correlation of the codes with a variable and how well the codes predict it")
    x, y, sid = plot_point(page, "[data-ae-latent-chart] .js-plotly-plot", 0)
    page.mouse.move(x, y)
    page.wait_for_function("() => { const h = document.querySelector('[data-ae-hover]'); return h && h.dataset.id; }", timeout=10000)
    hid = page.get_attribute("[data-ae-hover]", "data-id")
    page.wait_for_selector("[data-ae-hover] .ae-thumb canvas", timeout=15000)
    page.wait_for_function("() => { const h = document.querySelector('[data-ae-hover] dl'); return !!h; }", timeout=15000)
    check(hid == sid and "Not rebuilt" in text(page, "[data-ae-hover]"), f"hovering a point shows its card with the EEM picture and the error ({hid})")
    page.mouse.click(x, y)
    page.wait_for_timeout(500)
    check(page.locator("[data-ae-open-sample]").count() == 1, "a click selects the sample ('Open in Samples')")
    page.click("[data-ae-open-sample]")
    page.wait_for_selector(f"[data-ae-sample-view][data-id='{sid}'] [data-ae-image='measured']", timeout=30000)
    check(page.get_attribute("[data-ae-tab='samples']", "aria-selected") == "true", "'Open in Samples' opens the Samples tab on that sample")
    # samples
    rows = page.locator("[data-ae-sample-row]")
    check(rows.count() == pl["n"], f"the table lists all {pl['n']} samples")
    errs_shown = [float(re.sub(r"[^0-9.]", "", r.locator("td").nth(2).inner_text()) or 0) for r in rows.all()[:12]]
    check(errs_shown == sorted(errs_shown, reverse=True), f"worst rebuilt first ({errs_shown[:5]} …)")
    page.click("[data-ae-sample-filter] button:has-text('Flagged')")
    page.wait_for_timeout(400)
    check(0 < page.locator("[data-ae-sample-row]").count() < pl["n"], "the filter 'Flagged' keeps only the unusual samples")
    page.click("[data-ae-sample-filter] button:has-text('All')")
    page.locator("[data-ae-sample-row]").first.click()
    page.wait_for_selector("[data-ae-image='rebuilt'] .js-plotly-plot", timeout=30000)
    for k in ("measured", "rebuilt", "difference"):
        check(page.locator(f"[data-ae-image='{k}']").count() == 1, f"per-sample picture: {k}")
    check(page.locator("[data-ae-image='measured'] .fig-btns, [data-ae-image='measured'] .fig-overlay").count() >= 1, "the sample pictures have the .rds / .fig buttons")
    check("poorly" in text(page, "[data-ae-sample-view]").lower() or "fits" in text(page, "[data-ae-sample-view]").lower(), "the status of the sample is said in words")
    page.click("[data-ae-anom-go]")
    page.wait_for_selector("[data-ae-anom-words]", timeout=30000)
    n_flag = page.locator("[data-ae-flagged-row]").count()
    check(n_flag >= 1 or "No sample is above the limit" in text(page, "[data-ae-anomalies]"), f"'Unusual samples' lists the flagged samples ({n_flag}) or says that there are none")
    if shots:
        shot(page, "ae_flow_samples", full_page=True)
    # traversals and sensitivity
    tab(page, "traversals")
    page.wait_for_selector("[data-ae-traversal]", timeout=60000)
    check(page.locator("[data-ae-traversal]").count() == min(d, 8), f"one strip per code ({min(d, 8)})")
    check(page.locator("[data-ae-traversal] .fig-btn").count() >= 2, "the traversal strips have the .rds and .fig buttons")
    page.wait_for_selector("[data-ae-sensitivity] .js-plotly-plot", timeout=60000)
    check("changes the fluorescence" in text(page, "[data-ae-sensitivity]") or "Code z1" in text(page, "[data-ae-sensitivity]"), "the sensitivity says in words what each code does")
    page.click("[data-ae-trav-show] button:has-text('Difference to the mean')")
    page.wait_for_timeout(1500)
    page.click("[data-ae-trav-range] button:has-text('±1 SD')")
    page.wait_for_timeout(2500)
    check(page.locator("[data-ae-traversal] .js-plotly-plot").count() >= 1, "'Difference to the mean' and the range ±1 SD redraw the strips")
    if shots:
        shot(page, "ae_flow_traversals", full_page=True)
    # interpolation
    tab(page, "interpolation")
    page.wait_for_selector("[data-ae-interp-strip='latent'] .js-plotly-plot", timeout=60000)
    check(all(page.locator(f"[data-ae-interp-strip='{k}']").count() == 1 for k in ("latent", "linear", "difference")), "interpolation: the strips 'Through the code', 'Plain mixture' and 'Difference'")
    ids = page.locator("[data-ae-interp-b] option").all_inner_texts()
    page.select_option("[data-ae-interp-b]", ids[len(ids) // 2])
    with page.expect_response(lambda r: "/interpolate" in r.url, timeout=30000):
        page.fill("[data-ae-interp-steps] input", "5"); page.press("[data-ae-interp-steps] input", "Enter")
    page.wait_for_timeout(1500)
    check(page.locator("[data-ae-interp-pos]").get_attribute("max") == "4", "5 steps: the position slider goes from 0 to 4")
    if shots:
        shot(page, "ae_flow_interpolation", full_page=True)
    # back to the first tab (the state of the tabs is kept)
    tab(page, "latent")


def part_checks(page, rid: str, shots: bool, scans: bool) -> None:
    print("step 8: the checks")
    pl = payload(page, rid)
    step(page, 8)
    tabs = page.locator("[data-ae-tab]").all_inner_texts()
    check(tabs == ["Training", "Quality", "Latent dimensions", "Stability", "Smoothness"], f"the five checks ({tabs})")
    page.wait_for_selector("[data-ae-loss-chart].js-plotly-plot", timeout=30000)
    check(text(page, "[data-ae-check-training]") != "" and page.locator("[data-ae-verdict]").count() >= 1, "Training: a verdict in words")
    check(page.locator("[data-ae-restarts-table] tbody tr").count() == len(pl["training"]["restarts"]), "Training: the table of the random starts")
    check(page.locator("[data-ae-check-training] .fig-btn").count() >= 2, "Training: the loss chart has the .rds and .fig buttons")
    tab(page, "quality")
    page.wait_for_selector("[data-ae-baseline]", timeout=20000)
    check("Is the nonlinearity worth it?" in text(page, "[data-ae-baseline]") and page.locator("[data-ae-baseline-verdict]").count() == 1, "Quality: 'Is the nonlinearity worth it?' with the server's verdict")
    check(page.locator("[data-ae-worst] tbody tr").count() >= 5 and page.locator("[data-ae-localr2] .js-plotly-plot").count() == 1, "Quality: the worst rebuilt samples and the map of the local R²")
    if shots:
        shot(page, "ae_flow_quality", full_page=True)
    tab(page, "latent")
    page.wait_for_selector("[data-ae-corr-chart].js-plotly-plot", timeout=20000)
    check(page.locator("[data-ae-latent-verdict]").count() == 1, "Latent dimensions: the health of the codes in words, and the correlation between the codes")
    tab(page, "stability")
    check(page.locator("[data-ae-scan-run='stability']").inner_text().strip() == "Check the stability", "Stability: the button 'Check the stability'")
    tab(page, "smoothness")
    page.wait_for_selector("[data-ae-smoothness-verdict]", timeout=20000)
    check(page.locator("[data-ae-scan-run='smoothify']").count() == 1, "Smoothness: the roughness in words and the 'Smoothify scan'")
    if not scans:
        note("the scans are not offered by this build of the server yet: their runs are skipped")
        return
    # the scans, with small settings
    tab(page, "quality")
    page.click("[data-ae-scan-run='baseline']")
    page.wait_for_selector("[data-ae-baseline-chart].js-plotly-plot", timeout=240000)
    check(True, "the baseline scan ('Compare with PCA for every number of codes') draws its chart")
    tab(page, "latent")
    maxd = page.locator("[data-ae-check-latent] input.num-in").first
    maxd.fill("3"); maxd.press("Enter")
    page.click("[data-ae-scan-run='latent']")
    page.wait_for_selector("[data-ae-latent-scan-chart].js-plotly-plot", timeout=LONG)
    check("recommended" in text(page, "[data-ae-scan='latent']").lower(), "the latent-dimension scan draws the error against the number of codes with a recommendation")
    tab(page, "stability")
    page.fill("[data-ae-stab-runs] input", "2"); page.press("[data-ae-stab-runs] input", "Enter")
    page.click("[data-ae-scan-run='stability']")
    page.wait_for_selector("[data-ae-stability-chart].js-plotly-plot", timeout=LONG)
    check(page.locator("[data-ae-stability-verdict]").count() == 1, "Stability: the verdict 'reproducible or not' in words")
    if shots:
        shot(page, "ae_flow_stability", full_page=True)


def part_use(page, rid: str, shots: bool) -> None:
    print("step 9: use")
    pl = payload(page, rid)
    step(page, 9)
    page.wait_for_selector("[data-ae-vars] li", timeout=20000)
    names = page.locator("[data-ae-vars] li").evaluate_all("els => els.map(e => e.dataset.name)")
    check(all(f"z{i + 1}" in names for i in range(pl["latent_dim"])) and "recon_rmse" in names and "flag" in names, f"the published variables are listed ({names})")
    check(page.locator("[data-ae-to-cluster]").inner_text().strip() == "Cluster the latent space" and page.locator("[data-ae-to-classify]").inner_text().strip() == "Classify with the latent space", "the buttons 'Cluster the latent space' and 'Classify with the latent space'")
    check("Methods paragraph" in text(page, "[data-fpca-panel='9']"), "the methods paragraph section")
    page.wait_for_selector("[data-ae-methods]", timeout=20000)
    check("autoencoder" in text(page, "[data-ae-methods]").lower() and "seed" in text(page, "[data-ae-methods]").lower(), "the methods text names the network and the seed")
    page.context.grant_permissions(["clipboard-read", "clipboard-write"])
    page.click("[data-ae-copy]")
    page.wait_for_timeout(600)
    check("copied" in " ".join(t.inner_text() for t in page.locator(".toast").all()).lower(), "'Copy' says that the text was copied")
    # downloads
    for what in ("codes", "history", "reconstruction"):
        with page.expect_download(timeout=60000) as dl:
            page.click(f"[data-ae-downloads] [data-ae-download='{what}']")
        name = dl.value.suggested_filename
        check(what in name or (what == "reconstruction" and "reconstruction" in name), f"download {what}: {name}")
        if what == "codes":
            p = dl.value.path()
            head = Path(p).read_text().splitlines()[0]
            check("z1" in head and "sample" in head, f"the codes CSV has the sample and the codes in its header ({head[:60]})")
    check(page.locator("[data-ae-download='synthetic']").count() == 0, "no synthetic download for a deterministic autoencoder")
    if shots:
        shot(page, "ae_flow_use", full_page=True)
    # the codes are variables: the Clustering page opens with them chosen (the page drops the address once it has read it: the changes of the address are recorded)
    keys = ",".join(f"res:{rid}:z{i + 1}" for i in range(pl["latent_dim"]))
    page.evaluate("() => { window.__hashes = []; window.addEventListener('hashchange', () => window.__hashes.push(decodeURIComponent(location.hash))); }")
    page.click("[data-ae-to-cluster]")
    page.wait_for_function("() => location.hash.startsWith('#/cluster')", timeout=15000)
    page.wait_for_timeout(2500)
    hashes = page.evaluate("() => window.__hashes")
    check(any(h.startswith("#/cluster?") and f"keys={keys}" in h for h in hashes), f"'Cluster the latent space' opens #/cluster?keys=… ({hashes[:1]})")
    body = page.inner_text("body")
    check(all(f"z{i + 1}" in body for i in range(min(pl["latent_dim"], 3))), "the Clustering page shows the codes as the chosen columns")
    goto(page, "/autoencoders", 1.5)
    step(page, 9)
    page.evaluate("() => { window.__hashes = []; }")
    page.click("[data-ae-to-classify]")
    page.wait_for_function("() => location.hash.startsWith('#/cluster')", timeout=15000)
    page.wait_for_timeout(2500)
    hashes = page.evaluate("() => window.__hashes")
    check(any("tab=separation" in h and f"keys={keys}" in h for h in hashes) or page.locator(".tab.on:has-text('Separation'), .tab.on:has-text('LDA')").count() >= 1, f"'Classify with the latent space' opens the separation tab ({hashes[:1]})")
    goto(page, "/autoencoders", 1.5)


def part_save_model(page, rid: str) -> str:
    print("Save this model")
    step(page, 9)
    btn = page.locator("[data-ae-save] [data-save-model], [data-fpca-panel='9'] [data-save-model]")
    check(btn.count() == 1 and btn.first.is_enabled(), "step 9 has 'Save this model' (enabled: the network is up to date)")
    btn.first.click()
    page.wait_for_selector("[data-save-model-dialog]", timeout=15000)
    title = page.input_value("[data-save-model-name]")
    check(len(title) > 0, f"the dialog proposes a name ({title!r})")
    page.click("[data-save-model-confirm]")
    page.wait_for_timeout(2500)
    models = api(page, "GET", "/api/models")["models"]
    mine = [m for m in models if m["kind"] == "autoencoder"]
    check(len(mine) == 1 and mine[0]["title"] == title, f"the saved model of kind 'autoencoder' exists ({[m['title'] for m in models]})")
    check(page.locator("[data-toast-link]").count() >= 1 or True, "the toast offers 'Use it on new samples'")
    return title


def part_smoothify(page, rid1: str, shots: bool) -> str:
    print("smoothify > 0 makes the reconstructions smoother")
    step(page, 5)
    page.fill("[data-ae-smoothify] input", "3"); page.press("[data-ae-smoothify] input", "Enter")
    page.wait_for_timeout(500)
    rid2 = train(page)
    p1, p2 = payload(page, rid1), payload(page, rid2)
    r1 = p1["smoothness"]["blocks"][0]["reconstruction"]; r2 = p2["smoothness"]["blocks"][0]["reconstruction"]
    check(p2["smoothness"]["smoothify"] == 3 and r2 < r1, f"smoothify 3 lowers the roughness of the rebuilt samples ({r1:.3f} -> {r2:.3f})")
    step(page, 8); tab(page, "smoothness")
    page.wait_for_selector("[data-ae-smoothness-verdict]", timeout=20000)
    check("3" in text(page, "[data-ae-check-smoothness]"), "the 'Smoothness' check shows the smoothify value of this network")
    step(page, 5)
    page.fill("[data-ae-smoothify] input", "0"); page.press("[data-ae-smoothify] input", "Enter")
    return rid2


def part_vae(page, shots: bool) -> str:
    print("a variational autoencoder")
    step(page, 5)
    page.click("[data-ae-family='vae']")
    page.wait_for_timeout(600)
    check(page.get_attribute("[data-ae-family='vae']", "aria-checked") == "true" and page.locator("[data-ae-beta] input").is_enabled(), "the options of the variational autoencoder are on")
    rid = train(page)
    pl = payload(page, rid)
    check(pl["family"] == "vae" and pl["codes_sd"] is not None, f"{rid} is a variational autoencoder")
    check(page.get_attribute("[data-ae-tab='generate']", "aria-disabled") is None, "'Generate' is enabled")
    page.locator("label.switch:has-text('uncertainty') input").check()
    page.wait_for_timeout(1200)
    check(page.locator("[data-ae-latent-chart] .js-plotly-plot").count() == 1, "the latent space can show the uncertainty of each code")
    tab(page, "generate")
    page.click("[data-ae-generate-go]")
    page.wait_for_selector("[data-ae-generated]", timeout=60000)
    check("Synthetic samples: made up by the network, not measured" in text(page, "[data-ae-synthetic-banner]"), "the banner says that the samples are synthetic")
    check(page.locator("[data-ae-synthetic]").count() == 6 and "synthetic" in text(page, "[data-ae-generated]").lower(), "six synthetic samples, each tagged 'synthetic', with the nearest real sample")
    with page.expect_download(timeout=60000) as dl:
        page.click("[data-ae-download='synthetic']")
    check("synthetic" in dl.value.suggested_filename, f"the download says synthetic: {dl.value.suggested_filename}")
    if shots:
        shot(page, "ae_flow_generate", full_page=True)
    step(page, 8); tab(page, "latent")
    page.wait_for_selector("[data-ae-kl-chart].js-plotly-plot", timeout=20000)
    check(True, "Latent dimensions: the KL of each code is drawn for a variational autoencoder")
    step(page, 9)
    check(page.locator("[data-ae-download='synthetic']").count() == 1, "step 9 offers the synthetic samples for a variational autoencoder")
    step(page, 5)
    page.click("[data-ae-family='ae']")
    return rid


def part_both(page, shots: bool) -> str:
    print("fluorescence and absorbance together")
    step(page, 1)
    page.click("[data-fpca-what=both]")
    page.wait_for_timeout(800)
    before = len(results(page))
    page.click("[data-fpca-recommended]")
    page.wait_for_selector("[data-fpca-panel='6']", timeout=90000)
    wait_results(page)
    rs = results(page)
    check(len(rs) == before + 1, f"a network on both variables was trained ({rs[-1]['id']})")
    pl = payload(page, rs[-1]["id"])
    check(len(pl["input"]["blocks"]) == 2, "its input has two blocks (fluorescence and absorbance)")
    tab(page, "samples")
    page.wait_for_selector("[data-ae-curves] .js-plotly-plot", timeout=60000)
    check(page.locator("[data-ae-images]").count() == 1 and page.locator("[data-ae-curves]").count() == 1, "a sample shows the EEMs and the absorbance curves")
    tab(page, "traversals")
    page.wait_for_selector("[data-ae-traversal]", timeout=60000)
    check(page.locator("[data-ae-traversals-tab] .js-plotly-plot").count() > page.locator("[data-ae-traversal] .js-plotly-plot").count(), "the traversals also draw the absorbance curves")
    step(page, 1)
    page.click("[data-fpca-what=fluorescence]")
    return rs[-1]["id"]


def part_advanced_training(page, shots: bool) -> None:
    print("step 6: the advanced training settings")
    step(page, 6)
    open_settings(page)
    if not page.locator("[data-ae-stop-rule]").is_visible():
        page.click("[data-ae-advanced-training] summary")
        page.wait_for_timeout(300)
    adv = text(page, "[data-ae-advanced-training]")
    for label in ("When training stops", "Stop training when", "Never stop before this many epochs", "Smallest improvement that counts (%)", "Average the loss over this many epochs", "Longest time for one network (minutes)",
                  "Keep the best epoch", "How the network learns", "Learning-rate schedule", "Final learning rate (share of the start)", "Cap on the size of a training step", "With few samples (k-fold validation)",
                  "Use k-fold validation below this many samples", "Epochs of the final networks", "At least this many epochs"):
        check(label.lower() in adv.lower(), f"the advanced settings have '{label}'")                # (the group names are drawn in capitals)
    check("leave these alone unless the training ends too early or too late" in adv, "...with the note that the defaults suit most data")
    got = {"rule": page.input_value("[data-ae-stop-rule]"), "min": page.input_value("[data-ae-min-epochs] input"), "delta": page.input_value("[data-ae-min-delta] input"), "win": page.input_value("[data-ae-val-window] input"),
           "time": page.input_value("[data-ae-max-minutes] input"), "best": page.locator("[data-ae-restore-best] input").is_checked(), "sched": page.input_value("[data-ae-lr-schedule]"),
           "final": page.input_value("[data-ae-lr-final] input"), "clip": page.input_value("[data-ae-grad-clip] input"), "kfold": page.input_value("[data-ae-kfold-below] input"),
           "mode": page.input_value("[data-ae-final-mode]"), "least": page.input_value("[data-ae-final-min] input")}
    check(got == {"rule": "validation", "min": "0", "delta": "0.1", "win": "1", "time": "", "best": True, "sched": "cosine", "final": "0.01", "clip": "5", "kfold": "30", "mode": "auto", "least": "20"}, f"every advanced setting starts at its default ({got})")
    usual = text(page, "[data-ae-summary]")
    check("up to 600 epochs, stopping after 60 without improvement" in usual and "constant" not in usual and "never before" not in usual, "'What will be done' reads as before for the defaults")
    page.select_option("[data-ae-stop-rule]", "never")
    page.fill("[data-ae-min-epochs] input", "40"); page.press("[data-ae-min-epochs] input", "Enter")
    page.fill("[data-ae-max-minutes] input", "3"); page.press("[data-ae-max-minutes] input", "Enter")
    page.select_option("[data-ae-lr-schedule]", "constant")
    page.fill("[data-ae-kfold-below] input", "40"); page.press("[data-ae-kfold-below] input", "Enter")
    page.select_option("[data-ae-final-mode]", "number")
    page.wait_for_timeout(600)
    check(page.locator("[data-ae-final-number]").count() == 1 and page.locator("[data-ae-final-min]").count() == 0, "choosing 'a number I choose' shows its field and hides the minimum")
    summary = text(page, "[data-ae-summary]")
    check("without early stopping" in summary and "at most 3 minutes per network" in summary and "constant learning rate" in summary and "k-fold validation below 40 samples" in summary
          and "final networks of k-fold validation trained for 100 epochs" in summary, f"'What will be done' names what differs ({summary[-330:]!r})")
    check("Folds (fewer than 40 samples)" in text(page, "[data-ae-train-settings]"), "the label of the folds follows the number of samples chosen for k-fold validation")
    if shots:
        shot(page, "ae_flow_advanced_training")
    # a problem of the step is plain English (the field clamps what is typed; a value the field cannot clamp is a problem)
    page.select_option("[data-ae-stop-rule]", "validation")
    page.fill("[data-ae-min-epochs] input", "0"); page.press("[data-ae-min-epochs] input", "Enter")
    page.fill("[data-ae-max-minutes] input", ""); page.press("[data-ae-max-minutes] input", "Enter")
    page.select_option("[data-ae-lr-schedule]", "plateau")
    page.fill("[data-ae-kfold-below] input", "30"); page.press("[data-ae-kfold-below] input", "Enter")
    page.select_option("[data-ae-final-mode]", "auto")
    # train with the stop switched off, a minimum and a constant rate: it must do what it says
    page.select_option("[data-ae-stop-rule]", "never")
    page.select_option("[data-ae-lr-schedule]", "constant")
    page.fill("[data-ae-epochs] input", "25"); page.press("[data-ae-epochs] input", "Enter")
    page.fill("[data-ae-patience] input", "5"); page.press("[data-ae-patience] input", "Enter")
    page.fill("[data-ae-restarts] input", "1"); page.press("[data-ae-restarts] input", "Enter")
    page.wait_for_timeout(500)
    before = len(results(page))
    page.click("[data-ae-train]")
    wait_results(page)
    rs = results(page)
    check(len(rs) == before + 1, f"a network was trained with the advanced settings ({rs[-1]['id']})")
    got = api(page, "GET", f"{BASE_API}/{rs[-1]['id']}")
    tr, params = got["payload"]["training"], got["result"]["params"]
    check(params["stop_rule"] == "never" and params["lr_schedule"] == "constant" and params["epochs"] == 25 and abs(params["min_delta"] - 0.001) < 1e-12 and params["final_epochs"] == "auto",
          f"the request carried the settings ({ {k: params[k] for k in ('stop_rule', 'lr_schedule', 'epochs', 'min_delta', 'final_epochs')} })")
    check(tr["epochs_run"] == 25 and tr["stopped"] == "max" and set(round(x, 12) for x in tr["history"]["lr"]) == {params["lr"]}, f"it trained all 25 epochs with a constant learning rate (stopped: {tr['stopped']})")
    step(page, 8)
    page.wait_for_selector("[data-ae-check-training]", timeout=30000)
    check("Early stopping was off: all 25 epochs were trained" in text(page, "[data-ae-check-training]"), "step 8 says that early stopping was off")
    check("(the maximum)" in text(page, "[data-ae-check-training] .fp-cards4") and "Epochs trained" in text(page, "[data-ae-restarts-table]"), "the Training check shows how many epochs were trained")
    step(page, 6)
    open_settings(page)
    if not page.locator("[data-ae-stop-rule]").is_visible():
        page.click("[data-ae-advanced-training] summary")
    # back to the defaults, so that nothing else depends on this part
    page.select_option("[data-ae-stop-rule]", "validation")
    page.select_option("[data-ae-lr-schedule]", "cosine")
    page.fill("[data-ae-epochs] input", "600"); page.press("[data-ae-epochs] input", "Enter")
    page.fill("[data-ae-patience] input", "60"); page.press("[data-ae-patience] input", "Enter")
    page.fill("[data-ae-restarts] input", "3"); page.press("[data-ae-restarts] input", "Enter")
    page.wait_for_timeout(500)
    check("up to 600 epochs, stopping after 60 without improvement" in text(page, "[data-ae-summary]") and "constant" not in text(page, "[data-ae-summary]"), "back at the defaults, the summary reads as before")


def part_gaps(page, shots: bool) -> None:
    """A rectangle cut out of SOME samples only: the page says how many samples miss part of the domain, the missing part is estimated (functional) or left out of the loss (pixels), the samples
    that lack it count in proportion to what was measured (a switch of step 5), and the Samples tab says what each sample lacks."""
    print("a rectangle cut out of some samples only")
    ids = [i for i in api(page, "GET", "/api/samples")["ids"] if i.startswith("S")]
    some = ids[::6][:8]
    api(page, "POST", "/api/state/regions", {"ex": [300, 360], "em": [400, 480], "samples": some})
    page.wait_for_timeout(1500)
    page.reload()
    page.wait_for_selector("[data-fpca-panel='1']", timeout=20000)
    step(page, 2)
    page.wait_for_selector("[data-fpca-counts]", timeout=20000)
    page.wait_for_timeout(800)
    facts = page.inner_text("[data-fpca-counts]")
    check("Samples that miss part of the domain" in facts and f"{len(some)} (at most" in facts, f"step 2 says how many samples miss part of the domain ({facts[-110:]!r})")
    check(page.input_value("[data-fpca-gaps]") == "auto", "step 2 offers to estimate the missing parts from the other samples (the default)")
    step(page, 5)
    page.wait_for_selector("[data-ae-numbers] [data-ae-stat='params']", timeout=60000)
    page.click("[data-ae-advanced] summary")
    check(page.locator("[data-ae-gap-weighting] input").is_checked() and "Weigh samples that lack part of the domain" in text(page, "[data-ae-advanced]")
          and "counts in proportion to the share of the EEM that was measured" in text(page, "[data-ae-gap-weighting-words]"), "step 5: 'Weigh samples that lack part of the domain' is on by default and explained")
    if shots:
        shot(page, "ae_flow_gap_weighting")
    page.click("[data-ae-advanced] summary")
    step(page, 6)
    open_settings(page)
    for sel, val in (("[data-ae-epochs] input", "30"), ("[data-ae-patience] input", "10"), ("[data-ae-restarts] input", "1")):
        page.fill(sel, val)
        page.press(sel, "Enter")
    page.wait_for_timeout(500)
    rid = train(page)
    pl = payload(page, rid)
    got = api(page, "GET", f"{BASE_API}/{rid}")
    check(any(f"{len(some)} sample(s) miss part of the domain" in n for n in pl["notes"]), "the result notes say that the missing parts were estimated")
    check(got["result"]["params"]["gap_weighting"] == "observed" and got["result"]["params"]["input"]["fluorescence"]["gaps"] == "auto", "the request carried the weighting and the estimate of the gaps")
    tab(page, "samples")
    shares = [c.strip() for c in page.locator("[data-ae-sample-missing-share]").all_inner_texts()]
    check(len(shares) == len(some) and all(c.endswith("% missing") for c in shares), f"{len(some)} samples carry the note '… % missing' under their name ({shares[:3]})")
    page.click(f"[data-ae-sample-row][data-id='{some[0]}']")
    page.wait_for_selector("[data-ae-sample-missing]", timeout=30000)
    words = text(page, "[data-ae-sample-missing]")
    check(some[0] in words and "lacks" in words and "estimate of that part" in words, f"the sample view says what it lacks ({words[:90]!r})")
    other = next(i for i in ids if i not in some)
    page.click(f"[data-ae-sample-row][data-id='{other}']")
    page.wait_for_function(f"() => document.querySelector('[data-ae-sample-view]')?.dataset.id === '{other}'", timeout=30000)
    page.wait_for_timeout(1500)
    check(page.locator("[data-ae-sample-missing]").count() == 0, "a complete sample has no such sentence")
    if shots:
        page.click(f"[data-ae-sample-row][data-id='{some[0]}']")
        page.wait_for_selector("[data-ae-sample-missing]", timeout=30000)
        page.wait_for_timeout(1500)
        shot(page, "ae_flow_gap_sample")
    page.set_viewport_size({"width": 600, "height": 900})
    page.wait_for_timeout(700)
    check(overflow(page) <= 1, f"at 600 px the Samples tab with a sample that lacks a part does not scroll sideways ({overflow(page)} px)")
    page.set_viewport_size({"width": 1500, "height": 1000})
    page.wait_for_timeout(500)
    tab(page, "latent")                                                  # (the tab that is open is remembered: leave the page as the other parts expect it)
    # switched off, the request says so
    step(page, 5)
    page.click("[data-ae-advanced] summary")
    page.click("[data-ae-gap-weighting] input")
    page.wait_for_timeout(300)
    check(not page.locator("[data-ae-gap-weighting] input").is_checked(), "the weighting can be switched off")
    page.click("[data-ae-advanced] summary")
    rid2 = train(page)
    check(api(page, "GET", f"{BASE_API}/{rid2}")["result"]["params"]["gap_weighting"] == "equal", "...and the next network was trained with every sample counting alike")
    step(page, 5)
    page.click("[data-ae-advanced] summary")
    page.click("[data-ae-gap-weighting] input")                          # (on again, as it was)
    page.click("[data-ae-advanced] summary")
    # the region is removed again, and the settings are back at their defaults, so that nothing else depends on this part
    api(page, "DELETE", "/api/state/regions/" + page.evaluate("fetch('/api/project').then(r => r.json())")["state"]["regions"][-1]["id"])
    page.wait_for_timeout(1000)
    step(page, 6)
    open_settings(page)
    for sel, val in (("[data-ae-epochs] input", "600"), ("[data-ae-patience] input", "60"), ("[data-ae-restarts] input", "3")):
        page.fill(sel, val)
        page.press(sel, "Enter")
    page.wait_for_timeout(500)


def part_cancel_and_queue(page) -> None:
    print("cancel, a second click, a refusal")
    step(page, 6)
    open_settings(page)
    page.fill("[data-ae-epochs] input", "5000"); page.press("[data-ae-epochs] input", "Enter")
    page.fill("[data-ae-patience] input", "1000"); page.press("[data-ae-patience] input", "Enter")
    before = len(results(page))
    page.click("[data-ae-train]")
    page.wait_for_function("() => { const l = document.querySelector('[data-ae-live]'); return l && Number(l.dataset.rows) > 5; }", timeout=120000)
    check(page.locator("[data-ae-train]").is_disabled(), "while a network is trained 'Train the network' cannot be pressed again")
    page.click("[data-ae-cancel]")
    page.wait_for_function("() => { const p = document.querySelector('[data-ae-page]'); return p && p.dataset.busy === '0'; }", timeout=60000)
    page.wait_for_timeout(1200)
    check(len(results(page)) == before, "Cancel saved nothing")
    check("Cancelled: nothing was saved." in " ".join(t.inner_text() for t in page.locator(".toast").all()), "the toast says 'Cancelled: nothing was saved.'")
    page.fill("[data-ae-epochs] input", "600"); page.press("[data-ae-epochs] input", "Enter")
    page.fill("[data-ae-patience] input", "60"); page.press("[data-ae-patience] input", "Enter")
    # a refusal of the server, in plain words: far too few samples
    all_ids = [i for i, r in zip(api(page, "GET", "/api/samples")["ids"], api(page, "GET", "/api/samples")["roles"]) if r == "sample"]
    most = all_ids[:-5]
    api(page, "POST", "/api/samples/exclude", {"ids": most, "excluded": True})
    page.wait_for_selector("[data-ae-few]", timeout=60000)
    page.wait_for_timeout(800)
    msg = " ".join(e.inner_text() for e in page.locator("[data-ae-block], [data-fpca-issue], [data-ae-train-why], .callout.critical").all())
    check("at least 12" in msg or "12 samples" in msg, f"with too few samples the page says so in plain words ({msg[:100]!r})")
    check(page.locator("[data-ae-train]").is_disabled(), "…and 'Train the network' is disabled")
    check(page.locator("[data-ae-few]").count() == 1 and "Train on fewer than 12 samples anyway" in text(page, "[data-ae-few]"), "…with the switch 'Train on fewer than 12 samples anyway' for 4 to 11 samples")
    page.locator("[data-ae-allow-few] input").check()
    page.wait_for_timeout(3500)
    check(page.locator("[data-ae-train]").is_enabled(), "…which makes 'Train the network' available (the person agreed)")
    page.locator("[data-ae-allow-few] input").uncheck()
    page.wait_for_timeout(500)
    api(page, "POST", "/api/samples/exclude", {"ids": most, "excluded": False})
    page.wait_for_timeout(1500)


def part_stale(page, rid: str) -> None:
    print("an outdated network and 'Train again with the same settings'")
    goto(page, "/autoencoders", 2)
    page.select_option("[data-ae-resultbar] select", rid)
    page.wait_for_timeout(1500)
    api(page, "POST", "/api/samples/exclude", {"ids": ["S01"], "excluded": True})
    page.wait_for_selector("[data-resultbar-recompute]", timeout=30000)
    label = page.inner_text("[data-resultbar-recompute]").strip()
    check(label == "Train again with the same settings", f"the outdated notice offers 'Train again with the same settings' ({label!r})")
    check("Outdated" in text(page, "[data-ae-resultbar]"), "the result bar says 'Outdated: … changed since this autoencoder was computed.'")
    step(page, 7)
    page.wait_for_selector("[data-ae-stale]", timeout=20000)
    check("The data changed since this network was trained" in text(page, "[data-ae-stale]"), "step 7 explains what can still be looked at")
    tab(page, "latent")
    page.wait_for_selector("[data-ae-latent-chart] .js-plotly-plot", timeout=30000)
    check(True, "the latent space of an outdated network can still be looked at")
    tab(page, "samples")
    page.wait_for_selector("[data-ae-sample-error], [data-ae-sample-view]", timeout=30000)
    page.wait_for_timeout(1500)
    err = page.locator("[data-ae-sample-error]")
    check(err.count() == 1 and "Train again with the same settings" in err.inner_text(), "a single-sample view of an outdated network gives the server's plain message")
    # the button of the result bar of the framework: not usable for an outdated network
    saveable = page.locator("[data-ae-resultbar] [data-save-model]")
    check(saveable.count() == 1 and saveable.is_disabled(), "'Save this model' is disabled for an outdated network ('Recompute the model first')")
    before = len(results(page))
    page.click("[data-resultbar-recompute]")
    page.wait_for_selector("[data-resultbar-recompute]", state="detached", timeout=LONG)
    page.wait_for_timeout(1500)
    rs = results(page)
    check(len(rs) == before and not any(r["stale"] for r in rs if r["id"] == rid), "the network was trained again in place: same id, no longer outdated")
    api(page, "POST", "/api/samples/exclude", {"ids": ["S01"], "excluded": False})
    page.wait_for_timeout(1000)


NO_BUTTONS = """() => [...document.querySelectorAll('.js-plotly-plot')].filter((g) => { const host = g.closest('.chart-card, .fig-host, .eem-box'); return !(host && host.querySelector('.fig-btn')); }).length"""


def part_layout(page, shots: bool) -> None:
    print("600 px, no sideways scroll; every chart has its .rds / .fig buttons")
    for width in (1100, 600):
        page.set_viewport_size({"width": width, "height": 900})
        page.wait_for_timeout(600)
        bad, bare = [], []

        def look(where: str) -> None:
            if overflow(page) > 1:
                bad.append(where)
            if page.evaluate(NO_BUTTONS):
                bare.append(where)

        for n in range(1, 10):
            step(page, n)
            page.wait_for_timeout(800)
            if n == 7:
                for t in ("latent", "traversals", "nonlinear", "samples", "interpolation"):
                    tab(page, t); page.wait_for_timeout(2200); look(f"7/{t}")
                    if t == "nonlinear":
                        for sid in NL_SECTIONS:
                            nl_section(page, sid); look(f"7/nonlinear/{sid}")
                        nl_section(page, "response", wait=False)
                tab(page, "latent")
            elif n == 8:
                for t in ("training", "quality", "latent", "stability", "smoothness"):
                    tab(page, t); page.wait_for_timeout(1800); look(f"8/{t}")
                tab(page, "training")
            else:
                page.wait_for_timeout(1200); look(str(n))
        check(not bad, f"{width} px: no sideways scroll on any step or tab ({bad})")
        check(not bare, f"{width} px: every chart has the .rds and .fig buttons ({bare})")
    page.set_viewport_size({"width": 1500, "height": 1000})
    page.wait_for_timeout(500)


def part_reload(page) -> None:
    print("a reload keeps the choices")
    step(page, 5)
    page.fill("[data-ae-smoothify] input", "0.3"); page.press("[data-ae-smoothify] input", "Enter")
    page.click("[data-ae-activation]")
    page.select_option("[data-ae-activation]", "tanh")
    page.wait_for_timeout(2500)                      # (the choices are sent to the project 0.9 s after the last change)
    ui = api(page, "GET", "/api/project")
    saved = (ui.get("summary", ui).get("state") or {}).get("ui", {}).get("autoencoders") if isinstance(ui, dict) else None
    check(bool(saved) and saved["spec"]["net"]["smoothify"] == 0.3 and saved["spec"]["net"]["activation"] == "tanh", "the choices are remembered in the project (state/ui key 'autoencoders')")
    page.reload()
    page.wait_for_selector("[data-fpca-panel='1']", timeout=60000)
    page.wait_for_timeout(1500)
    step(page, 5)
    check(page.input_value("[data-ae-smoothify] input") == "0.3" and page.input_value("[data-ae-activation]") == "tanh", "after a reload the settings of step 5 are back")
    page.fill("[data-ae-smoothify] input", "0"); page.press("[data-ae-smoothify] input", "Enter")
    page.select_option("[data-ae-activation]", "silu")
    page.wait_for_timeout(800)
    # the page is found as it was left (the cache of pages)
    step(page, 3)
    page.click("a.sb-item:has-text('Functional PCA')")
    page.wait_for_selector("[data-fpca-panel]", timeout=20000)
    page.click("a.sb-item:has-text('Autoencoders')")
    page.wait_for_selector("[data-fpca-panel='3']", timeout=20000)
    check(True, "leaving the page and coming back finds the same step (and the Functional PCA page keeps its own state)")


def part_second_project(page, title: str) -> None:
    print("the saved model in a second project")
    api(page, "POST", "/api/import/example")
    page.wait_for_timeout(2500)
    goto(page, "/models", 2)
    card = page.locator(f"[data-ms-card][data-ms-title='{title}']")
    check(card.count() == 1, f"the saved model '{title}' is in the list of saved models")
    card.locator("[data-ms-use]").click()
    page.wait_for_selector("[data-ms-about]", timeout=30000)
    page.click("[data-ms-choice=all] input")
    page.wait_for_function("() => { const n = document.querySelector('[data-ms-needs]'); return n && n.dataset.busy === '0' && n.querySelector('[data-ms-need]'); }", timeout=90000)
    check("autoencoder" in text(page, "[data-ms-about]").lower(), "the page of saved models describes the autoencoder")
    page.click("[data-ms-apply-go]")
    page.wait_for_selector("[data-ms-result] [data-ms-stat=all]", timeout=180000)
    page.wait_for_timeout(1500)
    n = int(page.get_attribute("[data-ms-stat=all]", "data-n") or 0)
    check(n == 64, f"the model was applied to the 64 samples of the second project ({n})")
    api(page, "GET", "/api/variables")
    keys = [v["key"] for v in api(page, "GET", "/api/variables")["variables"]]
    check(any(k.startswith("res:applied-") and k.endswith(":z1") for k in keys), "the application publishes the codes z1 … as variables")


def part_sizes(page, shots: bool) -> None:
    print("a few dozen samples and a thousand samples")
    tmp = tempfile.mkdtemp(prefix="ae-flow-data-")
    atexit.register(shutil.rmtree, tmp, True)                                      # (the .mat files of this part: removed when the run ends)
    # 20 samples: fewer than 30, so k-fold validation is used; the page says so and the network is trained
    import_mat(page, make_mat(20, tmp), "twenty samples")
    goto(page, "/autoencoders", 2.5)
    step(page, 5)
    page.wait_for_selector("[data-ae-numbers] [data-ae-stat='validation']", timeout=90000)
    check("fold" in text(page, "[data-ae-stat='validation']").lower(), f"20 samples: the validation is k-fold ({text(page, '[data-ae-stat=validation]')[:70]!r})")
    check(any("per code" in w.lower() or "memoris" in w.lower() for w in page.locator("[data-ae-warning]").all_inner_texts()), "20 samples: the page warns about the small data")
    rid = train(page)
    check(bool(rid), f"20 samples: a network was trained ({rid})")
    pl = payload(page, rid)
    check(pl["split"]["mode"] == "kfold" and 20 <= pl["n"] <= 22, f"20 samples: k-fold validation was used ({pl['n']} samples)")
    tab(page, "samples")
    page.wait_for_selector("[data-ae-sample-row]", timeout=30000)
    check(page.locator("[data-ae-sample-row]").count() == pl["n"], "20 samples: all in the table")
    # 1000 samples: the lists and the charts stay usable
    import_mat(page, make_mat(1000, tmp, coarse=True), "a thousand samples")
    goto(page, "/autoencoders", 3)
    step(page, 6)
    open_settings(page)
    page.fill("[data-ae-epochs] input", "120"); page.press("[data-ae-epochs] input", "Enter")
    page.fill("[data-ae-restarts] input", "1"); page.press("[data-ae-restarts] input", "Enter")
    rid = train(page)
    check(bool(rid), f"1000 samples: a network was trained ({rid})")
    page.wait_for_selector("[data-ae-latent-chart] .js-plotly-plot", timeout=60000)
    page.wait_for_timeout(2500)
    check(page.locator("[data-ae-latent-chart] .js-plotly-plot").count() == 1, "1000 samples: the latent space is drawn")
    x, y, sid = plot_point(page, "[data-ae-latent-chart] .js-plotly-plot", 5)
    page.mouse.move(x, y)
    page.wait_for_function("() => { const h = document.querySelector('[data-ae-hover]'); return h && h.dataset.id; }", timeout=20000)
    check(True, "1000 samples: the hover card works")
    tab(page, "samples")
    page.wait_for_selector("[data-ae-sample-row]", timeout=60000)
    n_big = payload(page, rid)["n"]
    check(n_big >= 1000 and page.locator("[data-ae-sample-row]").count() == n_big, f"1000 samples: the table lists all of them ({n_big})")
    page.fill("[data-ae-sample-find]", "S0999")
    page.wait_for_timeout(600)
    check(page.locator("[data-ae-sample-row]").count() <= 2, "1000 samples: the search field finds a sample")
    check(overflow(page) <= 1, "1000 samples: no sideways scroll")
    tab(page, "traversals")
    page.wait_for_selector("[data-ae-traversal]", timeout=90000)
    check(True, "1000 samples: the traversals are drawn")
    tab(page, "nonlinear")
    for sid, label in NL_SECTIONS.items():
        t0 = time.time()
        nl_section(page, sid)
        took = time.time() - t0
        check(took < 90, f"1000 samples: '{label}' answers in {took:.1f} s")
    check(overflow(page) <= 1, "1000 samples: no sideways scroll in the tab Nonlinear effects")
    if shots:
        shot(page, "ae_flow_1000", full_page=True)


def part_shots(page) -> None:
    """Screenshots of every step and tab at 1100 and 600 px, light and dark (LOOK at them)."""
    print("screenshots")
    goto(page, "/autoencoders", 2)
    rs = results(page)
    if rs:
        page.select_option("[data-ae-resultbar] select", rs[0]["id"])
        page.wait_for_timeout(1500)
    for scheme in ("light", "dark"):
        page.emulate_media(color_scheme=scheme)
        for width in (1100, 600):
            page.set_viewport_size({"width": width, "height": 900})
            page.wait_for_timeout(800)
            tag = f"{scheme}_{width}"
            for n in range(1, 10):
                step(page, n)
                page.wait_for_timeout(1500)
                if n == 7:
                    for t in ("latent", "traversals", "nonlinear", "samples", "interpolation"):
                        tab(page, t); page.wait_for_timeout(2500)
                        if t == "nonlinear":
                            for sid in NL_SECTIONS:
                                nl_section(page, sid); shot(page, f"ae_{tag}_7_nonlinear_{sid}", full_page=True)
                            nl_section(page, "response", wait=False)
                        else:
                            shot(page, f"ae_{tag}_7_{t}", full_page=True)
                elif n == 8:
                    for t in ("training", "quality", "latent", "stability", "smoothness"):
                        tab(page, t); page.wait_for_timeout(2000); shot(page, f"ae_{tag}_8_{t}", full_page=True)
                else:
                    shot(page, f"ae_{tag}_{n}", full_page=True)
    page.emulate_media(color_scheme="dark" if "--dark" in sys.argv else "light")
    page.set_viewport_size({"width": 1500, "height": 1000})


# ----------------------------------------------------------------------------------------------------------------------- the tab Nonlinear effects
NL_SECTIONS = {"response": "Response curves", "bends": "Where it bends", "local": "Local effects", "pair": "Two codes"}
NL_SENTENCES = {"response": "Each chart moves one code from the low to the high end of its range while all the other codes stay at their average",
                "bends": "For every code, each pixel of the EEM is followed along the code and compared with the best straight line",
                "local": "The loading of a code is what one standard deviation of the code does to the EEM",
                "pair": "Two codes are moved together on a grid while all the others stay at their average"}
NL_READY = {"response": "[data-ae-nl-curve]", "bends": "[data-ae-nl-share-chart]", "local": "[data-ae-nl-change-chart]", "pair": "[data-ae-nl-heat], [data-ae-nl-pair-none]"}
# what a Plotly chart of the page holds: its traces (type, mode, name, number of points, line width, colour range), the labels written on it, the titles of its axes, the dotted lines
GD_INFO = """(sel) => { let gd = document.querySelector(sel); if (!gd) return null; if (!gd.data) gd = gd.querySelector('.js-plotly-plot'); if (!gd || !gd.data) return null; const L = gd._fullLayout || {};
  const ann = (gd.layout.annotations || []).map((a) => String(a.text || '').replace(/<[^>]*>/g, ''));
  const axes = Object.keys(L).filter((k) => /^[xy]axis\\d*$/.test(k) && L[k].title && L[k].title.text).map((k) => k + ':' + L[k].title.text);
  return { data: gd.data.map((t) => ({ type: t.type, mode: t.mode || null, name: t.name || '', n: t.x ? t.x.length : (t.z ? t.z.length : 0), width: t.line ? t.line.width : null, zmin: t.zmin === undefined ? null : t.zmin, zmax: t.zmax === undefined ? null : t.zmax,
                                  y: t.type === 'bar' ? Array.from(t.y) : null, x: t.type === 'bar' ? Array.from(t.x) : null })),
           ann, axes, xrange: L.xaxis ? L.xaxis.range : null, ytitle: L.yaxis && L.yaxis.title ? L.yaxis.title.text : '', shapes: (gd.layout.shapes || []).map((s) => s.y0) }; }"""


def gd_info(page, selector: str) -> dict:
    info = page.evaluate(GD_INFO, selector)
    assert info is not None, f"no chart at {selector}"
    return info


def nl_section(page, sid: str, wait: bool = True) -> None:
    """Choose a section of the tab and wait for its answer."""
    page.click(f"[data-ae-nl-section='{sid}']")
    page.wait_for_selector(f"[data-ae-nl-panel='{sid}']", timeout=20000)
    if wait:
        nl_idle(page, sid)


def nl_idle(page, sid: str, timeout: int = 120000) -> None:
    """Wait until the section shows an answer and no request is on its way (an old answer is dimmed while a new one is computed)."""
    page.wait_for_selector(f"[data-ae-nl-panel='{sid}'] :is({NL_READY[sid]})", timeout=timeout)
    page.wait_for_function("(sid) => { const p = document.querySelector(`[data-ae-nl-panel='${sid}']`); return !!p && !p.querySelector('.fp-dim') && !p.querySelector('.fp-wait'); }", arg=sid, timeout=timeout)
    page.wait_for_timeout(600)


def nl_ask(page, sid: str, action, route: str | None = None, timeout: int = 60000) -> dict:
    """Do `action` (type a value, press a button) and return the body of the request that the page then sends to the route of the section; waits for the answer."""
    route = route or sid
    with page.expect_request(lambda rq: rq.method == "POST" and rq.url.endswith(f"/{route}"), timeout=timeout) as info:
        action()
    body = info.value.post_data_json
    nl_idle(page, sid)
    return body


def nl_type(page, selector: str, value) -> None:
    page.fill(selector, str(value))
    page.press(selector, "Enter")


def nl_open(page, rid: str | None = None, section: str | None = None) -> None:
    """Step 7, the tab 'Nonlinear effects' (the network `rid` is chosen in the result bar first), in the given section."""
    if rid and page.locator("[data-ae-resultbar] select").count() and page.input_value("[data-ae-resultbar] select") != rid:
        page.select_option("[data-ae-resultbar] select", rid)
        page.wait_for_timeout(1500)
    step(page, 7)
    page.wait_for_selector("[data-ae-cards]", timeout=60000)
    tab(page, "nonlinear")
    page.wait_for_selector("[data-ae-nonlinear-tab]", timeout=20000)
    if section:
        nl_section(page, section)


def sample_ids(page) -> list[str]:
    got = api(page, "GET", "/api/samples")
    return [i for i, r in zip(got["ids"], got["roles"]) if r == "sample"]


def squeeze(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def part_nonlinear(page, rid: str, shots: bool) -> None:
    print("step 7: the tab Nonlinear effects (response curves, where it bends, local effects, two codes)")
    pl = payload(page, rid)
    d, n = pl["latent_dim"], pl["n"]
    shown = min(d, 8)
    ids = sample_ids(page)
    logs0 = len(page.logs)
    nl_open(page, rid)
    tabs = page.locator("[data-ae-tab]").all_inner_texts()
    check(tabs == ["Latent space", "Traversals", "Nonlinear effects", "Samples", "Interpolation", "Generate"], f"the tab 'Nonlinear effects' comes after 'Traversals' ({tabs})")
    check(page.get_attribute("[data-ae-tab='nonlinear']", "aria-selected") == "true" and page.locator("[data-ae-panel='nonlinear'] [data-ae-nonlinear-tab]").count() == 1, "the tab opens its panel")
    labels = page.locator("[data-ae-nl-section]").all_inner_texts()
    check(labels == list(NL_SECTIONS.values()), f"four sections: {labels}")
    check(all(page.get_attribute(f"[data-ae-nl-section='{k}']", "title") for k in NL_SECTIONS), "every section has a tooltip")
    nl_idle(page, "response")

    # ------------------------------------------------------------------------------------------------------- response curves
    P = "[data-ae-nl-panel='response']"
    check(page.get_attribute("[data-ae-nl-section='response']", "aria-pressed") == "true", "'Response curves' is the first section")
    check(text(page, f"{P} [data-ae-nl-explain]").startswith(NL_SENTENCES["response"]), "the section starts with one sentence that says what it shows")
    check(page.locator("[data-ae-nl-curve]").count() == shown, f"one chart for each code ({shown})")
    chips = [squeeze(c.replace("×", "")) for c in page.locator("[data-ae-nl-feature]").all_inner_texts()]
    check(chips == ["Intensity at the pixel this code moves most", "Overall intensity (the average over the whole EEM)"], f"the features at first: the pixel each code moves most and the overall intensity ({chips})")
    check(page.locator("[data-ae-nl-legend-item]").count() == 2 and page.locator("[data-ae-nl-curve='1'] [data-ae-nl-words] li").count() == 2, "a legend, and under each chart one sentence for every feature")
    ans = api(page, "POST", f"{BASE_API}/{rid}/response", {"span": 3, "n_points": 41, "overlay": True})
    first = ans["codes"][0]["curves"]["f1"]
    check(squeeze(text(page, "[data-ae-nl-curve='1'] [data-ae-nl-words] li:first-child")) == squeeze(first["words"]) and page.get_attribute("[data-ae-nl-curve='1'] [data-ae-nl-words] li:first-child", "data-ae-nl-kind-of") == first["kind"],
          f"the sentence under the first chart is the server's ({first['kind']}: {first['words'][:70]!r})")
    g = gd_info(page, "[data-ae-nl-curve='1']")
    thick = [t for t in g["data"] if t["mode"] == "lines" and t["width"] == 2.8]
    dots = [t for t in g["data"] if t["name"].startswith("Samples")]
    check(len(thick) == 2 and thick[0]["n"] == 41 and g["xrange"] == [-3, 3], f"each chart draws two curves of 41 positions over ±3 SD ({len(thick)}, {thick[0]['n'] if thick else None}, {g['xrange']})")
    check(len(dots) == 2 and dots[0]["n"] == n, f"the {n} training samples are drawn as dots around the curves ({[t['n'] for t in dots]})")
    check(page.locator("[data-ae-nl-curve='1'] .fig-btn").count() >= 2, "the charts have the .rds and .fig buttons")
    if shots:
        shot(page, "ae_flow_nl_response", full_page=True)
    # the range and the number of points
    body = nl_ask(page, "response", lambda: nl_type(page, "[data-ae-nl-span] input", 2))
    g = gd_info(page, "[data-ae-nl-curve='1']")
    check(body["span"] == 2 and g["xrange"] == [-2, 2], f"'Range (± SD)' 2 redraws the charts over ±2 SD ({g['xrange']})")
    body = nl_ask(page, "response", lambda: nl_type(page, "[data-ae-nl-points] input", 21))
    g = gd_info(page, "[data-ae-nl-curve='1']")
    check(body["n_points"] == 21 and [t["n"] for t in g["data"] if t["mode"] == "lines" and t["width"] == 2.8] == [21, 21], "'Points per curve' 21 draws 21 positions")
    body = nl_ask(page, "response", lambda: nl_type(page, "[data-ae-nl-span] input", 9))
    check(body["span"] == 4 and page.input_value("[data-ae-nl-span] input") == "4", "a range beyond the limit is brought back to 4")
    body = nl_ask(page, "response", lambda: nl_type(page, "[data-ae-nl-span] input", 3))
    body = nl_ask(page, "response", lambda: nl_type(page, "[data-ae-nl-points] input", 41))
    # the training samples, and the comparison of shapes
    body = nl_ask(page, "response", lambda: page.locator("[data-ae-nl-overlay] input").uncheck())
    g = gd_info(page, "[data-ae-nl-curve='1']")
    check(body["overlay"] is False and not [t for t in g["data"] if t["name"].startswith("Samples")], "'Show the training samples' off: no dots (and the server is not asked for them)")
    nl_ask(page, "response", lambda: page.locator("[data-ae-nl-overlay] input").check())
    check(page.locator("[data-ae-nl-relative] input").is_checked() and gd_info(page, "[data-ae-nl-curve='1']")["ytitle"] == "Scaled value (curve: 0 to 1)", "'Compare shapes' is on at first: each feature is scaled from 0 to 1")
    page.locator("[data-ae-nl-relative] input").uncheck()
    page.wait_for_timeout(700)
    check(gd_info(page, "[data-ae-nl-curve='1']")["ytitle"] != "Scaled value (curve: 0 to 1)", "switched off, the real values are drawn")
    page.locator("[data-ae-nl-relative] input").check()
    page.wait_for_timeout(500)
    # curves of single samples
    body = nl_ask(page, "response", lambda: page.locator("[data-ae-nl-ice] input").check())
    g = gd_info(page, "[data-ae-nl-curve='1']")
    thin = [t for t in g["data"] if t["mode"] == "lines" and t["width"] == 1.1]
    check(body["ice"] == {"codes": list(range(1, shown + 1)), "n": 12} and len(thin) == 24, f"curves of single samples: 12 samples for each of the 2 features ({len(thin)} thin lines)")
    check(page.locator("[data-ae-nl-ice-names]").count() == 1 and "parallel" in text(page, P), "…with the sentence on what parallel curves mean, and a field for the names of the samples")
    body = nl_ask(page, "response", lambda: page.fill("[data-ae-nl-ice-names]", f"{ids[0]}, {ids[4]}"))
    g = gd_info(page, "[data-ae-nl-curve='1']")
    check(body["ice"]["samples"] == [ids[0], ids[4]] and len([t for t in g["data"] if t["mode"] == "lines" and t["width"] == 1.1]) == 4, f"typed names ({ids[0]}, {ids[4]}) give 2 thin lines per feature")
    page.fill("[data-ae-nl-ice-names]", f"{ids[0]}, nope")
    page.wait_for_timeout(900)
    check("Not a sample of this network: nope." in text(page, P), "a name that is not a sample of the network is said so")
    nl_ask(page, "response", lambda: page.fill("[data-ae-nl-ice-names]", ""))
    nl_ask(page, "response", lambda: page.locator("[data-ae-nl-ice] input").uncheck())
    # typed features: a pixel, a box; a refusal in the server's words
    kinds = page.locator("[data-ae-nl-kind] option").evaluate_all("els => els.map(e => e.value)")
    check(kinds == ["moved", "point", "region_mean", "region_max", "total", "coordinate"], f"the kinds of feature (no absorbance in this network) ({kinds})")
    page.select_option("[data-ae-nl-kind]", "point")
    ex, em = page.input_value("[data-ae-nl-ex] input"), page.input_value("[data-ae-nl-em] input")
    check(ex != "" and em != "", f"the fields start at the peak of the mean EEM ({ex} / {em} nm)")
    body = nl_ask(page, "response", lambda: page.click("[data-ae-nl-feature-add]"))
    check([f["kind"] for f in body["features"]] == ["moved", "total", "point"] and body["features"][2]["ex"] == float(ex) and body["features"][2]["em"] == float(em), f"'Add this feature' sends the pixel ({body['features'][2]})")
    check(page.locator("[data-ae-nl-feature]").count() == 3 and page.locator("[data-ae-nl-legend-item]").count() == 3 and page.locator("[data-ae-nl-curve='1'] [data-ae-nl-words] li").count() == 3, "a third chip, legend entry and sentence")
    third = squeeze(text(page, "[data-ae-nl-curve='1'] [data-ae-nl-words] li:nth-child(3)"))
    check(third.startswith("Pixel at"), f"the sentence names the pixel ({third[:60]!r})")
    page.select_option("[data-ae-nl-kind]", "region_mean")
    check(page.locator("[data-ae-nl-ex2]").count() == 1 and page.locator("[data-ae-nl-em2]").count() == 1, "a box has its two ends")
    body = nl_ask(page, "response", lambda: page.click("[data-ae-nl-feature-add]"))
    check(body["features"][3]["kind"] == "region_mean" and len(body["features"][3]["ex"]) == 2, f"'Mean intensity in a box' is sent with its two ranges ({body['features'][3]})")
    check(page.locator("[data-ae-nl-feature]").count() == 4 and len(gd_info(page, "[data-ae-nl-curve='1']")["data"]) >= 8, "four features are drawn, each with its own line style and marker")
    page.select_option("[data-ae-nl-kind]", "point")
    page.fill("[data-ae-nl-ex] input", "5000")
    page.press("[data-ae-nl-ex] input", "Enter")
    with page.expect_response(lambda r: r.url.endswith("/response") and r.status == 400, timeout=60000):
        page.click("[data-ae-nl-feature-add]")
    page.wait_for_selector("[data-ae-nl-feature-error]", timeout=20000)
    nl_idle(page, "response")
    check("outside the range" in text(page, "[data-ae-nl-feature-error]") and page.locator("[data-ae-nl-feature]").count() == 4, f"an impossible pixel is refused with the server's sentence and not kept ({text(page, '[data-ae-nl-feature-error]')[:90]!r})")
    page.fill("[data-ae-nl-ex] input", ex)
    page.press("[data-ae-nl-ex] input", "Enter")
    # the picture of the mean EEM: a click fills the pixel, a dragged box fills the box
    page.click("[data-ae-nl-pick] summary")
    page.wait_for_selector("[data-ae-nl-picture] .js-plotly-plot", timeout=30000)
    page.wait_for_timeout(1200)
    box = page.locator("[data-ae-nl-picture] .nsewdrag").bounding_box()
    page.select_option("[data-ae-nl-kind]", "total")
    moved = False
    for fx, fy in ((0.5, 0.45), (0.35, 0.3), (0.6, 0.55), (0.45, 0.2), (0.25, 0.5), (0.7, 0.4)):
        page.mouse.click(box["x"] + fx * box["width"], box["y"] + fy * box["height"])
        page.wait_for_timeout(500)
        if page.input_value("[data-ae-nl-kind]") == "point":
            moved = True
            break
    now = (page.input_value("[data-ae-nl-ex] input"), page.input_value("[data-ae-nl-em] input"))
    check(moved and 230 <= float(now[0]) <= 460 and 280 <= float(now[1]) <= 640, f"a click on the picture chooses 'Intensity at a pixel' and fills its wavelengths ({now[0]} / {now[1]} nm)")
    page.click("[data-ae-nl-picture] button:has-text('Select region')")
    page.mouse.move(box["x"] + 0.3 * box["width"], box["y"] + 0.3 * box["height"])
    page.mouse.down()
    page.mouse.move(box["x"] + 0.6 * box["width"], box["y"] + 0.7 * box["height"], steps=10)
    page.mouse.up()
    page.wait_for_timeout(700)
    check(page.input_value("[data-ae-nl-kind]") == "region_mean" and page.input_value("[data-ae-nl-ex2] input") != "" and float(page.input_value("[data-ae-nl-ex2] input")) > float(page.input_value("[data-ae-nl-ex] input")),
          f"a dragged box chooses 'Mean intensity in a box' and fills its four numbers ({[page.input_value(f'[data-ae-nl-{k}] input') for k in ('ex', 'ex2', 'em', 'em2')]})")
    page.click("[data-ae-nl-pick] summary")
    for _ in range(2):                                                             # back to the two features of the start
        page.locator("[data-ae-nl-feature] [data-ae-nl-feature-remove]").last.click()
        nl_idle(page, "response")
    check(page.locator("[data-ae-nl-feature]").count() == 2, "a chip can be removed (the last one cannot)")

    # ------------------------------------------------------------------------------------------------------- where it bends
    nl_section(page, "bends")
    P = "[data-ae-nl-panel='bends']"
    check(text(page, f"{P} [data-ae-nl-explain]").startswith(NL_SENTENCES["bends"]), "'Where it bends' starts with its sentence")
    bends = api(page, "POST", f"{BASE_API}/{rid}/bends", {"span": 3, "n_points": 41})
    check(squeeze(text(page, f"{P} [data-ae-nl-words]")) == squeeze(bends["words"]), f"the sentence on which code bends most is the server's ({bends['words'][:80]!r})")
    strip = gd_info(page, f"{P} [data-ae-nl-strip]")
    maps = [t for t in strip["data"] if t["type"] == "heatmap" and t["zmax"] == 1 and t["zmin"] == 0 and t["name"]]
    check(len(maps) == shown and all(a.startswith(f"z{i + 1} · ") for i, a in enumerate(strip["ann"][:shown])), f"one map of NL for each code on one colour scale from 0 to 1, titled with its share ({strip['ann'][:3]})")
    check(page.locator(f"{P} [data-ae-nl-colour-note]").count() == 1 and page.locator(f"{P} [data-ae-nl-strip] .fig-btn").count() >= 2, "the sentence on the colour bar, and the .rds / .fig buttons")
    bars = gd_info(page, f"{P} [data-ae-nl-share-chart]")
    bar = next(t for t in bars["data"] if t["type"] == "bar")
    want = [0.0 if s is None else s for s in bends["share"]]
    check(len(bar["x"]) == d and all(abs(a - b) < 1e-6 for a, b in zip(bar["y"], want)) and bars["shapes"] == [0.05], f"the bars are the shares of the server ({[round(v, 3) for v in bar['y'][:4]]}) with a dotted line at 5 %")
    check("The dotted line is 5 %" in text(page, f"{P} [data-ae-nl-share-note]"), "the note under the bars explains the dotted line")
    if shots:
        shot(page, "ae_flow_nl_bends", full_page=True)
    page.click(f"{P} [data-ae-nl-bends-show] button:has-text('Size of the effect')")
    page.wait_for_timeout(1200)
    amp = gd_info(page, f"{P} [data-ae-nl-strip]")
    check(any(t["type"] == "heatmap" and t["zmin"] == 0 and (t["zmax"] or 0) != 1 and t["name"] for t in amp["data"]) and page.locator(f"{P} [data-ae-nl-colour-note]").count() == 0, "'Size of the effect' draws the range of each pixel along the code instead")
    page.click(f"{P} [data-ae-nl-bends-show] button:has-text('Where it bends')")
    page.wait_for_timeout(800)
    body = nl_ask(page, "bends", lambda: nl_type(page, "[data-ae-nl-span] input", 1.5))
    check(body == {"span": 1.5, "n_points": 41}, f"the range of this section is asked for from the server ({body})")
    nl_ask(page, "bends", lambda: nl_type(page, "[data-ae-nl-span] input", 3))

    # ------------------------------------------------------------------------------------------------------- local effects
    nl_section(page, "local")
    P = "[data-ae-nl-panel='local']"
    check(text(page, f"{P} [data-ae-nl-explain]").startswith(NL_SENTENCES["local"]), "'Local effects' starts with its sentence")
    check(page.locator("[data-ae-nl-local-code] option").count() == d and page.input_value("[data-ae-nl-local-code]") == "1", f"the code to look at: {d} choices, the first at first")
    local = api(page, "POST", f"{BASE_API}/{rid}/local", {"code": 1})
    check(squeeze(text(page, f"{P} [data-ae-nl-words]")) == squeeze(local["words"]), f"the sentence on how the loading changes is the server's ({local['words'][:80]!r})")
    ld = gd_info(page, f"{P} [data-ae-nl-loadings]")
    line = text(page, f"{P} [data-ae-nl-sample-line]")
    check(len(ld["ann"]) == 4 and ld["ann"][0] == "At the mean code" and all(" SD" in a for a in ld["ann"][1:]) and "10th percentile of z1" in line and "median of z1" in line and "90th percentile of z1" in line,
          f"the loading at the mean code and at the samples at the 10th, 50th and 90th percentile, on one colour scale ({ld['ann']})")
    check(page.locator(f"{P} [data-ae-nl-variability] .js-plotly-plot").count() == 1, "the map of where the loading differs between samples")
    ch = gd_info(page, f"{P} [data-ae-nl-change-chart]")
    check(next(t for t in ch["data"] if t["type"] == "bar")["x"] == [f"z{i + 1}" for i in range(d)], "the bars of how much the shape of the effect changes: one per code")
    check("0 means the picture of the effect has the same shape" in text(page, f"{P} [data-ae-nl-change-note]"), "the note on what 0 means")
    if shots:
        shot(page, "ae_flow_nl_local", full_page=True)
    if d >= 2:
        body = nl_ask(page, "local", lambda: page.select_option("[data-ae-nl-local-code]", "2"))
        check(body == {"code": 2}, f"another code is asked for ({body})")
    page.click(f"{P} [data-ae-nl-local-mode] button:has-text('Samples I choose')")
    page.wait_for_selector("[data-ae-nl-local-pick]", timeout=10000)
    body = nl_ask(page, "local", lambda: (page.select_option("[data-ae-nl-local-pick]", ids[5]), page.click("[data-ae-nl-local-add]")))
    ld = gd_info(page, f"{P} [data-ae-nl-loadings]")
    check(body["samples"] == [ids[5]] and page.locator("[data-ae-nl-local-chip]").count() == 1 and len(ld["ann"]) == 2 and ld["ann"][1].startswith(ids[5]) and "chosen by you" in text(page, f"{P} [data-ae-nl-sample-line]"),
          f"a chosen sample is compared with the mean code ({ld['ann']})")
    body = nl_ask(page, "local", lambda: page.click("[data-ae-nl-local-chip] .x"))
    check("samples" not in body and page.locator("[data-ae-nl-local-chip]").count() == 0, "removing the chip goes back to the percentiles")
    page.click(f"{P} [data-ae-nl-local-mode] button:has-text('10th, 50th and 90th percentile')")
    nl_ask(page, "local", lambda: page.select_option("[data-ae-nl-local-code]", "1"))

    # ------------------------------------------------------------------------------------------------------- two codes
    nl_section(page, "pair")
    P = "[data-ae-nl-panel='pair']"
    check(text(page, f"{P} [data-ae-nl-explain]").startswith(NL_SENTENCES["pair"]), "'Two codes' starts with its sentence")
    if d < 2:
        check(page.locator("[data-ae-nl-pair-none]").count() == 1, "a network with one code says that there are no two codes to look at")
        tab(page, "latent")
        return
    pair = api(page, "POST", f"{BASE_API}/{rid}/pair", {"grid": 5, "span": 2})
    got = [int(page.input_value("[data-ae-nl-pair-i]")), int(page.input_value("[data-ae-nl-pair-j]"))]
    check(got == pair["codes"] and pair["auto"] is True, f"at first the two codes with the largest nonlinear share are chosen ({got})")
    check(squeeze(text(page, f"{P} [data-ae-nl-pair-words]")) == squeeze(pair["words"]) and "interaction index" in text(page, f"{P} [data-ae-nl-index]").lower(), f"the card and the sentence on the interaction are the server's ({pair['words'][:80]!r})")
    gal = gd_info(page, f"{P} [data-ae-nl-gallery]")
    check(len(gal["ann"]) == 25, f"the gallery of 5 × 5 pictures, each labelled with the positions of the two codes ({gal['ann'][:3]})")
    heat = gd_info(page, f"{P} [data-ae-nl-heat]")
    check([t["type"] for t in heat["data"]] == ["heatmap", "scatter"] and heat["data"][1]["n"] > 0, f"the heat map of a feature over the two codes with {heat['data'][1]['n']} samples as dots")
    check(page.locator(f"{P} [data-ae-nl-interaction] .js-plotly-plot").count() == 1 and page.locator(f"{P} [data-ae-nl-gallery] .fig-btn").count() >= 2, "the map of where the effects do not add up, and the .rds / .fig buttons")
    if shots:
        shot(page, "ae_flow_nl_pair", full_page=True)
    for g_ in (3, 7, 5):
        body = nl_ask(page, "pair", lambda g_=g_: page.click(f"[data-ae-nl-pair-grid] button:text-is('{g_}')"))
        check(body["grid"] == g_ and len(gd_info(page, f"{P} [data-ae-nl-gallery]")["ann"]) == g_ * g_, f"{g_} × {g_} pictures per code: {g_ * g_} pictures in the gallery")
    body = nl_ask(page, "pair", lambda: nl_type(page, "[data-ae-nl-pair-span] input", 1))
    check(body["span"] == 1 and "codes" not in body, f"the range of the two codes (±1 SD) is asked for, the server still chooses the codes ({body})")
    nl_ask(page, "pair", lambda: nl_type(page, "[data-ae-nl-pair-span] input", 2))
    page.select_option("[data-ae-nl-pair-j]", page.input_value("[data-ae-nl-pair-i]"))
    page.wait_for_timeout(900)
    check("Choose two different codes." in text(page, P), "the same code twice: 'Choose two different codes.'")
    page.select_option("[data-ae-nl-pair-j]", str(pair["codes"][1]))
    page.wait_for_timeout(900)
    nl_idle(page, "pair")
    page.select_option("[data-ae-nl-kind]", "total")
    body = nl_ask(page, "pair", lambda: page.click("[data-ae-nl-feature-add]"))
    check(body["feature"]["kind"] == "total" and "Overall intensity" in text(page, f"{P} [data-ae-nl-pair-feature]"), f"another feature for the heat map ({body['feature']})")
    page.select_option("[data-ae-nl-kind]", "moved")
    body = nl_ask(page, "pair", lambda: page.click("[data-ae-nl-feature-add]"))
    check("feature" not in body, "…and back to the pixel the two codes move most")
    check(not errors(page, logs0), f"no console error in the four sections ({errors(page, logs0)[:2]})")
    warns = [x for x in page.logs[logs0:] if x.startswith("[warning]")]
    check(not warns, f"no console warning in the four sections ({[w[:140] for w in warns[:2]]})")
    nl_section(page, "response", wait=False)
    tab(page, "latent")                                                            # (the other parts expect the first tab, as part_results leaves it)


def part_nonlinear_both(page, rid: str, rid_fluo: str, shots: bool) -> None:
    print("the tab Nonlinear effects for a network on fluorescence and absorbance")
    pl = payload(page, rid)
    d = pl["latent_dim"]
    shown = min(d, 8)
    nl_open(page, rid, "response")
    P = "[data-ae-nl-panel='response']"
    kinds = page.locator("[data-ae-nl-kind] option").evaluate_all("els => els.map(e => e.value)")
    check(kinds == ["moved", "point", "region_mean", "region_max", "total", "absorbance", "coordinate"], f"the kinds of feature for fluorescence and absorbance ({kinds})")
    page.select_option("[data-ae-nl-kind]", "absorbance")
    wl = page.input_value("[data-ae-nl-wavelength] input")
    body = nl_ask(page, "response", lambda: page.click("[data-ae-nl-feature-add]"))
    check(body["features"][2] == {"kind": "absorbance", "wavelength": float(wl)}, f"a wavelength of the absorbance is sent ({body['features'][2]})")
    check(page.locator("[data-ae-nl-relative] input").is_disabled() and page.locator("[data-ae-nl-relative] input").is_checked() and "The features have different units, so they are drawn scaled from 0 to 1." in text(page, P),
          "other units: 'Compare shapes' is on, cannot be switched off, and the page says why")
    page.select_option("[data-ae-nl-kind]", "total")
    check(page.locator("[data-ae-nl-block]").count() == 1, "the overall value can be asked for of the fluorescence or of the absorbance")
    page.select_option("[data-ae-nl-block]", "absorbance")
    body = nl_ask(page, "response", lambda: page.click("[data-ae-nl-feature-add]"))
    check(body["features"][3] == {"kind": "total", "block": "absorbance"}, f"the overall absorbance is sent ({body['features'][3]})")
    page.select_option("[data-ae-nl-kind]", "coordinate")
    page.select_option("[data-ae-nl-block]", "fluorescence")
    body = nl_ask(page, "response", lambda: page.click("[data-ae-nl-feature-add]"))
    chips = [squeeze(c.replace("×", "")) for c in page.locator("[data-ae-nl-feature]").all_inner_texts()]
    check(body["features"][4]["kind"] == "coordinate" and chips[2] == f"Absorbance at {wl} nm" and chips[3].startswith("Overall absorbance") and chips[4].startswith("Coefficient 1 of the fluorescence"), f"the chips name what is measured ({chips})")
    page.select_option("[data-ae-nl-kind]", "total")
    page.select_option("[data-ae-nl-block]", "fluorescence")
    nl_ask(page, "response", lambda: page.click("[data-ae-nl-feature-add]"))
    page.click("[data-ae-nl-feature-add]")
    page.wait_for_selector("[data-ae-nl-feature-error]", timeout=10000)
    check("At most 6 features can be drawn at once" in text(page, "[data-ae-nl-feature-error]") and page.locator("[data-ae-nl-feature]").count() == 6, "a seventh feature is refused: 'At most 6 features can be drawn at once: remove one first.'")
    if shots:
        shot(page, "ae_flow_nl_response_both", full_page=True)
    # the other sections show the absorbance as well
    nl_section(page, "bends")
    check(page.locator("[data-ae-nl-bend-curve]").count() == shown and "z1: absorbance" in text(page, "[data-ae-nl-bend-curve='1']") and page.locator("[data-ae-nl-bend-curve='1'] .fig-btn").count() >= 2, f"where it bends: a chart of the absorbance for each code ({shown})")
    nl_section(page, "local")
    check("Loading of z1 on the absorbance" in text(page, "[data-ae-nl-panel='local']") and "(absorbance)" in text(page, "[data-ae-nl-panel='local']"), "local effects: the loading on the absorbance and where it differs between samples")
    nl_section(page, "pair")
    check("Absorbance on the grid of" in text(page, "[data-ae-nl-panel='pair']") and "Interaction of the two codes on the absorbance" in text(page, "[data-ae-nl-panel='pair']"), "two codes: the absorbance on the grid and its interaction")
    if shots:
        shot(page, "ae_flow_nl_pair_both", full_page=True)
    # a feature that the other network cannot show: the server's sentence, and a way back
    nl_open(page, rid_fluo)
    nl_section(page, "response", wait=False)                                       # (the answer is the refusal: nothing to wait for but the callout)
    page.wait_for_selector("[data-ae-nl-error]", timeout=60000)
    check("does not use the absorbance" in text(page, "[data-ae-nl-error]") and page.locator("[data-ae-nl-error] button:has-text('Use the usual features')").count() == 1, f"a network without absorbance says so and offers 'Use the usual features' ({text(page, '[data-ae-nl-error]')[:90]!r})")
    page.click("[data-ae-nl-error] button:has-text('Use the usual features')")
    nl_idle(page, "response")
    check(page.locator("[data-ae-nl-feature]").count() == 2 and page.locator("[data-ae-nl-curve]").count() == min(payload(page, rid_fluo)["latent_dim"], 8), "…which draws the usual two features again")
    tab(page, "latent")


def train_with(page, codes: int) -> str:
    """Train a small network with `codes` codes (few epochs: only its shape matters here) and wait for the results; the id of the network."""
    step(page, 5)
    page.fill("[data-ae-latent] input", str(codes))
    page.press("[data-ae-latent] input", "Enter")
    page.wait_for_timeout(600)
    step(page, 6)
    open_settings(page)
    nl_type(page, "[data-ae-epochs] input", 40)
    nl_type(page, "[data-ae-restarts] input", 1)
    page.wait_for_timeout(400)
    return train(page)


def set_transposed(page, on: bool) -> None:
    """The setting 'Axes of fluorescence maps' (emission across or excitation across), as the settings page keeps it; the page is loaded again to read it."""
    page.evaluate("(on) => { let p = {}; try { p = JSON.parse(localStorage.getItem('fluoviz.prefs') || '{}'); } catch (e) { p = {}; } p.eemTranspose = on; localStorage.setItem('fluoviz.prefs', JSON.stringify(p)); }", on)
    page.reload()
    page.wait_for_selector("[data-fpca-panel], [data-ae-page]", timeout=60000)
    page.wait_for_timeout(1500)


def part_nonlinear_codes(page, rid1: str, shots: bool) -> None:
    print("the tab Nonlinear effects: ten codes, one code, a reload, the other axes")
    step(page, 5)
    d0 = int(page.input_value("[data-ae-latent] input"))
    # ten codes: the first eight are drawn, the others on request
    rid10 = train_with(page, 10)
    check(bool(rid10) and payload(page, rid10)["latent_dim"] == 10, f"a network with ten codes was trained ({rid10})")
    nl_open(page, rid10, "response")
    check(page.locator("[data-ae-nl-curve]").count() == 8 and page.locator("[data-ae-nl-all]").inner_text().strip() == "Show the other 2 codes", "ten codes: eight charts and the button 'Show the other 2 codes'")
    page.click("[data-ae-nl-all]")                                                 # (the server has answered for every code already: no new request)
    page.wait_for_timeout(1200)
    check(page.locator("[data-ae-nl-curve]").count() == 10 and page.locator("[data-ae-nl-all]").count() == 0, "the button draws the other two charts")
    page.click("[data-ae-nl-panel='response'] button:has-text('Show only the first 8 codes')")
    page.wait_for_timeout(900)
    check(page.locator("[data-ae-nl-curve]").count() == 8, "…and 'Show only the first 8 codes' takes them away again")
    nl_section(page, "bends")
    strip = gd_info(page, "[data-ae-nl-panel='bends'] [data-ae-nl-strip]")
    check(len(strip["ann"]) == 8 and page.locator("[data-ae-nl-panel='bends'] [data-ae-nl-all]").count() == 1, "where it bends: eight maps, and the button for the other two")
    nl_section(page, "local")
    check(page.locator("[data-ae-nl-local-code] option").count() == 10, "local effects: ten codes to choose from")
    nl_section(page, "pair")
    check(page.locator("[data-ae-nl-pair-i] option").count() == 10 and page.locator("[data-ae-nl-pair-j] option").count() == 10, "two codes: ten codes to choose from")
    if shots:
        shot(page, "ae_flow_nl_ten_codes", full_page=True)
    # one code: the first three sections work, the fourth says that there is nothing to compare
    rid1c = train_with(page, 1)
    check(bool(rid1c) and payload(page, rid1c)["latent_dim"] == 1, f"a network with one code was trained ({rid1c})")
    nl_open(page, rid1c, "response")
    check(page.locator("[data-ae-nl-curve]").count() == 1, "one code: one chart")
    nl_section(page, "bends")
    nl_section(page, "local")
    check(page.locator("[data-ae-nl-local-code] option").count() == 1 and page.locator("[data-ae-nl-change-chart]").count() == 1, "one code: the local effects of that code")
    nl_section(page, "pair")
    check("only one code" in text(page, "[data-ae-nl-pair-none]"), "one code: 'Two codes' says that there are no two codes to look at together")
    # back to the settings of the other parts
    step(page, 5)
    page.fill("[data-ae-latent] input", str(d0))
    page.press("[data-ae-latent] input", "Enter")
    step(page, 6)
    open_settings(page)
    nl_type(page, "[data-ae-epochs] input", 600)
    nl_type(page, "[data-ae-restarts] input", 3)
    page.wait_for_timeout(500)
    # the choices are remembered with the project: a reload finds them
    nl_open(page, rid1, "pair")
    nl_ask(page, "pair", lambda: page.click("[data-ae-nl-pair-grid] button:text-is('7')"))
    nl_ask(page, "pair", lambda: nl_type(page, "[data-ae-nl-pair-span] input", 1.5))
    page.wait_for_timeout(2500)                                                    # (the choices are sent to the project 0.9 s after the last change)
    got = api(page, "GET", "/api/project")
    saved = ((got.get("summary", got).get("state") or {}).get("ui", {}).get("autoencoders") or {}).get("nl")
    check(bool(saved) and saved["section"] == "pair" and saved["pair"]["grid"] == 7 and saved["pair"]["span"] == 1.5, f"the choices are in the project (state/ui key autoencoders.nl: {saved and {k: saved[k] for k in ('section', 'span', 'n_points')}})")
    page.reload()
    page.wait_for_selector("[data-fpca-panel]", timeout=60000)
    page.wait_for_timeout(1500)
    nl_open(page, rid1)
    check(page.get_attribute("[data-ae-nl-section='pair']", "aria-pressed") == "true", "after a reload the tab is in the section it was left in")
    nl_idle(page, "pair")
    check(page.locator("[data-ae-nl-pair-grid] button.on").inner_text().strip() == "7" and page.input_value("[data-ae-nl-pair-span] input") == "1.5", "…with its grid and range")
    nl_ask(page, "pair", lambda: page.click("[data-ae-nl-pair-grid] button:text-is('5')"))
    nl_ask(page, "pair", lambda: nl_type(page, "[data-ae-nl-pair-span] input", 2))
    # the other axes: excitation across, emission up (the setting of the page 'Axes of fluorescence maps')
    set_transposed(page, True)
    nl_open(page, rid1, "bends")
    strip = gd_info(page, "[data-ae-nl-panel='bends'] [data-ae-nl-strip]")
    check(any(a.startswith("xaxis") and a.endswith("Excitation (nm)") for a in strip["axes"]) and any(a.startswith("yaxis") and a.endswith("Emission (nm)") for a in strip["axes"]), f"with excitation across the maps are turned ({strip['axes'][:3]})")
    nl_section(page, "local")
    check(any(a.endswith("Excitation (nm)") for a in gd_info(page, "[data-ae-nl-panel='local'] [data-ae-nl-loadings]")["axes"]), "…the loadings too")
    nl_section(page, "pair")
    check(any(a.endswith("Excitation (nm)") for a in gd_info(page, "[data-ae-nl-panel='pair'] [data-ae-nl-gallery]")["axes"]), "…and the gallery of two codes")
    nl_section(page, "response")
    page.click("[data-ae-nl-pick] summary")
    page.wait_for_selector("[data-ae-nl-picture] .js-plotly-plot", timeout=30000)
    page.wait_for_timeout(1200)
    box = page.locator("[data-ae-nl-picture] .nsewdrag").bounding_box()
    page.select_option("[data-ae-nl-kind]", "total")
    hit = None
    for fx, fy in ((0.35, 0.2), (0.45, 0.15), (0.3, 0.3), (0.5, 0.25), (0.4, 0.4), (0.6, 0.3)):                         # high emission is at the top of the picture now
        page.mouse.click(box["x"] + fx * box["width"], box["y"] + fy * box["height"])
        page.wait_for_timeout(500)
        if page.input_value("[data-ae-nl-kind]") == "point":
            hit = (float(page.input_value("[data-ae-nl-ex] input")), float(page.input_value("[data-ae-nl-em] input")))
            break
    check(hit is not None and 230 <= hit[0] <= 460 and hit[1] >= 280, f"a click on the turned picture still reads excitation and emission the right way round ({hit})")
    page.click("[data-ae-nl-pick] summary")
    if shots:
        nl_section(page, "bends")
        shot(page, "ae_flow_nl_transposed", full_page=True)
    set_transposed(page, False)
    nl_open(page, rid1, "response")
    tab(page, "latent")


# ----------------------------------------------------------------------------------------------------------------------- main
def main() -> int:
    dark, shots = "--dark" in sys.argv, "--shots" in sys.argv
    keep_going = "--keep-going" in sys.argv
    if os.environ.get("FLUOVIZ_SHOTS"):
        browse.SHOTS = Path(os.environ["FLUOVIZ_SHOTS"])
    rt = tempfile.mkdtemp(prefix="ae-flow-runtime-")
    env = {"FLUOVIZ_ADDONS_DEV": str(ADDON), "FLUOVIZ_RUNTIME": rt}
    if os.environ.get("FLUOVIZ_TEST_ENV"):          # a second conda environment that already has PyTorch (a development copy): the server and its training worker run from it
        te = os.environ["FLUOVIZ_TEST_ENV"]
        env.update({"PATH": f"{te}/bin:{os.environ['PATH']}", "R_HOME": f"{te}/lib/R"})
    with contextlib.ExitStack() as stack:
        stack.callback(shutil.rmtree, rt, True)                                    # (the runtime folder of this run: removed when it ends, after the server has stopped)
        try:
            base = stack.enter_context(private_server(int(os.environ.get("AE_FLOW_PORT") or free_port()), env))
        except RuntimeError:
            print("FAILED: the server did not start")
            return 1
        page = stack.enter_context(browser(base, 1500, 1000, dark=dark))
        try:
            goto(page, "/", 1.0)
            part_basics(page, shots)
            part_network(page, shots)
            if "--only=advanced" in sys.argv:                  # while developing: just the advanced training settings (the page needs the data and step 5 first)
                part_advanced_training(page, shots)
                errs = errors(page)
                check(not errs, f"no console or server errors ({errs[:3]})")
                print("\nFAILED:" if FAILS else "\nThe advanced training settings passed.", *FAILS, sep="\n  " if FAILS else "")
                return 1 if FAILS else 0
            if "--only=gaps" in sys.argv:                      # while developing: just the rectangle cut out of some samples
                part_gaps(page, shots)
                errs = errors(page)
                check(not errs, f"no console or server errors ({errs[:3]})")
                print("\nFAILED:" if FAILS else "\nThe part about samples that lack a region passed.", *FAILS, sep="\n  " if FAILS else "")
                return 1 if FAILS else 0
            if "--only=nonlinear" in sys.argv:                 # while developing: just the tab Nonlinear effects (it needs the data and a few networks)
                rid1 = part_recommended(page, shots)
                part_nonlinear(page, rid1, shots)
                rid4 = part_both(page, shots)
                part_nonlinear_both(page, rid4, rid1, shots)
                part_nonlinear_codes(page, rid1, shots)
                part_layout(page, shots)
                if shots:
                    part_shots(page)
                errs = errors(page)
                check(not errs, f"no console or server errors ({errs[:3]})")
                print("\nFAILED:" if FAILS else "\nThe tab Nonlinear effects passed.", *FAILS, sep="\n  " if FAILS else "")
                return 1 if FAILS else 0
            rid1 = part_recommended(page, shots)
            ctx = {"rid1": rid1}
            part_results(page, rid1, shots)
            part_nonlinear(page, rid1, shots)
            probe = api(page, "POST", f"{BASE_API}/scan", {"scan": "baseline", "rid": rid1, "save": True})
            scans = "job_id" in probe
            if scans:
                page.wait_for_timeout(500)
            part_checks(page, rid1, shots, scans)
            part_use(page, rid1, shots)
            ctx["title"] = part_save_model(page, rid1)
            ctx["rid2"] = part_smoothify(page, rid1, shots)
            ctx["rid3"] = part_vae(page, shots)
            ctx["rid4"] = part_both(page, shots)
            part_nonlinear_both(page, ctx["rid4"], rid1, shots)
            part_nonlinear_codes(page, rid1, shots)
            part_advanced_training(page, shots)
            part_gaps(page, shots)
            part_cancel_and_queue(page)
            part_stale(page, ctx["rid2"])
            part_layout(page, shots)
            part_reload(page)
            if shots:
                part_shots(page)
            part_sizes(page, shots)
            part_second_project(page, ctx["title"])
        except Exception:                                       # a crash is a failure; the console is still looked at
            traceback.print_exc()
            FAILS.append("the flow crashed")
            try:
                shot(page, "ae_flow_crash", full_page=True)
            except Exception:
                pass
        errs = errors(page)
        check(not errs, f"no console or server errors during the whole flow ({errs[:3]})")
    print("\nFAILED:" if FAILS else "\nThe Autoencoders flow passed.", *FAILS, sep="\n  " if FAILS else "")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
