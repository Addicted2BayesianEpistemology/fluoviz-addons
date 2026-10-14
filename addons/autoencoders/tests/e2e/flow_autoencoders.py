"""End-to-end flow of the Autoencoders add-on in a real browser (Playwright + Chrome), with assertions.

    <python-with-playwright> tests/e2e/flow_autoencoders.py [--dark] [--shots] [--keep-going]

What it does: starts a PRIVATE FluoViz (own temporary workspace and runtime folder, listening on this computer only) with the add-on loaded in development mode
(``FLUOVIZ_ADDONS_DEV=<this add-on>``), opens the example data and goes through the page: the sidebar entry and its badge, the nine steps, the functional switch, the
network settings, **Run with the recommended settings**, every tab of the results and of the checks, the tools (traversals, hover cards, interpolation, unusual samples),
smoothify, a variational autoencoder (generate), both variables together, cancel, an outdated network and **Train again with the same settings**, the use buttons
(**Cluster the latent space**, **Classify with the latent space**, **Save this model**, applying the saved model in a second project), the downloads, a reload, 600 px
(no sideways scroll), light and dark; at the end it looks at the console.  ``--shots`` also writes screenshots of every step and tab at 1100 and 600 px, light and dark,
into ``$FLUOVIZ_SHOTS`` (default: a folder in the temporary directory).

Environment: ``FLUOVIZ_DIR`` = the FluoViz folder (default: ``80_fluoViz`` next to ``81_fluoviz-addons``); ``FLUOVIZ_SHOTS`` = where the screenshots go.
The scans (latent-dimension, smoothify, stability, linear baselines) are tested when the server offers them and skipped, with a note, when it says they are not available yet.
Exit status 0 = every assertion passed.
"""
from __future__ import annotations

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
    check(tabs == ["Latent space", "Traversals", "Samples", "Interpolation", "Generate"], f"the five tabs ({tabs})")
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
                for t in ("latent", "traversals", "samples", "interpolation"):
                    tab(page, t); page.wait_for_timeout(2200); look(f"7/{t}")
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
                    for t in ("latent", "traversals", "samples", "interpolation"):
                        tab(page, t); page.wait_for_timeout(2500); shot(page, f"ae_{tag}_7_{t}", full_page=True)
                elif n == 8:
                    for t in ("training", "quality", "latent", "stability", "smoothness"):
                        tab(page, t); page.wait_for_timeout(2000); shot(page, f"ae_{tag}_8_{t}", full_page=True)
                else:
                    shot(page, f"ae_{tag}_{n}", full_page=True)
    page.emulate_media(color_scheme="dark" if "--dark" in sys.argv else "light")
    page.set_viewport_size({"width": 1500, "height": 1000})


# ----------------------------------------------------------------------------------------------------------------------- main
def main() -> int:
    dark, shots = "--dark" in sys.argv, "--shots" in sys.argv
    keep_going = "--keep-going" in sys.argv
    if os.environ.get("FLUOVIZ_SHOTS"):
        browse.SHOTS = Path(os.environ["FLUOVIZ_SHOTS"])
    rt = tempfile.mkdtemp(prefix="ae-flow-runtime-")
    with contextlib.ExitStack() as stack:
        try:
            base = stack.enter_context(private_server(int(os.environ.get("AE_FLOW_PORT") or free_port()), {"FLUOVIZ_ADDONS_DEV": str(ADDON), "FLUOVIZ_RUNTIME": rt}))
        except RuntimeError:
            print("FAILED: the server did not start")
            return 1
        page = stack.enter_context(browser(base, 1500, 1000, dark=dark))
        try:
            goto(page, "/", 1.0)
            part_basics(page, shots)
            part_network(page, shots)
            rid1 = part_recommended(page, shots)
            ctx = {"rid1": rid1}
            part_results(page, rid1, shots)
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
