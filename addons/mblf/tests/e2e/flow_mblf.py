"""End-to-end flow of the Multimodal latent factors add-on in a real browser (Chrome through the DevTools protocol, ``tests/e2e/cdp.py``), with assertions.

    FLUOVIZ_MBLF_ENGINE=<env with cmdstan> <FluoViz python> tests/e2e/flow_mblf.py [--dark] [--shots DIR] [--width 1300]

It starts FluoViz with the add-on loaded in development mode in this process, on a free port, with the small synthetic project (EEMs, absorbance spectra, a DOC value with gaps, a
site), walks the eight steps (choosing a value and a covariate in step 1), fits with the quick setting, opens every tab of the results, predicts a sample, runs the held-out comparison,
opens every tab of the checks, the Help guide, a 600 px window (no sideways scroll), and finally looks at the console.  Exit status 0 = every assertion passed.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ADDON = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ADDON / "tests"), str(ADDON), str(ADDON.parent / "common")]

from cdp import Chrome, free_port        # noqa: E402

FAILS: list[str] = []
SHOTS: Path | None = None
LONG = 600


def check(cond, what):
    if not cond:
        FAILS.append(what)
        print("FAIL:", what, flush=True)
    else:
        print("ok  :", what, flush=True)


def shot(ch, name):
    if SHOTS:
        ch.screenshot(str(SHOTS / f"{name}.png"))


def start(port: int):
    import uvicorn
    import api_support as A
    S, app, active = A.make_app(Path(tempfile.mkdtemp(prefix="mblf-ws-")))
    assert active, "the add-on did not start"
    A.load_project(S)
    S.save_now()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.1)
    return S, server


def step(ch, n):
    ch.click(f'[data-fpca-step="{n}"]')
    ch.wait(f"document.querySelector('[data-fpca-panel=\"{n}\"]')", 30, f"step {n}")


def tab(ch, label):
    ok = ch.eval(f"(() => {{ const b = [...document.querySelectorAll('.tabs button, .tab, .seg button')].find((x) => x.textContent.trim().startsWith({label!r})); if (b) b.click(); return !!b; }})()")
    time.sleep(0.6)
    return ok


def main() -> int:
    global SHOTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--dark", action="store_true")
    ap.add_argument("--shots", default=None)
    ap.add_argument("--width", type=int, default=1300)
    a = ap.parse_args()
    if a.shots:
        SHOTS = Path(a.shots)
        SHOTS.mkdir(parents=True, exist_ok=True)
    port = free_port()
    S, server = start(port)
    url = f"http://127.0.0.1:{port}/#/mblf"
    try:
        with Chrome(width=a.width, dark=a.dark) as ch:
            ch.goto(url)
            ch.wait("document.querySelector('[data-mblf-page]')", 60, "the page")
            check("Multimodal latent factors" in ch.text("h1"), "the page title")
            check(ch.count("[data-mblf-addon-pill]") == 1, "the add-on pill")
            ch.wait("document.querySelector('[data-mblf-what]')", 30, "step 1")
            shot(ch, "01-what")
            # a value and a covariate through the variable pickers (the catalogue keys)
            ch.pick_var("[data-mblf-add-value]", "meta:doc")
            check(ch.last_kept_open, "the picker that adds a value stays open after a choice (keep-open) and counts what was added")
            ch.pick_var("[data-mblf-add-covariate]", "meta:site")
            time.sleep(0.5)
            check(ch.eval("document.querySelector('[data-mblf-add-value]').getAttribute('data-value')") == "", "the pickers that add a variable are empty again")
            check(ch.count("[data-mblf-value]") == 1 and ch.count("[data-mblf-covariate]") == 1, "a value and a covariate are listed in step 1")
            check("covariate" in ch.text("[data-mblf-combo]"), "the sentence of the combination")
            step(ch, 2)
            shot(ch, "02-domain")
            step(ch, 3)
            ch.wait("document.querySelector('[data-mblf-basis-note]')", 30, "the basis note")
            shot(ch, "03-basis")
            ch.eval("""(async () => { const m = await import('/addon-web/mblf/web/components/MblfState.js'); const b = m.fp.spec; for (const n of ['fluorescence', 'absorbance']) { b[n].basis.lmode = 'typed'; b[n].basis.k_max = n === 'fluorescence' ? 12 : 10; b[n].basis.share = 98; } b.fluorescence.basis.typed = [60, 40]; b.absorbance.basis.typed = [40]; b.fluorescence.min_coverage = 0.9; b.absorbance.processed = false; b.mblf.K = 3; b.mblf.preset = 'quick'; return true; })()""", await_promise=True)
            step(ch, 4)
            ch.wait("document.querySelector('[data-mblf-engine][data-state=\"ready\"]')", 60, "the engine is ready")
            shot(ch, "04-model")
            step(ch, 5)
            ch.wait("document.querySelector('[data-mblf-summary] .fp-stat')", 120, "the check of the request")
            check("functions" in ch.text("[data-mblf-summary]"), "the check lists the bases")
            shot(ch, "05-fit")
            ch.wait("!document.querySelector('[data-mblf-run]').disabled", 60, "the fit button is enabled")
            ch.click("[data-mblf-run]")
            ch.wait("document.querySelector('[data-fpca-panel=\"6\"]') && document.querySelector('[data-mblf-overview]')", LONG, "the results after the fit")
            shot(ch, "06-overview")
            check("Convergence" in ch.text("[data-mblf-overview]"), "the convergence verdict")
            for label, sel in (("Factors", "[data-mblf-factors]"), ("What tells what", "[data-mblf-tells]"), ("How the variation splits", "[data-mblf-variance]"),
                               ("Covariates", "[data-mblf-covariates]"), ("Samples", "[data-mblf-samples]")):
                check(tab(ch, label), f"the tab {label}")
                ch.wait(f"document.querySelector('{sel}')", 30, label)
                time.sleep(1.0)
                shot(ch, f"06-{label.split()[0].lower()}")
            tab(ch, "Factors")
            ch.eval("(() => { const b = [...document.querySelectorAll('[data-mblf-stat] button')].find((x) => x.textContent.includes('P(> 0)')); if (b) b.click(); return !!b; })()")
            time.sleep(0.8)
            shot(ch, "06-factors-ppos")
            tab(ch, "What tells what")
            ch.wait("document.querySelector('[data-mblf-matrix] tbody tr')", 30, "the matrix")
            ch.eval("(() => { const r = document.querySelector('[data-mblf-matrix] tbody tr[data-key=\"absorbance\"]'); if (r) r.click(); return !!r; })()")
            time.sleep(2.0)
            shot(ch, "06-tells-absorbance")
            step(ch, 7)
            ch.wait("document.querySelector('[data-mblf-pred-text]')", 120, "a prediction")
            shot(ch, "07-one")
            check(ch.count("[data-mblf-budget]") == 1, "the uncertainty budget")
            ch.eval("(() => { const b = [...document.querySelectorAll('[data-mblf-pred-target] button')].find((x) => x.textContent.includes('Values')); if (b) b.click(); return !!b; })()")
            ch.wait("document.querySelector('[data-mblf-pred-text]') && document.querySelector('[data-mblf-pred-text]').textContent.includes('values')", 60, "a prediction of the values")
            shot(ch, "07-values")
            tab(ch, "Every choice")
            ch.wait("document.querySelector('[data-mblf-compare-run]')", 30, "the compare tab")
            ch.click("[data-mblf-compare-run]")
            ch.wait("document.querySelector('[data-mblf-cmp-table] tbody tr')", LONG, "the comparison table")
            time.sleep(1.0)
            shot(ch, "07-compare")
            check("inside" in ch.text("[data-mblf-cmp-verdict]").lower() or "contain" in ch.text("[data-mblf-cmp-verdict]"), "the comparison verdict")
            step(ch, 8)
            ch.wait("document.querySelector('[data-mblf-convergence]')", 30, "the convergence tab")
            shot(ch, "08-convergence")
            tab(ch, "Basis")
            ch.wait("document.querySelector('[data-mblf-basis]')", 30, "the basis tab")
            shot(ch, "08-basis")
            tab(ch, "Use the model")
            ch.wait("document.querySelector('[data-mblf-methods]')", 30, "the methods paragraph")
            shot(ch, "08-use")
            check(ch.count("[data-mblf-vars] li") >= 4, "the variables of the model")
            ch.goto(f"http://127.0.0.1:{port}/#/help?tab=addon:mblf:mblf&topic=budget")
            ch.wait("document.querySelector('[data-mblf-guide-title]')", 30, "the guide")
            check("four parts" in ch.text("[data-mblf-guide-title]"), "the guide page about the budget")
            shot(ch, "09-guide")
            ch.goto(url)
            ch.wait("document.querySelector('[data-mblf-page]')", 30, "back to the page")
            ch.call("Emulation.setDeviceMetricsOverride", width=600, height=900, deviceScaleFactor=1, mobile=False)
            time.sleep(1.0)
            check(not ch.hscroll(), "no sideways scroll at 600 px")
            shot(ch, "10-narrow")
            errs = [e for e in ch.errors if "favicon" not in e]
            check(not errs, "no console errors" + (f": {errs[:3]}" if errs else ""))
    except Exception:                                                             # noqa: BLE001
        traceback.print_exc()
        FAILS.append("the flow stopped")
    finally:
        server.should_exit = True
        S.shutdown()
    print(f"\n{len(FAILS)} failure(s)" if FAILS else "\nall passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
