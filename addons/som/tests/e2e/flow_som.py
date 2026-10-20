"""End-to-end flow of the Self-organising map add-on in a real browser (Chrome through the DevTools protocol, ``tests/e2e/cdp.py``), with assertions.

    <FluoViz python> tests/e2e/flow_som.py [--dark] [--shots DIR] [--only=NAME] [--keep-going]

What it does: starts FluoViz with the add-on loaded in development mode (``FLUOVIZ_ADDONS_DEV``) in this process, on a free port, with the small synthetic project already loaded and saved, and goes
through the page: the sidebar entry and its badge, the nine steps, the weights of two variables, the four distances and the three sizes of step 5, **Run with the recommended settings**, every tab
of the results (clicking a unit, the prototypes, adding a plane, grouping again, trajectories, a sample, the unusual samples, the projection), every tab of the checks (with a stability scan),
the buttons of step 9 and the methods paragraph, both variables together, the pixel version, the standardised distance, a map of other variables, a reload, 600 px (no sideways scroll)
and the dark theme; at the end it looks at the console.  ``--shots`` writes a screenshot of every step and tab.  Exit status 0 = every assertion passed.

Run it with the FluoViz environment and FluoViz on ``PYTHONPATH`` (``scripts/env.sh``; ``PYTHONPATH=<FluoViz folder>``): ``Chrome`` needs a short TMPDIR, which ``cdp.py`` sets for it.
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ADDON = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ADDON / "tests"), str(ADDON)]

import serve_for_browser as SB          # noqa: E402
from cdp import Chrome, free_port        # noqa: E402

FAILS: list[str] = []
SHOTS: Path | None = None
LONG = 180


def check(cond, msg: str) -> bool:
    if not cond:
        FAILS.append(msg)
        print("FAIL:", msg, flush=True)
    return bool(cond)


def shot(ch: Chrome, name: str) -> None:
    if SHOTS:
        SHOTS.mkdir(parents=True, exist_ok=True)
        ch.screenshot(str(SHOTS / f"{name}_{ch.width}{'_dark' if ch.dark else ''}.png"))


def step(ch: Chrome, n: int) -> None:
    ch.click(f'[data-fpca-step="{n}"]')
    ch.wait(f'document.querySelector(\'[data-fpca-panel="{n}"]\')', 20, f"step {n}")
    time.sleep(0.4)


def tab(ch: Chrome, group: str, label: str, wait_js: str | None = None) -> None:
    ch.click_text(f"[data-som-tabs{group}] .tab", label)
    if wait_js:
        ch.wait(wait_js, 40, f"tab {label}")
    time.sleep(0.8)


def run_recommended(ch: Chrome) -> None:
    step(ch, 1)
    ch.click("[data-fpca-recommended]")
    ch.wait("document.querySelector('[data-som-cards]')", LONG, "the results")
    time.sleep(1.5)


def compute_here(ch: Chrome) -> None:
    """From step 6: compute with the settings as they are and wait for the results."""
    step(ch, 6)
    ch.wait("document.querySelector('[data-som-run]') && !document.querySelector('[data-som-run]').disabled", 60, "the Compute button")
    ch.click("[data-som-run]")
    ch.wait("document.querySelector('[data-som-cards]')", LONG, "the results")
    time.sleep(1.2)


def part_page(ch: Chrome) -> None:
    ch.wait("document.querySelector('[data-som-page]')", 40, "the page")
    check(ch.count("[data-fpca-step]") == 9, "nine steps")
    check("ADD-ON" in ch.text(".sb-item.on, nav a.active, [aria-current='page']").upper() or "ADD-ON" in ch.text("nav").upper() or ch.count("[data-som-addon-pill]") == 1, "the add-on badge")
    labels = ch.eval("[...document.querySelectorAll('[data-fpca-step] .fp-lbl')].map(x=>x.textContent.trim())")
    check(labels == ["1 · What to map", "2 · Domain", "3 · Representation", "4 · Smoothing", "5 · Map and distance", "6 · Compute", "7 · Results", "8 · Checks", "9 · Use"], f"the steps: {labels}")
    check(ch.count("[data-fpca-what]") == 3, "three cards in step 1")
    check(ch.count("[data-som-other-switch]") == 1, "the switch for other variables")
    shot(ch, "step1")
    ch.click('[data-fpca-what="both"]')
    time.sleep(0.4)
    check(ch.count("[data-fpca-weights]") == 1, "the weights of two variables are offered")
    ch.click('[data-fpca-what="fluorescence"]')
    for n in (2, 3, 4):
        step(ch, n)
        if n == 3:
            check(ch.count("[data-som-functional-seg]") == 1, "the functional switch in step 3")
        shot(ch, f"step{n}")


def part_step5(ch: Chrome) -> None:
    step(ch, 5)
    check(ch.count("[data-som-metric]") == 4 and ch.count("[data-som-size]") == 4, "four distances, four sizes")
    ch.wait("document.querySelector('[data-som-size-sentence]')", 60, "the map in numbers")
    ch.click('[data-som-metric="mahalanobis"]')
    time.sleep(0.5)
    check(ch.count("[data-som-mahalanobis]") == 1, "the options of the standardised distance")
    ch.click('[data-som-metric="l2"]')
    ch.click('[data-som-size="compact"]')
    ch.wait("document.querySelector('[data-som-size-sentence]') && /units/.test(document.querySelector('[data-som-size-sentence]').textContent)", 60)
    ch.click('[data-som-size="standard"]')
    shot(ch, "step5")
    ch.click("[data-som-size-scan]")
    ch.wait("document.querySelector('[data-som-use-size]')", LONG, "the size scan")
    check(True, "size scan")
    shot(ch, "step5_scan")
    ch.click("[data-som-metric-scan]")
    ch.wait("document.querySelector('[data-som-metric-table]')", LONG, "the metric comparison")
    check(ch.count("[data-som-metric-table] tbody tr") >= 4, "the metric table")


def part_results(ch: Chrome) -> None:
    run_recommended(ch)
    check(ch.count("[data-som-cards] .fp-stat") == 4, "four number cards")
    check(ch.count("[data-som-grid]") >= 1, "the map is drawn")
    shot(ch, "step7_map")
    # a click on a unit opens its card
    ch.eval("""(()=>{const g=document.querySelector('[data-som-grid] .js-plotly-plot'); const pts=g.data.find(t=>t.name==='Units'); const ev={points:[{data:pts, curveNumber:0, pointNumber:3, customdata:3}]}; g.emit('plotly_click', ev); return true;})()""")
    ch.wait("document.querySelector('[data-som-unit-card]')", 30, "the unit card")
    check(True, "unit card")
    tab(ch, "7", "Prototypes", "document.querySelector('[data-som-protos]')")
    n_tiles = ch.count("[data-som-protos] .som-proto")
    units = ch.eval("document.querySelector('[data-som-protos]').getAttribute('data-units')")
    check(str(n_tiles) == str(units), f"one tile per unit ({n_tiles} vs {units})")
    wsizes = ch.eval("[...new Set([...document.querySelectorAll('[data-som-protos] .som-proto canvas')].map(c=>Math.round(c.getBoundingClientRect().width)))]")
    check(len(wsizes) == 1, f"every prototype picture has the same size: {wsizes}")
    ch.click("[data-som-protos] .som-proto")
    ch.wait("document.querySelector('[data-som-unit-card]')", 30)
    ch.click_text("[data-som-prototypes-tab] .seg button", "The mean of the samples")
    ch.wait("document.querySelector('[data-som-protos]')", 30)
    time.sleep(1.5)
    shot(ch, "tab_prototypes")
    tab(ch, "7", "Planes", "document.querySelector('[data-som-planes-tab]')")
    ch.set_value("[data-som-plane-index]", "total_fluorescence")
    ch.click("[data-som-add-plane]")
    ch.wait("document.querySelectorAll('[data-som-planes] .chart-card').length >= 3", 30, "the added plane")
    ch.set_value("[data-som-plane-kind]", "point")
    time.sleep(0.4)
    ch.click("[data-som-add-plane]")
    ch.wait("document.querySelectorAll('[data-som-planes] .chart-card').length >= 4", 30, "a pixel plane")
    shot(ch, "tab_planes")
    tab(ch, "7", "Groups", "document.querySelector('[data-som-groups-tab]')")
    ch.wait("document.querySelector('[data-som-group-images]')", 30, "the group pictures")
    ch.set_value("[data-som-group-method]", "kmeans")
    ch.click_text("[data-som-groups-tab] .seg button", "I choose")
    time.sleep(0.4)
    ch.set_value("[data-som-group-k] input", "3", "change")
    ch.click("[data-som-cluster-go]")
    ch.wait("document.querySelectorAll('[data-som-groups-tab] .som-groups-grid tbody tr').length === 3", 60, "three groups")
    shot(ch, "tab_groups")
    tab(ch, "7", "Trajectories", "document.querySelector('[data-som-trajectories-tab]')")
    ch.wait("document.querySelector('[data-som-traj-order]')", 20)
    ch.set_value("[data-som-traj-order]", "meta:day")
    ch.set_value("[data-som-traj-group]", "meta:site")
    ch.click("[data-som-traj-go]")
    ch.wait("document.querySelector('[data-som-trajectories-tab] .js-plotly-plot') && document.querySelector('[data-som-trajectories-tab] .js-plotly-plot').data.some(t=>t.mode==='lines+markers')", 40, "the paths")
    shot(ch, "tab_trajectories")
    tab(ch, "7", "Samples", "document.querySelector('[data-som-samples-tab]')")
    ch.click("[data-som-sample-row]")
    ch.wait("document.querySelector('[data-som-sample-text]')", 40, "a sample")
    check("falls on unit" in ch.text("[data-som-sample-card]"), "the sample text")
    ch.click("[data-som-anom-go]")
    ch.wait("document.querySelector('[data-som-anomalies] .chart-card')", 40, "unusual samples")
    shot(ch, "tab_samples")
    tab(ch, "7", "Projection", "document.querySelector('[data-som-projection-words]')")
    shot(ch, "tab_projection")


def part_checks(ch: Chrome) -> None:
    step(ch, 8)
    ch.wait("document.querySelector('[data-som-quality]')", 30)
    check(ch.count("[data-som-quality] .callout") >= 3, "verdicts in the quality tab")
    shot(ch, "step8_quality")
    tab(ch, "8", "Held-out error", "document.querySelector('[data-som-heldout] .js-plotly-plot')")
    shot(ch, "step8_heldout")
    tab(ch, "8", "Stability", "document.querySelector('[data-som-stability]')")
    ch.set_value("[data-som-stability] input", "3", "change", 0)
    ch.click("[data-som-run-stability]")
    ch.wait("document.querySelector('[data-som-v-stability]')", LONG, "the stability")
    shot(ch, "step8_stability")
    tab(ch, "8", "Training", "document.querySelector('[data-som-training] .js-plotly-plot')")
    shot(ch, "step8_training")
    tab(ch, "8", "Map size", "document.querySelector('[data-som-size-tab]')")
    tab(ch, "8", "Distances", "document.querySelector('[data-som-distances]')")


def part_use(ch: Chrome) -> None:
    step(ch, 9)
    ch.wait("document.querySelector('[data-som-methods]')", 40, "the methods paragraph")
    check("Kohonen" in ch.text("[data-som-methods]") or "self-organising map" in ch.text("[data-som-methods]").lower(), "the methods text")
    names = ch.eval("[...document.querySelectorAll('[data-som-vars] li')].map(x=>x.getAttribute('data-name'))")
    check({"som_x", "som_y", "som_unit", "som_cluster", "status"} <= set(names), f"the published variables: {names}")
    check(ch.count("[data-som-download]") == 5, "five downloads")
    check(ch.count("[data-som-to-cluster]") == 1 and ch.count("[data-som-copy]") == 1, "the buttons of step 9")
    shot(ch, "step9")
    ch.click("[data-som-to-cluster]")
    ch.wait("location.hash.indexOf('/cluster') >= 0 && location.hash.indexOf('som_x') >= 0", 20, "the clustering page")
    ch.goto(ch.url_som)
    ch.wait("document.querySelector('[data-som-page]')", 40)
    time.sleep(1.5)


def part_both_pixels_other(ch: Chrome) -> None:
    # both variables, functional
    step(ch, 1)
    ch.click('[data-fpca-what="both"]')
    time.sleep(0.3)
    compute_here(ch)
    tab(ch, "7", "Prototypes", "document.querySelector('[data-som-protos]')")
    check(ch.count("[data-som-protos-curves]") == 1, "a grid of absorbance curves next to the EEMs")
    shot(ch, "both_prototypes")
    tab(ch, "7", "Samples", "document.querySelector('[data-som-samples-tab]')")
    ch.click("[data-som-sample-row]")
    ch.wait("document.querySelector('[data-som-sample-text]')", 40)
    check(ch.count("[data-som-sample-card] .js-plotly-plot") >= 1, "the absorbance of the sample against its prototype")
    # pixels, standardised distance
    step(ch, 3)
    ch.click_text("[data-som-functional-seg] button", "On the pixels")
    time.sleep(0.5)
    step(ch, 5)
    ch.click('[data-som-metric="standardize"]')
    compute_here(ch)
    check(ch.count("[data-som-cards] .fp-stat") == 4, "pixels + standardised distance")
    shot(ch, "pixels_standardized")
    # the standardised (Mahalanobis) distance on the functional version
    step(ch, 3)
    ch.click_text("[data-som-functional-seg] button", "On the coefficients")
    time.sleep(0.5)
    step(ch, 5)
    ch.click('[data-som-metric="mahalanobis"]')
    compute_here(ch)
    check(ch.count("[data-som-cards] .fp-stat") == 4, "functional + Mahalanobis")
    # other variables
    step(ch, 1)
    ch.click("[data-som-other-switch] input" if False else "[data-som-other-switch]")
    time.sleep(0.5)
    ch.eval("(()=>{const sel=document.querySelector('[data-som-other] select'); const opt=[...sel.options].find(o=>o.value.indexOf('som_x')>=0); if(!opt) return false; sel.value=opt.value; sel.dispatchEvent(new Event('change',{bubbles:true})); return true;})()")
    ch.eval("(()=>{const sel=document.querySelector('[data-som-other] select'); const opt=[...sel.options].find(o=>o.value==='meta:doc'); if(!opt) return false; sel.value=opt.value; sel.dispatchEvent(new Event('change',{bubbles:true})); return true;})()")
    time.sleep(0.5)
    check(ch.count("[data-som-other] .chip") >= 2, "variables chosen for the map of other variables")
    compute_here(ch)
    tab(ch, "7", "Prototypes", "document.querySelector('[data-som-protos]')")
    shot(ch, "other_variables")
    step(ch, 9)
    ch.wait("document.querySelector('[data-som-nosave]')", 30, "the note that a map of other variables cannot be saved")


def part_layout(ch: Chrome) -> None:
    for w in (600,):
        ch.call("Emulation.setDeviceMetricsOverride", width=w, height=900, deviceScaleFactor=1, mobile=False)
        ch.width = w
        for n in (1, 5, 7, 8, 9):
            step(ch, n)
            time.sleep(0.6)
            check(not ch.hscroll(), f"no sideways scroll at {w} px in step {n}")
        shot(ch, "narrow_step9")
        ch.call("Emulation.setDeviceMetricsOverride", width=1300, height=900, deviceScaleFactor=1, mobile=False)
        ch.width = 1300


def part_help(ch: Chrome) -> None:
    ids = ["what", "pipeline", "distance", "size", "reading", "groups", "checks", "use", "literature"]
    for w in (1300, 600):
        ch.call("Emulation.setDeviceMetricsOverride", width=w, height=900, deviceScaleFactor=1, mobile=False)
        ch.width = w
        for topic in ids:
            ch.goto(f"{ch.base}/#/help?tab=addon:som:som&topic={topic}")
            ch.wait("document.querySelector('[data-som-guide]')", 30, "the guide")
            ch.wait(f"document.querySelector('.guide-link.on') && document.querySelector('.guide-link.on').getAttribute('data-topic') === '{topic}'", 20)
            time.sleep(0.5)
            txt = ch.text("[data-som-guide] .guide-body")
            check(len(txt) > 400, f"the guide page {topic} has text")
            for bad in ("{{", "undefined", "NaN"):
                check(bad not in txt, f"'{bad}' in the guide page {topic}")
            check(not ch.hscroll(), f"no sideways scroll in the guide page {topic} at {w} px")
            if topic in ("what", "distance"):
                check(ch.count("[data-som-guide] svg") >= 1, f"a picture on the guide page {topic}")
                shot(ch, f"help_{topic}")
    ch.call("Emulation.setDeviceMetricsOverride", width=1300, height=900, deviceScaleFactor=1, mobile=False)
    ch.width = 1300
    # the button of the page opens the guide
    ch.goto(ch.url_som)
    ch.wait("document.querySelector('[data-som-howto]')", 30)
    ch.click("[data-som-howto]")
    ch.wait("document.querySelector('[data-som-guide]')", 30, "the guide from the page")


def main() -> int:
    global SHOTS
    argv = sys.argv[1:]
    dark = "--dark" in argv
    only = next((a.split("=", 1)[1] for a in argv if a.startswith("--only=")), None)
    keep = "--keep-going" in argv
    if "--shots" in argv:
        SHOTS = Path(argv[argv.index("--shots") + 1])
    port = free_port()
    S, server = SB.start(port, 36, True)
    parts = [("page", part_page), ("step5", part_step5), ("results", part_results), ("checks", part_checks), ("use", part_use), ("variables", part_both_pixels_other), ("layout", part_layout), ("help", part_help)]
    try:
        with Chrome(1300, 900, dark=dark) as ch:
            ch.base = f"http://127.0.0.1:{port}"
            ch.url_som = f"{ch.base}/#/som"
            ch.goto(ch.url_som)
            for name, fn in parts:
                if only and only != name:
                    if name not in ("results",) or only in ("checks", "use", "variables", "layout"):
                        pass
                if only and only != name and not (only in ("checks", "use", "variables", "layout") and name == "results"):
                    continue
                print("part", name, flush=True)
                try:
                    fn(ch)
                except Exception as exc:                                      # noqa: BLE001
                    FAILS.append(f"part {name}: {exc}")
                    print("FAIL part", name, "-", exc, flush=True)
                    traceback.print_exc()
                    shot(ch, f"fail_{name}")
                    if not keep:
                        break
            errs = [e for e in ch.errors if "favicon" not in e]
            check(not errs, f"console errors: {errs[:6]}")
    finally:
        server.should_exit = True
    print("PASSED" if not FAILS else f"FAILED: {len(FAILS)}")
    for f in FAILS:
        print(" -", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
