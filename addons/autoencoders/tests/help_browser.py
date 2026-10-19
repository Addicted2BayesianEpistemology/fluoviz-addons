"""A browser check of the Help guide "Autoencoders" (run by ``tests/test_help_autoencoders.py`` when Playwright is available; not a test file itself).

    <python-with-playwright> tests/help_browser.py --fluoviz <FluoViz folder> [--shots DIR] [--json]

Starts a PRIVATE server of FluoViz (own workspace, this computer only: ``tests/e2e/browse.py: server``) with this add-on loaded in place (``FLUOVIZ_ADDONS_DEV``), opens every page of the tab
``#/help?tab=addon:autoencoders:autoencoders`` in the light and in the dark theme at 1300 px and at 600 px and checks: the tab is on the Help page, no console error, no ``{{`` / ``undefined`` /
``NaN`` in the text, no horizontal scroll, every picture is drawn inside the page and no text of a picture leaves it, every link of the guide leads to a page that exists.  Screenshots (``--shots``) go to
the given folder.  The result is printed as JSON (``--json``); the exit code is 0 when there are no problems.  Chrome needs a short temporary folder: run it with ``TMPDIR=/tmp``.
"""
from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

ADDON = Path(__file__).resolve().parents[1]
FLUOVIZ = Path(sys.argv[sys.argv.index("--fluoviz") + 1]) if "--fluoviz" in sys.argv else ADDON.parents[2] / "80_fluoViz"
sys.path.insert(0, str(FLUOVIZ / "tests" / "e2e"))
from browse import goto                                                               # noqa: E402
import browse                                                                         # noqa: E402
import os                                                                             # noqa: E402
import shlex                                                                          # noqa: E402
import shutil                                                                         # noqa: E402
import subprocess                                                                     # noqa: E402
import tempfile                                                                       # noqa: E402
import time                                                                           # noqa: E402
import urllib.error                                                                   # noqa: E402
import urllib.request                                                                 # noqa: E402
from contextlib import contextmanager                                                 # noqa: E402


@contextmanager
def browser(base: str, width: int = 1300, height: int = 900, dark: bool = False, launch_timeout: float = 900.0):
    """``browse.browser`` with a patient launch: Chrome can need minutes to start on a busy computer (``browse.browser`` gives up after 3 minutes)."""
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        kw = {"executable_path": browse.CHROME} if browse.CHROME else {}
        b = pw.chromium.launch(headless=True, args=["--no-sandbox", "--disable-gpu"], timeout=launch_timeout * 1000, **kw)
        ctx = b.new_context(viewport={"width": width, "height": height}, color_scheme="dark" if dark else "light")
        ctx.set_default_timeout(120000)
        page = ctx.new_page()
        page.logs = []
        page.base = base
        page.on("console", lambda m: page.logs.append(f"[{m.type}] {m.text}") if m.type in ("error", "warning") else None)
        page.on("pageerror", lambda e: page.logs.append(f"[pageerror] {e}"))
        page.on("response", lambda r: page.logs.append(f"[http {r.status}] {r.url}") if r.status >= 500 else None)
        try:
            yield page
        finally:
            b.close()


@contextmanager
def server(port: int, env: dict[str, str] | None = None, wait: float = 900.0):
    """The private server of ``browse.server`` (this computer only, a temporary workspace, ``env`` exported after ``scripts/env.sh``), with a patient wait: on a busy computer FluoViz needs
    more than the 30 seconds ``browse.server`` gives it.  Uses ``browse.server`` itself when its signature says it can wait."""
    import inspect
    if "wait" in inspect.signature(browse.server).parameters:
        with browse.server(port=port, env=env, wait=wait) as base:
            yield base
        return
    root = Path(browse.ROOT)
    ws = tempfile.mkdtemp(prefix="fluoviz-e2e-")
    exports = "".join(f"export {k}={shlex.quote(str(v))} && " for k, v in (env or {}).items())
    cmd = f'cd "{root}" && . scripts/env.sh && fluoviz_isolate && {exports}exec python -m fluoviz --no-browser --port {port} --local-only --workspace "{ws}"'
    proc = subprocess.Popen(["bash", "-c", cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    try:
        t0 = time.time()
        while True:
            try:
                urllib.request.urlopen(base + "/api/system", timeout=2).read()
                break
            except urllib.error.HTTPError:
                break
            except Exception:
                if proc.poll() is not None or time.time() - t0 > wait:
                    raise RuntimeError("the server did not start")
                time.sleep(0.5)
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(ws, ignore_errors=True)

TAB = "addon:autoencoders:autoencoders"
N_PAGES = 9

TEXT_OUT = """() => {
  const out = [];
  const body = document.querySelector('.guide-body').getBoundingClientRect();
  document.querySelectorAll('.guide-body figure.sheet-fig svg').forEach((svg, k) => {
    const box = svg.getBoundingClientRect();
    if (box.width < 50 || box.height < 50) out.push({ svg: k, problem: 'empty picture', w: box.width, h: box.height });
    if (box.right > body.right + 1.5 || box.left < body.left - 1.5) out.push({ svg: k, problem: 'picture leaves the page', right: Math.round(box.right - body.right) });
    svg.querySelectorAll('text').forEach((t) => {
      const r = t.getBoundingClientRect();
      if (!t.textContent.trim() || r.width === 0) return;
      if (r.right > box.right + 1.5 || r.left < box.left - 1.5 || r.bottom > box.bottom + 1.5 || r.top < box.top - 1.5) out.push({ svg: k, problem: 'text leaves the picture', text: t.textContent.trim().slice(0, 60), right: Math.round(r.right - box.right), left: Math.round(box.left - r.left), bottom: Math.round(r.bottom - box.bottom) });
    });
  });
  return out;
}"""

PAGE_FACTS = """() => {
  const body = document.querySelector('.guide-body');
  const text = body ? body.innerText : '';
  return {
    text_len: text.length,
    title: (document.querySelector('[data-ae-guide-title]') || {}).innerText || '',
    state: (document.querySelector('[data-addon-guide]') || {}).dataset ? document.querySelector('[data-addon-guide]').dataset.addonGuideState : '',
    braces: /\\{\\{|\\}\\}/.test(text), undef: /undefined/.test(text), nan: /\\bNaN\\b/.test(text),
    scroll: document.documentElement.scrollWidth - window.innerWidth,
    svgs: document.querySelectorAll('.guide-body figure.sheet-fig svg').length,
    rects: document.querySelectorAll('.guide-body figure.sheet-fig svg rect').length,
    links: document.querySelectorAll('.guide-body a.g-link').length,
    h2: document.querySelectorAll('.guide-body h2').length,
    ui: document.querySelectorAll('.guide-body i.ui').length,
    tables_wide: [...document.querySelectorAll('.guide-body .table-wrap')].filter((w) => w.scrollWidth > w.clientWidth + 1).length,
  };
}"""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run(shots: Path | None) -> dict:
    problems: list[str] = []
    report: dict = {"pages": {}, "links": 0, "figures": {}}
    with server(port=free_port(), env={"FLUOVIZ_ADDONS_DEV": str(ADDON)}) as base:
        for dark in (False, True):
            with browser(base, width=1300, height=900, dark=dark) as page:
                goto(page, f"/help?tab={TAB}&topic=what", wait=1.5)
                page.wait_for_selector("[data-ae-guide]", timeout=120000)
                tabs = page.evaluate("() => [...document.querySelectorAll('.tabs [role=tab], .tabs button, .tabs a')].map((b) => b.innerText.trim())")
                if not any(t == "Autoencoders" for t in tabs):
                    problems.append(f"the tab 'Autoencoders' is not on the Help page: {tabs}")
                topics = page.evaluate("() => [...document.querySelectorAll('[data-ae-guide] .guide-link')].map((b) => b.dataset.topic)")
                if len(topics) < N_PAGES:
                    problems.append(f"the guide has {len(topics)} pages in its list")
                for tid in topics:
                    for width in (1300, 600):
                        page.set_viewport_size({"width": width, "height": 900})
                        goto(page, f"/help?tab={TAB}&topic={tid}", wait=0.8)
                        page.wait_for_selector("[data-ae-guide-title]", timeout=120000)
                        facts = page.evaluate(PAGE_FACTS)
                        out = page.evaluate(TEXT_OUT)
                        key = f"{tid}-{'dark' if dark else 'light'}-{width}"
                        report["pages"][key] = {**facts, "text_out": out}
                        if facts["state"] != "ready":
                            problems.append(f"{key}: the guide is '{facts['state']}'")
                        if facts["braces"] or facts["undef"] or facts["nan"]:
                            problems.append(f"{key}: template leftovers (braces {facts['braces']}, undefined {facts['undef']}, NaN {facts['nan']})")
                        if facts["scroll"] > 1:
                            problems.append(f"{key}: horizontal scroll of {facts['scroll']} px")
                        if facts["text_len"] < 1800 or facts["h2"] < 3:
                            problems.append(f"{key}: the page is short ({facts['text_len']} characters, {facts['h2']} headings)")
                        if facts["tables_wide"]:
                            problems.append(f"{key}: {facts['tables_wide']} table(s) wider than the page")
                        for o in out:
                            problems.append(f"{key}: picture {o}")
                        report["figures"][key] = facts["svgs"]
                        if shots:
                            shots.mkdir(parents=True, exist_ok=True)
                            page.screenshot(path=str(shots / f"{key}.png"), full_page=True)
                            for fi, fig in enumerate(page.query_selector_all(".guide-body figure.sheet-fig")):
                                fig.screenshot(path=str(shots / f"fig-{key}-{fi + 1}.png"))
                # every link of the guide leads to a page that exists
                if not dark:
                    page.set_viewport_size({"width": 1300, "height": 900})
                    for tid in topics:
                        goto(page, f"/help?tab={TAB}&topic={tid}", wait=0.5)
                        n = page.evaluate("() => document.querySelectorAll('.guide-body a.g-link').length")
                        for i in range(n):
                            goto(page, f"/help?tab={TAB}&topic={tid}", wait=0.4)
                            info = page.evaluate("(i) => { const a = document.querySelectorAll('.guide-body a.g-link')[i]; const p = a.dataset.plink || ''; a.click(); return { plink: p, text: a.innerText }; }", i)
                            page.wait_for_timeout(500)
                            where = page.evaluate("""() => ({ hash: decodeURIComponent(location.hash),
                                ae: !!document.querySelector('[data-ae-guide]'), exp: !!document.querySelector('[data-experimental]'), saved: !!document.querySelector('[data-models-guide]'),
                                title: (document.querySelector('[data-ae-guide-title], [data-experimental-title], [data-models-title]') || {}).innerText || '',
                                h1: (document.querySelector('h1') || {}).innerText || '' })""")
                            report["links"] += 1
                            if info["plink"]:
                                if not where["hash"].startswith("#" + info["plink"]):
                                    problems.append(f"{tid}: the link '{info['text']}' to {info['plink']} leads to {where['hash']}")
                            elif not where["title"] or not (where["ae"] or where["exp"] or where["saved"]):
                                problems.append(f"{tid}: the link '{info['text']}' leads nowhere ({where})")
                # the pager buttons work
                if not dark:
                    goto(page, f"/help?tab={TAB}&topic=what", wait=0.6)
                    page.click(".guide-pager .btn.primary")
                    page.wait_for_timeout(400)
                    now = page.evaluate("() => decodeURIComponent(location.hash)")
                    if "topic=pipeline" not in now:
                        problems.append(f"the Next button of the pager leads to {now}")
                logs = [m for m in page.logs if not m.startswith("[warning]")]
                if logs:
                    problems.append(f"console ({'dark' if dark else 'light'}): {logs[:5]}")
    report["problems"] = problems
    return report


if __name__ == "__main__":
    shots = Path(sys.argv[sys.argv.index("--shots") + 1]) if "--shots" in sys.argv else None
    rep = run(shots)
    if "--json" in sys.argv:
        print(json.dumps(rep))
    else:
        print(json.dumps({k: v for k, v in rep.items() if k != "pages"}, indent=1))
        print(len(rep["pages"]), "page views;", rep["links"], "links followed")
    sys.exit(1 if rep["problems"] else 0)
