"""A very small Chrome DevTools Protocol client (``websockets`` + Chrome): enough to drive the page of the add-on in a real browser without Playwright.

    with Chrome(width=1300, dark=False) as ch:
        ch.goto("http://127.0.0.1:8765/#/som")
        ch.wait("document.querySelector('[data-som-page]')")
        ch.click("[data-fpca-recommended]")
        ch.screenshot("page.png")
        print(ch.errors)           # console errors and uncaught exceptions since the start

Everything is synchronous (a private event loop runs in a thread).  ``wait(js, timeout)`` polls a JavaScript expression until it is truthy.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import urllib.request


def free_port() -> int:
    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0))
        return sk.getsockname()[1]


class Chrome:
    def __init__(self, width: int = 1300, height: int = 900, dark: bool = False, binary: str | None = None):
        self.width, self.height, self.dark = width, height, dark
        self.binary = binary or os.environ.get("CHROME") or shutil.which("google-chrome") or shutil.which("chromium") or "google-chrome"
        self.errors: list[str] = []
        self.logs: list[str] = []
        self._id = 0
        self._pending: dict = {}

    # ------------------------------------------------------------------ life cycle
    def __enter__(self):
        self.port = free_port()
        self.profile = tempfile.mkdtemp(prefix="som-chrome-", dir="/tmp")
        args = [self.binary, "--headless=new", f"--remote-debugging-port={self.port}", "--no-sandbox", "--disable-gpu", "--no-first-run", "--disable-extensions", f"--user-data-dir={self.profile}",
                f"--window-size={self.width},{self.height}", "about:blank"]
        env = {**os.environ, "TMPDIR": "/tmp"}                      # (Chrome needs a short temporary folder)
        self.proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
        ws = None
        for _ in range(200):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{self.port}/json/list", timeout=2))
                pages = [t for t in tabs if t.get("type") == "page"]
                if pages:
                    ws = pages[0]["webSocketDebuggerUrl"]
                    break
            except Exception:
                time.sleep(0.2)
        if not ws:
            self.close()
            raise RuntimeError("Chrome did not start")
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self._thread.start()
        self._conn = self._run(self._connect(ws))
        self.call("Page.enable"); self.call("Runtime.enable"); self.call("Log.enable")
        self.call("Emulation.setDeviceMetricsOverride", width=self.width, height=self.height, deviceScaleFactor=1, mobile=False)
        self.call("Emulation.setEmulatedMedia", features=[{"name": "prefers-color-scheme", "value": "dark" if self.dark else "light"}])
        return self

    def __exit__(self, *a):
        self.close()

    def close(self):
        try:
            if hasattr(self, "loop"):
                self.loop.call_soon_threadsafe(self.loop.stop)
        except Exception:
            pass
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:
            pass
        shutil.rmtree(getattr(self, "profile", ""), ignore_errors=True)

    # ------------------------------------------------------------------ plumbing
    async def _connect(self, ws):
        import websockets
        conn = await websockets.connect(ws, max_size=64 * 2 ** 20)
        asyncio.ensure_future(self._reader(conn))
        return conn

    async def _reader(self, conn):
        async for raw in conn:
            m = json.loads(raw)
            if "id" in m and m["id"] in self._pending:
                fut = self._pending.pop(m["id"])
                if not fut.done():
                    fut.set_result(m)
            elif m.get("method") == "Runtime.consoleAPICalled":
                p = m["params"]
                text = " ".join(str(a.get("value", a.get("description", ""))) for a in p.get("args", []))
                (self.errors if p.get("type") in ("error", "assert") else self.logs).append(f"[{p.get('type')}] {text}")
            elif m.get("method") == "Runtime.exceptionThrown":
                d = m["params"]["exceptionDetails"]
                self.errors.append("[exception] " + (d.get("exception", {}).get("description") or d.get("text", "")))
            elif m.get("method") == "Log.entryAdded":
                e = m["params"]["entry"]
                if e.get("level") == "error":
                    self.errors.append(f"[log] {e.get('text')} {e.get('url', '')}")

    def _run(self, coro, timeout=120):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def call(self, method: str, **params):
        self._id += 1
        mid = self._id

        async def go():
            fut = self.loop.create_future()
            self._pending[mid] = fut
            await self._conn.send(json.dumps({"id": mid, "method": method, "params": params}))
            return await asyncio.wait_for(fut, 120)

        r = self._run(go())
        if "error" in r:
            raise RuntimeError(f"{method}: {r['error']}")
        return r.get("result", {})

    # ------------------------------------------------------------------ page helpers
    def eval(self, js: str, await_promise: bool = False):
        r = self.call("Runtime.evaluate", expression=js, returnByValue=True, awaitPromise=await_promise)
        if "exceptionDetails" in r:
            raise RuntimeError("JavaScript error: " + (r["exceptionDetails"].get("exception", {}).get("description") or str(r["exceptionDetails"])))
        return r["result"].get("value")

    def goto(self, url: str):
        self.call("Page.navigate", url=url)
        time.sleep(0.5)

    def wait(self, js: str, timeout: float = 60.0, what: str = ""):
        t0 = time.time()
        last = None
        while time.time() - t0 < timeout:
            try:
                last = self.eval(f"!!({js})")
                if last:
                    return True
            except RuntimeError as exc:
                last = str(exc)
            time.sleep(0.2)
        raise TimeoutError(f"timeout waiting for {what or js}: {last}")

    def click(self, selector: str, index: int = 0):
        ok = self.eval(f"(()=>{{const e=document.querySelectorAll({json.dumps(selector)})[{index}]; if(!e) return false; e.scrollIntoView({{block:'center'}}); e.click(); return true;}})()")
        if not ok:
            raise RuntimeError(f"nothing to click: {selector}")

    def click_text(self, selector: str, text: str):
        ok = self.eval(f"(()=>{{const e=[...document.querySelectorAll({json.dumps(selector)})].find(x=>x.textContent.trim().startsWith({json.dumps(text)})); if(!e) return false; e.scrollIntoView({{block:'center'}}); e.click(); return true;}})()")
        if not ok:
            raise RuntimeError(f"nothing to click: {selector} with text {text}")

    def count(self, selector: str) -> int:
        return int(self.eval(f"document.querySelectorAll({json.dumps(selector)}).length"))

    def text(self, selector: str = "body") -> str:
        return self.eval(f"(document.querySelector({json.dumps(selector)})||{{}}).innerText||''") or ""

    def screenshot(self, path: str, full: bool = True):
        if full:
            h = self.eval("Math.max(document.documentElement.scrollHeight, document.body.scrollHeight)")
            self.call("Emulation.setDeviceMetricsOverride", width=self.width, height=min(int(h), 6000), deviceScaleFactor=1, mobile=False)
        r = self.call("Page.captureScreenshot", format="png")
        with open(path, "wb") as f:
            f.write(base64.b64decode(r["data"]))
        if full:
            self.call("Emulation.setDeviceMetricsOverride", width=self.width, height=self.height, deviceScaleFactor=1, mobile=False)

    def hscroll(self) -> bool:
        """True when the page scrolls sideways."""
        return bool(self.eval("document.documentElement.scrollWidth > document.documentElement.clientWidth + 2"))


def _set_value(self, selector: str, value, event: str = "change", index: int = 0):
    """Set the value of a form control (select / input) the way a person would and tell Vue about it."""
    ok = self.eval(f"(()=>{{const e=document.querySelectorAll({json.dumps(selector)})[{index}]; if(!e) return false; e.value={json.dumps(value)}; e.dispatchEvent(new Event('input',{{bubbles:true}})); e.dispatchEvent(new Event({json.dumps(event)},{{bubbles:true}})); return true;}})()")
    if not ok:
        raise RuntimeError(f"no such control: {selector}")


Chrome.set_value = _set_value
