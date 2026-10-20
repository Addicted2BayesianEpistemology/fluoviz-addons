"""Start FluoViz with the add-on in development mode on a port, with the small synthetic project already loaded (for the browser flow and for looking at the page by hand).

    python tests/e2e/serve_for_browser.py [--port 8765] [--n 36] [--absorbance]

Run it inside the FluoViz environment (``scripts/run_tests.sh`` shows how: ``. scripts/env.sh && fluoviz_isolate``) with FluoViz on ``PYTHONPATH``.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parent), str(HERE.parents[2] / "common")]


def start(port: int, n: int = 36, absorbance: bool = True):
    """Start the server in a thread; returns ``(state, server)``."""
    import uvicorn
    import api_support as A
    S, app, active = A.make_app(Path(tempfile.mkdtemp(prefix="som-ws-")))
    assert active, "the add-on did not start"
    A.load_project(S, n=n, absorbance=absorbance)
    import numpy as np
    import pandas as pd
    site = np.where(np.arange(n) % 3 == 0, "up", np.where(np.arange(n) % 3 == 1, "mid", "down"))
    S.project.merge_metadata(pd.DataFrame({"site": site, "doc": np.linspace(1.0, 9.0, n), "day": np.arange(n) % 12}, index=[f"S{i:02d}" for i in range(n)]))
    S.save_now()                                   # a project that is already saved (as one opened or imported by a person is), so that the page is not reset by the first save
    cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(cfg)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.1)
    return S, server


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--n", type=int, default=36)
    ap.add_argument("--absorbance", action="store_true")
    a = ap.parse_args()
    S, server = start(a.port, a.n, a.absorbance)
    print(f"http://127.0.0.1:{a.port}/#/som", flush=True)
    while True:
        time.sleep(3600)
