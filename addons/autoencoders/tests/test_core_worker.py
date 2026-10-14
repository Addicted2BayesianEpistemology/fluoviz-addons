"""The worker process: a fresh ``spawn`` process per job (as on Windows and macOS), progress and results coming back, errors in plain English, cancellation that leaves nothing behind,
and PyTorch never imported by the process that waits."""
import multiprocessing as mp
import os
import threading
import time

import numpy as np
import pytest

from core_support import planted
from fluoviz.analysis.common import Cancelled, Progress
from fluoviz_addon_autoencoders.core import data as D, train as T, worker as W
from fluoviz_addon_autoencoders.core.spec import AutoencoderError, Config


def small_data(n=40):
    return D.TrainData.from_array(planted(n=n, P=20, k=2)[0])


def test_a_job_runs_in_another_process_and_reports_progress(tmp_path):
    seen = []
    out = W.run_in_worker("selftest", {"seconds": 0.4, "steps": 4}, progress=lambda f, m: seen.append((f, m)), workdir=tmp_path)
    assert out["pid"] != os.getpid() and out["torch_loaded"] is False and seen[-1][0] == pytest.approx(1.0) and seen[0][1] == "step 1"
    assert list(tmp_path.iterdir()) == []                                                               # the result file is removed once read
    big = W.run_in_worker("selftest", {"seconds": 0.1, "big": 5_000_000}, workdir=tmp_path)
    assert len(big["blob"]) == 5_000_000                                                                # a big result does not block the pipe


def test_errors_come_back_in_plain_english_and_unknown_jobs_are_refused(tmp_path):
    with pytest.raises(AutoencoderError, match="latent dimension is too large"):
        W.run_in_worker("selftest", {"fail": "user", "seconds": 0.05}, workdir=tmp_path)
    with pytest.raises(W.WorkerError, match="The training failed: ZeroDivisionError: boom") as e:
        W.run_in_worker("selftest", {"fail": "bug", "seconds": 0.05}, workdir=tmp_path)
    assert "ZeroDivisionError" in e.value.detail and "Traceback" in e.value.detail
    with pytest.raises(AutoencoderError, match="Unknown job"):
        W.run_in_worker("nothing", {})
    with pytest.raises(AutoencoderError, match="Unknown job"):
        W.run_in_worker("os:system", {"command": "echo"})                                                # only functions of this add-on can be started
    assert not [p for p in tmp_path.iterdir()]


def test_a_process_that_dies_or_takes_too_long_is_reported(tmp_path):
    with pytest.raises(W.WorkerError, match=r"stopped unexpectedly \(exit code 9\)"):
        W.run_in_worker("selftest", {"die": True}, workdir=tmp_path)
    with pytest.raises(W.WorkerError, match="longer than 0.3 seconds"):
        W.run_in_worker("selftest", {"seconds": 20.0, "steps": 1}, workdir=tmp_path, timeout=0.3)
    time.sleep(0.3)
    assert not [c for c in mp.active_children() if c.is_alive()] and not list(tmp_path.iterdir())


def test_torch_missing_is_said_in_words(monkeypatch):
    monkeypatch.setattr(W, "torch_available", lambda: False)
    with pytest.raises(AutoencoderError, match="PyTorch is not installed: open Settings"):
        W.run_in_worker("fit", {})


def test_training_in_the_worker_gives_the_same_network_as_in_process_and_writes_the_history(tmp_path):
    pytest.importorskip("torch")
    td = small_data()
    cfg = Config(latent_dim=2, epochs=60, n_restarts=2, seed=5)
    hist = tmp_path / "hist.jsonl"
    prog = []
    res = W.run_in_worker("fit", {"config": cfg, "data": td}, progress=lambda f, m: prog.append((f, m)), history_path=hist, workdir=tmp_path)
    here = T.fit(cfg, td)
    assert all(np.allclose(res.arrays[k], here.arrays[k], rtol=1e-5, atol=1e-6) for k in res.arrays) and res.best_epoch == here.best_epoch
    recs, _ = T.read_history(hist)
    assert recs[0]["event"] == "start" and recs[-1]["event"] == "done" and prog[-1] == (1.0, "Done") and any("start 2 of 2" in m for _, m in prog)
    assert [f for f, _ in prog] == sorted(f for f, _ in prog)


def test_cancelling_stops_the_process_and_saves_nothing(tmp_path):
    pytest.importorskip("torch")
    td = small_data()
    calls = []
    cancel = threading.Event()

    def progress(f, m):
        calls.append(f)
        if len(calls) >= 3:
            cancel.set()

    prog = Progress(progress, cancel.is_set)
    hist = tmp_path / "hist.jsonl"
    t0 = time.time()
    with pytest.raises(Cancelled):
        W.run_in_worker("fit", {"config": Config(latent_dim=2, epochs=600, patience=600, n_restarts=3), "data": td}, progress=prog, history_path=hist, workdir=tmp_path)
    assert time.time() - t0 < 60
    time.sleep(0.3)
    assert not [c for c in mp.active_children() if c.is_alive()]                                         # no process is left running
    assert not [p for p in tmp_path.iterdir() if p.suffix in (".pkl", ".part")]                          # and no half-written result
    recs, _ = T.read_history(hist)
    assert recs[-1]["event"] == "cancelled" and not any(r["event"] == "done" for r in recs)


def test_the_process_that_waits_never_imports_torch():
    import subprocess
    import sys
    import pathlib
    import fluoviz
    root = pathlib.Path(__file__).resolve().parents[1]
    code = ("import sys\nfrom fluoviz_addon_autoencoders.core import worker as W\n"
            "out = W.run_in_worker('selftest', {'seconds': 0.1})\n"
            "assert 'torch' not in sys.modules and out['torch_loaded'] is False\nprint('ok')\n")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(root), str(pathlib.Path(fluoviz.__file__).resolve().parents[1])]))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-1500:]
