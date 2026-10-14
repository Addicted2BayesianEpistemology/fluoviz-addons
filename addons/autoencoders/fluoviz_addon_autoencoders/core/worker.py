"""Running the heavy jobs (training, scans) in a separate process.

Why: PyTorch takes seconds and about 300 MB to import and must never be loaded by the FluoViz server; a crash or an out-of-memory in a training run must not take the server
down; and *Cancel* must really stop the computation.  So every job runs in a fresh **spawn** process (works on Windows and macOS: nothing is inherited, only top-level picklable
functions are used, PyTorch is imported inside the job).  The parent (this module, called from a FluoViz job thread) only waits, relays progress and cancellation, and loads the result.

    result = run_in_worker("fit", {"config": cfg, "data": prepared.train_data(cfg.smoothify_metric)}, progress=progress, history_path=path)

``progress`` is a ``fluoviz.analysis.common.Progress`` (or any ``progress(fraction, message)``): when it raises (the person pressed Cancel) the process is terminated and nothing is
returned or saved.  The result comes back through a temporary file that is written completely before it is announced, so there is never a half-saved result.  The per-epoch history
is written by the job itself to ``history_path`` (JSON lines, :func:`read_history`).

``JOBS`` names the jobs: ``fit`` (``train.fit``) and the scans of ``diagnostics`` (``latent_scan``, ``smoothify_scan``, ``stability_scan``) and ``selftest``.
"""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import multiprocessing as mp
import os
import pickle
import sys
import tempfile
import time
import traceback

from .spec import AutoencoderError

PKG = "fluoviz_addon_autoencoders.core"
JOBS = {
    "fit": f"{PKG}.train:fit",
    "latent_scan": f"{PKG}.diagnostics:latent_scan",
    "smoothify_scan": f"{PKG}.diagnostics:smoothify_scan",
    "stability_scan": f"{PKG}.diagnostics:stability_scan",
    "selftest": f"{PKG}.worker:selftest_job",
}

try:                                                                    # the exception FluoViz's job runner understands as "cancelled"
    from fluoviz.analysis.common import Cancelled
except Exception:                                                       # noqa: BLE001 - used outside FluoViz (a script): a local stand-in
    class Cancelled(Exception):                                         # type: ignore[no-redef]
        """The job was cancelled."""


class WorkerError(RuntimeError):
    """The training process failed (not something the person can fix by changing a setting); ``detail`` has the technical text."""

    def __init__(self, message: str, detail: str = ""):
        super().__init__(message)
        self.detail = detail


def torch_available() -> bool:
    """Is PyTorch installed?  (Looked up without importing it.)"""
    return importlib.util.find_spec("torch") is not None


# ----------------------------------------------------------------------------- the child
class _ChildProgress:
    """What the job in the worker process sees as ``progress``: it sends messages (at most ~10 a second) to the parent."""

    def __init__(self, conn, lo: float = 0.0, hi: float = 1.0, state: dict | None = None):
        self.conn, self.lo, self.hi = conn, lo, hi
        self._state = state if state is not None else {"t": 0.0}

    def __call__(self, fraction: float, message: str = "") -> None:
        now = time.monotonic()
        f = self.lo + (self.hi - self.lo) * float(min(max(fraction, 0.0), 1.0))
        if now - self._state["t"] >= 0.1 or fraction >= 1.0:
            self._state["t"] = now
            try:
                self.conn.send(("progress", f, message))
            except (OSError, ValueError):
                pass

    def check(self) -> None:                                            # cancellation is done by terminating the process
        return None

    def sub(self, lo: float, hi: float) -> "_ChildProgress":
        span = self.hi - self.lo
        return _ChildProgress(self.conn, self.lo + span * lo, self.lo + span * hi, self._state)


def _resolve(target: str):
    mod, _, fn = target.partition(":")
    if not mod.startswith("fluoviz_addon_autoencoders."):
        raise AutoencoderError("Unknown job.")
    return getattr(importlib.import_module(mod), fn)


def _child_main(conn, target: str, kwargs: dict, history_path, result_path: str) -> None:
    """Entry point of the worker process (top level, so that ``spawn`` can import it)."""
    try:
        fn = _resolve(target)
        params = inspect.signature(fn).parameters
        extra = {}
        if "progress" in params:
            extra["progress"] = _ChildProgress(conn)
        if "history_path" in params and history_path is not None:
            extra["history_path"] = history_path
        out = fn(**kwargs, **extra)
        tmp = result_path + ".part"
        with open(tmp, "wb") as fh:
            pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, result_path)                                    # complete or absent, never half written
        conn.send(("done", result_path))
    except AutoencoderError as exc:                                     # something the person can fix: say it as it is
        conn.send(("user_error", str(exc), ""))
    except BaseException as exc:                                        # noqa: BLE001 - everything is reported to the parent
        try:
            conn.send(("error", f"{type(exc).__name__}: {exc}", traceback.format_exc()))
        except Exception:                                               # noqa: BLE001
            pass
    finally:
        try:
            conn.close()
        except Exception:                                               # noqa: BLE001
            pass


# ----------------------------------------------------------------------------- the parent
def run_in_worker(name: str, kwargs: dict, progress=None, history_path=None, workdir=None, poll: float = 0.05, timeout: float | None = None):
    """Run the job ``name`` (a key of :data:`JOBS`, or ``"fluoviz_addon_autoencoders.<module>:<function>"``) with ``kwargs`` in a fresh process and return its result.

    ``progress(fraction, message)`` is called with what the job reports; if it raises (e.g. ``Cancelled``) the process is terminated and the exception propagates.
    Raises :class:`AutoencoderError` (a plain-English problem with the settings), :class:`WorkerError` (the process failed or died) or :class:`Cancelled`."""
    if not torch_available() and name != "selftest":
        raise AutoencoderError("PyTorch is not installed: open Settings, then Add-ons, and press Install (or Repair) for Autoencoders.")
    target = JOBS.get(name, name)
    if ":" not in target:
        raise AutoencoderError(f"Unknown job '{name}'.")
    ctx = mp.get_context("spawn")
    recv, send = ctx.Pipe(duplex=False)
    fd, result_path = tempfile.mkstemp(prefix="ae_result_", suffix=".pkl", dir=None if workdir is None else str(workdir))
    os.close(fd)
    os.remove(result_path)
    # spawn pickles the target BY NAME: take the function from the module object that is in sys.modules now (a test run or an add-on reload can leave an older copy of this
    # module behind, and pickling that copy's function fails with "it's not the same object")
    current = sys.modules.get(__name__)
    target_fn = getattr(current, "_child_main", _child_main) if current is not None else _child_main
    proc = ctx.Process(target=target_fn, args=(send, target, kwargs, None if history_path is None else str(history_path), result_path), daemon=True, name="fluoviz-autoencoder")
    t0 = time.monotonic()
    proc.start()
    send.close()
    outcome = None
    try:
        while True:
            if recv.poll(poll):
                try:
                    msg = recv.recv()
                except (EOFError, OSError):
                    msg = None
                if msg is None:
                    break
                kind = msg[0]
                if kind == "progress":
                    if progress is not None:
                        progress(msg[1], msg[2])
                    continue
                outcome = msg
                break
            if progress is not None:
                progress.check() if hasattr(progress, "check") else None            # a pending cancel is noticed even when the job is silent
            if not proc.is_alive() and not recv.poll():
                break
            if timeout is not None and time.monotonic() - t0 > timeout:
                raise WorkerError(f"The job took longer than {timeout:g} seconds and was stopped.")
    except BaseException:
        _stop(proc)
        _remove(result_path)
        _note_history(history_path, "cancelled")
        raise
    if outcome is None:                                                  # the process ended without a word
        proc.join(2)
        code = proc.exitcode
        _remove(result_path)
        _note_history(history_path, "error")
        hint = " It was probably stopped by the system because the computer ran out of memory." if code in (-9, 137) else ""
        raise WorkerError(f"The training process stopped unexpectedly (exit code {code}).{hint}", f"exit code {code}")
    proc.join(5)
    _stop(proc)
    kind = outcome[0]
    if kind == "done":
        try:
            with open(result_path, "rb") as fh:
                return pickle.load(fh)
        finally:
            _remove(result_path)
    _remove(result_path)
    _note_history(history_path, "error", outcome[1])
    if kind == "user_error":
        raise AutoencoderError(outcome[1])
    raise WorkerError("The training failed: " + outcome[1], outcome[2])


def _stop(proc) -> None:
    if proc.is_alive():
        proc.terminate()
        proc.join(3)
        if proc.is_alive():
            proc.kill()
            proc.join(3)


def _remove(path) -> None:
    for p in (path, path + ".part"):
        try:
            os.remove(p)
        except OSError:
            pass


def _note_history(path, event: str, message: str = "") -> None:
    if path is None:
        return
    try:
        from .train import HistoryWriter
        w = HistoryWriter(path)
        w.write({"event": event, "message": message})
        w.close()
    except Exception:                                                   # noqa: BLE001 - the history is a courtesy
        pass


# ----------------------------------------------------------------------------- a harmless job for tests of the machinery
def selftest_job(seconds: float = 0.2, steps: int = 4, fail: str | None = None, big: int = 0, die: bool = False, progress=None, history_path=None):
    """Sleeps ``seconds`` in ``steps`` steps reporting progress; ``fail="user"`` / ``"bug"`` raises, ``die`` ends the process without a word; returns facts about the process (is PyTorch loaded?)."""
    import sys
    if die:
        os._exit(9)
    for i in range(int(steps)):
        time.sleep(float(seconds) / max(int(steps), 1))
        if progress is not None:
            progress((i + 1) / max(int(steps), 1), f"step {i + 1}")
    if fail == "user":
        raise AutoencoderError("The latent dimension is too large.")
    if fail == "bug":
        raise ZeroDivisionError("boom")
    return {"pid": os.getpid(), "torch_loaded": "torch" in sys.modules, "fluoviz_loaded": "fluoviz" in sys.modules, "blob": b"x" * int(big)}
