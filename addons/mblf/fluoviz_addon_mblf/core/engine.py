"""The Stan engine: a private conda environment with CmdStan, cmdstanpy, xarray and a C++ compiler, in which the posterior is sampled (SPEC section 3).

Why a private environment and not the add-on's ``conda`` list: CmdStan pins the compiler of conda-forge (gcc 15 on Linux, clang 21 on macOS) and installing it into FluoViz's own
environment would replace the compilers FluoViz already has (the framework forbids an add-on to change the packages of the base installation).  So the engine lives in
``<data folder of the add-on>/engine/env`` (inside ``runtime/``, kept between versions of the add-on), is prepared ONCE from the page with FluoViz's own micromamba and the same
private settings the installer uses, and every fit runs there in a separate process (``micromamba run -p <env> python engine_job.py <job folder>``, which also does what conda's
activation does: the compiler, ``CMDSTAN``).  The FluoViz server never imports cmdstanpy.

Development and tests: ``FLUOVIZ_MBLF_ENGINE=<prefix of an environment that has cmdstan, cmdstanpy, xarray, numpy and scipy>`` uses that one instead (nothing is prepared);
``FLUOVIZ_MBLF_ENGINE_DIR`` puts the engine somewhere else (CmdStan cannot compile in a folder whose path contains a space).

A job: the parent writes ``input.json`` + ``input.npz`` into a fresh folder, starts the child, relays its progress (lines ``@@ <fraction> <text>`` and CmdStan's own
``Chain [c] Iteration: i / n`` lines), stops the whole process tree when the person cancels, and reads ``output.json`` + ``output.npz``, which the child writes completely before it says
``@@done``.  Plain English everywhere.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

ENGINE_SPECS = ["python=3.12", "cmdstan>=2.36", "cmdstanpy>=1.2", "xarray", "numpy", "scipy", "pandas"]
ENGINE_MB = 450                      # what the download is, roughly (shown on the button)
JOB_SCRIPT = Path(__file__).resolve().parents[1] / "engine_job.py"
_CHAIN = re.compile(r"Chain \[(\d+)\]\s+Iteration:\s*(\d+)\s*/\s*(\d+)")
_LOCK = threading.Lock()


try:                                                                    # the exception FluoViz's job runner understands as "cancelled"
    from fluoviz.analysis.common import Cancelled
except Exception:                                                       # noqa: BLE001 - outside FluoViz (a script): a local stand-in
    class Cancelled(Exception):                                         # type: ignore[no-redef]
        """The job was cancelled."""


class EngineError(RuntimeError):
    """The engine is missing, broken, or the job failed for a reason that is not a setting; ``detail`` has the technical text."""

    def __init__(self, message: str, detail: str = ""):
        super().__init__(message)
        self.detail = detail


@dataclass
class Engine:
    state: str                     # ready | missing | unavailable
    prefix: Path | None            # the conda environment
    micromamba: Path | None
    message: str
    source: str                    # private | given
    size_mb: float = 0.0

    def public(self) -> dict:
        return {"state": self.state, "message": self.message, "source": self.source, "size_mb": round(self.size_mb, 1), "download_mb": ENGINE_MB,
                "prefix": None if self.prefix is None else str(self.prefix)}


# ----------------------------------------------------------------------------- where it is
def _micromamba() -> Path | None:
    try:
        from fluoviz.addons.conda import mm_path
        return mm_path()
    except Exception:                                                   # noqa: BLE001 - outside FluoViz
        given = os.environ.get("FLUOVIZ_MM")
        return Path(given) if given and Path(given).is_file() else None


def engine_dir(data_dir: Path) -> Path:
    given = os.environ.get("FLUOVIZ_MBLF_ENGINE_DIR")
    return Path(given) if given else Path(data_dir) / "engine"


def _python_of(prefix: Path) -> Path:
    return prefix / ("python.exe" if os.name == "nt" else "bin/python")


def _has_cmdstan(prefix: Path) -> bool:
    return any((prefix / sub / "cmdstan" / "makefile").is_file() for sub in ("bin", "Library/bin"))


def _size_mb(prefix: Path) -> float:
    total = 0
    for root, _dirs, files in os.walk(prefix):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total / 1e6


def locate(data_dir: Path, size: bool = False) -> Engine:
    """Where the engine is and whether it can be used (quick: no process is started)."""
    mm = _micromamba()
    given = os.environ.get("FLUOVIZ_MBLF_ENGINE")
    if given:
        prefix = Path(given)
        ok = _python_of(prefix).is_file() and _has_cmdstan(prefix)
        return Engine("ready" if ok and mm else "unavailable", prefix, mm,
                      "The Stan engine of FLUOVIZ_MBLF_ENGINE is used." if ok and mm else
                      f"FLUOVIZ_MBLF_ENGINE names '{prefix}', which is not an environment with CmdStan (or micromamba is missing).", "given")
    prefix = engine_dir(data_dir) / "env"
    if mm is None:
        return Engine("unavailable", prefix, None, "This FluoViz was not installed with its own installer, so the Stan engine cannot be prepared automatically. Install FluoViz with its "
                      "installer, or point FLUOVIZ_MBLF_ENGINE at a conda environment that has cmdstan, cmdstanpy, xarray, numpy and scipy.", "private")
    if " " in str(prefix):
        return Engine("unavailable", prefix, mm, f"CmdStan cannot compile in a folder whose path contains a space ('{prefix}'). Move FluoViz to a folder without spaces, or set "
                      "FLUOVIZ_MBLF_ENGINE_DIR to one.", "private")
    if _python_of(prefix).is_file() and _has_cmdstan(prefix) and (prefix.parent / "engine.json").is_file():
        return Engine("ready", prefix, mm, "The Stan engine is ready.", "private", _size_mb(prefix) if size else 0.0)
    return Engine("missing", prefix, mm, f"The Stan engine is not prepared yet. Preparing it downloads about {ENGINE_MB} MB once (CmdStan and a C++ compiler) into the add-on's own "
                  "folder; FluoViz itself is not changed.", "private")


# ----------------------------------------------------------------------------- preparing it (a FluoViz job)
def _env_for_mamba() -> dict:
    try:
        from fluoviz.addons.conda import mamba_env
        return mamba_env()
    except Exception:                                                   # noqa: BLE001 - outside FluoViz
        env = dict(os.environ)
        for k in ("CONDARC", "MAMBARC", "CONDA_PREFIX", "PYTHONPATH", "PYTHONHOME"):
            env.pop(k, None)
        return env


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        import psutil
        parent = psutil.Process(proc.pid)
        kids = parent.children(recursive=True)
        for k in kids:
            try:
                k.kill()
            except psutil.Error:
                pass
        parent.kill()
    except Exception:                                                   # noqa: BLE001 - psutil missing or the process is gone
        try:
            proc.kill()
        except OSError:
            pass


def _pump(proc: subprocess.Popen, on_line, progress) -> list[str]:
    """Read the child's output line by line (a thread), call ``on_line`` for each one, and stop the child when ``progress`` raises (Cancel).  Returns the last lines."""
    tail: list[str] = []
    lines: list[str] = []
    lock = threading.Lock()

    def reader():
        for raw in iter(proc.stdout.readline, ""):
            with lock:
                lines.append(raw.rstrip("\n"))

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    try:
        while True:
            with lock:
                batch, lines[:] = list(lines), []
            for ln in batch:
                tail.append(ln)
                del tail[:-60]
                on_line(ln)
            if proc.poll() is not None and not t.is_alive():
                with lock:
                    rest = list(lines)
                for ln in rest:
                    tail.append(ln)
                    on_line(ln)
                break
            if progress is not None:
                (progress.check() if hasattr(progress, "check") else None)
            time.sleep(0.1)
    except BaseException:
        _kill_tree(proc)
        raise
    return tail[-60:]


def prepare(data_dir: Path, progress=None) -> Engine:
    """Create the engine (micromamba ``create``); ``progress(fraction, text)`` reports in words, raising in it cancels (the half-made environment is removed)."""
    eng = locate(data_dir)
    if eng.state == "ready":
        return eng
    if eng.state == "unavailable":
        raise EngineError(eng.message)
    if not _LOCK.acquire(blocking=False):
        raise EngineError("The Stan engine is already being prepared. Wait until it has finished.")
    try:
        prefix = eng.prefix
        if prefix.exists():
            shutil.rmtree(prefix, ignore_errors=True)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        say = progress or (lambda f, t="": None)
        say(0.02, "Finding the right versions of CmdStan and its compiler")
        args = [str(eng.micromamba), "create", "-y", "-p", str(prefix), *ENGINE_SPECS, "--no-rc", "-c", "conda-forge", "--override-channels"]
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=_env_for_mamba(), bufsize=1)
        seen = {"n": 0}

        def on_line(ln: str):
            low = ln.lower()
            if "linking" in low or "extract" in low:
                seen["n"] += 1
                say(min(0.3 + 0.006 * seen["n"], 0.9), "Setting up the packages")
            elif "download" in low or "fetch" in low or "mb" in low:
                say(0.2, f"Downloading CmdStan and its compiler (about {ENGINE_MB} MB)")

        try:
            tail = _pump(proc, on_line, progress)
        except BaseException:
            shutil.rmtree(prefix, ignore_errors=True)
            raise
        if proc.returncode != 0:
            shutil.rmtree(prefix, ignore_errors=True)
            text = "\n".join(tail)
            if "connection" in text.lower() or "network" in text.lower() or "resolve host" in text.lower():
                raise EngineError("The Stan engine could not be downloaded. Check that this computer is connected to the internet, then try again. Nothing was changed.", text)
            raise EngineError("The Stan engine could not be prepared. Nothing was changed in FluoViz. (micromamba said: " + " ".join(tail[-3:])[-400:] + ")", text)
        say(0.93, "Checking that the engine works")
        check = run_python(Engine("ready", prefix, eng.micromamba, "", "private"), ["-c", "import cmdstanpy, xarray, scipy; print(cmdstanpy.cmdstan_path())"], timeout=300)
        if check.returncode != 0:
            shutil.rmtree(prefix, ignore_errors=True)
            raise EngineError("The Stan engine was downloaded but does not start, so it was removed again. Try again; if it fails again, report it.", check.stdout[-2000:])
        (prefix.parent / "engine.json").write_text(json.dumps({"specs": ENGINE_SPECS, "prepared": time.strftime("%Y-%m-%dT%H:%M:%S"), "cmdstan": check.stdout.strip().splitlines()[-1]},
                                                              indent=1), encoding="utf-8")
        say(1.0, "The Stan engine is ready")
        return locate(data_dir)
    finally:
        _LOCK.release()


def remove(data_dir: Path) -> None:
    """Delete the private engine (the compiled programs too).  An engine given by FLUOVIZ_MBLF_ENGINE is never touched."""
    if os.environ.get("FLUOVIZ_MBLF_ENGINE"):
        raise EngineError("The engine is given by FLUOVIZ_MBLF_ENGINE, so it is not removed from here.")
    d = engine_dir(data_dir)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    if d.exists():
        raise EngineError("The Stan engine could not be removed completely (a file is in use). Restart FluoViz and try again.")


# ----------------------------------------------------------------------------- running a job in it
def _child_env(eng: Engine, data_dir: Path) -> dict:
    env = _env_for_mamba()
    for k in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        env.pop(k, None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    cache = engine_dir(data_dir) / "stan-cache"
    if " " in str(cache):
        import tempfile
        cache = Path(tempfile.gettempdir()) / "fluoviz-mblf-stan-cache"
    cache.mkdir(parents=True, exist_ok=True)
    env["MBLF_CACHE_DIR"] = str(cache)
    if os.environ.get("FLUOVIZ_MAMBA_ROOT"):                                  # tests: a root prefix of their own
        env["MAMBA_ROOT_PREFIX"] = os.environ["FLUOVIZ_MAMBA_ROOT"]
    return env


def _command(eng: Engine, args: list[str]) -> list[str]:
    return [str(eng.micromamba), "run", "-p", str(eng.prefix), "python", *args]


def run_python(eng: Engine, args: list[str], timeout: float = 600, data_dir: Path | None = None) -> subprocess.CompletedProcess:
    env = _child_env(eng, data_dir or (eng.prefix.parent.parent if eng.prefix else Path.cwd()))
    return subprocess.run(_command(eng, args), capture_output=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, timeout=timeout)


def run_job(eng: Engine, data_dir: Path, job_dir: Path, progress=None) -> dict:
    """Run ``engine_job.py`` on ``job_dir`` (``input.json`` / ``input.npz`` written by the caller).  Returns the parsed ``output.json`` (the arrays are in ``output.npz``).

    ``progress(fraction, text)``: what the child reports; raising in it (Cancel) stops the child and everything it started, and re-raises."""
    if eng.state != "ready":
        raise EngineError(eng.message)
    say = progress or (lambda f, t="": None)
    meta = json.loads((job_dir / "input.json").read_text(encoding="utf-8"))
    total_iter = {"n": max(1, int(meta.get("chains", 1)))}
    chains: dict[int, float] = {}
    state = {"phase": ""}

    def on_line(ln: str):
        if ln.startswith("@@"):
            parts = ln[2:].strip().split(" ", 1)
            try:
                frac = float(parts[0])
            except ValueError:
                return
            state["phase"] = parts[1] if len(parts) > 1 else ""
            say(frac, state["phase"])
            return
        m = _CHAIN.search(ln)
        if m:
            c, i, n = int(m.group(1)), int(m.group(2)), int(m.group(3))
            chains[c] = i / max(n, 1)
            done = sum(chains.values()) / total_iter["n"]
            warm = "warm-up" if "Warmup" in ln else "sampling"
            say(0.08 + 0.8 * done, f"Sampling the posterior: {100 * done:.0f} % ({len(chains)} of {total_iter['n']} chains running; {warm})")

    proc = subprocess.Popen(_command(eng, ["-u", str(JOB_SCRIPT), str(job_dir)]), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=_child_env(eng, data_dir), bufsize=1,
                            cwd=str(job_dir))
    try:
        tail = _pump(proc, on_line, progress)
    except BaseException:
        _note(job_dir, "cancelled")
        raise
    out = job_dir / "output.json"
    if proc.returncode != 0 or not out.is_file():
        text = "\n".join(tail)
        err = job_dir / "error.json"
        if err.is_file():
            e = json.loads(err.read_text(encoding="utf-8"))
            raise EngineError(e.get("message") or "The fit failed.", e.get("detail", text))
        raise EngineError("The Stan engine stopped without a result. " + (" ".join(tail[-2:])[-300:] if tail else ""), text)
    return json.loads(out.read_text(encoding="utf-8"))


def _note(job_dir: Path, status: str) -> None:
    try:
        (job_dir / "status.txt").write_text(status, encoding="utf-8")
    except OSError:
        pass


def python_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"
