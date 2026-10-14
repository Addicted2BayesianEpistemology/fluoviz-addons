"""The live training history: the worker appends one JSON line per event to a small file in ``runtime/tmp/``; the page polls ``GET /fit/{job}/history``.  The server only reads the file
(it never imports the training code or PyTorch).  After the job ends the rows are kept in memory (the last few fits) and the file is removed."""
from __future__ import annotations

import json
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

from fluoviz import paths

MAX_KEPT = 12
_LOCK = threading.Lock()


@dataclass
class Fit:
    token: str
    path: Path
    job_id: str | None = None
    rows: list = field(default_factory=list)         # every record of the file, once the job has ended (kept in memory)
    finished: bool = False


FITS: "OrderedDict[str, Fit]" = OrderedDict()          # job id -> Fit
_PENDING: dict[str, Fit] = {}                          # token -> Fit (before the job id is known)


def new_fit() -> Fit:
    """A history file for a job that is about to start (the folder is ``runtime/tmp``, inside FluoViz's own folder)."""
    d = Path(paths.RUNTIME) / "tmp"
    d.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:12]
    fit = Fit(token=token, path=d / f"autoencoders-{token}.jsonl")
    with _LOCK:
        _PENDING[token] = fit
    return fit


def attach(fit: Fit, job_id: str) -> None:
    """Remember which job writes ``fit`` (called right after the job was submitted)."""
    with _LOCK:
        fit.job_id = job_id
        FITS[job_id] = fit
        _PENDING.pop(fit.token, None)
        while len(FITS) > MAX_KEPT:
            old = FITS.popitem(last=False)[1]
            old.path.unlink(missing_ok=True)


def finish(fit: Fit) -> None:
    """The job ended: keep its rows in memory and remove the file."""
    rows, _ = read_rows(fit.path, 0)
    with _LOCK:
        fit.rows = rows
        fit.finished = True
    fit.path.unlink(missing_ok=True)


def read_rows(path, since: int = 0) -> tuple[list[dict], int]:
    """The records of the file after the first ``since`` records and the index to ask for next (an incomplete last line is left for the next call)."""
    try:
        raw = Path(path).read_bytes()
    except OSError:
        return [], since
    complete = raw.split(b"\n")[:-1]
    out = []
    for ln in complete[since:]:
        try:
            out.append(json.loads(ln.decode("utf-8")))
        except ValueError:
            continue
    return out, len(complete)


def rows_after(fit: Fit, since: int) -> tuple[list[dict], int]:
    if fit.finished:
        return fit.rows[since:], len(fit.rows)
    return read_rows(fit.path, since)


def history_view(fit: Fit, job, since: int) -> dict:
    """The answer of ``GET /fit/{job}/history``: the epoch rows after ``since`` and where the training is."""
    rows, nxt = rows_after(fit, since)
    epochs = [{k: v for k, v in r.items() if k != "event"} for r in rows if r.get("event") == "epoch"]
    events = [r for r in rows if r.get("event") not in (None, "epoch")]
    info = {"restart": None, "fold": None, "n_restarts": None, "n_folds": None}
    allrows = fit.rows if fit.finished else read_rows(fit.path, 0)[0]
    for r in allrows:
        if r.get("event") == "start":
            info["n_restarts"], info["n_folds"] = r.get("restarts"), (r.get("folds") if r.get("mode") == "kfold" else None)
    cur = [r for r in allrows if r.get("event") == "epoch"]
    if cur:
        info["restart"], info["fold"] = cur[-1].get("restart"), cur[-1].get("fold")
    best = None
    ends = [r for r in allrows if r.get("event") == "run_end" and r.get("restart") == info["restart"] and r.get("fold") == info["fold"] and r.get("best_epoch") is not None]
    if ends:                                                            # the run that is (or was last) training: where its best weights are (-1 = the start, which no epoch improved on)
        e = ends[-1]
        best = {"restart": e.get("restart"), "fold": e.get("fold"), "epoch": e.get("best_epoch"), "val": e.get("val")}
    else:
        for r in cur:
            if r.get("best") and r.get("restart") == info["restart"] and r.get("fold") == info["fold"]:
                best = {"restart": r.get("restart"), "fold": r.get("fold"), "epoch": r.get("epoch"), "val": r.get("val")}
    status = job.status if job is not None else ("done" if fit.finished else "running")
    phase = "done" if status == "done" else ("checking" if any(r.get("event") == "done" for r in allrows) else ("training" if cur else "preparing"))
    if status in ("error", "cancelled"):
        phase = status
    return {"job": fit.job_id, "status": status, "phase": phase, **info, "epochs": epochs, "events": events, "next": nxt, "best": best, "message": getattr(job, "message", "") if job is not None else ""}
