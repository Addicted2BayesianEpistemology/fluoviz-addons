"""Downloads of a result: the codes (CSV), the rebuilt samples (``.npz``), the training history (CSV) and synthetic samples (``.npz`` / CSV, always labelled synthetic)."""
from __future__ import annotations

import csv
import io
import re

import numpy as np

from . import prep, views
from .results import model_of

SYNTH_README = ("SYNTHETIC samples: made up by a variational autoencoder, NOT measured. Do not mix them with measurements without saying so. "
                "'codes' are the latent codes drawn from the prior (seed and temperature are stored), '<variable>_synthetic' the decoded functions "
                "(NaN outside the domain of the network, units of the pre-treated data).")


def slug(text: str, default: str) -> str:
    return re.sub(r"[^\w\-]+", "_", str(text)).strip("_") or default


def _cell(v):
    if v is None:
        return ""
    if isinstance(v, (float, np.floating)):
        return "" if not np.isfinite(v) else f"{float(v):.8g}"
    if isinstance(v, (np.integer, int)) and not isinstance(v, bool):
        return int(v)
    return str(v)


def csv_bytes(header: list[str], rows) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    for row in rows:
        w.writerow([_cell(v) for v in row])
    return buf.getvalue().encode("utf-8")


def codes_csv(r) -> bytes:
    """One row per sample: split, codes (and their SDs), the reconstruction error and the flags."""
    ids = list(r.sample_ids or [])
    cols = [c for c in r.variables if c not in ("status", "split")]
    header = ["sample", "split"] + cols + ["status"]
    split = [views.split_words(r, i) for i in range(len(ids))]
    rows = []
    for i, sid in enumerate(ids):
        rows.append([sid, split[i]] + [np.asarray(r.variables[c])[i] for c in cols] + [np.asarray(r.variables["status"])[i] if "status" in r.variables else ""])
    return csv_bytes(header, rows)


def history_csv(r) -> bytes:
    h = (r.payload.get("training") or {}).get("history") or {}
    keys = [k for k in ("restart", "fold", "epoch", "train", "val", "rec", "kl", "smooth", "beta", "sigma2", "lr") if k in h]
    n = len(h[keys[0]]) if keys else 0
    rows = [[(h[k][i] if not isinstance(h[k], np.ndarray) else h[k][i]) for k in keys] for i in range(n)]
    return csv_bytes(keys, rows)


def reconstruction_npz(p, r) -> bytes:
    """The rebuilt functions of every sample on the grid of each variable (NaN outside the domain; module orientation ``(n, n_em, n_ex)`` / ``(n, n_wl)``), the mean, the axes and the
    codes; with current data also the observed (pre-treated) values and the residual."""
    model = model_of(r)
    ids = list(r.sample_ids or [])
    codes = views.codes_of(r)
    arrs: dict = {"ids": np.array(ids), "codes": codes, "code_names": np.array(model.code_names)}
    F = model.decode_functions(codes)
    mean = model.mean_functions()
    current = not p.stale_parts(r)
    ev = model.evaluate(prep.apply_data(p, ids, model.names), 0.0) if current else None
    for b in model.frame.blocks:
        arrs[f"{b.name}_reconstruction"] = b.domain.embed(F[b.name])
        arrs[f"{b.name}_mean"] = b.domain.embed(mean[b.name])
        if ev is not None:
            arrs[f"{b.name}_observed"] = b.domain.embed(ev.treated[b.name])
            arrs[f"{b.name}_residual"] = b.domain.embed(ev.treated[b.name] - F[b.name])
        for a, nm in zip(b.element.axes, b.element.axis_names):
            arrs[f"{b.name}_{nm}_nm"] = np.asarray(a, float)
    arrs["observed_included"] = np.array(bool(current))
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrs)
    return buf.getvalue()


def synthetic_export(r, n: int, seed: int, temperature: float, fmt: str) -> tuple[bytes, str, str]:
    """``(bytes, media type, file suffix)`` of ``n`` synthetic samples (VAE only)."""
    from fastapi import HTTPException
    model = model_of(r)
    if not model.is_vae:
        raise HTTPException(400, "Only variational autoencoders can draw new samples (this network is a deterministic autoencoder). Train a variational autoencoder.")
    out = views.synthetic(r, model, n, seed, temperature, images=False)
    if fmt == "csv":
        header = ["sample", "synthetic"] + model.code_names + ["nearest_real_sample", "distance_in_sd"]
        rows = [[out["ids"][k], "yes: made up by the network, not measured"] + list(out["codes"][k]) + [out["nearest"][k]["id"], out["nearest"][k]["distance"]] for k in range(n)]
        return csv_bytes(header, rows), "text/csv", "csv"
    F = model.decode_functions(out["codes"])
    arrs: dict = {"synthetic": np.array(True), "README": np.array(SYNTH_README), "ids": np.array(out["ids"]), "codes": out["codes"], "code_names": np.array(model.code_names),
                  "seed": np.array(seed), "temperature": np.array(temperature), "nearest_real_sample": np.array([x["id"] for x in out["nearest"]])}
    for b in model.frame.blocks:
        arrs[f"{b.name}_synthetic"] = b.domain.embed(F[b.name])
        for a, nm in zip(b.element.axes, b.element.axis_names):
            arrs[f"{b.name}_{nm}_nm"] = np.asarray(a, float)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrs)
    return buf.getvalue(), "application/octet-stream", "npz"
