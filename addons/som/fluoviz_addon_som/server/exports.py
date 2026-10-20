"""Downloads: positions, planes, groups, history (CSV) and prototypes (.npz).  Files keep the orientation of the data."""
from __future__ import annotations

import csv
import io

import numpy as np
from fastapi import HTTPException

from .results import is_variables, model_of


def _csv(header: list, rows: list) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    for r in rows:
        w.writerow(["" if (isinstance(x, float) and not np.isfinite(x)) else x for x in r])
    return buf.getvalue().encode("utf-8")


def export(r, what: str) -> tuple:
    """``(bytes, filename, media type)``."""
    pl = r.payload
    S = pl["samples"]
    ids = [str(i) for i in r.sample_ids]
    names = pl["map"]["unit_names"]
    stem = r.id
    if what == "positions":
        cl = pl["clusters"]["sample_labels"]
        rows = [[ids[i], names[S["unit"][i]], int(S["unit"][i] // pl["map"]["cols"] + 1), int(S["unit"][i] % pl["map"]["cols"] + 1), float(S["pos"][i, 0]), float(S["pos"][i, 1]), float(S["qe"][i]),
                 float(S["qe_rel"][i]), float(S["heldout_qe"][i]), f"group {int(cl[i]) + 1}", str(S["status"][i])] for i in range(len(ids))]
        return _csv(["sample", "unit", "row", "column", "x", "y", "distance_to_unit", "share_not_captured", "heldout_distance", "group", "status"], rows), f"{stem}-positions.csv", "text/csv"
    if what == "planes":
        keys = list(pl["planes"])
        rows = [[names[j]] + [float(pl["planes"][k]["values"][j]) if pl["planes"][k]["kind"] == "numeric" else "" for k in keys] for j in range(pl["map"]["units"])]
        return _csv(["unit"] + keys, rows), f"{stem}-planes.csv", "text/csv"
    if what == "clusters":
        lab = pl["clusters"]["unit_labels"]
        rows = [[names[j], int(pl["map"]["hits"][j]), f"group {int(lab[j]) + 1}"] for j in range(pl["map"]["units"])]
        return _csv(["unit", "samples", "group"], rows), f"{stem}-groups.csv", "text/csv"
    if what == "history":
        h = pl["training"]["history"]
        rows = [[i + 1, h["phase"][i], h["radius"][i], h["energy"][i], h["qe"][i]] for i in range(len(h["energy"]))]
        return _csv(["epoch", "phase", "radius", "energy", "mean_distance_to_unit"], rows), f"{stem}-history.csv", "text/csv"
    if what == "prototypes":
        arrays: dict = {"unit_names": np.array(names), "hits": np.asarray(pl["map"]["hits"]), "coords": np.asarray(pl["map"]["coords"])}
        if is_variables(r):
            arrays["prototypes"] = np.asarray(pl["prototype_values"], float)
            arrays["columns"] = np.array(pl["columns"])
        else:
            model = model_of(r)
            for b in model.frame.blocks:
                if b.element.kind == "surface":
                    arrays["prototype_eems"] = np.asarray(pl["prototypes"][b.name], np.float32)          # (units, n_em, n_ex), NaN outside the domain
                    arrays["em"], arrays["ex"] = np.asarray(pl["em"]), np.asarray(pl["ex"])
                else:
                    arrays["prototype_absorbance"] = np.asarray(pl["prototype_curves"][b.name], np.float32)
                    arrays["wavelength"] = np.asarray(pl["abs_wl"])
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        return buf.getvalue(), f"{stem}-prototypes.npz", "application/octet-stream"
    raise HTTPException(400, "'what' must be positions, planes, clusters, history or prototypes.")
