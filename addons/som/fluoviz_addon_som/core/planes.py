"""Planes: one number per unit, drawn as a coloured map (SPEC section 5.3).  Pure numpy (the indices call FluoViz's feature module).

A plane is a dictionary ``{"key", "label", "values" (M,), "kind": "numeric" | "category", "units", "words", "signed"}``; a category plane also has ``"categories"`` and the unit labels.
"""
from __future__ import annotations

import numpy as np

from .metric import pca_basis, weighted_centre
from .spec import SomError


# ----------------------------------------------------------------------------- planes of the map itself
def basic_planes(protos_t: np.ndarray, mean_t: np.ndarray, hits: np.ndarray, umat: np.ndarray, qe: np.ndarray, c1: np.ndarray) -> dict:
    M = protos_t.shape[0]
    qe_unit = np.full(M, np.nan)
    for j in np.flatnonzero(hits > 0):
        qe_unit[j] = float(np.mean(qe[c1 == j]))
    dmean = np.sqrt(((protos_t - mean_t) ** 2).sum(axis=1))
    return {
        "umatrix": {"key": "umatrix", "label": "U-matrix (distance to the neighbouring units)", "values": umat, "kind": "numeric", "signed": False,
                    "words": "High values (dark ridges) separate regions of the map whose prototypes differ a lot; low values are plateaus of similar samples."},
        "hits": {"key": "hits", "label": "Hits (samples on the unit)", "values": hits.astype(float), "kind": "numeric", "signed": False,
                 "words": "How many samples have this unit as their best unit. Units with no sample interpolate between their neighbours."},
        "dmean": {"key": "dmean", "label": "Distance of the prototype to the average sample", "values": dmean, "kind": "numeric", "signed": False,
                  "words": "How unlike the average sample the prototype is: the edges of the map hold the most unusual samples."},
        "qe_unit": {"key": "qe_unit", "label": "Quantisation error of the unit", "values": qe_unit, "kind": "numeric", "signed": False,
                    "words": "The mean distance of the samples of the unit to its prototype: high where the prototype stands for its samples poorly."},
    }


def score_planes(R_train: np.ndarray, protos_raw: np.ndarray, l2_gain: np.ndarray, w: np.ndarray | None = None, k: int = 8) -> dict:
    """The projection of every prototype on the leading principal directions of the training coordinates in the plain L2 metric (the multivariate fPCA / PCA scores)."""
    X = np.asarray(R_train, float) * l2_gain
    mu = np.nanmean(X, axis=0)
    Xf = np.where(np.isfinite(X), X, mu)
    Xc, mu2, dof = weighted_centre(Xf, w)
    V, s = pca_basis(Xc, k)
    total = float((Xc ** 2).sum())
    S = (np.asarray(protos_raw, float) * l2_gain - mu2) @ V
    out = {}
    for j in range(V.shape[1]):
        share = float(s[j] ** 2 / total) if total > 0 else float("nan")
        out[f"score{j + 1}"] = {"key": f"score{j + 1}", "label": f"Principal direction {j + 1} ({100 * share:.0f} % of the variation)", "values": S[:, j], "kind": "numeric", "signed": True,
                                "words": "The position of the prototype along the j-th direction of largest variation of the samples (the score of the functional PCA / PCA)."}
    return out


# ----------------------------------------------------------------------------- planes of the prototypes' functions
def _surface_block(frame):
    for b in frame.blocks:
        if b.element.kind == "surface":
            return b
    return None


def _curve_block(frame):
    for b in frame.blocks:
        if b.element.kind == "curve":
            return b
    return None


def _nearest(axis: np.ndarray, x: float, what: str) -> int:
    axis = np.asarray(axis, float)
    if not axis[0] - 1e-9 <= x <= axis[-1] + 1e-9:
        raise SomError(f"The {what} {x:g} nm is outside the measured range ({axis[0]:g} to {axis[-1]:g} nm).")
    return int(np.argmin(np.abs(axis - x)))


def feature_plane(frame, funcs: dict, spec: dict) -> dict:
    """A feature of the prototype EEM / spectrum: ``{"kind": "point", "ex", "em"}``, ``{"kind": "region_mean"|"region_max", "ex": [a, b], "em": [a, b]}``, ``{"kind": "total"}``,
    ``{"kind": "absorbance", "wl"}``."""
    kind = spec.get("kind")
    if kind == "absorbance":
        b = _curve_block(frame)
        if b is None:
            raise SomError("This map does not read the absorbance.")
        wl = np.asarray(b.element.axes[0], float)
        i = _nearest(wl, float(spec["wl"]), "wavelength")
        G = b.domain.embed(funcs[b.name])
        v = G[:, i]
        if not np.isfinite(v).all():
            raise SomError(f"The absorbance at {wl[i]:g} nm is not part of the map's domain.")
        return {"key": f"absorbance@{wl[i]:g}", "label": f"Absorbance at {wl[i]:g} nm", "values": v, "kind": "numeric", "signed": False, "units": "absorbance",
                "words": "The absorbance of the prototype at that wavelength."}
    b = _surface_block(frame)
    if b is None:
        raise SomError("This map does not read the fluorescence.")
    em = np.asarray(b.element.axes[0], float)
    ex = np.asarray(b.element.axes[1], float)
    G = b.domain.embed(funcs[b.name])                                  # (M, n_em, n_ex)
    units = str(b.element.units or "")
    if kind == "total":
        w = np.asarray(b.domain.w, float)
        v = funcs[b.name] @ w
        return {"key": "total", "label": "Overall intensity", "values": v, "kind": "numeric", "signed": False, "units": units, "words": "The mean intensity of the prototype over the whole EEM."}
    if kind == "point":
        i, j = _nearest(ex, float(spec["ex"]), "excitation"), _nearest(em, float(spec["em"]), "emission")
        v = G[:, j, i]
        if not np.isfinite(v).all():
            raise SomError(f"The pixel at excitation {ex[i]:g} nm and emission {em[j]:g} nm is not part of the map's domain (scatter, a cut region or below the Rayleigh line).")
        return {"key": f"point@{ex[i]:g}/{em[j]:g}", "label": f"Intensity at excitation {ex[i]:g} nm, emission {em[j]:g} nm", "values": v, "kind": "numeric", "signed": False, "units": units,
                "words": "The intensity of the prototype at that pixel."}
    if kind in ("region_mean", "region_max"):
        e0, e1 = sorted(float(x) for x in spec["ex"])
        m0, m1 = sorted(float(x) for x in spec["em"])
        sel = ((ex >= e0) & (ex <= e1))[None, None, :] & ((em >= m0) & (em <= m1))[None, :, None]
        sub = np.where(sel, G, np.nan)
        if not np.isfinite(sub).any():
            raise SomError("The box holds no pixel of the map's domain.")
        with np.errstate(invalid="ignore"):
            v = np.nanmean(sub, axis=(1, 2)) if kind == "region_mean" else np.nanmax(sub, axis=(1, 2))
        what = "Mean" if kind == "region_mean" else "Highest"
        return {"key": f"{kind}@{e0:g}-{e1:g}/{m0:g}-{m1:g}", "label": f"{what} intensity in the box excitation {e0:g}–{e1:g} nm, emission {m0:g}–{m1:g} nm", "values": v, "kind": "numeric",
                "signed": False, "units": units, "words": f"The {what.lower()} intensity of the prototype inside the box."}
    raise SomError("Unknown kind of feature.")


def index_catalogue(frame) -> list:
    """The fluorescence and absorbance indices that can be computed for the prototypes of this map."""
    from fluoviz.analysis.features import list_indices
    has_eem = _surface_block(frame) is not None
    has_abs = _curve_block(frame) is not None
    out = []
    for d in list_indices():
        req = set(d.get("requires") or [])
        if any(r.startswith("meta:") for r in req):
            continue
        if ("eem" in req and not has_eem) or ("absorbance" in req and not has_abs):
            continue
        for o in d["outputs"]:
            suf = o.get("suffix", "")
            out.append({"key": d["key"] + suf if suf else d["key"], "index": d["key"], "suffix": suf, "name": d["name"] + (f" ({o.get('label', suf)})" if suf else ""), "group": d["group"],
                        "kind": d["kind"], "description": d.get("description", "")})
    return out


def index_plane(frame, funcs: dict, key: str, ids: list) -> dict:
    """An index of the prototypes: ``key`` is an index key, with the output suffix for those with several outputs (``FRI_Phi_I``)."""
    from fluoviz.analysis.features import get_index, make_inputs
    from fluoviz.analysis.features.base import FeatureInputError
    cat = {c["key"]: c for c in index_catalogue(frame)}
    if key not in cat:
        raise SomError(f"The index '{key}' cannot be computed for this map (it needs data the map does not read).")
    c = cat[key]
    sb, cb = _surface_block(frame), _curve_block(frame)
    view = None
    if sb is not None:
        view = {"X": sb.domain.embed(funcs[sb.name]), "ex": np.asarray(sb.element.axes[1], float), "em": np.asarray(sb.element.axes[0], float), "ids": list(ids), "units": str(sb.element.units or "")}
    ab = None
    if cb is not None:
        ab = (np.asarray(cb.element.axes[0], float), cb.domain.embed(funcs[cb.name]), list(ids))
    try:
        inputs = make_inputs(view=view, absorbance=ab, ids=list(ids))
        res = get_index(c["index"]).compute_all(inputs)
        vals = res.values[c["suffix"]] if c["suffix"] in res.values else next(iter(res.values.values()))
    except (FeatureInputError, ValueError, KeyError) as exc:
        raise SomError(f"The index '{key}' could not be computed for the prototypes: {exc}") from None
    v = np.asarray(vals, float)
    return {"key": f"index:{key}", "label": c["name"], "values": v, "kind": "numeric", "signed": False, "units": "",
            "words": (c["description"] + " Computed on the prototype of each unit: an index of a prototype (an average, possibly smoothed or shape-only) is not the index of a sample."
                      ).strip()}


# ----------------------------------------------------------------------------- planes of a variable of the samples
def variable_plane(values, c1: np.ndarray, M: int, lat_d2: np.ndarray, label: str, smooth: bool = False, kind: str | None = None) -> dict:
    """The mean of a numeric variable over the samples of every unit (NaN for an empty unit; with ``smooth`` the mean is weighted by the closeness of the neighbouring units, so empty
    units get a value too), or for a category the label map: the most frequent label of every unit, its share and the counts."""
    vals = np.asarray(values)
    numeric = kind == "numeric" if kind else vals.dtype.kind in "fiub"
    n = c1.shape[0]
    if numeric:
        v = vals.astype(float)
        ok = np.isfinite(v)
        out = np.full(M, np.nan)
        if smooth:
            H = np.exp(-lat_d2 / 2.0)
            num = H[:, c1[ok]] @ v[ok]
            den = H[:, c1[ok]].sum(axis=1)
            out = np.where(den > 1e-9, num / np.maximum(den, 1e-300), np.nan)
        else:
            for j in range(M):
                sel = (c1 == j) & ok
                if sel.any():
                    out[j] = float(v[sel].mean())
        n_valid = np.bincount(c1[ok], minlength=M)
        return {"key": f"var:{label}", "label": label, "values": out, "kind": "numeric", "signed": False, "units": "", "n": n_valid,
                "words": "The mean of the variable over the samples of the unit" + (", smoothed over the neighbouring units." if smooth else ". Units with no sample are blank.")}
    s = np.array([("" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)) for x in vals], dtype=object)
    cats = sorted({x for x in s if x != ""})
    labels = np.full(M, "", dtype=object)
    purity = np.full(M, np.nan)
    counts = np.zeros((M, len(cats)), int)
    for j in range(M):
        sel = s[c1 == j]
        for k, c in enumerate(cats):
            counts[j, k] = int((sel == c).sum())
        tot = counts[j].sum()
        if tot:
            k = int(np.argmax(counts[j]))
            labels[j] = cats[k]
            purity[j] = counts[j, k] / tot
    code = np.array([cats.index(x) if x in cats else -1 for x in labels], float)
    code[code < 0] = np.nan
    return {"key": f"var:{label}", "label": label, "values": code, "kind": "category", "signed": False, "categories": cats, "unit_labels": [str(x) for x in labels], "purity": purity,
            "counts": counts, "words": "The most frequent value of the variable among the samples of the unit; the purity is the share of those samples that have it."}


def sample_scores(R: np.ndarray, l2_gain: np.ndarray, w: np.ndarray | None = None, k: int = 4) -> np.ndarray:
    """The scores ``(n, k)`` of the samples on the leading principal directions of the plain L2 metric (the same directions :func:`score_planes` projects the prototypes on)."""
    X = np.asarray(R, float) * l2_gain
    mu = np.nanmean(X, axis=0)
    Xf = np.where(np.isfinite(X), X, mu)
    Xc, mu2, dof = weighted_centre(Xf, w)
    V, s = pca_basis(Xc, k)
    return (Xf - mu2) @ V
