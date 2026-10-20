"""Looking at a result: one sample, one unit, the prototypes, the planes, the groups, unusual samples, trajectories, the projection.  Images leave in browser orientation."""
from __future__ import annotations

import numpy as np
from fastapi import HTTPException

from fluoviz.analysis.frozen import STATUS_WORDS
from fluoviz.analysis.frozen.kinds import _decomp as D
from fluoviz.server.analysis import plot_image, plot_images, plot_mask

from ..core import groups as G
from ..core import planes as PLN
from ..core import quality as Q
from ..core.spec import SomError
from ..kinds import som as KS
from . import prep
from .results import browser_images, frozen_of, is_variables, model_of, require_current


def sample_index(r, sid: str) -> int:
    ids = [str(i) for i in (r.sample_ids or [])]
    if sid not in ids:
        raise HTTPException(404, f"The sample '{sid}' is not part of this map.")
    return ids.index(sid)


def opts_of(r) -> dict:
    o = D.settings(KS.options_for(), None)
    o["limit"] = str((r.params or {}).get("validation", {}).get("limit", 99))
    return o


def unit_name(r, u) -> str:
    return r.payload["map"]["unit_names"][int(u)]


def axes_of(model) -> dict:
    out: dict = {}
    for b in model.frame.blocks:
        fe = b.element
        if fe.kind == "surface":
            out["em"], out["ex"] = np.asarray(fe.axes[0], float), np.asarray(fe.axes[1], float)
            out["mask"] = plot_mask(~np.asarray(fe.obs, bool))
            out["units"] = str(fe.units or "")
        else:
            out["abs_wl"] = np.asarray(fe.axes[0], float)
    return out


# ----------------------------------------------------------------------------- one sample
def sample_view(p, r, sid: str, thumb: bool = False) -> dict:
    i = sample_index(r, sid)
    S = r.payload["samples"]
    u = int(S["unit"][i])
    base = {"id": sid, "title": f"{sid} on the {r.title}", "unit": u, "unit_name": unit_name(r, u), "row": u // r.payload["map"]["cols"] + 1, "col": u % r.payload["map"]["cols"] + 1,
            "pos": S["pos"][i], "qe": float(S["qe"][i]), "qe_rel": float(S["qe_rel"][i]), "heldout_qe": float(S["heldout_qe"][i]), "status": str(S["status"][i]), "words": STATUS_WORDS[str(S["status"][i])],
            "flag": str(S["flag"][i]), "cluster": int(r.payload["clusters"]["sample_labels"][i])}
    if is_variables(r):
        cols = r.payload["columns"]
        vals = prep.table_for(p, r.params["input"]["keys"])
        ids = vals[1]
        row = {c: (float(vals[0][c][ids.index(sid)]) if sid in ids else None) for c in cols}
        base["variables"] = [{"name": c, "value": row[c], "prototype": float(r.payload["prototype_values"][u, k])} for k, c in enumerate(cols)]
        base["text"] = f"{sid} falls on unit {base['unit_name']}."
        return base
    require_current(p, r, "a single sample")
    model = model_of(r)
    ref = frozen_of(r)["reference"]
    data = prep.apply_data(p, [sid], model.names)
    ev = model.evaluate(data, 0.0)
    if not ev.usable[0]:
        raise HTTPException(400, " ".join(ev.why[0]) or "This sample cannot be shown against the map.")
    opt = opts_of(r)
    jd = KS.judge(model, ev, ref, opt)
    u = int(ev.unit[0])
    protos = model.prototype_functions([u])
    mean = model.mean_functions()
    blocks = []
    for k, b in enumerate(model.frame.blocks):
        y, pr = ev.treated[b.name][0], protos[b.name][0]
        entry = {"name": b.name, "kind": b.element.kind, "units": str(b.element.units or ""), "qe": float(ev.qe_blocks[0, k])}
        if b.element.kind == "surface":
            entry["measured"], entry["prototype"] = browser_images(b.domain, y, True)[0], browser_images(b.domain, pr, True)[0]
            if not thumb:
                entry["difference"] = browser_images(b.domain, np.where(np.isfinite(y), y - pr, np.nan), True)[0]
                entry["mean"] = browser_images(b.domain, mean[b.name], True)[0]
            ax = axes_of(model)
            entry.update({"ex": ax["ex"], "em": ax["em"], "mask": ax["mask"]})
        else:
            entry["x"] = np.asarray(b.element.axes[0], float)
            entry["measured"], entry["prototype"] = browser_images(b.domain, y, False)[0], browser_images(b.domain, pr, False)[0]
            if not thumb:
                entry["difference"] = browser_images(b.domain, np.where(np.isfinite(y), y - pr, np.nan), False)[0]
                entry["mean"] = browser_images(b.domain, mean[b.name], False)[0]
        blocks.append(entry)
    base.update({"blocks": blocks, "reasons": list(jd.reasons[0]), "limit": float(ref["limits"][str(opt["limit"])]), "limit_level": str(opt["limit"])})
    base["text"] = (f"{sid} falls on unit {base['unit_name']} at distance {D.g(float(ev.qe[0]))} from its prototype (inside the {opt['limit']} % limit {D.g(base['limit'])})." if jd.status[0] == "ok"
                    else f"{sid}: " + " ".join(jd.reasons[0]))
    lack = _lacking(model, ev, sid)
    if lack:
        base["missing"], base["missing_words"] = lack
    return base


def _lacking(model, ev, sid: str):
    share, words = {}, []
    for b in model.frame.blocks:
        y = np.asarray(ev.treated[b.name][0], float)
        w = np.asarray(b.domain.w, float)
        s = float(w[~np.isfinite(y)].sum() / max(w.sum(), 1e-300))
        if s < 0.005:
            continue
        share[b.name] = s
        if b.kind == "pixels":
            words.append(f"The {b.name} of {sid} lacks {100 * s:.0f} % of the domain (not measured, or cut out): the distance to the prototypes uses the pixels it has.")
        else:
            how = "from the pixels the sample has" + (" and from what the other samples show" if getattr(b.element, "gap", None) is not None else "")
            words.append(f"The {b.name} of {sid} lacks {100 * s:.0f} % of the domain (not measured, or cut out): its coordinates were estimated {how}. The measured picture is blank there.")
    return (share, " ".join(words)) if share else None


# ----------------------------------------------------------------------------- prototypes and units
def prototypes_view(p, r, what: str = "prototype") -> dict:
    """The prototype of every unit as pictures: ``prototype`` (the stored ones), ``members`` (the mean of the samples of the unit, from the current data) or ``difference`` (the second minus
    the first).  Surfaces ``(M, n_ex, n_em)``, curves ``(M, n_wl)``."""
    if what not in ("prototype", "members", "difference"):
        raise HTTPException(400, "'what' must be prototype, members or difference.")
    pl = r.payload
    M = pl["map"]["units"]
    out: dict = {"what": what, "units": M, "hits": pl["map"]["hits"]}
    if is_variables(r):
        return _variables_prototypes(p, r, what, out)
    model = model_of(r)
    out.update(axes_of(model))
    mem = None
    if what in ("members", "difference"):
        require_current(p, r, "the mean of the samples of a unit")
        mem = _member_means(p, r, model)
    out["surfaces"], out["curves"] = {}, {}
    for b in model.frame.blocks:
        surf = b.element.kind == "surface"
        proto = (pl["prototypes"] if surf else pl["prototype_curves"])[b.name]
        proto = np.asarray(proto, np.float32)
        if what == "prototype":
            arr = proto
        elif what == "members":
            arr = mem[b.name]
        else:
            arr = np.where(np.isfinite(mem[b.name]), mem[b.name] - proto, np.nan)
        if surf:
            out["surfaces"][b.name] = plot_images(np.asarray(arr, np.float32))
        else:
            out["curves"][b.name] = np.asarray(arr, np.float32)
    return out


def _member_means(p, r, model) -> dict:
    """``{block: (M, n_em, n_ex) or (M, n_wl)}`` the mean of the pre-treated samples of every unit (NaN for an empty unit), on the grid."""
    ids = list(r.sample_ids)
    ev = model.evaluate(prep.apply_data(p, ids, model.names), 0.0)
    unit = r.payload["samples"]["unit"]
    M = r.payload["map"]["units"]
    out = {}
    for b in model.frame.blocks:
        T = np.asarray(ev.treated[b.name], float)
        F = np.full((M, T.shape[1]), np.nan)
        for j in range(M):
            sel = unit == j
            if sel.any():
                with np.errstate(invalid="ignore"):
                    F[j] = np.nanmean(T[sel], axis=0)
        out[b.name] = b.domain.embed(F)
    return out


def _variables_prototypes(p, r, what: str, out: dict) -> dict:
    """A map of other variables: each unit's picture is the mean EEM of the samples of the unit and of its neighbours (weighted by closeness on the map)."""
    require_current(p, r, "the pictures of the units")
    try:
        view = p.view(only=list(r.sample_ids))
    except Exception as exc:                                                  # noqa: BLE001
        raise HTTPException(400, f"The EEMs of the samples cannot be read: {exc}") from None
    X = np.asarray(view.X, float)
    unit = r.payload["samples"]["unit"]
    coords = r.payload["map"]["coords"]
    M = r.payload["map"]["units"]
    d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2)
    H = np.exp(-d2 / 2.0)
    Z = np.full((M,) + X.shape[1:], np.nan, np.float32)
    for j in range(M):
        w = H[j][unit]
        ok = np.isfinite(X)
        num = np.nansum(X * w[:, None, None], axis=0)
        den = (ok * w[:, None, None]).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            Z[j] = np.where(den > 1e-9, num / den, np.nan)
    out.update({"ex": np.asarray(view.ex, float), "em": np.asarray(view.em, float), "units": M, "unit_units": str(view.units or ""),
                "surfaces": {"fluorescence": plot_images(Z)}, "curves": {}, "note": "Each picture is the mean EEM of the samples of the unit and of its neighbours, weighted by their closeness on the map."})
    return out


def unit_view(p, r, index: int) -> dict:
    pl = r.payload
    M = pl["map"]["units"]
    if not 0 <= index < M:
        raise HTTPException(404, f"There is no unit {index} on this map (it has {M}).")
    S = pl["samples"]
    ids = [str(x) for x in r.sample_ids]
    members = [i for i in range(len(ids)) if int(S["unit"][i]) == index]
    cols = pl["map"]["cols"]
    out = {"unit": index, "name": unit_name(r, index), "row": index // cols + 1, "col": index % cols + 1, "hits": int(pl["map"]["hits"][index]), "members": [ids[i] for i in members],
           "qe_members": [float(S["qe"][i]) for i in members], "cluster": int(pl["clusters"]["unit_labels"][index]),
           "planes": {k: (None if not np.isfinite(v["values"][index]) else float(v["values"][index])) for k, v in pl["planes"].items() if v.get("kind") == "numeric"},
           "plane_labels": {k: v["label"] for k, v in pl["planes"].items()}}
    if is_variables(r):
        out["variables"] = [{"name": c, "prototype": float(pl["prototype_values"][index, k])} for k, c in enumerate(pl["columns"])]
        return out
    model = model_of(r)
    protos = model.prototype_functions([index])
    out.update(axes_of(model))
    blocks = []
    mem = None
    if members:
        try:
            require_current(p, r, "the mean of the samples of the unit")
            ev = model.evaluate(prep.apply_data(p, [ids[i] for i in members], model.names), 0.0)
            mem = {b.name: np.nanmean(np.asarray(ev.treated[b.name], float), axis=0) for b in model.frame.blocks}
        except HTTPException:
            mem = None
    for b in model.frame.blocks:
        surf = b.element.kind == "surface"
        e = {"name": b.name, "kind": b.element.kind, "prototype": browser_images(b.domain, protos[b.name][0], surf)[0]}
        if surf:
            pass
        else:
            e["x"] = np.asarray(b.element.axes[0], float)
        if mem is not None:
            e["members_mean"] = browser_images(b.domain, mem[b.name], surf)[0]
            e["difference"] = browser_images(b.domain, mem[b.name] - protos[b.name][0], surf)[0]
        blocks.append(e)
    out["blocks"] = blocks
    return out


# ----------------------------------------------------------------------------- planes
FEATURE_KINDS = ("point", "region_mean", "region_max", "total", "absorbance")


def plane_view(p, r, body: dict) -> dict:
    """Compute or fetch a plane: ``{"kind": "stored", "key": "umatrix"}``, a feature (``point``, ``region_mean``, ``region_max``, ``total``, ``absorbance``), ``{"kind": "index", "key": "HIX"}`` or
    ``{"kind": "var", "key": "meta:site", "smooth": true}``."""
    kind = body.get("kind")
    pl = r.payload
    M = pl["map"]["units"]
    if kind == "stored":
        key = body.get("key")
        if key not in pl["planes"]:
            raise HTTPException(404, f"There is no plane '{key}' (the planes are: {', '.join(pl['planes'])}).")
        return {**pl["planes"][key], "cached": True}
    try:
        if kind in FEATURE_KINDS or kind == "index":
            if is_variables(r):
                raise SomError("Pixels, boxes and indices need a map of EEMs or spectra; for a map of other variables use the planes of the variables.")
            model = model_of(r)
            funcs = model.prototype_functions(np.arange(M))
            if kind == "index":
                plane = PLN.index_plane(model.frame, funcs, str(body.get("key", "")), pl["map"]["unit_names"])
            else:
                plane = PLN.feature_plane(model.frame, funcs, body)
        elif kind == "var":
            key = body.get("key")
            if not isinstance(key, str) or not key:
                raise SomError("Name the variable to draw (for example meta:site or res:parafac-1:C1_Fmax).")
            from fluoviz.server.routers.samples import qc_columns
            vals = p.variable_values([key], qc_columns(p) if key.startswith("qc:") else None)["columns"][key]
            pos = {s: k for k, s in enumerate(p.ids)}
            idx = [pos.get(str(s)) for s in r.sample_ids]
            if any(i is None for i in idx):
                raise SomError("Some samples of this map are no longer in the project.")
            arr = np.asarray(vals, dtype=object) if not isinstance(vals, np.ndarray) else vals
            sub = arr[idx]
            label = next((c["label"] for c in p.variable_catalogue(qc_columns(p) if key.startswith("qc:") else None) if c["key"] == key), key)
            d2 = ((pl["map"]["coords"][:, None, :] - pl["map"]["coords"][None, :, :]) ** 2).sum(axis=2)
            if pl["map"]["period"] is not None:
                Lx, Ly = pl["map"]["period"]
                dd = pl["map"]["coords"][:, None, :] - pl["map"]["coords"][None, :, :]
                for sx in (-1, 0, 1):
                    for sy in (-1, 0, 1):
                        d2 = np.minimum(d2, ((dd + np.array([sx * Lx, sy * Ly])) ** 2).sum(axis=2))
            plane = PLN.variable_plane(sub, pl["samples"]["unit"], M, d2, label, bool(body.get("smooth", False)))
            plane["variable"] = key
        else:
            raise SomError("Unknown kind of plane: use stored, point, region_mean, region_max, total, absorbance, index or var.")
    except SomError as exc:
        raise HTTPException(400, str(exc)) from None
    return plane


def planes_list(r) -> dict:
    return {"stored": [{"key": k, "label": v["label"]} for k, v in r.payload["planes"].items()]}


def index_list(r) -> list:
    if is_variables(r):
        return []
    return PLN.index_catalogue(model_of(r).frame)


# ----------------------------------------------------------------------------- groups
def groups_view(p, r) -> dict:
    """The groups with their interpretation on the current data: mean EEM / z-map of every group, profiles of the scores, associations with the metadata."""
    from fluoviz.analysis.clustering import cluster_profiles, metadata_associations
    cl = r.payload["clusters"]
    labels = np.asarray(cl["sample_labels"], int)
    ids = [str(i) for i in r.sample_ids]
    out: dict = {"k": cl["k"], "sizes": cl["sizes"], "labels": labels, "unit_labels": cl["unit_labels"], "reason": cl["reason"], "warnings": cl["warnings"], "k_values": cl["k_values"],
                 "silhouette": cl["silhouette"]}
    try:
        sc = np.column_stack([r.payload["samples"]["pos"], r.payload["samples"]["scores"][:, :3]])
        names = ["Map position, across", "Map position, down"] + [f"Principal direction {j + 1}" for j in range(r.payload["samples"]["scores"][:, :3].shape[1])]
        out["profiles"] = cluster_profiles(sc, labels, names).to_payload()
    except Exception as exc:                                                  # noqa: BLE001 - the interpretation is a nicety
        out["profiles_note"] = f"The profiles could not be computed ({exc})."
    try:
        meta = p.meta.loc[[i for i in ids if i in p.meta.index]]
        keep = [k for k, i in enumerate(ids) if i in p.meta.index]
        if len(meta.columns) and len(keep) >= 3:
            out["associations"] = metadata_associations(labels[keep], meta).to_payload()
    except Exception as exc:                                                  # noqa: BLE001
        out["associations_note"] = f"The associations with the metadata could not be computed ({exc})."
    if not is_variables(r) and "fluorescence" in r.payload["variables"]:
        try:
            require_current(p, r, "the mean EEM of a group")
            from fluoviz.analysis.clustering import cluster_mean_images
            view = p.view(only=ids)
            im = cluster_mean_images(np.asarray(view.X, float), labels, z_map=True)
            ip = im.to_payload() if hasattr(im, "to_payload") else {}
            out["images"] = {"mean": plot_images(np.asarray(im.mean_images, np.float32)), "difference": plot_images(np.asarray(im.diff_images, np.float32)),
                             "z": None if im.z_images is None else plot_images(np.asarray(im.z_images, np.float32)), "ex": np.asarray(view.ex, float), "em": np.asarray(view.em, float),
                             "n_samples": [int(x) for x in im.n_samples]}
        except HTTPException as exc:
            out["images_note"] = exc.detail
        except Exception as exc:                                              # noqa: BLE001
            out["images_note"] = f"The group pictures could not be computed ({exc})."
    return out


def protos_t_of(r) -> np.ndarray:
    if is_variables(r):
        return np.asarray(r.payload["map"]["protos_t"], float)
    return model_of(r).protos_t


def recluster(p, r, body: dict) -> dict:
    """Group the units again with other settings; the result keeps the new groups and publishes ``som_cluster``."""
    from ..core.spec import Config
    from .payload import cluster_payload
    cur = Config.from_dict({"input": (r.params or {}).get("input", {}), **{k: v for k, v in (r.params or {}).items() if k in ("cluster",)}})
    new = Config.from_dict({"cluster": {**cur.to_dict()["cluster"], **(body or {})}})
    pl = r.payload
    lat_adj = _adjacency(pl)
    grp = G.cluster_units(protos_t_of(r), pl["samples"]["unit"], lat_adj, new)
    pl["clusters"] = cluster_payload(grp, new)
    r.params = {**(r.params or {}), "cluster": new.to_dict()["cluster"]}
    lab = np.asarray(grp["unit_labels"], int)
    var = np.array([f"group {int(k) + 1}" for k in lab[pl["samples"]["unit"]]])
    r.variables["som_cluster"] = var
    r.var_info["som_cluster"] = {"kind": "category", "label": "Map group", "description": "The group of units (second-level clustering of the prototypes) the sample's unit belongs to."}
    if not is_variables(r):
        model = model_of(r)
        model.unit_cluster = lab
        from ..kinds.freeze import pack_frozen
        from ..kinds.model import SomModel
        old = pl["frozen"]
        pl["frozen"] = pack_frozen(model.to_tree(), old["reference"], old["validation"], old["description"], model)
    p.dirty.add("results")
    return {"clusters": pl["clusters"]}


def _adjacency(pl) -> np.ndarray:
    from ..core import som as S
    m = pl["map"]
    return S.make_lattice(m["rows"], m["cols"], m["lattice"], m["shape"]).adj


# ----------------------------------------------------------------------------- unusual samples, trajectories, projection
def anomalies_view(p, r, body: dict) -> dict:
    S = r.payload["samples"]
    held = np.asarray(S["heldout_qe"], float)
    qe = np.asarray(S["qe"], float)
    use = held if np.isfinite(held).any() else qe
    method = body.get("method", "mad")
    if method == "mad":
        c = float(body.get("c", 3.0))
        if not 0.5 <= c <= 20:
            raise HTTPException(400, "'c' must be between 0.5 and 20.")
        lim = D.robust_upper_limit(use[np.isfinite(use)], c)
        words = f"The limit is the median plus {c:g} times the robust spread (MAD) of the distances to the best unit."
    elif method == "quantile":
        q = float(body.get("q", 0.99))
        if not 0.5 <= q < 1:
            raise HTTPException(400, "'q' must be between 0.5 and 1.")
        lim = float(np.nanquantile(use, q))
        words = f"The limit is the {100 * q:g} % quantile of the distances to the best unit."
    elif method == "limit":
        lev = str(body.get("level", (r.params or {}).get("validation", {}).get("limit", 99)))
        if lev not in ("95", "99"):
            raise HTTPException(400, "'level' must be 95 or 99.")
        lim = float(frozen_of(r)["reference"]["limits"][lev]) if not is_variables(r) else float("nan")
        words = f"The limit is the {lev} % limit of the saved model (from samples the map was not trained on)."
    else:
        raise HTTPException(400, "'method' must be mad, quantile or limit.")
    ids = [str(i) for i in r.sample_ids]
    flagged = [{"id": ids[i], "qe": float(use[i]), "unit": unit_name(r, S["unit"][i])} for i in np.argsort(-np.nan_to_num(use, nan=-1)) if np.isfinite(use[i]) and np.isfinite(lim) and use[i] > lim]
    counts, edges = np.histogram(use[np.isfinite(use)], bins=min(30, max(5, len(use) // 3)))
    return {"method": method, "limit": lim, "words": words, "flagged": flagged, "histogram": {"counts": counts, "edges": edges}, "uses_heldout": bool(np.isfinite(held).any())}


def trajectories_view(p, r, body: dict) -> dict:
    import pandas as pd
    from fluoviz.server.routers.samples import qc_columns
    order_key = body.get("order_by")
    group_key = body.get("group_by")
    if not isinstance(order_key, str) or not order_key:
        raise HTTPException(400, "Choose the variable to order the samples by (for example a date or a time).")
    keys = [order_key] + ([group_key] if isinstance(group_key, str) and group_key else [])
    need_qc = any(k.startswith("qc:") for k in keys)
    vals = p.variable_values(keys, qc_columns(p) if need_qc else None)["columns"]
    pos = {s: k for k, s in enumerate(p.ids)}
    ids = [str(i) for i in r.sample_ids]
    idx = [pos.get(i) for i in ids]
    if any(i is None for i in idx):
        raise HTTPException(400, "Some samples of this map are no longer in the project.")
    o = np.asarray(vals[order_key], dtype=object)[idx]
    try:
        order = np.asarray([np.nan if x is None else float(x) for x in o], float)
    except (TypeError, ValueError):
        dt = pd.to_datetime(pd.Series(list(o)), errors="coerce")
        if dt.notna().sum() < 2:
            raise HTTPException(400, f"The variable '{order_key}' is neither a number nor a date, so the samples cannot be put in order by it.") from None
        order = dt.astype("int64").where(dt.notna(), np.nan).to_numpy(float)
    grp = np.array(["all"] * len(ids), dtype=object) if len(keys) == 1 else np.array([("" if x is None else str(x)) for x in np.asarray(vals[group_key], dtype=object)[idx]], dtype=object)
    posm = r.payload["samples"]["pos"]
    lines = []
    for g in sorted(set(grp.tolist())):
        sel = [i for i in range(len(ids)) if grp[i] == g and np.isfinite(order[i])]
        if len(sel) < 2:
            continue
        sel.sort(key=lambda i: order[i])
        lines.append({"group": g or "(no value)", "ids": [ids[i] for i in sel], "pos": posm[sel], "order": order[sel]})
    if not lines:
        raise HTTPException(400, "No group has two samples with a value to order by: there is no path to draw.")
    return {"lines": lines, "order_by": order_key, "group_by": group_key}


def projection_view(r) -> dict:
    pl = r.payload
    Dm = np.asarray(pl["map"]["proto_dist"], float)
    xy, stress = Q.sammon(Dm, np.asarray(pl["map"]["coords"], float))
    edges = np.asarray(pl["map"]["edges"], float)
    cross = Q.edge_crossings(xy, edges)
    words = ("The map is not folded: no two lattice edges cross when the prototypes are drawn by their distances." if cross == 0 else
             f"{cross} pair(s) of lattice edges cross when the prototypes are drawn by their distances: the map is folded in places.")
    return {"xy": xy, "stress": stress, "crossings": cross, "edges": edges[:, :2].astype(int), "words": words, "hits": pl["map"]["hits"]}
