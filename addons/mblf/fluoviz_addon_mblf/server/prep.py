"""The data of the open project as the model sees it: the modes (``core.modes``) of the included samples — the fluorescence and absorbance as the Functional PCA reads them (the
``common`` add-on), the values to model and the covariates from the catalogue of variables — cached, because the basis takes seconds."""
from __future__ import annotations

import numpy as np
from fluoviz_addon_common import project_data as C

from ..core import modes as MD
from ..core.spec import MblfError
from .request import Request

_CACHE = C.PreparedCache()


def labels_of(p, keys: list[str]) -> dict:
    """What a person reads for every variable key (the catalogue's label when there is one)."""
    out = {}
    for k in keys:
        kind, _, rest = k.partition(":")
        if kind == "meta":
            out[k] = rest
        elif kind == "res":
            rid, _, name = rest.partition(":")
            try:
                r = p.get_result(rid)
                out[k] = (r.var_info.get(name) or {}).get("label") or f"{name} ({rid})"
            except Exception:                                                   # noqa: BLE001
                out[k] = f"{name} ({rid})"
        else:
            out[k] = rest or k
    return out


def values_for(p, keys: list[str], ids: list[str]) -> dict:
    """``{key: (n,) values}`` of the variables ``keys`` (numbers as floats, categories as text; missing = NaN / None) for the samples ``ids``."""
    if not keys:
        return {}
    from fluoviz.server.routers.samples import qc_columns
    need_qc = any(k.startswith("qc:") for k in keys)
    try:
        vals = p.variable_values(keys, qc_columns(p) if need_qc else None)
    except Exception as exc:                                                    # noqa: BLE001 - ProjectError and the like carry a plain sentence
        raise MblfError(str(exc)) from None
    pos = {s: k for k, s in enumerate(vals["ids"])}
    out = {}
    for k in keys:
        col = vals["columns"][k]
        arr = col if isinstance(col, np.ndarray) else np.asarray(col, dtype=object)
        out[k] = [arr[pos[s]] if s in pos else None for s in ids]
        if isinstance(col, np.ndarray):
            out[k] = np.asarray([np.nan if v is None else v for v in out[k]], float)
    return out


def build(p, req: Request, progress=None) -> MD.Prepared:
    """The modes of the request on the current data (see ``core.modes``)."""
    from fluoviz.analysis.common import Progress
    progress = progress or Progress()
    inp = req.input
    included = set(p.included_ids())
    variables = C.make_variables(p, {"variables": inp["variables"], **{n: inp[n] for n in inp["variables"]}}) if inp["variables"] else []
    notes, dropped = [], []
    for v in variables:
        dropped += [tuple(d) for d in v.dropped]
        notes += [f"{v.name}: {t}" for t in v.notes]
    # the samples: every included sample that has at least one mode (or all of them, when samples that lack a mode are not used)
    sets = [[s for s in v.ids if s in included] for v in variables]
    order = [s for s in p.ids if s in included]
    vals = values_for(p, inp["values"], order) if inp["values"] else {}
    if inp["values"]:
        has_v = np.column_stack([np.isfinite(np.asarray(vals[k], float)) for k in inp["values"]]).any(axis=1)
        sets.append([s for s, h in zip(order, has_v) if h])
    if not sets:
        raise MblfError("Nothing to model.")
    if inp["partial"]:
        keep = set().union(*map(set, sets))
    else:
        keep = set(sets[0]).intersection(*map(set, sets[1:]))
    ids = [s for s in order if s in keep]
    cov = None
    if inp["covariates"]:
        cvals = values_for(p, inp["covariates"], ids)
        cov, ok, cnotes = MD.covariates_from(cvals, labels_of(p, inp["covariates"]), ids)
        notes += cnotes
        dropped += [(s, "a covariate is missing") for s, k in zip(ids, ok) if not k]
        ids = [s for s, k in zip(ids, ok) if k]
    if len(ids) < 2:
        raise MblfError(f"Only {len(ids)} sample(s) have what the model needs.")
    modes, Y = [], []
    nf = len(variables) + (1 if inp["values"] else 0)
    for j, v in enumerate(variables):
        sub = progress.sub(j / nf, (j + 1) / nf)
        sub(0.0, f"Basis for the {v.name}")
        m, y, mn = MD.functional_mode(v, ids, inp[v.name]["basis"], sub, C.ife_of(p, {"variables": inp["variables"], **{n: inp[n] for n in inp["variables"]}}))
        modes.append(m)
        Y.append(y)
        notes += [f"{v.name}: {t}" for t in mn]
    if inp["values"]:
        vv = values_for(p, inp["values"], ids)
        bad = [k for k in inp["values"] if not isinstance(vv[k], np.ndarray)]
        if bad:
            raise MblfError(f"The variable {labels_of(p, bad)[bad[0]]} is a category or text: only numbers can be modelled as values (use it as a covariate instead).")
        m, y, mn = MD.values_mode(vv, labels_of(p, inp["values"]), ids)
        modes.append(m)
        Y.append(y)
        notes += mn
    prepared = MD.Prepared(ids, modes, Y, cov, inp, notes, dropped)
    pres = prepared.presence()
    lacking = int((~pres.all(axis=1)).sum())
    if lacking and len(modes) > 1:
        prepared.notes.append(f"{lacking} sample(s) lack a whole mode: the model uses what each of them has and predicts what it lacks.")
    progress(1.0, "Data prepared")
    return prepared


def get_prepared(p, req: Request, progress=None) -> MD.Prepared:
    """The prepared modes of the request on the current data; the last one is kept, so ``fit`` right after ``check`` does not redo the basis."""
    key = C.request_key(p, req.input, 0, res=sorted(p.results))
    return _CACHE.get(key, lambda: build(p, req, progress))


def forget() -> None:
    _CACHE.forget()
