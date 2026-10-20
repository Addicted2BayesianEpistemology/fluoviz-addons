"""The data of the open project as the map sees it: the functional variables (the Functional PCA's own), the prepared coordinates (``core.data.prepare``, cached because the basis and the
smoothing take seconds), a table of other variables, and the samples of a result as an ``ApplyData``.  What the autoencoders add-on does the same way is in the ``common`` add-on
(``fluoviz_addon_common.project_data``)."""
from __future__ import annotations

import numpy as np
from fluoviz_addon_common import project_data as C
from fluoviz_addon_common.project_data import abs_matrix, ife_of, make_variables  # noqa: F401  (part of this module's interface)

from fluoviz.analysis.frozen import ApplyData

from ..core import data as cdata
from ..core.spec import SomError
from .request import Request

_CACHE = C.PreparedCache()


def table_for(p, keys: list[str]) -> tuple[dict, list]:
    """``({key: (n,) numeric values}, ids)`` of the variables ``keys`` for the included samples (the keys of the variable catalogue: ``res:ID:name``, ``meta:column``, ``qc:key``)."""
    from fluoviz.server.routers.samples import qc_columns
    need_qc = any(k.startswith("qc:") for k in keys)
    try:
        vals = p.variable_values(keys, qc_columns(p) if need_qc else None)["columns"]
    except Exception as exc:                                                  # noqa: BLE001 - ProjectError and the like carry a plain sentence
        raise SomError(str(exc)) from None
    all_ids = list(p.ids)
    pos = {s: k for k, s in enumerate(all_ids)}
    ids = [s for s in p.included_ids() if s in pos]
    cols = {}
    for k in keys:
        v = vals[k]
        arr = np.asarray([np.nan if x is None else x for x in v]) if not isinstance(v, np.ndarray) else v
        try:
            arr = arr.astype(float)
        except (TypeError, ValueError):
            raise SomError(f"The variable '{k}' is not a number (it is a category or text): only numeric variables can be mapped.") from None
        cols[k] = arr[[pos[s] for s in ids]]
    keep = np.isfinite(np.column_stack(list(cols.values()))).any(axis=1)
    ids = [s for s, k in zip(ids, keep) if k]
    cols = {k: v[keep] for k, v in cols.items()}
    return cols, ids


def get_prepared(p, req: Request, progress=None):
    """``core.data.Prepared`` of the request on the current data; the last one is kept, so ``fit`` right after ``check`` does not redo the basis and the smoothing."""
    key = C.request_key(p, req.input, req.cfg.seed, res=sorted(p.results))

    def build():
        if req.variables_mode:
            cols, ids = table_for(p, req.input["keys"])
            return cdata.prepare_table(cols, ids, req.input["standardize"], req.cfg.seed)
        variables = make_variables(p, req.input)
        return cdata.prepare(variables, req.input, progress, seed=req.cfg.seed, ife=ife_of(p, req.input))

    return _CACHE.get(key, build)


def forget() -> None:
    _CACHE.forget()


def apply_data(p, ids: list[str], names: list[str]) -> ApplyData:
    """The samples ``ids`` of the open project in the state the map of a result (not outdated) reads them."""
    return C.apply_data(p, ids, names, "map")
