"""Two-level clustering: group the prototypes (Vesanto & Alhoniemi 2000), then every sample takes the group of its unit (SPEC section 6)."""
from __future__ import annotations

import numpy as np

from .spec import Config, SomError


def _components(adj: np.ndarray, members: np.ndarray) -> int:
    """Number of connected pieces of the units ``members`` (a bool mask) on the lattice."""
    idx = np.flatnonzero(members)
    if idx.size == 0:
        return 0
    seen = np.zeros(adj.shape[0], bool)
    pieces = 0
    for s in idx:
        if seen[s]:
            continue
        pieces += 1
        stack = [s]
        seen[s] = True
        while stack:
            u = stack.pop()
            for v in np.flatnonzero(adj[u] & members & ~seen):
                seen[v] = True
                stack.append(v)
    return pieces


def cluster_units(protos_t: np.ndarray, c1: np.ndarray, adj: np.ndarray, cfg: Config, sample_w: np.ndarray | None = None, progress=None) -> dict:
    """``{"unit_labels" (M,), "sample_labels" (n,), "k", "reason", "k_values", "silhouette", "skipped_empty", "warnings", "sizes"}``; the groups are numbered 0 … k-1 by decreasing number of
    samples."""
    from fluoviz.analysis.clustering import choose_k, cluster_scores
    M = protos_t.shape[0]
    n = c1.shape[0]
    cnt = np.bincount(c1, minlength=M)
    occ = cnt > 0
    skip = cfg.cluster_skip_empty
    if skip == "auto":
        skip = bool((~occ).mean() > 0.5)
    use = occ if skip else np.ones(M, bool)
    S = protos_t[use]
    warnings: list = []
    if S.shape[0] < 3:
        labels_u = np.zeros(M, int)
        return {"unit_labels": labels_u, "sample_labels": labels_u[c1], "k": 1, "reason": "There are too few units with samples to form groups: everything is one group.",
                "k_values": [], "silhouette": [], "skipped_empty": bool(skip), "warnings": ["Too few occupied units to group."], "sizes": [int(n)]}
    kr = (cfg.cluster_k_range[0], int(min(cfg.cluster_k_range[1], S.shape[0] - 1)))
    reason, kvals, sil = "", [], []
    if cfg.cluster_k == "auto":
        if kr[0] > kr[1]:
            kr = (2, max(2, S.shape[0] - 1))
        try:
            ck = choose_k(S, cfg.cluster_method, k_range=kr, standardize=False, seed=cfg.seed, stability=False, gap=False, progress=progress)
            k = int(ck.suggested_k or 2)
            reason = str(ck.reason or "")
            kvals = [int(v) for v in ck.k_values]
            sil = [float(v) for v in ck.silhouette]
        except Exception as exc:                                   # noqa: BLE001 - any failure of the selection falls back to a plain default
            k = 2
            reason = f"The number of groups could not be chosen automatically ({exc}); two groups were used."
    else:
        k = int(min(cfg.cluster_k, S.shape[0] - 1))
        reason = f"{k} groups, as asked."
    res = cluster_scores(S, cfg.cluster_method, n_clusters=k, standardize=False, seed=cfg.seed)
    lab = np.asarray(res.labels, int)
    lab = np.where(lab < 0, 0, lab)
    labels_u = np.full(M, -1)
    labels_u[use] = lab
    if (labels_u < 0).any():                                       # empty units that were left out join the group of the nearest unit that took part
        P = protos_t
        for j in np.flatnonzero(labels_u < 0):
            d = ((P[use] - P[j]) ** 2).sum(axis=1)
            labels_u[j] = lab[int(np.argmin(d))]
    # renumber by decreasing number of samples
    ks = sorted(set(labels_u.tolist()))
    sizes = {g: int(cnt[labels_u == g].sum()) for g in ks}
    order = sorted(ks, key=lambda g: (-sizes[g], g))
    remap = {g: i for i, g in enumerate(order)}
    labels_u = np.array([remap[g] for g in labels_u])
    for g in range(len(order)):
        if _components(adj, labels_u == g) > 1:
            warnings.append(f"Group {g + 1} is not in one piece on the map: its units are separated by units of other groups.")
    return {"unit_labels": labels_u, "sample_labels": labels_u[c1], "k": len(order), "reason": reason, "k_values": kvals, "silhouette": sil, "skipped_empty": bool(skip),
            "warnings": warnings, "sizes": [int(sizes[g]) for g in order]}
