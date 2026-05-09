"""Personalized PageRank over the saved code graph using scipy sparse power iteration."""
from __future__ import annotations

import json

import numpy as np
import scipy.sparse as sp


def load_graph(graph_path: str) -> tuple[sp.csr_matrix, dict, dict]:
    """Load node_link_data graph JSON, build column-stochastic CSR adjacency.

    Returns (A_sparse, id_to_idx, idx_to_id).
    """
    with open(graph_path) as f:
        data = json.load(f)

    nodes = data.get("nodes", [])
    links = data.get("links", data.get("edges", []))

    id_to_idx: dict[str, int] = {n["id"]: i for i, n in enumerate(nodes)}
    idx_to_id: dict[int, str] = {i: n["id"] for i, n in enumerate(nodes)}
    n = len(nodes)

    if n == 0:
        return sp.csr_matrix((0, 0)), id_to_idx, idx_to_id

    rows, cols = [], []
    for link in links:
        src = link.get("source")
        tgt = link.get("target")
        if src in id_to_idx and tgt in id_to_idx:
            # column j → row i means column-stochastic A@r propagates mass from j to i
            rows.append(id_to_idx[tgt])
            cols.append(id_to_idx[src])

    if rows:
        data_vals = np.ones(len(rows), dtype=np.float64)
        A = sp.csr_matrix((data_vals, (rows, cols)), shape=(n, n))
    else:
        A = sp.csr_matrix((n, n), dtype=np.float64)

    # Column-normalize
    col_sums = np.asarray(A.sum(axis=0)).flatten()
    col_sums[col_sums == 0] = 1.0
    inv_col = sp.diags(1.0 / col_sums)
    A = A @ inv_col
    A = A.tocsr()

    return A, id_to_idx, idx_to_id


def ppr(
    A: sp.csr_matrix,
    seed_ids: list[str],
    id_to_idx: dict[str, int],
    idx_to_id: dict[int, str],
    alpha: float = 0.85,
    top_k: int = 50,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> list[dict]:
    """Power-iteration PPR seeded on seed_ids.

    Returns [{"id": str, "score": float}] sorted descending, top_k non-seed nodes.
    """
    n = A.shape[0]
    if n == 0:
        return []

    valid_indices = [id_to_idx[sid] for sid in seed_ids if sid in id_to_idx]
    if not valid_indices:
        return []

    teleport = np.zeros(n, dtype=np.float64)
    for idx in valid_indices:
        teleport[idx] = 1.0
    teleport /= teleport.sum()

    r = teleport.copy()
    for _ in range(max_iter):
        r_new = alpha * A @ r + (1.0 - alpha) * teleport
        if np.linalg.norm(r_new - r, ord=1) < tol:
            r = r_new
            break
        r = r_new

    seed_set = set(valid_indices)
    results = [
        {"id": idx_to_id[i], "score": float(r[i])}
        for i in range(n)
        if i not in seed_set
    ]
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]
