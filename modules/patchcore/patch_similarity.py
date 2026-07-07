# modules/patchcore/patch_similarity.py
"""Optimized similarity and distance computation for PatchCore patch search."""

from __future__ import annotations

import numpy as np


def search_patch_neighbors(
    query_embeddings: np.ndarray,
    memory_embeddings: np.ndarray,
    metric: str = "cosine",
    k: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find the *k* nearest reference patches for each query patch.

    Parameters
    ----------
    query_embeddings : np.ndarray
        Query patch embeddings, shape (Q, D), usually (196, D).
    memory_embeddings : np.ndarray
        Reference patch database, shape (M, D).
    metric : {"cosine", "euclidean"}
        The similarity metric to use.
    k : int
        Number of nearest neighbors to retrieve.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (indices, distances, similarities) each shape (Q, k).
        Sorted ascending by distance (closest first).
    """
    q_len = query_embeddings.shape[0]
    m_len = memory_embeddings.shape[0]
    k = max(1, min(k, m_len))

    if metric == "cosine":
        # Ensure query is normalized
        norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        q_norm = query_embeddings / norms

        # Cosine similarity matrix via matrix multiplication
        # shape: (Q, M)
        scores = q_norm @ memory_embeddings.T
        distances = 1.0 - scores

        # Find top k closest reference patches (smallest distance / highest score)
        top_k_idx = np.argsort(distances, axis=1)[:, :k]

        # Gather distance values
        row_indices = np.arange(q_len)[:, np.newaxis]
        top_k_dist = distances[row_indices, top_k_idx]
        top_k_sim = np.clip(1.0 - top_k_dist / 2.0, 0.0, 1.0)

        return top_k_idx.astype(np.int64), top_k_dist.astype(np.float32), top_k_sim.astype(np.float32)

    else:
        # Euclidean L2 distance
        # Use ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b
        a_sq = np.sum(query_embeddings ** 2, axis=1, keepdims=True)  # (Q, 1)
        b_sq = np.sum(memory_embeddings ** 2, axis=1, keepdims=True).T  # (1, M)
        ab = query_embeddings @ memory_embeddings.T  # (Q, M)
        
        dist_sq = np.clip(a_sq + b_sq - 2 * ab, 0.0, None)
        distances = np.sqrt(dist_sq)

        # Get top k closest reference patches
        top_k_idx = np.argsort(distances, axis=1)[:, :k]
        row_indices = np.arange(q_len)[:, np.newaxis]
        top_k_dist = distances[row_indices, top_k_idx]
        top_k_sim = 1.0 / (1.0 + top_k_dist)

        return top_k_idx.astype(np.int64), top_k_dist.astype(np.float32), top_k_sim.astype(np.float32)
