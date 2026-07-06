# modules/anomaly_detection/similarity.py
"""Reusable similarity and nearest-neighbour utilities for FewVision.

All functions are stateless and operate on NumPy arrays, making them trivially
testable and easy to swap for FAISS-backed implementations without touching
any calling code.

Supported metrics
-----------------
``"cosine"``
    Cosine similarity converted to a distance in ``[0, 2]``.
    A distance of ``0`` means identical vectors; ``2`` means opposite.
    **Assumes L2-normalised inputs** — if inputs are not pre-normalised,
    call :func:`modules.anomaly_detection.memory_bank.MemoryBank.search`
    which normalises the query automatically.

``"euclidean"``
    Standard Euclidean (L2) distance.

Public API
----------
compute_similarity(a, b, metric)    → float
nearest_neighbor(query, memory, metric) → (index, distance, score)
top_k_neighbors(query, memory, k, metric) → (indices, distances, scores)
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

logger = logging.getLogger("fewvision.anomaly_detection.similarity")

# Type alias for the two supported metrics
Metric = Literal["cosine", "euclidean"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_metric(metric: str) -> None:
    """Raise ``ValueError`` if *metric* is not a supported option."""
    if metric not in ("cosine", "euclidean"):
        raise ValueError(
            f"Unsupported similarity metric '{metric}'. "
            "Choose 'cosine' or 'euclidean'."
        )


def _l2_distances(query: np.ndarray, memory: np.ndarray) -> np.ndarray:
    """Compute Euclidean distances from *query* (1-D) to every row of *memory*.

    Parameters
    ----------
    query : np.ndarray
        Shape ``(D,)``.
    memory : np.ndarray
        Shape ``(N, D)``.

    Returns
    -------
    np.ndarray
        Shape ``(N,)``, float32 distances.
    """
    diff = memory - query[np.newaxis, :]          # (N, D)
    return np.sqrt(np.sum(diff ** 2, axis=1))     # (N,)


def _cosine_distances(query: np.ndarray, memory: np.ndarray) -> np.ndarray:
    """Compute cosine distances from *query* (1-D) to every row of *memory*.

    For L2-normalised vectors this reduces to ``1 - dot(query, memory[i])``,
    which is O(N·D) with no extra allocations beyond the dot product.

    Parameters
    ----------
    query : np.ndarray
        Shape ``(D,)``.  Assumed to be L2-normalised.
    memory : np.ndarray
        Shape ``(N, D)``.  Assumed to be L2-normalised.

    Returns
    -------
    np.ndarray
        Shape ``(N,)``, cosine distances in ``[0, 2]``.
    """
    dot_products = memory @ query                 # (N,)  fast BLAS path
    return 1.0 - dot_products                     # cosine distance


def _distances_to_scores(distances: np.ndarray, metric: str) -> np.ndarray:
    """Convert raw distances to similarity scores in ``[0, 1]``.

    For cosine: distance is already in ``[0, 2]``, so score = 1 - dist / 2.
    For euclidean: score = 1 / (1 + dist)  (bounded in (0, 1]).

    Parameters
    ----------
    distances : np.ndarray
        Raw distance values.
    metric : str
        ``"cosine"`` or ``"euclidean"``.

    Returns
    -------
    np.ndarray
        Similarity scores in ``[0, 1]``.
    """
    if metric == "cosine":
        return np.clip(1.0 - distances / 2.0, 0.0, 1.0)
    # euclidean
    return 1.0 / (1.0 + distances)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_similarity(
    a: np.ndarray,
    b: np.ndarray,
    metric: Metric = "cosine",
) -> float:
    """Compute the similarity score between two embedding vectors.

    Parameters
    ----------
    a : np.ndarray
        1-D embedding vector, shape ``(D,)``.
    b : np.ndarray
        1-D embedding vector, shape ``(D,)``.
    metric : {"cosine", "euclidean"}
        Distance metric to use.

    Returns
    -------
    float
        Similarity score in ``[0, 1]`` (higher = more similar).

    Raises
    ------
    ValueError
        If *metric* is not supported or vectors have mismatched shapes.
    """
    _validate_metric(metric)

    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()

    if a.shape != b.shape:
        raise ValueError(
            f"Embedding shape mismatch: {a.shape} vs {b.shape}."
        )

    if metric == "cosine":
        # Robust cosine similarity (handles non-normalised inputs)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    # euclidean → convert distance to score
    dist = float(np.linalg.norm(a - b))
    return float(1.0 / (1.0 + dist))


def nearest_neighbor(
    query: np.ndarray,
    memory: np.ndarray,
    metric: Metric = "cosine",
) -> tuple[int, float, float]:
    """Find the single nearest neighbour in *memory* to *query*.

    Parameters
    ----------
    query : np.ndarray
        1-D query embedding, shape ``(D,)``.
        **Must be L2-normalised** for cosine metric.
    memory : np.ndarray
        2-D memory matrix, shape ``(N, D)``.
        **Must be L2-normalised** for cosine metric.
    metric : {"cosine", "euclidean"}
        Distance metric to use.

    Returns
    -------
    tuple[int, float, float]
        ``(index, distance, score)`` where:
        - ``index`` — row index of the nearest neighbour in *memory*
        - ``distance`` — raw distance value
        - ``score`` — similarity score in ``[0, 1]``

    Raises
    ------
    ValueError
        If *metric* is unsupported or *memory* is empty.
    """
    indices, distances, scores = top_k_neighbors(query, memory, k=1, metric=metric)
    return int(indices[0]), float(distances[0]), float(scores[0])


def top_k_neighbors(
    query: np.ndarray,
    memory: np.ndarray,
    k: int = 5,
    metric: Metric = "cosine",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find the *k* nearest neighbours in *memory* to *query*.

    Parameters
    ----------
    query : np.ndarray
        1-D query embedding, shape ``(D,)``.
        **Must be L2-normalised** for cosine metric.
    memory : np.ndarray
        2-D memory matrix, shape ``(N, D)``.
        **Must be L2-normalised** for cosine metric.
    k : int
        Number of nearest neighbours to return.  Clamped to ``[1, N]``.
    metric : {"cosine", "euclidean"}
        Distance metric to use.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(indices, distances, scores)`` where each array has shape ``(k,)``
        and is sorted ascending by distance (nearest first).

    Raises
    ------
    ValueError
        If *metric* is unsupported, *memory* is empty, or shapes are
        incompatible.
    """
    _validate_metric(metric)

    query = np.asarray(query, dtype=np.float32).ravel()
    memory = np.asarray(memory, dtype=np.float32)

    if memory.ndim != 2:
        raise ValueError(
            f"memory must be 2-D (N, D), got shape {memory.shape}."
        )

    n = memory.shape[0]
    if n == 0:
        raise ValueError("memory is empty — cannot perform nearest-neighbour search.")

    if query.shape[0] != memory.shape[1]:
        raise ValueError(
            f"Dimension mismatch: query has dim {query.shape[0]}, "
            f"memory has dim {memory.shape[1]}."
        )

    k = max(1, min(k, n))

    # Compute all pairwise distances in one vectorised pass
    if metric == "cosine":
        distances = _cosine_distances(query, memory)
    else:
        distances = _l2_distances(query, memory)

    # Partial sort: only fully sort the top-k portion
    if k < n:
        top_k_idx = np.argpartition(distances, k)[:k]
        top_k_idx = top_k_idx[np.argsort(distances[top_k_idx])]
    else:
        top_k_idx = np.argsort(distances)

    top_k_distances = distances[top_k_idx]
    top_k_scores = _distances_to_scores(top_k_distances, metric)

    return (
        top_k_idx.astype(np.int64),
        top_k_distances.astype(np.float32),
        top_k_scores.astype(np.float32),
    )
