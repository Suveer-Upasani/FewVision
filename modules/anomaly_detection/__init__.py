# modules/anomaly_detection/__init__.py
"""Anomaly detection package for FewVision.

This package provides the Memory Bank construction stage and supporting
utilities that bridge the Embedding Database with future anomaly detection
algorithms (PatchCore, PaDiM, etc.).

Pipeline position
-----------------
    Embedding Database
          │
          ▼
    Memory Bank  ←  this package
          │
          ▼
    Similarity Engine  ←  this package
          │
          ▼
    [ PatchCore / PaDiM — future ]

Public API
----------
>>> from modules.anomaly_detection import MemoryBank
>>> bank = MemoryBank()
>>> bank.build(session_id)
>>> bank.save(session_id)
>>> results = bank.search(query_embedding, k=5)

>>> from modules.anomaly_detection import compute_anomaly_score
>>> score = compute_anomaly_score(distances=[0.12, 0.15], nearest_index=0)
"""

from modules.anomaly_detection.memory_bank import MemoryBank
from modules.anomaly_detection.anomaly_score import compute_anomaly_score
from modules.anomaly_detection.similarity import (
    compute_similarity,
    nearest_neighbor,
    top_k_neighbors,
)

__all__ = [
    "MemoryBank",
    "compute_anomaly_score",
    "compute_similarity",
    "nearest_neighbor",
    "top_k_neighbors",
]
