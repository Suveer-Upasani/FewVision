# modules/feature_extraction/__init__.py
"""Feature extraction package for FewVision.

Converts reference images into high-dimensional embeddings for downstream
anomaly detection (Memory Bank / PatchCore / PaDiM).

Public API
----------
>>> from modules.feature_extraction import get_extractor
>>> extractor = get_extractor()          # uses config.FEATURE_EXTRACTOR
>>> extractor.load_model()
>>> embedding = extractor.extract(image) # np.ndarray shape (D,)
"""

from modules.feature_extraction.extractor_factory import get_extractor
from modules.feature_extraction.embedding_database import (
    save_embeddings,
    load_embeddings,
    list_sessions,
)

__all__ = [
    "get_extractor",
    "save_embeddings",
    "load_embeddings",
    "list_sessions",
]
