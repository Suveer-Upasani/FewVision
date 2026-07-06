# modules/feature_extraction/embedding_database.py
"""Embedding database — saves and loads the FewVision embedding store.

All embeddings for a session are stored under::

    data/embeddings/{session_id}/
        embeddings.npy       — float32, shape (N, D)
        filenames.json       — list of N filenames
        metadata.json        — per-image quality / content / augmentation info
        extractor_info.json  — extractor name, dim, model variant, preprocessing

Public API
----------
save_embeddings(session_id, embeddings, filenames, metadata, extractor_info)
    → str   (path to embeddings.npy)

load_embeddings(session_id)
    → tuple[np.ndarray, list[str], dict, dict]

list_sessions()
    → list[str]
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np

logger = logging.getLogger("fewvision.feature_extraction.db")

# ---------------------------------------------------------------------------
# Resolve embedding root from config (with fallback)
# ---------------------------------------------------------------------------

def _embeddings_root() -> str:
    try:
        import config as _cfg
        return _cfg.EMBEDDINGS_FOLDER
    except (ImportError, AttributeError):
        return os.path.join(os.path.dirname(__file__), "..", "..", "data", "embeddings")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _session_dir(session_id: str) -> str:
    return os.path.join(_embeddings_root(), session_id)


def _ensure(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_embeddings(
    session_id: str,
    embeddings: np.ndarray,
    filenames: list[str],
    metadata: list[dict],
    extractor_info: dict,
) -> str:
    """Persist the embedding database to disk.

    Parameters
    ----------
    session_id : str
        Session identifier (matches the upload / augmentation session).
    embeddings : np.ndarray
        2-D float32 array of shape ``(N, D)``.
    filenames : list[str]
        Ordered list of N image filenames corresponding to each row.
    metadata : list[dict]
        Per-image metadata (quality score, content score, augmentations …).
        Must have the same length as ``filenames``.
    extractor_info : dict
        Extractor configuration from :attr:`BaseExtractor.info`.

    Returns
    -------
    str
        Absolute path to the saved ``embeddings.npy`` file.

    Raises
    ------
    ValueError
        If ``embeddings``, ``filenames``, and ``metadata`` have mismatched
        lengths, or if ``embeddings`` is not a 2-D array.
    """
    # --- Validation ---
    if embeddings.ndim != 2:
        raise ValueError(
            f"embeddings must be 2-D (N, D), got shape {embeddings.shape}"
        )
    n = len(filenames)
    if embeddings.shape[0] != n:
        raise ValueError(
            f"embeddings has {embeddings.shape[0]} rows but filenames has {n} items."
        )
    if len(metadata) != n:
        raise ValueError(
            f"metadata has {len(metadata)} items but filenames has {n} items."
        )

    session_dir = _ensure(_session_dir(session_id))

    # --- embeddings.npy ---
    npy_path = os.path.join(session_dir, "embeddings.npy")
    np.save(npy_path, embeddings.astype(np.float32))

    # --- filenames.json ---
    filenames_path = os.path.join(session_dir, "filenames.json")
    with open(filenames_path, "w") as f:
        json.dump(filenames, f, indent=2)

    # --- metadata.json ---
    metadata_path = os.path.join(session_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # --- extractor_info.json ---
    info_path = os.path.join(session_dir, "extractor_info.json")
    with open(info_path, "w") as f:
        json.dump(extractor_info, f, indent=2)

    logger.info(
        "Embedding database saved — session=%s, shape=%s, dir=%s",
        session_id,
        embeddings.shape,
        session_dir,
    )
    return npy_path


def load_embeddings(
    session_id: str,
) -> tuple[np.ndarray, list[str], list[dict], dict]:
    """Load a previously saved embedding database.

    Parameters
    ----------
    session_id : str
        Session identifier.

    Returns
    -------
    tuple
        ``(embeddings, filenames, metadata, extractor_info)``

    Raises
    ------
    FileNotFoundError
        If the session directory or any required file is missing.
    """
    session_dir = _session_dir(session_id)

    if not os.path.isdir(session_dir):
        raise FileNotFoundError(
            f"No embedding database found for session '{session_id}' "
            f"at '{session_dir}'."
        )

    # --- embeddings.npy ---
    npy_path = os.path.join(session_dir, "embeddings.npy")
    if not os.path.isfile(npy_path):
        raise FileNotFoundError(f"embeddings.npy missing in '{session_dir}'.")
    embeddings = np.load(npy_path)

    # --- filenames.json ---
    filenames_path = os.path.join(session_dir, "filenames.json")
    if not os.path.isfile(filenames_path):
        raise FileNotFoundError(f"filenames.json missing in '{session_dir}'.")
    with open(filenames_path) as f:
        filenames: list[str] = json.load(f)

    # --- metadata.json ---
    metadata_path = os.path.join(session_dir, "metadata.json")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"metadata.json missing in '{session_dir}'.")
    with open(metadata_path) as f:
        metadata: list[dict] = json.load(f)

    # --- extractor_info.json ---
    info_path = os.path.join(session_dir, "extractor_info.json")
    extractor_info: dict = {}
    if os.path.isfile(info_path):
        with open(info_path) as f:
            extractor_info = json.load(f)

    logger.info(
        "Embedding database loaded — session=%s, shape=%s",
        session_id,
        embeddings.shape,
    )
    return embeddings, filenames, metadata, extractor_info


def list_sessions() -> list[str]:
    """Return all session IDs that have a saved embedding database.

    Returns
    -------
    list[str]
        Sorted list of session directory names.
    """
    root = _embeddings_root()
    if not os.path.isdir(root):
        return []
    return sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    )


def embedding_summary(session_id: str) -> dict:
    """Return a lightweight summary without loading the full array.

    Reads only the JSON sidecar files, not the ``.npy`` binary.

    Parameters
    ----------
    session_id : str
        Session identifier.

    Returns
    -------
    dict
        Keys: ``session_id``, ``count``, ``extractor_name``,
        ``embedding_dim``, ``npy_size_mb``.
    """
    session_dir = _session_dir(session_id)

    result: dict = {"session_id": session_id}

    # Filenames count (lightweight)
    filenames_path = os.path.join(session_dir, "filenames.json")
    if os.path.isfile(filenames_path):
        with open(filenames_path) as f:
            result["count"] = len(json.load(f))

    # Extractor info
    info_path = os.path.join(session_dir, "extractor_info.json")
    if os.path.isfile(info_path):
        with open(info_path) as f:
            info = json.load(f)
        result["extractor_name"] = info.get("extractor_name", "unknown")
        result["embedding_dim"]  = info.get("embedding_dim", "unknown")

    # File size
    npy_path = os.path.join(session_dir, "embeddings.npy")
    if os.path.isfile(npy_path):
        result["npy_size_mb"] = round(os.path.getsize(npy_path) / 1_048_576, 2)

    return result
