# modules/anomaly_detection/memory_bank.py
"""Memory Bank for FewVision anomaly detection.

The Memory Bank is a searchable, L2-normalised embedding store constructed
from the Embedding Database.  It serves as the reference library that future
anomaly detection algorithms (PatchCore, PaDiM) will query at inference time.

Storage layout
--------------
    data/memory_bank/{session_id}/
        memory.npy              — float32, shape (N, D), L2-normalised
        memory_metadata.json    — session_id, filenames, extractor info, counts
        config.json             — metric, k, thresholds captured at build time

Design decisions
----------------
* **FAISS-ready API**: the internal index is currently a plain NumPy array.
  Replacing it with a FAISS ``IndexFlatIP`` index only requires changing the
  private ``_build_index`` and ``_search_index`` methods — zero pipeline code
  changes are needed.

* **Dimension-agnostic**: the embedding dimension is read from the extractor
  info stored by the Embedding Database.  No dimension is hardcoded.

* **Idempotent build**: calling :meth:`build` twice is safe — the second call
  simply rebuilds from the same on-disk embeddings.

* **Independent storage**: the Memory Bank is written to a separate directory
  from the Embedding Database so both can coexist without conflict.

Public API
----------
MemoryBank()
    .build(session_id)           → MemoryBank (chainable)
    .save(session_id)            → str  (path to memory.npy)
    .load(session_id)            → MemoryBank (chainable)
    .search(query, k)            → dict
    .size()                      → int
    .embedding_dimension()       → int
    .is_built                    → bool
    .session_id                  → str | None
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger("fewvision.anomaly_detection.memory_bank")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _memory_bank_root() -> str:
    """Return the root directory for all memory bank data.

    Reads from :attr:`config.MEMORY_BANK_FOLDER` with a safe fallback.
    """
    try:
        import config as _cfg
        return _cfg.MEMORY_BANK_FOLDER
    except (ImportError, AttributeError):
        return os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "memory_bank"
        )


def _session_dir(session_id: str) -> str:
    return os.path.join(_memory_bank_root(), session_id)


def _ensure(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _get_config_snapshot() -> dict:
    """Capture the current similarity config for traceability."""
    try:
        import config as _cfg
        return {
            "similarity_metric": getattr(_cfg, "SIMILARITY_METRIC", "cosine"),
            "top_k_neighbors": getattr(_cfg, "TOP_K_NEIGHBORS", 5),
            "anomaly_thresholds": getattr(
                _cfg, "ANOMALY_THRESHOLDS",
                {"normal_max": 0.20, "suspicious_max": 0.50},
            ),
        }
    except ImportError:
        return {
            "similarity_metric": "cosine",
            "top_k_neighbors": 5,
            "anomaly_thresholds": {"normal_max": 0.20, "suspicious_max": 0.50},
        }


# ---------------------------------------------------------------------------
# Internal index helpers  (swap these two methods to integrate FAISS)
# ---------------------------------------------------------------------------

def _build_index(embeddings: np.ndarray) -> np.ndarray:
    """Build the internal search index from normalised embeddings.

    Currently returns the numpy array directly (brute-force search).
    Replace with ``faiss.IndexFlatIP(dim)`` + ``index.add(embeddings)``
    when FAISS is available.

    Parameters
    ----------
    embeddings : np.ndarray
        L2-normalised float32 array, shape ``(N, D)``.

    Returns
    -------
    np.ndarray
        The index object (currently the same array).
    """
    return embeddings


def _search_index(index: np.ndarray, query: np.ndarray, k: int) -> tuple:
    """Search the internal index for the *k* nearest neighbours.

    Currently delegates to :func:`modules.anomaly_detection.similarity.top_k_neighbors`.
    Replace the body with ``index.search(query[np.newaxis, :], k)`` when FAISS
    is integrated.

    Parameters
    ----------
    index : np.ndarray
        The index returned by :func:`_build_index`.
    query : np.ndarray
        L2-normalised 1-D query vector, shape ``(D,)``.
    k : int
        Number of nearest neighbours.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(indices, distances, scores)`` — each shape ``(k,)``.
    """
    from modules.anomaly_detection.similarity import top_k_neighbors

    try:
        import config as _cfg
        metric = getattr(_cfg, "SIMILARITY_METRIC", "cosine")
    except ImportError:
        metric = "cosine"

    return top_k_neighbors(query, index, k=k, metric=metric)


# ---------------------------------------------------------------------------
# MemoryBank
# ---------------------------------------------------------------------------

class MemoryBank:
    """Searchable reference memory built from a FewVision embedding session.

    Lifecycle
    ---------
    1. Construct → ``MemoryBank()``
    2. Build from embedding DB → ``.build(session_id)``
    3. Persist to disk → ``.save(session_id)``
    4. (Later) Load from disk → ``.load(session_id)``
    5. Query → ``.search(query_embedding, k=5)``

    Thread-safety
    -------------
    Read operations (:meth:`search`, :meth:`size`, :meth:`embedding_dimension`)
    are safe to call from multiple threads after :meth:`build` or :meth:`load`.
    Write operations (:meth:`build`, :meth:`save`, :meth:`load`) are not
    thread-safe; guard them externally if concurrent access is required.
    """

    def __init__(self) -> None:
        self._embeddings: Optional[np.ndarray] = None   # (N, D), float32, L2-norm
        self._index: Optional[np.ndarray] = None        # FAISS-ready slot
        self._filenames: list[str] = []
        self._extractor_info: dict = {}
        self._session_id: Optional[str] = None
        self._config_snapshot: dict = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_built(self) -> bool:
        """``True`` if the memory bank has been built or loaded."""
        return self._embeddings is not None

    @property
    def session_id(self) -> Optional[str]:
        """Session ID associated with the current memory bank, or ``None``."""
        return self._session_id

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, session_id: str) -> "MemoryBank":
        """Construct the memory bank from an existing embedding session.

        Loads the embedding database produced by the feature extraction stage,
        L2-normalises all embeddings, and builds the search index.

        Parameters
        ----------
        session_id : str
            Session identifier that has a completed embedding database at
            ``data/embeddings/{session_id}/``.

        Returns
        -------
        MemoryBank
            ``self`` — allows method chaining.

        Raises
        ------
        FileNotFoundError
            If no embedding database exists for *session_id*.
        ValueError
            If the loaded embeddings array is not 2-D.
        """
        logger.info("Starting Memory Bank construction for session %s …", session_id)

        # Import here to avoid circular imports at module load time
        from modules.feature_extraction.embedding_database import load_embeddings

        embeddings, filenames, _metadata, extractor_info = load_embeddings(session_id)

        if embeddings.ndim != 2:
            raise ValueError(
                f"Expected 2-D embedding array, got shape {embeddings.shape}."
            )

        n, d = embeddings.shape
        logger.info("Loaded %d embeddings (dim=%d) from session %s", n, d, session_id)

        # L2 normalise — required for cosine distance via dot product
        logger.info("Normalizing embeddings …")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Guard against zero-norm vectors (degenerate embeddings)
        norms = np.where(norms == 0.0, 1.0, norms)
        normalised = (embeddings / norms).astype(np.float32)

        # Build the search index
        logger.info("Building searchable memory …")
        self._index = _build_index(normalised)
        self._embeddings = normalised
        self._filenames = filenames
        self._extractor_info = extractor_info
        self._session_id = session_id
        self._config_snapshot = _get_config_snapshot()

        logger.info(
            "Memory Bank built — %d embeddings, dim=%d, metric=%s",
            n, d, self._config_snapshot.get("similarity_metric", "cosine"),
        )
        return self

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, session_id: str) -> str:
        """Persist the memory bank to disk.

        Writes three files to ``data/memory_bank/{session_id}/``:

        * ``memory.npy``             — L2-normalised embeddings
        * ``memory_metadata.json``   — filenames, extractor info, counts
        * ``config.json``            — metric, k, thresholds at build time

        Parameters
        ----------
        session_id : str
            Session identifier.  Used to construct the output directory.

        Returns
        -------
        str
            Absolute path to the saved ``memory.npy`` file.

        Raises
        ------
        RuntimeError
            If :meth:`build` (or :meth:`load`) has not been called first.
        """
        if not self.is_built:
            raise RuntimeError(
                "MemoryBank.save() called before build() or load(). "
                "Call build(session_id) first."
            )

        logger.info("Saving Memory Bank for session %s …", session_id)

        out_dir = _ensure(_session_dir(session_id))

        # --- memory.npy ---
        npy_path = os.path.join(out_dir, "memory.npy")
        np.save(npy_path, self._embeddings)

        # --- memory_metadata.json ---
        meta = {
            "session_id": session_id,
            "count": int(self._embeddings.shape[0]),
            "embedding_dim": int(self._embeddings.shape[1]),
            "filenames": self._filenames,
            "extractor_info": self._extractor_info,
        }
        meta_path = os.path.join(out_dir, "memory_metadata.json")
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2, default=str)

        # --- config.json ---
        cfg_path = os.path.join(out_dir, "config.json")
        with open(cfg_path, "w") as fh:
            json.dump(self._config_snapshot, fh, indent=2)

        logger.info(
            "Memory Bank saved — %d embeddings → %s",
            self._embeddings.shape[0], out_dir,
        )
        return npy_path

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, session_id: str) -> "MemoryBank":
        """Load a previously saved memory bank from disk.

        Parameters
        ----------
        session_id : str
            Session identifier.

        Returns
        -------
        MemoryBank
            ``self`` — allows method chaining.

        Raises
        ------
        FileNotFoundError
            If the memory bank directory or ``memory.npy`` is missing.
        """
        session_dir = _session_dir(session_id)

        if not os.path.isdir(session_dir):
            raise FileNotFoundError(
                f"No memory bank found for session '{session_id}' "
                f"at '{session_dir}'."
            )

        npy_path = os.path.join(session_dir, "memory.npy")
        if not os.path.isfile(npy_path):
            raise FileNotFoundError(
                f"memory.npy missing in '{session_dir}'."
            )

        embeddings = np.load(npy_path)
        self._embeddings = embeddings.astype(np.float32)
        self._index = _build_index(self._embeddings)
        self._session_id = session_id

        # Load metadata if present
        meta_path = os.path.join(session_dir, "memory_metadata.json")
        if os.path.isfile(meta_path):
            with open(meta_path) as fh:
                meta = json.load(fh)
            self._filenames = meta.get("filenames", [])
            self._extractor_info = meta.get("extractor_info", {})

        # Load config snapshot if present
        cfg_path = os.path.join(session_dir, "config.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path) as fh:
                self._config_snapshot = json.load(fh)

        logger.info(
            "Memory Bank loaded — session=%s, shape=%s",
            session_id, self._embeddings.shape,
        )
        return self

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: np.ndarray,
        k: Optional[int] = None,
    ) -> dict:
        """Search the memory bank for the *k* nearest neighbours of *query*.

        The query vector is L2-normalised before searching, matching the
        normalisation applied during :meth:`build`.

        Parameters
        ----------
        query : np.ndarray
            1-D raw (un-normalised) query embedding, shape ``(D,)``.
        k : int, optional
            Number of neighbours to return.  Defaults to
            ``config.TOP_K_NEIGHBORS`` (or 5 if config is unavailable).

        Returns
        -------
        dict
            ``{
                "indices":   list[int],    # memory bank row indices
                "distances": list[float],  # sorted ascending
                "scores":    list[float],  # similarity scores in [0,1]
            }``

        Raises
        ------
        RuntimeError
            If :meth:`build` or :meth:`load` has not been called.
        ValueError
            If *query* has a different dimension from the stored embeddings.
        """
        if not self.is_built:
            raise RuntimeError(
                "MemoryBank.search() called before build() or load()."
            )

        if k is None:
            k = self._config_snapshot.get("top_k_neighbors", 5)
            try:
                import config as _cfg
                k = getattr(_cfg, "TOP_K_NEIGHBORS", k)
            except ImportError:
                pass

        query_arr = np.asarray(query, dtype=np.float32).ravel()
        norm = np.linalg.norm(query_arr)
        if norm > 0.0:
            query_arr = query_arr / norm

        indices, distances, scores = _search_index(self._index, query_arr, k=k)

        return {
            "indices": indices.tolist(),
            "distances": distances.tolist(),
            "scores": scores.tolist(),
        }

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def size(self) -> int:
        """Return the number of embeddings stored in the memory bank.

        Returns
        -------
        int
            Number of reference embeddings.  ``0`` if not yet built.
        """
        if self._embeddings is None:
            return 0
        return int(self._embeddings.shape[0])

    def embedding_dimension(self) -> int:
        """Return the dimensionality of the stored embeddings.

        Returns
        -------
        int
            Embedding dimension ``D``.  ``0`` if not yet built.
        """
        if self._embeddings is None:
            return 0
        return int(self._embeddings.shape[1])

    def summary(self) -> dict:
        """Return a JSON-serialisable summary dict.

        Useful for API responses and dashboard rendering.

        Returns
        -------
        dict
            Keys: ``session_id``, ``count``, ``embedding_dim``,
            ``similarity_metric``, ``top_k``, ``is_built``.
        """
        return {
            "session_id": self._session_id,
            "count": self.size(),
            "embedding_dim": self.embedding_dimension(),
            "similarity_metric": self._config_snapshot.get("similarity_metric", "cosine"),
            "top_k": self._config_snapshot.get("top_k_neighbors", 5),
            "is_built": self.is_built,
        }

    def __repr__(self) -> str:
        status = (
            f"built, n={self.size()}, dim={self.embedding_dimension()}"
            if self.is_built
            else "not built"
        )
        return f"MemoryBank(session='{self._session_id}', {status})"


# ---------------------------------------------------------------------------
# Module-level convenience helpers
# ---------------------------------------------------------------------------

def load_memory_bank_summary(session_id: str) -> dict:
    """Return a lightweight summary for a saved memory bank.

    Reads only the JSON sidecar files — does **not** load the ``.npy`` binary.

    Parameters
    ----------
    session_id : str
        Session identifier.

    Returns
    -------
    dict
        ``{"session_id", "count", "embedding_dim", "similarity_metric",
           "top_k", "npy_size_mb", "location"}``
        Missing fields are omitted if the files are not present.
    """
    session_dir = _session_dir(session_id)
    result: dict = {"session_id": session_id, "location": session_dir}

    meta_path = os.path.join(session_dir, "memory_metadata.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as fh:
            meta = json.load(fh)
        result["count"] = meta.get("count", 0)
        result["embedding_dim"] = meta.get("embedding_dim", 0)

    cfg_path = os.path.join(session_dir, "config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path) as fh:
            cfg = json.load(fh)
        result["similarity_metric"] = cfg.get("similarity_metric", "cosine")
        result["top_k"] = cfg.get("top_k_neighbors", 5)

    npy_path = os.path.join(session_dir, "memory.npy")
    if os.path.isfile(npy_path):
        result["npy_size_mb"] = round(os.path.getsize(npy_path) / 1_048_576, 2)

    return result


def list_memory_bank_sessions() -> list[str]:
    """Return all session IDs that have a saved memory bank.

    Returns
    -------
    list[str]
        Sorted list of session directory names.
    """
    root = _memory_bank_root()
    if not os.path.isdir(root):
        return []
    return sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    )
