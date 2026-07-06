# modules/feature_extraction/base_extractor.py
"""Abstract base class for all feature extractors in FewVision.

Every concrete extractor (DINOv2, CLIP, ResNet50, ViT …) must inherit from
:class:`BaseExtractor` and implement the four abstract methods.  Pipeline
code depends **only** on this interface — never on a concrete implementation.

Design rationale
----------------
* ``load_model`` is kept separate from ``__init__`` so the model weights are
  not downloaded/loaded until explicitly requested.  This lets the factory
  create extractor objects without triggering a network download.
* ``preprocess`` is exposed as an abstract method so each extractor can own
  the exact preprocessing spec required by its model.
* ``extract_batch`` provides the primary performance path; ``extract`` is a
  convenience wrapper for single-image use.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

logger = logging.getLogger("fewvision.feature_extraction")


class BaseExtractor(ABC):
    """Abstract interface for image feature extractors.

    Parameters
    ----------
    device : str or None
        PyTorch device string (``"cuda"``, ``"mps"``, ``"cpu"``).
        ``None`` triggers automatic detection at :meth:`load_model` time.

    Attributes
    ----------
    _model : Any or None
        The loaded model.  ``None`` before :meth:`load_model` is called.
    _device : str
        Resolved device string (set during :meth:`load_model`).
    """

    def __init__(self, device: str | None = None) -> None:
        self._model: Any = None
        self._device: str = device or self._auto_device()

    # ------------------------------------------------------------------
    # Device helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _auto_device() -> str:
        """Return the best available torch device.

        Priority: ``cuda`` → ``mps`` (Apple Silicon) → ``cpu``.
        """
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    # ------------------------------------------------------------------
    # Abstract interface — all subclasses must implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def load_model(self) -> None:
        """Download (if necessary) and load the model into memory.

        Must set ``self._model`` to a non-None value.
        Must be idempotent — calling it twice should not reload the model.
        """

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> "torch.Tensor":
        """Transform a raw BGR image into a model-ready tensor.

        Parameters
        ----------
        image : np.ndarray
            BGR image loaded by OpenCV, shape ``(H, W, 3)`` or ``(H, W)``.

        Returns
        -------
        torch.Tensor
            Batched tensor ready for model inference, shape ``(1, C, H, W)``.
        """

    @abstractmethod
    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract a single embedding from one image.

        Parameters
        ----------
        image : np.ndarray
            BGR image, shape ``(H, W, 3)``.

        Returns
        -------
        np.ndarray
            1-D embedding vector, shape ``(D,)`` where D = :attr:`embedding_dim`.
        """

    @abstractmethod
    def extract_batch(
        self,
        images: list[np.ndarray],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Extract embeddings from a list of images.

        The model is called once per batch.  A progress bar is written to
        the logger at INFO level.

        Parameters
        ----------
        images : list[np.ndarray]
            List of BGR images (may have different sizes — preprocessing
            normalises them).
        batch_size : int
            Number of images per forward pass.

        Returns
        -------
        np.ndarray
            2-D array of shape ``(N, D)``.
        """

    # ------------------------------------------------------------------
    # Abstract properties
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Output embedding dimensionality (e.g. 384 for DINOv2 ViT-S/14)."""

    @property
    @abstractmethod
    def extractor_name(self) -> str:
        """Human-readable extractor identifier stored in ``extractor_info.json``."""

    # ------------------------------------------------------------------
    # Concrete helpers available to all subclasses
    # ------------------------------------------------------------------

    def _require_model(self) -> None:
        """Raise if :meth:`load_model` has not been called yet."""
        if self._model is None:
            raise RuntimeError(
                f"{self.__class__.__name__}.load_model() must be called before "
                "extract() or extract_batch()."
            )

    @property
    def info(self) -> dict:
        """Return a JSON-serialisable dict for ``extractor_info.json``."""
        return {
            "extractor_name": self.extractor_name,
            "embedding_dim": self.embedding_dim,
            "device": self._device,
        }

    def __repr__(self) -> str:
        status = "loaded" if self._model is not None else "not loaded"
        return (
            f"{self.__class__.__name__}("
            f"extractor='{self.extractor_name}', "
            f"dim={self.embedding_dim}, "
            f"device='{self._device}', "
            f"model={status})"
        )
