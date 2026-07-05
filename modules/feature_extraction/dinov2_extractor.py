# modules/feature_extraction/dinov2_extractor.py
"""DINOv2 feature extractor for FewVision.

Loads a pretrained DINOv2 model via ``torch.hub`` and extracts CLS-token
embeddings from images.  The model is loaded **once** and reused for every
subsequent call, making batch extraction efficient.

Model variants
--------------
+-------------------+-------+----------+-------+
| model_variant     | Arch  | Params   | Dim   |
+===================+=======+==========+=======+
| dinov2_vits14     | ViT-S |  21 M    |  384  |
| dinov2_vitb14     | ViT-B |  86 M    |  768  |
| dinov2_vitl14     | ViT-L | 307 M    | 1024  |
| dinov2_vitg14     | ViT-G | 1.1 B    | 1536  |
+-------------------+-------+----------+-------+

Default: ``dinov2_vits14`` — fastest model, excellent quality for industrial
anomaly detection baselines on datasets with < 1000 images.

References
----------
* Meta AI DINOv2 — https://github.com/facebookresearch/dinov2
* Oquab et al. (2023) — https://arxiv.org/abs/2304.07193
"""

from __future__ import annotations

import logging
import math
from typing import Any

import cv2
import numpy as np
import torch

from modules.feature_extraction.base_extractor import BaseExtractor
from modules.feature_extraction.preprocessing import (
    preprocess_image,
    preprocess_batch,
    preprocessing_config,
)

logger = logging.getLogger("fewvision.feature_extraction.dinov2")

# Maps model variant string → output embedding dimension
_VARIANT_TO_DIM: dict[str, int] = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}

_HUB_REPO = "facebookresearch/dinov2"


class DINOv2Extractor(BaseExtractor):
    """DINOv2 feature extractor.

    Parameters
    ----------
    model_variant : str
        One of the DINOv2 model identifiers (default: ``"dinov2_vits14"``).
    device : str or None
        Target device.  ``None`` triggers automatic detection.

    Raises
    ------
    ValueError
        If ``model_variant`` is not a known DINOv2 variant.
    """

    def __init__(
        self,
        model_variant: str = "dinov2_vits14",
        device: str | None = None,
    ) -> None:
        super().__init__(device=device)

        if model_variant not in _VARIANT_TO_DIM:
            raise ValueError(
                f"Unknown DINOv2 variant '{model_variant}'. "
                f"Valid options: {list(_VARIANT_TO_DIM)}"
            )

        self._model_variant = model_variant
        self._dim = _VARIANT_TO_DIM[model_variant]
        self._model: torch.nn.Module | None = None

    # ------------------------------------------------------------------
    # BaseExtractor interface
    # ------------------------------------------------------------------

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def extractor_name(self) -> str:
        return f"dinov2/{self._model_variant}"

    @property
    def info(self) -> dict:
        """Extended info dict for ``extractor_info.json``."""
        base = super().info
        base.update({
            "model_variant": self._model_variant,
            "hub_repo": _HUB_REPO,
            "output_token": "cls_token",
            "preprocessing": preprocessing_config(),
        })
        return base

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load the DINOv2 model via ``torch.hub``.

        The weights are downloaded on the first call (~85 MB for ViT-S/14)
        and cached in ``~/.cache/torch/hub``.  Subsequent calls are instant.

        This method is **idempotent** — calling it more than once is safe.
        """
        if self._model is not None:
            logger.debug("DINOv2 model already loaded — skipping.")
            return

        logger.info(
            "Loading DINOv2 model '%s' onto device '%s' …",
            self._model_variant,
            self._device,
        )

        try:
            model = torch.hub.load(
                _HUB_REPO,
                self._model_variant,
                pretrained=True,
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load DINOv2 model '{self._model_variant}' via "
                f"torch.hub. Check your internet connection on first run. "
                f"Original error: {exc}"
            ) from exc

        model.eval()
        model.to(self._device)
        self._model = model

        logger.info(
            "DINOv2 '%s' loaded — dim=%d, device=%s",
            self._model_variant,
            self._dim,
            self._device,
        )

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess a single BGR image into a model-ready tensor.

        Parameters
        ----------
        image : np.ndarray
            BGR image from ``cv2.imread``.

        Returns
        -------
        torch.Tensor
            Shape ``(1, 3, 224, 224)``, on ``self._device``.
        """
        return preprocess_image(image, device=self._device)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract a CLS-token embedding from a single image.

        Parameters
        ----------
        image : np.ndarray
            BGR image from ``cv2.imread``.

        Returns
        -------
        np.ndarray
            1-D float32 array, shape ``(D,)``.
        """
        self._require_model()

        tensor = self.preprocess(image)          # (1, 3, 224, 224)

        with torch.no_grad():
            embedding = self._model(tensor)      # (1, D)

        return embedding.squeeze(0).cpu().numpy().astype(np.float32)  # (D,)

    def extract_batch(
        self,
        images: list[np.ndarray],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Extract embeddings for a list of images in mini-batches.

        Parameters
        ----------
        images : list[np.ndarray]
            List of BGR images.
        batch_size : int
            Number of images processed per forward pass.

        Returns
        -------
        np.ndarray
            Shape ``(N, D)``, float32.
        """
        self._require_model()

        n = len(images)
        if n == 0:
            return np.empty((0, self._dim), dtype=np.float32)

        n_batches = math.ceil(n / batch_size)
        all_embeddings: list[np.ndarray] = []

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end   = min(start + batch_size, n)
            batch_imgs = images[start:end]

            logger.info(
                "Extracting batch %d/%d  (images %d–%d)",
                batch_idx + 1, n_batches, start + 1, end,
            )

            batch_tensor = preprocess_batch(batch_imgs, device=self._device)

            with torch.no_grad():
                batch_emb = self._model(batch_tensor)   # (B, D)

            all_embeddings.append(batch_emb.cpu().numpy().astype(np.float32))

        return np.concatenate(all_embeddings, axis=0)   # (N, D)
