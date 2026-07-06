# modules/feature_extraction/resnet_extractor.py
"""ResNet50 feature extractor for FewVision.

Loads a pretrained ResNet50 model from ``torchvision.models`` and extracts
global average-pooled embeddings (2048-dim) by replacing the classification
head with ``torch.nn.Identity()``.  The model is loaded **once** and reused
for every subsequent call, making batch extraction efficient.

Model variants
--------------
+------------------+--------+----------+------+
| model_variant    | Arch   | Weights  | Dim  |
+==================+========+==========+======+
| resnet50         | ResNet | IN1K V1  | 2048 |
+------------------+--------+----------+------+

Default: ``resnet50`` — strong supervised ImageNet-1K baseline, excellent
for industrial inspection tasks with texture-rich surfaces.

References
----------
* He et al. (2016) — https://arxiv.org/abs/1512.03385
* torchvision ResNet — https://pytorch.org/vision/stable/models/resnet.html
"""

from __future__ import annotations

import logging
import math
from typing import Any

import cv2
import numpy as np
import torch
import torchvision.transforms as T

from modules.feature_extraction.base_extractor import BaseExtractor
from modules.feature_extraction.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    RESIZE_SIZE,
    CROP_SIZE,
)

logger = logging.getLogger("fewvision.feature_extraction.resnet")

# Supported variants and their embedding dimensions
_VARIANT_TO_DIM: dict[str, int] = {
    "resnet50": 2048,
}


class ResNet50Extractor(BaseExtractor):
    """ResNet50 feature extractor.

    Parameters
    ----------
    model_variant : str
        The model variant string. Currently only ``"resnet50"`` is supported.
    device : str or None
        Target device.  ``None`` triggers automatic detection.

    Raises
    ------
    ValueError
        If ``model_variant`` is not in :data:`_VARIANT_TO_DIM`.
    """

    def __init__(
        self,
        model_variant: str = "resnet50",
        device: str | None = None,
    ) -> None:
        super().__init__(device=device)

        if model_variant not in _VARIANT_TO_DIM:
            raise ValueError(
                f"Unknown ResNet variant '{model_variant}'. "
                f"Valid options: {list(_VARIANT_TO_DIM)}"
            )

        self._model_variant = model_variant
        self._dim = _VARIANT_TO_DIM[model_variant]
        self._model: torch.nn.Module | None = None

        # ResNet uses BILINEAR resize (torchvision default for ResNet eval)
        self._transform = T.Compose([
            T.Resize(RESIZE_SIZE, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(CROP_SIZE),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    # ------------------------------------------------------------------
    # BaseExtractor interface
    # ------------------------------------------------------------------

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def extractor_name(self) -> str:
        return f"resnet/{self._model_variant}"

    @property
    def info(self) -> dict:
        """Extended info dict for ``extractor_info.json``."""
        base = super().info
        base.update({
            "model_variant": self._model_variant,
            "hub_repo": "torchvision.models",
            "output_token": "global_avg_pool",
            "preprocessing": {
                "resize_size": RESIZE_SIZE,
                "crop_size": CROP_SIZE,
                "mean": list(IMAGENET_MEAN),
                "std": list(IMAGENET_STD),
                "color_space": "RGB",
                "interpolation": "BILINEAR",
            },
        })
        return base

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load the pretrained torchvision ResNet50 model.

        Model loading is idempotent.  The classification head (``model.fc``)
        is replaced with ``torch.nn.Identity()`` so the forward pass returns
        raw 2048-dim global average-pooled embeddings.

        Weights source: ``ResNet50_Weights.IMAGENET1K_V1`` (~98 MB).
        Downloaded once and cached in ``~/.cache/torch``.
        """
        if self._model is not None:
            logger.debug("ResNet50 model already loaded — skipping.")
            return

        logger.info(
            "Loading torchvision ResNet model '%s' onto device '%s' …",
            self._model_variant,
            self._device,
        )

        try:
            import torchvision.models as models

            weights = models.ResNet50_Weights.IMAGENET1K_V1
            model = models.resnet50(weights=weights)

            # Strip classification head — output is now (B, 2048) from avgpool
            model.fc = torch.nn.Identity()

        except Exception as exc:
            raise RuntimeError(
                f"Failed to load torchvision ResNet model '{self._model_variant}'. "
                f"Original error: {exc}"
            ) from exc

        model.eval()
        model.to(self._device)
        self._model = model

        logger.info(
            "ResNet '%s' loaded — dim=%d, device=%s",
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
        if image is None or image.size == 0:
            raise ValueError("ResNet50Extractor.preprocess received an empty image array.")

        # Ensure 3-channel BGR
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif image.shape[2] != 3:
            raise ValueError(f"Unsupported channel count: {image.shape[2]}")

        # BGR → RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image for torchvision transforms
        from PIL import Image as _PILImage
        pil_img = _PILImage.fromarray(rgb)

        tensor = self._transform(pil_img)
        return tensor.unsqueeze(0).to(self._device)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract a global-average-pool embedding from a single image.

        Parameters
        ----------
        image : np.ndarray
            BGR image from ``cv2.imread``.

        Returns
        -------
        np.ndarray
            1-D float32 array, shape ``(2048,)``.
        """
        self._require_model()

        tensor = self.preprocess(image)          # (1, 3, 224, 224)

        with torch.no_grad():
            embedding = self._model(tensor)      # (1, 2048)

        return embedding.squeeze(0).cpu().numpy().astype(np.float32)  # (2048,)

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
            Shape ``(N, 2048)``, float32.
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

            # Preprocess all images in the current batch
            tensors = [self.preprocess(img).squeeze(0) for img in batch_imgs]
            batch_tensor = torch.stack(tensors, dim=0).to(self._device)

            with torch.no_grad():
                batch_emb = self._model(batch_tensor)   # (B, 2048)

            all_embeddings.append(batch_emb.cpu().numpy().astype(np.float32))

        return np.concatenate(all_embeddings, axis=0)   # (N, 2048)
