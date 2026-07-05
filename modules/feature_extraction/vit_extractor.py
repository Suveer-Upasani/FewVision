# modules/feature_extraction/vit_extractor.py
"""Vision Transformer (ViT) feature extractor for FewVision.

Loads a pretrained torchvision ViT-B/16 model and extracts CLS-token
embeddings. The model is loaded once and reused for every subsequent call.
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

logger = logging.getLogger("fewvision.feature_extraction.vit")


class ViTExtractor(BaseExtractor):
    """Vision Transformer (ViT) feature extractor.

    Parameters
    ----------
    model_variant : str
        The model variant string. Supported options: ``"vit_b_16"``.
    device : str or None
        Target device. ``None`` triggers automatic detection.

    Raises
    ------
    ValueError
        If ``model_variant`` is not ``"vit_b_16"``.
    """

    def __init__(
        self,
        model_variant: str = "vit_b_16",
        device: str | None = None,
    ) -> None:
        super().__init__(device=device)

        if model_variant != "vit_b_16":
            raise ValueError(
                f"Unsupported model variant '{model_variant}'. "
                "Only 'vit_b_16' is supported by this extractor."
            )

        self._model_variant = model_variant
        self._dim = 768
        self._model: torch.nn.Module | None = None

        # Build transform with BILINEAR interpolation to match torchvision's ViT training config
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
        return f"vit/{self._model_variant}"

    @property
    def info(self) -> dict:
        """Extended info dict for ``extractor_info.json``."""
        base = super().info
        base.update({
            "model_variant": self._model_variant,
            "hub_repo": "torchvision.models",
            "output_token": "cls_token",
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
        """Load the pretrained torchvision ViT-B/16 model.

        Model loading is idempotent. The classification head is replaced
        with Identity to return raw CLS token embeddings.
        """
        if self._model is not None:
            logger.debug("ViT model already loaded — skipping.")
            return

        logger.info(
            "Loading torchvision ViT model '%s' onto device '%s' …",
            self._model_variant,
            self._device,
        )

        try:
            import torchvision.models as models

            # Load model with explicit IMAGENET1K_V1 weights
            weights = models.ViT_B_16_Weights.IMAGENET1K_V1
            model = models.vit_b_16(weights=weights)

            # Replace heads with Identity to output CLS token features directly
            model.heads = torch.nn.Identity()

        except Exception as exc:
            raise RuntimeError(
                f"Failed to load torchvision ViT model '{self._model_variant}'. "
                f"Original error: {exc}"
            ) from exc

        model.eval()
        model.to(self._device)
        self._model = model

        logger.info(
            "ViT '%s' loaded — dim=%d, device=%s",
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
            raise ValueError("ViTExtractor.preprocess received an empty image array.")

        # Ensure 3-channel BGR
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif image.shape[2] != 3:
            raise ValueError(f"Unsupported channel count: {image.shape[2]}")

        # BGR -> RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image for torchvision transforms
        from PIL import Image as _PILImage
        pil_img = _PILImage.fromarray(rgb)

        # Apply transforms and add batch dimension
        tensor = self._transform(pil_img)
        return tensor.unsqueeze(0).to(self._device)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract CLS-token embedding from a single image.

        Parameters
        ----------
        image : np.ndarray
            BGR image from ``cv2.imread``.

        Returns
        -------
        np.ndarray
            1-D float32 array, shape ``(768,)``.
        """
        self._require_model()

        tensor = self.preprocess(image)

        with torch.no_grad():
            embedding = self._model(tensor)

        return embedding.squeeze(0).cpu().numpy().astype(np.float32)

    def extract_batch(
        self,
        images: list[np.ndarray],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Extract CLS-token embeddings in mini-batches.

        Parameters
        ----------
        images : list[np.ndarray]
            List of BGR images.
        batch_size : int
            Number of images processed per forward pass.

        Returns
        -------
        np.ndarray
            Shape ``(N, 768)``, float32.
        """
        self._require_model()

        n = len(images)
        if n == 0:
            return np.empty((0, self._dim), dtype=np.float32)

        n_batches = math.ceil(n / batch_size)
        all_embeddings: list[np.ndarray] = []

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n)
            batch_imgs = images[start:end]

            logger.info(
                "Extracting batch %d/%d (images %d–%d)",
                batch_idx + 1, n_batches, start + 1, end,
            )

            # Preprocess all images in the current batch
            tensors = [self.preprocess(img).squeeze(0) for img in batch_imgs]
            batch_tensor = torch.stack(tensors, dim=0).to(self._device)

            with torch.no_grad():
                batch_emb = self._model(batch_tensor)

            all_embeddings.append(batch_emb.cpu().numpy().astype(np.float32))

        return np.concatenate(all_embeddings, axis=0)
