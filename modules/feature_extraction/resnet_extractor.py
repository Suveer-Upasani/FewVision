# modules/feature_extraction/resnet_extractor.py
"""ResNet50 feature extractor for FewVision."""

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


class ResNet50Extractor(BaseExtractor):
    """ResNet50 feature extractor.

    Parameters
    ----------
    model_variant : str
        The model variant string. Supported options: ``"resnet50"``.
    device : str or None
        Target device. ``None`` triggers automatic detection.

    Raises
    ------
    ValueError
        If ``model_variant`` is not ``"resnet50"``.
    """

    def __init__(
        self,
        model_variant: str = "resnet50",
        device: str | None = None,
    ) -> None:
        super().__init__(device=device)

        if model_variant != "resnet50":
            raise ValueError(
                f"Unsupported model variant '{model_variant}'. "
                "Only 'resnet50' is supported by this extractor."
            )

        self._model_variant = model_variant
        self._dim = 2048
        self._model: torch.nn.Module | None = None

        self._transform = T.Compose([
            T.Resize(RESIZE_SIZE, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(CROP_SIZE),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def extractor_name(self) -> str:
        return f"resnet/{self._model_variant}"

    @property
    def info(self) -> dict:
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

    def load_model(self) -> None:
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

            # Replace classification head with Identity
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

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        if image is None or image.size == 0:
            raise ValueError("ResNet50Extractor.preprocess received an empty image array.")

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif image.shape[2] != 3:
            raise ValueError(f"Unsupported channel count: {image.shape[2]}")

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        from PIL import Image as _PILImage
        pil_img = _PILImage.fromarray(rgb)

        tensor = self._transform(pil_img)
        return tensor.unsqueeze(0).to(self._device)

    def extract(self, image: np.ndarray) -> np.ndarray:
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

            tensors = [self.preprocess(img).squeeze(0) for img in batch_imgs]
            batch_tensor = torch.stack(tensors, dim=0).to(self._device)

            with torch.no_grad():
                batch_emb = self._model(batch_tensor)

            all_embeddings.append(batch_emb.cpu().numpy().astype(np.float32))

        return np.concatenate(all_embeddings, axis=0)
