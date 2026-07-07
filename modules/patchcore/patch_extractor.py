# modules/patchcore/patch_extractor.py
"""DINOv2 patch extractor for PatchCore industrial anomaly localization.

Extracts local patch features of shape (number_of_patches, embedding_dim)
from images in evaluation mode with gradients disabled.
"""

from __future__ import annotations

import logging
import math
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from modules.feature_extraction.dinov2_extractor import DINOv2Extractor

logger = logging.getLogger("fewvision.patchcore.extractor")

# ImageNet standard normalization stats
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PatchExtractor:
    """Extracts patch embeddings from BGR images using DINOv2.

    Parameters
    ----------
    extractor : DINOv2Extractor or None
        An existing DINOv2Extractor instance to reuse. If None, a new one is created.
    patch_size : int
        The model's patch size (default: 14).
    device : str or None
        Target device for torch models.
    """

    def __init__(
        self,
        extractor: DINOv2Extractor | None = None,
        patch_size: int = 14,
        device: str | None = None,
    ) -> None:
        if extractor is None:
            self.extractor = DINOv2Extractor(device=device)
            self.extractor.load_model()
        else:
            self.extractor = extractor
            if not self.extractor._model:
                self.extractor.load_model()

        self.patch_size = patch_size
        self.device = self.extractor._device
        self._dim = self.extractor.embedding_dim

        # Target size for 14x14 grid = 196 patches (14 * 14 = 196)
        self.target_size = (196, 196)

        # Preprocessing pipeline
        self.transform = T.Compose([
            T.Resize(self.target_size, interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess a single BGR image into a model-ready tensor.

        Parameters
        ----------
        image : np.ndarray
            BGR image.

        Returns
        -------
        torch.Tensor
            Preprocessed tensor, shape (1, 3, 196, 196), on target device.
        """
        if image is None or image.size == 0:
            raise ValueError("PatchExtractor received an empty image.")

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        # BGR -> RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        return self.transform(pil_img).unsqueeze(0).to(self.device)

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract patch features from a single BGR image.

        Parameters
        ----------
        image : np.ndarray
            BGR image.

        Returns
        -------
        np.ndarray
            Shape (196, embedding_dim), float32.
        """
        self.extractor._require_model()
        tensor = self.preprocess(image)

        with torch.no_grad():
            features = self.extractor._model.forward_features(tensor)
            # x_norm_patchtokens shape is (1, 196, dim)
            patch_tokens = features["x_norm_patchtokens"].squeeze(0)

        return patch_tokens.cpu().numpy().astype(np.float32)

    def extract_batch(
        self,
        images: list[np.ndarray],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Extract patch features from a list of images in mini-batches.

        Parameters
        ----------
        images : list[np.ndarray]
            List of BGR images.
        batch_size : int
            Number of images processed per forward pass.

        Returns
        -------
        np.ndarray
            Shape (N, 196, embedding_dim), float32.
        """
        self.extractor._require_model()
        n = len(images)
        if n == 0:
            return np.empty((0, 196, self._dim), dtype=np.float32)

        n_batches = math.ceil(n / batch_size)
        all_embeddings: list[np.ndarray] = []

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n)
            batch_imgs = images[start:end]

            tensors = [self.preprocess(img).squeeze(0) for img in batch_imgs]
            batch_tensor = torch.stack(tensors, dim=0)

            with torch.no_grad():
                features = self.extractor._model.forward_features(batch_tensor)
                # x_norm_patchtokens shape: (B, 196, dim)
                batch_patches = features["x_norm_patchtokens"]

            all_embeddings.append(batch_patches.cpu().numpy().astype(np.float32))

        return np.concatenate(all_embeddings, axis=0)
