# modules/feature_extraction/preprocessing.py
"""Image preprocessing pipeline for DINOv2-compatible feature extraction.

Implements the **exact** preprocessing spec used when training DINOv2
(ViT-S/14, ViT-B/14, ViT-L/14):

    1. Convert BGR (OpenCV) → RGB
    2. Resize shortest side to 256 px  (preserving aspect ratio)
    3. Centre-crop to 224 × 224 px
    4. Convert to float tensor in [0, 1]
    5. Normalise with ImageNet mean / std

References
----------
* https://github.com/facebookresearch/dinov2
* Meta AI DINOv2 paper (Oquab et al., 2023)

Notes
-----
This module is intentionally **stateless** — it exports a single function
``preprocess_image`` and a ``build_transform`` helper.  Any extractor that
shares the same spec (CLIP ViT-L/14, standard ViT-B/16) can reuse this
directly.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torchvision.transforms as T

# ---------------------------------------------------------------------------
# DINOv2 preprocessing constants (must match training configuration exactly)
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
RESIZE_SIZE   = 256   # resize shorter side to this
CROP_SIZE     = 224   # centre-crop to this


def build_transform() -> T.Compose:
    """Return the DINOv2 evaluation preprocessing pipeline.

    Builds a ``torchvision.transforms.Compose`` object that accepts a
    ``PIL.Image`` or ``torch.Tensor`` and returns a normalised tensor.
    The output shape is ``(3, 224, 224)``.

    Returns
    -------
    T.Compose
        Preprocessing transform.
    """
    return T.Compose([
        T.Resize(RESIZE_SIZE, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(CROP_SIZE),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# Module-level singleton — built once and reused for every call
_TRANSFORM: T.Compose | None = None


def _get_transform() -> T.Compose:
    global _TRANSFORM
    if _TRANSFORM is None:
        _TRANSFORM = build_transform()
    return _TRANSFORM


def preprocess_image(
    image: np.ndarray,
    device: str = "cpu",
) -> torch.Tensor:
    """Preprocess a single BGR image into a DINOv2-ready tensor.

    Parameters
    ----------
    image : np.ndarray
        Input image in **BGR** format (as returned by ``cv2.imread``).
        Shape must be ``(H, W, 3)`` or ``(H, W)`` (greyscale, will be
        converted to 3-channel).
    device : str
        Target device for the output tensor (``"cpu"``, ``"cuda"``, ``"mps"``).

    Returns
    -------
    torch.Tensor
        Batched tensor ready for model inference, shape ``(1, 3, 224, 224)``.

    Raises
    ------
    ValueError
        If the image array is empty or has an unsupported number of channels.
    """
    if image is None or image.size == 0:
        raise ValueError("preprocess_image received an empty image array.")

    # --- Ensure 3-channel BGR ---
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.shape[2] != 3:
        raise ValueError(f"Unsupported channel count: {image.shape[2]}")

    # --- BGR → RGB (OpenCV loads BGR; DINOv2 was trained on RGB) ---
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # --- Convert to PIL for torchvision transforms ---
    from PIL import Image as _PILImage
    pil_img = _PILImage.fromarray(rgb)

    # --- Apply transform ---
    tensor = _get_transform()(pil_img)          # (3, 224, 224)
    return tensor.unsqueeze(0).to(device)        # (1, 3, 224, 224)


def preprocess_batch(
    images: list[np.ndarray],
    device: str = "cpu",
) -> torch.Tensor:
    """Preprocess a list of BGR images into a batched tensor.

    Parameters
    ----------
    images : list[np.ndarray]
        List of BGR images.  Each image may have a different size — they
        are all resized and cropped to ``(224, 224)`` during preprocessing.
    device : str
        Target device.

    Returns
    -------
    torch.Tensor
        Batched tensor, shape ``(N, 3, 224, 224)``.
    """
    tensors = [preprocess_image(img, device="cpu").squeeze(0) for img in images]
    batch = torch.stack(tensors, dim=0)   # (N, 3, 224, 224)
    return batch.to(device)


def preprocessing_config() -> dict:
    """Return the preprocessing configuration as a JSON-serialisable dict.

    Intended for inclusion in ``extractor_info.json``.
    """
    return {
        "resize_size": RESIZE_SIZE,
        "crop_size": CROP_SIZE,
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "color_space": "RGB",
        "interpolation": "BICUBIC",
    }
