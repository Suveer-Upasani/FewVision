# modules/utils/image_utils.py
"""Image utility helpers for the FewVision pipeline."""

import os
import cv2
import numpy as np
from typing import Optional

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_image(path: str) -> np.ndarray:
    """Load an image from disk using OpenCV.

    Parameters
    ----------
    path : str
        Absolute or relative path to the image file.

    Returns
    -------
    np.ndarray
        Image in BGR format.

    Raises
    ------
    FileNotFoundError
        If the image cannot be loaded.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return img


def load_image_rgb(path: str) -> np.ndarray:
    """Load an image and convert to RGB.

    Parameters
    ----------
    path : str
        Path to the image file.

    Returns
    -------
    np.ndarray
        Image in RGB format.
    """
    img = load_image(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def is_valid_image(filename: str) -> bool:
    """Check whether a filename has a supported image extension.

    Parameters
    ----------
    filename : str
        Filename or path to check.

    Returns
    -------
    bool
    """
    return os.path.splitext(filename)[1].lower() in VALID_EXTENSIONS


def collect_images(folder: str) -> list[str]:
    """Recursively collect all valid image paths from a directory.

    Parameters
    ----------
    folder : str
        Root directory to search.

    Returns
    -------
    list[str]
        Sorted list of absolute image paths.
    """
    paths = []
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            if is_valid_image(f):
                paths.append(os.path.join(root, f))
    return paths


def resize_to_fit(img: np.ndarray, max_dim: int = 1024) -> np.ndarray:
    """Resize an image so its longest dimension is at most ``max_dim`` pixels.

    Aspect ratio is preserved.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    max_dim : int
        Maximum allowed size for width or height.

    Returns
    -------
    np.ndarray
        Resized image, or original if already within bounds.
    """
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
