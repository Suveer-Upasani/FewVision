# modules/patchcore/heatmap.py
"""Visualizes anomaly distance maps by creating high-resolution color heatmaps and blended overlays."""

from __future__ import annotations

import cv2
import numpy as np


def generate_heatmap(
    image_path: str,
    distance_map: np.ndarray,
    alpha: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    """Upscale anomaly distance map, apply colormap, and generate blended overlay.

    Parameters
    ----------
    image_path : str
        Path to original BGR test image.
    distance_map : np.ndarray
        Distance map of shape (14, 14) from patch similarities.
    alpha : float
        Blending factor for the heatmap overlay (0.0 = only original image, 1.0 = only heatmap).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (color_heatmap, overlay_image) - both in BGR format.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image at {image_path}")

    h, w, c = img.shape

    # 1. Upscale distance map to original image shape
    # Using INTER_CUBIC for premium look (smooth gradients)
    upscaled = cv2.resize(distance_map, (w, h), interpolation=cv2.INTER_CUBIC)

    # 2. Min-max normalize distance map to [0, 255] for visual contrast
    min_val = upscaled.min()
    max_val = upscaled.max()
    if max_val - min_val > 1e-5:
        norm_map = ((upscaled - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        norm_map = np.zeros_like(upscaled, dtype=np.uint8)

    # 3. Apply JET colormap to single-channel normalized map
    color_heatmap = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)

    # 4. Blend colormap overlay with original image
    # overlay = (1 - alpha) * original + alpha * heatmap
    overlay = cv2.addWeighted(img, 1.0 - alpha, color_heatmap, alpha, 0)

    return color_heatmap, overlay
