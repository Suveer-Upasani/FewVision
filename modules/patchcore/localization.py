# modules/patchcore/localization.py
"""Defect localization by thresholding distance maps, extracting contours, and computing region statistics."""

from __future__ import annotations

import cv2
import numpy as np


def localize_defects(
    distance_map: np.ndarray,
    original_shape: tuple[int, int],
    threshold: float = 0.5,
) -> dict:
    """Threshold upscaled distance map, extract contours, and compute anomaly statistics.

    Parameters
    ----------
    distance_map : np.ndarray
        Distance map of shape (14, 14) from patch similarities.
    original_shape : tuple[int, int]
        Shape (H, W) of the original image.
    threshold : float
        Anomalous patch distance threshold.

    Returns
    -------
    dict
        Contains:
        - "bbox": [ymin, xmin, ymax, xmax] of the largest defect contour (or [0, 0, 0, 0])
        - "area_percent": percentage of total image area marked as anomalous
        - "max_score": the maximum patch distance value (float)
        - "center": [cy, cx] coordinates of the largest defect's centroid
    """
    h, w = original_shape

    # 1. Upscale distance map to original image shape
    upscaled = cv2.resize(distance_map, (w, h), interpolation=cv2.INTER_CUBIC)

    # 2. Threshold to create binary defect mask
    binary_mask = (upscaled > threshold).astype(np.uint8) * 255

    # 3. Find defect contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bbox = [0, 0, 0, 0]
    center = [0, 0]

    if contours:
        # Find largest defect contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w_box, h_box = cv2.boundingRect(largest_contour)
        
        # Format as [ymin, xmin, ymax, xmax]
        bbox = [int(y), int(x), int(y + h_box), int(x + w_box)]

        # Calculate centroid moments
        M = cv2.moments(largest_contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = int(x + w_box / 2), int(y + h_box / 2)
        center = [cy, cx]

    # Calculate overall anomalous area percentage
    total_pixels = h * w
    anomalous_pixels = np.sum(binary_mask > 0)
    area_percent = float((anomalous_pixels / total_pixels) * 100.0)
    max_score = float(np.max(distance_map))

    return {
        "bbox": bbox,
        "area_percent": area_percent,
        "max_score": max_score,
        "center": center,
    }
