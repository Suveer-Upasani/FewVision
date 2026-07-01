# content_analysis.py
"""Content analysis utilities for the computer‑vision pipeline.

The module provides a `ContentAnalyzer` class that extracts a set of
high‑level descriptors from a single image.  All methods operate on the
image loaded from `image_path` using OpenCV (cv2) and NumPy.

The returned dictionary matches the specification in the user's
proposal.
"""

import cv2
import numpy as np
import os
from typing import Dict

class ContentAnalyzer:
    """Analyze the *content* of an image.

    Mirrors the structure of `ImageQualityChecker` – instantiated with the
    path to an image and provides an ``analyze`` method that returns a flat
    dictionary of metrics.
    """

    def __init__(self, image_path: str):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Cannot open image: {image_path}")
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.height, self.width = self.gray.shape

    # ---------------------------------------------------------------------
    # 1. Background complexity – Simple / Moderate / Complex
    # ---------------------------------------------------------------------
    def _background_complexity(self) -> str:
        # Edge density using Canny
        edges = cv2.Canny(self.gray, 100, 200)
        edge_density = np.sum(edges > 0) / edges.size
        # Entropy of grayscale histogram
        hist = cv2.calcHist([self.gray], [0], None, [256], [0, 256]).ravel()
        hist_norm = hist / hist.sum()
        entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-12))
        # Simple heuristic thresholds
        if edge_density < 0.02 and entropy < 3.5:
            return "Simple"
        if edge_density < 0.07 and entropy < 5.5:
            return "Moderate"
        return "Complex"

    # ---------------------------------------------------------------------
    # 2. Lighting – Dark / Normal / Bright (based on brightness value)
    # ---------------------------------------------------------------------
    def _lighting_category(self, brightness: float) -> str:
        if brightness < 80:
            return "Dark"
        if brightness > 180:
            return "Bright"
        return "Normal"

    # ---------------------------------------------------------------------
    # 3. Object coverage – percentage of image occupied by the largest contour
    # ---------------------------------------------------------------------
    def _object_coverage(self) -> float:
        _, thresh = cv2.threshold(self.gray, 0, 255, cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return 0.0
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        return (area / (self.width * self.height)) * 100.0

    # ---------------------------------------------------------------------
    # 4. Orientation – dominant angle of the largest contour
    # ---------------------------------------------------------------------
    def _orientation(self) -> float:
        _, thresh = cv2.threshold(self.gray, 0, 255, cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return 0.0
        largest = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest)
        angle = rect[2]
        if angle < -45:
            angle = 90 + angle
        return round(angle, 2)

    # ---------------------------------------------------------------------
    # 5. Aspect ratio – width / height of the bounding box of the largest contour
    # ---------------------------------------------------------------------
    def _aspect_ratio(self) -> float:
        _, thresh = cv2.threshold(self.gray, 0, 255, cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return 1.0
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        if h == 0:
            return 0.0
        return round(w / h, 2)

    # ---------------------------------------------------------------------
    # 6. Centeredness – distance of contour centroid from image centre (percent)
    # ---------------------------------------------------------------------
    def _centeredness(self) -> float:
        _, thresh = cv2.threshold(self.gray, 0, 255, cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return 0.0
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return 0.0
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        center_x = self.width / 2.0
        center_y = self.height / 2.0
        offset = np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)
        diag = np.sqrt(self.width ** 2 + self.height ** 2)
        return round((offset / diag) * 100.0, 2)

    # ---------------------------------------------------------------------
    # Public API – combine everything
    # ---------------------------------------------------------------------
    def analyze(self) -> Dict[str, object]:
        """Return a dictionary with all content‑related metrics."""
        brightness = np.mean(self.gray)
        return {
            "background": self._background_complexity(),
            "lighting": self._lighting_category(brightness),
            "object_coverage": round(self._object_coverage(), 2),
            "orientation": self._orientation(),
            "aspect_ratio": self._aspect_ratio(),
            "center_offset": self._centeredness(),
        }

# Quick test when run directly
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python content_analysis.py <image_path>")
        sys.exit(1)
    analyzer = ContentAnalyzer(sys.argv[1])
    print(analyzer.analyze())
