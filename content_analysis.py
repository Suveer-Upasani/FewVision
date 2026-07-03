# content_analysis.py
"""Content analysis module for extracting scene and object metrics from images.

Provides the `ContentAnalyzer` class with an `analyze(image_path)` method returning a dictionary
with the following keys:
- background: one of "Simple", "Moderate", "Complex" based on edge density.
- lighting: one of "Dark", "Normal", "Bright" based on mean brightness.
- object_coverage: percentage of image area covered by primary object (0-100).
- orientation: rotation angle of the main object's bounding box in degrees.
- aspect_ratio: width/height ratio of the main object.
- center_offset: distance (in % of image diagonal) between image centre and object centre.
"""

import cv2
import numpy as np
import os


class ContentAnalyzer:
    """Analyze content characteristics of an image."""

    def __init__(self, image_path: str):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Cannot open image: {image_path}")
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.height, self.width = self.gray.shape
        
        # Cache for efficiency so we don't calculate expensive masks twice
        self._mask = None
        self._largest_contour = None

    def _get_object_mask_and_contour(self):
        """Compute and cache the object mask and largest contour efficiently with morphological cleaning."""
        if self._mask is not None and self._largest_contour is not None:
            return self._mask, self._largest_contour

        # 1. Apply Gaussian blur to remove high-frequency noise/grain before thresholding
        blurred = cv2.GaussianBlur(self.gray, (5, 5), 0)
        
        # 2. Otsu's thresholding
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Assume the object is smaller than the background. 
        # If white pixels dominate (more than 50%), invert the mask to isolate the object.
        if np.sum(mask == 255) > (self.width * self.height) / 2:
            mask = cv2.bitwise_not(mask)
            
        # 3. Morphological Operations: Fill holes inside the object (shiny metal reflections) and remove tiny noise spots
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        self._mask = mask

        # Find largest contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            self._largest_contour = max(contours, key=cv2.contourArea)
        else:
            self._largest_contour = None

        return self._mask, self._largest_contour

    # ---------------------------------------------------------------------
    # Background complexity – based on edge density
    # ---------------------------------------------------------------------
    def _background_complexity(self) -> str:
        # Blur first to ensure image grain/sensor noise isn't mistakenly counted as complex background edges
        blurred = cv2.GaussianBlur(self.gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 150)
        edge_density = np.sum(edges > 0) / (self.width * self.height)
        
        if edge_density < 0.05:
            return "Simple"
        elif edge_density < 0.15:
            return "Moderate"
        else:
            return "Complex"

    # ---------------------------------------------------------------------
    # Lighting – mean brightness categorisation
    # ---------------------------------------------------------------------
    def _lighting(self) -> str:
        mean_val = np.mean(self.gray)
        if mean_val < 70:
            return "Dark"
        elif mean_val > 180:
            return "Bright"
        else:
            return "Normal"

    # ---------------------------------------------------------------------
    # Object coverage – Efficiently calculate from cached mask
    # ---------------------------------------------------------------------
    def _object_coverage(self) -> float:
        mask, _ = self._get_object_mask_and_contour()
        coverage = np.sum(mask == 255) / (self.width * self.height) * 100.0
        return round(coverage, 2)

    # ---------------------------------------------------------------------
    # Orientation, aspect ratio and centre offset – based on cached contour
    # ---------------------------------------------------------------------
    def _object_geometry(self):
        _, largest_contour = self._get_object_mask_and_contour()
        
        if largest_contour is None:
            return 0.0, 1.0, 0.0
            
        rect = cv2.minAreaRect(largest_contour)
        (cx, cy), (w, h), angle = rect
        if w < h:
            angle = angle + 90
            
        orientation = round(angle, 2)
        aspect_ratio = round(w / h if h != 0 else 0, 2)
        
        img_cx, img_cy = self.width / 2.0, self.height / 2.0
        diag = np.sqrt(self.width ** 2 + self.height ** 2)
        offset_px = np.sqrt((cx - img_cx) ** 2 + (cy - img_cy) ** 2)
        offset_pct = round((offset_px / diag) * 100, 2)
        
        return orientation, aspect_ratio, offset_pct

    def analyze(self) -> dict:
        """Return a dictionary with all content metrics for the image."""
        background = self._background_complexity()
        lighting = self._lighting()
        coverage = self._object_coverage()
        orientation, aspect_ratio, center_offset = self._object_geometry()
        return {
            "background": background,
            "lighting": lighting,
            "object_coverage": coverage,
            "orientation": orientation,
            "aspect_ratio": aspect_ratio,
            "center_offset": center_offset,
        }

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python content_analysis.py <image_path>")
        sys.exit(1)
    analyzer = ContentAnalyzer(sys.argv[1])
    print(analyzer.analyze())
