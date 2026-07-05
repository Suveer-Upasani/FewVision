# modules/content/content_analysis.py
"""Content analysis module for extracting scene and object metrics from images.

Analyses background complexity, lighting conditions, object coverage,
orientation, aspect ratio, and centre offset. Produces a composite content
score (0–100) suitable for dataset suitability assessment.

Enhanced with:
  - GrabCut segmentation (Otsu fallback)
  - Multi-feature background analysis (edge + entropy + colour)
  - Confidence scores
  - Content score (0–100)
  - Bounding box data for visualisation
"""

import math
import cv2
import numpy as np

from modules.utils.models import ContentMetrics


class ContentAnalyzer:
    """Analyse content characteristics of an image.

    Parameters
    ----------
    image_path : str
        Path to the image file to analyse.

    Raises
    ------
    ValueError
        If the image cannot be opened.
    """

    def __init__(self, image_path: str):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Cannot open image: {image_path}")
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.height, self.width = self.gray.shape
        self._fg_mask = None
        self._bounding_rect = None

    # ------------------------------------------------------------------
    # Background complexity – multi-feature
    # ------------------------------------------------------------------
    def _edge_density(self) -> float:
        edges = cv2.Canny(self.gray, 100, 200)
        return float(np.sum(edges > 0) / (self.width * self.height))

    def _texture_entropy(self) -> float:
        """Approximate local entropy using Laplacian standard deviation."""
        lap = cv2.Laplacian(self.gray, cv2.CV_64F)
        return float(np.std(lap))

    def _colour_variance(self) -> float:
        """Mean per-channel standard deviation in BGR."""
        stds = [float(np.std(self.image[:, :, c])) for c in range(3)]
        return float(np.mean(stds))

    def _background_complexity(self) -> tuple[str, float]:
        """Classify background using edge density + texture entropy + colour variance.

        Returns
        -------
        tuple[str, float]
            ``(label, confidence)`` where label is one of ``"Simple"``,
            ``"Moderate"``, or ``"Complex"``.
        """
        ed = self._edge_density()
        te = self._texture_entropy()
        cv_val = self._colour_variance()

        # Normalise each to 0–1
        ed_score = min(ed / 0.15, 1.0)
        te_score = min(te / 60.0, 1.0)
        cv_score = min(cv_val / 80.0, 1.0)

        # Weighted vote
        complexity = 0.40 * ed_score + 0.30 * te_score + 0.30 * cv_score

        if complexity < 0.25:
            label = "Simple"
        elif complexity < 0.55:
            label = "Moderate"
        else:
            label = "Complex"

        # Confidence: distance from nearest boundary
        distances = [abs(complexity - 0.25), abs(complexity - 0.55)]
        confidence = round(min(1.0, min(distances) / 0.15), 4)
        return label, confidence

    # ------------------------------------------------------------------
    # Lighting
    # ------------------------------------------------------------------
    def _lighting(self) -> str:
        """Classify lighting as ``"Dark"``, ``"Normal"``, or ``"Bright"``."""
        mean_val = np.mean(self.gray)
        if mean_val < 70:
            return "Dark"
        elif mean_val > 180:
            return "Bright"
        return "Normal"

    # ------------------------------------------------------------------
    # Object segmentation – GrabCut with Otsu fallback
    # ------------------------------------------------------------------
    def _segment_foreground(self) -> np.ndarray:
        """Return a binary mask (0/255) of the foreground object."""
        if self._fg_mask is not None:
            return self._fg_mask

        # --- Try GrabCut ---
        try:
            mask = np.zeros(self.gray.shape[:2], np.uint8)
            rect = (
                int(0.1 * self.width),
                int(0.1 * self.height),
                int(0.8 * self.width),
                int(0.8 * self.height),
            )
            bgd = np.zeros((1, 65), np.float64)
            fgd = np.zeros((1, 65), np.float64)
            cv2.grabCut(self.image, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
            fg = np.where(
                (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
            ).astype(np.uint8)
            fg_ratio = np.sum(fg == 255) / (self.width * self.height)
            if 0.01 < fg_ratio < 0.99:
                self._fg_mask = fg
                return fg
        except Exception:
            pass

        # --- Otsu fallback ---
        _, mask = cv2.threshold(
            self.gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        if np.sum(mask == 255) < np.sum(mask == 0):
            mask = cv2.bitwise_not(mask)
        self._fg_mask = mask
        return mask

    # ------------------------------------------------------------------
    # Object coverage
    # ------------------------------------------------------------------
    def _object_coverage(self) -> tuple[float, float]:
        """Return ``(coverage_pct, confidence)``."""
        mask = self._segment_foreground()
        total = self.width * self.height
        coverage = float(np.sum(mask == 255) / total * 100.0)

        # Confidence proxy: edge crispness of the mask boundary
        edges = cv2.Canny(mask, 100, 200)
        edge_sharpness = float(np.sum(edges > 0) / max(1, np.sum(mask == 255)))
        confidence = round(min(1.0, edge_sharpness * 50), 4)
        return round(coverage, 2), confidence

    # ------------------------------------------------------------------
    # Object geometry
    # ------------------------------------------------------------------
    def _object_geometry(self) -> tuple[float, float, float, object]:
        """Return ``(orientation, aspect_ratio, center_offset, bounding_rect)``."""
        mask = self._segment_foreground()
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return 0.0, 1.0, 0.0, None

        largest = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest)
        (cx, cy), (w, h), angle = rect
        if w < h:
            angle = angle + 90
        orientation = round(angle, 2)
        aspect_ratio = round(w / h if h != 0 else 0, 2)

        img_cx, img_cy = self.width / 2.0, self.height / 2.0
        diag = math.sqrt(self.width ** 2 + self.height ** 2)
        offset_px = math.sqrt((cx - img_cx) ** 2 + (cy - img_cy) ** 2)
        offset_pct = round((offset_px / diag) * 100, 2)

        self._bounding_rect = rect
        return orientation, aspect_ratio, offset_pct, rect

    # ------------------------------------------------------------------
    # Content Score (0–100)
    # ------------------------------------------------------------------
    @staticmethod
    def _content_score(
        coverage: float,
        center_offset: float,
        background: str,
        lighting: str,
    ) -> float:
        """Weighted content score — higher = more suitable for training.

        Weights: coverage 35%, centre offset 25%, background 20%, lighting 20%.
        """
        # Coverage: ideal 40–80 %
        if 40 <= coverage <= 80:
            s_cov = 1.0
        elif coverage < 40:
            s_cov = coverage / 40.0
        else:
            s_cov = max(0.0, 1.0 - (coverage - 80) / 20.0)

        s_offset = max(0.0, 1.0 - center_offset / 20.0)

        bg_map = {"Simple": 1.0, "Moderate": 0.6, "Complex": 0.3}
        s_bg = bg_map.get(background, 0.5)

        lt_map = {"Normal": 1.0, "Bright": 0.6, "Dark": 0.5}
        s_lt = lt_map.get(lighting, 0.5)

        score = (0.35 * s_cov + 0.25 * s_offset + 0.20 * s_bg + 0.20 * s_lt) * 100
        return round(score, 2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(self) -> ContentMetrics:
        """Run all content checks and return a :class:`ContentMetrics` instance."""
        background, bg_conf = self._background_complexity()
        lighting = self._lighting()
        coverage, cov_conf = self._object_coverage()
        orientation, aspect_ratio, center_offset, brect = self._object_geometry()
        content_score = self._content_score(
            coverage, center_offset, background, lighting
        )

        return ContentMetrics(
            background=background,
            background_confidence=bg_conf,
            lighting=lighting,
            object_coverage=coverage,
            coverage_confidence=cov_conf,
            orientation=orientation,
            aspect_ratio=aspect_ratio,
            center_offset=center_offset,
            content_score=content_score,
            foreground_mask=self._fg_mask,
            bounding_rect=brect,
        )
