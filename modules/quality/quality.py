# modules/quality/quality.py
"""Image quality analysis for the FewVision pipeline.

Analyses a single image and returns a :class:`QualityMetrics` dataclass
containing blur, brightness, contrast, noise, resolution, exposure clipping,
a composite quality score, confidence estimates, and augmentation
recommendations.
"""

import os
import math
import cv2
import numpy as np

from modules.utils.models import QualityMetrics


class ImageQualityChecker:
    """Analyse the technical quality of an image.

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

    # ----------------------------
    # Blur Detection
    # ----------------------------
    def blur_score(self) -> float:
        """Return the Laplacian variance as a sharpness proxy.

        Higher = sharper.
        """
        return cv2.Laplacian(self.gray, cv2.CV_64F).var()

    # ----------------------------
    # Brightness
    # ----------------------------
    def brightness(self) -> float:
        """Return mean pixel intensity of the greyscale image (0–255)."""
        return np.mean(self.gray)

    # ----------------------------
    # Contrast
    # ----------------------------
    def contrast(self) -> float:
        """Return standard deviation of the greyscale image (0–128)."""
        return np.std(self.gray)

    # ----------------------------
    # Resolution
    # ----------------------------
    def resolution(self) -> tuple[int, int]:
        """Return ``(width, height)`` of the image in pixels."""
        h, w = self.gray.shape
        return w, h

    # ----------------------------
    # Noise Estimate (MAD-based)
    # ----------------------------
    def noise_score(self) -> float:
        """Estimate noise using Median Absolute Deviation of high-frequency content.

        More stable than simple Gaussian subtraction.
        """
        lap = cv2.Laplacian(self.gray, cv2.CV_64F)
        # MAD estimator: sigma ≈ MAD / 0.6745
        mad = float(np.median(np.abs(lap - np.median(lap))))
        sigma = mad / 0.6745 if mad > 0 else 0.0
        return sigma

    # ----------------------------
    # Exposure Clipping
    # ----------------------------
    def exposure_clipping(self) -> tuple[float, float]:
        """Return ``(underexposed_pct, overexposed_pct)``.

        Percentages of pixels clipped to near-black (< 5) or near-white (> 250).
        """
        total = self.width * self.height
        under = float(np.sum(self.gray < 5) / total * 100)
        over = float(np.sum(self.gray > 250) / total * 100)
        return round(under, 2), round(over, 2)

    # ----------------------------
    # Sharpness Heatmap
    # ----------------------------
    def sharpness_heatmap(self, block_size: int = 32) -> np.ndarray:
        """Block-wise Laplacian variance heatmap.

        Parameters
        ----------
        block_size : int
            Size of each block in pixels (default: 32).

        Returns
        -------
        np.ndarray
            2-D array where each cell holds the sharpness of that image region.
        """
        rows = math.ceil(self.height / block_size)
        cols = math.ceil(self.width / block_size)
        hmap = np.zeros((rows, cols), dtype=np.float64)
        for r in range(rows):
            for c in range(cols):
                y0, y1 = r * block_size, min((r + 1) * block_size, self.height)
                x0, x1 = c * block_size, min((c + 1) * block_size, self.width)
                patch = self.gray[y0:y1, x0:x1]
                hmap[r, c] = cv2.Laplacian(patch, cv2.CV_64F).var()
        return hmap

    # ----------------------------
    # Confidence Scores
    # ----------------------------
    def _blur_confidence(self, blur: float) -> float:
        """Higher blur variance → higher confidence the image is sharp."""
        return round(1.0 - math.exp(-blur / 200.0), 4)

    def _noise_confidence(self) -> float:
        """Stability of noise estimate across image quadrants."""
        h2, w2 = self.height // 2, self.width // 2
        quarters = [
            self.gray[:h2, :w2], self.gray[:h2, w2:],
            self.gray[h2:, :w2], self.gray[h2:, w2:],
        ]
        stds = []
        for q in quarters:
            blurred = cv2.GaussianBlur(q, (3, 3), 0)
            diff = q.astype(np.float32) - blurred.astype(np.float32)
            stds.append(float(np.std(diff)))
        mean_std = np.mean(stds)
        if mean_std == 0:
            return 1.0
        cv = float(np.std(stds) / mean_std)
        return round(max(0.0, min(1.0, 1.0 - cv)), 4)

    # ----------------------------
    # Quality Score
    # ----------------------------
    def quality_score(
        self,
        blur: float,
        bright: float,
        contrast: float,
        noise: float,
        w: int,
        h: int,
        under_pct: float,
        over_pct: float,
    ) -> tuple[float, str]:
        """Weighted composite quality score (0–100) and rating string.

        Weights: blur 30%, brightness 20%, contrast 20%, noise 15%, resolution 15%.
        Penalised if > 5% pixels are under/over-exposed.
        """

        def sigmoid(val: float, mid: float, steep: float = 0.05) -> float:
            x = steep * (val - mid)
            return 1.0 / (1.0 + math.exp(-x))

        s_blur = sigmoid(blur, 150, 0.02)                          # higher = sharper
        s_bright = max(0.0, 1.0 - abs(bright - 128) / 128.0)      # optimal ~128
        s_contrast = sigmoid(contrast, 50, 0.04)                   # higher = better
        s_noise = 1.0 - sigmoid(noise, 15, 0.15)                   # lower = better
        s_res = min(1.0, (w * h) / 640_000)                        # ≥800×800 = 1.0

        raw = (0.30 * s_blur + 0.20 * s_bright + 0.20 * s_contrast
               + 0.15 * s_noise + 0.15 * s_res)

        if under_pct > 5.0:
            raw *= 0.9
        if over_pct > 5.0:
            raw *= 0.9

        score = round(raw * 100, 2)
        if score >= 85:
            rating = "Excellent"
        elif score >= 60:
            rating = "Good"
        elif score >= 40:
            rating = "Fair"
        else:
            rating = "Poor"
        return score, rating

    # ----------------------------
    # Recommendations
    # ----------------------------
    def recommendations(self) -> list[str]:
        """Return a list of suggested augmentations based on quality metrics."""
        rec = []

        blur = self.blur_score()
        bright = self.brightness()
        contrast = self.contrast()
        under, over = self.exposure_clipping()

        if blur < 100:
            rec.append("Sharpen Image")
        else:
            rec.extend(["Small Rotation", "Horizontal Flip"])

        if bright < 70:
            rec.append("Increase Brightness")
        elif bright > 180:
            rec.append("Decrease Brightness")

        if contrast < 40:
            rec.append("Increase Contrast")

        if under > 5.0:
            rec.append("Fix Underexposure")
        if over > 5.0:
            rec.append("Fix Overexposure")

        return rec

    # ----------------------------
    # Public API
    # ----------------------------
    def analyze(self) -> QualityMetrics:
        """Run all quality checks and return a :class:`QualityMetrics` instance."""
        blur = self.blur_score()
        bright = self.brightness()
        cont = self.contrast()
        noise = self.noise_score()
        width, height = self.resolution()
        under, over = self.exposure_clipping()
        score, rating = self.quality_score(blur, bright, cont, noise,
                                           width, height, under, over)
        rec = self.recommendations()
        s_map = self.sharpness_heatmap()

        return QualityMetrics(
            blur=round(float(blur), 2),
            brightness=round(float(bright), 2),
            contrast=round(float(cont), 2),
            noise=round(float(noise), 4),
            resolution=f"{width}x{height}",
            underexposed_pct=under,
            overexposed_pct=over,
            quality_score=score,
            quality_rating=rating,
            blur_confidence=self._blur_confidence(blur),
            noise_confidence=self._noise_confidence(),
            recommendations=rec,
            sharpness_map=s_map,
        )
