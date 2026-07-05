# modules/utils/models.py
"""Typed dataclasses for the FewVision pipeline.

These replace raw dictionaries, making the code safer, easier to extend,
and less error-prone.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class QualityMetrics:
    """Results from the Image Quality Check stage."""

    blur: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    noise: float = 0.0
    resolution: str = "0x0"

    underexposed_pct: float = 0.0       # % pixels < 5
    overexposed_pct: float = 0.0        # % pixels > 250
    quality_score: float = 0.0          # weighted 0–100
    quality_rating: str = "Unknown"     # Excellent / Good / Fair / Poor

    blur_confidence: float = 0.0        # 0–1
    noise_confidence: float = 0.0       # 0–1

    recommendations: list[str] = field(default_factory=list)

    # Sharpness heatmap stored as a numpy array (not serialised to JSON)
    sharpness_map: Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        """Serialise to a flat dictionary (excludes numpy arrays)."""
        return {
            "blur": self.blur,
            "brightness": self.brightness,
            "contrast": self.contrast,
            "noise": self.noise,
            "resolution": self.resolution,
            "underexposed_pct": self.underexposed_pct,
            "overexposed_pct": self.overexposed_pct,
            "quality_score": self.quality_score,
            "quality_rating": self.quality_rating,
            "blur_confidence": self.blur_confidence,
            "noise_confidence": self.noise_confidence,
            "recommendations": self.recommendations,
        }


@dataclass
class ContentMetrics:
    """Results from the Content Analysis stage."""

    background: str = "Unknown"
    background_confidence: float = 0.0
    lighting: str = "Unknown"
    object_coverage: float = 0.0
    coverage_confidence: float = 0.0
    orientation: float = 0.0
    aspect_ratio: float = 0.0
    center_offset: float = 0.0
    content_score: float = 0.0          # 0–100

    # Internal: foreground mask for report visualisation (not serialised)
    foreground_mask: Optional[np.ndarray] = field(default=None, repr=False)
    # Bounding box as ((cx, cy), (w, h), angle) from minAreaRect
    bounding_rect: Optional[tuple] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        """Serialise to a flat dictionary (excludes numpy arrays)."""
        return {
            "background": self.background,
            "background_confidence": self.background_confidence,
            "lighting": self.lighting,
            "object_coverage": self.object_coverage,
            "coverage_confidence": self.coverage_confidence,
            "orientation": self.orientation,
            "aspect_ratio": self.aspect_ratio,
            "center_offset": self.center_offset,
            "content_score": self.content_score,
        }


@dataclass
class AnalysisResult:
    """Combined analysis result for a single image."""

    image: str = ""                     # basename
    image_path: str = ""                # absolute path
    quality: QualityMetrics = field(default_factory=QualityMetrics)
    content: ContentMetrics = field(default_factory=ContentMetrics)
    suitability_score: float = 0.0      # 0–100
    suitability_rating: str = "Unknown" # Ready / Marginal / Unsuitable
    augmentations: list[str] = field(default_factory=list)

    def to_flat_dict(self) -> dict:
        """Merge quality + content + top-level into a single flat dict for CSV/JSON."""
        d = {
            "image": self.image,
            "image_path": self.image_path,
        }
        d.update(self.quality.to_dict())
        d.update(self.content.to_dict())
        d["suitability_score"] = self.suitability_score
        d["suitability_rating"] = self.suitability_rating
        d["augmentations"] = ", ".join(self.augmentations)
        d["recommendations"] = ", ".join(self.quality.recommendations)
        return d


@dataclass
class DatasetResult:
    """Combined result for an entire uploaded dataset batch."""

    results: list[AnalysisResult] = field(default_factory=list)
    analytics: dict = field(default_factory=dict)
    augmented_dir: str = ""
    report_dir: str = ""
    session_id: str = ""
    total_images: int = 0
    augmented_count: int = 0
    ready_count: int = 0
    marginal_count: int = 0
    unsuitable_count: int = 0

    def summary_dict(self) -> dict:
        """Return a JSON-safe summary for API responses."""
        return {
            "session_id": self.session_id,
            "total_images": self.total_images,
            "augmented_count": self.augmented_count,
            "ready_count": self.ready_count,
            "marginal_count": self.marginal_count,
            "unsuitable_count": self.unsuitable_count,
            "augmented_dir": self.augmented_dir,
            "report_dir": self.report_dir,
        }
