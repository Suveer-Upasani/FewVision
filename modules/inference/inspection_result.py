# modules/inference/inspection_result.py
"""Inspection result model for FewVision.

Represents a single query image's complete anomaly detection, similarity,
and quality/content assessment data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class InspectionResult:
    """Detailed anomaly detection and part assessment results for a single test image.

    Attributes
    ----------
    image_name : str
        Filename of the test image.
    image_path : str
        Absolute path to the test image.
    prediction : str
        Anomaly label, one of "Normal", "Suspicious", or "Anomalous".
    anomaly_score : float
        Composite anomaly score in [0, 100].
    confidence : float
        Confidence of the prediction label in [0, 1].
    nearest_reference : str
        Filename of the closest normal reference image in the Memory Bank.
    top_k_neighbors : List[Dict[str, Any]]
        List of dicts representing the nearest neighbors, each containing:
        - "rank": int (1-based index)
        - "filename": str
        - "distance": float
        - "similarity": float
    quality_score : float
        Quality score of the test image in [0, 100].
    content_score : float
        Content score of the test image in [0, 100].
    quality_metrics : Dict[str, Any]
        Raw quality metrics dict (blur, brightness, contrast, noise, etc.).
    content_metrics : Dict[str, Any]
        Raw content metrics dict (background, lighting, coverage, etc.).
    """

    image_name: str
    image_path: str
    prediction: str
    anomaly_score: float
    confidence: float
    nearest_reference: str
    top_k_neighbors: List[Dict[str, Any]] = field(default_factory=list)
    quality_score: float = 0.0
    content_score: float = 0.0
    quality_metrics: Dict[str, Any] = field(default_factory=dict)
    content_metrics: Dict[str, Any] = field(default_factory=dict)

    # PatchCore fields
    patchcore_enabled: bool = False
    max_patch_score: float = 0.0
    anomaly_area_percent: float = 0.0
    bounding_box: List[int] = field(default_factory=list)  # [ymin, xmin, ymax, xmax]
    centroid: List[int] = field(default_factory=list)  # [cy, cx]
    heatmap_url: str = ""
    overlay_url: str = ""
    original_url: str = ""
    top_5_patch_matches: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the inspection result into a JSON-serialisable dictionary."""
        return {
            "image_name": self.image_name,
            "image_path": self.image_path,
            "prediction": self.prediction,
            "anomaly_score": self.anomaly_score,
            "confidence": self.confidence,
            "nearest_reference": self.nearest_reference,
            "top_k_neighbors": self.top_k_neighbors,
            "quality_score": self.quality_score,
            "content_score": self.content_score,
            "quality_metrics": self.quality_metrics,
            "content_metrics": self.content_metrics,
            "patchcore_enabled": self.patchcore_enabled,
            "max_patch_score": self.max_patch_score,
            "anomaly_area_percent": self.anomaly_area_percent,
            "bounding_box": self.bounding_box,
            "centroid": self.centroid,
            "heatmap_url": self.heatmap_url,
            "overlay_url": self.overlay_url,
            "original_url": self.original_url,
            "top_5_patch_matches": self.top_5_patch_matches,
        }

