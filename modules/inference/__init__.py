# modules/inference/__init__.py
"""Inference pipeline package for FewVision.

Coordinates testing/inspection of product images against a saved Memory Bank,
providing anomaly scoring, label prediction, and confidence estimates.

Public API
----------
>>> from modules.inference import InferenceEngine, InspectionResult
"""

from modules.inference.inspection_result import InspectionResult
from modules.inference.inference_engine import InferenceEngine

__all__ = [
    "InferenceEngine",
    "InspectionResult",
]
