# modules/webcam/__init__.py
"""FewVision Live Webcam Inspection module.

Provides real-time camera management, quality validation, object ROI extraction,
PatchCore anomaly scoring, visual overlay generation, human expert review,
and inspection feedback storage.
"""

from modules.webcam.webcam_manager import WebcamManager, list_available_cameras
from modules.webcam.frame_processor import WebcamFrameProcessor
from modules.webcam.live_inspector import LiveWebcamInspector
from modules.webcam.feedback_store import FeedbackStore

__all__ = [
    "WebcamManager",
    "list_available_cameras",
    "WebcamFrameProcessor",
    "LiveWebcamInspector",
    "FeedbackStore",
]
