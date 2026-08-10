# modules/inference/video_result.py
"""Result models for multi-frame video inspection in FewVision.

Provides two dataclasses:
- FrameResult   — per-frame anomaly detection output
- VideoInspectionResult — aggregated video-level result
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FrameResult:
    """Anomaly detection result for a single sampled video frame.

    Attributes
    ----------
    frame_id : int
        Sequential ID of this frame within the processed set (0-based).
    frame_index : int
        Original frame index in the source video.
    timestamp : float
        Timestamp in seconds within the video.
    score : float
        PatchCore max patch score for this frame (raw distance units).
    is_anomaly : bool
        True when score >= PATCH_THRESHOLD.
    centroid : List[int]
        [cy, cx] centroid of the largest detected defect region (or [0, 0]).
    bounding_box : List[int]
        [ymin, xmin, ymax, xmax] of the largest defect contour (or [0,0,0,0]).
    heatmap_url : str
        URL path to the saved heatmap image (empty if SAVE_VIDEO_FRAMES=False).
    overlay_url : str
        URL path to the saved overlay image (empty if SAVE_VIDEO_FRAMES=False).
    original_url : str
        URL path to the saved original frame image.
    patchcore_enabled : bool
        Whether PatchCore ran successfully for this frame.
    padim : Dict[str, Any]
        PaDiM localization results (empty dict if not enabled).
    anomaly_area_percent : float
        Percentage of frame area flagged as anomalous.
    error : Optional[str]
        If frame processing failed, the error message; otherwise None.
    """

    frame_id: int
    frame_index: int
    timestamp: float
    score: float
    is_anomaly: bool
    centroid: List[int] = field(default_factory=list)
    bounding_box: List[int] = field(default_factory=list)
    heatmap_url: str = ""
    overlay_url: str = ""
    original_url: str = ""
    patchcore_enabled: bool = False
    padim: Dict[str, Any] = field(default_factory=dict)
    anomaly_area_percent: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "frame_id": self.frame_id,
            "frame_index": self.frame_index,
            "timestamp": round(self.timestamp, 3),
            "score": round(self.score, 6),
            "is_anomaly": self.is_anomaly,
            "centroid": self.centroid,
            "bounding_box": self.bounding_box,
            "heatmap_url": self.heatmap_url,
            "overlay_url": self.overlay_url,
            "original_url": self.original_url,
            "patchcore_enabled": self.patchcore_enabled,
            "padim": self.padim,
            "anomaly_area_percent": round(self.anomaly_area_percent, 4),
            "error": self.error,
        }


@dataclass
class VideoMetadata:
    """Basic metadata extracted from the source video file.

    Attributes
    ----------
    filename : str
        Original upload filename.
    fps : float
        Frames per second of the video.
    total_frames : int
        Total frame count reported by OpenCV.
    duration : float
        Duration in seconds.
    width : int
        Frame width in pixels.
    height : int
        Frame height in pixels.
    sample_interval : int
        Configured sampling interval used during inspection.
    processed_frames : int
        Actual number of frames inspected.
    """

    filename: str
    fps: float
    total_frames: int
    duration: float
    width: int
    height: int
    sample_interval: int
    processed_frames: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "fps": round(self.fps, 3),
            "total_frames": self.total_frames,
            "duration": round(self.duration, 3),
            "width": self.width,
            "height": self.height,
            "sample_interval": self.sample_interval,
            "processed_frames": self.processed_frames,
        }


@dataclass
class VideoInspectionResult:
    """Aggregated anomaly detection result for an entire video clip.

    Attributes
    ----------
    session_id : str
        Reference memory bank session used for inspection.
    run_id : str
        Unique run identifier.
    video_score : float
        Aggregated video-level anomaly score [0, 100].
    anomalous_frames : int
        Number of frames individually flagged as anomalous.
    processed_frames : int
        Total number of frames actually inspected.
    anomaly_persistence : float
        Ratio of anomalous to processed frames in [0.0, 1.0].
    status : str
        Final status: ``"NORMAL"``, ``"SUSPICIOUS"``, or ``"ANOMALY"``.
    aggregation_method : str
        The aggregation method used (e.g. ``"top_k"``).
    metadata : VideoMetadata
        Source video metadata.
    frame_results : List[FrameResult]
        Per-frame inspection results.
    """

    session_id: str
    run_id: str
    video_score: float
    anomalous_frames: int
    processed_frames: int
    anomaly_persistence: float
    status: str
    aggregation_method: str
    metadata: VideoMetadata
    frame_results: List[FrameResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_type": "video",
            "session_id": self.session_id,
            "run_id": self.run_id,
            "video_score": round(self.video_score, 4),
            "anomalous_frames": self.anomalous_frames,
            "processed_frames": self.processed_frames,
            "anomaly_persistence": round(self.anomaly_persistence, 6),
            "status": self.status,
            "aggregation_method": self.aggregation_method,
            "metadata": self.metadata.to_dict(),
            "frame_results": [f.to_dict() for f in self.frame_results],
        }
