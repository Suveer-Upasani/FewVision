# modules/utils/video_utils.py
"""Video file utilities for FewVision.

Provides helpers for:
- Detecting whether an uploaded file is a video or image based on extension.
- Opening an OpenCV VideoCapture and reading basic metadata.
- Validating that the opened capture is readable and non-empty.
"""

from __future__ import annotations

import logging
import os
from typing import Tuple

import cv2

logger = logging.getLogger("fewvision.utils.video")


# ---------------------------------------------------------------------------
# Extension-based type detection
# ---------------------------------------------------------------------------

def is_video_file(filename: str) -> bool:
    """Return True if the filename has a supported video extension.

    Parameters
    ----------
    filename : str
        The filename (with or without directory path).

    Returns
    -------
    bool
    """
    import config
    ext = os.path.splitext(filename)[1].lower()
    return ext in config.VALID_VIDEO_EXTENSIONS


def is_image_file(filename: str) -> bool:
    """Return True if the filename has a supported image extension.

    Parameters
    ----------
    filename : str
        The filename (with or without directory path).

    Returns
    -------
    bool
    """
    import config
    ext = os.path.splitext(filename)[1].lower()
    return ext in config.VALID_EXTENSIONS


# ---------------------------------------------------------------------------
# VideoCapture wrapper
# ---------------------------------------------------------------------------

class VideoMetadataRaw:
    """Lightweight container for raw OpenCV video metadata."""

    __slots__ = ("fps", "total_frames", "width", "height", "duration")

    def __init__(self, fps: float, total_frames: int, width: int, height: int) -> None:
        self.fps = fps
        self.total_frames = total_frames
        self.width = width
        self.height = height
        self.duration = (total_frames / fps) if fps > 0 else 0.0


def open_video(path: str) -> Tuple[cv2.VideoCapture, VideoMetadataRaw]:
    """Open a video file and return the capture object along with metadata.

    Parameters
    ----------
    path : str
        Absolute path to the video file.

    Returns
    -------
    Tuple[cv2.VideoCapture, VideoMetadataRaw]
        An open capture handle and a metadata snapshot.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If OpenCV cannot open the file or the video has no frames.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(
            f"OpenCV failed to open video '{path}'. "
            "The file may be corrupted or use an unsupported codec."
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames <= 0:
        # Some containers don't report frame count; attempt a read to confirm.
        ret, _ = cap.read()
        if not ret:
            cap.release()
            raise ValueError(
                f"Video '{path}' reports zero frames and the first read failed."
            )
        # Rewind to start.
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        total_frames = 1  # at least one readable frame

    meta = VideoMetadataRaw(
        fps=fps,
        total_frames=total_frames,
        width=width,
        height=height,
    )

    logger.info(
        "Opened video '%s': fps=%.2f total_frames=%d size=%dx%d duration=%.2fs",
        os.path.basename(path),
        meta.fps,
        meta.total_frames,
        meta.width,
        meta.height,
        meta.duration,
    )
    return cap, meta


def validate_frame(frame, frame_index: int) -> bool:
    """Return True if the frame is a valid non-empty numpy array.

    Parameters
    ----------
    frame : np.ndarray or None
        The frame returned by cap.read().
    frame_index : int
        Index of this frame (used for logging only).

    Returns
    -------
    bool
    """
    if frame is None:
        logger.warning("Frame %d is None — skipping.", frame_index)
        return False
    if frame.size == 0:
        logger.warning("Frame %d is empty — skipping.", frame_index)
        return False
    return True
