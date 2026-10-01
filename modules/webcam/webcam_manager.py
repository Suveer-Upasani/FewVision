# modules/webcam/webcam_manager.py
"""Robust Camera and Live Video Stream Manager for FewVision.

Provides resilient camera initialization, resolution and FPS negotiation,
thread-safe latest-frame capturing (zero latency backlog), camera discovery,
and simulated test-stream mode for headless environments and continuous testing.
"""

from __future__ import annotations

import glob
import logging
import os
import threading
import time
from typing import Any, Generator, List, Optional, Tuple

import cv2
import numpy as np

import config

logger = logging.getLogger("fewvision.webcam.manager")

# Silence OpenCV internal C++ backend warnings (e.g. DSHOW probe on empty indices)
try:
    import cv2.utils.logging as cvlog
    cvlog.setLogLevel(cvlog.LOG_LEVEL_SILENT)
except Exception:
    pass


def list_available_cameras(max_probe: int = 4) -> List[dict[str, Any]]:
    """Probe system camera indices to discover connected cameras without crashing.

    Parameters
    ----------
    max_probe : int
        Maximum number of camera indices to probe (default: 4).

    Returns
    -------
    List[dict[str, Any]]
        List of dicts with keys: ``index``, ``name``, ``width``, ``height``, ``fps``.
    """
    available = []
    # Ensure OpenCV internal backend logging is completely silent during index probing
    try:
        import cv2.utils.logging as cvlog
        cvlog.setLogLevel(cvlog.LOG_LEVEL_SILENT)
    except Exception:
        pass

    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY

    for idx in range(max_probe):
        cap = None
        try:
            cap = cv2.VideoCapture(idx, backend)
            if cap is not None and cap.isOpened():
                # Read a test frame to ensure it actually transmits
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = float(cap.get(cv2.CAP_PROP_FPS))
                    available.append({
                        "index": idx,
                        "name": f"Camera {idx} ({w}x{h})",
                        "width": w,
                        "height": h,
                        "fps": fps if fps > 0 else 30.0,
                    })
        except Exception as exc:
            logger.debug("Camera probe at index %d failed: %s", idx, exc)
        finally:
            if cap is not None:
                cap.release()

    logger.info("Discovered %d active cameras on the system.", len(available))
    return available


class SimulatedStream:
    """Simulates a live webcam feed from a directory of static image frames.

    Used for automated testing, debugging, and environments without physical cameras.
    """

    def __init__(self, folder_path: str, loop: bool = True, target_fps: float = 15.0):
        self.folder_path = folder_path
        self.loop = loop
        self.target_fps = max(1.0, float(target_fps))
        self.delay = 1.0 / self.target_fps
        self._image_paths = []
        self._current_idx = 0
        self._last_read_time = 0.0

        valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
        for ext in valid_exts:
            self._image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
            self._image_paths.extend(glob.glob(os.path.join(folder_path, ext.upper())))

        self._image_paths.sort()
        if not self._image_paths:
            raise FileNotFoundError(
                f"No test images found in simulated stream directory: {folder_path}"
            )

        # Pre-read first frame to determine dimensions
        first_img = cv2.imread(self._image_paths[0])
        if first_img is None:
            raise ValueError(f"Failed to read image from {self._image_paths[0]}")
        self.height, self.width = first_img.shape[:2]
        logger.info(
            "Simulated stream initialized with %d images from '%s' (%dx%d, %.1f FPS)",
            len(self._image_paths),
            folder_path,
            self.width,
            self.height,
            self.target_fps,
        )

    def is_opened(self) -> bool:
        return len(self._image_paths) > 0

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the next simulated frame, respecting target FPS timing."""
        if not self._image_paths:
            return False, None

        # Pace to target FPS
        now = time.time()
        elapsed = now - self._last_read_time
        if elapsed < self.delay:
            time.sleep(max(0.001, self.delay - elapsed))
        self._last_read_time = time.time()

        if self._current_idx >= len(self._image_paths):
            if self.loop:
                self._current_idx = 0
            else:
                return False, None

        img_path = self._image_paths[self._current_idx]
        self._current_idx += 1

        img = cv2.imread(img_path)
        if img is None:
            logger.warning("Corrupt image in simulated stream: %s", img_path)
            return False, None

        return True, img

    @property
    def total_frames(self) -> int:
        return len(self._image_paths)

    def release(self) -> None:
        self._image_paths.clear()
        self._current_idx = 0


class WebcamManager:
    """Thread-safe webcam capture and streaming manager.

    Maintains a continuous background reading loop to ensure the frame buffer
    is always flushed, preventing latency buildup. Inference can consume
    the latest frame at any rate without backlog.
    """

    def __init__(
        self,
        camera_index: int | str = 0,
        width: int = 1280,
        height: int = 720,
        target_fps: int = 15,
        mode: str = "hardware",  # "hardware" or "simulated"
        simulated_folder: Optional[str] = None,
        simulated_dir: Optional[str] = None,
    ):
        if str(camera_index).lower() == "simulated" or mode == "simulated":
            self.mode = "simulated"
            self.camera_index = 0
        else:
            self.mode = mode.lower()
            try:
                self.camera_index = int(camera_index)
            except (ValueError, TypeError):
                self.camera_index = 0

        self.requested_width = width
        self.requested_height = height
        self.target_fps = target_fps
        self.simulated_folder = simulated_folder or simulated_dir

        self.cap: Optional[cv2.VideoCapture | SimulatedStream] = None
        self.is_running = False
        self.is_paused = False

        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

        self._latest_raw_frame: Optional[np.ndarray] = None
        self._latest_annotated_frame: Optional[np.ndarray] = None
        self._latest_result: Optional[dict[str, Any]] = None
        self._frame_count = 0
        self._dropped_frames = 0
        self._last_frame_timestamp = 0.0
        self._measured_fps = 0.0
        self._fps_counter = 0
        self._fps_start_time = time.time()

        self.actual_width = 0
        self.actual_height = 0
        self.actual_fps = 0.0
        self.error_message: Optional[str] = None

    def start(self) -> bool:
        """Initialize the video capture device and launch the background worker thread.

        Returns
        -------
        bool
            True if started successfully, False otherwise.
        """
        with self._lock:
            if self.is_running:
                logger.info("WebcamManager is already running.")
                return True

            self.error_message = None

            if self.mode == "simulated":
                folder = self.simulated_folder or getattr(
                    config, "TEST_STREAM_FOLDER", os.path.join(config.BASE_DIR, "Tiles_dataset")
                )
                if not os.path.isdir(folder):
                    # Fallback to parent Tiles_dataset if subfolder not found
                    alt_folder = os.path.join(config.BASE_DIR, "Tiles_dataset")
                    if os.path.isdir(alt_folder):
                        folder = alt_folder
                    else:
                        self.error_message = f"Simulated stream directory not found: {folder}"
                        logger.error(self.error_message)
                        return False

                try:
                    self.cap = SimulatedStream(folder, loop=True, target_fps=self.target_fps)
                    self.actual_width = self.cap.width
                    self.actual_height = self.cap.height
                    self.actual_fps = self.target_fps
                    logger.info("Started simulated camera stream from '%s'", folder)
                except Exception as exc:
                    self.error_message = f"Failed to initialize simulated stream: {exc}"
                    logger.error(self.error_message)
                    return False

            else:
                # Hardware OpenCV camera
                backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
                logger.info("Opening hardware camera index %d...", self.camera_index)

                try:
                    cap = cv2.VideoCapture(self.camera_index, backend)
                    if not cap.isOpened():
                        # Try without CAP_DSHOW as fallback
                        cap = cv2.VideoCapture(self.camera_index)

                    if not cap.isOpened():
                        self.error_message = (
                            f"Unable to access camera index {self.camera_index}. "
                            "Check camera connection, index, or permissions."
                        )
                        logger.error(self.error_message)
                        return False

                    # Configure resolution and FPS
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)
                    if self.target_fps > 0:
                        cap.set(cv2.CAP_PROP_FPS, self.target_fps)

                    # Query negotiated parameters
                    self.actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    self.actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    raw_fps = cap.get(cv2.CAP_PROP_FPS)
                    self.actual_fps = float(raw_fps) if raw_fps > 0 else float(self.target_fps)

                    self.cap = cap
                    logger.info(
                        "Camera %d opened: %dx%d @ %.1f FPS (requested: %dx%d @ %d FPS)",
                        self.camera_index,
                        self.actual_width,
                        self.actual_height,
                        self.actual_fps,
                        self.requested_width,
                        self.requested_height,
                        self.target_fps,
                    )
                except Exception as exc:
                    self.error_message = f"Exception opening camera {self.camera_index}: {exc}"
                    logger.error(self.error_message)
                    return False

            self.is_running = True
            self.is_paused = False
            self._frame_count = 0
            self._dropped_frames = 0
            self._fps_counter = 0
            self._fps_start_time = time.time()

            # Start background reader thread
            self._thread = threading.Thread(
                target=self._capture_worker, name="FewVisionWebcamCapture", daemon=True
            )
            self._thread.start()
            return True

    def _capture_worker(self) -> None:
        """Background thread continuously pulling frames to discard stale buffer frames."""
        logger.info("Webcam capture worker thread started.")

        while self.is_running:
            if self.cap is None:
                break

            if self.is_paused:
                time.sleep(0.05)
                continue

            try:
                ret, frame = self.cap.read()
                if not ret or frame is None or frame.size == 0:
                    logger.warning("Failed to grab frame from camera.")
                    time.sleep(0.02)
                    continue

                now = time.time()
                with self._lock:
                    self._latest_raw_frame = frame
                    self._last_frame_timestamp = now
                    self._frame_count += 1
                    self._fps_counter += 1

                    # Compute measured capture FPS every second
                    elapsed = now - self._fps_start_time
                    if elapsed >= 1.0:
                        self._measured_fps = round(self._fps_counter / elapsed, 1)
                        self._fps_counter = 0
                        self._fps_start_time = now

            except Exception as exc:
                logger.error("Error in webcam capture worker: %s", exc)
                time.sleep(0.05)

        logger.info("Webcam capture worker thread exiting.")

    def get_latest_frame(self, timeout: float = 0.0) -> Tuple[bool, Optional[np.ndarray], float]:
        """Retrieve the most recent frame captured without blocking (or up to timeout seconds).

        Returns
        -------
        Tuple[bool, Optional[np.ndarray], float]
            (success, frame_bgr_copy, timestamp)
        """
        start = time.time()
        while True:
            with self._lock:
                if self._latest_raw_frame is not None:
                    return True, self._latest_raw_frame.copy(), self._last_frame_timestamp
                if not self.is_running:
                    return False, None, 0.0

            if timeout <= 0.0 or (time.time() - start) >= timeout:
                break
            time.sleep(0.02)

        with self._lock:
            if self._latest_raw_frame is not None:
                return True, self._latest_raw_frame.copy(), self._last_frame_timestamp
            return False, None, 0.0

    def update_latest_result(
        self, annotated_frame: np.ndarray, result: dict[str, Any]
    ) -> None:
        """Store the latest inference result and annotated visualization."""
        with self._lock:
            self._latest_annotated_frame = annotated_frame.copy()
            self._latest_result = result

    def get_latest_annotated_frame(self) -> Optional[np.ndarray]:
        """Get the latest annotated overlay frame (for streaming)."""
        with self._lock:
            if self._latest_annotated_frame is not None:
                return self._latest_annotated_frame.copy()
            if self._latest_raw_frame is not None:
                return self._latest_raw_frame.copy()
            return None

    def get_latest_result(self) -> Optional[dict[str, Any]]:
        """Get the latest calculated inspection telemetry."""
        with self._lock:
            return dict(self._latest_result) if self._latest_result is not None else None

    def pause(self) -> bool:
        """Toggle pause state."""
        with self._lock:
            self.is_paused = not self.is_paused
            logger.info("Webcam inspection %s", "paused" if self.is_paused else "resumed")
            return self.is_paused

    def stop(self) -> None:
        """Safely stop the capture worker and release camera hardware."""
        logger.info("Stopping WebcamManager...")
        self.is_running = False

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            self._thread = None

        with self._lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception as exc:
                    logger.warning("Error releasing camera: %s", exc)
                self.cap = None

            self._latest_raw_frame = None
            self._latest_annotated_frame = None
            self._latest_result = None
            logger.info("WebcamManager stopped and camera released.")

    def get_status(self) -> dict[str, Any]:
        """Return diagnostic status information."""
        with self._lock:
            return {
                "is_running": self.is_running,
                "is_paused": self.is_paused,
                "mode": self.mode,
                "camera_index": self.camera_index,
                "actual_width": self.actual_width,
                "actual_height": self.actual_height,
                "target_fps": self.target_fps,
                "measured_fps": self._measured_fps,
                "frame_count": self._frame_count,
                "error_message": self.error_message,
            }
