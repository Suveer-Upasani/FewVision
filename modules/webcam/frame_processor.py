# modules/webcam/frame_processor.py
"""Frame Quality Assessment and ROI Selection for FewVision Live Webcam.

Performs low-latency in-memory technical quality assessment (blur, brightness,
contrast, noise, exposure clipping) and extracts/manages region-of-interest (ROI)
boundaries for focused industrial part anomaly inspection.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

import config

logger = logging.getLogger("fewvision.webcam.frame_processor")


class WebcamFrameProcessor:
    """Processes in-memory webcam frames for quality and spatial inspection boundaries."""

    def __init__(self, quality_config: Optional[dict[str, float]] = None):
        cfg = quality_config or getattr(config, "QUALITY_CONFIG", {})
        self.blur_threshold = float(cfg.get("blur_threshold", 30.0))
        self.brightness_min = float(cfg.get("brightness_min", 40.0))
        self.brightness_max = float(cfg.get("brightness_max", 220.0))
        self.contrast_threshold = float(cfg.get("contrast_threshold", 20.0))

    def evaluate_quality(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        """Assess the technical quality of an in-memory BGR video frame.

        Parameters
        ----------
        frame_bgr : np.ndarray
            BGR image frame.

        Returns
        -------
        dict[str, Any]
            Quality telemetry including:
            - ``status``: ``"GOOD"``, ``"WARNING"``, or ``"BAD"``
            - ``message``: Human-readable status note
            - ``blur``: Laplacian variance (higher = sharper)
            - ``brightness``: Mean greyscale intensity (0-255)
            - ``contrast``: Standard deviation of pixel intensities
            - ``noise``: MAD-based noise estimate
            - ``underexposed_pct``: Percentage of clipped dark pixels (<5)
            - ``overexposed_pct``: Percentage of clipped bright pixels (>250)
            - ``quality_score``: 0-100 score
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return {
                "status": "BAD",
                "message": "Empty or invalid frame",
                "quality_score": 0.0,
                "blur": 0.0,
                "brightness": 0.0,
                "contrast": 0.0,
                "noise": 0.0,
                "underexposed_pct": 0.0,
                "overexposed_pct": 0.0,
            }

        # Convert to greyscale
        if frame_bgr.ndim == 2:
            gray = frame_bgr
        else:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape
        total_pixels = max(1, h * w)

        # 1. Sharpness (Laplacian variance)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # 2. Brightness
        brightness = float(np.mean(gray))

        # 3. Contrast (Standard deviation)
        contrast = float(np.std(gray))

        # 4. Noise estimate (Median Absolute Deviation of Laplacian)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        mad = float(np.median(np.abs(lap - np.median(lap))))
        noise = float(mad / 0.6745 if mad > 0 else 0.0)

        # 5. Exposure clipping
        under_pct = float((np.sum(gray < 5) / total_pixels) * 100.0)
        over_pct = float((np.sum(gray > 250) / total_pixels) * 100.0)

        # 6. Composite Score (0 - 100)
        def _sigmoid(val: float, mid: float, steep: float = 0.05) -> float:
            x = steep * (val - mid)
            return 1.0 / (1.0 + math.exp(-x))

        s_blur = _sigmoid(blur, self.blur_threshold, 0.10)
        s_bright = max(0.0, 1.0 - abs(brightness - 128.0) / 128.0)
        s_contrast = _sigmoid(contrast, self.contrast_threshold, 0.15)
        s_noise = 1.0 - _sigmoid(noise, 15.0, 0.15)
        s_res = min(1.0, total_pixels / 250_000.0)

        raw = (0.30 * s_blur + 0.20 * s_bright + 0.20 * s_contrast
               + 0.15 * s_noise + 0.15 * s_res)

        if under_pct > 8.0:
            raw *= 0.85
        if over_pct > 8.0:
            raw *= 0.85

        score = round(raw * 100.0, 1)

        # 7. Classification: GOOD / WARNING / BAD
        reasons = []
        is_bad = False
        is_warning = False

        # Extreme conditions -> BAD
        if blur < (self.blur_threshold * 0.4):
            is_bad = True
            reasons.append("Severe blur detected")
        elif blur < self.blur_threshold:
            is_warning = True
            reasons.append("Mild blur")

        if brightness < (self.brightness_min * 0.5):
            is_bad = True
            reasons.append("Frame is extremely dark")
        elif brightness < self.brightness_min:
            is_warning = True
            reasons.append("Low lighting")

        if brightness > (self.brightness_max + 15.0):
            is_bad = True
            reasons.append("Frame is severely overexposed")
        elif brightness > self.brightness_max:
            is_warning = True
            reasons.append("High brightness")

        if contrast < (self.contrast_threshold * 0.5):
            is_bad = True
            reasons.append("Very low contrast")
        elif contrast < self.contrast_threshold:
            is_warning = True
            reasons.append("Sub-optimal contrast")

        if under_pct > 25.0:
            is_bad = True
            reasons.append("Heavy shadow/dark clipping")
        if over_pct > 25.0:
            is_bad = True
            reasons.append("Heavy glare/saturation")

        if is_bad:
            status = "BAD"
            message = f"IMAGE QUALITY TOO LOW: {'; '.join(reasons)}"
        elif is_warning:
            status = "WARNING"
            message = f"Quality warning: {'; '.join(reasons)}"
        else:
            status = "GOOD"
            message = "Quality optimal for inspection"

        return {
            "status": status,
            "message": message,
            "quality_score": score,
            "blur": round(blur, 2),
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "noise": round(noise, 2),
            "underexposed_pct": round(under_pct, 2),
            "overexposed_pct": round(over_pct, 2),
        }

    def resolve_roi(
        self,
        frame_bgr: np.ndarray,
        roi_spec: Optional[list[float] | tuple[float, ...]] = None,
        auto_detect: bool = False,
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Extract the inspection Region Of Interest (ROI) from a frame.

        Parameters
        ----------
        frame_bgr : np.ndarray
            The full camera frame.
        roi_spec : list or tuple of 4 floats, optional
            Normalized bounding box ``[ymin, xmin, ymax, xmax]`` in range ``[0.0, 1.0]``.
            If None, defaults to full frame or auto-detected object contour.
        auto_detect : bool
            If True, attempts to detect the primary industrial object contour
            inside the frame using morphological edge segmentation.

        Returns
        -------
        Tuple[np.ndarray, Tuple[int, int, int, int]]
            ``(roi_crop_bgr, (xmin, ymin, xmax, ymax))`` in absolute pixel coordinates.
        """
        h, w = frame_bgr.shape[:2]

        if auto_detect:
            detected_box = self._detect_object_box(frame_bgr)
            if detected_box is not None:
                xmin, ymin, xmax, ymax = detected_box
                return frame_bgr[ymin:ymax, xmin:xmax].copy(), (xmin, ymin, xmax, ymax)

        if roi_spec is not None and len(roi_spec) == 4:
            # Normalized coordinates: [ymin, xmin, ymax, xmax]
            ymin_norm, xmin_norm, ymax_norm, xmax_norm = roi_spec
            xmin = max(0, min(w - 10, int(xmin_norm * w)))
            ymin = max(0, min(h - 10, int(ymin_norm * h)))
            xmax = max(xmin + 10, min(w, int(xmax_norm * w)))
            ymax = max(ymin + 10, min(h, int(ymax_norm * h)))
            return frame_bgr[ymin:ymax, xmin:xmax].copy(), (xmin, ymin, xmax, ymax)

        # Default to full frame
        return frame_bgr.copy(), (0, 0, w, h)

    def _detect_object_box(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Heuristic object detector for parts on industrial surface or conveyor."""
        try:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            # Otsu automatic thresholding
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None

            h, w = gray.shape
            min_area = (h * w) * 0.05
            max_area = (h * w) * 0.95

            valid = [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]
            if not valid:
                return None

            largest = max(valid, key=cv2.contourArea)
            x, y, bw, bh = cv2.boundingRect(largest)

            # Add 5% padding
            pad_x = int(bw * 0.05)
            pad_y = int(bh * 0.05)
            xmin = max(0, x - pad_x)
            ymin = max(0, y - pad_y)
            xmax = min(w, x + bw + pad_x)
            ymax = min(h, y + bh + pad_y)

            return (xmin, ymin, xmax, ymax)
        except Exception as exc:
            logger.debug("Auto object detection fallback: %s", exc)
            return None

    def draw_roi_bracket(
        self,
        canvas: np.ndarray,
        roi_box: Tuple[int, int, int, int],
        status: str = "GOOD",
        color: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """Draw an industrial bracket overlay indicating the active inspection ROI."""
        xmin, ymin, xmax, ymax = roi_box
        h, w = canvas.shape[:2]

        if color is None:
            if status == "GOOD":
                color = (0, 220, 100)    # Emerald green
            elif status == "WARNING":
                color = (0, 180, 255)    # Amber
            else:
                color = (60, 60, 230)    # Red

        # Corner bracket length
        arm = min(30, max(10, int(min(xmax - xmin, ymax - ymin) * 0.15)))
        thick = 2

        # Top-left
        cv2.line(canvas, (xmin, ymin), (xmin + arm, ymin), color, thick)
        cv2.line(canvas, (xmin, ymin), (xmin, ymin + arm), color, thick)

        # Top-right
        cv2.line(canvas, (xmax, ymin), (xmax - arm, ymin), color, thick)
        cv2.line(canvas, (xmax, ymin), (xmax, ymin + arm), color, thick)

        # Bottom-left
        cv2.line(canvas, (xmin, ymax), (xmin + arm, ymax), color, thick)
        cv2.line(canvas, (xmin, ymax), (xmin, ymax - arm), color, thick)

        # Bottom-right
        cv2.line(canvas, (xmax, ymax), (xmax - arm, ymax), color, thick)
        cv2.line(canvas, (xmax, ymax), (xmax, ymax - arm), color, thick)

        # Label tag above ROI
        label = "INSPECTION ROI"
        cv2.putText(
            canvas,
            label,
            (xmin + 6, max(20, ymin - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    def assess_quality(self, frame_bgr: Optional[np.ndarray]) -> Tuple[dict[str, Any], str, str]:
        """Convenience wrapper returning (metrics_dict, status_str, reason_str)."""
        metrics = self.evaluate_quality(frame_bgr)
        metrics["blur_score"] = metrics.get("blur", 0.0)
        metrics["mean_brightness"] = metrics.get("brightness", 0.0)
        metrics["contrast_std"] = metrics.get("contrast", 0.0)
        status = metrics.get("status", "BAD")
        reason = "OK" if status == "GOOD" else metrics.get("message", "Quality below threshold")
        return metrics, status, reason

    def extract_roi(
        self,
        frame_bgr: np.ndarray,
        mode: str = "center",
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Extract ROI based on mode ('center', 'full', 'auto').
        
        Returns (cropped_bgr, (x, y, w, h)).
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return np.array([], dtype=np.uint8), (0, 0, 0, 0)

        h, w = frame_bgr.shape[:2]
        if mode == "center":
            roi_w = int(w * 0.8)
            roi_h = int(h * 0.8)
            roi_x = (w - roi_w) // 2
            roi_y = (h - roi_h) // 2
            return frame_bgr[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w].copy(), (roi_x, roi_y, roi_w, roi_h)
        elif mode == "full":
            return frame_bgr.copy(), (0, 0, w, h)
        else:
            cropped, box = self.resolve_roi(frame_bgr, auto_detect=(mode == "auto"))
            xmin, ymin, xmax, ymax = box
            return cropped, (xmin, ymin, xmax - xmin, ymax - ymin)

