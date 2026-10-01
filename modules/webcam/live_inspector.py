# modules/webcam/live_inspector.py
"""Production-grade Live Webcam Inference Engine for FewVision.

Coordinates single-load DINOv2 feature extraction, PatchCore memory bank
nearest-neighbor comparisons, temporal score stabilization, real-time heatmap
localization, and multi-component performance latency measurement.
"""

from __future__ import annotations

import base64
import collections
import datetime
import logging
import time
from typing import Any, Deque, List, Optional, Tuple

import cv2
import numpy as np
import torch

import config
from modules.inference.inference_engine import InferenceEngine
from modules.patchcore.heatmap import generate_heatmap
from modules.patchcore.localization import localize_defects
from modules.patchcore.patch_similarity import search_patch_neighbors
from modules.webcam.frame_processor import WebcamFrameProcessor

logger = logging.getLogger("fewvision.webcam.live_inspector")

# Auto-detect compute hardware
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_NAME = "CUDA" if torch.cuda.is_available() else "CPU"


class LiveWebcamInspector:
    """Manages real-time webcam frame inspection with single-load model lifecycle.

    Reuses the existing FewVision InferenceEngine so models and memory banks
    are NEVER reloaded on per-frame calls.
    """

    def __init__(
        self,
        session_id: str,
        engine: Optional[InferenceEngine] = None,
        patch_threshold: Optional[float] = None,
        temporal_window: Optional[int] = None,
        vote_threshold: Optional[float] = None,
        anomaly_vote_threshold: Optional[float] = None,
        quality_config: Optional[dict[str, float]] = None,
    ):
        self.session_id = session_id
        logger.info(
            "Initializing LiveWebcamInspector for session '%s' on %s...",
            session_id,
            DEVICE_NAME,
        )

        # 1. Load or reuse InferenceEngine
        if engine is not None:
            self.engine = engine
        else:
            self.engine = InferenceEngine(session_id)

        # Validate that reference memory bank is ready
        if (
            not self.engine.patchcore_enabled
            or self.engine.patch_memory_bank is None
            or self.engine.patch_memory_bank._embeddings is None
            or len(self.engine.patch_memory_bank._embeddings) == 0
        ):
            raise RuntimeError(
                f"Live inspection unavailable: insufficient normal reference data "
                f"or missing PatchCore memory bank for session '{session_id}'."
            )

        # 2. Quality and ROI processor
        self.frame_processor = WebcamFrameProcessor(quality_config)

        # 3. Tunable parameters
        self.patch_threshold = float(
            patch_threshold
            if patch_threshold is not None
            else getattr(config, "PATCH_THRESHOLD", 0.5)
        )
        self.temporal_window = int(
            temporal_window
            if temporal_window is not None
            else getattr(config, "TEMPORAL_WINDOW", 5)
        )
        v_thresh = vote_threshold if vote_threshold is not None else anomaly_vote_threshold
        self.vote_threshold = float(
            v_thresh
            if v_thresh is not None
            else getattr(config, "ANOMALY_VOTE_THRESHOLD", 0.6)
        )

        # 4. Temporal stabilization ring buffers
        self._score_history: Deque[float] = collections.deque(maxlen=self.temporal_window)
        self._vote_history: Deque[bool] = collections.deque(maxlen=self.temporal_window)

        # 5. Live performance tracking
        self.frame_id = 0
        self.total_inspections = 0
        self.total_anomalies = 0
        self._fps_tracker: Deque[float] = collections.deque(maxlen=15)
        self._last_inference_time = 0.0

        # 6. Model warmup
        self._warmup()

    @property
    def score_history(self) -> Deque[float]:
        return self._score_history

    @property
    def threshold(self) -> float:
        return self.patch_threshold

    @property
    def device(self) -> torch.device:
        return DEVICE

    def _warmup(self) -> None:
        """Run a single dummy inference to pre-warm PyTorch caches and GPU/CPU kernels."""
        logger.info("Warming up model on %s...", DEVICE_NAME)
        try:
            dummy_frame = np.zeros((224, 224, 3), dtype=np.uint8) + 128
            # Run one forward pass through patch extractor
            from modules.patchcore.patch_extractor import PatchExtractor
            pe = PatchExtractor(extractor=self.engine.extractor, patch_size=config.PATCH_SIZE)
            _ = pe.extract(dummy_frame)
            logger.info("Model warmup completed. Model ready for live inspection.")
        except Exception as exc:
            logger.warning("Warmup encountered an issue (non-fatal): %s", exc)

    def inspect_frame(
        self,
        frame_bgr: np.ndarray,
        roi_spec: Optional[list[float] | tuple[float, ...]] = None,
        auto_roi: bool = False,
        roi_mode: Optional[str] = None,
        suppress_low_quality: bool = True,
    ) -> dict[str, Any]:
        """Inspect a single live webcam frame through the full FewVision pipeline.

        Parameters
        ----------
        frame_bgr : np.ndarray
            Raw BGR image frame from the webcam.
        roi_spec : list or tuple of 4 floats, optional
            Normalized ROI ``[ymin, xmin, ymax, xmax]``.
        auto_roi : bool
            Whether to attempt automated part segmentation for ROI.
        roi_mode : str, optional
            Convenience selector ('center', 'full', 'auto').
        suppress_low_quality : bool
            If True, treats BAD quality frames as QUALITY_WARNING rather than
            making unreliable anomaly decisions.

        Returns
        -------
        dict[str, Any]
            Full inspection telemetry and visualization artifacts.
        """
        if roi_mode == "center" and roi_spec is None:
            roi_spec = [0.1, 0.1, 0.9, 0.9]
        elif roi_mode == "full" and roi_spec is None:
            roi_spec = [0.0, 0.0, 1.0, 1.0]
        elif roi_mode == "auto":
            auto_roi = True

        t_start = time.perf_counter()
        self.frame_id += 1
        now_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 1. Check frame validity
        if frame_bgr is None or frame_bgr.size == 0:
            return {
                "frame_id": self.frame_id,
                "timestamp": now_ts,
                "status": "QUALITY WARNING",
                "raw_status": "QUALITY WARNING",
                "error": "Empty or corrupt frame",
                "score": 0.0,
                "raw_score": 0.0,
                "stabilized_score": 0.0,
                "threshold": self.patch_threshold,
                "quality": "BAD",
                "quality_metrics": {"status": "BAD", "message": "Invalid frame"},
                "latency_ms": 0.0,
                "latency": {"preprocess_ms": 0.0, "dinov2_ms": 0.0, "patchcore_ms": 0.0, "total_ms": 0.0},
                "timings": {"preprocess_ms": 0.0, "dinov2_ms": 0.0, "patchcore_ms": 0.0, "total_ms": 0.0},
                "fps": 0.0,
                "device": DEVICE_NAME,
                "roi": [0, 0, 0, 0],
                "bbox": [],
                "bounding_box": [],
                "centroid": [],
                "anomaly_area_percent": 0.0,
                "anomaly_detected": False,
                "overlay_base64": "",
                "heatmap_base64": "",
            }

        # 2. Quality assessment
        t_prep_start = time.perf_counter()
        q_result = self.frame_processor.evaluate_quality(frame_bgr)

        # 3. Resolve ROI
        roi_crop, roi_box = self.frame_processor.resolve_roi(
            frame_bgr, roi_spec=roi_spec, auto_detect=auto_roi
        )
        xmin_r, ymin_r, xmax_r, ymax_r = roi_box
        t_prep_end = time.perf_counter()
        t_prep_ms = (t_prep_end - t_prep_start) * 1000.0

        # Check if quality is too low to produce reliable anomaly scoring
        if suppress_low_quality and q_result["status"] == "BAD":
            t_total_ms = (time.perf_counter() - t_start) * 1000.0
            banner_overlay = frame_bgr.copy()
            self._draw_hud_banner(
                banner_overlay,
                status="QUALITY WARNING",
                score=0.0,
                threshold=self.patch_threshold,
                fps=0.0,
                quality_status="BAD",
            )
            _, overlay_buf = cv2.imencode(".jpg", banner_overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
            overlay_b64 = "data:image/jpeg;base64," + base64.b64encode(overlay_buf).decode("utf-8")

            return {
                "frame_id": self.frame_id,
                "timestamp": now_ts,
                "status": "QUALITY WARNING",
                "raw_status": "QUALITY WARNING",
                "message": q_result["message"],
                "score": 0.0,
                "raw_score": 0.0,
                "stabilized_score": 0.0,
                "threshold": self.patch_threshold,
                "quality": q_result.get("status", "BAD"),
                "quality_metrics": q_result,
                "latency_ms": round(t_total_ms, 2),
                "latency": {
                    "preprocess_ms": round(t_prep_ms, 2),
                    "dinov2_ms": 0.0,
                    "patchcore_ms": 0.0,
                    "total_ms": round(t_total_ms, 2),
                },
                "timings": {
                    "preprocess_ms": round(t_prep_ms, 2),
                    "dinov2_ms": 0.0,
                    "patchcore_ms": 0.0,
                    "total_ms": round(t_total_ms, 2),
                },
                "fps": self._compute_live_fps(t_start),
                "device": DEVICE_NAME,
                "bounding_box": [],
                "bbox": [],
                "centroid": [],
                "anomaly_area_percent": 0.0,
                "anomaly_detected": False,
                "roi": list(roi_box),
                "overlay_base64": overlay_b64,
                "heatmap_base64": "",
                "_raw_frame": frame_bgr,
                "_annotated_frame": banner_overlay,
                "_heatmap_roi": None,
                "_distance_map": None,
            }

        # 4. Extract patch embeddings with DINOv2
        t_dino_start = time.perf_counter()
        from modules.patchcore.patch_extractor import PatchExtractor

        patch_extractor = PatchExtractor(
            extractor=self.engine.extractor, patch_size=config.PATCH_SIZE
        )
        patch_embeddings = patch_extractor.extract(roi_crop)  # Shape (196, D)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_dino_end = time.perf_counter()
        t_dino_ms = (t_dino_end - t_dino_start) * 1000.0

        # 5. Search PatchCore Memory Bank
        t_patch_start = time.perf_counter()
        p_indices, p_distances, p_similarities = search_patch_neighbors(
            patch_embeddings,
            self.engine.patch_memory_bank._embeddings,
            metric=getattr(config, "PATCH_SIMILARITY", "cosine"),
            k=5,
        )

        # 14x14 distance map from nearest-neighbor (k=1)
        distance_map = p_distances[:, 0].reshape(14, 14)
        t_patch_end = time.perf_counter()
        t_patch_ms = (t_patch_end - t_patch_start) * 1000.0

        # 6. Anomaly scoring & defect localization
        raw_score = float(np.max(distance_map))
        # Defect localization on ROI crop shape
        loc = localize_defects(
            distance_map, roi_crop.shape[:2], threshold=self.patch_threshold
        )
        bbox = loc["bbox"]  # [ymin, xmin, ymax, xmax] relative to ROI
        area_pct = float(loc["area_percent"])
        centroid = loc["center"]  # [cy, cx] relative to ROI

        # Map ROI bbox and centroid back to full frame coordinates
        full_bbox = []
        full_centroid = []
        if bbox != [0, 0, 0, 0]:
            by_min, bx_min, by_max, bx_max = bbox
            full_bbox = [
                ymin_r + by_min,
                xmin_r + bx_min,
                ymin_r + by_max,
                xmin_r + bx_max,
            ]
            cy, cx = centroid
            full_centroid = [ymin_r + cy, xmin_r + cx]

        # 7. Raw frame decision
        raw_is_anomaly = raw_score >= self.patch_threshold

        # 8. Temporal stabilization (moving average & voting)
        self._score_history.append(raw_score)
        self._vote_history.append(raw_is_anomaly)

        stabilized_score = float(np.mean(self._score_history))
        anomaly_votes = sum(self._vote_history)
        vote_ratio = anomaly_votes / len(self._vote_history)

        if vote_ratio >= self.vote_threshold:
            stabilized_status = "ANOMALY"
        elif q_result["status"] == "WARNING":
            stabilized_status = "QUALITY WARNING"
        else:
            stabilized_status = "NORMAL"

        self.total_inspections += 1
        if stabilized_status == "ANOMALY":
            self.total_anomalies += 1

        # 9. Generate Anomaly Heatmap and Overlays
        # Heatmap for the ROI crop
        heatmap_roi, overlay_roi = self._render_roi_heatmap(
            roi_crop, distance_map, alpha=getattr(config, "HEATMAP_ALPHA", 0.6)
        )

        # Full frame composite visualization
        full_overlay = frame_bgr.copy()
        # Splice heatmap overlay into full frame at ROI position
        full_overlay[ymin_r:ymax_r, xmin_r:xmax_r] = overlay_roi

        # Draw ROI brackets
        self.frame_processor.draw_roi_bracket(
            full_overlay, roi_box, status=q_result["status"]
        )

        # Draw defect bounding box and centroid if anomalous
        if full_bbox:
            f_ymin, f_xmin, f_ymax, f_xmax = full_bbox
            cv2.rectangle(full_overlay, (f_xmin, f_ymin), (f_xmax, f_ymax), (0, 0, 255), 2)
            cv2.circle(full_overlay, (full_centroid[1], full_centroid[0]), 6, (0, 0, 255), -1)
            cv2.putText(
                full_overlay,
                f"DEFECT {raw_score:.2f}",
                (f_xmin, max(20, f_ymin - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        # Draw professional Industrial HUD banner on top
        self._draw_hud_banner(
            full_overlay,
            status=stabilized_status,
            score=stabilized_score,
            threshold=self.patch_threshold,
            fps=self._fps_tracker[-1] if self._fps_tracker else 0.0,
            quality_status=q_result["status"],
        )

        t_total_ms = (time.perf_counter() - t_start) * 1000.0
        fps = self._compute_live_fps(t_start)

        # Base64 encode for direct API/UI rendering
        _, overlay_buf = cv2.imencode(".jpg", full_overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
        overlay_b64 = "data:image/jpeg;base64," + base64.b64encode(overlay_buf).decode("utf-8")

        _, hm_buf = cv2.imencode(".jpg", heatmap_roi, [cv2.IMWRITE_JPEG_QUALITY, 85])
        heatmap_b64 = "data:image/jpeg;base64," + base64.b64encode(hm_buf).decode("utf-8")

        timings_dict = {
            "preprocess_ms": round(t_prep_ms, 2),
            "dinov2_ms": round(t_dino_ms, 2),
            "patchcore_ms": round(t_patch_ms, 2),
            "total_ms": round(t_total_ms, 2),
        }

        return {
            "frame_id": self.frame_id,
            "timestamp": now_ts,
            "status": stabilized_status,
            "raw_status": "ANOMALY" if raw_is_anomaly else "NORMAL",
            "score": round(raw_score, 4),
            "raw_score": round(raw_score, 4),
            "stabilized_score": round(stabilized_score, 4),
            "threshold": round(self.patch_threshold, 4),
            "quality": q_result.get("status", "GOOD"),
            "quality_metrics": q_result,
            "latency_ms": round(t_total_ms, 2),
            "latency": timings_dict,
            "timings": timings_dict,
            "fps": fps,
            "device": DEVICE_NAME,
            "anomaly_area_percent": round(area_pct, 2),
            "bounding_box": full_bbox,
            "bbox": full_bbox,
            "centroid": full_centroid,
            "anomaly_detected": (stabilized_status == "ANOMALY"),
            "roi": list(roi_box),
            "total_inspections": self.total_inspections,
            "total_anomalies": self.total_anomalies,
            "overlay_base64": overlay_b64,
            "heatmap_base64": heatmap_b64,
            # Raw image references (used by capture or streaming)
            "_raw_frame": frame_bgr,
            "_annotated_frame": full_overlay,
            "_heatmap_roi": heatmap_roi,
            "_distance_map": distance_map,
        }

    def _render_roi_heatmap(
        self, roi_bgr: np.ndarray, distance_map: np.ndarray, alpha: float = 0.6
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate high-contrast JET colormap and blended overlay for the ROI."""
        h, w = roi_bgr.shape[:2]
        upscaled = cv2.resize(distance_map, (w, h), interpolation=cv2.INTER_CUBIC)

        min_val = float(upscaled.min())
        max_val = float(upscaled.max())
        if max_val - min_val > 1e-5:
            norm_map = ((upscaled - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
        else:
            norm_map = np.zeros((h, w), dtype=np.uint8)

        color_heatmap = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(roi_bgr, 1.0 - alpha, color_heatmap, alpha, 0)
        return color_heatmap, overlay

    def _draw_hud_banner(
        self,
        canvas: np.ndarray,
        status: str,
        score: float,
        threshold: float,
        fps: float,
        quality_status: str,
    ) -> None:
        """Draw a sleek, industrial telemetry banner at the top of the video frame."""
        h, w = canvas.shape[:2]
        banner_h = 38

        # Dark glass translucent overlay bar
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (12, 14, 20), -1)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)

        # Status badge color
        if status == "ANOMALY":
            badge_color = (40, 50, 240)    # Industrial Red
            text_color = (255, 255, 255)
        elif status == "QUALITY WARNING":
            badge_color = (0, 160, 240)   # Industrial Amber
            text_color = (20, 20, 20)
        else:
            badge_color = (0, 200, 90)     # Emerald Green
            text_color = (20, 20, 20)

        # Status block on left
        badge_w = 110
        cv2.rectangle(canvas, (8, 6), (8 + badge_w, banner_h - 6), badge_color, -1)
        cv2.putText(
            canvas,
            status,
            (16, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            text_color,
            2,
            cv2.LINE_AA,
        )

        # Score & Threshold
        score_text = f"SCORE: {score:.3f} / {threshold:.2f}"
        cv2.putText(
            canvas,
            score_text,
            (130, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 230, 240),
            1,
            cv2.LINE_AA,
        )

        # Hardware & FPS on right
        stats_text = f"{DEVICE_NAME} | {fps:.1f} FPS | Q: {quality_status}"
        (tw, _), _ = cv2.getTextSize(stats_text, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        cv2.putText(
            canvas,
            stats_text,
            (w - tw - 12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (170, 185, 200),
            1,
            cv2.LINE_AA,
        )

    def _compute_live_fps(self, t_start: float) -> float:
        """Calculate exponential moving average FPS for live display."""
        elapsed = time.perf_counter() - t_start
        fps = 1.0 / max(0.001, elapsed)
        self._fps_tracker.append(fps)
        return round(float(np.mean(self._fps_tracker)), 1)

    def set_threshold(self, threshold: float) -> None:
        """Update anomaly decision threshold dynamically."""
        self.patch_threshold = float(threshold)
        if hasattr(self.engine, "threshold"):
            self.engine.threshold = float(threshold)
        logger.info("Updated patch anomaly threshold to: %.4f", self.patch_threshold)
