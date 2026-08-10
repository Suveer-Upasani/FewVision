# modules/inference/video_inspector.py
"""Video Inspection Orchestrator for FewVision.

Reads a video file frame-by-frame using OpenCV, samples frames according to
FRAME_SAMPLE_INTERVAL, calls the shared InferenceEngine.inspect_frame_array()
on each sampled frame, aggregates per-frame results via temporal_aggregation,
and returns a VideoInspectionResult.

Memory usage is bounded: raw frames are discarded immediately after processing.
Only results and (optionally) saved images remain on disk.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import uuid
from typing import TYPE_CHECKING

import cv2
import numpy as np

import config
from modules.anomaly_detection.temporal_aggregation import (
    aggregate_frame_scores,
    compute_temporal_persistence,
    make_video_status,
)
from modules.inference.video_result import FrameResult, VideoInspectionResult, VideoMetadata
from modules.utils.file_utils import ensure_dir
from modules.utils.video_utils import open_video, validate_frame

if TYPE_CHECKING:
    from modules.inference.inference_engine import InferenceEngine

logger = logging.getLogger("fewvision.inference.video")


class VideoInspector:
    """Orchestrates multi-frame video anomaly inspection.

    Parameters
    ----------
    session_id : str
        Reference memory bank session ID.
    engine : InferenceEngine
        An already-loaded InferenceEngine instance.  The caller owns the engine
        lifecycle; VideoInspector never creates or destroys it.
    """

    def __init__(self, session_id: str, engine: "InferenceEngine") -> None:
        self.session_id = session_id
        self.engine = engine

        # Read video config once at construction time.
        self.sample_interval = getattr(config, "FRAME_SAMPLE_INTERVAL", 5)
        self.max_frames = getattr(config, "MAX_FRAMES", 100)
        self.aggregation_method = getattr(config, "FRAME_AGGREGATION", "top_k")
        self.top_k_frames = getattr(config, "TOP_K_FRAMES", 5)
        self.min_anomalous_frames = getattr(config, "MIN_ANOMALOUS_FRAMES", 3)
        self.temporal_threshold = getattr(config, "TEMPORAL_THRESHOLD", 0.5)
        self.save_frames = getattr(config, "SAVE_VIDEO_FRAMES", True)
        self.patch_threshold = getattr(config, "PATCH_THRESHOLD", 0.5)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inspect_video(
        self,
        video_path: str,
        run_id: str | None = None,
    ) -> VideoInspectionResult:
        """Inspect a video file and return an aggregated VideoInspectionResult.

        Parameters
        ----------
        video_path : str
            Absolute path to the video file.
        run_id : str or None
            Unique run identifier; generated automatically if not provided.

        Returns
        -------
        VideoInspectionResult
        """
        if run_id is None:
            run_id = uuid.uuid4().hex[:12]

        filename = os.path.basename(video_path)
        logger.info("Starting video inspection: %s (run_id=%s)", filename, run_id)

        # Open video and read metadata.
        cap, raw_meta = open_video(video_path)

        video_meta = VideoMetadata(
            filename=filename,
            fps=raw_meta.fps,
            total_frames=raw_meta.total_frames,
            duration=raw_meta.duration,
            width=raw_meta.width,
            height=raw_meta.height,
            sample_interval=self.sample_interval,
        )

        logger.info(
            "Video FPS: %.2f | Total frames: %d | Duration: %.2fs | "
            "Sampling interval: %d | Max frames: %d",
            raw_meta.fps,
            raw_meta.total_frames,
            raw_meta.duration,
            self.sample_interval,
            self.max_frames,
        )

        # Prepare output directories.
        base_inspect_dir = ensure_dir(
            os.path.join(config.DATA_FOLDER, "inspection", self.session_id, "video", run_id)
        )
        frames_dir = ensure_dir(os.path.join(base_inspect_dir, "frames"))
        heatmaps_dir = ensure_dir(os.path.join(base_inspect_dir, "heatmaps"))
        overlays_dir = ensure_dir(os.path.join(base_inspect_dir, "overlays"))

        frame_results: list[FrameResult] = []
        processed_count = 0

        try:
            frame_index = 0
            frame_id = 0  # sequential ID across processed frames

            while processed_count < self.max_frames:
                # Seek to the correct frame position.
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()

                if not ret:
                    # End of video or read error.
                    logger.info("No more frames to read at index %d.", frame_index)
                    break

                if not validate_frame(frame, frame_index):
                    frame_index += self.sample_interval
                    continue

                # Compute timestamp.
                fps = raw_meta.fps if raw_meta.fps > 0 else 1.0
                timestamp = frame_index / fps

                logger.info(
                    "Processing frame %d (timestamp=%.2fs, id=%d/%d)",
                    frame_index, timestamp, processed_count + 1, self.max_frames,
                )

                frame_result = self._process_frame(
                    frame=frame,
                    frame_id=frame_id,
                    frame_index=frame_index,
                    timestamp=timestamp,
                    frames_dir=frames_dir if self.save_frames else None,
                    heatmaps_dir=heatmaps_dir if self.save_frames else None,
                    overlays_dir=overlays_dir if self.save_frames else None,
                    run_id=run_id,
                )

                frame_results.append(frame_result)
                processed_count += 1
                frame_id += 1
                frame_index += self.sample_interval

        finally:
            cap.release()

        # Update processed_frames in metadata.
        video_meta.processed_frames = processed_count

        logger.info(
            "Video inspection complete — processed %d frames.",
            processed_count,
        )

        if not frame_results:
            logger.warning("No frames were successfully processed for video: %s", filename)
            return VideoInspectionResult(
                session_id=self.session_id,
                run_id=run_id,
                video_score=0.0,
                anomalous_frames=0,
                processed_frames=0,
                anomaly_persistence=0.0,
                status="NORMAL",
                aggregation_method=self.aggregation_method,
                metadata=video_meta,
                frame_results=[],
            )

        # ------------------------------------------------------------------
        # Temporal aggregation
        # ------------------------------------------------------------------
        # Use max_patch_score as the raw per-frame anomaly signal.
        raw_scores = [fr.score for fr in frame_results]

        video_score = aggregate_frame_scores(
            scores=raw_scores,
            method=self.aggregation_method,
            k=self.top_k_frames,
        )

        persistence_info = compute_temporal_persistence(
            frame_scores=raw_scores,
            threshold=self.patch_threshold,
        )

        status = make_video_status(
            video_score=video_score,
            anomalous_count=persistence_info["anomalous_frames"],
            min_anomalous_frames=self.min_anomalous_frames,
            anomaly_persistence=persistence_info["anomaly_persistence"],
            temporal_threshold=self.temporal_threshold,
            patch_threshold_score=self.patch_threshold,
        )

        logger.info(
            "Processed frames: %d | Anomalous frames: %d | "
            "Persistence: %.1f%% | Score: %.4f | Status: %s",
            processed_count,
            persistence_info["anomalous_frames"],
            persistence_info["anomaly_persistence"] * 100,
            video_score,
            status,
        )

        result = VideoInspectionResult(
            session_id=self.session_id,
            run_id=run_id,
            video_score=video_score,
            anomalous_frames=persistence_info["anomalous_frames"],
            processed_frames=processed_count,
            anomaly_persistence=persistence_info["anomaly_persistence"],
            status=status,
            aggregation_method=self.aggregation_method,
            metadata=video_meta,
            frame_results=frame_results,
        )

        # Save results.json to disk.
        self._save_results(result, base_inspect_dir)

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        frame_index: int,
        timestamp: float,
        frames_dir: str | None,
        heatmaps_dir: str | None,
        overlays_dir: str | None,
        run_id: str,
    ) -> FrameResult:
        """Run inspect_frame_array on a single BGR frame and return a FrameResult.

        Errors are caught per-frame and stored in FrameResult.error so that a
        single bad frame never crashes the entire run.

        Parameters
        ----------
        frame : np.ndarray
            BGR image from VideoCapture.
        frame_id : int
            Sequential processed-frame ID.
        frame_index : int
            Original frame index in the video.
        timestamp : float
            Timestamp in seconds.
        frames_dir : str or None
            Directory to save the original frame (None → skip).
        heatmaps_dir : str or None
            Directory to save heatmap images (None → skip).
        overlays_dir : str or None
            Directory to save overlay images (None → skip).
        run_id : str
            Current run identifier (used in URL construction).

        Returns
        -------
        FrameResult
        """
        stem = f"frame_{frame_index:06d}"

        try:
            # If saving is enabled, write the frame to the frames dir and pass
            # the per-frame output dirs to inspect_frame_array.
            if frames_dir is not None:
                frame_path = os.path.join(frames_dir, f"{stem}.jpg")
                cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

            # We route all output to a single flat directory per run so that
            # the URL prefix computed inside inspect_frame_array resolves correctly.
            output_dir = frames_dir if frames_dir is not None else ensure_dir(
                os.path.join(config.DATA_FOLDER, "inspection", self.session_id, "video", run_id, "frames")
            )

            frame_data = self.engine.inspect_frame_array(
                img_bgr=frame,
                stem=stem,
                output_dir=output_dir,
            )

            # Relocate heatmap / overlay files to their respective subdirs
            # if saving is enabled, then update the URL.
            heatmap_url = frame_data.get("heatmap_url", "")
            overlay_url = frame_data.get("overlay_url", "")

            if heatmaps_dir is not None and frame_data.get("patchcore_enabled"):
                src_heatmap = os.path.join(output_dir, f"{stem}_heatmap.png")
                dst_heatmap = os.path.join(heatmaps_dir, f"{stem}_heatmap.png")
                if os.path.isfile(src_heatmap) and src_heatmap != dst_heatmap:
                    import shutil
                    shutil.move(src_heatmap, dst_heatmap)
                    # Update URL to use heatmaps subdir
                    heatmap_url = (
                        f"/inspection/{self.session_id}/video/{run_id}/heatmaps/{stem}_heatmap.png"
                    )

            if overlays_dir is not None and frame_data.get("patchcore_enabled"):
                src_overlay = os.path.join(output_dir, f"{stem}_overlay.png")
                dst_overlay = os.path.join(overlays_dir, f"{stem}_overlay.png")
                if os.path.isfile(src_overlay) and src_overlay != dst_overlay:
                    import shutil
                    shutil.move(src_overlay, dst_overlay)
                    overlay_url = (
                        f"/inspection/{self.session_id}/video/{run_id}/overlays/{stem}_overlay.png"
                    )

            score = frame_data.get("max_patch_score", 0.0)
            is_anomaly = score >= self.patch_threshold

            return FrameResult(
                frame_id=frame_id,
                frame_index=frame_index,
                timestamp=timestamp,
                score=score,
                is_anomaly=is_anomaly,
                centroid=frame_data.get("centroid", []),
                bounding_box=frame_data.get("bounding_box", []),
                heatmap_url=heatmap_url,
                overlay_url=overlay_url,
                original_url=frame_data.get("original_url", ""),
                patchcore_enabled=frame_data.get("patchcore_enabled", False),
                padim=frame_data.get("padim", {}),
                anomaly_area_percent=frame_data.get("anomaly_area_percent", 0.0),
                error=None,
            )

        except Exception as exc:
            logger.error(
                "Failed to process frame %d (id=%d): %s",
                frame_index, frame_id, exc, exc_info=True,
            )
            return FrameResult(
                frame_id=frame_id,
                frame_index=frame_index,
                timestamp=timestamp,
                score=0.0,
                is_anomaly=False,
                error=str(exc),
            )

    def _save_results(self, result: VideoInspectionResult, output_dir: str) -> None:
        """Write results.json and inspection_summary.json to output_dir."""
        results_path = os.path.join(output_dir, "results.json")
        summary_path = os.path.join(output_dir, "inspection_summary.json")

        with open(results_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

        summary = {
            "input_type": "video",
            "session_id": result.session_id,
            "run_id": result.run_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "video_score": result.video_score,
            "status": result.status,
            "anomalous_frames": result.anomalous_frames,
            "processed_frames": result.processed_frames,
            "anomaly_persistence": result.anomaly_persistence,
            "aggregation_method": result.aggregation_method,
            "metadata": result.metadata.to_dict(),
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("Video run results saved to: %s", output_dir)
