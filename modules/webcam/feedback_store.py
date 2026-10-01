# modules/webcam/feedback_store.py
"""Inspection Result Capture, Human-in-the-Loop Review, and Feedback Store.

Implements Phase 16, 17, 18, 19:
- Capturing raw/annotated/heatmap frames with result JSON
- Storing expert validation into partitioned feedback datasets (normal/anomaly/uncertain)
- Maintaining in-memory and on-disk inspection history with CSV export
- Ensuring privacy: frames are only persisted upon explicit capture or review
"""

from __future__ import annotations

import csv
import datetime
import io
import json
import logging
import os
import shutil
import uuid
from typing import Any, List, Optional

import cv2
import numpy as np

import config
from modules.utils.file_utils import ensure_dir

logger = logging.getLogger("fewvision.webcam.feedback")


class FeedbackStore:
    """Manages captures, human expert reviews, feedback datasets, and inspection history."""

    def __init__(
        self,
        captures_dir: Optional[str] = None,
        feedback_dir: Optional[str] = None,
        max_history_entries: int = 150,
    ):
        self.captures_dir = ensure_dir(
            captures_dir or getattr(config, "CAPTURES_FOLDER", os.path.join(config.DATA_FOLDER, "inspection_results"))
        )
        self.feedback_dir = ensure_dir(
            feedback_dir or getattr(config, "FEEDBACK_FOLDER", os.path.join(config.DATA_FOLDER, "feedback"))
        )
        self.feedback_normal_dir = ensure_dir(os.path.join(self.feedback_dir, "normal"))
        self.feedback_anomaly_dir = ensure_dir(os.path.join(self.feedback_dir, "anomaly"))
        self.feedback_uncertain_dir = ensure_dir(os.path.join(self.feedback_dir, "uncertain"))

        self.max_history_entries = max_history_entries
        self.history: List[dict[str, Any]] = []

    def save_capture(
        self,
        raw_frame: np.ndarray,
        annotated_frame: np.ndarray,
        heatmap_frame: Optional[np.ndarray],
        result_meta: dict[str, Any],
    ) -> dict[str, Any]:
        """Save a snapshot of the current inspection state to disk.

        Creates:
        - raw.jpg
        - annotated.jpg
        - heatmap.jpg (if available)
        - result.json

        Parameters
        ----------
        raw_frame : np.ndarray
            The un-annotated camera frame.
        annotated_frame : np.ndarray
            Frame containing heatmap overlay and bounding boxes.
        heatmap_frame : np.ndarray, optional
            The isolated heatmap colormap image.
        result_meta : dict[str, Any]
            Calculated score, status, timings, and quality metrics.

        Returns
        -------
        dict[str, Any]
            Paths and capture identifier.
        """
        now = datetime.datetime.now()
        ts_slug = now.strftime("%Y-%m-%d_%H%M%S")
        capture_id = f"cap_{uuid.uuid4().hex[:8]}"
        capture_dir_name = f"{ts_slug}_{capture_id}"
        dest_dir = ensure_dir(os.path.join(self.captures_dir, capture_dir_name))

        raw_path = os.path.join(dest_dir, "raw.jpg")
        annotated_path = os.path.join(dest_dir, "annotated.jpg")
        heatmap_path = os.path.join(dest_dir, "heatmap.jpg")
        result_path = os.path.join(dest_dir, "result.json")

        cv2.imwrite(raw_path, raw_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(annotated_path, annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        if heatmap_frame is not None and heatmap_frame.size > 0:
            cv2.imwrite(heatmap_path, heatmap_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Filter out numpy arrays from JSON metadata
        cleaned_meta = {}
        for k, v in result_meta.items():
            if not k.startswith("_") and not isinstance(v, np.ndarray):
                cleaned_meta[k] = v

        cleaned_meta.update({
            "capture_id": capture_id,
            "saved_at": now.isoformat(),
            "raw_image": "raw.jpg",
            "annotated_image": "annotated.jpg",
            "heatmap_image": "heatmap.jpg" if heatmap_frame is not None else None,
            "expert_label": None,
            "review_status": "UNREVIEWED",
        })

        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_meta, f, indent=2)

        # Log into history
        qual_val = cleaned_meta.get("quality", "UNKNOWN")
        if isinstance(qual_val, dict):
            quality_str = qual_val.get("status", "UNKNOWN")
        else:
            quality_str = str(qual_val)

        self.add_history_entry({
            "id": capture_id,
            "timestamp": now.strftime("%H:%M:%S"),
            "status": cleaned_meta.get("status", "UNKNOWN"),
            "score": cleaned_meta.get("score", 0.0),
            "quality": quality_str,
            "latency_ms": cleaned_meta.get("latency_ms", 0.0),
            "expert_decision": "Unreviewed",
            "capture_dir": capture_dir_name,
        })

        logger.info("Saved inspection capture '%s' to '%s'", capture_id, dest_dir)
        return {
            "success": True,
            "capture_id": capture_id,
            "directory": dest_dir,
            "folder_name": capture_dir_name,
            "result": cleaned_meta,
        }

    def record_expert_review(
        self,
        frame_bgr: np.ndarray,
        model_prediction: str,
        model_score: float,
        expert_label: str,  # "CONFIRM_ANOMALY", "MARK_NORMAL", "UNCERTAIN", "FALSE_POSITIVE", "FALSE_NEGATIVE"
        notes: str = "",
        quality_score: float = 0.0,
        capture_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Record human expert review and save labeled image into the feedback dataset.

        Parameters
        ----------
        frame_bgr : np.ndarray
            The image frame to save as training/validation candidate.
        model_prediction : str
            The original model prediction (e.g. "ANOMALY" or "NORMAL").
        model_score : float
            Anomaly distance score from model.
        expert_label : str
            Expert verdict.
        notes : str
            Optional notes from the expert inspector.
        quality_score : float
            Quality rating of the sample.
        capture_id : str, optional
            Related capture ID if reviewing a saved snapshot.

        Returns
        -------
        dict[str, Any]
            Feedback item summary and storage location.
        """
        now = datetime.datetime.now()
        ts_slug = now.strftime("%Y-%m-%d_%H%M%S")
        review_id = f"rev_{uuid.uuid4().hex[:8]}"
        expert_label_norm = expert_label.upper().strip()

        # Categorize into target folder
        if expert_label_norm in ("CONFIRM_ANOMALY", "FALSE_NEGATIVE", "ANOMALY"):
            category = "anomaly"
            dest_dir = self.feedback_anomaly_dir
            review_status = "CONFIRMED_ANOMALY" if expert_label_norm != "FALSE_NEGATIVE" else "FALSE_NEGATIVE"
        elif expert_label_norm in ("MARK_NORMAL", "FALSE_POSITIVE", "NORMAL"):
            category = "normal"
            dest_dir = self.feedback_normal_dir
            review_status = "CONFIRMED_NORMAL" if expert_label_norm != "FALSE_POSITIVE" else "FALSE_POSITIVE"
        else:
            category = "uncertain"
            dest_dir = self.feedback_uncertain_dir
            review_status = "UNCERTAIN"

        img_filename = f"{ts_slug}_{review_id}.jpg"
        json_filename = f"{ts_slug}_{review_id}.json"

        img_path = os.path.join(dest_dir, img_filename)
        json_path = os.path.join(dest_dir, json_filename)

        cv2.imwrite(img_path, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

        try:
            q_score_val = float(quality_score)
        except (ValueError, TypeError):
            q_score_val = 100.0 if str(quality_score).upper() == "GOOD" else (50.0 if str(quality_score).upper() == "WARNING" else 0.0)

        review_data = {
            "review_id": review_id,
            "timestamp": now.isoformat(),
            "capture_id": capture_id,
            "category": category,
            "expert_label": expert_label_norm,
            "review_status": review_status,
            "model_prediction": model_prediction,
            "model_score": float(model_score),
            "quality_score": q_score_val,
            "quality_label": str(quality_score),
            "notes": notes.strip(),
            "image_path": img_filename,
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(review_data, f, indent=2)

        # Update capture result.json if capture_id provided
        if capture_id:
            self._update_capture_review(capture_id, expert_label_norm, review_status, notes)

        # Update matching entry in history if exists
        for entry in self.history:
            if capture_id and entry.get("id") == capture_id:
                entry["expert_decision"] = review_status
                break

        logger.info(
            "Saved expert review '%s' categorized under '%s' (verdict: %s)",
            review_id,
            category,
            review_status,
        )
        return {
            "success": True,
            "review_id": review_id,
            "category": category,
            "review_status": review_status,
            "saved_file": img_path,
            "saved_path": img_path,
        }

    def _update_capture_review(
        self, capture_id: str, expert_label: str, review_status: str, notes: str
    ) -> None:
        """Update result.json in a previously saved capture directory."""
        try:
            for d in os.listdir(self.captures_dir):
                if capture_id in d:
                    r_file = os.path.join(self.captures_dir, d, "result.json")
                    if os.path.isfile(r_file):
                        with open(r_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        data["expert_label"] = expert_label
                        data["review_status"] = review_status
                        data["notes"] = notes
                        with open(r_file, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=2)
                        break
        except Exception as exc:
            logger.warning("Failed to update capture review metadata: %s", exc)

    def add_history_entry(self, entry: dict[str, Any]) -> None:
        """Add an inspection event to the rolling in-memory history table."""
        self.history.insert(0, entry)
        if len(self.history) > self.max_history_entries:
            self.history.pop()

    def get_history(self, limit: int = 50) -> List[dict[str, Any]]:
        """Retrieve recent inspections."""
        return self.history[:limit]

    def clear_history(self) -> None:
        """Clear recent history in memory."""
        self.history.clear()
        logger.info("Cleared inspection history.")

    def export_csv(self) -> str:
        """Generate a CSV string representation of the inspection history."""
        output = io.StringIO()
        fieldnames = [
            "id",
            "timestamp",
            "status",
            "score",
            "quality",
            "latency_ms",
            "expert_decision",
            "capture_dir",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in self.history:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
        return output.getvalue()
