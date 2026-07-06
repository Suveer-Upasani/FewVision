# modules/inference/inference_engine.py
"""Inference Engine for FewVision.

Coordinates image inspection pipeline by running image quality checkers,
content analyzers, extracting features using DINOv2 / ViT, searching
the Memory Bank, and computing final anomaly scores.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import uuid
from typing import List, Optional

import cv2
import numpy as np

import config
from modules.anomaly_detection.anomaly_score import compute_anomaly_score
from modules.anomaly_detection.memory_bank import MemoryBank
from modules.content.content_analysis import ContentAnalyzer
from modules.feature_extraction import get_extractor
from modules.inference.inspection_result import InspectionResult
from modules.quality.quality import ImageQualityChecker
from modules.utils.file_utils import ensure_dir

logger = logging.getLogger("fewvision.inference.engine")


class InferenceEngine:
    """Production-ready inference engine for industrial part anomaly detection.

    Loads the feature extractor and memory bank for a given session, then runs
    quality checker, content analyzer, feature extractor, and similarity searches.
    """

    def __init__(self, session_id: str):
        """Initialise the inference engine for a given session.

        Parameters
        ----------
        session_id : str
            Session ID containing the built Memory Bank.
        """
        logger.info("Initializing InferenceEngine for session %s...", session_id)
        self.session_id = session_id

        # 1. Load Memory Bank
        self.memory_bank = MemoryBank()
        self.memory_bank.load(session_id)

        # 2. Resolve and Load Feature Extractor
        extractor_name = self.memory_bank._extractor_info.get(
            "extractor_name", config.FEATURE_EXTRACTOR
        )
        logger.info("Loading feature extractor: %s", extractor_name)
        self.extractor = get_extractor(extractor_name)
        self.extractor.load_model()

        # 3. Retrieve settings
        self.top_k = getattr(config, "DEFAULT_TOP_K", 5)
        self.similarity_metric = getattr(config, "DEFAULT_SIMILARITY", "cosine")

    def predict(self, image_path: str) -> InspectionResult:
        """Run the complete inference pipeline on a single image.

        Parameters
        ----------
        image_path : str
            Path to the test image.

        Returns
        -------
        InspectionResult
            The structured inspection result containing anomaly score, labels, and metrics.
        """
        logger.info("Running quality assessment for: %s", image_path)
        # Quality assessment
        q_checker = ImageQualityChecker(image_path)
        q_metrics = q_checker.analyze()

        logger.info("Running content analysis for: %s", image_path)
        # Content analysis
        c_analyzer = ContentAnalyzer(image_path)
        c_metrics = c_analyzer.analyze()

        logger.info("Extracting features for: %s", image_path)
        # Load image as BGR BGR for model input
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image at {image_path}")

        # Extract features
        embedding = self.extractor.extract(img_bgr)

        logger.info("Searching memory bank...")
        # Search the memory bank
        search_res = self.memory_bank.search(embedding, k=self.top_k)

        indices = search_res["indices"]
        distances = search_res["distances"]
        scores = search_res["scores"]

        if not indices:
            raise ValueError("Memory bank search returned zero neighbors.")

        nearest_index = indices[0]
        nearest_reference = self.memory_bank._filenames[nearest_index]

        # Top-K neighbors dict preparation
        top_k_neighbors = []
        for rank, (idx, dist, score) in enumerate(zip(indices, distances, scores), 1):
            top_k_neighbors.append(
                {
                    "rank": rank,
                    "filename": self.memory_bank._filenames[idx],
                    "distance": float(dist),
                    "similarity": float(score),
                }
            )

        logger.info("Computing anomaly score...")
        # Anomaly scoring logic
        score_res = compute_anomaly_score(
            distances=distances, nearest_index=nearest_index
        )

        return InspectionResult(
            image_name=os.path.basename(image_path),
            image_path=image_path,
            prediction=score_res["label"],
            anomaly_score=score_res["score"],
            confidence=score_res["confidence"],
            nearest_reference=nearest_reference,
            top_k_neighbors=top_k_neighbors,
            quality_score=q_metrics.quality_score,
            content_score=c_metrics.content_score,
            quality_metrics=q_metrics.to_dict(),
            content_metrics=c_metrics.to_dict(),
        )

    def predict_batch(self, image_paths: List[str]) -> List[InspectionResult]:
        """Run the complete inference pipeline on multiple images.

        Parameters
        ----------
        image_paths : List[str]
            List of test image paths.

        Returns
        -------
        List[InspectionResult]
            List of structured inspection results.
        """
        results = []
        for idx, path in enumerate(image_paths, 1):
            try:
                logger.info(
                    "[%d/%d] Inspecting image: %s", idx, len(image_paths), path
                )
                res = self.predict(path)
                results.append(res)
            except Exception as e:
                logger.error("Failed to inspect image %s: %s", path, e, exc_info=True)
        return results

    def save_run(self, results: List[InspectionResult]) -> dict:
        """Save run details to a unique, non-overwriting run folder.

        Stores results.json, inspection_summary.json, and annotated_image.png.

        Parameters
        ----------
        results : List[InspectionResult]
            List of inspection results for the current run.

        Returns
        -------
        dict
            The contents of inspection_summary.json.
        """
        run_id = uuid.uuid4().hex[:12]
        run_dir = ensure_dir(
            os.path.join(config.INFERENCE_FOLDER, self.session_id, run_id)
        )
        logger.info("Saving inference run %s to %s", run_id, run_dir)

        # 1. results.json
        results_serialized = [r.to_dict() for r in results]
        with open(os.path.join(run_dir, "results.json"), "w") as f:
            json.dump(results_serialized, f, indent=2, default=str)

        # 2. Calculate summary statistics
        total = len(results)
        normal = sum(1 for r in results if r.prediction == "Normal")
        suspicious = sum(1 for r in results if r.prediction == "Suspicious")
        anomalous = sum(1 for r in results if r.prediction == "Anomalous")

        summary = {
            "session_id": self.session_id,
            "run_id": run_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "total_images": total,
            "normal_count": normal,
            "suspicious_count": suspicious,
            "anomalous_count": anomalous,
            "extractor_used": self.memory_bank._extractor_info.get(
                "extractor_name", "unknown"
            ),
            "metric_used": self.similarity_metric,
        }

        # Save inspection_summary.json
        with open(os.path.join(run_dir, "inspection_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        # 3. Create annotated_image.png placeholder canvas
        placeholder = np.zeros((400, 600, 3), dtype=np.uint8) + 40  # dark background
        cv2.putText(
            placeholder,
            "Inspection Heatmap Placeholder",
            (50, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (180, 180, 180),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            placeholder,
            f"Run: {run_id}",
            (50, 230),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (120, 120, 120),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            placeholder,
            "PatchCore / PaDiM visualization will render here.",
            (50, 270),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (100, 100, 100),
            1,
            cv2.LINE_AA,
        )
        cv2.imwrite(os.path.join(run_dir, "annotated_image.png"), placeholder)

        logger.info("Inspection run %s successfully saved.", run_id)
        return summary
