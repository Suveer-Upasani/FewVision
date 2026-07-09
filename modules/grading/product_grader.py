# modules/grading/product_grader.py
"""Deterministic Product Grading Engine for FewVision.

Evaluates an InspectionResult against rule-based criteria combining
global anomaly prediction, localization evidence, quality, and content scores.
"""

from typing import Any, Dict, List
import logging

import config
from modules.inference.inspection_result import InspectionResult
from modules.grading.grading_result import ProductGrade

logger = logging.getLogger("fewvision.grading.grader")


class ProductGrader:
    """Evaluates inspection results to assign PASS/REVIEW/FAIL grades."""

    def __init__(
        self,
        quality_threshold: float | None = None,
        content_threshold: float | None = None,
        patchcore_area_threshold: float | None = None,
        padim_area_threshold: float | None = None,
        severe_patchcore_area: float | None = None,
        severe_padim_area: float | None = None,
    ) -> None:
        # Load thresholds with fallbacks to config.py values
        self.quality_threshold = (
            quality_threshold if quality_threshold is not None else getattr(config, "GRADING_QUALITY_THRESHOLD", 70.0)
        )
        self.content_threshold = (
            content_threshold if content_threshold is not None else getattr(config, "GRADING_CONTENT_THRESHOLD", 80.0)
        )
        self.patchcore_area_threshold = (
            patchcore_area_threshold
            if patchcore_area_threshold is not None
            else getattr(config, "GRADING_PATCHCORE_AREA_THRESHOLD", 1.0)
        )
        self.padim_area_threshold = (
            padim_area_threshold
            if padim_area_threshold is not None
            else getattr(config, "GRADING_PADIM_AREA_THRESHOLD", 1.0)
        )
        self.severe_patchcore_area = (
            severe_patchcore_area
            if severe_patchcore_area is not None
            else getattr(config, "GRADING_SEVERE_PATCHCORE_AREA", 5.0)
        )
        self.severe_padim_area = (
            severe_padim_area
            if severe_padim_area is not None
            else getattr(config, "GRADING_SEVERE_PADIM_AREA", 5.0)
        )

    def grade_product(self, result: InspectionResult) -> ProductGrade:
        """Evaluate an InspectionResult and return a structured ProductGrade.

        Parameters
        ----------
        result : InspectionResult
            The inspection result object to evaluate.

        Returns
        -------
        ProductGrade
            The assigned grade, confidence, reason codes, and evidence summary.
        """
        # 1. Extract and normalize evidence fields
        prediction = result.prediction
        global_conf = result.confidence
        
        quality_score = result.quality_score
        content_score = result.content_score

        patchcore_enabled = result.patchcore_enabled
        patchcore_area = result.anomaly_area_percent
        
        padim_enabled = result.padim.get("enabled", False) if isinstance(result.padim, dict) else False
        padim_area = result.padim.get("anomaly_area_percent", 0.0) if isinstance(result.padim, dict) else 0.0

        # Determine logical defect flags
        pc_defect = patchcore_enabled and (patchcore_area >= self.patchcore_area_threshold)
        padim_defect = padim_enabled and (padim_area >= self.padim_area_threshold)
        
        pc_severe = patchcore_enabled and (patchcore_area >= self.severe_patchcore_area)
        padim_severe = padim_enabled and (padim_area >= self.severe_padim_area)

        # 2. Build structured evidence dictionary
        evidence = {
            "global_prediction": prediction,
            "global_confidence": global_conf,
            "quality_score": quality_score,
            "content_score": content_score,
            "patchcore_enabled": patchcore_enabled,
            "patchcore_area_percent": patchcore_area,
            "patchcore_defect_detected": pc_defect,
            "padim_enabled": padim_enabled,
            "padim_area_percent": padim_area,
            "padim_defect_detected": padim_defect,
        }

        # Initialize collections
        reason_codes: List[str] = []
        reasons: List[str] = []

        # 3. Evaluate rules (FAIL -> REVIEW -> PASS hierarchy)
        
        # Rule check: Agreement between both localization methods
        agreement = pc_defect and padim_defect
        
        # Rule check: Severe area defect
        severe_defect = pc_severe or padim_severe

        # Rule check: Strongly anomalous global status with localization confirmation
        global_anomalous = (prediction == "Anomalous")
        global_anomaly_confirmed = global_anomalous and (pc_defect or padim_defect)

        # A. FAIL Rules
        if agreement or severe_defect or global_anomaly_confirmed:
            grade = "FAIL"
            if agreement:
                reason_codes.append("LOCALIZATION_AGREEMENT")
                reasons.append("Both PatchCore and PaDiM localization algorithms detected a localized defect.")
            if pc_severe:
                reason_codes.append("SEVERE_PATCHCORE_AREA")
                reasons.append(f"PatchCore detected a severe anomalous defect area of {patchcore_area:.2f}%.")
            if padim_severe:
                reason_codes.append("SEVERE_PADIM_AREA")
                reasons.append(f"PaDiM detected a severe anomalous defect area of {padim_area:.2f}%.")
            if global_anomaly_confirmed and not agreement and not severe_defect:
                reason_codes.append("GLOBAL_ANOMALY_WITH_LOCALIZATION")
                reasons.append("Global anomaly classification is confirmed by localized defect evidence.")

            # Compute FAIL confidence
            if agreement:
                conf = 0.90 + 0.05 * min(1.0, max(patchcore_area, padim_area) / 50.0)
            elif severe_defect:
                conf = 0.85
            else:
                conf = 0.80 + 0.08 * global_conf

        # B. REVIEW Rules
        elif (
            (patchcore_enabled and padim_enabled and (pc_defect != padim_defect))
            or (prediction == "Suspicious")
            or (global_anomalous and not pc_defect and not padim_defect)
            or (prediction == "Normal" and (pc_defect or padim_defect))
            or (quality_score < self.quality_threshold)
            or (content_score < self.content_threshold)
            or (not patchcore_enabled or not padim_enabled)
        ):
            grade = "REVIEW"
            conf = 0.70  # Default baseline for review

            # Determine specific reasons and adjust confidence for conflicts
            if patchcore_enabled and padim_enabled and (pc_defect != padim_defect):
                conf = 0.60  # Conflict lowers confidence
                if pc_defect:
                    reason_codes.append("PATCHCORE_ONLY_DEFECT")
                    reasons.append("PatchCore detected a local anomaly, but PaDiM did not confirm it.")
                else:
                    reason_codes.append("PADIM_ONLY_DEFECT")
                    reasons.append("PaDiM detected a local anomaly, but PatchCore did not confirm it.")

            if prediction == "Suspicious":
                reason_codes.append("GLOBAL_SUSPICIOUS")
                reasons.append("Global classification status is Suspicious.")
                conf = 0.65 + 0.10 * global_conf

            if global_anomalous and not pc_defect and not padim_defect:
                reason_codes.append("GLOBAL_LOCALIZATION_CONFLICT")
                reasons.append("Global status is Anomalous, but localization did not detect any defect.")
                conf = 0.55

            if prediction == "Normal" and (pc_defect or padim_defect):
                reason_codes.append("GLOBAL_LOCALIZATION_CONFLICT")
                reasons.append("Global status is Normal, but localization detected anomalous patches.")
                conf = 0.60

            if quality_score < self.quality_threshold:
                reason_codes.append("LOW_QUALITY")
                reasons.append(f"Image quality score ({quality_score:.1f}) is below acceptable limit ({self.quality_threshold:.1f}).")
                conf = min(conf, 0.70 + 0.10 * (quality_score / 100.0))

            if content_score < self.content_threshold:
                reason_codes.append("CONTENT_MISMATCH")
                reasons.append(f"Content alignment score ({content_score:.1f}) is below acceptable limit ({self.content_threshold:.1f}).")
                conf = min(conf, 0.70 + 0.10 * (content_score / 100.0))

            if not patchcore_enabled or not padim_enabled:
                reason_codes.append("INCOMPLETE_EVIDENCE")
                reasons.append("Incomplete localization evidence; one or both models are disabled.")

        # C. PASS Rules
        else:
            grade = "PASS"
            reason_codes.append("ALL_SIGNALS_CLEAN")
            reasons.append("All anomaly detection and quality metrics satisfy acceptable limits.")
            conf = 0.85 + 0.10 * global_conf + 0.05 * (quality_score / 100.0)

        # Enforce exact bounds [0.0, 1.0] and round to 2 decimals
        bounded_conf = round(max(0.0, min(1.0, conf)), 2)

        return ProductGrade(
            grade=grade,
            confidence=bounded_conf,
            reason_codes=reason_codes,
            reasons=reasons,
            evidence=evidence,
        )
