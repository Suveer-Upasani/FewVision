# tests/test_product_grader.py
"""Unit tests for Product Grading Core decision engine."""

import unittest
from typing import Any, Dict

from modules.inference.inspection_result import InspectionResult
from modules.grading.product_grader import ProductGrader


class TestProductGrader(unittest.TestCase):
    """Unit test suite for rule-based Product Grading Core."""

    def setUp(self) -> None:
        # Default grader instance under default config/thresholds
        self.grader = ProductGrader(
            quality_threshold=70.0,
            content_threshold=80.0,
            patchcore_area_threshold=1.0,
            padim_area_threshold=1.0,
            severe_patchcore_area=5.0,
            severe_padim_area=5.0,
        )

    def _create_mock_result(
        self,
        prediction: str = "Normal",
        confidence: float = 0.90,
        quality_score: float = 85.0,
        content_score: float = 90.0,
        patchcore_enabled: bool = True,
        patchcore_area: float = 0.0,
        padim_enabled: bool = True,
        padim_area: float = 0.0,
    ) -> InspectionResult:
        """Helper to create a mocked InspectionResult."""
        padim_res = {
            "enabled": padim_enabled,
            "anomaly_area_percent": padim_area,
            "image_score": 120.0,
        }
        return InspectionResult(
            image_name="test_tile.png",
            image_path="/dummy/path/test_tile.png",
            prediction=prediction,
            anomaly_score=10.0 if prediction == "Normal" else 60.0,
            confidence=confidence,
            nearest_reference="ref_tile.png",
            quality_score=quality_score,
            content_score=content_score,
            patchcore_enabled=patchcore_enabled,
            anomaly_area_percent=patchcore_area,
            padim=padim_res,
        )

    def test_clean_normal_image_pass(self) -> None:
        """1. Clean normal image with no anomalies and good quality should result in PASS."""
        res = self._create_mock_result(
            prediction="Normal",
            confidence=0.95,
            quality_score=95.0,
            content_score=98.0,
            patchcore_area=0.2,
            padim_area=0.1,
        )
        grade = self.grader.grade_product(res)
        self.assertEqual(grade.grade, "PASS")
        self.assertIn("ALL_SIGNALS_CLEAN", grade.reason_codes)
        self.assertGreater(grade.confidence, 0.85)
        self.assertLessEqual(grade.confidence, 1.00)

    def test_patchcore_and_padim_defects_fail(self) -> None:
        """2. When both PatchCore and PaDiM detect localized defects, result must be FAIL."""
        res = self._create_mock_result(
            prediction="Normal",
            patchcore_area=1.5,  # >= 1.0% threshold
            padim_area=1.8,      # >= 1.0% threshold
        )
        grade = self.grader.grade_product(res)
        self.assertEqual(grade.grade, "FAIL")
        self.assertIn("LOCALIZATION_AGREEMENT", grade.reason_codes)
        self.assertGreaterEqual(grade.confidence, 0.90)

    def test_patchcore_defect_only_review(self) -> None:
        """3. When only PatchCore detects a defect and global status is Normal, result is REVIEW."""
        res = self._create_mock_result(
            prediction="Normal",
            patchcore_area=1.5,
            padim_area=0.2,
        )
        grade = self.grader.grade_product(res)
        self.assertEqual(grade.grade, "REVIEW")
        self.assertIn("PATCHCORE_ONLY_DEFECT", grade.reason_codes)
        self.assertEqual(grade.confidence, 0.60)

    def test_padim_defect_only_review(self) -> None:
        """4. When only PaDiM detects a defect and global status is Normal, result is REVIEW."""
        res = self._create_mock_result(
            prediction="Normal",
            patchcore_area=0.2,
            padim_area=1.5,
        )
        grade = self.grader.grade_product(res)
        self.assertEqual(grade.grade, "REVIEW")
        self.assertIn("PADIM_ONLY_DEFECT", grade.reason_codes)
        self.assertEqual(grade.confidence, 0.60)

    def test_global_anomalous_with_localization_fail(self) -> None:
        """5. Global status Anomalous + at least one localization method detects defect -> FAIL."""
        res = self._create_mock_result(
            prediction="Anomalous",
            confidence=0.85,
            patchcore_area=1.2,
            padim_area=0.2,
        )
        grade = self.grader.grade_product(res)
        self.assertEqual(grade.grade, "FAIL")
        self.assertIn("GLOBAL_ANOMALY_WITH_LOCALIZATION", grade.reason_codes)
        self.assertGreaterEqual(grade.confidence, 0.80)

    def test_global_suspicious_no_localization_review(self) -> None:
        """6. Global status Suspicious with clean localization results in REVIEW."""
        res = self._create_mock_result(
            prediction="Suspicious",
            confidence=0.75,
            patchcore_area=0.1,
            padim_area=0.2,
        )
        grade = self.grader.grade_product(res)
        self.assertEqual(grade.grade, "REVIEW")
        self.assertIn("GLOBAL_SUSPICIOUS", grade.reason_codes)
        self.assertGreater(grade.confidence, 0.60)

    def test_global_normal_low_quality_only_review(self) -> None:
        """7. Global Normal + quality below threshold results in REVIEW."""
        res = self._create_mock_result(
            prediction="Normal",
            quality_score=65.0,  # Below 70
        )
        grade = self.grader.grade_product(res)
        self.assertEqual(grade.grade, "REVIEW")
        self.assertIn("LOW_QUALITY", grade.reason_codes)

    def test_global_normal_content_mismatch_only_review(self) -> None:
        """8. Global Normal + content below threshold results in REVIEW."""
        res = self._create_mock_result(
            prediction="Normal",
            content_score=75.0,  # Below 80
        )
        grade = self.grader.grade_product(res)
        self.assertEqual(grade.grade, "REVIEW")
        self.assertIn("CONTENT_MISMATCH", grade.reason_codes)

    def test_conflicting_global_anomalous_no_localization_review(self) -> None:
        """9. Global status Anomalous but no localization defect detected results in REVIEW."""
        res = self._create_mock_result(
            prediction="Anomalous",
            patchcore_area=0.2,
            padim_area=0.3,
        )
        grade = self.grader.grade_product(res)
        self.assertEqual(grade.grade, "REVIEW")
        self.assertIn("GLOBAL_LOCALIZATION_CONFLICT", grade.reason_codes)

    def test_missing_padim_result_review(self) -> None:
        """10. Missing PaDiM result (disabled/None) defaults gracefully to REVIEW."""
        res = self._create_mock_result(
            prediction="Normal",
            padim_enabled=False,
        )
        grade = self.grader.grade_product(res)
        self.assertEqual(grade.grade, "REVIEW")
        self.assertIn("INCOMPLETE_EVIDENCE", grade.reason_codes)

    def test_missing_patchcore_result_review(self) -> None:
        """11. Missing PatchCore result (disabled) defaults gracefully to REVIEW."""
        res = self._create_mock_result(
            prediction="Normal",
            patchcore_enabled=False,
        )
        grade = self.grader.grade_product(res)
        self.assertEqual(grade.grade, "REVIEW")
        self.assertIn("INCOMPLETE_EVIDENCE", grade.reason_codes)

    def test_boundary_values_area_thresholds(self) -> None:
        """12. Boundary values for PatchCore and PaDiM anomaly area thresholds."""
        # Just below threshold (0.99%) -> Normal
        res_below = self._create_mock_result(patchcore_area=0.99, padim_area=0.99)
        grade_below = self.grader.grade_product(res_below)
        self.assertEqual(grade_below.grade, "PASS")

        # Exactly at threshold (1.00%) -> agreement -> FAIL
        res_exact = self._create_mock_result(patchcore_area=1.00, padim_area=1.00)
        grade_exact = self.grader.grade_product(res_exact)
        self.assertEqual(grade_exact.grade, "FAIL")

    def test_boundary_values_quality_and_content_thresholds(self) -> None:
        """12. Boundary values for quality and content thresholds."""
        # Exactly at quality threshold (70.0) -> PASS
        res_q_exact = self._create_mock_result(quality_score=70.0, content_score=80.0)
        grade_q_exact = self.grader.grade_product(res_q_exact)
        self.assertEqual(grade_q_exact.grade, "PASS")

        # Just below quality threshold (69.9) -> REVIEW
        res_q_below = self._create_mock_result(quality_score=69.9, content_score=80.0)
        grade_q_below = self.grader.grade_product(res_q_below)
        self.assertEqual(grade_q_below.grade, "REVIEW")

    def test_confidence_range_enforcement(self) -> None:
        """13. Enforce that confidence is always within [0.0, 1.0]."""
        res = self._create_mock_result(prediction="Anomalous", confidence=0.0)
        grade = self.grader.grade_product(res)
        self.assertTrue(0.0 <= grade.confidence <= 1.0)

        res_max = self._create_mock_result(prediction="Normal", confidence=1.0, quality_score=100.0)
        grade_max = self.grader.grade_product(res_max)
        self.assertTrue(0.0 <= grade_max.confidence <= 1.0)

    def test_reasons_and_codes_are_deterministic(self) -> None:
        """14. Reasons and codes are deterministic and unchanged across multiple evaluations."""
        res = self._create_mock_result(prediction="Suspicious")
        grade1 = self.grader.grade_product(res)
        grade2 = self.grader.grade_product(res)
        self.assertEqual(grade1.grade, grade2.grade)
        self.assertEqual(grade1.reason_codes, grade2.reason_codes)
        self.assertEqual(grade1.reasons, grade2.reasons)

    def test_input_objects_are_not_mutated(self) -> None:
        """15. Evaluation does not mutate original InspectionResult."""
        res = self._create_mock_result(prediction="Normal")
        original_dict = res.to_dict()
        _ = self.grader.grade_product(res)
        self.assertEqual(res.to_dict(), original_dict)


if __name__ == "__main__":
    unittest.main()
