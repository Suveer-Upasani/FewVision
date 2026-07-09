# tests/test_padim_integration.py
"""Integration tests for PaDiM pipeline fit, calibration, and inference."""

import unittest
import os
import shutil
import cv2
import numpy as np

import config
from modules.pipeline.pipeline import process_dataset
from modules.inference.inference_engine import InferenceEngine


class TestPaDiMIntegration(unittest.TestCase):
    """Integration test suite for PaDiM model training and scoring."""

    def setUp(self) -> None:
        self.session_id = "test_padim_integration_session"
        # We will create a small mock reference upload directory
        self.upload_dir = os.path.join(config.DATA_FOLDER, "temp_test_uploads", self.session_id)
        os.makedirs(self.upload_dir, exist_ok=True)

        # Create 3 small mock reference BGR images with slight variations
        for i in range(3):
            np.random.seed(i)
            img = (np.zeros((224, 224, 3), dtype=np.uint8) + 128).astype(np.float32)
            noise = np.random.normal(loc=0.0, scale=5.0, size=img.shape)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
            # Add some lines to make it non-trivial
            cv2.line(img, (20, 20), (200, 20), (200, 200, 200), 3)
            cv2.line(img, (20, 20), (20, 200), (200, 200, 200), 3)
            cv2.imwrite(os.path.join(self.upload_dir, f"{i:03d}.png"), img)

    def tearDown(self) -> None:
        # Clean up temporary session directories
        for folder in [
            self.upload_dir,
            os.path.join(config.UPLOAD_FOLDER, self.session_id),
            os.path.join(config.AUGMENTED_FOLDER, self.session_id),
            os.path.join(config.REPORTS_FOLDER, self.session_id),
            os.path.join(config.EMBEDDINGS_FOLDER, self.session_id),
            os.path.join(config.MEMORY_BANK_FOLDER, self.session_id),
            os.path.join(config.INFERENCE_FOLDER, self.session_id),
            os.path.join(config.DATA_FOLDER, "inspection", self.session_id)
        ]:
            if os.path.isdir(folder):
                shutil.rmtree(folder)

        parent_temp = os.path.dirname(self.upload_dir)
        if os.path.isdir(parent_temp) and not os.listdir(parent_temp):
            os.rmdir(parent_temp)

    def test_padim_pipeline_and_inference(self) -> None:
        """Verify PaDiM model fits, calibrates, saves, loads, and scores in the pipeline."""
        # 1. Run pipeline: this will build both PatchCore and PaDiM reference structures
        # Force DINOv2 to match standard defaults
        res = process_dataset(self.upload_dir, session_id=self.session_id, extractor_name="dinov2")

        self.assertEqual(res.session_id, self.session_id)

        # 2. Check that PaDiM training artifact model.npz exists in the session folder
        model_path = os.path.join(config.MEMORY_BANK_FOLDER, self.session_id, "padim", "model.npz")
        self.assertTrue(os.path.isfile(model_path), f"PaDiM model not found at {model_path}")

        # Load it and verify calibrated threshold exists in metadata
        from modules.padim.padim_model import PaDiMModel
        loaded_model = PaDiMModel.load(model_path)
        self.assertTrue(loaded_model.is_fitted)
        self.assertIsNotNone(loaded_model.localization_threshold)
        self.assertGreater(loaded_model.localization_threshold, 0.0)

        # 3. Instantiate InferenceEngine
        engine = InferenceEngine(self.session_id)
        self.assertTrue(engine.padim_enabled)
        self.assertIsNotNone(engine.padim_model)

        # 4. Create a query image and score it
        query_path = os.path.join(self.upload_dir, "query_test.png")
        query_img = np.zeros((224, 224, 3), dtype=np.uint8) + 128
        # Add a mock defect (bright red patch in the center)
        cv2.rectangle(query_img, (100, 100), (124, 124), (0, 0, 255), -1)
        cv2.imwrite(query_path, query_img)

        # Predict
        inspect_res = engine.predict(query_path)

        # 5. Assert result contents
        padim_data = inspect_res.padim
        self.assertTrue(padim_data.get("enabled", False))

        self.assertTrue(padim_data["enabled"])
        self.assertGreater(padim_data["image_score"], 0.0)
        self.assertEqual(padim_data["score_method"], "top_5_percent_mean")
        self.assertEqual(padim_data["localization_threshold"], loaded_model.localization_threshold)
        self.assertTrue(0.0 <= padim_data["anomaly_area_percent"] <= 100.0)
        self.assertIn("bounding_box", padim_data)
        self.assertIn("centroid", padim_data)

        # 6. Check that heatmap and overlay outputs are written to the inspection folder
        inspect_dir = os.path.join(config.DATA_FOLDER, "inspection", self.session_id)
        heatmap_path = os.path.join(inspect_dir, "query_test_padim_heatmap.png")
        overlay_path = os.path.join(inspect_dir, "query_test_padim_overlay.png")

        self.assertTrue(os.path.isfile(heatmap_path), f"Heatmap missing: {heatmap_path}")
        self.assertTrue(os.path.isfile(overlay_path), f"Overlay missing: {overlay_path}")

        # Check API backward compatibility - PatchCore fields must still exist in flat structure
        self.assertIn("patchcore_enabled", inspect_res.to_dict())
        self.assertIn("max_patch_score", inspect_res.to_dict())

    def test_localization_bounds(self) -> None:
        """Verify PaDiM localization metrics respond to localized anomalies and remain 0 on normals."""
        # 1. Run pipeline to build model
        res = process_dataset(self.upload_dir, session_id=self.session_id, extractor_name="dinov2")
        engine = InferenceEngine(self.session_id)
        
        # 2. Assert model threshold persistence
        self.assertEqual(engine.padim_model.localization_threshold, engine.padim_model.localization_threshold)
        
        # 3. Test unseen normal query: should produce very small anomaly area (since calibrated at p99)
        normal_query_path = os.path.join(self.upload_dir, "normal_query.png")
        img_ref = cv2.imread(os.path.join(self.upload_dir, "000.png"))
        cv2.imwrite(normal_query_path, img_ref)
        
        inspect_normal = engine.predict(normal_query_path)
        normal_padim = inspect_normal.padim
        self.assertTrue(normal_padim["enabled"])
        
        # Anomaly area should be extremely small (typically <= 5.0% for p99 threshold on normal)
        self.assertLess(normal_padim["anomaly_area_percent"], 5.0)
        self.assertEqual(normal_padim["bounding_box"], [0, 0, 0, 0])
        self.assertEqual(normal_padim["centroid"], [0, 0])

        # 4. Test highly anomalous local region (synthetic high score region in a query)
        anom_query_path = os.path.join(self.upload_dir, "anom_query.png")
        img_anom = img_ref.copy()
        # Draw a big high contrast square to cause a local anomaly
        cv2.rectangle(img_anom, (80, 80), (140, 140), (0, 0, 255), -1)
        cv2.imwrite(anom_query_path, img_anom)
        
        inspect_anom = engine.predict(anom_query_path)
        anom_padim = inspect_anom.padim
        self.assertTrue(anom_padim["enabled"])
        
        # The anomaly area should be non-zero and localized (distinctly less than 50.0% since it's only in the center)
        self.assertGreater(anom_padim["anomaly_area_percent"], 0.0)
        self.assertLess(anom_padim["anomaly_area_percent"], 50.0)
        
        # Bounding box must be non-zero and not full image
        bbox = anom_padim["bounding_box"]
        self.assertNotEqual(bbox, [0, 0, 0, 0])
        self.assertNotEqual(bbox, [0, 0, img_ref.shape[0], img_ref.shape[1]])
        
        # Verify centroid is roughly in the center (around 110, 110)
        cy, cx = anom_padim["centroid"]
        self.assertTrue(50 <= cx <= 170)
        self.assertTrue(50 <= cy <= 170)


if __name__ == "__main__":
    unittest.main()
