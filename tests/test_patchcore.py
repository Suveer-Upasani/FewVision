# tests/test_patchcore.py
"""Unit tests for the PatchCore anomaly localization system."""

import unittest
import numpy as np
import os
import cv2

from modules.patchcore.patch_extractor import PatchExtractor
from modules.patchcore.patch_similarity import search_patch_neighbors
from modules.patchcore.heatmap import generate_heatmap
from modules.patchcore.localization import localize_defects


class TestPatchCore(unittest.TestCase):
    """Test suite for PatchCore modules."""

    def test_similarity_cosine(self) -> None:
        """Verify cosine similarity distance search matches shapes and bounds."""
        q = np.random.randn(196, 384).astype(np.float32)
        m = np.random.randn(1000, 384).astype(np.float32)

        # L2 normalize memory since search_patch_neighbors assumes normalized reference
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        m = m / np.where(norms == 0.0, 1.0, norms)

        idx, dist, sim = search_patch_neighbors(q, m, metric="cosine", k=5)

        self.assertEqual(idx.shape, (196, 5))
        self.assertEqual(dist.shape, (196, 5))
        self.assertEqual(sim.shape, (196, 5))

        # Cosine distance is in [0, 2]
        self.assertTrue(np.all(dist >= -1e-5))
        self.assertTrue(np.all(dist <= 2.0 + 1e-5))
        # Similarity is in [0, 1]
        self.assertTrue(np.all(sim >= 0.0))
        self.assertTrue(np.all(sim <= 1.0))

    def test_similarity_euclidean(self) -> None:
        """Verify Euclidean similarity distance search shapes."""
        q = np.random.randn(196, 384).astype(np.float32)
        m = np.random.randn(1000, 384).astype(np.float32)

        idx, dist, sim = search_patch_neighbors(q, m, metric="euclidean", k=5)

        self.assertEqual(idx.shape, (196, 5))
        self.assertEqual(dist.shape, (196, 5))
        self.assertEqual(sim.shape, (196, 5))
        self.assertTrue(np.all(dist >= 0.0))

    def test_defect_localization(self) -> None:
        """Test defect localization with thresholding."""
        # Setup dummy 14x14 distance map
        dist_map = np.zeros((14, 14), dtype=np.float32)
        # Add a simulated defect area in the center (4x4)
        dist_map[5:9, 5:9] = 0.8

        loc = localize_defects(dist_map, (224, 224), threshold=0.5)

        self.assertIn("bbox", loc)
        self.assertIn("area_percent", loc)
        self.assertIn("max_score", loc)
        self.assertIn("center", loc)

        self.assertAlmostEqual(loc["max_score"], 0.8)
        # 16 patches out of 196 are active, which is about 8.16%
        # The upscaled binary mask will match this region
        self.assertGreater(loc["area_percent"], 5.0)
        self.assertLess(loc["area_percent"], 12.0)

        # Centroid should be near the center of the image (112, 112)
        cy, cx = loc["center"]
        self.assertTrue(90 <= cy <= 130)
        self.assertTrue(90 <= cx <= 130)

    def test_heatmap_generation(self) -> None:
        """Verify heatmap creation and blending."""
        # Create a temporary dummy image
        temp_img_path = "temp_test_image.png"
        dummy_img = np.zeros((224, 224, 3), dtype=np.uint8) + 128
        cv2.imwrite(temp_img_path, dummy_img)

        try:
            dist_map = np.random.rand(14, 14).astype(np.float32)
            heatmap, overlay = generate_heatmap(temp_img_path, dist_map, alpha=0.6)

            self.assertEqual(heatmap.shape, (224, 224, 3))
            self.assertEqual(overlay.shape, (224, 224, 3))
        finally:
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)


if __name__ == "__main__":
    unittest.main()
