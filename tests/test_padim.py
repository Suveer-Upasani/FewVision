# tests/test_padim.py
"""Unit tests for the isolated PaDiM anomaly detection model core."""

import unittest
import os
import tempfile
import numpy as np

from modules.padim.padim_model import PaDiMModel


class TestPaDiMModel(unittest.TestCase):
    """Test suite for PaDiMModel core logic, statistics, and numeric correctness."""

    def setUp(self) -> None:
        self.random_seed = 42
        self.N = 10
        self.H = 14
        self.W = 14
        self.D = 384  # DINOv2-S default dimension
        self.d = 50   # target reduced dimension

        # Generate synthetic normal training data: shape (N, H, W, D)
        np.random.seed(self.random_seed)
        self.ref_data = np.random.normal(loc=0.0, scale=1.0, size=(self.N, self.H, self.W, self.D)).astype(np.float32)

    def test_fit_shape_validation(self) -> None:
        """1. Verify fit shape validation checks dimensions."""
        model = PaDiMModel(d=self.d, random_seed=self.random_seed)

        # Invalid 3D array
        with self.assertRaises(ValueError):
            model.fit(np.zeros((self.H, self.W, self.D), dtype=np.float32))

        # Invalid 5D array
        with self.assertRaises(ValueError):
            model.fit(np.zeros((1, self.N, self.H, self.W, self.D), dtype=np.float32))

    def test_query_shape_validation(self) -> None:
        """2. Verify query shape validation checks dimensions and grid compatibility."""
        model = PaDiMModel(d=self.d, random_seed=self.random_seed)
        model.fit(self.ref_data)

        # Query before fit raises error
        unfitted_model = PaDiMModel(d=self.d, random_seed=self.random_seed)
        with self.assertRaises(RuntimeError):
            unfitted_model.score(np.zeros((self.H, self.W, self.D), dtype=np.float32))

        # Query is 4D (expects 3D)
        with self.assertRaises(ValueError):
            model.score(np.zeros((1, self.H, self.W, self.D), dtype=np.float32))

        # Grid mismatch (e.g. 10x10 instead of 14x14)
        with self.assertRaises(ValueError):
            model.score(np.zeros((10, 10, self.D), dtype=np.float32))

        # Dimension mismatch (e.g. 128 channels instead of 384)
        with self.assertRaises(ValueError):
            model.score(np.zeros((self.H, self.W, 128), dtype=np.float32))

    def test_deterministic_channel_selection(self) -> None:
        """3. Verify deterministic channel selection is repeatable."""
        model1 = PaDiMModel(d=self.d, random_seed=self.random_seed)
        model1.fit(self.ref_data)

        model2 = PaDiMModel(d=self.d, random_seed=self.random_seed)
        model2.fit(self.ref_data)

        # Verify selected channels match exactly
        np.testing.assert_array_equal(model1.selected_channels, model2.selected_channels)
        self.assertEqual(len(model1.selected_channels), self.d)

        # Different seed chooses different channels
        model3 = PaDiMModel(d=self.d, random_seed=100)
        model3.fit(self.ref_data)
        self.assertFalse(np.array_equal(model1.selected_channels, model3.selected_channels))

    def test_expected_statistics_shape(self) -> None:
        """4. Verify expected shapes of mean and covariance parameters."""
        model = PaDiMModel(d=self.d, random_seed=self.random_seed)
        model.fit(self.ref_data)

        # means shape should be (H, W, d)
        self.assertEqual(model.means.shape, (self.H, self.W, self.d))
        # inv_covs shape should be (H, W, d, d)
        self.assertEqual(model.inv_covs.shape, (self.H, self.W, self.d, self.d))

    def test_finite_anomaly_scores(self) -> None:
        """5. Verify scores contain only finite float32 values."""
        model = PaDiMModel(d=self.d, random_seed=self.random_seed)
        model.fit(self.ref_data)

        query = np.random.normal(loc=0.0, scale=1.0, size=(self.H, self.W, self.D)).astype(np.float32)
        score_map = model.score(query)

        self.assertEqual(score_map.shape, (self.H, self.W))
        self.assertEqual(score_map.dtype, np.float32)
        self.assertTrue(np.isfinite(score_map).all())
        self.assertTrue((score_map >= 0.0).all())

    def test_normal_vs_anomalous_scoring(self) -> None:
        """6. Verify normal query scores lower than anomalous query."""
        # Create normal training data with low variance around mean=0
        ref = np.random.normal(loc=0.0, scale=0.1, size=(self.N, self.H, self.W, self.D)).astype(np.float32)

        model = PaDiMModel(d=self.d, random_seed=self.random_seed)
        model.fit(ref)

        # Normal query matches the normal distribution (mean=0)
        normal_query = np.random.normal(loc=0.0, scale=0.1, size=(self.H, self.W, self.D)).astype(np.float32)
        # Anomalous query is a significant outlier (mean=10)
        anomalous_query = np.random.normal(loc=10.0, scale=0.1, size=(self.H, self.W, self.D)).astype(np.float32)

        normal_score = model.score(normal_query)
        anomalous_score = model.score(anomalous_query)

        # Anomaly scores should reflect the statistical distance
        self.assertLess(normal_score.mean(), anomalous_score.mean())

    def test_n_less_than_d(self) -> None:
        """7. Verify covariance regularization holds and doesn't crash when N < d."""
        small_N = 3
        ref_small = np.random.normal(loc=0.0, scale=1.0, size=(small_N, self.H, self.W, self.D)).astype(np.float32)

        # Here small_N = 3, d = 50. Since N < d, empirical covariance is singular (rank deficiency).
        model = PaDiMModel(d=self.d, regularization=0.01, random_seed=self.random_seed)
        
        # Fit should complete without LinAlgError due to diagonal regularization
        model.fit(ref_small)
        self.assertTrue(model.is_fitted)

        # Score must also succeed
        query = np.random.normal(loc=0.0, scale=1.0, size=(self.H, self.W, self.D)).astype(np.float32)
        score_map = model.score(query)
        self.assertTrue(np.isfinite(score_map).all())

    def test_nan_rejection(self) -> None:
        """8. Verify input features containing NaNs are rejected."""
        model = PaDiMModel(d=self.d, random_seed=self.random_seed)

        # NaN in fit
        bad_ref = self.ref_data.copy()
        bad_ref[0, 0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            model.fit(bad_ref)

        # NaN in score
        model.fit(self.ref_data)
        bad_query = np.zeros((self.H, self.W, self.D), dtype=np.float32)
        bad_query[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            model.score(bad_query)

    def test_inf_rejection(self) -> None:
        """9. Verify input features containing Infs are rejected."""
        model = PaDiMModel(d=self.d, random_seed=self.random_seed)

        # Inf in fit
        bad_ref = self.ref_data.copy()
        bad_ref[0, 0, 0, 0] = np.inf
        with self.assertRaises(ValueError):
            model.fit(bad_ref)

        # Inf in score
        model.fit(self.ref_data)
        bad_query = np.zeros((self.H, self.W, self.D), dtype=np.float32)
        bad_query[0, 0, 0] = np.inf
        with self.assertRaises(ValueError):
            model.score(bad_query)

    def test_save_load_score_equivalence(self) -> None:
        """10. Verify model persistence roundtrip preserves output scores identically."""
        model = PaDiMModel(d=self.d, random_seed=self.random_seed)
        model.fit(self.ref_data)

        query = np.random.normal(loc=0.5, scale=1.0, size=(self.H, self.W, self.D)).astype(np.float32)
        original_score = model.score(query)

        # Save to temp file
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, "test_padim_save.npz")
        try:
            model.save(temp_path)
            self.assertTrue(os.path.exists(temp_path))

            # Load model
            loaded_model = PaDiMModel.load(temp_path)
            loaded_score = loaded_model.score(query)

            # Assert complete equality within tolerance
            np.testing.assert_allclose(original_score, loaded_score, rtol=1e-5, atol=1e-6)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_output_anomaly_map_shape(self) -> None:
        """11. Verify output map shape equals spatial grid shape."""
        model = PaDiMModel(d=self.d, random_seed=self.random_seed)
        model.fit(self.ref_data)

        query = np.random.normal(loc=0.0, scale=1.0, size=(self.H, self.W, self.D)).astype(np.float32)
        score_map = model.score(query)

        self.assertEqual(score_map.shape, (self.H, self.W))


if __name__ == "__main__":
    unittest.main()
