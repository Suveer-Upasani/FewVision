"""Integration tests for the FewVision live webcam inspection pipeline.

Tests end-to-end inference flow:
Frame -> Quality Check -> ROI -> DINOv2 -> PatchCore Memory Bank -> Anomaly Score -> Heatmap -> Stabilization
"""

import unittest
import os
import numpy as np
import cv2

from modules.webcam.live_inspector import LiveWebcamInspector
from modules.webcam.frame_processor import WebcamFrameProcessor
import config


class TestLiveWebcamInspectorIntegration(unittest.TestCase):
    """End-to-end integration test for LiveWebcamInspector."""

    @classmethod
    def setUpClass(cls):
        # Locate an existing session with a memory bank or use default
        mb_dir = getattr(config, "MEMORY_BANKS_FOLDER", config.MEMORY_BANK_FOLDER)
        available_sessions = [
            d for d in os.listdir(mb_dir)
            if os.path.isfile(os.path.join(mb_dir, d, "memory.npy"))
            or os.path.isfile(os.path.join(mb_dir, d, "patchcore", "memory.npy"))
            or os.path.isfile(os.path.join(mb_dir, d, "memory_bank.npy"))
        ] if os.path.isdir(mb_dir) else []

        if not available_sessions:
            raise unittest.SkipTest("No trained memory bank session found for integration testing.")

        cls.session_id = available_sessions[0]
        cls.inspector = LiveWebcamInspector(
            session_id=cls.session_id,
            temporal_window=3,
            anomaly_vote_threshold=0.5,
        )

    def test_single_load_and_warmup(self):
        """Inspector must initialize engine once, warm up, and record device."""
        self.assertIsNotNone(self.inspector.engine)
        self.assertIn(str(self.inspector.device), ["cpu", "cuda"])

    def test_end_to_end_frame_inspection(self):
        """Pass a real industrial test image through inspect_frame and verify result schema."""
        test_img_path = os.path.join(config.BASE_DIR, "Tiles_dataset", "000.png")
        if not os.path.isfile(test_img_path):
            test_img_path = os.path.join(config.BASE_DIR, "Tiles_dataset", "test", "cracks", "000.png")

        if os.path.isfile(test_img_path):
            frame = cv2.imread(test_img_path)
        else:
            frame = np.full((480, 640, 3), 128, dtype=np.uint8)
            cv2.circle(frame, (320, 240), 100, (200, 200, 200), -1)

        result = self.inspector.inspect_frame(frame, roi_mode="center")

        # Verify required keys
        expected_keys = [
            "frame_id", "timestamp", "status", "score", "raw_score",
            "threshold", "quality", "quality_metrics", "roi", "bbox",
            "device", "fps", "latency", "overlay_base64", "heatmap_base64",
            "anomaly_detected"
        ]
        for key in expected_keys:
            self.assertIn(key, result, f"Missing key in inspection result: {key}")

        # Check numeric validity
        self.assertIsInstance(result["score"], (float, int))
        self.assertFalse(np.isnan(result["score"]), "Score must not be NaN")
        self.assertFalse(np.isinf(result["score"]), "Score must not be infinite")

        # Verify latencies
        lat = result["latency"]
        self.assertIn("preprocess_ms", lat)
        self.assertIn("dinov2_ms", lat)
        self.assertIn("patchcore_ms", lat)
        self.assertIn("total_ms", lat)
        self.assertGreater(lat["total_ms"], 0.0)

        # Verify status is one of the permitted states
        self.assertIn(result["status"], ["NORMAL", "ANOMALY", "QUALITY WARNING"])

        # Verify overlay and heatmap base64 strings
        self.assertIsInstance(result["overlay_base64"], str)
        self.assertGreater(len(result["overlay_base64"]), 50)
        self.assertIsInstance(result["heatmap_base64"], str)
        self.assertGreater(len(result["heatmap_base64"]), 50)

    def test_temporal_stabilization(self):
        """Sequential frames should update rolling scores and smoothed values."""
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        cv2.circle(frame, (320, 240), 120, (220, 220, 220), -1)
        noise = np.random.randint(0, 40, (480, 640, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)

        r1 = self.inspector.inspect_frame(frame)
        r2 = self.inspector.inspect_frame(frame)
        r3 = self.inspector.inspect_frame(frame)

        self.assertEqual(len(self.inspector.score_history), 3)
        self.assertFalse(np.isnan(r3["score"]))

    def test_bad_quality_frame_handling(self):
        """Completely black frame should be flagged as BAD quality without crashing."""
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.inspector.inspect_frame(black_frame)

        self.assertEqual(result["quality"], "BAD")
        self.assertEqual(result["status"], "QUALITY WARNING")
        self.assertIn("IMAGE QUALITY TOO LOW", result["message"])

    def test_dynamic_threshold_update(self):
        """Updating threshold must update both inspector and threshold property."""
        self.inspector.set_threshold(0.75)
        self.assertEqual(self.inspector.threshold, 0.75)


if __name__ == "__main__":
    unittest.main()
