"""Unit tests for the FewVision live webcam inspection module."""

import unittest
import os
import shutil
import tempfile
import numpy as np
import cv2

from modules.webcam.webcam_manager import WebcamManager, SimulatedStream
from modules.webcam.frame_processor import WebcamFrameProcessor
from modules.webcam.feedback_store import FeedbackStore
import config


class TestWebcamFrameProcessor(unittest.TestCase):
    """Unit tests for frame quality assessment and ROI cropping."""

    def setUp(self):
        self.processor = WebcamFrameProcessor()

    def test_good_quality_frame(self):
        """A normal textured frame should pass as GOOD quality."""
        # Create a synthetic image with good contrast and edges
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (100, 100), (300, 300), (200, 200, 200), -1)
        cv2.circle(frame, (450, 250), 80, (150, 150, 50), -1)
        # Add high-frequency texture to ensure high Laplacian variance
        noise = np.random.randint(0, 50, (480, 640, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)

        metrics, status, reason = self.processor.assess_quality(frame)
        self.assertIn("blur_score", metrics)
        self.assertIn("mean_brightness", metrics)
        self.assertIn("contrast_std", metrics)
        self.assertEqual(status, "GOOD")
        self.assertEqual(reason, "OK")

    def test_very_dark_frame_bad_quality(self):
        """A very dark frame (< brightness_min) should be flagged as BAD."""
        dark_frame = np.full((480, 640, 3), 10, dtype=np.uint8)
        metrics, status, reason = self.processor.assess_quality(dark_frame)
        self.assertEqual(status, "BAD")
        self.assertIn("dark", reason.lower())

    def test_overexposed_frame_bad_quality(self):
        """An overexposed frame (> brightness_max) should be flagged as BAD."""
        bright_frame = np.full((480, 640, 3), 250, dtype=np.uint8)
        metrics, status, reason = self.processor.assess_quality(bright_frame)
        self.assertEqual(status, "BAD")
        self.assertTrue("overexposed" in reason.lower() or "bright" in reason.lower())

    def test_blurry_frame_bad_quality(self):
        """A flat or completely blurred frame should fail the blur threshold."""
        blurred_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        metrics, status, reason = self.processor.assess_quality(blurred_frame)
        self.assertEqual(status, "BAD")
        self.assertIn("blur", reason.lower())

    def test_invalid_none_frame(self):
        """Passing None or an empty frame should gracefully return BAD."""
        metrics, status, reason = self.processor.assess_quality(None)
        self.assertEqual(status, "BAD")
        self.assertIn("invalid", reason.lower())

        empty_frame = np.array([], dtype=np.uint8)
        metrics2, status2, reason2 = self.processor.assess_quality(empty_frame)
        self.assertEqual(status2, "BAD")

    def test_roi_cropping_center(self):
        """Centered 80% ROI should correctly slice the frame and return valid bbox."""
        frame = np.zeros((400, 500, 3), dtype=np.uint8)
        roi_img, bbox = self.processor.extract_roi(frame, mode="center")
        x, y, w, h = bbox
        self.assertEqual(w, 400)  # 500 * 0.8
        self.assertEqual(h, 320)  # 400 * 0.8
        self.assertEqual(x, 50)
        self.assertEqual(y, 40)
        self.assertEqual(roi_img.shape, (320, 400, 3))

    def test_roi_cropping_full(self):
        """Full frame ROI should return entire dimensions."""
        frame = np.zeros((300, 400, 3), dtype=np.uint8)
        roi_img, bbox = self.processor.extract_roi(frame, mode="full")
        self.assertEqual(bbox, (0, 0, 400, 300))
        self.assertEqual(roi_img.shape, (300, 400, 3))


class TestWebcamManagerAndStream(unittest.TestCase):
    """Unit tests for WebcamManager and SimulatedStream."""

    def test_simulated_stream_from_dataset(self):
        """SimulatedStream should yield sequential valid frames from a directory."""
        stream_dir = os.path.join(config.BASE_DIR, "Tiles_dataset", "test", "cracks")
        if not os.path.isdir(stream_dir):
            self.skipTest(f"Directory {stream_dir} does not exist for simulated stream test.")

        sim = SimulatedStream(stream_dir, target_fps=100)
        self.assertGreater(sim.total_frames, 0)

        ret, frame = sim.read()
        self.assertTrue(ret)
        self.assertIsNotNone(frame)
        self.assertEqual(len(frame.shape), 3)

        ret2, frame2 = sim.read()
        self.assertTrue(ret2)
        self.assertIsNotNone(frame2)
        sim.release()

    def test_webcam_manager_invalid_camera_fails_gracefully(self):
        """Requesting an invalid camera index (e.g. 99) should fail gracefully without crashing."""
        manager = WebcamManager(camera_index=99)
        ok = manager.start()
        self.assertFalse(ok)
        self.assertIn("Unable to access camera", manager.error_message)
        self.assertFalse(manager.is_running)
        manager.stop()

    def test_webcam_manager_simulated_mode(self):
        """WebcamManager should run cleanly in simulated mode."""
        sim_dir = os.path.join(config.BASE_DIR, "Tiles_dataset", "test", "cracks")
        if not os.path.isdir(sim_dir):
            self.skipTest("Tiles_dataset not available.")

        manager = WebcamManager(camera_index="simulated", simulated_dir=sim_dir, target_fps=30)
        ok = manager.start()
        self.assertTrue(ok)
        self.assertTrue(manager.is_running)

        # Grab a frame
        ret, frame, ts = manager.get_latest_frame(timeout=1.0)
        self.assertTrue(ret)
        self.assertIsNotNone(frame)
        self.assertEqual(len(frame.shape), 3)

        # Stop
        manager.stop()
        self.assertFalse(manager.is_running)


class TestFeedbackStore(unittest.TestCase):
    """Unit tests for feedback dataset storage, captures, and history."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.captures_dir = os.path.join(self.temp_dir, "captures")
        self.feedback_dir = os.path.join(self.temp_dir, "feedback")
        self.store = FeedbackStore(captures_dir=self.captures_dir, feedback_dir=self.feedback_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_capture(self):
        """Verifies saving raw frame, annotated frame, heatmap, and result.json."""
        raw = np.full((100, 100, 3), 120, dtype=np.uint8)
        annotated = np.full((100, 100, 3), 150, dtype=np.uint8)
        heatmap = np.full((100, 100, 3), 200, dtype=np.uint8)
        result = {
            "status": "ANOMALY",
            "score": 0.825,
            "threshold": 0.50,
            "quality": "GOOD",
            "latency_ms": 42.1,
            "fps": 12.0,
            "roi": [10, 10, 80, 80],
        }

        cap_info = self.store.save_capture(raw, annotated, heatmap, result)
        self.assertTrue(cap_info["success"])
        target_folder = cap_info["directory"]

        self.assertTrue(os.path.isdir(target_folder))
        self.assertTrue(os.path.isfile(os.path.join(target_folder, "raw.jpg")))
        self.assertTrue(os.path.isfile(os.path.join(target_folder, "annotated.jpg")))
        self.assertTrue(os.path.isfile(os.path.join(target_folder, "heatmap.jpg")))
        self.assertTrue(os.path.isfile(os.path.join(target_folder, "result.json")))

    def test_record_expert_review_and_feedback_file(self):
        """Expert review should persist images into partitioned feedback directories."""
        frame = np.full((100, 100, 3), 80, dtype=np.uint8)
        res = self.store.record_expert_review(
            frame_bgr=frame,
            model_prediction="ANOMALY",
            model_score=0.88,
            expert_label="CONFIRM_ANOMALY",
            notes="Real crack near upper corner",
            quality_score="GOOD",
        )
        self.assertTrue(res["success"])
        self.assertEqual(res["category"], "anomaly")
        self.assertTrue(os.path.isfile(res["saved_path"]))

        # Review as False Positive -> normal category
        res2 = self.store.record_expert_review(
            frame_bgr=frame,
            model_prediction="ANOMALY",
            model_score=0.62,
            expert_label="FALSE_POSITIVE",
            notes="Glare caused false positive",
            quality_score="GOOD",
        )
        self.assertTrue(res2["success"])
        self.assertEqual(res2["category"], "normal")
        self.assertTrue(os.path.isfile(res2["saved_path"]))

    def test_history_and_csv_export(self):
        """Inspection history must track items, allow clearing, and export valid CSV."""
        self.store.add_history_entry({
            "status": "NORMAL",
            "score": 0.22,
            "threshold": 0.50,
            "quality": "GOOD",
            "latency_ms": 35.0,
            "expert_label": "None",
        })
        self.store.add_history_entry({
            "status": "ANOMALY",
            "score": 0.91,
            "threshold": 0.50,
            "quality": "GOOD",
            "latency_ms": 40.0,
            "expert_label": "CONFIRM_ANOMALY",
        })

        hist = self.store.get_history(limit=10)
        self.assertEqual(len(hist), 2)
        self.assertEqual(hist[0]["status"], "ANOMALY")  # latest first

        csv_text = self.store.export_csv()
        self.assertIn("status", csv_text)
        self.assertIn("ANOMALY", csv_text)
        self.assertIn("0.91", csv_text)

        self.store.clear_history()
        self.assertEqual(len(self.store.get_history()), 0)


if __name__ == "__main__":
    unittest.main()
