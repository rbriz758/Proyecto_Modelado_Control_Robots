import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from sensor_camara import TurtleBot3Camera


class TestRedTargetDetection(unittest.TestCase):
    def setUp(self):
        self.camera = TurtleBot3Camera()

    def test_detects_red_target_and_extracts_features(self):
        frame = np.zeros((20, 40, 3), dtype=np.uint8)
        frame[6:14, 24:34] = [220, 20, 20]

        features = self.camera.extract_red_target_features(frame)

        self.assertTrue(features.detected)
        self.assertEqual(features.pixel_count, 80)
        self.assertAlmostEqual(features.center_x, 0.425)
        self.assertAlmostEqual(features.center_y, -0.05)
        self.assertAlmostEqual(features.area_ratio, 0.1)

    def test_ignores_non_red_frame(self):
        frame = np.zeros((20, 40, 3), dtype=np.uint8)
        frame[6:14, 24:34] = [20, 220, 20]

        features = self.camera.extract_red_target_features(frame)

        self.assertFalse(features.detected)
        self.assertEqual(features.pixel_count, 0)

    def test_rejects_small_red_region(self):
        frame = np.zeros((20, 40, 3), dtype=np.uint8)
        frame[9:11, 19:26] = [220, 20, 20]

        features = self.camera.extract_red_target_features(frame)

        self.assertFalse(features.detected)
        self.assertEqual(features.pixel_count, 14)


if __name__ == "__main__":
    unittest.main()