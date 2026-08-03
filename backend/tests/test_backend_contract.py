import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

SRC = Path(__file__).resolve().parents[1] / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app.api.main import app
from app.core.config import CONFIG
from app.repositories.db import DetectionDB
from app.services.detector import _correct_camera_orientation, _decode_input_image, _fuzzy_egg_size


class BackendContractTests(unittest.TestCase):
    def test_detection_routes_are_available(self) -> None:
        paths = app.openapi()['paths']
        self.assertIn('/detect', paths)
        self.assertIn('/detect/camera', paths)
        self.assertIn('/detect/camera/multi', paths)
        self.assertIn('/focus/score', paths)
        self.assertIn('/calibration', paths)
        self.assertIn('/calibration/manual', paths)
        self.assertIn('/calibration/reference', paths)
        self.assertFalse(any(path.startswith('/camera') for path in paths))

    def test_image_orientation_is_preserved(self) -> None:
        source = np.zeros((240, 120, 3), dtype=np.uint8)
        source[:80, :40] = (10, 80, 230)
        ok, encoded = cv2.imencode('.png', source)
        self.assertTrue(ok)
        decoded = _decode_input_image(encoded.tobytes())
        self.assertEqual(decoded.shape[:2], (240, 120))
        self.assertGreater(int(decoded[20, 20, 2]), 200)
        self.assertLess(int(decoded[20, 100, 2]), 20)

    def test_camera_capture_preserves_source_orientation_by_default(self) -> None:
        source = np.zeros((6, 4, 3), dtype=np.uint8)
        source[0, 0] = (10, 80, 230)
        corrected = _correct_camera_orientation(source, CONFIG)
        self.assertEqual(corrected.shape[:2], (6, 4))
        self.assertGreater(int(corrected[0, 0, 2]), 200)
        self.assertLess(int(corrected[0, 3, 2]), 20)
        self.assertIsNot(corrected, source)

    def test_egg_size_memberships_are_normalized(self) -> None:
        label, confidence, memberships, score = _fuzzy_egg_size(
            43.0,
            57.0,
            CONFIG,
        )
        self.assertEqual(label, 'medium')
        self.assertGreater(confidence, 0.70)
        self.assertAlmostEqual(sum(memberships.values()), 1.0, places=5)
        self.assertGreater(score, 0.4)
        self.assertLess(score, 0.7)

    def test_sqlite_save_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = DetectionDB(root / 'detections.db', root / 'images')
            created = database.add({'id': 'one', 'is_crack': False})
            replaced = database.add({'id': 'one', 'is_crack': True})
            self.assertTrue(created)
            self.assertFalse(replaced)
            self.assertEqual(len(database.list()), 1)
            self.assertTrue(database.list()[0]['is_crack'])


if __name__ == '__main__':
    unittest.main()
