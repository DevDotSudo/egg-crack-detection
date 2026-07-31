import tempfile
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

SRC = Path(__file__).resolve().parents[1] / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app.api.main import app
from app.core.config import CONFIG
from app.repositories.history_store import HistoryStore
from app.services.detector import _decode_input_image, _fuzzy_egg_size


class BackendContractTests(unittest.TestCase):
    def test_backend_exposes_no_camera_routes(self) -> None:
        paths = app.openapi()['paths']
        self.assertFalse(any(path.startswith('/camera') for path in paths))

    def test_camera_detection_endpoint_is_renamed(self) -> None:
        paths = app.openapi()['paths']
        self.assertIn('/detect/camera', paths)
        self.assertIn('/detect/camera/multi', paths)
        self.assertIn('/focus/score', paths)
        self.assertNotIn('/detect/consensus', paths)

    def test_camera_image_orientation_is_preserved(self) -> None:
        source = np.zeros((240, 120, 3), dtype=np.uint8)
        source[:80, :40] = (10, 80, 230)
        ok, encoded = cv2.imencode('.png', source)
        self.assertTrue(ok)
        decoded = _decode_input_image(encoded.tobytes())
        self.assertEqual(decoded.shape[:2], (240, 120))
        self.assertGreater(int(decoded[20, 20, 2]), 200)
        self.assertLess(int(decoded[20, 100, 2]), 20)

    def test_fuzzy_egg_size_returns_normalized_memberships(self) -> None:
        label, confidence, memberships, score = _fuzzy_egg_size(
            0.27,
            CONFIG,
            egg_width_ratio=0.43,
            egg_length_ratio=0.57,
        )
        self.assertEqual(label, 'medium')
        self.assertGreater(confidence, 0.8)
        self.assertAlmostEqual(sum(memberships.values()), 1.0, places=3)
        self.assertGreater(score, 0.4)
        self.assertLess(score, 0.7)

    def test_history_save_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = HistoryStore(Path(directory) / 'history.json')
            created = store.add({'id': 'one', 'is_crack': False})
            replaced = store.add({'id': 'one', 'is_crack': True})
            self.assertTrue(created)
            self.assertFalse(replaced)
            self.assertEqual(len(store.list()), 1)
            self.assertTrue(store.list()[0]['is_crack'])


if __name__ == '__main__':
    unittest.main()
