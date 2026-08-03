import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

SRC = Path(__file__).resolve().parents[1] / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app.detection.models import EggRegion
from app.detection.rendering import OverlayRenderer


class RenderingTests(unittest.TestCase):
    def test_egg_border_and_crack_polygon_are_one_pixel(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        contour = np.array([[[10, 10]], [[90, 10]], [[90, 90]], [[10, 90]]], dtype=np.int32)
        full_mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.fillPoly(full_mask, [contour], 255)
        egg = EggRegion(
            full_mask=full_mask,
            inner_mask=full_mask.copy(),
            rim_mask=np.zeros_like(full_mask),
            contour=contour,
            bbox=(10, 10, 81, 81),
            center=(50.0, 50.0),
            width=80.0,
            length=80.0,
            minor_axis=80.0,
            major_axis=80.0,
            area_ratio=0.64,
            score=1.0,
        )
        crack_mask = np.zeros((100, 100), dtype=np.uint8)
        crack_mask[30:41, 30:61] = 255
        overlay = OverlayRenderer().render(image, egg, crack_mask, [])
        self.assertTupleEqual(tuple(int(value) for value in overlay[10, 50]), (0, 220, 0))
        self.assertTupleEqual(tuple(int(value) for value in overlay[11, 50]), (0, 0, 0))
        self.assertTupleEqual(tuple(int(value) for value in overlay[30, 45]), (0, 0, 255))
        self.assertTupleEqual(tuple(int(value) for value in overlay[31, 45]), (0, 0, 0))
        self.assertTupleEqual(tuple(int(value) for value in overlay[35, 45]), (0, 0, 0))


if __name__ == '__main__':
    unittest.main()
