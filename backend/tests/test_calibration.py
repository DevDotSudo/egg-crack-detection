import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

SRC = Path(__file__).resolve().parents[1] / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app.core.config import CONFIG
from app.detection.calibration import CalibrationStore, CameraCalibrator, EggMeasurementService
from app.detection.models import EggRegion


class CalibrationTests(unittest.TestCase):
    def test_manual_profile_is_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = CalibrationStore(Path(directory) / 'calibration.json')
            profile = CameraCalibrator(CONFIG).manual_profile(30.0, 300.0, 1280, 720)
            store.save(profile)
            loaded = store.load()
            self.assertIsNotNone(loaded)
            self.assertAlmostEqual(loaded.pixels_per_mm, 10.0)
            self.assertEqual(loaded.camera_distance_inches, 4.0)

    def test_reference_circle_calibration(self) -> None:
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.circle(image, (640, 360), 150, (255, 255, 255), -1)
        profile, overlay = CameraCalibrator(CONFIG).image_profile(image, 30.0, 'circle')
        self.assertEqual(overlay.shape, image.shape)
        self.assertAlmostEqual(profile.reference_width_pixels, 300.0, delta=4.0)
        self.assertAlmostEqual(profile.pixels_per_mm, 10.0, delta=0.2)

    def test_egg_measurement_uses_calibrated_height_and_width(self) -> None:
        mask = np.zeros((720, 1280), dtype=np.uint8)
        contour = np.array([[[425, 75]], [[855, 75]], [[855, 645]], [[425, 645]]], dtype=np.int32)
        egg = EggRegion(
            full_mask=mask,
            inner_mask=mask,
            rim_mask=mask,
            contour=contour,
            bbox=(425, 75, 430, 570),
            center=(640.0, 360.0),
            width=430.0,
            length=570.0,
            minor_axis=430.0,
            major_axis=570.0,
            area_ratio=0.25,
            score=4.0,
        )
        profile = CameraCalibrator(CONFIG).manual_profile(30.0, 300.0, 1280, 720)
        measurement = EggMeasurementService(CONFIG.calibration).measure(egg, (720, 1280), profile)
        self.assertTrue(measurement.valid)
        self.assertAlmostEqual(measurement.width_mm, 43.0)
        self.assertAlmostEqual(measurement.height_mm, 57.0)

    def test_rotated_resolution_keeps_existing_pixel_scale(self) -> None:
        mask = np.zeros((1280, 720), dtype=np.uint8)
        contour = np.array([[[145, 355]], [[575, 355]], [[575, 925]], [[145, 925]]], dtype=np.int32)
        egg = EggRegion(
            mask,
            mask,
            mask,
            contour,
            (145, 355, 430, 570),
            (360.0, 640.0),
            430.0,
            570.0,
            430.0,
            570.0,
            0.25,
            4.0,
        )
        profile = CameraCalibrator(CONFIG).manual_profile(30.0, 300.0, 1280, 720)
        measurement = EggMeasurementService(CONFIG.calibration).measure(egg, (1280, 720), profile)
        self.assertTrue(measurement.valid)
        self.assertAlmostEqual(measurement.width_mm, 43.0)
        self.assertAlmostEqual(measurement.height_mm, 57.0)

    def test_resolution_mismatch_invalidates_size(self) -> None:
        mask = np.zeros((720, 1280), dtype=np.uint8)
        contour = np.array([[[425, 75]], [[855, 75]], [[855, 645]], [[425, 645]]], dtype=np.int32)
        egg = EggRegion(mask, mask, mask, contour, (425, 75, 430, 570), (640.0, 360.0), 430.0, 570.0, 430.0, 570.0, 0.25, 4.0)
        profile = CameraCalibrator(CONFIG).manual_profile(30.0, 300.0, 1280, 720)
        measurement = EggMeasurementService(CONFIG.calibration).measure(egg, (700, 1200), profile)
        self.assertFalse(measurement.valid)
        self.assertIn('resolution', measurement.message.lower())


if __name__ == '__main__':
    unittest.main()
