from __future__ import annotations

import json
import math
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from app.detection.config import CalibrationConfig, DetectionConfig
from app.detection.models import EggRegion


class CalibrationError(ValueError):
    pass


@dataclass(frozen=True)
class CalibrationProfile:
    camera_distance_inches: float
    reference_width_mm: float
    reference_width_pixels: float
    pixels_per_mm: float
    processed_width: int
    processed_height: int
    created_at: str
    method: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> 'CalibrationProfile':
        return cls(
            camera_distance_inches=float(value['camera_distance_inches']),
            reference_width_mm=float(value['reference_width_mm']),
            reference_width_pixels=float(value['reference_width_pixels']),
            pixels_per_mm=float(value['pixels_per_mm']),
            processed_width=int(value['processed_width']),
            processed_height=int(value['processed_height']),
            created_at=str(value['created_at']),
            method=str(value.get('method', 'manual')),
        )


@dataclass(frozen=True)
class EggMeasurement:
    valid: bool
    message: str
    width_pixels: float
    height_pixels: float
    width_mm: float | None
    height_mm: float | None
    pixels_per_mm: float | None
    camera_distance_inches: float


class CalibrationStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def load(self) -> CalibrationProfile | None:
        if not self.path.exists():
            return None
        try:
            raw = json.loads(self.path.read_text(encoding='utf-8'))
            return CalibrationProfile.from_dict(raw)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None

    def save(self, profile: CalibrationProfile) -> CalibrationProfile:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix('.tmp')
            temporary.write_text(json.dumps(profile.to_dict(), indent=2), encoding='utf-8')
            temporary.replace(self.path)
        return profile

    def clear(self) -> bool:
        with self._lock:
            existed = self.path.exists()
            self.path.unlink(missing_ok=True)
        return existed


class CameraCalibrator:
    def __init__(self, detection_config: DetectionConfig) -> None:
        self.detection_config = detection_config
        self.config = detection_config.calibration

    def working_image(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        scale = min(
            1.0,
            self.detection_config.target_width / max(float(width), 1.0),
            self.detection_config.target_height / max(float(height), 1.0),
        )
        if scale >= 1.0:
            return image.copy()
        target = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        return cv2.resize(image, target, interpolation=cv2.INTER_AREA)

    def manual_profile(
        self,
        reference_width_mm: float,
        reference_width_pixels: float,
        processed_width: int,
        processed_height: int,
    ) -> CalibrationProfile:
        self._validate_inputs(reference_width_mm, reference_width_pixels, processed_width, processed_height)
        return self._profile(
            reference_width_mm,
            reference_width_pixels,
            processed_width,
            processed_height,
            'manual',
        )

    def image_profile(
        self,
        image: np.ndarray,
        reference_width_mm: float,
        reference_shape: str = 'circle',
    ) -> tuple[CalibrationProfile, np.ndarray]:
        working = self.working_image(image)
        contour = self._detect_reference_contour(working)
        area = float(cv2.contourArea(contour))
        shape = reference_shape.strip().lower()
        if shape == 'circle':
            reference_width_pixels = 2.0 * math.sqrt(max(area, 1.0) / math.pi)
        elif shape == 'square':
            reference_width_pixels = math.sqrt(max(area, 1.0))
        else:
            rect = cv2.minAreaRect(contour)
            side_a, side_b = rect[1]
            reference_width_pixels = max(float(side_a), float(side_b))
        height, width = working.shape[:2]
        profile = self.manual_profile(reference_width_mm, reference_width_pixels, width, height)
        profile = CalibrationProfile(**{**profile.to_dict(), 'method': f'image-{shape}'})
        overlay = working.copy()
        cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 1, cv2.LINE_8)
        x, y, box_width, box_height = cv2.boundingRect(contour)
        cv2.putText(
            overlay,
            f'{reference_width_pixels:.1f}px = {reference_width_mm:.1f}mm',
            (x, max(24, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        return profile, overlay

    def _profile(
        self,
        reference_width_mm: float,
        reference_width_pixels: float,
        processed_width: int,
        processed_height: int,
        method: str,
    ) -> CalibrationProfile:
        return CalibrationProfile(
            camera_distance_inches=self.config.camera_distance_inches,
            reference_width_mm=float(reference_width_mm),
            reference_width_pixels=float(reference_width_pixels),
            pixels_per_mm=float(reference_width_pixels / reference_width_mm),
            processed_width=int(processed_width),
            processed_height=int(processed_height),
            created_at=datetime.now(timezone.utc).isoformat(),
            method=method,
        )

    def _validate_inputs(
        self,
        reference_width_mm: float,
        reference_width_pixels: float,
        processed_width: int,
        processed_height: int,
    ) -> None:
        if reference_width_mm <= 0:
            raise CalibrationError('Reference width must be greater than zero millimeters')
        if not self.config.minimum_reference_pixels <= reference_width_pixels <= self.config.maximum_reference_pixels:
            raise CalibrationError('Reference width in pixels is outside the supported calibration range')
        if processed_width <= 0 or processed_height <= 0:
            raise CalibrationError('Calibration image dimensions must be greater than zero')

    def _detect_reference_contour(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        candidates = [binary, cv2.bitwise_not(binary)]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        frame_height, frame_width = gray.shape
        frame_area = float(frame_height * frame_width)
        best: np.ndarray | None = None
        best_score = -1.0
        for candidate in candidates:
            cleaned = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = float(cv2.contourArea(contour))
                ratio = area / max(frame_area, 1.0)
                if not 0.001 <= ratio <= 0.35:
                    continue
                x, y, width, height = cv2.boundingRect(contour)
                if x <= 2 or y <= 2 or x + width >= frame_width - 2 or y + height >= frame_height - 2:
                    continue
                aspect = max(width, height) / max(float(min(width, height)), 1.0)
                if aspect > 1.5:
                    continue
                hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
                solidity = area / max(hull_area, 1.0)
                if solidity < 0.85:
                    continue
                moments = cv2.moments(contour)
                if moments['m00'] <= 0:
                    continue
                center_x = moments['m10'] / moments['m00']
                center_y = moments['m01'] / moments['m00']
                distance = math.hypot(center_x - frame_width / 2.0, center_y - frame_height / 2.0)
                center_score = 1.0 - min(1.0, distance / max(math.hypot(frame_width, frame_height) * 0.5, 1.0))
                score = 2.0 * solidity + center_score + min(1.0, ratio / 0.05)
                if score > best_score:
                    best = contour
                    best_score = score
        if best is None:
            raise CalibrationError('No calibration marker was found. Use one solid circle or square on a plain background')
        return best


class EggMeasurementService:
    def __init__(self, config: CalibrationConfig) -> None:
        self.config = config

    def measure(
        self,
        egg: EggRegion,
        frame_shape: tuple[int, int],
        profile: CalibrationProfile | None,
    ) -> EggMeasurement:
        rotated = cv2.minAreaRect(egg.contour)
        side_a, side_b = rotated[1]
        if side_a > 0 and side_b > 0:
            height_pixels = float(max(side_a, side_b))
            width_pixels = float(min(side_a, side_b))
        else:
            height_pixels = float(max(egg.width, egg.length))
            width_pixels = float(min(egg.width, egg.length))
        distance = self.config.camera_distance_inches
        if profile is None:
            return EggMeasurement(
                False,
                'Egg size is unavailable until the 4-inch camera setup is calibrated',
                width_pixels,
                height_pixels,
                None,
                None,
                None,
                distance,
            )
        if profile.pixels_per_mm <= 0:
            return EggMeasurement(
                False,
                'The saved calibration scale is invalid',
                width_pixels,
                height_pixels,
                None,
                None,
                None,
                distance,
            )
        if abs(profile.camera_distance_inches - self.config.camera_distance_inches) > 0.01:
            return EggMeasurement(
                False,
                'The saved calibration does not match the required 4-inch camera distance',
                width_pixels,
                height_pixels,
                None,
                None,
                profile.pixels_per_mm,
                distance,
            )
        frame_height, frame_width = frame_shape
        tolerance = self.config.resolution_tolerance_pixels
        direct_resolution_match = (
            abs(frame_width - profile.processed_width) <= tolerance
            and abs(frame_height - profile.processed_height) <= tolerance
        )
        rotated_resolution_match = (
            abs(frame_width - profile.processed_height) <= tolerance
            and abs(frame_height - profile.processed_width) <= tolerance
        )
        if not direct_resolution_match and not rotated_resolution_match:
            return EggMeasurement(
                False,
                'Camera resolution does not match the saved calibration',
                width_pixels,
                height_pixels,
                None,
                None,
                profile.pixels_per_mm,
                profile.camera_distance_inches,
            )
        x, y, box_width, box_height = egg.bbox
        margin_x = max(2, int(round(frame_width * self.config.border_margin_ratio)))
        margin_y = max(2, int(round(frame_height * self.config.border_margin_ratio)))
        if x <= margin_x or y <= margin_y or x + box_width >= frame_width - margin_x or y + box_height >= frame_height - margin_y:
            return EggMeasurement(
                False,
                'The full egg must be visible and separated from the image border',
                width_pixels,
                height_pixels,
                None,
                None,
                profile.pixels_per_mm,
                profile.camera_distance_inches,
            )
        offset_x = abs(egg.center[0] - frame_width / 2.0) / max(frame_width / 2.0, 1.0)
        offset_y = abs(egg.center[1] - frame_height / 2.0) / max(frame_height / 2.0, 1.0)
        if max(offset_x, offset_y) > self.config.center_tolerance_ratio:
            return EggMeasurement(
                False,
                'Place the egg at the calibrated center position',
                width_pixels,
                height_pixels,
                None,
                None,
                profile.pixels_per_mm,
                profile.camera_distance_inches,
            )
        width_mm = width_pixels / profile.pixels_per_mm
        height_mm = height_pixels / profile.pixels_per_mm
        if not self.config.minimum_egg_width_mm <= width_mm <= self.config.maximum_egg_width_mm:
            return EggMeasurement(
                False,
                'Measured egg width is outside the valid calibrated range. Check the 4-inch distance',
                width_pixels,
                height_pixels,
                width_mm,
                height_mm,
                profile.pixels_per_mm,
                profile.camera_distance_inches,
            )
        if not self.config.minimum_egg_height_mm <= height_mm <= self.config.maximum_egg_height_mm:
            return EggMeasurement(
                False,
                'Measured egg height is outside the valid calibrated range. Check the 4-inch distance',
                width_pixels,
                height_pixels,
                width_mm,
                height_mm,
                profile.pixels_per_mm,
                profile.camera_distance_inches,
            )
        return EggMeasurement(
            True,
            'Egg height and width were measured using the saved 4-inch calibration',
            width_pixels,
            height_pixels,
            width_mm,
            height_mm,
            profile.pixels_per_mm,
            profile.camera_distance_inches,
        )
