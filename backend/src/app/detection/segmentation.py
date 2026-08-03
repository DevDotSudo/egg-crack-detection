import cv2
import numpy as np

from app.detection.config import DetectionConfig
from app.detection.models import EggRegion


class EggSegmentationError(ValueError):
    pass


def fill_holes(mask: np.ndarray) -> np.ndarray:
    flood = mask.copy()
    padded = cv2.copyMakeBorder(flood, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    cv2.floodFill(padded, None, (0, 0), 255)
    holes = cv2.bitwise_not(padded[1:-1, 1:-1])
    return cv2.bitwise_or(mask, holes)


class EggSegmenter:
    def __init__(self, config: DetectionConfig) -> None:
        self.config = config

    def _candidate_masks(self, image: np.ndarray) -> list[np.ndarray]:
        red = image[:, :, 2]
        green = image[:, :, 1]
        value = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        channels = [red, green, value, gray]
        masks: list[np.ndarray] = []
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        for channel in channels:
            blurred = cv2.GaussianBlur(channel, (11, 11), 0)
            _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            adaptive = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                51,
                -2,
            )
            for mask in (otsu, adaptive):
                cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
                cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
                masks.append(fill_holes(cleaned))
        return masks

    def _score_contour(self, contour: np.ndarray, shape: tuple[int, int]) -> float:
        height, width = shape
        frame_area = float(height * width)
        area = float(cv2.contourArea(contour))
        if area <= 0:
            return -1.0
        area_ratio = area / frame_area
        if not self.config.segmentation.minimum_area_ratio <= area_ratio <= self.config.segmentation.maximum_area_ratio:
            return -1.0
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if box_width < self.config.min_egg_width or box_height < self.config.min_egg_height:
            return -1.0
        aspect = max(box_width, box_height) / max(1.0, min(box_width, box_height))
        if aspect > self.config.segmentation.maximum_aspect_ratio:
            return -1.0
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / max(hull_area, 1.0)
        if solidity < self.config.segmentation.minimum_solidity:
            return -1.0
        moments = cv2.moments(contour)
        if moments['m00'] <= 0:
            return -1.0
        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']
        center_distance = np.hypot(cx - width / 2.0, cy - height / 2.0)
        center_score = 1.0 - min(1.0, center_distance / max(np.hypot(width, height) * 0.5, 1.0))
        border_touch = int(x <= 1) + int(y <= 1) + int(x + box_width >= width - 1) + int(y + box_height >= height - 1)
        ellipse_score = 0.0
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            axes = ellipse[1]
            ellipse_area = np.pi * axes[0] * axes[1] / 4.0
            ellipse_score = 1.0 - min(1.0, abs(area - ellipse_area) / max(ellipse_area, 1.0))
        return 2.2 * solidity + 1.2 * center_score + 1.0 * ellipse_score + 0.8 * min(area_ratio / 0.25, 1.0) - 1.6 * border_touch

    def segment(self, image: np.ndarray) -> EggRegion:
        best_contour: np.ndarray | None = None
        best_score = -1.0
        for mask in self._candidate_masks(image):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                score = self._score_contour(contour, image.shape[:2])
                if score > best_score:
                    best_score = score
                    best_contour = contour
        if best_contour is None:
            raise EggSegmentationError('No egg was detected. Use a dark background and keep one full egg in view')
        full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(full_mask, [best_contour], -1, 255, -1)
        full_mask = fill_holes(full_mask)
        area = int(cv2.countNonZero(full_mask))
        if area < self.config.min_egg_pixels:
            raise EggSegmentationError('The egg is too small in the image')
        x, y, box_width, box_height = cv2.boundingRect(best_contour)
        if len(best_contour) >= 5:
            center, axes, _ = cv2.fitEllipse(best_contour)
            width_value = float(min(axes))
            length_value = float(max(axes))
        else:
            center = (x + box_width / 2.0, y + box_height / 2.0)
            width_value = float(min(box_width, box_height))
            length_value = float(max(box_width, box_height))
        minor_axis = max(width_value, 1.0)
        major_axis = max(length_value, minor_axis)
        inner_pixels = max(1, int(round(minor_axis * self.config.segmentation.inner_margin_ratio)))
        rim_pixels = max(inner_pixels + 1, int(round(minor_axis * self.config.segmentation.rim_width_ratio)))
        inner_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inner_pixels * 2 + 1, inner_pixels * 2 + 1))
        rim_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rim_pixels * 2 + 1, rim_pixels * 2 + 1))
        inner_mask = cv2.erode(full_mask, inner_kernel)
        deep_inner = cv2.erode(full_mask, rim_kernel)
        rim_mask = cv2.subtract(full_mask, deep_inner)
        if cv2.countNonZero(inner_mask) < self.config.min_inner_pixels:
            raise EggSegmentationError('The visible egg area is too small for crack detection')
        return EggRegion(
            full_mask=full_mask,
            inner_mask=inner_mask,
            rim_mask=rim_mask,
            contour=best_contour,
            bbox=(x, y, box_width, box_height),
            center=(float(center[0]), float(center[1])),
            width=width_value,
            length=length_value,
            minor_axis=minor_axis,
            major_axis=major_axis,
            area_ratio=area / float(image.shape[0] * image.shape[1]),
            score=float(best_score),
        )
