import cv2
import numpy as np

from app.detection.config import DetectionConfig
from app.detection.models import ThresholdResult


class ResponseThreshold:
    def __init__(self, config: DetectionConfig) -> None:
        self.config = config

    def apply(
        self,
        response: np.ndarray,
        mask: np.ndarray,
        minimum_weak: float,
        minimum_strong: float,
    ) -> ThresholdResult:
        values = response[mask > 0]
        if values.size == 0:
            empty = np.zeros_like(response, dtype=np.uint8)
            return ThresholdResult(empty, empty, empty, minimum_weak, minimum_strong)
        nonzero = values[values > 0]
        source = nonzero if nonzero.size >= 64 else values
        weak_threshold = max(minimum_weak, float(np.percentile(source, self.config.threshold.weak_percentile)))
        strong_threshold = max(minimum_strong, float(np.percentile(source, self.config.threshold.strong_percentile)))
        if strong_threshold < weak_threshold + 1.0:
            strong_threshold = weak_threshold + 1.0
        weak = np.where((response >= weak_threshold) & (mask > 0), 255, 0).astype(np.uint8)
        strong = np.where((response >= strong_threshold) & (mask > 0), 255, 0).astype(np.uint8)
        grown = strong.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        for _ in range(self.config.threshold.maximum_growth_iterations):
            previous = grown
            grown = cv2.bitwise_and(cv2.dilate(grown, kernel), weak)
            if np.array_equal(previous, grown):
                break
        return ThresholdResult(
            weak_mask=weak,
            strong_mask=strong,
            grown_mask=grown,
            weak_threshold=weak_threshold,
            strong_threshold=strong_threshold,
        )


def connect_small_gaps(mask: np.ndarray) -> np.ndarray:
    output = mask.copy()
    kernels = [
        np.array([[1, 1, 1]], dtype=np.uint8),
        np.array([[1], [1], [1]], dtype=np.uint8),
        np.eye(3, dtype=np.uint8),
        np.fliplr(np.eye(3, dtype=np.uint8)),
    ]
    for kernel in kernels:
        output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel, iterations=1)
    return output
