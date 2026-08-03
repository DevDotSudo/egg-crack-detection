import cv2
import numpy as np

from app.detection.config import DetectionConfig
from app.detection.models import EggRegion, EnhancementResult, QualityReport


def normalize_u8(
    image: np.ndarray,
    mask: np.ndarray,
    mask_output: bool = True,
) -> np.ndarray:
    values = image[mask > 0]
    if values.size == 0:
        return np.zeros_like(image, dtype=np.uint8)
    low, high = np.percentile(values, (1.0, 99.0))
    if high <= low + 1e-6:
        return np.zeros_like(image, dtype=np.uint8)
    output = (image.astype(np.float32) - float(low)) * 255.0 / float(high - low)
    output = np.clip(output, 0, 255).astype(np.uint8)
    if not mask_output:
        return output
    return cv2.bitwise_and(output, output, mask=mask)


def extend_nearest_inside(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if cv2.countNonZero(mask) == 0:
        return image.copy()
    source = np.where(mask > 0, 0, 255).astype(np.uint8)
    _, labels = cv2.distanceTransformWithLabels(
        source,
        cv2.DIST_L2,
        5,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    inside_labels = labels[mask > 0]
    inside_values = image[mask > 0]
    maximum_label = int(labels.max())
    lookup = np.zeros(maximum_label + 1, dtype=image.dtype)
    lookup[inside_labels] = inside_values
    output = lookup[labels]
    output[mask > 0] = image[mask > 0]
    return output


class QualityAssessor:
    def __init__(self, config: DetectionConfig) -> None:
        self.config = config

    def assess(self, image: np.ndarray, egg: EggRegion) -> QualityReport:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        values = gray[egg.inner_mask > 0]
        dynamic_range = float(np.percentile(values, 95) - np.percentile(values, 5))
        laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        sharpness = float(np.var(laplacian[egg.inner_mask > 0]))
        detail = cv2.GaussianBlur(gray, (0, 0), 1.2)
        detail_variance = float(np.var(cv2.subtract(gray, detail)[egg.inner_mask > 0]))
        saturation = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2]
        glare_mask = np.where((saturation >= 248) & (egg.inner_mask > 0), 255, 0).astype(np.uint8)
        saturated_ratio = cv2.countNonZero(glare_mask) / max(float(cv2.countNonZero(egg.inner_mask)), 1.0)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(glare_mask, connectivity=8)
        largest_area = 0
        largest_width = 0.0
        for index in range(1, count):
            area = int(stats[index, cv2.CC_STAT_AREA])
            if area > largest_area:
                largest_area = area
                component = np.where(labels == index, 255, 0).astype(np.uint8)
                distance = cv2.distanceTransform(component, cv2.DIST_L2, 5)
                largest_width = float(distance.max() * 2.0)
        glare_ratio = largest_area / max(float(cv2.countNonZero(egg.inner_mask)), 1.0)
        q = self.config.quality
        acceptable = True
        message = 'Image quality is suitable for detection'
        if dynamic_range < q.minimum_dynamic_range:
            acceptable = False
            message = 'The egg lighting has too little contrast. Increase the candling light'
        elif sharpness < q.minimum_sharpness:
            acceptable = False
            message = 'The egg image is blurry. Hold the camera steady and refocus'
        elif saturated_ratio > q.maximum_saturated_ratio or glare_ratio > q.maximum_glare_ratio or largest_width > egg.minor_axis * q.maximum_glare_width_ratio:
            acceptable = False
            message = 'Strong flashlight glare covers the egg. Diffuse or move the light source'
        sharpness_score = min(1.0, sharpness / max(q.minimum_sharpness * 5.0, 1.0))
        range_score = min(1.0, dynamic_range / 65.0)
        glare_score = max(0.0, 1.0 - glare_ratio / max(q.maximum_glare_ratio, 1e-6))
        score = float(np.clip(0.45 * sharpness_score + 0.35 * range_score + 0.20 * glare_score, 0.0, 1.0))
        return QualityReport(
            score=score,
            sharpness=sharpness,
            detail_variance=detail_variance,
            saturated_ratio=float(saturated_ratio),
            glare_ratio=float(glare_ratio),
            glare_width=float(largest_width),
            dynamic_range=dynamic_range,
            glare_mask=glare_mask,
            acceptable=acceptable,
            message=message,
        )


class DualPolarityEnhancer:
    def __init__(self, config: DetectionConfig) -> None:
        self.config = config
        e = config.enhancement
        self.clahe = cv2.createCLAHE(
            clipLimit=e.clahe_clip_limit,
            tileGridSize=(e.clahe_grid_size, e.clahe_grid_size),
        )

    def enhance(
        self,
        image: np.ndarray,
        egg: EggRegion,
        quality: QualityReport,
        detection_mask: np.ndarray | None = None,
    ) -> EnhancementResult:
        analysis_mask = egg.inner_mask if detection_mask is None else detection_mask
        green = image[:, :, 1]
        lab_light = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0]
        raw = cv2.addWeighted(green, 0.65, lab_light, 0.35, 0)
        extended_raw = extend_nearest_inside(raw, egg.full_mask)
        sigma = max(9.0, egg.minor_axis * self.config.enhancement.flatfield_sigma_ratio)
        background = cv2.GaussianBlur(extended_raw, (0, 0), sigma)
        center = float(np.median(background[egg.inner_mask > 0]))
        corrected = extended_raw.astype(np.float32)
        corrected -= self.config.enhancement.flatfield_strength * background.astype(np.float32)
        corrected += self.config.enhancement.flatfield_strength * center
        corrected = normalize_u8(np.clip(corrected, 0, 255), analysis_mask, mask_output=False)
        normalized_full = self.clahe.apply(corrected)
        normalized = cv2.bitwise_and(normalized_full, normalized_full, mask=analysis_mask)
        dark_maps: list[np.ndarray] = []
        bright_maps: list[np.ndarray] = []
        for small_sigma, large_sigma in zip(self.config.enhancement.small_sigmas, self.config.enhancement.large_sigmas):
            small = cv2.GaussianBlur(normalized_full, (0, 0), small_sigma)
            large = cv2.GaussianBlur(normalized_full, (0, 0), large_sigma)
            dark_maps.append(cv2.subtract(large, small))
            bright_maps.append(cv2.subtract(small, large))
        for ratio in self.config.enhancement.morphology_ratios:
            size = max(3, int(round(egg.minor_axis * ratio)))
            if size % 2 == 0:
                size += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            dark_maps.append(cv2.morphologyEx(normalized_full, cv2.MORPH_BLACKHAT, kernel))
            bright_maps.append(cv2.morphologyEx(normalized_full, cv2.MORPH_TOPHAT, kernel))
        dark_response = dark_maps[0]
        bright_response = bright_maps[0]
        for response in dark_maps[1:]:
            dark_response = cv2.max(dark_response, response)
        for response in bright_maps[1:]:
            bright_response = cv2.max(bright_response, response)
        gx = cv2.Scharr(normalized_full, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(normalized_full, cv2.CV_32F, 0, 1)
        edge = cv2.magnitude(gx, gy)
        edge_response = normalize_u8(edge, analysis_mask)
        dark_response = cv2.bitwise_and(dark_response, dark_response, mask=analysis_mask)
        bright_response = cv2.bitwise_and(bright_response, bright_response, mask=analysis_mask)
        return EnhancementResult(
            normalized=normalized,
            raw_channel=raw,
            dark_response=dark_response,
            bright_response=bright_response,
            edge_response=edge_response,
            glare_mask=quality.glare_mask,
        )
