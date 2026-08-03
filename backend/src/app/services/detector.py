from __future__ import annotations

import base64
import math
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np

from app.core.config import CONFIG, DetectionConfig
from app.core.paths import runtime_root
from app.detection.calibration import CalibrationStore, EggMeasurementService
from app.detection.fuzzy import CrackSizeMamdaniClassifier, EggSizeApparentMamdaniClassifier, EggSizeMamdaniClassifier
from app.detection.models import ComponentFeatures, CrackPolarity
from app.detection.pipeline import EggCrackPipeline
from app.detection.rendering import OverlayRenderer
from app.detection.segmentation import EggSegmentationError


class DetectionError(ValueError):
    pass

_CALIBRATION_PATH = runtime_root() / 'data' / 'calibration.json'
CALIBRATION_STORE = CalibrationStore(_CALIBRATION_PATH)


@dataclass
class CrackComponent:
    x: int
    y: int
    mask: np.ndarray
    support_mask: np.ndarray
    metrics: dict[str, float]
    channel: str


def _encode_image(image: np.ndarray, extension: str = '.jpg') -> str:
    parameters = [cv2.IMWRITE_JPEG_QUALITY, 94] if extension.lower() in {'.jpg', '.jpeg'} else []
    ok, encoded = cv2.imencode(extension, image, parameters)
    if not ok:
        raise DetectionError('Could not encode the processed image')
    return base64.b64encode(encoded.tobytes()).decode('ascii')


def _decode_input_image(data: bytes) -> np.ndarray:
    if not data:
        raise DetectionError('The image is empty')
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise DetectionError('The uploaded file is not a valid image')
    return image


def _correct_camera_orientation(image: np.ndarray, cfg: DetectionConfig) -> np.ndarray:
    mode = str(cfg.camera_orientation_fix).strip().lower()
    if mode in {'flip_horizontal', 'mirror_horizontal', 'horizontal'}:
        return cv2.flip(image, 1)
    if mode in {'flip_vertical', 'vertical'}:
        return cv2.flip(image, 0)
    if mode in {'rotate_180', '180'}:
        return cv2.rotate(image, cv2.ROTATE_180)
    if mode in {'rotate_90_clockwise', '90_clockwise', 'cw'}:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if mode in {'rotate_90_counterclockwise', '90_counterclockwise', 'ccw'}:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image.copy()


def _fuzzy_egg_size(
    egg_width_mm: float,
    egg_height_mm: float,
    cfg: DetectionConfig = CONFIG,
) -> tuple[str, float, dict[str, float], float]:
    result = EggSizeMamdaniClassifier(cfg.fuzzy).classify(
        float(egg_width_mm),
        float(egg_height_mm),
    )
    return result.label, result.confidence, result.memberships, result.score


def _fuzzy_apparent_egg_size(
    egg_width_pixels: float,
    egg_height_pixels: float,
    frame_shape: tuple[int, int],
    cfg: DetectionConfig = CONFIG,
) -> tuple[str, float, dict[str, float], float, float, float]:
    frame_height, frame_width = frame_shape
    short_side = max(float(min(frame_width, frame_height)), 1.0)
    width_ratio = float(egg_width_pixels) / short_side
    height_ratio = float(egg_height_pixels) / short_side
    result = EggSizeApparentMamdaniClassifier(cfg.fuzzy).classify(width_ratio, height_ratio)
    return (
        result.label,
        result.confidence,
        result.memberships,
        result.score,
        width_ratio,
        height_ratio,
    )


def _fuzzy_crack_size(
    is_crack: bool,
    traced_length: float,
    traced_pixels: int,
    egg_area: float,
    component_count: int,
    strongest_response: float,
    cfg: DetectionConfig = CONFIG,
) -> tuple[str, float, dict[str, float], float]:
    result = CrackSizeMamdaniClassifier(cfg.fuzzy).classify(
        is_crack=is_crack,
        traced_length=traced_length,
        traced_pixels=traced_pixels,
        egg_area=egg_area,
        component_count=component_count,
        strongest_response=strongest_response,
    )
    return result.label, result.confidence, result.memberships, result.score


def _component_location(index: int, component: ComponentFeatures) -> dict[str, Any]:
    x, y, width, height = component.bbox
    return {
        'id': index,
        'polarity': component.polarity.value,
        'bounding_box': [x, y, width, height],
        'pixel_area': component.area,
        'length_pixels': component.skeleton_length,
        'mean_thickness': round(component.mean_thickness, 3),
        'maximum_thickness': round(component.maximum_thickness, 3),
        'elongation': round(component.elongation, 3),
        'roughness': round(component.roughness, 4),
        'score': round(component.score, 4),
        'reasons': list(component.reasons),
    }


def _build_response(pipeline_result, include_steps: bool, cfg: DetectionConfig) -> dict[str, Any]:
    renderer = OverlayRenderer()
    accepted = [component for component in pipeline_result.components if component.accepted]
    is_crack = bool(accepted)
    overlay = renderer.render(
        pipeline_result.working,
        pipeline_result.egg,
        pipeline_result.crack_mask,
        pipeline_result.components,
    )
    egg_pixels = max(float(cv2.countNonZero(pipeline_result.egg.full_mask)), 1.0)
    candidate_pixels = int(cv2.countNonZero(pipeline_result.crack_mask))
    area_ratio = candidate_pixels / egg_pixels
    contour_length = float(sum(component.skeleton_length for component in accepted))
    longest_candidate = float(max((component.skeleton_length for component in accepted), default=0.0))
    strengths = [component.response_p90 * 255.0 for component in accepted]
    mean_strength = float(np.mean(strengths)) if strengths else 0.0
    strongest_strength = float(max(strengths, default=0.0))
    detection_score = float(max((component.score for component in accepted), default=0.0))
    dominant = max(accepted, key=lambda component: component.score, default=None)
    channel = dominant.polarity.value if dominant is not None else 'none'
    confidence = float(np.clip(0.50 + detection_score * 0.48, 0.0, 0.99)) if is_crack else float(np.clip(0.55 + pipeline_result.quality.score * 0.38, 0.0, 0.97))
    egg = pipeline_result.egg
    height, width = pipeline_result.working.shape[:2]
    egg_width_ratio = egg.width / max(float(width), 1.0)
    egg_length_ratio = egg.length / max(float(height), 1.0)
    calibration_profile = CALIBRATION_STORE.load()
    measurement = EggMeasurementService(cfg.calibration).measure(
        egg,
        (height, width),
        calibration_profile,
    )
    if measurement.valid and measurement.width_mm is not None and measurement.height_mm is not None:
        egg_size, egg_size_confidence, memberships, egg_size_score = _fuzzy_egg_size(
            measurement.width_mm,
            measurement.height_mm,
            cfg,
        )
        egg_size_mode = 'calibrated_mm'
        apparent_width_ratio = measurement.width_pixels / max(float(min(width, height)), 1.0)
        apparent_height_ratio = measurement.height_pixels / max(float(min(width, height)), 1.0)
        size_message = measurement.message
        size_requires_calibration = False
        size_measurement_valid = True
    else:
        (
            egg_size,
            egg_size_confidence,
            memberships,
            egg_size_score,
            apparent_width_ratio,
            apparent_height_ratio,
        ) = _fuzzy_apparent_egg_size(
            measurement.width_pixels,
            measurement.height_pixels,
            (height, width),
            cfg,
        )
        egg_size_mode = 'apparent_4_inch'
        size_message = (
            'Egg size was classified by Mamdani fuzzy logic from the detected height and width '
            'at the fixed 4-inch camera setup. Calibrate once to also return millimeters.'
        )
        size_requires_calibration = False
        size_measurement_valid = True
    crack_size, crack_size_confidence, crack_size_memberships, crack_size_score = _fuzzy_crack_size(
        is_crack=is_crack,
        traced_length=contour_length,
        traced_pixels=candidate_pixels,
        egg_area=egg_pixels,
        component_count=len(accepted),
        strongest_response=strongest_strength,
        cfg=cfg,
    )
    intermediate_steps = None
    if include_steps:
        intermediate_steps = {name: _encode_image(value, '.png') for name, value in pipeline_result.steps.items()}
    raw_count = pipeline_result.raw_component_count
    fragmentation_suppressed = raw_count > max(12, len(accepted) * 6) and not is_crack
    texture_pixels = int(cv2.countNonZero(pipeline_result.support_mask))
    texture_anomaly_ratio = texture_pixels / egg_pixels
    texture_score = float(np.clip(texture_anomaly_ratio / 0.08, 0.0, 1.0))
    result = {
        'id': str(uuid.uuid4()),
        'is_crack': is_crack,
        'confidence': confidence,
        'area_ratio': float(area_ratio),
        'contour_length': contour_length,
        'processing_time_ms': pipeline_result.processing_time_ms,
        'original_image_b64': _encode_image(pipeline_result.working, '.jpg'),
        'overlay_image_b64': _encode_image(overlay, '.png'),
        'intermediate_steps': intermediate_steps,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'candidate_components': len(accepted),
        'raw_candidate_components': raw_count,
        'dominant_crack_override': any(
            'coherent_fragment_group' in component.reasons
            or 'dominant_texture_survivor' in component.reasons
            or 'paper_guided_full_trace' in component.reasons
            for component in accepted
        ),
        'candidate_pixels': candidate_pixels,
        'longest_candidate': longest_candidate,
        'mean_candidate_strength': mean_strength,
        'detection_score': detection_score,
        'primary_detection_channel': channel,
        'pale_surface_score': detection_score if channel == CrackPolarity.BRIGHT.value else 0.0,
        'spatial_chain_score': detection_score if any('coherent_fragment_group' in component.reasons for component in accepted) else 0.0,
        'fragmentation_suppressed': fragmentation_suppressed,
        'threshold_used': int(round(max(*pipeline_result.dark_thresholds, *pipeline_result.bright_thresholds))),
        'paper_method_used': True,
        'paper_method_crack': pipeline_result.paper.crack,
        'paper_method_score': pipeline_result.paper.score,
        'paper_method_components': pipeline_result.paper.components,
        'shell_texture_score': texture_score,
        'shell_texture_uniformity': float(1.0 - texture_score),
        'texture_anomaly_ratio': float(texture_anomaly_ratio),
        'texture_candidate_pixels': texture_pixels,
        'thin_crack_score': detection_score,
        'thin_crack_detected': is_crack and max((component.mean_thickness for component in accepted), default=999.0) <= egg.minor_axis * 0.015,
        'image_quality_score': pipeline_result.quality.score,
        'image_sharpness': pipeline_result.quality.sharpness,
        'image_detail_variance': pipeline_result.quality.detail_variance,
        'image_saturated_ratio': pipeline_result.quality.saturated_ratio,
        'image_glare_ratio': pipeline_result.quality.glare_ratio,
        'image_dynamic_range': pipeline_result.quality.dynamic_range,
        'requires_recapture': False,
        'quality_message': pipeline_result.quality.message,
        'egg_detected': True,
        'egg_score': egg.score,
        'egg_size': egg_size,
        'egg_size_confidence': egg_size_confidence,
        'egg_area_ratio': egg.area_ratio,
        'egg_width_pixels': measurement.width_pixels,
        'egg_height_pixels': measurement.height_pixels,
        'egg_length_pixels': measurement.height_pixels,
        'egg_width_mm': measurement.width_mm,
        'egg_height_mm': measurement.height_mm,
        'egg_width_ratio': egg_width_ratio,
        'egg_length_ratio': egg_length_ratio,
        'egg_measurement_valid': size_measurement_valid,
        'egg_physical_measurement_valid': measurement.valid,
        'egg_measurement_message': size_message,
        'egg_size_requires_calibration': size_requires_calibration,
        'egg_size_mode': egg_size_mode,
        'egg_apparent_width_ratio': apparent_width_ratio,
        'egg_apparent_height_ratio': apparent_height_ratio,
        'camera_distance_inches': measurement.camera_distance_inches,
        'calibration_available': calibration_profile is not None,
        'calibration_pixels_per_mm': measurement.pixels_per_mm,
        'calibration_reference_width_mm': calibration_profile.reference_width_mm if calibration_profile is not None else None,
        'calibration_reference_width_pixels': calibration_profile.reference_width_pixels if calibration_profile is not None else None,
        'calibration_processed_width': calibration_profile.processed_width if calibration_profile is not None else None,
        'calibration_processed_height': calibration_profile.processed_height if calibration_profile is not None else None,
        'egg_size_score': egg_size_score,
        'egg_size_memberships': memberships,
        'crack_size': crack_size,
        'crack_size_confidence': crack_size_confidence,
        'crack_size_score': crack_size_score,
        'crack_size_memberships': crack_size_memberships,
        'crack_mask_b64': _encode_image(pipeline_result.crack_mask, '.png'),
        'crack_locations': [_component_location(index, component) for index, component in enumerate(accepted, start=1)],
        'detection_iterations': len(accepted),
        'search_iterations': len(accepted) + 1,
        'termination_reason': 'no_more_cracks',
        'sample_count': 1,
        'crack_votes': 1 if is_crack else 0,
        'no_crack_votes': 0 if is_crack else 1,
        'decision_consistency': 1.0,
        'area_consistent': True,
        'area_consistency': 1.0,
        'area_mean_ratio': float(area_ratio),
        'area_spread_ratio': 0.0,
        'area_samples': [float(area_ratio)] if is_crack else [],
        '_internal_crack_mask': pipeline_result.crack_mask.copy(),
        '_internal_support_mask': pipeline_result.support_mask.copy(),
        '_internal_egg_contour': pipeline_result.egg.contour.copy(),
    }
    return result


def detect_image_bytes(
    data: bytes,
    include_steps: bool = False,
    cfg: DetectionConfig = CONFIG,
) -> dict[str, Any]:
    image = _decode_input_image(data)
    try:
        pipeline_result = EggCrackPipeline(cfg).detect(image, include_steps)
    except (EggSegmentationError, ValueError) as exc:
        raise DetectionError(str(exc)) from exc
    result = _build_response(pipeline_result, include_steps, cfg)
    result['camera_orientation_fix'] = 'none'
    return result


def detect_iterative_image_bytes(
    data: bytes,
    include_steps: bool = False,
    cfg: DetectionConfig = CONFIG,
) -> dict[str, Any]:
    return detect_image_bytes(data, include_steps, cfg)


def detect_camera_image_bytes(
    data: bytes,
    include_steps: bool = False,
    cfg: DetectionConfig = CONFIG,
) -> dict[str, Any]:
    image = _correct_camera_orientation(_decode_input_image(data), cfg)
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise DetectionError('Could not prepare the camera image')
    result = detect_image_bytes(encoded.tobytes(), include_steps, cfg)
    result['camera_orientation_fix'] = cfg.camera_orientation_fix
    return result


def score_camera_focus_image_bytes(data: bytes, cfg: DetectionConfig = CONFIG) -> dict[str, Any]:
    image = _correct_camera_orientation(_decode_input_image(data), cfg)
    started = time.perf_counter()
    try:
        pipeline = EggCrackPipeline(cfg)
        working = pipeline._working_image(image)
        egg = pipeline.segmenter.segment(working)
    except (EggSegmentationError, ValueError) as exc:
        raise DetectionError(str(exc)) from exc

    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    focus_mask = egg.inner_mask.copy()
    erosion_size = max(3, int(round(egg.minor_axis * 0.035)))
    if erosion_size % 2 == 0:
        erosion_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size))
    reduced_mask = cv2.erode(focus_mask, kernel)
    if cv2.countNonZero(reduced_mask) >= max(cfg.min_inner_pixels, 500):
        focus_mask = reduced_mask

    inner_pixels = focus_mask > 0
    inner_values = gray[inner_pixels]
    if inner_values.size < 100:
        raise DetectionError('The illuminated egg region is too small')

    focus_gray = cv2.GaussianBlur(gray, (3, 3), 0)
    laplacian = cv2.Laplacian(focus_gray, cv2.CV_64F, ksize=3)
    detail_variance = float(np.var(laplacian[inner_pixels]))
    gradient_x = cv2.Sobel(focus_gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(focus_gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gradient_x, gradient_y)
    texture_sharpness = float(np.percentile(gradient[inner_pixels], 90.0))
    focus_score = float(
        np.log1p(max(detail_variance, 0.0)) * 0.72
        + np.log1p(max(texture_sharpness, 0.0)) * 0.28
    )
    normalized_focus_score = float(1.0 - np.exp(-focus_score / 7.0))
    median_brightness = float(np.percentile(inner_values, 50.0))
    p95_brightness = float(np.percentile(inner_values, 95.0))
    dynamic_range = float(np.percentile(inner_values, 95.0) - np.percentile(inner_values, 5.0))

    return {
        'egg_detected': True,
        'focus_score': round(focus_score, 6),
        'normalized_focus_score': round(float(np.clip(normalized_focus_score, 0.0, 1.0)), 6),
        'sharpness': round(detail_variance, 3),
        'detail_variance': round(detail_variance, 3),
        'texture_sharpness': round(texture_sharpness, 3),
        'dynamic_range': round(dynamic_range, 3),
        'egg_brightness': round(median_brightness, 3),
        'egg_p95_brightness': round(p95_brightness, 3),
        'egg_area_ratio': round(float(egg.area_ratio), 6),
        'focus_region': 'inner_egg',
        'message': 'Focus score measured only inside the egg',
        'processing_time_ms': int(round((time.perf_counter() - started) * 1000.0)),
    }


def _normalized_trace(result: dict[str, Any], key: str, cfg: DetectionConfig) -> np.ndarray:
    mask = result.get(key)
    if not isinstance(mask, np.ndarray):
        return np.zeros((cfg.temporal.normalized_height, cfg.temporal.normalized_width), dtype=np.uint8)
    contour = result.get('_internal_egg_contour')
    if isinstance(contour, np.ndarray) and contour.size:
        x, y, width, height = cv2.boundingRect(contour)
    else:
        points = cv2.findNonZero(mask)
        if points is None:
            return np.zeros((cfg.temporal.normalized_height, cfg.temporal.normalized_width), dtype=np.uint8)
        x, y, width, height = cv2.boundingRect(points)
    x = max(0, x)
    y = max(0, y)
    width = max(1, min(width, mask.shape[1] - x))
    height = max(1, min(height, mask.shape[0] - y))
    crop = mask[y:y + height, x:x + width]
    return cv2.resize(
        crop,
        (cfg.temporal.normalized_width, cfg.temporal.normalized_height),
        interpolation=cv2.INTER_NEAREST,
    )


def _trace_recall(reference: np.ndarray, support: np.ndarray) -> float:
    reference_pixels = cv2.countNonZero(reference)
    if reference_pixels == 0:
        return 0.0
    dilated = cv2.dilate(support, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    intersection = cv2.countNonZero(cv2.bitwise_and(reference, dilated))
    return intersection / float(reference_pixels)


def _strip_internal(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if not key.startswith('_internal_')}


def _clear_crack(result: dict[str, Any], message: str) -> dict[str, Any]:
    output = dict(result)
    mask = output.get('_internal_crack_mask')
    if isinstance(mask, np.ndarray):
        output['crack_mask_b64'] = _encode_image(np.zeros_like(mask), '.png')
    output.update({
        'is_crack': False,
        'candidate_components': 0,
        'candidate_pixels': 0,
        'contour_length': 0.0,
        'area_ratio': 0.0,
        'crack_size': 'none',
        'crack_size_confidence': 1.0,
        'crack_size_score': 0.0,
        'crack_size_memberships': {'none': 1.0},
        'crack_locations': [],
        'detection_iterations': 0,
        'primary_detection_channel': 'none',
        'quality_message': message,
    })
    return output


def detect_camera_images_bytes(
    frames: list[bytes],
    include_steps: bool = False,
    cfg: DetectionConfig = CONFIG,
) -> dict[str, Any]:
    if not frames:
        raise DetectionError('At least one camera frame is required')
    results = [detect_camera_image_bytes(frame, include_steps, cfg) for frame in frames]
    strong_indices = [index for index, result in enumerate(results) if result['is_crack']]
    required_votes = max(2 if len(results) > 1 else 1, int(math.ceil(len(results) * cfg.temporal.minimum_vote_ratio)))
    if not strong_indices:
        output = dict(max(results, key=lambda result: result['image_quality_score']))
        output.update({
            'sample_count': len(results),
            'crack_votes': 0,
            'no_crack_votes': len(results),
            'decision_consistency': 1.0,
            'termination_reason': 'no_more_cracks',
        })
        return _strip_internal(output)
    reference_index = max(strong_indices, key=lambda index: results[index]['detection_score'])
    reference = _normalized_trace(results[reference_index], '_internal_crack_mask', cfg)
    votes = 0
    area_samples: list[float] = []
    weak_support_used = False
    for index, result in enumerate(results):
        crack_trace = _normalized_trace(result, '_internal_crack_mask', cfg)
        support_trace = _normalized_trace(result, '_internal_support_mask', cfg)
        crack_recall = _trace_recall(reference, crack_trace)
        support_recall = _trace_recall(reference, support_trace)
        if result['is_crack'] and (index == reference_index or crack_recall >= cfg.temporal.minimum_overlap):
            votes += 1
            area_samples.append(float(result['area_ratio']))
        elif not result['is_crack'] and support_recall >= cfg.temporal.weak_support_overlap:
            votes += 1
            weak_support_used = True
    best = dict(results[reference_index])
    consistency = votes / float(len(results))
    if votes >= required_votes:
        mean_area = float(np.mean(area_samples)) if area_samples else float(best['area_ratio'])
        spread = float(np.ptp(area_samples)) if len(area_samples) > 1 else 0.0
        best.update({
            'sample_count': len(results),
            'crack_votes': votes,
            'no_crack_votes': len(results) - votes,
            'decision_consistency': consistency,
            'area_mean_ratio': mean_area,
            'area_spread_ratio': spread,
            'area_samples': area_samples,
            'area_consistent': spread <= max(mean_area * 0.8, 0.002),
            'area_consistency': float(max(0.0, 1.0 - spread / max(mean_area, 0.002))),
            'termination_reason': 'no_more_cracks',
            'quality_message': best['quality_message'] + ('; confirmed by weak support in another frame' if weak_support_used else '; confirmed across camera frames'),
        })
        return _strip_internal(best)
    cleared = _clear_crack(best, 'The possible crack was not stable across camera frames')
    cleared.update({
        'sample_count': len(results),
        'crack_votes': 0,
        'no_crack_votes': len(results),
        'decision_consistency': 1.0 - consistency,
        'area_mean_ratio': 0.0,
        'area_spread_ratio': 0.0,
        'area_samples': [],
        'termination_reason': 'multi_frame_disagreement',
    })
    return _strip_internal(cleared)


def _is_dominant_crack_component(component: CrackComponent, cfg: DetectionConfig = CONFIG) -> bool:
    metrics = component.metrics
    return (
        metrics.get('skeleton_length', 0.0) >= 0.18 * 550.0
        and metrics.get('span', 0.0) >= 0.12 * 550.0
        and metrics.get('density', 1.0) <= 0.45
        and metrics.get('average_thickness', 999.0) <= 12.0
    )


def _dominant_fragment_group(components: list[CrackComponent], cfg: DetectionConfig = CONFIG) -> list[CrackComponent]:
    if len(components) < 2:
        return []
    ordered = sorted(components, key=lambda component: component.x)
    group = [ordered[0]]
    for component in ordered[1:]:
        previous = group[-1]
        gap = component.x - (previous.x + previous.mask.shape[1])
        axis_a = np.array([previous.metrics.get('axis_x', 1.0), previous.metrics.get('axis_y', 0.0)])
        axis_b = np.array([component.metrics.get('axis_x', 1.0), component.metrics.get('axis_y', 0.0)])
        alignment = abs(float(np.dot(axis_a, axis_b)))
        if gap <= 80 and alignment >= 0.70:
            group.append(component)
    return group if len(group) >= 2 else []
