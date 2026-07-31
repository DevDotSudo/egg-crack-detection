import base64
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np

from app.core.config import CONFIG, DetectionConfig


class DetectionError(ValueError):
    pass


@dataclass
class CrackComponent:
    x: int
    y: int
    mask: np.ndarray
    skeleton: np.ndarray
    metrics: dict[str, float]
    source: str


def _paste_component(
    target: np.ndarray,
    component: CrackComponent,
    use_skeleton: bool = False,
) -> None:
    source = component.skeleton if use_skeleton else component.mask
    h, w = source.shape
    region = target[component.y:component.y + h, component.x:component.x + w]
    cv2.bitwise_or(region, source, dst=region)


def _encode_image(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 94])
    if not ok:
        raise DetectionError('Could not encode processed image')
    return base64.b64encode(encoded.tobytes()).decode('ascii')


def _encode_png_image(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise DetectionError('Could not encode processed image')
    return base64.b64encode(encoded.tobytes()).decode('ascii')


def _decode_input_image(data: bytes) -> np.ndarray:
    image = cv2.imdecode(
        np.frombuffer(data, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if image is None:
        raise DetectionError('Unsupported or corrupted image')
    return image


def _resize_letterbox(image: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h < 10 or w < 10:
        raise DetectionError('Image dimensions are too small')

    scale = min(width / w, height / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (nw, nh), interpolation=interpolation)

    edges = np.concatenate((
        resized[:4].reshape(-1, 3),
        resized[-4:].reshape(-1, 3),
        resized[:, :4].reshape(-1, 3),
        resized[:, -4:].reshape(-1, 3),
    ))
    border_color = np.median(edges, axis=0).astype(np.uint8)
    canvas = np.empty((height, width, 3), dtype=np.uint8)
    canvas[:] = border_color
    x = (width - nw) // 2
    y = (height - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def _prepare_working_image(
    image: np.ndarray,
    cfg: DetectionConfig,
) -> np.ndarray:
    h, w = image.shape[:2]
    if h < 10 or w < 10:
        raise DetectionError('Image dimensions are too small')

    max_width = cfg.target_width if w >= h else cfg.target_height
    max_height = cfg.target_height if w >= h else cfg.target_width
    scale = min(1.0, max_width / float(w), max_height / float(h))
    if scale >= 0.999:
        return image.copy()

    resized_width = max(1, int(round(w * scale)))
    resized_height = max(1, int(round(h * scale)))
    return cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )


def _egg_axes(egg_contour: np.ndarray) -> tuple[float, float]:
    if len(egg_contour) >= 5:
        axes = cv2.fitEllipse(egg_contour)[1]
        return float(min(axes)), float(max(axes))
    _, _, width, height = cv2.boundingRect(egg_contour)
    return float(min(width, height)), float(max(width, height))


def _refine_working_image(
    original: np.ndarray,
    working: np.ndarray,
    egg_contour: np.ndarray,
    cfg: DetectionConfig,
) -> np.ndarray | None:
    """Recover source detail when the fast camera resize makes the egg small."""
    original_h, original_w = original.shape[:2]
    working_h, working_w = working.shape[:2]
    if original_h <= working_h and original_w <= working_w:
        return None

    egg_minor_axis, _ = _egg_axes(egg_contour)
    required_scale = (
        cfg.detail_refinement_target_egg_minor_axis
        / max(egg_minor_axis, 1.0)
    )
    source_scale = min(
        original_w / max(float(working_w), 1.0),
        original_h / max(float(working_h), 1.0),
    )
    refinement_scale = min(
        required_scale,
        source_scale,
        cfg.detail_refinement_max_scale,
    )
    if refinement_scale < 1.08:
        return None

    target_width = max(working_w, int(round(working_w * refinement_scale)))
    target_height = max(working_h, int(round(working_h * refinement_scale)))
    return cv2.resize(
        original,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )


def _scaled_odd(value: int, scale: float, minimum: int = 3) -> int:
    scaled = max(minimum, int(round(value * scale)))
    return scaled if scaled % 2 == 1 else scaled + 1


def _scale_detection_config(
    cfg: DetectionConfig,
    egg_contour: np.ndarray,
) -> DetectionConfig:
    """Scale pixel geometry against the apparent width of the detected egg."""
    egg_minor_axis, _ = _egg_axes(egg_contour)
    scale = float(np.clip(
        egg_minor_axis / max(cfg.geometry_reference_egg_minor_axis, 1.0),
        cfg.geometry_min_scale,
        cfg.geometry_max_scale,
    ))

    length_fields = (
        'min_inner_margin',
        'local_hairline_min_length',
        'local_hairline_min_span',
        'local_hairline_max_thickness',
        'persistent_trace_min_skeleton_length',
        'persistent_trace_min_span',
        'persistent_trace_max_average_thickness',
        'pale_surface_min_skeleton_length',
        'pale_surface_max_skeleton_length',
        'pale_surface_max_average_thickness',
        'pale_surface_min_total_length',
        'pale_surface_min_group_span',
        'pale_surface_max_group_gap',
        'smooth_band_min_length',
        'smooth_band_min_thickness',
        'smooth_arc_min_length',
        'min_component_span',
        'min_skeleton_length',
        'preferred_max_thickness',
        'max_component_thickness',
        'support_min_span',
        'support_min_skeleton_length',
        'trace_geometry_min_skeleton_length',
        'trace_geometry_max_average_thickness',
        'trace_broad_min_skeleton_length',
        'trace_broad_max_skeleton_length',
        'trace_broad_max_average_thickness',
        'texture_seed_min_length',
        'texture_seed_min_span',
        'texture_seed_max_thickness',
        'fragmented_texture_crack_min_skeleton_length',
        'fragmented_texture_crack_max_thickness',
        'thin_crack_min_length',
        'thin_crack_min_span',
        'thin_crack_max_thickness',
        'dominant_min_skeleton_length',
        'dominant_min_span',
        'dominant_max_average_thickness',
        'dominant_network_min_skeleton_length',
        'dominant_network_min_span',
        'dominant_network_max_average_thickness',
        'texture_dominant_min_skeleton_length',
        'texture_dominant_min_span',
        'fragment_link_min_skeleton_length',
        'fragment_link_min_span',
        'fragment_link_max_average_thickness',
        'fragment_link_max_endpoint_gap',
        'fragment_group_min_total_length',
        'fragment_group_min_span',
        'spatial_chain_max_gap',
        'spatial_chain_min_total_length',
        'spatial_chain_min_span',
        'spatial_chain_max_thickness',
        'line_min_span',
        'line_min_skeleton_length',
        'line_max_thickness',
        'dark_crack_max_thickness',
        'dark_crack_max_max_thickness',
        'decision_min_longest',
        'decision_min_total_length',
        'fuzzy_length_scale',
    )
    values: dict[str, Any] = {
        name: float(getattr(cfg, name)) * scale
        for name in length_fields
    }
    area_scale = scale * scale
    values.update({
        'geometry_scale': scale,
        'pale_surface_min_pixels': max(
            1, int(round(cfg.pale_surface_min_pixels * area_scale)),
        ),
        'min_component_pixels': max(
            1, int(round(cfg.min_component_pixels * area_scale)),
        ),
        'iterative_min_new_pixels': max(
            1, int(round(cfg.iterative_min_new_pixels * area_scale)),
        ),
        'pale_surface_background_window': _scaled_odd(
            cfg.pale_surface_background_window, scale,
        ),
        'persistent_bright_kernel_length': _scaled_odd(
            cfg.persistent_bright_kernel_length, scale,
        ),
        'texture_window_size': _scaled_odd(cfg.texture_window_size, scale),
        'texture_coherence_window': _scaled_odd(
            cfg.texture_coherence_window, scale,
        ),
        'morph_kernel_length': _scaled_odd(cfg.morph_kernel_length, scale),
        'dark_valley_line_length': _scaled_odd(
            cfg.dark_valley_line_length, scale,
        ),
        'dark_valley_dilation_size': _scaled_odd(
            cfg.dark_valley_dilation_size, scale,
        ),
        'support_radius': max(1, int(round(cfg.support_radius * scale))),
        'texture_support_radius': max(
            1, int(round(cfg.texture_support_radius * scale)),
        ),
        'rim_band_thickness': max(
            1, int(round(cfg.rim_band_thickness * scale)),
        ),
        'iterative_exclusion_padding': max(
            1, int(round(cfg.iterative_exclusion_padding * scale)),
        ),
        'line_sigmas': tuple(sigma * scale for sigma in cfg.line_sigmas),
        'log_sigmas': tuple(sigma * scale for sigma in cfg.log_sigmas),
        'texture_ridge_sigmas': tuple(
            sigma * scale for sigma in cfg.texture_ridge_sigmas
        ),
        'local_hairline_windows': tuple(
            _scaled_odd(window, scale) for window in cfg.local_hairline_windows
        ),
        'persistent_bright_windows': tuple(
            _scaled_odd(window, scale)
            for window in cfg.persistent_bright_windows
        ),
        'morphology_sizes': tuple(
            _scaled_odd(size, scale) for size in cfg.morphology_sizes
        ),
        'tophat_line_kernels': tuple(
            (
                1 if width == 1 else _scaled_odd(width, scale, 1),
                1 if height == 1 else _scaled_odd(height, scale, 1),
            )
            for width, height in cfg.tophat_line_kernels
        ),
    })
    return replace(cfg, **values)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    flood = mask.copy()
    h, w = mask.shape
    cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    return cv2.bitwise_or(mask, cv2.bitwise_not(flood))


def _candidate_egg_masks(image: np.ndarray) -> list[np.ndarray]:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    red = cv2.GaussianBlur(image[:, :, 2], (11, 11), 0)
    green = cv2.GaussianBlur(image[:, :, 1], (11, 11), 0)

    h, w = gray.shape
    border = max(8, min(h, w) // 30)
    border_lab = np.concatenate((
        lab[:border].reshape(-1, 3),
        lab[-border:].reshape(-1, 3),
        lab[:, :border].reshape(-1, 3),
        lab[:, -border:].reshape(-1, 3),
    ))
    background = np.median(border_lab, axis=0)
    distance = np.linalg.norm(
        lab.astype(np.float32) - background.astype(np.float32), axis=2,
    )
    maximum = float(distance.max())
    if maximum > 0:
        distance = np.clip(distance * (255.0 / maximum), 0, 255)
    distance = distance.astype(np.uint8)

    border_gray = np.concatenate((
        gray[:border].reshape(-1),
        gray[-border:].reshape(-1),
        gray[:, :border].reshape(-1),
        gray[:, -border:].reshape(-1),
    ))
    background_gray = float(np.median(border_gray))
    gray_difference = np.clip(
        np.abs(gray.astype(np.float32) - background_gray) * 2.2,
        0,
        255,
    ).astype(np.uint8)
    value_difference = np.clip(
        (hsv[:, :, 2].astype(np.float32) - background_gray) * 2.0,
        0,
        255,
    ).astype(np.uint8)

    sources = (
        distance,
        gray_difference,
        value_difference,
        hsv[:, :, 1],
        gray,
        red,
        green,
    )
    raw_masks: list[np.ndarray] = []
    for source in sources:
        for mode in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
            _, binary = cv2.threshold(source, 0, 255, mode + cv2.THRESH_OTSU)
            raw_masks.append(binary)

    combined = cv2.max(distance, hsv[:, :, 1])
    _, combined_mask = cv2.threshold(
        combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    raw_masks.append(combined_mask)

    center_y, center_x = h // 2, w // 2
    center_patch = lab[
        max(0, center_y - max(4, h // 40)):min(h, center_y + max(4, h // 40) + 1),
        max(0, center_x - max(4, w // 40)):min(w, center_x + max(4, w // 40) + 1),
    ]
    if center_patch.size:
        center_color = np.median(center_patch.reshape(-1, 3), axis=0)
        center_distance = np.linalg.norm(
            lab.astype(np.float32) - center_color.astype(np.float32),
            axis=2,
        )
        center_threshold = max(12.0, float(np.percentile(center_distance, 32.0)))
        center_mask = np.where(center_distance <= center_threshold, 255, 0).astype(np.uint8)
        raw_masks.append(center_mask)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    masks: list[np.ndarray] = []
    for mask in raw_masks:
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_kernel)
        cleaned[[0, -1], :] = 0
        cleaned[:, [0, -1]] = 0
        masks.append(_fill_holes(cleaned))
    return masks


def _egg_score(
    contour: np.ndarray,
    shape: tuple[int, int],
    cfg: DetectionConfig,
) -> tuple[float, dict[str, float]]:
    h, w = shape
    area = float(cv2.contourArea(contour))
    area_ratio = area / max(float(h * w), 1.0)
    if area_ratio < cfg.min_egg_area_ratio or area_ratio > cfg.max_egg_area_ratio:
        return -1.0, {}

    x, y, bw, bh = cv2.boundingRect(contour)
    minor_dimension = min(bw, bh)
    major_dimension = max(bw, bh)
    if (
        minor_dimension < cfg.min_egg_width
        or major_dimension < cfg.min_egg_height
    ):
        return -1.0, {}
    touches = (
        int(x <= 4)
        + int(y <= 4)
        + int(x + bw >= w - 4)
        + int(y + bh >= h - 4)
    )
    if touches >= 2:
        return -1.0, {}

    hull_area = max(float(cv2.contourArea(cv2.convexHull(contour))), 1.0)
    solidity = area / hull_area
    perimeter = max(float(cv2.arcLength(contour, True)), 1.0)
    circularity = 4.0 * np.pi * area / (perimeter * perimeter)
    ellipse_fit = 0.0
    aspect = major_dimension / max(float(minor_dimension), 1.0)
    if len(contour) >= 5:
        axes = cv2.fitEllipse(contour)[1]
        minor_axis = max(min(axes), 1.0)
        major_axis = max(max(axes), 1.0)
        aspect = major_axis / minor_axis
        ellipse_area = np.pi * minor_axis * major_axis / 4.0
        ellipse_fit = min(area / ellipse_area, ellipse_area / area)

    if aspect < cfg.min_egg_aspect or aspect > cfg.max_egg_aspect:
        return -1.0, {}

    cx = x + bw / 2.0
    cy = y + bh / 2.0
    center_distance = np.hypot(cx - w / 2.0, cy - h / 2.0)
    center_distance /= max(np.hypot(w / 2.0, h / 2.0), 1.0)
    center_score = 1.0 - min(center_distance, 1.0)
    aspect_score = 1.0 - min(abs(aspect - 1.35) / 0.9, 1.0)

    score = (
        area_ratio * 5.0
        + solidity * 2.1
        + ellipse_fit * 2.0
        + center_score * 1.2
        + aspect_score
        + circularity * 0.5
        - touches * 0.8
    )
    return score, {
        'area_ratio': area_ratio,
        'aspect': aspect,
        'solidity': solidity,
        'ellipse_fit': ellipse_fit,
        'center_score': center_score,
        'touches': float(touches),
        'horizontal': float(bw > bh),
    }


def _detect_egg(
    image: np.ndarray,
    cfg: DetectionConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    best_contour: np.ndarray | None = None
    best_score = -1.0
    best_data: dict[str, float] = {}
    for mask in _candidate_egg_masks(image):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            score, data = _egg_score(contour, mask.shape, cfg)
            if score > best_score:
                best_contour = contour
                best_score = score
                best_data = data

    if best_contour is None or best_score < cfg.min_egg_score:
        raise DetectionError(
            'No egg detected. Use one fully visible egg on a plain dark background',
        )

    hull = cv2.convexHull(best_contour)
    hull_area = float(cv2.contourArea(hull))
    contour_area = max(float(cv2.contourArea(best_contour)), 1.0)
    if hull_area / contour_area <= 1.22:
        best_contour = hull

    if len(best_contour) >= 5:
        center, axes, angle = cv2.fitEllipse(best_contour)
        axis_x = max(2, int(round(axes[0] / 2.0)))
        axis_y = max(2, int(round(axes[1] / 2.0)))
        ellipse_points = cv2.ellipse2Poly(
            (int(round(center[0])), int(round(center[1]))),
            (axis_x, axis_y),
            int(round(angle)),
            0,
            360,
            2,
        )
        ellipse_contour = ellipse_points.reshape(-1, 1, 2).astype(np.int32)
        ellipse_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(ellipse_mask, [ellipse_contour], 255)
        overlap = cv2.countNonZero(cv2.bitwise_and(ellipse_mask, np.where(
            cv2.drawContours(np.zeros(image.shape[:2], dtype=np.uint8), [best_contour], -1, 255, cv2.FILLED) > 0,
            255,
            0,
        ).astype(np.uint8)))
        ellipse_pixels = max(cv2.countNonZero(ellipse_mask), 1)
        if overlap / ellipse_pixels >= 0.82:
            best_contour = ellipse_contour

    egg_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(egg_mask, [best_contour], 255)
    if cv2.countNonZero(egg_mask) < cfg.min_egg_pixels:
        raise DetectionError('The detected egg is too small. Move it closer to the camera')
    best_data['score'] = best_score
    return egg_mask, best_contour, best_data


def _inner_egg_mask(egg_mask: np.ndarray, cfg: DetectionConfig) -> np.ndarray:
    distance = cv2.distanceTransform(egg_mask, cv2.DIST_L2, 5)
    max_distance = float(distance.max())
    if max_distance < 8.0:
        raise DetectionError('The detected egg region is too small')
    margin = max(cfg.min_inner_margin, max_distance * cfg.inner_margin_ratio)
    inner = np.where(distance > margin, 255, 0).astype(np.uint8)
    if cv2.countNonZero(inner) < cfg.min_inner_pixels:
        raise DetectionError('The egg is not suitable for reliable crack analysis')
    return inner


def _select_coherent_paper_fragments(
    mask: np.ndarray,
    support_mask: np.ndarray | None = None,
    min_connector_support: float = 0.0,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Reject scattered shell markings while retaining a broken crack line."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    fragments: list[dict[str, float]] = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < 5:
            continue
        ys, xs = np.where(labels == index)
        if xs.size < 3:
            continue
        points = np.column_stack((xs, ys)).astype(np.float32)
        center = points.mean(axis=0)
        centered = points - center
        covariance = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        projections = centered @ axis
        transverse = centered @ np.array([-axis[1], axis[0]], dtype=np.float32)
        span = float(projections.max() - projections.min())
        width = float(transverse.max() - transverse.min() + 1.0)
        component = np.where(labels == index, 255, 0).astype(np.uint8)
        fragments.append({
            'label': float(index),
            'area': float(area),
            'center_x': float(center[0]),
            'center_y': float(center[1]),
            'axis_x': float(axis[0]),
            'axis_y': float(axis[1]),
            'span': span,
            'width': width,
            'elongation': span / max(width, 1.0),
            'skeleton_length': float(cv2.countNonZero(_skeletonize(component))),
        })

    if not fragments:
        return np.zeros_like(mask), []

    coherent_group: list[int] = []
    scale = max(
        float(np.hypot(*mask.shape)) / float(np.hypot(540.0, 960.0)),
        1.0,
    )
    if len(fragments) == 1:
        selected_indices = {0} if fragments[0]['span'] >= 10.0 * scale else set()
    else:
        selected_indices = {
            index
            for index, fragment in enumerate(fragments)
            if fragment['span'] >= 75.0 * scale
            and fragment['elongation'] >= 2.0
        }

        adjacency = [set() for _ in fragments]
        expanded_support = None
        if support_mask is not None:
            expanded_support = cv2.dilate(
                np.where(support_mask > 0, 255, 0).astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            )
        for left in range(len(fragments)):
            left_fragment = fragments[left]
            left_center = np.array([
                left_fragment['center_x'], left_fragment['center_y'],
            ], dtype=np.float32)
            left_axis = np.array([
                left_fragment['axis_x'], left_fragment['axis_y'],
            ], dtype=np.float32)
            for right in range(left + 1, len(fragments)):
                right_fragment = fragments[right]
                right_center = np.array([
                    right_fragment['center_x'], right_fragment['center_y'],
                ], dtype=np.float32)
                connector = right_center - left_center
                gap = float(np.linalg.norm(connector))
                if gap <= 1e-6 or gap > 70.0 * scale:
                    continue
                connector /= gap
                right_axis = np.array([
                    right_fragment['axis_x'], right_fragment['axis_y'],
                ], dtype=np.float32)
                if abs(float(np.dot(left_axis, right_axis))) < 0.40:
                    continue
                if min(
                    abs(float(np.dot(connector, left_axis))),
                    abs(float(np.dot(connector, right_axis))),
                ) < 0.57:
                    continue
                if expanded_support is not None:
                    connector_line = np.zeros_like(mask)
                    start_point = left_center + connector * (gap * 0.25)
                    end_point = left_center + connector * (gap * 0.75)
                    cv2.line(
                        connector_line,
                        tuple(np.rint(start_point).astype(int)),
                        tuple(np.rint(end_point).astype(int)),
                        255,
                        1,
                        cv2.LINE_8,
                    )
                    line_pixels = max(cv2.countNonZero(connector_line), 1)
                    supported_pixels = cv2.countNonZero(cv2.bitwise_and(
                        connector_line, expanded_support,
                    ))
                    if supported_pixels / line_pixels < min_connector_support:
                        continue
                adjacency[left].add(right)
                adjacency[right].add(left)

        groups: list[list[int]] = []
        visited: set[int] = set()
        for start in range(len(fragments)):
            if start in visited:
                continue
            stack = [start]
            visited.add(start)
            group: list[int] = []
            while stack:
                current = stack.pop()
                group.append(current)
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
            groups.append(group)

        total_area = max(sum(fragment['area'] for fragment in fragments), 1.0)
        qualifying_groups: list[tuple[float, list[int]]] = []
        for group in groups:
            if len(group) < 3:
                continue
            centers = np.asarray([
                [fragments[index]['center_x'], fragments[index]['center_y']]
                for index in group
            ], dtype=np.float32)
            distances = np.linalg.norm(
                centers[:, None, :] - centers[None, :, :], axis=2,
            )
            group_span = float(distances.max()) if distances.size else 0.0
            group_area = sum(fragments[index]['area'] for index in group)
            if (
                group_span >= 70.0 * scale
                and (
                    group_area / total_area >= 0.26
                    or len(group) / len(fragments) >= 0.50
                )
            ):
                qualifying_groups.append((group_span + group_area * 0.1, group))
        if qualifying_groups:
            coherent_group = max(qualifying_groups, key=lambda item: item[0])[1]
            selected_indices.update(coherent_group)

    if not selected_indices:
        return np.zeros_like(mask), []
    selected_labels = {
        int(round(fragments[index]['label'])) for index in selected_indices
    }
    selected = np.where(np.isin(labels, list(selected_labels)), 255, 0).astype(np.uint8)
    return selected, [fragments[index] for index in sorted(selected_indices)]




def _paper_method_crack_detection(
    image: np.ndarray,
    egg_mask: np.ndarray,
    inner_mask: np.ndarray,
    cfg: DetectionConfig = CONFIG,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    target_w, target_h = 1147, 633
    original_h, original_w = image.shape[:2]
    resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
    resized_egg = cv2.resize(egg_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    red_channel = resized[:, :, 2]
    green_channel = resized[:, :, 1]
    red_blurred = cv2.GaussianBlur(red_channel, (11, 11), 0)

    _, paper_binary = cv2.threshold(
        red_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    if cv2.countNonZero(paper_binary) > paper_binary.size * 0.5:
        paper_binary = cv2.bitwise_not(paper_binary)

    binary_egg = cv2.bitwise_and(paper_binary, resized_egg)
    binary_egg = cv2.morphologyEx(
        binary_egg,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    binary_egg = cv2.bitwise_or(binary_egg, resized_egg)

    safe_inner = cv2.resize(
        inner_mask,
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST,
    )
    safe_inner = cv2.bitwise_and(safe_inner, binary_egg)

    green_roi = cv2.multiply(green_channel, binary_egg, scale=1.0 / 255.0)
    local_background = cv2.GaussianBlur(green_roi, (0, 0), 9.0)
    bright_response = cv2.subtract(green_roi, local_background)
    dark_response = cv2.subtract(local_background, green_roi)
    # Amplify the dark side to compensate for its naturally weaker signal
    # so that surface cracks not transmitting light can reach threshold.
    boosted_dark = np.clip(
        dark_response.astype(np.float32) * 1.8, 0, 255,
    ).astype(np.uint8)
    contrast_response = cv2.max(bright_response, boosted_dark)
    contrast_response = cv2.bitwise_and(contrast_response, contrast_response, mask=safe_inner)

    values = contrast_response[safe_inner > 0]
    if values.size:
        dark_dominant = bool(
            float(np.mean(dark_response[safe_inner > 0]))
            > float(np.mean(bright_response[safe_inner > 0]))
        )
        percentile = 95.5 if dark_dominant else 96.8
        contrast_threshold = int(max(5, np.percentile(values, percentile)))
    else:
        contrast_threshold = 255
    _, contrast_mask = cv2.threshold(
        contrast_response, contrast_threshold, 255, cv2.THRESH_BINARY,
    )

    directional_response = np.zeros_like(green_roi)
    line_length = 21
    center = line_length // 2
    radius = center - 1
    for angle in (0, 30, 60, 90, 120, 150):
        radians = np.deg2rad(angle)
        dx = int(round(np.cos(radians) * radius))
        dy = int(round(np.sin(radians) * radius))
        line_kernel = np.zeros((line_length, line_length), dtype=np.uint8)
        cv2.line(
            line_kernel,
            (center - dx, center - dy),
            (center + dx, center + dy),
            1,
            1,
        )
        opened = cv2.morphologyEx(green_roi, cv2.MORPH_OPEN, line_kernel)
        directional_response = cv2.max(
            directional_response, cv2.subtract(green_roi, opened),
        )
    directional_response = cv2.bitwise_and(
        directional_response, directional_response, mask=safe_inner,
    )
    directional_values = directional_response[safe_inner > 0]
    directional_threshold = int(max(3, np.percentile(directional_values, 98.0))) if directional_values.size else 255
    _, directional_mask = cv2.threshold(
        directional_response, directional_threshold, 255, cv2.THRESH_BINARY,
    )

    median_value = float(np.median(green_roi[safe_inner > 0])) if cv2.countNonZero(safe_inner) else 80.0
    lower = int(max(12, 0.55 * median_value))
    upper = int(min(220, max(lower + 24, 1.65 * median_value)))
    edges = cv2.Canny(green_roi, lower, upper, L2gradient=True)
    edges = cv2.bitwise_and(edges, safe_inner)

    kernel_e = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    morphology = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_e, iterations=1)
    morphology = cv2.bitwise_and(morphology, safe_inner)

    subtraction = cv2.subtract(binary_egg, morphology)
    binary_inverse = cv2.bitwise_not(binary_egg)
    subtraction_inverse = cv2.bitwise_not(subtraction)
    paper_crack = cv2.subtract(subtraction_inverse, binary_inverse)
    paper_crack = cv2.bitwise_and(paper_crack, safe_inner)

    candidate = cv2.bitwise_or(paper_crack, cv2.bitwise_and(contrast_mask, morphology))
    candidate = cv2.bitwise_or(candidate, directional_mask)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel_e, iterations=1)
    candidate = cv2.bitwise_and(candidate, safe_inner)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    accepted_mask = np.zeros_like(candidate)
    areas: list[float] = []
    lengths: list[float] = []

    egg_pixels = max(float(cv2.countNonZero(binary_egg)), 1.0)
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        w = int(stats[index, cv2.CC_STAT_WIDTH])
        h = int(stats[index, cv2.CC_STAT_HEIGHT])
        if area < 7 or area > egg_pixels * 0.012:
            continue
        span = float(np.hypot(w, h))
        if span < 18.0:
            continue

        component = np.where(labels[y:y + h, x:x + w] == index, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(contour)
        rw, rh = rect[1]
        major = max(float(rw), float(rh), 1.0)
        minor = max(min(float(rw), float(rh)), 1.0)
        elongation = major / minor

        padded = cv2.copyMakeBorder(component, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        thickness = float(cv2.distanceTransform(padded, cv2.DIST_L2, 3).max() * 2.0)
        perimeter = float(cv2.arcLength(contour, False))
        density = area / max(float(w * h), 1.0)

        if elongation < 2.5 and perimeter < 32.0:
            continue
        if thickness > 8.0 or density > 0.58:
            continue

        region = accepted_mask[y:y + h, x:x + w]
        cv2.bitwise_or(region, component, dst=region)
        areas.append(float(area))
        lengths.append(max(perimeter, span))

    accepted_mask = cv2.morphologyEx(accepted_mask, cv2.MORPH_CLOSE, kernel_e, iterations=1)
    accepted_mask = cv2.bitwise_and(accepted_mask, safe_inner)

    polygon_source = cv2.dilate(
        accepted_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    polygon_contours, _ = cv2.findContours(
        polygon_source, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    filled_polygon = np.zeros_like(binary_egg)
    valid_polygons: list[np.ndarray] = []
    for contour in polygon_contours:
        if cv2.contourArea(contour) < 5.0:
            continue
        epsilon = max(0.8, 0.006 * cv2.arcLength(contour, True))
        polygon = cv2.approxPolyDP(contour, epsilon, True)
        if len(polygon) >= 3:
            valid_polygons.append(polygon)
    if valid_polygons:
        cv2.fillPoly(filled_polygon, valid_polygons, 255)
        filled_polygon = cv2.bitwise_and(filled_polygon, safe_inner)

    def restore(mask: np.ndarray, interpolation: int = cv2.INTER_NEAREST) -> np.ndarray:
        return cv2.resize(mask, (original_w, original_h), interpolation=interpolation)

    restored_polygon = restore(filled_polygon)
    restored_polygon = cv2.bitwise_and(restored_polygon, inner_mask)
    unfiltered_polygon = restored_polygon.copy()
    inner_distance = cv2.distanceTransform(inner_mask, cv2.DIST_L2, 5)
    maximum_depth = float(inner_distance.max())
    if maximum_depth > 0.0 and cfg.paper_min_depth_ratio > 0.0:
        paper_core = np.where(
            inner_distance >= maximum_depth * cfg.paper_min_depth_ratio,
            255,
            0,
        ).astype(np.uint8)
        restored_polygon = cv2.bitwise_and(restored_polygon, paper_core)
    restored_polygon, _ = _select_coherent_paper_fragments(
        restored_polygon,
    )
    raw_paper_components = _mask_to_components(
        restored_polygon, 'paper', cfg,
    )
    paper_components = [
        component
        for component in raw_paper_components
        if _is_crack_seed(component.metrics, cfg, 'paper')
    ]
    if _is_coherent_paper_line_group(raw_paper_components):
        paper_components = raw_paper_components
    restored_polygon[:] = 0
    for component in paper_components:
        _paste_component(restored_polygon, component)
    total_area = float(cv2.countNonZero(restored_polygon))
    fragment_lengths = [
        float(component.metrics['skeleton_length'])
        for component in paper_components
    ]
    total_length = float(sum(fragment_lengths))
    longest = float(max(fragment_lengths, default=0.0))
    is_crack = bool(paper_components and total_area >= 10.0)
    if not is_crack:
        restored_polygon[:] = 0
    score = min(
        1.0,
        (longest / 90.0) * 0.6 + (total_length / 220.0) * 0.4,
    ) if is_crack else 0.0
    return restored_polygon, {
        'is_crack': is_crack,
        'component_count': len(paper_components) if is_crack else 0,
        'total_area': total_area if is_crack else 0.0,
        'total_length': total_length if is_crack else 0.0,
        'longest_length': longest if is_crack else 0.0,
        'score': score,
    }, {
        'paper_red_blur': restore(red_blurred, cv2.INTER_LINEAR),
        'paper_binary_egg': restore(binary_egg),
        'paper_green_roi': restore(green_roi, cv2.INTER_LINEAR),
        'paper_edges': restore(edges),
        'paper_morphology': restore(morphology),
        'paper_subtraction': restore(subtraction),
        'paper_binary_inverse': restore(binary_inverse),
        'paper_subtraction_inverse': restore(subtraction_inverse),
        'paper_unfiltered_polygon': unfiltered_polygon,
    }


def _capture_quality_metrics(
    image: np.ndarray,
    egg_mask: np.ndarray,
    inner_mask: np.ndarray,
    egg_contour: np.ndarray,
    cfg: DetectionConfig,
) -> dict[str, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inner_values = gray[inner_mask > 0]
    if inner_values.size < 100:
        raise DetectionError('Move the egg closer and capture it again')

    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    laplacian_variance = float(np.var(laplacian[inner_mask > 0]))

    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gradient_x, gradient_y)
    boundary_band = np.zeros_like(gray)
    boundary_thickness = max(5, int(round(min(gray.shape) / 110.0)))
    cv2.drawContours(
        boundary_band,
        [egg_contour],
        -1,
        255,
        boundary_thickness,
        cv2.LINE_AA,
    )
    boundary_values = gradient[boundary_band > 0]
    boundary_sharpness = float(np.percentile(boundary_values, 90.0)) \
        if boundary_values.size else 0.0

    p05 = float(np.percentile(inner_values, 5.0))
    p50 = float(np.percentile(inner_values, 50.0))
    p95 = float(np.percentile(inner_values, 95.0))
    p99 = float(np.percentile(inner_values, 99.0))
    dynamic_range = p95 - p05
    saturated_ratio = float(np.mean(inner_values >= cfg.quality_saturation_value))

    glare_binary = np.where(
        (gray >= cfg.quality_glare_value) & (inner_mask > 0),
        255,
        0,
    ).astype(np.uint8)
    glare_binary = cv2.morphologyEx(
        glare_binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        glare_binary, connectivity=8,
    )
    largest_glare_ratio = 0.0
    largest_glare_thickness = 0.0
    inner_pixels = max(float(cv2.countNonZero(inner_mask)), 1.0)
    for index in range(1, count):
        area = float(stats[index, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        w = int(stats[index, cv2.CC_STAT_WIDTH])
        h = int(stats[index, cv2.CC_STAT_HEIGHT])
        component = np.where(
            labels[y:y + h, x:x + w] == index, 255, 0,
        ).astype(np.uint8)
        padded = cv2.copyMakeBorder(
            component, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0,
        )
        thickness = float(
            cv2.distanceTransform(padded, cv2.DIST_L2, 3).max() * 2.0,
        )
        ratio = area / inner_pixels
        if ratio > largest_glare_ratio:
            largest_glare_ratio = ratio
            largest_glare_thickness = thickness

    blurry = bool(
        boundary_sharpness < cfg.quality_hard_min_boundary_sharpness
        or (
            laplacian_variance < cfg.quality_good_laplacian_variance
            and boundary_sharpness < cfg.quality_min_boundary_sharpness
        )
    )
    broad_glare = bool(
        largest_glare_ratio >= cfg.quality_max_glare_component_ratio
        and largest_glare_thickness >= cfg.quality_max_glare_thickness
    )
    overexposed = bool(
        saturated_ratio >= cfg.quality_max_saturated_ratio
        or broad_glare
    )
    underexposed = p50 < cfg.quality_min_median
    low_contrast = dynamic_range < cfg.quality_min_dynamic_range

    if blurry:
        raise DetectionError(
            'The image is blurry. Keep the egg still, add more light, and capture again',
        )
    if overexposed:
        raise DetectionError(
            'The flashlight glare is too strong. Diffuse or reduce the light and capture again',
        )
    if underexposed:
        raise DetectionError(
            'The egg is too dark. Increase the light and capture again',
        )
    if low_contrast:
        raise DetectionError(
            'The shell detail is not clear. Adjust the light and capture again',
        )

    sharpness_score = float(np.clip(
        boundary_sharpness / max(cfg.quality_good_boundary_sharpness, 1.0),
        0.0,
        1.0,
    ))
    detail_score = float(np.clip(
        laplacian_variance / max(cfg.quality_good_laplacian_variance, 1.0),
        0.0,
        1.0,
    ))
    exposure_score = float(np.clip(
        1.0 - saturated_ratio / max(cfg.quality_max_saturated_ratio, 1e-6),
        0.0,
        1.0,
    ))
    contrast_score = float(np.clip(
        dynamic_range / max(cfg.quality_good_dynamic_range, 1.0),
        0.0,
        1.0,
    ))
    quality_score = (
        sharpness_score * 0.38
        + detail_score * 0.27
        + exposure_score * 0.20
        + contrast_score * 0.15
    )
    return {
        'quality_score': quality_score,
        'laplacian_variance': laplacian_variance,
        'boundary_sharpness': boundary_sharpness,
        'saturated_ratio': saturated_ratio,
        'glare_component_ratio': largest_glare_ratio,
        'glare_thickness': largest_glare_thickness,
        'dynamic_range': dynamic_range,
        'median_brightness': p50,
        'p99_brightness': p99,
    }


def _correct_overexposure(
    image: np.ndarray,
    egg_mask: np.ndarray,
    cfg: DetectionConfig,
) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    values = gray[egg_mask > 0]
    if values.size == 0:
        return image
    mean = float(np.mean(values))
    p95 = float(np.percentile(values, 95))
    if mean <= cfg.overexposure_mean_threshold and p95 <= cfg.overexposure_p95_threshold:
        return image

    severity = max(
        (mean - cfg.overexposure_mean_threshold) / 50.0,
        (p95 - cfg.overexposure_p95_threshold) / 10.0,
    )
    gamma = float(np.clip(1.0 + severity * 0.45, 1.0, cfg.overexposure_max_gamma))
    lookup = np.array([
        round(((value / 255.0) ** gamma) * 255.0) for value in range(256)
    ], dtype=np.uint8)
    return cv2.LUT(image, lookup)


def _prepare_candling_image(
    image: np.ndarray,
    egg_mask: np.ndarray,
    cfg: DetectionConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    corrected_input = _correct_overexposure(image, egg_mask, cfg)
    lab = cv2.cvtColor(corrected_input, cv2.COLOR_BGR2LAB)
    luminance = lab[:, :, 0]
    values = luminance[egg_mask > 0]
    fill = int(np.median(values)) if values.size else int(np.median(luminance))
    illumination_source = luminance.copy()
    illumination_source[egg_mask == 0] = fill
    maximum_dimension = max(illumination_source.shape)
    illumination_scale = min(
        1.0,
        cfg.flatfield_max_dimension / max(float(maximum_dimension), 1.0),
    )
    if illumination_scale < 0.999:
        small_width = max(
            1, int(round(illumination_source.shape[1] * illumination_scale)),
        )
        small_height = max(
            1, int(round(illumination_source.shape[0] * illumination_scale)),
        )
        small_source = cv2.resize(
            illumination_source,
            (small_width, small_height),
            interpolation=cv2.INTER_AREA,
        )
        small_illumination = cv2.GaussianBlur(
            small_source,
            (0, 0),
            max(cfg.flatfield_sigma * illumination_scale, 1.0),
        )
        illumination = cv2.resize(
            small_illumination,
            (illumination_source.shape[1], illumination_source.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    else:
        illumination = cv2.GaussianBlur(
            illumination_source, (0, 0), cfg.flatfield_sigma,
        )
    target = max(float(np.median(values)) if values.size else float(fill), 1.0)
    normalized = (
        luminance.astype(np.float32)
        / np.maximum(illumination.astype(np.float32), 1.0)
        * target
    )
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    blended = cv2.addWeighted(
        normalized,
        cfg.flatfield_strength,
        luminance,
        1.0 - cfg.flatfield_strength,
        0,
    )
    lab[:, :, 0] = blended
    corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    corrected[egg_mask == 0] = corrected_input[egg_mask == 0]

    tile = max(2, cfg.clahe_tile_size)
    clahe = cv2.createCLAHE(
        clipLimit=cfg.clahe_clip_limit,
        tileGridSize=(tile, tile),
    )
    enhanced = clahe.apply(blended)
    detail = enhanced
    # Only a light bilateral — no edgePreservingFilter. The EPF was
    # destroying crack edges that are clearly visible to the eye. CLAHE
    # already enhanced local contrast; the bilateral just removes speckle.
    detail = cv2.bilateralFilter(
        detail, 5, cfg.bilateral_sigma_color, cfg.bilateral_sigma_space,
    )
    detail = cv2.bitwise_and(detail, detail, mask=egg_mask)
    # Dark-preserved detail: minimal smoothing keeps faint dark cracks
    # intact for the dark valley detector channel.
    detail_dark_preserved = cv2.medianBlur(enhanced, 3)
    detail_dark_preserved = cv2.bitwise_and(
        detail_dark_preserved, detail_dark_preserved, mask=egg_mask,
    )
    illumination = cv2.bitwise_and(illumination, illumination, mask=egg_mask)
    return corrected, illumination, detail, detail_dark_preserved


def _masked_channel(channel: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = channel[mask > 0]
    fill = int(np.median(values)) if values.size else int(np.median(channel))
    output = channel.copy()
    output[mask == 0] = fill
    return output


def _local_hairline_responses(
    image: np.ndarray,
    inner_mask: np.ndarray,
    cfg: DetectionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    source = _masked_channel(gray, inner_mask)
    dark_best = np.zeros_like(gray)
    bright_best = np.zeros_like(gray)
    for requested in cfg.local_hairline_windows:
        window = max(3, int(requested) | 1)
        median = cv2.medianBlur(source, window)
        dark_best = cv2.max(dark_best, cv2.subtract(median, source))
        bright_best = cv2.max(bright_best, cv2.subtract(source, median))
    dark_best = cv2.bitwise_and(dark_best, dark_best, mask=inner_mask)
    bright_best = cv2.bitwise_and(bright_best, bright_best, mask=inner_mask)
    return dark_best, bright_best


def _persistent_hairline_trace(
    response: np.ndarray,
    inner_mask: np.ndarray,
    cfg: DetectionConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Find a faint light or dark crack persistent along many angles."""
    values = response[inner_mask > 0]
    if values.size == 0:
        return np.zeros_like(inner_mask), np.zeros_like(response), 0.0

    length = max(5, int(cfg.persistent_bright_kernel_length) | 1)
    center = length // 2
    persistence = np.zeros(response.shape, dtype=np.float32)
    source = response.astype(np.float32)
    for angle in cfg.morph_angles:
        radians = np.deg2rad(angle)
        dx = int(round(np.cos(radians) * center))
        dy = int(round(np.sin(radians) * center))
        kernel = np.zeros((length, length), dtype=np.float32)
        cv2.line(
            kernel,
            (center - dx, center - dy),
            (center + dx, center + dy),
            1.0,
            1,
        )
        kernel /= max(float(kernel.sum()), 1.0)
        persistence = np.maximum(
            persistence,
            cv2.filter2D(source, cv2.CV_32F, kernel),
        )
    persistence[inner_mask == 0] = 0.0

    threshold = float(np.percentile(
        persistence[inner_mask > 0], cfg.persistent_bright_percentile,
    ))
    distance = cv2.distanceTransform(inner_mask, cv2.DIST_L2, 5)
    maximum_depth = float(distance.max())
    if maximum_depth <= 0.0:
        return np.zeros_like(inner_mask), persistence, threshold
    central_core = distance >= maximum_depth * cfg.persistent_bright_core_ratio
    candidates = np.where(
        (persistence >= threshold)
        & (response >= cfg.persistent_bright_min_contrast)
        & central_core,
        255,
        0,
    ).astype(np.uint8)
    candidates = _connect_small_gaps(
        candidates,
        np.where(central_core, 255, 0).astype(np.uint8),
    )
    weak_support = np.where(
        (response >= cfg.persistent_bright_support_contrast)
        & central_core,
        255,
        0,
    ).astype(np.uint8)
    selected, _ = _select_coherent_paper_fragments(
        candidates,
        support_mask=weak_support,
        min_connector_support=cfg.persistent_bright_connector_support,
    )
    return selected, persistence, threshold


def _persistent_blue_response(
    image: np.ndarray,
    inner_mask: np.ndarray,
    cfg: DetectionConfig,
) -> np.ndarray:
    """Expose whitish transmitted-light cracks against a yellow shell."""
    source = _masked_channel(image[:, :, 0], inner_mask)
    response = np.zeros_like(source)
    for requested in cfg.persistent_bright_windows:
        window = max(3, int(requested) | 1)
        response = cv2.max(
            response,
            cv2.subtract(source, cv2.medianBlur(source, window)),
        )
    return cv2.bitwise_and(response, response, mask=inner_mask)


def _persistent_dark_response(
    image: np.ndarray,
    inner_mask: np.ndarray,
    cfg: DetectionConfig,
) -> np.ndarray:
    """Expose dark shell hairlines against the locally illuminated egg."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    source = _masked_channel(gray, inner_mask)
    response = np.zeros_like(source)
    for requested in cfg.persistent_bright_windows:
        window = max(3, int(requested) | 1)
        response = cv2.max(
            response,
            cv2.subtract(cv2.medianBlur(source, window), source),
        )
    return cv2.bitwise_and(response, response, mask=inner_mask)


def _dark_valley_response(
    detail_dark_preserved: np.ndarray,
    inner_mask: np.ndarray,
    cfg: DetectionConfig,
) -> np.ndarray:
    """Detect faint dark cracks as narrow valleys in the dark-preserved detail.

    A valley detector (local_max - pixel) highlights dark lines in locally
    brighter regions while suppressing uniformly dark areas (pores, texture).
    Directional filtering keeps only responses that appear in at least
    ``dark_valley_min_directional_count`` orientations — round pores produce
    responses in all directions so they dominate fewer; cracks appear only
    perpendicular to their axis.
    """
    source = _masked_channel(detail_dark_preserved, inner_mask)
    dilation_size = max(3, cfg.dark_valley_dilation_size | 1)
    local_max = cv2.dilate(
        source,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilation_size, dilation_size),
        ),
    )
    valley = cv2.subtract(local_max, source)
    valley = cv2.bitwise_and(valley, valley, mask=inner_mask)

    # Directional vote: keep only valleys confirmed along >= N orientations.
    length = max(5, cfg.dark_valley_line_length | 1)
    center = length // 2
    radius = center - 1
    vote = np.zeros(source.shape, dtype=np.uint8)
    for angle in (0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5):
        radians = np.deg2rad(angle)
        dx = int(round(np.cos(radians) * radius))
        dy = int(round(np.sin(radians) * radius))
        kernel = np.zeros((length, length), dtype=np.uint8)
        cv2.line(
            kernel,
            (center - dx, center - dy),
            (center + dx, center + dy),
            1,
            1,
        )
        opened = cv2.morphologyEx(source, cv2.MORPH_OPEN, kernel)
        directional_valley = cv2.subtract(source, opened)
        directional_valley = cv2.bitwise_and(
            directional_valley, directional_valley, mask=inner_mask,
        )
        # Count this angle as a vote wherever the response is non-trivial.
        vote += np.where(directional_valley >= 2, 1, 0).astype(np.uint8)

    # Suppress pixels that triggered in too few directions (round pores
    # trigger in many; genuine cracks trigger in few).  We keep pixels
    # that triggered in >= min_count AND <= 5 directions.
    min_count = max(1, cfg.dark_valley_min_directional_count)
    max_count = cfg.dark_valley_max_directional_count
    directional_mask = np.where(
        (vote >= min_count) & (vote <= max_count), 255, 0,
    ).astype(np.uint8)
    result = cv2.bitwise_and(valley, valley, mask=directional_mask)
    return cv2.bitwise_and(result, result, mask=inner_mask)


def _local_hairline_masks(
    response: np.ndarray,
    inner_mask: np.ndarray,
    cfg: DetectionConfig,
) -> tuple[np.ndarray, np.ndarray, int]:
    values = response[inner_mask > 0]
    if values.size == 0:
        empty = np.zeros_like(inner_mask)
        return empty, empty, cfg.local_hairline_min_strong_contrast
    noise = float(np.percentile(values, cfg.local_hairline_noise_percentile))
    strong_threshold = int(round(max(
        cfg.local_hairline_min_strong_contrast,
        noise * cfg.local_hairline_noise_scale,
    )))
    strong_threshold = int(np.clip(strong_threshold, 1, 255))
    weak_threshold = max(
        cfg.local_hairline_min_weak_contrast,
        int(round(strong_threshold * 0.45)),
    )
    weak = np.where(
        (response >= weak_threshold) & (inner_mask > 0), 255, 0,
    ).astype(np.uint8)
    strong = np.where(
        (response >= strong_threshold) & (inner_mask > 0), 255, 0,
    ).astype(np.uint8)
    return weak, strong, strong_threshold


def _is_local_hairline_component(
    component: CrackComponent,
    cfg: DetectionConfig,
) -> bool:
    metrics = component.metrics
    return bool(
        metrics['skeleton_length'] >= cfg.local_hairline_min_length
        and metrics['span'] >= cfg.local_hairline_min_span
        and metrics['average_thickness'] <= cfg.local_hairline_max_thickness
        and metrics['density'] <= cfg.local_hairline_max_density
        and metrics['elongation'] >= cfg.local_hairline_min_elongation
        and metrics['strength_p90'] >= cfg.local_hairline_min_strength
        and metrics['score'] >= cfg.local_hairline_min_score
        and metrics['rim_overlap'] < cfg.rim_overlap_reject_ratio
    )


def _is_dominant_local_hairline_component(
    component: CrackComponent,
    cfg: DetectionConfig,
) -> bool:
    """Identify a long raw-camera hairline before noisier maps can absorb it."""
    metrics = component.metrics
    return bool(
        _is_local_hairline_component(component, cfg)
        and metrics['skeleton_length'] >= cfg.texture_seed_min_length
        and metrics['span'] >= cfg.texture_seed_min_span
        and metrics['elongation'] >= 2.4
        and metrics['average_thickness'] <= cfg.local_hairline_max_thickness
        and metrics['strength_p90'] >= 12.0
    )


def _is_persistent_luminance_component(
    component: CrackComponent,
    cfg: DetectionConfig,
) -> bool:
    """Keep only true hairline-like traces from raw luminance persistence."""
    metrics = component.metrics
    return bool(
        metrics['skeleton_length'] >= cfg.persistent_trace_min_skeleton_length
        and metrics['span'] >= cfg.persistent_trace_min_span
        and metrics['elongation'] >= cfg.persistent_trace_min_elongation
        and metrics['average_thickness']
        <= cfg.persistent_trace_max_average_thickness
        and metrics['density'] <= cfg.persistent_trace_max_density
        and metrics['branch_ratio'] <= cfg.persistent_trace_max_branch_ratio
        and metrics['extent_ratio'] >= cfg.persistent_trace_min_extent_ratio
        and metrics['rim_overlap'] < cfg.rim_overlap_reject_ratio
    )


def _component_mask_overlap(
    component: CrackComponent,
    mask: np.ndarray,
) -> float:
    height, width = component.mask.shape
    region = mask[
        component.y:component.y + height,
        component.x:component.x + width,
    ]
    pixels = max(float(cv2.countNonZero(component.mask)), 1.0)
    return float(cv2.countNonZero(cv2.bitwise_and(component.mask, region))) / pixels


def _directional_tophat_response(
    channel: np.ndarray,
    inner_mask: np.ndarray,
    cfg: DetectionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply morphological top-hat and black-hat with directional (elongated)
    structuring elements to detect linear hairlines running along any axis.
    A 1×N kernel isolates nearly-horizontal hairlines; N×1 catches vertical
    ones. Two intermediate sizes cover diagonal cracks."""
    dark_best = np.zeros(channel.shape, dtype=np.float32)
    bright_best = np.zeros(channel.shape, dtype=np.float32)
    length = max(3, cfg.morph_kernel_length | 1)
    center = length // 2
    radius = center - 1
    for angle in cfg.morph_angles:
        radians = np.deg2rad(angle)
        dx = int(round(np.cos(radians) * radius))
        dy = int(round(np.sin(radians) * radius))
        kernel = np.zeros((length, length), dtype=np.uint8)
        cv2.line(
            kernel,
            (center - dx, center - dy),
            (center + dx, center + dy),
            1,
            1,
        )
        blackhat = cv2.morphologyEx(channel, cv2.MORPH_BLACKHAT, kernel)
        tophat = cv2.morphologyEx(channel, cv2.MORPH_TOPHAT, kernel)
        dark_best = np.maximum(dark_best, blackhat.astype(np.float32))
        bright_best = np.maximum(bright_best, tophat.astype(np.float32))

    dark = np.clip(dark_best * cfg.morph_amplify, 0, 255).astype(np.uint8)
    bright = np.clip(bright_best * cfg.morph_amplify, 0, 255).astype(np.uint8)
    dark = cv2.bitwise_and(dark, dark, mask=inner_mask)
    bright = cv2.bitwise_and(bright, bright, mask=inner_mask)
    return dark, bright


def _log_zero_crossing_response(
    detail: np.ndarray,
    inner_mask: np.ndarray,
    cfg: DetectionConfig,
) -> np.ndarray:
    """Laplacian-of-Gaussian (LoG) zero-crossing detector.

    A crack appears as a narrow dark trough between two bright flanks in the
    shell detail image. The LoG of that trough crosses zero exactly at the
    crack edges, producing a two-pixel-wide zero-crossing band centred on the
    crack. This is the theoretically optimal thin-line detector.

    The response image is a *strength* map: each zero-crossing pixel carries
    the magnitude of the gradient across the crossing so that noisy crossings
    (gradient near zero) are naturally suppressed by the downstream adaptive
    threshold.
    """
    source = _masked_channel(detail, inner_mask).astype(np.float32) / 255.0
    strength_best = np.zeros_like(source)

    for sigma in cfg.log_sigmas:
        # Compute LoG = Laplacian( Gaussian(source, sigma) )
        smooth = cv2.GaussianBlur(source, (0, 0), sigma)
        # Use CV_32F second-order Sobel as a fast LoG approximation.
        lap = cv2.Laplacian(smooth, cv2.CV_32F, ksize=3) * (sigma * sigma)

        # Detect sign changes between horizontally and vertically adjacent
        # pixel pairs. Where the LoG crosses zero, one neighbour is positive
        # and the other negative.
        pos = lap > 0
        neg = lap < 0
        # Horizontal crossings
        h_cross = (pos[:, :-1] & neg[:, 1:]) | (neg[:, :-1] & pos[:, 1:])
        # Vertical crossings
        v_cross = (pos[:-1, :] & neg[1:, :]) | (neg[:-1, :] & pos[1:, :])

        crossing = np.zeros_like(lap, dtype=bool)
        crossing[:, :-1] |= h_cross
        crossing[:, 1:] |= h_cross
        crossing[:-1, :] |= v_cross
        crossing[1:, :] |= v_cross

        # Strength = local gradient magnitude where a crossing is detected.
        gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(gx, gy)

        strength = np.where(crossing, grad, 0.0)
        strength_best = np.maximum(strength_best, strength)

    # Normalise to uint8. Use the 99th percentile inside the egg mask so
    # a single bright artefact does not compress the entire map to near-zero.
    values = strength_best[inner_mask > 0]
    p99 = float(np.percentile(values, 99.0)) if values.size else 1e-6
    p99 = max(p99, 1e-6)
    normalized = np.clip(strength_best * (255.0 / p99), 0, 255).astype(np.uint8)
    return cv2.bitwise_and(normalized, normalized, mask=inner_mask)


def _gradient_magnitude_response(
    detail: np.ndarray,
    inner_mask: np.ndarray,
    cfg: DetectionConfig,
) -> np.ndarray:
    """Gradient magnitude channel for detecting strong light/dark borders.

    In a candled egg, cracks appear as abrupt brightness transitions — the
    shell transmits light differently on each side of a fracture. This
    produces very strong gradient magnitudes along the crack, much stronger
    than the gradients from the mottled shell texture.

    Multi-scale Sobel gradients are computed and the maximum across scales
    is taken. A morphological thinning pass (non-maximum suppression via
    Canny-style NMS) keeps only the ridge of the gradient, producing thin
    response lines along crack borders.
    """
    source = _masked_channel(detail, inner_mask).astype(np.float32)
    best = np.zeros_like(source)

    for sigma in (0.8, 1.5, 2.5):
        smooth = cv2.GaussianBlur(source, (0, 0), sigma)
        gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        best = np.maximum(best, mag)

    # Normalise to uint8 using 99th percentile inside the egg mask.
    values = best[inner_mask > 0]
    p99 = float(np.percentile(values, 99.0)) if values.size else 1e-6
    p99 = max(p99, 1e-6)
    normalized = np.clip(best * (255.0 / p99), 0, 255).astype(np.uint8)
    return cv2.bitwise_and(normalized, normalized, mask=inner_mask)


def _line_responses(
    image: np.ndarray,
    detail: np.ndarray,
    inner_mask: np.ndarray,
    cfg: DetectionConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    green = image[:, :, 1]
    channels = (
        _masked_channel(green, inner_mask),
        _masked_channel(detail, inner_mask),
    )

    dark_responses: list[np.ndarray] = []
    bright_responses: list[np.ndarray] = []
    for channel in channels:
        # Light bilateral only — preserves crack edges that are clearly
        # visible in the CLAHE-enhanced image.
        smooth = cv2.bilateralFilter(channel, 5, 25, 25)
        for sigma in cfg.line_sigmas:
            background = cv2.GaussianBlur(smooth, (0, 0), sigma)
            dark = cv2.subtract(background, smooth)
            bright = cv2.subtract(smooth, background)
            # Fine-sigma passes (σ < 1.0) amplified more strongly since the
            # absolute difference is smaller for sub-pixel features.
            scale = 6.0 if sigma < 1.0 else 4.0
            dark_responses.append(
                np.clip(dark.astype(np.float32) * scale, 0, 255).astype(np.uint8),
            )
            bright_responses.append(
                np.clip(bright.astype(np.float32) * scale, 0, 255).astype(np.uint8),
            )

        for size in cfg.morphology_sizes:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            blackhat = cv2.morphologyEx(smooth, cv2.MORPH_BLACKHAT, kernel)
            tophat = cv2.morphologyEx(smooth, cv2.MORPH_TOPHAT, kernel)
            dark_responses.append(
                np.clip(blackhat.astype(np.float32) * 2.0, 0, 255).astype(np.uint8),
            )
            bright_responses.append(
                np.clip(tophat.astype(np.float32) * 2.0, 0, 255).astype(np.uint8),
            )

    # Directional top-hat pass: elongated rectangular kernels expose linear
    # hairlines that run along or near the horizontal / vertical axes.
    detail_masked = _masked_channel(detail, inner_mask)
    dir_dark, dir_bright = _directional_tophat_response(
        detail_masked, inner_mask, cfg,
    )
    dark_responses.append(dir_dark)
    bright_responses.append(dir_bright)

    def persistent_response(responses: list[np.ndarray]) -> np.ndarray:
        best = np.zeros_like(responses[0])
        second = np.zeros_like(responses[0])
        for current in responses:
            second = cv2.max(second, cv2.min(best, current))
            best = cv2.max(best, current)
        return cv2.addWeighted(best, 0.58, second, 0.42, 0)

    dark_response = persistent_response(dark_responses)
    bright_response = persistent_response(bright_responses)

    # Reinforce only features that survive directional black-hat/top-hat
    # extraction in the enhanced luminance and color channels. Unlike Canny,
    # these maps preserve crack centerlines instead of generic shell edges.
    color = _masked_channel(image[:, :, 1], inner_mask)
    color_dark, color_bright = _directional_tophat_response(
        color, inner_mask, cfg,
    )
    morphology_support = cv2.max(
        cv2.max(dir_dark, dir_bright),
        cv2.max(color_dark, color_bright),
    )
    dark_response = cv2.addWeighted(dark_response, 1.0, color_dark, 0.35, 0)
    bright_response = cv2.addWeighted(bright_response, 1.0, color_bright, 0.35, 0)
    dark_response = cv2.bitwise_and(dark_response, dark_response, mask=inner_mask)
    bright_response = cv2.bitwise_and(bright_response, bright_response, mask=inner_mask)
    return dark_response, bright_response, morphology_support


def _normalize_texture_response(
    response: np.ndarray,
    mask: np.ndarray,
    high_percentile: float = 99.6,
) -> np.ndarray:
    values = response[mask > 0]
    if values.size == 0:
        return np.zeros(mask.shape, dtype=np.uint8)
    low = float(np.percentile(values, 55.0))
    high = float(np.percentile(values, high_percentile))
    if high <= low + 1e-6:
        return np.zeros(mask.shape, dtype=np.uint8)
    normalized = (response.astype(np.float32) - low) * (255.0 / (high - low))
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    return cv2.bitwise_and(normalized, normalized, mask=mask)


def _hessian_ridge_responses(
    channel: np.ndarray,
    inner_mask: np.ndarray,
    cfg: DetectionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    source = _masked_channel(channel, inner_mask).astype(np.float32) / 255.0
    dark_best = np.zeros_like(source, dtype=np.float32)
    bright_best = np.zeros_like(source, dtype=np.float32)

    for sigma in cfg.texture_ridge_sigmas:
        smooth = cv2.GaussianBlur(source, (0, 0), sigma)
        scale = sigma * sigma
        dxx = cv2.Sobel(smooth, cv2.CV_32F, 2, 0, ksize=3) * scale
        dyy = cv2.Sobel(smooth, cv2.CV_32F, 0, 2, ksize=3) * scale
        dxy = cv2.Sobel(smooth, cv2.CV_32F, 1, 1, ksize=3) * scale

        trace = dxx + dyy
        difference = np.sqrt(np.maximum((dxx - dyy) ** 2 + 4.0 * dxy * dxy, 0.0))
        lambda_a = 0.5 * (trace - difference)
        lambda_b = 0.5 * (trace + difference)
        swap = np.abs(lambda_a) > np.abs(lambda_b)
        lambda_small = np.where(swap, lambda_b, lambda_a)
        lambda_large = np.where(swap, lambda_a, lambda_b)

        ratio = np.abs(lambda_small) / (np.abs(lambda_large) + 1e-6)
        energy = np.sqrt(lambda_small * lambda_small + lambda_large * lambda_large)
        valid_energy = energy[inner_mask > 0]
        scale_energy = float(np.percentile(valid_energy, 92.0)) if valid_energy.size else 0.02
        scale_energy = max(scale_energy, 1e-4)
        line_likeness = np.exp(-(ratio * ratio) / (2.0 * 0.45 * 0.45))
        strength = 1.0 - np.exp(-(energy * energy) / (2.0 * scale_energy * scale_energy))
        vesselness = line_likeness * strength

        dark_best = np.maximum(dark_best, np.where(lambda_large > 0, vesselness, 0.0))
        bright_best = np.maximum(bright_best, np.where(lambda_large < 0, vesselness, 0.0))

    dark = _normalize_texture_response(dark_best, inner_mask, 99.4)
    bright = _normalize_texture_response(bright_best, inner_mask, 99.4)
    return dark, bright


def _shell_texture_responses(
    detail: np.ndarray,
    inner_mask: np.ndarray,
    cfg: DetectionConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source = _masked_channel(detail, inner_mask).astype(np.float32)
    window = max(5, cfg.texture_window_size | 1)
    local_mean = cv2.boxFilter(source, cv2.CV_32F, (window, window), normalize=True)
    local_square = cv2.boxFilter(source * source, cv2.CV_32F, (window, window), normalize=True)
    local_std = np.sqrt(np.maximum(local_square - local_mean * local_mean, 0.0))
    denominator = local_std + 2.5
    dark_contrast = np.clip((local_mean - source) / denominator, 0.0, 6.0)
    bright_contrast = np.clip((source - local_mean) / denominator, 0.0, 6.0)
    dark_contrast = _normalize_texture_response(dark_contrast, inner_mask, 99.5)
    bright_contrast = _normalize_texture_response(bright_contrast, inner_mask, 99.5)

    smoothed = cv2.GaussianBlur(source, (0, 0), 0.8)
    gx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
    coherence_window = max(3, cfg.texture_coherence_window | 1)
    jxx = cv2.boxFilter(gx * gx, cv2.CV_32F, (coherence_window, coherence_window))
    jyy = cv2.boxFilter(gy * gy, cv2.CV_32F, (coherence_window, coherence_window))
    jxy = cv2.boxFilter(gx * gy, cv2.CV_32F, (coherence_window, coherence_window))
    coherence = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4.0 * jxy * jxy, 0.0))
    coherence /= jxx + jyy + 1e-6
    coherence = np.clip(coherence, 0.0, 1.0)
    coherence[inner_mask == 0] = 0.0

    ridge_dark, ridge_bright = _hessian_ridge_responses(detail, inner_mask, cfg)
    coherence_weight = np.clip(
        (coherence - cfg.texture_min_coherence)
        / max(1.0 - cfg.texture_min_coherence, 1e-6),
        0.0,
        1.0,
    )
    coherence_weight = 0.35 + 0.65 * coherence_weight

    dark = (
        ridge_dark.astype(np.float32) * cfg.texture_ridge_weight
        + dark_contrast.astype(np.float32) * cfg.texture_contrast_weight
    ) * coherence_weight
    bright = (
        ridge_bright.astype(np.float32) * cfg.texture_ridge_weight
        + bright_contrast.astype(np.float32) * cfg.texture_contrast_weight
    ) * coherence_weight

    dark = np.clip(dark, 0, 255).astype(np.uint8)
    bright = np.clip(bright, 0, 255).astype(np.uint8)
    dark = cv2.bitwise_and(dark, dark, mask=inner_mask)
    bright = cv2.bitwise_and(bright, bright, mask=inner_mask)
    anomaly = cv2.max(dark, bright)
    coherence_image = np.clip(coherence * 255.0, 0, 255).astype(np.uint8)
    coherence_image = cv2.bitwise_and(coherence_image, coherence_image, mask=inner_mask)
    return dark, bright, anomaly, coherence_image


def _attach_texture_metrics(
    components: list[CrackComponent],
    texture_response: np.ndarray,
    texture_strong: np.ndarray,
) -> None:
    for component in components:
        h, w = component.mask.shape
        response = texture_response[component.y:component.y + h, component.x:component.x + w]
        strong = texture_strong[component.y:component.y + h, component.x:component.x + w]
        values = response[component.mask > 0]
        if values.size == 0:
            component.metrics['texture_strength'] = 0.0
            component.metrics['texture_strength_p90'] = 0.0
            component.metrics['texture_overlap'] = 0.0
            continue
        texture_overlap = float(
            cv2.countNonZero(cv2.bitwise_and(component.mask, strong)),
        ) / max(float(cv2.countNonZero(component.mask)), 1.0)
        component.metrics['texture_strength'] = float(np.mean(values))
        component.metrics['texture_strength_p90'] = float(np.percentile(values, 90.0))
        component.metrics['texture_overlap'] = texture_overlap
        component.metrics['score'] += min(
            component.metrics['texture_strength_p90'] / 90.0,
            0.8,
        ) + min(texture_overlap * 1.2, 0.6)


def _is_thin_texture_crack(
    component: CrackComponent,
    cfg: DetectionConfig,
) -> bool:
    metrics = component.metrics
    return bool(
        _has_valid_trace_geometry(metrics, cfg)
        and metrics['skeleton_length'] >= cfg.thin_crack_min_length
        and metrics['span'] >= cfg.thin_crack_min_span
        and metrics['average_thickness'] <= cfg.thin_crack_max_thickness
        and metrics['elongation'] >= cfg.thin_crack_min_elongation
        and metrics.get('texture_strength_p90', 0.0)
        >= cfg.thin_crack_min_texture_strength
        and metrics.get('texture_overlap', 0.0)
        >= cfg.thin_crack_min_texture_overlap
        and metrics['score'] >= cfg.thin_crack_min_score
    )


def _is_standalone_texture_component(
    component: CrackComponent,
    cfg: DetectionConfig,
) -> bool:
    metrics = component.metrics
    ellipse_like_artifact = bool(
        metrics['ellipse_axis_ratio'] <= 2.2
        and metrics['ellipse_residual'] <= 0.035
        and metrics['ellipse_coverage'] >= 0.12
        and metrics['branch_ratio'] >= 0.38
    )
    if ellipse_like_artifact:
        return False
    return bool(
        metrics['skeleton_length'] >= cfg.texture_seed_min_length
        and metrics['span'] >= cfg.texture_seed_min_span
        and metrics['elongation'] >= cfg.texture_seed_min_elongation
        and metrics['average_thickness'] <= cfg.texture_seed_max_thickness
        and metrics['density'] <= cfg.texture_seed_max_density
        and metrics['strength_p90'] >= cfg.texture_seed_min_strength
        and metrics['strong_overlap'] >= cfg.texture_seed_min_strong_overlap
        and metrics['extent_ratio'] >= cfg.texture_seed_min_extent_ratio
        and metrics['rim_overlap'] < cfg.rim_overlap_reject_ratio
        and metrics['score'] >= cfg.texture_seed_min_score
    )


def _response_masks(
    response: np.ndarray,
    inner_mask: np.ndarray,
    min_weak: int,
    min_strong: int,
    cfg: DetectionConfig,
    *,
    is_dark_channel: bool = False,
) -> tuple[np.ndarray, np.ndarray, int]:
    values = response[inner_mask > 0]
    if values.size == 0:
        raise DetectionError('No valid egg interior was available for crack analysis')

    median = float(np.median(values))
    mad = float(np.median(np.abs(values.astype(np.float32) - median)))
    noise = max(mad, 1.0)
    weak_percentile = float(np.percentile(values, cfg.weak_percentile))
    strong_percentile = float(np.percentile(values, cfg.strong_percentile))
    mad_weak = cfg.dark_weak_mad_factor if is_dark_channel else cfg.weak_mad_factor
    mad_strong = cfg.dark_strong_mad_factor if is_dark_channel else cfg.strong_mad_factor
    pct_weak_scale = cfg.dark_weak_percentile_scale if is_dark_channel else cfg.weak_percentile_scale
    pct_strong_scale = cfg.dark_strong_percentile_scale if is_dark_channel else cfg.strong_percentile_scale
    weak_threshold = int(max(
        min_weak,
        median + mad_weak * noise,
        weak_percentile * pct_weak_scale,
    ))
    strong_threshold = int(max(
        min_strong,
        weak_threshold + 2,
        median + mad_strong * noise,
        strong_percentile * pct_strong_scale,
    ))
    weak_threshold = min(weak_threshold, cfg.max_response_threshold - 2)
    strong_threshold = min(strong_threshold, cfg.max_response_threshold)

    _, weak = cv2.threshold(response, weak_threshold, 255, cv2.THRESH_BINARY)
    _, strong = cv2.threshold(response, strong_threshold, 255, cv2.THRESH_BINARY)
    weak = cv2.bitwise_and(weak, weak, mask=inner_mask)
    strong = cv2.bitwise_and(strong, strong, mask=inner_mask)
    return weak, strong, strong_threshold


def _connect_small_gaps(binary: np.ndarray, inner_mask: np.ndarray) -> np.ndarray:
    connected = binary.copy()
    # 7px kernels bridge gaps up to ~3px that edge-preserving smoothing can
    # introduce in fine crack lines. The four orientations cover horizontal,
    # vertical, and both diagonal directions.
    kernels = (
        np.ones((1, 7), dtype=np.uint8),
        np.ones((7, 1), dtype=np.uint8),
        np.eye(7, dtype=np.uint8),
        np.fliplr(np.eye(7, dtype=np.uint8)),
    )
    for kernel in kernels:
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        connected = cv2.bitwise_or(connected, closed)
    return cv2.bitwise_and(connected, connected, mask=inner_mask)


def _skeletonize(binary: np.ndarray) -> np.ndarray:
    points = cv2.findNonZero(binary)
    output = np.zeros_like(binary)
    if points is None:
        return output

    x, y, w, h = cv2.boundingRect(points)
    x0 = max(0, x - 1)
    y0 = max(0, y - 1)
    x1 = min(binary.shape[1], x + w + 1)
    y1 = min(binary.shape[0], y + h + 1)
    image = binary[y0:y1, x0:x1].copy()
    skeleton = np.zeros_like(image)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    # Cap iterations to avoid stalling on large/thick components while still
    # fully thinning any genuine crack shape up to ~300 pixels thick.
    max_iters = min(max(image.shape) + 2, 320)
    for _ in range(max_iters):
        current_pixels = cv2.countNonZero(image)
        if current_pixels == 0:
            break
        eroded = cv2.erode(
            image, element, borderType=cv2.BORDER_CONSTANT, borderValue=0,
        )
        opened = cv2.dilate(
            eroded, element, borderType=cv2.BORDER_CONSTANT, borderValue=0,
        )
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(image, opened))
        if cv2.countNonZero(eroded) >= current_pixels:
            skeleton = cv2.bitwise_or(skeleton, eroded)
            break
        image = eroded
    output[y0:y1, x0:x1] = skeleton
    return output


def _rim_band(inner_mask: np.ndarray, thickness: int) -> np.ndarray:
    contours, _ = cv2.findContours(inner_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    band = np.zeros_like(inner_mask)
    cv2.drawContours(band, contours, -1, 255, thickness)
    return band


def _perimeter_zone(
    egg_mask: np.ndarray,
    inner_mask: np.ndarray,
    cfg: DetectionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    shell_depth = cv2.distanceTransform(egg_mask, cv2.DIST_L2, 5)
    maximum_depth = float(shell_depth.max())
    perimeter_depth = max(2.0, maximum_depth * cfg.perimeter_depth_ratio)
    perimeter = np.where(
        (inner_mask > 0) & (shell_depth <= perimeter_depth),
        255,
        0,
    ).astype(np.uint8)
    return perimeter, shell_depth


def _annotate_perimeter_geometry(
    component: CrackComponent,
    perimeter: np.ndarray,
    shell_depth: np.ndarray,
    egg_mask: np.ndarray,
) -> None:
    skeleton_y, skeleton_x = np.where(component.skeleton > 0)
    if skeleton_x.size == 0:
        component.metrics.update({
            'perimeter_overlap': 0.0,
            'perimeter_penetration': 0.0,
            'perimeter_tangent_alignment': 0.0,
        })
        return

    global_x = skeleton_x + component.x
    global_y = skeleton_y + component.y
    valid = (
        (global_x >= 0)
        & (global_x < perimeter.shape[1])
        & (global_y >= 0)
        & (global_y < perimeter.shape[0])
    )
    global_x = global_x[valid]
    global_y = global_y[valid]
    if global_x.size == 0:
        return

    perimeter_overlap = float(np.mean(perimeter[global_y, global_x] > 0))
    depths = shell_depth[global_y, global_x]
    penetration = float(depths.max() - depths.min()) if depths.size else 0.0

    moments = cv2.moments(egg_mask, binaryImage=True)
    if abs(moments['m00']) > 1e-6:
        egg_center = np.array([
            moments['m10'] / moments['m00'],
            moments['m01'] / moments['m00'],
        ], dtype=np.float32)
    else:
        egg_center = np.array([
            egg_mask.shape[1] / 2.0,
            egg_mask.shape[0] / 2.0,
        ], dtype=np.float32)
    component_center = np.array([
        component.x + component.metrics.get('center_x', 0.0),
        component.y + component.metrics.get('center_y', 0.0),
    ], dtype=np.float32)
    radial = component_center - egg_center
    radial_length = float(np.linalg.norm(radial))
    axis = np.array([
        component.metrics.get('axis_x', 0.0),
        component.metrics.get('axis_y', 0.0),
    ], dtype=np.float32)
    axis_length = float(np.linalg.norm(axis))
    tangent_alignment = 0.0
    if radial_length > 1e-6 and axis_length > 1e-6:
        radial /= radial_length
        axis /= axis_length
        radial_alignment = float(np.clip(abs(np.dot(axis, radial)), 0.0, 1.0))
        tangent_alignment = float(np.sqrt(max(0.0, 1.0 - radial_alignment ** 2)))

    component.metrics.update({
        'perimeter_overlap': perimeter_overlap,
        'perimeter_penetration': penetration,
        'perimeter_tangent_alignment': tangent_alignment,
    })


def _has_valid_perimeter_geometry(
    component: CrackComponent,
    shell_depth: np.ndarray,
    cfg: DetectionConfig,
) -> bool:
    metrics = component.metrics
    if metrics.get('perimeter_overlap', 0.0) < cfg.perimeter_min_component_overlap:
        return True
    minimum_penetration = max(
        2.0 * cfg.geometry_scale,
        float(shell_depth.max()) * cfg.perimeter_min_penetration_ratio,
    )
    return bool(
        metrics.get('perimeter_penetration', 0.0) >= minimum_penetration
        or metrics.get('perimeter_tangent_alignment', 0.0)
        < cfg.perimeter_max_tangent_alignment
    )


def _component_metrics(
    component: np.ndarray,
    response: np.ndarray,
    strong_mask: np.ndarray,
    rim_band: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    ys, xs = np.where(component > 0)
    pixels = len(xs)
    if pixels == 0:
        return {}, np.zeros_like(component)

    points = np.column_stack((xs, ys)).astype(np.float32)
    x, y, w, h = cv2.boundingRect(points.astype(np.int32))
    centered = points - points.mean(axis=0, keepdims=True)
    covariance = np.cov(centered, rowvar=False) if pixels >= 3 else np.eye(2)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    principal_axis = eigenvectors[:, -1]
    center = points.mean(axis=0)
    projections = centered @ principal_axis
    endpoint_a = center + principal_axis * float(projections.min())
    endpoint_b = center + principal_axis * float(projections.max())
    elongation = float(np.sqrt(
        max(float(eigenvalues[-1]), 1e-6)
        / max(float(eigenvalues[0]), 1e-6),
    ))

    skeleton = _skeletonize(component)
    skeleton_pixels = cv2.countNonZero(skeleton)
    skeleton_length = float(skeleton_pixels)
    span = float(np.hypot(w, h))
    extent_ratio = span / max(skeleton_length, 1.0)
    average_thickness = pixels / max(skeleton_length, 1.0)
    padded_component = cv2.copyMakeBorder(
        component, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0,
    )
    distance = cv2.distanceTransform(padded_component, cv2.DIST_L2, 3)
    max_thickness = float(distance[1:-1, 1:-1].max()) * 2.0
    density = pixels / max(float(w * h), 1.0)

    response_values = response[component > 0]
    strength = float(np.mean(response_values))
    strength_p90 = float(np.percentile(response_values, 90))
    strong_overlap = float(
        cv2.countNonZero(cv2.bitwise_and(component, strong_mask)),
    ) / max(float(pixels), 1.0)
    rim_overlap = float(
        cv2.countNonZero(cv2.bitwise_and(component, rim_band)),
    ) / max(float(pixels), 1.0)

    binary_skeleton = (skeleton > 0).astype(np.uint8)
    neighbors = cv2.filter2D(
        binary_skeleton, cv2.CV_16S, np.ones((3, 3), dtype=np.int16),
    ) - binary_skeleton.astype(np.int16)
    endpoint_mask = (binary_skeleton > 0) & (neighbors == 1)
    endpoints = int(np.count_nonzero(endpoint_mask))
    branchpoints = int(np.count_nonzero((binary_skeleton > 0) & (neighbors >= 3)))
    branch_ratio = branchpoints / max(float(skeleton_pixels), 1.0)

    skeleton_y, skeleton_x = np.where(binary_skeleton > 0)
    endpoint_chord_ratio = 1.0
    if endpoints == 2:
        endpoint_y, endpoint_x = np.where(endpoint_mask)
        chord = float(np.hypot(
            endpoint_x[1] - endpoint_x[0],
            endpoint_y[1] - endpoint_y[0],
        ))
        endpoint_chord_ratio = chord / max(skeleton_length, 1.0)

    axis_deviation_ratio = 0.0
    ellipse_axis_ratio = 999.0
    ellipse_residual = 1.0
    ellipse_coverage = 0.0
    if skeleton_pixels >= 12:
        skeleton_points = np.column_stack((skeleton_x, skeleton_y)).astype(np.float32)
        skeleton_centered = skeleton_points - skeleton_points.mean(
            axis=0, keepdims=True,
        )
        perpendicular = np.array([-principal_axis[1], principal_axis[0]])
        axis_deviation_ratio = float(np.mean(np.abs(
            skeleton_centered @ perpendicular,
        ))) / max(span, 1.0)

        if skeleton_pixels >= 20:
            try:
                ellipse = cv2.fitEllipse(skeleton_points.reshape(-1, 1, 2))
                (ellipse_cx, ellipse_cy), (ellipse_w, ellipse_h), angle = ellipse
                minimum_axis = max(min(ellipse_w, ellipse_h), 1e-6)
                maximum_axis = max(ellipse_w, ellipse_h)
                ellipse_axis_ratio = maximum_axis / minimum_axis
                if minimum_axis >= 5.0 and ellipse_axis_ratio <= 8.0:
                    radians = np.deg2rad(angle)
                    cosine = float(np.cos(radians))
                    sine = float(np.sin(radians))
                    shifted_x = skeleton_points[:, 0] - ellipse_cx
                    shifted_y = skeleton_points[:, 1] - ellipse_cy
                    rotated_x = shifted_x * cosine + shifted_y * sine
                    rotated_y = -shifted_x * sine + shifted_y * cosine
                    normalized_x = rotated_x / max(ellipse_w / 2.0, 1e-6)
                    normalized_y = rotated_y / max(ellipse_h / 2.0, 1e-6)
                    radius = np.sqrt(normalized_x * normalized_x + normalized_y * normalized_y)
                    ellipse_residual = float(np.median(np.abs(radius - 1.0)))
                    angles = np.mod(np.arctan2(normalized_y, normalized_x), 2.0 * np.pi)
                    sorted_angles = np.sort(angles)
                    wrapped = np.concatenate((sorted_angles, sorted_angles[:1] + 2.0 * np.pi))
                    largest_gap = float(np.max(np.diff(wrapped)))
                    ellipse_coverage = float((2.0 * np.pi - largest_gap) / (2.0 * np.pi))
            except cv2.error:
                pass

    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    area = float(sum(cv2.contourArea(contour) for contour in contours))
    perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
    circularity = 4.0 * np.pi * area / max(perimeter * perimeter, 1.0)
    return {
        'pixels': float(pixels),
        'width': float(w),
        'height': float(h),
        'span': span,
        'elongation': elongation,
        'skeleton_length': skeleton_length,
        'extent_ratio': extent_ratio,
        'average_thickness': average_thickness,
        'max_thickness': max_thickness,
        'density': density,
        'strength': strength,
        'strength_p90': strength_p90,
        'strong_overlap': strong_overlap,
        'rim_overlap': rim_overlap,
        'endpoints': float(endpoints),
        'branchpoints': float(branchpoints),
        'branch_ratio': branch_ratio,
        'endpoint_chord_ratio': endpoint_chord_ratio,
        'axis_deviation_ratio': axis_deviation_ratio,
        'ellipse_axis_ratio': ellipse_axis_ratio,
        'ellipse_residual': ellipse_residual,
        'ellipse_coverage': ellipse_coverage,
        'circularity': circularity,
        'axis_x': float(principal_axis[0]),
        'axis_y': float(principal_axis[1]),
        'center_x': float(center[0]),
        'center_y': float(center[1]),
        'endpoint_a_x': float(endpoint_a[0]),
        'endpoint_a_y': float(endpoint_a[1]),
        'endpoint_b_x': float(endpoint_b[0]),
        'endpoint_b_y': float(endpoint_b[1]),
    }, skeleton


def _component_score(metrics: dict[str, float], cfg: DetectionConfig) -> float:
    scale = max(cfg.geometry_scale, 1e-6)
    normalized_length = metrics['skeleton_length'] / scale
    normalized_span = metrics['span'] / scale
    normalized_thickness = metrics['average_thickness'] / scale
    score = 0.0
    score += min(normalized_length / 70.0, 1.8)
    score += min(normalized_span / 65.0, 1.4)
    score += min(metrics['elongation'] / 4.0, 1.1)
    score += min(metrics['strength_p90'] / 55.0, 1.0)
    score += min(metrics['strong_overlap'] * 2.0, 0.7)
    score += min(metrics['branchpoints'] / 4.0, 0.35)
    score -= max(
        0.0,
        normalized_thickness - cfg.preferred_max_thickness / scale,
    ) * 0.16
    score -= max(0.0, metrics['density'] - 0.45) * 2.6
    score -= metrics['rim_overlap'] * 1.5
    return score


def _has_strict_line_geometry(
    metrics: dict[str, float],
    cfg: DetectionConfig,
) -> bool:
    """Require an actual thin trace, not merely a locally dark shell region."""
    scale = cfg.geometry_scale
    return bool(
        metrics['average_thickness'] <= 4.0 * scale
        and metrics['max_thickness'] <= 7.0 * scale
        and metrics['density'] <= 0.34
        and metrics['branch_ratio'] <= 0.32
        and (
            metrics['elongation'] >= 2.2
            or metrics['extent_ratio'] >= 0.65
        )
    )


def _has_relaxed_dark_geometry(
    metrics: dict[str, float],
    cfg: DetectionConfig,
) -> bool:
    """Looser geometry gate for dark crack sources that are naturally noisier."""
    return bool(
        metrics['average_thickness'] <= cfg.dark_crack_max_thickness
        and metrics['max_thickness'] <= cfg.dark_crack_max_max_thickness
        and metrics['density'] <= cfg.dark_crack_max_density
        and metrics['branch_ratio'] <= cfg.dark_crack_max_branch_ratio
        and (
            metrics['elongation'] >= 1.8
            or metrics['extent_ratio'] >= 0.55
        )
    )


def _has_valid_trace_geometry(
    metrics: dict[str, float],
    cfg: DetectionConfig,
) -> bool:
    """Reject dense shell ridges that only resemble cracks after enhancement."""
    if metrics['skeleton_length'] < cfg.trace_geometry_min_skeleton_length:
        return True
    thin_axis_line = (
        metrics['average_thickness'] <= 2.4 * cfg.geometry_scale
        and metrics['max_thickness'] <= 6.0 * cfg.geometry_scale
        and metrics['branch_ratio'] <= 0.25
        and metrics['axis_deviation_ratio'] <= 0.05
        and (
            metrics['elongation'] >= 3.0
            or metrics['extent_ratio'] >= 0.85
        )
    )
    if thin_axis_line:
        return True
    broad_low_contrast_trace = (
        cfg.trace_broad_min_skeleton_length
        <= metrics['skeleton_length']
        <= cfg.trace_broad_max_skeleton_length
        and metrics['elongation'] >= cfg.trace_broad_min_elongation
        and metrics['average_thickness']
        <= cfg.trace_broad_max_average_thickness
        and metrics['density'] <= cfg.trace_broad_max_density
        and metrics['branch_ratio'] <= cfg.trace_broad_max_branch_ratio
        and metrics['extent_ratio'] >= cfg.trace_broad_min_extent_ratio
        and metrics['axis_deviation_ratio']
        <= cfg.trace_broad_max_axis_deviation
        and metrics['strong_overlap'] >= cfg.trace_broad_min_strong_overlap
        and cfg.trace_broad_min_texture_overlap
        <= metrics.get('texture_overlap', 0.0)
        <= cfg.trace_broad_max_texture_overlap
        and metrics.get('texture_strength_p90', 0.0)
        <= cfg.trace_broad_max_texture_strength
    )
    if broad_low_contrast_trace:
        return True
    return bool(
        metrics['average_thickness']
        <= cfg.trace_geometry_max_average_thickness
        and metrics['density'] <= cfg.trace_geometry_max_density
        and metrics['branch_ratio'] <= cfg.trace_geometry_max_branch_ratio
        and metrics['extent_ratio'] >= cfg.trace_geometry_min_extent_ratio
        and metrics['axis_deviation_ratio']
        <= cfg.trace_geometry_max_axis_deviation
    )


def _is_coherent_paper_line_group(
    components: list[CrackComponent],
) -> bool:
    """Accept a broken faint line only when its fragments form one long axis."""
    if len(components) < 3:
        return False

    centers = np.asarray([
        [
            component.x + component.metrics['center_x'],
            component.y + component.metrics['center_y'],
        ]
        for component in components
    ], dtype=np.float32)
    centered = centers - centers.mean(axis=0, keepdims=True)
    covariance = np.cov(centered, rowvar=False)
    eigenvalues = np.sort(np.linalg.eigvalsh(covariance))
    elongation = float(np.sqrt(
        max(float(eigenvalues[-1]), 1e-6)
        / max(float(eigenvalues[0]), 1e-6),
    ))
    distances = np.linalg.norm(
        centers[:, None, :] - centers[None, :, :], axis=2,
    )
    span = float(distances.max()) if distances.size else 0.0
    total_length = float(sum(
        component.metrics['skeleton_length'] for component in components
    ))
    average_thickness = float(sum(
        component.metrics['average_thickness']
        * component.metrics['skeleton_length']
        for component in components
    )) / max(total_length, 1.0)
    return bool(
        total_length >= 90.0
        and span >= 65.0
        and elongation >= 3.0
        and average_thickness <= 8.0
    )


def _is_crack_seed(
    metrics: dict[str, float],
    cfg: DetectionConfig,
    source: str = '',
) -> bool:
    if metrics['rim_overlap'] >= cfg.rim_overlap_reject_ratio:
        return False
    if metrics['pixels'] < cfg.min_component_pixels:
        return False
    if (
        metrics['span'] < cfg.min_component_span
        or metrics['skeleton_length'] < cfg.min_skeleton_length
    ):
        return False
    if source == 'paper' and not _has_strict_line_geometry(metrics, cfg):
        return False
    if source == 'persistent_dark' and not _has_relaxed_dark_geometry(
        metrics, cfg,
    ):
        return False
    if metrics['skeleton_length'] >= 12.0 and metrics['branch_ratio'] > 0.75:
        return False
    if metrics['skeleton_length'] >= 35.0 and metrics['extent_ratio'] < 0.56:
        return False
    smooth_broad_band = (
        metrics['skeleton_length'] >= cfg.smooth_band_min_length
        and metrics['branch_ratio'] >= cfg.smooth_band_min_branch_ratio
        and metrics['average_thickness'] >= cfg.smooth_band_min_thickness
        and metrics['branchpoints'] >= cfg.smooth_band_min_branchpoints
    )
    smooth_ellipse_arc = (
        metrics['skeleton_length'] >= cfg.smooth_arc_min_length
        and metrics['branchpoints'] == 0
        and metrics['endpoints'] <= 2
        and metrics['ellipse_axis_ratio'] <= cfg.smooth_arc_max_axis_ratio
        and metrics['ellipse_residual'] <= cfg.smooth_arc_max_residual
        and metrics['ellipse_coverage'] >= cfg.smooth_arc_min_coverage
        and metrics['endpoint_chord_ratio'] <= cfg.smooth_arc_max_chord_ratio
    )
    if smooth_broad_band or smooth_ellipse_arc:
        return False

    compact_loop = (
        metrics['span'] < 42.0
        and metrics['circularity'] > 0.48
        and metrics['endpoints'] == 0
        and metrics['branchpoints'] == 0
    )
    if compact_loop:
        return False

    thin = (
        metrics['average_thickness'] <= cfg.max_component_thickness
        and (
            metrics['density'] <= cfg.max_component_density
            or metrics['elongation'] >= 3.0
        )
    )
    line_like = (
        metrics['elongation'] >= cfg.min_elongation
        or metrics['endpoints'] >= 2
        or metrics['branchpoints'] >= 1
        or metrics['skeleton_length'] >= 55.0
    )
    is_dark_hairline_source = source in (
        'local',
        'local_dark',
        'blackhat',
        'dark',
        'persistent_dark',
        'dark_valley',
    ) and metrics['elongation'] >= 2.4
    min_required_score = 0.25 if is_dark_hairline_source else cfg.min_component_score
    return thin and line_like and metrics['score'] >= min_required_score


def _is_crack_support(metrics: dict[str, float], cfg: DetectionConfig) -> bool:
    if metrics['rim_overlap'] >= min(cfg.rim_overlap_reject_ratio, 0.50):
        return False
    if (
        metrics['span'] < cfg.support_min_span
        or metrics['skeleton_length'] < cfg.support_min_skeleton_length
    ):
        return False
    if metrics['average_thickness'] > cfg.max_component_thickness * 1.25:
        return False
    if metrics['skeleton_length'] >= 35.0 and metrics['extent_ratio'] < 0.56:
        return False
    if metrics['skeleton_length'] >= 12.0 and metrics['branch_ratio'] > 0.75:
        return False
    if (
        metrics['skeleton_length'] >= cfg.smooth_band_min_length
        and metrics['branch_ratio'] >= cfg.smooth_band_min_branch_ratio
        and metrics['average_thickness'] >= cfg.smooth_band_min_thickness
        and metrics['branchpoints'] >= cfg.smooth_band_min_branchpoints
    ):
        return False
    if (
        metrics['skeleton_length'] >= cfg.smooth_arc_min_length
        and metrics['branchpoints'] == 0
        and metrics['endpoints'] <= 2
        and metrics['ellipse_axis_ratio'] <= cfg.smooth_arc_max_axis_ratio
        and metrics['ellipse_residual'] <= cfg.smooth_arc_max_residual
        and metrics['ellipse_coverage'] >= cfg.smooth_arc_min_coverage
        and metrics['endpoint_chord_ratio'] <= cfg.smooth_arc_max_chord_ratio
    ):
        return False
    if metrics['density'] > 0.78 and metrics['elongation'] < 2.2:
        return False
    line_like = (
        metrics['elongation'] >= 1.18
        or metrics['endpoints'] >= 2
        or metrics['branchpoints'] >= 1
    )
    return line_like and metrics['score'] >= 0.55


def _extract_crack_components(
    binary: np.ndarray,
    response: np.ndarray,
    strong_mask: np.ndarray,
    rim_band: np.ndarray,
    source: str,
    cfg: DetectionConfig,
) -> tuple[np.ndarray, list[CrackComponent]]:
    # Strong pixels establish independent crack seeds. Measuring weak pixels
    # first can merge a real line with thousands of nearby shell pores into
    # one dense component, which is the main reason hairlines were discarded.
    seed_binary = _connect_small_gaps(strong_mask, binary)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        seed_binary, connectivity=8,
    )
    seeds: list[CrackComponent] = []
    max_candidate_pixels = max(3500, int(binary.size * 0.006))

    for index in range(1, count):
        pixels = int(stats[index, cv2.CC_STAT_AREA])
        if pixels < cfg.min_component_pixels:
            continue
        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        w = int(stats[index, cv2.CC_STAT_WIDTH])
        h = int(stats[index, cv2.CC_STAT_HEIGHT])
        if np.hypot(w, h) < cfg.support_min_span:
            continue
        box_density = pixels / max(float(w * h), 1.0)
        box_aspect = max(w / max(float(h), 1.0), h / max(float(w), 1.0))
        if box_density > 0.70 and box_aspect < 2.2:
            continue
        if pixels > max_candidate_pixels:
            continue
        component_mask = np.where(
            labels[y:y + h, x:x + w] == index, 255, 0,
        ).astype(np.uint8)
        metrics, skeleton = _component_metrics(
            component_mask,
            response[y:y + h, x:x + w],
            strong_mask[y:y + h, x:x + w],
            rim_band[y:y + h, x:x + w],
        )
        if not metrics:
            continue
        metrics['score'] = _component_score(metrics, cfg)
        component = CrackComponent(x, y, component_mask, skeleton, metrics, source)
        if _is_crack_seed(component.metrics, cfg, source):
            seeds.append(component)

    if not seeds:
        return np.zeros_like(binary), []

    accepted = np.zeros_like(binary)
    accepted_components = list(seeds)
    for seed in seeds:
        _paste_component(accepted, seed)

    # Recover faint pixels connected to a validated strong seed without
    # swallowing an entire weak shell-texture component.
    growth_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(cfg.support_radius):
        grown = cv2.bitwise_and(cv2.dilate(accepted, growth_kernel), binary)
        updated = cv2.bitwise_or(accepted, grown)
        if cv2.countNonZero(updated) == cv2.countNonZero(accepted):
            break
        accepted = updated
    return accepted, accepted_components


def _deduplicate_components(
    accepted_mask: np.ndarray,
    response: np.ndarray,
    strong_mask: np.ndarray,
    rim_band: np.ndarray,
    cfg: DetectionConfig,
) -> list[CrackComponent]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        accepted_mask, connectivity=8,
    )
    components: list[CrackComponent] = []
    max_component_pixels = max(12000, int(accepted_mask.size * 0.04))
    for index in range(1, count):
        pixels = int(stats[index, cv2.CC_STAT_AREA])
        if pixels < cfg.min_component_pixels:
            continue
        if pixels > max_component_pixels:
            continue
        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        w = int(stats[index, cv2.CC_STAT_WIDTH])
        h = int(stats[index, cv2.CC_STAT_HEIGHT])
        component_mask = np.where(
            labels[y:y + h, x:x + w] == index, 255, 0,
        ).astype(np.uint8)
        metrics, skeleton = _component_metrics(
            component_mask,
            response[y:y + h, x:x + w],
            strong_mask[y:y + h, x:x + w],
            rim_band[y:y + h, x:x + w],
        )
        if not metrics:
            continue
        metrics['score'] = _component_score(metrics, cfg)
        if _is_crack_support(metrics, cfg):
            components.append(CrackComponent(
                x, y, component_mask, skeleton, metrics, 'combined',
            ))
    return components


def _mask_to_components(
    mask: np.ndarray,
    source: str,
    cfg: DetectionConfig,
) -> list[CrackComponent]:
    """Represent an already validated binary trace in the common component form."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    components: list[CrackComponent] = []
    empty_rim = np.zeros_like(mask)
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) < cfg.min_component_pixels:
            continue
        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        component_mask = np.where(
            labels[y:y + height, x:x + width] == index, 255, 0,
        ).astype(np.uint8)
        metrics, skeleton = _component_metrics(
            component_mask,
            component_mask,
            component_mask,
            empty_rim[y:y + height, x:x + width],
        )
        if not metrics:
            continue
        metrics['score'] = _component_score(metrics, cfg)
        components.append(CrackComponent(
            x, y, component_mask, skeleton, metrics, source,
        ))
    return components


def _is_dominant_crack_component(
    component: CrackComponent,
    cfg: DetectionConfig,
) -> bool:
    metrics = component.metrics
    dominant_line = bool(
        metrics['skeleton_length'] >= cfg.dominant_min_skeleton_length
        and metrics['span'] >= cfg.dominant_min_span
        and metrics['elongation'] >= cfg.dominant_min_elongation
        and metrics['average_thickness'] <= cfg.dominant_max_average_thickness
        and metrics['density'] <= cfg.dominant_max_density
        and metrics['strength_p90'] >= cfg.dominant_min_strength_p90
        and metrics['strong_overlap'] >= cfg.dominant_min_strong_overlap
        and metrics['score'] >= cfg.dominant_min_component_score
    )
    sparse_strong_network = bool(
        metrics['skeleton_length'] >= cfg.dominant_network_min_skeleton_length
        and metrics['span'] >= cfg.dominant_network_min_span
        and metrics['elongation'] >= cfg.dominant_network_min_elongation
        and metrics['average_thickness']
        <= cfg.dominant_network_max_average_thickness
        and metrics['density'] <= cfg.dominant_network_max_density
        and metrics['strength_p90'] >= cfg.dominant_network_min_strength_p90
        and metrics['strong_overlap']
        >= cfg.dominant_network_min_strong_overlap
        and metrics['score'] >= cfg.dominant_network_min_component_score
    )
    texture_supported_ridge = bool(
        metrics['skeleton_length']
        >= cfg.texture_dominant_min_skeleton_length
        and metrics['span'] >= cfg.texture_dominant_min_span
        and metrics['elongation'] >= cfg.texture_dominant_min_elongation
        and metrics['density'] <= cfg.texture_dominant_max_density
        and metrics['strength_p90']
        >= cfg.texture_dominant_min_strength_p90
        and metrics['strong_overlap']
        >= cfg.texture_dominant_min_strong_overlap
        and metrics.get('texture_strength_p90', 0.0)
        >= cfg.texture_dominant_min_texture_strength
        and metrics.get('texture_overlap', 0.0)
        >= cfg.texture_dominant_min_texture_overlap
    )
    return dominant_line or sparse_strong_network or texture_supported_ridge


def _dominant_fragment_group(
    components: list[CrackComponent],
    cfg: DetectionConfig,
) -> list[CrackComponent]:
    candidates = [
        component
        for component in components
        if component.metrics['skeleton_length']
        >= cfg.fragment_link_min_skeleton_length
        and component.metrics['span'] >= cfg.fragment_link_min_span
        and component.metrics['elongation'] >= cfg.fragment_link_min_elongation
        and component.metrics['average_thickness']
        <= cfg.fragment_link_max_average_thickness
        and component.metrics['density'] <= cfg.fragment_link_max_density
        and component.metrics['strength_p90']
        >= cfg.fragment_link_min_strength_p90
    ]
    if len(candidates) < cfg.fragment_group_min_components:
        return []

    def global_point(component: CrackComponent, prefix: str) -> np.ndarray:
        return np.array([
            component.x + component.metrics[f'{prefix}_x'],
            component.y + component.metrics[f'{prefix}_y'],
        ], dtype=np.float32)

    def aligns_with_anchor(
        anchor: CrackComponent,
        other: CrackComponent,
    ) -> bool:
        anchor_axis = np.array([
            anchor.metrics['axis_x'], anchor.metrics['axis_y'],
        ], dtype=np.float32)
        other_axis = np.array([
            other.metrics['axis_x'], other.metrics['axis_y'],
        ], dtype=np.float32)
        axis_alignment = abs(float(np.dot(anchor_axis, other_axis)))
        if axis_alignment < cfg.fragment_link_min_axis_alignment:
            return False

        anchor_endpoints = (
            global_point(anchor, 'endpoint_a'),
            global_point(anchor, 'endpoint_b'),
        )
        other_endpoints = (
            global_point(other, 'endpoint_a'),
            global_point(other, 'endpoint_b'),
        )
        endpoint_gap = min(
            float(np.linalg.norm(anchor_point - other_point))
            for anchor_point in anchor_endpoints
            for other_point in other_endpoints
        )
        if endpoint_gap > cfg.fragment_link_max_endpoint_gap:
            return False

        connector = (
            global_point(other, 'center') - global_point(anchor, 'center')
        )
        connector_length = float(np.linalg.norm(connector))
        if connector_length <= 1e-6:
            return False
        connector /= connector_length
        connector_alignment = abs(float(np.dot(connector, anchor_axis)))
        return (
            connector_alignment
            >= cfg.fragment_link_min_connector_alignment
        )

    # Build each group around one straight-line anchor. A transitive graph can
    # incorrectly walk through curved shell texture until it reaches a crack;
    # direct anchor membership keeps all retained pieces on the same line.
    groups: list[list[CrackComponent]] = []
    for anchor in candidates:
        groups.append([
            component
            for component in candidates
            if component is anchor or aligns_with_anchor(anchor, component)
        ])

    qualifying: list[tuple[float, list[CrackComponent]]] = []
    for group in groups:
        if len(group) < cfg.fragment_group_min_components:
            continue
        total_length = sum(
            component.metrics['skeleton_length'] for component in group
        )
        min_x = min(component.x for component in group)
        min_y = min(component.y for component in group)
        max_x = max(component.x + component.mask.shape[1] for component in group)
        max_y = max(component.y + component.mask.shape[0] for component in group)
        group_span = float(np.hypot(max_x - min_x, max_y - min_y))
        mean_strength = float(np.mean([
            component.metrics['strength_p90'] for component in group
        ]))
        if (
            total_length >= cfg.fragment_group_min_total_length
            and group_span >= cfg.fragment_group_min_span
            and mean_strength >= cfg.fragment_group_min_mean_strength
        ):
            qualifying.append((total_length + group_span, group))
    if not qualifying:
        return []
    return max(qualifying, key=lambda item: item[0])[1]


def _merge_component_group(
    group: list[CrackComponent],
    shape: tuple[int, int],
    response: np.ndarray,
    strong_mask: np.ndarray,
    rim_band: np.ndarray,
    cfg: DetectionConfig,
) -> CrackComponent | None:
    if not group:
        return None

    combined = np.zeros(shape, dtype=np.uint8)
    combined_skeleton = np.zeros(shape, dtype=np.uint8)
    total_source_length = 0.0
    weighted_strength = 0.0
    weighted_overlap = 0.0
    weighted_rim = 0.0
    max_thickness = 0.0
    endpoints = 0.0
    branchpoints = 0.0
    points: list[list[float]] = []

    for component in group:
        _paste_component(combined, component)
        _paste_component(combined_skeleton, component, use_skeleton=True)
        metrics = component.metrics
        length = max(float(metrics.get('skeleton_length', 0.0)), 1.0)
        total_source_length += length
        weighted_strength += float(metrics.get('strength', 0.0)) * length
        weighted_overlap += float(metrics.get('strong_overlap', 0.0)) * length
        weighted_rim += float(metrics.get('rim_overlap', 0.0)) * length
        max_thickness = max(max_thickness, float(metrics.get('max_thickness', 0.0)))
        endpoints += float(metrics.get('endpoints', 0.0))
        branchpoints += float(metrics.get('branchpoints', 0.0))
        points.extend([
            [metrics.get('endpoint_a_x', 0.0) + component.x, metrics.get('endpoint_a_y', 0.0) + component.y],
            [metrics.get('endpoint_b_x', 0.0) + component.x, metrics.get('endpoint_b_y', 0.0) + component.y],
            [metrics.get('center_x', 0.0) + component.x, metrics.get('center_y', 0.0) + component.y],
        ])

    nonzero = cv2.findNonZero(combined)
    if nonzero is None:
        return None
    x, y, w, h = cv2.boundingRect(nonzero)
    component_mask = combined[y:y + h, x:x + w].copy()
    skeleton = combined_skeleton[y:y + h, x:x + w].copy()
    pixels = float(cv2.countNonZero(component_mask))
    skeleton_length = float(cv2.countNonZero(skeleton))
    if skeleton_length <= 0.0:
        skeleton = _skeletonize(component_mask)
        skeleton_length = float(cv2.countNonZero(skeleton))
    if skeleton_length <= 0.0:
        return None

    point_array = np.asarray(points, dtype=np.float32)
    centered = point_array - point_array.mean(axis=0, keepdims=True)
    covariance = np.cov(centered, rowvar=False) if len(point_array) >= 3 else np.eye(2)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    principal_axis = eigenvectors[:, order[-1]]
    elongation = float((eigenvalues[-1] + 1e-3) / (eigenvalues[0] + 1e-3))
    distances = np.linalg.norm(
        point_array[:, None, :] - point_array[None, :, :],
        axis=2,
    )
    span = float(np.max(distances)) if distances.size else float(np.hypot(w, h))
    average_thickness = pixels / max(skeleton_length, 1.0)
    density = pixels / max(float(w * h), 1.0)
    center = point_array.mean(axis=0)
    projections = centered @ principal_axis
    endpoint_a = point_array[int(np.argmin(projections))]
    endpoint_b = point_array[int(np.argmax(projections))]
    length_weight = max(total_source_length, 1.0)

    metrics = {
        'pixels': pixels,
        'span': span,
        'elongation': elongation,
        'skeleton_length': skeleton_length,
        'extent_ratio': skeleton_length / max(span, 1.0),
        'average_thickness': average_thickness,
        'max_thickness': max_thickness,
        'density': density,
        'strength': weighted_strength / length_weight,
        'strength_p90': max(float(component.metrics.get('strength_p90', 0.0)) for component in group),
        'strong_overlap': weighted_overlap / length_weight,
        'rim_overlap': weighted_rim / length_weight,
        'endpoints': endpoints,
        'branchpoints': branchpoints,
        'branch_ratio': branchpoints / max(skeleton_length, 1.0),
        'endpoint_chord_ratio': float(np.linalg.norm(endpoint_b - endpoint_a)) / max(span, 1.0),
        'axis_deviation_ratio': 0.0,
        'ellipse_axis_ratio': 999.0,
        'ellipse_residual': 1.0,
        'ellipse_coverage': 0.0,
        'circularity': 0.0,
        'axis_x': float(principal_axis[0]),
        'axis_y': float(principal_axis[1]),
        'center_x': float(center[0] - x),
        'center_y': float(center[1] - y),
        'endpoint_a_x': float(endpoint_a[0] - x),
        'endpoint_a_y': float(endpoint_a[1] - y),
        'endpoint_b_x': float(endpoint_b[0] - x),
        'endpoint_b_y': float(endpoint_b[1] - y),
    }
    metrics['score'] = _component_score(metrics, cfg)
    return CrackComponent(x, y, component_mask, skeleton, metrics, 'spatial_chain')


def _dominant_spatial_chain(
    components: list[CrackComponent],
    cfg: DetectionConfig,
) -> list[CrackComponent]:
    if len(components) < cfg.spatial_chain_min_components:
        return []

    candidates = [
        component
        for component in components
        if component.metrics.get('skeleton_length', 0.0) >= 8.0
        and component.metrics.get('span', 0.0) >= 8.0
        and component.metrics.get('elongation', 0.0) >= 1.35
        and component.metrics.get('average_thickness', 99.0) <= cfg.spatial_chain_max_thickness * 1.7
        and component.metrics.get('density', 99.0) <= 0.75
    ]
    candidates.sort(
        key=lambda component: (
            component.metrics.get('skeleton_length', 0.0)
            * max(component.metrics.get('strength_p90', 0.0), 1.0)
        ),
        reverse=True,
    )
    components = candidates[:96]
    if len(components) < cfg.spatial_chain_min_components:
        return []

    count = len(components)
    adjacency = [set() for _ in range(count)]
    for i in range(count):
        a = components[i].metrics
        a_points = np.array([
            [a['endpoint_a_x'] + components[i].x, a['endpoint_a_y'] + components[i].y],
            [a['endpoint_b_x'] + components[i].x, a['endpoint_b_y'] + components[i].y],
        ], dtype=np.float32)
        for j in range(i + 1, count):
            b = components[j].metrics
            b_points = np.array([
                [b['endpoint_a_x'] + components[j].x, b['endpoint_a_y'] + components[j].y],
                [b['endpoint_b_x'] + components[j].x, b['endpoint_b_y'] + components[j].y],
            ], dtype=np.float32)
            gap = float(np.min(np.linalg.norm(a_points[:, None, :] - b_points[None, :, :], axis=2)))
            if gap <= cfg.spatial_chain_max_gap:
                adjacency[i].add(j)
                adjacency[j].add(i)

    visited: set[int] = set()
    qualifying: list[tuple[float, list[CrackComponent]]] = []
    for start in range(count):
        if start in visited:
            continue
        stack = [start]
        group_indices: list[int] = []
        visited.add(start)
        while stack:
            current = stack.pop()
            group_indices.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        if len(group_indices) < cfg.spatial_chain_min_components:
            continue

        group = [components[index] for index in group_indices]
        points: list[list[float]] = []
        total_length = 0.0
        weighted_thickness = 0.0
        weighted_density = 0.0
        strengths: list[float] = []
        for component in group:
            metrics = component.metrics
            length = metrics['skeleton_length']
            total_length += length
            weighted_thickness += metrics['average_thickness'] * length
            weighted_density += metrics['density'] * length
            strengths.append(metrics['strength_p90'])
            points.extend([
                [metrics['endpoint_a_x'] + component.x, metrics['endpoint_a_y'] + component.y],
                [metrics['endpoint_b_x'] + component.x, metrics['endpoint_b_y'] + component.y],
                [metrics['center_x'] + component.x, metrics['center_y'] + component.y],
            ])
        if total_length <= 0:
            continue
        array = np.asarray(points, dtype=np.float32)
        centered = array - array.mean(axis=0, keepdims=True)
        covariance = np.cov(centered, rowvar=False) if len(array) >= 3 else np.eye(2)
        eigenvalues = np.sort(np.linalg.eigvalsh(covariance))
        elongation = float((eigenvalues[-1] + 1e-3) / (eigenvalues[0] + 1e-3))
        span = float(np.max(np.linalg.norm(array[:, None, :] - array[None, :, :], axis=2)))
        average_thickness = weighted_thickness / total_length
        average_density = weighted_density / total_length
        mean_strength = float(np.mean(strengths)) if strengths else 0.0
        if (
            total_length >= cfg.spatial_chain_min_total_length
            and span >= cfg.spatial_chain_min_span
            and elongation >= cfg.spatial_chain_min_elongation
            and average_thickness <= cfg.spatial_chain_max_thickness
            and average_density <= cfg.spatial_chain_max_density
            and mean_strength >= cfg.spatial_chain_min_strength
        ):
            score = total_length + span + elongation * 20.0
            qualifying.append((score, group))

    if not qualifying:
        return []
    return max(qualifying, key=lambda item: item[0])[1]


def _pale_surface_crack_components(
    image: np.ndarray,
    inner_mask: np.ndarray,
    rim_band: np.ndarray,
    cfg: DetectionConfig,
) -> tuple[list[CrackComponent], np.ndarray, np.ndarray]:
    """Detect whitish shell fractures that appear as pale low-saturation ridges."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lightness = lab[:, :, 0]
    saturation = hsv[:, :, 1]
    window = max(3, cfg.pale_surface_background_window | 1)
    lightness_background = cv2.medianBlur(lightness, window)
    saturation_background = cv2.medianBlur(saturation, window)
    bright_ridge = cv2.subtract(lightness, lightness_background)
    pale_ridge = cv2.subtract(saturation_background, saturation)
    pale_score = np.clip(
        bright_ridge.astype(np.float32) * cfg.pale_surface_lightness_weight
        + pale_ridge.astype(np.float32) * cfg.pale_surface_saturation_weight,
        0,
        255,
    ).astype(np.uint8)
    pale_score = cv2.bitwise_and(pale_score, pale_score, mask=inner_mask)
    response, _ = _directional_tophat_response(pale_score, inner_mask, cfg)
    values = response[inner_mask > 0]
    if values.size == 0:
        return [], response, np.zeros_like(inner_mask)

    threshold = int(max(
        cfg.pale_surface_min_threshold,
        np.percentile(values, cfg.pale_surface_percentile),
    ))
    strong = np.where(
        (response >= threshold) & (inner_mask > 0), 255, 0,
    ).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(strong, 8)
    components: list[CrackComponent] = []
    for index in range(1, count):
        pixels = int(stats[index, cv2.CC_STAT_AREA])
        if pixels < cfg.pale_surface_min_pixels:
            continue
        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        component_mask = np.where(
            labels[y:y + height, x:x + width] == index, 255, 0,
        ).astype(np.uint8)
        metrics, skeleton = _component_metrics(
            component_mask,
            response[y:y + height, x:x + width],
            strong[y:y + height, x:x + width],
            rim_band[y:y + height, x:x + width],
        )
        if not metrics:
            continue
        metrics['score'] = _component_score(metrics, cfg)
        if not (
            cfg.pale_surface_min_skeleton_length
            <= metrics['skeleton_length']
            <= cfg.pale_surface_max_skeleton_length
            and metrics['average_thickness']
            <= cfg.pale_surface_max_average_thickness
            and metrics['density'] <= cfg.pale_surface_max_density
            and metrics['branch_ratio'] <= cfg.pale_surface_max_branch_ratio
            and metrics['extent_ratio'] >= cfg.pale_surface_min_extent_ratio
            and metrics['axis_deviation_ratio']
            <= cfg.pale_surface_max_axis_deviation
            and metrics['strength_p90'] >= cfg.pale_surface_min_strength_p90
            and metrics['rim_overlap'] < cfg.rim_overlap_reject_ratio
        ):
            continue
        components.append(CrackComponent(
            x, y, component_mask, skeleton, metrics, 'pale_surface',
        ))

    if len(components) < cfg.pale_surface_min_components:
        return [], response, strong

    def global_point(component: CrackComponent, prefix: str) -> np.ndarray:
        return np.array([
            component.x + component.metrics[f'{prefix}_x'],
            component.y + component.metrics[f'{prefix}_y'],
        ], dtype=np.float32)

    def components_link(left: CrackComponent, right: CrackComponent) -> bool:
        left_axis = np.array([
            left.metrics['axis_x'], left.metrics['axis_y'],
        ], dtype=np.float32)
        right_axis = np.array([
            right.metrics['axis_x'], right.metrics['axis_y'],
        ], dtype=np.float32)
        axis_alignment = abs(float(np.dot(left_axis, right_axis)))
        if axis_alignment < cfg.pale_surface_min_axis_alignment:
            return False

        left_endpoints = (
            global_point(left, 'endpoint_a'),
            global_point(left, 'endpoint_b'),
        )
        right_endpoints = (
            global_point(right, 'endpoint_a'),
            global_point(right, 'endpoint_b'),
        )
        endpoint_gap = min(
            float(np.linalg.norm(left_point - right_point))
            for left_point in left_endpoints
            for right_point in right_endpoints
        )
        if endpoint_gap > cfg.pale_surface_max_group_gap:
            return False

        connector = global_point(right, 'center') - global_point(left, 'center')
        connector_length = float(np.linalg.norm(connector))
        if connector_length <= 1e-6:
            return False
        connector /= connector_length
        connector_alignment = max(
            abs(float(np.dot(connector, left_axis))),
            abs(float(np.dot(connector, right_axis))),
        )
        left_thickness = max(left.metrics['average_thickness'], 1e-6)
        right_thickness = max(right.metrics['average_thickness'], 1e-6)
        thickness_ratio = max(left_thickness, right_thickness) / min(
            left_thickness, right_thickness,
        )
        return bool(
            connector_alignment >= cfg.pale_surface_min_connector_alignment
            and thickness_ratio <= cfg.pale_surface_max_thickness_ratio
        )

    adjacency = [set() for _ in components]
    for left_index, left in enumerate(components):
        for right_index in range(left_index + 1, len(components)):
            if components_link(left, components[right_index]):
                adjacency[left_index].add(right_index)
                adjacency[right_index].add(left_index)

    groups: list[list[int]] = []
    visited: set[int] = set()
    for start in range(len(components)):
        if start in visited:
            continue
        stack = [start]
        group: list[int] = []
        visited.add(start)
        while stack:
            current = stack.pop()
            group.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        if len(group) >= cfg.pale_surface_min_components:
            groups.append(group)

    qualifying: list[tuple[float, list[CrackComponent]]] = []
    for group_indices in groups:
        group = [components[index] for index in group_indices]
        total_length = sum(
            component.metrics['skeleton_length'] for component in group
        )
        min_x = min(component.x for component in group)
        min_y = min(component.y for component in group)
        max_x = max(component.x + component.mask.shape[1] for component in group)
        max_y = max(component.y + component.mask.shape[0] for component in group)
        span = float(np.hypot(max_x - min_x, max_y - min_y))
        if (
            total_length >= cfg.pale_surface_min_total_length
            and span >= cfg.pale_surface_min_group_span
        ):
            qualifying.append((total_length + span, group))

    if not qualifying:
        return [], response, strong

    accepted: list[CrackComponent] = []
    for _, group in sorted(qualifying, key=lambda item: item[0], reverse=True):
        merged = _merge_component_group(
            group,
            inner_mask.shape,
            response,
            strong,
            rim_band,
            cfg,
        )
        if merged is not None:
            merged.source = 'pale_surface'
            accepted.append(merged)
        else:
            accepted.extend(group)
    return accepted, response, strong


def _overlay_line_sizes(image: np.ndarray) -> tuple[int, int]:
    """Scale overlay line thickness to the image size instead of using a
    fixed pixel width, which looks disproportionately thick on lower
    resolution webcam frames and disproportionately thin on 4K captures."""
    diagonal = float(np.hypot(*image.shape[:2]))
    contour_thickness = 1
    trace_dilation = 1 if diagonal < 1000.0 else 2
    return contour_thickness, trace_dilation


def _draw_trace_overlay(
    image: np.ndarray,
    trace_mask: np.ndarray,
    crack_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw crack regions as a thin 1px bright red line trace following the exact crack centerline."""
    overlay = image.copy()

    # Skeletonize to obtain exact 1px thin crack centerline trace
    skeleton = _skeletonize(crack_mask)

    # Draw exact 1px thin crack line in bright red (0, 0, 255)
    overlay[skeleton > 0] = (0, 0, 255)

    # Build a visible mask from the 1px thin crack trace
    visible = skeleton.copy()

    return overlay, visible


def _falling_membership(value: float, full_until: float, empty_at: float) -> float:
    if value <= full_until:
        return 1.0
    if value >= empty_at:
        return 0.0
    return float((empty_at - value) / max(empty_at - full_until, 1e-6))


def _rising_membership(value: float, empty_until: float, full_at: float) -> float:
    if value <= empty_until:
        return 0.0
    if value >= full_at:
        return 1.0
    return float((value - empty_until) / max(full_at - empty_until, 1e-6))


def _triangle_membership(value: float, left: float, peak: float, right: float) -> float:
    if value <= left or value >= right:
        return 0.0
    if value == peak:
        return 1.0
    if value < peak:
        return float((value - left) / max(peak - left, 1e-6))
    return float((right - value) / max(right - peak, 1e-6))


def _fuzzy_egg_size(
    egg_area_ratio: float,
    cfg: DetectionConfig,
    egg_width_ratio: float | None = None,
    egg_length_ratio: float | None = None,
) -> tuple[str, float, dict[str, float], float]:
    feature_memberships: list[tuple[float, dict[str, float]]] = [
        (
            0.70,
            {
                'small': _falling_membership(
                    egg_area_ratio,
                    cfg.egg_size_small_full_ratio,
                    cfg.egg_size_small_empty_ratio,
                ),
                'medium': _triangle_membership(
                    egg_area_ratio,
                    cfg.egg_size_medium_left_ratio,
                    cfg.egg_size_medium_peak_ratio,
                    cfg.egg_size_medium_right_ratio,
                ),
                'large': _rising_membership(
                    egg_area_ratio,
                    cfg.egg_size_large_empty_ratio,
                    cfg.egg_size_large_full_ratio,
                ),
            },
        ),
    ]

    if egg_width_ratio is not None:
        feature_memberships.append((
            0.15,
            {
                'small': _falling_membership(
                    egg_width_ratio,
                    cfg.egg_width_small_full_ratio,
                    cfg.egg_width_small_empty_ratio,
                ),
                'medium': _triangle_membership(
                    egg_width_ratio,
                    cfg.egg_width_medium_left_ratio,
                    cfg.egg_width_medium_peak_ratio,
                    cfg.egg_width_medium_right_ratio,
                ),
                'large': _rising_membership(
                    egg_width_ratio,
                    cfg.egg_width_large_empty_ratio,
                    cfg.egg_width_large_full_ratio,
                ),
            },
        ))

    if egg_length_ratio is not None:
        feature_memberships.append((
            0.15,
            {
                'small': _falling_membership(
                    egg_length_ratio,
                    cfg.egg_length_small_full_ratio,
                    cfg.egg_length_small_empty_ratio,
                ),
                'medium': _triangle_membership(
                    egg_length_ratio,
                    cfg.egg_length_medium_left_ratio,
                    cfg.egg_length_medium_peak_ratio,
                    cfg.egg_length_medium_right_ratio,
                ),
                'large': _rising_membership(
                    egg_length_ratio,
                    cfg.egg_length_large_empty_ratio,
                    cfg.egg_length_large_full_ratio,
                ),
            },
        ))

    total_weight = sum(weight for weight, _ in feature_memberships)
    raw = {label: 0.0 for label in ('small', 'medium', 'large')}
    for weight, memberships in feature_memberships:
        for label in raw:
            raw[label] += weight * memberships[label]
    raw = {label: value / max(total_weight, 1e-6) for label, value in raw.items()}

    total = sum(raw.values())
    if total <= 0:
        if egg_area_ratio < cfg.egg_size_medium_peak_ratio:
            normalized = {'small': 1.0, 'medium': 0.0, 'large': 0.0}
        elif egg_area_ratio < cfg.egg_size_large_full_ratio:
            normalized = {'small': 0.0, 'medium': 1.0, 'large': 0.0}
        else:
            normalized = {'small': 0.0, 'medium': 0.0, 'large': 1.0}
    else:
        normalized = {label: value / total for label, value in raw.items()}

    label = max(normalized, key=normalized.get)
    confidence = float(np.clip(normalized[label], 0.0, 1.0))
    size_score = (
        normalized['small'] * 0.20
        + normalized['medium'] * 0.55
        + normalized['large'] * 0.90
    )
    rounded_memberships = {
        key: round(float(np.clip(value, 0.0, 1.0)), 4)
        for key, value in normalized.items()
    }
    return (
        label,
        round(confidence, 4),
        rounded_memberships,
        round(float(np.clip(size_score, 0.0, 1.0)), 4),
    )


def _fuzzy_crack_size(
    is_crack: bool,
    traced_length: float,
    traced_pixels: int,
    egg_area: float,
    components: list[CrackComponent],
    strongest: float,
    cfg: DetectionConfig,
) -> tuple[str, float]:
    if not is_crack:
        return 'none', 1.0

    area_ratio = traced_pixels / max(egg_area, 1.0)
    length = float(np.clip(traced_length / cfg.fuzzy_length_scale, 0.0, 1.0))
    area = float(np.clip(area_ratio / cfg.fuzzy_area_scale, 0.0, 1.0))
    strength = float(np.clip(strongest / cfg.fuzzy_strength_scale, 0.0, 1.0))
    count = float(np.clip(len(components) / cfg.fuzzy_component_scale, 0.0, 1.0))

    memberships = {
        'small': max(
            min(_falling_membership(length, 0.08, 0.50), _falling_membership(area, 0.04, 0.18)),
            min(_falling_membership(length, 0.08, 0.50), _falling_membership(strength, 0.20, 0.60)),
            min(_falling_membership(area, 0.04, 0.18), _falling_membership(count, 0.15, 0.65)),
        ),
        'medium': max(
            min(_triangle_membership(length, 0.12, 0.42, 0.78), _triangle_membership(area, 0.05, 0.30, 0.72)),
            min(_triangle_membership(length, 0.12, 0.42, 0.78), _triangle_membership(strength, 0.18, 0.50, 0.88)),
            min(_triangle_membership(area, 0.05, 0.30, 0.72), _triangle_membership(count, 0.12, 0.42, 0.86)),
        ),
        'large': max(
            min(_rising_membership(length, 0.48, 0.90), _rising_membership(area, 0.42, 0.86)),
            min(_rising_membership(length, 0.48, 0.90), _rising_membership(strength, 0.55, 0.95)),
            min(_rising_membership(area, 0.42, 0.86), _rising_membership(count, 0.58, 1.0)),
        ),
    }
    total = sum(memberships.values())
    if total <= 0:
        label = 'large' if area >= 0.5 or length >= 0.55 else 'medium'
        return label, 0.34
    label = max(memberships, key=memberships.get)
    return label, round(float(np.clip(memberships[label] / total, 0.0, 1.0)), 4)


def fuzzy_area_consistency(
    area_ratios: list[float],
    cfg: DetectionConfig = CONFIG,
) -> tuple[bool, float, float, float]:
    if not area_ratios:
        return False, 0.0, 0.0, 1.0
    values = np.asarray(area_ratios, dtype=np.float32)
    mean = float(np.mean(values))
    median = float(np.median(values))
    spread = float((values.max() - values.min()) / max(abs(median), 1e-6))
    consistency = _falling_membership(
        spread,
        cfg.area_consistency_full_spread,
        cfg.area_consistency_max_spread,
    )
    return consistency >= 0.5, round(consistency, 4), mean, spread


def detect_image_bytes(
    data: bytes,
    include_steps: bool = False,
    cfg: DetectionConfig = CONFIG,
    *,
    _include_internal: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    original = _decode_input_image(data)

    image = _prepare_working_image(original, cfg)
    egg_mask, egg_contour, egg_data = _detect_egg(image, cfg)
    refined = _refine_working_image(original, image, egg_contour, cfg)
    if refined is not None:
        image = refined
        egg_mask, egg_contour, egg_data = _detect_egg(image, cfg)
    cfg = _scale_detection_config(cfg, egg_contour)
    inner_mask = _inner_egg_mask(egg_mask, cfg)
    quality = _capture_quality_metrics(
        image, egg_mask, inner_mask, egg_contour, cfg,
    )
    paper_mask, paper_metrics, paper_steps = _paper_method_crack_detection(
        image, egg_mask, inner_mask, cfg,
    )

    analysis, illumination, detail, detail_dark_preserved = _prepare_candling_image(
        image, egg_mask, cfg,
    )

    base_dark_response, base_bright_response, edge_response = _line_responses(
        analysis, detail, inner_mask, cfg,
    )
    (
        texture_dark_response,
        texture_bright_response,
        texture_response,
        texture_coherence,
    ) = _shell_texture_responses(detail, inner_mask, cfg)

    # LoG zero-crossing response: catches hairlines as sign changes in the
    # scale-normalised Laplacian. This is the 4th independent crack channel
    # alongside dark DoG, bright DoG, and Hessian texture ridges.
    log_response = _log_zero_crossing_response(detail, inner_mask, cfg)

    # Gradient magnitude response: the 5th channel. Detects strong light/dark
    # borders where the crack causes an abrupt brightness transition in the
    # candled image. This is the primary detector for cracks that show up as
    # step-edges rather than narrow dark/bright lines.
    grad_response = _gradient_magnitude_response(detail, inner_mask, cfg)

    # Dark valley response: the 6th channel.  Operates on the dark-preserved
    # detail (median-only smoothing) to catch faint dark cracks that the
    # bilateral filter would otherwise erase.
    dark_valley_resp = _dark_valley_response(
        detail_dark_preserved, inner_mask, cfg,
    )

    local_dark_response, local_bright_response = _local_hairline_responses(
        image, inner_mask, cfg,
    )
    (
        persistent_dark_mask,
        persistent_dark_response,
        persistent_dark_threshold,
    ) = _persistent_hairline_trace(
        _persistent_dark_response(image, inner_mask, cfg), inner_mask, cfg,
    )
    (
        persistent_bright_mask,
        persistent_bright_response,
        persistent_bright_threshold,
    ) = _persistent_hairline_trace(
        _persistent_blue_response(image, inner_mask, cfg), inner_mask, cfg,
    )

    dark_response = base_dark_response
    bright_response = base_bright_response

    dark_weak, dark_strong, dark_threshold = _response_masks(
        dark_response,
        inner_mask,
        cfg.dark_min_weak_threshold,
        cfg.dark_min_strong_threshold,
        cfg,
    )
    bright_weak, bright_strong, bright_threshold = _response_masks(
        bright_response,
        inner_mask,
        cfg.bright_min_weak_threshold,
        cfg.bright_min_strong_threshold,
        cfg,
    )
    texture_weak, texture_strong, texture_threshold = _response_masks(
        texture_response,
        inner_mask,
        cfg.texture_min_weak_threshold,
        cfg.texture_min_strong_threshold,
        cfg,
    )
    log_weak, log_strong, log_threshold = _response_masks(
        log_response,
        inner_mask,
        cfg.log_min_weak_threshold,
        cfg.log_min_strong_threshold,
        cfg,
    )
    grad_weak, grad_strong, grad_threshold = _response_masks(
        grad_response,
        inner_mask,
        cfg.grad_min_weak_threshold,
        cfg.grad_min_strong_threshold,
        cfg,
    )
    local_dark_weak, local_dark_strong, local_dark_threshold = _local_hairline_masks(
        local_dark_response, inner_mask, cfg,
    )
    local_bright_weak, local_bright_strong, local_bright_threshold = _local_hairline_masks(
        local_bright_response, inner_mask, cfg,
    )
    dark_candidates = _connect_small_gaps(dark_weak, inner_mask)
    bright_candidates = _connect_small_gaps(bright_weak, inner_mask)
    texture_candidates = _connect_small_gaps(texture_weak, inner_mask)
    log_candidates = _connect_small_gaps(log_weak, inner_mask)
    grad_candidates = _connect_small_gaps(grad_weak, inner_mask)
    local_dark_candidates = _connect_small_gaps(local_dark_weak, inner_mask)
    local_bright_candidates = _connect_small_gaps(local_bright_weak, inner_mask)
    dark_valley_weak, dark_valley_strong, dark_valley_threshold = _response_masks(
        dark_valley_resp,
        inner_mask,
        cfg.dark_valley_min_weak_threshold,
        cfg.dark_valley_min_strong_threshold,
        cfg,
        is_dark_channel=True,
    )
    dark_valley_candidates = _connect_small_gaps(dark_valley_weak, inner_mask)

    rim_band = _rim_band(inner_mask, cfg.rim_band_thickness)
    perimeter_zone, shell_depth = _perimeter_zone(
        egg_mask, inner_mask, cfg,
    )
    accepted_dark, _ = _extract_crack_components(
        dark_candidates,
        dark_response,
        dark_strong,
        rim_band,
        'dark',
        cfg,
    )
    accepted_bright, _ = _extract_crack_components(
        bright_candidates,
        bright_response,
        bright_strong,
        rim_band,
        'bright',
        cfg,
    )
    raw_texture_mask, texture_seed_components = _extract_crack_components(
        texture_candidates,
        texture_response,
        texture_strong,
        rim_band,
        'texture',
        cfg,
    )
    # LoG channel: treated like the dark channel — strong seeds only,
    # with a weak-pixel growth pass. It does not participate in the
    # texture-standalone path (which is reserved for Hessian hairlines).
    accepted_log, _ = _extract_crack_components(
        log_candidates,
        log_response,
        log_strong,
        rim_band,
        'log',
        cfg,
    )
    # Gradient magnitude channel: detects step-edge cracks (light/dark
    # borders). Treated like the dark channel with strong seed extraction.
    accepted_grad, _ = _extract_crack_components(
        grad_candidates,
        grad_response,
        grad_strong,
        rim_band,
        'grad',
        cfg,
    )
    # Dark valley channel: dedicated path for faint dark cracks that other
    # channels miss due to bilateral smoothing and high adaptive thresholds.
    accepted_dark_valley, _ = _extract_crack_components(
        dark_valley_candidates,
        dark_valley_resp,
        dark_valley_strong,
        rim_band,
        'dark_valley',
        cfg,
    )
    accepted_local_dark, local_dark_components = _extract_crack_components(
        local_dark_candidates,
        local_dark_response,
        local_dark_strong,
        rim_band,
        'local_dark',
        cfg,
    )
    accepted_local_bright, local_bright_components = _extract_crack_components(
        local_bright_candidates,
        local_bright_response,
        local_bright_strong,
        rim_band,
        'local_bright',
        cfg,
    )
    local_hairline_components = [
        component
        for component in local_dark_components + local_bright_components
        if _is_local_hairline_component(component, cfg)
    ]
    dominant_local_components = [
        component
        for component in local_hairline_components
        if _is_dominant_local_hairline_component(component, cfg)
    ]
    persistent_bright_components = [
        component
        for component in _mask_to_components(
            persistent_bright_mask, 'persistent_bright', cfg,
        )
        if _is_persistent_luminance_component(component, cfg)
    ]
    (
        pale_surface_components,
        pale_surface_response,
        pale_surface_strong,
    ) = _pale_surface_crack_components(
        image,
        inner_mask,
        rim_band,
        cfg,
    )
    trusted_local_mask = np.zeros_like(inner_mask)
    for component in local_hairline_components:
        _paste_component(trusted_local_mask, component)
    nearby_local_mask = cv2.dilate(
        trusted_local_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    persistent_dark_components = [
        component
        for component in _mask_to_components(
            persistent_dark_mask, 'persistent_dark', cfg,
        )
        if _is_crack_seed(component.metrics, cfg, 'persistent_dark')
        and _is_persistent_luminance_component(component, cfg)
        and _component_mask_overlap(component, nearby_local_mask) < 0.60
    ]
    persistent_bright_components = [
        component
        for component in persistent_bright_components
        if _component_mask_overlap(component, nearby_local_mask) < 0.60
    ]
    dominant_local_components.extend(persistent_dark_components)
    dominant_local_components.extend(persistent_bright_components)
    dominant_local_components.extend(pale_surface_components)
    for component in dominant_local_components:
        _annotate_perimeter_geometry(
            component, perimeter_zone, shell_depth, egg_mask,
        )
    dominant_local_components = [
        component
        for component in dominant_local_components
        if _has_valid_perimeter_geometry(component, shell_depth, cfg)
    ]
    for component in (
        persistent_dark_components
        + persistent_bright_components
        + pale_surface_components
    ):
        _paste_component(trusted_local_mask, component)
    accepted_texture = np.zeros_like(inner_mask)
    qualifying_texture_components = [
        component
        for component in texture_seed_components
        if _is_standalone_texture_component(component, cfg)
    ]
    base_seed_pixels = (
        cv2.countNonZero(accepted_dark)
        + cv2.countNonZero(accepted_bright)
        + cv2.countNonZero(accepted_log)
        + cv2.countNonZero(accepted_grad)
        + cv2.countNonZero(accepted_dark_valley)
        + cv2.countNonZero(trusted_local_mask)
    )
    if (
        base_seed_pixels == 0
        and 0 < len(qualifying_texture_components)
        <= cfg.texture_max_standalone_components
    ):
        texture_seed_mask = np.zeros_like(inner_mask)
        for component in qualifying_texture_components:
            _paste_component(texture_seed_mask, component)
        accepted_texture = texture_seed_mask.copy()
        growth_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        for _ in range(cfg.texture_support_radius):
            grown = cv2.bitwise_and(
                cv2.dilate(accepted_texture, growth_kernel),
                raw_texture_mask,
            )
            updated = cv2.bitwise_or(accepted_texture, grown)
            if cv2.countNonZero(updated) == cv2.countNonZero(accepted_texture):
                break
            accepted_texture = updated

    # A dark crack commonly creates two bright edge responses. Keep the dark
    # centerline and remove only bright pixels in its immediate neighborhood;
    # a separate bright light-leak crack remains available to the bright pass.
    dark_neighborhood = cv2.dilate(
        accepted_dark,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    unique_bright = cv2.bitwise_and(
        accepted_bright, cv2.bitwise_not(dark_neighborhood),
    )
    accepted_mask = cv2.bitwise_or(accepted_dark, unique_bright)
    accepted_mask = cv2.bitwise_or(accepted_mask, accepted_texture)
    # Merge LoG zero-crossing pixels.
    accepted_mask = cv2.bitwise_or(accepted_mask, accepted_log)
    # Merge gradient magnitude pixels — strong light/dark borders.
    accepted_mask = cv2.bitwise_or(accepted_mask, accepted_grad)
    # Merge dark valley pixels — faint dark cracks from the preserved detail.
    accepted_mask = cv2.bitwise_or(accepted_mask, accepted_dark_valley)
    accepted_mask = cv2.bitwise_or(accepted_mask, trusted_local_mask)
    local_response = cv2.max(local_dark_response, local_bright_response)
    local_strong = cv2.bitwise_or(local_dark_strong, local_bright_strong)
    combined_response = cv2.max(
        pale_surface_response,
        cv2.max(dark_response, bright_response),
    )
    combined_response = cv2.max(
        combined_response,
        cv2.max(
            cv2.max(texture_response, log_response),
            cv2.max(
                cv2.max(grad_response, local_response),
                dark_valley_resp,
            ),
        ),
    )
    combined_strong = cv2.bitwise_or(
        pale_surface_strong,
        cv2.bitwise_or(dark_strong, bright_strong),
    )
    combined_strong = cv2.bitwise_or(
        combined_strong,
        cv2.bitwise_or(
            cv2.bitwise_or(texture_strong, log_strong),
            cv2.bitwise_or(
                cv2.bitwise_or(grad_strong, local_strong),
                dark_valley_strong,
            ),
        ),
    )
    components = _deduplicate_components(
        accepted_mask, combined_response, combined_strong, rim_band, cfg,
    )
    _attach_texture_metrics(components, texture_response, texture_strong)
    raw_component_count = len(components)
    components = [
        component
        for component in components
        if _has_valid_trace_geometry(component.metrics, cfg)
    ]
    for component in components:
        _annotate_perimeter_geometry(
            component, perimeter_zone, shell_depth, egg_mask,
        )
    components = [
        component
        for component in components
        if _has_valid_perimeter_geometry(component, shell_depth, cfg)
    ]
    fragmented_candidates = raw_component_count >= cfg.max_fragmented_components
    dominant_override = False
    if dominant_local_components:
        # These components come from the untouched camera luminance. Preserve
        # them before the union of enhanced response maps can merge a genuine
        # one-pixel crack into a much thicker texture blob.
        components = dominant_local_components
        _attach_texture_metrics(components, texture_response, texture_strong)
        dominant_override = True
    elif fragmented_candidates:
        dominant_components = [
            component
            for component in components
            if _is_dominant_crack_component(component, cfg)
        ]
        if dominant_local_components:
            dominant_components.extend(dominant_local_components)
        if dominant_components:
            # Keep only the exceptional line components. The numerous small
            # shell-texture fragments must not enter the overlay or metrics.
            components = dominant_components
            dominant_override = True
        else:
            texture_supported_components = [
                component
                for component in components
                if component.metrics['skeleton_length']
                >= cfg.fragmented_texture_crack_min_skeleton_length
                and component.metrics['elongation']
                >= cfg.fragmented_texture_crack_min_elongation
                and component.metrics['average_thickness']
                <= cfg.fragmented_texture_crack_max_thickness
                and component.metrics['density']
                <= cfg.fragmented_texture_crack_max_density
                and component.metrics.get('texture_overlap', 0.0)
                >= cfg.fragmented_texture_crack_min_texture_overlap
                and component.metrics['score']
                >= cfg.fragmented_texture_crack_min_score
            ]
            if texture_supported_components:
                components = texture_supported_components
                dominant_override = True
            else:
                dominant_group = _dominant_fragment_group(components, cfg)
                if dominant_group:
                    merged_group = _merge_component_group(
                        dominant_group,
                        inner_mask.shape,
                        combined_response,
                        combined_strong,
                        rim_band,
                        cfg,
                    )
                    components = [merged_group] if merged_group is not None else dominant_group
                    if merged_group is not None:
                        _attach_texture_metrics(components, texture_response, texture_strong)
                    dominant_override = True
    validated_mask = np.zeros_like(inner_mask)
    for component in components:
        _paste_component(validated_mask, component)
    accepted_mask = validated_mask
    if paper_metrics['is_crack']:
        accepted_mask = cv2.bitwise_or(accepted_mask, paper_mask)
    consolidated_trace = cv2.morphologyEx(
        accepted_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    trace_mask = _skeletonize(consolidated_trace)
    traced_pixels = int(cv2.countNonZero(validated_mask))
    traced_length = float(cv2.countNonZero(trace_mask))
    total_length = float(sum(
        component.metrics['skeleton_length'] for component in components
    ))
    longest = float(max((
        component.metrics['skeleton_length'] for component in components
    ), default=0.0))
    strongest = float(max((
        component.metrics['strength_p90'] for component in components
    ), default=0.0))
    best_score = float(max((
        component.metrics['score'] for component in components
    ), default=0.0))
    if paper_metrics['is_crack']:
        traced_pixels = int(cv2.countNonZero(accepted_mask))
        traced_length = max(traced_length, float(cv2.countNonZero(trace_mask)))
        total_length = max(total_length, float(paper_metrics['total_length']))
        longest = max(longest, float(paper_metrics['longest_length']))
        strongest = max(strongest, 65.0)
        best_score = max(best_score, float(paper_metrics['score']) * 4.0)

    decision_score = 0.0
    if components:
        geometry_scale = max(cfg.geometry_scale, 1e-6)
        decision_score += min(longest / geometry_scale / 85.0, 1.8)
        decision_score += min(total_length / geometry_scale / 150.0, 1.2)
        decision_score += min(strongest / 55.0, 0.8)
        decision_score += min(best_score / 4.0, 1.0)
        decision_score += min(
            traced_pixels / (geometry_scale * geometry_scale) / 160.0,
            0.6,
        )

    thin_texture_components = [
        component for component in components
        if _is_thin_texture_crack(component, cfg)
    ]
    thin_crack_score = float(max((
        min(
            component.metrics['skeleton_length']
            / max(cfg.geometry_scale, 1e-6)
            / 45.0,
            1.0,
        ) * 0.28
        + min(
            component.metrics['span']
            / max(cfg.geometry_scale, 1e-6)
            / 45.0,
            1.0,
        ) * 0.20
        + min(component.metrics.get('texture_strength_p90', 0.0) / 90.0, 1.0) * 0.24
        + min(component.metrics.get('texture_overlap', 0.0), 1.0) * 0.18
        + min(component.metrics['elongation'] / 5.0, 1.0) * 0.10
        for component in thin_texture_components
    ), default=0.0))
    texture_like_fragmentation = fragmented_candidates and not dominant_override
    if texture_like_fragmentation:
        decision_score = min(decision_score, 0.35)
        thin_crack_score = 0.0

    normal_crack = bool(
        components
        and not texture_like_fragmentation
        and longest >= cfg.decision_min_longest
        and total_length >= cfg.decision_min_total_length
        and decision_score >= cfg.decision_min_score
    )
    thin_crack = bool(
        thin_texture_components
        and not texture_like_fragmentation
        and thin_crack_score >= cfg.thin_crack_min_score
    )
    component_crack = normal_crack or thin_crack
    paper_crack = bool(paper_metrics['is_crack'])
    is_crack = component_crack or paper_crack
    if component_crack:
        accepted_mask = validated_mask.copy()
        if paper_crack:
            accepted_mask = cv2.bitwise_or(accepted_mask, paper_mask)
            # Rebuild from the union so two channels tracing the same physical
            # fracture are reported as one camera location.
            components = _mask_to_components(accepted_mask, 'combined', cfg)
    elif paper_crack:
        accepted_mask = paper_mask.copy()
        components = _mask_to_components(paper_mask, 'paper', cfg)
    else:
        accepted_mask = np.zeros_like(paper_mask)
        components = []
    trace_mask = _skeletonize(accepted_mask)
    traced_pixels = int(cv2.countNonZero(accepted_mask))
    traced_length = float(cv2.countNonZero(trace_mask))

    quality_score = float(quality['quality_score'])
    if is_crack:
        confidence = float(np.clip(
            (0.56 + decision_score * 0.085) * (0.82 + quality_score * 0.18),
            0.58,
            0.95,
        ))
    else:
        confidence = float(np.clip(
            (0.90 - decision_score * 0.12) * (0.80 + quality_score * 0.20),
            0.52,
            0.92,
        ))
        accepted_mask[:] = 0
        trace_mask[:] = 0
        traced_pixels = 0
        traced_length = 0.0

    egg_area = max(float(cv2.contourArea(egg_contour)), 1.0)
    egg_area_ratio = egg_area / max(float(image.shape[0] * image.shape[1]), 1.0)
    if len(egg_contour) >= 5:
        fitted_axes = cv2.fitEllipse(egg_contour)[1]
        egg_width_pixels = float(min(fitted_axes))
        egg_length_pixels = float(max(fitted_axes))
    else:
        _, _, box_width, box_height = cv2.boundingRect(egg_contour)
        egg_width_pixels = float(min(box_width, box_height))
        egg_length_pixels = float(max(box_width, box_height))
    egg_width_ratio = egg_width_pixels / max(float(image.shape[1]), 1.0)
    egg_length_ratio = egg_length_pixels / max(float(image.shape[0]), 1.0)
    (
        egg_size,
        egg_size_confidence,
        egg_size_memberships,
        egg_size_score,
    ) = _fuzzy_egg_size(
        egg_area_ratio,
        cfg,
        egg_width_ratio=egg_width_ratio,
        egg_length_ratio=egg_length_ratio,
    )
    crack_size, crack_size_confidence = _fuzzy_crack_size(
        is_crack,
        traced_length,
        traced_pixels,
        egg_area,
        components,
        strongest,
        cfg,
    )

    overlay = image.copy()
    visible_trace = np.zeros_like(trace_mask)
    if is_crack:
        overlay, visible_trace = _draw_trace_overlay(image, trace_mask, accepted_mask)
    contour_thickness, _ = _overlay_line_sizes(image)
    cv2.drawContours(overlay, [egg_contour], -1, (0, 255, 0), contour_thickness, cv2.LINE_AA)

    inner_pixels = max(float(cv2.countNonZero(inner_mask)), 1.0)
    texture_candidate_pixels = int(cv2.countNonZero(texture_strong))
    texture_anomaly_ratio = texture_candidate_pixels / inner_pixels
    shell_texture_uniformity = float(np.clip(1.0 - texture_anomaly_ratio * 4.0, 0.0, 1.0))
    component_texture_score = float(max((
        component.metrics.get('texture_strength_p90', 0.0) / 255.0
        for component in components
    ), default=0.0))
    shell_texture_score = (
        max(component_texture_score, min(texture_anomaly_ratio * 4.0, 1.0))
        if is_crack else 0.0
    )

    result: dict[str, Any] = {
        'id': str(uuid.uuid4()),
        'is_crack': is_crack,
        'confidence': round(confidence, 4),
        'area_ratio': round(traced_pixels / egg_area, 6),
        'contour_length': round(traced_length, 2),
        'processing_time_ms': int((time.perf_counter() - started) * 1000),
        'original_image_b64': _encode_image(image),
        'overlay_image_b64': _encode_png_image(overlay),
        'intermediate_steps': None,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'egg_detected': True,
        'egg_score': round(float(egg_data['score']), 3),
        'egg_size': egg_size,
        'egg_size_confidence': egg_size_confidence,
        'egg_area_ratio': round(egg_area_ratio, 6),
        'egg_width_pixels': round(egg_width_pixels, 2),
        'egg_length_pixels': round(egg_length_pixels, 2),
        'egg_width_ratio': round(egg_width_ratio, 6),
        'egg_length_ratio': round(egg_length_ratio, 6),
        'egg_size_score': egg_size_score,
        'egg_size_memberships': egg_size_memberships,
        'crack_size': crack_size,
        'crack_size_confidence': crack_size_confidence,
        'candidate_components': len(components) if is_crack else 0,
        'raw_candidate_components': raw_component_count,
        'dominant_crack_override': dominant_override,
        'candidate_pixels': traced_pixels,
        'longest_candidate': round(longest, 2) if is_crack else 0.0,
        'mean_candidate_strength': round(float(np.mean([
            component.metrics['strength'] for component in components
        ])), 2) if is_crack and components else 0.0,
        'detection_score': round(decision_score, 3),
        'paper_method_used': True,
        'paper_method_crack': bool(paper_metrics['is_crack']),
        'paper_method_score': round(float(paper_metrics['score']), 4),
        'paper_method_components': int(paper_metrics['component_count']),
        'threshold_used': min(
            dark_threshold,
            bright_threshold,
            texture_threshold,
            log_threshold,
            int(round(persistent_dark_threshold)),
            int(round(persistent_bright_threshold)),
        ),
        'shell_texture_score': round(shell_texture_score, 4),
        'shell_texture_uniformity': round(shell_texture_uniformity, 4),
        'texture_anomaly_ratio': round(texture_anomaly_ratio, 6),
        'texture_candidate_pixels': texture_candidate_pixels,
        'thin_crack_score': round(thin_crack_score, 4),
        'thin_crack_detected': thin_crack,
        'image_quality_score': round(quality_score, 4),
        'image_sharpness': round(float(quality['boundary_sharpness']), 3),
        'image_detail_variance': round(float(quality['laplacian_variance']), 3),
        'image_saturated_ratio': round(float(quality['saturated_ratio']), 6),
        'image_glare_ratio': round(float(quality['glare_component_ratio']), 6),
        'image_dynamic_range': round(float(quality['dynamic_range']), 3),
        'requires_recapture': False,
        'quality_message': 'Image quality is suitable for detection',
        'sample_count': 1,
        'crack_votes': 1 if is_crack else 0,
        'no_crack_votes': 0 if is_crack else 1,
        'decision_consistency': 1.0,
        'area_consistent': True,
        'area_consistency': 1.0,
        'area_mean_ratio': round(traced_pixels / egg_area, 6),
        'area_spread_ratio': 0.0,
        'area_samples': [round(traced_pixels / egg_area, 6)],
        'crack_mask_b64': _encode_png_image(accepted_mask),
        'crack_locations': [
            _crack_location_from_component(component, index)
            for index, component in enumerate(components, start=1)
        ] if is_crack else [],
        'detection_iterations': len(components) if is_crack else 0,
        'search_iterations': 1,
        'termination_reason': 'single_pass_complete',
    }
    if _include_internal:
        result['_internal_crack_mask'] = accepted_mask.copy()
        result['_internal_trace_mask'] = trace_mask.copy()
        result['_internal_components'] = list(components)
        result['_internal_image'] = image.copy()
        result['_internal_egg_contour'] = egg_contour.copy()
        result['_internal_egg_mask'] = egg_mask.copy()
        result['_internal_inner_mask'] = inner_mask.copy()
        result['_internal_egg_area'] = float(egg_area)

    if include_steps:
        result['intermediate_steps'] = {
            'egg_mask': _encode_image(egg_mask),
            'inner_egg_mask': _encode_image(inner_mask),
            'perimeter_shell_zone': _encode_image(perimeter_zone),
            'candling_illumination': _encode_image(illumination),
            'candling_detail_image': _encode_image(detail),
            'dark_line_response': _encode_image(dark_response),
            'bright_line_response': _encode_image(bright_response),
            'paper_channel_response': _encode_image(edge_response),
            'shell_texture_dark_response': _encode_image(texture_dark_response),
            'shell_texture_bright_response': _encode_image(texture_bright_response),
            'shell_texture_anomaly_response': _encode_image(texture_response),
            'shell_texture_coherence': _encode_image(texture_coherence),
            'shell_texture_candidates': _encode_image(texture_candidates),
            'shell_texture_strong_candidates': _encode_image(texture_strong),
            'log_zero_crossing_response': _encode_image(log_response),
            'log_crack_candidates': _encode_image(log_candidates),
            'log_strong_candidates': _encode_image(log_strong),
            'dark_crack_candidates': _encode_image(dark_candidates),
            'bright_crack_candidates': _encode_image(bright_candidates),
            'dark_strong_candidates': _encode_image(dark_strong),
            'bright_strong_candidates': _encode_image(bright_strong),
            'persistent_dark_response': _encode_image(
                np.clip(persistent_dark_response, 0, 255).astype(np.uint8),
            ),
            'persistent_dark_trace': _encode_image(persistent_dark_mask),
            'persistent_bright_response': _encode_image(
                np.clip(persistent_bright_response, 0, 255).astype(np.uint8),
            ),
            'persistent_bright_trace': _encode_image(persistent_bright_mask),
            'pale_surface_response': _encode_image(pale_surface_response),
            'pale_surface_strong_candidates': _encode_image(
                pale_surface_strong,
            ),
            'accepted_dark_trace': _encode_image(accepted_dark),
            'accepted_bright_trace': _encode_image(accepted_bright),
            'accepted_texture_trace': _encode_image(accepted_texture),
            'accepted_log_trace': _encode_image(accepted_log),
            'dark_valley_response': _encode_image(dark_valley_resp),
            'accepted_dark_valley_trace': _encode_image(accepted_dark_valley),
            'accepted_crack_trace': _encode_image(accepted_mask),
            'fused_crack_response': _encode_image(combined_response),
            'full_crack_trace': _encode_image(trace_mask),
            'traced_crack_overlay_mask': _encode_image(visible_trace),
            'paper_red_blur': _encode_image(paper_steps['paper_red_blur']),
            'paper_binary_egg': _encode_image(paper_steps['paper_binary_egg']),
            'paper_green_roi': _encode_image(paper_steps['paper_green_roi']),
            'paper_edges': _encode_image(paper_steps['paper_edges']),
            'paper_morphology': _encode_image(paper_steps['paper_morphology']),
            'paper_subtraction': _encode_image(paper_steps['paper_subtraction']),
            'paper_binary_inverse': _encode_image(paper_steps['paper_binary_inverse']),
            'paper_subtraction_inverse': _encode_image(paper_steps['paper_subtraction_inverse']),
            'paper_unfiltered_polygon': _encode_image(paper_steps['paper_unfiltered_polygon']),
            'paper_filled_crack_polygon': _encode_image(paper_mask),
        }
    return result





def _crack_location_from_component(
    component: CrackComponent,
    iteration: int,
) -> dict[str, Any]:
    ys, xs = np.where(component.mask > 0)
    if xs.size == 0 or ys.size == 0:
        return {
            'iteration': iteration,
            'bounding_box': {'x': component.x, 'y': component.y, 'width': 0, 'height': 0},
            'center': {'x': component.x, 'y': component.y},
            'pixel_count': 0,
            'trace_length': 0.0,
            'score': 0.0,
        }

    min_x = int(xs.min()) + component.x
    max_x = int(xs.max()) + component.x
    min_y = int(ys.min()) + component.y
    max_y = int(ys.max()) + component.y
    center_x = float(xs.mean()) + component.x
    center_y = float(ys.mean()) + component.y
    return {
        'iteration': iteration,
        'bounding_box': {
            'x': min_x,
            'y': min_y,
            'width': max_x - min_x + 1,
            'height': max_y - min_y + 1,
        },
        'center': {
            'x': round(center_x, 2),
            'y': round(center_y, 2),
        },
        'pixel_count': int(xs.size),
        'trace_length': round(float(component.metrics.get('skeleton_length', 0.0)), 2),
        'span': round(float(component.metrics.get('span', 0.0)), 2),
        'score': round(float(component.metrics.get('score', 0.0)), 3),
        'source': component.source,
    }


def _remove_internal_values(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if not key.startswith('_internal_')
    }


def detect_iterative_image_bytes(
    data: bytes,
    include_steps: bool = False,
    cfg: DetectionConfig = CONFIG,
) -> dict[str, Any]:
    started = time.perf_counter()
    detected = detect_image_bytes(
        data,
        include_steps,
        cfg,
        _include_internal=True,
    )

    image = detected['_internal_image'].copy()
    egg_contour = detected['_internal_egg_contour']
    egg_area = max(float(detected['_internal_egg_area']), 1.0)
    components = list(detected.get('_internal_components', []))
    components.sort(
        key=lambda component: (
            component.metrics.get('score', 0.0),
            component.metrics.get('skeleton_length', 0.0),
            component.metrics.get('strength_p90', 0.0),
        ),
        reverse=True,
    )

    accumulated_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    excluded_mask = np.zeros_like(accumulated_mask)
    selected_components: list[CrackComponent] = []
    crack_locations: list[dict[str, Any]] = []
    primary_length = float(components[0].metrics.get('skeleton_length', 0.0)) if components else 0.0
    minimum_iteration_length = max(
        cfg.decision_min_longest,
        primary_length * cfg.iterative_min_relative_length,
    )
    remaining = [
        component
        for component in components
        if component.metrics.get('skeleton_length', 0.0) >= minimum_iteration_length
    ]
    search_iterations = 0

    while remaining and len(selected_components) < cfg.iterative_max_iterations:
        search_iterations += 1
        selected: CrackComponent | None = None
        selected_mask: np.ndarray | None = None
        next_remaining: list[CrackComponent] = []

        for component in remaining:
            component_global = np.zeros_like(accumulated_mask)
            _paste_component(component_global, component)
            new_mask = cv2.bitwise_and(
                component_global,
                cv2.bitwise_not(excluded_mask),
            )
            new_pixels = int(cv2.countNonZero(new_mask))
            if selected is None and new_pixels >= cfg.iterative_min_new_pixels:
                selected = component
                selected_mask = new_mask
                continue
            next_remaining.append(component)

        if selected is None or selected_mask is None:
            break

        selected_components.append(selected)
        crack_locations.append(
            _crack_location_from_component(
                selected,
                len(selected_components),
            ),
        )
        accumulated_mask = cv2.bitwise_or(accumulated_mask, selected_mask)

        kernel_size = max(3, cfg.iterative_exclusion_padding * 2 + 1)
        exclusion_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        excluded_mask = cv2.dilate(accumulated_mask, exclusion_kernel)
        remaining = next_remaining

    search_iterations += 1
    if not bool(detected['is_crack']) or not selected_components:
        final_result = _remove_internal_values(detected)
        final_result.update({
            'id': str(uuid.uuid4()),
            'processing_time_ms': int((time.perf_counter() - started) * 1000),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'crack_mask_b64': _encode_png_image(accumulated_mask),
            'crack_locations': [],
            'detection_iterations': 0,
            'search_iterations': search_iterations,
            'termination_reason': 'no_crack_found',
            'sample_count': 1,
            'crack_votes': 0,
            'no_crack_votes': 1,
            'decision_consistency': 1.0,
        })
        return final_result

    consolidated = cv2.morphologyEx(
        accumulated_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    trace_mask = _skeletonize(consolidated)
    traced_pixels = int(cv2.countNonZero(accumulated_mask))
    traced_length = float(cv2.countNonZero(trace_mask))
    total_component_length = float(sum(
        component.metrics.get('skeleton_length', 0.0)
        for component in selected_components
    ))
    longest = float(max((
        component.metrics.get('skeleton_length', 0.0)
        for component in selected_components
    ), default=0.0))
    strongest = float(max((
        component.metrics.get('strength_p90', 0.0)
        for component in selected_components
    ), default=0.0))
    thin_crack = any(
        _is_thin_texture_crack(component, cfg)
        for component in selected_components
    )

    crack_size, crack_size_confidence = _fuzzy_crack_size(
        True,
        max(traced_length, total_component_length),
        traced_pixels,
        egg_area,
        selected_components,
        strongest,
        cfg,
    )

    overlay, visible_trace = _draw_trace_overlay(
        image,
        trace_mask,
        accumulated_mask,
    )
    contour_thickness, _ = _overlay_line_sizes(image)
    cv2.drawContours(overlay, [egg_contour], -1, (0, 255, 0), contour_thickness, cv2.LINE_AA)

    final_result = _remove_internal_values(detected)
    final_result.update({
        'id': str(uuid.uuid4()),
        'is_crack': True,
        'area_ratio': round(traced_pixels / egg_area, 6),
        'contour_length': round(traced_length, 2),
        'processing_time_ms': int((time.perf_counter() - started) * 1000),
        'overlay_image_b64': _encode_png_image(overlay),
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'candidate_components': len(selected_components),
        'candidate_pixels': traced_pixels,
        'longest_candidate': round(longest, 2),
        'mean_candidate_strength': round(float(np.mean([
            component.metrics.get('strength', 0.0)
            for component in selected_components
        ])), 2),
        'thin_crack_detected': thin_crack,
        'crack_size': crack_size,
        'crack_size_confidence': crack_size_confidence,
        'crack_mask_b64': _encode_png_image(accumulated_mask),
        'crack_locations': crack_locations,
        'detection_iterations': len(selected_components),
        'search_iterations': search_iterations,
        'termination_reason': (
            'maximum_iterations_reached'
            if len(selected_components) >= cfg.iterative_max_iterations
            and remaining
            else 'no_more_cracks'
        ),
        'sample_count': 1,
        'crack_votes': 1,
        'no_crack_votes': 0,
        'decision_consistency': 1.0,
        'area_consistent': True,
        'area_consistency': 1.0,
        'area_mean_ratio': round(traced_pixels / egg_area, 6),
        'area_spread_ratio': 0.0,
        'area_samples': [round(traced_pixels / egg_area, 6)],
    })
    if include_steps:
        steps = dict(final_result.get('intermediate_steps') or {})
        steps['iterative_accumulated_crack_mask'] = _encode_image(accumulated_mask)
        steps['iterative_final_trace'] = _encode_image(trace_mask)
        steps['iterative_visible_overlay_mask'] = _encode_image(visible_trace)
        final_result['intermediate_steps'] = steps
    return final_result


def _correct_camera_orientation(data: bytes, cfg: DetectionConfig) -> bytes:
    mode = cfg.camera_orientation_fix
    if mode == 'none':
        return data

    image = _decode_input_image(data)
    if mode == 'rotate_180':
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif mode == 'flip_vertical':
        image = cv2.flip(image, 0)
    elif mode == 'flip_horizontal':
        image = cv2.flip(image, 1)
    else:
        raise DetectionError(f'Unknown camera_orientation_fix mode: {mode}')

    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise DetectionError('Could not correct camera image orientation')
    return encoded.tobytes()


def _camera_detection_config(cfg: DetectionConfig) -> DetectionConfig:
    if not cfg.camera_fast_mode:
        return cfg
    return replace(
        cfg,
        target_width=cfg.camera_target_width,
        target_height=cfg.camera_target_height,
        line_sigmas=(0.7, 1.2, 2.8),
        morphology_sizes=(3, 7),
        tophat_line_kernels=((1, 9), (9, 1), (1, 13), (13, 1)),
        log_sigmas=(0.8, 1.4),
        texture_ridge_sigmas=(0.65, 0.9, 1.25, 1.8),
        local_hairline_windows=(7, 11),
        iterative_max_iterations=1,
    )


def score_camera_focus_image_bytes(
    data: bytes,
    cfg: DetectionConfig = CONFIG,
) -> dict[str, float | bool]:
    """Locate the illuminated egg and score sharpness only inside its shell.

    The score is intended for comparing frames from the same manual-focus
    sweep. Brightness identifies the egg; Laplacian and gradient energy decide
    which lens position resolves the most shell detail.
    """
    corrected = _correct_camera_orientation(data, cfg)
    image = _prepare_working_image(
        _decode_input_image(corrected),
        _camera_detection_config(cfg),
    )
    egg_mask, _, egg_data = _detect_egg(image, cfg)
    inner_mask = _inner_egg_mask(egg_mask, cfg)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inner_pixels = inner_mask > 0
    inner_values = gray[inner_pixels]
    if inner_values.size < 100:
        raise DetectionError('The illuminated egg region is too small')

    # Suppress single-pixel sensor noise before measuring focus. Otherwise a
    # noisy but defocused bright frame can beat a genuinely sharp shell.
    focus_gray = cv2.GaussianBlur(gray, (3, 3), 0)
    laplacian = cv2.Laplacian(focus_gray, cv2.CV_64F, ksize=3)
    detail_variance = float(np.var(laplacian[inner_pixels]))
    gradient_x = cv2.Sobel(focus_gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(focus_gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gradient_x, gradient_y)
    texture_sharpness = float(np.percentile(gradient[inner_pixels], 90.0))

    # Log scaling keeps one high-contrast crack or reflection from dominating
    # the whole sweep while preserving the ordering between focus positions.
    focus_score = (
        np.log1p(max(detail_variance, 0.0)) * 0.72
        + np.log1p(max(texture_sharpness, 0.0)) * 0.28
    )
    median_brightness = float(np.percentile(inner_values, 50.0))
    p95_brightness = float(np.percentile(inner_values, 95.0))

    return {
        'egg_detected': True,
        'focus_score': round(float(focus_score), 6),
        'detail_variance': round(detail_variance, 3),
        'texture_sharpness': round(texture_sharpness, 3),
        'egg_brightness': round(median_brightness, 3),
        'egg_p95_brightness': round(p95_brightness, 3),
        'egg_area_ratio': round(float(egg_data.get('area_ratio', 0.0)), 6),
    }


def detect_camera_image_bytes(
    data: bytes,
    include_steps: bool = False,
    cfg: DetectionConfig = CONFIG,
    *,
    _include_internal: bool = False,
) -> dict[str, Any]:
    data = _correct_camera_orientation(data, cfg)
    camera_cfg = _camera_detection_config(cfg)
    if camera_cfg.camera_fast_mode:
        result = detect_image_bytes(
            data,
            include_steps,
            camera_cfg,
            _include_internal=_include_internal,
        )
        detection_count = len(result.get('crack_locations', []))
        result['termination_reason'] = 'no_more_cracks'
        result['detection_iterations'] = detection_count
        result['search_iterations'] = detection_count + 1
        result['sample_count'] = 1
        result['crack_votes'] = 1 if result.get('is_crack') else 0
        result['no_crack_votes'] = 0 if result.get('is_crack') else 1
        result['decision_consistency'] = 1.0
        return result
    return detect_iterative_image_bytes(data, include_steps, camera_cfg)


def _decode_crack_mask(result: dict[str, Any]) -> np.ndarray | None:
    encoded = result.get('crack_mask_b64')
    if not isinstance(encoded, str) or not encoded:
        return None
    mask = cv2.imdecode(
        np.frombuffer(base64.b64decode(encoded), dtype=np.uint8),
        cv2.IMREAD_GRAYSCALE,
    )
    if mask is None:
        return None
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _registered_egg_trace(
    result: dict[str, Any],
    cfg: DetectionConfig,
) -> np.ndarray | None:
    mask = result.get('_internal_crack_mask')
    contour = result.get('_internal_egg_contour')
    if not isinstance(mask, np.ndarray):
        mask = _decode_crack_mask(result)
    if mask is None:
        return None

    if not isinstance(contour, np.ndarray):
        encoded = result.get('original_image_b64')
        if not isinstance(encoded, str) or not encoded:
            return None
        image = cv2.imdecode(
            np.frombuffer(base64.b64decode(encoded), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if image is None:
            return None
        try:
            _, contour, _ = _detect_egg(image, cfg)
        except (DetectionError, cv2.error):
            return None

    target_width = cfg.multi_frame_registration_width
    target_height = cfg.multi_frame_registration_height
    target_center = np.array([
        target_width / 2.0,
        target_height / 2.0,
    ], dtype=np.float32)

    if len(contour) < 5:
        x, y, width, height = cv2.boundingRect(contour)
        if width <= 0 or height <= 0:
            return None
        crop = mask[y:y + height, x:x + width]
        return cv2.resize(
            crop,
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST,
        )

    center, axes, angle = cv2.fitEllipse(contour)
    axis_a, axis_b = float(axes[0]), float(axes[1])
    radians = np.deg2rad(angle)
    first_axis = np.array([
        np.cos(radians), np.sin(radians),
    ], dtype=np.float32)
    second_axis = np.array([
        -np.sin(radians), np.cos(radians),
    ], dtype=np.float32)
    if axis_a >= axis_b:
        major_axis = first_axis
        major_length = axis_a
        minor_length = axis_b
    else:
        major_axis = second_axis
        major_length = axis_b
        minor_length = axis_a

    if (
        abs(float(major_axis[1])) >= abs(float(major_axis[0]))
        and major_axis[1] < 0
    ) or (
        abs(float(major_axis[0])) > abs(float(major_axis[1]))
        and major_axis[0] < 0
    ):
        major_axis *= -1.0
    minor_axis = np.array([
        major_axis[1], -major_axis[0],
    ], dtype=np.float32)
    source_center = np.asarray(center, dtype=np.float32)
    source_points = np.asarray([
        source_center,
        source_center + minor_axis * (minor_length * 0.5),
        source_center + major_axis * (major_length * 0.5),
    ], dtype=np.float32)
    target_points = np.asarray([
        target_center,
        target_center + np.array([target_width * 0.45, 0.0], dtype=np.float32),
        target_center + np.array([0.0, target_height * 0.45], dtype=np.float32),
    ], dtype=np.float32)
    transform = cv2.getAffineTransform(source_points, target_points)
    registered = cv2.warpAffine(
        mask,
        transform,
        (target_width, target_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return np.where(registered > 0, 255, 0).astype(np.uint8)


def _clear_unconfirmed_crack(
    result: dict[str, Any],
    mask: np.ndarray | None,
    sample_count: int,
) -> dict[str, Any]:
    result = dict(result)
    if mask is None:
        image = cv2.imdecode(
            np.frombuffer(base64.b64decode(result['original_image_b64']), dtype=np.uint8),
            cv2.IMREAD_GRAYSCALE,
        )
        mask = np.zeros(image.shape, dtype=np.uint8) if image is not None else np.zeros((1, 1), dtype=np.uint8)
    result.update({
        'is_crack': False,
        'confidence': min(float(result['confidence']), 0.70),
        'area_ratio': 0.0,
        'contour_length': 0.0,
        'overlay_image_b64': result['original_image_b64'],
        'crack_size': 'none',
        'crack_size_confidence': 0.0,
        'crack_mask_b64': _encode_png_image(np.zeros_like(mask)),
        'crack_locations': [],
        'candidate_components': 0,
        'candidate_pixels': 0,
        'longest_candidate': 0.0,
        'mean_candidate_strength': 0.0,
        'thin_crack_detected': False,
        'thin_crack_score': 0.0,
        'detection_iterations': 0,
        'termination_reason': 'multi_frame_disagreement',
        'sample_count': sample_count,
        'crack_votes': 0,
        'no_crack_votes': sample_count,
        'decision_consistency': 0.0,
        'area_consistent': False,
        'area_consistency': 0.0,
        'area_mean_ratio': 0.0,
        'area_spread_ratio': 1.0,
        'area_samples': [],
        'quality_message': 'No stable crack trace was present across the captured frames',
    })
    return result


def detect_camera_images_bytes(
    frames: list[bytes],
    include_steps: bool = False,
    cfg: DetectionConfig = CONFIG,
) -> dict[str, Any]:
    """Validate camera detections using repeated, spatially matching traces."""
    started = time.perf_counter()
    minimum_samples = max(cfg.multi_frame_min_agreement, 2)
    if len(frames) < minimum_samples:
        raise DetectionError(f'Capture at least {minimum_samples} camera frames')

    def detect_frame(frame: bytes) -> dict[str, Any] | None:
        try:
            return detect_camera_image_bytes(
                frame,
                include_steps,
                cfg,
                _include_internal=True,
            )
        except (DetectionError, cv2.error):
            return None

    selected_frames = frames[:cfg.multi_frame_count]
    with ThreadPoolExecutor(max_workers=len(selected_frames)) as executor:
        frame_results = [
            result
            for result in executor.map(detect_frame, selected_frames)
            if result is not None
        ]
    if len(frame_results) < minimum_samples:
        raise DetectionError('Too few usable camera frames. Keep the egg still and capture again')

    candidates = [
        (index, result, _registered_egg_trace(result, cfg))
        for index, result in enumerate(frame_results)
        if result['is_crack']
    ]
    candidates = [candidate for candidate in candidates if candidate[2] is not None]
    stable_indices: set[int] = set()
    dilation_size = max(1, cfg.multi_frame_dilation | 1)
    dilation_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilation_size, dilation_size),
    )
    for left, (_, _, left_mask) in enumerate(candidates):
        for _, (_, _, right_mask) in enumerate(candidates[left + 1:], start=left + 1):
            if left_mask.shape != right_mask.shape:
                continue
            left_area = max(float(cv2.countNonZero(left_mask)), 1.0)
            right_area = max(float(cv2.countNonZero(right_mask)), 1.0)
            left_overlap = cv2.countNonZero(cv2.bitwise_and(
                left_mask, cv2.dilate(right_mask, dilation_kernel),
            )) / left_area
            right_overlap = cv2.countNonZero(cv2.bitwise_and(
                right_mask, cv2.dilate(left_mask, dilation_kernel),
            )) / right_area
            if min(left_overlap, right_overlap) >= cfg.multi_frame_min_overlap:
                stable_indices.update((candidates[left][0], candidates[left + 1][0]))

    sample_count = len(frame_results)
    stable_results = [frame_results[index] for index in stable_indices]
    if len(stable_results) < cfg.multi_frame_min_agreement:
        representative = max(
            frame_results,
            key=lambda result: (result['image_quality_score'], -result['detection_score']),
        )
        result = _clear_unconfirmed_crack(
            representative,
            _decode_crack_mask(representative),
            sample_count,
        )
        result['processing_time_ms'] = int(
            (time.perf_counter() - started) * 1000,
        )
        return _remove_internal_values(result)

    representative = max(
        stable_results,
        key=lambda result: (result['image_quality_score'], result['detection_score']),
    )
    area_samples = [float(result['area_ratio']) for result in stable_results]
    area_consistent, area_consistency, area_mean, area_spread = fuzzy_area_consistency(
        area_samples, cfg,
    )
    crack_votes = len(stable_results)
    result = dict(representative)
    result.update({
        'processing_time_ms': int((time.perf_counter() - started) * 1000),
        'sample_count': sample_count,
        'crack_votes': crack_votes,
        'no_crack_votes': sample_count - crack_votes,
        'decision_consistency': round(crack_votes / sample_count, 4),
        'area_consistent': area_consistent,
        'area_consistency': area_consistency,
        'area_mean_ratio': round(area_mean, 6),
        'area_spread_ratio': round(area_spread, 6),
        'area_samples': [round(area, 6) for area in area_samples],
        'quality_message': 'Crack trace validated across multiple camera frames',
    })
    return _remove_internal_values(result)
