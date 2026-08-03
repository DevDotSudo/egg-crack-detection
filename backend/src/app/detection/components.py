import math

import cv2
import numpy as np

from app.detection.config import DetectionConfig
from app.detection.models import ComponentFeatures, CrackPolarity, EggRegion, EnhancementResult


class ComponentAnalyzer:
    def __init__(self, config: DetectionConfig) -> None:
        self.config = config

    def _extract_one(
        self,
        label: int,
        component_mask: np.ndarray,
        polarity: CrackPolarity,
        response: np.ndarray,
        enhancement: EnhancementResult,
        egg: EggRegion,
    ) -> ComponentFeatures:
        points = cv2.findNonZero(component_mask)
        x, y, width, height = cv2.boundingRect(points)
        area = int(cv2.countNonZero(component_mask))
        span = float(np.hypot(width, height))
        density = area / max(float(width * height), 1.0)
        coordinates_yx = np.column_stack(np.where(component_mask > 0)).astype(np.float32)
        coordinates_xy = coordinates_yx[:, ::-1]
        center_xy = coordinates_xy.mean(axis=0)
        if len(coordinates_xy) >= 2:
            covariance = np.cov(coordinates_xy, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            axis = eigenvectors[:, -1].astype(np.float32)
            axis /= max(float(np.linalg.norm(axis)), 1e-6)
            perpendicular = np.array([-axis[1], axis[0]], dtype=np.float32)
            elongation = float(np.sqrt(max(eigenvalues[-1], 1e-6) / max(eigenvalues[0], 1e-6)))
        else:
            axis = np.array([1.0, 0.0], dtype=np.float32)
            perpendicular = np.array([0.0, 1.0], dtype=np.float32)
            elongation = 1.0
        centered = coordinates_xy - center_xy
        along = centered @ axis
        across = centered @ perpendicular
        bins = np.rint(along).astype(np.int32)
        unique_bins = np.unique(bins)
        centerline_points: list[np.ndarray] = []
        cross_ranges: list[float] = []
        cross_medians: list[float] = []
        for value in unique_bins:
            values = across[bins == value]
            median = float(np.median(values))
            cross_medians.append(median)
            cross_ranges.append(float(values.max() - values.min() + 1.0))
            centerline_points.append(center_xy + axis * float(value) + perpendicular * median)
        if len(centerline_points) >= 2:
            centerline = np.vstack(centerline_points)
            segment_lengths = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
            skeleton_length = int(round(float(segment_lengths.sum())))
        else:
            skeleton_length = max(1, len(centerline_points))
        mean_thickness = area / max(float(skeleton_length), 1.0)
        distance = cv2.distanceTransform(component_mask, cv2.DIST_L2, 5)
        maximum_thickness = float(distance.max() * 2.0)
        medians = np.asarray(cross_medians, dtype=np.float32)
        if medians.size >= 7:
            smooth = cv2.GaussianBlur(medians.reshape(-1, 1), (1, 5), 0).reshape(-1)
            second = np.abs(np.diff(smooth, n=2))
            roughness = float(np.percentile(second, 75) / max(mean_thickness * 3.0, 1.0)) if second.size else 0.0
        else:
            roughness = 0.0
        branch_count = int(np.count_nonzero(np.asarray(cross_ranges) > max(mean_thickness * 2.4, 4.0)))
        endpoint_count = 2 if skeleton_length > 1 else 0
        component_pixels = component_mask > 0
        values = response[component_pixels]
        mean_response = float(np.mean(values) / 255.0) if values.size else 0.0
        response_p90 = float(np.percentile(values, 90) / 255.0) if values.size else 0.0
        edge_values = enhancement.edge_response[component_pixels]
        edge_support = float(np.mean(edge_values >= 24)) if edge_values.size else 0.0
        glare_overlap = float(np.mean(enhancement.glare_mask[component_pixels] > 0)) if values.size else 0.0
        rim_overlap = float(np.mean(egg.rim_mask[component_pixels] > 0)) if values.size else 0.0
        return ComponentFeatures(
            label=label,
            polarity=polarity,
            mask=component_mask,
            bbox=(x, y, width, height),
            area=area,
            span=span,
            skeleton_length=skeleton_length,
            mean_thickness=mean_thickness,
            maximum_thickness=maximum_thickness,
            elongation=elongation,
            density=density,
            edge_support=edge_support,
            glare_overlap=glare_overlap,
            rim_overlap=rim_overlap,
            mean_response=mean_response,
            response_p90=response_p90,
            roughness=roughness,
            endpoint_count=endpoint_count,
            branch_count=branch_count,
            axis=(float(axis[0]), float(axis[1])),
            center=(float(center_xy[0]), float(center_xy[1])),
        )

    def analyze_mask(
        self,
        mask: np.ndarray,
        polarity: CrackPolarity,
        response: np.ndarray,
        enhancement: EnhancementResult,
        egg: EggRegion,
        label: int = -1,
    ) -> ComponentFeatures:
        binary = np.where(mask > 0, 255, 0).astype(np.uint8)
        if cv2.countNonZero(binary) == 0:
            raise ValueError('Cannot analyze an empty component mask')
        return self._extract_one(label, binary, polarity, response, enhancement, egg)

    def extract(
        self,
        mask: np.ndarray,
        polarity: CrackPolarity,
        response: np.ndarray,
        enhancement: EnhancementResult,
        egg: EggRegion,
    ) -> list[ComponentFeatures]:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        ranked: list[tuple[float, int]] = []
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self.config.components.minimum_area:
                continue
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            span = float(np.hypot(width, height))
            ranked.append((span * math.sqrt(max(area, 1)), label))
        ranked.sort(reverse=True)
        if len(ranked) > 64:
            ranked = ranked[:64]
        output: list[ComponentFeatures] = []
        for _, label in ranked:
            component_mask = np.where(labels == label, 255, 0).astype(np.uint8)
            output.append(self._extract_one(label, component_mask, polarity, response, enhancement, egg))
        return output


class ArtifactRejector:
    def __init__(self, config: DetectionConfig) -> None:
        self.config = config

    def classify(self, component: ComponentFeatures, egg: EggRegion) -> ComponentFeatures:
        c = self.config.components
        reasons: list[str] = []
        minimum_span = egg.minor_axis * c.minimum_span_ratio
        minimum_length = egg.minor_axis * c.minimum_length_ratio
        maximum_mean_thickness = egg.minor_axis * c.maximum_mean_thickness_ratio
        maximum_thickness = egg.minor_axis * c.maximum_thickness_ratio
        if component.span < minimum_span:
            reasons.append('short_span')
        if component.skeleton_length < minimum_length:
            reasons.append('short_length')
        if component.mean_thickness > maximum_mean_thickness:
            reasons.append('broad_component')
        if component.maximum_thickness > maximum_thickness:
            reasons.append('excessive_width')
        if component.density > c.maximum_density:
            reasons.append('blob_like')
        if component.elongation < c.minimum_elongation and component.branch_count < 2:
            reasons.append('low_elongation')
        narrow_bright = (
            component.polarity == CrackPolarity.BRIGHT
            and component.mean_thickness <= egg.minor_axis * 0.014
            and component.elongation >= 3.0
        )
        if component.glare_overlap > c.maximum_glare_overlap and not narrow_bright:
            reasons.append('glare_overlap')
        radial_vector = np.array([
            component.center[0] - egg.center[0],
            component.center[1] - egg.center[1],
        ], dtype=np.float32)
        radial_norm = float(np.linalg.norm(radial_vector))
        if radial_norm > 1e-6:
            radial_vector /= radial_norm
        component_axis = np.array(component.axis, dtype=np.float32)
        radial_alignment = float(abs(np.dot(component_axis, radial_vector))) if radial_norm > 1e-6 else 0.0
        inward_fraction = max(0.0, 1.0 - component.rim_overlap)
        rim_thin = component.mean_thickness <= egg.minor_axis * c.rim_maximum_mean_thickness_ratio
        rim_long = component.skeleton_length >= egg.minor_axis * c.rim_minimum_length_ratio
        rim_enters_shell = inward_fraction >= c.rim_minimum_inward_fraction
        rim_radial = radial_alignment >= c.rim_minimum_radial_alignment
        near_rim = component.rim_overlap >= 0.20
        rim_crack_supported = near_rim and rim_thin and rim_long and (rim_enters_shell or rim_radial)
        if component.rim_overlap > c.maximum_rim_overlap and not rim_crack_supported:
            reasons.append('rim_artifact')
        if component.response_p90 < c.minimum_response:
            reasons.append('weak_response')
        if component.edge_support < c.minimum_edge_support:
            reasons.append('weak_edge_support')
        smooth_single = (
            component.polarity == CrackPolarity.DARK
            and component.endpoint_count <= 2
            and component.roughness < c.smooth_line_maximum_roughness
            and component.mean_thickness > egg.minor_axis * 0.009
            and component.skeleton_length < egg.minor_axis * 0.95
        )
        if smooth_single:
            reasons.append('smooth_shell_mark')
        short_smooth_bright = (
            component.polarity == CrackPolarity.BRIGHT
            and component.skeleton_length < egg.minor_axis * 0.25
            and component.endpoint_count <= 2
            and component.roughness < 0.025
        )
        if short_smooth_bright and not rim_crack_supported:
            reasons.append('smooth_bright_arc')
        length_score = min(1.0, component.skeleton_length / max(egg.minor_axis * c.preferred_length_ratio, 1.0))
        thin_limit = max(maximum_mean_thickness, 1.0)
        thin_score = max(0.0, 1.0 - component.mean_thickness / thin_limit)
        shape_score = min(1.0, max(0.0, (component.elongation - 1.0) / 5.0))
        roughness_score = min(1.0, component.roughness / max(c.minimum_roughness * 3.0, 1e-6))
        branch_score = min(1.0, component.branch_count / 6.0)
        rim_score = 1.0 if rim_crack_supported else max(0.0, 1.0 - component.rim_overlap / max(c.maximum_rim_overlap, 1e-6))
        rim_entry_score = 1.0 if rim_crack_supported else 0.0
        score = (
            0.21 * component.response_p90
            + 0.18 * length_score
            + 0.14 * thin_score
            + 0.13 * component.edge_support
            + 0.10 * shape_score
            + 0.10 * max(roughness_score, branch_score)
            + 0.07 * (1.0 - min(component.density / max(c.maximum_density, 1e-6), 1.0))
            + 0.04 * rim_score
            + 0.03 * (1.0 - component.glare_overlap)
            + 0.06 * rim_entry_score
        )
        hard_failures = {
            'short_span',
            'short_length',
            'broad_component',
            'excessive_width',
            'blob_like',
            'glare_overlap',
            'weak_response',
        }
        accepted = score >= c.acceptance_score and not any(reason in hard_failures for reason in reasons)
        if 'smooth_shell_mark' in reasons:
            accepted = False
        if 'smooth_bright_arc' in reasons:
            accepted = False
        if 'rim_artifact' in reasons:
            accepted = False
        component.accepted = accepted
        component.score = float(score)
        component.reasons = tuple(reasons)
        return component


    def is_dominant_survivor(self, component: ComponentFeatures, egg: EggRegion) -> bool:
        c = self.config.components
        narrow_bright = (
            component.polarity == CrackPolarity.BRIGHT
            and component.mean_thickness <= egg.minor_axis * 0.014
            and component.elongation >= 3.0
        )
        glare_is_safe = component.glare_overlap < c.maximum_glare_overlap or narrow_bright
        radial_vector = np.array([
            component.center[0] - egg.center[0],
            component.center[1] - egg.center[1],
        ], dtype=np.float32)
        radial_norm = float(np.linalg.norm(radial_vector))
        radial_alignment = 0.0
        if radial_norm > 1e-6:
            radial_vector /= radial_norm
            radial_alignment = float(abs(np.dot(np.array(component.axis, dtype=np.float32), radial_vector)))
        rim_supported = (
            component.rim_overlap >= 0.20
            and component.mean_thickness <= egg.minor_axis * c.rim_maximum_mean_thickness_ratio
            and component.skeleton_length >= egg.minor_axis * c.rim_minimum_length_ratio
            and (
                1.0 - component.rim_overlap >= c.rim_minimum_inward_fraction
                or radial_alignment >= c.rim_minimum_radial_alignment
            )
        )
        dark_shape_is_crack_like = (
            component.polarity != CrackPolarity.DARK
            or component.roughness >= c.minimum_roughness
            or component.branch_count >= 2
        )
        return (
            dark_shape_is_crack_like
            and component.score >= max(c.acceptance_score + 0.18, 0.68)
            and component.skeleton_length >= egg.minor_axis * 0.30
            and (
                component.span >= egg.minor_axis * 0.25
                or (
                    component.span >= egg.minor_axis * 0.20
                    and component.mean_thickness <= egg.minor_axis * 0.010
                    and component.density <= 0.16
                )
            )
            and component.elongation >= 5.0
            and component.mean_thickness <= egg.minor_axis * 0.022
            and component.density <= 0.30
            and component.edge_support >= 0.45
            and component.response_p90 >= 0.35
            and (component.rim_overlap < 0.40 or rim_supported)
            and glare_is_safe
        )


class FragmentGrouper:
    def __init__(self, config: DetectionConfig) -> None:
        self.config = config

    @staticmethod
    def _alignment(first: ComponentFeatures, second: ComponentFeatures) -> float:
        first_axis = np.array(first.axis, dtype=np.float32)
        second_axis = np.array(second.axis, dtype=np.float32)
        return float(abs(np.dot(first_axis, second_axis)))

    @staticmethod
    def _gap(first: ComponentFeatures, second: ComponentFeatures) -> float:
        return float(np.hypot(first.center[0] - second.center[0], first.center[1] - second.center[1]) - 0.5 * first.span - 0.5 * second.span)

    def groups(self, components: list[ComponentFeatures], egg: EggRegion) -> list[list[ComponentFeatures]]:
        if len(components) < 2:
            return []
        c = self.config.components
        maximum_gap = egg.minor_axis * c.group_maximum_gap_ratio
        unused = set(range(len(components)))
        groups: list[list[ComponentFeatures]] = []
        while unused:
            seed = unused.pop()
            group = {seed}
            changed = True
            while changed:
                changed = False
                for index in list(unused):
                    if any(
                        self._gap(components[index], components[other]) <= maximum_gap
                        and self._alignment(components[index], components[other]) >= c.group_minimum_alignment
                        for other in group
                    ):
                        group.add(index)
                        unused.remove(index)
                        changed = True
            if len(group) >= 2:
                groups.append([components[index] for index in group])
        return groups

    def accept_group(self, group: list[ComponentFeatures], egg: EggRegion) -> bool:
        total_length = sum(component.skeleton_length for component in group)
        total_response = sum(component.response_p90 * component.skeleton_length for component in group) / max(total_length, 1)
        centers = np.array([component.center for component in group], dtype=np.float32)
        spread = float(np.hypot(*(centers.max(axis=0) - centers.min(axis=0)))) if len(centers) > 1 else 0.0
        average_glare = float(np.mean([component.glare_overlap for component in group]))
        return (
            total_length >= egg.minor_axis * self.config.components.group_minimum_length_ratio
            and spread >= egg.minor_axis * 0.04
            and total_response >= self.config.components.minimum_response
            and average_glare < self.config.components.maximum_glare_overlap
        )


def deduplicate_components(components: list[ComponentFeatures]) -> list[ComponentFeatures]:
    ordered = sorted(components, key=lambda value: (value.accepted, value.score, value.skeleton_length), reverse=True)
    kept: list[ComponentFeatures] = []
    for component in ordered:
        duplicate = False
        for existing in kept:
            overlap = cv2.countNonZero(cv2.bitwise_and(component.mask, existing.mask))
            smaller = max(1, min(component.area, existing.area))
            if overlap / smaller >= 0.35:
                duplicate = True
                break
        if not duplicate:
            kept.append(component)
    return kept


def merge_group_mask(group: list[ComponentFeatures]) -> np.ndarray:
    mask = np.zeros_like(group[0].mask)
    for component in group:
        mask = cv2.bitwise_or(mask, component.mask)
    return mask
