import time

import cv2
import numpy as np

from app.detection.components import (
    ArtifactRejector,
    ComponentAnalyzer,
    FragmentGrouper,
    deduplicate_components,
)
from app.detection.config import DetectionConfig
from app.detection.models import ComponentFeatures, CrackPolarity, PipelineResult
from app.detection.paper_baseline import PaperBaselineDetector
from app.detection.preprocessing import DualPolarityEnhancer, QualityAssessor
from app.detection.segmentation import EggSegmenter
from app.detection.thresholding import ResponseThreshold, connect_small_gaps


class EggCrackPipeline:
    def __init__(self, config: DetectionConfig) -> None:
        self.config = config
        self.segmenter = EggSegmenter(config)
        self.quality_assessor = QualityAssessor(config)
        self.enhancer = DualPolarityEnhancer(config)
        self.threshold = ResponseThreshold(config)
        self.analyzer = ComponentAnalyzer(config)
        self.rejector = ArtifactRejector(config)
        self.grouper = FragmentGrouper(config)
        self.paper = PaperBaselineDetector()

    def _working_image(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        scale = min(
            1.0,
            self.config.target_width / max(float(width), 1.0),
            self.config.target_height / max(float(height), 1.0),
        )
        if scale >= 1.0:
            return image.copy()
        target = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        return cv2.resize(image, target, interpolation=cv2.INTER_AREA)

    def _whole_egg_detection_mask(self, full_mask: np.ndarray) -> np.ndarray:
        pixels = max(0, int(self.config.segmentation.detection_border_pixels))
        if pixels == 0:
            return full_mask.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.erode(full_mask, kernel, iterations=pixels)
        if cv2.countNonZero(mask) < self.config.min_inner_pixels:
            return full_mask.copy()
        return mask

    @staticmethod
    def _overlap_ratio(first: ComponentFeatures, second: ComponentFeatures) -> float:
        overlap = cv2.countNonZero(cv2.bitwise_and(first.mask, second.mask))
        smaller = max(1, min(first.area, second.area))
        return overlap / float(smaller)

    def _texture_survivor(self, component: ComponentFeatures, egg) -> bool:
        c = self.config.components
        return (
            component.skeleton_length >= egg.minor_axis * 0.60
            and component.span >= egg.minor_axis * 0.35
            and component.elongation >= 3.0
            and component.mean_thickness <= egg.minor_axis * 0.009
            and component.density <= 0.18
            and component.response_p90 >= c.minimum_response
            and component.glare_overlap <= c.maximum_glare_overlap
            and 'rim_artifact' not in component.reasons
        )

    def _keep_multiple_texture_survivors(
        self,
        components: list[ComponentFeatures],
        egg,
    ) -> None:
        c = self.config.components
        if len(components) <= c.texture_overload_threshold:
            return
        survivors = [
            component
            for component in components
            if self._texture_survivor(component, egg)
            or self.rejector.is_dominant_survivor(component, egg)
        ]
        survivors.sort(
            key=lambda component: (
                component.score,
                component.edge_support,
                component.response_p90,
                component.skeleton_length,
            ),
            reverse=True,
        )
        for component in components:
            component.accepted = False
        kept: list[ComponentFeatures] = []
        for component in survivors:
            if any(self._overlap_ratio(component, existing) >= 0.35 for existing in kept):
                continue
            component.accepted = True
            component.score = max(component.score, c.acceptance_score + 0.08)
            component.reasons = tuple(
                reason
                for reason in component.reasons
                if reason not in {'smooth_shell_mark', 'smooth_bright_arc'}
            ) + ('dominant_texture_survivor',)
            kept.append(component)
            if len(kept) >= c.maximum_accepted_components:
                break

    def _paper_component_candidates(
        self,
        paper_mask: np.ndarray,
        enhancement,
        egg,
    ) -> list[ComponentFeatures]:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (paper_mask > 0).astype(np.uint8),
            connectivity=8,
        )
        output: list[ComponentFeatures] = []
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self.config.components.minimum_area:
                continue
            mask = np.where(labels == label, 255, 0).astype(np.uint8)
            pixels = mask > 0
            bright_mean = float(np.mean(enhancement.bright_response[pixels])) if np.any(pixels) else 0.0
            dark_mean = float(np.mean(enhancement.dark_response[pixels])) if np.any(pixels) else 0.0
            polarity = CrackPolarity.BRIGHT if bright_mean >= dark_mean else CrackPolarity.DARK
            response = enhancement.bright_response if polarity == CrackPolarity.BRIGHT else enhancement.dark_response
            component = self.analyzer.analyze_mask(
                mask,
                polarity,
                response,
                enhancement,
                egg,
                label=100000 + label,
            )
            output.append(self.rejector.classify(component, egg))
        output.sort(
            key=lambda component: (
                component.skeleton_length * max(component.score, 0.01),
                component.span,
                component.area,
            ),
            reverse=True,
        )
        return output

    def _recovery_shape_is_valid(self, component: ComponentFeatures, egg) -> bool:
        c = self.config.components
        return (
            component.skeleton_length >= egg.minor_axis * c.recovery_minimum_length_ratio
            and component.span >= egg.minor_axis * c.recovery_minimum_span_ratio
            and component.mean_thickness <= egg.minor_axis * c.recovery_maximum_mean_thickness_ratio
            and component.density <= c.recovery_maximum_density
            and component.edge_support >= c.recovery_minimum_edge_support
            and component.response_p90 >= c.recovery_minimum_response
            and component.glare_overlap <= max(c.maximum_glare_overlap, 0.25)
            and component.rim_overlap <= 0.88
        )

    def _recover_full_traces(
        self,
        classified: list[ComponentFeatures],
        paper_mask: np.ndarray,
        enhancement,
        egg,
        detection_mask: np.ndarray,
    ) -> tuple[list[ComponentFeatures], np.ndarray, int]:
        anchors = [component for component in classified if component.accepted]
        recovered_mask = np.zeros_like(detection_mask)
        if not anchors or cv2.countNonZero(paper_mask) == 0:
            return classified, recovered_mask, 0
        c = self.config.components
        radius = max(1, int(c.recovery_anchor_dilation_pixels))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        used_anchor_ids: set[int] = set()
        recovered_components: list[ComponentFeatures] = []
        next_label = 200000
        for paper_component in self._paper_component_candidates(paper_mask, enhancement, egg):
            if not self._recovery_shape_is_valid(paper_component, egg):
                continue
            overlapping: list[ComponentFeatures] = []
            for anchor in anchors:
                if id(anchor) in used_anchor_ids:
                    continue
                expanded_anchor = cv2.dilate(anchor.mask, kernel, iterations=1)
                overlap_pixels = cv2.countNonZero(cv2.bitwise_and(paper_component.mask, expanded_anchor))
                minimum_overlap = max(
                    c.recovery_minimum_overlap_pixels,
                    int(round(min(anchor.area, paper_component.area) * 0.008)),
                )
                if overlap_pixels >= minimum_overlap:
                    overlapping.append(anchor)
            if not overlapping:
                continue
            longest_anchor = max(overlapping, key=lambda value: value.skeleton_length)
            extends_anchor = (
                paper_component.skeleton_length >= longest_anchor.skeleton_length * 1.08
                or paper_component.span >= longest_anchor.span * 1.08
            )
            if not extends_anchor:
                continue
            fused_mask = cv2.bitwise_and(paper_component.mask, detection_mask)
            for anchor in overlapping:
                fused_mask = cv2.bitwise_or(fused_mask, anchor.mask)
            pixels = fused_mask > 0
            bright_mean = float(np.mean(enhancement.bright_response[pixels])) if np.any(pixels) else 0.0
            dark_mean = float(np.mean(enhancement.dark_response[pixels])) if np.any(pixels) else 0.0
            polarity = CrackPolarity.BRIGHT if bright_mean >= dark_mean else CrackPolarity.DARK
            response = enhancement.bright_response if polarity == CrackPolarity.BRIGHT else enhancement.dark_response
            recovered = self.analyzer.analyze_mask(
                fused_mask,
                polarity,
                response,
                enhancement,
                egg,
                label=next_label,
            )
            next_label += 1
            recovered = self.rejector.classify(recovered, egg)
            if not self._recovery_shape_is_valid(recovered, egg):
                continue
            recovered.accepted = True
            recovered.score = max(
                recovered.score,
                max(anchor.score for anchor in overlapping),
                c.acceptance_score + 0.12,
            )
            recovered.reasons = tuple(
                reason
                for reason in recovered.reasons
                if reason not in {'smooth_shell_mark', 'smooth_bright_arc'}
            ) + ('paper_guided_full_trace',)
            for anchor in overlapping:
                anchor.accepted = False
                used_anchor_ids.add(id(anchor))
            recovered_components.append(recovered)
            recovered_mask = cv2.bitwise_or(recovered_mask, recovered.mask)
        classified.extend(recovered_components)
        return classified, recovered_mask, len(recovered_components)

    def _directional_extension_mask(
        self,
        component: ComponentFeatures,
        response: np.ndarray,
        edge_response: np.ndarray,
        detection_mask: np.ndarray,
        egg,
        weak_threshold: float,
    ) -> np.ndarray:
        c = self.config.components
        output = np.zeros_like(detection_mask)
        if not c.directional_extension_enabled:
            return output
        if component.elongation < c.directional_extension_minimum_elongation:
            return output
        if component.skeleton_length < egg.minor_axis * c.directional_extension_minimum_length_ratio:
            return output

        coordinates_yx = np.column_stack(np.where(component.mask > 0)).astype(np.float32)
        if coordinates_yx.shape[0] < 2:
            return output
        coordinates_xy = coordinates_yx[:, ::-1]
        center = np.asarray(component.center, dtype=np.float32)
        axis = np.asarray(component.axis, dtype=np.float32)
        axis /= max(float(np.linalg.norm(axis)), 1e-6)
        perpendicular = np.array([-axis[1], axis[0]], dtype=np.float32)
        relative = coordinates_xy - center
        along = relative @ axis
        across = relative @ perpendicular
        minimum_t = float(along.min())
        maximum_t = float(along.max())
        corridor_half_width = max(6.0, egg.minor_axis * c.directional_extension_corridor_ratio)
        maximum_steps = max(
            18,
            int(round(min(
                egg.minor_axis * c.directional_extension_maximum_ratio,
                max(component.span * 0.80, 18.0),
            ))),
        )
        minimum_extension = max(
            10.0,
            component.span * c.directional_extension_minimum_span_ratio,
        )
        minimum_response = max(
            24.0,
            float(weak_threshold) * c.directional_extension_response_ratio,
        )
        minimum_edge = float(c.directional_extension_minimum_edge)
        lateral_step = max(1, int(c.directional_extension_lateral_step))
        gap_limit = max(1, int(c.directional_extension_gap_limit))
        image_height, image_width = detection_mask.shape

        boundary_distance = cv2.distanceTransform((detection_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
        endpoints = [(-1.0, minimum_t), (1.0, maximum_t)]
        endpoint_distances: list[float] = []
        for _, endpoint_t in endpoints:
            endpoint_band = np.abs(along - endpoint_t) <= 5.0
            endpoint_points = coordinates_xy[endpoint_band].astype(np.int32)
            distances = [
                float(boundary_distance[y, x])
                for x, y in endpoint_points
                if 0 <= x < image_width and 0 <= y < image_height
            ]
            endpoint_distances.append(float(np.median(distances)) if distances else float('inf'))
        if len(endpoint_distances) == 2:
            nearest_index = int(np.argmin(endpoint_distances))
            farther_index = 1 - nearest_index
            distance_gap = endpoint_distances[farther_index] - endpoint_distances[nearest_index]
            if distance_gap >= egg.minor_axis * 0.05:
                endpoints = [endpoints[nearest_index]]

        for direction, endpoint_t in endpoints:
            endpoint_band = np.abs(along - endpoint_t) <= 5.0
            current_offset = float(np.median(across[endpoint_band])) if np.any(endpoint_band) else 0.0
            consecutive_gap = 0
            good_points: list[tuple[int, int, float]] = []
            last_good_t: float | None = None

            for step in range(1, maximum_steps + 1):
                current_t = endpoint_t + direction * float(step)
                best: tuple[float, int, int, float, float, float] | None = None
                for delta in range(-lateral_step, lateral_step + 1):
                    offset = current_offset + float(delta)
                    if abs(offset) > corridor_half_width:
                        continue
                    point = center + axis * current_t + perpendicular * offset
                    x = int(round(float(point[0])))
                    y = int(round(float(point[1])))
                    if x < 0 or x >= image_width or y < 0 or y >= image_height:
                        continue
                    if detection_mask[y, x] == 0:
                        continue
                    response_value = float(response[y, x])
                    edge_value = float(edge_response[y, x])
                    score = response_value + edge_value * 0.25
                    if best is None or score > best[0]:
                        best = (score, x, y, offset, response_value, edge_value)
                if best is None:
                    break

                _, x, y, offset, response_value, edge_value = best
                is_supported = response_value >= minimum_response and edge_value >= minimum_edge
                if is_supported:
                    current_offset = offset
                    consecutive_gap = 0
                    good_points.append((x, y, current_t))
                    last_good_t = current_t
                else:
                    consecutive_gap += 1
                    if consecutive_gap > gap_limit:
                        break

            if last_good_t is None:
                continue
            extension_span = abs(last_good_t - endpoint_t)
            if extension_span < minimum_extension:
                continue
            previous: tuple[int, int, float] | None = None
            for point in good_points:
                if previous is not None and abs(point[2] - previous[2]) <= gap_limit + 1:
                    cv2.line(output, (previous[0], previous[1]), (point[0], point[1]), 255, 1, cv2.LINE_8)
                else:
                    output[point[1], point[0]] = 255
                previous = point

        return cv2.bitwise_and(output, detection_mask)

    def _extend_accepted_components(
        self,
        components: list[ComponentFeatures],
        enhancement,
        egg,
        detection_mask: np.ndarray,
        dark_weak_threshold: float,
        bright_weak_threshold: float,
    ) -> tuple[list[ComponentFeatures], np.ndarray, int]:
        extended_components: list[ComponentFeatures] = []
        extension_mask = np.zeros_like(detection_mask)
        extension_count = 0
        for component in components:
            if not component.accepted:
                extended_components.append(component)
                continue
            response = (
                enhancement.bright_response
                if component.polarity == CrackPolarity.BRIGHT
                else enhancement.dark_response
            )
            weak_threshold = (
                bright_weak_threshold
                if component.polarity == CrackPolarity.BRIGHT
                else dark_weak_threshold
            )
            recovered = self._directional_extension_mask(
                component,
                response,
                enhancement.edge_response,
                detection_mask,
                egg,
                weak_threshold,
            )
            if cv2.countNonZero(recovered) == 0:
                extended_components.append(component)
                continue
            fused = cv2.bitwise_or(component.mask, recovered)
            updated = self.analyzer.analyze_mask(
                fused,
                component.polarity,
                response,
                enhancement,
                egg,
                label=component.label,
            )
            updated = self.rejector.classify(updated, egg)
            updated.accepted = True
            updated.score = max(component.score, updated.score)
            updated.reasons = tuple(
                reason
                for reason in component.reasons
                if reason not in {'smooth_shell_mark', 'smooth_bright_arc'}
            ) + ('directional_ridge_extension',)
            extended_components.append(updated)
            extension_mask = cv2.bitwise_or(extension_mask, recovered)
            extension_count += 1
        return extended_components, extension_mask, extension_count

    def detect(self, image: np.ndarray, include_steps: bool = False) -> PipelineResult:
        started = time.perf_counter()
        working = self._working_image(image)
        egg = self.segmenter.segment(working)
        quality = self.quality_assessor.assess(working, egg)
        if not quality.acceptable:
            raise ValueError(quality.message)
        detection_mask = self._whole_egg_detection_mask(egg.full_mask)
        enhancement = self.enhancer.enhance(working, egg, quality, detection_mask)
        dark = self.threshold.apply(
            enhancement.dark_response,
            detection_mask,
            self.config.threshold.dark_minimum_weak,
            self.config.threshold.dark_minimum_strong,
        )
        bright = self.threshold.apply(
            enhancement.bright_response,
            detection_mask,
            self.config.threshold.bright_minimum_weak,
            self.config.threshold.bright_minimum_strong,
        )
        dark_candidates = cv2.bitwise_and(connect_small_gaps(dark.grown_mask), detection_mask)
        bright_candidates = cv2.bitwise_and(connect_small_gaps(bright.grown_mask), detection_mask)
        dark_components = self.analyzer.extract(
            dark_candidates,
            CrackPolarity.DARK,
            enhancement.dark_response,
            enhancement,
            egg,
        )
        bright_components = self.analyzer.extract(
            bright_candidates,
            CrackPolarity.BRIGHT,
            enhancement.bright_response,
            enhancement,
            egg,
        )
        raw_components = dark_components + bright_components
        classified = [self.rejector.classify(component, egg) for component in raw_components]
        for polarity in (CrackPolarity.DARK, CrackPolarity.BRIGHT):
            polarity_components = [component for component in classified if component.polarity == polarity]
            if len(polarity_components) <= 20:
                candidates = [component for component in polarity_components if not component.accepted]
                for group in self.grouper.groups(candidates, egg):
                    if self.grouper.accept_group(group, egg):
                        for component in group:
                            component.accepted = True
                            component.score = max(component.score, self.config.components.acceptance_score + 0.03)
                            component.reasons = tuple(
                                reason
                                for reason in component.reasons
                                if reason not in {'smooth_shell_mark', 'smooth_bright_arc'}
                            ) + ('coherent_fragment_group',)
            self._keep_multiple_texture_survivors(polarity_components, egg)
        paper = self.paper.detect(working, egg)
        classified, recovered_trace_mask, recovered_count = self._extend_accepted_components(
            classified,
            enhancement,
            egg,
            detection_mask,
            dark.weak_threshold,
            bright.weak_threshold,
        )
        components = deduplicate_components(classified)
        accepted = [component for component in components if component.accepted]
        crack_mask = np.zeros_like(detection_mask)
        for component in accepted:
            crack_mask = cv2.bitwise_or(crack_mask, component.mask)
        crack_mask = cv2.bitwise_and(crack_mask, detection_mask)
        support_mask = cv2.bitwise_or(dark.grown_mask, bright.grown_mask)
        support_mask = cv2.bitwise_and(support_mask, detection_mask)
        steps: dict[str, np.ndarray] = {}
        if include_steps:
            steps = {
                'egg_mask': egg.full_mask,
                'inner_egg_mask': egg.inner_mask,
                'whole_egg_detection_mask': detection_mask,
                'perimeter_shell_zone': egg.rim_mask,
                'normalized_shell': enhancement.normalized,
                'dark_crack_response': enhancement.dark_response,
                'bright_crack_response': enhancement.bright_response,
                'edge_support': enhancement.edge_response,
                'dark_weak_candidates': dark.weak_mask,
                'dark_strong_candidates': dark.strong_mask,
                'bright_weak_candidates': bright.weak_mask,
                'bright_strong_candidates': bright.strong_mask,
                'pale_surface_response': enhancement.bright_response,
                'pale_surface_weak_candidates': bright.weak_mask,
                'spatial_chain_candidates': cv2.bitwise_or(dark_candidates, bright_candidates),
                'fused_crack_response': cv2.max(enhancement.dark_response, enhancement.bright_response),
                'directional_ridge_extension_mask': recovered_trace_mask,
                'paper_guided_recovered_mask': np.zeros_like(recovered_trace_mask),
                'accepted_crack_mask': crack_mask,
                **paper.steps,
            }
        processing_time_ms = int(round((time.perf_counter() - started) * 1000.0))
        metadata = {
            'dark_candidates': len(dark_components),
            'bright_candidates': len(bright_components),
            'accepted_dark': sum(component.accepted and component.polarity == CrackPolarity.DARK for component in components),
            'accepted_bright': sum(component.accepted and component.polarity == CrackPolarity.BRIGHT for component in components),
            'directional_ridge_extensions': recovered_count,
            'paper_guided_recovered': 0,
        }
        return PipelineResult(
            original=image,
            working=working,
            egg=egg,
            quality=quality,
            crack_mask=crack_mask,
            support_mask=support_mask,
            components=components,
            raw_component_count=len(raw_components),
            dark_thresholds=(dark.weak_threshold, dark.strong_threshold),
            bright_thresholds=(bright.weak_threshold, bright.strong_threshold),
            paper=paper,
            steps=steps,
            processing_time_ms=processing_time_ms,
            metadata=metadata,
        )
