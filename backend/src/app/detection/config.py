import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SegmentationConfig:
    minimum_area_ratio: float = 0.02
    maximum_area_ratio: float = 0.88
    minimum_solidity: float = 0.82
    maximum_aspect_ratio: float = 2.35
    inner_margin_ratio: float = 0.012
    rim_width_ratio: float = 0.045
    detection_border_pixels: int = 1


@dataclass(frozen=True)
class QualityConfig:
    minimum_dynamic_range: float = 12.0
    minimum_sharpness: float = 18.0
    maximum_saturated_ratio: float = 0.055
    maximum_glare_ratio: float = 0.022
    maximum_glare_width_ratio: float = 0.045


@dataclass(frozen=True)
class EnhancementConfig:
    flatfield_sigma_ratio: float = 0.14
    flatfield_strength: float = 0.85
    clahe_clip_limit: float = 2.4
    clahe_grid_size: int = 8
    small_sigmas: tuple[float, ...] = (0.65, 1.0, 1.45)
    large_sigmas: tuple[float, ...] = (2.4, 4.0, 6.0)
    morphology_ratios: tuple[float, ...] = (0.009, 0.015, 0.024)


@dataclass(frozen=True)
class ThresholdConfig:
    weak_percentile: float = 91.5
    strong_percentile: float = 97.8
    dark_minimum_weak: float = 45.0
    dark_minimum_strong: float = 65.0
    bright_minimum_weak: float = 65.0
    bright_minimum_strong: float = 95.0
    maximum_growth_iterations: int = 8


@dataclass(frozen=True)
class ComponentConfig:
    minimum_area: int = 4
    minimum_span_ratio: float = 0.035
    minimum_length_ratio: float = 0.055
    preferred_length_ratio: float = 0.18
    maximum_mean_thickness_ratio: float = 0.020
    maximum_thickness_ratio: float = 0.045
    maximum_density: float = 0.48
    minimum_elongation: float = 1.35
    maximum_glare_overlap: float = 0.18
    maximum_rim_overlap: float = 0.70
    minimum_response: float = 0.10
    minimum_edge_support: float = 0.035
    minimum_roughness: float = 0.020
    smooth_line_maximum_roughness: float = 0.014
    acceptance_score: float = 0.50
    group_maximum_gap_ratio: float = 0.12
    group_minimum_length_ratio: float = 0.15
    group_minimum_alignment: float = 0.65
    rim_minimum_inward_fraction: float = 0.08
    rim_minimum_radial_alignment: float = 0.30
    rim_minimum_length_ratio: float = 0.10
    rim_maximum_mean_thickness_ratio: float = 0.016
    texture_overload_threshold: int = 12
    maximum_accepted_components: int = 8
    recovery_anchor_dilation_pixels: int = 5
    recovery_minimum_overlap_pixels: int = 3
    recovery_minimum_length_ratio: float = 0.12
    recovery_minimum_span_ratio: float = 0.10
    recovery_maximum_mean_thickness_ratio: float = 0.026
    recovery_maximum_density: float = 0.32
    recovery_minimum_edge_support: float = 0.14
    recovery_minimum_response: float = 0.10
    directional_extension_enabled: bool = True
    directional_extension_minimum_elongation: float = 4.0
    directional_extension_minimum_length_ratio: float = 0.16
    directional_extension_maximum_ratio: float = 0.34
    directional_extension_minimum_span_ratio: float = 0.15
    directional_extension_corridor_ratio: float = 0.025
    directional_extension_lateral_step: int = 4
    directional_extension_gap_limit: int = 8
    directional_extension_response_ratio: float = 0.43
    directional_extension_minimum_edge: int = 45


@dataclass(frozen=True)
class TemporalConfig:
    minimum_overlap: float = 0.12
    weak_support_overlap: float = 0.80
    minimum_vote_ratio: float = 0.50
    normalized_width: int = 256
    normalized_height: int = 320


@dataclass(frozen=True)
class CalibrationConfig:
    camera_distance_inches: float = 4.0
    default_reference_width_mm: float = 30.0
    minimum_reference_pixels: float = 40.0
    maximum_reference_pixels: float = 1100.0
    resolution_tolerance_pixels: int = 2
    center_tolerance_ratio: float = 0.28
    border_margin_ratio: float = 0.012
    minimum_egg_width_mm: float = 25.0
    maximum_egg_width_mm: float = 65.0
    minimum_egg_height_mm: float = 35.0
    maximum_egg_height_mm: float = 85.0


@dataclass(frozen=True)
class FuzzyConfig:
    universe_points: int = 1001
    output_memberships: dict[str, tuple[float, float, float, float]] = field(default_factory=lambda: {
        'small': (0.00, 0.00, 0.25, 0.45),
        'medium': (0.30, 0.45, 0.55, 0.70),
        'large': (0.55, 0.75, 1.00, 1.00),
    })
    egg_width_mm_memberships: dict[str, tuple[float, float, float, float]] = field(default_factory=lambda: {
        'small': (25.0, 25.0, 37.0, 41.5),
        'medium': (38.5, 41.5, 44.5, 47.5),
        'large': (44.5, 48.0, 65.0, 65.0),
    })
    egg_height_mm_memberships: dict[str, tuple[float, float, float, float]] = field(default_factory=lambda: {
        'small': (35.0, 35.0, 50.0, 54.5),
        'medium': (51.0, 54.0, 59.0, 62.5),
        'large': (59.0, 63.0, 85.0, 85.0),
    })
    egg_width_frame_ratio_memberships: dict[str, tuple[float, float, float, float]] = field(default_factory=lambda: {
        'small': (0.00, 0.00, 0.36, 0.55),
        'medium': (0.40, 0.52, 0.62, 0.82),
        'large': (0.72, 0.92, 1.00, 1.00),
    })
    egg_height_frame_ratio_memberships: dict[str, tuple[float, float, float, float]] = field(default_factory=lambda: {
        'small': (0.00, 0.00, 0.50, 0.70),
        'medium': (0.55, 0.68, 0.78, 0.92),
        'large': (0.82, 0.98, 1.00, 1.00),
    })
    crack_length_memberships: dict[str, tuple[float, float, float, float]] = field(default_factory=lambda: {
        'small': (0.00, 0.00, 0.08, 0.35),
        'medium': (0.15, 0.35, 0.55, 0.75),
        'large': (0.55, 0.80, 1.00, 1.00),
    })
    crack_area_memberships: dict[str, tuple[float, float, float, float]] = field(default_factory=lambda: {
        'small': (0.00, 0.00, 0.04, 0.18),
        'medium': (0.08, 0.22, 0.40, 0.65),
        'large': (0.48, 0.72, 1.00, 1.00),
    })
    crack_strength_memberships: dict[str, tuple[float, float, float, float]] = field(default_factory=lambda: {
        'small': (0.00, 0.00, 0.20, 0.50),
        'medium': (0.25, 0.45, 0.65, 0.85),
        'large': (0.65, 0.88, 1.00, 1.00),
    })
    crack_count_memberships: dict[str, tuple[float, float, float, float]] = field(default_factory=lambda: {
        'small': (0.00, 0.00, 0.15, 0.55),
        'medium': (0.20, 0.40, 0.60, 0.80),
        'large': (0.60, 0.85, 1.00, 1.00),
    })
    crack_length_scale: float = 520.0
    crack_area_scale: float = 0.035
    crack_strength_scale: float = 100.0
    crack_component_scale: float = 8.0


@dataclass(frozen=True)
class DetectionConfig:
    target_width: int = 1280
    target_height: int = 720
    camera_orientation_fix: str = field(default_factory=lambda: os.getenv('EGG_CAMERA_ORIENTATION_FIX', 'none'))
    min_egg_width: int = 50
    min_egg_height: int = 70
    min_egg_pixels: int = 3000
    min_inner_pixels: int = 1800
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    enhancement: EnhancementConfig = field(default_factory=EnhancementConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    components: ComponentConfig = field(default_factory=ComponentConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    fuzzy: FuzzyConfig = field(default_factory=FuzzyConfig)


CONFIG = DetectionConfig()
