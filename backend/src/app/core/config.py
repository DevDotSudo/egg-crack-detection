from dataclasses import dataclass


@dataclass(frozen=True)
class DetectionConfig:
    # The Windows webcam capture path (camera_windows / DirectShow) hands back
    # frames upside-down for some cameras. This is corrected once, right after
    # the JPEG/PNG bytes are decoded, before any egg or crack analysis runs.
    # Options: 'none', 'rotate_180' (upside-down), 'flip_vertical'
    # (top/bottom mirrored only), 'flip_horizontal' (left/right mirrored
    # only). If the preview still looks wrong after this fix, change this
    # value to match what you actually see, not what you'd expect.
    camera_orientation_fix: str = 'flip_horizontal'

    # A 4K capture is reduced only to full HD, preserving a two-pixel source
    # fracture as a traceable one-pixel feature without stalling live use.
    target_width: int = 1920
    target_height: int = 1080

    # Live-camera mode uses a smaller working frame and one detection pass.
    # Uploaded images still use target_width/target_height for maximum detail.
    camera_target_width: int = 960
    camera_target_height: int = 540
    camera_fast_mode: bool = True

    # Geometric thresholds are calibrated against this apparent egg width.
    # Larger source frames keep their extra detail while line lengths, gaps,
    # kernels, and thickness limits scale with the detected shell.
    geometry_reference_egg_minor_axis: float = 370.0
    geometry_min_scale: float = 0.65
    geometry_max_scale: float = 2.25
    geometry_scale: float = 1.0
    detail_refinement_target_egg_minor_axis: float = 560.0
    detail_refinement_max_scale: float = 1.50

    # Egg isolation and validation.
    min_egg_area_ratio: float = 0.02
    max_egg_area_ratio: float = 0.88
    # These are the minimum minor/major dimensions, independent of whether the
    # egg is standing vertically or lying horizontally.
    min_egg_width: int = 60
    min_egg_height: int = 80
    min_egg_aspect: float = 1.0
    max_egg_aspect: float = 2.20
    min_egg_score: float = 3.2
    min_egg_pixels: int = 4000
    # Keep nearly the full egg available for crack analysis. A tiny boundary
    # guard avoids background bleed without limiting detection to the center.
    inner_margin_ratio: float = 0.01
    min_inner_margin: float = 1.0
    min_inner_pixels: int = 4000

    # Fuzzy apparent egg-size grading. These normalized contour-area bands
    # assume the camera and egg distance stay approximately fixed.
    egg_size_small_full_ratio: float = 0.12
    egg_size_small_empty_ratio: float = 0.24
    egg_size_medium_left_ratio: float = 0.14
    egg_size_medium_peak_ratio: float = 0.27
    egg_size_medium_right_ratio: float = 0.44
    egg_size_large_empty_ratio: float = 0.34
    egg_size_large_full_ratio: float = 0.50

    # Fuzzy size also considers the egg axes relative to the normalized frame.
    egg_width_small_full_ratio: float = 0.24
    egg_width_small_empty_ratio: float = 0.39
    egg_width_medium_left_ratio: float = 0.25
    egg_width_medium_peak_ratio: float = 0.43
    egg_width_medium_right_ratio: float = 0.61
    egg_width_large_empty_ratio: float = 0.52
    egg_width_large_full_ratio: float = 0.68
    egg_length_small_full_ratio: float = 0.36
    egg_length_small_empty_ratio: float = 0.53
    egg_length_medium_left_ratio: float = 0.37
    egg_length_medium_peak_ratio: float = 0.57
    egg_length_medium_right_ratio: float = 0.77
    egg_length_large_empty_ratio: float = 0.68
    egg_length_large_full_ratio: float = 0.84

    # Capture-quality validation. Poor images are rejected instead of being
    # confidently labeled as cracked or clean.
    quality_min_laplacian_variance: float = 25.0
    quality_good_laplacian_variance: float = 85.0
    quality_hard_min_boundary_sharpness: float = 68.0
    quality_min_boundary_sharpness: float = 190.0
    quality_good_boundary_sharpness: float = 430.0
    quality_saturation_value: int = 250
    quality_glare_value: int = 247
    quality_max_saturated_ratio: float = 0.055
    quality_max_glare_component_ratio: float = 0.012
    quality_max_glare_thickness: float = 9.0
    quality_min_median: float = 18.0
    quality_min_dynamic_range: float = 12.0
    quality_good_dynamic_range: float = 58.0

    # Candling illumination correction and local detail enhancement.
    overexposure_mean_threshold: float = 205.0
    overexposure_p95_threshold: float = 245.0
    overexposure_max_gamma: float = 2.0
    # Flatfield correction: a larger sigma is needed to capture the wide-scale
    # dark-top / bright-bottom illumination gradient produced by candling. The
    # original sigma=55 was too small, leaving a strong residual gradient that
    # the crack detectors misread as a broad band of candidates.
    flatfield_sigma: float = 90.0
    flatfield_strength: float = 0.82
    flatfield_max_dimension: int = 480

    # CLAHE with higher clip limit and finer tile grid exposes faint crack
    # valleys that the original gentle settings left hidden in the shell
    # texture background.
    clahe_clip_limit: float = 4.0
    clahe_tile_size: int = 8
    bilateral_sigma_color: float = 15.0
    bilateral_sigma_space: float = 15.0

    # Response construction and adaptive hysteresis thresholds.
    line_sigmas: tuple[float, ...] = (0.7, 1.1, 2.8, 6.0)
    morphology_sizes: tuple[int, ...] = (3, 7, 13)
    tophat_line_kernels: tuple[tuple[int, int], ...] = (
        (1, 9), (9, 1), (1, 13), (13, 1),
    )
    log_sigmas: tuple[float, ...] = (0.8, 1.4, 2.2)
    texture_window_size: int = 17
    texture_coherence_window: int = 7
    texture_ridge_sigmas: tuple[float, ...] = (0.45, 0.65, 0.9, 1.25, 1.8, 2.5)
    texture_min_weak_threshold: int = 3
    texture_min_strong_threshold: int = 6
    texture_min_coherence: float = 0.16
    texture_contrast_weight: float = 0.44
    texture_ridge_weight: float = 0.56
    log_min_weak_threshold: int = 3
    log_min_strong_threshold: int = 9
    grad_min_weak_threshold: int = 5
    grad_min_strong_threshold: int = 9
    weak_percentile: float = 88.0
    strong_percentile: float = 97.0
    weak_percentile_scale: float = 0.46
    strong_percentile_scale: float = 0.72
    weak_mad_factor: float = 0.9
    strong_mad_factor: float = 2.2
    dark_min_weak_threshold: int = 2
    dark_min_strong_threshold: int = 5
    bright_min_weak_threshold: int = 2
    bright_min_strong_threshold: int = 7
    max_response_threshold: int = 150

    # Dark-channel-specific threshold tuning (lower than bright defaults)
    dark_weak_mad_factor: float = 0.6
    dark_strong_mad_factor: float = 1.6
    dark_weak_percentile_scale: float = 0.36
    dark_strong_percentile_scale: float = 0.58

    # Direct local-contrast hairline detection. A median background model
    # exposes one-pixel dark or bright cracks without amplifying shell texture.
    local_hairline_windows: tuple[int, ...] = (7, 11)
    local_hairline_min_weak_contrast: int = 2
    local_hairline_min_strong_contrast: int = 5
    local_hairline_noise_percentile: float = 99.4
    local_hairline_noise_scale: float = 1.7
    local_hairline_min_length: float = 9.0
    local_hairline_min_span: float = 8.0
    local_hairline_min_elongation: float = 1.45
    local_hairline_max_thickness: float = 5.0
    local_hairline_max_density: float = 0.58
    local_hairline_min_strength: float = 10.0
    local_hairline_min_score: float = 0.45
    local_hairline_component_overlap: float = 0.08

    # A faint transmitted-light crack may be only a few grey levels brighter
    # than mottled shell texture. Average the raw bright response along many
    # orientations and keep only long, aligned traces deep inside the egg.
    persistent_bright_windows: tuple[int, ...] = (7, 11, 21, 31)
    persistent_bright_kernel_length: int = 31
    persistent_bright_percentile: float = 98.0
    persistent_bright_min_contrast: int = 4
    persistent_bright_support_contrast: int = 2
    persistent_bright_connector_support: float = 0.45
    persistent_bright_core_ratio: float = 0.28
    persistent_trace_min_skeleton_length: float = 55.0
    persistent_trace_min_span: float = 45.0
    persistent_trace_min_elongation: float = 3.0
    persistent_trace_max_average_thickness: float = 3.2
    persistent_trace_max_density: float = 0.22
    persistent_trace_max_branch_ratio: float = 0.25
    persistent_trace_min_extent_ratio: float = 0.65

    # Pale surface cracks can appear as whitish, low-saturation ridges under
    # candling. They are kept only when multiple nearby fragments form a
    # stronger local ridge than the surrounding shell texture.
    pale_surface_percentile: float = 98.0
    pale_surface_weak_percentile: float = 85.0
    pale_surface_min_threshold: int = 65
    pale_surface_min_weak_threshold: int = 18
    pale_surface_background_window: int = 31
    pale_surface_lightness_weight: float = 1.4
    pale_surface_saturation_weight: float = 1.1
    pale_surface_support_radius: int = 10
    pale_surface_min_pixels: int = 90
    pale_surface_min_skeleton_length: float = 55.0
    pale_surface_max_skeleton_length: float = 95.0
    pale_surface_max_average_thickness: float = 6.2
    pale_surface_max_density: float = 0.58
    pale_surface_max_branch_ratio: float = 0.35
    pale_surface_min_extent_ratio: float = 0.60
    pale_surface_max_axis_deviation: float = 0.10
    pale_surface_min_strength_p90: float = 95.0
    pale_surface_min_components: int = 2
    pale_surface_min_total_length: float = 75.0
    pale_surface_min_group_span: float = 80.0
    pale_surface_max_group_gap: float = 115.0
    pale_surface_min_axis_alignment: float = 0.45
    pale_surface_min_connector_alignment: float = 0.25
    pale_surface_max_thickness_ratio: float = 1.90
    pale_surface_branch_min_length_ratio: float = 0.72
    pale_surface_branch_min_span_ratio: float = 0.62
    pale_surface_branch_max_axis_deviation: float = 0.28
    pale_surface_branch_max_density: float = 0.66
    pale_surface_branch_min_strength_ratio: float = 0.70
    pale_recovery_min_fragment_length: float = 35.0
    pale_recovery_min_fragment_span: float = 20.0
    pale_recovery_max_endpoint_gap: float = 36.0
    pale_recovery_anchor_min_length: float = 80.0
    pale_recovery_anchor_min_span: float = 60.0
    pale_recovery_anchor_max_endpoint_gap: float = 64.0
    pale_recovery_anchor_min_thickness_ratio: float = 1.15
    pale_recovery_min_total_length: float = 220.0
    pale_recovery_min_network_span: float = 120.0
    pale_recovery_min_shell_depth_ratio: float = 0.04
    pale_recovery_min_strength_p90: float = 112.0
    pale_recovery_max_branch_ratio: float = 0.72
    pale_recovery_max_axis_deviation: float = 0.16
    pale_recovery_min_axis_alignment: float = 0.30
    pale_recovery_min_connector_alignment: float = 0.75
    pale_recovery_candidate_limit: int = 64

    # The paper-style channel uses the same full-egg analysis area as the main
    # detector instead of a center-only fallback.
    paper_min_depth_ratio: float = 0.05

    # Component filtering and shell-artifact rejection.
    smooth_band_min_length: float = 75.0
    smooth_band_min_branch_ratio: float = 0.46
    smooth_band_min_thickness: float = 3.8
    smooth_band_min_branchpoints: float = 20.0
    smooth_arc_min_length: float = 70.0
    smooth_arc_max_axis_ratio: float = 3.2
    smooth_arc_max_residual: float = 0.12
    smooth_arc_min_coverage: float = 0.14
    smooth_arc_max_chord_ratio: float = 0.76
    smooth_short_arc_min_length: float = 24.0
    smooth_short_arc_min_axis_deviation: float = 0.030
    smooth_short_arc_max_chord_ratio: float = 0.86
    smooth_short_arc_max_residual: float = 0.18
    min_component_pixels: int = 12
    min_component_span: float = 28.0
    min_skeleton_length: float = 35.0
    min_elongation: float = 1.3
    preferred_max_thickness: float = 7.5
    max_component_thickness: float = 8.0
    max_component_density: float = 0.62
    min_component_score: float = 1.05
    support_min_span: float = 6.0
    support_min_skeleton_length: float = 4.0
    trace_geometry_min_skeleton_length: float = 12.0
    trace_geometry_max_average_thickness: float = 4.0
    trace_geometry_max_density: float = 0.28
    trace_geometry_max_branch_ratio: float = 0.32
    trace_geometry_min_extent_ratio: float = 0.62
    trace_geometry_max_axis_deviation: float = 0.075
    trace_broad_min_skeleton_length: float = 70.0
    trace_broad_max_skeleton_length: float = 220.0
    trace_broad_min_elongation: float = 4.0
    trace_broad_max_average_thickness: float = 8.0
    trace_broad_max_density: float = 0.45
    trace_broad_max_branch_ratio: float = 0.50
    trace_broad_min_extent_ratio: float = 0.80
    trace_broad_max_axis_deviation: float = 0.035
    trace_broad_min_strong_overlap: float = 0.45
    trace_broad_min_texture_overlap: float = 0.25
    trace_broad_max_texture_overlap: float = 0.75
    trace_broad_max_texture_strength: float = 230.0

    texture_seed_min_length: float = 55.0
    texture_seed_min_span: float = 48.0
    texture_seed_min_elongation: float = 3.4
    texture_seed_max_thickness: float = 5.4
    texture_seed_max_density: float = 0.42
    texture_seed_min_strength: float = 62.0
    texture_seed_min_strong_overlap: float = 0.26
    texture_seed_min_extent_ratio: float = 0.62
    texture_seed_min_score: float = 2.0
    texture_support_radius: int = 3
    texture_max_standalone_components: int = 3
    fragmented_texture_crack_min_skeleton_length: float = 90.0
    fragmented_texture_crack_min_elongation: float = 4.5
    fragmented_texture_crack_max_thickness: float = 8.0
    fragmented_texture_crack_max_density: float = 0.42
    fragmented_texture_crack_min_texture_overlap: float = 0.20
    fragmented_texture_crack_min_score: float = 4.5

    thin_crack_min_length: float = 55.0
    thin_crack_min_span: float = 45.0
    thin_crack_max_thickness: float = 5.5
    thin_crack_min_elongation: float = 1.6
    thin_crack_min_texture_strength: float = 28.0
    thin_crack_min_texture_overlap: float = 0.08
    thin_crack_min_score: float = 0.50
    max_fragmented_components: int = 8

    dominant_min_skeleton_length: float = 85.0
    dominant_min_span: float = 70.0
    dominant_min_elongation: float = 4.0
    dominant_max_average_thickness: float = 8.5
    dominant_max_density: float = 0.34
    dominant_min_strength_p90: float = 50.0
    dominant_min_strong_overlap: float = 0.30
    dominant_min_component_score: float = 3.2
    dominant_network_min_skeleton_length: float = 450.0
    dominant_network_min_span: float = 300.0
    dominant_network_min_elongation: float = 2.5
    dominant_network_max_average_thickness: float = 8.5
    dominant_network_max_density: float = 0.12
    dominant_network_min_strength_p90: float = 100.0
    dominant_network_min_strong_overlap: float = 0.50
    dominant_network_min_component_score: float = 5.0
    texture_dominant_min_skeleton_length: float = 420.0
    texture_dominant_min_span: float = 320.0
    texture_dominant_min_elongation: float = 4.5
    texture_dominant_max_density: float = 0.22
    texture_dominant_min_strength_p90: float = 90.0
    texture_dominant_min_strong_overlap: float = 0.40
    texture_dominant_min_texture_strength: float = 150.0
    texture_dominant_min_texture_overlap: float = 0.20
    fragment_link_min_skeleton_length: float = 40.0
    fragment_link_min_span: float = 30.0
    fragment_link_min_elongation: float = 3.0
    fragment_link_max_average_thickness: float = 8.0
    fragment_link_max_density: float = 0.45
    fragment_link_min_strength_p90: float = 35.0
    fragment_link_max_endpoint_gap: float = 130.0
    fragment_link_min_axis_alignment: float = 0.90
    fragment_link_min_connector_alignment: float = 0.90
    fragment_group_min_components: int = 3
    fragment_group_min_total_length: float = 200.0
    fragment_group_min_span: float = 155.0
    fragment_group_min_mean_strength: float = 60.0
    spatial_chain_max_gap: float = 48.0
    spatial_chain_min_components: int = 3
    spatial_chain_min_total_length: float = 120.0
    spatial_chain_min_span: float = 180.0
    spatial_chain_min_elongation: float = 2.0
    spatial_chain_max_thickness: float = 6.2
    spatial_chain_max_density: float = 0.55
    spatial_chain_min_strength: float = 48.0

    # ── Morphological crack extraction ──
    # Directional line kernels at 8 angles (0° to 157.5° in 22.5° steps).
    # Blackhat/tophat with these kernels finds narrow dark/bright lines while
    # rejecting round shell pores.
    morph_kernel_length: int = 21
    morph_angles: tuple[float, ...] = (0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5)
    morph_amplify: float = 3.0

    # Dark valley response channel
    dark_valley_dilation_size: int = 3
    dark_valley_line_length: int = 11
    dark_valley_min_directional_count: int = 3
    dark_valley_max_directional_count: int = 4
    dark_valley_min_weak_threshold: int = 2
    dark_valley_min_strong_threshold: int = 4

    # Adaptive threshold for the morphological response.
    morph_weak_percentile: float = 92.0
    morph_strong_percentile: float = 97.0
    morph_min_weak_threshold: int = 6
    morph_min_strong_threshold: int = 14

    # ── Line filtering ──
    # Connected components from the morphological extraction are kept only
    # if they look like lines, not blobs.
    line_min_span: float = 32.0
    line_min_skeleton_length: float = 40.0
    line_min_elongation: float = 2.5
    line_max_density: float = 0.45
    line_min_score: float = 0.5
    line_max_thickness: float = 18.0

    # Support growth: seeds grow into adjacent weak pixels.
    support_radius: int = 4
    rim_band_thickness: int = 3
    rim_overlap_reject_ratio: float = 0.92
    perimeter_depth_ratio: float = 0.08
    perimeter_min_component_overlap: float = 0.28
    perimeter_min_penetration_ratio: float = 0.025
    perimeter_max_tangent_alignment: float = 0.92

    # Relaxed geometry for dark crack sources
    dark_crack_max_thickness: float = 6.0
    dark_crack_max_max_thickness: float = 10.0
    dark_crack_max_density: float = 0.45
    dark_crack_max_branch_ratio: float = 0.40

    # ── Crack decision ──
    decision_min_longest: float = 70.0
    decision_min_total_length: float = 100.0
    decision_min_score: float = 0.50

    # ── Multi-frame validation ──
    multi_frame_count: int = 3
    multi_frame_dilation: int = 5
    multi_frame_min_agreement: int = 2
    multi_frame_min_overlap: float = 0.18
    multi_frame_min_weak_overlap: float = 0.14
    multi_frame_max_support_area_ratio: float = 12.0
    multi_frame_registration_width: int = 512
    multi_frame_registration_height: int = 768

    # ── Crack size fuzzy classification ──
    fuzzy_length_scale: float = 520.0
    fuzzy_area_scale: float = 0.035
    fuzzy_strength_scale: float = 100.0
    fuzzy_component_scale: float = 8.0

    # Retained for backward compatibility with saved records.
    area_consistency_full_spread: float = 0.05
    area_consistency_max_spread: float = 0.20

    # Iterative camera trace extraction.
    iterative_max_iterations: int = 3
    iterative_exclusion_padding: int = 14
    iterative_min_new_pixels: int = 6
    iterative_min_relative_length: float = 0.08


CONFIG = DetectionConfig()
