from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DetectionResponse(BaseModel):
    id: str
    is_crack: bool
    confidence: float
    area_ratio: float
    contour_length: float
    processing_time_ms: int
    original_image_b64: str
    overlay_image_b64: str
    intermediate_steps: Optional[Dict[str, str]] = None
    timestamp: str
    candidate_components: int = 0
    raw_candidate_components: int = 0
    dominant_crack_override: bool = False
    candidate_pixels: int = 0
    longest_candidate: float = 0.0
    mean_candidate_strength: float = 0.0
    detection_score: float = 0.0
    threshold_used: int = 0
    paper_method_used: bool = True
    paper_method_crack: bool = False
    paper_method_score: float = 0.0
    paper_method_components: int = 0
    shell_texture_score: float = 0.0
    shell_texture_uniformity: float = 1.0
    texture_anomaly_ratio: float = 0.0
    texture_candidate_pixels: int = 0
    thin_crack_score: float = 0.0
    thin_crack_detected: bool = False
    image_quality_score: float = 0.0
    image_sharpness: float = 0.0
    image_detail_variance: float = 0.0
    image_saturated_ratio: float = 0.0
    image_glare_ratio: float = 0.0
    image_dynamic_range: float = 0.0
    requires_recapture: bool = False
    quality_message: str = ''
    egg_detected: bool = True
    egg_score: float = 0.0
    egg_size: str = 'unknown'
    egg_size_confidence: float = 0.0
    egg_area_ratio: float = 0.0
    egg_width_pixels: float = 0.0
    egg_length_pixels: float = 0.0
    egg_width_ratio: float = 0.0
    egg_length_ratio: float = 0.0
    egg_size_score: float = 0.0
    egg_size_memberships: Dict[str, float] = Field(default_factory=dict)
    crack_size: str = 'none'
    crack_size_confidence: float = 0.0
    crack_mask_b64: str = ''
    crack_locations: List[Dict[str, Any]] = Field(default_factory=list)
    detection_iterations: int = 0
    search_iterations: int = 1
    termination_reason: str = 'no_more_cracks'
    sample_count: int = 1
    crack_votes: int = 0
    no_crack_votes: int = 1
    decision_consistency: float = 1.0
    area_consistent: bool = True
    area_consistency: float = 1.0
    area_mean_ratio: float = 0.0
    area_spread_ratio: float = 0.0
    area_samples: List[float] = Field(default_factory=list)


class HistorySaveRequest(BaseModel):
    result: DetectionResponse
    source_name: str = 'camera'
