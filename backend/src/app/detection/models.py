from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class CrackPolarity(str, Enum):
    DARK = 'dark'
    BRIGHT = 'bright'
    PAPER = 'paper'


@dataclass(frozen=True)
class EggRegion:
    full_mask: np.ndarray
    inner_mask: np.ndarray
    rim_mask: np.ndarray
    contour: np.ndarray
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    width: float
    length: float
    minor_axis: float
    major_axis: float
    area_ratio: float
    score: float


@dataclass(frozen=True)
class QualityReport:
    score: float
    sharpness: float
    detail_variance: float
    saturated_ratio: float
    glare_ratio: float
    glare_width: float
    dynamic_range: float
    glare_mask: np.ndarray
    acceptable: bool
    message: str


@dataclass(frozen=True)
class EnhancementResult:
    normalized: np.ndarray
    raw_channel: np.ndarray
    dark_response: np.ndarray
    bright_response: np.ndarray
    edge_response: np.ndarray
    glare_mask: np.ndarray


@dataclass(frozen=True)
class ThresholdResult:
    weak_mask: np.ndarray
    strong_mask: np.ndarray
    grown_mask: np.ndarray
    weak_threshold: float
    strong_threshold: float


@dataclass
class ComponentFeatures:
    label: int
    polarity: CrackPolarity
    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    area: int
    span: float
    skeleton_length: int
    mean_thickness: float
    maximum_thickness: float
    elongation: float
    density: float
    edge_support: float
    glare_overlap: float
    rim_overlap: float
    mean_response: float
    response_p90: float
    roughness: float
    endpoint_count: int
    branch_count: int
    axis: tuple[float, float]
    center: tuple[float, float]
    accepted: bool = False
    score: float = 0.0
    reasons: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PaperBaselineResult:
    mask: np.ndarray
    crack: bool
    score: float
    components: int
    steps: dict[str, np.ndarray]


@dataclass
class PipelineResult:
    original: np.ndarray
    working: np.ndarray
    egg: EggRegion
    quality: QualityReport
    crack_mask: np.ndarray
    support_mask: np.ndarray
    components: list[ComponentFeatures]
    raw_component_count: int
    dark_thresholds: tuple[float, float]
    bright_thresholds: tuple[float, float]
    paper: PaperBaselineResult
    steps: dict[str, np.ndarray]
    processing_time_ms: int
    metadata: dict[str, Any]
