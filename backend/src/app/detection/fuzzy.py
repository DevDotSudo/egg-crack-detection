from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.detection.config import FuzzyConfig


@dataclass(frozen=True)
class FuzzyResult:
    label: str
    confidence: float
    memberships: dict[str, float]
    score: float


class TrapezoidalMembership:
    @staticmethod
    def evaluate(value: float, points: tuple[float, float, float, float]) -> float:
        a, b, c, d = points
        value = float(value)
        if a == b and value <= b:
            return 1.0
        if c == d and value >= c:
            return 1.0
        if value <= a or value >= d:
            return 0.0
        if b <= value <= c:
            return 1.0
        if value < b:
            return float((value - a) / max(b - a, 1e-9))
        return float((d - value) / max(d - c, 1e-9))


class MamdaniEngine:
    def __init__(self, config: FuzzyConfig) -> None:
        self.config = config
        self.universe = np.linspace(0.0, 1.0, config.universe_points, dtype=np.float64)
        self.output_memberships = {
            label: np.asarray(
                [TrapezoidalMembership.evaluate(value, points) for value in self.universe],
                dtype=np.float64,
            )
            for label, points in config.output_memberships.items()
        }

    def infer(self, rules: list[tuple[float, str]]) -> FuzzyResult:
        aggregated = np.zeros_like(self.universe)
        for firing_strength, label in rules:
            if label not in self.output_memberships:
                raise ValueError(f'Unknown fuzzy output label: {label}')
            alpha = float(np.clip(firing_strength, 0.0, 1.0))
            if alpha > 0.0:
                aggregated = np.maximum(
                    aggregated,
                    np.minimum(self.output_memberships[label], alpha),
                )
        denominator = float(aggregated.sum())
        if denominator <= 1e-12:
            label = max(rules, key=lambda rule: rule[0], default=(0.0, 'medium'))[1]
            memberships = {name: 1.0 if name == label else 0.0 for name in self.output_memberships}
            fallback_scores = {'small': 0.20, 'medium': 0.50, 'large': 0.85}
            return FuzzyResult(label, 1.0, memberships, fallback_scores.get(label, 0.50))
        score = float(np.sum(self.universe * aggregated) / denominator)
        areas = {
            label: float(np.minimum(membership, aggregated).sum())
            for label, membership in self.output_memberships.items()
        }
        total_area = max(sum(areas.values()), 1e-12)
        memberships = {label: area / total_area for label, area in areas.items()}
        label = max(memberships, key=memberships.get)
        return FuzzyResult(
            label=label,
            confidence=float(memberships[label]),
            memberships=memberships,
            score=float(np.clip(score, 0.0, 1.0)),
        )


class EggSizeMamdaniClassifier:
    def __init__(self, config: FuzzyConfig) -> None:
        self.config = config
        self.engine = MamdaniEngine(config)

    def classify(self, width_mm: float, height_mm: float) -> FuzzyResult:
        width = self._fuzzify(width_mm, self.config.egg_width_mm_memberships)
        height = self._fuzzify(height_mm, self.config.egg_height_mm_memberships)
        rules = [
            (min(width['small'], height['small']), 'small'),
            (min(width['medium'], height['medium']), 'medium'),
            (min(width['large'], height['large']), 'large'),
            (min(width['small'], height['medium']), 'small'),
            (min(width['medium'], height['small']), 'small'),
            (min(width['medium'], height['large']), 'large'),
            (min(width['large'], height['medium']), 'large'),
            (min(width['small'], height['large']), 'medium'),
            (min(width['large'], height['small']), 'medium'),
        ]
        if max((strength for strength, _ in rules), default=0.0) <= 1e-9:
            for label in ('small', 'medium', 'large'):
                rules.extend([
                    (width[label] * 0.65, label),
                    (height[label] * 0.65, label),
                ])
        return self.engine.infer(rules)

    @staticmethod
    def _fuzzify(value: float, definitions: dict[str, tuple[float, float, float, float]]) -> dict[str, float]:
        return {
            label: TrapezoidalMembership.evaluate(float(value), points)
            for label, points in definitions.items()
        }


class EggSizeApparentMamdaniClassifier:
    def __init__(self, config: FuzzyConfig) -> None:
        self.config = config
        self.engine = MamdaniEngine(config)

    def classify(self, width_ratio: float, height_ratio: float) -> FuzzyResult:
        width = self._fuzzify(width_ratio, self.config.egg_width_frame_ratio_memberships)
        height = self._fuzzify(height_ratio, self.config.egg_height_frame_ratio_memberships)
        rules = [
            (min(width['small'], height['small']), 'small'),
            (min(width['medium'], height['medium']), 'medium'),
            (min(width['large'], height['large']), 'large'),
            (min(width['small'], height['medium']), 'small'),
            (min(width['medium'], height['small']), 'small'),
            (min(width['medium'], height['large']), 'large'),
            (min(width['large'], height['medium']), 'large'),
            (min(width['small'], height['large']), 'medium'),
            (min(width['large'], height['small']), 'medium'),
        ]
        if max((strength for strength, _ in rules), default=0.0) <= 1e-9:
            for label in ('small', 'medium', 'large'):
                rules.extend([
                    (width[label] * 0.65, label),
                    (height[label] * 0.65, label),
                ])
        return self.engine.infer(rules)

    @staticmethod
    def _fuzzify(value: float, definitions: dict[str, tuple[float, float, float, float]]) -> dict[str, float]:
        return {
            label: TrapezoidalMembership.evaluate(float(value), points)
            for label, points in definitions.items()
        }


class CrackSizeMamdaniClassifier:
    def __init__(self, config: FuzzyConfig) -> None:
        self.config = config
        self.engine = MamdaniEngine(config)

    def classify(
        self,
        is_crack: bool,
        traced_length: float,
        traced_pixels: int,
        egg_area: float,
        component_count: int,
        strongest_response: float,
    ) -> FuzzyResult:
        if not is_crack:
            return FuzzyResult('none', 1.0, {'none': 1.0}, 0.0)
        area_ratio = float(traced_pixels) / max(float(egg_area), 1.0)
        normalized = {
            'length': float(np.clip(traced_length / self.config.crack_length_scale, 0.0, 1.0)),
            'area': float(np.clip(area_ratio / self.config.crack_area_scale, 0.0, 1.0)),
            'strength': float(np.clip(strongest_response / self.config.crack_strength_scale, 0.0, 1.0)),
            'count': float(np.clip(component_count / self.config.crack_component_scale, 0.0, 1.0)),
        }
        memberships = {
            'length': self._fuzzify(normalized['length'], self.config.crack_length_memberships),
            'area': self._fuzzify(normalized['area'], self.config.crack_area_memberships),
            'strength': self._fuzzify(normalized['strength'], self.config.crack_strength_memberships),
            'count': self._fuzzify(normalized['count'], self.config.crack_count_memberships),
        }
        length = memberships['length']
        area = memberships['area']
        strength = memberships['strength']
        count = memberships['count']
        rules = [
            (min(length['small'], area['small']), 'small'),
            (min(length['small'], strength['small']), 'small'),
            (min(strength['small'], count['small']), 'small'),
            (min(length['medium'], area['medium']), 'medium'),
            (min(length['medium'], strength['medium']), 'medium'),
            (min(area['medium'], count['medium']), 'medium'),
            (min(count['medium'], strength['medium']), 'medium'),
            (min(length['large'], area['large']), 'large'),
            (min(length['large'], strength['large']), 'large'),
            (min(area['large'], count['large']), 'large'),
            (min(strength['large'], count['large']), 'large'),
            (min(area['large'], strength['large']), 'large'),
        ]
        if max((firing for firing, _ in rules), default=0.0) <= 1e-9:
            for label in ('small', 'medium', 'large'):
                rules.extend([
                    (length[label], label),
                    (area[label] * 0.85, label),
                    (strength[label] * 0.45, label),
                    (count[label] * 0.35, label),
                ])
        return self.engine.infer(rules)

    @staticmethod
    def _fuzzify(value: float, definitions: dict[str, tuple[float, float, float, float]]) -> dict[str, float]:
        return {
            label: TrapezoidalMembership.evaluate(value, points)
            for label, points in definitions.items()
        }
