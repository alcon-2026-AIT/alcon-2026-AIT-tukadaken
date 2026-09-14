"""Pure candidate-mask scoring used by every extraction backend."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Iterable


@dataclass(frozen=True)
class CandidateMetrics:
    """Normalised measurements for one proposed paddy mask."""

    index: int
    detector_score: float
    area_fraction: float
    centroid_x: float
    centroid_y: float
    bottom_contact: float
    sky_overlap: float = 0.0
    forest_overlap: float = 0.0
    road_overlap: float = 0.0
    building_overlap: float = 0.0


@dataclass(frozen=True)
class ScoredCandidate:
    """A candidate accompanied by its deterministic ranking score."""

    metrics: CandidateMetrics
    score: float


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def centrality_score(centroid_x: float, centroid_y: float) -> float:
    """Prefer a subject near the lower centre where a photographed paddy lies."""

    target_x, target_y = 0.5, 0.58
    max_distance = hypot(max(target_x, 1.0 - target_x), max(target_y, 1.0 - target_y))
    return _clamp(1.0 - hypot(centroid_x - target_x, centroid_y - target_y) / max_distance)


def area_score(area_fraction: float) -> float:
    """Reward substantial regions while rejecting masks covering nearly all pixels."""

    area = _clamp(area_fraction)
    return min(area / 0.18, 1.0) * min((1.0 - area) / 0.18, 1.0)


def score_candidate(metrics: CandidateMetrics) -> float:
    """Return a score where larger values indicate the primary photographed paddy."""

    penalty = (
        0.30 * _clamp(metrics.sky_overlap)
        + 0.26 * _clamp(metrics.forest_overlap)
        + 0.24 * _clamp(metrics.road_overlap)
        + 0.20 * _clamp(metrics.building_overlap)
    )
    return (
        0.23 * _clamp(metrics.detector_score)
        + 0.25 * centrality_score(metrics.centroid_x, metrics.centroid_y)
        + 0.22 * area_score(metrics.area_fraction)
        + 0.15 * _clamp(metrics.bottom_contact)
        - penalty
    )


def select_primary_candidate(candidates: Iterable[CandidateMetrics]) -> ScoredCandidate:
    """Select the highest-scoring candidate, breaking ties by candidate index."""

    scored = [ScoredCandidate(metrics, score_candidate(metrics)) for metrics in candidates]
    if not scored:
        raise ValueError("No paddy-mask candidates were supplied.")
    return max(scored, key=lambda item: (item.score, -item.metrics.index))
