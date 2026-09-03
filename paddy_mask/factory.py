"""Shared construction of extraction backends for command-line entry points."""

from __future__ import annotations

import sys

from .backends import (
    HeuristicPaddyExtractor,
    ZeroShotDependenciesUnavailable,
    ZeroShotPaddyExtractor,
)


def build_extractor(
    backend: str,
    device: str,
    grounding_model_id: str,
    sam_model_id: str,
    box_threshold: float,
    text_threshold: float,
) -> tuple[object, str]:
    """Create the requested extractor without disguising a semantic fallback."""

    if backend == "heuristic":
        print("Using heuristic mode: output is non-semantic and may confuse forest with paddy.", file=sys.stderr)
        return HeuristicPaddyExtractor(), "heuristic"
    try:
        return (
            ZeroShotPaddyExtractor(
                device=device,
                grounding_model_id=grounding_model_id,
                sam_model_id=sam_model_id,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            ),
            "zero-shot",
        )
    except ZeroShotDependenciesUnavailable:
        if backend == "zero-shot":
            raise
        print(
            "WARNING: zero-shot packages are unavailable; using the non-semantic heuristic fallback. "
            "Install requirements-paddy-mask-zero-shot.txt for semantic paddy extraction.",
            file=sys.stderr,
        )
        return HeuristicPaddyExtractor(), "heuristic-fallback"
