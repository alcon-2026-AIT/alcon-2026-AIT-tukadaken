"""Segmentation backends for automatic primary paddy-mask extraction."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Iterable, Sequence

import numpy as np

from .scoring import CandidateMetrics, ScoredCandidate, select_primary_candidate


class ZeroShotDependenciesUnavailable(RuntimeError):
    """Raised only when the optional zero-shot packages are not installed."""


class NoPaddyDetected(RuntimeError):
    """Raised when a functioning semantic model finds no paddy candidates."""


@dataclass(frozen=True)
class ExtractionResult:
    """The selected binary mask and the backend that generated it."""

    mask: np.ndarray
    selected: ScoredCandidate
    backend: str


@dataclass(frozen=True)
class Detection:
    """One Grounding DINO detection in image pixel coordinates."""

    label: str
    score: float
    box: tuple[float, float, float, float]


_PROMPT = (
    "rice paddy. paddy field. rice field. agricultural field. farmland. "
    "forest. trees. sky. road. building."
)
_NEGATIVE_LABELS = {
    "sky": ("sky",),
    "forest": ("forest", "tree", "woods"),
    "road": ("road", "street", "path", "driveway"),
    "building": ("building", "house", "shed"),
}


def _normalise_label(label: str) -> str:
    return " ".join(label.lower().replace("_", " ").split())


def _is_paddy_label(label: str) -> bool:
    normalised = _normalise_label(label)
    return any(
        phrase in normalised
        for phrase in ("rice paddy", "paddy field", "rice field", "agricultural field", "farmland")
    )


def paddy_detection_score(label: str, detector_score: float) -> float:
    """Prefer a specific paddy phrase over a broad agricultural-region proposal."""

    normalised = _normalise_label(label)
    if "rice paddy" in normalised:
        specificity_bonus = 0.24
    elif "paddy field" in normalised:
        specificity_bonus = 0.22
    elif "rice field" in normalised:
        specificity_bonus = 0.20
    else:
        specificity_bonus = 0.0
    return min(1.0, detector_score + specificity_bonus)


def _negative_category(label: str) -> str | None:
    normalised = _normalise_label(label)
    for category, terms in _NEGATIVE_LABELS.items():
        if any(term in normalised for term in terms):
            return category
    return None


def _box_mask(box: tuple[float, float, float, float], shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    x1, y1, x2, y2 = box
    left = max(0, min(width, round(min(x1, x2))))
    right = max(0, min(width, round(max(x1, x2))))
    top = max(0, min(height, round(min(y1, y2))))
    bottom = max(0, min(height, round(max(y1, y2))))
    region = np.zeros(shape, dtype=bool)
    region[top:bottom, left:right] = True
    return region


def candidate_metrics(
    mask: np.ndarray,
    index: int,
    detector_score: float,
    negative_regions: dict[str, Sequence[np.ndarray]],
) -> CandidateMetrics:
    """Measure a binary candidate against scene-layout and negative detections."""

    candidate = np.asarray(mask, dtype=bool)
    if candidate.ndim != 2:
        raise ValueError("A candidate mask must be a two-dimensional array.")
    height, width = candidate.shape
    area_pixels = int(candidate.sum())
    if area_pixels == 0:
        raise ValueError("A candidate mask cannot be empty.")

    y_coords, x_coords = np.nonzero(candidate)
    bottom_start = int(height * 0.90)
    bottom_columns = np.any(candidate[bottom_start:, :], axis=0)
    overlaps = {
        category: max(
            (float(np.count_nonzero(candidate & region)) / area_pixels for region in regions),
            default=0.0,
        )
        for category, regions in negative_regions.items()
    }
    return CandidateMetrics(
        index=index,
        detector_score=detector_score,
        area_fraction=area_pixels / candidate.size,
        centroid_x=float(x_coords.mean() / max(width - 1, 1)),
        centroid_y=float(y_coords.mean() / max(height - 1, 1)),
        bottom_contact=float(bottom_columns.mean()),
        sky_overlap=overlaps.get("sky", 0.0),
        forest_overlap=overlaps.get("forest", 0.0),
        road_overlap=overlaps.get("road", 0.0),
        building_overlap=overlaps.get("building", 0.0),
    )


def _select_mask(
    masks: Iterable[np.ndarray],
    detector_scores: Iterable[float],
    negative_regions: dict[str, Sequence[np.ndarray]],
    backend: str,
) -> ExtractionResult:
    candidate_masks = [np.asarray(mask, dtype=bool) for mask in masks]
    metrics = [
        candidate_metrics(mask, index, score, negative_regions)
        for index, (mask, score) in enumerate(zip(candidate_masks, detector_scores, strict=True))
        if np.any(mask)
    ]
    selected = select_primary_candidate(metrics)
    return ExtractionResult(candidate_masks[selected.metrics.index], selected, backend)


class ZeroShotPaddyExtractor:
    """Grounding DINO proposes paddy regions and SAM turns them into masks."""

    def __init__(
        self,
        device: str = "auto",
        grounding_model_id: str = "IDEA-Research/grounding-dino-tiny",
        sam_model_id: str = "facebook/sam-vit-base",
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
    ) -> None:
        try:
            import torch
            from transformers import (
                AutoModelForZeroShotObjectDetection,
                AutoProcessor,
                SamModel,
                SamProcessor,
            )
        except ModuleNotFoundError as error:
            raise ZeroShotDependenciesUnavailable(
                "Zero-shot mode requires torch and transformers. "
                "Install requirements-paddy-mask-zero-shot.txt."
            ) from error

        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(grounding_model_id)
        self._grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_model_id)
        self._sam_processor = SamProcessor.from_pretrained(sam_model_id)
        self._sam_model = SamModel.from_pretrained(sam_model_id)
        self._device = self._resolve_device(device)
        self._grounding_model.to(self._device).eval()
        self._sam_model.to(self._device).eval()
        self._box_threshold = box_threshold
        self._text_threshold = text_threshold

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        if self._torch.cuda.is_available():
            return "cuda"
        mps = getattr(self._torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    def _detect(self, image: object) -> list[Detection]:
        inputs = self._processor(images=image, text=_PROMPT, return_tensors="pt").to(self._device)
        with self._torch.no_grad():
            outputs = self._grounding_model(**inputs)

        postprocess = self._processor.post_process_grounded_object_detection
        kwargs = {
            "target_sizes": [image.size[::-1]],
            "box_threshold": self._box_threshold,
            "text_threshold": self._text_threshold,
        }
        if "box_threshold" not in inspect.signature(postprocess).parameters:
            kwargs["threshold"] = kwargs.pop("box_threshold")
            kwargs.pop("text_threshold")
        result = postprocess(outputs, inputs.input_ids, **kwargs)[0]
        labels = result.get("text_labels", result["labels"])
        return [
            Detection(
                label=str(label),
                score=float(score),
                box=tuple(float(value) for value in box.tolist()),
            )
            for label, score, box in zip(labels, result["scores"], result["boxes"], strict=True)
        ]

    def _segment(self, image: object, boxes: Sequence[tuple[float, float, float, float]]) -> list[np.ndarray]:
        input_boxes = [[list(box) for box in boxes]]
        inputs = self._sam_processor(images=image, input_boxes=input_boxes, return_tensors="pt")
        inputs["input_boxes"] = inputs["input_boxes"].to(dtype=self._torch.float32)
        inputs = inputs.to(self._device)
        with self._torch.no_grad():
            outputs = self._sam_model(**inputs)
        processed = self._sam_processor.image_processor.post_process_masks(
            outputs.pred_masks,
            inputs["original_sizes"],
            inputs["reshaped_input_sizes"],
        )[0]
        masks = processed.detach().cpu().numpy()
        qualities = outputs.iou_scores[0].detach().cpu().numpy()
        return [
            masks[index, int(np.argmax(qualities[index]))].astype(bool)
            for index in range(len(boxes))
        ]

    def extract(self, image: object) -> ExtractionResult:
        detections = self._detect(image)
        paddy_detections = [detection for detection in detections if _is_paddy_label(detection.label)]
        if not paddy_detections:
            raise NoPaddyDetected(
                "The zero-shot detector ran successfully but did not find a rice-paddy candidate."
            )

        negative_regions: dict[str, list[np.ndarray]] = {}
        for detection in detections:
            category = _negative_category(detection.label)
            if category:
                negative_regions.setdefault(category, []).append(
                    _box_mask(detection.box, image.size[::-1])
                )
        masks = self._segment(image, [detection.box for detection in paddy_detections])
        return _select_mask(
            masks,
            [
                paddy_detection_score(detection.label, detection.score)
                for detection in paddy_detections
            ],
            negative_regions,
            backend="zero-shot",
        )


class HeuristicPaddyExtractor:
    """Non-semantic emergency fallback for systems without zero-shot dependencies."""

    def extract(self, image: object) -> ExtractionResult:
        try:
            import cv2
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Heuristic fallback requires opencv-python. Install requirements-paddy-mask.txt."
            ) from error

        rgb = np.asarray(image.convert("RGB"))
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hue, saturation, value = cv2.split(hsv)
        green = (hue >= 25) & (hue <= 95) & (saturation >= 45) & (value >= 30)
        cleaned = cv2.morphologyEx(green.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)

        masks = [
            labels == label
            for label in range(1, count)
            if stats[label, cv2.CC_STAT_AREA] >= max(100, cleaned.size // 2000)
        ]
        if not masks:
            raise NoPaddyDetected("The heuristic fallback could not find a substantial green region.")

        height = cleaned.shape[0]
        forest_region = np.zeros_like(cleaned, dtype=bool)
        forest_region[: int(height * 0.42), :] = True
        return _select_mask(
            masks,
            [0.0] * len(masks),
            {"forest": [forest_region]},
            backend="heuristic-fallback",
        )
