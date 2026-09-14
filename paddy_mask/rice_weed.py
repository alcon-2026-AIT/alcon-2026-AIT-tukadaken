"""Model-backed rice/weed inference used after primary-paddy extraction.

Feature extraction is adapted from the user-supplied ``detector.py`` received
with this task. No license accompanied that source; keep its provenance when
redistributing this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


RICE_LABEL = 0
WEED_LABEL = 1


@dataclass(frozen=True)
class RiceWeedPrediction:
    """Full-frame, mutually exclusive rice and weed classifications."""

    rice_mask: np.ndarray
    weed_mask: np.ndarray


@dataclass(frozen=True)
class RiceWeedSummary:
    """Contest metadata derived only from model-classified foreground pixels."""

    level: int
    plant_pixels: int
    weed_pixels: int
    weed_ratio_percent: float


def classify_level(weed_ratio_percent: float) -> int:
    """Apply the four thresholds from the supplied contest classifier program."""

    if weed_ratio_percent < 8:
        return 0
    if weed_ratio_percent < 20:
        return 1
    if weed_ratio_percent < 40:
        return 2
    return 3


def summarize_prediction(prediction: RiceWeedPrediction) -> RiceWeedSummary:
    """Calculate contest fields p, w, r, and level from the classifier masks."""

    if prediction.rice_mask.shape != prediction.weed_mask.shape:
        raise ValueError("rice_mask and weed_mask must have matching dimensions.")
    if np.any(prediction.rice_mask & prediction.weed_mask):
        raise ValueError("rice_mask and weed_mask must not overlap.")
    rice_pixels = int(np.count_nonzero(prediction.rice_mask))
    weed_pixels = int(np.count_nonzero(prediction.weed_mask))
    plant_pixels = rice_pixels + weed_pixels
    weed_ratio_percent = 0.0 if plant_pixels == 0 else weed_pixels / plant_pixels * 100.0
    return RiceWeedSummary(
        level=classify_level(weed_ratio_percent),
        plant_pixels=plant_pixels,
        weed_pixels=weed_pixels,
        weed_ratio_percent=weed_ratio_percent,
    )


class RiceWeedClassifier:
    """Run the supplied two-class joblib model on paddy foreground pixels only."""

    def __init__(self, model_path: Path, batch_size: int = 250_000) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        try:
            import joblib
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Rice/weed model inference requires joblib and scikit-learn. "
                "Install requirements-paddy-mask-contest.txt."
            ) from error
        if not model_path.is_file():
            raise ValueError(
                f"Rice/weed model is required for final contest output but was not found: {model_path}"
            )
        self._model = joblib.load(model_path)
        expected_features = getattr(self._model, "n_features_in_", 7)
        if expected_features != 7:
            raise ValueError(
                f"Rice/weed model must accept the supplied detector's 7 features, "
                f"but {model_path} expects {expected_features}."
            )
        self._batch_size = batch_size

    @staticmethod
    def _features_for_indices(
        bgr: np.ndarray, lab: np.ndarray, flat_indices: np.ndarray
    ) -> np.ndarray:
        import cv2

        b, g, r = (channel.reshape(-1)[flat_indices] for channel in cv2.split(bgr))
        l, a, lab_b = (channel.reshape(-1)[flat_indices] for channel in cv2.split(lab))
        exg = 2 * g.astype(np.float32) - r.astype(np.float32) - b.astype(np.float32)
        return np.column_stack((b, g, r, l, a, lab_b, exg)).reshape(-1, 7)

    def predict(self, rgb_image: np.ndarray, foreground_mask: np.ndarray) -> RiceWeedPrediction:
        """Classify only foreground pixels; the background remains unclassified."""

        import cv2

        rgb = np.asarray(rgb_image)
        foreground = np.asarray(foreground_mask, dtype=bool)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("rgb_image must have shape (height, width, 3).")
        if foreground.shape != rgb.shape[:2]:
            raise ValueError("foreground_mask dimensions must match rgb_image.")

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        flat_indices = np.flatnonzero(foreground)
        labels = np.empty(flat_indices.size, dtype=np.int64)
        for start in range(0, flat_indices.size, self._batch_size):
            indexes = flat_indices[start : start + self._batch_size]
            predicted = np.asarray(self._model.predict(self._features_for_indices(bgr, lab, indexes)))
            if not np.isin(predicted, (RICE_LABEL, WEED_LABEL)).all():
                raise ValueError(
                    "Rice/weed model returned a label other than 0 (rice) or 1 (weed)."
                )
            labels[start : start + indexes.size] = predicted

        rice_mask = np.zeros(foreground.shape, dtype=bool)
        weed_mask = np.zeros(foreground.shape, dtype=bool)
        rice_mask.flat[flat_indices] = labels == RICE_LABEL
        weed_mask.flat[flat_indices] = labels == WEED_LABEL
        return RiceWeedPrediction(rice_mask=rice_mask, weed_mask=weed_mask)
