import unittest

import numpy as np

from paddy_mask.backends import paddy_detection_score
from paddy_mask.rice_weed import RiceWeedPrediction, classify_level, summarize_prediction
from paddy_mask.scoring import CandidateMetrics, select_primary_candidate


def candidate(index: int, **changes: float) -> CandidateMetrics:
    values = {
        "index": index,
        "detector_score": 0.80,
        "area_fraction": 0.32,
        "centroid_x": 0.50,
        "centroid_y": 0.58,
        "bottom_contact": 0.55,
        "sky_overlap": 0.0,
        "forest_overlap": 0.0,
        "road_overlap": 0.0,
        "building_overlap": 0.0,
    }
    values.update(changes)
    return CandidateMetrics(**values)


class PrimaryCandidateSelectionTests(unittest.TestCase):
    def test_prefers_specific_paddy_detection_over_generic_agriculture(self) -> None:
        paddy_confidence = paddy_detection_score("paddy field", 0.34)
        agriculture_confidence = paddy_detection_score("agricultural field", 0.42)

        self.assertGreater(paddy_confidence, agriculture_confidence)

    def test_prefers_central_substantial_bottom_connected_paddy(self) -> None:
        primary = candidate(0)
        edge_forest = candidate(
            1,
            detector_score=0.95,
            centroid_x=0.08,
            centroid_y=0.20,
            bottom_contact=0.0,
            forest_overlap=0.75,
        )

        selected = select_primary_candidate([edge_forest, primary])

        self.assertEqual(selected.metrics.index, 0)

    def test_penalises_sky_and_road_overlap(self) -> None:
        clear_field = candidate(0, detector_score=0.65)
        obstructed_field = candidate(
            1,
            detector_score=0.95,
            sky_overlap=0.50,
            road_overlap=0.50,
        )

        selected = select_primary_candidate([clear_field, obstructed_field])

        self.assertEqual(selected.metrics.index, 0)

    def test_breaks_equal_scores_by_input_index(self) -> None:
        selected = select_primary_candidate([candidate(3), candidate(1)])

        self.assertEqual(selected.metrics.index, 1)

    def test_rejects_empty_candidate_list(self) -> None:
        with self.assertRaisesRegex(ValueError, "No paddy-mask candidates"):
            select_primary_candidate([])


class RiceWeedSummaryTests(unittest.TestCase):
    def test_uses_supplied_thresholds_and_counts(self) -> None:
        prediction = RiceWeedPrediction(
            rice_mask=np.array([[True, True, True, True, True, True, True, True, True, False]]),
            weed_mask=np.array([[False, False, False, False, False, False, False, False, False, True]]),
        )

        summary = summarize_prediction(prediction)

        self.assertEqual(summary.plant_pixels, 10)
        self.assertEqual(summary.weed_pixels, 1)
        self.assertEqual(summary.weed_ratio_percent, 10)
        self.assertEqual(summary.level, 1)

    def test_classification_level_boundaries(self) -> None:
        self.assertEqual([classify_level(value) for value in (0, 7.99, 8, 19.99, 20, 39.99, 40)], [0, 0, 1, 1, 2, 2, 3])


if __name__ == "__main__":
    unittest.main()
