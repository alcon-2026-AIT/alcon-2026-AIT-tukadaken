import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from paddy_mask.contest import (
    ContestInput,
    ContestOutput,
    build_foreground_roi,
    compose_contest_image,
    final_output_path,
    input_image_path,
    output_relative_path,
    read_input_csv,
    write_output_csv,
)


class ContestPathTests(unittest.TestCase):
    def test_supports_windows_input_paths_and_required_output_name(self) -> None:
        self.assertEqual(output_relative_path(r"photos\plot.one.JPG"), r"photos\plot.one-output.JPG")
        self.assertEqual(
            input_image_path(Path("/contest"), r"photos\plot.one.JPG"),
            Path("/contest/photos/plot.one.JPG"),
        )
        self.assertEqual(
            final_output_path(Path("/contest"), r"photos\plot.one.JPG"),
            Path("/contest/photos/plot.one-output.JPG"),
        )

    def test_rejects_path_escaping_contest_directory(self) -> None:
        with self.assertRaisesRegex(ValueError, "relative path"):
            input_image_path(Path("/contest"), r"..\outside.JPG")

    def test_reads_input_and_writes_downstream_owned_output_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            input_csv = directory / "input.csv"
            input_csv.write_text("2\n640,480,photos/a.JPG\n800,600,photos\\b.png\n", encoding="utf-8")

            records = read_input_csv(input_csv)
            self.assertEqual(records[1], ContestInput(800, 600, r"photos\b.png"))

            output_csv = directory / "output.csv"
            write_output_csv(
                output_csv,
                [
                    ContestOutput(
                        records[0],
                        output_relative_path(records[0].relative_image_path),
                        "DOWNSTREAM_LEVEL",
                        "DOWNSTREAM_P",
                        "DOWNSTREAM_W",
                        "DOWNSTREAM_R",
                    )
                ],
            )
            with output_csv.open(newline="", encoding="utf-8") as csv_file:
                self.assertEqual(
                    list(csv.reader(csv_file)),
                    [
                        ["1"],
                        [
                            "640",
                            "480",
                            "photos/a-output.JPG",
                            "DOWNSTREAM_LEVEL",
                            "DOWNSTREAM_P",
                            "DOWNSTREAM_W",
                            "DOWNSTREAM_R",
                        ],
                    ],
                )


class ForegroundCompositionTests(unittest.TestCase):
    def test_background_is_black_and_roi_is_preserved(self) -> None:
        image = np.array(
            [
                [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
                [[100, 110, 120], [130, 140, 150], [160, 170, 180]],
            ],
            dtype=np.uint8,
        )
        mask = np.array([[False, True, False], [True, True, False]])

        roi = build_foreground_roi(image, mask)

        self.assertEqual(roi.bounding_box, (0, 0, 2, 2))
        np.testing.assert_array_equal(roi.masked_rgb[0, 0], [0, 0, 0])
        np.testing.assert_array_equal(roi.masked_rgb[1, 1], [130, 140, 150])
        self.assertEqual(roi.crop_mask.shape, (2, 2))

    def test_composes_only_downstream_classified_foreground_pixels(self) -> None:
        foreground = build_foreground_roi(
            np.full((2, 3, 3), 25, dtype=np.uint8),
            np.array([[True, True, False], [True, False, False]]),
        )
        rice = np.array([[True, False, False], [False, False, False]])
        weed = np.array([[False, True, False], [False, False, False]])

        result = compose_contest_image(foreground, rice, weed)

        np.testing.assert_array_equal(result[0, 0], [128, 128, 128])
        np.testing.assert_array_equal(result[0, 1], [255, 255, 255])
        np.testing.assert_array_equal(result[1, 0], [0, 0, 0])
        np.testing.assert_array_equal(result[1, 2], [0, 0, 0])

    def test_rejects_downstream_classification_outside_roi(self) -> None:
        foreground = build_foreground_roi(
            np.zeros((1, 2, 3), dtype=np.uint8),
            np.array([[True, False]]),
        )
        with self.assertRaisesRegex(ValueError, "outside the foreground"):
            compose_contest_image(foreground, np.array([[False, True]]), np.array([[False, False]]))
