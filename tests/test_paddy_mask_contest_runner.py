import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from paddy_mask import contest_runner
from paddy_mask.rice_weed import RiceWeedPrediction


class _FakeExtractor:
    def extract(self, image: object) -> object:
        return type(
            "Extraction",
            (),
            {
                "mask": np.array([[True, True], [False, True]]),
                "backend": "zero-shot",
            },
        )()


class _FakeClassifier:
    def __init__(self, model_path: Path, batch_size: int) -> None:
        self.model_path = model_path
        self.batch_size = batch_size

    def predict(self, rgb: np.ndarray, foreground: np.ndarray) -> RiceWeedPrediction:
        if foreground.shape != rgb.shape[:2]:
            raise AssertionError("Foreground shape must match the input image.")
        return RiceWeedPrediction(
            rice_mask=np.array([[True, False], [False, False]]),
            weed_mask=np.array([[False, True], [False, True]]),
        )


class ContestRunnerTests(unittest.TestCase):
    def test_creates_required_images_and_output_csv(self) -> None:
        try:
            from PIL import Image
        except ModuleNotFoundError:
            self.skipTest("Pillow is provided by requirements-paddy-mask-contest.txt")

        with tempfile.TemporaryDirectory() as temporary_directory:
            working_directory = Path(temporary_directory)
            image_path = working_directory / "images" / "field.photo.png"
            image_path.parent.mkdir()
            Image.fromarray(np.full((2, 2, 3), 50, dtype=np.uint8), mode="RGB").save(image_path)
            (working_directory / "input.csv").write_text(
                "1\n2,2,images/field.photo.png\n", encoding="utf-8"
            )

            with (
                patch(
                    "paddy_mask.contest_runner.build_extractor",
                    return_value=(_FakeExtractor(), "zero-shot"),
                ),
                patch("paddy_mask.contest_runner.RiceWeedClassifier", _FakeClassifier),
            ):
                exit_code = contest_runner.main(
                    ["--input-csv", str(working_directory / "input.csv"), "--model", "model.pkl"]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue((working_directory / "images" / "field.photo-output.JPG").is_file())
            with (working_directory / "output.csv").open(newline="", encoding="utf-8") as output_file:
                self.assertEqual(
                    list(csv.reader(output_file)),
                    [
                        ["1"],
                        [
                            "2",
                            "2",
                            "images/field.photo-output.JPG",
                            "3",
                            "3",
                            "2",
                            "66.666667",
                        ],
                    ],
                )
