from pathlib import Path
import tempfile
import unittest

import numpy as np

from paddy_mask.rice_weed import RiceWeedClassifier


class RiceWeedClassifierTests(unittest.TestCase):
    def test_classifies_only_foreground_with_supplied_seven_feature_contract(self) -> None:
        try:
            import joblib
            from sklearn.dummy import DummyClassifier
        except ModuleNotFoundError:
            self.skipTest("joblib and scikit-learn are provided by requirements-paddy-mask-contest.txt")

        model = DummyClassifier(strategy="constant", constant=1).fit(
            np.zeros((2, 7)),
            np.array([0, 1]),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "rice_weed_model.pkl"
            joblib.dump(model, model_path)
            prediction = RiceWeedClassifier(model_path, batch_size=1).predict(
                np.full((2, 2, 3), 100, dtype=np.uint8),
                np.array([[True, False], [True, True]]),
            )

        np.testing.assert_array_equal(prediction.rice_mask, [[False, False], [False, False]])
        np.testing.assert_array_equal(prediction.weed_mask, [[True, False], [True, True]])


if __name__ == "__main__":
    unittest.main()
