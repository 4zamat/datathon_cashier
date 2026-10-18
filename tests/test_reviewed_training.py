import copy
from pathlib import Path
import unittest

import numpy as np

from scripts.laptop_common import LinearHead
from scripts.reviewed_common import (fit_linear, frame_targets, interval_error, scores_for,
                                     temporal_features, validate_dataset)


class ReviewedTrainingTests(unittest.TestCase):
    def test_temporal_context_never_crosses_video_boundary(self):
        a = np.array([[1.0], [3.0]], dtype=np.float32)
        b = np.array([[100.0]], dtype=np.float32)
        self.assertTrue(np.array_equal(temporal_features(b), [[100, 100, 0]]))
        result = temporal_features(a)
        self.assertEqual(result.shape, (2, 3))
        np.testing.assert_allclose(result[:, 1], [1.8, 2.2])
        np.testing.assert_allclose(result[:, 2], [2, 2])

    def test_uncertainty_is_ignored_and_positive_overrides_normal(self):
        video = {"has_violation": 1, "events": [
            {"start_sec": 0, "end_sec": 8, "assessment": "no_violation"},
            {"start_sec": 2, "end_sec": 4, "assessment": "uncertain"}],
            "positive_windows": [{"start_sec": 5, "end_sec": 7, "weight": .35}]}
        y, w = frame_targets(np.arange(9, dtype=np.float64), video)
        self.assertEqual(y.tolist(), [0, 0, -1, -1, -1, 1, 1, 1, 0])
        self.assertEqual(w[3], 0)
        self.assertAlmostEqual(w[6], .35)

    def test_video_negative_does_not_override_uncertain_note(self):
        y, w = frame_targets(np.array([0., 1., 2.]), {"has_violation": 0,
            "events": [{"start_sec": .8, "end_sec": 1.2, "assessment": "uncertain"}], "positive_windows": []})
        self.assertEqual(y.tolist(), [0, -1, 0])
        self.assertAlmostEqual(w[0], .35)

    def test_fit_and_head_state_reload_agree(self):
        x = np.repeat(np.array([[-2., -1., 0.], [2., 1., 0.]], dtype=np.float32), 8, axis=0)
        y = np.repeat(np.array([0., 1.], dtype=np.float32), 8)
        w = np.ones(16, dtype=np.float32) / 16
        head, history, epoch = fit_linear(x, y, w, x, y, w, .01, epochs=25)
        self.assertGreater(epoch, 0)
        self.assertLess(history[-1]["train_loss"], history[0]["train_loss"])
        clone = LinearHead.from_state(head.state())
        np.testing.assert_array_equal(head.logits(x), clone.logits(x))
        self.assertLess(head.logits(x)[0], head.logits(x)[-1])

    def test_scene_and_hash_leakage_rejected(self):
        videos = []
        for si, split in enumerate(("train", "val", "test")):
            for label in (0, 1):
                videos.append({"video": f"videos/{split}{label}.mov", "split": split,
                    "sha256": f"{si}{label}", "scene_group_id": split, "split_group_id": split,
                    "duration_sec": 3, "has_violation": label, "events": [],
                    "positive_windows": [{"start_sec": 1., "end_sec": 2., "weight": .35}] if label else []})
        dataset = {"format": "cashier-reviewed-v3", "videos": videos}
        validate_dataset(dataset, Path("data"), verify_files=False)
        broken = copy.deepcopy(dataset)
        broken["videos"][-1]["split_group_id"] = "train"
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_dataset(broken, Path("data"), verify_files=False)
        broken = copy.deepcopy(dataset)
        broken["videos"][1]["sha256"] = broken["videos"][0]["sha256"]
        with self.assertRaisesRegex(ValueError, "duplicates"):
            validate_dataset(broken, Path("data"), verify_files=False)

    def test_interval_error_does_not_require_one_arbitrary_timecode(self):
        intervals = [{"start_sec": 54, "end_sec": 72}]
        self.assertEqual(interval_error(57, intervals), 0)
        self.assertEqual(interval_error(68, intervals), 0)
        self.assertEqual(interval_error(50, intervals), 4)
        self.assertIsNone(interval_error(50, []))


if __name__ == "__main__":
    unittest.main()
