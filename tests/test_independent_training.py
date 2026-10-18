import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.independent_common import (fit_head, frame_views, letterbox, metrics, nested_evaluation,
    predict, temporal_features, training_arrays, validate_dataset)
from scripts.laptop_common import LinearHead
from scripts.prepare_independent_review import compile_review


def synthetic_dataset():
    videos, store = [], {}
    for group in range(4):
        # Last outer group has only positive examples, as in the real review.
        for label in ([0, 1] if group < 3 else [1]):
            name = f"videos/Х-{group}-{label}.mov"
            row = {"video": name, "reviewed_label": label, "eligible": True, "duration_sec": 5.,
                   "source_sha256": f"{group}-{label}", "layout_group": str(group), "scene_group_id": str(group),
                   "positive_windows": [{"start_sec": 2., "end_sec": 4., "confidence": .8}] if label else []}
            times = np.arange(10) / 2
            features = np.full((10, 4), label * 2 - 1, dtype=np.float32)
            features[:, 1] = group / 10
            store[name] = times, features
            videos.append(row)
    return {"format": "cashier-independent-v4", "organizer_labels_used_for_training": False,
            "ui_assessments_used_for_decisions": False, "videos": videos}, store


class IndependentTrainingTests(unittest.TestCase):
    def test_compile_never_falls_back_to_organizer_labels_or_ui_status(self):
        common = {"reviewed_label": None, "reason": "Не видна сумма", "positive_windows": [],
                  "source_sha256": "a", "layout_group": "A", "scene_group_id": "scene", "observed_event": "unknown"}
        decisions = {"videos": {"videos/a.mov": common}, "ui_assessments_used_for_decisions": False,
                     "organizer_labels_used_for_decisions": False, "policy": "unknown excluded",
                     "reviewer": "review", "review_method": "frames"}
        for old_label in (0, 1):
            manifest, _ = compile_review(decisions, [{"video": "videos/a.mov", "has_violation": str(old_label), "timecode_sec": "1"}])
            self.assertIsNone(manifest["videos"][0]["reviewed_label"])
            self.assertFalse(manifest["videos"][0]["eligible"])
        bad = copy.deepcopy(decisions)
        bad["ui_assessments_used_for_decisions"] = True
        with self.assertRaises(ValueError):
            compile_review(bad, [])

    def test_duplicate_conflict_rejected(self):
        a = {"reviewed_label": 0, "reason": "normal", "positive_windows": [], "source_sha256": "same",
             "layout_group": "A", "scene_group_id": "s", "observed_event": "normal"}
        b = {**a, "reviewed_label": None}
        decisions = {"videos": {"a": a, "b": b}, "ui_assessments_used_for_decisions": False,
                     "organizer_labels_used_for_decisions": False}
        with self.assertRaisesRegex(ValueError, "duplicate"):
            compile_review(decisions, [{"video": n, "has_violation": "0", "timecode_sec": ""} for n in ("a", "b")])

    def test_unknown_video_cannot_become_negative_and_positive_background_is_ignored(self):
        dataset, store = synthetic_dataset()
        videos = dataset["videos"][:2]
        x, y, weights = training_arrays(videos, store)
        self.assertEqual(sum(y == 1), 5)  # 2, 2.5, 3, 3.5, 4; other positive-video frames ignored
        self.assertEqual(sum(y == 0), 10)
        self.assertAlmostEqual(float(weights[y == 0].sum()), .5)
        self.assertEqual(x.shape[1], 8)
        videos = copy.deepcopy(videos)
        videos[0].update(reviewed_label=None, eligible=False)
        with self.assertRaisesRegex(ValueError, "Unresolved"):
            training_arrays(videos, store)

    def test_letterbox_keeps_top_and_bottom_of_portrait(self):
        frame = np.zeros((448, 224, 3), dtype=np.uint8)
        frame[:60] = (255, 0, 0)
        frame[-60:] = (0, 0, 255)
        result = letterbox(frame)
        np.testing.assert_array_equal(result[0, 112], (255, 0, 0))
        np.testing.assert_array_equal(result[-1, 112], (0, 0, 255))
        np.testing.assert_array_equal(result[100, 0], (104, 116, 124))
        views = frame_views(frame)
        self.assertEqual(len(views), 4)
        np.testing.assert_array_equal(views[1], views[0][:, ::-1])
        np.testing.assert_array_equal(views[3], views[2][:, ::-1])

    def test_temporal_baseline_is_per_video(self):
        first = np.full((6, 2), 17, dtype=np.float32)
        second = np.full((6, 2), -50, dtype=np.float32)
        np.testing.assert_array_equal(temporal_features(first)[:, 2:], 0)
        np.testing.assert_array_equal(temporal_features(second)[:, 2:], 0)
        second[5] = 100
        self.assertTrue((temporal_features(second)[-1, 2:] > 0).all())

    def test_nested_folds_never_train_or_calibrate_on_outer_videos(self):
        dataset, store = synthetic_dataset()
        validate_dataset(dataset, Path("data"), verify_files=False)
        predictions, folds = nested_evaluation(dataset["videos"], store, progress=lambda *a, **k: None)
        self.assertEqual(sorted(r["video"] for r in predictions), sorted(store))
        self.assertEqual(len(folds), 4)
        for fold in folds:
            held = set(fold["test_videos"])
            self.assertFalse(held.intersection(fold["train_videos"]))
            self.assertFalse(held.intersection(r["video"] for r in fold["inner_predictions"]))
            for inner in fold["inner_folds"]:
                self.assertFalse(held.intersection(inner["train_videos"] + inner["calibration_videos"]))
                self.assertFalse(set(inner["train_videos"]).intersection(inner["calibration_videos"]))
        self.assertIsNone(folds[-1]["test_metrics"]["balanced_accuracy"])
        self.assertGreater(metrics(predictions)["f1"], .8)

    def test_saved_numpy_head_preserves_predictions(self):
        dataset, store = synthetic_dataset()
        head, _ = fit_head(dataset["videos"], store)
        sample = next(iter(store.values()))
        expected = predict(head, *sample, .5)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "head.npz"
            np.savez(path, **head.state())
            with np.load(path, allow_pickle=False) as arrays:
                restored = LinearHead.from_state(dict(arrays))
        self.assertEqual(predict(restored, *sample, .5), expected)

    def test_related_scenes_cannot_cross_layout_groups(self):
        dataset, _ = synthetic_dataset()
        dataset["videos"][-1]["scene_group_id"] = "0"
        with self.assertRaisesRegex(ValueError, "Related scene"):
            validate_dataset(dataset, Path("data"), verify_files=False)


if __name__ == "__main__":
    unittest.main()
