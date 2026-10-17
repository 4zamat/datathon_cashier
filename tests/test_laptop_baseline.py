"""Lightweight checks for the laptop baseline helpers. No training, no video decoding, no GPU."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

try:
    import numpy as np
    from scripts import laptop_common as lc
except ModuleNotFoundError as exc:  # service-only environment without ML extras
    np = None
    lc = None
    SKIP_REASON = str(exc)
else:
    SKIP_REASON = ""


@unittest.skipIf(lc is None, f"ML dependencies missing: {SKIP_REASON}")
class SplitTests(unittest.TestCase):
    def groups(self):
        return [{"group_id": f"g{i:02d}", "has_violation": i % 2, "sha256": f"s{i}"} for i in range(40)]

    def test_split_is_disjoint_complete_and_stratified(self):
        groups = self.groups()
        parts = lc.stratified_group_split(groups, seed=42)
        ids = [g["group_id"] for name in lc.SPLITS for g in parts[name]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {g["group_id"] for g in groups})
        for members in parts.values():
            self.assertEqual({g["has_violation"] for g in members}, {0, 1})
        self.assertEqual([len(parts[n]) for n in lc.SPLITS], [24, 8, 8])

    def test_split_is_reproducible(self):
        first = lc.stratified_group_split(self.groups(), seed=42)
        second = lc.stratified_group_split(self.groups(), seed=42)
        self.assertEqual(first, second)

    def test_saved_splits_have_no_overlap_if_present(self):
        if not (lc.SPLIT_DIR / "train.csv").is_file():
            self.skipTest("split files not generated")
        seen: dict[str, str] = {}
        for name in lc.SPLITS:
            for row in lc.read_csv(lc.SPLIT_DIR / f"{name}.csv"):
                for sha in (row["sha256"],):
                    self.assertNotIn(sha, seen, f"{row['video']} also in {seen.get(sha)}")
                    seen[sha] = name
                for duplicate in filter(None, row["duplicates"].split(";")):
                    self.assertNotIn(duplicate, seen)
                    seen[duplicate] = name
                if row["has_violation"] == "1":
                    self.assertGreaterEqual(float(row["timecode_sec"]), 0.0)
        for row in lc.read_csv(lc.SPLIT_DIR / "excluded.csv"):
            self.assertNotIn(row["sha256"], seen)


@unittest.skipIf(lc is None, f"ML dependencies missing: {SKIP_REASON}")
class MetricTests(unittest.TestCase):
    def test_metrics_and_timecode_error_only_on_true_positives(self):
        rows = [
            {"actual": 1, "predicted": 1, "true_timecode_sec": 10.0, "predicted_timecode_sec": 12.0},
            {"actual": 1, "predicted": 0, "true_timecode_sec": 5.0, "predicted_timecode_sec": 99.0},
            {"actual": 0, "predicted": 1, "true_timecode_sec": 0.0, "predicted_timecode_sec": 3.0},
            {"actual": 0, "predicted": 0, "true_timecode_sec": 0.0, "predicted_timecode_sec": 0.0},
        ]
        m = lc.video_metrics(rows)
        self.assertEqual((m["tp"], m["fp"], m["fn"], m["tn"]), (1, 1, 1, 1))
        self.assertEqual(m["mean_timecode_error_sec"], 2.0)
        self.assertEqual(m["localized_true_positives"], 1)

    def test_no_true_positive_gives_null_error(self):
        m = lc.video_metrics([{"actual": 1, "predicted": 0, "true_timecode_sec": 1.0, "predicted_timecode_sec": 0.0}])
        self.assertIsNone(m["mean_timecode_error_sec"])

    def test_threshold_rule_prefers_f1_then_recall(self):
        threshold, metrics = lc.choose_threshold([0.9, 0.8, 0.3, 0.2], [1, 1, 0, 0])
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(threshold, 0.8)

    def test_aggregation_uses_max_and_its_time(self):
        score, when = lc.aggregate(np.array([0.0, 0.5, 1.0]), np.array([0.1, 0.7, 0.4]))
        self.assertEqual((score, when), (0.7, 0.5))


@unittest.skipIf(lc is None, f"ML dependencies missing: {SKIP_REASON}")
class HeadTests(unittest.TestCase):
    def test_head_state_roundtrip(self):
        rng = np.random.default_rng(0)
        head = lc.LinearHead(rng.normal(size=4).astype("f4"), np.ones(4, "f4") * 2, rng.normal(size=4).astype("f4"), 0.3)
        restored = lc.LinearHead.from_state(head.state())
        x = rng.normal(size=(5, 4)).astype("f4")
        np.testing.assert_allclose(head.probabilities(x), restored.probabilities(x), rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
