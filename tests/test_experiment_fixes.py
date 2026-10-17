import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts import laptop_common as lc
from scripts import prepare_laptop_data as prep
from scripts.export_experiment import MODEL_FILES, RUN_FILES, SPLIT_FILES, export_experiment, verify_archive
from scripts.train_laptop import frame_dataset
from scripts.validate_annotations import validate
from scripts.verify_laptop_model import compare_prediction


class ExperimentFixTests(unittest.TestCase):
    def test_cached_and_fresh_features_are_bitwise_equal(self):
        class Extractor:
            cache_identity = lc.CACHE_VERSION

            def extract(self, path):
                return np.array([0.0, 0.5]), np.array([[0.1234567], [1.1234567]], dtype=np.float32)

        with tempfile.TemporaryDirectory() as directory, patch.object(lc, "FEATURE_DIR", Path(directory)):
            a = lc.cached_features(Extractor(), Path("unused"), "hash")
            with patch.object(Extractor, "extract", side_effect=AssertionError("must hit cache")):
                b = lc.cached_features(Extractor(), Path("unused"), "hash")
            np.testing.assert_array_equal(a[1], b[1])
            self.assertEqual(b[2], 0)

    def test_mixed_label_scene_never_crosses_split(self):
        rows = [{"group_id": f"scene{i}", "has_violation": label, "sha256": f"{i}-{label}"}
                for i in range(15) for label in (0, 1)]
        result = lc.stratified_group_split(rows)
        self.assertEqual(result, lc.stratified_group_split(rows))
        seen = {}
        self.assertEqual(sum(map(len, result.values())), 30)
        for split, members in result.items():
            self.assertEqual({r["has_violation"] for r in members}, {0, 1})
            for row in members:
                self.assertEqual(seen.setdefault(row["group_id"], split), split)

    def test_stored_timecode_mismatch_is_rejected(self):
        a = {"score": 0.8, "predicted": 1, "predicted_timecode_sec": 5.0}
        ref = {"score": "0.8", "predicted": "1", "predicted_timecode_sec": "9.0"}
        self.assertFalse(compare_prediction(a, a, ref)["stored_timecode_matches"])

    def test_frame_weights_balance_classes_and_videos(self):
        videos = [{"video": "a", "has_violation": 1, "timecode_sec": 1},
                  {"video": "b", "has_violation": 0}, {"video": "c", "has_violation": 0}]
        store = {"a": (np.arange(3), np.zeros((3, 4))), "b": (np.arange(2), np.zeros((2, 4))),
                 "c": (np.arange(10), np.zeros((10, 4)))}
        _, y, weights = frame_dataset(videos, store)
        np.testing.assert_allclose([weights[y == 1].sum(), weights[3:5].sum(), weights[5:].sum()],
                                   [0.5, 0.25, 0.25], atol=1e-6)

    def test_two_conflicting_contents_count_as_two_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, checks = [], []
            for i in range(10):
                content = str(i // 2 if i < 4 else i).encode()
                video = f"v{i}.mov"
                (root / video).write_bytes(content)
                rows.append({"video": video, "has_violation": i % 2, "timecode_sec": "1" if i % 2 else ""})
                checks.append({"video": video, "ok": True, "duration_sec": 10, "sha256": lc.sha256_file(root / video)})
            lc.write_csv(root / "labels.csv", ["video", "has_violation", "timecode_sec"], rows)
            lc.dump_json(root / "check.json", {"results": checks})
            with patch.object(prep, "DATA_DIR", root):
                result = prep.prepare(root / "labels.csv", root / "check.json", 42, root / "splits")
            self.assertEqual(result["excluded_groups"], 2)
            self.assertEqual(result["excluded_videos"], 4)
            with self.assertRaises(ValueError):
                prep.prepare(root / "labels.csv", root / "check.json", 42, root / "splits")

    def test_preprocessing_changes_fail_instead_of_silently_loading(self):
        config = {"preprocessing": {"color": "BGR->RGB", "resize": 256, "center_crop": 224,
                                    "transform": "ResNet18_Weights.IMAGENET1K_V1.transforms()",
                                    "normalize": "ImageNet mean/std", "sample_fps": 2},
                  "encoder": {"arch": "resnet18", "feature_dim": 512, "file": "encoder_resnet18.pt"},
                  "head": {"file": "head.npz"}, "threshold": 0.5}
        lc.validate_model_config(config)
        config["preprocessing"]["center_crop"] = 112
        with self.assertRaises(ValueError):
            lc.validate_model_config(config)

    def test_annotation_rejects_time_beyond_video(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lc.dump_json(root / "check.json", {"results": [{"video": "v.mov", "ok": True, "duration_sec": 10}]})
            (root / "rules.yaml").write_text("event: {}", encoding="utf-8")
            row = dict(video="v.mov", event_id="1", start_sec=2, end_sec=11, event_type="event",
                       observation="visible action", assessment="possible_violation", reviewer="a", scene_group_id="s1")
            lc.write_csv(root / "events.csv", list(row), [row])
            with self.assertRaises(ValueError):
                validate(root / "events.csv", root / "check.json", root / "rules.yaml")
            row["end_sec"] = 10
            lc.write_csv(root / "events.csv", list(row), [row])
            self.assertEqual(validate(root / "events.csv", root / "check.json", root / "rules.yaml")["videos_reviewed"], 1)


class ArtifactTests(unittest.TestCase):
    def test_export_is_complete_and_does_not_include_private_inputs_or_env(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder, files in (("model", MODEL_FILES), ("run", RUN_FILES), ("splits", SPLIT_FILES)):
                (root / folder).mkdir()
                for name in files:
                    (root / folder / name).write_text("{}" if name.endswith("json") else "fixture", encoding="utf-8")
            (root / ".env").write_text("SECRET=never-copy")
            (root / "video.mov").write_bytes(b"private video")
            archive = root / "export.zip"
            result = export_experiment(root, root / "model", root / "run", root / "splits", archive)
            self.assertTrue(result["complete"])
            self.assertEqual(verify_archive(archive), result)
            with zipfile.ZipFile(archive) as z:
                self.assertNotIn(".env", z.namelist())
                self.assertNotIn("video.mov", z.namelist())
            with self.assertRaises(FileExistsError):
                export_experiment(root, root / "model", root / "run", root / "splits", archive)

    def test_partial_backup_is_saved_and_marked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = export_experiment(root, root / "model", root / "run", root / "splits", root / "partial.zip")
            self.assertFalse(result["complete"])
            self.assertIn("model/head.npz", result["missing_required"])
            self.assertFalse(verify_archive(root / "partial.zip")["complete"])

    def test_checksum_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("head.npz", "changed")
                archive.writestr("manifest.json", json.dumps({"files": {"head.npz": {"bytes": 7, "sha256": "wrong"}}}))
            with self.assertRaises(ValueError):
                verify_archive(path)


if __name__ == "__main__":
    unittest.main()
