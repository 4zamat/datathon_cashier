import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from app.rules import RuleCatalog
from pipeline.vlm import (EVENT_TYPES, LocalVlmEngine, Sample, check_compiler, parse_report,
                          sample_sequence, to_result, validate_vision_control)
from scripts.vlm_probe import summarize


def clean_report():
    return {"checks": [{"type": kind, "status": "not_observed", "start_frame": None,
            "end_frame": None, "evidence_ids": [], "description": "Рабочее место пустое на всех кадрах."}
            for kind in sorted(EVENT_TYPES)]}


def event(raw, kind):
    return next(v for v in raw["checks"] if v["type"] == kind)


class VlmTests(unittest.TestCase):
    def test_extra_real_evidence_is_reduced_without_changing_event(self):
        raw = clean_report()
        event(raw, "cash_left_unattended").update(status="observed", start_frame=8, end_frame=19,
            evidence_ids=list(range(8, 20)), description="Купюра осталась после ухода сотрудницы.")
        report = parse_report(json.dumps(raw), 20)
        check = next(c for c in report.checks if c.type == "cash_left_unattended")
        self.assertEqual(check.evidence_ids, [8, 14, 19])
        self.assertEqual((check.status, check.start_frame, check.end_frame), ("observed", 8, 19))
        event(raw, "cash_left_unattended")["evidence_ids"].append(50)
        with self.assertRaisesRegex(ValueError, "unsupplied"):
            parse_report(json.dumps(raw), 20)

    def test_extra_evidence_does_not_hide_duplicate_references(self):
        raw = clean_report()
        event(raw, "personal_items_in_cash_area").update(status="observed", start_frame=1, end_frame=5,
            evidence_ids=[1, 2, 2, 3, 5], description="Одежда на рабочем столе")
        with self.assertRaisesRegex(ValueError, "unique"):
            parse_report(json.dumps(raw), 20)

    def test_empty_copied_template_is_rejected(self):
        raw = clean_report()
        for check in raw["checks"]:
            check["description"] = ""
        with self.assertRaisesRegex(ValueError, "visual explanation"):
            parse_report(json.dumps(raw), 20)

    def test_control_requires_correct_image_order_and_colors(self):
        self.assertTrue(validate_vision_control('{"first":"red","second":"blue"}'))
        self.assertFalse(validate_vision_control('{"first":"blue","second":"red"}'))
        self.assertFalse(validate_vision_control('{"first":"red","second":"red"}'))
        self.assertFalse(validate_vision_control('I cannot see images.'))

    def test_observation_precedes_assessment_without_video_filename(self):
        engine = object.__new__(LocalVlmEngine)
        engine.max_frames = 20
        engine.prompt, engine.observation_prompt = "assessment prompt", "observation prompt"
        frames = [Sample(i, float(i), np.zeros((32, 32, 3), dtype=np.uint8)) for i in range(3)]
        metadata = {"generation_reached_token_limit": False, "peak_cuda_memory_gb": 1.}
        with patch("pipeline.vlm.sample_sequence", return_value=(frames, 3.)), \
                patch.object(engine, "_generate", side_effect=[("The desk is empty.", metadata),
                    (json.dumps(clean_report()), metadata)]) as generate:
            report, _, result = engine.analyze(Path("label-positive-40.mov"))
        self.assertEqual(len(report.checks), 5)
        self.assertEqual(result["observation_response"], "The desk is empty.")
        self.assertIn("The desk is empty.", generate.call_args_list[1].args[2])
        self.assertNotIn("label-positive", str(generate.call_args_list))

    def test_missing_linux_compiler_fails_before_model_load(self):
        with patch("pipeline.vlm.sys.platform", "linux"), patch("pipeline.vlm.shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "build-essential"):
                check_compiler()

    def test_zero_predictions_are_not_reported_as_measured_zero_quality(self):
        summary = summarize([{"actual": 1, "predicted_visual": None, "status": "failed"}], "smoke", 6)
        self.assertIsNone(summary["precision_on_scored"])
        self.assertIsNone(summary["recall_on_all_known_positives"])
        self.assertIsNone(summary["f1_on_scored"])

    def test_absence_is_valid_and_uncertainty_is_not_clean(self):
        raw = clean_report()
        self.assertEqual(to_result(parse_report(json.dumps(raw), 3), []).verdict, "no_violation")
        event(raw, "personal_items_in_cash_area").update(status="uncertain", description="Не видно владельца")
        with self.assertRaisesRegex(ValueError, "ручная"):
            to_result(parse_report(json.dumps(raw), 3), [])

    def test_invalid_json_unknown_type_and_hallucinated_frame_rejected(self):
        with self.assertRaises(ValueError):
            parse_report('I think everything is fine.', 3)
        raw = clean_report()
        raw["checks"][0]["type"] = "theft"
        with self.assertRaises(ValueError):
            parse_report(json.dumps(raw), 3)
        raw = clean_report()
        event(raw, "personal_items_in_cash_area").update(status="observed", start_frame=1, end_frame=10,
            evidence_ids=[1, 10], description="Предмет")
        with self.assertRaisesRegex(ValueError, "unsupplied"):
            parse_report(json.dumps(raw), 3)

    def test_real_evidence_time_and_rule_come_from_inputs_and_catalog(self):
        raw = clean_report()
        event(raw, "previous_cash_not_stored").update(status="observed", start_frame=1, end_frame=2,
            evidence_ids=[0, 2], description="Деньги остались после смены клиента")
        frames = [Sample(i, t, np.zeros((32, 32, 3), dtype=np.uint8)) for i,t in enumerate([0., 4., 8.])]
        result = to_result(parse_report(json.dumps(raw), 3), frames)
        self.assertEqual(result.violations[0].timecode_sec, 4.)
        self.assertEqual([f.timestamp_sec for f in result.violations[0].evidence_frames], [0., 8.])
        enriched = RuleCatalog(Path(__file__).resolve().parents[1] / "rules/rules.yaml").enrich(result)
        self.assertEqual(enriched[0].rule.clause, "п. 20, абзац 2")

    def test_unattended_cash_is_not_assigned_unproven_rule(self):
        raw = clean_report()
        event(raw, "cash_left_unattended").update(status="observed", start_frame=1, end_frame=2,
            evidence_ids=[0, 2], description="Деньги оставлены")
        with self.assertRaisesRegex(ValueError, "ручная"):
            to_result(parse_report(json.dumps(raw), 3), [])

    def test_temporal_event_requires_sequence(self):
        raw = clean_report()
        event(raw, "cash_count_without_client").update(status="observed", start_frame=1, end_frame=2,
            evidence_ids=[2], description="Клиента нет")
        with self.assertRaisesRegex(ValueError, "before/after"):
            parse_report(json.dumps(raw), 3)

    def test_abstentions_are_visible_in_metrics(self):
        rows = [{"actual": 1, "predicted_visual": None, "status": "failed"},
                {"actual": 1, "predicted_visual": 1, "status": "done"},
                {"actual": 0, "predicted_visual": 0, "status": "done"}]
        measured = summarize(rows, "smoke", 6)
        self.assertEqual(measured["recall_on_all_known_positives"], .5)
        self.assertEqual(measured["abstentions_or_errors_with_known_label"], 1)
        self.assertFalse(measured["all_selected_processed_without_errors"])

    def test_sampling_bounded_and_independent_of_filename(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "unknown.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10., (64, 96))
            self.assertTrue(writer.isOpened())
            for i in range(60):
                writer.write(np.full((96, 64, 3), i * 4, dtype=np.uint8))
            writer.release()
            frames, duration = sample_sequence(path, 8)
            renamed = path.with_name("positive-label-but-ignored.avi")
            path.rename(renamed)
            other, _ = sample_sequence(renamed, 8)
            self.assertEqual(len(frames), 8)
            self.assertAlmostEqual(duration, 6.)
            self.assertEqual([f.timestamp_sec for f in frames], [f.timestamp_sec for f in other])
            self.assertTrue(all(a.timestamp_sec < b.timestamp_sec for a,b in zip(frames, frames[1:])))
            np.testing.assert_array_equal(frames[-1].bgr, other[-1].bgr)


if __name__ == "__main__":
    unittest.main()
