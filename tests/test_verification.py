import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pipeline.verification import CONDITIONS, apply_verification
from pipeline.vlm import LocalVlmEngine, Sample, parse_report, to_result
from tests.test_vlm import clean_report, event


class VerificationTests(unittest.TestCase):
    def proposed(self, kind):
        raw = clean_report()
        event(raw, kind).update(status="observed", start_frame=1, end_frame=3,
                               evidence_ids=[1, 3], description="Detector candidate")
        return parse_report(json.dumps(raw), 20)

    def facts(self, kind, first="visible", second="visible"):
        return {"checks": [{"type": kind, "facts": [
            {"condition": CONDITIONS[kind][0], "status": first, "frame_ids": [1], "description": "Context"},
            {"condition": CONDITIONS[kind][1], "status": second, "frame_ids": [12, 15], "description": "Event"},
        ]}]}

    def test_leaving_without_customer_cannot_get_service_rule(self):
        kind = "cashier_left_during_service"
        facts = self.facts(kind, first="absent")
        result = apply_verification(self.proposed(kind), json.dumps(facts), 20)
        self.assertEqual(next(c for c in result.checks if c.type == kind).status, "not_observed")
        self.assertEqual(to_result(result, []).verdict, "no_violation")

    def test_unknown_customer_service_remains_unresolved(self):
        kind = "cashier_left_during_service"
        result = apply_verification(self.proposed(kind), json.dumps(self.facts(kind, first="uncertain")), 20)
        with self.assertRaisesRegex(ValueError, "ручная"):
            to_result(result, [])

    def test_personal_item_evidence_and_time_use_placement_not_wearing(self):
        kind = "personal_items_in_cash_area"
        result = apply_verification(self.proposed(kind), json.dumps(self.facts(kind)), 20)
        frames = [Sample(i, float(i), np.zeros((32, 32, 3), dtype=np.uint8)) for i in range(20)]
        detection = to_result(result, frames).violations[0]
        self.assertEqual(detection.timecode_sec, 12.)
        self.assertEqual([f.timestamp_sec for f in detection.evidence_frames], [1., 12., 15.])

    def test_verifier_cannot_add_new_categories_or_omit_conditions(self):
        kind = "cashier_left_during_service"
        invalid = self.facts("personal_items_in_cash_area")
        with self.assertRaisesRegex(ValueError, "exactly"):
            apply_verification(self.proposed(kind), json.dumps(invalid), 20)
        invalid = self.facts(kind)
        invalid["checks"][0]["facts"][0]["condition"] = "employee_is_walking"
        with self.assertRaisesRegex(ValueError, "condition"):
            apply_verification(self.proposed(kind), json.dumps(invalid), 20)

    def test_entire_verification_evidence_is_checked_before_selection(self):
        kind = "personal_items_in_cash_area"
        for ids in ([1, 99, 12, 15], [], [12, 12], [-1, 15]):
            invalid = self.facts(kind)
            invalid["checks"][0]["facts"][1]["frame_ids"] = ids
            with self.assertRaises(ValueError):
                apply_verification(self.proposed(kind), json.dumps(invalid), 20)

    def test_live_engine_verifies_candidate_and_preserves_both_outputs(self):
        kind = "cashier_left_during_service"
        engine = object.__new__(LocalVlmEngine)
        engine.max_frames = 20
        engine.prompt, engine.observation_prompt = "assessment", "observation"
        samples = [Sample(i, float(i), np.zeros((32, 32, 3), dtype=np.uint8)) for i in range(20)]
        meta = {"generation_reached_token_limit": False, "peak_cuda_memory_gb": 1.}
        proposed = self.proposed(kind).model_dump_json()
        verification = json.dumps(self.facts(kind, first="absent"))
        with patch("pipeline.vlm.sample_sequence", return_value=(samples, 20.)), \
                patch.object(engine, "_generate", side_effect=[("An employee leaves.", meta),
                    (proposed, meta), (verification, meta)]) as generate:
            result, _, audit = engine.analyze(Path("never-send-this-name.mov"))
        self.assertEqual(generate.call_count, 3)
        self.assertEqual(to_result(result, samples).verdict, "no_violation")
        self.assertEqual(audit["raw_response"], proposed)
        self.assertEqual(audit["verification_response"], verification)
        self.assertNotIn("never-send-this-name", str(generate.call_args_list))


if __name__ == "__main__":
    unittest.main()
