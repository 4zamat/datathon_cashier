import unittest
from pathlib import Path

from app.rules import RuleCatalog
from pipeline.base import Result
from tests.helpers import detection


class VerifiedRulesTests(unittest.TestCase):
    def test_verified_rules_are_enriched_from_catalog(self):
        catalog = RuleCatalog(Path(__file__).resolve().parents[1] / "rules/rules.yaml")
        clauses = {"previous_cash_not_stored": "п. 20, абзац 2", "cash_count_without_client": "п. 20, абзац 1",
                   "cashier_left_during_service": "п. 22", "personal_items_in_cash_area": "п. 12"}
        for kind, clause in clauses.items():
            with self.subTest(kind=kind):
                event = detection().model_copy(update={"type": kind})
                result = catalog.enrich(Result(violations=[event]))[0]
                self.assertEqual(result.rule.clause, clause)
                self.assertIn("29.11.2019 № 231", result.rule.document)
                self.assertIn(clause, result.description)
                self.assertIn(result.rule.document, result.description)


if __name__ == "__main__":
    unittest.main()
