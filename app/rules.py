from pathlib import Path

import yaml
from pydantic import TypeAdapter

from app.models import NonemptyText, RuleReference, Violation
from pipeline.base import Result


class RuleDefinition(RuleReference):
    summary: NonemptyText


class RuleCatalog:
    def __init__(self, path: Path) -> None:
        with path.open(encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
        self._rules = TypeAdapter(dict[str, RuleDefinition]).validate_python(raw)
        if not self._rules:
            raise ValueError("Rules catalog must not be empty")

    def enrich(self, result: Result) -> list[Violation]:
        violations = []
        for detection in result.violations:
            if detection.type not in self._rules:
                raise ValueError(f"Unknown violation type in rules catalog: {detection.type}")
            definition = self._rules[detection.type]
            payload = detection.model_dump()
            if detection.type != "demo_violation":
                payload["description"] += f" Нормативное основание: {definition.document}, {definition.clause}."
            violations.append(Violation(
                **payload,
                rule=RuleReference(document=definition.document, clause=definition.clause),
            ))
        return violations
