"""Require visible prerequisites before attaching a catalog-backed rule to an event."""
from __future__ import annotations

from typing import Literal

from pydantic import Field

from app.models import Model
from pipeline.vlm import EventCheck, ObservationReport, strip_fence

CONDITIONS = {
    "cashier_left_during_service": (
        "customer_is_visibly_being_served",
        "employee_is_absent_while_that_customer_waits_for_unfinished_service",
    ),
    "personal_items_in_cash_area": (
        "item_visibly_belongs_to_employee_and_is_not_work_equipment",
        "item_is_placed_or_stored_in_cash_work_area_not_just_worn_or_carried",
    ),
    "previous_cash_not_stored": (
        "previous_customer_transaction_visibly_finishes",
        "cash_from_that_transaction_remains_out_when_next_customer_is_served",
    ),
    "cash_count_without_client": (
        "cash_visibly_belongs_to_identifiable_current_customer",
        "employee_counts_that_cash_while_its_owner_is_absent",
    ),
}


class Fact(Model):
    condition: str
    status: Literal["visible", "absent", "uncertain"]
    frame_ids: list[int] = Field(default_factory=list, max_length=32)
    description: str = Field(min_length=1, max_length=600)


class Verification(Model):
    type: str
    facts: list[Fact] = Field(min_length=2, max_length=2)


class VerificationReport(Model):
    checks: list[Verification] = Field(min_length=1, max_length=4)


def verification_request(report: ObservationReport) -> str | None:
    import json

    candidates = [{"type": c.type, "conditions": CONDITIONS[c.type]}
                  for c in report.checks if c.status == "observed" and c.type in CONDITIONS]
    if not candidates:
        return None
    return (
        "Check each required condition independently against the images. A prior detector proposed these "
        "categories; its proposal is NOT evidence. Do not assume that an employee leaving means unfinished "
        "customer service. An empty customer chair is not proof of a previous customer. Internal counting "
        "is not customer cash acceptance. For clothing, cite an image AFTER the item was actually placed, "
        "not while it was still worn or held. Mark ambiguous objects, ownership, or occluded actions uncertain. "
        "Use visible only when the images positively support the entire condition; absent when the images "
        "contradict it. Include actual supporting frame IDs and a short description. Use no legal citations. "
        "Return only JSON: {\"checks\":[{\"type\":\"category\",\"facts\":[{\"condition\":\"exact condition\","
        "\"status\":\"visible|absent|uncertain\",\"frame_ids\":[0],\"description\":\"visible facts\"}]}]}. "
        "Return exactly two facts per category with the condition names specified below. "
        "Descriptions should be in Russian, at most 15 words each. Cite at most three frames per fact. "
        "Categories and conditions: " + json.dumps(candidates)
    )


def apply_verification(report: ObservationReport, raw: str, sample_count: int) -> ObservationReport:
    verified = VerificationReport.model_validate_json(strip_fence(raw))
    expected = {c.type for c in report.checks if c.status == "observed" and c.type in CONDITIONS}
    if len(verified.checks) != len(expected) or {c.type for c in verified.checks} != expected:
        raise ValueError("Verification must cover exactly the proposed legal categories")
    replacements = {}
    for check in verified.checks:
        required = CONDITIONS[check.type]
        if {f.condition for f in check.facts} != set(required):
            raise ValueError("Verification omitted a required factual condition")
        facts = {f.condition: f for f in check.facts}
        for fact in check.facts:
            if (any(i < 0 or i >= sample_count for i in fact.frame_ids)
                    or len(set(fact.frame_ids)) != len(fact.frame_ids)):
                raise ValueError("Invalid verification frame references")
            if not fact.description.strip() or (fact.status == "visible" and not fact.frame_ids):
                raise ValueError("Visible prerequisite needs actual evidence and explanation")
        description = " ".join(facts[name].description for name in required)
        statuses = [f.status for f in check.facts]
        status = "not_observed" if "absent" in statuses else ("uncertain" if "uncertain" in statuses else "observed")
        # One frame for context, then the actual event and its final visible state.
        context = min(facts[required[0]].frame_ids, default=0)
        event_ids = sorted(facts[required[1]].frame_ids)
        ids = sorted({context, *([event_ids[0], event_ids[-1]] if event_ids else [])})
        if status == "observed" and check.type != "personal_items_in_cash_area" and len(ids) < 2:
            status = "uncertain"
        replacements[check.type] = EventCheck(
            type=check.type, status=status, description=description,
            start_frame=event_ids[0] if status == "observed" else None,
            end_frame=event_ids[-1] if status == "observed" else None,
            evidence_ids=ids if status == "observed" else [],
        )
    return ObservationReport(checks=[replacements.get(c.type, c) for c in report.checks])
