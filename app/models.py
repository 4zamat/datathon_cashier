"""Validated API and pipeline contracts."""

import base64
import binascii
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator

Seconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]
NonemptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
JobId = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]
Verdict = Literal["violation", "no_violation"]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EvidenceFrame(Model):
    timestamp_sec: Seconds
    image_base64: NonemptyText

    @field_validator("image_base64")
    @classmethod
    def jpeg_base64(cls, value: str) -> str:
        try:
            image = base64.b64decode(value, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("Evidence must be valid base64") from exc
        if not image.startswith(b"\xff\xd8\xff") or not image.endswith(b"\xff\xd9"):
            raise ValueError("Evidence must be JPEG, without a data URL prefix")
        return value


class Detection(Model):
    """Pipeline output has no rule field; only the catalog can supply it."""

    type: NonemptyText
    timecode_sec: Seconds
    description: NonemptyText
    evidence_frames: list[EvidenceFrame] = Field(min_length=1, max_length=3)


class RuleReference(Model):
    document: NonemptyText
    clause: NonemptyText


class Violation(Detection):
    rule: RuleReference


class AcceptedJob(Model):
    job_id: JobId


class Job(AcceptedJob):
    status: Literal["processing", "done", "failed"] = "processing"
    # Required wire enum; a verdict is meaningful only when status == "done".
    verdict: Verdict = "no_violation"
    violations: list[Violation] = Field(default_factory=list)
    error: str | None = None

    @model_validator(mode="after")
    def consistent_result(self) -> Self:
        if self.status == "done":
            expected = "violation" if self.violations else "no_violation"
            if self.verdict != expected or self.error is not None:
                raise ValueError("Completed job must have a consistent verdict and no error")
        elif self.verdict != "no_violation" or self.violations:
            raise ValueError("An unfinished/failed job must use the placeholder verdict and no violations")
        return self
