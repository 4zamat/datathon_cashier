"""Versioned annotations in source-frame coordinates."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)


class Event(Record):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    start_sec: float = Field(ge=0)
    end_sec: float = Field(ge=0)
    event_type: str = "unknown"
    assessment: Literal["possible_violation", "no_violation", "uncertain"] = "uncertain"
    observation: str = Field(min_length=1, max_length=10000)

    @model_validator(mode="after")
    def ordered(self) -> "Event":
        if self.end_sec < self.start_sec:
            raise ValueError("Конец события должен быть не раньше начала")
        if self.assessment == "possible_violation" and self.event_type in {"normal", "unknown"}:
            raise ValueError("Для признаков нарушения выберите конкретный тип события")
        if self.event_type == "normal" and self.assessment != "no_violation":
            raise ValueError("Нормальный фрагмент требует оценки «Нарушений не видно»")
        return self


LEGACY_YOLO_CLASSES = ["person", "cash", "hand", "phone", "tray", "storage"]
YOLO_CLASSES = [*LEGACY_YOLO_CLASSES, "laptop", "cash_counter", "banknote_detector", "bag"]
OBJECT_CLASSES = [*YOLO_CLASSES, "other"]
EQUIPMENT_CLASSES = ["storage", "laptop", "cash_counter", "banknote_detector"]
REVIEW_PROFILES = {"equipment": EQUIPMENT_CLASSES, "all": YOLO_CLASSES}


class Box(Record):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    label: Literal["person", "cash", "hand", "phone", "tray", "storage", "other",
                   "laptop", "cash_counter", "banknote_detector", "bag"]
    x: float = Field(ge=0, lt=1)
    y: float = Field(ge=0, lt=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)
    occluded: bool = False
    comment: str = Field(default="", max_length=3000)
    annotator: str = Field(default="", max_length=200)
    source: Literal["human", "assistant"] = "human"

    @model_validator(mode="after")
    def inside_frame(self) -> "Box":
        if self.x + self.width > 1.000001 or self.y + self.height > 1.000001:
            raise ValueError("Рамка выходит за границы кадра")
        return self


class Frame(Record):
    id: str = Field(pattern=r"^f[0-9]+$")
    frame_index: int = Field(ge=0)
    timestamp_sec: float = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    reviewed: bool = False
    boxes: list[Box] = Field(default_factory=list, max_length=500)
    reviewed_classes: list[str] = Field(default_factory=list, max_length=20)
    reviewed_by: str = Field(default="", max_length=200)
    annotation_notes: str = Field(default="", max_length=5000)

    @model_validator(mode="after")
    def confirmed_classes(self) -> "Frame":
        if any(name not in YOLO_CLASSES for name in self.reviewed_classes):
            raise ValueError("Неизвестный класс в области проверки кадра")
        if len(self.reviewed_classes) != len(set(self.reviewed_classes)):
            raise ValueError("Повторяющиеся классы в области проверки кадра")
        if self.reviewed and not self.reviewed_classes:
            # A legacy check covered six classes; adding classes must not silently expand it.
            self.reviewed_classes = list(LEGACY_YOLO_CLASSES)
        if self.reviewed and set(self.reviewed_classes) != set(EQUIPMENT_CLASSES) and any(box.label == "other" for box in self.boxes):
            raise ValueError("Уточните класс «Другое» перед подтверждением кадра для YOLO")
        return self


class EventDraft(Record):
    id: str = ""
    start: str = Field(default="", max_length=40)
    end: str = Field(default="", max_length=40)
    event_type: str = "unknown"
    assessment: Literal["possible_violation", "no_violation", "uncertain"] = "uncertain"
    observation: str = Field(default="", max_length=10000)


class AnnotationDocument(Record):
    schema_version: Literal[1] = 1
    video_id: str = Field(pattern=r"^[a-f0-9]{20}$")
    video: str
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    revision: int = Field(default=0, ge=0)
    updated_at: str = ""
    reviewer: str = Field(default="", max_length=200)
    scene_group_id: str = Field(default="", max_length=200)
    layout_group_id: str = Field(default="", max_length=200)
    layout_notes: str = Field(default="", max_length=2000)
    watched_full: bool = False
    notes: str = Field(default="", max_length=20000)
    events: list[Event] = Field(default_factory=list, max_length=1000)
    frames: list[Frame] = Field(default_factory=list, max_length=1000)
    event_draft: EventDraft | None = None


class CaptureRequest(Record):
    timestamp_sec: float = Field(ge=0)
