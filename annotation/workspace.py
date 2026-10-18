"""Video catalog, atomic annotation storage, frame capture and portable export."""

import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import threading
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import yaml
from fastapi import HTTPException

from annotation.models import AnnotationDocument, Frame, OBJECT_CLASSES, YOLO_CLASSES, EQUIPMENT_CLASSES

ROOT = Path(__file__).resolve().parents[1]
EVENT_NAMES = {
    "cashier_left_during_service": "Уход во время обслуживания",
    "previous_cash_not_stored": "Ранее принятые деньги не убраны",
    "cash_count_without_client": "Пересчёт без клиента",
    "personal_items_in_cash_area": "Личные вещи в кассовой зоне",
    "normal": "Нормальный фрагмент",
    "unknown": "Неизвестно / нужна проверка",
}
EVENT_FIELDS = ["video", "event_id", "start_sec", "end_sec", "event_type", "observation",
                "assessment", "reviewer", "scene_group_id"]


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def csv_bytes(fields: list[str], rows: list[dict]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")


class Workspace:
    def __init__(self, data_dir: Path, rules_path: Path = ROOT / "rules/rules.yaml") -> None:
        self.data_dir = data_dir.resolve()
        self.output = self.data_dir / "annotations/local"
        self.cache = self.data_dir / "annotation_cache"
        self.lock = threading.RLock()
        self.media_lock = threading.Lock()
        self.rules = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
        self.rules.pop("demo_violation", None)
        self.videos: dict[str, dict] = {}
        self.paths: dict[str, Path] = {}
        self.fingerprints: dict[str, tuple[int, int]] = {}
        self.scan_errors: list[str] = []
        check_path = self.data_dir / "dataset_check.json"
        checked = {}
        if check_path.exists():
            try:
                checked = {row["video"]: row for row in json.loads(check_path.read_text(encoding="utf-8"))["results"]}
            except (ValueError, KeyError, TypeError):
                self.scan_errors.append("dataset_check.json не удалось прочитать; длительности определены по видео")
        for path in sorted((self.data_dir / "videos").glob("*"), key=self.sort_key):
            if path.suffix.lower() not in {".mov", ".mp4"} or not path.is_file():
                continue
            if not path.resolve().is_relative_to(self.data_dir / "videos"):
                continue
            relative = path.relative_to(self.data_dir).as_posix()
            video_id = hashlib.sha256(relative.encode()).hexdigest()[:20]
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            capture = cv2.VideoCapture(str(path))
            try:
                fps = capture.get(cv2.CAP_PROP_FPS)
                count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                ok, frame = capture.read()
                if not ok or not math.isfinite(fps) or fps <= 0 or count <= 0:
                    self.scan_errors.append(f"Не удалось открыть {path.name}")
                    continue
                height, width = frame.shape[:2]
            finally:
                capture.release()
            self.paths[video_id] = path
            stat = path.stat()
            self.fingerprints[video_id] = (stat.st_size, stat.st_mtime_ns)
            reference = checked.get(relative, {})
            duration = count / fps
            if reference.get("sha256") == digest and reference.get("ok"):
                duration = float(reference["duration_sec"])
            self.videos[video_id] = {
                "id": video_id, "video": relative, "name": path.name, "sha256": digest,
                "duration_sec": duration, "fps": fps, "frame_count": count,
                "width": width, "height": height,
            }

    @staticmethod
    def sort_key(path: Path) -> tuple[int, str]:
        match = re.search(r"(\d+)", path.stem)
        return (int(match[1]) if match else 999999, path.name)

    def info(self, video_id: str) -> dict:
        if video_id not in self.videos:
            raise HTTPException(404, "Видео не найдено")
        try:
            stat = self.paths[video_id].stat()
            fingerprint = (stat.st_size, stat.st_mtime_ns)
            if fingerprint != self.fingerprints[video_id]:
                with self.paths[video_id].open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if digest != self.videos[video_id]["sha256"]:
                    raise HTTPException(409, "Исходное видео изменилось. Перезапустите сервер и проверьте файл")
                self.fingerprints[video_id] = fingerprint
        except OSError as exc:
            raise HTTPException(409, "Исходное видео недоступно") from exc
        return self.videos[video_id]

    def path(self, video_id: str) -> Path:
        self.info(video_id)
        return self.output / "documents" / f"{video_id}.json"

    def load(self, video_id: str) -> AnnotationDocument:
        info = self.info(video_id)
        path = self.path(video_id)
        with self.lock:
            if path.exists():
                try:
                    document = AnnotationDocument.model_validate_json(path.read_bytes())
                except (ValueError, OSError) as exc:
                    raise HTTPException(409, "Сохранённая разметка повреждена. Проверьте резервные копии") from exc
                if document.source_sha256 != info["sha256"]:
                    raise HTTPException(409, "Видео изменилось после разметки; сначала проверьте исходный файл")
                return document
        return AnnotationDocument(video_id=video_id, video=info["video"], source_sha256=info["sha256"])

    def save(self, video_id: str, document: AnnotationDocument) -> AnnotationDocument:
        info = self.info(video_id)
        if (document.video_id, document.video, document.source_sha256) != (
            video_id, info["video"], info["sha256"]
        ):
            raise HTTPException(422, "Разметка относится к другому видео")
        if (document.events or document.frames or document.watched_full) and not document.reviewer:
            raise HTTPException(422, "Укажите, кто выполняет разметку")
        self.unique([event.id for event in document.events])
        self.unique([frame.id for frame in document.frames])
        for event in document.events:
            if event.end_sec > info["duration_sec"] or event.event_type not in {*self.rules, "normal", "unknown"}:
                raise HTTPException(422, "Неизвестный тип события или время за пределами видео")
        for frame in document.frames:
            stored = self.frame_info(video_id, frame.id)
            if any(getattr(frame, key) != stored[key] for key in ("frame_index", "timestamp_sec", "width", "height")):
                raise HTTPException(422, "Параметры кадра не совпадают с исходным изображением")
            self.unique([box.id for box in frame.boxes])
        with self.lock:
            current = self.load(video_id)
            if current.revision != document.revision:
                raise HTTPException(409, "Это видео изменено в другой вкладке. Скачайте свой черновик и обновите страницу")
            path = self.path(video_id)
            if path.exists():
                backup = self.output / "history" / video_id / f"r{current.revision}-{uuid4().hex[:8]}.json"
                atomic_write(backup, path.read_bytes())
            saved = document.model_copy(update={
                "revision": current.revision + 1, "updated_at": datetime.now(timezone.utc).isoformat(),
            })
            atomic_write(path, saved.model_dump_json(indent=2).encode("utf-8"))
            return saved

    @staticmethod
    def unique(ids: list[str]) -> None:
        if len(ids) != len(set(ids)):
            raise HTTPException(422, "Повторяющийся идентификатор события, кадра или рамки")

    def frame_path(self, video_id: str, frame_id: str, suffix: str) -> Path:
        info = self.info(video_id)
        if not re.fullmatch(r"f[0-9]+", frame_id):
            raise HTTPException(404, "Кадр не найден")
        return self.output / "frames" / video_id / info["sha256"] / f"{frame_id}.{suffix}"

    def frame_info(self, video_id: str, frame_id: str) -> dict:
        path = self.frame_path(video_id, frame_id, "json")
        if not path.is_file():
            raise HTTPException(422, "Сначала зафиксируйте кадр из видео")
        return json.loads(path.read_text(encoding="utf-8"))

    def capture(self, video_id: str, timestamp: float) -> dict:
        info = self.info(video_id)
        if timestamp >= info["duration_sec"]:
            timestamp = max(0, info["duration_sec"] - 1 / info["fps"])
        capture = cv2.VideoCapture(str(self.paths[video_id]))
        try:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, image = capture.read()
            if not ok:
                raise HTTPException(422, "Не удалось извлечь кадр. Попробуйте чуть более ранний момент")
            index = max(0, int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
            actual_time = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if not math.isfinite(actual_time) or (actual_time <= 0 and index > 0):
                actual_time = index / info["fps"]
            height, width = image.shape[:2]
            frame = Frame(id=f"f{index}", frame_index=index, timestamp_sec=actual_time, width=width, height=height)
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                raise HTTPException(500, "Не удалось сохранить JPEG")
            with self.lock:
                # Once created, a frame and its coordinates retain exactly the same reference image.
                if not self.frame_path(video_id, frame.id, "json").exists():
                    atomic_write(self.frame_path(video_id, frame.id, "jpg"), encoded.tobytes())
                    atomic_write(self.frame_path(video_id, frame.id, "json"), json_bytes(frame.model_dump()))
                return self.frame_info(video_id, frame.id)
        finally:
            capture.release()

    def media(self, video_id: str) -> tuple[Path, str]:
        info = self.info(video_id)
        source = self.paths[video_id]
        ffmpeg = shutil.which("ffmpeg")
        if source.suffix.lower() == ".mp4" or not ffmpeg:
            return source, "video/mp4" if source.suffix.lower() == ".mp4" else "video/quicktime"
        target = self.cache / f"{info['sha256']}.mp4"
        with self.media_lock:
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(f"{target.stem}-{uuid4().hex}.mp4")
                try:
                    result = subprocess.run(
                        [ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(source), "-map", "0:v:0",
                         "-an", "-c:v", "copy", "-movflags", "+faststart", str(temporary)],
                        capture_output=True, timeout=120, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                    if result.returncode:
                        raise HTTPException(422, "Не удалось подготовить MP4 для браузера; проверьте кодек видео")
                    os.replace(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)
        return target, "video/mp4"

    def export(self) -> Path:
        """Export a consistent snapshot; drafts and checked YOLO frames remain distinguishable."""
        export_dir = self.output / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        destination = export_dir / f"annotations-{stamp}-{uuid4().hex[:6]}.zip"
        documents = []
        events, drafts, reviews, objects = [], [], [], []
        warnings: list[str] = []
        with self.lock:
            for video_id in self.videos:
                document = self.load(video_id)
                if not document.revision:
                    continue
                documents.append(document)
                reviews.append({"video": document.video, "watched_full": document.watched_full,
                                "reviewer": document.reviewer, "scene_group_id": document.scene_group_id,
                                "layout_group_id": document.layout_group_id, "layout_notes": document.layout_notes,
                                "notes": document.notes, "source_sha256": document.source_sha256})
                for event in document.events:
                    row = {"video": document.video, "event_id": event.id,
                           **event.model_dump(exclude={"id"}), "reviewer": document.reviewer,
                           "scene_group_id": document.scene_group_id}
                    (events if document.scene_group_id else drafts).append(row)
                if document.events and not document.scene_group_id:
                    warnings.append(f"{document.video}: укажите группу сцены для экспорта в events.csv")
            if not documents:
                raise HTTPException(422, "Пока нет сохранённой разметки")
            with zipfile.ZipFile(destination, "x", zipfile.ZIP_DEFLATED) as archive:
                for document in documents:
                    archive.writestr(f"documents/{document.video_id}.json", json_bytes(document.model_dump()))
                    for frame in document.frames:
                        name = f"{document.video_id}_{frame.id}"
                        archive.write(self.frame_path(document.video_id, frame.id, "jpg"), f"frames/{name}.jpg")
                        for box in frame.boxes:
                            objects.append({"video": document.video, "frame_id": frame.id,
                                            "timestamp_sec": frame.timestamp_sec, "image": f"frames/{name}.jpg",
                                            "image_width": frame.width, "image_height": frame.height,
                                            "frame_reviewed": frame.reviewed, "reviewed_classes": "|".join(frame.reviewed_classes),
                                            "reviewed_by": frame.reviewed_by, **box.model_dump()})
                        for prefix, classes in (("yolo", YOLO_CLASSES), ("equipment_yolo", EQUIPMENT_CLASSES)):
                            if frame.reviewed and set(classes).issubset(frame.reviewed_classes):
                                archive.write(self.frame_path(document.video_id, frame.id, "jpg"), f"{prefix}/images/{name}.jpg")
                                lines = [f"{classes.index(b.label)} {b.x+b.width/2:.8f} {b.y+b.height/2:.8f} "
                                         f"{b.width:.8f} {b.height:.8f}" for b in frame.boxes if b.label in classes]
                                archive.writestr(f"{prefix}/labels/{name}.txt", "\n".join(lines) + ("\n" if lines else ""))
                archive.writestr("events.csv", csv_bytes(EVENT_FIELDS, events))
                archive.writestr("events_draft.csv", csv_bytes(EVENT_FIELDS, drafts))
                archive.writestr("video_reviews.csv", csv_bytes(
                    ["video", "watched_full", "reviewer", "scene_group_id", "layout_group_id", "layout_notes", "notes", "source_sha256"], reviews))
                archive.writestr("objects.csv", csv_bytes(
                    ["video", "frame_id", "timestamp_sec", "image", "image_width", "image_height", "frame_reviewed",
                     "reviewed_classes", "reviewed_by", "id", "label", "x", "y", "width", "height", "occluded", "comment", "annotator", "source"], objects))
                archive.writestr("yolo/classes.txt", "\n".join(YOLO_CLASSES) + "\n")
                archive.writestr("equipment_yolo/classes.txt", "\n".join(EQUIPMENT_CLASSES) + "\n")
                archive.writestr("rules_reference.json", json_bytes(self.rules))
                archive.writestr("manifest.json", json_bytes({
                    "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                    "videos_saved": len(documents), "events_exported": len(events), "events_draft": len(drafts),
                    "unfinished_event_forms": sum(d.event_draft is not None for d in documents),
                    "yolo_reviewed_frames": sum(f.reviewed and set(YOLO_CLASSES).issubset(f.reviewed_classes) for d in documents for f in d.frames),
                    "equipment_reviewed_frames": sum(f.reviewed and set(EQUIPMENT_CLASSES).issubset(f.reviewed_classes) for d in documents for f in d.frames),
                    "object_classes": OBJECT_CLASSES, "yolo_classes": YOLO_CLASSES,
                    "equipment_classes": EQUIPMENT_CLASSES, "warnings": warnings,
                    "coordinate_system": "normalized top-left x,y,width,height in exported JPEG; YOLO uses center x,y",
                    "training_ready": False,
                }))
                archive.writestr("README.txt", (
                    "Ручная разметка, не итоговый обучающий датасет. Исходные видео и labels.csv не изменялись.\n"
                    "events.csv совместим с scripts/validate_annotations.py --allow-partial.\n"
                    "events_draft.csv содержит события без группы сцены; все записи есть в documents/.\n"
                    "Незавершённая форма события сохраняется в documents/*:event_draft; в CSV она не включена.\n"
                    "watched_full означает полный просмотр, а не автоматическую метку отсутствия нарушений.\n"
                    "YOLO содержит только кадры с явным подтверждением полной разметки выбранных классов.\n"
                    "equipment_yolo/ содержит отдельную выборку только для 4 классов оборудования.\n"
                    "Порядок классов в equipment_yolo/classes.txt отличается от yolo/classes.txt.\n"
                    "layout_group_id — расстановка оборудования, не группа независимых эпизодов для train/test.\n"
                    "source/annotator/reviewed_by отделяют разметку человека от визуальной разметки ассистента.\n"
                    "Пустой TXT — подтверждённое отсутствие объектов этих классов, а не отсутствие нарушения.\n"
                    "Кадры до подтверждения полноты остаются в frames/ и objects.csv.\n"
                    "Проверьте разметку и группы сцен до разделения train/val/test. Соседние кадры держите вместе.\n"
                    "Текущий train_laptop.py не читает события или рамки. Для них нужен отдельный этап обучения.\n"
                    "other — черновой общий класс; для полного yolo/ его нужно уточнить.\n"
                    "Для equipment_yolo/ подтвердите все экземпляры 4 классов оборудования; прочие классы в него не входят.\n"
                ))
        return destination
