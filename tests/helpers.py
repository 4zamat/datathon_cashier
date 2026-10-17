import base64
import time
from pathlib import Path

import cv2
import numpy as np

from app.models import Detection, EvidenceFrame
from pipeline.base import Pipeline, Result


def make_video(path: Path) -> bytes:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24, (64, 48))
    if not writer.isOpened():
        raise RuntimeError("OpenCV test MOV encoder is unavailable")
    try:
        for index in range(96):
            frame = np.full((48, 64, 3), (index, 30, 200), dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()
    return path.read_bytes()


def detection(timestamp: float = 2.0, kind: str = "demo_violation") -> Detection:
    ok, jpeg = cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))
    assert ok
    return Detection(type=kind, timecode_sec=timestamp, description="Test detection", evidence_frames=[
        EvidenceFrame(timestamp_sec=timestamp, image_base64=base64.b64encode(jpeg.tobytes()).decode())
    ])


class ConditionalPipeline(Pipeline):
    def analyze(self, video_path: Path) -> Result:
        if video_path.read_bytes() == b"hang":
            (video_path.parent.parent / "started").touch()
            time.sleep(60)
        return Result()


class RecordingPipeline(Pipeline):
    def analyze(self, video_path: Path) -> Result:
        log = video_path.parent.parent / "execution.log"
        with log.open("a") as stream:
            stream.write(f"start {video_path.stem}\n")
        time.sleep(0.2)
        with log.open("a") as stream:
            stream.write(f"end {video_path.stem}\n")
        return Result()


class UnknownTypePipeline(Pipeline):
    def analyze(self, video_path: Path) -> Result:
        return Result(violations=[detection(kind="unknown_type")])


class EvaluationPipeline(Pipeline):
    def analyze(self, video_path: Path) -> Result:
        return Result(violations=[detection(4), detection(20)]) if video_path.stem in {"tp", "fp"} else Result()
