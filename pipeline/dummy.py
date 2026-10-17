"""A demonstration pipeline. It does not recognize cashier violations."""

import base64
import math
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from app.models import Detection, EvidenceFrame
from pipeline.base import Pipeline, Result

Image = NDArray[np.uint8]


def sample_frames(video_path: Path, sample_fps: float = 2.0) -> Iterator[tuple[float, Image]]:
    """Decode sequentially, yielding the first frame at/after each 0.5s boundary.

    Presentation timestamps support variable-rate videos. The frame-index clock
    is a fallback for backends which report zero/nonmonotonic timestamps.
    """
    if not math.isfinite(sample_fps) or sample_fps <= 0:
        raise ValueError("sample_fps must be positive and finite")
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError("Video could not be opened")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video has no usable frame rate")
        next_sample = 0.0
        previous_time = -1.0
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if not math.isfinite(timestamp) or timestamp < 0 or timestamp <= previous_time:
                timestamp = max(frame_index / fps, previous_time + 1.0 / fps)
            previous_time = timestamp
            frame_index += 1
            if timestamp + 1e-7 >= next_sample:
                yield timestamp, frame
                next_sample = (math.floor((timestamp + 1e-7) * sample_fps) + 1) / sample_fps
    finally:
        capture.release()


class DummyPipeline(Pipeline):
    def analyze(self, video_path: Path) -> Result:
        cv2.setNumThreads(1)
        # Two passes keep memory bounded and avoid trusting MOV frame-count metadata.
        sample_count = sum(1 for _ in sample_frames(video_path))
        if sample_count == 0:
            raise ValueError("Video has no decodable frames")
        middle_index = sample_count // 2
        frames = sample_frames(video_path)
        try:
            for index, (timestamp, frame) in enumerate(frames):
                if index != middle_index:
                    continue
                height, width = frame.shape[:2]
                if max(height, width) > 1280:
                    scale = 1280 / max(height, width)
                    frame = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    raise ValueError("Could not encode evidence as JPEG")
                return Result(violations=[Detection(
                    type="demo_violation",
                    timecode_sec=timestamp,
                    description=(
                        "ДЕМОНСТРАЦИЯ: фиктивное нарушение для проверки сервиса. "
                        "Показан реальный кадр из середины видео; действия кассира не анализировались."
                    ),
                    evidence_frames=[EvidenceFrame(
                        timestamp_sec=timestamp,
                        image_base64=base64.b64encode(encoded.tobytes()).decode("ascii"),
                    )],
                )])
        finally:
            frames.close()
        raise ValueError("Video changed or could not be decoded on the second pass")
