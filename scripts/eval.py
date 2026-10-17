"""Evaluate video-level predictions without requiring a running API."""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from statistics import mean

# Support both `python scripts/eval.py` and `python -m scripts.eval`.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import PROJECT_ROOT
from app.models import Model
from pipeline.base import Pipeline, Result, load_pipeline


class Metrics(Model):
    videos: int
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float
    recall: float
    mean_timecode_error_sec: float | None
    localized_true_positives: int


def evaluate(labels_path: Path, pipeline: Pipeline) -> Metrics:
    tp = fp = fn = tn = 0
    errors: list[float] = []
    seen: set[Path] = set()
    with labels_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = csv.DictReader(stream)
        required = {"video", "has_violation", "timecode_sec"}
        if not required.issubset(rows.fieldnames or []):
            raise ValueError("CSV must contain video,has_violation,timecode_sec")
        for line, row in enumerate(rows, 2):
            video = (row.get("video") or "").strip()
            label = (row.get("has_violation") or "").strip().lower()
            if not video or label not in {"0", "1", "false", "true"}:
                raise ValueError(f"Line {line}: expected a video path and a 0/1 or false/true label")
            path = Path(video)
            if not path.is_absolute():
                path = labels_path.parent / path
            path = path.resolve()
            if path in seen:
                raise ValueError(f"Line {line}: one CSV row per video is required")
            if not path.is_file():
                raise ValueError(f"Line {line}: video not found: {path}")
            seen.add(path)
            actual = label in {"1", "true"}
            timestamp_text = (row.get("timecode_sec") or "").strip()
            timestamp = float(timestamp_text) if timestamp_text else None
            if timestamp is not None and (not math.isfinite(timestamp) or timestamp < 0):
                raise ValueError(f"Line {line}: timecode must be finite and nonnegative")
            if actual and timestamp is None:
                raise ValueError(f"Line {line}: positive videos require a timecode")
            result = Result.model_validate(pipeline.analyze(path))
            predicted = bool(result.violations)
            if actual and predicted:
                tp += 1
                assert timestamp is not None
                errors.append(min(abs(item.timecode_sec - timestamp) for item in result.violations))
            elif predicted:
                fp += 1
            elif actual:
                fn += 1
            else:
                tn += 1
    if not seen:
        raise ValueError("Labels CSV contains no videos")
    return Metrics(
        videos=len(seen), true_positives=tp, false_positives=fp, false_negatives=fn, true_negatives=tn,
        precision=tp / (tp + fp) if tp + fp else 0.0,
        recall=tp / (tp + fn) if tp + fn else 0.0,
        mean_timecode_error_sec=mean(errors) if errors else None,
        localized_true_positives=len(errors),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=PROJECT_ROOT / "data" / "labels.csv")
    parser.add_argument("--pipeline", default=os.getenv("PIPELINE_CLASS", "pipeline.dummy:DummyPipeline"))
    args = parser.parse_args()
    try:
        metrics = evaluate(args.labels, load_pipeline(args.pipeline))
    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(metrics.model_dump(), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
