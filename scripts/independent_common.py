"""Independent v4 targets, full-scene features and nested layout evaluation.

No UI status, organizer target, timestamp or rule enters inference.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from scripts.laptop_common import LinearHead, sample_frames, sha256_file, video_metrics

FEATURE_VERSION = "v4-letterbox224-full-top70-flipmean-l2-fps2"
EPOCHS = 60
WEIGHT_DECAY = 0.2


def validate_dataset(dataset: dict, data_dir: Path, verify_files: bool = True) -> None:
    if dataset.get("format") != "cashier-independent-v4":
        raise ValueError("Expected independent v4 dataset")
    if dataset.get("organizer_labels_used_for_training") is not False:
        raise ValueError("Organizer targets must not be used")
    if dataset.get("ui_assessments_used_for_decisions") is not False:
        raise ValueError("UI assessments must not be used")
    names, hashes, scenes = set(), set(), {}
    for row in dataset["videos"]:
        label = row["reviewed_label"]
        if label not in (None, 0, 1) or row["eligible"] != (label is not None):
            raise ValueError("Unresolved targets must be excluded")
        if row["video"] in names or row["source_sha256"] in hashes:
            raise ValueError("Duplicate video in canonical dataset")
        names.add(row["video"])
        hashes.add(row["source_sha256"])
        group, scene = row["layout_group"], row["scene_group_id"]
        if scene in scenes and scenes[scene] != group:
            raise ValueError("Related scene split across layouts")
        scenes[scene] = group
        if bool(row["positive_windows"]) != (label == 1):
            raise ValueError("Only positive videos may have positive intervals")
        for window in row["positive_windows"]:
            start, end = window["start_sec"], window["end_sec"]
            if not 0 <= start < end <= row["duration_sec"] + .05:
                raise ValueError("Invalid event interval")
        path = (data_dir / row["video"]).resolve()
        if not path.is_relative_to(data_dir.resolve()):
            raise ValueError("Video path must stay in data directory")
        if verify_files and (not path.is_file() or sha256_file(path) != row["source_sha256"]):
            raise ValueError(f"Missing or changed video: {row['video']}")
    videos = [v for v in dataset["videos"] if v["eligible"]]
    groups = sorted({v["layout_group"] for v in videos})
    if len(groups) < 4:
        raise ValueError("Need at least four layout groups for nested calibration")
    for outer in groups:
        for inner in (g for g in groups if g != outer):
            train = [v for v in videos if v["layout_group"] not in (outer, inner)]
            if {v["reviewed_label"] for v in train} != {0, 1}:
                raise ValueError(f"Inner training lacks a class: {outer}/{inner}")


def letterbox(frame: np.ndarray, size: int = 224) -> np.ndarray:
    """Preserve every pixel region, padding with approximate ImageNet mean (BGR)."""
    import cv2

    h, w = frame.shape[:2]
    if min(h, w) < 1 or size < 1:
        raise ValueError("Invalid image size")
    scale = min(size / h, size / w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.empty((size, size, 3), dtype=np.uint8)
    canvas[:] = (104, 116, 124)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def frame_views(frame: np.ndarray) -> list[np.ndarray]:
    full = letterbox(frame)
    work = letterbox(frame[:max(1, round(frame.shape[0] * .7))])
    return [full, full[:, ::-1].copy(), work, work[:, ::-1].copy()]


class SceneExtractor:
    def __init__(self, encoder, device: str, batch_size: int = 8):
        import torch

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.torch, self.device, self.batch_size = torch, device, batch_size
        self.encoder = encoder.to(device).eval()

    def _flush(self, frames: list[list[np.ndarray]]) -> np.ndarray:
        torch = self.torch
        rgb = np.ascontiguousarray(np.stack([v for views in frames for v in views])[..., ::-1])
        tensor = torch.from_numpy(rgb).to(self.device).permute(0, 3, 1, 2).float() / 255
        mean = tensor.new_tensor([.485, .456, .406])[None, :, None, None]
        std = tensor.new_tensor([.229, .224, .225])[None, :, None, None]
        with torch.inference_mode():
            features = self.encoder((tensor - mean) / std).float().cpu().numpy()
        features = features.reshape(len(frames), 2, 2, -1).mean(axis=2)
        features /= np.maximum(np.linalg.norm(features, axis=2, keepdims=True), 1e-8)
        return features.reshape(len(frames), -1).astype(np.float32)

    def extract(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        times, chunks, frames = [], [], []
        for timestamp, frame in sample_frames(path, 2.0):
            times.append(timestamp)
            frames.append(frame_views(frame))
            if len(frames) == self.batch_size:
                chunks.append(self._flush(frames))
                frames = []
        if frames:
            chunks.append(self._flush(frames))
        if not times:
            raise ValueError(f"No decoded frames: {path.name}")
        return np.asarray(times, dtype=np.float64), np.concatenate(chunks)


def temporal_features(features: np.ndarray) -> np.ndarray:
    """Local appearance plus change relative to the first two seconds of this clip."""
    if features.ndim != 2 or not len(features) or not np.isfinite(features).all():
        raise ValueError("Expected finite nonempty feature matrix")
    padded = np.pad(features, ((1, 1), (0, 0)), mode="edge")
    local = (padded[:-2] + padded[1:-1] + padded[2:]) / 3
    baseline = features[:4].mean(axis=0)
    return np.concatenate([local, local - baseline], axis=1).astype(np.float32)


def training_arrays(videos: list[dict], store: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, ws = [], [], []
    for video in videos:
        label = video["reviewed_label"]
        if label is None or not video["eligible"]:
            raise ValueError("Unresolved video must not enter supervised training")
        times, features = store[video["video"]]
        confidence = np.ones(len(times), dtype=np.float32) if label == 0 else np.zeros(len(times), dtype=np.float32)
        for window in video["positive_windows"]:
            mask = (times >= window["start_sec"]) & (times <= window["end_sec"])
            confidence[mask] = np.maximum(confidence[mask], window.get("confidence", 1.0))
        valid = confidence > 0
        if not valid.any():
            raise ValueError(f"No reviewed training frames: {video['video']}")
        xs.append(temporal_features(features)[valid])
        ys.append(np.full(valid.sum(), label, dtype=np.float32))
        ws.append(confidence[valid] / valid.sum())
    x, y, weights = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Training requires both classes")
    for label in (0, 1):
        mask = y == label
        weights[mask] /= weights[mask].sum()
    return x, y, (weights / weights.sum()).astype(np.float32)


def fit_head(videos: list[dict], store: dict) -> tuple[LinearHead, dict]:
    """Fixed 60-epoch weighted ridge logistic head; no test-dependent early stopping."""
    x, y, weights = training_arrays(videos, store)
    mean = (x * weights[:, None]).sum(axis=0).astype(np.float32)
    variance = ((x - mean) ** 2 * weights[:, None]).sum(axis=0)
    std = np.maximum(np.sqrt(variance), .01).astype(np.float32)
    a = ((x - mean) / std).astype(np.float32)
    theta, bias = np.zeros(a.shape[1], dtype=np.float32), 0.0
    m, v, mb, vb = np.zeros_like(theta), np.zeros_like(theta), 0., 0.
    history = []
    for epoch in range(1, EPOCHS + 1):
        z = a @ theta + bias
        residual = (1 / (1 + np.exp(-np.clip(z, -60, 60))) - y) * weights
        grad, gb = a.T @ residual + WEIGHT_DECAY * theta, float(residual.sum())
        m, v = .9 * m + .1 * grad, .999 * v + .001 * grad * grad
        mb, vb = .9 * mb + .1 * gb, .999 * vb + .001 * gb * gb
        theta -= .001 * (m / (1 - .9 ** epoch)) / (np.sqrt(v / (1 - .999 ** epoch)) + 1e-8)
        bias -= .001 * (mb / (1 - .9 ** epoch)) / (math.sqrt(vb / (1 - .999 ** epoch)) + 1e-8)
        history.append(float(((np.logaddexp(0, z) - y * z) * weights).sum()))
    if not np.isfinite(theta).all() or not math.isfinite(bias):
        raise ValueError("Non-finite training result")
    return LinearHead(mean, std, theta, bias), {"frames": len(y), "videos": len(videos), "loss": history}


def predict(head: LinearHead, times: np.ndarray, features: np.ndarray, threshold: float) -> dict:
    logits = head.logits(temporal_features(features))
    scores = 1 / (1 + np.exp(-np.clip(logits, -60, 60)))
    best = int(np.argmax(scores))
    score = float(scores[best])
    return {"score": score, "predicted": int(score >= threshold), "predicted_timecode_sec": float(times[best])}


def metrics(rows: list[dict]) -> dict:
    result = video_metrics(rows)
    result["specificity"] = result["tn"] / (result["tn"] + result["fp"]) if result["tn"] + result["fp"] else None
    sensitivity = result["recall"] if result["tp"] + result["fn"] else None
    result["balanced_accuracy"] = (sensitivity + result["specificity"]) / 2 if sensitivity is not None and result["specificity"] is not None else None
    return result


def select_threshold(rows: list[dict]) -> tuple[float, dict]:
    """Inner-group predictions only: maximize balanced accuracy, then F1, recall."""
    if {r["actual"] for r in rows} != {0, 1}:
        raise ValueError("Threshold calibration needs both classes")
    candidates = sorted({r["score"] for r in rows} | {1.000001})
    evaluated = []
    for threshold in candidates:
        result = metrics([{**r, "predicted": int(r["score"] >= threshold)} for r in rows])
        evaluated.append(((result["balanced_accuracy"], result["f1"], result["recall"], threshold), result))
    key, result = max(evaluated, key=lambda item: item[0])
    return float(key[-1]), result


def row_prediction(head: LinearHead, video: dict, store: dict, threshold: float) -> dict:
    prediction = predict(head, *store[video["video"]], threshold)
    windows = video["positive_windows"]
    timestamp = prediction["predicted_timecode_sec"]
    interval_error = min(max(w["start_sec"] - timestamp, 0, timestamp - w["end_sec"]) for w in windows) if windows else None
    return {"video": video["video"], "layout_group": video["layout_group"], "actual": video["reviewed_label"],
            "threshold": threshold, **prediction, "true_timecode_sec": windows[0]["start_sec"] if windows else 0.,
            "interval_error_sec": interval_error}


def nested_evaluation(videos: list[dict], store: dict, progress=print) -> tuple[list[dict], list[dict]]:
    """Each outer layout excluded from training AND threshold calibration."""
    groups = sorted({v["layout_group"] for v in videos})
    predictions, folds = [], []
    for outer in groups:
        development = [v for v in videos if v["layout_group"] != outer]
        held_out = [v for v in videos if v["layout_group"] == outer]
        inner_predictions, inner_audit = [], []
        for inner in (g for g in groups if g != outer):
            train = [v for v in development if v["layout_group"] != inner]
            calibration = [v for v in development if v["layout_group"] == inner]
            head, _ = fit_head(train, store)
            inner_predictions.extend(row_prediction(head, v, store, .5) for v in calibration)
            inner_audit.append({"train_videos": [v["video"] for v in train], "calibration_videos": [v["video"] for v in calibration]})
        threshold, calibration_metrics = select_threshold(inner_predictions)
        head, log = fit_head(development, store)
        rows = [row_prediction(head, v, store, threshold) for v in held_out]
        predictions.extend(rows)
        folds.append({"held_out_layout": outer, "test_videos": [v["video"] for v in held_out],
                      "train_videos": [v["video"] for v in development], "inner_folds": inner_audit,
                      "inner_predictions": inner_predictions, "threshold": threshold,
                      "inner_calibration_metrics": calibration_metrics, "test_metrics": metrics(rows), "training": log})
        progress(f"FOLD {outer}: threshold={threshold:.4f}, {metrics(rows)}", flush=True)
    return predictions, folds
