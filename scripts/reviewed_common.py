"""Event-window supervision and temporal features for a diagnostic binary model.

This model predicts the dataset target, not a legal conclusion or a rule type.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from scripts.laptop_common import LinearHead, sha256_file


def temporal_features(features: np.ndarray, radius: int = 2) -> np.ndarray:
    """At 2 fps: current appearance, two-second local mean, and local change.

    Context never crosses a video boundary. Edge padding repeats endpoint frames.
    """
    if features.ndim != 2 or len(features) == 0 or radius < 1:
        raise ValueError("Expected nonempty frame-feature matrix and positive radius")
    padded = np.pad(features, ((radius, radius), (0, 0)), mode="edge")
    mean = sum(padded[k:k + len(features)] for k in range(2 * radius + 1)) / (2 * radius + 1)
    delta = padded[2 * radius:2 * radius + len(features)] - padded[:len(features)]
    return np.concatenate([features, mean, delta], axis=1).astype(np.float32)


def frame_targets(times: np.ndarray, video: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return labels (-1 means ignored) and supervision confidence.

    Positive intervals override conflicting normal notes. Unresolved intervals are
    never silently converted to clean negatives. Organizer-only labels stay weak.
    """
    labels = np.full(len(times), -1, dtype=np.int64)
    weights = np.zeros(len(times), dtype=np.float32)
    if video["has_violation"] == 0:
        labels[:] = 0
        weights[:] = 0.35  # video-level organizer supervision, not human-confirmed frames
    for event in video["events"]:
        mask = (times >= event["start_sec"]) & (times <= event["end_sec"])
        if event["assessment"] == "no_violation":
            labels[mask], weights[mask] = 0, 1.0
    for event in video["events"]:
        if event["assessment"] != "no_violation":
            mask = (times >= event["start_sec"] - 0.5) & (times <= event["end_sec"] + 0.5)
            labels[mask], weights[mask] = -1, 0
    for window in video["positive_windows"]:
        # Guard bands keep the same persistent event from being taught as normal.
        guard = (times >= window["start_sec"] - 0.5) & (times <= window["end_sec"] + 0.5)
        labels[guard], weights[guard] = -1, 0
        mask = (times >= window["start_sec"]) & (times <= window["end_sec"])
        labels[mask], weights[mask] = 1, window["weight"]
    return labels, weights


def training_arrays(videos: list[dict], store: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, ws = [], [], []
    for video in videos:
        times, features = store[video["video"]]
        labels, confidence = frame_targets(times, video)
        valid = labels >= 0
        if not valid.any():
            raise ValueError(f"No supervised frames: {video['video']}")
        xs.append(temporal_features(features)[valid])
        ys.append(labels[valid].astype(np.float32))
        # Equal upper contribution per video; retain weaker confidence for weak labels.
        ws.append(confidence[valid] / valid.sum())
    x, y, w = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Training/validation windows must contain both classes")
    for label in (0, 1):
        mask = y == label
        w[mask] /= w[mask].sum()
    return x, y, w / w.sum()


def fit_linear(x: np.ndarray, y: np.ndarray, w: np.ndarray,
               xv: np.ndarray, yv: np.ndarray, wv: np.ndarray,
               decay: float, epochs: int = 60, patience: int = 8) -> tuple[LinearHead, list[dict], int]:
    """Deterministic weighted full-batch Adam; normalization is fitted on train only."""
    mean, std = x.mean(0), np.maximum(x.std(0), 1e-3)
    a, av = ((x - mean) / std).astype(np.float32), ((xv - mean) / std).astype(np.float32)
    theta = np.zeros(a.shape[1], dtype=np.float32)
    bias = 0.0
    m, v = np.zeros_like(theta), np.zeros_like(theta)
    mb = vb = 0.0
    best_loss, best_epoch, best = math.inf, 0, None
    history = []
    for epoch in range(1, epochs + 1):
        z = a @ theta + bias
        probability = 1 / (1 + np.exp(-np.clip(z, -60, 60)))
        residual = (probability - y) * w
        grad = a.T @ residual + decay * theta
        gb = float(residual.sum())
        m, v = 0.9 * m + 0.1 * grad, 0.999 * v + 0.001 * grad * grad
        mb, vb = 0.9 * mb + 0.1 * gb, 0.999 * vb + 0.001 * gb * gb
        theta -= 1e-3 * (m / (1 - 0.9 ** epoch)) / (np.sqrt(v / (1 - 0.999 ** epoch)) + 1e-8)
        bias -= 1e-3 * (mb / (1 - 0.9 ** epoch)) / (math.sqrt(vb / (1 - 0.999 ** epoch)) + 1e-8)
        zv = av @ theta + bias
        val_loss = float(((np.logaddexp(0, zv) - yv * zv) * wv).sum())
        train_loss = float(((np.logaddexp(0, z) - y * z) * w).sum())
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_loss - 1e-6:
            best_loss, best_epoch = val_loss, epoch
            best = LinearHead(mean.copy(), std.copy(), theta.copy(), bias)
        elif epoch - best_epoch >= patience:
            break
    if best is None:
        raise ValueError("Training produced no finite validation loss")
    return best, history, best_epoch


def scores_for(head: LinearHead, features: np.ndarray) -> np.ndarray:
    z = head.logits(temporal_features(features))
    return 1 / (1 + np.exp(-np.clip(z, -60, 60)))


def interval_error(timestamp: float, windows: list[dict]) -> float | None:
    if not windows:
        return None
    return min(max(w["start_sec"] - timestamp, 0, timestamp - w["end_sec"]) for w in windows)


def validate_dataset(dataset: dict, data_dir: Path, verify_files: bool = True) -> None:
    if dataset.get("format") != "cashier-reviewed-v3":
        raise ValueError("Unsupported reviewed dataset format")
    seen_names, seen_hashes, seen_groups, seen_scenes = set(), {}, {}, {}
    for split in ("train", "val", "test"):
        videos = [v for v in dataset["videos"] if v["split"] == split]
        if {v["has_violation"] for v in videos} != {0, 1}:
            raise ValueError(f"{split}: both classes required")
        for video in videos:
            if video["video"] in seen_names:
                raise ValueError("Duplicate video row")
            seen_names.add(video["video"])
            for field, index in (("sha256", seen_hashes), ("split_group_id", seen_groups), ("scene_group_id", seen_scenes)):
                value = video[field]
                if not value or value in index and index[value] != split:
                    raise ValueError(f"Split leakage: {field}")
                if field == "sha256" and value in index:
                    raise ValueError("Exact duplicates must have a single representative")
                index[value] = split
            duration = video["duration_sec"]
            for event in [*video["events"], *video["positive_windows"]]:
                if not 0 <= event["start_sec"] < event["end_sec"] <= duration:
                    raise ValueError("Invalid interval")
            if bool(video["positive_windows"]) != bool(video["has_violation"]):
                raise ValueError("Positive window/label mismatch")
            for window in video["positive_windows"]:
                if not 0 < window["weight"] <= 1:
                    raise ValueError("Invalid supervision weight")
            path = (data_dir / video["video"]).resolve()
            if not path.is_relative_to(data_dir.resolve()):
                raise ValueError("Video path leaves data directory")
            if verify_files and sha256_file(path) != video["sha256"]:
                raise ValueError(f"Source changed: {video['video']}")
