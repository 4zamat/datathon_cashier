"""Shared helpers for the laptop baseline experiment (data, frames, features, model)."""

import csv
import hashlib
import json
import math
import random
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SPLIT_DIR = DATA_DIR / "splits" / "laptop_baseline"
FEATURE_DIR = DATA_DIR / "features"
RUN_DIR = DATA_DIR / "runs" / "laptop_baseline"
MODEL_DIR = PROJECT_ROOT / "models" / "laptop_baseline"

SEED = 42
SAMPLE_FPS = 2.0
SPLITS = ("train", "val", "test")
SPLIT_FIELDS = ["video", "has_violation", "timecode_sec", "group_id", "sha256", "duplicates"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_split(split: str) -> list[dict]:
    rows = []
    for row in read_csv(SPLIT_DIR / f"{split}.csv"):
        rows.append({
            **row,
            "has_violation": int(row["has_violation"]),
            "timecode_sec": float(row["timecode_sec"]) if row["timecode_sec"] else None,
            "path": DATA_DIR / row["video"],
        })
    return rows


def stratified_group_split(
    groups: list[dict], seed: int = SEED, fractions: tuple[float, float] = (0.2, 0.2)
) -> dict[str, list[dict]]:
    """Split whole groups (label is group-constant) into train/val/test, stratified by label."""
    result: dict[str, list[dict]] = {name: [] for name in SPLITS}
    rng = random.Random(seed)
    for label in (0, 1):
        members = sorted((g for g in groups if g["has_violation"] == label), key=lambda g: g["group_id"])
        rng.shuffle(members)
        count = len(members)
        n_val = max(1, round(count * fractions[0]))
        n_test = max(1, round(count * fractions[1]))
        result["val"] += members[:n_val]
        result["test"] += members[n_val:n_val + n_test]
        result["train"] += members[n_val + n_test:]
    return result


# ---------------------------------------------------------------- video frames

def sample_frames(video_path: Path, sample_fps: float = SAMPLE_FPS) -> Iterator[tuple[float, np.ndarray]]:
    """Sequentially decode; yield the first frame at/after each 1/sample_fps boundary.

    Uses decoder timestamps, falling back to frame-index / fps when they are not monotonic.
    Frames are BGR uint8 as returned by OpenCV.
    """
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Video could not be opened: {video_path.name}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video has no usable frame rate")
        next_sample = 0.0
        previous_time = -1.0
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if not math.isfinite(timestamp) or timestamp < 0 or timestamp <= previous_time:
                timestamp = max(index / fps, previous_time + 1.0 / fps)
            previous_time = timestamp
            index += 1
            if timestamp + 1e-7 >= next_sample:
                yield timestamp, frame
                next_sample = (math.floor((timestamp + 1e-7) * sample_fps) + 1) / sample_fps
    finally:
        capture.release()


# ------------------------------------------------------------- encoder / model

def build_encoder(weights_path: Path | None = None):
    """ResNet18 trunk (512-d pooled features), frozen, eval mode.

    With weights_path=None the torchvision ImageNet weights are used (downloaded once);
    otherwise the saved state dict is loaded without any network access.
    """
    import torch
    import torchvision

    if weights_path is None:
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        net = torchvision.models.resnet18(weights=weights)
    else:
        net = torchvision.models.resnet18(weights=None)
        net.fc = torch.nn.Identity()  # saved trunk has no classifier layer
        net.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    net.fc = torch.nn.Identity()
    net.eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    return net


def build_transform():
    """Preprocessing of ResNet18_Weights.IMAGENET1K_V1 (resize 256, center crop 224, ImageNet norm)."""
    import torchvision

    return torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms()


class FeatureExtractor:
    """Decode a video at ~2 fps and compute frozen-encoder features in batches."""

    def __init__(self, encoder, device: str, batch_size: int = 16):
        import torch

        self.torch = torch
        self.encoder = encoder.to(device)
        self.device = device
        self.batch_size = batch_size
        self.transform = build_transform()

    def _flush(self, batch: list[np.ndarray]) -> np.ndarray:
        torch = self.torch
        rgb = np.ascontiguousarray(np.stack(batch)[..., ::-1])  # BGR -> RGB, (B, H, W, 3)
        tensor = torch.from_numpy(rgb).to(self.device).permute(0, 3, 1, 2)
        with torch.inference_mode():
            tensor = self.transform(tensor)
            return self.encoder(tensor).float().cpu().numpy()

    def extract(self, video_path: Path) -> tuple[np.ndarray, np.ndarray]:
        times: list[float] = []
        chunks: list[np.ndarray] = []
        batch: list[np.ndarray] = []
        for timestamp, frame in sample_frames(video_path):
            times.append(timestamp)
            batch.append(frame)
            if len(batch) == self.batch_size:
                chunks.append(self._flush(batch))
                batch = []
        if batch:
            chunks.append(self._flush(batch))
        if not times:
            raise ValueError("Video has no decodable frames")
        return np.asarray(times, dtype=np.float64), np.concatenate(chunks)


def cached_features(extractor: FeatureExtractor, video_path: Path, sha: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (times, features, seconds spent extracting; 0.0 on cache hit)."""
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    cache = FEATURE_DIR / f"{sha}_resnet18_{SAMPLE_FPS:g}fps.npz"
    if cache.is_file():
        with np.load(cache) as stored:
            return stored["times"], stored["features"].astype(np.float32), 0.0
    start = time.perf_counter()
    times, features = extractor.extract(video_path)
    elapsed = time.perf_counter() - start
    np.savez(cache, times=times, features=features.astype(np.float16))
    return times, features, elapsed


class LinearHead:
    """Standardize + linear logit; stored as plain tensors so it loads without training code."""

    def __init__(self, mean: np.ndarray, std: np.ndarray, weight: np.ndarray, bias: float):
        self.mean, self.std, self.weight, self.bias = mean, std, weight, float(bias)

    def logits(self, features: np.ndarray) -> np.ndarray:
        return ((features - self.mean) / self.std) @ self.weight + self.bias

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.logits(features)))

    def state(self) -> dict:
        return {"mean": self.mean, "std": self.std, "weight": self.weight, "bias": self.bias}

    @classmethod
    def from_state(cls, state: dict) -> "LinearHead":
        return cls(*(np.asarray(state[k], dtype=np.float32) for k in ("mean", "std", "weight")), float(state["bias"]))


def aggregate(times: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Video score = max frame score; predicted timecode = time of that maximum."""
    best = int(np.argmax(scores))
    return float(scores[best]), float(times[best])


def video_metrics(rows: list[dict]) -> dict:
    """rows: dicts with actual (0/1), predicted (0/1), true_timecode_sec, predicted_timecode_sec."""
    tp = sum(1 for r in rows if r["actual"] == 1 and r["predicted"] == 1)
    fp = sum(1 for r in rows if r["actual"] == 0 and r["predicted"] == 1)
    fn = sum(1 for r in rows if r["actual"] == 1 and r["predicted"] == 0)
    tn = sum(1 for r in rows if r["actual"] == 0 and r["predicted"] == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    errors = [abs(r["predicted_timecode_sec"] - r["true_timecode_sec"]) for r in rows
              if r["actual"] == 1 and r["predicted"] == 1]
    return {
        "videos": len(rows), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall": recall, "precision": precision, "f1": f1,
        "accuracy": (tp + tn) / len(rows) if rows else 0.0,
        "mean_timecode_error_sec": float(np.mean(errors)) if errors else None,
        "localized_true_positives": len(errors),
    }


def choose_threshold(scores: list[float], actual: list[int]) -> tuple[float, dict]:
    """Max F1 on validation; ties -> larger recall; further ties -> higher (stricter) threshold.

    Candidates are the observed video scores (predict violation when score >= threshold).
    """
    best = None
    for candidate in sorted(set(scores)):
        rows = [{"actual": a, "predicted": int(s >= candidate), "true_timecode_sec": 0.0,
                 "predicted_timecode_sec": 0.0} for s, a in zip(scores, actual)]
        metrics = video_metrics(rows)
        key = (metrics["f1"], metrics["recall"], candidate)
        if best is None or key > best[0]:
            best = (key, candidate, metrics)
    return float(best[1]), best[2]


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ model I/O

def git_sha() -> str:
    import subprocess

    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def environment_versions() -> dict:
    import platform

    import cv2
    import torch
    import torchvision

    return {
        "python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
        "numpy": np.__version__, "opencv": cv2.__version__, "cuda_build": torch.version.cuda,
    }


def save_model(model_dir: Path, encoder, head: LinearHead, config: dict) -> None:
    import torch

    model_dir.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in encoder.state_dict().items()}
    torch.save(state, model_dir / "encoder_resnet18.pt")
    np.savez(model_dir / "head.npz", **{k: np.asarray(v) for k, v in head.state().items()})
    dump_json(model_dir / "config.json", config)


def load_model(model_dir: Path, device: str, batch_size: int = 16):
    """Load encoder + head from local files only (no downloads). Returns (extractor, head, config)."""
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    encoder = build_encoder(model_dir / "encoder_resnet18.pt")
    with np.load(model_dir / "head.npz") as stored:
        head = LinearHead.from_state({k: stored[k] for k in stored.files})
    return FeatureExtractor(encoder, device, batch_size), head, config


def analyze_video(extractor: FeatureExtractor, head: LinearHead, threshold: float, video_path: Path) -> dict:
    """Analyze the whole video without any ground truth; no feature cache is read or written."""
    times, features = extractor.extract(video_path)
    score, timecode = aggregate(times, head.probabilities(features))
    return {"score": score, "predicted": int(score >= threshold), "predicted_timecode_sec": timecode,
            "frames": len(times)}
