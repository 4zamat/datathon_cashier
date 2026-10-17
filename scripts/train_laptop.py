"""Train the linear head on frozen ResNet18 features. Uses only train + validation videos.

Test videos are never touched here. Validation picks weight decay, epoch count and threshold.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from laptop_common import (  # noqa: E402
    MODEL_DIR, RUN_DIR, SAMPLE_FPS, SEED, FeatureExtractor, LinearHead, aggregate, build_encoder,
    cached_features, choose_threshold, dump_json, environment_versions, git_sha, load_split,
    save_model, video_metrics,
)

POSITIVE_WINDOW_SEC = 1.0
WEIGHT_DECAYS = (0.0, 1e-3, 1e-2, 1e-1)
LEARNING_RATE = 1e-3
MINIBATCH = 64


def frame_dataset(videos: list[dict], store: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Frame features, labels and weights. Positives: +-1 s around the label; negatives: all frames
    of negative videos. Every video gets equal total weight, then classes are balanced."""
    xs, ys, ws = [], [], []
    for video in videos:
        times, features = store[video["video"]]
        if video["has_violation"]:
            mask = np.abs(times - video["timecode_sec"]) <= POSITIVE_WINDOW_SEC
            if not mask.any():
                raise ValueError(f"{video['video']}: no sampled frame near the label")
            features, label = features[mask], 1.0
        else:
            label = 0.0
        xs.append(features)
        ys.append(np.full(len(features), label, dtype=np.float32))
        ws.append(np.full(len(features), 1.0 / len(features), dtype=np.float32))
    x, y, w = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
    for label in (0.0, 1.0):
        w[y == label] /= w[y == label].sum()
    return x, y, w / w.sum()


def fit_head(x, y, w, x_val, y_val, w_val, weight_decay: float, epochs: int, patience: int, seed: int):
    import torch

    torch.manual_seed(seed)
    mean = x.mean(axis=0)
    std = x.std(axis=0) + 1e-6
    xt = torch.from_numpy((x - mean) / std)
    yt, wt = torch.from_numpy(y), torch.from_numpy(w * len(w))
    xv = torch.from_numpy((x_val - mean) / std)
    yv, wv = torch.from_numpy(y_val), torch.from_numpy(w_val)
    layer = torch.nn.Linear(x.shape[1], 1)
    optimizer = torch.optim.Adam(layer.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
    generator = torch.Generator().manual_seed(seed)
    best = (float("inf"), 0, None)
    history = []
    for epoch in range(1, epochs + 1):
        layer.train()
        order = torch.randperm(len(xt), generator=generator)
        for start in range(0, len(order), MINIBATCH):
            index = order[start:start + MINIBATCH]
            loss = (loss_fn(layer(xt[index]).squeeze(1), yt[index]) * wt[index]).sum() / wt[index].sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        layer.eval()
        with torch.no_grad():
            val_loss = float((loss_fn(layer(xv).squeeze(1), yv) * wv).sum() / wv.sum())
            train_loss = float((loss_fn(layer(xt).squeeze(1), yt) * (wt / wt.sum())).sum())
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best[0] - 1e-6:
            best = (val_loss, epoch, {k: v.clone() for k, v in layer.state_dict().items()})
        elif epoch - best[1] >= patience:
            break
    state = best[2]
    head = LinearHead(mean.astype(np.float32), std.astype(np.float32),
                      state["weight"].numpy()[0].astype(np.float32), float(state["bias"][0]))
    return head, best[0], best[1], history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--smoke", action="store_true",
                        help="short trial: 3 train + 2 val videos, 2 epochs, output under data/runs/.../smoke")
    args = parser.parse_args()

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; pass --device cpu explicitly to fall back")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train, val = load_split("train"), load_split("val")
    out_run, out_model, epochs = RUN_DIR, MODEL_DIR, args.epochs
    if args.smoke:
        train = [v for v in train if v["has_violation"]][:2] + [v for v in train if not v["has_violation"]][:2]
        val = [v for v in val if v["has_violation"]][:1] + [v for v in val if not v["has_violation"]][:1]
        out_run, out_model, epochs = RUN_DIR / "smoke", RUN_DIR / "smoke" / "model", 2
    out_run.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    encoder = build_encoder()  # one-time torchvision weight download if not yet cached
    extractor = FeatureExtractor(encoder, args.device, args.batch_size)
    store, feature_seconds = {}, {}
    for video in train + val:
        times, features, seconds = cached_features(extractor, video["path"], video["sha256"])
        store[video["video"]] = (times, features)
        feature_seconds[video["video"]] = seconds
        print(f"features {video['video']}: {len(times)} frames, {seconds:.1f}s", flush=True)

    train_start = time.perf_counter()
    x, y, w = frame_dataset(train, store)
    x_val, y_val, w_val = frame_dataset(val, store)
    trials = []
    for weight_decay in WEIGHT_DECAYS:
        head, val_loss, best_epoch, history = fit_head(
            x, y, w, x_val, y_val, w_val, weight_decay, epochs, args.patience, args.seed)
        trials.append({"weight_decay": weight_decay, "val_loss": val_loss, "best_epoch": best_epoch,
                       "epochs_run": len(history), "history": history, "head": head})
        print(f"wd={weight_decay}: val_loss={val_loss:.4f} best_epoch={best_epoch}", flush=True)
    chosen = min(trials, key=lambda t: (t["val_loss"], t["weight_decay"]))
    head = chosen["head"]

    # Threshold: video score = max frame probability over the WHOLE validation video.
    val_scores = []
    for video in val:
        times, features = store[video["video"]]
        val_scores.append(aggregate(times, head.probabilities(features))[0])
    threshold, val_metrics = choose_threshold(val_scores, [v["has_violation"] for v in val])
    train_seconds = time.perf_counter() - train_start

    config = {
        "name": "laptop_baseline", "seed": args.seed, "code_commit": git_sha(),
        "encoder": {"arch": "resnet18", "weights": "torchvision ResNet18_Weights.IMAGENET1K_V1",
                    "frozen": True, "feature_dim": 512, "file": "encoder_resnet18.pt"},
        "preprocessing": {"sample_fps": SAMPLE_FPS, "color": "BGR->RGB", "transform": "ResNet18_Weights.IMAGENET1K_V1.transforms()",
                          "resize": 256, "center_crop": 224, "normalize": "ImageNet mean/std",
                          "batch_size": args.batch_size, "num_workers": 0},
        "head": {"file": "head.npz", "type": "standardize + linear logit + sigmoid",
                 "selected_weight_decay": chosen["weight_decay"], "learning_rate": LEARNING_RATE,
                 "best_epoch": chosen["best_epoch"], "max_epochs": epochs, "patience": args.patience,
                 "minibatch": MINIBATCH, "weight_decay_grid": list(WEIGHT_DECAYS)},
        "labels": {"positive_window_sec": POSITIVE_WINDOW_SEC, "class_mapping": {"0": "no_violation_event", "1": "violation_event"},
                   "note": "Binary presence of an event only; no violation types or normative clauses."},
        "aggregation": "video score = max frame probability; predicted timecode = time of that frame",
        "threshold": threshold,
        "threshold_rule": "max validation F1; tie -> larger recall; tie -> higher threshold; predict violation when score >= threshold",
        "validation_metrics_at_threshold": val_metrics,
        "train_videos": [v["video"] for v in train], "val_videos": [v["video"] for v in val],
        "versions": environment_versions(),
    }
    save_model(out_model, encoder, head, config)
    dump_json(out_run / "train_log.json", {
        "smoke": args.smoke,
        "feature_extraction_sec_total": sum(feature_seconds.values()),
        "feature_extraction_sec_per_video": feature_seconds,
        "head_training_and_threshold_sec": train_seconds,
        "wall_clock_sec": time.perf_counter() - started,
        "trials": [{k: v for k, v in t.items() if k != "head"} for t in trials],
        "chosen_weight_decay": chosen["weight_decay"], "threshold": threshold,
        "validation_metrics_at_threshold": val_metrics,
    })
    print(json.dumps({"chosen_weight_decay": chosen["weight_decay"], "best_epoch": chosen["best_epoch"],
                      "threshold": threshold, "val_metrics": val_metrics,
                      "feature_extraction_sec": round(sum(feature_seconds.values()), 1),
                      "head_sec": round(train_seconds, 2)}, indent=2))


if __name__ == "__main__":
    main()
