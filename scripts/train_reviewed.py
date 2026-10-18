"""Train and evaluate one frozen-ResNet18 temporal baseline on reviewed windows."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import time

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import laptop_common as common
from scripts.reviewed_common import (fit_linear, interval_error, scores_for, training_arrays,
                                     validate_dataset)


def predict(head: common.LinearHead, times, features, threshold: float) -> dict:
    score, timecode = common.aggregate(times, scores_for(head, features))
    return {"score": score, "predicted": int(score >= threshold), "predicted_timecode_sec": timecode}


def load_reviewed_model(model_dir: Path, device: str, batch_size: int = 16):
    import numpy as np

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("format") != "cashier-temporal-v3" or config.get("context_radius") != 2:
        raise ValueError("Unsupported temporal model")
    for name, digest in config["files_sha256"].items():
        if common.sha256_file(model_dir / name) != digest:
            raise ValueError(f"Model checksum mismatch: {name}")
    encoder = common.build_encoder(model_dir / "encoder_resnet18.pt")
    with np.load(model_dir / "head.npz", allow_pickle=False) as stored:
        head = common.LinearHead.from_state(dict(stored))
    if head.weight.shape != (1536,):
        raise ValueError("Temporal head must have 1536 input features")
    return common.FeatureExtractor(encoder, device, batch_size), head, config


def reload_child(args) -> None:
    def blocked(*_args, **_kwargs):
        raise RuntimeError("Network access is forbidden during inference")
    socket.create_connection = blocked
    socket.socket.connect = blocked
    extractor, head, config = load_reviewed_model(args.out_dir / "model", args.device, args.batch_size)
    times, features = extractor.extract(args.data_dir / args.reload_video)
    result = predict(head, times, features, config["threshold"])
    common.dump_json(args.out_dir / "reload_prediction.json", result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True, help="Prepared dataset manifest JSON")
    parser.add_argument("--encoder-weights", type=Path, required=True, help="Existing local ResNet18 trunk")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--reload-video", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.reload_video:
        reload_child(args)
        return
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("Output already exists; use a new run directory")
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    validate_dataset(dataset, args.data_dir)
    import numpy as np
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing an implicit CPU fallback")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    np.random.seed(42)
    started = time.perf_counter()
    args.out_dir.mkdir(parents=True)
    shutil.copy2(args.dataset, args.out_dir / "dataset_manifest.json")
    common.FEATURE_DIR = args.data_dir / "features"
    encoder = common.build_encoder(args.encoder_weights)
    extractor = common.FeatureExtractor(encoder, args.device, args.batch_size)
    extractor.cache_identity += "-" + common.sha256_file(args.encoder_weights)
    groups = {s: [v for v in dataset["videos"] if v["split"] == s] for s in ("train", "val", "test")}
    store, extraction = {}, {}
    for video in groups["train"] + groups["val"]:
        times, features, elapsed = common.cached_features(extractor, args.data_dir / video["video"], video["sha256"])
        store[video["video"]] = (times, features)
        extraction[video["video"]] = elapsed
        print(f"FEATURES {video['split']:5} {video['video']}: {len(times)} frames, {elapsed:.1f}s", flush=True)
    x, y, w = training_arrays(groups["train"], store)
    xv, yv, wv = training_arrays(groups["val"], store)
    trials = []
    for decay in (0.001, 0.01, 0.1):
        head, history, best_epoch = fit_linear(x, y, w, xv, yv, wv, decay, args.epochs)
        val_loss = history[best_epoch - 1]["val_loss"]
        trials.append({"decay": decay, "head": head, "history": history, "best_epoch": best_epoch, "val_loss": val_loss})
        print(f"FIT decay={decay}: val_loss={val_loss:.5f}, epoch={best_epoch}", flush=True)
    chosen = min(trials, key=lambda t: (t["val_loss"], t["decay"]))
    head = chosen["head"]
    val_scores = [common.aggregate(store[v["video"]][0], scores_for(head, store[v["video"]][1]))[0] for v in groups["val"]]
    threshold, validation = common.choose_threshold(val_scores, [v["has_violation"] for v in groups["val"]])
    model_dir = args.out_dir / "model"
    model_dir.mkdir()
    # Preserve exact pretrained trunk bytes, avoiding ambiguous encoder provenance.
    shutil.copy2(args.encoder_weights, model_dir / "encoder_resnet18.pt")
    np.savez(model_dir / "head.npz", **{k: np.asarray(v) for k, v in head.state().items()})
    config = {
        "format": "cashier-temporal-v3", "context_radius": 2, "sample_fps": 2.0,
        "encoder": "frozen ResNet18 ImageNet trunk", "head_features": 1536,
        "temporal_features": "current + centered 5-frame mean + endpoint difference",
        "threshold": threshold, "validation_metrics": validation,
        "selected_weight_decay": chosen["decay"], "selected_epoch": chosen["best_epoch"],
        "preprocessing": "torchvision ResNet18 IMAGENET1K_V1 transforms; 2 fps",
        "dataset_sha256": common.sha256_file(args.dataset), "source_provenance": common.source_provenance(),
        "versions": common.environment_versions(),
        "target": "binary organizer dataset event with reviewed intervals and confidence weighting",
        "legal_rule_classifier": False, "evaluation_role": "development holdout; all source clips previously inspected",
        "files_sha256": {n: common.sha256_file(model_dir / n) for n in ("encoder_resnet18.pt", "head.npz")},
    }
    common.dump_json(model_dir / "config.json", config)
    common.dump_json(args.out_dir / "train_log.json", {
        "feature_seconds": extraction, "train_frames": len(y), "val_frames": len(yv),
        "trials": [{k: v for k, v in t.items() if k != "head"} for t in trials],
        "test_used_for_selection": False,
    })
    # Evaluate the SAVED model, decoding every whole video. No test features were
    # extracted before head/threshold selection. Targets are joined only afterwards.
    extractor, head, config = load_reviewed_model(model_dir, args.device, args.batch_size)
    predictions = []
    for split in ("train", "val", "test"):
        for video in groups[split]:
            start = time.perf_counter()
            times, features = extractor.extract(args.data_dir / video["video"])
            predicted = predict(head, times, features, threshold)
            row = {"video": video["video"], "split": split, "actual": video["has_violation"], **predicted,
                   "true_timecode_sec": video["timecode_sec"], "elapsed_sec": time.perf_counter() - start,
                   "interval_error_sec": interval_error(predicted["predicted_timecode_sec"], video["positive_windows"])}
            predictions.append(row)
            print(f"EVAL {split:5} {video['video']}: pred={row['predicted']} score={row['score']:.3f}", flush=True)
    common.write_csv(args.out_dir / "predictions.csv", list(predictions[0]), predictions)
    metrics = {"complete": True, "threshold": threshold, "dataset_sha256": config["dataset_sha256"],
               "evaluation_role": config["evaluation_role"], "legal_rule_classifier": False,
               "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None, "splits": {}}
    for split in ("train", "val", "test"):
        rows = [r for r in predictions if r["split"] == split]
        base = [{**r, "predicted": 1, "predicted_timecode_sec": 0.0} for r in rows]
        constant = common.video_metrics(base)
        constant.pop("mean_timecode_error_sec")
        constant.pop("localized_true_positives")
        errors = [r["interval_error_sec"] for r in rows if r["actual"] and r["predicted"]]
        metrics["splits"][split] = {"model": common.video_metrics(rows), "always_violation": constant,
            "mean_interval_error_on_tp_sec": float(np.mean(errors)) if errors else None,
            "max_video_processing_sec": max(r["elapsed_sec"] for r in rows)}
    metrics["wall_clock_sec"] = time.perf_counter() - started
    common.dump_json(args.out_dir / "metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    main()
