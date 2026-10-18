"""Run the v4 experiment using explicit visual review targets and nested layout CV."""
from __future__ import annotations

import argparse
import base64
from html import escape
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import time

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scripts import laptop_common as common
from scripts.independent_common import (EPOCHS, FEATURE_VERSION, WEIGHT_DECAY, SceneExtractor,
    fit_head, metrics, nested_evaluation, predict, row_prediction, validate_dataset)


def load_model(model_dir: Path, device: str):
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("format") != "cashier-independent-model-v4" or config.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Unsupported model preprocessing")
    for name, expected in config["files_sha256"].items():
        if common.sha256_file(model_dir / name) != expected:
            raise ValueError(f"Model checksum mismatch: {name}")
    with np.load(model_dir / "head.npz", allow_pickle=False) as arrays:
        head = common.LinearHead.from_state(dict(arrays))
    if any(array.shape != (2048,) for array in (head.mean, head.std, head.weight)) or (head.std <= 0).any():
        raise ValueError("Invalid v4 head shape/normalization")
    encoder = common.build_encoder(model_dir / "encoder_resnet18.pt")
    return SceneExtractor(encoder, device), head, config


def preview(path: Path, timestamp: float) -> str:
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        return "<p>Кадр не удалось прочитать</p>"
    h, w = frame.shape[:2]
    frame = cv2.resize(frame, (round(w * min(1., 480 / h)), round(h * min(1., 480 / h))))
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return '<img alt="Кадр в момент максимального score" src="data:image/jpeg;base64,' + base64.b64encode(jpeg).decode("ascii") + '">'


def write_report(out: Path, data: Path, dataset: dict, predictions: list[dict], result: dict) -> None:
    by_name = {v["video"]: v for v in dataset["videos"]}
    cards = []
    for row in sorted(predictions, key=lambda r: (r["actual"] == r["predicted"], r["video"])):
        video = by_name[row["video"]]
        ok = row["actual"] == row["predicted"]
        cards.append(f'<article class="{"ok" if ok else "error"}"><h2>{escape(row["video"])} — {"совпало" if ok else "ошибка"}</h2>'
            f'<p>Разметка: {row["actual"]}; прогноз: {row["predicted"]}; score {row["score"]:.3f}, порог {row["threshold"]:.3f}.</p>'
            f'<p>Время прогноза: {row["predicted_timecode_sec"]:.2f} с. Группа: {escape(row["layout_group"])}.</p>'
            f'<p>{escape(video["reason"])}</p><p>Размеченные интервалы: {escape(json.dumps(video["positive_windows"], ensure_ascii=False))}</p>'
            + preview(data / row["video"], row["predicted_timecode_sec"]) + '</article>')
    excluded = "".join(f'<li><b>{escape(v["video"])}</b>: {escape(v["reason"])}</li>' for v in dataset["videos"] if not v["eligible"])
    html = ('<!doctype html><html lang="ru"><meta charset="utf-8"><title>Проверка v4</title>'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><style>'
        'body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 16px;background:#f4f5f7;color:#182132}'
        'article{background:white;border-left:6px solid #34865c;padding:18px;margin:20px 0;border-radius:8px}'
        '.error{border-color:#bd284b}img{max-width:100%;max-height:480px}pre{white-space:pre-wrap;overflow-wrap:anywhere}'
        '</style><h1>Независимая разметка: эксперимент v4</h1>'
        '<p>Показаны прогнозы на исключённом из обучения ракурсе. Метка 1 означает признаки возможного нарушения. '
        'Все исходные ролики уже просмотрены при разметке: это проверка разработки, а не независимый финальный тест. '
        'Бинарная модель не определяет тип нарушения и пункт правила.</p><pre>'
        + escape(json.dumps(result, ensure_ascii=False, indent=2)) + '</pre>' + ''.join(cards)
        + '<h2>Исключены из обучения и оценки</h2><ul>' + excluded + '</ul></html>')
    (out / "review.html").write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--encoder-weights", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--reload-video", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.reload_video:
        def blocked(*_args, **_kwargs):
            raise RuntimeError("Network forbidden in saved-model check")
        socket.create_connection = blocked
        socket.socket.connect = blocked
        extractor, head, config = load_model(args.out_dir / "model", args.device)
        started = time.perf_counter()
        times, features = extractor.extract(args.data_dir / args.reload_video)
        common.dump_json(args.out_dir / "reload_prediction.json", {
            **predict(head, times, features, config["threshold"]), "elapsed_sec": time.perf_counter() - started})
        return

    import torch

    started = time.perf_counter()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available in this Python environment")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    validate_dataset(dataset, args.data_dir)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.dataset, args.out_dir / "dataset_manifest.json")
    videos = [v for v in dataset["videos"] if v["eligible"]]
    encoder_sha = common.sha256_file(args.encoder_weights)
    extractor = SceneExtractor(common.build_encoder(args.encoder_weights), args.device)
    cache = args.data_dir / "features" / FEATURE_VERSION / encoder_sha
    cache.mkdir(parents=True, exist_ok=True)
    store, feature_log = {}, []
    for index, video in enumerate(videos, 1):
        tick = time.perf_counter()
        file = cache / f"{video['source_sha256']}.npz"
        hit = file.is_file()
        if hit:
            with np.load(file, allow_pickle=False) as arrays:
                times, features = arrays["times"], arrays["features"]
        else:
            times, features = extractor.extract(args.data_dir / video["video"])
            temporary = file.with_suffix(".tmp.npz")
            np.savez(temporary, times=times, features=features)
            temporary.replace(file)
        if features.shape != (len(times), 1024) or not len(times) or not np.isfinite(features).all():
            raise ValueError(f"Invalid feature cache: {file}")
        store[video["video"]] = times, features
        feature_log.append({"video": video["video"], "seconds": time.perf_counter() - tick, "cache_hit": hit, "frames": len(times)})
        print(f"FEATURES {index}/{len(videos)} {video['video']}: {len(times)} frames", flush=True)
    predictions, folds = nested_evaluation(videos, store)
    common.write_csv(args.out_dir / "predictions_oof.csv", list(predictions[0]), predictions)
    common.dump_json(args.out_dir / "folds.json", {"folds": folds})
    measured = metrics(predictions)
    errors = [r["interval_error_sec"] for r in predictions if r["actual"] and r["predicted"]]
    measured["mean_interval_error_on_tp_sec"] = float(np.mean(errors)) if errors else None
    baselines = {}
    for label, name in ((0, "always_normal"), (1, "always_violation")):
        baseline = metrics([{**r, "predicted": label, "predicted_timecode_sec": 0.} for r in predictions])
        for key in ("mean_timecode_error_sec", "localized_true_positives"):
            baseline.pop(key)
        baselines[name] = baseline
    # Predeclared screen, not certification; do not change after looking at OOF results.
    gate = (measured["recall"] >= .8 and measured["precision"] >= .7
            and measured["balanced_accuracy"] >= .7 and measured["f1"] > baselines["always_violation"]["f1"])
    final_head, final_log = fit_head(videos, store)
    threshold = float(np.median([fold["threshold"] for fold in folds]))
    model_dir = args.out_dir / "model"
    model_dir.mkdir()
    shutil.copy2(args.encoder_weights, model_dir / "encoder_resnet18.pt")
    np.savez(model_dir / "head.npz", **final_head.state())
    config = {"format": "cashier-independent-model-v4", "feature_version": FEATURE_VERSION,
        "threshold": threshold, "threshold_policy": "median of four inner-only calibrated thresholds",
        "features": "2 views x 512; local mean + change from first 4 frames = 2048; 2 fps",
        "epochs": EPOCHS, "weight_decay": WEIGHT_DECAY, "legal_rule_classifier": False,
        "dataset_sha256": common.sha256_file(args.dataset), "versions": common.environment_versions(),
        "source_provenance": common.source_provenance(), "quality_screen_passed": gate,
        "files_sha256": {n: common.sha256_file(model_dir / n) for n in ("encoder_resnet18.pt", "head.npz")}}
    common.dump_json(model_dir / "config.json", config)
    final_predictions = [row_prediction(final_head, v, store, threshold) for v in videos]
    # These are for reload verification ONLY, never reported as generalization metrics.
    common.write_csv(args.out_dir / "predictions_final_resubstitution.csv", list(final_predictions[0]), final_predictions)
    result = {"training_complete": True, "quality_screen_passed": gate, "model_ready_for_production": False,
        "quality_screen": "OOF recall >= .8, precision >= .7, balanced_accuracy >= .7, F1 > always_violation",
        "evaluation_role": dataset["evaluation_role"], "oof": measured, "baselines": baselines,
        "coverage": {"eligible": len(videos), "unique_total": len(dataset["videos"]),
                     "excluded": len(dataset["videos"]) - len(videos), "positives": sum(v["reviewed_label"] for v in videos)},
        "final_model_threshold": threshold, "legal_rule_classifier": False,
        "dataset_sha256": config["dataset_sha256"], "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "wall_clock_sec": time.perf_counter() - started}
    common.dump_json(args.out_dir / "metrics.json", result)
    common.dump_json(args.out_dir / "train_log.json", {"feature_extraction": feature_log, "final_training": final_log,
        "outer_labels_used_for_threshold_or_model_selection": False, "fixed_epochs": EPOCHS})
    write_report(args.out_dir, args.data_dir, dataset, predictions, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    main()
