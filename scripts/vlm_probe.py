"""Evaluate a frozen local VLM on videos; labels are joined only after inference."""
from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
import os
from pathlib import Path
import platform
import socket
import sys
import time

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rules import RuleCatalog
from pipeline.vlm import (EXPERIMENT_VERSION, MODEL_ID, MODEL_REVISION, WEIGHT_HASHES, InvalidModelResponse,
                          LocalVlmEngine, jpeg, to_result)
from scripts.laptop_common import dump_json, sha256_file

SMOKE_NUMBERS = {4, 5, 26, 27, 40, 43}


def apply_observations(row: dict, observations, samples, video: dict, catalog: RuleCatalog) -> None:
    """One adapter for live inference and explicitly labelled saved-response replay."""
    observed = [c for c in observations.checks if c.status == "observed"]
    uncertain = [c for c in observations.checks if c.status == "uncertain"]
    row.update(status="done", checks=observations.model_dump()["checks"])
    row["preview_frames"] = [jpeg(samples[i]).model_dump() for i in (0, len(samples) // 2, len(samples)-1)]
    row["predicted_visual"] = 1 if observed else (None if uncertain else 0)
    frame_ids = sorted({i for c in observed for i in c.evidence_ids})
    row["evidence"] = [jpeg(samples[i]).model_dump() for i in frame_ids]
    row["api_status"] = "done"
    try:
        public = to_result(observations, samples)
        row["api_violations"] = [v.model_dump() for v in catalog.enrich(public)]
        row["api_verdict"] = public.verdict
    except ValueError as error:
        row.update(api_status="needs_review", api_error=str(error))
    # Ground truth is joined after inference; never used by parse_report or to_result.
    aliases = {"personal_drink_in_cash_area": "personal_items_in_cash_area"}
    matches = [(c, w) for c in observed for w in video["positive_windows"]
               if c.type == aliases.get(w["event_type"], w["event_type"])]
    row["matching_event_type"] = bool(matches)
    row["matching_interval_error_sec"] = min(max(w["start_sec"] - samples[c.start_frame].timestamp_sec, 0,
        samples[c.start_frame].timestamp_sec - w["end_sec"]) for c, w in matches) if matches else None


def selection(dataset: dict, scope: str) -> list[dict]:
    eligible = [v for v in dataset["videos"] if v["eligible"]]
    if scope == "smoke":
        return [v for v in eligible if int(Path(v["video"]).stem.split("-")[-1]) in SMOKE_NUMBERS]
    return eligible if scope == "reviewed" else dataset["videos"]


def summarize(rows: list[dict], scope: str, expected_count: int) -> dict:
    known = [r for r in rows if r["actual"] is not None]
    scored = [r for r in known if r.get("predicted_visual") is not None]
    tp = sum(r["actual"] == 1 and r["predicted_visual"] == 1 for r in scored)
    fp = sum(r["actual"] == 0 and r["predicted_visual"] == 1 for r in scored)
    fn = sum(r["actual"] == 1 and r["predicted_visual"] == 0 for r in scored)
    tn = sum(r["actual"] == 0 and r["predicted_visual"] == 0 for r in scored)
    unresolved_positive = sum(r["actual"] == 1 and r.get("predicted_visual") is None for r in known)
    interval_errors = [r["matching_interval_error_sec"] for r in scored if r.get("matching_interval_error_sec") is not None]
    return {"scope": scope, "evaluation_role": "development diagnostic; previously reviewed videos, no training",
        "requested": expected_count, "processed": len(rows), "known_labels": len(known), "scored": len(scored),
        "abstentions_or_errors_with_known_label": len(known) - len(scored), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision_on_scored": (tp / (tp + fp) if tp + fp else 0.) if scored else None,
        "recall_on_all_known_positives": (tp / (tp + fn + unresolved_positive) if tp + fn + unresolved_positive else 0.) if scored else None,
        "f1_on_scored": (2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.) if scored else None,
        "matching_event_type_videos": sum(r.get("matching_event_type", False) for r in scored),
        "mean_interval_error_matching_type_sec": sum(interval_errors) / len(interval_errors) if interval_errors else None,
        "api_ready_verdicts": sum(r.get("api_status") == "done" for r in rows),
        "errors": sum(r.get("status") == "failed" for r in rows),
        "distinct_raw_responses": len({r["raw_response"] for r in rows if r.get("raw_response")}),
        "all_selected_processed_without_errors": len(rows) == expected_count and all(r.get("status") == "done" for r in rows),
        "production_quality_demonstrated": False}


def report_html(rows: list[dict], summary: dict) -> str:
    cards = []
    for row in rows:
        detail = {k: v for k, v in row.items() if k not in {"evidence", "preview_frames", "api_violations"}}
        frames = ''.join(f'<figure><img src="data:image/jpeg;base64,{f["image_base64"]}"><figcaption>{f["timestamp_sec"]:.2f} с</figcaption></figure>' for f in row.get("evidence", []))
        previews = ''.join(f'<figure><img src="data:image/jpeg;base64,{f["image_base64"]}"><figcaption>{f["timestamp_sec"]:.2f} с</figcaption></figure>' for f in row.get("preview_frames", []))
        cards.append(f'<article><h2>{escape(row["video"])}</h2><p>Разметка: {row["actual"]}; визуальный кандидат: {row.get("predicted_visual")}; API: {row.get("api_status", "failed")}</p>'
                     + '<p>Обзорные кадры (не доказательство нарушения):</p><div class="frames">' + previews
                     + '</div><p>Выбранные моделью доказательства:</p><div class="frames">' + frames
                     + '</div><pre>' + escape(json.dumps(detail, ensure_ascii=False, indent=2)) + '</pre></article>')
    return ('<!doctype html><html lang="ru"><meta charset="utf-8"><title>VLM: проверка событий</title>'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><style>'
        'body{font:16px system-ui;max-width:1100px;margin:30px auto;padding:16px;background:#eef1f6;color:#182132}'
        'article{background:white;padding:20px;margin:20px 0;border-radius:10px}.frames{display:flex;flex-wrap:wrap;gap:12px}'
        'figure{margin:0}img{height:320px;max-width:100%;object-fit:contain}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>'
        '<h1>Готовая VLM: события и реальные кадры</h1><p>Модель не обучалась на этих видео. В её запрос не передаются имена файлов, метки, комментарии или размеченные времена. '
        'Проверка smoke выбрана для диагностики старых ошибок и не является независимым тестом качества. '
        'Визуальный кандидат без применимого правила остаётся для ручной проверки.</p><pre>'
        + escape(json.dumps(summary, ensure_ascii=False, indent=2)) + '</pre>' + ''.join(cards) + '</html>')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=["smoke", "reviewed", "all"], default="smoke")
    args = parser.parse_args()
    # Prevent Python network connections in the process that reads private videos.
    def blocked(*_args, **_kwargs):
        raise RuntimeError("Network disabled during local VLM inference")
    socket.create_connection = blocked
    socket.socket.connect = blocked
    args.out_dir.mkdir(parents=True, exist_ok=False)
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    videos = selection(dataset, args.scope)
    if args.scope == "smoke" and len(videos) != 6:
        raise ValueError("Smoke test requires six specific reviewed clips")
    for video in videos:
        path = (args.data_dir / video["video"]).resolve()
        if not path.is_relative_to(args.data_dir.resolve()) or sha256_file(path) != video["source_sha256"]:
            raise ValueError(f"Missing or changed video: {video['video']}")
    print("Checking downloaded weight checksums (8.9 GB)...", flush=True)
    for name, expected in WEIGHT_HASHES.items():
        if sha256_file(args.model_dir / name) != expected:
            raise ValueError(f"Downloaded weight checksum differs: {name}")
    started = time.perf_counter()
    print("Loading Qwen3-VL-4B on CUDA...", flush=True)
    engine = LocalVlmEngine(args.model_dir)
    import torch, transformers

    catalog = RuleCatalog(Path(__file__).resolve().parents[1] / "rules/rules.yaml")
    config = {"model": MODEL_ID, "revision": MODEL_REVISION, "weights_sha256": WEIGHT_HASHES,
        "experiment_version": EXPERIMENT_VERSION,
        "observation_prompt_sha256": hashlib.sha256(engine.observation_prompt.encode()).hexdigest(),
        "prompt_sha256": hashlib.sha256(engine.prompt.encode()).hexdigest(), "max_frames": engine.max_frames,
        "max_pixels": engine.max_pixels, "labels_sent_to_model": False, "network_python_connections_blocked": True,
        "dataset_sha256": sha256_file(args.dataset), "load_seconds": time.perf_counter() - started,
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__}}
    dump_json(args.out_dir / "config.json", config)
    print("VISION CONTROL: checking two synthetic images before private videos...", flush=True)
    try:
        control = engine.vision_control()
    except Exception as error:
        control = {"passed": False, "error": f"{type(error).__name__}: {error}"}
    dump_json(args.out_dir / "vision_control.json", control)
    print(f"VISION CONTROL: {control['passed']}", flush=True)
    if not control["passed"]:
        raise RuntimeError("Vision control failed; inspect vision_control.json before processing videos")
    rows = []
    for index, video in enumerate(videos, 1):
        print(f"ANALYZE {index}/{len(videos)} {video['video']}", flush=True)
        row = {"video": video["video"], "actual": video["reviewed_label"], "predicted_visual": None}
        try:
            observations, samples, metadata = engine.analyze(args.data_dir / video["video"])
            row.update(metadata)
            apply_observations(row, observations, samples, video, catalog)
        except Exception as error:
            row.update(status="failed", error=f"{type(error).__name__}: {error}")
            if isinstance(error, InvalidModelResponse):
                row.update(error.metadata)
            torch.cuda.empty_cache()
        rows.append(row)
        summary = summarize(rows, args.scope, len(videos))
        dump_json(args.out_dir / "predictions.json", {"videos": rows})
        dump_json(args.out_dir / "metrics.json", summary)
        (args.out_dir / "review.html").write_text(report_html(rows, summary), encoding="utf-8")
        print(json.dumps({k:v for k,v in row.items() if k in {"video","status","predicted_visual","api_status","elapsed_sec","error","matching_event_type"}},ensure_ascii=False), flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if summary["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    main()
