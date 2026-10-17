"""Evaluate the saved baseline by analyzing whole videos with no feature cache and no ground truth.

Two passes over the same videos: pass 1 (cold: includes model load and CUDA warm-up) and pass 2 (warm).
Only aggregates go to the public metrics JSON; per-video rows go to data/runs/.
"""

import argparse
import platform
import sys
import time
import traceback
from pathlib import Path
from statistics import mean

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from laptop_common import (  # noqa: E402
    MODEL_DIR, PROJECT_ROOT, RUN_DIR, SPLITS, analyze_video, dump_json, environment_versions, git_sha,
    load_model, load_split, video_metrics, write_csv,
)

PREDICTION_FIELDS = ["video", "split", "actual", "predicted", "score", "true_timecode_sec",
                     "predicted_timecode_sec", "elapsed_sec", "error"]


def constant_baselines(rows: list[dict]) -> dict:
    out = {}
    for name, value in (("always_violation", 1), ("never_violation", 0)):
        out[name] = video_metrics([
            {"actual": r["actual"], "predicted": value, "true_timecode_sec": r["true_timecode_sec"] or 0.0,
             "predicted_timecode_sec": 0.0} for r in rows])
        out[name].pop("mean_timecode_error_sec"), out[name].pop("localized_true_positives")
    return out


def run_pass(extractor, head, threshold, videos, split_of) -> list[dict]:
    rows = []
    for video in videos:
        start = time.perf_counter()
        row = {"video": video["video"], "split": split_of[video["video"]], "actual": video["has_violation"],
               "true_timecode_sec": video["timecode_sec"], "score": None, "predicted": None,
               "predicted_timecode_sec": None, "error": ""}
        try:
            row.update({k: v for k, v in analyze_video(extractor, head, threshold, video["path"]).items()
                        if k != "frames"})
        except Exception as exc:  # recorded, never counted as a negative prediction
            row["error"] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        row["elapsed_sec"] = time.perf_counter() - start
        rows.append(row)
        print(f"{row['split']:5s} {row['video']:18s} score={row['score']} pred={row['predicted']} "
              f"{row['elapsed_sec']:.2f}s {row['error']}", flush=True)
    return rows


def timing(rows: list[dict]) -> dict:
    values = [r["elapsed_sec"] for r in rows]
    return {"mean_sec": mean(values), "max_sec": max(values), "min_sec": min(values), "total_sec": sum(values)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    parser.add_argument("--out-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--metrics-json", type=Path, default=PROJECT_ROOT / "reports" / "laptop_baseline_metrics.json")
    args = parser.parse_args()

    import psutil
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    process = psutil.Process()
    videos, split_of = [], {}
    for split in args.splits:
        for video in load_split(split):
            videos.append(video)
            split_of[video["video"]] = split

    load_start = time.perf_counter()
    extractor, head, config = load_model(args.model_dir, args.device, args.batch_size)
    load_seconds = time.perf_counter() - load_start
    threshold = config["threshold"]
    cold = run_pass(extractor, head, threshold, videos, split_of)
    peak_rss_mib = process.memory_info().peak_wset / 2**20 if hasattr(process.memory_info(), "peak_wset") else None
    warm = run_pass(extractor, head, threshold, videos, split_of)
    peak_vram_mib = torch.cuda.max_memory_allocated() / 2**20 if args.device == "cuda" else None

    deviation = max((abs(a["score"] - b["score"]) for a, b in zip(cold, warm)
                     if a["score"] is not None and b["score"] is not None), default=None)
    predictions = [{**row, "elapsed_sec": round(row["elapsed_sec"], 3)} for row in warm]
    write_csv(args.out_dir / "predictions.csv", PREDICTION_FIELDS, predictions)
    write_csv(args.out_dir / "predictions_cold_pass.csv", PREDICTION_FIELDS, cold)

    def split_summary(split: str) -> dict:
        rows = [r for r in warm if r["split"] == split]
        ok = [r for r in rows if not r["error"]]
        cold_rows = [r for r in cold if r["split"] == split]
        summary = {
            "videos": len(rows), "processing_errors": len(rows) - len(ok),
            "positive": sum(r["actual"] for r in rows), "negative": sum(1 - r["actual"] for r in rows),
            "model": video_metrics(ok), "constant_baselines": constant_baselines(rows),
            "analysis_time_warm": timing(rows), "analysis_time_cold_pass": timing(cold_rows),
        }
        positives = [r for r in ok if r["actual"] == 1]
        summary["missed_positive_videos"] = sum(1 for r in positives if r["predicted"] == 0)
        if split == "train":
            summary["note"] = "resubstitution on training videos; NOT a result on new videos"
        if split == "val":
            summary["note"] = "validation videos were used to choose the weight decay, epochs and threshold"
        return summary

    metrics = {
        "model": args.model_dir.name, "code_commit": git_sha(), "threshold": threshold,
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
            "vram_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2) if args.device == "cuda" else None,
            "cpu_logical_cores": psutil.cpu_count(), "ram_gib": round(psutil.virtual_memory().total / 2**30, 1),
            "os": platform.platform(), "device_used": args.device,
        },
        "versions": environment_versions(),
        "model_load_sec": load_seconds,
        "cold_first_video_sec": cold[0]["elapsed_sec"],
        "peak_process_rss_mib_after_cold_pass": peak_rss_mib,
        "peak_cuda_allocated_mib": peak_vram_mib,
        "max_abs_score_difference_between_passes": deviation,
        "weights_bytes": {p.name: p.stat().st_size for p in sorted(args.model_dir.glob("*")) if p.is_file()},
        "splits": {split: split_summary(split) for split in args.splits},
    }
    dump_json(args.metrics_json, metrics)
    print(f"wrote {args.metrics_json} and {args.out_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
