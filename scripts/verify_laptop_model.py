"""Reload the saved model in fresh processes with networking disabled and compare predictions.

The parent starts two child processes; each loads models/laptop_baseline from local files only and
re-analyzes a few videos. Scores must match each other and the stored predictions.csv.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.laptop_common import MODEL_DIR, RUN_DIR, SPLIT_DIR, dump_json, load_split, read_csv  # noqa: E402

TOLERANCE = 1e-4


def child(model_dir: Path, device: str, names: list[str], split_dir: Path) -> None:
    import socket

    def blocked(*_args, **_kwargs):
        raise RuntimeError("network access is disabled during model verification")

    socket.socket.connect = blocked  # any download attempt would fail loudly
    socket.create_connection = blocked

    from scripts.laptop_common import analyze_video, load_model

    extractor, head, config = load_model(model_dir, device)
    by_name = {v["video"]: v for split in ("train", "val", "test") for v in load_split(split, split_dir)}
    result = {name: analyze_video(extractor, head, config["threshold"], by_name[name].get("path"))
              for name in names}
    print("RESULT " + json.dumps(result))


def compare_prediction(a: dict, b: dict, stored: dict) -> dict:
    return {
        "child_vs_child": abs(a["score"] - b["score"]),
        "child_vs_stored_predictions": max(abs(r["score"] - float(stored["score"])) for r in (a, b)),
        "same_decision": a["predicted"] == b["predicted"] == int(stored["predicted"]),
        "same_timecode": abs(a["predicted_timecode_sec"] - b["predicted_timecode_sec"]) <= TOLERANCE,
        "stored_timecode_matches": all(abs(r["predicted_timecode_sec"] - float(stored["predicted_timecode_sec"]))
                                       <= TOLERANCE for r in (a, b)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--videos", type=int, default=3, help="number of test videos to re-analyze")
    parser.add_argument("--child", nargs="*", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.videos < 1:
        parser.error("--videos must be positive")
    if args.child is not None:
        child(args.model_dir, args.device, args.child, args.split_dir)
        return

    stored = {r["video"]: r for r in read_csv(args.run_dir / "predictions.csv")}
    names = [v["video"] for v in load_split("test", args.split_dir)][:args.videos]
    if not names or any(name not in stored or stored[name].get("error") for name in names):
        raise ValueError("No test videos or stored predictions are missing/failed")
    runs = []
    for _ in range(2):
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--model-dir", str(args.model_dir),
             "--device", args.device, "--split-dir", str(args.split_dir), "--child", *names],
            capture_output=True, text=True, encoding="utf-8", check=True, timeout=270)
        line = next(l for l in completed.stdout.splitlines() if l.startswith("RESULT "))
        runs.append(json.loads(line[len("RESULT "):]))
    checks = []
    for name in names:
        checks.append(compare_prediction(runs[0][name], runs[1][name], stored[name]))
    ok = all(c["child_vs_child"] <= TOLERANCE and c["child_vs_stored_predictions"] <= TOLERANCE
             and c["same_decision"] and c["same_timecode"] and c["stored_timecode_matches"] for c in checks)
    report = {"videos_checked": len(names), "tolerance": TOLERANCE, "network_disabled_in_child": True,
              "max_child_vs_child": max(c["child_vs_child"] for c in checks),
              "max_child_vs_stored": max(c["child_vs_stored_predictions"] for c in checks),
              "checks": checks, "passed": ok}
    dump_json(args.run_dir / "reload_check.json", report)
    print(json.dumps(report, indent=2))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
