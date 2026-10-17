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
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from laptop_common import MODEL_DIR, RUN_DIR, dump_json, load_split, read_csv  # noqa: E402

TOLERANCE = 1e-4


def child(model_dir: Path, device: str, names: list[str]) -> None:
    import socket

    def blocked(*_args, **_kwargs):
        raise RuntimeError("network access is disabled during model verification")

    socket.socket.connect = blocked  # any download attempt would fail loudly
    socket.create_connection = blocked

    from laptop_common import analyze_video, load_model

    extractor, head, config = load_model(model_dir, device)
    by_name = {v["video"]: v for split in ("train", "val", "test") for v in load_split(split)}
    result = {name: analyze_video(extractor, head, config["threshold"], by_name[name].get("path"))
              for name in names}
    print("RESULT " + json.dumps(result))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--videos", type=int, default=3, help="number of test videos to re-analyze")
    parser.add_argument("--child", nargs="*", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child is not None:
        child(args.model_dir, args.device, args.child)
        return

    stored = {r["video"]: r for r in read_csv(RUN_DIR / "predictions.csv")}
    names = [v["video"] for v in load_split("test")][:args.videos]
    runs = []
    for _ in range(2):
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--model-dir", str(args.model_dir),
             "--device", args.device, "--child", *names],
            capture_output=True, text=True, encoding="utf-8", check=True)
        line = next(l for l in completed.stdout.splitlines() if l.startswith("RESULT "))
        runs.append(json.loads(line[len("RESULT "):]))
    checks = []
    for name in names:
        a, b, ref = runs[0][name], runs[1][name], float(stored[name]["score"])
        checks.append({
            "child_vs_child": abs(a["score"] - b["score"]),
            "child_vs_stored_predictions": abs(a["score"] - ref),
            "same_decision": a["predicted"] == b["predicted"] == int(stored[name]["predicted"]),
            "same_timecode": abs(a["predicted_timecode_sec"] - b["predicted_timecode_sec"]) < 1e-9,
        })
    ok = all(c["child_vs_child"] <= TOLERANCE and c["child_vs_stored_predictions"] <= TOLERANCE
             and c["same_decision"] and c["same_timecode"] for c in checks)
    report = {"videos_checked": len(names), "tolerance": TOLERANCE, "network_disabled_in_child": True,
              "max_child_vs_child": max(c["child_vs_child"] for c in checks),
              "max_child_vs_stored": max(c["child_vs_stored_predictions"] for c in checks), "passed": ok}
    dump_json(RUN_DIR / "reload_check.json", report)
    print(json.dumps(report, indent=2))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
