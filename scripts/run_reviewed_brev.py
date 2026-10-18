"""One-command local-data Brev run, cold offline reload, and private result export."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run_logged(command: list[str], log_path: Path, env: dict) -> None:
    with log_path.open("a", encoding="utf-8") as stream:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", env=env, cwd=ROOT)
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
            stream.flush()
        if process.wait():
            raise RuntimeError(f"Command failed ({process.returncode}); see {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path.cwd() / "data")
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset/dataset_manifest.json")
    parser.add_argument("--encoder-weights", type=Path)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    data = args.data_dir.resolve()
    dataset_path = args.dataset.resolve()
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    # Confirm the supplied bundle before execution; never install packages or download weights.
    bundle_manifest = ROOT / "bundle_manifest.json"
    if bundle_manifest.exists():
        for relative, expected in json.loads(bundle_manifest.read_text(encoding="utf-8"))["files_sha256"].items():
            if digest(ROOT / relative) != expected:
                raise ValueError(f"Bundle file changed: {relative}")
    weights = args.encoder_weights
    if weights is None:
        candidates = sorted((data.parent / "models").glob("*/encoder_resnet18.pt"))
        candidates += sorted((data / "imports").glob("*/model/encoder_resnet18.pt"))
        if not candidates:
            raise FileNotFoundError("No existing encoder found. Pass --encoder-weights path/to/encoder_resnet18.pt from the saved Brev v2 run.")
        weights = candidates[0]
    weights = weights.resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("reviewed-v3-%Y%m%d-%H%M%S")
    if Path(run_id).name != run_id or not run_id:
        raise ValueError("run-id must be a directory name")
    out = data / "runs" / run_id
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    log = out.parent / f"{run_id}.log"
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONUNBUFFERED": "1",
           "PYTHONUTF8": "1", "OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    common_command = [sys.executable, "-u", str(ROOT / "scripts/train_reviewed.py"),
        "--data-dir", str(data), "--dataset", str(dataset_path), "--encoder-weights", str(weights),
        "--out-dir", str(out), "--device", args.device, "--epochs", str(args.epochs)]
    print(f"DATA: {data}\nENCODER: {weights}\nOUTPUT: {out}\nLOG: {log}", flush=True)
    status = {"complete": False, "dataset_sha256": digest(dataset_path), "encoder_sha256": digest(weights)}
    try:
        run_logged(common_command, log, env)
        predictions_path = out / "predictions.csv"
        import csv
        with predictions_path.open(encoding="utf-8", newline="") as stream:
            predictions = list(csv.DictReader(stream))
        sample = next(v for v in predictions if v["split"] == "test")
        run_logged([*common_command, "--reload-video", sample["video"]], log, env)
        reloaded = json.loads((out / "reload_prediction.json").read_text(encoding="utf-8"))
        reload_ok = (abs(float(sample["score"]) - reloaded["score"]) <= 1e-5
                     and int(sample["predicted"]) == reloaded["predicted"]
                     and abs(float(sample["predicted_timecode_sec"]) - reloaded["predicted_timecode_sec"]) <= .51)
        (out / "reload_check.json").write_text(json.dumps({"passed": reload_ok, "network_disabled": True,
            "new_process": True, "video": sample["video"], "prediction": reloaded}, ensure_ascii=False, indent=2), encoding="utf-8")
        if not reload_ok:
            raise RuntimeError("Saved-model reload prediction differs")
        run_logged([sys.executable, str(ROOT / "scripts/report_reviewed.py"), "--data-dir", str(data), "--run-dir", str(out)], log, env)
        status["complete"] = True
    except Exception as error:
        status["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    finally:
        out.mkdir(parents=True, exist_ok=True)
        source = out / "source"
        for pattern in ("scripts/*.py", "requirements*.txt"):
            for path in ROOT.glob(pattern):
                destination = source / path.relative_to(ROOT)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
        if not (out / "dataset_manifest.json").exists():
            shutil.copy2(dataset_path, out / "dataset_manifest.json")
        for name in ("summary.json", "review_decisions.json", "events.csv", "scene_groups.csv", "event_review.csv",
                     "human_observations.json", "labels_organizer_original.csv", "duplicates.csv"):
            source_input = dataset_path.parent / name
            if source_input.is_file():
                (out / "inputs").mkdir(exist_ok=True)
                shutil.copy2(source_input, out / "inputs" / name)
        if log.exists():
            shutil.copy2(log, out / "console.log")
        status["files_sha256"] = {p.relative_to(out).as_posix(): digest(p) for p in out.rglob("*") if p.is_file()}
        (out / "run_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        export = data / "exports" / f"{run_id}.zip"
        export.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(export, "x", zipfile.ZIP_DEFLATED) as archive:
            for path in out.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(out))
        if (out / "metrics.json").exists():
            shutil.copy2(out / "metrics.json", export.with_suffix(".metrics.json"))
        print(f"\nCOMPLETE: {status['complete']}\nZIP: {export}\nSHA256: {digest(export)}", flush=True)
        for label, artifact in (("METRICS", export.with_suffix(".metrics.json")),
                                ("VISUAL REPORT", out / "review.html")):
            print(f"{label}: {artifact if artifact.is_file() else 'not created'}", flush=True)
    if not status["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
