"""One-command v4 Brev run; explicit interpreter, immutable inputs, private exports."""
from __future__ import annotations

import argparse
import csv
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


def run_logged(command: list[str], log: Path, env: dict) -> None:
    with log.open("a", encoding="utf-8") as stream:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", env=env, cwd=ROOT)
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
            stream.flush()
        if process.wait():
            raise RuntimeError(f"Command failed ({process.returncode}); see {log}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path.cwd() / "data")
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset/dataset_manifest.json")
    parser.add_argument("--encoder-weights", type=Path)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    data, dataset = args.data_dir.resolve(), args.dataset.resolve()
    bundle = ROOT / "bundle_manifest.json"
    if bundle.exists():
        for relative, expected in json.loads(bundle.read_text(encoding="utf-8"))["files_sha256"].items():
            path = (ROOT / relative).resolve()
            if not path.is_relative_to(ROOT) or digest(path) != expected:
                raise ValueError(f"Changed bundle file: {relative}")
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    weights = args.encoder_weights or data.parent / "models/brev_v2/encoder_resnet18.pt"
    weights = weights.resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Saved encoder not found: {weights}. Pass --encoder-weights explicitly.")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("reviewed-v4-%Y%m%d-%H%M%S")
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run-id must be a directory name")
    out = data / "runs" / run_id
    exports = data / "exports"
    log = out.with_suffix(".log")
    if out.exists() or log.exists() or (exports / f"{run_id}.zip").exists():
        raise FileExistsError(f"Refusing to overwrite run: {run_id}")
    out.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONUTF8": "1",
           "PYTHONUNBUFFERED": "1", "OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    command = [sys.executable, "-u", str(ROOT / "scripts/train_independent.py"), "--data-dir", str(data),
        "--dataset", str(dataset), "--encoder-weights", str(weights), "--out-dir", str(out), "--device", args.device]
    print(f"PYTHON: {sys.executable}\nDATA: {data}\nOUTPUT: {out}\nLOG: {log}", flush=True)
    status = {"complete": False, "dataset_sha256": digest(dataset), "encoder_sha256": digest(weights)}
    try:
        run_logged([sys.executable, "-c", "import numpy,cv2,torch,torchvision; print('DEPENDENCIES OK',torch.__version__); "
            + ("assert torch.cuda.is_available(), 'CUDA unavailable'" if args.device == "cuda" else "print('CPU requested')")], log, env)
        run_logged(command, log, env)
        with (out / "predictions_final_resubstitution.csv").open(encoding="utf-8", newline="") as stream:
            predictions = list(csv.DictReader(stream))
        sample = next(r for r in predictions if r["actual"] == "1")
        run_logged([*command, "--reload-video", sample["video"]], log, env)
        reloaded = json.loads((out / "reload_prediction.json").read_text(encoding="utf-8"))
        passed = (abs(float(sample["score"]) - reloaded["score"]) <= 1e-5
                  and int(sample["predicted"]) == reloaded["predicted"]
                  and abs(float(sample["predicted_timecode_sec"]) - reloaded["predicted_timecode_sec"]) <= .51)
        (out / "reload_check.json").write_text(json.dumps({"passed": passed, "new_process": True,
            "python_socket_connections_blocked": True, "video": sample["video"], "prediction": reloaded}, indent=2), encoding="utf-8")
        if not passed:
            raise RuntimeError("Saved-model predictions differ after offline reload")
        measured = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        status.update(complete=True, quality_screen_passed=measured["quality_screen_passed"])
    except Exception as error:
        status["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    finally:
        out.mkdir(parents=True, exist_ok=True)
        for path in [*ROOT.glob("scripts/*.py"), *ROOT.glob("requirements*.txt")]:
            destination = out / "source" / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        inputs = out / "inputs"
        inputs.mkdir(exist_ok=True)
        for name in ("dataset_manifest.json", "decisions_v4.json", "summary.json", "video_review.csv", "excluded.csv", "labels_reviewed.csv", "labels_organizer_original.csv"):
            path = dataset.parent / name
            if path.is_file():
                shutil.copy2(path, inputs / name)
        if log.exists():
            shutil.copy2(log, out / "console.log")
        status["files_sha256"] = {p.relative_to(out).as_posix(): digest(p) for p in out.rglob("*") if p.is_file()}
        (out / "run_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        exports.mkdir(parents=True, exist_ok=True)
        full = exports / f"{run_id}.zip"
        small = exports / f"{run_id}.small.zip"
        for target, include_encoder in ((full, True), (small, False)):
            with zipfile.ZipFile(target, "x", zipfile.ZIP_DEFLATED) as archive:
                for path in out.rglob("*"):
                    if path.is_file() and (include_encoder or path.name != "encoder_resnet18.pt"):
                        archive.write(path, path.relative_to(out))
                archive.writestr("export_manifest.json", json.dumps({"includes_encoder": include_encoder,
                    "encoder_sha256": status["encoder_sha256"], "complete": status["complete"]}, indent=2))
        metrics_path = exports / f"{run_id}.metrics.json"
        if (out / "metrics.json").is_file():
            shutil.copy2(out / "metrics.json", metrics_path)
        print(f"\nCOMPLETE: {status['complete']}\nQUALITY SCREEN: {status.get('quality_screen_passed', 'not measured')}"
              f"\nFULL ZIP: {full}\nSMALL ZIP (send this): {small}\nSMALL ZIP SHA256: {digest(small)}", flush=True)
        print(f"METRICS: {metrics_path if metrics_path.is_file() else 'not created'}", flush=True)
        print(f"VISUAL REPORT: {out / 'review.html' if (out / 'review.html').is_file() else 'not created'}", flush=True)
    if not status["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
