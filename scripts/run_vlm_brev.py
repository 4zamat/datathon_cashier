"""Run a private offline VLM probe and export diagnostics, without model weights."""
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


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path.cwd() / "data")
    parser.add_argument("--model-dir", type=Path, default=Path.cwd() / "models/qwen3-vl-4b")
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset/dataset_manifest.json")
    parser.add_argument("--scope", choices=["smoke", "reviewed", "all"], default="smoke")
    args = parser.parse_args()
    data, model, dataset = args.data_dir.resolve(), args.model_dir.resolve(), args.dataset.resolve()
    manifest = ROOT / "bundle_manifest.json"
    if manifest.is_file():
        for relative, expected in json.loads(manifest.read_text(encoding="utf-8"))["files_sha256"].items():
            path = (ROOT / relative).resolve()
            if not path.is_relative_to(ROOT) or sha(path) != expected:
                raise ValueError(f"Changed bundle file: {relative}")
    if not model.is_dir():
        raise FileNotFoundError("Download the pinned Qwen model first; see README.md")
    run_id = datetime.now(timezone.utc).strftime(f"vlm-v3-verify-{args.scope}-%Y%m%d-%H%M%S")
    out = data / "runs" / run_id
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    log = out.parent / f"{run_id}.log"
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
           "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false"}
    cmd = [sys.executable, "-u", str(ROOT / "scripts/vlm_probe.py"), "--data-dir", str(data),
           "--model-dir", str(model), "--dataset", str(dataset), "--scope", args.scope, "--out-dir", str(out)]
    status = {"complete": False, "scope": args.scope, "quality_approved": False}
    print(f"PYTHON: {sys.executable}\nMODEL: {model}\nOUTPUT: {out}\nLOG: {log}", flush=True)
    try:
        with log.open("w", encoding="utf-8") as stream:
            child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace")
            try:
                for line in child.stdout:
                    print(line, end="", flush=True)
                    stream.write(line)
                    stream.flush()
                code = child.wait()
            except BaseException:
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                raise
        status["complete"] = code == 0
        status["exit_code"] = code
    except BaseException as error:
        status["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    finally:
        out.mkdir(exist_ok=True)
        for pattern in ("scripts/*.py", "pipeline/*.py", "app/*.py", "prompts/*.txt", "rules/*.yaml", "requirements*.txt"):
            for path in ROOT.glob(pattern):
                target = out / "source" / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        shutil.copy2(dataset, out / "dataset_manifest.json")
        if log.is_file():
            shutil.copy2(log, out / "console.log")
        status["files_sha256"] = {p.relative_to(out).as_posix(): sha(p) for p in out.rglob("*") if p.is_file()}
        (out / "run_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        exports = data / "exports"
        exports.mkdir(exist_ok=True)
        archive = exports / f"{run_id}.zip"
        with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as z:
            for path in out.rglob("*"):
                if path.is_file():
                    z.write(path, path.relative_to(out))
        metrics = exports / f"{run_id}.metrics.json"
        if (out / "metrics.json").is_file():
            shutil.copy2(out / "metrics.json", metrics)
        print(f"\nCOMPLETE: {status['complete']}\nQUALITY: requires inspection, not approved"
              f"\nZIP (send this): {archive}\nSHA256: {sha(archive)}", flush=True)
        print(f"METRICS: {metrics if metrics.is_file() else 'not created'}", flush=True)
        print(f"VISUAL REPORT: {out / 'review.html' if (out / 'review.html').is_file() else 'not created'}", flush=True)
    if not status["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
