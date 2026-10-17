"""Export/verify a private experiment backup without importing Torch or copying videos.

Run before another training attempt or losing access to a laptop/VM. Source files are
the export-time snapshot; a legacy model's training commit is not silently corrected.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILES = ("encoder_resnet18.pt", "head.npz", "config.json")
SPLIT_FILES = ("train.csv", "val.csv", "test.csv", "excluded.csv", "summary.json")
RUN_FILES = ("train_log.json", "predictions.csv", "predictions_cold_pass.csv", "reload_check.json")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_archive(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or "manifest.json" not in names:
            raise ValueError("Duplicate paths or missing manifest")
        for name in names:
            parts = PurePosixPath(name)
            if parts.is_absolute() or ".." in parts.parts or "\\" in name or ":" in name:
                raise ValueError("Unsafe archive path")
        manifest = json.loads(archive.read("manifest.json"))
        if set(names) != set(manifest["files"]) | {"manifest.json"}:
            raise ValueError("Archive contents differ from manifest")
        for name, expected in manifest["files"].items():
            h, size = hashlib.sha256(), 0
            with archive.open(name) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(chunk)
                    size += len(chunk)
            if h.hexdigest() != expected["sha256"] or size != expected["bytes"]:
                raise ValueError(f"Checksum mismatch: {name}")
        return manifest


def export_experiment(root: Path, model: Path, run: Path, splits: Path, output: Path) -> dict:
    root = root.resolve()
    files: dict[str, Path] = {}
    missing = []
    for prefix, directory, required in (("model", model, MODEL_FILES), ("run", run, RUN_FILES),
                                         ("splits", splits, SPLIT_FILES)):
        for name in required:
            source = directory / name
            if source.is_file():
                files[f"{prefix}/{name}"] = source
            else:
                missing.append(f"{prefix}/{name}")
    # Explicit allowlist: never copy .env, credentials, raw video, crops, caches or .venv.
    for pattern in ("scripts/*.py", "app/*.py", "pipeline/*.py", "tests/*.py", "rules/*.yaml",
                    "requirements*.txt", "Dockerfile", "README.md", "docs/*.md"):
        for source in root.glob(pattern):
            files["source/" + source.relative_to(root).as_posix()] = source
    for source in (run / "metrics.json", root / "reports/laptop_baseline_metrics.json",
                   root / "reports/laptop_baseline.md"):
        if source.is_file():
            files[("run/" if source.parent == run else "historical_reports/") + source.name] = source
    for source in files.values():
        if not source.resolve().is_relative_to(root):
            raise ValueError(f"Artifact must be inside project: {source}")
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        head = "unknown"
    config_path = model / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    manifest = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "complete": not missing, "missing_required": missing,
        "export_head_commit": head, "training_commit_claimed_by_model": config.get("code_commit"),
        "training_provenance": config.get("provenance"),
        "source_snapshot_note": "Export-time source. For legacy models, exact training source is unproven.",
        "python": platform.python_version(), "platform": platform.platform(),
        "installed_packages": sorted({f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()}),
        "files": {name: {"bytes": source.stat().st_size, "sha256": digest(source)}
                  for name, source in sorted(files.items())},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, source in sorted(files.items()):
            archive.write(source, name)
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    verify_archive(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/laptop_baseline")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "data/runs/laptop_baseline")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data/splits/laptop_baseline")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path, help="Verify hashes/CRC without extracting or loading a model")
    args = parser.parse_args()
    if args.verify:
        report = verify_archive(args.verify)
        path = args.verify
    else:
        if args.output is None:
            parser.error("Specify --output data/exports/<unique-name>.zip or --verify <archive>")
        path = args.output
        report = export_experiment(ROOT, args.model_dir, args.run_dir, args.split_dir, path)
    print(json.dumps({"archive": str(path), "sha256": digest(path), "complete": report["complete"],
                      "missing_required": report["missing_required"], "files": len(report["files"])}, indent=2))
    if not report["complete"]:
        print("Partial backup preserved. Retrieve the missing files if available; do not delete this backup.", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
