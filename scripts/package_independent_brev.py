"""Build the private v4 code/metadata bundle, without videos, frames or weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import zipfile

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.laptop_common import sha256_file

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ["scripts/__init__.py", "scripts/laptop_common.py", "scripts/independent_common.py", "scripts/train_independent.py",
           "scripts/run_independent_brev.py", "scripts/prepare_independent_review.py", "requirements.txt", "requirements-ml.txt"]
INPUTS = ["dataset_manifest.json", "decisions_v4.json", "summary.json", "video_review.csv",
          "excluded.csv", "labels_reviewed.csv", "labels_organizer_original.csv"]


def package(dataset: Path, output: Path) -> None:
    files = {name: ROOT / name for name in SOURCES}
    files.update({f"dataset/{name}": dataset / name for name in INPUTS})
    files["README.md"] = ROOT / "docs/TRAIN_INDEPENDENT.md"
    manifest = {"format": "cashier-independent-brev-bundle-v4", "contains_videos": False,
        "contains_frames": False, "contains_weights": False,
        "files_sha256": {name: sha256_file(path) for name, path in files.items()}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED) as archive:
        for name, path in files.items():
            archive.write(path, "brev-reviewed-v4/" + name)
        archive.writestr("brev-reviewed-v4/bundle_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({"path": str(output.resolve()), "bytes": output.stat().st_size, "sha256": sha256_file(output)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data/annotations/reconciled_v4")
    parser.add_argument("--output", type=Path, default=ROOT / "data/exports/brev-reviewed-v4.zip")
    args = parser.parse_args()
    package(args.dataset_dir, args.output)
