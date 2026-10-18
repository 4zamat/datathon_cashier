"""Build a small private handoff: source + reviewed metadata, no videos/frames/weights."""
import argparse
import json
from pathlib import Path
import sys
import zipfile

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.laptop_common import sha256_file

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = ["scripts/laptop_common.py", "scripts/reviewed_common.py", "scripts/train_reviewed.py",
                "scripts/run_reviewed_brev.py", "scripts/report_reviewed.py", "requirements.txt", "requirements-ml.txt"]
INPUT_FILES = ["dataset_manifest.json", "summary.json", "review_decisions.json", "human_observations.json",
               "events.csv", "event_review.csv", "scene_groups.csv", "duplicates.csv", "labels_organizer_original.csv",
               "train.csv", "val.csv", "test.csv"]


def package(dataset_dir: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    files = {name: ROOT / name for name in SOURCE_FILES}
    files.update({f"dataset/{name}": dataset_dir / name for name in INPUT_FILES})
    files["README.md"] = ROOT / "docs/TRAIN_REVIEWED.md"
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = {"format": "cashier-reviewed-brev-bundle-v3", "files_sha256": {n: sha256_file(p) for n, p in files.items()},
                "contains_source_videos": False, "contains_weights": False}
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED) as archive:
        for name, path in files.items():
            archive.write(path, f"brev-reviewed-v3/{name}")
        archive.writestr("brev-reviewed-v3/bundle_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({"path": str(output.resolve()), "bytes": output.stat().st_size, "sha256": sha256_file(output)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data/annotations/reconciled_v3")
    parser.add_argument("--output", type=Path, default=ROOT / "data/exports/brev-reviewed-v3.zip")
    args = parser.parse_args()
    package(args.dataset_dir, args.output)
