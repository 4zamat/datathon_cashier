"""Create a small, immutable service handoff without weights, data or credentials."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/exports/brev-service-v1.zip")
    args = parser.parse_args()
    names = ["requirements.txt", "requirements-vlm.txt", "pipeline/__init__.py", "pipeline/base.py",
             "pipeline/dummy.py", "pipeline/vlm.py", "pipeline/verification.py", "rules/rules.yaml",
             "scripts/__init__.py", "scripts/serve_vlm.py", "scripts/check_live_api.py",
             "prompts/cashier_vlm_v2.txt", "prompts/cashier_observe_v2.txt"]
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT / "app").glob("*.py")]
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT / "static").rglob("*") if p.is_file()]
    files = {name: ROOT / name for name in names}
    files["README.md"] = ROOT / "docs/SERVICE_BREV.md"
    manifest = {"format": "cashier-service-v1", "contains_videos": False, "contains_weights": False,
                "gpu_validation_pending": True,
                "files_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "x", zipfile.ZIP_DEFLATED) as archive:
        for name, path in files.items():
            archive.write(path, "brev-service-v1/" + name)
        archive.writestr("brev-service-v1/bundle_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({"path": str(args.output), "bytes": args.output.stat().st_size,
                      "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
