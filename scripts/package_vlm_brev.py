"""Package an isolated VLM probe and existing API, without private videos or weights."""
import argparse
import json
from pathlib import Path
import sys
import zipfile

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.laptop_common import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/exports/brev-vlm-v3-verify.zip")
    args = parser.parse_args()
    names = ["scripts/__init__.py", "scripts/laptop_common.py", "scripts/vlm_probe.py", "scripts/run_vlm_brev.py", "scripts/replay_vlm.py",
             "requirements.txt", "requirements-vlm.txt", "pipeline/__init__.py", "pipeline/base.py", "pipeline/vlm.py",
             "pipeline/dummy.py", "pipeline/verification.py", "prompts/cashier_vlm_v2.txt", "prompts/cashier_observe_v2.txt", "rules/rules.yaml"]
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT / "app").glob("*.py")]
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT / "static").rglob("*") if p.is_file()]
    files = {name: ROOT / name for name in names}
    files["dataset/dataset_manifest.json"] = ROOT / "data/annotations/reconciled_v4/dataset_manifest.json"
    files["README.md"] = ROOT / "docs/VLM_VERIFY.md"
    files["docs/VLM_QUICKSTART.md"] = ROOT / "docs/VLM_QUICKSTART.md"
    manifest = {"format": "cashier-vlm-probe-v3-prerequisites", "contains_videos": False, "contains_weights": False,
                "files_sha256": {name: sha256_file(path) for name, path in files.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "x", zipfile.ZIP_DEFLATED) as z:
        for name, path in files.items():
            z.write(path, "brev-vlm-v3-verify/" + name)
        z.writestr("brev-vlm-v3-verify/bundle_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({"path": str(args.output), "bytes": args.output.stat().st_size, "sha256": sha256_file(args.output)}, indent=2))


if __name__ == "__main__":
    main()
