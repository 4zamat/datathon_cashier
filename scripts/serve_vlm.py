"""Start the GPU API with local weights, isolated runtime files, and a persistent API key."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path.cwd() / "models/qwen3-vl-4b")
    parser.add_argument("--runtime-dir", type=Path, default=Path.cwd() / "data/service_v1")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="2")
    from pipeline.vlm import WEIGHT_HASHES, check_compiler
    import torch
    import transformers
    import uvicorn

    check_compiler()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment")
    manifest_path = ROOT / "bundle_manifest.json"
    if manifest_path.is_file():
        for name, expected in json.loads(manifest_path.read_text(encoding="utf-8"))["files_sha256"].items():
            path = (ROOT / name).resolve()
            if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Bundle integrity check failed: {name}")
    model = args.model_dir.resolve()
    print("Checking local model weights…", flush=True)
    for name, expected in WEIGHT_HASHES.items():
        with (model / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"Model checksum mismatch: {name}")
    runtime = args.runtime_dir.resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    key_file = runtime / "api-key.txt"
    key = os.environ.get("API_KEY", "").strip()
    if not key:
        if key_file.exists():
            key = key_file.read_text(encoding="utf-8").strip()
        else:
            key = secrets.token_urlsafe(32)
            descriptor = os.open(key_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(key + "\n")
        if not key:
            raise ValueError("Saved API key is empty")
        print(f"API key file: {key_file} (keep private)", flush=True)
    else:
        print("Using API_KEY from environment.", flush=True)
    os.environ.update(API_KEY=key, PIPELINE_CLASS="pipeline.vlm:LocalVlmPipeline",
                      VLM_MODEL_DIR=str(model), DATA_DIR=str(runtime / "jobs_runtime"),
                      VLM_DIAGNOSTICS_DIR=str(runtime / "diagnostics"),
                      RULES_PATH=str(ROOT / "rules/rules.yaml"), PROCESSING_TIMEOUT_SEC="270")
    print(f"GPU: {torch.cuda.get_device_name(0)}; torch={torch.__version__}; transformers={transformers.__version__}", flush=True)
    print(f"UI: http://127.0.0.1:{args.port}/ ; model loads when a job starts. Quality not certified.", flush=True)
    uvicorn.run("app.main:app", host=args.host, port=args.port, workers=1, reload=False)


if __name__ == "__main__":
    main()
