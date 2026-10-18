"""Test authentication, multipart upload and polling against the actual running service."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.models import Job


def request(base: str, path: str, key: str = "", data: bytes | None = None,
            headers: dict | None = None) -> tuple[int, dict]:
    combined = {**(headers or {}), **({"X-API-Key": key} if key else {})}
    req = Request(base.rstrip("/") + path, data=data, headers=combined)
    try:
        with urlopen(req, timeout=120) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.load(error)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--data-dir", type=Path, default=Path.cwd() / "data")
    parser.add_argument("--key-file", type=Path, default=Path.cwd() / "data/service_v1/api-key.txt")
    parser.add_argument("--numbers", type=int, nargs="+", default=[4, 5, 26, 27, 40, 43])
    args = parser.parse_args()
    key = os.environ.get("API_KEY", "").strip() or args.key_file.read_text(encoding="utf-8").strip()
    data = args.data_dir.resolve()
    if not key:
        raise ValueError("API key is empty")
    paths = []
    for number in args.numbers:
        matches = [p for p in (data / "videos").iterdir() if p.suffix.lower() == ".mov"
                   and p.stem.rsplit("-", 1)[-1] == str(number)]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one video numbered {number}, found {len(matches)}")
        paths.append(matches[0])
    code, health = request(args.base_url, "/health")
    if code != 200 or health.get("demo") is not False:
        raise RuntimeError(f"Real pipeline is not configured: {health}")
    for path, payload in [("/api/v1/analyze", b"invalid"), ("/api/v1/jobs/missing", None)]:
        for bad_key in ("", "deliberately-incorrect-key"):
            code, _ = request(args.base_url, path, key=bad_key, data=payload)
            if code != 401:
                raise AssertionError(f"Authentication check failed: {path}: {code}")
    run_id = datetime.now(timezone.utc).strftime("api-service-v1-%Y%m%d-%H%M%S")
    out = data / "runs" / run_id
    out.mkdir(parents=True, exist_ok=False)
    report = {"health": health, "authentication_passed": True,
              "evaluation_role": "development API smoke; no independent quality claim", "videos": []}
    try:
        for video in paths:
            start = time.monotonic()
            row = {"video": video.name}
            try:
                boundary = uuid.uuid4().hex
                # Neutral filename deliberately excludes the original label-bearing video name.
                body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="upload.mov"\r\n'
                        'Content-Type: video/quicktime\r\n\r\n').encode() + video.read_bytes() + f'\r\n--{boundary}--\r\n'.encode()
                code, accepted = request(args.base_url, "/api/v1/analyze", key, body,
                    {"Content-Type": f"multipart/form-data; boundary={boundary}"})
                del body
                if code != 202 or not accepted.get("job_id"):
                    raise RuntimeError(f"Upload failed: {code}: {accepted}")
                job_id = accepted["job_id"]
                print(f"{video.name}: processing ({job_id})", flush=True)
                deadline = time.monotonic() + 285
                while time.monotonic() < deadline:
                    code, payload = request(args.base_url, f"/api/v1/jobs/{job_id}", key)
                    if code != 200:
                        raise RuntimeError(f"Polling failed: {code}")
                    job = Job.model_validate(payload)
                    if job.job_id != job_id:
                        raise ValueError("Returned job ID does not match")
                    if job.status != "processing":
                        (out / f"{video.stem}.json").write_text(job.model_dump_json(indent=2), encoding="utf-8")
                        diagnostic = data / "service_v1/diagnostics" / f"{job_id}.json"
                        if diagnostic.is_file():
                            shutil.copy2(diagnostic, out / f"{video.stem}.diagnostics.json")
                        row.update(status=job.status, verdict=job.verdict if job.status == "done" else None,
                                   types=[v.type for v in job.violations], error=job.error)
                        break
                    time.sleep(2)
                else:
                    raise TimeoutError("Server did not finish within its processing deadline")
            except Exception as error:
                row.update(status="failed", error=f"{type(error).__name__}: {error}")
            row["elapsed_sec"] = round(time.monotonic() - start, 2)
            report["videos"].append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            (out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        report["all_jobs_done"] = len(report["videos"]) == len(paths) and all(r["status"] == "done" for r in report["videos"])
        (out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        exports = data / "exports"
        exports.mkdir(exist_ok=True)
        archive = exports / f"{run_id}.zip"
        with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as z:
            for path in out.iterdir():
                z.write(path, path.name)
        print(f"\nSEND THIS ZIP: {archive}", flush=True)
    if not report["all_jobs_done"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
