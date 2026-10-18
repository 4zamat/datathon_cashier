"""Reprocess saved VLM responses without inference; never overwrite the source run."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import zipfile

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rules import RuleCatalog
from pipeline.vlm import parse_report, sample_sequence, strip_fence
from scripts.laptop_common import dump_json, sha256_file
from scripts.vlm_probe import apply_observations, report_html, summarize

ROOT = Path(__file__).resolve().parents[1]


def replay(source: Path, data: Path, output: Path) -> dict:
    source, data, output = source.resolve(), data.resolve(), output.resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError("Replay must be outside the preserved source run")
    manifest = json.loads((source / "run_status.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files_sha256"].items():
        path = (source / name).resolve()
        if not path.is_relative_to(source) or sha256_file(path) != expected:
            raise ValueError(f"Changed source artifact: {name}")
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if sha256_file(source / "dataset_manifest.json") != config["dataset_sha256"]:
        raise ValueError("Source dataset checksum differs")
    dataset = json.loads((source / "dataset_manifest.json").read_text(encoding="utf-8"))
    lookup = {v["video"]: v for v in dataset["videos"]}
    originals = json.loads((source / "predictions.json").read_text(encoding="utf-8"))["videos"]
    if any(row.get("verification_response") for row in originals):
        raise ValueError("This format-only replay supports pre-verification runs only; use the saved verified checks")
    if len({r["video"] for r in originals}) != len(originals):
        raise ValueError("Duplicate predictions in source run")
    output.mkdir(parents=True, exist_ok=False)
    catalog = RuleCatalog(ROOT / "rules/rules.yaml")
    rows = []
    for original in originals:
        video = lookup[original["video"]]
        row = {k: v for k, v in original.items() if k not in {
            "error", "checks", "evidence", "preview_frames", "api_status", "api_error", "api_violations",
            "api_verdict", "matching_event_type", "matching_interval_error_sec"}}
        row.update(original_status=original["status"], original_error=original.get("error"),
                   status="failed", predicted_visual=None, replay_without_inference=True)
        try:
            path = (data / video["video"]).resolve()
            if not path.is_relative_to(data) or sha256_file(path) != video["source_sha256"]:
                raise ValueError("Source video checksum differs")
            if row["actual"] != video["reviewed_label"]:
                raise ValueError("Saved label differs from source dataset")
            samples, _ = sample_sequence(path, config["max_frames"])
            times = original["frame_timestamps"]
            if len(samples) != len(times) or any(not math.isclose(s.timestamp_sec, t, abs_tol=.002)
                                                for s, t in zip(samples, times)):
                raise ValueError("Decoded frame timestamps differ from inference inputs")
            row["decoded_pixels_match_source"] = original.get("frame_sha256") == [
                hashlib.sha256(s.bgr.tobytes()).hexdigest() for s in samples]
            observations = parse_report(original["raw_response"], len(samples))
            raw = json.loads(strip_fence(original["raw_response"]))
            row["evidence_selection"] = [{"type": c.type, "source_ids": raw["checks"][i]["evidence_ids"],
                "selected_ids": c.evidence_ids} for i, c in enumerate(observations.checks)
                if raw["checks"][i]["evidence_ids"] != c.evidence_ids]
            apply_observations(row, observations, samples, video, catalog)
        except Exception as error:
            row.update(status="failed", error=f"{type(error).__name__}: {error}")
        rows.append(row)
        print(json.dumps({k: row.get(k) for k in ("video", "status", "predicted_visual", "error")}, ensure_ascii=False))
    summary = summarize(rows, "saved-response-replay", len(originals))
    summary.update(evaluation_role="post-hoc format-only repair of saved development smoke responses; NOT a new GPU test",
                   source_run=source.name, model_rerun=False, labels_changed=False)
    dump_json(output / "predictions.json", {"videos": rows})
    dump_json(output / "metrics.json", summary)
    (output / "review.html").write_text(report_html(rows, summary), encoding="utf-8")
    for relative in ("pipeline/vlm.py", "scripts/replay_vlm.py", "scripts/vlm_probe.py", "rules/rules.yaml"):
        target = output / "processor_source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    dump_json(output / "replay_audit.json", {"source_run": str(source),
        "source_predictions_sha256": sha256_file(source / "predictions.json"),
        "source_manifest_sha256": sha256_file(source / "run_status.json"),
        "source_config": config, "selection": "first, middle and last cited frame in chronological order",
        "all_original_references_validated_before_selection": True,
        "raw_responses_unchanged": all(a.get("raw_response") == b.get("raw_response") for a, b in zip(originals, rows)),
        "files_sha256": {p.relative_to(output).as_posix(): sha256_file(p) for p in output.rglob("*") if p.is_file()}})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path.cwd() / "data")
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    run_id = datetime.now(timezone.utc).strftime("vlm-v2-format-replay-%Y%m%d-%H%M%S")
    output = args.out_dir or args.data_dir / "runs" / run_id
    summary = replay(args.run_dir, args.data_dir, output)
    exports = args.data_dir / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    archive = exports / (output.name + ".zip")
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as z:
        for path in output.rglob("*"):
            if path.is_file():
                z.write(path, path.relative_to(output))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"ZIP: {archive.resolve()}\nREPORT: {(output / 'review.html').resolve()}")
    if summary["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
