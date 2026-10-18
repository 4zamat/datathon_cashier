"""Compile an explicit review decision file; never overwrite source observations."""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.laptop_common import dump_json, read_csv, sha256_file, stratified_group_split, write_csv
from scripts.reviewed_common import validate_dataset
from scripts.validate_annotations import Annotation


def prepare(data_dir: Path, decisions_path: Path, output: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output exists; preserve the previous reviewed version")
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    checks = {v["video"]: v for v in json.loads((data_dir / "dataset_check.json").read_text(encoding="utf-8"))["results"]}
    source_labels = read_csv(data_dir / "labels.csv")
    docs = {}
    source_hashes = {}
    for path in (data_dir / "annotations/local/documents").glob("*.json"):
        document = json.loads(path.read_text(encoding="utf-8"))
        docs[document["video"]] = document
        source_hashes[path.name] = sha256_file(path)
    if set(decisions["videos"]) != {r["video"] for r in source_labels}:
        raise ValueError("Review decisions must cover every source video")
    output.mkdir(parents=True)
    dump_json(output / "review_decisions.json", decisions)
    dump_json(output / "human_observations.json", {k: v for k, v in docs.items()})
    all_events, audit, scene_rows, reconciled = [], [], [], []
    for label in source_labels:
        name = label["video"]
        info, doc, decision = checks[name], docs[name], decisions["videos"][name]
        if sha256_file(data_dir / name) != info["sha256"] or doc["source_sha256"] != info["sha256"]:
            raise ValueError(f"Source changed: {name}")
        events = []
        overrides = decision.get("events", {})
        if set(overrides) - {e["id"] for e in doc["events"]}:
            raise ValueError("Review refers to missing human event")
        for original in doc["events"]:
            event = copy.deepcopy(original)
            override = overrides.get(event["id"], {})
            for key in ("start_sec", "end_sec", "event_type", "assessment", "observation"):
                if key in override:
                    event[key] = override[key]
            event["review_reason"] = override.get("reason", decision["review_note"])
            event["rule_support"] = override.get("rule_support", "not_established" if event["assessment"] == "uncertain" else "observation_only")
            events.append(event)
            row = {"video": name, "event_id": event["id"], "start_sec": event["start_sec"], "end_sec": event["end_sec"],
                   "event_type": event["event_type"], "observation": event["observation"], "assessment": event["assessment"],
                   "reviewer": f"{doc['reviewer']}; Codex (reconciliation)", "scene_group_id": decision["scene_group_id"]}
            Annotation.model_validate(row)
            all_events.append(row)
            audit.append({**row, "source_assessment": original["assessment"], "source_type": original["event_type"],
                          "source_start_sec": original["start_sec"], "source_end_sec": original["end_sec"],
                          "source_observation": original["observation"],
                          "review_reason": event["review_reason"], "rule_support": event["rule_support"]})
        scene_rows.append({"video": name, "scene_group_id": decision["scene_group_id"], "status": "include",
                           "reason": decision["scene_reason"], "split_group_id": decision["split_group_id"]})
        reconciled.append({"video": name, "sha256": info["sha256"], "duration_sec": info["duration_sec"],
            "has_violation": int(label["has_violation"]),
            "timecode_sec": float(label["timecode_sec"]) if label["timecode_sec"] else None,
            "original_timecode_sec": float(label["timecode_sec"]) if label["timecode_sec"] else None,
            "scene_group_id": decision["scene_group_id"], "split_group_id": decision["split_group_id"],
            "review_note": decision["review_note"], "events": events, "positive_windows": decision["positive_windows"]})
    by_sha = defaultdict(list)
    for row in reconciled:
        by_sha[row["sha256"]].append(row)
    unique, duplicates = [], []
    for rows in by_sha.values():
        if len({(r["has_violation"], r["scene_group_id"], r["split_group_id"]) for r in rows}) != 1:
            raise ValueError("Duplicate label/group conflict needs explicit adjudication")
        canonical = rows[0]  # source-label order is stable; lowest-number original appears first
        canonical["aliases"] = [r["video"] for r in rows[1:]]
        for alias in rows[1:]:
            duplicates.append({"video": alias["video"], "representative": canonical["video"], "sha256": alias["sha256"],
                "original_timecode_sec": alias["timecode_sec"], "representative_timecode_sec": canonical["timecode_sec"],
                "reason": "Exact duplicate; both source timecodes retained; one sustained-event interval used"})
        unique.append(canonical)
    split_input = [{**row, "group_id": row["split_group_id"]} for row in unique]
    splits = stratified_group_split(split_input, seed=42)
    for split, rows in splits.items():
        for row in rows:
            row["split"] = split
            row.pop("group_id")
    manifest = {"format": "cashier-reviewed-v3", "seed": 42,
        "target": "organizer binary labels + reviewed temporal windows; weak labels explicitly downweighted",
        "evaluation_role": "development holdout; all source videos inspected before this experiment",
        "group_policy": "8 conservative take families; 4 broader layouts kept whole across train/val/test",
        "legal_rule_classifier": False, "videos": [v for rows in splits.values() for v in rows],
        "source_labels_sha256": sha256_file(data_dir / "labels.csv"),
        "dataset_check_sha256": sha256_file(data_dir / "dataset_check.json"),
        "human_document_sha256": source_hashes, "decision_sha256": sha256_file(decisions_path),
        "duplicate_resolution": duplicates}
    validate_dataset(manifest, data_dir)
    dump_json(output / "dataset_manifest.json", manifest)
    write_csv(output / "events.csv", list(all_events[0]), all_events)
    write_csv(output / "event_review.csv", list(audit[0]), audit)
    write_csv(output / "scene_groups.csv", list(scene_rows[0]), scene_rows)
    write_csv(output / "duplicates.csv", list(duplicates[0]), duplicates)
    write_csv(output / "labels_organizer_original.csv", list(source_labels[0]), source_labels)
    for split, rows in splits.items():
        write_csv(output / f"{split}.csv", ["video", "has_violation", "timecode_sec", "scene_group_id", "split_group_id", "sha256"], rows)
    summary = {"source_videos": len(reconciled), "unique_videos": len(unique), "events": len(all_events),
        "assessments": dict(Counter(r["assessment"] for r in all_events)), "duplicates_merged": len(duplicates),
        "scene_groups": len({r["scene_group_id"] for r in unique}), "split_groups": len({r["split_group_id"] for r in unique}),
        "strong_positive_windows": sum(w["weight"] == 1 for r in unique for w in r["positive_windows"]),
        "weak_positive_windows": sum(w["weight"] < 1 for r in unique for w in r["positive_windows"]),
        "splits": {s: {"videos": len(v), "positive": sum(r["has_violation"] for r in v),
                        "groups": sorted({r["split_group_id"] for r in v})} for s, v in splits.items()},
        "ready_for_diagnostic_training": True, "ready_for_legal_rule_inference": False,
        "pending": decisions["pending"], "source_labels_overwritten": False}
    dump_json(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.data_dir, args.decisions, args.out_dir), ensure_ascii=False, indent=2))
