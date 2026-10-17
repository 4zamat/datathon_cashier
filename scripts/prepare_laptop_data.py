"""Validate labels, group exact duplicates, exclude conflicting groups, write seed-42 splits.

Source videos and data/labels.csv are only read, never modified.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from laptop_common import (  # noqa: E402
    DATA_DIR, SEED, SPLIT_DIR, SPLIT_FIELDS, SPLITS, dump_json, read_csv, sha256_file,
    stratified_group_split, write_csv,
)


def prepare(labels_path: Path, check_path: Path, seed: int) -> dict:
    rows = read_csv(labels_path)
    check = {r["video"]: r for r in json.loads(check_path.read_text(encoding="utf-8"))["results"]}
    entries = []
    for line, row in enumerate(rows, 2):
        video = row["video"].strip()
        label = row["has_violation"].strip()
        if label not in {"0", "1"}:
            raise ValueError(f"line {line}: label must be 0/1")
        path = DATA_DIR / video
        if not path.is_file():
            raise FileNotFoundError(f"line {line}: missing video {video!r}")
        timecode = row["timecode_sec"].strip()
        if label == "1" and not timecode:
            raise ValueError(f"line {line}: positive label without timecode")
        info = check.get(video)
        if info is None or not info.get("ok"):
            raise ValueError(f"line {line}: {video!r} is not a passed entry of dataset_check.json")
        if timecode and not 0 <= float(timecode) <= info["duration_sec"]:
            raise ValueError(f"line {line}: timecode outside the video duration")
        sha = sha256_file(path)
        if sha != info["sha256"]:
            raise ValueError(f"line {line}: SHA-256 differs from dataset_check.json")
        entries.append({
            "video": video, "has_violation": int(label),
            "timecode_sec": timecode, "sha256": sha, "duration_sec": info["duration_sec"],
        })
    if len({e["video"] for e in entries}) != len(entries):
        raise ValueError("duplicate video rows in labels.csv")

    by_sha: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        by_sha[entry["sha256"]].append(entry)

    groups, excluded = [], []
    for sha, members in sorted(by_sha.items(), key=lambda kv: min(m["video"] for m in kv[1])):
        members = sorted(members, key=lambda m: m["video"])
        variants = {(m["has_violation"], m["timecode_sec"]) for m in members}
        if len(variants) > 1:
            for m in members:
                excluded.append({**m, "reason": "identical video content with conflicting labels/timecodes"})
            continue
        representative = members[0]
        groups.append({
            **representative, "group_id": f"g{len(groups) + 1:02d}",
            "duplicates": ";".join(m["video"] for m in members[1:]),
        })

    splits = stratified_group_split(groups, seed=seed)
    for name, members in splits.items():
        labels = {g["has_violation"] for g in members}
        if labels != {0, 1}:
            raise RuntimeError(f"split {name} lacks a class")
    assigned = [g["sha256"] for name in SPLITS for g in splits[name]]
    if len(assigned) != len(set(assigned)):
        raise RuntimeError("a group landed in more than one split")

    for name in SPLITS:
        write_csv(SPLIT_DIR / f"{name}.csv", SPLIT_FIELDS,
                  sorted(splits[name], key=lambda g: g["group_id"]))
    write_csv(SPLIT_DIR / "excluded.csv", ["video", "has_violation", "timecode_sec", "sha256", "reason"],
              [{k: e[k] for k in ("video", "has_violation", "timecode_sec", "sha256", "reason")} for e in excluded])
    summary = {
        "seed": seed,
        "label_rows": len(entries),
        "unique_contents": len(by_sha),
        "excluded_videos": len(excluded),
        "excluded_groups": 1 if excluded else 0,
        "usable_unique_videos": len(groups),
        "merged_duplicate_files": sum(len(g["duplicates"].split(";")) for g in groups if g["duplicates"]),
        "splits": {
            name: {
                "videos": len(splits[name]),
                "positive": sum(g["has_violation"] for g in splits[name]),
                "negative": sum(1 - g["has_violation"] for g in splits[name]),
            } for name in SPLITS
        },
    }
    dump_json(SPLIT_DIR / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DATA_DIR / "labels.csv")
    parser.add_argument("--check", type=Path, default=DATA_DIR / "dataset_check.json")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    print(json.dumps(prepare(args.labels, args.check, args.seed), indent=2))


if __name__ == "__main__":
    main()
