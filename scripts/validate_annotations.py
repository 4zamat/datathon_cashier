"""Validate team observation CSVs; never rewrite original labels or infer legal guilt."""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parents[1]


class Annotation(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)
    video: str = Field(min_length=1)
    event_id: str = Field(min_length=1)
    start_sec: float = Field(ge=0)
    end_sec: float = Field(ge=0)
    event_type: str = Field(min_length=1)
    observation: str = Field(min_length=1)
    assessment: Literal["possible_violation", "no_violation", "uncertain"]
    reviewer: str = Field(min_length=1)
    scene_group_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def ordered_interval(self) -> "Annotation":
        if self.end_sec < self.start_sec:
            raise ValueError("end_sec must be >= start_sec")
        return self


def validate(path: Path, check_path: Path, rules_path: Path, allow_partial: bool = False) -> dict:
    videos = {r["video"]: r for r in json.loads(check_path.read_text(encoding="utf-8"))["results"]}
    rules = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
    seen, groups, reviewed = set(), {}, set()
    count = 0
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for line, raw in enumerate(csv.DictReader(stream), 2):
            row = Annotation.model_validate(raw)
            info = videos.get(row.video)
            if info is None or not info["ok"] or row.end_sec > info["duration_sec"]:
                raise ValueError(f"line {line}: unknown video, failed decode check or interval outside video")
            if row.event_type not in (set(rules) - {"demo_violation"}) | {"normal", "unknown"}:
                raise ValueError(f"line {line}: unknown event type")
            if row.assessment == "possible_violation" and row.event_type in {"normal", "unknown"}:
                raise ValueError(f"line {line}: possible violation requires a candidate rule type")
            key = (row.video, row.event_id, row.reviewer)
            if key in seen:
                raise ValueError(f"line {line}: duplicate event/reviewer")
            if row.video in groups and groups[row.video] != row.scene_group_id:
                raise ValueError(f"line {line}: inconsistent scene_group_id for one video")
            seen.add(key)
            groups[row.video] = row.scene_group_id
            reviewed.add(row.video)
            count += 1
    missing = sorted(set(videos) - reviewed)
    if not count or (missing and not allow_partial):
        raise ValueError(f"Empty/incomplete annotation: {len(missing)} videos still unreviewed")
    return {"rows": count, "videos_reviewed": len(reviewed), "videos_missing": len(missing),
            "note": "Format and bounds validated; labels and legal interpretation still need human review"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--check", type=Path, default=ROOT / "data/dataset_check.json")
    parser.add_argument("--rules", type=Path, default=ROOT / "rules/rules.yaml")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(validate(args.events, args.check, args.rules, args.allow_partial), indent=2))
    except (ValueError, OSError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
