"""Compile explicit visual decisions into v4; never fall back to organizer/UI labels."""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import csv
from html import escape
import json
from pathlib import Path
import shutil
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.laptop_common import dump_json, read_csv, sha256_file, write_csv


def compile_review(decisions: dict, organizer_rows: list[dict]) -> tuple[dict, list[dict]]:
    if decisions.get("ui_assessments_used_for_decisions") is not False:
        raise ValueError("Decisions must explicitly disregard UI assessment")
    if decisions.get("organizer_labels_used_for_decisions") is not False:
        raise ValueError("Decisions must be independent of organizer target")
    if set(decisions["videos"]) != {r["video"] for r in organizer_rows}:
        raise ValueError("Every source video requires an explicit review")
    videos, audit, canonical = [], [], {}
    for source in organizer_rows:
        name = source["video"]
        d = decisions["videos"][name]
        label = d["reviewed_label"]
        if label not in (0, 1, None) or not d["reason"].strip():
            raise ValueError("Missing decision/reason")
        if bool(d["positive_windows"]) != (label == 1):
            raise ValueError("Positive decision must have reviewed intervals")
        representative = canonical.get(d["source_sha256"])
        if representative:
            prior = next(v for v in videos if v["video"] == representative)
            if prior["reviewed_label"] != label or prior["layout_group"] != d["layout_group"]:
                raise ValueError("Exact duplicate has conflicting independent decisions")
            if prior["positive_windows"] != d["positive_windows"]:
                raise ValueError("Exact duplicate intervals disagree")
            prior["aliases"].append(name)
        else:
            representative = name
            canonical[d["source_sha256"]] = name
            videos.append({"video": name, **d, "aliases": [],
                           "organizer_label": int(source["has_violation"]),
                           "organizer_timecode_sec": float(source["timecode_sec"]) if source["timecode_sec"] else None,
                           "eligible": label is not None})
        audit.append({"video": name, "organizer_label": source["has_violation"],
            "reviewed_label": "unknown" if label is None else label,
            "decision": "excluded" if label is None else ("possible_violation" if label else "no_visible_violation"),
            "event": d["observed_event"], "reason": d["reason"], "layout_group": d["layout_group"],
            "scene_group_id": d["scene_group_id"], "representative": representative,
            "start_sec": d["positive_windows"][0]["start_sec"] if label == 1 else "",
            "rule_candidate": d["positive_windows"][0]["rule_candidate"] if label == 1 else "",
            "rule_status": d["positive_windows"][0]["rule_status"] if label == 1 else ""})
    manifest = {"format": "cashier-independent-v4", "target": decisions["policy"],
        "ui_assessments_used_for_decisions": False, "organizer_labels_used_for_training": False,
        "review_method": decisions["review_method"], "reviewer": decisions["reviewer"],
        "evaluation_role": "development nested leave-layout-out; all clips seen during dataset review",
        "legal_rule_classifier": False, "videos": videos}
    return manifest, audit


def write_html(audit: list[dict], decisions: dict, directory: Path, source_dir: Path) -> None:
    cards = []
    for row in audit:
        decision = decisions["videos"][row["video"]]
        picture = source_dir / decision["review_sheet"]
        source = "data:image/jpeg;base64," + base64.b64encode(picture.read_bytes()).decode("ascii")
        comments = "".join(f"<li>{c['start_sec']:g}–{c['end_sec']:g} с: {escape(c['text'])}</li>" for c in decision["source_comments"])
        cards.append(f"<article><h2>{escape(row['video'])} — {escape(str(row['decision']))}</h2>"
            f"<p>Исходная метка: {row['organizer_label']}; новая: {row['reviewed_label']}; группа: {row['layout_group']}.</p>"
            f"<p>{escape(row['reason'])}</p><details><summary>Исходные комментарии</summary><ul>{comments}</ul></details>"
            f"<details><summary>Просмотренные кадры</summary><img src='{source}' alt='Выборка кадров'></details></article>")
    html = "<!doctype html><html lang='ru'><meta charset='utf-8'><title>Независимый пересмотр v4</title><style>body{font:16px system-ui;background:#f3f4f6;color:#17212b;max-width:1000px;margin:auto;padding:24px}article{background:white;padding:20px;margin:16px 0;border-radius:10px}img{max-width:100%;height:auto}summary{cursor:pointer;padding:8px}li{white-space:pre-wrap}</style><h1>Пересмотр всех видео — v4</h1><p>Статусы формы не использованы для решений. Unknown исключён из обучения и бинарных метрик. Метка 1 — признаки для проверки, а не юридический вывод. Кадры просмотрены выборочно по всей длительности и комментариям; это не просмотр каждого исходного кадра.</p>" + "".join(cards) + "</html>"
    (directory / "review_all_videos.html").write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError("Preserve old datasets; use a new output directory")
    decisions = json.loads(args.decisions.read_text(encoding="utf-8"))
    manifest, audit = compile_review(decisions, read_csv(args.data_dir / "labels.csv"))
    for video in manifest["videos"]:
        if sha256_file(args.data_dir / video["video"]) != video["source_sha256"]:
            raise ValueError(f"Video changed: {video['video']}")
    args.out_dir.mkdir(parents=True)
    manifest["decisions_sha256"] = sha256_file(args.decisions)
    manifest["organizer_labels_sha256"] = sha256_file(args.data_dir / "labels.csv")
    dump_json(args.out_dir / "dataset_manifest.json", manifest)
    shutil.copy2(args.decisions, args.out_dir / "decisions_v4.json")
    shutil.copy2(args.data_dir / "labels.csv", args.out_dir / "labels_organizer_original.csv")
    write_csv(args.out_dir / "video_review.csv", list(audit[0]), audit)
    eligible = [v for v in manifest["videos"] if v["eligible"]]
    labels = [{"video": v["video"], "has_violation": v["reviewed_label"],
               "timecode_sec": v["positive_windows"][0]["start_sec"] if v["reviewed_label"] else ""} for v in eligible]
    write_csv(args.out_dir / "labels_reviewed.csv", ["video", "has_violation", "timecode_sec"], labels)
    excluded = [r for r in audit if r["decision"] == "excluded"]
    write_csv(args.out_dir / "excluded.csv", list(audit[0]), excluded)
    summary = {"source_videos": len(audit), "unique_videos": len(manifest["videos"]),
        "eligible_unique": len(eligible), "positive_unique": sum(v["reviewed_label"] == 1 for v in eligible),
        "negative_unique": sum(v["reviewed_label"] == 0 for v in eligible),
        "excluded_unique": sum(not v["eligible"] for v in manifest["videos"]),
        "groups": {g: dict(Counter(str(v["reviewed_label"]) for v in manifest["videos"] if v["layout_group"] == g)) for g in sorted({v["layout_group"] for v in manifest["videos"]})},
        "source_notes_preserved": True, "ground_truth_independently_adjudicated_by_bank": False,
        "ready_for_training_experiment": True, "working_model_quality_demonstrated": False}
    dump_json(args.out_dir / "summary.json", summary)
    write_html(audit, decisions, args.out_dir, args.decisions.parent)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
