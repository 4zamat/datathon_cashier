"""Private, self-contained HTML for visual inspection of development holdout errors."""
import argparse
import base64
import csv
from html import escape
import json
from pathlib import Path

import cv2


def render(data_dir: Path, run_dir: Path) -> Path:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    with (run_dir / "predictions.csv").open(encoding="utf-8", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == "test"]
    cards = []
    for row in rows:
        capture = cv2.VideoCapture(str(data_dir / row["video"]))
        capture.set(cv2.CAP_PROP_POS_MSEC, float(row["predicted_timecode_sec"]) * 1000)
        ok, frame = capture.read()
        capture.release()
        if not ok:
            raise ValueError(f"Could not read report evidence: {row['video']}")
        encoded_ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not encoded_ok:
            raise ValueError("JPEG encoding failed")
        picture = base64.b64encode(encoded).decode("ascii")
        correct = row["actual"] == row["predicted"]
        cards.append(f'''<article class="{'correct' if correct else 'error'}">
<h2>{escape(row['video'])} · {'совпало' if correct else 'ошибка'}</h2>
<p>Метка датасета: <b>{row['actual']}</b> · Предсказание: <b>{row['predicted']}</b></p>
<p>Score {float(row['score']):.3f} · Время пика {float(row['predicted_timecode_sec']):.2f} с</p>
<p>Исходный таймкод: {escape(row['true_timecode_sec'] or '—')} · Расстояние до интервала: {escape(row['interval_error_sec'] or '—')} с</p>
<img src="data:image/jpeg;base64,{picture}" alt="Кадр в момент максимального score"></article>''')
    score = metrics["splits"]["test"]["model"]
    baseline = metrics["splits"]["test"]["always_violation"]
    text = f'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Проверка эксперимента v3</title><style>
body{{font:16px system-ui;margin:32px;background:#f5f5f9;color:#202332}}main{{max-width:1150px;margin:auto}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px}}article{{background:white;padding:18px;border-radius:12px;border-top:5px solid #53a07c}}article.error{{border-color:#b93065}}h2{{font-size:18px}}img{{max-width:100%;max-height:600px;display:block;margin:auto}}p{{line-height:1.5}}.note{{padding:18px;background:#fff4df;border-radius:12px}}
</style><main><h1>Визуальная проверка v3</h1>
<p class="note">Это проверка на отложенных группах уже изученного датасета. 0/1 — бинарная метка организаторов, а не доказательство юридического нарушения. Score не является откалиброванной вероятностью. Этот отчёт нельзя использовать для повторного подбора порога по test.</p>
<p>F1: <b>{score['f1']:.3f}</b> · Precision: {score['precision']:.3f} · Recall: {score['recall']:.3f} · «Всегда нарушение»: F1 {baseline['f1']:.3f}</p>
<p>TP {score['tp']} · FP {score['fp']} · FN {score['fn']} · TN {score['tn']} · Порог выбран по validation: {metrics['threshold']:.4f}</p>
<div class="grid">{''.join(cards)}</div></main></html>'''
    path = run_dir / "review.html"
    path.write_text(text, encoding="utf-8")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(render(args.data_dir, args.run_dir))
