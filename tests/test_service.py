import base64
import json
import tempfile
import time
import unittest
from pathlib import Path
from queue import Full

import cv2
import numpy as np
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import Settings
from app.main import create_app
from app.models import Detection, EvidenceFrame, Job
from app.rules import RuleCatalog
from app.storage import JobStore
from app.worker import JobWorker
from pipeline.base import Result
from pipeline.dummy import sample_frames
from scripts.eval import evaluate
from tests.helpers import EvaluationPipeline, detection, make_video

HEADERS = {"X-API-Key": "test-key"}


def wait_for_job(client: TestClient, job_id: str, timeout: float = 15) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/jobs/{job_id}", headers=HEADERS)
        response.raise_for_status()
        job = response.json()
        if job["status"] != "processing":
            return job
        time.sleep(0.05)
    raise AssertionError("Job did not reach a terminal state")


def submit(client: TestClient, contents: bytes) -> str:
    response = client.post("/api/v1/analyze", headers=HEADERS, files={"file": ("test.mov", contents, "video/quicktime")})
    if response.status_code != 202:
        raise AssertionError(response.text)
    return response.json()["job_id"]


class ServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.video = make_video(self.root / "sample.mov")
        self.settings = Settings(api_key="test-key", data_dir=self.root / "data")

    def test_authentication_precedes_multipart_parsing(self) -> None:
        with TestClient(create_app(self.settings)) as client:
            for headers in ({}, {"X-API-Key": "wrong"}):
                self.assertEqual(client.post("/api/v1/analyze", headers=headers, content=b"invalid").status_code, 401)
                self.assertEqual(client.get("/api/v1/jobs/missing", headers=headers).status_code, 401)
            self.assertEqual(client.get("/api/v1/jobs/missing", headers=HEADERS).status_code, 404)
            self.assertEqual(client.get("/").status_code, 200)
            self.assertEqual(client.get("/static/app.js").status_code, 200)

    def test_real_mov_end_to_end_and_restart(self) -> None:
        with TestClient(create_app(self.settings)) as client:
            job_id = submit(client, self.video)
            job = wait_for_job(client, job_id)
            self.assertEqual(job["status"], "done")
            self.assertEqual(job["verdict"], "violation")
            self.assertEqual(len(job["violations"]), 1)
            violation = job["violations"][0]
            self.assertAlmostEqual(violation["timecode_sec"], 2, places=2)
            expected_rule = RuleCatalog(self.settings.rules_path).enrich(Result(violations=[detection()]))[0].rule
            self.assertEqual(violation["rule"], expected_rule.model_dump())
            frame = violation["evidence_frames"][0]
            pixels = cv2.imdecode(np.frombuffer(base64.b64decode(frame["image_base64"]), np.uint8), cv2.IMREAD_COLOR)
            self.assertIsNotNone(pixels)
            np.testing.assert_allclose(pixels[20, 20], [48, 30, 200], atol=8)
            self.assertEqual(list((self.settings.data_dir / "uploads").iterdir()), [])
            self.assertEqual(list((self.settings.data_dir / "work").iterdir()), [])
            saved = json.loads((self.settings.data_dir / "jobs" / f"{job_id}.json").read_text(encoding="utf-8"))
            self.assertEqual(saved, job)
        with TestClient(create_app(self.settings)) as client:
            self.assertEqual(client.get(f"/api/v1/jobs/{job_id}", headers=HEADERS).json(), job)

    def test_30_mib_mov_upload(self) -> None:
        # Padding after the valid MOV atoms exercises realistic upload size without private footage.
        payload = self.video + bytes(30 * 1024 * 1024 - len(self.video))
        with TestClient(create_app(self.settings)) as client:
            job = wait_for_job(client, submit(client, payload))
            self.assertEqual(job["status"], "done")
            self.assertEqual(list((self.settings.data_dir / "uploads").iterdir()), [])

    def test_validation_and_size_limits(self) -> None:
        settings = self.settings.model_copy(update={"max_upload_mb": 1})
        with TestClient(create_app(settings)) as client:
            self.assertEqual(client.post("/api/v1/analyze", headers=HEADERS).status_code, 422)
            for name, payload, code in [("video.mp4", b"abc", 415), ("empty.mov", b"", 400),
                                        ("large.mov", bytes(1024 * 1024 + 1), 413)]:
                response = client.post("/api/v1/analyze", headers=HEADERS, files={"file": (name, payload)})
                self.assertEqual(response.status_code, code)
            # No Content-Length: the streaming multipart guard must still reject the request.
            body = (b'--boundary\r\nContent-Disposition: form-data; name="file"; filename="large.mov"\r\n'
                    b'Content-Type: video/quicktime\r\n\r\n' + bytes(3 * 1024 * 1024) + b'\r\n--boundary--\r\n')
            response = client.post("/api/v1/analyze", headers={**HEADERS, "Content-Type": "multipart/form-data; boundary=boundary"},
                                   content=iter([body[:1024], body[1024:]]))
            self.assertEqual(response.status_code, 413)
            self.assertEqual(list((settings.data_dir / "uploads").iterdir()), [])

    def test_corrupt_video_fails_and_is_removed(self) -> None:
        with TestClient(create_app(self.settings)) as client:
            job = wait_for_job(client, submit(client, b"not a movie"))
            self.assertEqual(job["status"], "failed")
            self.assertEqual(job["verdict"], "no_violation")
            self.assertEqual(job["violations"], [])
            self.assertEqual(list((self.settings.data_dir / "uploads").iterdir()), [])

    def test_timeout_kills_pipeline_and_queue_remains_usable(self) -> None:
        settings = self.settings.model_copy(update={"pipeline_class": "tests.helpers:ConditionalPipeline", "processing_timeout_sec": 3})
        with TestClient(create_app(settings)) as client:
            start = time.monotonic()
            job = wait_for_job(client, submit(client, b"hang"))
            self.assertTrue((settings.data_dir / "started").exists())
            self.assertEqual(job["status"], "failed")
            self.assertIn("timeout", job["error"])
            self.assertLess(time.monotonic() - start, 8)
            self.assertEqual(list((settings.data_dir / "uploads").iterdir()), [])
            next_job = wait_for_job(client, submit(client, b"ok"))
            self.assertEqual(next_job["status"], "done")
            self.assertEqual(next_job["verdict"], "no_violation")
            self.assertEqual(next_job["violations"], [])

    def test_processing_is_sequential(self) -> None:
        settings = self.settings.model_copy(update={"pipeline_class": "tests.helpers:RecordingPipeline"})
        with TestClient(create_app(settings)) as client:
            first = submit(client, b"first")
            second = submit(client, b"second")
            self.assertEqual(wait_for_job(client, first)["status"], "done")
            self.assertEqual(wait_for_job(client, second)["status"], "done")
        lines = (settings.data_dir / "execution.log").read_text().splitlines()
        self.assertEqual(lines, [f"start {first}", f"end {first}", f"start {second}", f"end {second}"])

    def test_unknown_rule_cannot_be_published(self) -> None:
        settings = self.settings.model_copy(update={"pipeline_class": "tests.helpers:UnknownTypePipeline"})
        with TestClient(create_app(settings)) as client:
            job = wait_for_job(client, submit(client, b"test"))
            self.assertEqual(job["status"], "failed")
            self.assertEqual(job["violations"], [])

    def test_shutdown_cleans_active_and_queued_jobs(self) -> None:
        settings = self.settings.model_copy(update={"pipeline_class": "tests.helpers:ConditionalPipeline"})
        with TestClient(create_app(settings)) as client:
            first = submit(client, b"hang")
            second = submit(client, b"ok")
        store = JobStore(settings.data_dir / "jobs")
        self.assertEqual(store.get(first).status, "failed")
        self.assertEqual(store.get(second).status, "failed")
        self.assertEqual(list((settings.data_dir / "uploads").iterdir()), [])

    def test_restart_marks_interrupted_jobs_failed_and_cleans_orphans(self) -> None:
        store = JobStore(self.settings.data_dir / "jobs")
        job_id = "a" * 32
        store.save(Job(job_id=job_id))
        upload_dir = self.settings.data_dir / "uploads"
        upload_dir.mkdir()
        (upload_dir / f"{job_id}.mov").write_bytes(b"old")
        with TestClient(create_app(self.settings)) as client:
            job = client.get(f"/api/v1/jobs/{job_id}", headers=HEADERS).json()
            self.assertEqual(job["status"], "failed")
            self.assertEqual(list(upload_dir.iterdir()), [])

    def test_sampling_and_evidence_contract(self) -> None:
        frames = list(sample_frames(self.root / "sample.mov"))
        np.testing.assert_allclose([time for time, _ in frames], np.arange(0, 4, .5), atol=.01)
        item = detection()
        for evidence in ([], [item.evidence_frames[0]] * 4):
            with self.assertRaises(ValidationError):
                Detection(**{**item.model_dump(), "evidence_frames": evidence})
        with self.assertRaises(ValidationError):
            Detection(**item.model_dump(), rule={"document": "fabricated", "clause": "1"})
        with self.assertRaises(ValidationError):
            EvidenceFrame(timestamp_sec=0, image_base64="invalid")

    def test_queue_capacity(self) -> None:
        settings = self.settings.model_copy(update={"max_pending_jobs": 1})
        store = JobStore(settings.data_dir / "jobs")
        worker = JobWorker(settings, store, RuleCatalog(settings.rules_path))
        video = worker.upload_dir / "test.mov"
        video.write_bytes(self.video)
        worker.submit("a" * 32, video)
        with self.assertRaises(Full):
            worker.submit("b" * 32, video)
        self.assertEqual(store.get("b" * 32).status, "failed")

    def test_queue_wait_counts_toward_timeout(self) -> None:
        settings = self.settings.model_copy(update={"processing_timeout_sec": .05})
        store = JobStore(settings.data_dir / "jobs")
        worker = JobWorker(settings, store, RuleCatalog(settings.rules_path))
        video = worker.upload_dir / "queued.mov"
        video.write_bytes(self.video)
        job_id = "c" * 32
        worker.submit(job_id, video)
        time.sleep(.08)
        worker.start()
        try:
            deadline = time.monotonic() + 2
            while store.get(job_id).status == "processing" and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertEqual(store.get(job_id).status, "failed")
            self.assertIn("timeout", store.get(job_id).error)
            self.assertFalse(video.exists())
        finally:
            worker.stop()

    def test_evaluation_metrics_and_csv_validation(self) -> None:
        labels = self.root / "labels.csv"
        for name in ("tp", "fp", "fn", "tn"):
            (self.root / f"{name}.mov").touch()
        labels.write_text("video,has_violation,timecode_sec\ntp.mov,1,5\nfp.mov,0,\nfn.mov,1,6\ntn.mov,0,\n", encoding="utf-8")
        metrics = evaluate(labels, EvaluationPipeline())
        self.assertEqual((metrics.true_positives, metrics.false_positives, metrics.false_negatives, metrics.true_negatives), (1, 1, 1, 1))
        self.assertEqual(metrics.precision, .5)
        self.assertEqual(metrics.recall, .5)
        self.assertEqual(metrics.mean_timecode_error_sec, 1)
        labels.write_text("video,has_violation,timecode_sec\nfn.mov,1,6\n", encoding="utf-8")
        metrics = evaluate(labels, EvaluationPipeline())
        self.assertEqual(metrics.precision, 0)
        self.assertIsNone(metrics.mean_timecode_error_sec)
        labels.write_text("video,has_violation,timecode_sec\ntp.mov,1,NaN\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            evaluate(labels, EvaluationPipeline())


if __name__ == "__main__":
    unittest.main()
