"""One queue thread, at most one killable pipeline process at a time."""

import logging
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
from time import monotonic

from app.config import Settings
from app.models import Job
from app.rules import RuleCatalog
from app.storage import JobStore
from pipeline.base import Result, load_pipeline

logger = logging.getLogger(__name__)


def run_pipeline(spec: str, video_path: Path, output_path: Path) -> None:
    """Spawn entrypoint: only JSON crosses the process boundary."""
    result = Result.model_validate(load_pipeline(spec).analyze(video_path))
    output_path.write_text(result.model_dump_json(), encoding="utf-8")


@dataclass(frozen=True)
class PendingJob:
    job_id: str
    video_path: Path
    deadline: float


class JobWorker:
    def __init__(self, settings: Settings, store: JobStore, rules: RuleCatalog) -> None:
        self.settings = settings
        self.store = store
        self.rules = rules
        self.upload_dir = settings.data_dir / "uploads"
        self.result_dir = settings.data_dir / "work"
        for directory in (self.upload_dir, self.result_dir):
            directory.mkdir(parents=True, exist_ok=True)
            # These private directories contain only disposable service-owned files.
            for path in directory.iterdir():
                if path.is_file():
                    path.unlink()
        self._queue: Queue[PendingJob] = Queue(maxsize=settings.max_pending_jobs)
        self._stop = Event()
        self._thread = Thread(target=self._loop, name="analysis-worker", daemon=True)
        self._context = mp.get_context("spawn")

    def start(self) -> None:
        self._thread.start()

    def submit(self, job_id: str, video_path: Path) -> None:
        if self._stop.is_set():
            raise RuntimeError("Worker is shutting down")
        pending = PendingJob(job_id, video_path, monotonic() + self.settings.processing_timeout_sec)
        self.store.save(Job(job_id=job_id))
        try:
            self._queue.put_nowait(pending)
        except Full:
            self.store.save(Job(job_id=job_id, status="failed", error="Queue is full"))
            raise

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()

    def _loop(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                pending = self._queue.get(timeout=0.1)
            except Empty:
                continue
            output_path = self.result_dir / f"{pending.job_id}.json"
            try:
                job = self._analyze(pending, output_path)
            except Exception:
                logger.exception("Analysis failed for job %s", pending.job_id)
                job = Job(job_id=pending.job_id, status="failed", error="Analysis failed; see server logs")
            finally:
                # A terminal status is published only after video cleanup is attempted.
                try:
                    pending.video_path.unlink(missing_ok=True)
                    output_path.unlink(missing_ok=True)
                except OSError:
                    logger.exception("Cleanup failed for job %s", pending.job_id)
                    job = Job(job_id=pending.job_id, status="failed", error="Could not remove temporary analysis files")
            try:
                self.store.save(job)
            except OSError:
                logger.exception("Could not persist terminal status for %s", pending.job_id)
            finally:
                self._queue.task_done()

    def _analyze(self, pending: PendingJob, output_path: Path) -> Job:
        if self._stop.is_set():
            return Job(job_id=pending.job_id, status="failed", error="Service is shutting down")
        if monotonic() >= pending.deadline:
            return self._timeout(pending)
        process = self._context.Process(
            target=run_pipeline,
            args=(self.settings.pipeline_class, pending.video_path, output_path),
            name=f"pipeline-{pending.job_id}",
            daemon=True,
        )
        process.start()
        try:
            while process.is_alive():
                remaining = pending.deadline - monotonic()
                if self._stop.is_set():
                    return Job(job_id=pending.job_id, status="failed", error="Service is shutting down")
                if remaining <= 0:
                    return self._timeout(pending)
                process.join(timeout=min(0.1, remaining))
            if monotonic() >= pending.deadline:
                return self._timeout(pending)
            if process.exitcode != 0:
                raise RuntimeError(f"Pipeline exited with code {process.exitcode}")
            result = Result.model_validate_json(output_path.read_text(encoding="utf-8"))
            violations = self.rules.enrich(result)
            if monotonic() >= pending.deadline:
                return self._timeout(pending)
            return Job(job_id=pending.job_id, status="done", verdict=result.verdict, violations=violations)
        finally:
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)
                if process.is_alive():
                    process.kill()
                    process.join()
            process.close()

    def _timeout(self, pending: PendingJob) -> Job:
        return Job(
            job_id=pending.job_id,
            status="failed",
            error=f"Processing timeout ({self.settings.processing_timeout_sec:g} seconds, including queue wait)",
        )
