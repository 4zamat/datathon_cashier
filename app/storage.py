"""Thread-safe memory cache backed by atomically replaced JSON files."""

import logging
import os
from pathlib import Path
from threading import RLock

from app.models import Job

logger = logging.getLogger(__name__)


class JobStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._jobs: dict[str, Job] = {}
        for path in self.directory.glob("*.json"):
            try:
                job = Job.model_validate_json(path.read_text(encoding="utf-8"))
                if path.stem != job.job_id:
                    raise ValueError("Job ID does not match its filename")
                self._jobs[job.job_id] = job
            except (OSError, ValueError):
                logger.exception("Could not restore job file %s", path.name)
        for job in list(self._jobs.values()):
            if job.status == "processing":
                self.save(Job(job_id=job.job_id, status="failed", error="Service restarted before analysis completed"))

    def save(self, job: Job) -> None:
        with self._lock:
            target = self.directory / f"{job.job_id}.json"
            temporary = target.with_suffix(".tmp")
            try:
                with temporary.open("w", encoding="utf-8") as stream:
                    stream.write(job.model_dump_json())
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, target)
                self._jobs[job.job_id] = job.model_copy(deep=True)
            finally:
                temporary.unlink(missing_ok=True)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.model_copy(deep=True) if job else None
