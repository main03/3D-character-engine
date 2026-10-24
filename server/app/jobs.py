"""In-process job queue. One worker: CPU inference saturates the machine anyway."""

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

from charengine import CharacterPipeline, CharEngineError

log = logging.getLogger(__name__)


@dataclass
class Job:
    id: str
    status: str = "queued"  # queued | running | succeeded | failed
    stage: Optional[str] = None
    progress: float = 0.0
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    glb: Optional[bytes] = field(default=None, repr=False)
    stats: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    reconstructor: Optional[str] = None


class QueueFull(Exception):
    pass


class JobManager:
    def __init__(self, pipeline: CharacterPipeline, max_pending: int = 8, result_ttl: float = 3600.0):
        self.pipeline = pipeline
        self.max_pending = max_pending
        self.result_ttl = result_ttl
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="charengine")

    def submit(self, data: bytes) -> Job:
        self._expire()
        with self._lock:
            pending = sum(j.status in ("queued", "running") for j in self._jobs.values())
            if pending >= self.max_pending:
                raise QueueFull()
            job = Job(id=uuid.uuid4().hex)
            self._jobs[job.id] = job
        self._executor.submit(self._run, job, data)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        self._expire()
        with self._lock:
            return self._jobs.get(job_id)

    def queue_position(self, job: Job) -> int:
        with self._lock:
            return sum(j.status == "queued" and j.created_at < job.created_at for j in self._jobs.values())

    def shutdown(self):
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _run(self, job: Job, data: bytes):
        job.status = "running"

        def on_progress(stage: str, fraction: float):
            job.stage, job.progress = stage, fraction

        try:
            result = self.pipeline.generate(data, on_progress)
            job.glb, job.stats, job.timings = result.glb, result.stats, result.timings
            job.warnings, job.reconstructor = result.warnings, result.reconstructor
            job.status, job.progress = "succeeded", 1.0
        except CharEngineError as exc:
            job.status, job.error = "failed", str(exc)
        except Exception:
            log.exception("job %s crashed", job.id)
            job.status, job.error = "failed", "Internal error while generating the character."
        finally:
            del data
            job.finished_at = time.time()

    def _expire(self):
        cutoff = time.time() - self.result_ttl
        with self._lock:
            for job_id in [k for k, j in self._jobs.items() if j.finished_at and j.finished_at < cutoff]:
                del self._jobs[job_id]
