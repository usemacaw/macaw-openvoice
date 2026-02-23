"""In-memory async job manager for webhook-based transcription."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from macaw.logging import get_logger

if TYPE_CHECKING:
    import asyncio

logger = get_logger("scheduler.async_jobs")


class JobStatus(Enum):
    """Status of an async transcription job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AsyncJob:
    """Tracks state and result of an async transcription job."""

    job_id: str
    request_id: str
    status: JobStatus = JobStatus.PENDING
    result: dict[str, Any] | None = None
    error: str | None = None
    webhook_url: str = ""
    webhook_secret: str | None = None
    metadata: dict[str, object] | None = None


class AsyncJobManager:
    """Manages async transcription jobs with webhook delivery.

    Tracks pending/running/completed jobs in-memory (bounded).
    """

    def __init__(self, max_jobs: int = 1000) -> None:
        self._jobs: dict[str, AsyncJob] = {}
        self._max_jobs = max_jobs
        self._tasks: dict[str, asyncio.Task[None]] = {}

    def submit(
        self,
        request_id: str,
        webhook_url: str,
        webhook_secret: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> str:
        """Submit a new async job. Returns job_id."""
        if len(self._jobs) >= self._max_jobs:
            self._evict_completed()

        job_id = str(uuid.uuid4())
        job = AsyncJob(
            job_id=job_id,
            request_id=request_id,
            webhook_url=webhook_url,
            webhook_secret=webhook_secret,
            metadata=metadata,
        )
        self._jobs[job_id] = job
        logger.info("async_job_submitted", job_id=job_id, request_id=request_id)
        return job_id

    def get(self, job_id: str) -> AsyncJob | None:
        """Return job by ID, or None if not found."""
        return self._jobs.get(job_id)

    async def run_job(
        self,
        job_id: str,
        coro: Any,  # coroutine that returns transcription result
        delivery: Any,  # WebhookDelivery instance
    ) -> None:
        """Run transcription and deliver result via webhook."""
        job = self._jobs.get(job_id)
        if not job:
            return

        job.status = JobStatus.RUNNING
        try:
            result = await coro
            job.status = JobStatus.COMPLETED
            job.result = result

            payload: dict[str, object] = {
                "status": "completed",
                "request_id": job.request_id,
                "job_id": job_id,
                "result": result,
            }
            if job.metadata:
                payload["metadata"] = job.metadata

            await delivery.deliver(
                url=job.webhook_url,
                payload=payload,
                secret=job.webhook_secret,
            )
        except Exception as exc:
            job.status = JobStatus.FAILED
            job.error = str(exc)
            logger.error("async_job_failed", job_id=job_id, error=str(exc))

            error_payload: dict[str, object] = {
                "status": "failed",
                "request_id": job.request_id,
                "job_id": job_id,
                "error": str(exc),
            }
            if job.metadata:
                error_payload["metadata"] = job.metadata

            await delivery.deliver(
                url=job.webhook_url,
                payload=error_payload,
                secret=job.webhook_secret,
            )

    def track_task(self, job_id: str, task: asyncio.Task[None]) -> None:
        """Store a reference to a background task for graceful shutdown."""
        self._tasks[job_id] = task

    def _evict_completed(self) -> None:
        """Remove oldest completed/failed jobs to make room."""
        to_remove = [
            jid
            for jid, j in self._jobs.items()
            if j.status in (JobStatus.COMPLETED, JobStatus.FAILED)
        ]
        for jid in to_remove[: max(1, len(to_remove) // 2)]:
            del self._jobs[jid]

    async def shutdown(self) -> None:
        """Cancel all running tasks."""
        for task in self._tasks.values():
            task.cancel()
        self._tasks.clear()
