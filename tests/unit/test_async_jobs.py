"""Tests for the async job manager (webhook-based transcription).

Covers job submission, status tracking, webhook delivery on success/failure,
metadata forwarding, and bounded job tracking with eviction.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from macaw.scheduler.async_jobs import AsyncJob, AsyncJobManager, JobStatus


class TestSubmit:
    """AsyncJobManager.submit() creates and tracks jobs."""

    def test_returns_job_id(self) -> None:
        manager = AsyncJobManager()
        job_id = manager.submit(
            request_id="req-1",
            webhook_url="https://example.com/hook",
        )
        assert isinstance(job_id, str)
        assert len(job_id) > 0

    def test_creates_pending_job(self) -> None:
        manager = AsyncJobManager()
        job_id = manager.submit(
            request_id="req-1",
            webhook_url="https://example.com/hook",
        )
        job = manager.get(job_id)
        assert job is not None
        assert job.status == JobStatus.PENDING
        assert job.request_id == "req-1"
        assert job.webhook_url == "https://example.com/hook"

    def test_stores_webhook_secret(self) -> None:
        manager = AsyncJobManager()
        job_id = manager.submit(
            request_id="req-1",
            webhook_url="https://example.com/hook",
            webhook_secret="my-secret-key-1234",  # pragma: allowlist secret
        )
        job = manager.get(job_id)
        assert job is not None
        assert job.webhook_secret == "my-secret-key-1234"  # pragma: allowlist secret

    def test_stores_metadata(self) -> None:
        manager = AsyncJobManager()
        metadata = {"user_id": "u-123", "batch": "b-456"}
        job_id = manager.submit(
            request_id="req-1",
            webhook_url="https://example.com/hook",
            metadata=metadata,
        )
        job = manager.get(job_id)
        assert job is not None
        assert job.metadata == metadata


class TestGet:
    """AsyncJobManager.get() retrieves jobs."""

    def test_returns_job_by_id(self) -> None:
        manager = AsyncJobManager()
        job_id = manager.submit(
            request_id="req-1",
            webhook_url="https://example.com/hook",
        )
        job = manager.get(job_id)
        assert job is not None
        assert job.job_id == job_id

    def test_returns_none_for_unknown_id(self) -> None:
        manager = AsyncJobManager()
        assert manager.get("nonexistent-id") is None


class TestRunJobSuccess:
    """AsyncJobManager.run_job() on successful transcription."""

    async def test_sets_status_completed(self) -> None:
        manager = AsyncJobManager()
        job_id = manager.submit(
            request_id="req-1",
            webhook_url="https://example.com/hook",
        )

        delivery = AsyncMock()
        delivery.deliver = AsyncMock(return_value=True)

        async def fake_transcription() -> dict[str, str]:
            return {"text": "hello world"}

        await manager.run_job(job_id, fake_transcription(), delivery)

        job = manager.get(job_id)
        assert job is not None
        assert job.status == JobStatus.COMPLETED
        assert job.result == {"text": "hello world"}

    async def test_delivers_success_payload(self) -> None:
        manager = AsyncJobManager()
        job_id = manager.submit(
            request_id="req-1",
            webhook_url="https://example.com/hook",
            webhook_secret="secret-1234567890",  # pragma: allowlist secret
        )

        delivery = AsyncMock()
        delivery.deliver = AsyncMock(return_value=True)

        async def fake_transcription() -> dict[str, str]:
            return {"text": "hello"}

        await manager.run_job(job_id, fake_transcription(), delivery)

        delivery.deliver.assert_called_once()
        call_kwargs = delivery.deliver.call_args
        assert call_kwargs.kwargs["url"] == "https://example.com/hook"
        assert call_kwargs.kwargs["secret"] == "secret-1234567890"

        payload = call_kwargs.kwargs["payload"]
        assert payload["status"] == "completed"
        assert payload["request_id"] == "req-1"
        assert payload["job_id"] == job_id
        assert payload["result"] == {"text": "hello"}

    async def test_includes_metadata_in_payload(self) -> None:
        manager = AsyncJobManager()
        metadata = {"user_id": "u-123"}
        job_id = manager.submit(
            request_id="req-1",
            webhook_url="https://example.com/hook",
            metadata=metadata,
        )

        delivery = AsyncMock()
        delivery.deliver = AsyncMock(return_value=True)

        async def fake_transcription() -> dict[str, str]:
            return {"text": "hello"}

        await manager.run_job(job_id, fake_transcription(), delivery)

        call_kwargs = delivery.deliver.call_args
        payload = call_kwargs.kwargs["payload"]
        assert payload["metadata"] == metadata


class TestRunJobFailure:
    """AsyncJobManager.run_job() on failed transcription."""

    async def test_sets_status_failed(self) -> None:
        manager = AsyncJobManager()
        job_id = manager.submit(
            request_id="req-1",
            webhook_url="https://example.com/hook",
        )

        delivery = AsyncMock()
        delivery.deliver = AsyncMock(return_value=True)

        async def failing_transcription() -> dict[str, str]:
            msg = "Worker crashed"
            raise RuntimeError(msg)

        await manager.run_job(job_id, failing_transcription(), delivery)

        job = manager.get(job_id)
        assert job is not None
        assert job.status == JobStatus.FAILED
        assert job.error == "Worker crashed"

    async def test_delivers_error_payload(self) -> None:
        manager = AsyncJobManager()
        job_id = manager.submit(
            request_id="req-1",
            webhook_url="https://example.com/hook",
        )

        delivery = AsyncMock()
        delivery.deliver = AsyncMock(return_value=True)

        async def failing_transcription() -> dict[str, str]:
            msg = "Worker crashed"
            raise RuntimeError(msg)

        await manager.run_job(job_id, failing_transcription(), delivery)

        delivery.deliver.assert_called_once()
        call_kwargs = delivery.deliver.call_args
        payload = call_kwargs.kwargs["payload"]
        assert payload["status"] == "failed"
        assert payload["error"] == "Worker crashed"
        assert payload["request_id"] == "req-1"
        assert payload["job_id"] == job_id

    async def test_includes_metadata_in_error_payload(self) -> None:
        manager = AsyncJobManager()
        metadata = {"batch": "b-1"}
        job_id = manager.submit(
            request_id="req-1",
            webhook_url="https://example.com/hook",
            metadata=metadata,
        )

        delivery = AsyncMock()
        delivery.deliver = AsyncMock(return_value=True)

        async def failing_transcription() -> dict[str, str]:
            msg = "fail"
            raise RuntimeError(msg)

        await manager.run_job(job_id, failing_transcription(), delivery)

        call_kwargs = delivery.deliver.call_args
        payload = call_kwargs.kwargs["payload"]
        assert payload["metadata"] == metadata


class TestRunJobEdgeCases:
    """Edge cases for run_job."""

    async def test_unknown_job_id_is_noop(self) -> None:
        manager = AsyncJobManager()
        delivery = AsyncMock()

        # Use an already-resolved future to avoid unawaited coroutine warning
        future: asyncio.Future[None] = asyncio.get_event_loop().create_future()
        future.set_result(None)

        # Should not raise
        await manager.run_job("nonexistent", future, delivery)
        delivery.deliver.assert_not_called()


class TestBoundedJobTracking:
    """AsyncJobManager respects max_jobs bound."""

    def test_evicts_completed_when_full(self) -> None:
        manager = AsyncJobManager(max_jobs=2)

        # Submit 2 jobs
        jid1 = manager.submit(request_id="r1", webhook_url="https://example.com")
        jid2 = manager.submit(request_id="r2", webhook_url="https://example.com")

        # Mark first as completed
        job1 = manager.get(jid1)
        assert job1 is not None
        job1.status = JobStatus.COMPLETED

        # Submit a third -- should evict jid1
        jid3 = manager.submit(request_id="r3", webhook_url="https://example.com")

        assert manager.get(jid1) is None
        assert manager.get(jid2) is not None
        assert manager.get(jid3) is not None


class TestShutdown:
    """AsyncJobManager.shutdown() cancels running tasks."""

    async def test_shutdown_clears_tasks(self) -> None:
        manager = AsyncJobManager()

        # Add a mock task
        mock_task = AsyncMock(spec=asyncio.Task)
        manager._tasks["t1"] = mock_task

        await manager.shutdown()

        mock_task.cancel.assert_called_once()
        assert len(manager._tasks) == 0


class TestAsyncJobDataclass:
    """AsyncJob dataclass defaults."""

    def test_default_values(self) -> None:
        job = AsyncJob(job_id="j1", request_id="r1")
        assert job.status == JobStatus.PENDING
        assert job.result is None
        assert job.error is None
        assert job.webhook_url == ""
        assert job.webhook_secret is None
        assert job.metadata is None


class TestJobStatusEnum:
    """JobStatus enum values."""

    def test_values(self) -> None:
        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
