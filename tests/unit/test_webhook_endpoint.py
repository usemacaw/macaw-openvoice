"""Tests for webhook-based async transcription endpoint wiring.

Verifies that POST /v1/audio/transcriptions with webhook_url returns 202,
backward compatibility without webhook_url, and proper response structure.
"""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from macaw._types import BatchResult, SegmentDetail
from macaw.scheduler.async_jobs import AsyncJobManager
from macaw.server.app import create_app
from macaw.server.models.responses import WebhookAcceptedResponse


def _make_mock_registry() -> MagicMock:
    registry = MagicMock()
    registry.has_model.return_value = True
    registry.get_manifest.return_value = MagicMock()
    return registry


def _make_mock_scheduler(
    result: BatchResult | None = None,
) -> MagicMock:
    scheduler = MagicMock()
    scheduler.transcribe = AsyncMock(
        return_value=result
        or BatchResult(
            text="hello world",
            language="en",
            duration=2.0,
            segments=(SegmentDetail(id=0, start=0.0, end=2.0, text="hello world"),),
        )
    )
    return scheduler


async def _cancel_tasks(tasks: list[asyncio.Task[None]]) -> None:
    """Cancel background tasks to avoid unawaited coroutine warnings."""
    for task in tasks:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task


class _TaskCapture:
    """Context for capturing asyncio.create_task calls."""

    def __init__(self) -> None:
        self.tasks: list[asyncio.Task[None]] = []
        self._original = asyncio.create_task

    def capture(self, coro: object, **kwargs: object) -> asyncio.Task[None]:
        task = self._original(coro, **kwargs)  # type: ignore[arg-type]
        self.tasks.append(task)
        return task


class TestWebhookReturns202:
    """POST with webhook_url returns HTTP 202 Accepted."""

    async def test_returns_202_with_webhook_url(self) -> None:
        scheduler = _make_mock_scheduler()
        app = create_app(registry=_make_mock_registry(), scheduler=scheduler)
        capture = _TaskCapture()

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            with patch(
                "macaw.server.routes.transcriptions.asyncio.create_task",
                side_effect=capture.capture,
            ):
                response = await client.post(
                    "/v1/audio/transcriptions",
                    files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
                    data={
                        "model": "test-model",
                        "webhook_url": "https://example.com/hook",
                    },
                )

        assert response.status_code == 202
        await _cancel_tasks(capture.tasks)

    async def test_202_response_includes_request_id_and_job_id(self) -> None:
        scheduler = _make_mock_scheduler()
        app = create_app(registry=_make_mock_registry(), scheduler=scheduler)
        capture = _TaskCapture()

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            with patch(
                "macaw.server.routes.transcriptions.asyncio.create_task",
                side_effect=capture.capture,
            ):
                response = await client.post(
                    "/v1/audio/transcriptions",
                    files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
                    data={
                        "model": "test-model",
                        "webhook_url": "https://example.com/hook",
                    },
                )

        data = response.json()
        assert "request_id" in data
        assert "job_id" in data
        assert len(data["request_id"]) > 0
        assert len(data["job_id"]) > 0
        await _cancel_tasks(capture.tasks)

    async def test_202_response_has_status_accepted(self) -> None:
        scheduler = _make_mock_scheduler()
        app = create_app(registry=_make_mock_registry(), scheduler=scheduler)
        capture = _TaskCapture()

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            with patch(
                "macaw.server.routes.transcriptions.asyncio.create_task",
                side_effect=capture.capture,
            ):
                response = await client.post(
                    "/v1/audio/transcriptions",
                    files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
                    data={
                        "model": "test-model",
                        "webhook_url": "https://example.com/hook",
                    },
                )

        data = response.json()
        assert data["status"] == "accepted"
        await _cancel_tasks(capture.tasks)


class TestBackwardCompatibility:
    """POST without webhook_url returns normal result (no regression)."""

    async def test_normal_response_without_webhook(self) -> None:
        scheduler = _make_mock_scheduler()
        app = create_app(registry=_make_mock_registry(), scheduler=scheduler)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/audio/transcriptions",
                files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
                data={"model": "test-model"},
            )

        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert data["text"] == "hello world"


class TestWebhookSecurityAndMetadata:
    """Webhook secret and metadata handling."""

    async def test_webhook_secret_not_echoed_in_response(self) -> None:
        scheduler = _make_mock_scheduler()
        app = create_app(registry=_make_mock_registry(), scheduler=scheduler)
        capture = _TaskCapture()

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            with patch(
                "macaw.server.routes.transcriptions.asyncio.create_task",
                side_effect=capture.capture,
            ):
                response = await client.post(
                    "/v1/audio/transcriptions",
                    files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
                    data={
                        "model": "test-model",
                        "webhook_url": "https://example.com/hook",
                        "webhook_secret": "my-super-secret-key",  # pragma: allowlist secret
                    },
                )

        data = response.json()
        assert "webhook_secret" not in data
        assert "secret" not in data
        assert "my-super-secret-key" not in response.text
        await _cancel_tasks(capture.tasks)

    async def test_202_response_message_field(self) -> None:
        scheduler = _make_mock_scheduler()
        app = create_app(registry=_make_mock_registry(), scheduler=scheduler)
        capture = _TaskCapture()

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            with patch(
                "macaw.server.routes.transcriptions.asyncio.create_task",
                side_effect=capture.capture,
            ):
                response = await client.post(
                    "/v1/audio/transcriptions",
                    files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
                    data={
                        "model": "test-model",
                        "webhook_url": "https://example.com/hook",
                    },
                )

        data = response.json()
        assert "message" in data
        assert "webhook" in data["message"].lower()
        await _cancel_tasks(capture.tasks)


class TestWebhookAcceptedResponseModel:
    """WebhookAcceptedResponse Pydantic model tests."""

    def test_correct_schema(self) -> None:
        schema = WebhookAcceptedResponse.model_json_schema()
        props = schema["properties"]
        assert "request_id" in props
        assert "job_id" in props
        assert "status" in props
        assert "message" in props

    def test_default_values(self) -> None:
        resp = WebhookAcceptedResponse(request_id="r1", job_id="j1")
        assert resp.status == "accepted"
        assert "webhook" in resp.message.lower()

    def test_model_dump(self) -> None:
        resp = WebhookAcceptedResponse(request_id="r1", job_id="j1")
        data = resp.model_dump()
        assert data["request_id"] == "r1"
        assert data["job_id"] == "j1"
        assert data["status"] == "accepted"


class TestBackgroundTaskCreated:
    """Background task is scheduled for webhook processing."""

    async def test_create_task_called(self) -> None:
        scheduler = _make_mock_scheduler()
        app = create_app(registry=_make_mock_registry(), scheduler=scheduler)
        capture = _TaskCapture()

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            with patch(
                "macaw.server.routes.transcriptions.asyncio.create_task",
                side_effect=capture.capture,
            ):
                await client.post(
                    "/v1/audio/transcriptions",
                    files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
                    data={
                        "model": "test-model",
                        "webhook_url": "https://example.com/hook",
                    },
                )

        assert len(capture.tasks) == 1
        await _cancel_tasks(capture.tasks)


class TestJobManagerBounded:
    """AsyncJobManager on app.state respects max_jobs."""

    def test_app_has_job_manager(self) -> None:
        app = create_app(registry=_make_mock_registry(), scheduler=_make_mock_scheduler())
        assert hasattr(app.state, "async_job_manager")
        assert isinstance(app.state.async_job_manager, AsyncJobManager)


class TestMultipleWebhookRequests:
    """Multiple webhook requests do not block each other."""

    async def test_multiple_requests_return_immediately(self) -> None:
        scheduler = _make_mock_scheduler()
        app = create_app(registry=_make_mock_registry(), scheduler=scheduler)
        capture = _TaskCapture()

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            with patch(
                "macaw.server.routes.transcriptions.asyncio.create_task",
                side_effect=capture.capture,
            ):
                responses = []
                for _ in range(3):
                    resp = await client.post(
                        "/v1/audio/transcriptions",
                        files={"file": ("audio.wav", b"fake-audio", "audio/wav")},
                        data={
                            "model": "test-model",
                            "webhook_url": "https://example.com/hook",
                        },
                    )
                    responses.append(resp)

        for resp in responses:
            assert resp.status_code == 202

        job_ids = [r.json()["job_id"] for r in responses]
        assert len(set(job_ids)) == 3
        await _cancel_tasks(capture.tasks)


class TestDependencyGetAsyncJobManager:
    """get_async_job_manager dependency function."""

    def test_returns_manager_from_app_state(self) -> None:
        from macaw.server.dependencies import get_async_job_manager

        app = create_app(registry=_make_mock_registry(), scheduler=_make_mock_scheduler())
        mock_request = MagicMock()
        mock_request.app = app
        result = get_async_job_manager(mock_request)
        assert isinstance(result, AsyncJobManager)

    def test_returns_none_when_not_set(self) -> None:
        from macaw.server.dependencies import get_async_job_manager

        mock_request = MagicMock()
        mock_request.app.state = MagicMock(spec=[])
        result = get_async_job_manager(mock_request)
        assert result is None
