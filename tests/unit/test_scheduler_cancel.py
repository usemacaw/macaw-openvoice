"""Tests for CancellationManager and Scheduler.cancel() (M8-03).

Tests cover: cancel in queue, cancel of nonexistent request, cancel after
completion, concurrent submit+cancel, cancel before dequeue, idempotency,
CancellationManager lifecycle, cancel endpoint.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from macaw._types import ResponseFormat
from macaw.proto.stt_worker_pb2 import Segment, TranscribeFileResponse, Word
from macaw.scheduler.cancel import CancellationManager
from macaw.scheduler.scheduler import Scheduler
from macaw.server.models.requests import TranscribeRequest
from macaw.workers.manager import WorkerHandle, WorkerState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(
    request_id: str = "req_1", model: str = "faster-whisper-tiny"
) -> TranscribeRequest:
    return TranscribeRequest(
        request_id=request_id,
        model_name=model,
        audio_data=b"\x00" * 3200,
        language="pt",
        response_format=ResponseFormat.JSON,
    )


def _make_worker(port: int = 50051) -> WorkerHandle:
    return WorkerHandle(
        worker_id=f"worker-{port}",
        port=port,
        model_name="faster-whisper-tiny",
        engine="faster-whisper",
        state=WorkerState.READY,
    )


def _make_proto_response(text: str = "hello") -> TranscribeFileResponse:
    return TranscribeFileResponse(
        text=text,
        language="pt",
        duration=1.0,
        segments=[
            Segment(
                id=0,
                start=0.0,
                end=1.0,
                text=text,
                avg_logprob=-0.3,
                no_speech_prob=0.02,
                compression_ratio=1.1,
            ),
        ],
        words=[Word(word=text, start=0.0, end=1.0, probability=0.95)],
    )


def _make_scheduler(
    worker: WorkerHandle | None = None,
) -> tuple[Scheduler, MagicMock, MagicMock]:
    mock_manager = MagicMock()
    mock_registry = MagicMock()
    mock_registry.get_manifest.return_value = MagicMock()

    mock_manager.get_ready_worker.return_value = worker

    return Scheduler(mock_manager, mock_registry), mock_manager, mock_registry


# ---------------------------------------------------------------------------
# CancellationManager unit tests
# ---------------------------------------------------------------------------


class TestCancellationManagerRegister:
    def test_register_adds_to_tracking(self) -> None:
        cm = CancellationManager()
        event = asyncio.Event()
        cm.register("req_1", event, None)
        assert cm.pending_count == 1

    def test_register_multiple(self) -> None:
        cm = CancellationManager()
        cm.register("req_1", asyncio.Event(), None)
        cm.register("req_2", asyncio.Event(), None)
        assert cm.pending_count == 2


class TestCancellationManagerCancel:
    def test_cancel_sets_event(self) -> None:
        cm = CancellationManager()
        event = asyncio.Event()
        cm.register("req_1", event, None)
        result = cm.cancel("req_1")
        assert result is True
        assert event.is_set()

    async def test_cancel_cancels_future(self) -> None:
        cm = CancellationManager()
        event = asyncio.Event()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        cm.register("req_1", event, future)
        cm.cancel("req_1")
        assert future.cancelled()

    def test_cancel_nonexistent_returns_false(self) -> None:
        cm = CancellationManager()
        assert cm.cancel("nonexistent") is False

    def test_cancel_removes_from_tracking(self) -> None:
        cm = CancellationManager()
        cm.register("req_1", asyncio.Event(), None)
        cm.cancel("req_1")
        assert cm.pending_count == 0

    def test_cancel_idempotent(self) -> None:
        cm = CancellationManager()
        cm.register("req_1", asyncio.Event(), None)
        assert cm.cancel("req_1") is True
        assert cm.cancel("req_1") is False  # Already removed

    async def test_cancel_does_not_fail_on_done_future(self) -> None:
        cm = CancellationManager()
        event = asyncio.Event()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        future.set_result("done")  # Already resolved
        cm.register("req_1", event, future)
        result = cm.cancel("req_1")
        assert result is True
        assert event.is_set()
        # Future was already done, so cancel() should not raise
        assert future.result() == "done"


class TestCancellationManagerUnregister:
    def test_unregister_removes(self) -> None:
        cm = CancellationManager()
        cm.register("req_1", asyncio.Event(), None)
        cm.unregister("req_1")
        assert cm.pending_count == 0

    def test_unregister_nonexistent_is_noop(self) -> None:
        cm = CancellationManager()
        cm.unregister("nonexistent")  # Should not raise
        assert cm.pending_count == 0


class TestCancellationManagerMarkInFlight:
    def test_mark_in_flight(self) -> None:
        cm = CancellationManager()
        cm.register("req_1", asyncio.Event(), None)
        cm.mark_in_flight("req_1", "localhost:50051")
        assert cm.get_worker_address("req_1") == "localhost:50051"

    def test_mark_in_flight_nonexistent_is_noop(self) -> None:
        cm = CancellationManager()
        cm.mark_in_flight("nonexistent", "localhost:50051")  # No-op

    def test_get_worker_address_none_when_queued(self) -> None:
        cm = CancellationManager()
        cm.register("req_1", asyncio.Event(), None)
        assert cm.get_worker_address("req_1") is None

    def test_get_worker_address_none_when_unknown(self) -> None:
        cm = CancellationManager()
        assert cm.get_worker_address("nonexistent") is None


class TestCancellationManagerIsCancelled:
    def test_not_cancelled_when_registered(self) -> None:
        cm = CancellationManager()
        cm.register("req_1", asyncio.Event(), None)
        assert cm.is_cancelled("req_1") is False

    def test_cancelled_after_cancel(self) -> None:
        cm = CancellationManager()
        cm.register("req_1", asyncio.Event(), None)
        cm.cancel("req_1")
        # After cancel, entry is removed → is_cancelled returns True
        assert cm.is_cancelled("req_1") is True

    def test_cancelled_for_unknown(self) -> None:
        cm = CancellationManager()
        # Unknown request treated as cancelled/completed
        assert cm.is_cancelled("nonexistent") is True


# ---------------------------------------------------------------------------
# Scheduler.cancel() integration tests
# ---------------------------------------------------------------------------


class TestSchedulerCancel:
    async def test_cancel_in_queue(self) -> None:
        """Cancel request while still in queue."""
        worker = _make_worker()
        scheduler, _, _ = _make_scheduler(worker)

        mock_stub = AsyncMock()
        mock_stub.TranscribeFile.return_value = _make_proto_response()

        with (
            patch("macaw.scheduler.scheduler.grpc.aio.insecure_channel", return_value=AsyncMock()),
            patch("macaw.scheduler.scheduler.STTWorkerStub", return_value=mock_stub),
        ):
            await scheduler.start()
            try:
                # Submit but don't let dispatch loop pick it up yet
                future = await scheduler.submit(_make_request("req_cancel"))

                # Cancel immediately
                result = scheduler.cancel("req_cancel")
                assert result is True
                assert future.cancelled()

                # Verify CancellationManager cleaned up
                assert scheduler.cancellation.pending_count == 0
            finally:
                await scheduler.stop()

    async def test_cancel_nonexistent_returns_false(self) -> None:
        scheduler, _, _ = _make_scheduler()
        assert scheduler.cancel("nonexistent") is False

    async def test_cancel_idempotent(self) -> None:
        scheduler, _, _ = _make_scheduler()
        await scheduler.start()
        try:
            await scheduler.submit(_make_request("req_1"))
            assert scheduler.cancel("req_1") is True
            assert scheduler.cancel("req_1") is False
        finally:
            await scheduler.stop()

    async def test_cancel_after_completion(self) -> None:
        """Cancel after request completed returns False (unregistered)."""
        worker = _make_worker()
        scheduler, _, _ = _make_scheduler(worker)

        mock_stub = AsyncMock()
        mock_stub.TranscribeFile.return_value = _make_proto_response()

        with (
            patch("macaw.scheduler.scheduler.grpc.aio.insecure_channel", return_value=AsyncMock()),
            patch("macaw.scheduler.scheduler.STTWorkerStub", return_value=mock_stub),
        ):
            await scheduler.start()
            try:
                future = await scheduler.submit(_make_request("req_done"))
                result = await future  # Wait for completion
                assert result.text == "hello"

                # Cancel after completion is no-op
                assert scheduler.cancel("req_done") is False
            finally:
                await scheduler.stop()

    async def test_cancelled_request_skipped_by_dispatch_loop(self) -> None:
        """Dispatch loop skips cancelled requests without sending to worker."""
        worker = _make_worker()
        scheduler, _, _ = _make_scheduler(worker)

        mock_stub = AsyncMock()
        mock_stub.TranscribeFile.return_value = _make_proto_response()

        with (
            patch("macaw.scheduler.scheduler.grpc.aio.insecure_channel", return_value=AsyncMock()),
            patch("macaw.scheduler.scheduler.STTWorkerStub", return_value=mock_stub),
        ):
            # Submit and cancel BEFORE starting dispatch loop
            await scheduler.submit(_make_request("req_skip"))
            scheduler.cancel("req_skip")

            # Now start — dispatch loop should skip the cancelled request
            await scheduler.start()
            await asyncio.sleep(0.1)  # Give dispatch loop time to process
            await scheduler.stop()

            # TranscribeFile should NOT have been called for cancelled request
            mock_stub.TranscribeFile.assert_not_called()

    async def test_concurrent_submit_and_cancel(self) -> None:
        """Submit and cancel concurrently without errors."""
        scheduler, _, _ = _make_scheduler()
        await scheduler.start()
        try:
            n = 10
            futures = []
            for i in range(n):
                f = await scheduler.submit(_make_request(f"req_{i}"))
                futures.append((f"req_{i}", f))

            # Cancel half
            for i in range(0, n, 2):
                scheduler.cancel(f"req_{i}")

            # Verify cancelled ones are done
            for req_id, future in futures:
                idx = int(req_id.split("_")[1])
                if idx % 2 == 0:
                    assert future.cancelled()
        finally:
            await scheduler.stop()


# ---------------------------------------------------------------------------
# Cancel endpoint tests
# ---------------------------------------------------------------------------


class TestCancelEndpoint:
    async def test_cancel_endpoint_returns_json(self) -> None:
        """POST /v1/audio/transcriptions/{request_id}/cancel returns JSON."""
        from httpx import ASGITransport, AsyncClient

        from macaw.server.app import create_app

        worker = _make_worker()
        scheduler, _, _ = _make_scheduler(worker)
        app = create_app(scheduler=scheduler)

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            # Cancel nonexistent → cancelled: false
            resp = await client.post("/v1/audio/transcriptions/req_x/cancel")
            assert resp.status_code == 200
            data = resp.json()
            assert data["request_id"] == "req_x"
            assert data["cancelled"] is False

    async def test_cancel_endpoint_cancels_queued_request(self) -> None:
        """Cancel endpoint cancels a queued request."""
        from httpx import ASGITransport, AsyncClient

        from macaw.server.app import create_app

        worker = _make_worker()
        scheduler, _, _ = _make_scheduler(worker)
        app = create_app(scheduler=scheduler)

        # Submit a request to the scheduler queue
        await scheduler.start()
        try:
            future = await scheduler.submit(_make_request("req_cancel_api"))

            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                resp = await client.post("/v1/audio/transcriptions/req_cancel_api/cancel")
                assert resp.status_code == 200
                data = resp.json()
                assert data["request_id"] == "req_cancel_api"
                assert data["cancelled"] is True

            assert future.cancelled()
        finally:
            await scheduler.stop()
