"""Tests for Scheduler async dispatch loop (M8-02).

Tests cover: start/stop lifecycle, dispatch loop, priority ordering,
FIFO within level, worker unavailable re-enqueue, worker crash,
timeout, graceful shutdown, concurrent requests, channel pool,
cancel skip, inline fallback.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from macaw._types import ResponseFormat
from macaw.exceptions import (
    ModelNotFoundError,
    WorkerCrashError,
    WorkerTimeoutError,
)
from macaw.proto import Segment, TranscribeFileResponse, Word
from macaw.scheduler.queue import RequestPriority
from macaw.scheduler.scheduler import Scheduler
from macaw.server.models.requests import TranscribeRequest
from macaw.workers.manager import WorkerHandle, WorkerState


def _make_request(
    request_id: str = "req_1",
    model_name: str = "faster-whisper-tiny",
) -> TranscribeRequest:
    return TranscribeRequest(
        request_id=request_id,
        model_name=model_name,
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


def _make_proto_response(text: str = "Ola mundo") -> TranscribeFileResponse:
    return TranscribeFileResponse(
        text=text,
        language="pt",
        duration=1.5,
        segments=[
            Segment(
                id=0,
                start=0.0,
                end=1.5,
                text=text,
                avg_logprob=-0.3,
                no_speech_prob=0.02,
                compression_ratio=1.1,
            ),
        ],
        words=[
            Word(word=text, start=0.0, end=1.5, probability=0.95),
        ],
    )


def _make_scheduler(
    worker: WorkerHandle | None = None,
    *,
    model_not_found: bool = False,
) -> tuple[Scheduler, MagicMock, MagicMock]:
    """Create scheduler with mocked dependencies."""
    registry = MagicMock()
    if model_not_found:
        registry.get_manifest.side_effect = ModelNotFoundError("test-model")
    else:
        registry.get_manifest.return_value = MagicMock()

    worker_manager = MagicMock()
    worker_manager.get_ready_worker.return_value = worker

    scheduler = Scheduler(worker_manager, registry)
    return scheduler, worker_manager, registry


class TestSchedulerLifecycle:
    async def test_start_sets_running(self) -> None:
        scheduler, _, _ = _make_scheduler()
        assert scheduler.running is False
        await scheduler.start()
        assert scheduler.running is True
        await scheduler.stop()

    async def test_stop_clears_running(self) -> None:
        scheduler, _, _ = _make_scheduler()
        await scheduler.start()
        await scheduler.stop()
        assert scheduler.running is False

    async def test_start_is_idempotent(self) -> None:
        scheduler, _, _ = _make_scheduler()
        await scheduler.start()
        await scheduler.start()  # no-op
        assert scheduler.running is True
        await scheduler.stop()

    async def test_stop_is_idempotent(self) -> None:
        scheduler, _, _ = _make_scheduler()
        await scheduler.stop()  # no-op when not running
        assert scheduler.running is False

    async def test_stop_without_start(self) -> None:
        scheduler, _, _ = _make_scheduler()
        await scheduler.stop()  # should not raise


class TestSchedulerDispatch:
    async def test_submit_and_await_result(self) -> None:
        """Submit a request via the dispatch loop and get result."""
        worker = _make_worker()
        scheduler, _, _ = _make_scheduler(worker)

        proto_response = _make_proto_response()
        mock_stub = AsyncMock()
        mock_stub.TranscribeFile.return_value = proto_response

        with (
            patch(
                "macaw.scheduler.scheduler.grpc.aio.insecure_channel",
                return_value=AsyncMock(),
            ),
            patch(
                "macaw.scheduler.scheduler.STTWorkerStub",
                return_value=mock_stub,
            ),
        ):
            await scheduler.start()
            try:
                result = await scheduler.transcribe(_make_request())
                assert result.text == "Ola mundo"
                assert result.language == "pt"
            finally:
                await scheduler.stop()

    async def test_submit_with_priority(self) -> None:
        """Submit with explicit priority returns future."""
        worker = _make_worker()
        scheduler, _, _ = _make_scheduler(worker)

        proto_response = _make_proto_response()
        mock_stub = AsyncMock()
        mock_stub.TranscribeFile.return_value = proto_response

        with (
            patch(
                "macaw.scheduler.scheduler.grpc.aio.insecure_channel",
                return_value=AsyncMock(),
            ),
            patch(
                "macaw.scheduler.scheduler.STTWorkerStub",
                return_value=mock_stub,
            ),
        ):
            await scheduler.start()
            try:
                future = await scheduler.submit(
                    _make_request(),
                    priority=RequestPriority.REALTIME,
                )
                assert isinstance(future, asyncio.Future)
                result = await asyncio.wait_for(future, timeout=5.0)
                assert result.text == "Ola mundo"
            finally:
                await scheduler.stop()

    async def test_model_not_found_raises_before_enqueue(self) -> None:
        """ModelNotFoundError is raised before enqueuing."""
        scheduler, _, _ = _make_scheduler(model_not_found=True)
        await scheduler.start()
        try:
            with pytest.raises(ModelNotFoundError):
                await scheduler.transcribe(_make_request())
        finally:
            await scheduler.stop()

    async def test_dispatch_skips_cancelled_request(self) -> None:
        """Dispatch loop skips requests with cancel_event set."""
        worker = _make_worker()
        scheduler, _, _ = _make_scheduler(worker)

        proto_response = _make_proto_response()
        mock_stub = AsyncMock()
        mock_stub.TranscribeFile.return_value = proto_response

        with (
            patch(
                "macaw.scheduler.scheduler.grpc.aio.insecure_channel",
                return_value=AsyncMock(),
            ),
            patch(
                "macaw.scheduler.scheduler.STTWorkerStub",
                return_value=mock_stub,
            ),
        ):
            await scheduler.start()
            try:
                # Submit and immediately cancel
                future = await scheduler.submit(_make_request("r1"))
                scheduler.queue.cancel("r1")
                assert future.cancelled()

                # Submit a second request that should be processed
                result = await scheduler.transcribe(_make_request("r2"))
                assert result.text == "Ola mundo"
            finally:
                await scheduler.stop()


class TestSchedulerWorkerUnavailable:
    async def test_requeue_when_no_worker(self) -> None:
        """Request is re-enqueued when no worker is available, then processed."""
        worker = _make_worker()
        scheduler, worker_manager, _ = _make_scheduler()

        # First call: no worker. Second call: worker available.
        worker_manager.get_ready_worker.side_effect = [None, worker]

        proto_response = _make_proto_response()
        mock_stub = AsyncMock()
        mock_stub.TranscribeFile.return_value = proto_response

        # Speed up backoff for test
        scheduler._settings = type(scheduler._settings).model_validate(
            {**scheduler._settings.model_dump(), "no_worker_backoff_s": 0.01}
        )

        with (
            patch(
                "macaw.scheduler.scheduler.grpc.aio.insecure_channel",
                return_value=AsyncMock(),
            ),
            patch(
                "macaw.scheduler.scheduler.STTWorkerStub",
                return_value=mock_stub,
            ),
        ):
            await scheduler.start()
            try:
                result = await asyncio.wait_for(
                    scheduler.transcribe(_make_request()),
                    timeout=5.0,
                )
                assert result.text == "Ola mundo"
                assert worker_manager.get_ready_worker.call_count == 2
            finally:
                await scheduler.stop()


class TestSchedulerErrors:
    async def test_worker_crash_rejects_future(self) -> None:
        """gRPC UNAVAILABLE error rejects the future with WorkerCrashError."""
        import grpc
        import grpc.aio

        worker = _make_worker()
        scheduler, _, _ = _make_scheduler(worker)

        mock_stub = AsyncMock()
        mock_stub.TranscribeFile.side_effect = grpc.aio.AioRpcError(
            code=grpc.StatusCode.UNAVAILABLE,
            initial_metadata=grpc.aio.Metadata(),
            trailing_metadata=grpc.aio.Metadata(),
            details="Connection refused",
        )

        with (
            patch(
                "macaw.scheduler.scheduler.grpc.aio.insecure_channel",
                return_value=AsyncMock(),
            ),
            patch(
                "macaw.scheduler.scheduler.STTWorkerStub",
                return_value=mock_stub,
            ),
        ):
            await scheduler.start()
            try:
                with pytest.raises(WorkerCrashError):
                    await asyncio.wait_for(
                        scheduler.transcribe(_make_request()),
                        timeout=5.0,
                    )
            finally:
                await scheduler.stop()

    async def test_worker_timeout_rejects_future(self) -> None:
        """gRPC DEADLINE_EXCEEDED error rejects future with WorkerTimeoutError."""
        import grpc
        import grpc.aio

        worker = _make_worker()
        scheduler, _, _ = _make_scheduler(worker)

        mock_stub = AsyncMock()
        mock_stub.TranscribeFile.side_effect = grpc.aio.AioRpcError(
            code=grpc.StatusCode.DEADLINE_EXCEEDED,
            initial_metadata=grpc.aio.Metadata(),
            trailing_metadata=grpc.aio.Metadata(),
            details="Deadline Exceeded",
        )

        with (
            patch(
                "macaw.scheduler.scheduler.grpc.aio.insecure_channel",
                return_value=AsyncMock(),
            ),
            patch(
                "macaw.scheduler.scheduler.STTWorkerStub",
                return_value=mock_stub,
            ),
        ):
            await scheduler.start()
            try:
                with pytest.raises(WorkerTimeoutError):
                    await asyncio.wait_for(
                        scheduler.transcribe(_make_request()),
                        timeout=5.0,
                    )
            finally:
                await scheduler.stop()

    async def test_generic_exception_rejects_future(self) -> None:
        """Non-gRPC exception rejects the future."""
        worker = _make_worker()
        scheduler, _, _ = _make_scheduler(worker)

        mock_stub = AsyncMock()
        mock_stub.TranscribeFile.side_effect = RuntimeError("unexpected")

        with (
            patch(
                "macaw.scheduler.scheduler.grpc.aio.insecure_channel",
                return_value=AsyncMock(),
            ),
            patch(
                "macaw.scheduler.scheduler.STTWorkerStub",
                return_value=mock_stub,
            ),
        ):
            await scheduler.start()
            try:
                with pytest.raises(RuntimeError, match="unexpected"):
                    await asyncio.wait_for(
                        scheduler.transcribe(_make_request()),
                        timeout=5.0,
                    )
            finally:
                await scheduler.stop()


class TestSchedulerPriority:
    async def test_realtime_processed_before_batch(self) -> None:
        """REALTIME requests are dispatched before BATCH."""
        worker = _make_worker()
        scheduler, _, _ = _make_scheduler(worker)

        results_order: list[str] = []

        async def fake_transcribe(
            proto_req: object, timeout: float = 30
        ) -> TranscribeFileResponse:
            # Extract request_id from proto
            req_id = proto_req.request_id  # type: ignore[union-attr]
            results_order.append(req_id)
            return _make_proto_response(text=req_id)

        mock_stub = AsyncMock()
        mock_stub.TranscribeFile.side_effect = fake_transcribe

        with (
            patch(
                "macaw.scheduler.scheduler.grpc.aio.insecure_channel",
                return_value=AsyncMock(),
            ),
            patch(
                "macaw.scheduler.scheduler.STTWorkerStub",
                return_value=mock_stub,
            ),
        ):
            # Submit batch first, then realtime — but DON'T start yet
            batch_future = await scheduler.submit(
                _make_request("batch_1"),
                RequestPriority.BATCH,
            )
            rt_future = await scheduler.submit(
                _make_request("rt_1"),
                RequestPriority.REALTIME,
            )

            # Now start: dispatch loop processes realtime first
            await scheduler.start()
            try:
                await asyncio.wait_for(batch_future, timeout=5.0)
                await asyncio.wait_for(rt_future, timeout=5.0)
            finally:
                await scheduler.stop()

            # REALTIME should be processed before BATCH
            assert results_order[0] == "rt_1"
            assert results_order[1] == "batch_1"

    async def test_fifo_within_same_priority(self) -> None:
        """Requests within same priority are processed in FIFO order."""
        worker = _make_worker()
        scheduler, _, _ = _make_scheduler(worker)

        results_order: list[str] = []

        async def fake_transcribe(
            proto_req: object, timeout: float = 30
        ) -> TranscribeFileResponse:
            req_id = proto_req.request_id  # type: ignore[union-attr]
            results_order.append(req_id)
            return _make_proto_response(text=req_id)

        mock_stub = AsyncMock()
        mock_stub.TranscribeFile.side_effect = fake_transcribe

        with (
            patch(
                "macaw.scheduler.scheduler.grpc.aio.insecure_channel",
                return_value=AsyncMock(),
            ),
            patch(
                "macaw.scheduler.scheduler.STTWorkerStub",
                return_value=mock_stub,
            ),
        ):
            futures = []
            for i in range(3):
                f = await scheduler.submit(
                    _make_request(f"r{i}"),
                    RequestPriority.BATCH,
                )
                futures.append(f)

            await scheduler.start()
            try:
                for f in futures:
                    await asyncio.wait_for(f, timeout=5.0)
            finally:
                await scheduler.stop()

            assert results_order == ["r0", "r1", "r2"]


class TestSchedulerChannelPool:
    async def test_channel_reused_for_same_address(self) -> None:
        """Same worker address reuses the same channel."""
        worker = _make_worker()
        scheduler, _, _ = _make_scheduler(worker)

        proto_response = _make_proto_response()
        mock_stub = AsyncMock()
        mock_stub.TranscribeFile.return_value = proto_response

        mock_channel = AsyncMock()
        channel_create_count = 0

        def make_channel(*args: object, **kwargs: object) -> AsyncMock:
            nonlocal channel_create_count
            channel_create_count += 1
            return mock_channel

        with (
            patch(
                "macaw.scheduler.scheduler.grpc.aio.insecure_channel",
                side_effect=make_channel,
            ),
            patch(
                "macaw.scheduler.scheduler.STTWorkerStub",
                return_value=mock_stub,
            ),
        ):
            await scheduler.start()
            try:
                # Two requests to same worker = same channel
                await scheduler.transcribe(_make_request("r1"))
                await scheduler.transcribe(_make_request("r2"))
            finally:
                await scheduler.stop()

            # Channel created only once (pooled)
            assert channel_create_count == 1

    async def test_channels_closed_on_stop(self) -> None:
        """All pooled channels are closed when scheduler stops."""
        worker = _make_worker()
        scheduler, _, _ = _make_scheduler(worker)

        proto_response = _make_proto_response()
        mock_stub = AsyncMock()
        mock_stub.TranscribeFile.return_value = proto_response

        mock_channel = AsyncMock()

        with (
            patch(
                "macaw.scheduler.scheduler.grpc.aio.insecure_channel",
                return_value=mock_channel,
            ),
            patch(
                "macaw.scheduler.scheduler.STTWorkerStub",
                return_value=mock_stub,
            ),
        ):
            await scheduler.start()
            await scheduler.transcribe(_make_request())
            await scheduler.stop()

            mock_channel.close.assert_awaited_once()

    async def test_close_channel_removes_from_pool(self) -> None:
        """close_channel() removes the channel from the pool."""
        scheduler, _, _ = _make_scheduler()
        mock_channel = AsyncMock()

        with patch(
            "macaw.scheduler.scheduler.grpc.aio.insecure_channel",
            return_value=mock_channel,
        ):
            # Create a channel via pool
            scheduler._get_or_create_channel("localhost:50051")
            assert "localhost:50051" in scheduler._channels

            await scheduler.close_channel("localhost:50051")
            assert "localhost:50051" not in scheduler._channels
            mock_channel.close.assert_awaited_once()


class TestSchedulerConcurrent:
    async def test_concurrent_requests_all_complete(self) -> None:
        """Multiple concurrent requests all complete successfully."""
        worker = _make_worker()
        scheduler, _, _ = _make_scheduler(worker)

        proto_response = _make_proto_response()
        mock_stub = AsyncMock()
        mock_stub.TranscribeFile.return_value = proto_response

        with (
            patch(
                "macaw.scheduler.scheduler.grpc.aio.insecure_channel",
                return_value=AsyncMock(),
            ),
            patch(
                "macaw.scheduler.scheduler.STTWorkerStub",
                return_value=mock_stub,
            ),
        ):
            await scheduler.start()
            try:
                # Submit 10 concurrent requests
                tasks = [
                    asyncio.create_task(
                        scheduler.transcribe(_make_request(f"r{i}")),
                    )
                    for i in range(10)
                ]
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks),
                    timeout=10.0,
                )
                assert len(results) == 10
                assert all(r.text == "Ola mundo" for r in results)
            finally:
                await scheduler.stop()


class TestSchedulerGracefulShutdown:
    async def test_stop_waits_for_in_flight(self) -> None:
        """stop() waits for in-flight requests to complete."""
        worker = _make_worker()
        scheduler, _, _ = _make_scheduler(worker)

        # Slow response (simulates long transcription)
        async def slow_transcribe(*args: object, **kwargs: object) -> TranscribeFileResponse:
            await asyncio.sleep(0.2)
            return _make_proto_response()

        mock_stub = AsyncMock()
        mock_stub.TranscribeFile.side_effect = slow_transcribe

        with (
            patch(
                "macaw.scheduler.scheduler.grpc.aio.insecure_channel",
                return_value=AsyncMock(),
            ),
            patch(
                "macaw.scheduler.scheduler.STTWorkerStub",
                return_value=mock_stub,
            ),
        ):
            await scheduler.start()

            # Submit request (don't await yet)
            future = await scheduler.submit(_make_request())

            # Give dispatch loop time to pick it up
            await asyncio.sleep(0.05)

            # Stop should wait for in-flight
            await scheduler.stop()

            # Request should have completed
            assert future.done()
            assert future.result().text == "Ola mundo"


class TestSchedulerInlineFallback:
    async def test_transcribe_works_without_start(self) -> None:
        """transcribe() works in inline mode when dispatch loop is not started."""
        worker = _make_worker()
        scheduler, _, _ = _make_scheduler(worker)

        proto_response = _make_proto_response()
        mock_stub = AsyncMock()
        mock_stub.TranscribeFile.return_value = proto_response
        mock_channel = AsyncMock()

        with (
            patch(
                "macaw.scheduler.scheduler.grpc.aio.insecure_channel",
                return_value=mock_channel,
            ),
            patch(
                "macaw.scheduler.scheduler.STTWorkerStub",
                return_value=mock_stub,
            ),
        ):
            # No start() — uses inline path
            result = await scheduler.transcribe(_make_request())
            assert result.text == "Ola mundo"
            mock_channel.close.assert_awaited_once()

    async def test_inline_no_worker_raises(self) -> None:
        """Inline mode raises WorkerUnavailableError when no worker."""
        from macaw.exceptions import WorkerUnavailableError

        scheduler, _, _ = _make_scheduler(worker=None)

        with pytest.raises(WorkerUnavailableError):
            await scheduler.transcribe(_make_request())


class TestSchedulerQueueAccess:
    async def test_queue_property(self) -> None:
        scheduler, _, _ = _make_scheduler()
        assert scheduler.queue is scheduler._queue

    async def test_queue_depth_increases_on_submit(self) -> None:
        scheduler, _, _ = _make_scheduler()
        assert scheduler.queue.depth == 0
        await scheduler.submit(_make_request("r1"))
        assert scheduler.queue.depth == 1
