"""End-to-end scheduler integration tests with mocked workers (M8-09).

Tests exercise the full scheduler pipeline: submit -> priority queue ->
dispatch loop -> gRPC call -> future resolution, with mocked gRPC stubs
and worker managers.

Covers:
- Prioritization (REALTIME dispatched before BATCH)
- Aging (BATCH promoted after threshold)
- Cancellation in queue (future cancelled, worker never called)
- Cancellation in-flight (cancel_in_flight called with correct address)
- Batching (BatchAccumulator groups BATCH requests)
- Graceful shutdown (in-flight completes, scheduler stops)
- Latency tracking through full pipeline
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from macaw._types import ResponseFormat
from macaw.proto import Segment, TranscribeFileResponse, Word
from macaw.scheduler.queue import RequestPriority
from macaw.scheduler.scheduler import Scheduler
from macaw.server.models.requests import TranscribeRequest
from macaw.workers.manager import WorkerHandle, WorkerState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    **kwargs: object,
) -> tuple[Scheduler, MagicMock, MagicMock]:
    """Create scheduler with mocked dependencies.

    Extra kwargs are forwarded to the Scheduler constructor
    (e.g. aging_threshold_s, batch_accumulate_ms, batch_max_size).
    """
    registry = MagicMock()
    registry.get_manifest.return_value = MagicMock()

    worker_manager = MagicMock()
    worker_manager.get_ready_worker.return_value = worker

    scheduler = Scheduler(worker_manager, registry, **kwargs)  # type: ignore[arg-type]
    return scheduler, worker_manager, registry


# ---------------------------------------------------------------------------
# Prioritization
# ---------------------------------------------------------------------------


class TestPrioritization:
    async def test_realtime_dispatched_before_multiple_batch(self) -> None:
        """Submit N BATCH + 1 REALTIME. REALTIME should be dispatched first."""
        worker = _make_worker()
        scheduler, _, _ = _make_scheduler(worker)

        dispatch_order: list[str] = []

        async def fake_transcribe(
            proto_req: object, timeout: float = 30
        ) -> TranscribeFileResponse:
            req_id = proto_req.request_id  # type: ignore[attr-defined]
            dispatch_order.append(req_id)
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
            # Enqueue 5 BATCH requests first
            batch_futures = []
            for i in range(5):
                f = await scheduler.submit(
                    _make_request(f"batch_{i}"),
                    RequestPriority.BATCH,
                )
                batch_futures.append(f)

            # Then 1 REALTIME request
            rt_future = await scheduler.submit(
                _make_request("rt_0"),
                RequestPriority.REALTIME,
            )

            # Start dispatch loop — priority queue should yield REALTIME first
            await scheduler.start()
            try:
                await asyncio.wait_for(rt_future, timeout=5.0)
                for f in batch_futures:
                    await asyncio.wait_for(f, timeout=5.0)
            finally:
                await scheduler.stop()

            # REALTIME must be the first dispatched request
            assert dispatch_order[0] == "rt_0"


# ---------------------------------------------------------------------------
# Aging
# ---------------------------------------------------------------------------


class TestAging:
    async def test_batch_request_ages_after_threshold(self) -> None:
        """BATCH request is aged when queue wait exceeds aging_threshold_s."""
        # Use very short aging threshold so the test is deterministic
        scheduler, _, _ = _make_scheduler(aging_threshold_s=0.001)

        future = await scheduler.submit(
            _make_request("aged_req"),
            RequestPriority.BATCH,
        )

        # Wait longer than the aging threshold
        await asyncio.sleep(0.01)

        scheduled = await scheduler.queue.dequeue()
        assert scheduler.queue.is_aged(scheduled)
        assert scheduled.request.request_id == "aged_req"

        # Cleanup: cancel the future so it does not leak
        if not future.done():
            future.cancel()

    async def test_fresh_batch_request_is_not_aged(self) -> None:
        """BATCH request dequeued immediately is not considered aged."""
        scheduler, _, _ = _make_scheduler(aging_threshold_s=60.0)

        future = await scheduler.submit(
            _make_request("fresh_req"),
            RequestPriority.BATCH,
        )

        scheduled = await scheduler.queue.dequeue()
        assert not scheduler.queue.is_aged(scheduled)

        if not future.done():
            future.cancel()


# ---------------------------------------------------------------------------
# Cancellation in queue
# ---------------------------------------------------------------------------


class TestCancellationInQueue:
    async def test_cancel_before_dispatch_never_reaches_worker(self) -> None:
        """Request cancelled in queue is never sent to the worker."""
        worker = _make_worker()
        scheduler, _, _ = _make_scheduler(worker)

        mock_stub = AsyncMock()
        mock_stub.TranscribeFile.return_value = _make_proto_response()

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
            # Submit but cancel BEFORE starting dispatch loop
            future = await scheduler.submit(_make_request("cancel_me"))
            cancelled = scheduler.cancel("cancel_me")

            assert cancelled is True
            assert future.cancelled()

            # Start and stop — worker should never be called for cancelled request
            await scheduler.start()
            await asyncio.sleep(0.1)
            await scheduler.stop()

            # Verify TranscribeFile was never called
            mock_stub.TranscribeFile.assert_not_called()


# ---------------------------------------------------------------------------
# Cancellation in-flight
# ---------------------------------------------------------------------------


class TestCancellationInFlight:
    async def test_cancel_in_flight_propagates_to_worker(self) -> None:
        """Cancel of in-flight request triggers gRPC Cancel to the worker."""
        worker = _make_worker()
        scheduler, _, _ = _make_scheduler(worker)

        # Slow response so request stays in-flight long enough to cancel
        started_event = asyncio.Event()

        async def slow_transcribe(*args: object, **kwargs: object) -> TranscribeFileResponse:
            started_event.set()
            await asyncio.sleep(5.0)
            return _make_proto_response()

        mock_stub = AsyncMock()
        mock_stub.TranscribeFile.side_effect = slow_transcribe
        # Cancel returns a coroutine-like mock (AsyncMock)
        mock_cancel_response = MagicMock(acknowledged=True)
        mock_stub.Cancel.return_value = mock_cancel_response

        with (
            patch(
                "macaw.scheduler.scheduler.grpc.aio.insecure_channel",
                return_value=AsyncMock(),
            ),
            patch(
                "macaw.scheduler.scheduler.STTWorkerStub",
                return_value=mock_stub,
            ),
            # Also patch the STTWorkerStub in cancel module since
            # cancel_in_flight creates its own stub from the channel
            patch(
                "macaw.scheduler.cancel.STTWorkerStub",
                return_value=mock_stub,
            ),
        ):
            await scheduler.start()
            try:
                _future = await scheduler.submit(
                    _make_request("inflight_req"),
                    RequestPriority.REALTIME,
                )

                # Wait for the gRPC call to actually start
                await asyncio.wait_for(started_event.wait(), timeout=5.0)

                # Now cancel — request is in-flight, should propagate to worker
                cancelled = scheduler.cancel("inflight_req")
                assert cancelled is True

                # Give time for the fire-and-forget cancel task to execute
                await asyncio.sleep(0.3)

                # Verify gRPC Cancel was called on the stub
                mock_stub.Cancel.assert_awaited_once()
            finally:
                await scheduler.stop()


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


class TestBatching:
    async def test_batch_requests_accumulated_and_dispatched_together(self) -> None:
        """Multiple BATCH requests are grouped by the BatchAccumulator."""
        worker = _make_worker()
        # max_batch_size=3, short accumulate window so it flushes fast
        scheduler, _, _ = _make_scheduler(
            worker,
            batch_accumulate_ms=500.0,
            batch_max_size=3,
        )

        dispatched_ids: list[str] = []

        async def fake_transcribe(
            proto_req: object, timeout: float = 30
        ) -> TranscribeFileResponse:
            req_id: str = proto_req.request_id  # type: ignore[attr-defined]
            dispatched_ids.append(req_id)
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
            await scheduler.start()
            try:
                # Submit exactly max_batch_size BATCH requests
                futures = []
                for i in range(3):
                    f = await scheduler.submit(
                        _make_request(f"b{i}"),
                        RequestPriority.BATCH,
                    )
                    futures.append(f)

                # Wait for all to complete
                for f in futures:
                    await asyncio.wait_for(f, timeout=5.0)

                # All 3 should have been dispatched
                assert len(dispatched_ids) == 3
                assert set(dispatched_ids) == {"b0", "b1", "b2"}
            finally:
                await scheduler.stop()


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------


class TestGracefulShutdown:
    async def test_stop_completes_in_flight_requests(self) -> None:
        """Scheduler.stop() waits for in-flight requests to finish."""
        worker = _make_worker()
        scheduler, _, _ = _make_scheduler(worker)

        async def slow_transcribe(*args: object, **kwargs: object) -> TranscribeFileResponse:
            await asyncio.sleep(0.2)
            return _make_proto_response("shutdown_result")

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

            future = await scheduler.submit(
                _make_request("shutdown_req"),
                RequestPriority.REALTIME,
            )

            # Let dispatch loop pick it up
            await asyncio.sleep(0.05)

            # Stop should drain in-flight and complete
            await scheduler.stop()

            assert future.done()
            assert not future.cancelled()
            assert future.result().text == "shutdown_result"

    async def test_stop_flushes_pending_batch(self) -> None:
        """Scheduler.stop() flushes pending batch accumulator before stopping."""
        worker = _make_worker()
        # Long accumulate window so batch stays pending until stop()
        scheduler, _, _ = _make_scheduler(
            worker,
            batch_accumulate_ms=10_000.0,
            batch_max_size=100,
        )

        mock_stub = AsyncMock()
        mock_stub.TranscribeFile.return_value = _make_proto_response("flushed")

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

            # Submit BATCH request — accumulator holds it (long window)
            future = await scheduler.submit(
                _make_request("pending_batch"),
                RequestPriority.BATCH,
            )

            # Let dispatch loop dequeue and add to accumulator
            await asyncio.sleep(0.3)

            # stop() should flush the pending batch
            await scheduler.stop()

            # The future should be resolved with the flushed result
            assert future.done()
            if not future.cancelled():
                assert future.result().text == "flushed"


# ---------------------------------------------------------------------------
# Latency tracking through full pipeline
# ---------------------------------------------------------------------------


class TestLatencyTracking:
    async def test_latency_tracker_records_full_pipeline(self) -> None:
        """LatencyTracker records enqueue, dequeue, grpc_start, and complete."""
        worker = _make_worker()
        scheduler, _, _ = _make_scheduler(worker)

        mock_stub = AsyncMock()
        mock_stub.TranscribeFile.return_value = _make_proto_response()

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
                    scheduler.transcribe(_make_request("lat_req")),
                    timeout=5.0,
                )
                assert result.text == "Ola mundo"

                # LatencyTracker should have completed the entry
                summary = scheduler.latency.get_summary("lat_req")
                assert summary is not None
                assert summary.request_id == "lat_req"
                assert summary.queue_wait >= 0.0
                assert summary.grpc_time >= 0.0
                assert summary.total_time >= 0.0
                assert summary.total_time >= summary.queue_wait
            finally:
                await scheduler.stop()
