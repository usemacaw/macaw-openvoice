"""Tests for Scheduler contention scenarios (M8-09).

Tests cover: batch queued while streaming occupies workers, batch requests
don't starve under continuous realtime, cancel during contention, and
batch accumulation under load.
"""

from __future__ import annotations

import asyncio
from typing import Any
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


def _patch_metrics_and_grpc(mock_stub: AsyncMock) -> tuple[Any, ...]:
    """Return a tuple of context managers that patch all metrics and gRPC.

    Prevents Prometheus side effects and wires the mock stub.
    Returns the patches as a tuple to be used with ``with`` statement.
    """
    return (
        patch("macaw.scheduler.scheduler.scheduler_queue_depth"),
        patch("macaw.scheduler.scheduler.scheduler_queue_wait_seconds"),
        patch("macaw.scheduler.scheduler.scheduler_requests_total"),
        patch("macaw.scheduler.scheduler.scheduler_grpc_duration_seconds"),
        patch("macaw.scheduler.scheduler.scheduler_batch_size"),
        patch("macaw.scheduler.scheduler.scheduler_aging_promotions_total"),
        patch(
            "macaw.scheduler.scheduler.grpc.aio.insecure_channel",
            return_value=AsyncMock(),
        ),
        patch(
            "macaw.scheduler.scheduler.STTWorkerStub",
            return_value=mock_stub,
        ),
    )


# ---------------------------------------------------------------------------
# Contention tests
# ---------------------------------------------------------------------------


class TestBatchQueuedWhileRealtimeOccupiesWorker:
    """When a worker is busy with a REALTIME request, BATCH requests wait."""

    async def test_batch_waits_until_realtime_completes(self) -> None:
        """BATCH request is dispatched only after slow REALTIME finishes."""
        worker = _make_worker()
        scheduler, _, _ = _make_scheduler(worker)

        dispatch_order: list[str] = []
        realtime_started = asyncio.Event()

        async def _slow_transcribe(
            proto_req: object,
            timeout: float = 30,
        ) -> TranscribeFileResponse:
            req_id: str = proto_req.request_id  # type: ignore[attr-defined]
            dispatch_order.append(req_id)

            if req_id == "rt_slow":
                realtime_started.set()
                await asyncio.sleep(0.3)

            return _make_proto_response(text=req_id)

        mock_stub = AsyncMock()
        mock_stub.TranscribeFile.side_effect = _slow_transcribe

        with _apply_patches(_patch_metrics_and_grpc(mock_stub)):
            await scheduler.start()
            try:
                # Submit REALTIME first (slow) then BATCH
                rt_future = await scheduler.submit(
                    _make_request("rt_slow"),
                    RequestPriority.REALTIME,
                )

                # Wait until realtime starts processing
                await asyncio.wait_for(realtime_started.wait(), timeout=2.0)

                # Submit batch — it should queue behind realtime
                batch_future = await scheduler.submit(
                    _make_request("batch_1"),
                    RequestPriority.BATCH,
                )

                rt_result = await asyncio.wait_for(rt_future, timeout=5.0)
                batch_result = await asyncio.wait_for(batch_future, timeout=5.0)

                assert rt_result.text == "rt_slow"
                assert batch_result.text == "batch_1"
                assert dispatch_order.index("rt_slow") < dispatch_order.index("batch_1")
            finally:
                await scheduler.stop()


class TestBatchRequestsDoNotStarve:
    """BATCH requests are processed even with continuous REALTIME submissions."""

    async def test_batch_processed_when_worker_becomes_free(self) -> None:
        """After REALTIME requests finish, pending BATCH requests complete."""
        worker = _make_worker()
        scheduler, _, _ = _make_scheduler(worker)

        completed: list[str] = []

        async def _track_transcribe(
            proto_req: object,
            timeout: float = 30,
        ) -> TranscribeFileResponse:
            req_id: str = proto_req.request_id  # type: ignore[attr-defined]
            if req_id.startswith("rt_"):
                await asyncio.sleep(0.05)
            completed.append(req_id)
            return _make_proto_response(text=req_id)

        mock_stub = AsyncMock()
        mock_stub.TranscribeFile.side_effect = _track_transcribe

        with _apply_patches(_patch_metrics_and_grpc(mock_stub)):
            await scheduler.start()
            try:
                # Submit 3 REALTIME and 2 BATCH requests
                rt_futures = [
                    await scheduler.submit(
                        _make_request(f"rt_{i}"),
                        RequestPriority.REALTIME,
                    )
                    for i in range(3)
                ]
                batch_futures = [
                    await scheduler.submit(
                        _make_request(f"batch_{i}"),
                        RequestPriority.BATCH,
                    )
                    for i in range(2)
                ]

                # Wait for all to complete
                all_futures = rt_futures + batch_futures
                for f in all_futures:
                    await asyncio.wait_for(f, timeout=5.0)

                # All BATCH requests should have completed
                batch_ids = {req_id for req_id in completed if req_id.startswith("batch_")}
                assert batch_ids == {"batch_0", "batch_1"}
            finally:
                await scheduler.stop()


class TestCancelDuringContention:
    """Cancel a queued BATCH request while a REALTIME request is executing."""

    async def test_cancel_queued_batch_during_realtime(self) -> None:
        """Cancelling a queued BATCH request resolves its future immediately."""
        worker = _make_worker()
        scheduler, _, _ = _make_scheduler(worker)

        realtime_started = asyncio.Event()

        async def _slow_realtime(
            proto_req: object,
            timeout: float = 30,
        ) -> TranscribeFileResponse:
            req_id: str = proto_req.request_id  # type: ignore[attr-defined]
            if req_id == "rt_blocking":
                realtime_started.set()
                await asyncio.sleep(0.3)
            return _make_proto_response(text=req_id)

        mock_stub = AsyncMock()
        mock_stub.TranscribeFile.side_effect = _slow_realtime

        with _apply_patches(_patch_metrics_and_grpc(mock_stub)):
            await scheduler.start()
            try:
                # Submit REALTIME (slow) then BATCH
                rt_future = await scheduler.submit(
                    _make_request("rt_blocking"),
                    RequestPriority.REALTIME,
                )
                await asyncio.wait_for(realtime_started.wait(), timeout=2.0)

                batch_future = await scheduler.submit(
                    _make_request("batch_cancel_me"),
                    RequestPriority.BATCH,
                )

                # Cancel the batch request while it's queued
                cancelled = scheduler.cancel("batch_cancel_me")
                assert cancelled is True

                # The future should be resolved (cancelled)
                assert batch_future.done()
                assert batch_future.cancelled()

                # REALTIME should still complete normally
                rt_result = await asyncio.wait_for(rt_future, timeout=5.0)
                assert rt_result.text == "rt_blocking"
            finally:
                await scheduler.stop()


class TestBatchAccumulationUnderLoad:
    """Multiple rapid BATCH requests are accumulated into a single batch."""

    async def test_rapid_batch_requests_are_batched(self) -> None:
        """Batch accumulator groups rapid BATCH requests (batch_size > 1)."""
        worker = _make_worker()
        # Short accumulate window so the test runs fast, max_batch_size=4
        scheduler, _, _ = _make_scheduler(
            worker,
            batch_accumulate_ms=100.0,
            batch_max_size=4,
        )

        mock_stub = AsyncMock()
        mock_stub.TranscribeFile.return_value = _make_proto_response()

        observed_batch_sizes: list[int] = []
        original_dispatch_batch = scheduler._dispatch_batch

        async def _spy_dispatch_batch(batch: Any) -> None:
            observed_batch_sizes.append(len(batch))
            await original_dispatch_batch(batch)

        scheduler._dispatch_batch = _spy_dispatch_batch  # type: ignore[method-assign]
        scheduler._batch_accumulator._on_flush = _spy_dispatch_batch

        with _apply_patches(_patch_metrics_and_grpc(mock_stub)):
            await scheduler.start()
            try:
                # Submit 4 BATCH requests rapidly (should hit max_batch_size)
                futures = []
                for i in range(4):
                    f = await scheduler.submit(
                        _make_request(f"batch_{i}"),
                        RequestPriority.BATCH,
                    )
                    futures.append(f)

                # Wait for all to complete
                for f in futures:
                    await asyncio.wait_for(f, timeout=5.0)

                # At least one batch should have size > 1
                assert any(size > 1 for size in observed_batch_sizes), (
                    f"Expected at least one batch with size > 1, got: {observed_batch_sizes}"
                )
            finally:
                await scheduler.stop()


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


class _MultiPatch:
    """Applies multiple context managers as a single context manager."""

    def __init__(self, patches: tuple[Any, ...]) -> None:
        self._patches = patches
        self._stack: list[object] = []

    def __enter__(self) -> _MultiPatch:
        for p in self._patches:
            self._stack.append(p.__enter__())
        return self

    def __exit__(self, *exc: object) -> None:
        for p in reversed(self._patches):
            p.__exit__(None, None, None)


def _apply_patches(patches: tuple[Any, ...]) -> _MultiPatch:
    """Wrap a tuple of patch context managers into a single context manager."""
    return _MultiPatch(patches)
