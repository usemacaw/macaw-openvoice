"""Tests for M8-05: BatchAccumulator.

Tests cover: accumulation by time, accumulation by count, flush with
cancelled requests, flush manual, mixed models, timer reset, empty flush,
concurrent add, integration with Scheduler dispatch loop.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from macaw._types import ResponseFormat
from macaw.scheduler.batching import BatchAccumulator
from macaw.scheduler.queue import RequestPriority, ScheduledRequest
from macaw.server.models.requests import TranscribeRequest

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


def _make_scheduled(
    request_id: str = "req_1",
    model: str = "faster-whisper-tiny",
    priority: RequestPriority = RequestPriority.BATCH,
) -> ScheduledRequest:
    return ScheduledRequest(
        request=_make_request(request_id, model),
        priority=priority,
    )


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


class TestBatchAccumulatorInit:
    def test_default_params(self) -> None:
        """BatchAccumulator accepts default params."""
        acc = BatchAccumulator(on_flush=AsyncMock())
        assert acc.pending_count == 0
        assert acc.model_name is None

    def test_custom_params(self) -> None:
        """BatchAccumulator accepts custom accumulate_ms and max_batch_size."""
        acc = BatchAccumulator(
            accumulate_ms=100.0,
            max_batch_size=4,
            on_flush=AsyncMock(),
        )
        assert acc.pending_count == 0

    def test_invalid_accumulate_ms(self) -> None:
        """accumulate_ms <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="accumulate_ms"):
            BatchAccumulator(accumulate_ms=0, on_flush=AsyncMock())

    def test_invalid_max_batch_size(self) -> None:
        """max_batch_size < 1 raises ValueError."""
        with pytest.raises(ValueError, match="max_batch_size"):
            BatchAccumulator(max_batch_size=0, on_flush=AsyncMock())


# ---------------------------------------------------------------------------
# Add and flush tests
# ---------------------------------------------------------------------------


class TestAddAndFlush:
    async def test_add_increments_pending(self) -> None:
        """add() increments pending_count."""
        acc = BatchAccumulator(
            accumulate_ms=500,  # long timer to avoid auto-flush
            on_flush=AsyncMock(),
        )
        acc.add(_make_scheduled("req_1"))
        assert acc.pending_count == 1
        assert acc.model_name == "faster-whisper-tiny"

    async def test_flush_returns_added_requests(self) -> None:
        """flush() returns all added requests and resets."""
        acc = BatchAccumulator(
            accumulate_ms=500,
            on_flush=AsyncMock(),
        )
        s1 = _make_scheduled("req_1")
        s2 = _make_scheduled("req_2")
        acc.add(s1)
        acc.add(s2)

        batch = acc.flush()

        assert len(batch) == 2
        assert batch[0] is s1
        assert batch[1] is s2
        assert acc.pending_count == 0
        assert acc.model_name is None

    async def test_flush_empty_returns_empty_list(self) -> None:
        """flush() on empty accumulator returns empty list."""
        acc = BatchAccumulator(
            accumulate_ms=500,
            on_flush=AsyncMock(),
        )
        batch = acc.flush()
        assert batch == []

    async def test_flush_removes_cancelled_requests(self) -> None:
        """flush() filters out requests whose cancel_event is set."""
        acc = BatchAccumulator(
            accumulate_ms=500,
            on_flush=AsyncMock(),
        )
        s1 = _make_scheduled("req_1")
        s2 = _make_scheduled("req_2")
        s3 = _make_scheduled("req_3")

        acc.add(s1)
        acc.add(s2)
        acc.add(s3)

        # Cancel req_2
        s2.cancel_event.set()

        batch = acc.flush()
        assert len(batch) == 2
        assert batch[0] is s1
        assert batch[1] is s3

    async def test_flush_all_cancelled_returns_empty(self) -> None:
        """flush() returns empty list when all requests are cancelled."""
        acc = BatchAccumulator(
            accumulate_ms=500,
            on_flush=AsyncMock(),
        )
        s1 = _make_scheduled("req_1")
        s2 = _make_scheduled("req_2")
        s1.cancel_event.set()
        s2.cancel_event.set()

        acc.add(s1)
        acc.add(s2)

        batch = acc.flush()
        assert batch == []


# ---------------------------------------------------------------------------
# Timer auto-flush tests
# ---------------------------------------------------------------------------


class TestTimerFlush:
    async def test_timer_auto_flush(self) -> None:
        """Requests are auto-flushed after accumulate_ms."""
        flushed: list[Any] = []

        async def on_flush(batch: list[ScheduledRequest]) -> None:
            flushed.append(batch)

        acc = BatchAccumulator(
            accumulate_ms=30,  # 30ms
            max_batch_size=100,  # won't trigger by count
            on_flush=on_flush,
        )
        acc.add(_make_scheduled("req_1"))
        acc.add(_make_scheduled("req_2"))

        # Wait for timer to fire
        await asyncio.sleep(0.06)

        assert len(flushed) == 1
        assert len(flushed[0]) == 2

    async def test_timer_resets_on_add(self) -> None:
        """Adding a request when buffer is empty starts a fresh timer."""
        flushed: list[Any] = []

        async def on_flush(batch: list[ScheduledRequest]) -> None:
            flushed.append(batch)

        acc = BatchAccumulator(
            accumulate_ms=50,
            max_batch_size=100,
            on_flush=on_flush,
        )

        # First batch
        acc.add(_make_scheduled("req_1"))
        await asyncio.sleep(0.07)  # timer fires
        assert len(flushed) == 1

        # Second batch — fresh timer
        acc.add(_make_scheduled("req_2"))
        await asyncio.sleep(0.07)  # timer fires again
        assert len(flushed) == 2
        assert len(flushed[1]) == 1

    async def test_manual_flush_cancels_timer(self) -> None:
        """Manual flush() cancels the pending timer."""
        flushed: list[Any] = []

        async def on_flush(batch: list[ScheduledRequest]) -> None:
            flushed.append(batch)

        acc = BatchAccumulator(
            accumulate_ms=50,
            max_batch_size=100,
            on_flush=on_flush,
        )
        acc.add(_make_scheduled("req_1"))

        # Flush manually before timer
        batch = acc.flush()
        assert len(batch) == 1

        # Wait to ensure timer doesn't fire
        await asyncio.sleep(0.08)
        assert len(flushed) == 0  # on_flush NOT called (we flushed manually)


# ---------------------------------------------------------------------------
# Max batch size auto-flush tests
# ---------------------------------------------------------------------------


class TestMaxBatchSizeFlush:
    async def test_max_batch_size_triggers_flush(self) -> None:
        """Reaching max_batch_size triggers immediate flush."""
        flushed: list[Any] = []

        async def on_flush(batch: list[ScheduledRequest]) -> None:
            flushed.append(batch)

        acc = BatchAccumulator(
            accumulate_ms=5000,  # long timer — won't trigger
            max_batch_size=3,
            on_flush=on_flush,
        )
        acc.add(_make_scheduled("req_1"))
        acc.add(_make_scheduled("req_2"))
        assert len(flushed) == 0  # not yet

        acc.add(_make_scheduled("req_3"))  # hits max_batch_size

        # Give asyncio.create_task a chance to run
        await asyncio.sleep(0.01)

        assert len(flushed) == 1
        assert len(flushed[0]) == 3

    async def test_max_batch_size_one_flushes_every_request(self) -> None:
        """max_batch_size=1 flushes on every add()."""
        flushed: list[Any] = []

        async def on_flush(batch: list[ScheduledRequest]) -> None:
            flushed.append(batch)

        acc = BatchAccumulator(
            accumulate_ms=5000,
            max_batch_size=1,
            on_flush=on_flush,
        )
        acc.add(_make_scheduled("req_1"))
        await asyncio.sleep(0.01)

        acc.add(_make_scheduled("req_2"))
        await asyncio.sleep(0.01)

        assert len(flushed) == 2


# ---------------------------------------------------------------------------
# Model name validation tests
# ---------------------------------------------------------------------------


class TestModelValidation:
    async def test_same_model_accumulates(self) -> None:
        """Requests for the same model accumulate in the buffer."""
        acc = BatchAccumulator(
            accumulate_ms=500,
            on_flush=AsyncMock(),
        )
        acc.add(_make_scheduled("req_1", model="model-a"))
        acc.add(_make_scheduled("req_2", model="model-a"))
        assert acc.pending_count == 2
        assert acc.model_name == "model-a"

    async def test_different_model_flushes_previous_batch(self) -> None:
        """Adding a request for a different model flushes the current batch."""
        flushed: list[Any] = []

        async def on_flush(batch: list[ScheduledRequest]) -> None:
            flushed.append(batch)

        acc = BatchAccumulator(
            accumulate_ms=5000,
            max_batch_size=100,
            on_flush=on_flush,
        )
        acc.add(_make_scheduled("req_1", model="model-a"))
        acc.add(_make_scheduled("req_2", model="model-a"))

        # Different model — should flush model-a batch first
        acc.add(_make_scheduled("req_3", model="model-b"))
        await asyncio.sleep(0.01)

        # First flush: model-a (2 requests)
        assert len(flushed) == 1
        assert len(flushed[0]) == 2
        assert flushed[0][0].request.model_name == "model-a"

        # model-b request is now in the buffer
        assert acc.pending_count == 1
        assert acc.model_name == "model-b"


# ---------------------------------------------------------------------------
# Integration with Scheduler tests
# ---------------------------------------------------------------------------


class TestSchedulerIntegration:
    async def test_realtime_bypasses_accumulator(self) -> None:
        """REALTIME requests are dispatched directly, not accumulated."""
        from macaw.scheduler.scheduler import Scheduler

        mock_manager = MagicMock()
        mock_registry = MagicMock()
        mock_registry.get_manifest.return_value = MagicMock()
        mock_manager.get_ready_worker.return_value = None  # no worker

        scheduler = Scheduler(mock_manager, mock_registry)
        await scheduler.start()
        try:
            await scheduler.submit(_make_request("req_rt"), RequestPriority.REALTIME)
            await asyncio.sleep(0.1)

            # REALTIME request should NOT be in the batch accumulator
            assert scheduler._batch_accumulator.pending_count == 0
        finally:
            await scheduler.stop()

    async def test_batch_goes_through_accumulator(self) -> None:
        """BATCH requests go through the BatchAccumulator."""
        from macaw.scheduler.scheduler import Scheduler

        mock_manager = MagicMock()
        mock_registry = MagicMock()
        mock_registry.get_manifest.return_value = MagicMock()
        mock_manager.get_ready_worker.return_value = None

        # Use very long accumulate_ms so timer doesn't fire during test
        scheduler = Scheduler(
            mock_manager,
            mock_registry,
            batch_accumulate_ms=5000,
        )
        await scheduler.start()
        try:
            await scheduler.submit(_make_request("req_batch_1"), RequestPriority.BATCH)
            # Give dispatch loop time to dequeue and add to accumulator
            await asyncio.sleep(0.1)

            # BATCH request should be in the accumulator
            assert scheduler._batch_accumulator.pending_count == 1
        finally:
            await scheduler.stop()

    async def test_stop_flushes_pending_batch(self) -> None:
        """Scheduler.stop() flushes any pending batch requests."""
        from unittest.mock import patch

        from macaw.scheduler.scheduler import Scheduler

        mock_manager = MagicMock()
        mock_registry = MagicMock()
        mock_registry.get_manifest.return_value = MagicMock()
        mock_manager.get_ready_worker.return_value = None

        scheduler = Scheduler(
            mock_manager,
            mock_registry,
            batch_accumulate_ms=5000,  # won't fire during test
        )
        await scheduler.start()

        await scheduler.submit(_make_request("req_pending"), RequestPriority.BATCH)
        await asyncio.sleep(0.1)
        assert scheduler._batch_accumulator.pending_count == 1

        # Patch _dispatch_batch to capture calls
        with patch.object(
            scheduler,
            "_dispatch_batch",
            new_callable=AsyncMock,
        ) as mock_dispatch_batch:
            await scheduler.stop()

            # _dispatch_batch should have been called with pending batch
            mock_dispatch_batch.assert_called_once()
            batch_arg = mock_dispatch_batch.call_args[0][0]
            assert len(batch_arg) == 1


# ---------------------------------------------------------------------------
# Concurrent add tests
# ---------------------------------------------------------------------------


class TestConcurrentAdd:
    async def test_multiple_adds_before_flush(self) -> None:
        """Multiple add() calls accumulate correctly."""
        acc = BatchAccumulator(
            accumulate_ms=500,
            on_flush=AsyncMock(),
        )
        for i in range(5):
            acc.add(_make_scheduled(f"req_{i}"))

        assert acc.pending_count == 5
        batch = acc.flush()
        assert len(batch) == 5

    async def test_add_after_flush_starts_fresh(self) -> None:
        """add() after flush() starts a new batch."""
        acc = BatchAccumulator(
            accumulate_ms=500,
            on_flush=AsyncMock(),
        )
        acc.add(_make_scheduled("req_1"))
        acc.flush()

        acc.add(_make_scheduled("req_2"))
        assert acc.pending_count == 1
        assert acc.model_name == "faster-whisper-tiny"

        batch = acc.flush()
        assert len(batch) == 1
        assert batch[0].request.request_id == "req_2"


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    async def test_double_flush_returns_empty(self) -> None:
        """Second flush() after first returns empty list."""
        acc = BatchAccumulator(
            accumulate_ms=500,
            on_flush=AsyncMock(),
        )
        acc.add(_make_scheduled("req_1"))
        batch1 = acc.flush()
        batch2 = acc.flush()
        assert len(batch1) == 1
        assert len(batch2) == 0

    async def test_timer_with_all_cancelled_does_not_call_on_flush(self) -> None:
        """Timer flush with all-cancelled batch does not call on_flush."""
        flushed: list[Any] = []

        async def on_flush(batch: list[ScheduledRequest]) -> None:
            flushed.append(batch)

        acc = BatchAccumulator(
            accumulate_ms=30,
            max_batch_size=100,
            on_flush=on_flush,
        )
        s1 = _make_scheduled("req_1")
        s1.cancel_event.set()
        acc.add(s1)

        # Wait for timer to fire
        await asyncio.sleep(0.06)

        # on_flush should NOT be called (all requests cancelled = empty batch)
        assert len(flushed) == 0


# ---------------------------------------------------------------------------
# _on_flush_done callback tests
# ---------------------------------------------------------------------------


class TestOnFlushDone:
    async def test_flush_done_removes_task_from_set(self) -> None:
        """_on_flush_done removes completed task from _flush_tasks."""
        flushed: list[Any] = []

        async def on_flush(batch: list[ScheduledRequest]) -> None:
            flushed.append(batch)

        acc = BatchAccumulator(
            accumulate_ms=30,
            max_batch_size=100,
            on_flush=on_flush,
        )
        acc.add(_make_scheduled("req_1"))

        # Wait for timer to fire and task to complete
        await asyncio.sleep(0.06)

        assert len(flushed) == 1
        # All flush tasks should have been cleaned up
        assert len(acc._flush_tasks) == 0

    async def test_flush_done_logs_exception(self) -> None:
        """_on_flush_done logs exceptions from flush tasks without raising."""

        async def failing_flush(batch: list[ScheduledRequest]) -> None:
            msg = "flush failed"
            raise RuntimeError(msg)

        acc = BatchAccumulator(
            accumulate_ms=30,
            max_batch_size=100,
            on_flush=failing_flush,
        )
        acc.add(_make_scheduled("req_1"))

        # Wait for timer to fire; exception should be logged, not propagated
        await asyncio.sleep(0.06)

        # Task should still be cleaned up despite the exception
        assert len(acc._flush_tasks) == 0

    async def test_flush_done_handles_cancelled_task(self) -> None:
        """_on_flush_done handles cancelled flush tasks gracefully."""
        never_completes: asyncio.Future[None] = asyncio.get_running_loop().create_future()

        async def slow_flush(batch: list[ScheduledRequest]) -> None:
            await never_completes

        acc = BatchAccumulator(
            accumulate_ms=30,
            max_batch_size=100,
            on_flush=slow_flush,
        )
        acc.add(_make_scheduled("req_1"))

        # Wait for timer to fire and task to be created
        await asyncio.sleep(0.06)

        # Cancel all pending flush tasks
        for task in list(acc._flush_tasks):
            task.cancel()

        # Give cancellation a chance to propagate
        await asyncio.sleep(0.01)

        # Should not raise; tasks cleaned up
        assert len(acc._flush_tasks) == 0
