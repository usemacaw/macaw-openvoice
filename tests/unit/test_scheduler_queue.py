"""Tests for PriorityQueue (M8-01).

Tests cover: submit, dequeue by priority, FIFO within level, cancel,
aging, empty queue, concurrent submit/dequeue, future resolution.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from macaw._types import ResponseFormat
from macaw.scheduler.queue import RequestPriority, ScheduledRequest, SchedulerQueue
from macaw.server.models.requests import TranscribeRequest


def _make_request(request_id: str = "req_1", model: str = "test-model") -> TranscribeRequest:
    """Helper to create a TranscribeRequest for testing."""
    return TranscribeRequest(
        request_id=request_id,
        model_name=model,
        audio_data=b"\x00" * 3200,
        language="pt",
        response_format=ResponseFormat.JSON,
    )


class TestRequestPriority:
    def test_realtime_has_higher_priority(self) -> None:
        assert RequestPriority.REALTIME < RequestPriority.BATCH

    def test_realtime_value_is_zero(self) -> None:
        assert RequestPriority.REALTIME.value == 0

    def test_batch_value_is_one(self) -> None:
        assert RequestPriority.BATCH.value == 1


class TestScheduledRequest:
    def test_lt_priority_ordering(self) -> None:
        now = time.monotonic()
        rt = ScheduledRequest(
            request=_make_request("a"), priority=RequestPriority.REALTIME, enqueued_at=now
        )
        batch = ScheduledRequest(
            request=_make_request("b"), priority=RequestPriority.BATCH, enqueued_at=now
        )
        assert rt < batch

    def test_lt_fifo_within_same_priority(self) -> None:
        earlier = ScheduledRequest(
            request=_make_request("a"),
            priority=RequestPriority.BATCH,
            enqueued_at=1000.0,
        )
        later = ScheduledRequest(
            request=_make_request("b"),
            priority=RequestPriority.BATCH,
            enqueued_at=1001.0,
        )
        assert earlier < later

    def test_cancel_event_not_set_by_default(self) -> None:
        sr = ScheduledRequest(request=_make_request(), priority=RequestPriority.BATCH)
        assert not sr.cancel_event.is_set()


class TestSchedulerQueueSubmit:
    async def test_submit_returns_future(self) -> None:
        queue = SchedulerQueue()
        future = await queue.submit(_make_request(), RequestPriority.BATCH)
        assert isinstance(future, asyncio.Future)
        assert not future.done()

    async def test_submit_increases_depth(self) -> None:
        queue = SchedulerQueue()
        assert queue.depth == 0
        await queue.submit(_make_request("r1"), RequestPriority.BATCH)
        assert queue.depth == 1
        await queue.submit(_make_request("r2"), RequestPriority.BATCH)
        assert queue.depth == 2

    async def test_submit_default_priority_is_batch(self) -> None:
        queue = SchedulerQueue()
        await queue.submit(_make_request("r1"))
        counts = queue.depth_by_priority
        assert counts["BATCH"] == 1
        assert counts["REALTIME"] == 0

    async def test_submit_realtime_priority(self) -> None:
        queue = SchedulerQueue()
        await queue.submit(_make_request("r1"), RequestPriority.REALTIME)
        counts = queue.depth_by_priority
        assert counts["REALTIME"] == 1
        assert counts["BATCH"] == 0


class TestSchedulerQueueDequeue:
    async def test_dequeue_returns_scheduled_request(self) -> None:
        queue = SchedulerQueue()
        await queue.submit(_make_request("r1"), RequestPriority.BATCH)
        scheduled = await queue.dequeue()
        assert isinstance(scheduled, ScheduledRequest)
        assert scheduled.request.request_id == "r1"

    async def test_dequeue_removes_from_pending(self) -> None:
        queue = SchedulerQueue()
        await queue.submit(_make_request("r1"), RequestPriority.BATCH)
        assert queue.depth == 1
        await queue.dequeue()
        assert queue.depth == 0

    async def test_dequeue_realtime_before_batch(self) -> None:
        queue = SchedulerQueue()
        # Submit batch first, then realtime
        await queue.submit(_make_request("batch_1"), RequestPriority.BATCH)
        await queue.submit(_make_request("rt_1"), RequestPriority.REALTIME)

        first = await queue.dequeue()
        assert first.request.request_id == "rt_1"
        assert first.priority == RequestPriority.REALTIME

        second = await queue.dequeue()
        assert second.request.request_id == "batch_1"
        assert second.priority == RequestPriority.BATCH

    async def test_dequeue_fifo_within_same_priority(self) -> None:
        queue = SchedulerQueue()
        await queue.submit(_make_request("r1"), RequestPriority.BATCH)
        await queue.submit(_make_request("r2"), RequestPriority.BATCH)
        await queue.submit(_make_request("r3"), RequestPriority.BATCH)

        first = await queue.dequeue()
        second = await queue.dequeue()
        third = await queue.dequeue()

        assert first.request.request_id == "r1"
        assert second.request.request_id == "r2"
        assert third.request.request_id == "r3"

    async def test_dequeue_multiple_realtime_fifo(self) -> None:
        queue = SchedulerQueue()
        await queue.submit(_make_request("rt_1"), RequestPriority.REALTIME)
        await queue.submit(_make_request("rt_2"), RequestPriority.REALTIME)

        first = await queue.dequeue()
        second = await queue.dequeue()

        assert first.request.request_id == "rt_1"
        assert second.request.request_id == "rt_2"

    async def test_dequeue_interleaved_priorities(self) -> None:
        queue = SchedulerQueue()
        await queue.submit(_make_request("b1"), RequestPriority.BATCH)
        await queue.submit(_make_request("r1"), RequestPriority.REALTIME)
        await queue.submit(_make_request("b2"), RequestPriority.BATCH)
        await queue.submit(_make_request("r2"), RequestPriority.REALTIME)

        results = []
        for _ in range(4):
            sr = await queue.dequeue()
            results.append(sr.request.request_id)

        # REALTIME first (FIFO), then BATCH (FIFO)
        assert results == ["r1", "r2", "b1", "b2"]


class TestSchedulerQueueCancel:
    async def test_cancel_sets_event(self) -> None:
        queue = SchedulerQueue()
        await queue.submit(_make_request("r1"), RequestPriority.BATCH)
        result = queue.cancel("r1")
        assert result is True

    async def test_cancel_cancels_future(self) -> None:
        queue = SchedulerQueue()
        future = await queue.submit(_make_request("r1"), RequestPriority.BATCH)
        queue.cancel("r1")
        assert future.cancelled()

    async def test_cancel_nonexistent_returns_false(self) -> None:
        queue = SchedulerQueue()
        result = queue.cancel("nonexistent")
        assert result is False

    async def test_cancel_removes_from_pending(self) -> None:
        queue = SchedulerQueue()
        await queue.submit(_make_request("r1"), RequestPriority.BATCH)
        assert queue.depth == 1
        queue.cancel("r1")
        assert queue.depth == 0

    async def test_cancel_idempotent(self) -> None:
        queue = SchedulerQueue()
        await queue.submit(_make_request("r1"), RequestPriority.BATCH)
        assert queue.cancel("r1") is True
        assert queue.cancel("r1") is False  # Already cancelled

    async def test_cancelled_request_has_event_set_on_dequeue(self) -> None:
        queue = SchedulerQueue()
        await queue.submit(_make_request("r1"), RequestPriority.BATCH)
        queue.cancel("r1")

        # The request is still in the asyncio queue, but cancel_event is set
        scheduled = await queue.dequeue()
        assert scheduled.cancel_event.is_set()


class TestSchedulerQueueAging:
    async def test_not_aged_when_fresh(self) -> None:
        queue = SchedulerQueue(aging_threshold_s=30.0)
        await queue.submit(_make_request("r1"), RequestPriority.BATCH)
        scheduled = await queue.dequeue()
        assert queue.is_aged(scheduled) is False

    async def test_aged_after_threshold(self) -> None:
        queue = SchedulerQueue(aging_threshold_s=30.0)
        await queue.submit(_make_request("r1"), RequestPriority.BATCH)
        scheduled = await queue.dequeue()

        # Simulate passage of time
        scheduled.enqueued_at = time.monotonic() - 31.0
        assert queue.is_aged(scheduled) is True

    async def test_realtime_never_aged(self) -> None:
        queue = SchedulerQueue(aging_threshold_s=30.0)
        await queue.submit(_make_request("r1"), RequestPriority.REALTIME)
        scheduled = await queue.dequeue()

        # Even after threshold, REALTIME is not "aged"
        scheduled.enqueued_at = time.monotonic() - 100.0
        assert queue.is_aged(scheduled) is False

    async def test_aging_threshold_configurable(self) -> None:
        queue = SchedulerQueue(aging_threshold_s=5.0)
        await queue.submit(_make_request("r1"), RequestPriority.BATCH)
        scheduled = await queue.dequeue()

        scheduled.enqueued_at = time.monotonic() - 6.0
        assert queue.is_aged(scheduled) is True


class TestSchedulerQueueDepth:
    async def test_empty_queue(self) -> None:
        queue = SchedulerQueue()
        assert queue.depth == 0
        assert queue.empty is True
        assert queue.depth_by_priority == {"REALTIME": 0, "BATCH": 0}

    async def test_depth_by_priority(self) -> None:
        queue = SchedulerQueue()
        await queue.submit(_make_request("r1"), RequestPriority.REALTIME)
        await queue.submit(_make_request("r2"), RequestPriority.BATCH)
        await queue.submit(_make_request("r3"), RequestPriority.BATCH)

        assert queue.depth == 3
        assert queue.empty is False
        counts = queue.depth_by_priority
        assert counts["REALTIME"] == 1
        assert counts["BATCH"] == 2


class TestSchedulerQueueFutureResolution:
    async def test_future_resolved_on_set_result(self) -> None:
        queue = SchedulerQueue()
        future = await queue.submit(_make_request("r1"), RequestPriority.BATCH)
        scheduled = await queue.dequeue()

        # Simulate scheduler resolving the future with a result
        from macaw._types import BatchResult

        result = BatchResult(
            text="hello",
            language="en",
            duration=1.0,
            segments=(),
        )
        scheduled.result_future.set_result(result)

        assert future.done()
        assert future.result() is result

    async def test_future_rejected_on_exception(self) -> None:
        queue = SchedulerQueue()
        future = await queue.submit(_make_request("r1"), RequestPriority.BATCH)
        scheduled = await queue.dequeue()

        error = RuntimeError("worker crash")
        scheduled.result_future.set_exception(error)

        assert future.done()
        with pytest.raises(RuntimeError, match="worker crash"):
            future.result()


class TestSchedulerQueueConcurrent:
    async def test_concurrent_submit_and_dequeue(self) -> None:
        """Submit and dequeue concurrently without errors."""
        queue = SchedulerQueue()
        n = 20
        results: list[str] = []

        async def producer() -> None:
            for i in range(n):
                await queue.submit(_make_request(f"r{i}"), RequestPriority.BATCH)
                await asyncio.sleep(0)  # yield control

        async def consumer() -> None:
            for _ in range(n):
                sr = await queue.dequeue()
                results.append(sr.request.request_id)

        await asyncio.gather(producer(), consumer())
        assert len(results) == n

    async def test_dequeue_blocks_on_empty_queue(self) -> None:
        """Dequeue blocks until an item is available."""
        queue = SchedulerQueue()
        dequeued: list[ScheduledRequest] = []

        async def delayed_submit() -> None:
            await asyncio.sleep(0.05)
            await queue.submit(_make_request("r1"), RequestPriority.BATCH)

        async def blocking_dequeue() -> None:
            sr = await queue.dequeue()
            dequeued.append(sr)

        await asyncio.gather(delayed_submit(), blocking_dequeue())
        assert len(dequeued) == 1
        assert dequeued[0].request.request_id == "r1"
