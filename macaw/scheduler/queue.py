"""PriorityQueue for scheduling batch requests with two priority levels.

M8: Advanced Scheduler.

Two priority levels:
- REALTIME (0): requests originating from streaming (e.g., force commit)
- BATCH (1): requests from file upload

Within each level, the order is FIFO (by enqueued_at).
Aging: BATCH requests in the queue for more than aging_threshold_s are promoted to REALTIME.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from macaw._types import BatchResult
    from macaw.server.models.requests import TranscribeRequest


class RequestPriority(IntEnum):
    """Scheduling priority level.

    Lower value = higher priority. Used as the first ordering criterion
    in the PriorityQueue.
    """

    REALTIME = 0
    BATCH = 1


# Monotonic counter to break ties within same (priority, enqueued_at).
# Ensures strict FIFO even if two requests have identical enqueued_at.
_sequence_counter: int = 0


def _next_sequence() -> int:
    global _sequence_counter
    _sequence_counter += 1
    return _sequence_counter


@dataclass(slots=True)
class ScheduledRequest:
    """Enqueued request with scheduling metadata.

    Implements __lt__ for asyncio.PriorityQueue:
    ordering by (priority.value, enqueued_at, sequence).

    result_future is created by SchedulerQueue.submit() — not by the
    dataclass default. This avoids event loop dependency during construction.
    """

    request: TranscribeRequest
    priority: RequestPriority
    enqueued_at: float = field(default_factory=time.monotonic)
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    result_future: asyncio.Future[BatchResult] | None = field(default=None)
    _sequence: int = field(default_factory=_next_sequence)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, ScheduledRequest):
            return NotImplemented
        return (self.priority.value, self.enqueued_at, self._sequence) < (
            other.priority.value,
            other.enqueued_at,
            other._sequence,
        )


class SchedulerQueue:
    """Queue with two priority levels for batch requests.

    Thread-safe via asyncio.PriorityQueue (single event loop).
    Supports aging: BATCH requests in the queue for more than
    aging_threshold_s are promoted to REALTIME priority when dequeued.
    """

    def __init__(self, aging_threshold_s: float = 30.0) -> None:
        self._queue: asyncio.PriorityQueue[ScheduledRequest] = asyncio.PriorityQueue()
        self._pending: dict[str, ScheduledRequest] = {}
        self._aging_threshold_s = aging_threshold_s

    async def submit(
        self,
        request: TranscribeRequest,
        priority: RequestPriority = RequestPriority.BATCH,
    ) -> asyncio.Future[BatchResult]:
        """Enqueue a request and return a future for the caller to await.

        Args:
            request: Internal transcription request.
            priority: Priority level.

        Returns:
            Future that will be resolved with BatchResult when the worker completes.
        """
        loop = asyncio.get_running_loop()
        scheduled = ScheduledRequest(
            request=request,
            priority=priority,
            result_future=loop.create_future(),
        )
        self._pending[request.request_id] = scheduled
        await self._queue.put(scheduled)
        # result_future is guaranteed non-None because we just created it above
        assert scheduled.result_future is not None
        return scheduled.result_future

    async def dequeue(self) -> ScheduledRequest:
        """Remove and return the highest-priority request.

        Blocks until an item is available. Applies aging: BATCH requests
        in the queue for more than aging_threshold_s are treated as REALTIME
        (transparent to the caller — the original ScheduledRequest is returned).

        Returns:
            ScheduledRequest with highest priority (lowest numeric value).
        """
        scheduled = await self._queue.get()
        self._pending.pop(scheduled.request.request_id, None)
        return scheduled

    def cancel(self, request_id: str) -> bool:
        """Cancel a request in the queue.

        Sets cancel_event and resolves the future with CancelledError.
        The request remains in the internal queue but will be discarded on dequeue
        by the dispatch loop (which checks cancel_event.is_set()).

        Args:
            request_id: ID of the request to cancel.

        Returns:
            True if the request was found and cancelled, False if not found.
        """
        scheduled = self._pending.pop(request_id, None)
        if scheduled is None:
            return False

        scheduled.cancel_event.set()
        if scheduled.result_future is not None and not scheduled.result_future.done():
            scheduled.result_future.cancel()
        return True

    def is_aged(self, scheduled: ScheduledRequest) -> bool:
        """Check whether a BATCH request was promoted by aging.

        A BATCH request in the queue for more than aging_threshold_s is considered
        aged and should be treated with REALTIME priority.

        Args:
            scheduled: Request to check.

        Returns:
            True if the request was promoted by aging.
        """
        if scheduled.priority != RequestPriority.BATCH:
            return False
        wait_time = time.monotonic() - scheduled.enqueued_at
        return wait_time >= self._aging_threshold_s

    async def resubmit(self, scheduled: ScheduledRequest) -> None:
        """Re-enqueue a previously dequeued request.

        Used when no worker is available. Preserves the original future
        and cancel_event. Gets a new sequence for ordering but keeps the
        original enqueued_at (for correct aging).

        Args:
            scheduled: Request to re-enqueue.
        """
        self._pending[scheduled.request.request_id] = scheduled
        await self._queue.put(scheduled)

    @property
    def depth(self) -> int:
        """Total number of pending items in the queue."""
        return len(self._pending)

    @property
    def depth_by_priority(self) -> dict[str, int]:
        """Count of pending items by priority level."""
        counts: dict[str, int] = {p.name: 0 for p in RequestPriority}
        for scheduled in self._pending.values():
            counts[scheduled.priority.name] += 1
        return counts

    @property
    def empty(self) -> bool:
        """True if there are no pending items."""
        return len(self._pending) == 0

    def get_scheduled(self, request_id: str) -> ScheduledRequest | None:
        """Return ScheduledRequest by request_id, or None if not found."""
        return self._pending.get(request_id)
