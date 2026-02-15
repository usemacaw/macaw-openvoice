"""LatencyTracker — per-request latency budget.

M8-07: Records timestamps for each batch request pipeline phase
and calculates latency metrics (queue_wait, grpc_time, total_time).

Used by the Scheduler for instrumentation. Entries are automatically
removed after complete() or after a 60s TTL to prevent memory leaks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from macaw.logging import get_logger

logger = get_logger("scheduler.latency")

# TTL for incomplete entries (prevents leaks on crash without cancel)
_DEFAULT_TTL_S = 60.0


@dataclass(slots=True)
class _RequestTimestamps:
    """Timestamps for each pipeline phase of a request."""

    enqueue_time: float = 0.0
    dequeue_time: float = 0.0
    grpc_start_time: float = 0.0
    complete_time: float = 0.0


@dataclass(frozen=True, slots=True)
class LatencySummary:
    """Latency summary for a completed request.

    All values are in seconds.
    """

    request_id: str
    queue_wait: float
    grpc_time: float
    total_time: float
    enqueue_time: float
    dequeue_time: float
    grpc_start_time: float
    complete_time: float


class LatencyTracker:
    """Track per-request latency by pipeline phase.

    Recorded phases:
    1. ``start()``       — request enqueued (enqueue_time)
    2. ``dequeued()``    — request dequeued (dequeue_time)
    3. ``grpc_started()``— gRPC call started (grpc_start_time)
    4. ``complete()``    — response received from worker (complete_time)

    After ``complete()``, derived metrics are computed:
    - queue_wait = dequeue_time - enqueue_time
    - grpc_time  = complete_time - grpc_start_time
    - total_time = complete_time - enqueue_time

    Entries are removed automatically after ``complete()`` and
    ``get_summary()`` or after TTL via ``cleanup()``.

    Args:
        ttl_s: Seconds before removing incomplete entries.
    """

    def __init__(self, *, ttl_s: float = _DEFAULT_TTL_S) -> None:
        if ttl_s <= 0:
            msg = f"ttl_s must be positive, got {ttl_s}"
            raise ValueError(msg)
        self._ttl_s = ttl_s
        self._entries: dict[str, _RequestTimestamps] = {}
        self._summaries: dict[str, LatencySummary] = {}

    @property
    def active_count(self) -> int:
        """Number of tracked requests (not completed)."""
        return len(self._entries)

    def start(self, request_id: str) -> None:
        """Record enqueue_time for a request.

        Called when the request is submitted to the scheduler.
        """
        ts = _RequestTimestamps(enqueue_time=time.monotonic())
        self._entries[request_id] = ts

    def dequeued(self, request_id: str) -> None:
        """Record dequeue_time for a request.

        Called when the request is dequeued from the priority queue.
        """
        entry = self._entries.get(request_id)
        if entry is None:
            logger.debug("latency_dequeued_unknown", request_id=request_id)
            return
        entry.dequeue_time = time.monotonic()

    def grpc_started(self, request_id: str) -> None:
        """Record grpc_start_time for a request.

        Called immediately before the gRPC TranscribeFile.
        """
        entry = self._entries.get(request_id)
        if entry is None:
            logger.debug("latency_grpc_started_unknown", request_id=request_id)
            return
        entry.grpc_start_time = time.monotonic()

    def complete(self, request_id: str) -> LatencySummary | None:
        """Record complete_time and compute summary.

        Called when the gRPC response is received (success or error).
        Removes the active entry and stores the summary for retrieval.

        Returns:
            LatencySummary if the request was tracked, None otherwise.
        """
        entry = self._entries.pop(request_id, None)
        if entry is None:
            logger.debug("latency_complete_unknown", request_id=request_id)
            return None

        entry.complete_time = time.monotonic()

        summary = LatencySummary(
            request_id=request_id,
            queue_wait=entry.dequeue_time - entry.enqueue_time if entry.dequeue_time > 0 else 0.0,
            grpc_time=entry.complete_time - entry.grpc_start_time
            if entry.grpc_start_time > 0
            else 0.0,
            total_time=entry.complete_time - entry.enqueue_time,
            enqueue_time=entry.enqueue_time,
            dequeue_time=entry.dequeue_time,
            grpc_start_time=entry.grpc_start_time,
            complete_time=entry.complete_time,
        )

        self._summaries[request_id] = summary

        logger.debug(
            "latency_complete",
            request_id=request_id,
            queue_wait_ms=round(summary.queue_wait * 1000, 1),
            grpc_time_ms=round(summary.grpc_time * 1000, 1),
            total_time_ms=round(summary.total_time * 1000, 1),
        )

        return summary

    def get_summary(self, request_id: str) -> LatencySummary | None:
        """Return summary for a completed request.

        Removes the summary after returning (one-shot read).

        Returns:
            LatencySummary if available, None otherwise.
        """
        return self._summaries.pop(request_id, None)

    def discard(self, request_id: str) -> None:
        """Remove a request from the tracker without completing.

        Used for cancelled requests that never reach complete().
        """
        self._entries.pop(request_id, None)
        self._summaries.pop(request_id, None)

    def cleanup(self) -> int:
        """Remove entries older than TTL.

        Prevents memory leaks for requests that never complete
        (e.g., worker crash without cancel propagation).

        Returns:
            Number of removed entries.
        """
        now = time.monotonic()
        cutoff = now - self._ttl_s

        expired = [rid for rid, ts in self._entries.items() if ts.enqueue_time < cutoff]

        for rid in expired:
            self._entries.pop(rid, None)
            logger.debug("latency_ttl_expired", request_id=rid)

        # Cleanup unconsumed summaries as well
        expired_summaries = [
            rid for rid, summary in self._summaries.items() if summary.enqueue_time < cutoff
        ]
        for rid in expired_summaries:
            self._summaries.pop(rid, None)

        total_removed = len(expired) + len(expired_summaries)
        if total_removed > 0:
            logger.debug(
                "latency_cleanup",
                entries_removed=len(expired),
                summaries_removed=len(expired_summaries),
            )
        return total_removed
