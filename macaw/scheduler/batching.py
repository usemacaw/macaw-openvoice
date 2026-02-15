"""BatchAccumulator — accumulates batch requests for grouped dispatch.

M8-05: Accumulation of Requests for Batch Inference.

BatchAccumulator groups BATCH requests by model, dispatching them as a
batch after ``accumulate_ms`` or when ``max_batch_size`` is reached.
REALTIME requests do NOT pass through the accumulator — they are sent directly.

The accumulator is a *scheduler* component, not a worker component. The scheduler
decides WHEN to group; the worker decides HOW to process the group (M8-06).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from macaw.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from macaw.scheduler.queue import ScheduledRequest

logger = get_logger("scheduler.batching")


class BatchAccumulator:
    """Accumulate batch requests by time or count before flushing.

    Flush occurs automatically in two situations:
    1. ``accumulate_ms`` timer expires (flush what is available).
    2. ``max_batch_size`` is reached (immediate flush).

    Cancelled requests are removed before flush.
    All requests in the batch must share the same ``model_name``.

    Args:
        accumulate_ms: Maximum accumulation time before flush (ms).
        max_batch_size: Maximum batch size before immediate flush.
        on_flush: Async callback invoked with the list of accumulated requests.
    """

    def __init__(
        self,
        *,
        accumulate_ms: float = 50.0,
        max_batch_size: int = 8,
        on_flush: Callable[[list[ScheduledRequest]], Coroutine[object, object, None]],
    ) -> None:
        if accumulate_ms <= 0:
            msg = f"accumulate_ms must be positive, got {accumulate_ms}"
            raise ValueError(msg)
        if max_batch_size < 1:
            msg = f"max_batch_size must be >= 1, got {max_batch_size}"
            raise ValueError(msg)

        self._accumulate_s = accumulate_ms / 1000.0
        self._max_batch_size = max_batch_size
        self._on_flush = on_flush

        self._buffer: list[ScheduledRequest] = []
        self._model_name: str | None = None
        self._timer_handle: asyncio.TimerHandle | None = None
        self._flush_tasks: set[asyncio.Task[None]] = set()

    @property
    def pending_count(self) -> int:
        """Number of requests in the buffer awaiting flush."""
        return len(self._buffer)

    @property
    def model_name(self) -> str | None:
        """Model for the current batch, or None if empty."""
        return self._model_name

    def add(self, scheduled: ScheduledRequest) -> None:
        """Add a request to the current batch.

        If the batch is empty, sets the batch ``model_name``.
        If the request is for a different model, flush the current batch
        before adding.

        Starts the flush timer if this is the first item.
        If ``max_batch_size`` is reached, forces immediate flush.

        Args:
            scheduled: Request to add to the batch.

        Raises:
            RuntimeError: If called without an active event loop.
        """
        request_model = scheduled.request.model_name

        # If current batch is for another model, flush before adding
        if self._model_name is not None and self._model_name != request_model:
            logger.debug(
                "batch_model_mismatch_flush",
                current_model=self._model_name,
                new_model=request_model,
                flushing_count=len(self._buffer),
            )
            self._schedule_flush_now()

        # First item: set model and start timer
        if not self._buffer:
            self._model_name = request_model
            self._start_timer()

        self._buffer.append(scheduled)

        logger.debug(
            "batch_add",
            request_id=scheduled.request.request_id,
            model=request_model,
            batch_size=len(self._buffer),
        )

        # max_batch_size reached: immediate flush
        if len(self._buffer) >= self._max_batch_size:
            logger.debug(
                "batch_max_size_flush",
                batch_size=len(self._buffer),
                max_batch_size=self._max_batch_size,
            )
            self._schedule_flush_now()

    def flush(self) -> list[ScheduledRequest]:
        """Return accumulated requests and reset the buffer.

        Removes cancelled requests before returning.
        Cancels any pending timer.

        Returns:
            List of non-cancelled ScheduledRequests (may be empty).
        """
        self._cancel_timer()

        # Filter cancelled requests
        batch = [s for s in self._buffer if not s.cancel_event.is_set()]
        cancelled_count = len(self._buffer) - len(batch)

        if cancelled_count > 0:
            logger.debug(
                "batch_flush_removed_cancelled",
                removed=cancelled_count,
                remaining=len(batch),
            )

        # Reset
        self._buffer.clear()
        self._model_name = None

        return batch

    def _start_timer(self) -> None:
        """Start flush timer after accumulate_ms."""
        self._cancel_timer()
        loop = asyncio.get_running_loop()
        self._timer_handle = loop.call_later(
            self._accumulate_s,
            self._on_timer_expired,
        )

    def _cancel_timer(self) -> None:
        """Cancel pending timer if it exists."""
        if self._timer_handle is not None:
            self._timer_handle.cancel()
            self._timer_handle = None

    def _on_timer_expired(self) -> None:
        """Timer callback: flush and invoke on_flush."""
        self._timer_handle = None
        batch = self.flush()
        if batch:
            logger.info(
                "batch_timer_flush",
                batch_size=len(batch),
                model=batch[0].request.model_name,
            )
            # Schedule on_flush as a task (timer callback is synchronous)
            task = asyncio.create_task(self._on_flush(batch))
            self._flush_tasks.add(task)
            task.add_done_callback(self._flush_tasks.discard)

    def _schedule_flush_now(self) -> None:
        """Schedule immediate flush via call_soon (does not block add()).

        Necessary because add() is synchronous but on_flush is async.
        """
        batch = self.flush()
        if batch:
            task = asyncio.create_task(self._on_flush(batch))
            self._flush_tasks.add(task)
            task.add_done_callback(self._flush_tasks.discard)
