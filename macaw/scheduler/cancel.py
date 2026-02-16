"""CancellationManager — tracks and cancels batch requests (queued and in-flight).

M8-03: Cancellation of Queued Requests.
M8-04: Cancellation of In-Flight Requests (gRPC Cancel).

Responsibilities:
- Register cancellable requests (cancel_event + future).
- Cancel instantly in queue (<1ms): set cancel_event, resolve future.
- Cancel in-flight requests via gRPC Cancel RPC to the worker.
- Idempotent: cancel of non-existent/completed request is no-op.
- Lifecycle: unregister after completion (success, error, or cancel).
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import grpc.aio

from macaw.logging import get_logger
from macaw.proto.stt_worker_pb2 import CancelRequest
from macaw.proto.stt_worker_pb2_grpc import STTWorkerStub
from macaw.scheduler.metrics import scheduler_cancel_latency_seconds

if TYPE_CHECKING:
    from macaw._types import BatchResult

logger = get_logger("scheduler.cancel")

# Timeout for cancel propagation via gRPC to the worker (seconds)
_CANCEL_PROPAGATION_TIMEOUT_S = 0.1


@dataclass(slots=True)
class _CancellableRequest:
    """Tracking info for a cancellable request."""

    request_id: str
    cancel_event: asyncio.Event
    result_future: asyncio.Future[BatchResult] | None
    worker_address: str | None = field(default=None)


class CancellationManager:
    """Manages cancellation of batch requests.

    Thread-safe via single event loop (asyncio single-threaded).

    Flow:
    1. ``register()`` when request enters the queue.
    2. ``mark_in_flight()`` when request is dispatched to the worker.
    3. ``cancel()`` at any time -- sets cancel_event and resolves future.
    4. ``unregister()`` after completion (success, error, or cancel).
    """

    def __init__(self) -> None:
        self._requests: dict[str, _CancellableRequest] = {}

    def register(
        self,
        request_id: str,
        cancel_event: asyncio.Event,
        result_future: asyncio.Future[BatchResult] | None,
    ) -> None:
        """Register request as cancellable.

        Called when request enters the queue (after submit).

        Args:
            request_id: Unique request ID.
            cancel_event: Cancellation event shared with ScheduledRequest.
            result_future: Caller's future (resolved with CancelledError on cancel).
        """
        self._requests[request_id] = _CancellableRequest(
            request_id=request_id,
            cancel_event=cancel_event,
            result_future=result_future,
        )

    def mark_in_flight(self, request_id: str, worker_address: str) -> None:
        """Mark request as in-flight (dispatched to the worker).

        Used by M8-04 for cancel propagation via gRPC.

        Args:
            request_id: Request ID.
            worker_address: Worker address (e.g. ``localhost:50051``).
        """
        entry = self._requests.get(request_id)
        if entry is not None:
            entry.worker_address = worker_address

    def cancel(self, request_id: str) -> bool:
        """Cancel request (queued or in-flight).

        Sets cancel_event and resolves future with CancelledError.
        For in-flight requests, only sets local flags -- gRPC propagation
        to the worker must be done via ``cancel_in_flight()`` by the caller.

        Idempotent: cancel of non-existent/completed request returns False.

        Args:
            request_id: ID of the request to cancel.

        Returns:
            True if request was found and cancelled, False otherwise.
        """
        entry = self._requests.get(request_id)
        if entry is None:
            return False

        # Set cancel_event (dispatch loop checks before gRPC call)
        entry.cancel_event.set()

        # Resolve future with CancelledError
        if entry.result_future is not None and not entry.result_future.done():
            entry.result_future.cancel()

        logger.info(
            "request_cancelled",
            request_id=request_id,
            in_flight=entry.worker_address is not None,
        )

        # Remove from tracking (cancel is terminal)
        self._requests.pop(request_id, None)

        return True

    async def cancel_in_flight(
        self,
        request_id: str,
        worker_address: str,
        channel: grpc.aio.Channel | None = None,
    ) -> bool:
        """Propagate cancellation via gRPC Cancel RPC to the worker.

        Called by the Scheduler when the request is in-flight.
        Best-effort: if worker does not respond in 100ms, gives up silently.

        Args:
            request_id: ID of the request to cancel.
            worker_address: gRPC worker address (e.g. ``localhost:50051``).
            channel: Existing gRPC channel (reused from pool). If None, creates temporary.

        Returns:
            True if worker confirmed cancel (acknowledged), False otherwise.
        """
        start = time.monotonic()
        close_channel = False

        try:
            if channel is None:
                channel = grpc.aio.insecure_channel(worker_address)
                close_channel = True

            stub = STTWorkerStub(channel)  # type: ignore[no-untyped-call]
            response = await asyncio.wait_for(
                stub.Cancel(CancelRequest(request_id=request_id)),
                timeout=_CANCEL_PROPAGATION_TIMEOUT_S,
            )

            elapsed = time.monotonic() - start
            elapsed_ms = elapsed * 1000

            # Observe cancel latency metric
            if scheduler_cancel_latency_seconds is not None:
                scheduler_cancel_latency_seconds.observe(elapsed)

            logger.info(
                "cancel_propagated",
                request_id=request_id,
                worker_address=worker_address,
                acknowledged=response.acknowledged,
                elapsed_ms=round(elapsed_ms, 1),
            )
            return bool(response.acknowledged)

        except (TimeoutError, grpc.aio.AioRpcError) as exc:
            elapsed = time.monotonic() - start
            elapsed_ms = elapsed * 1000

            # Observe metric even on failure
            if scheduler_cancel_latency_seconds is not None:
                scheduler_cancel_latency_seconds.observe(elapsed)

            logger.warning(
                "cancel_propagation_failed",
                request_id=request_id,
                worker_address=worker_address,
                error=str(exc),
                elapsed_ms=round(elapsed_ms, 1),
            )
            return False

        finally:
            if close_channel and channel is not None:
                with contextlib.suppress(Exception):
                    await channel.close()

    def unregister(self, request_id: str) -> None:
        """Remove request from tracking after completion.

        Called when request completes (success or error).
        No-op if request was already removed (by cancel).

        Args:
            request_id: ID of the request to remove.
        """
        self._requests.pop(request_id, None)

    def is_cancelled(self, request_id: str) -> bool:
        """Check if request was cancelled.

        An unknown request (never registered or already unregistered) is
        considered **not cancelled** -- it was never tracked, so it cannot
        have been cancelled. Only requests whose cancel_event is set
        return True.

        Args:
            request_id: Request ID.

        Returns:
            True if the request was explicitly cancelled, False otherwise
            (including unknown/unregistered requests).
        """
        entry = self._requests.get(request_id)
        if entry is None:
            return False
        return entry.cancel_event.is_set()

    def get_worker_address(self, request_id: str) -> str | None:
        """Return worker address of in-flight request.

        Used by M8-04 for cancel propagation via gRPC.

        Args:
            request_id: Request ID.

        Returns:
            Worker address if request is in-flight, None otherwise.
        """
        entry = self._requests.get(request_id)
        if entry is None:
            return None
        return entry.worker_address

    @property
    def pending_count(self) -> int:
        """Total registered requests (queued + in-flight)."""
        return len(self._requests)
