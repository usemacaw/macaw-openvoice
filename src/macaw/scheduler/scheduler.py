"""Scheduler — routes transcription requests to gRPC workers.

M8: Advanced Scheduler with PriorityQueue, async dispatch loop,
gRPC channel pool, and graceful shutdown.

Streaming (WebSocket) does NOT go through this scheduler — it uses
StreamingGRPCClient directly (M5). This scheduler only manages batch requests.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING, Any

import grpc.aio

from macaw.exceptions import WorkerCrashError, WorkerTimeoutError, WorkerUnavailableError
from macaw.logging import get_logger
from macaw.proto.stt_worker_pb2_grpc import STTWorkerStub
from macaw.scheduler.batching import BatchAccumulator
from macaw.scheduler.cancel import CancellationManager
from macaw.scheduler.converters import build_proto_request, proto_response_to_batch_result
from macaw.scheduler.latency import LatencyTracker
from macaw.scheduler.metrics import (
    scheduler_aging_promotions_total,
    scheduler_batch_size,
    scheduler_grpc_duration_seconds,
    scheduler_queue_depth,
    scheduler_queue_wait_seconds,
    scheduler_requests_total,
)
from macaw.scheduler.queue import RequestPriority, SchedulerQueue

if TYPE_CHECKING:
    from macaw._types import BatchResult
    from macaw.registry.registry import ModelRegistry
    from macaw.scheduler.queue import ScheduledRequest
    from macaw.server.models.requests import TranscribeRequest
    from macaw.workers.manager import WorkerManager

logger = get_logger("scheduler")

# gRPC channel options — gRPC defaults are 4MB (insufficient for 25MB audio)
_GRPC_CHANNEL_OPTIONS = [
    ("grpc.max_send_message_length", 30 * 1024 * 1024),
    ("grpc.max_receive_message_length", 30 * 1024 * 1024),
    ("grpc.keepalive_time_ms", 30_000),
    ("grpc.keepalive_timeout_ms", 10_000),
    ("grpc.keepalive_permit_without_calls", 1),
]

# Minimum timeout for TranscribeFile (seconds)
_MIN_GRPC_TIMEOUT = 30.0

# Factor applied to estimated audio duration to compute timeout
_TIMEOUT_FACTOR = 2.0

# Backoff when no worker is available (seconds)
_NO_WORKER_BACKOFF_S = 0.1

# Timeout for graceful shutdown (seconds)
_SHUTDOWN_TIMEOUT_S = 10.0


class Scheduler:
    """Routes transcription requests to gRPC workers.

    M8: Scheduler with PriorityQueue, async dispatch loop, and channel pool.

    Flow:
    1. ``transcribe(request)`` enqueues into PriorityQueue and awaits a future.
    2. ``_dispatch_loop()`` consumes the queue and dispatches to workers via gRPC.
    3. Result resolves the future; caller receives ``BatchResult``.

    Lifecycle:
    - ``start()`` starts the dispatch loop as a background task.
    - ``stop()`` stops the loop and waits for in-flight requests (graceful, 10s timeout).
    """

    def __init__(
        self,
        worker_manager: WorkerManager,
        registry: ModelRegistry,
        *,
        aging_threshold_s: float = 30.0,
        batch_accumulate_ms: float = 50.0,
        batch_max_size: int = 8,
    ) -> None:
        self._worker_manager = worker_manager
        self._registry = registry
        self._queue = SchedulerQueue(aging_threshold_s=aging_threshold_s)
        self._channels: dict[str, grpc.aio.Channel] = {}
        self._dispatch_task: asyncio.Task[None] | None = None
        self._running = False
        self._in_flight: set[str] = set()
        self._in_flight_tasks: set[asyncio.Task[Any]] = set()
        self._cancellation = CancellationManager()
        self._batch_accumulator = BatchAccumulator(
            accumulate_ms=batch_accumulate_ms,
            max_batch_size=batch_max_size,
            on_flush=self._dispatch_batch,
        )
        self._latency = LatencyTracker()

    @property
    def queue(self) -> SchedulerQueue:
        """Access the queue for inspection (metrics)."""
        return self._queue

    @property
    def cancellation(self) -> CancellationManager:
        """Access CancellationManager (for inspection/metrics)."""
        return self._cancellation

    @property
    def batch_accumulator(self) -> BatchAccumulator:
        """Access BatchAccumulator (for inspection/metrics)."""
        return self._batch_accumulator

    @property
    def latency(self) -> LatencyTracker:
        """Access LatencyTracker (for inspection/metrics)."""
        return self._latency

    @property
    def running(self) -> bool:
        """True if the dispatch loop is active."""
        return self._running

    def cancel(self, request_id: str) -> bool:
        """Cancel a request in the queue or in flight.

        Sets cancel_event and resolves the future with CancelledError.
        For in-flight requests, starts gRPC propagation to the worker as a
        fire-and-forget task (does not block the caller).

        Idempotent: cancelling a missing/completed request returns False.

        Args:
            request_id: ID of the request to cancel.

        Returns:
            True if the request was found and cancelled, False otherwise.
        """
        # Captura worker_address ANTES de cancelar (cancel remove a entry)
        worker_address = self._cancellation.get_worker_address(request_id)
        channel = self._channels.get(worker_address) if worker_address else None

        # Capture priority for metrics BEFORE cancelling
        scheduled_for_metric = self._queue.get_scheduled(request_id)

        # Cancel via CancellationManager (sets event + resolves future)
        if not self._cancellation.cancel(request_id):
            return False

        # Remove from queue tracking (for correct depth_by_priority)
        cancelled_in_queue = self._queue.cancel(request_id)

        # Update queue depth if the request was queued
        if (
            cancelled_in_queue
            and scheduler_queue_depth is not None
            and scheduled_for_metric is not None
        ):
            scheduler_queue_depth.labels(
                priority=scheduled_for_metric.priority.name,
            ).dec()

        # Remove from latency tracker (will not complete)
        self._latency.discard(request_id)

        # If in-flight, propagate gRPC Cancel to the worker (fire-and-forget)
        if worker_address is not None:
            task = asyncio.create_task(
                self._cancellation.cancel_in_flight(request_id, worker_address, channel)
            )
            self._in_flight_tasks.add(task)
            task.add_done_callback(self._in_flight_tasks.discard)

        return True

    async def start(self) -> None:
        """Start the dispatch loop as a background task.

        Idempotent: calling start() when already running is a no-op.
        """
        if self._running:
            return
        self._running = True
        self._dispatch_task = asyncio.create_task(self._dispatch_loop())
        logger.info("scheduler_started")

    async def stop(self) -> None:
        """Stop the dispatch loop and wait for in-flight requests.

        Graceful shutdown with a 10s timeout. Requests not completed
        after the timeout are cancelled.
        """
        if not self._running:
            return
        self._running = False

        # Flush batch accumulator (dispatch pending requests)
        pending_batch = self._batch_accumulator.flush()
        if pending_batch:
            await self._dispatch_batch(pending_batch)

        # Signal the dispatch loop to stop
        if self._dispatch_task is not None:
            self._dispatch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._dispatch_task
            self._dispatch_task = None

        # Wait for in-flight tasks with timeout
        if self._in_flight_tasks:
            logger.info("scheduler_draining", in_flight=len(self._in_flight_tasks))
            _done, pending = await asyncio.wait(
                self._in_flight_tasks,
                timeout=_SHUTDOWN_TIMEOUT_S,
            )
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        # Close gRPC channels
        close_tasks = [channel.close() for channel in self._channels.values()]
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
        self._channels.clear()

        logger.info("scheduler_stopped")

    async def transcribe(self, request: TranscribeRequest) -> BatchResult:
        """Send request to a worker and return the result.

        Keeps the same external signature as M3 for compatibility.

        If the dispatch loop is active (``start()`` called), it enqueues in the
        PriorityQueue and awaits the future. Otherwise, it executes inline
        (compatibility with M3 and existing tests).

        Raises:
            ModelNotFoundError: Model does not exist in the registry.
            WorkerUnavailableError: No READY worker for the model.
            WorkerCrashError: Worker returned an unrecoverable error.
            WorkerTimeoutError: Worker did not respond within the timeout.
            asyncio.CancelledError: Request was cancelled.
        """
        # Validate that the model exists (raises ModelNotFoundError if not)
        self._registry.get_manifest(request.model_name)

        if not self._running:
            # Inline mode (M3 compat): execute directly without queue
            return await self._transcribe_inline(request)

        # M8 mode: enqueue with BATCH priority and await result
        future = await self.submit(request, RequestPriority.BATCH)
        return await future

    async def _transcribe_inline(self, request: TranscribeRequest) -> BatchResult:
        """Inline execution without queue (M3 compatibility).

        Used when the dispatch loop is not active. Creates and closes a channel
        per request (behavior identical to the original M3).
        """
        worker = self._worker_manager.get_ready_worker(request.model_name)
        if worker is None:
            raise WorkerUnavailableError(request.model_name)

        logger.info(
            "transcribe_start",
            request_id=request.request_id,
            model=request.model_name,
            worker_id=worker.worker_id,
            audio_bytes=len(request.audio_data),
            task=request.task,
        )

        proto_request = build_proto_request(request)

        audio_duration_estimate = len(request.audio_data) / (16_000 * 2)
        timeout = max(_MIN_GRPC_TIMEOUT, audio_duration_estimate * _TIMEOUT_FACTOR)

        channel = grpc.aio.insecure_channel(
            f"localhost:{worker.port}",
            options=_GRPC_CHANNEL_OPTIONS,
        )
        try:
            stub = STTWorkerStub(channel)  # type: ignore[no-untyped-call]
            proto_response = await stub.TranscribeFile(
                proto_request,
                timeout=timeout,
            )
        except grpc.aio.AioRpcError as exc:
            _translate_grpc_error(exc, worker.worker_id, timeout)
            raise  # pragma: no cover — _translate_grpc_error sempre levanta
        finally:
            try:
                await channel.close()
            except Exception:
                logger.warning("channel_close_error", worker_id=worker.worker_id)

        result = proto_response_to_batch_result(proto_response)

        logger.info(
            "transcribe_done",
            request_id=request.request_id,
            text_length=len(result.text),
            segments=len(result.segments),
        )

        return result

    async def submit(
        self,
        request: TranscribeRequest,
        priority: RequestPriority = RequestPriority.BATCH,
    ) -> asyncio.Future[BatchResult]:
        """Enqueue a request and return a future for the caller to await.

        Unlike ``transcribe()``, it does not block until the result.
        Allows the caller to cancel or monitor progress.
        Automatically registers in CancellationManager.
        """
        # Record enqueue timestamp
        self._latency.start(request.request_id)

        future = await self._queue.submit(request, priority)

        # Update queue depth metric
        if scheduler_queue_depth is not None:
            scheduler_queue_depth.labels(priority=priority.name).inc()

        # Register in CancellationManager to allow cancel
        scheduled = self._queue.get_scheduled(request.request_id)
        if scheduled is not None:
            self._cancellation.register(
                request.request_id,
                scheduled.cancel_event,
                future,
            )

        return future

    async def _dispatch_loop(self) -> None:
        """Main dispatch loop for requests.

        Consumes the PriorityQueue and dispatches to workers via gRPC.
        Runs as a background task until ``stop()`` is called.
        """
        logger.info("dispatch_loop_started")
        try:
            while self._running:
                try:
                    scheduled = await asyncio.wait_for(
                        self._queue.dequeue(),
                        timeout=0.5,
                    )
                except TimeoutError:
                    continue

                # Record dequeue timestamp
                self._latency.dequeued(scheduled.request.request_id)

                # Update queue depth (decrement for original priority)
                if scheduler_queue_depth is not None:
                    scheduler_queue_depth.labels(
                        priority=scheduled.priority.name,
                    ).dec()

                # Observe queue wait time
                if scheduler_queue_wait_seconds is not None:
                    wait = time.monotonic() - scheduled.enqueued_at
                    scheduler_queue_wait_seconds.observe(wait)

                # Check aging (BATCH promoted to REALTIME)
                if self._queue.is_aged(scheduled):
                    if scheduler_aging_promotions_total is not None:
                        scheduler_aging_promotions_total.inc()
                    logger.info(
                        "dispatch_aged_promotion",
                        request_id=scheduled.request.request_id,
                    )

                # Check if request was cancelled while queued
                if scheduled.cancel_event.is_set():
                    self._latency.discard(scheduled.request.request_id)
                    if scheduler_requests_total is not None:
                        scheduler_requests_total.labels(
                            priority=scheduled.priority.name,
                            status="cancelled",
                        ).inc()
                    logger.debug(
                        "dispatch_skip_cancelled",
                        request_id=scheduled.request.request_id,
                    )
                    continue

                # REALTIME: dispatch immediately (no batching)
                # BATCH: accumulate in BatchAccumulator (flush by timer or max_size)
                if scheduled.priority == RequestPriority.REALTIME:
                    task = asyncio.create_task(self._dispatch_request(scheduled))
                    self._in_flight_tasks.add(task)
                    task.add_done_callback(self._in_flight_tasks.discard)
                else:
                    self._batch_accumulator.add(scheduled)
        except asyncio.CancelledError:
            logger.info("dispatch_loop_cancelled")

    async def _dispatch_request(self, scheduled: ScheduledRequest) -> None:
        """Dispatch a request to the worker via gRPC.

        If no worker is available, re-enqueue with backoff.
        """
        request = scheduled.request
        future = scheduled.result_future

        # Find READY worker for the model
        worker = self._worker_manager.get_ready_worker(request.model_name)
        if worker is None:
            # Re-enqueue: preserve future and cancel_event
            await asyncio.sleep(_NO_WORKER_BACKOFF_S)

            # Check if it was cancelled during backoff
            if scheduled.cancel_event.is_set():
                return

            # Re-enqueue into PriorityQueue
            await self._queue.resubmit(scheduled)
            logger.debug(
                "dispatch_requeue_no_worker",
                request_id=request.request_id,
                model=request.model_name,
            )
            return

        # Mark as in-flight
        address = f"localhost:{worker.port}"
        self._in_flight.add(request.request_id)
        self._cancellation.mark_in_flight(request.request_id, address)

        logger.info(
            "transcribe_start",
            request_id=request.request_id,
            model=request.model_name,
            worker_id=worker.worker_id,
            audio_bytes=len(request.audio_data),
            task=request.task,
        )

        grpc_start = time.monotonic()
        status = "ok"

        try:
            # Send gRPC TranscribeFile to the worker
            proto_request = build_proto_request(request)

            # Timeout proportional to audio size
            audio_duration_estimate = len(request.audio_data) / (16_000 * 2)
            timeout = max(_MIN_GRPC_TIMEOUT, audio_duration_estimate * _TIMEOUT_FACTOR)

            channel = self._get_or_create_channel(address)
            stub = STTWorkerStub(channel)  # type: ignore[no-untyped-call]

            self._latency.grpc_started(request.request_id)
            proto_response = await stub.TranscribeFile(
                proto_request,
                timeout=timeout,
            )

            # Convert proto -> BatchResult
            result = proto_response_to_batch_result(proto_response)

            # Resolve future with result
            if future is not None and not future.done():
                future.set_result(result)

            logger.info(
                "transcribe_done",
                request_id=request.request_id,
                text_length=len(result.text),
                segments=len(result.segments),
            )

        except grpc.aio.AioRpcError as exc:
            status = "error"
            error = _make_domain_error(exc, worker.worker_id, timeout)
            if future is not None and not future.done():
                future.set_exception(error)

        except Exception as exc:
            status = "error"
            if future is not None and not future.done():
                future.set_exception(exc)

        finally:
            self._latency.complete(request.request_id)
            self._in_flight.discard(request.request_id)
            self._cancellation.unregister(request.request_id)

            # Observe gRPC metrics and status
            grpc_elapsed = time.monotonic() - grpc_start
            if scheduler_grpc_duration_seconds is not None:
                scheduler_grpc_duration_seconds.observe(grpc_elapsed)
            if scheduler_requests_total is not None:
                scheduler_requests_total.labels(
                    priority=scheduled.priority.name,
                    status=status,
                ).inc()

    async def _dispatch_batch(self, batch: list[ScheduledRequest]) -> None:
        """Dispatch a batch of requests in parallel via asyncio.gather.

        BatchAccumulator callback. All requests in the batch are sent in
        parallel to the same worker (HTTP/2 multiplexes via the gRPC channel).
        An error in one request does not affect others (return_exceptions=True
        in the internal gather of _dispatch_request, which resolves individual futures).
        """
        # Filter cancelled requests between add() and flush()
        active = [s for s in batch if not s.cancel_event.is_set()]
        if not active:
            return

        # Observe batch size
        if scheduler_batch_size is not None:
            scheduler_batch_size.observe(len(active))

        logger.info(
            "dispatch_batch",
            batch_size=len(active),
            model=active[0].request.model_name,
        )

        # Dispatch all in parallel — each _dispatch_request resolves its future
        tasks = []
        for scheduled in active:
            task = asyncio.create_task(self._dispatch_request(scheduled))
            self._in_flight_tasks.add(task)
            task.add_done_callback(self._in_flight_tasks.discard)
            tasks.append(task)

        # Wait for all to complete (errors already resolved in individual futures)
        await asyncio.gather(*tasks, return_exceptions=True)

    def _get_or_create_channel(self, address: str) -> grpc.aio.Channel:
        """Return gRPC channel for the address, creating if needed."""
        channel = self._channels.get(address)
        if channel is None:
            channel = grpc.aio.insecure_channel(
                address,
                options=_GRPC_CHANNEL_OPTIONS,
            )
            self._channels[address] = channel
        return channel

    async def close_channel(self, address: str) -> None:
        """Close and remove gRPC channel for a specific address.

        Used when a worker dies and the channel should be discarded.
        """
        channel = self._channels.pop(address, None)
        if channel is not None:
            try:
                await channel.close()
            except Exception:
                logger.warning("channel_close_error", address=address)


def _make_domain_error(
    exc: grpc.aio.AioRpcError,
    worker_id: str,
    timeout: float,
) -> WorkerCrashError | WorkerTimeoutError:
    """Translate gRPC errors into Macaw domain exceptions."""
    code = exc.code()

    if code == grpc.StatusCode.DEADLINE_EXCEEDED:
        return WorkerTimeoutError(worker_id, timeout)

    if code == grpc.StatusCode.UNAVAILABLE:
        return WorkerCrashError(worker_id)

    if code == grpc.StatusCode.CANCELLED:
        return WorkerCrashError(worker_id)

    # Remaining gRPC errors -> generic WorkerCrashError
    logger.error(
        "grpc_error",
        worker_id=worker_id,
        grpc_code=code.name if code else "UNKNOWN",
        grpc_details=exc.details(),
    )
    return WorkerCrashError(worker_id)


# Backward compat: re-export for existing code that imports _translate_grpc_error
def _translate_grpc_error(
    exc: grpc.aio.AioRpcError,
    worker_id: str,
    timeout: float,
) -> None:
    """Translate gRPC errors into Macaw domain exceptions.

    Always raises — never returns normally.
    Kept for backward compatibility with existing tests.
    """
    raise _make_domain_error(exc, worker_id, timeout) from exc
