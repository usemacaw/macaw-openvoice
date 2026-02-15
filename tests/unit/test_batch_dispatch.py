"""Tests for M8-06: Worker Batch Inference — Parallel Dispatch.

Tests cover: parallel dispatch via asyncio.gather, semaphore concurrency
limiting in servicer, error isolation between requests in a batch,
batch dispatch with cancelled requests, gather with return_exceptions,
engine without batch support (semaphore=1), and scheduler integration.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import grpc.aio
import pytest

from macaw._types import (
    BatchResult,
    ResponseFormat,
    SegmentDetail,
    STTArchitecture,
)
from macaw.exceptions import WorkerCrashError
from macaw.proto.stt_worker_pb2 import (
    Segment,
    TranscribeFileRequest,
    TranscribeFileResponse,
    Word,
)
from macaw.scheduler.queue import RequestPriority, ScheduledRequest
from macaw.server.models.requests import TranscribeRequest
from macaw.workers.stt.servicer import STTWorkerServicer

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


def _make_batch_result(text: str = "hello") -> BatchResult:
    return BatchResult(
        text=text,
        language="pt",
        duration=1.0,
        segments=(
            SegmentDetail(
                id=0,
                start=0.0,
                end=1.0,
                text=text,
                avg_logprob=-0.3,
                no_speech_prob=0.02,
                compression_ratio=1.1,
            ),
        ),
        words=None,
    )


def _make_proto_response(text: str = "hello") -> TranscribeFileResponse:
    return TranscribeFileResponse(
        text=text,
        language="pt",
        duration=1.0,
        segments=[
            Segment(
                id=0,
                start=0.0,
                end=1.0,
                text=text,
                avg_logprob=-0.3,
                no_speech_prob=0.02,
                compression_ratio=1.1,
            ),
        ],
        words=[Word(word=text, start=0.0, end=1.0, probability=0.95)],
    )


class _MockBackend:
    """Lightweight mock backend for servicer tests."""

    def __init__(self, result: BatchResult | None = None, delay: float = 0.0) -> None:
        self._result = result or _make_batch_result()
        self._delay = delay
        self.call_count = 0

    @property
    def architecture(self) -> STTArchitecture:
        return STTArchitecture.ENCODER_DECODER

    async def transcribe_file(self, **kwargs: Any) -> BatchResult:
        self.call_count += 1
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        return self._result

    async def health(self) -> dict[str, str]:
        return {"status": "ok"}


def _make_servicer(
    backend: Any = None,
    max_concurrent: int = 1,
) -> STTWorkerServicer:
    if backend is None:
        backend = _MockBackend()
    return STTWorkerServicer(
        backend=backend,
        model_name="faster-whisper-tiny",
        engine="faster-whisper",
        max_concurrent=max_concurrent,
    )


def _make_context() -> AsyncMock:
    ctx = AsyncMock()
    ctx.abort = AsyncMock()
    ctx.cancelled.return_value = False
    return ctx


# ---------------------------------------------------------------------------
# Servicer Semaphore Tests
# ---------------------------------------------------------------------------


class TestServicerSemaphore:
    async def test_semaphore_created_with_max_concurrent(self) -> None:
        """Semaphore is initialized with max_concurrent value."""
        servicer = _make_servicer(max_concurrent=4)
        assert servicer._max_concurrent == 4
        # Semaphore internal value should be 4
        assert servicer._inference_semaphore._value == 4

    async def test_semaphore_defaults_to_one(self) -> None:
        """Default max_concurrent is 1."""
        servicer = _make_servicer()
        assert servicer._max_concurrent == 1
        assert servicer._inference_semaphore._value == 1

    async def test_semaphore_minimum_is_one(self) -> None:
        """max_concurrent below 1 is clamped to 1."""
        servicer = _make_servicer(max_concurrent=0)
        assert servicer._max_concurrent == 1
        servicer2 = _make_servicer(max_concurrent=-5)
        assert servicer2._max_concurrent == 1

    async def test_single_request_passes_semaphore(self) -> None:
        """Single request acquires and releases semaphore normally."""
        backend = _MockBackend()
        servicer = _make_servicer(backend=backend, max_concurrent=1)
        context = _make_context()

        request = TranscribeFileRequest(
            request_id="req_1",
            audio_data=b"\x00" * 3200,
        )

        response = await servicer.TranscribeFile(request, context)

        assert response.text == "hello"
        assert backend.call_count == 1
        context.abort.assert_not_called()

    async def test_concurrent_requests_limited_by_semaphore(self) -> None:
        """Semaphore limits concurrent inference calls."""
        backend = _MockBackend(delay=0.05)
        servicer = _make_servicer(backend=backend, max_concurrent=2)

        max_concurrent_observed = 0
        current_concurrent = 0
        original_transcribe = backend.transcribe_file

        async def tracking_transcribe(**kwargs: Any) -> BatchResult:
            nonlocal max_concurrent_observed, current_concurrent
            current_concurrent += 1
            if current_concurrent > max_concurrent_observed:
                max_concurrent_observed = current_concurrent
            try:
                return await original_transcribe(**kwargs)
            finally:
                current_concurrent -= 1

        backend.transcribe_file = tracking_transcribe  # type: ignore[assignment]

        # Send 4 requests concurrently with semaphore=2
        requests = []
        for i in range(4):
            req = TranscribeFileRequest(
                request_id=f"req_{i}",
                audio_data=b"\x00" * 3200,
            )
            requests.append(servicer.TranscribeFile(req, _make_context()))

        await asyncio.gather(*requests)

        # Semaphore should have limited to 2 concurrent
        assert max_concurrent_observed <= 2
        assert backend.call_count == 4

    async def test_semaphore_released_on_error(self) -> None:
        """Semaphore is released even if inference raises an exception."""
        backend = _MockBackend()

        async def failing_transcribe(**kwargs: Any) -> BatchResult:
            msg = "GPU OOM"
            raise RuntimeError(msg)

        backend.transcribe_file = failing_transcribe  # type: ignore[assignment]
        servicer = _make_servicer(backend=backend, max_concurrent=1)
        context = _make_context()

        request = TranscribeFileRequest(
            request_id="req_fail",
            audio_data=b"\x00" * 3200,
        )

        await servicer.TranscribeFile(request, context)

        # Semaphore should be released (value back to 1)
        assert servicer._inference_semaphore._value == 1
        context.abort.assert_called_once()

    async def test_cancelled_during_semaphore_wait(self) -> None:
        """Request cancelled while waiting for semaphore is detected."""
        backend = _MockBackend(delay=0.1)
        servicer = _make_servicer(backend=backend, max_concurrent=1)

        # First request holds semaphore
        req1 = TranscribeFileRequest(
            request_id="req_hold",
            audio_data=b"\x00" * 3200,
        )
        ctx1 = _make_context()

        # Second request will wait for semaphore; cancel it after a short delay
        req2 = TranscribeFileRequest(
            request_id="req_cancel",
            audio_data=b"\x00" * 3200,
        )
        ctx2 = _make_context()

        async def cancel_after_delay() -> None:
            await asyncio.sleep(0.02)
            servicer._cancelled_requests.add("req_cancel")

        # Run all concurrently
        results = await asyncio.gather(
            servicer.TranscribeFile(req1, ctx1),
            servicer.TranscribeFile(req2, ctx2),
            cancel_after_delay(),
        )

        # req1 should complete normally
        assert results[0].text == "hello"
        # req2 should have been aborted with CANCELLED
        ctx2.abort.assert_called_once_with(grpc.StatusCode.CANCELLED, "Request cancelled")


# ---------------------------------------------------------------------------
# Parallel Batch Dispatch Tests (Scheduler side)
# ---------------------------------------------------------------------------


class TestBatchDispatch:
    async def test_dispatch_batch_sends_parallel(self) -> None:
        """_dispatch_batch sends all requests in parallel."""
        from macaw.scheduler.scheduler import Scheduler

        mock_manager = MagicMock()
        mock_registry = MagicMock()
        mock_registry.get_manifest.return_value = MagicMock()

        worker = MagicMock()
        worker.worker_id = "worker-1"
        worker.port = 50051
        mock_manager.get_ready_worker.return_value = worker

        mock_stub = AsyncMock()
        mock_stub.TranscribeFile.return_value = _make_proto_response()

        scheduler = Scheduler(mock_manager, mock_registry)

        with (
            patch(
                "macaw.scheduler.scheduler.grpc.aio.insecure_channel",
                return_value=AsyncMock(),
            ),
            patch("macaw.scheduler.scheduler.STTWorkerStub", return_value=mock_stub),
        ):
            batch = [_make_scheduled(f"req_{i}") for i in range(3)]
            # Set up futures for each scheduled request
            loop = asyncio.get_running_loop()
            for s in batch:
                s.result_future = loop.create_future()

            await scheduler._dispatch_batch(batch)

            # All 3 requests should have been dispatched
            assert mock_stub.TranscribeFile.call_count == 3

            # All futures should be resolved
            for s in batch:
                assert s.result_future.done()
                result = s.result_future.result()
                assert result.text == "hello"

    async def test_dispatch_batch_skips_cancelled(self) -> None:
        """_dispatch_batch skips requests with cancel_event set."""
        from macaw.scheduler.scheduler import Scheduler

        mock_manager = MagicMock()
        mock_registry = MagicMock()
        mock_registry.get_manifest.return_value = MagicMock()

        worker = MagicMock()
        worker.worker_id = "worker-1"
        worker.port = 50051
        mock_manager.get_ready_worker.return_value = worker

        mock_stub = AsyncMock()
        mock_stub.TranscribeFile.return_value = _make_proto_response()

        scheduler = Scheduler(mock_manager, mock_registry)

        with (
            patch(
                "macaw.scheduler.scheduler.grpc.aio.insecure_channel",
                return_value=AsyncMock(),
            ),
            patch("macaw.scheduler.scheduler.STTWorkerStub", return_value=mock_stub),
        ):
            batch = [
                _make_scheduled("req_1"),
                _make_scheduled("req_2"),
                _make_scheduled("req_3"),
            ]
            loop = asyncio.get_running_loop()
            for s in batch:
                s.result_future = loop.create_future()

            # Cancel req_2
            batch[1].cancel_event.set()

            await scheduler._dispatch_batch(batch)

            # Only 2 requests dispatched (req_2 skipped)
            assert mock_stub.TranscribeFile.call_count == 2

    async def test_dispatch_batch_empty_after_cancel(self) -> None:
        """_dispatch_batch does nothing if all requests cancelled."""
        from macaw.scheduler.scheduler import Scheduler

        mock_manager = MagicMock()
        mock_registry = MagicMock()
        scheduler = Scheduler(mock_manager, mock_registry)

        batch = [_make_scheduled("req_1"), _make_scheduled("req_2")]
        batch[0].cancel_event.set()
        batch[1].cancel_event.set()

        # Should return without dispatching anything
        await scheduler._dispatch_batch(batch)

    async def test_dispatch_batch_error_isolation(self) -> None:
        """Error in one request of batch does not affect others."""
        from macaw.scheduler.scheduler import Scheduler

        mock_manager = MagicMock()
        mock_registry = MagicMock()
        mock_registry.get_manifest.return_value = MagicMock()

        worker = MagicMock()
        worker.worker_id = "worker-1"
        worker.port = 50051
        mock_manager.get_ready_worker.return_value = worker

        call_count = 0

        async def selective_failure(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise grpc.aio.AioRpcError(
                    code=grpc.StatusCode.INTERNAL,
                    initial_metadata=grpc.aio.Metadata(),
                    trailing_metadata=grpc.aio.Metadata(),
                    details="GPU error",
                )
            return _make_proto_response()

        mock_stub = AsyncMock()
        mock_stub.TranscribeFile.side_effect = selective_failure

        scheduler = Scheduler(mock_manager, mock_registry)

        with (
            patch(
                "macaw.scheduler.scheduler.grpc.aio.insecure_channel",
                return_value=AsyncMock(),
            ),
            patch("macaw.scheduler.scheduler.STTWorkerStub", return_value=mock_stub),
        ):
            batch = [
                _make_scheduled("req_ok_1"),
                _make_scheduled("req_fail"),
                _make_scheduled("req_ok_2"),
            ]
            loop = asyncio.get_running_loop()
            for s in batch:
                s.result_future = loop.create_future()

            await scheduler._dispatch_batch(batch)

            # req_ok_1 should succeed
            assert batch[0].result_future.done()
            result1 = batch[0].result_future.result()
            assert result1.text == "hello"

            # req_fail should have exception
            assert batch[1].result_future.done()
            with pytest.raises(WorkerCrashError):
                batch[1].result_future.result()

            # req_ok_2 should succeed (not affected by req_fail)
            assert batch[2].result_future.done()
            result3 = batch[2].result_future.result()
            assert result3.text == "hello"


# ---------------------------------------------------------------------------
# Engine Without Batch Support Tests
# ---------------------------------------------------------------------------


class TestEngineBatchSupport:
    async def test_semaphore_one_serializes_requests(self) -> None:
        """With semaphore=1, requests are serialized (no parallel inference)."""
        execution_order: list[str] = []
        backend = _MockBackend(delay=0.02)

        original_transcribe = backend.transcribe_file

        async def ordered_transcribe(**kwargs: Any) -> BatchResult:
            req_id = f"call_{backend.call_count}"
            execution_order.append(f"{req_id}_start")
            result = await original_transcribe(**kwargs)
            execution_order.append(f"{req_id}_end")
            return result

        backend.transcribe_file = ordered_transcribe  # type: ignore[assignment]
        servicer = _make_servicer(backend=backend, max_concurrent=1)

        requests = []
        for i in range(3):
            req = TranscribeFileRequest(
                request_id=f"req_{i}",
                audio_data=b"\x00" * 3200,
            )
            requests.append(servicer.TranscribeFile(req, _make_context()))

        await asyncio.gather(*requests)

        # With semaphore=1, each request should start after previous ends
        # execution_order should show interleaved start/end pairs without overlap
        assert backend.call_count == 3
        # Verify serial execution: start_1 before end_0 would mean overlap
        for i in range(0, len(execution_order) - 1, 2):
            start = execution_order[i]
            end = execution_order[i + 1]
            assert start.endswith("_start")
            assert end.endswith("_end")

    async def test_semaphore_allows_parallel_when_higher(self) -> None:
        """With semaphore>1, requests can run in parallel (observed via concurrency)."""
        max_concurrent_observed = 0
        current_concurrent = 0
        task_delay = 0.05
        task_count = 4
        backend = _MockBackend(delay=task_delay)
        original_transcribe = backend.transcribe_file

        async def tracking_transcribe(**kwargs: Any) -> BatchResult:
            nonlocal max_concurrent_observed, current_concurrent
            current_concurrent += 1
            if current_concurrent > max_concurrent_observed:
                max_concurrent_observed = current_concurrent
            try:
                return await original_transcribe(**kwargs)
            finally:
                current_concurrent -= 1

        backend.transcribe_file = tracking_transcribe  # type: ignore[assignment]
        servicer = _make_servicer(backend=backend, max_concurrent=task_count)

        requests = []
        for i in range(task_count):
            req = TranscribeFileRequest(
                request_id=f"req_{i}",
                audio_data=b"\x00" * 3200,
            )
            requests.append(servicer.TranscribeFile(req, _make_context()))

        await asyncio.gather(*requests)

        # Structural assertion: with semaphore=4 and 4 tasks, we should observe
        # more than 1 concurrent execution (proves parallelism is working).
        assert max_concurrent_observed > 1, (
            f"Expected parallel execution but max concurrency was {max_concurrent_observed}"
        )
        assert backend.call_count == task_count


# ---------------------------------------------------------------------------
# Integration: BatchAccumulator -> Scheduler -> Worker
# ---------------------------------------------------------------------------


class TestBatchAccumulatorIntegration:
    async def test_accumulator_flush_triggers_parallel_dispatch(self) -> None:
        """BatchAccumulator flush sends requests through _dispatch_batch."""
        from macaw.scheduler.scheduler import Scheduler

        mock_manager = MagicMock()
        mock_registry = MagicMock()
        mock_registry.get_manifest.return_value = MagicMock()

        worker = MagicMock()
        worker.worker_id = "worker-1"
        worker.port = 50051
        mock_manager.get_ready_worker.return_value = worker

        mock_stub = AsyncMock()
        mock_stub.TranscribeFile.return_value = _make_proto_response()

        scheduler = Scheduler(
            mock_manager,
            mock_registry,
            batch_accumulate_ms=5000,  # long timer
            batch_max_size=3,
        )

        with (
            patch(
                "macaw.scheduler.scheduler.grpc.aio.insecure_channel",
                return_value=AsyncMock(),
            ),
            patch("macaw.scheduler.scheduler.STTWorkerStub", return_value=mock_stub),
        ):
            await scheduler.start()
            try:
                # Submit 3 BATCH requests (hits max_batch_size)
                futures = []
                for i in range(3):
                    future = await scheduler.submit(
                        _make_request(f"req_batch_{i}"),
                        RequestPriority.BATCH,
                    )
                    futures.append(future)

                # Give dispatch loop and accumulator time to process
                await asyncio.sleep(0.3)

                # All 3 should be dispatched
                assert mock_stub.TranscribeFile.call_count == 3
            finally:
                await scheduler.stop()

    async def test_realtime_bypasses_batch_accumulator(self) -> None:
        """REALTIME requests go directly to dispatch, not through accumulator."""
        from macaw.scheduler.scheduler import Scheduler

        mock_manager = MagicMock()
        mock_registry = MagicMock()
        mock_registry.get_manifest.return_value = MagicMock()

        worker = MagicMock()
        worker.worker_id = "worker-1"
        worker.port = 50051
        mock_manager.get_ready_worker.return_value = worker

        mock_stub = AsyncMock()
        mock_stub.TranscribeFile.return_value = _make_proto_response()

        scheduler = Scheduler(
            mock_manager,
            mock_registry,
            batch_accumulate_ms=5000,  # long timer
        )

        with (
            patch(
                "macaw.scheduler.scheduler.grpc.aio.insecure_channel",
                return_value=AsyncMock(),
            ),
            patch("macaw.scheduler.scheduler.STTWorkerStub", return_value=mock_stub),
        ):
            await scheduler.start()
            try:
                await scheduler.submit(
                    _make_request("req_rt"),
                    RequestPriority.REALTIME,
                )
                await asyncio.sleep(0.2)

                # REALTIME should be dispatched immediately (not accumulated)
                assert scheduler._batch_accumulator.pending_count == 0
                assert mock_stub.TranscribeFile.call_count == 1
            finally:
                await scheduler.stop()
