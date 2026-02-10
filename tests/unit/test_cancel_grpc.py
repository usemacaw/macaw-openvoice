"""Tests for M8-04: gRPC Cancel propagation and cooperative cancellation.

Tests cover: STTWorkerServicer.Cancel RPC, cooperative cancellation in
TranscribeFile, CancellationManager.cancel_in_flight, Scheduler cancel of
in-flight requests, fire-and-forget propagation.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import grpc.aio

from macaw._types import ResponseFormat
from macaw.proto.stt_worker_pb2 import (
    CancelRequest,
    CancelResponse,
    Segment,
    TranscribeFileRequest,
    TranscribeFileResponse,
    Word,
)
from macaw.scheduler.cancel import CancellationManager
from macaw.scheduler.scheduler import Scheduler
from macaw.server.models.requests import TranscribeRequest
from macaw.workers.manager import WorkerHandle, WorkerState
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


def _make_worker(port: int = 50051) -> WorkerHandle:
    return WorkerHandle(
        worker_id=f"worker-{port}",
        port=port,
        model_name="faster-whisper-tiny",
        engine="faster-whisper",
        state=WorkerState.READY,
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


def _make_scheduler(
    worker: WorkerHandle | None = None,
) -> tuple[Scheduler, MagicMock, MagicMock]:
    mock_manager = MagicMock()
    mock_registry = MagicMock()
    mock_registry.get_manifest.return_value = MagicMock()
    mock_manager.get_ready_worker.return_value = worker
    return Scheduler(mock_manager, mock_registry), mock_manager, mock_registry


def _make_servicer() -> STTWorkerServicer:
    mock_backend = AsyncMock()
    return STTWorkerServicer(
        backend=mock_backend,
        model_name="faster-whisper-tiny",
        engine="faster-whisper",
    )


# ---------------------------------------------------------------------------
# STTWorkerServicer.Cancel RPC tests
# ---------------------------------------------------------------------------


class TestServicerCancelRPC:
    async def test_cancel_returns_acknowledged(self) -> None:
        """Cancel RPC returns acknowledged=True."""
        servicer = _make_servicer()
        context = AsyncMock()
        request = CancelRequest(request_id="req_1")

        response = await servicer.Cancel(request, context)

        assert response.acknowledged is True

    async def test_cancel_adds_to_cancelled_set(self) -> None:
        """Cancel RPC adds request_id to cancelled set."""
        servicer = _make_servicer()
        context = AsyncMock()
        await servicer.Cancel(CancelRequest(request_id="req_1"), context)

        assert servicer.is_cancelled("req_1")

    async def test_cancel_is_current_request(self) -> None:
        """Cancel of the current request is detected."""
        servicer = _make_servicer()
        context = AsyncMock()

        # Simulate that req_1 is the current request being processed
        servicer._current_request_id = "req_1"

        response = await servicer.Cancel(CancelRequest(request_id="req_1"), context)
        assert response.acknowledged is True

    async def test_is_cancelled_false_for_unknown(self) -> None:
        """is_cancelled returns False for unknown request_id."""
        servicer = _make_servicer()
        assert servicer.is_cancelled("unknown") is False


# ---------------------------------------------------------------------------
# STTWorkerServicer.TranscribeFile cooperative cancellation
# ---------------------------------------------------------------------------


class TestServicerCooperativeCancel:
    async def test_transcribe_file_cancelled_before_inference(self) -> None:
        """TranscribeFile aborts if request cancelled before inference starts.

        Simulates Cancel RPC arriving after TranscribeFile enters but before
        inference starts (between the lock block and the is_cancelled check).
        """
        servicer = _make_servicer()
        context = AsyncMock()

        # Patch is_cancelled to return True on first call (before inference)
        # This simulates Cancel RPC arriving between entry and inference
        with patch.object(servicer, "is_cancelled", return_value=True):
            request = TranscribeFileRequest(
                request_id="req_cancel",
                audio_data=b"\x00" * 3200,
            )

            await servicer.TranscribeFile(request, context)

        # context.abort should have been called with CANCELLED
        context.abort.assert_called_once_with(grpc.StatusCode.CANCELLED, "Request cancelled")

    async def test_transcribe_file_cancelled_after_inference(self) -> None:
        """TranscribeFile aborts if request cancelled after inference completes."""
        servicer = _make_servicer()
        context = AsyncMock()

        from macaw._types import BatchResult, SegmentDetail

        # Backend returns normally, but we cancel during inference
        servicer._backend.transcribe_file.return_value = BatchResult(
            text="hello",
            language="pt",
            duration=1.0,
            segments=(
                SegmentDetail(
                    id=0,
                    start=0.0,
                    end=1.0,
                    text="hello",
                    avg_logprob=-0.3,
                    no_speech_prob=0.02,
                    compression_ratio=1.1,
                ),
            ),
            words=None,
        )

        request = TranscribeFileRequest(
            request_id="req_cancel_after",
            audio_data=b"\x00" * 3200,
        )

        # Cancel happens while inference is running (simulate by setting flag
        # between transcribe_file start and check)
        original_transcribe = servicer._backend.transcribe_file

        async def cancel_during_inference(**kwargs: Any) -> Any:
            result = await original_transcribe(**kwargs)
            # Cancel flag set after inference completes
            servicer._cancelled_requests.add("req_cancel_after")
            return result

        servicer._backend.transcribe_file = cancel_during_inference

        await servicer.TranscribeFile(request, context)

        context.abort.assert_called_once_with(grpc.StatusCode.CANCELLED, "Request cancelled")

    async def test_transcribe_file_not_cancelled_completes_normally(self) -> None:
        """TranscribeFile completes normally when not cancelled."""
        servicer = _make_servicer()
        context = AsyncMock()

        from macaw._types import BatchResult, SegmentDetail

        servicer._backend.transcribe_file.return_value = BatchResult(
            text="hello",
            language="pt",
            duration=1.0,
            segments=(
                SegmentDetail(
                    id=0,
                    start=0.0,
                    end=1.0,
                    text="hello",
                    avg_logprob=-0.3,
                    no_speech_prob=0.02,
                    compression_ratio=1.1,
                ),
            ),
            words=None,
        )

        request = TranscribeFileRequest(
            request_id="req_normal",
            audio_data=b"\x00" * 3200,
        )

        response = await servicer.TranscribeFile(request, context)

        # Should NOT abort
        context.abort.assert_not_called()
        assert response.text == "hello"

    async def test_transcribe_file_cleans_up_cancel_flag(self) -> None:
        """TranscribeFile removes cancelled flag and current_request_id in finally."""
        servicer = _make_servicer()
        context = AsyncMock()

        from macaw._types import BatchResult, SegmentDetail

        servicer._backend.transcribe_file.return_value = BatchResult(
            text="hello",
            language="pt",
            duration=1.0,
            segments=(
                SegmentDetail(
                    id=0,
                    start=0.0,
                    end=1.0,
                    text="hello",
                    avg_logprob=-0.3,
                    no_speech_prob=0.02,
                    compression_ratio=1.1,
                ),
            ),
            words=None,
        )

        request = TranscribeFileRequest(
            request_id="req_cleanup",
            audio_data=b"\x00" * 3200,
        )

        await servicer.TranscribeFile(request, context)

        # After completion, cancel flag and current_request_id should be cleaned
        assert servicer._current_request_id is None
        assert "req_cleanup" not in servicer._cancelled_requests


# ---------------------------------------------------------------------------
# CancellationManager.cancel_in_flight tests
# ---------------------------------------------------------------------------


class TestCancelInFlight:
    async def test_cancel_in_flight_success(self) -> None:
        """cancel_in_flight sends gRPC Cancel and returns True on success."""
        cm = CancellationManager()

        mock_channel = AsyncMock()
        mock_stub = AsyncMock()
        mock_stub.Cancel.return_value = CancelResponse(acknowledged=True)

        with patch("macaw.scheduler.cancel.STTWorkerStub", return_value=mock_stub):
            result = await cm.cancel_in_flight("req_1", "localhost:50051", channel=mock_channel)

        assert result is True
        mock_stub.Cancel.assert_called_once()

    async def test_cancel_in_flight_not_acknowledged(self) -> None:
        """cancel_in_flight returns False when worker does not acknowledge."""
        cm = CancellationManager()

        mock_channel = AsyncMock()
        mock_stub = AsyncMock()
        mock_stub.Cancel.return_value = CancelResponse(acknowledged=False)

        with patch("macaw.scheduler.cancel.STTWorkerStub", return_value=mock_stub):
            result = await cm.cancel_in_flight("req_1", "localhost:50051", channel=mock_channel)

        assert result is False

    async def test_cancel_in_flight_timeout(self) -> None:
        """cancel_in_flight returns False on timeout."""
        cm = CancellationManager()

        mock_channel = AsyncMock()
        mock_stub = AsyncMock()
        mock_stub.Cancel.side_effect = TimeoutError("timeout")

        with patch("macaw.scheduler.cancel.STTWorkerStub", return_value=mock_stub):
            result = await cm.cancel_in_flight("req_1", "localhost:50051", channel=mock_channel)

        assert result is False

    async def test_cancel_in_flight_grpc_error(self) -> None:
        """cancel_in_flight returns False on gRPC error."""
        cm = CancellationManager()

        mock_channel = AsyncMock()
        mock_stub = AsyncMock()
        mock_stub.Cancel.side_effect = grpc.aio.AioRpcError(
            code=grpc.StatusCode.UNAVAILABLE,
            initial_metadata=grpc.aio.Metadata(),
            trailing_metadata=grpc.aio.Metadata(),
            details="Worker unavailable",
        )

        with patch("macaw.scheduler.cancel.STTWorkerStub", return_value=mock_stub):
            result = await cm.cancel_in_flight("req_1", "localhost:50051", channel=mock_channel)

        assert result is False

    async def test_cancel_in_flight_creates_channel_when_none(self) -> None:
        """cancel_in_flight creates temporary channel when none provided."""
        cm = CancellationManager()

        mock_stub = AsyncMock()
        mock_stub.Cancel.return_value = CancelResponse(acknowledged=True)
        mock_channel = AsyncMock()

        with (
            patch(
                "macaw.scheduler.cancel.grpc.aio.insecure_channel",
                return_value=mock_channel,
            ) as mock_create_channel,
            patch("macaw.scheduler.cancel.STTWorkerStub", return_value=mock_stub),
        ):
            result = await cm.cancel_in_flight("req_1", "localhost:50051", channel=None)

        assert result is True
        mock_create_channel.assert_called_once_with("localhost:50051")
        # Temporary channel should be closed
        mock_channel.close.assert_called_once()

    async def test_cancel_in_flight_does_not_close_provided_channel(self) -> None:
        """cancel_in_flight does NOT close a channel that was provided."""
        cm = CancellationManager()

        mock_channel = AsyncMock()
        mock_stub = AsyncMock()
        mock_stub.Cancel.return_value = CancelResponse(acknowledged=True)

        with patch("macaw.scheduler.cancel.STTWorkerStub", return_value=mock_stub):
            await cm.cancel_in_flight("req_1", "localhost:50051", channel=mock_channel)

        # Provided channel should NOT be closed
        mock_channel.close.assert_not_called()


# ---------------------------------------------------------------------------
# Scheduler cancel of in-flight request tests
# ---------------------------------------------------------------------------


class TestSchedulerCancelInFlight:
    async def test_cancel_in_flight_fires_grpc_cancel(self) -> None:
        """Scheduler.cancel() fires gRPC cancel for in-flight request."""
        worker = _make_worker()
        scheduler, _, _ = _make_scheduler(worker)

        mock_stub = AsyncMock()
        mock_stub.TranscribeFile.return_value = _make_proto_response()

        # We need the dispatch loop to pick up the request and mark it in-flight
        # Use a slow mock to keep it in-flight while we cancel
        slow_future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()

        async def slow_transcribe(*args: Any, **kwargs: Any) -> Any:
            return await slow_future

        mock_stub.TranscribeFile.side_effect = slow_transcribe

        with (
            patch(
                "macaw.scheduler.scheduler.grpc.aio.insecure_channel",
                return_value=AsyncMock(),
            ),
            patch("macaw.scheduler.scheduler.STTWorkerStub", return_value=mock_stub),
            patch.object(
                scheduler._cancellation,
                "cancel_in_flight",
                new_callable=AsyncMock,
            ) as mock_cancel_in_flight,
        ):
            await scheduler.start()
            try:
                await scheduler.submit(_make_request("req_inflight"))

                # Wait for dispatch loop to pick up and mark in-flight
                for _ in range(50):
                    await asyncio.sleep(0.01)
                    if "req_inflight" in scheduler._in_flight:
                        break

                # Now cancel the in-flight request
                result = scheduler.cancel("req_inflight")
                assert result is True

                # Give the fire-and-forget task a chance to run
                await asyncio.sleep(0.05)

                # cancel_in_flight should have been called
                mock_cancel_in_flight.assert_called_once()

                # Clean up: resolve the slow future to unblock dispatch
                slow_future.set_result(_make_proto_response())
            finally:
                await scheduler.stop()

    async def test_cancel_queued_does_not_fire_grpc_cancel(self) -> None:
        """Scheduler.cancel() of queued request does NOT fire gRPC cancel."""
        scheduler, _, _ = _make_scheduler(None)  # No worker → stays in queue

        with patch.object(
            scheduler._cancellation,
            "cancel_in_flight",
            new_callable=AsyncMock,
        ) as mock_cancel_in_flight:
            await scheduler.start()
            try:
                await scheduler.submit(_make_request("req_queued"))
                scheduler.cancel("req_queued")

                await asyncio.sleep(0.05)

                # cancel_in_flight should NOT be called for queued requests
                mock_cancel_in_flight.assert_not_called()
            finally:
                await scheduler.stop()

    async def test_cancel_in_flight_passes_channel_from_pool(self) -> None:
        """Scheduler.cancel() passes existing channel from pool."""
        worker = _make_worker()
        scheduler, _, _ = _make_scheduler(worker)

        # Pre-populate the channel pool
        mock_channel = AsyncMock()
        scheduler._channels["localhost:50051"] = mock_channel

        mock_stub = AsyncMock()
        slow_future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()

        async def slow_transcribe(*args: Any, **kwargs: Any) -> Any:
            return await slow_future

        mock_stub.TranscribeFile.side_effect = slow_transcribe

        with (
            patch("macaw.scheduler.scheduler.STTWorkerStub", return_value=mock_stub),
            patch.object(
                scheduler._cancellation,
                "cancel_in_flight",
                new_callable=AsyncMock,
            ) as mock_cancel_in_flight,
        ):
            await scheduler.start()
            try:
                await scheduler.submit(_make_request("req_pool"))

                # Wait for in-flight
                for _ in range(50):
                    await asyncio.sleep(0.01)
                    if "req_pool" in scheduler._in_flight:
                        break

                scheduler.cancel("req_pool")
                await asyncio.sleep(0.05)

                # Verify channel was passed
                if mock_cancel_in_flight.call_count > 0:
                    call_args = mock_cancel_in_flight.call_args
                    assert call_args[0][2] is mock_channel  # channel arg

                slow_future.set_result(_make_proto_response())
            finally:
                await scheduler.stop()
