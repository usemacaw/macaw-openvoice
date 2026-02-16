"""Tests for M8-08: Scheduler Prometheus Metrics.

Tests cover: metric definitions, lazy import pattern,
queue_depth inc/dec, queue_wait_seconds observation, grpc_duration_seconds
observation, cancel_latency_seconds observation, batch_size observation,
requests_total by priority+status, aging_promotions_total, and no-op
behavior when prometheus_client is not installed.
"""

from __future__ import annotations

import asyncio
import contextlib
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
    *,
    aging_threshold_s: float = 30.0,
) -> tuple[Scheduler, MagicMock, MagicMock]:
    """Create scheduler with mocked dependencies."""
    registry = MagicMock()
    registry.get_manifest.return_value = MagicMock()

    worker_manager = MagicMock()
    worker_manager.get_ready_worker.return_value = worker

    scheduler = Scheduler(
        worker_manager,
        registry,
        aging_threshold_s=aging_threshold_s,
    )
    return scheduler, worker_manager, registry


# ---------------------------------------------------------------------------
# Metric module tests
# ---------------------------------------------------------------------------


class TestMetricDefinitions:
    """Tests for the metrics module itself."""

    def test_all_metrics_defined(self) -> None:
        """All 7 metrics are defined (not None) when prometheus is available."""
        from macaw.scheduler import metrics

        assert metrics.scheduler_queue_depth is not None
        assert metrics.scheduler_queue_wait_seconds is not None
        assert metrics.scheduler_grpc_duration_seconds is not None
        assert metrics.scheduler_cancel_latency_seconds is not None
        assert metrics.scheduler_batch_size is not None
        assert metrics.scheduler_requests_total is not None
        assert metrics.scheduler_aging_promotions_total is not None

    def test_queue_depth_has_priority_label(self) -> None:
        """queue_depth gauge accepts 'priority' label."""
        from macaw.scheduler.metrics import scheduler_queue_depth

        assert scheduler_queue_depth is not None
        # Should not raise — valid label
        child = scheduler_queue_depth.labels(priority="BATCH")
        assert child is not None


# ---------------------------------------------------------------------------
# Queue depth metric tests
# ---------------------------------------------------------------------------


class TestQueueDepthMetric:
    async def test_submit_increments_queue_depth(self) -> None:
        """submit() increments queue_depth for the given priority."""
        scheduler, _, _ = _make_scheduler()

        with patch("macaw.scheduler.scheduler.scheduler_queue_depth") as mock_gauge:
            mock_child = MagicMock()
            mock_gauge.labels.return_value = mock_child

            await scheduler.submit(_make_request(), RequestPriority.BATCH)

            mock_gauge.labels.assert_called_with(priority="BATCH")
            mock_child.inc.assert_called_once()

    async def test_dispatch_decrements_queue_depth(self) -> None:
        """Dispatch loop decrements queue_depth when dequeuing."""
        worker = _make_worker()
        scheduler, _, _ = _make_scheduler(worker)

        mock_child = MagicMock()
        with (
            patch("macaw.scheduler.scheduler.scheduler_queue_depth") as mock_gauge,
            patch("macaw.scheduler.scheduler.scheduler_queue_wait_seconds"),
            patch("macaw.scheduler.scheduler.scheduler_requests_total"),
            patch("macaw.scheduler.scheduler.scheduler_grpc_duration_seconds"),
            patch("macaw.scheduler.scheduler.STTWorkerStub") as mock_stub_cls,
        ):
            mock_gauge.labels.return_value = mock_child
            mock_stub = AsyncMock()
            mock_stub.TranscribeFile = AsyncMock(
                return_value=_make_proto_response(),
            )
            mock_stub_cls.return_value = mock_stub

            await scheduler.start()
            await scheduler.submit(_make_request(), RequestPriority.BATCH)
            await asyncio.sleep(0.3)
            await scheduler.stop()

            # .dec() should have been called (dispatch dequeued the request)
            dec_calls = [c for c in mock_child.method_calls if c[0] == "dec"]
            assert len(dec_calls) >= 1

    async def test_cancel_decrements_queue_depth(self) -> None:
        """cancel() decrements queue_depth if request was in queue."""
        scheduler, _, _ = _make_scheduler()

        mock_child = MagicMock()
        with patch("macaw.scheduler.scheduler.scheduler_queue_depth") as mock_gauge:
            mock_gauge.labels.return_value = mock_child

            await scheduler.submit(
                _make_request("req_cancel"),
                RequestPriority.BATCH,
            )

            # Cancel before dispatch
            result = scheduler.cancel("req_cancel")
            assert result is True

            # dec() should be called (once for cancel removing from queue)
            dec_calls = [c for c in mock_child.method_calls if c[0] == "dec"]
            assert len(dec_calls) >= 1


# ---------------------------------------------------------------------------
# Queue wait seconds metric tests
# ---------------------------------------------------------------------------


class TestQueueWaitMetric:
    async def test_dequeue_observes_queue_wait(self) -> None:
        """Dispatch loop observes queue_wait_seconds on dequeue."""
        worker = _make_worker()
        scheduler, _, _ = _make_scheduler(worker)

        with (
            patch("macaw.scheduler.scheduler.scheduler_queue_depth"),
            patch("macaw.scheduler.scheduler.scheduler_queue_wait_seconds") as mock_hist,
            patch("macaw.scheduler.scheduler.scheduler_requests_total"),
            patch("macaw.scheduler.scheduler.scheduler_grpc_duration_seconds"),
            patch("macaw.scheduler.scheduler.STTWorkerStub") as mock_stub_cls,
        ):
            mock_stub = AsyncMock()
            mock_stub.TranscribeFile = AsyncMock(
                return_value=_make_proto_response(),
            )
            mock_stub_cls.return_value = mock_stub

            await scheduler.start()
            await scheduler.submit(_make_request(), RequestPriority.BATCH)
            await asyncio.sleep(0.3)
            await scheduler.stop()

            mock_hist.observe.assert_called()
            # Wait time should be non-negative
            observed_value = mock_hist.observe.call_args[0][0]
            assert observed_value >= 0


# ---------------------------------------------------------------------------
# gRPC duration metric tests
# ---------------------------------------------------------------------------


class TestGRPCDurationMetric:
    async def test_dispatch_observes_grpc_duration(self) -> None:
        """Dispatch observes grpc_duration_seconds after gRPC call."""
        worker = _make_worker()
        scheduler, _, _ = _make_scheduler(worker)

        with (
            patch("macaw.scheduler.scheduler.scheduler_queue_depth"),
            patch("macaw.scheduler.scheduler.scheduler_queue_wait_seconds"),
            patch("macaw.scheduler.scheduler.scheduler_requests_total"),
            patch("macaw.scheduler.scheduler.scheduler_grpc_duration_seconds") as mock_hist,
            patch("macaw.scheduler.scheduler.STTWorkerStub") as mock_stub_cls,
        ):
            mock_stub = AsyncMock()
            mock_stub.TranscribeFile = AsyncMock(
                return_value=_make_proto_response(),
            )
            mock_stub_cls.return_value = mock_stub

            await scheduler.start()
            await scheduler.submit(_make_request(), RequestPriority.REALTIME)
            await asyncio.sleep(0.3)
            await scheduler.stop()

            mock_hist.observe.assert_called()
            observed_value = mock_hist.observe.call_args[0][0]
            assert observed_value >= 0


# ---------------------------------------------------------------------------
# Batch size metric tests
# ---------------------------------------------------------------------------


class TestBatchSizeMetric:
    async def test_batch_dispatch_observes_batch_size(self) -> None:
        """Batch dispatch observes batch_size histogram."""
        worker = _make_worker()
        scheduler, _, _ = _make_scheduler(worker)

        with (
            patch("macaw.scheduler.scheduler.scheduler_queue_depth"),
            patch("macaw.scheduler.scheduler.scheduler_queue_wait_seconds"),
            patch("macaw.scheduler.scheduler.scheduler_requests_total"),
            patch("macaw.scheduler.scheduler.scheduler_grpc_duration_seconds"),
            patch("macaw.scheduler.scheduler.scheduler_batch_size") as mock_hist,
            patch("macaw.scheduler.scheduler.STTWorkerStub") as mock_stub_cls,
        ):
            mock_stub = AsyncMock()
            mock_stub.TranscribeFile = AsyncMock(
                return_value=_make_proto_response(),
            )
            mock_stub_cls.return_value = mock_stub

            await scheduler.start()
            # Submit 3 BATCH requests (accumulator will batch them)
            for i in range(3):
                await scheduler.submit(
                    _make_request(f"req_batch_{i}"),
                    RequestPriority.BATCH,
                )
            # Wait for accumulator flush + dispatch
            await asyncio.sleep(0.5)
            await scheduler.stop()

            # batch_size.observe should have been called at least once
            assert mock_hist.observe.called


# ---------------------------------------------------------------------------
# Requests total metric tests
# ---------------------------------------------------------------------------


class TestRequestsTotalMetric:
    async def test_successful_request_increments_ok(self) -> None:
        """Successful dispatch increments requests_total with status=ok."""
        worker = _make_worker()
        scheduler, _, _ = _make_scheduler(worker)

        with (
            patch("macaw.scheduler.scheduler.scheduler_queue_depth"),
            patch("macaw.scheduler.scheduler.scheduler_queue_wait_seconds"),
            patch("macaw.scheduler.scheduler.scheduler_requests_total") as mock_counter,
            patch("macaw.scheduler.scheduler.scheduler_grpc_duration_seconds"),
            patch("macaw.scheduler.scheduler.STTWorkerStub") as mock_stub_cls,
        ):
            mock_child = MagicMock()
            mock_counter.labels.return_value = mock_child
            mock_stub = AsyncMock()
            mock_stub.TranscribeFile = AsyncMock(
                return_value=_make_proto_response(),
            )
            mock_stub_cls.return_value = mock_stub

            await scheduler.start()
            await scheduler.submit(_make_request(), RequestPriority.REALTIME)
            await asyncio.sleep(0.3)
            await scheduler.stop()

            # Should have been called with status="ok"
            mock_counter.labels.assert_any_call(
                priority="REALTIME",
                status="ok",
            )

    async def test_cancelled_request_increments_cancelled(self) -> None:
        """Request cancelled in queue increments requests_total with status=cancelled."""
        scheduler, _, _ = _make_scheduler()

        with (
            patch("macaw.scheduler.scheduler.scheduler_queue_depth"),
            patch("macaw.scheduler.scheduler.scheduler_queue_wait_seconds"),
            patch("macaw.scheduler.scheduler.scheduler_requests_total") as mock_counter,
        ):
            mock_child = MagicMock()
            mock_counter.labels.return_value = mock_child

            await scheduler.start()
            await scheduler.submit(
                _make_request("req_cancel2"),
                RequestPriority.BATCH,
            )
            scheduler.cancel("req_cancel2")
            # Let dispatch loop pick it up and skip
            await asyncio.sleep(0.3)
            await scheduler.stop()

            # Should have been called with status="cancelled"
            calls = [
                c
                for c in mock_counter.labels.call_args_list
                if c == ((), {"priority": "BATCH", "status": "cancelled"})
            ]
            assert len(calls) >= 1

    async def test_error_request_increments_error(self) -> None:
        """gRPC error increments requests_total with status=error."""
        import grpc.aio

        worker = _make_worker()
        scheduler, _, _ = _make_scheduler(worker)

        with (
            patch("macaw.scheduler.scheduler.scheduler_queue_depth"),
            patch("macaw.scheduler.scheduler.scheduler_queue_wait_seconds"),
            patch("macaw.scheduler.scheduler.scheduler_requests_total") as mock_counter,
            patch("macaw.scheduler.scheduler.scheduler_grpc_duration_seconds"),
            patch("macaw.scheduler.scheduler.STTWorkerStub") as mock_stub_cls,
        ):
            mock_child = MagicMock()
            mock_counter.labels.return_value = mock_child
            mock_stub = AsyncMock()
            # Simulate gRPC error
            mock_error = grpc.aio.AioRpcError(
                code=grpc.StatusCode.INTERNAL,
                initial_metadata=grpc.aio.Metadata(),
                trailing_metadata=grpc.aio.Metadata(),
                details="internal error",
            )
            mock_stub.TranscribeFile = AsyncMock(side_effect=mock_error)
            mock_stub_cls.return_value = mock_stub

            await scheduler.start()
            await scheduler.submit(
                _make_request(),
                RequestPriority.REALTIME,
            )
            await asyncio.sleep(0.3)
            await scheduler.stop()

            mock_counter.labels.assert_any_call(
                priority="REALTIME",
                status="error",
            )


# ---------------------------------------------------------------------------
# Cancel latency metric tests
# ---------------------------------------------------------------------------


class TestCancelLatencyMetric:
    async def test_cancel_in_flight_observes_latency(self) -> None:
        """cancel_in_flight() observes cancel_latency_seconds."""
        from macaw.scheduler.cancel import CancellationManager

        mgr = CancellationManager()

        with patch("macaw.scheduler.cancel.scheduler_cancel_latency_seconds") as mock_hist:
            # Simulate gRPC cancel call — mock the channel and stub
            mock_channel = AsyncMock()
            mock_stub_instance = AsyncMock()
            mock_stub_instance.Cancel = AsyncMock(
                return_value=MagicMock(acknowledged=True),
            )

            with patch("macaw.scheduler.cancel.STTWorkerStub", return_value=mock_stub_instance):
                result = await mgr.cancel_in_flight(
                    "req_1",
                    "localhost:50051",
                    mock_channel,
                )

            assert result is True
            mock_hist.observe.assert_called_once()
            observed_value = mock_hist.observe.call_args[0][0]
            assert observed_value >= 0


# ---------------------------------------------------------------------------
# Aging promotions metric tests
# ---------------------------------------------------------------------------


class TestAgingPromotionsMetric:
    async def test_aged_request_increments_counter(self) -> None:
        """Aged request increments aging_promotions_total."""
        worker = _make_worker()
        # Use very short aging threshold to trigger aging
        scheduler, _, _ = _make_scheduler(worker, aging_threshold_s=0.001)

        with (
            patch("macaw.scheduler.scheduler.scheduler_queue_depth"),
            patch("macaw.scheduler.scheduler.scheduler_queue_wait_seconds"),
            patch("macaw.scheduler.scheduler.scheduler_requests_total"),
            patch("macaw.scheduler.scheduler.scheduler_grpc_duration_seconds"),
            patch("macaw.scheduler.scheduler.scheduler_aging_promotions_total") as mock_counter,
            patch("macaw.scheduler.scheduler.STTWorkerStub") as mock_stub_cls,
        ):
            mock_stub = AsyncMock()
            mock_stub.TranscribeFile = AsyncMock(
                return_value=_make_proto_response(),
            )
            mock_stub_cls.return_value = mock_stub

            # Submit BATCH request, then start scheduler after a small delay
            # so the request ages before being dequeued
            await scheduler.submit(_make_request(), RequestPriority.BATCH)
            await asyncio.sleep(0.01)  # Let it age past 1ms threshold
            await scheduler.start()
            await asyncio.sleep(0.3)
            await scheduler.stop()

            mock_counter.inc.assert_called()


# ---------------------------------------------------------------------------
# No-op without prometheus tests
# ---------------------------------------------------------------------------


class TestNoOpWithoutPrometheus:
    def test_metrics_null_without_prometheus(self) -> None:
        """When prometheus_client is not importable, metrics are NullMetric."""
        import importlib
        import sys

        from prometheus_client import REGISTRY

        import macaw.scheduler.metrics as metrics_mod
        from macaw._null_metrics import NullMetric

        # Unregister existing metrics to avoid "duplicated timeseries" on reload
        metric_names = [
            "macaw_scheduler_queue_depth",
            "macaw_scheduler_queue_wait_seconds",
            "macaw_scheduler_grpc_duration_seconds",
            "macaw_scheduler_cancel_latency_seconds",
            "macaw_scheduler_batch_size",
            "macaw_scheduler_requests_total",
            "macaw_scheduler_aging_promotions_total",
        ]
        for name in metric_names:
            with contextlib.suppress(KeyError):
                REGISTRY.unregister(REGISTRY._names_to_collectors[name])

        try:
            # Simulate prometheus_client not installed
            saved_module = sys.modules.get("prometheus_client")
            sys.modules["prometheus_client"] = None  # type: ignore[assignment]

            importlib.reload(metrics_mod)

            assert isinstance(metrics_mod.scheduler_queue_depth, NullMetric)
            assert isinstance(metrics_mod.scheduler_queue_wait_seconds, NullMetric)
            assert isinstance(metrics_mod.scheduler_grpc_duration_seconds, NullMetric)
            assert isinstance(metrics_mod.scheduler_cancel_latency_seconds, NullMetric)
            assert isinstance(metrics_mod.scheduler_batch_size, NullMetric)
            assert isinstance(metrics_mod.scheduler_requests_total, NullMetric)
            assert isinstance(metrics_mod.scheduler_aging_promotions_total, NullMetric)
        finally:
            # Restore prometheus_client and reload to re-register metrics
            if saved_module is not None:
                sys.modules["prometheus_client"] = saved_module
            else:
                sys.modules.pop("prometheus_client", None)
            importlib.reload(metrics_mod)

    def test_scheduler_works_without_metrics(self) -> None:
        """Scheduler works when metrics are NullMetric (no prometheus)."""
        from macaw._null_metrics import NullMetric

        null = NullMetric()
        with (
            patch("macaw.scheduler.scheduler.scheduler_queue_depth", null),
            patch("macaw.scheduler.scheduler.scheduler_queue_wait_seconds", null),
            patch("macaw.scheduler.scheduler.scheduler_grpc_duration_seconds", null),
            patch("macaw.scheduler.scheduler.scheduler_requests_total", null),
            patch("macaw.scheduler.scheduler.scheduler_aging_promotions_total", null),
            patch("macaw.scheduler.scheduler.scheduler_batch_size", null),
        ):
            scheduler, _, _ = _make_scheduler()
            # Should not raise
            assert scheduler is not None
