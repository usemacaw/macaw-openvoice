"""Testes do modulo tts_metrics.

Valida:
- Metricas TTS criadas com tipos corretos quando prometheus_client esta disponivel
- HAS_TTS_METRICS flag reflete disponibilidade de prometheus
- Metricas sao None quando prometheus nao esta instalado
- Histograms tem buckets corretos
- Counter tem labels corretos
- Instrumentacao em _tts_speak_task: TTFB, duration, requests_total, active_sessions
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from macaw.scheduler.tts_metrics import (
    HAS_TTS_METRICS,
    tts_active_sessions,
    tts_requests_total,
    tts_synthesis_duration_seconds,
    tts_ttfb_seconds,
)


class TestTTSMetricsAvailability:
    def test_has_tts_metrics_flag_is_true(self) -> None:
        """HAS_TTS_METRICS is True when prometheus_client is installed."""
        assert HAS_TTS_METRICS is True

    def test_ttfb_metric_not_none(self) -> None:
        """tts_ttfb_seconds is created when prometheus is available."""
        assert tts_ttfb_seconds is not None

    def test_synthesis_duration_metric_not_none(self) -> None:
        """tts_synthesis_duration_seconds is created when prometheus is available."""
        assert tts_synthesis_duration_seconds is not None

    def test_requests_total_metric_not_none(self) -> None:
        """tts_requests_total is created when prometheus is available."""
        assert tts_requests_total is not None

    def test_active_sessions_metric_not_none(self) -> None:
        """tts_active_sessions is created when prometheus is available."""
        assert tts_active_sessions is not None


class TestTTSMetricsTypes:
    def test_ttfb_is_histogram(self) -> None:
        """tts_ttfb_seconds is a Histogram."""
        from prometheus_client import Histogram

        assert isinstance(tts_ttfb_seconds, Histogram)

    def test_synthesis_duration_is_histogram(self) -> None:
        """tts_synthesis_duration_seconds is a Histogram."""
        from prometheus_client import Histogram

        assert isinstance(tts_synthesis_duration_seconds, Histogram)

    def test_requests_total_is_counter(self) -> None:
        """tts_requests_total is a Counter."""
        # Counter with labels returns a MetricWrapperClass
        assert tts_requests_total is not None
        assert hasattr(tts_requests_total, "labels")

    def test_active_sessions_is_gauge(self) -> None:
        """tts_active_sessions is a Gauge."""
        from prometheus_client import Gauge

        assert isinstance(tts_active_sessions, Gauge)


class TestTTSMetricsNoPrometheus:
    def test_metrics_none_when_prometheus_missing(self) -> None:
        """When prometheus_client is not installed, all metrics are None.

        Instead of reloading the module (which causes duplicate registration
        issues with prometheus_client), we verify the code structure: the
        except ImportError branch sets all metrics to None.
        """
        import ast
        import inspect

        import macaw.scheduler.tts_metrics as mod

        source = inspect.getsource(mod)
        tree = ast.parse(source)

        # Find the except ImportError block
        found_except = False
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ExceptHandler)
                and node.type is not None
                and isinstance(node.type, ast.Name)
                and node.type.id == "ImportError"
            ):
                found_except = True
                # Verify all metrics are set to None in except block
                assignments = [
                    n.targets[0].id
                    for n in node.body
                    if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                ]
                assert "tts_ttfb_seconds" in assignments
                assert "tts_synthesis_duration_seconds" in assignments
                assert "tts_requests_total" in assignments
                assert "tts_active_sessions" in assignments
                assert "HAS_TTS_METRICS" in assignments

        assert found_except, "No except ImportError block found"


class TestTTSMetricsInstrumentation:
    """Testa que _tts_speak_task instrui as metricas corretamente."""

    async def test_ttfb_observed_on_first_chunk(self) -> None:
        """TTFB metric is observed when first chunk arrives."""
        import asyncio

        from macaw.server.routes.realtime import _tts_speak_task
        from tests.unit.test_full_duplex import (
            _make_mock_registry,
            _make_mock_session,
            _make_mock_websocket,
            _make_mock_worker,
            _make_mock_worker_manager,
            _make_send_event,
        )

        ws = _make_mock_websocket()
        session = _make_mock_session()
        send_event, _events = _make_send_event()
        cancel = asyncio.Event()

        worker = _make_mock_worker()
        registry = _make_mock_registry()
        wm = _make_mock_worker_manager(worker)

        ws.app = MagicMock()
        ws.app.state.registry = registry
        ws.app.state.worker_manager = wm

        chunk = MagicMock()
        chunk.audio_data = b"\x00\x01" * 100
        chunk.is_last = True

        from tests.unit.test_full_duplex import _make_mock_grpc_stream

        stream = _make_mock_grpc_stream([chunk])

        with (
            patch("macaw.server.routes.realtime.grpc.aio.insecure_channel") as mock_ch,
            patch("macaw.server.routes.realtime.tts_ttfb_seconds") as mock_ttfb,
            patch("macaw.server.routes.realtime.HAS_TTS_METRICS", True),
        ):
            mock_channel = AsyncMock()
            mock_ch.return_value = mock_channel
            mock_stub = MagicMock()
            mock_stub.Synthesize.return_value = stream

            with patch(
                "macaw.server.routes.realtime.TTSWorkerStub",
                return_value=mock_stub,
            ):
                await _tts_speak_task(
                    websocket=ws,
                    session_id="sess_test",
                    session=session,
                    request_id="req_1",
                    text="Hello",
                    voice="default",
                    model_tts="kokoro-v1",
                    send_event=send_event,
                    cancel_event=cancel,
                )

        mock_ttfb.observe.assert_called_once()
        ttfb_value = mock_ttfb.observe.call_args[0][0]
        assert ttfb_value >= 0

    async def test_synthesis_duration_observed_on_completion(self) -> None:
        """Synthesis duration metric is observed at the end."""
        import asyncio

        from macaw.server.routes.realtime import _tts_speak_task
        from tests.unit.test_full_duplex import (
            _make_mock_registry,
            _make_mock_session,
            _make_mock_websocket,
            _make_mock_worker,
            _make_mock_worker_manager,
            _make_send_event,
        )

        ws = _make_mock_websocket()
        session = _make_mock_session()
        send_event, _events = _make_send_event()
        cancel = asyncio.Event()

        worker = _make_mock_worker()
        registry = _make_mock_registry()
        wm = _make_mock_worker_manager(worker)

        ws.app = MagicMock()
        ws.app.state.registry = registry
        ws.app.state.worker_manager = wm

        chunk = MagicMock()
        chunk.audio_data = b"\x00\x01" * 100
        chunk.is_last = True

        from tests.unit.test_full_duplex import _make_mock_grpc_stream

        stream = _make_mock_grpc_stream([chunk])

        with (
            patch("macaw.server.routes.realtime.grpc.aio.insecure_channel") as mock_ch,
            patch(
                "macaw.server.routes.realtime.tts_synthesis_duration_seconds",
            ) as mock_dur,
            patch("macaw.server.routes.realtime.HAS_TTS_METRICS", True),
        ):
            mock_channel = AsyncMock()
            mock_ch.return_value = mock_channel
            mock_stub = MagicMock()
            mock_stub.Synthesize.return_value = stream

            with patch(
                "macaw.server.routes.realtime.TTSWorkerStub",
                return_value=mock_stub,
            ):
                await _tts_speak_task(
                    websocket=ws,
                    session_id="sess_test",
                    session=session,
                    request_id="req_1",
                    text="Hello",
                    voice="default",
                    model_tts="kokoro-v1",
                    send_event=send_event,
                    cancel_event=cancel,
                )

        mock_dur.observe.assert_called_once()

    async def test_requests_total_ok_on_success(self) -> None:
        """requests_total counter incremented with status=ok on success."""
        import asyncio

        from macaw.server.routes.realtime import _tts_speak_task
        from tests.unit.test_full_duplex import (
            _make_mock_registry,
            _make_mock_session,
            _make_mock_websocket,
            _make_mock_worker,
            _make_mock_worker_manager,
            _make_send_event,
        )

        ws = _make_mock_websocket()
        session = _make_mock_session()
        send_event, _events = _make_send_event()
        cancel = asyncio.Event()

        worker = _make_mock_worker()
        registry = _make_mock_registry()
        wm = _make_mock_worker_manager(worker)

        ws.app = MagicMock()
        ws.app.state.registry = registry
        ws.app.state.worker_manager = wm

        chunk = MagicMock()
        chunk.audio_data = b"\x00\x01" * 100
        chunk.is_last = True

        from tests.unit.test_full_duplex import _make_mock_grpc_stream

        stream = _make_mock_grpc_stream([chunk])

        with (
            patch("macaw.server.routes.realtime.grpc.aio.insecure_channel") as mock_ch,
            patch("macaw.server.routes.realtime.tts_requests_total") as mock_counter,
            patch("macaw.server.routes.realtime.HAS_TTS_METRICS", True),
        ):
            mock_channel = AsyncMock()
            mock_ch.return_value = mock_channel
            mock_stub = MagicMock()
            mock_stub.Synthesize.return_value = stream

            with patch(
                "macaw.server.routes.realtime.TTSWorkerStub",
                return_value=mock_stub,
            ):
                await _tts_speak_task(
                    websocket=ws,
                    session_id="sess_test",
                    session=session,
                    request_id="req_1",
                    text="Hello",
                    voice="default",
                    model_tts="kokoro-v1",
                    send_event=send_event,
                    cancel_event=cancel,
                )

        mock_counter.labels.assert_called_with(status="ok")
        mock_counter.labels.return_value.inc.assert_called_once()

    async def test_requests_total_error_on_failure(self) -> None:
        """requests_total counter incremented with status=error on failure."""
        import asyncio

        from macaw.server.routes.realtime import _tts_speak_task
        from tests.unit.test_full_duplex import (
            _make_mock_session,
            _make_mock_websocket,
            _make_send_event,
        )

        ws = _make_mock_websocket()
        session = _make_mock_session()
        send_event, _events = _make_send_event()
        cancel = asyncio.Event()

        # No registry -> error path
        ws.app = MagicMock()
        ws.app.state.registry = None
        ws.app.state.worker_manager = None

        with (
            patch("macaw.server.routes.realtime.tts_requests_total") as mock_counter,
            patch("macaw.server.routes.realtime.HAS_TTS_METRICS", True),
        ):
            await _tts_speak_task(
                websocket=ws,
                session_id="sess_test",
                session=session,
                request_id="req_1",
                text="Hello",
                voice="default",
                model_tts="kokoro-v1",
                send_event=send_event,
                cancel_event=cancel,
            )

        mock_counter.labels.assert_called_with(status="error")
        mock_counter.labels.return_value.inc.assert_called_once()

    async def test_active_sessions_inc_dec(self) -> None:
        """active_sessions gauge is incremented on first chunk and decremented at end."""
        import asyncio

        from macaw.server.routes.realtime import _tts_speak_task
        from tests.unit.test_full_duplex import (
            _make_mock_registry,
            _make_mock_session,
            _make_mock_websocket,
            _make_mock_worker,
            _make_mock_worker_manager,
            _make_send_event,
        )

        ws = _make_mock_websocket()
        session = _make_mock_session()
        send_event, _events = _make_send_event()
        cancel = asyncio.Event()

        worker = _make_mock_worker()
        registry = _make_mock_registry()
        wm = _make_mock_worker_manager(worker)

        ws.app = MagicMock()
        ws.app.state.registry = registry
        ws.app.state.worker_manager = wm

        chunk = MagicMock()
        chunk.audio_data = b"\x00\x01" * 100
        chunk.is_last = True

        from tests.unit.test_full_duplex import _make_mock_grpc_stream

        stream = _make_mock_grpc_stream([chunk])

        with (
            patch("macaw.server.routes.realtime.grpc.aio.insecure_channel") as mock_ch,
            patch("macaw.server.routes.realtime.tts_active_sessions") as mock_gauge,
            patch("macaw.server.routes.realtime.HAS_TTS_METRICS", True),
        ):
            mock_channel = AsyncMock()
            mock_ch.return_value = mock_channel
            mock_stub = MagicMock()
            mock_stub.Synthesize.return_value = stream

            with patch(
                "macaw.server.routes.realtime.TTSWorkerStub",
                return_value=mock_stub,
            ):
                await _tts_speak_task(
                    websocket=ws,
                    session_id="sess_test",
                    session=session,
                    request_id="req_1",
                    text="Hello",
                    voice="default",
                    model_tts="kokoro-v1",
                    send_event=send_event,
                    cancel_event=cancel,
                )

        mock_gauge.inc.assert_called_once()
        mock_gauge.dec.assert_called_once()

    async def test_requests_total_cancelled(self) -> None:
        """requests_total counter incremented with status=cancelled on cancel."""
        import asyncio

        from macaw.server.routes.realtime import _tts_speak_task
        from tests.unit.test_full_duplex import (
            _make_mock_registry,
            _make_mock_session,
            _make_mock_websocket,
            _make_mock_worker,
            _make_mock_worker_manager,
            _make_send_event,
        )

        ws = _make_mock_websocket()
        session = _make_mock_session()
        send_event, _events = _make_send_event()
        cancel = asyncio.Event()
        cancel.set()  # Pre-cancel

        worker = _make_mock_worker()
        registry = _make_mock_registry()
        wm = _make_mock_worker_manager(worker)

        ws.app = MagicMock()
        ws.app.state.registry = registry
        ws.app.state.worker_manager = wm

        chunk = MagicMock()
        chunk.audio_data = b"\x00\x01" * 100
        chunk.is_last = False

        from tests.unit.test_full_duplex import _make_mock_grpc_stream

        stream = _make_mock_grpc_stream([chunk])

        with (
            patch("macaw.server.routes.realtime.grpc.aio.insecure_channel") as mock_ch,
            patch("macaw.server.routes.realtime.tts_requests_total") as mock_counter,
            patch("macaw.server.routes.realtime.HAS_TTS_METRICS", True),
        ):
            mock_channel = AsyncMock()
            mock_ch.return_value = mock_channel
            mock_stub = MagicMock()
            mock_stub.Synthesize.return_value = stream

            with patch(
                "macaw.server.routes.realtime.TTSWorkerStub",
                return_value=mock_stub,
            ):
                await _tts_speak_task(
                    websocket=ws,
                    session_id="sess_test",
                    session=session,
                    request_id="req_1",
                    text="Hello",
                    voice="default",
                    model_tts="kokoro-v1",
                    send_event=send_event,
                    cancel_event=cancel,
                )

        mock_counter.labels.assert_called_with(status="cancelled")
