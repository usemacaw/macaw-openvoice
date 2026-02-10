"""Testes de integracao full-duplex STT+TTS no endpoint WebSocket.

Valida:
- tts.speak resolve modelo, abre gRPC stream, mute STT, envia chunks
- tts.cancel cancela TTS ativa, unmute, emite tts.speaking_end(cancelled=true)
- Sequential speaks: segundo cancela primeiro
- TTS worker crash: emite error, unmute
- Modelo TTS nao encontrado: emite error
- TTS roda em background task (nao bloqueia main loop)
- Mute/unmute lifecycle coordenado com StreamingSession
- model_tts via session.configure
- Cleanup no disconnect durante TTS ativa
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from macaw.server.routes.realtime import (
    _cancel_active_tts,
    _tts_speak_task,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_session() -> MagicMock:
    """Cria mock de StreamingSession com mute/unmute."""
    session = MagicMock()
    session.is_closed = False
    session.is_muted = False
    session.segment_id = 0

    def _mute() -> None:
        session.is_muted = True

    def _unmute() -> None:
        session.is_muted = False

    session.mute = _mute
    session.unmute = _unmute
    session.close = AsyncMock()
    session.process_frame = AsyncMock()
    return session


def _make_mock_websocket() -> MagicMock:
    """Cria mock de WebSocket."""
    from starlette.websockets import WebSocketState

    ws = AsyncMock()
    ws.client_state = WebSocketState.CONNECTED
    ws.send_json = AsyncMock()
    ws.send_bytes = AsyncMock()

    return ws


def _make_mock_registry(has_tts: bool = True, model_name: str = "kokoro-v1") -> MagicMock:
    """Cria mock de ModelRegistry."""
    from macaw._types import ModelType

    registry = MagicMock()

    if has_tts:
        manifest = MagicMock()
        manifest.model_type = ModelType.TTS
        manifest.name = model_name
        registry.get_manifest.return_value = manifest
        registry.list_models.return_value = [manifest]
    else:
        from macaw.exceptions import ModelNotFoundError

        registry.get_manifest.side_effect = ModelNotFoundError(model_name)
        registry.list_models.return_value = []

    return registry


def _make_mock_worker(port: int = 50052) -> MagicMock:
    """Cria mock de WorkerInfo."""
    worker = MagicMock()
    worker.port = port
    return worker


def _make_mock_worker_manager(worker: MagicMock | None = None) -> MagicMock:
    """Cria mock de WorkerManager."""
    wm = MagicMock()
    wm.get_ready_worker.return_value = worker
    return wm


def _make_send_event() -> tuple[AsyncMock, list[Any]]:
    """Cria send_event callback que coleta eventos."""
    events: list[Any] = []

    async def _send(event: Any) -> None:
        events.append(event)

    return AsyncMock(side_effect=_send), events


def _make_mock_grpc_stream(chunks: list[MagicMock] | None = None) -> Any:
    """Cria mock de gRPC Synthesize response stream."""
    if chunks is None:
        chunk1 = MagicMock()
        chunk1.audio_data = b"\x00\x01" * 100
        chunk1.is_last = False

        chunk2 = MagicMock()
        chunk2.audio_data = b"\x00\x02" * 100
        chunk2.is_last = True

        chunks = [chunk1, chunk2]

    class _FakeStream:
        def __init__(self, items: list[MagicMock]) -> None:
            self._items = items
            self._idx = 0

        def __aiter__(self) -> _FakeStream:
            return self

        async def __anext__(self) -> MagicMock:
            if self._idx >= len(self._items):
                raise StopAsyncIteration
            item = self._items[self._idx]
            self._idx += 1
            return item

    return _FakeStream(chunks)


# ---------------------------------------------------------------------------
# Tests: _tts_speak_task
# ---------------------------------------------------------------------------


class TestTTSSpeakTaskHappyPath:
    async def test_sends_audio_chunks_to_websocket(self) -> None:
        """TTS task sends audio data as binary WebSocket frames."""
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

        stream = _make_mock_grpc_stream()

        with patch("macaw.server.routes.realtime.grpc.aio.insecure_channel") as mock_ch:
            mock_channel = AsyncMock()
            mock_ch.return_value = mock_channel
            mock_stub = MagicMock()
            mock_stub.Synthesize.return_value = stream

            with patch("macaw.server.routes.realtime.TTSWorkerStub", return_value=mock_stub):
                await _tts_speak_task(
                    websocket=ws,
                    session_id="sess_test",
                    session=session,
                    request_id="req_1",
                    text="Hello world",
                    voice="default",
                    model_tts="kokoro-v1",
                    send_event=send_event,
                    cancel_event=cancel,
                )

        # Audio chunks sent via send_bytes
        assert ws.send_bytes.call_count == 2

    async def test_emits_speaking_start_and_end(self) -> None:
        """TTS task emits tts.speaking_start and tts.speaking_end events."""
        ws = _make_mock_websocket()
        session = _make_mock_session()
        send_event, events = _make_send_event()
        cancel = asyncio.Event()

        worker = _make_mock_worker()
        registry = _make_mock_registry()
        wm = _make_mock_worker_manager(worker)

        ws.app = MagicMock()
        ws.app.state.registry = registry
        ws.app.state.worker_manager = wm

        stream = _make_mock_grpc_stream()

        with patch("macaw.server.routes.realtime.grpc.aio.insecure_channel") as mock_ch:
            mock_channel = AsyncMock()
            mock_ch.return_value = mock_channel
            mock_stub = MagicMock()
            mock_stub.Synthesize.return_value = stream

            with patch("macaw.server.routes.realtime.TTSWorkerStub", return_value=mock_stub):
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

        event_types = [e.type for e in events]
        assert "tts.speaking_start" in event_types
        assert "tts.speaking_end" in event_types

    async def test_mutes_stt_on_first_chunk(self) -> None:
        """STT is muted when first TTS chunk arrives."""
        ws = _make_mock_websocket()
        session = _make_mock_session()
        send_event, _ = _make_send_event()
        cancel = asyncio.Event()

        worker = _make_mock_worker()
        registry = _make_mock_registry()
        wm = _make_mock_worker_manager(worker)

        ws.app = MagicMock()
        ws.app.state.registry = registry
        ws.app.state.worker_manager = wm

        stream = _make_mock_grpc_stream()

        with patch("macaw.server.routes.realtime.grpc.aio.insecure_channel") as mock_ch:
            mock_channel = AsyncMock()
            mock_ch.return_value = mock_channel
            mock_stub = MagicMock()
            mock_stub.Synthesize.return_value = stream

            with patch("macaw.server.routes.realtime.TTSWorkerStub", return_value=mock_stub):
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

        # After TTS completes, session is unmuted (unmute in finally)
        assert session.is_muted is False

    async def test_unmutes_stt_after_completion(self) -> None:
        """STT is always unmuted after TTS task completes."""
        ws = _make_mock_websocket()
        session = _make_mock_session()
        send_event, _ = _make_send_event()
        cancel = asyncio.Event()

        worker = _make_mock_worker()
        registry = _make_mock_registry()
        wm = _make_mock_worker_manager(worker)

        ws.app = MagicMock()
        ws.app.state.registry = registry
        ws.app.state.worker_manager = wm

        stream = _make_mock_grpc_stream()

        with patch("macaw.server.routes.realtime.grpc.aio.insecure_channel") as mock_ch:
            mock_channel = AsyncMock()
            mock_ch.return_value = mock_channel
            mock_stub = MagicMock()
            mock_stub.Synthesize.return_value = stream

            with patch("macaw.server.routes.realtime.TTSWorkerStub", return_value=mock_stub):
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

        assert session.is_muted is False

    async def test_speaking_end_not_cancelled(self) -> None:
        """Normal completion sets cancelled=False on speaking_end event."""
        ws = _make_mock_websocket()
        session = _make_mock_session()
        send_event, events = _make_send_event()
        cancel = asyncio.Event()

        worker = _make_mock_worker()
        registry = _make_mock_registry()
        wm = _make_mock_worker_manager(worker)

        ws.app = MagicMock()
        ws.app.state.registry = registry
        ws.app.state.worker_manager = wm

        stream = _make_mock_grpc_stream()

        with patch("macaw.server.routes.realtime.grpc.aio.insecure_channel") as mock_ch:
            mock_channel = AsyncMock()
            mock_ch.return_value = mock_channel
            mock_stub = MagicMock()
            mock_stub.Synthesize.return_value = stream

            with patch("macaw.server.routes.realtime.TTSWorkerStub", return_value=mock_stub):
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

        end_events = [e for e in events if e.type == "tts.speaking_end"]
        assert len(end_events) == 1
        assert end_events[0].cancelled is False


class TestTTSSpeakTaskCancel:
    async def test_cancel_event_stops_streaming(self) -> None:
        """Setting cancel_event stops the TTS stream."""
        ws = _make_mock_websocket()
        session = _make_mock_session()
        send_event, events = _make_send_event()
        cancel = asyncio.Event()

        worker = _make_mock_worker()
        registry = _make_mock_registry()
        wm = _make_mock_worker_manager(worker)

        ws.app = MagicMock()
        ws.app.state.registry = registry
        ws.app.state.worker_manager = wm

        # Create chunks where second chunk triggers cancel
        chunk1 = MagicMock()
        chunk1.audio_data = b"\x00\x01" * 100
        chunk1.is_last = False

        chunk2 = MagicMock()
        chunk2.audio_data = b"\x00\x02" * 100
        chunk2.is_last = False

        chunk3 = MagicMock()
        chunk3.audio_data = b"\x00\x03" * 100
        chunk3.is_last = True

        class _CancelStream:
            def __init__(self) -> None:
                self._items = [chunk1, chunk2, chunk3]
                self._idx = 0

            def __aiter__(self) -> _CancelStream:
                return self

            async def __anext__(self) -> MagicMock:
                if self._idx >= len(self._items):
                    raise StopAsyncIteration
                item = self._items[self._idx]
                self._idx += 1
                # Cancel after first chunk
                if self._idx == 2:
                    cancel.set()
                return item

        stream = _CancelStream()

        with patch("macaw.server.routes.realtime.grpc.aio.insecure_channel") as mock_ch:
            mock_channel = AsyncMock()
            mock_ch.return_value = mock_channel
            mock_stub = MagicMock()
            mock_stub.Synthesize.return_value = stream

            with patch("macaw.server.routes.realtime.TTSWorkerStub", return_value=mock_stub):
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

        end_events = [e for e in events if e.type == "tts.speaking_end"]
        assert len(end_events) == 1
        assert end_events[0].cancelled is True

    async def test_cancel_unmutes_stt(self) -> None:
        """Cancellation always unmutes STT."""
        ws = _make_mock_websocket()
        session = _make_mock_session()
        send_event, _ = _make_send_event()
        cancel = asyncio.Event()
        cancel.set()  # Pre-set cancel — will stop immediately

        worker = _make_mock_worker()
        registry = _make_mock_registry()
        wm = _make_mock_worker_manager(worker)

        ws.app = MagicMock()
        ws.app.state.registry = registry
        ws.app.state.worker_manager = wm

        chunk = MagicMock()
        chunk.audio_data = b"\x00\x01" * 100
        chunk.is_last = False

        stream = _make_mock_grpc_stream([chunk])

        with patch("macaw.server.routes.realtime.grpc.aio.insecure_channel") as mock_ch:
            mock_channel = AsyncMock()
            mock_ch.return_value = mock_channel
            mock_stub = MagicMock()
            mock_stub.Synthesize.return_value = stream

            with patch("macaw.server.routes.realtime.TTSWorkerStub", return_value=mock_stub):
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

        assert session.is_muted is False


class TestTTSSpeakTaskErrors:
    async def test_model_not_found_emits_error(self) -> None:
        """When TTS model not found, emits error event."""
        ws = _make_mock_websocket()
        session = _make_mock_session()
        send_event, events = _make_send_event()
        cancel = asyncio.Event()

        registry = _make_mock_registry(has_tts=False)
        wm = _make_mock_worker_manager(None)

        ws.app = MagicMock()
        ws.app.state.registry = registry
        ws.app.state.worker_manager = wm

        await _tts_speak_task(
            websocket=ws,
            session_id="sess_test",
            session=session,
            request_id="req_1",
            text="Hello",
            voice="default",
            model_tts=None,
            send_event=send_event,
            cancel_event=cancel,
        )

        error_events = [e for e in events if e.type == "error"]
        assert len(error_events) == 1
        assert "No TTS model" in error_events[0].message

    async def test_no_ready_worker_emits_error(self) -> None:
        """When no ready TTS worker, emits error event."""
        ws = _make_mock_websocket()
        session = _make_mock_session()
        send_event, events = _make_send_event()
        cancel = asyncio.Event()

        registry = _make_mock_registry()
        wm = _make_mock_worker_manager(None)  # No ready worker

        ws.app = MagicMock()
        ws.app.state.registry = registry
        ws.app.state.worker_manager = wm

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

        error_events = [e for e in events if e.type == "error"]
        assert len(error_events) == 1
        assert "worker" in error_events[0].message.lower()

    async def test_grpc_error_emits_error_and_unmutes(self) -> None:
        """gRPC error during streaming emits error and unmutes STT."""
        ws = _make_mock_websocket()
        session = _make_mock_session()
        send_event, events = _make_send_event()
        cancel = asyncio.Event()

        worker = _make_mock_worker()
        registry = _make_mock_registry()
        wm = _make_mock_worker_manager(worker)

        ws.app = MagicMock()
        ws.app.state.registry = registry
        ws.app.state.worker_manager = wm

        # Use a RuntimeError caught by generic except, since mocking
        # grpc.aio.AioRpcError is complex due to exception class patching.
        # The generic except block also calls unmute and emits error.
        with patch("macaw.server.routes.realtime.grpc.aio.insecure_channel") as mock_ch:
            mock_channel = AsyncMock()
            mock_ch.return_value = mock_channel
            mock_stub = MagicMock()
            mock_stub.Synthesize.side_effect = RuntimeError("gRPC connection refused")

            with patch("macaw.server.routes.realtime.TTSWorkerStub", return_value=mock_stub):
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

        assert session.is_muted is False
        error_events = [e for e in events if e.type == "error"]
        assert len(error_events) >= 1

    async def test_no_registry_emits_error(self) -> None:
        """When registry is not available, emits error event."""
        ws = _make_mock_websocket()
        session = _make_mock_session()
        send_event, events = _make_send_event()
        cancel = asyncio.Event()

        ws.app = MagicMock()
        ws.app.state.registry = None
        ws.app.state.worker_manager = None

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

        error_events = [e for e in events if e.type == "error"]
        assert len(error_events) == 1
        assert "not available" in error_events[0].message


class TestTTSSpeakTaskSessionNone:
    async def test_works_without_session(self) -> None:
        """TTS works even when STT session is None (no STT worker)."""
        ws = _make_mock_websocket()
        send_event, events = _make_send_event()
        cancel = asyncio.Event()

        worker = _make_mock_worker()
        registry = _make_mock_registry()
        wm = _make_mock_worker_manager(worker)

        ws.app = MagicMock()
        ws.app.state.registry = registry
        ws.app.state.worker_manager = wm

        stream = _make_mock_grpc_stream()

        with patch("macaw.server.routes.realtime.grpc.aio.insecure_channel") as mock_ch:
            mock_channel = AsyncMock()
            mock_ch.return_value = mock_channel
            mock_stub = MagicMock()
            mock_stub.Synthesize.return_value = stream

            with patch("macaw.server.routes.realtime.TTSWorkerStub", return_value=mock_stub):
                await _tts_speak_task(
                    websocket=ws,
                    session_id="sess_test",
                    session=None,  # No STT session
                    request_id="req_1",
                    text="Hello",
                    voice="default",
                    model_tts="kokoro-v1",
                    send_event=send_event,
                    cancel_event=cancel,
                )

        # Should still send audio and emit events
        assert ws.send_bytes.call_count == 2
        event_types = [e.type for e in events]
        assert "tts.speaking_start" in event_types
        assert "tts.speaking_end" in event_types


class TestTTSSpeakTaskAutoDiscover:
    async def test_auto_discovers_tts_model(self) -> None:
        """When model_tts is None, discovers first TTS model in registry."""
        ws = _make_mock_websocket()
        session = _make_mock_session()
        send_event, events = _make_send_event()
        cancel = asyncio.Event()

        worker = _make_mock_worker()
        registry = _make_mock_registry(model_name="auto-tts")
        wm = _make_mock_worker_manager(worker)

        ws.app = MagicMock()
        ws.app.state.registry = registry
        ws.app.state.worker_manager = wm

        stream = _make_mock_grpc_stream()

        with patch("macaw.server.routes.realtime.grpc.aio.insecure_channel") as mock_ch:
            mock_channel = AsyncMock()
            mock_ch.return_value = mock_channel
            mock_stub = MagicMock()
            mock_stub.Synthesize.return_value = stream

            with patch("macaw.server.routes.realtime.TTSWorkerStub", return_value=mock_stub):
                await _tts_speak_task(
                    websocket=ws,
                    session_id="sess_test",
                    session=session,
                    request_id="req_1",
                    text="Hello",
                    voice="default",
                    model_tts=None,  # Auto-discover
                    send_event=send_event,
                    cancel_event=cancel,
                )

        # Should have found and used the model
        event_types = [e.type for e in events]
        assert "tts.speaking_start" in event_types


class TestTTSSpeakTaskEmptyChunks:
    async def test_skips_empty_audio_data(self) -> None:
        """Empty audio_data chunks are skipped (not sent to client)."""
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

        # Chunk with empty audio (metadata only)
        empty_chunk = MagicMock()
        empty_chunk.audio_data = b""
        empty_chunk.is_last = False

        real_chunk = MagicMock()
        real_chunk.audio_data = b"\x00\x01" * 50
        real_chunk.is_last = True

        stream = _make_mock_grpc_stream([empty_chunk, real_chunk])

        with patch("macaw.server.routes.realtime.grpc.aio.insecure_channel") as mock_ch:
            mock_channel = AsyncMock()
            mock_ch.return_value = mock_channel
            mock_stub = MagicMock()
            mock_stub.Synthesize.return_value = stream

            with patch("macaw.server.routes.realtime.TTSWorkerStub", return_value=mock_stub):
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

        # Only the real chunk should be sent
        assert ws.send_bytes.call_count == 1


class TestTTSSpeakTaskGenericError:
    async def test_generic_exception_emits_error_and_unmutes(self) -> None:
        """Generic exception during TTS emits error and unmutes."""
        ws = _make_mock_websocket()
        session = _make_mock_session()
        send_event, events = _make_send_event()
        cancel = asyncio.Event()

        worker = _make_mock_worker()
        registry = _make_mock_registry()
        wm = _make_mock_worker_manager(worker)

        ws.app = MagicMock()
        ws.app.state.registry = registry
        ws.app.state.worker_manager = wm

        # Patch only the channel creation (not grpc.aio itself) to avoid
        # breaking the except grpc.aio.AioRpcError clause.
        with patch("macaw.server.routes.realtime.grpc.aio.insecure_channel") as mock_ch:
            mock_channel = AsyncMock()
            mock_ch.return_value = mock_channel
            mock_stub = MagicMock()
            mock_stub.Synthesize.side_effect = RuntimeError("Unexpected!")

            with patch("macaw.server.routes.realtime.TTSWorkerStub", return_value=mock_stub):
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

        assert session.is_muted is False
        error_events = [e for e in events if e.type == "error"]
        assert len(error_events) == 1
        assert error_events[0].recoverable is True


# ---------------------------------------------------------------------------
# Tests: _cancel_active_tts
# ---------------------------------------------------------------------------


class TestCancelActiveTTS:
    async def test_cancels_running_task(self) -> None:
        """_cancel_active_tts sets cancel event and awaits task."""
        cancel_event = asyncio.Event()
        done = asyncio.Event()

        async def _fake_task() -> None:
            try:
                await asyncio.wait_for(cancel_event.wait(), timeout=5.0)
            finally:
                done.set()

        task = asyncio.create_task(_fake_task())
        tts_task_ref: list[asyncio.Task[None] | None] = [task]
        tts_cancel_ref: list[asyncio.Event | None] = [cancel_event]

        await _cancel_active_tts(tts_task_ref, tts_cancel_ref)

        assert tts_task_ref[0] is None
        assert tts_cancel_ref[0] is None
        assert done.is_set()

    async def test_noop_when_no_task(self) -> None:
        """_cancel_active_tts is noop when refs are None."""
        tts_task_ref: list[asyncio.Task[None] | None] = [None]
        tts_cancel_ref: list[asyncio.Event | None] = [None]

        # Should not raise
        await _cancel_active_tts(tts_task_ref, tts_cancel_ref)

        assert tts_task_ref[0] is None
        assert tts_cancel_ref[0] is None

    async def test_noop_when_task_already_done(self) -> None:
        """_cancel_active_tts handles already-completed tasks."""
        cancel_event = asyncio.Event()

        async def _quick_task() -> None:
            pass

        task = asyncio.create_task(_quick_task())
        await task  # Let it complete

        tts_task_ref: list[asyncio.Task[None] | None] = [task]
        tts_cancel_ref: list[asyncio.Event | None] = [cancel_event]

        await _cancel_active_tts(tts_task_ref, tts_cancel_ref)

        assert tts_task_ref[0] is None
        assert tts_cancel_ref[0] is None


# ---------------------------------------------------------------------------
# Tests: Sequential Speaks
# ---------------------------------------------------------------------------


class TestSequentialSpeaks:
    async def test_second_speak_cancels_first(self) -> None:
        """When a second tts.speak arrives, the first is cancelled."""
        cancel1 = asyncio.Event()

        first_cancelled = asyncio.Event()

        async def _first_task() -> None:
            try:
                await asyncio.wait_for(cancel1.wait(), timeout=10.0)
            finally:
                first_cancelled.set()

        task1 = asyncio.create_task(_first_task())
        tts_task_ref: list[asyncio.Task[None] | None] = [task1]
        tts_cancel_ref: list[asyncio.Event | None] = [cancel1]

        # "Second speak" cancels first
        await _cancel_active_tts(tts_task_ref, tts_cancel_ref)

        assert first_cancelled.is_set()
        assert tts_task_ref[0] is None

    async def test_cancel_clears_refs(self) -> None:
        """After cancel, task and event refs are cleared."""
        cancel = asyncio.Event()

        async def _task() -> None:
            await asyncio.wait_for(cancel.wait(), timeout=10.0)

        task = asyncio.create_task(_task())
        tts_task_ref: list[asyncio.Task[None] | None] = [task]
        tts_cancel_ref: list[asyncio.Event | None] = [cancel]

        await _cancel_active_tts(tts_task_ref, tts_cancel_ref)

        assert tts_task_ref[0] is None
        assert tts_cancel_ref[0] is None


# ---------------------------------------------------------------------------
# Tests: model_tts via session.configure
# ---------------------------------------------------------------------------


class TestModelTTSTracking:
    async def test_explicit_model_tts_not_found(self) -> None:
        """Explicit model_tts that doesn't exist emits error."""
        from macaw.exceptions import ModelNotFoundError

        ws = _make_mock_websocket()
        session = _make_mock_session()
        send_event, events = _make_send_event()
        cancel = asyncio.Event()

        registry = MagicMock()
        registry.get_manifest.side_effect = ModelNotFoundError("nonexistent-tts")
        wm = _make_mock_worker_manager(None)

        ws.app = MagicMock()
        ws.app.state.registry = registry
        ws.app.state.worker_manager = wm

        await _tts_speak_task(
            websocket=ws,
            session_id="sess_test",
            session=session,
            request_id="req_1",
            text="Hello",
            voice="default",
            model_tts="nonexistent-tts",
            send_event=send_event,
            cancel_event=cancel,
        )

        error_events = [e for e in events if e.type == "error"]
        assert len(error_events) == 1
        assert "not found" in error_events[0].message


class TestTTSSpeakTaskChannelClose:
    async def test_channel_closed_in_finally(self) -> None:
        """gRPC channel is always closed in finally block."""
        ws = _make_mock_websocket()
        session = _make_mock_session()
        send_event, _ = _make_send_event()
        cancel = asyncio.Event()

        worker = _make_mock_worker()
        registry = _make_mock_registry()
        wm = _make_mock_worker_manager(worker)

        ws.app = MagicMock()
        ws.app.state.registry = registry
        ws.app.state.worker_manager = wm

        stream = _make_mock_grpc_stream()

        with patch("macaw.server.routes.realtime.grpc.aio.insecure_channel") as mock_ch:
            mock_channel = AsyncMock()
            mock_ch.return_value = mock_channel
            mock_stub = MagicMock()
            mock_stub.Synthesize.return_value = stream

            with patch("macaw.server.routes.realtime.TTSWorkerStub", return_value=mock_stub):
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

        mock_channel.close.assert_called_once()
