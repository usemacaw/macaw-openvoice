"""Integration tests for the full-duplex STT+TTS flow.

These tests exercise composite scenarios that combine multiple operations
of the full-duplex flow: mute/unmute lifecycle, sequential cancellation,
error recovery and edge cases. They differ from test_full_duplex.py by testing
interactions between components, not isolated functions.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from macaw.server.models.events import TTSSpeakCommand
from macaw.server.routes.realtime import (
    SessionContext,
    _cancel_active_tts,
    _tts_speak_task,
)
from tests.unit.test_full_duplex import (
    _make_mock_grpc_stream,
    _make_mock_registry,
    _make_mock_session,
    _make_mock_websocket,
    _make_mock_worker,
    _make_mock_worker_manager,
    _make_send_event,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_tts_env(
    *,
    has_tts: bool = True,
    model_name: str = "kokoro-v1",
) -> tuple[MagicMock, MagicMock, MagicMock, MagicMock]:
    """Prepare ws, session, registry and worker_manager for TTS."""
    ws = _make_mock_websocket()
    session = _make_mock_session()
    worker = _make_mock_worker()
    registry = _make_mock_registry(has_tts=has_tts, model_name=model_name)
    wm = _make_mock_worker_manager(worker if has_tts else None)

    ws.app = MagicMock()
    ws.app.state.registry = registry
    ws.app.state.worker_manager = wm

    return ws, session, registry, wm


def _make_session_context(
    *,
    tts_task: asyncio.Task[None] | None = None,
    tts_cancel_event: asyncio.Event | None = None,
) -> SessionContext:
    """Build a minimal SessionContext for _cancel_active_tts tests."""
    ws = _make_mock_websocket()
    ctx = SessionContext(
        session_id="sess_test",
        session_start=0.0,
        websocket=ws,
    )
    ctx.tts_task = tts_task
    ctx.tts_cancel_event = tts_cancel_event
    return ctx


async def _run_tts_speak(
    ws: MagicMock,
    session: MagicMock | None,
    send_event: AsyncMock,
    cancel: asyncio.Event,
    *,
    stream: Any = None,
    model_tts: str | None = "kokoro-v1",
    request_id: str = "req_1",
    text: str = "Hello",
    stub_side_effect: Exception | None = None,
) -> None:
    """Execute _tts_speak_task with standard gRPC mocks."""
    if stream is None:
        stream = _make_mock_grpc_stream()

    with patch("macaw.server.routes.realtime.get_or_create_tts_channel") as mock_ch:
        mock_channel = AsyncMock()
        mock_ch.return_value = mock_channel
        mock_stub = MagicMock()

        if stub_side_effect is not None:
            mock_stub.Synthesize.side_effect = stub_side_effect
        else:
            mock_stub.Synthesize.return_value = stream

        with patch(
            "macaw.server.routes.realtime.TTSWorkerStub",
            return_value=mock_stub,
        ):
            await _tts_speak_task(
                websocket=ws,
                session_id="sess_test",
                session=session,
                request_id=request_id,
                cmd=TTSSpeakCommand(text=text),
                model_tts=model_tts,
                send_event=send_event,
                cancel_event=cancel,
            )


def _event_types(events: list[Any]) -> list[str]:
    """Extract list of event types."""
    return [e.type for e in events if hasattr(e, "type")]


# ---------------------------------------------------------------------------
# Tests: Mute Lifecycle (STT + TTS coordinated)
# ---------------------------------------------------------------------------


class TestFullDuplexMuteLifecycle:
    """Verifies that STT mute/unmute is coordinated with the TTS cycle."""

    async def test_mute_on_tts_start_unmute_on_tts_end(self) -> None:
        """TTS task mutes session during synthesis and unmutes after completion."""
        ws, session, _reg, _wm = _setup_tts_env()
        send_event, events = _make_send_event()
        cancel = asyncio.Event()

        assert session.is_muted is False

        await _run_tts_speak(ws, session, send_event, cancel)

        # After completion, session should be unmuted
        assert session.is_muted is False
        types = _event_types(events)
        assert "tts.speaking_start" in types
        assert "tts.speaking_end" in types

    async def test_frames_discarded_during_tts(self) -> None:
        """While TTS is active, audio frames are discarded by the session."""
        ws, session, _reg, _wm = _setup_tts_env()
        send_event, _events = _make_send_event()

        # Chunk stream that allows checking mute during iteration
        mute_observed = False

        chunk = MagicMock()
        chunk.audio_data = b"\x00\x01" * 100
        chunk.is_last = False

        last_chunk = MagicMock()
        last_chunk.audio_data = b"\x00\x02" * 50
        last_chunk.is_last = True

        class _MuteCheckStream:
            """Stream that checks mute during iteration."""

            def __init__(self) -> None:
                self._items = [chunk, last_chunk]
                self._idx = 0

            def __aiter__(self) -> _MuteCheckStream:
                return self

            async def __anext__(self) -> MagicMock:
                if self._idx >= len(self._items):
                    raise StopAsyncIteration
                item = self._items[self._idx]
                self._idx += 1
                # After first chunk is processed, session should be muted
                if self._idx == 2:
                    nonlocal mute_observed
                    mute_observed = session.is_muted
                return item

        cancel = asyncio.Event()
        await _run_tts_speak(
            ws,
            session,
            send_event,
            cancel,
            stream=_MuteCheckStream(),
        )

        # During streaming, session was muted
        assert mute_observed is True
        # After completion, session unmuted
        assert session.is_muted is False

    async def test_unmute_after_tts_completion_enables_process_frame(self) -> None:
        """After TTS completes, process_frame is callable again."""
        ws, session, _reg, _wm = _setup_tts_env()
        send_event, _events = _make_send_event()
        cancel = asyncio.Event()

        await _run_tts_speak(ws, session, send_event, cancel)

        assert session.is_muted is False
        # Simulate that process_frame works normally after unmute
        await session.process_frame(b"\x00\x01" * 100)
        session.process_frame.assert_called_once_with(b"\x00\x01" * 100)

    async def test_tts_cancel_unmutes_immediately(self) -> None:
        """Setting cancel_event before completion triggers immediate unmute."""
        ws, session, _reg, _wm = _setup_tts_env()
        send_event, events = _make_send_event()
        cancel = asyncio.Event()
        cancel.set()  # Pre-set: cancels on first check

        await _run_tts_speak(ws, session, send_event, cancel)

        assert session.is_muted is False
        # speaking_end with cancelled=True
        end_events = [e for e in events if hasattr(e, "type") and e.type == "tts.speaking_end"]
        if end_events:
            assert end_events[0].cancelled is True


# ---------------------------------------------------------------------------
# Tests: Sequential Speaks (cancel + speak composition)
# ---------------------------------------------------------------------------


class TestFullDuplexSequential:
    """Verifies sequential speaks and cancellation scenarios."""

    async def test_second_speak_replaces_first(self) -> None:
        """Starting second speak cancels the first and completes the second."""
        ws, session, _reg, _wm = _setup_tts_env()

        # First speak: slow task
        cancel1 = asyncio.Event()
        first_cancelled = asyncio.Event()

        async def _slow_first() -> None:
            try:
                await asyncio.wait_for(cancel1.wait(), timeout=10.0)
            finally:
                first_cancelled.set()

        task1 = asyncio.create_task(_slow_first())
        ctx = _make_session_context(tts_task=task1, tts_cancel_event=cancel1)

        # Cancel first (simulating arrival of second speak)
        await _cancel_active_tts(ctx)

        assert first_cancelled.is_set()
        assert ctx.tts_task is None
        assert ctx.tts_cancel_event is None

        # Execute second speak normally
        send_event2, events2 = _make_send_event()
        cancel2 = asyncio.Event()
        await _run_tts_speak(ws, session, send_event2, cancel2, request_id="req_2")

        types2 = _event_types(events2)
        assert "tts.speaking_start" in types2
        assert "tts.speaking_end" in types2
        assert session.is_muted is False

    async def test_rapid_speak_cancel_speak(self) -> None:
        """Speak -> cancel -> speak rapidly without leaving inconsistent state."""
        ws, session, _reg, _wm = _setup_tts_env()

        # First speak
        send1, events1 = _make_send_event()
        cancel1 = asyncio.Event()
        await _run_tts_speak(ws, session, send1, cancel1, request_id="req_1")
        assert session.is_muted is False

        # Cancel (noop -- first already completed)
        ctx = _make_session_context(tts_task=None, tts_cancel_event=None)
        await _cancel_active_tts(ctx)

        # Second speak
        send2, events2 = _make_send_event()
        cancel2 = asyncio.Event()
        await _run_tts_speak(ws, session, send2, cancel2, request_id="req_2")
        assert session.is_muted is False

        # Both completed with speaking_start and speaking_end
        assert "tts.speaking_start" in _event_types(events1)
        assert "tts.speaking_end" in _event_types(events1)
        assert "tts.speaking_start" in _event_types(events2)
        assert "tts.speaking_end" in _event_types(events2)

    async def test_cancel_noop_after_completion(self) -> None:
        """Cancel after TTS already completed is a no-op without error."""
        ws, session, _reg, _wm = _setup_tts_env()
        send_event, _events = _make_send_event()
        cancel = asyncio.Event()

        await _run_tts_speak(ws, session, send_event, cancel)

        # Task already completed, cancel is noop
        ctx = _make_session_context(tts_task=None, tts_cancel_event=cancel)
        await _cancel_active_tts(ctx)

        assert ctx.tts_task is None
        assert ctx.tts_cancel_event is None
        assert session.is_muted is False


# ---------------------------------------------------------------------------
# Tests: Error Recovery
# ---------------------------------------------------------------------------


class TestFullDuplexErrorRecovery:
    """Verifies that errors during TTS do not leave inconsistent state."""

    async def test_worker_crash_emits_error_and_unmutes(self) -> None:
        """gRPC error during synthesis emits error and unmutes session."""
        ws, session, _reg, _wm = _setup_tts_env()
        send_event, events = _make_send_event()
        cancel = asyncio.Event()

        await _run_tts_speak(
            ws,
            session,
            send_event,
            cancel,
            stub_side_effect=RuntimeError("Worker crashed"),
        )

        assert session.is_muted is False
        error_events = [e for e in events if hasattr(e, "type") and e.type == "error"]
        assert len(error_events) >= 1
        assert error_events[0].recoverable is True

    async def test_model_not_found_emits_error_no_mute(self) -> None:
        """Nonexistent TTS model emits error without muting session."""
        ws, session, _reg, _wm = _setup_tts_env(has_tts=False)
        send_event, events = _make_send_event()
        cancel = asyncio.Event()

        # model_tts=None and no TTS model in registry
        await _run_tts_speak(
            ws,
            session,
            send_event,
            cancel,
            model_tts=None,
        )

        assert session.is_muted is False
        error_events = [e for e in events if hasattr(e, "type") and e.type == "error"]
        assert len(error_events) == 1
        assert "No TTS model" in error_events[0].message
        # Should not have speaking_start (never started)
        assert "tts.speaking_start" not in _event_types(events)

    async def test_error_does_not_leave_session_muted(self) -> None:
        """Any error path guarantees session unmuted at the end."""
        ws, session, _reg, _wm = _setup_tts_env()
        send_event1, _events1 = _make_send_event()
        cancel1 = asyncio.Event()

        # gRPC error
        await _run_tts_speak(
            ws,
            session,
            send_event1,
            cancel1,
            stub_side_effect=RuntimeError("gRPC fail"),
        )
        assert session.is_muted is False

        # Registry None
        ws2 = _make_mock_websocket()
        ws2.app = MagicMock()
        ws2.app.state.registry = None
        ws2.app.state.worker_manager = None

        send_event2, _events2 = _make_send_event()
        cancel2 = asyncio.Event()

        await _tts_speak_task(
            websocket=ws2,
            session_id="sess_test",
            session=session,
            request_id="req_2",
            cmd=TTSSpeakCommand(text="Hello"),
            model_tts="kokoro-v1",
            send_event=send_event2,
            cancel_event=cancel2,
        )
        assert session.is_muted is False

    async def test_error_after_first_chunk_still_unmutes(self) -> None:
        """If error occurs after first chunk (mute already applied), unmute happens."""
        ws, session, _reg, _wm = _setup_tts_env()
        send_event, events = _make_send_event()
        cancel = asyncio.Event()

        # Stream that produces one chunk then fails
        chunk = MagicMock()
        chunk.audio_data = b"\x00\x01" * 100
        chunk.is_last = False

        class _FailAfterFirstStream:
            def __init__(self) -> None:
                self._sent_first = False

            def __aiter__(self) -> _FailAfterFirstStream:
                return self

            async def __anext__(self) -> MagicMock:
                if not self._sent_first:
                    self._sent_first = True
                    return chunk
                msg = "Connection lost"
                raise RuntimeError(msg)

        await _run_tts_speak(
            ws,
            session,
            send_event,
            cancel,
            stream=_FailAfterFirstStream(),
        )

        # Session muted on first chunk but unmuted in finally
        assert session.is_muted is False
        # speaking_start emitted, error emitted, speaking_end emitted
        types = _event_types(events)
        assert "tts.speaking_start" in types
        assert "error" in types
        assert "tts.speaking_end" in types


# ---------------------------------------------------------------------------
# Tests: Edge Cases
# ---------------------------------------------------------------------------


class TestFullDuplexEdgeCases:
    """Edge cases for the full-duplex flow."""

    async def test_tts_without_session_completes(self) -> None:
        """TTS works when session is None (no STT worker)."""
        ws, _session, _reg, _wm = _setup_tts_env()
        send_event, events = _make_send_event()
        cancel = asyncio.Event()

        await _run_tts_speak(ws, None, send_event, cancel)

        types = _event_types(events)
        assert "tts.speaking_start" in types
        assert "tts.speaking_end" in types
        # Audio sent to websocket
        assert ws.send_bytes.call_count == 2

    async def test_empty_audio_chunks_still_completes(self) -> None:
        """TTS with empty chunks (only is_last=True) completes without error."""
        ws, session, _reg, _wm = _setup_tts_env()
        send_event, events = _make_send_event()
        cancel = asyncio.Event()

        # Chunk with only is_last, no audio_data
        only_last = MagicMock()
        only_last.audio_data = b""
        only_last.is_last = True

        stream = _make_mock_grpc_stream([only_last])

        await _run_tts_speak(ws, session, send_event, cancel, stream=stream)

        # Should not have sent bytes to websocket (empty audio)
        assert ws.send_bytes.call_count == 0
        # Should not have emitted speaking_start (no chunk with audio)
        assert "tts.speaking_start" not in _event_types(events)
        # Session was never muted
        assert session.is_muted is False

    async def test_multiple_frames_during_tts_all_discarded(self) -> None:
        """Multiple audio frames sent during TTS are all discarded."""
        ws, session, _reg, _wm = _setup_tts_env()
        send_event, _events = _make_send_event()

        frames_during_mute: list[bool] = []

        chunk1 = MagicMock()
        chunk1.audio_data = b"\x00\x01" * 100
        chunk1.is_last = False

        chunk2 = MagicMock()
        chunk2.audio_data = b"\x00\x02" * 100
        chunk2.is_last = False

        last_chunk = MagicMock()
        last_chunk.audio_data = b"\x00\x03" * 100
        last_chunk.is_last = True

        class _MultiFrameCheckStream:
            """Stream that checks mute during iteration for multiple frames."""

            def __init__(self) -> None:
                self._items = [chunk1, chunk2, last_chunk]
                self._idx = 0

            def __aiter__(self) -> _MultiFrameCheckStream:
                return self

            async def __anext__(self) -> MagicMock:
                if self._idx >= len(self._items):
                    raise StopAsyncIteration
                item = self._items[self._idx]
                self._idx += 1
                # Record muted state on each iteration after first chunk
                if self._idx >= 2:
                    frames_during_mute.append(session.is_muted)
                return item

        cancel = asyncio.Event()
        await _run_tts_speak(
            ws,
            session,
            send_event,
            cancel,
            stream=_MultiFrameCheckStream(),
        )

        # All checks during streaming showed muted=True
        assert len(frames_during_mute) == 2
        assert all(frames_during_mute)
        # After TTS, unmuted
        assert session.is_muted is False

    async def test_tts_with_no_worker_available(self) -> None:
        """When no TTS worker is ready, emits error without muting."""
        ws = _make_mock_websocket()
        session = _make_mock_session()
        registry = _make_mock_registry()
        wm = _make_mock_worker_manager(None)  # No ready worker

        ws.app = MagicMock()
        ws.app.state.registry = registry
        ws.app.state.worker_manager = wm

        send_event, events = _make_send_event()
        cancel = asyncio.Event()

        await _tts_speak_task(
            websocket=ws,
            session_id="sess_test",
            session=session,
            request_id="req_1",
            cmd=TTSSpeakCommand(text="Hello"),
            model_tts="kokoro-v1",
            send_event=send_event,
            cancel_event=cancel,
        )

        assert session.is_muted is False
        error_events = [e for e in events if hasattr(e, "type") and e.type == "error"]
        assert len(error_events) == 1
        assert "worker" in error_events[0].message.lower()
        assert "tts.speaking_start" not in _event_types(events)

    async def test_cancel_active_tts_then_speak_again(self) -> None:
        """After cancel_active_tts, new speak works normally."""
        ws, session, _reg, _wm = _setup_tts_env()

        # Start TTS in background (slow)
        cancel1 = asyncio.Event()
        completed = asyncio.Event()

        async def _slow_tts() -> None:
            try:
                await asyncio.wait_for(cancel1.wait(), timeout=10.0)
            finally:
                completed.set()

        task1 = asyncio.create_task(_slow_tts())
        ctx = _make_session_context(tts_task=task1, tts_cancel_event=cancel1)

        # Cancel
        await _cancel_active_tts(ctx)
        assert completed.is_set()

        # New speak completes normally
        send_event, events = _make_send_event()
        cancel2 = asyncio.Event()
        await _run_tts_speak(ws, session, send_event, cancel2, request_id="req_new")

        types = _event_types(events)
        assert "tts.speaking_start" in types
        assert "tts.speaking_end" in types
        assert session.is_muted is False

    async def test_concurrent_cancel_events_safe(self) -> None:
        """Multiple concurrent cancels do not cause errors."""
        cancel = asyncio.Event()

        async def _waiter() -> None:
            await asyncio.wait_for(cancel.wait(), timeout=10.0)

        task = asyncio.create_task(_waiter())
        ctx1 = _make_session_context(tts_task=task, tts_cancel_event=cancel)
        ctx2 = _make_session_context(tts_task=None, tts_cancel_event=None)

        # Two concurrent cancels
        await asyncio.gather(
            _cancel_active_tts(ctx1),
            _cancel_active_tts(ctx2),  # Segundo sem task (simula race condition)
        )

        assert ctx1.tts_task is None
        assert ctx1.tts_cancel_event is None
