"""Integration tests: SessionStateMachine + StreamingSession.

Validates that the state machine (M6) is correctly integrated with the
streaming orchestrator. Focus on:
- State transitions triggered by VAD events
- Per-state timeouts with controllable clock
- State-dependent behavior (frames in HOLD not sent)
- SessionHoldEvent emission
- session.configure propagating timeouts
- Initial state (INIT, not ACTIVE)

All tests are deterministic -- they use an injectable clock.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

from macaw._types import SessionState
from macaw.server.models.events import SessionHoldEvent
from macaw.session.state_machine import SessionStateMachine, SessionTimeouts
from macaw.session.streaming import StreamingSession
from macaw.vad.detector import VADEvent, VADEventType
from tests.helpers import AsyncIterFromList, make_preprocessor_mock, make_raw_bytes, make_vad_mock


def _make_stream_handle_mock(events: list | None = None) -> Mock:
    """Create a StreamHandle mock."""
    handle = Mock()
    handle.is_closed = False
    handle.session_id = "test_session"
    if events is None:
        events = []
    handle.receive_events.return_value = AsyncIterFromList(events)
    handle.send_frame = AsyncMock()
    handle.close = AsyncMock()
    handle.cancel = AsyncMock()
    return handle


def _make_grpc_client_mock(stream_handle: Mock | None = None) -> AsyncMock:
    """Create a StreamingGRPCClient mock."""
    client = AsyncMock()
    if stream_handle is None:
        stream_handle = _make_stream_handle_mock()
    client.open_stream = AsyncMock(return_value=stream_handle)
    client.close = AsyncMock()
    return client


def _make_postprocessor_mock() -> Mock:
    """Create a PostProcessingPipeline mock."""
    mock = Mock()
    mock.process.side_effect = lambda text, **kwargs: f"ITN({text})"
    return mock


def _controllable_clock() -> tuple[list[float], object]:
    """Create a controllable clock for the state machine.

    Returns:
        (time_ref, clock_fn) where time_ref[0] controls the time
        and clock_fn can be passed as clock to SessionStateMachine.
    """
    time_ref = [0.0]

    def clock() -> float:
        return time_ref[0]

    return time_ref, clock


def _make_session_with_sm(
    *,
    timeouts: SessionTimeouts | None = None,
    clock_ref: list[float] | None = None,
    clock_fn: object | None = None,
    vad: Mock | None = None,
    grpc_client: AsyncMock | None = None,
    on_event: AsyncMock | None = None,
    hot_words: list[str] | None = None,
    enable_itn: bool = True,
) -> tuple[StreamingSession, SessionStateMachine, Mock, AsyncMock, AsyncMock]:
    """Create a StreamingSession with SessionStateMachine and controllable clock.

    Returns:
        (session, state_machine, vad, grpc_client, on_event)
    """
    if clock_ref is not None and clock_fn is None:

        def _clock() -> float:
            return clock_ref[0]

        clock_fn = _clock

    if clock_fn is None:
        _ref, clock_fn = _controllable_clock()

    sm = SessionStateMachine(timeouts=timeouts, clock=clock_fn)

    _vad = vad or make_vad_mock()
    _grpc_client = grpc_client or _make_grpc_client_mock()
    _on_event = on_event or AsyncMock()

    session = StreamingSession(
        session_id="test_session",
        preprocessor=make_preprocessor_mock(),
        vad=_vad,
        grpc_client=_grpc_client,
        postprocessor=_make_postprocessor_mock(),
        on_event=_on_event,
        hot_words=hot_words,
        enable_itn=enable_itn,
        state_machine=sm,
    )

    return session, sm, _vad, _grpc_client, _on_event


# ---------------------------------------------------------------------------
# Tests: Initial State
# ---------------------------------------------------------------------------


async def test_initial_state_is_init():
    """Initial session state must be INIT, not ACTIVE."""
    time_ref, _clock = _controllable_clock()
    session, sm, _, _, _ = _make_session_with_sm(clock_ref=time_ref)

    assert session.session_state == SessionState.INIT
    assert sm.state == SessionState.INIT


async def test_session_state_property_reflects_state_machine():
    """session_state must reflect the state machine's state."""
    time_ref, _clock = _controllable_clock()
    session, sm, _, _, _ = _make_session_with_sm(clock_ref=time_ref)

    assert session.session_state == SessionState.INIT

    # Transition manually
    sm.transition(SessionState.ACTIVE)
    assert session.session_state == SessionState.ACTIVE


# ---------------------------------------------------------------------------
# Tests: VAD Transitions
# ---------------------------------------------------------------------------


async def test_speech_start_transitions_init_to_active():
    """First frame with speech transitions INIT -> ACTIVE."""
    time_ref, _clock = _controllable_clock()
    vad = make_vad_mock()
    session, _sm, _, _, _ = _make_session_with_sm(clock_ref=time_ref, vad=vad)

    assert session.session_state == SessionState.INIT

    # Emit speech_start
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = True
    await session.process_frame(make_raw_bytes())
    await asyncio.sleep(0.01)

    assert session.session_state == SessionState.ACTIVE

    # Cleanup
    await session.close()


async def test_speech_end_transitions_active_to_silence():
    """VAD speech_end transitions ACTIVE -> SILENCE."""
    time_ref, _clock = _controllable_clock()
    vad = make_vad_mock()
    session, _sm, _, _, _ = _make_session_with_sm(clock_ref=time_ref, vad=vad)

    # First: INIT -> ACTIVE via speech_start
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = True
    await session.process_frame(make_raw_bytes())
    await asyncio.sleep(0.01)
    assert session.session_state == SessionState.ACTIVE

    # Now: ACTIVE -> SILENCE via speech_end
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_END,
        timestamp_ms=2000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())

    assert session.session_state == SessionState.SILENCE


async def test_speech_start_during_silence_transitions_to_active():
    """New speech during SILENCE transitions SILENCE -> ACTIVE."""
    time_ref, _clock = _controllable_clock()
    vad = make_vad_mock()
    grpc_client = _make_grpc_client_mock()
    session, _sm, _, _, _ = _make_session_with_sm(
        clock_ref=time_ref,
        vad=vad,
        grpc_client=grpc_client,
    )

    # INIT -> ACTIVE
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = True
    await session.process_frame(make_raw_bytes())
    await asyncio.sleep(0.01)

    # ACTIVE -> SILENCE
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_END,
        timestamp_ms=2000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())
    assert session.session_state == SessionState.SILENCE

    # New stream handle for next open_stream
    stream_handle2 = _make_stream_handle_mock()
    grpc_client.open_stream = AsyncMock(return_value=stream_handle2)

    # SILENCE -> ACTIVE (new speech)
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=3000,
    )
    vad.is_speaking = True
    await session.process_frame(make_raw_bytes())
    await asyncio.sleep(0.01)

    assert session.session_state == SessionState.ACTIVE

    # Cleanup
    await session.close()


async def test_speech_start_during_hold_transitions_to_active():
    """Speech during HOLD transitions HOLD -> ACTIVE."""
    time_ref = [0.0]
    vad = make_vad_mock()
    grpc_client = _make_grpc_client_mock()
    session, sm, _, _, _ = _make_session_with_sm(
        clock_ref=time_ref,
        vad=vad,
        grpc_client=grpc_client,
    )

    # INIT -> ACTIVE -> SILENCE
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = True
    await session.process_frame(make_raw_bytes())
    await asyncio.sleep(0.01)

    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_END,
        timestamp_ms=2000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())
    assert session.session_state == SessionState.SILENCE

    # SILENCE -> HOLD via timeout (30s default)
    time_ref[0] = 31.0
    sm.transition(SessionState.HOLD)
    assert session.session_state == SessionState.HOLD

    # New stream handle
    stream_handle2 = _make_stream_handle_mock()
    grpc_client.open_stream = AsyncMock(return_value=stream_handle2)

    # HOLD -> ACTIVE (speech detected)
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=35000,
    )
    vad.is_speaking = True
    await session.process_frame(make_raw_bytes())
    await asyncio.sleep(0.01)

    assert session.session_state == SessionState.ACTIVE

    # Cleanup
    await session.close()


# ---------------------------------------------------------------------------
# Tests: Timeouts
# ---------------------------------------------------------------------------


async def test_init_timeout_transitions_to_closed():
    """INIT timeout (30s default) transitions to CLOSED."""
    time_ref = [0.0]
    session, _sm, _, _, _ = _make_session_with_sm(clock_ref=time_ref)

    assert session.session_state == SessionState.INIT

    # Simulate 31s elapsed
    time_ref[0] = 31.0

    result = session.check_inactivity()
    assert result is True
    assert session.session_state == SessionState.CLOSED


async def test_silence_timeout_transitions_to_hold():
    """SILENCE timeout (30s default) transitions to HOLD via check_timeout."""
    time_ref = [0.0]
    vad = make_vad_mock()
    on_event = AsyncMock()
    session, _sm, _, _, _ = _make_session_with_sm(
        clock_ref=time_ref,
        vad=vad,
        on_event=on_event,
    )

    # INIT -> ACTIVE -> SILENCE
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = True
    time_ref[0] = 1.0
    await session.process_frame(make_raw_bytes())
    await asyncio.sleep(0.01)

    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_END,
        timestamp_ms=2000,
    )
    vad.is_speaking = False
    time_ref[0] = 2.0
    await session.process_frame(make_raw_bytes())
    assert session.session_state == SessionState.SILENCE

    # Simulate 31s in SILENCE state
    time_ref[0] = 33.0  # 2.0 + 31.0

    result = await session.check_timeout()
    assert result == SessionState.HOLD
    assert session.session_state == SessionState.HOLD


async def test_hold_timeout_transitions_to_closing():
    """HOLD timeout (5min default) transitions to CLOSING via check_timeout."""
    time_ref = [0.0]
    vad = make_vad_mock()
    session, sm, _, _, _ = _make_session_with_sm(clock_ref=time_ref, vad=vad)

    # INIT -> ACTIVE -> SILENCE -> HOLD
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = True
    time_ref[0] = 1.0
    await session.process_frame(make_raw_bytes())
    await asyncio.sleep(0.01)

    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_END,
        timestamp_ms=2000,
    )
    vad.is_speaking = False
    time_ref[0] = 2.0
    await session.process_frame(make_raw_bytes())

    # Transition manually to HOLD
    time_ref[0] = 3.0
    sm.transition(SessionState.HOLD)
    assert session.session_state == SessionState.HOLD

    # Simulate 301s in HOLD state (>300s default)
    time_ref[0] = 304.0  # 3.0 + 301.0

    result = await session.check_timeout()
    assert result == SessionState.CLOSING


async def test_closing_timeout_transitions_to_closed():
    """CLOSING timeout (2s default) transitions to CLOSED via check_timeout."""
    time_ref = [0.0]
    session, sm, _, _, _ = _make_session_with_sm(clock_ref=time_ref)

    # INIT -> ACTIVE -> CLOSING
    time_ref[0] = 1.0
    sm.transition(SessionState.ACTIVE)
    time_ref[0] = 2.0
    sm.transition(SessionState.CLOSING)
    assert session.session_state == SessionState.CLOSING

    # Simulate 3s in CLOSING state (>2s default)
    time_ref[0] = 5.0

    result = await session.check_timeout()
    assert result == SessionState.CLOSED
    assert session.is_closed


# ---------------------------------------------------------------------------
# Tests: SessionHoldEvent
# ---------------------------------------------------------------------------


async def test_silence_to_hold_emits_session_hold_event():
    """SILENCE -> HOLD transition via check_timeout emits SessionHoldEvent."""
    time_ref = [0.0]
    vad = make_vad_mock()
    on_event = AsyncMock()
    session, _sm, _, _, _ = _make_session_with_sm(
        clock_ref=time_ref,
        vad=vad,
        on_event=on_event,
    )

    # INIT -> ACTIVE -> SILENCE
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = True
    time_ref[0] = 1.0
    await session.process_frame(make_raw_bytes())
    await asyncio.sleep(0.01)

    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_END,
        timestamp_ms=2000,
    )
    vad.is_speaking = False
    time_ref[0] = 2.0
    await session.process_frame(make_raw_bytes())
    assert session.session_state == SessionState.SILENCE

    # Clear previous on_event calls to isolate verification
    on_event.reset_mock()

    # Simulate SILENCE timeout (31s > 30s default)
    time_ref[0] = 33.0

    await session.check_timeout()

    # Verify that SessionHoldEvent was emitted
    hold_calls = [
        call for call in on_event.call_args_list if isinstance(call.args[0], SessionHoldEvent)
    ]
    assert len(hold_calls) == 1

    hold_event = hold_calls[0].args[0]
    assert hold_event.type == "session.hold"
    assert hold_event.hold_timeout_ms == 300000  # 5min = 300s = 300000ms


async def test_session_hold_event_has_correct_hold_timeout_ms():
    """SessionHoldEvent contains correct hold_timeout_ms after session.configure."""
    time_ref = [0.0]
    vad = make_vad_mock()
    on_event = AsyncMock()
    custom_timeouts = SessionTimeouts(
        init_timeout_s=30.0,
        silence_timeout_s=5.0,  # Short for testing
        hold_timeout_s=120.0,  # 2min instead of 5min
        closing_timeout_s=2.0,
    )
    session, _sm, _, _, _ = _make_session_with_sm(
        clock_ref=time_ref,
        vad=vad,
        on_event=on_event,
        timeouts=custom_timeouts,
    )

    # INIT -> ACTIVE -> SILENCE
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = True
    time_ref[0] = 1.0
    await session.process_frame(make_raw_bytes())
    await asyncio.sleep(0.01)

    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_END,
        timestamp_ms=2000,
    )
    vad.is_speaking = False
    time_ref[0] = 2.0
    await session.process_frame(make_raw_bytes())

    on_event.reset_mock()

    # SILENCE timeout (6s > 5s custom)
    time_ref[0] = 8.0

    await session.check_timeout()

    hold_calls = [
        call for call in on_event.call_args_list if isinstance(call.args[0], SessionHoldEvent)
    ]
    assert len(hold_calls) == 1
    assert hold_calls[0].args[0].hold_timeout_ms == 120000  # 2min


# ---------------------------------------------------------------------------
# Tests: State-Based Behavior
# ---------------------------------------------------------------------------


async def test_frames_in_hold_not_sent_to_worker():
    """Frames in HOLD are not sent to worker (GPU savings)."""
    time_ref = [0.0]
    vad = make_vad_mock()
    stream_handle = _make_stream_handle_mock()
    grpc_client = _make_grpc_client_mock(stream_handle)
    session, sm, _, _, _ = _make_session_with_sm(
        clock_ref=time_ref,
        vad=vad,
        grpc_client=grpc_client,
    )

    # INIT -> ACTIVE (opens stream)
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = True
    time_ref[0] = 1.0
    await session.process_frame(make_raw_bytes())
    await asyncio.sleep(0.01)

    # ACTIVE -> SILENCE -> HOLD
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_END,
        timestamp_ms=2000,
    )
    vad.is_speaking = False
    time_ref[0] = 2.0
    await session.process_frame(make_raw_bytes())

    time_ref[0] = 3.0
    sm.transition(SessionState.HOLD)
    assert session.session_state == SessionState.HOLD

    # Reset sent frame counter
    stream_handle.send_frame.reset_mock()

    # New stream to simulate "speaking" but in HOLD
    stream_handle2 = _make_stream_handle_mock()
    grpc_client.open_stream = AsyncMock(return_value=stream_handle2)

    # Send frames in HOLD (vad.is_speaking=True simulated,
    # but without speech_start event - state remains HOLD)
    vad.process_frame.return_value = None
    vad.is_speaking = True  # VAD thinks it is speaking
    await session.process_frame(make_raw_bytes())
    await session.process_frame(make_raw_bytes())

    # Assert: no frames sent to worker (state is HOLD, not ACTIVE)
    stream_handle2.send_frame.assert_not_called()


async def test_frames_in_init_not_sent_to_worker():
    """Frames in INIT (before speech_start) are not sent to worker."""
    time_ref = [0.0]
    vad = make_vad_mock(is_speaking=False)
    stream_handle = _make_stream_handle_mock()
    grpc_client = _make_grpc_client_mock(stream_handle)
    session, _sm, _, _, _ = _make_session_with_sm(
        clock_ref=time_ref,
        vad=vad,
        grpc_client=grpc_client,
    )

    assert session.session_state == SessionState.INIT

    # Send frames without speech_start
    vad.process_frame.return_value = None
    await session.process_frame(make_raw_bytes())
    await session.process_frame(make_raw_bytes())

    # Assert: no frames sent
    stream_handle.send_frame.assert_not_called()


async def test_frames_in_closing_rejected():
    """Frames in CLOSING are ignored (does not accept new frames)."""
    time_ref = [0.0]
    vad = make_vad_mock()
    preprocessor = make_preprocessor_mock()
    session, sm, _, _, _ = _make_session_with_sm(
        clock_ref=time_ref,
        vad=vad,
    )
    # Replace preprocessor to verify it is not called
    session._preprocessor = preprocessor

    # Transition to CLOSING
    time_ref[0] = 1.0
    sm.transition(SessionState.ACTIVE)
    time_ref[0] = 2.0
    sm.transition(SessionState.CLOSING)
    assert session.session_state == SessionState.CLOSING

    preprocessor.process_frame.reset_mock()

    # Try to process frame
    await session.process_frame(make_raw_bytes())

    # Assert: preprocessor not called (frame rejected early)
    preprocessor.process_frame.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: session.configure / timeouts
# ---------------------------------------------------------------------------


async def test_update_session_timeouts():
    """update_session_timeouts() updates state machine timeouts."""
    time_ref = [0.0]
    session, sm, _, _, _ = _make_session_with_sm(clock_ref=time_ref)

    # Default timeouts: INIT=30s
    assert sm.timeouts.init_timeout_s == 30.0

    # Update timeouts
    new_timeouts = SessionTimeouts(
        init_timeout_s=10.0,
        silence_timeout_s=15.0,
        hold_timeout_s=60.0,
        closing_timeout_s=3.0,
    )
    session.update_session_timeouts(new_timeouts)

    assert sm.timeouts.init_timeout_s == 10.0
    assert sm.timeouts.silence_timeout_s == 15.0
    assert sm.timeouts.hold_timeout_s == 60.0
    assert sm.timeouts.closing_timeout_s == 3.0


async def test_updated_timeout_affects_check_timeout():
    """Updated timeouts are used immediately in check_timeout."""
    time_ref = [0.0]
    session, _sm, _, _, _ = _make_session_with_sm(clock_ref=time_ref)

    # Default INIT timeout: 30s
    time_ref[0] = 15.0  # 15s < 30s
    assert not session.check_inactivity()  # Not expired

    # Update to 10s
    new_timeouts = SessionTimeouts(
        init_timeout_s=10.0,
        silence_timeout_s=30.0,
        hold_timeout_s=300.0,
        closing_timeout_s=2.0,
    )
    session.update_session_timeouts(new_timeouts)

    # Now 15s > 10s, should expire
    assert session.check_inactivity()
    assert session.session_state == SessionState.CLOSED


# ---------------------------------------------------------------------------
# Tests: close()
# ---------------------------------------------------------------------------


async def test_close_transitions_to_closing_then_closed():
    """close() transitions to CLOSING -> CLOSED via state machine."""
    time_ref = [0.0]
    vad = make_vad_mock()
    session, _sm, _, _, _ = _make_session_with_sm(clock_ref=time_ref, vad=vad)

    # INIT -> ACTIVE
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = True
    time_ref[0] = 1.0
    await session.process_frame(make_raw_bytes())
    await asyncio.sleep(0.01)
    assert session.session_state == SessionState.ACTIVE

    # close()
    time_ref[0] = 2.0
    await session.close()

    assert session.session_state == SessionState.CLOSED
    assert session.is_closed


async def test_close_from_init_transitions_to_closed():
    """close() from INIT transitions directly via CLOSING -> CLOSED."""
    time_ref = [0.0]
    session, _sm, _, _, _ = _make_session_with_sm(clock_ref=time_ref)

    assert session.session_state == SessionState.INIT

    await session.close()

    # INIT has no direct transition to CLOSING in the state machine,
    # but close() tries CLOSING and if it fails, goes directly to CLOSED
    assert session.is_closed


# ---------------------------------------------------------------------------
# Tests: check_timeout returns None when there is no timeout
# ---------------------------------------------------------------------------


async def test_check_timeout_returns_none_when_active():
    """ACTIVE has no timeout, check_timeout returns None."""
    time_ref = [0.0]
    vad = make_vad_mock()
    session, _sm, _, _, _ = _make_session_with_sm(clock_ref=time_ref, vad=vad)

    # INIT -> ACTIVE
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = True
    time_ref[0] = 1.0
    await session.process_frame(make_raw_bytes())
    await asyncio.sleep(0.01)
    assert session.session_state == SessionState.ACTIVE

    # Even after a long time, ACTIVE does not expire
    time_ref[0] = 999.0
    result = await session.check_timeout()
    assert result is None
    assert session.session_state == SessionState.ACTIVE

    # Cleanup
    await session.close()


async def test_check_timeout_returns_none_when_closed():
    """CLOSED returns None from check_timeout."""
    time_ref = [0.0]
    session, _sm, _, _, _ = _make_session_with_sm(clock_ref=time_ref)

    await session.close()
    assert session.is_closed

    time_ref[0] = 999.0
    result = await session.check_timeout()
    assert result is None
