"""Tests for StreamingSession.

Validates that the streaming orchestrator correctly coordinates:
preprocessing -> VAD -> gRPC worker -> post-processing.

All tests use mocks for external dependencies.
Tests are deterministic -- no dependency on real timing.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

from macaw._types import TranscriptSegment, WordTimestamp
from macaw.exceptions import WorkerCrashError
from macaw.server.models.events import (
    StreamingErrorEvent,
    TranscriptFinalEvent,
    TranscriptPartialEvent,
    VADSpeechEndEvent,
    VADSpeechStartEvent,
)
from macaw.session.state_machine import SessionStateMachine
from macaw.session.streaming import StreamingSession
from macaw.vad.detector import VADEvent, VADEventType
from tests.helpers import (
    AsyncIterFromList,
    make_preprocessor_mock,
    make_raw_bytes,
    make_vad_mock,
)


def _make_stream_handle_mock(
    events: list | None = None,
) -> Mock:
    """Create StreamHandle mock.

    Args:
        events: List of TranscriptSegment or Exception for receive_events.
                If None, returns empty iterator.
    """
    handle = Mock()
    handle.is_closed = False
    handle.session_id = "test_session"

    # receive_events returns an async iterable
    if events is None:
        events = []
    handle.receive_events.return_value = AsyncIterFromList(events)

    handle.send_frame = AsyncMock()
    handle.close = AsyncMock()
    handle.cancel = AsyncMock()
    return handle


def _make_grpc_client_mock(stream_handle: Mock | None = None) -> AsyncMock:
    """Create StreamingGRPCClient mock."""
    client = AsyncMock()
    if stream_handle is None:
        stream_handle = _make_stream_handle_mock()
    client.open_stream = AsyncMock(return_value=stream_handle)
    client.close = AsyncMock()
    return client


def _make_postprocessor_mock() -> Mock:
    """Create PostProcessingPipeline mock."""
    mock = Mock()
    mock.process.side_effect = lambda text, **kwargs: f"ITN({text})"
    return mock


def _make_on_event() -> AsyncMock:
    """Create on_event mock callback."""
    return AsyncMock()


def _make_session(
    *,
    preprocessor: Mock | None = None,
    vad: Mock | None = None,
    grpc_client: AsyncMock | None = None,
    postprocessor: Mock | None = None,
    on_event: AsyncMock | None = None,
    hot_words: list[str] | None = None,
    enable_itn: bool = True,
    session_id: str = "test_session",
) -> tuple[StreamingSession, Mock, Mock, AsyncMock, Mock, AsyncMock]:
    """Create StreamingSession with configured mocks.

    Returns:
        (session, preprocessor, vad, grpc_client, postprocessor, on_event)
    """
    _preprocessor = preprocessor or make_preprocessor_mock()
    _vad = vad or make_vad_mock()
    _grpc_client = grpc_client or _make_grpc_client_mock()
    _postprocessor = postprocessor or _make_postprocessor_mock()
    _on_event = on_event or _make_on_event()

    session = StreamingSession(
        session_id=session_id,
        preprocessor=_preprocessor,
        vad=_vad,
        grpc_client=_grpc_client,
        postprocessor=_postprocessor,
        on_event=_on_event,
        hot_words=hot_words,
        enable_itn=enable_itn,
    )

    return session, _preprocessor, _vad, _grpc_client, _postprocessor, _on_event


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_speech_start_emits_vad_event():
    """VAD speech_start emits vad.speech_start via callback."""
    # Arrange
    vad = make_vad_mock()
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1500,
    )
    vad.is_speaking = False  # Before speech_start, not speaking

    session, _, _, _, _, on_event = _make_session(vad=vad)

    # Act
    await session.process_frame(make_raw_bytes())

    # Give time for receiver task to start
    await asyncio.sleep(0.01)

    # Assert
    on_event.assert_any_call(
        VADSpeechStartEvent(timestamp_ms=1500),
    )

    # Cleanup
    await session.close()


async def test_speech_end_emits_vad_event():
    """VAD speech_end emits vad.speech_end via callback."""
    # Arrange
    stream_handle = _make_stream_handle_mock()
    grpc_client = _make_grpc_client_mock(stream_handle)
    vad = make_vad_mock()
    on_event = _make_on_event()

    session = StreamingSession(
        session_id="test_session",
        preprocessor=make_preprocessor_mock(),
        vad=vad,
        grpc_client=grpc_client,
        postprocessor=_make_postprocessor_mock(),
        on_event=on_event,
    )

    # Simulate speech_start
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())
    await asyncio.sleep(0.01)

    # Simulate speech_end
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_END,
        timestamp_ms=2000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())

    # Assert
    on_event.assert_any_call(
        VADSpeechEndEvent(timestamp_ms=2000),
    )


async def test_final_transcript_applies_postprocessing():
    """ITN is applied only to transcript.final."""
    # Arrange
    final_segment = TranscriptSegment(
        text="dois mil e vinte e cinco",
        is_final=True,
        segment_id=0,
        start_ms=1000,
        end_ms=2000,
        language="pt",
        confidence=0.95,
    )

    stream_handle = _make_stream_handle_mock(events=[final_segment])
    grpc_client = _make_grpc_client_mock(stream_handle)
    vad = make_vad_mock()
    postprocessor = _make_postprocessor_mock()
    on_event = _make_on_event()

    session = StreamingSession(
        session_id="test_session",
        preprocessor=make_preprocessor_mock(),
        vad=vad,
        grpc_client=grpc_client,
        postprocessor=postprocessor,
        on_event=on_event,
        enable_itn=True,
    )

    # Trigger speech_start -> opens stream and starts receiver
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())

    # Give time for receiver task to process the event
    await asyncio.sleep(0.05)

    # Trigger speech_end para flush
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_END,
        timestamp_ms=2000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())

    # Assert: ITN was applied to the final text
    postprocessor.process.assert_called_once_with("dois mil e vinte e cinco", language="pt")

    # Verify that the final event has post-processed text
    final_calls = [
        call for call in on_event.call_args_list if isinstance(call.args[0], TranscriptFinalEvent)
    ]
    assert len(final_calls) == 1
    assert final_calls[0].args[0].text == "ITN(dois mil e vinte e cinco)"


async def test_partial_transcript_no_postprocessing():
    """Partial transcripts are NOT processed by ITN."""
    # Arrange
    partial_segment = TranscriptSegment(
        text="ola como",
        is_final=False,
        segment_id=0,
        start_ms=1000,
    )

    stream_handle = _make_stream_handle_mock(events=[partial_segment])
    grpc_client = _make_grpc_client_mock(stream_handle)
    vad = make_vad_mock()
    postprocessor = _make_postprocessor_mock()
    on_event = _make_on_event()

    session = StreamingSession(
        session_id="test_session",
        preprocessor=make_preprocessor_mock(),
        vad=vad,
        grpc_client=grpc_client,
        postprocessor=postprocessor,
        on_event=on_event,
    )

    # Trigger speech_start
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())

    # Give time for receiver task
    await asyncio.sleep(0.05)

    # Assert: ITN was NOT called
    postprocessor.process.assert_not_called()

    # Verify that the partial was emitted with original text
    partial_calls = [
        call
        for call in on_event.call_args_list
        if isinstance(call.args[0], TranscriptPartialEvent)
    ]
    assert len(partial_calls) == 1
    assert partial_calls[0].args[0].text == "ola como"

    # Cleanup
    await session.close()


async def test_segment_id_increments():
    """Each speech segment receives incremental segment_id."""
    # Arrange
    stream_handle1 = _make_stream_handle_mock()
    grpc_client = _make_grpc_client_mock(stream_handle1)
    vad = make_vad_mock()
    on_event = _make_on_event()

    session = StreamingSession(
        session_id="test_session",
        preprocessor=make_preprocessor_mock(),
        vad=vad,
        grpc_client=grpc_client,
        postprocessor=_make_postprocessor_mock(),
        on_event=on_event,
    )

    assert session.segment_id == 0

    # First segment: speech_start -> speech_end
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())
    await asyncio.sleep(0.01)

    # New stream_handle for next open_stream
    stream_handle2 = _make_stream_handle_mock()
    grpc_client.open_stream = AsyncMock(return_value=stream_handle2)

    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_END,
        timestamp_ms=2000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())

    assert session.segment_id == 1

    # Second segment: speech_start -> speech_end
    stream_handle3 = _make_stream_handle_mock()
    grpc_client.open_stream = AsyncMock(return_value=stream_handle3)

    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=3000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())
    await asyncio.sleep(0.01)

    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_END,
        timestamp_ms=4000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())

    assert session.segment_id == 2


async def test_close_cleans_up_resources():
    """close() closes gRPC stream and marks session as CLOSED."""
    # Arrange
    stream_handle = _make_stream_handle_mock()
    grpc_client = _make_grpc_client_mock(stream_handle)
    vad = make_vad_mock()

    session = StreamingSession(
        session_id="test_session",
        preprocessor=make_preprocessor_mock(),
        vad=vad,
        grpc_client=grpc_client,
        postprocessor=_make_postprocessor_mock(),
        on_event=_make_on_event(),
    )

    # Open stream (trigger speech_start)
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())
    await asyncio.sleep(0.01)
    assert not session.is_closed

    # Act
    await session.close()

    # Assert
    assert session.is_closed
    stream_handle.cancel.assert_called_once()


async def test_process_frame_during_closed_is_noop():
    """Frames received after close are ignored."""
    # Arrange
    preprocessor = make_preprocessor_mock()

    session, _, _, _, _, _ = _make_session(preprocessor=preprocessor)

    # Close session
    await session.close()
    assert session.is_closed

    # Reset mock counters
    preprocessor.process_frame.reset_mock()

    # Act: try to process frame
    await session.process_frame(make_raw_bytes())

    # Assert: preprocessor was NOT called
    preprocessor.process_frame.assert_not_called()


async def test_full_speech_cycle():
    """Full cycle: speech_start -> partials -> final -> speech_end in order."""
    # Arrange
    partial_seg = TranscriptSegment(
        text="ola",
        is_final=False,
        segment_id=0,
        start_ms=1000,
    )
    final_seg = TranscriptSegment(
        text="ola mundo",
        is_final=True,
        segment_id=0,
        start_ms=1000,
        end_ms=2000,
        language="pt",
        confidence=0.95,
    )

    stream_handle = _make_stream_handle_mock(events=[partial_seg, final_seg])
    grpc_client = _make_grpc_client_mock(stream_handle)
    vad = make_vad_mock()
    postprocessor = _make_postprocessor_mock()
    on_event = _make_on_event()

    session = StreamingSession(
        session_id="test_session",
        preprocessor=make_preprocessor_mock(),
        vad=vad,
        grpc_client=grpc_client,
        postprocessor=postprocessor,
        on_event=on_event,
        enable_itn=True,
    )

    # 1. Speech start
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())

    # Give time for receiver task to process both events
    await asyncio.sleep(0.05)

    # 2. Speech end
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_END,
        timestamp_ms=2000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())

    # Assert: events in correct order
    event_types = [type(call.args[0]).__name__ for call in on_event.call_args_list]

    assert "VADSpeechStartEvent" in event_types
    assert "TranscriptPartialEvent" in event_types
    assert "TranscriptFinalEvent" in event_types
    assert "VADSpeechEndEvent" in event_types

    # Verify order: speech_start < partial < final < speech_end
    start_idx = event_types.index("VADSpeechStartEvent")
    partial_idx = event_types.index("TranscriptPartialEvent")
    final_idx = event_types.index("TranscriptFinalEvent")
    end_idx = event_types.index("VADSpeechEndEvent")

    assert start_idx < partial_idx < final_idx < end_idx


async def test_hot_words_sent_only_on_first_frame():
    """Hot words are sent to worker only on the first frame of the segment."""
    # Arrange
    stream_handle = _make_stream_handle_mock()
    grpc_client = _make_grpc_client_mock(stream_handle)
    vad = make_vad_mock()
    preprocessor = make_preprocessor_mock()

    session = StreamingSession(
        session_id="test_session",
        preprocessor=preprocessor,
        vad=vad,
        grpc_client=grpc_client,
        postprocessor=_make_postprocessor_mock(),
        on_event=_make_on_event(),
        hot_words=["PIX", "TED", "Selic"],
    )

    # Trigger speech_start
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = True  # After speech_start, is_speaking = True
    await session.process_frame(make_raw_bytes())

    # Send more frames during speech
    vad.process_frame.return_value = None  # No transition
    await session.process_frame(make_raw_bytes())
    await session.process_frame(make_raw_bytes())

    # Assert: hot_words sent on first frame, None on subsequent ones
    calls = stream_handle.send_frame.call_args_list

    # First send_frame: should have hot_words
    assert calls[0].kwargs.get("hot_words") == ["PIX", "TED", "Selic"]

    # Second and third: no hot_words
    assert calls[1].kwargs.get("hot_words") is None
    assert calls[2].kwargs.get("hot_words") is None

    # Cleanup
    await session.close()


async def test_close_is_idempotent():
    """Calling close() multiple times does not cause error."""
    # Arrange
    session, _, _, _, _, _ = _make_session()

    # Act
    await session.close()
    await session.close()
    await session.close()

    # Assert
    assert session.is_closed


async def test_no_postprocessor_skips_itn():
    """If postprocessor is None, transcript.final is emitted without ITN."""
    # Arrange
    final_segment = TranscriptSegment(
        text="texto cru",
        is_final=True,
        segment_id=0,
        start_ms=1000,
        end_ms=2000,
    )

    stream_handle = _make_stream_handle_mock(events=[final_segment])
    grpc_client = _make_grpc_client_mock(stream_handle)
    vad = make_vad_mock()
    on_event = _make_on_event()

    session = StreamingSession(
        session_id="test_session",
        preprocessor=make_preprocessor_mock(),
        vad=vad,
        grpc_client=grpc_client,
        postprocessor=None,
        on_event=on_event,
        enable_itn=True,
    )

    # Trigger speech_start
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())

    await asyncio.sleep(0.05)

    # Trigger speech_end
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_END,
        timestamp_ms=2000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())

    # Assert: text without ITN
    final_calls = [
        call for call in on_event.call_args_list if isinstance(call.args[0], TranscriptFinalEvent)
    ]
    assert len(final_calls) == 1
    assert final_calls[0].args[0].text == "texto cru"


async def test_itn_disabled_skips_postprocessing():
    """If enable_itn=False, transcript.final is emitted without ITN."""
    # Arrange
    final_segment = TranscriptSegment(
        text="texto cru",
        is_final=True,
        segment_id=0,
        start_ms=1000,
        end_ms=2000,
    )

    stream_handle = _make_stream_handle_mock(events=[final_segment])
    grpc_client = _make_grpc_client_mock(stream_handle)
    vad = make_vad_mock()
    postprocessor = _make_postprocessor_mock()
    on_event = _make_on_event()

    session = StreamingSession(
        session_id="test_session",
        preprocessor=make_preprocessor_mock(),
        vad=vad,
        grpc_client=grpc_client,
        postprocessor=postprocessor,
        on_event=on_event,
        enable_itn=False,
    )

    # Trigger speech_start
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())

    await asyncio.sleep(0.05)

    # Trigger speech_end
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_END,
        timestamp_ms=2000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())

    # Assert: postprocessor was NOT called
    postprocessor.process.assert_not_called()

    # Original text emitted
    final_calls = [
        call for call in on_event.call_args_list if isinstance(call.args[0], TranscriptFinalEvent)
    ]
    assert len(final_calls) == 1
    assert final_calls[0].args[0].text == "texto cru"


async def test_worker_crash_emits_error_event():
    """Worker crash during streaming emits recoverable error."""
    # Arrange
    stream_handle = _make_stream_handle_mock(
        events=[WorkerCrashError("worker_1")],
    )

    grpc_client = _make_grpc_client_mock(stream_handle)
    vad = make_vad_mock()
    on_event = _make_on_event()

    session = StreamingSession(
        session_id="test_session",
        preprocessor=make_preprocessor_mock(),
        vad=vad,
        grpc_client=grpc_client,
        postprocessor=_make_postprocessor_mock(),
        on_event=on_event,
    )

    # Trigger speech_start (starts receiver task that will crash)
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())

    await asyncio.sleep(0.05)

    # Assert: error event emitted
    error_calls = [
        call for call in on_event.call_args_list if isinstance(call.args[0], StreamingErrorEvent)
    ]
    assert len(error_calls) >= 1
    error_event = error_calls[0].args[0]
    assert error_event.code == "worker_crash"
    assert error_event.recoverable is True

    # Cleanup
    await session.close()


async def test_inactivity_check():
    """check_inactivity() returns True after INIT timeout (30s default)."""
    # Arrange: use controllable clock in state machine
    current_time = [0.0]

    def fake_clock() -> float:
        return current_time[0]

    sm = SessionStateMachine(clock=fake_clock)
    session, _, _, _, _, _ = _make_session()
    # Inject state machine with controllable clock
    session._state_machine = sm

    # Recently created: not expired
    assert not session.check_inactivity()

    # Simulate 31s without audio (INIT timeout = 30s -> CLOSED)
    current_time[0] = 31.0
    assert session.check_inactivity()


async def test_inactivity_reset_on_frame():
    """process_frame() resets the inactivity timer via state machine transition.

    When a frame with speech_start is processed, the state machine transitions
    from INIT to ACTIVE (which has no timeout), preventing check_inactivity()
    from returning True. Frames in INIT (without speech) do not reset the state
    machine timer, but INIT state has a 30s timeout to CLOSED.
    """
    # Arrange: use controllable clock in state machine
    current_time = [0.0]

    def fake_clock() -> float:
        return current_time[0]

    sm = SessionStateMachine(clock=fake_clock)
    session, _, vad, _, _, _ = _make_session()
    session._state_machine = sm

    # Verify that in INIT with 25s elapsed, not expired
    current_time[0] = 25.0
    assert not session.check_inactivity()

    # Simulate speech_start -> transitions to ACTIVE (no timeout)
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = True
    current_time[0] = 26.0
    await session.process_frame(make_raw_bytes())
    await asyncio.sleep(0.01)

    # ACTIVE has no timeout, so check_inactivity returns False
    current_time[0] = 200.0
    assert not session.check_inactivity()

    # Cleanup
    await session.close()


async def test_word_timestamps_in_final():
    """Final transcript with word timestamps is correctly converted."""
    # Arrange
    final_segment = TranscriptSegment(
        text="ola mundo",
        is_final=True,
        segment_id=0,
        start_ms=1000,
        end_ms=2000,
        language="pt",
        confidence=0.95,
        words=(
            WordTimestamp(word="ola", start=1.0, end=1.5),
            WordTimestamp(word="mundo", start=1.5, end=2.0),
        ),
    )

    stream_handle = _make_stream_handle_mock(events=[final_segment])
    grpc_client = _make_grpc_client_mock(stream_handle)
    vad = make_vad_mock()
    on_event = _make_on_event()

    session = StreamingSession(
        session_id="test_session",
        preprocessor=make_preprocessor_mock(),
        vad=vad,
        grpc_client=grpc_client,
        postprocessor=_make_postprocessor_mock(),
        on_event=on_event,
        enable_itn=False,  # No ITN to simplify
    )

    # Trigger speech_start + speech_end
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())

    await asyncio.sleep(0.05)

    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_END,
        timestamp_ms=2000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())

    # Assert: word timestamps presentes no evento final
    final_calls = [
        call for call in on_event.call_args_list if isinstance(call.args[0], TranscriptFinalEvent)
    ]
    assert len(final_calls) == 1
    final_event = final_calls[0].args[0]
    assert final_event.words is not None
    assert len(final_event.words) == 2
    assert final_event.words[0].word == "ola"
    assert final_event.words[1].word == "mundo"


async def test_grpc_open_stream_failure_emits_error():
    """Failure to open gRPC stream emits recoverable error."""
    # Arrange
    grpc_client = AsyncMock()
    grpc_client.open_stream = AsyncMock(side_effect=WorkerCrashError("worker_1"))

    vad = make_vad_mock()
    on_event = _make_on_event()

    session = StreamingSession(
        session_id="test_session",
        preprocessor=make_preprocessor_mock(),
        vad=vad,
        grpc_client=grpc_client,
        postprocessor=_make_postprocessor_mock(),
        on_event=on_event,
    )

    # Trigger speech_start -> open_stream will fail
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())

    # Assert: error emitted
    error_calls = [
        call for call in on_event.call_args_list if isinstance(call.args[0], StreamingErrorEvent)
    ]
    assert len(error_calls) == 1
    assert error_calls[0].args[0].code == "worker_crash"
    assert error_calls[0].args[0].recoverable is True

    # Cleanup
    await session.close()


async def test_frames_sent_during_speech():
    """Audio frames are sent to worker during speech."""
    # Arrange
    stream_handle = _make_stream_handle_mock()
    grpc_client = _make_grpc_client_mock(stream_handle)
    vad = make_vad_mock()

    session = StreamingSession(
        session_id="test_session",
        preprocessor=make_preprocessor_mock(),
        vad=vad,
        grpc_client=grpc_client,
        postprocessor=_make_postprocessor_mock(),
        on_event=_make_on_event(),
    )

    # Trigger speech_start
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = True
    await session.process_frame(make_raw_bytes())

    # More frames during speech
    vad.process_frame.return_value = None
    await session.process_frame(make_raw_bytes())
    await session.process_frame(make_raw_bytes())

    # Assert: 3 frames enviados ao worker (1 do speech_start + 2)
    assert stream_handle.send_frame.call_count == 3

    # Cleanup
    await session.close()


async def test_no_frames_sent_during_silence():
    """Frames are NOT sent to worker when there is no speech."""
    # Arrange
    stream_handle = _make_stream_handle_mock()
    grpc_client = _make_grpc_client_mock(stream_handle)
    vad = make_vad_mock(is_speaking=False)

    session = StreamingSession(
        session_id="test_session",
        preprocessor=make_preprocessor_mock(),
        vad=vad,
        grpc_client=grpc_client,
        postprocessor=_make_postprocessor_mock(),
        on_event=_make_on_event(),
    )

    # Process frames in silence
    vad.process_frame.return_value = None
    await session.process_frame(make_raw_bytes())
    await session.process_frame(make_raw_bytes())

    # Assert: no frames sent
    stream_handle.send_frame.assert_not_called()


class TestEventOrdering:
    """Tests for event ordering: transcript.final BEFORE vad.speech_end."""

    async def test_transcript_final_emitted_before_speech_end(self) -> None:
        """transcript.final from worker is emitted BEFORE vad.speech_end.

        Ensures that the WebSocket protocol semantics are respected:
        the last transcript.final of a segment comes before vad.speech_end.
        """
        # Arrange: worker retorna um final transcript
        final_segment = TranscriptSegment(
            text="ola mundo",
            is_final=True,
            segment_id=0,
            start_ms=0,
            end_ms=1000,
            language="pt",
            confidence=0.95,
        )
        stream_handle = _make_stream_handle_mock(events=[final_segment])
        grpc_client = _make_grpc_client_mock(stream_handle)
        vad = make_vad_mock(is_speaking=False)

        events_emitted: list[object] = []

        async def capture_event(event: object) -> None:
            events_emitted.append(event)

        session = StreamingSession(
            session_id="test_session",
            preprocessor=make_preprocessor_mock(),
            vad=vad,
            grpc_client=grpc_client,
            postprocessor=None,
            on_event=capture_event,
            enable_itn=False,
        )

        # Act: simulate SPEECH_START -> frames -> SPEECH_END
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START,
            timestamp_ms=0,
        )
        await session.process_frame(make_raw_bytes())

        # Send frames during speech
        vad.process_frame.return_value = None
        vad.is_speaking = True
        await session.process_frame(make_raw_bytes())

        # Wait for receiver task to process the final (give time to event loop)
        await asyncio.sleep(0.05)

        # Emit SPEECH_END
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_END,
            timestamp_ms=2000,
        )
        vad.is_speaking = False
        await session.process_frame(make_raw_bytes())

        # Assert: verify ordering
        event_types = [type(e).__name__ for e in events_emitted]

        # Must have: speech_start, final, speech_end (in that order)
        assert "VADSpeechStartEvent" in event_types
        assert "TranscriptFinalEvent" in event_types
        assert "VADSpeechEndEvent" in event_types

        final_idx = event_types.index("TranscriptFinalEvent")
        end_idx = event_types.index("VADSpeechEndEvent")
        assert final_idx < end_idx, (
            f"transcript.final (idx={final_idx}) must come before "
            f"vad.speech_end (idx={end_idx}). Order: {event_types}"
        )

        await session.close()
