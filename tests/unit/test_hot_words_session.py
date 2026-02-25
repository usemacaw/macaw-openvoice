"""Tests for per-session hot words in StreamingSession.

Validates that hot words are:
- Stored correctly on init and via update_hot_words().
- Sent to the worker only on the first frame of each speech segment.
- Updated dynamically and used in the next segment.

All tests use mocks for external dependencies.
Tests are deterministic -- no dependency on real timing.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

from macaw.session.streaming import StreamingSession
from macaw.vad.detector import VADEvent, VADEventType
from tests.helpers import make_float32_frame, make_raw_bytes


def _make_stream_handle_mock() -> Mock:
    """Create StreamHandle mock."""
    handle = Mock()
    handle.is_closed = False
    handle.session_id = "test_session"

    # receive_events returns empty async iterator
    async def _empty_iter():
        return
        yield  # needed to make it an async generator

    handle.receive_events.return_value = _empty_iter()
    handle.send_frame = AsyncMock()
    handle.close = AsyncMock()
    handle.cancel = AsyncMock()
    return handle


def _make_session(
    hot_words: list[str] | None = None,
) -> tuple[StreamingSession, Mock, Mock, AsyncMock, AsyncMock]:
    """Create StreamingSession with configured mocks.

    Returns:
        (session, preprocessor, vad, stream_handle, on_event)
    """
    preprocessor = Mock()
    preprocessor.process_frame.return_value = make_float32_frame()

    vad = Mock()
    vad.process_frame.return_value = None
    vad.is_speaking = False
    vad.reset.return_value = None

    stream_handle = _make_stream_handle_mock()
    grpc_client = AsyncMock()
    grpc_client.open_stream = AsyncMock(return_value=stream_handle)

    on_event = AsyncMock()

    session = StreamingSession(
        session_id="test-session",
        preprocessor=preprocessor,
        vad=vad,
        grpc_client=grpc_client,
        postprocessor=None,
        on_event=on_event,
        hot_words=hot_words,
        enable_itn=False,
    )

    return session, preprocessor, vad, stream_handle, on_event


# ---------------------------------------------------------------------------
# Tests: hot words storage
# ---------------------------------------------------------------------------


async def test_session_created_with_hot_words():
    """Hot words provided on init are stored correctly."""
    # Arrange & Act
    session, _, _, _, _ = _make_session(hot_words=["PIX", "TED", "Selic"])

    # Assert
    assert session._hot_words == ["PIX", "TED", "Selic"]

    # Cleanup
    await session.close()


async def test_session_created_without_hot_words():
    """Session created without hot words has None as default."""
    # Arrange & Act
    session, _, _, _, _ = _make_session(hot_words=None)

    # Assert
    assert session._hot_words is None

    # Cleanup
    await session.close()


async def test_update_hot_words():
    """update_hot_words() changes the stored hot words."""
    # Arrange
    session, _, _, _, _ = _make_session(hot_words=["PIX"])

    # Act
    session.update_hot_words(["TED", "Selic", "CDI"])

    # Assert
    assert session._hot_words == ["TED", "Selic", "CDI"]

    # Cleanup
    await session.close()


async def test_update_hot_words_to_none():
    """update_hot_words(None) clears the hot words."""
    # Arrange
    session, _, _, _, _ = _make_session(hot_words=["PIX", "TED"])

    # Act
    session.update_hot_words(None)

    # Assert
    assert session._hot_words is None

    # Cleanup
    await session.close()


# ---------------------------------------------------------------------------
# Tests: sending hot words to worker
# ---------------------------------------------------------------------------


async def test_hot_words_sent_on_first_frame():
    """Hot words are sent to the worker on the first frame of the segment."""
    # Arrange
    session, _, vad, stream_handle, _ = _make_session(
        hot_words=["PIX", "TED", "Selic"],
    )

    # Act: trigger speech_start
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = True  # After speech_start
    await session.process_frame(make_raw_bytes())

    # Assert: first send_frame should include hot_words
    assert stream_handle.send_frame.call_count == 1
    first_call = stream_handle.send_frame.call_args_list[0]
    assert first_call.kwargs.get("hot_words") == ["PIX", "TED", "Selic"]

    # Cleanup
    await session.close()


async def test_hot_words_not_sent_on_subsequent_frames():
    """Hot words are NOT sent on subsequent frames of the same segment."""
    # Arrange
    session, _, vad, stream_handle, _ = _make_session(
        hot_words=["PIX", "TED"],
    )

    # Act: trigger speech_start + send 2 additional frames
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = True
    await session.process_frame(make_raw_bytes())

    vad.process_frame.return_value = None  # No transition
    await session.process_frame(make_raw_bytes())
    await session.process_frame(make_raw_bytes())

    # Assert: 3 frames sent
    assert stream_handle.send_frame.call_count == 3

    # Second and third frames: hot_words should be None
    second_call = stream_handle.send_frame.call_args_list[1]
    third_call = stream_handle.send_frame.call_args_list[2]
    assert second_call.kwargs.get("hot_words") is None
    assert third_call.kwargs.get("hot_words") is None

    # Cleanup
    await session.close()


async def test_hot_words_reset_on_new_segment():
    """Hot words are resent on the first frame of a new segment."""
    # Arrange
    session, _, vad, _, _ = _make_session(
        hot_words=["PIX"],
    )
    grpc_client = session._grpc_client

    # First segment: speech_start -> speech_end
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = True
    await session.process_frame(make_raw_bytes())
    await asyncio.sleep(0.01)

    # Prepare new stream_handle for next segment
    stream_handle_2 = _make_stream_handle_mock()
    grpc_client.open_stream = AsyncMock(return_value=stream_handle_2)

    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_END,
        timestamp_ms=2000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())

    # Second segment: speech_start
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=3000,
    )
    vad.is_speaking = True
    await session.process_frame(make_raw_bytes())

    # Assert: hot_words sent on first frame of the second segment
    assert stream_handle_2.send_frame.call_count == 1
    first_call_seg2 = stream_handle_2.send_frame.call_args_list[0]
    assert first_call_seg2.kwargs.get("hot_words") == ["PIX"]

    # Cleanup
    await session.close()


async def test_updated_hot_words_used_in_next_segment():
    """After update_hot_words(), the NEW hot words are used in the next segment."""
    # Arrange
    session, _, vad, stream_handle, _ = _make_session(
        hot_words=["PIX", "TED"],
    )
    grpc_client = session._grpc_client

    # First segment: speech_start -> frame -> speech_end
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = True
    await session.process_frame(make_raw_bytes())
    await asyncio.sleep(0.01)

    # Verify the original hot words were sent
    first_call = stream_handle.send_frame.call_args_list[0]
    assert first_call.kwargs.get("hot_words") == ["PIX", "TED"]

    # Prepare new stream_handle for next segment
    stream_handle_2 = _make_stream_handle_mock()
    grpc_client.open_stream = AsyncMock(return_value=stream_handle_2)

    # Speech end: closes first segment
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_END,
        timestamp_ms=2000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())

    # Act: update hot words between segments
    session.update_hot_words(["Selic", "CDI", "IPCA"])

    # Second segment: speech_start
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=3000,
    )
    vad.is_speaking = True
    await session.process_frame(make_raw_bytes())

    # Assert: UPDATED hot words sent on first frame of the second segment
    assert stream_handle_2.send_frame.call_count == 1
    first_call_seg2 = stream_handle_2.send_frame.call_args_list[0]
    assert first_call_seg2.kwargs.get("hot_words") == ["Selic", "CDI", "IPCA"]

    # Cleanup
    await session.close()


async def test_no_hot_words_sent_when_none():
    """When hot_words is None, no hot words are sent to the worker."""
    # Arrange
    session, _, vad, stream_handle, _ = _make_session(hot_words=None)

    # Act: trigger speech_start + frame
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = True
    await session.process_frame(make_raw_bytes())

    # Assert: send_frame called without hot_words
    assert stream_handle.send_frame.call_count == 1
    first_call = stream_handle.send_frame.call_args_list[0]
    assert first_call.kwargs.get("hot_words") is None

    # Cleanup
    await session.close()


async def test_update_hot_words_empty_list_sent_as_none():
    """Empty hot words list results in None sent to the worker.

    The logic in _send_frame_to_worker checks `if self._hot_words` (truthy),
    so an empty list is treated as falsy and is not sent.
    """
    # Arrange
    session, _, vad, stream_handle, _ = _make_session(hot_words=["PIX"])

    # Act: update to empty list
    session.update_hot_words([])

    # Trigger speech_start
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = True
    await session.process_frame(make_raw_bytes())

    # Assert: hot_words not sent (empty list is falsy)
    assert stream_handle.send_frame.call_count == 1
    first_call = stream_handle.send_frame.call_args_list[0]
    assert first_call.kwargs.get("hot_words") is None

    # Cleanup
    await session.close()
