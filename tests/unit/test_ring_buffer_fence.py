"""Ring Buffer tests -- read fence, force commit, and StreamingSession integration.

Covers:
- Read fence (commit, uncommitted_bytes, available_for_write_bytes)
- Uncommitted data protection (BufferOverrunError)
- Force commit callback (>90% uncommitted usage)
- Integration with StreamingSession (write to ring buffer, commit after final,
  force commit via sync callback -> async flag)

Basic ring buffer functionality tests (write, read, wrap-around)
are in test_ring_buffer.py.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from macaw._types import TranscriptSegment
from macaw.exceptions import BufferOverrunError
from macaw.server.models.events import TranscriptFinalEvent
from macaw.session.ring_buffer import _DEFAULT_FORCE_COMMIT_THRESHOLD, RingBuffer
from macaw.session.streaming import StreamingSession
from macaw.vad.detector import VADEvent, VADEventType
from tests.helpers import (
    FRAME_SIZE,
    AsyncIterFromList,
    make_preprocessor_mock,
    make_raw_bytes,
    make_vad_mock,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    mock.process.side_effect = lambda text, **kwargs: text  # identity (no transformation)
    return mock


# ---------------------------------------------------------------------------
# Read Fence: properties and commit
# ---------------------------------------------------------------------------


class TestReadFence:
    """Tests for read fence (last_committed_offset)."""

    def test_read_fence_starts_at_zero(self) -> None:
        """Read fence starts at 0 (nothing committed)."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        assert rb.read_fence == 0

    def test_commit_advances_fence(self) -> None:
        """commit() advances the read fence to the given offset."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        rb.write(b"\x01" * 5)
        rb.commit(3)
        assert rb.read_fence == 3

    def test_commit_to_total_written(self) -> None:
        """commit() can advance up to total_written (everything committed)."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        rb.write(b"\x01" * 7)
        rb.commit(rb.total_written)
        assert rb.read_fence == 7
        assert rb.uncommitted_bytes == 0

    def test_commit_below_current_fence_raises(self) -> None:
        """commit() with offset less than the current fence raises ValueError."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        rb.write(b"\x01" * 5)
        rb.commit(3)
        with pytest.raises(ValueError, match="cannot be less"):
            rb.commit(2)

    def test_commit_above_total_written_raises(self) -> None:
        """commit() with offset greater than total_written raises ValueError."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        rb.write(b"\x01" * 5)
        with pytest.raises(ValueError, match="cannot exceed"):
            rb.commit(10)

    def test_commit_same_fence_is_noop(self) -> None:
        """commit() with the same offset as the current fence is a no-op (valid)."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        rb.write(b"\x01" * 5)
        rb.commit(3)
        rb.commit(3)  # No-op, should not raise
        assert rb.read_fence == 3

    def test_commit_incremental(self) -> None:
        """commit() can be called incrementally."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        rb.write(b"\x01" * 10)
        rb.commit(2)
        assert rb.read_fence == 2
        rb.commit(5)
        assert rb.read_fence == 5
        rb.commit(10)
        assert rb.read_fence == 10


# ---------------------------------------------------------------------------
# Uncommitted bytes e available_for_write_bytes
# ---------------------------------------------------------------------------


class TestUncommittedBytes:
    """Tests for uncommitted_bytes and available_for_write_bytes."""

    def test_uncommitted_bytes_equals_total_written_when_no_commit(self) -> None:
        """Without commit, everything is uncommitted."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        rb.write(b"\x01" * 7)
        assert rb.uncommitted_bytes == 7

    def test_uncommitted_bytes_decreases_after_commit(self) -> None:
        """uncommitted_bytes decreases after commit."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        rb.write(b"\x01" * 7)
        rb.commit(3)
        assert rb.uncommitted_bytes == 4

    def test_uncommitted_bytes_zero_after_full_commit(self) -> None:
        """uncommitted_bytes = 0 after full commit."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        rb.write(b"\x01" * 7)
        rb.commit(rb.total_written)
        assert rb.uncommitted_bytes == 0

    def test_available_for_write_equals_capacity_when_all_committed(self) -> None:
        """When everything is committed, available = capacity."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        # capacity = 10
        rb.write(b"\x01" * 7)
        rb.commit(rb.total_written)
        assert rb.available_for_write_bytes == 10

    def test_available_for_write_decreases_with_uncommitted(self) -> None:
        """available_for_write decreases with uncommitted data."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        # capacity = 10
        rb.write(b"\x01" * 7)
        # uncommitted = 7, available = 10 - 7 = 3
        assert rb.available_for_write_bytes == 3

    def test_available_for_write_zero_when_uncommitted_fills_capacity(self) -> None:
        """available_for_write = 0 when uncommitted >= capacity."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        # capacity = 10
        rb.write(b"\x01" * 10)
        # uncommitted = 10 = capacity, available = 0
        assert rb.available_for_write_bytes == 0


# ---------------------------------------------------------------------------
# Write protection (fence protects uncommitted data)
# ---------------------------------------------------------------------------


class TestWriteProtection:
    """Write protection tests -- fence prevents overwriting uncommitted data."""

    def test_write_raises_when_would_overwrite_uncommitted(self) -> None:
        """Write that would overwrite uncommitted data raises BufferOverrunError."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        # capacity = 10
        rb.write(b"\x01" * 10)
        # uncommitted = 10, available = 0
        with pytest.raises(BufferOverrunError, match="would overwrite"):
            rb.write(b"\x02" * 1)

    def test_write_succeeds_after_partial_commit(self) -> None:
        """Write succeeds after partial commit frees space."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        # capacity = 10
        rb.write(b"\x01" * 10)
        rb.commit(5)  # frees 5 bytes
        # available = 10 - 5 = 5
        rb.write(b"\x02" * 5)  # Should not raise
        assert rb.total_written == 15

    def test_write_exceeding_available_raises(self) -> None:
        """Write larger than available_for_write raises BufferOverrunError."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        rb.write(b"\x01" * 8)
        # uncommitted = 8, available = 2
        with pytest.raises(BufferOverrunError, match="would overwrite"):
            rb.write(b"\x02" * 3)

    def test_write_exactly_available_succeeds(self) -> None:
        """Write exactly the size of available does not raise an error."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        rb.write(b"\x01" * 8)
        # uncommitted = 8, available = 2
        rb.write(b"\x02" * 2)  # Should not raise
        assert rb.total_written == 10

    def test_write_unrestricted_when_all_committed(self) -> None:
        """Write is free when everything is committed (nothing to protect)."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        rb.write(b"\x01" * 10)
        rb.commit(rb.total_written)
        # Everything committed -- write is free, even if > capacity
        rb.write(b"\x02" * 25)
        assert rb.total_written == 35

    def test_write_with_zero_uncommitted_allows_large_write(self) -> None:
        """Zero uncommitted allows writes larger than capacity."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        # fence=0, total_written=0 -> uncommitted=0
        data = bytes(range(25))
        rb.write(data)  # 25 > 10 (capacity), mas uncommitted=0 -> OK
        assert rb.total_written == 25


# ---------------------------------------------------------------------------
# Force commit callback
# ---------------------------------------------------------------------------


class TestForceCommit:
    """Tests for on_force_commit callback (>90% uncommitted usage)."""

    def test_force_commit_threshold_is_90_percent(self) -> None:
        """Force commit threshold is 90%."""
        assert pytest.approx(0.90) == _DEFAULT_FORCE_COMMIT_THRESHOLD

    def test_force_commit_triggered_above_90_percent(self) -> None:
        """Callback is invoked when uncommitted > 90% of capacity."""
        callback = Mock()
        rb = RingBuffer(
            duration_s=1.0,
            sample_rate=10,
            bytes_per_sample=1,
            on_force_commit=callback,
        )
        # capacity = 10, 90% = 9 bytes

        # Write 10 bytes (100% uncommitted) -> triggers callback
        rb.write(b"\x01" * 10)

        callback.assert_called_once_with(10)  # total_written = 10

    def test_force_commit_not_triggered_at_90_percent(self) -> None:
        """Callback is NOT invoked when uncommitted = exactly 90%."""
        callback = Mock()
        rb = RingBuffer(
            duration_s=1.0,
            sample_rate=10,
            bytes_per_sample=1,
            on_force_commit=callback,
        )
        # capacity = 10, 90% = 9 bytes. Threshold e STRICT > 0.90

        # Write exactly 9 bytes (90% = exactly threshold, not > threshold)
        rb.write(b"\x01" * 9)

        callback.assert_not_called()

    def test_force_commit_not_triggered_below_90_percent(self) -> None:
        """Callback is NOT invoked when uncommitted < 90%."""
        callback = Mock()
        rb = RingBuffer(
            duration_s=1.0,
            sample_rate=10,
            bytes_per_sample=1,
            on_force_commit=callback,
        )
        # capacity = 10

        rb.write(b"\x01" * 8)  # 80%
        callback.assert_not_called()

    def test_force_commit_not_triggered_when_committed(self) -> None:
        """Callback is NOT invoked when data is already committed."""
        callback = Mock()
        rb = RingBuffer(
            duration_s=1.0,
            sample_rate=10,
            bytes_per_sample=1,
            on_force_commit=callback,
        )
        # capacity = 10

        rb.write(b"\x01" * 5)
        rb.commit(rb.total_written)  # Everything committed

        # Write 5 more -- uncommitted = 5 (50%) -> no callback
        rb.write(b"\x02" * 5)
        callback.assert_not_called()

    def test_force_commit_not_triggered_without_callback(self) -> None:
        """Without a configured callback, no error occurs when exceeding 90%."""
        rb = RingBuffer(
            duration_s=1.0,
            sample_rate=10,
            bytes_per_sample=1,
            on_force_commit=None,
        )
        # capacity = 10

        # Write 10 bytes without callback -- should not raise an error
        rb.write(b"\x01" * 10)
        assert rb.total_written == 10

    def test_force_commit_called_multiple_times(self) -> None:
        """Callback can be called multiple times as buffer fills up."""
        callback = Mock()
        rb = RingBuffer(
            duration_s=1.0,
            sample_rate=100,
            bytes_per_sample=1,
            on_force_commit=callback,
        )
        # capacity = 100

        # Primeira escrita: 95 bytes (95% > 90%)
        rb.write(b"\x01" * 95)
        assert callback.call_count == 1

        # Commit everything and write again
        rb.commit(rb.total_written)
        rb.write(b"\x02" * 91)  # 91% > 90%
        assert callback.call_count == 2


# ---------------------------------------------------------------------------
# Integration: StreamingSession + Ring Buffer
# ---------------------------------------------------------------------------


class TestStreamingSessionRingBuffer:
    """Integration tests for StreamingSession with Ring Buffer."""

    async def test_session_writes_pcm_to_ring_buffer(self) -> None:
        """StreamingSession writes PCM frames to the ring buffer during speech."""
        # Arrange
        rb = RingBuffer(duration_s=5.0, sample_rate=16000, bytes_per_sample=2)
        stream_handle = _make_stream_handle_mock()
        grpc_client = _make_grpc_client_mock(stream_handle)
        vad = make_vad_mock()

        session = StreamingSession(
            session_id="test_session",
            preprocessor=make_preprocessor_mock(),
            vad=vad,
            grpc_client=grpc_client,
            postprocessor=_make_postprocessor_mock(),
            on_event=AsyncMock(),
            ring_buffer=rb,
        )

        # Trigger speech_start
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START,
            timestamp_ms=0,
        )
        vad.is_speaking = True
        await session.process_frame(make_raw_bytes())

        # Send more frames during speech
        vad.process_frame.return_value = None
        await session.process_frame(make_raw_bytes())
        await session.process_frame(make_raw_bytes())

        # Assert: ring buffer has written data.
        # Each frame is 1024 samples * 2 bytes = 2048 bytes PCM int16.
        expected_bytes = 3 * FRAME_SIZE * 2  # 3 frames
        assert rb.total_written == expected_bytes

        # Cleanup
        await session.close()

    async def test_session_without_ring_buffer_works(self) -> None:
        """StreamingSession works normally without ring buffer (backward compat)."""
        stream_handle = _make_stream_handle_mock()
        grpc_client = _make_grpc_client_mock(stream_handle)
        vad = make_vad_mock()

        session = StreamingSession(
            session_id="test_session",
            preprocessor=make_preprocessor_mock(),
            vad=vad,
            grpc_client=grpc_client,
            postprocessor=_make_postprocessor_mock(),
            on_event=AsyncMock(),
            ring_buffer=None,  # No ring buffer
        )

        # Trigger speech_start and send frames -- should not raise an error
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START,
            timestamp_ms=0,
        )
        vad.is_speaking = True
        await session.process_frame(make_raw_bytes())

        vad.process_frame.return_value = None
        await session.process_frame(make_raw_bytes())

        # Assert: frames sent to the worker normally
        assert stream_handle.send_frame.call_count == 2

        await session.close()

    async def test_ring_buffer_fence_advances_on_transcript_final(self) -> None:
        """Ring buffer read fence advances after transcript.final."""
        # Arrange
        final_segment = TranscriptSegment(
            text="ola mundo",
            is_final=True,
            segment_id=0,
            start_ms=0,
            end_ms=1000,
            language="pt",
            confidence=0.95,
        )

        rb = RingBuffer(duration_s=5.0, sample_rate=16000, bytes_per_sample=2)
        stream_handle = _make_stream_handle_mock(events=[final_segment])
        grpc_client = _make_grpc_client_mock(stream_handle)
        vad = make_vad_mock()
        on_event = AsyncMock()

        session = StreamingSession(
            session_id="test_session",
            preprocessor=make_preprocessor_mock(),
            vad=vad,
            grpc_client=grpc_client,
            postprocessor=_make_postprocessor_mock(),
            on_event=on_event,
            enable_itn=False,
            ring_buffer=rb,
        )

        # Trigger speech_start (opens stream, starts receiver)
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START,
            timestamp_ms=0,
        )
        vad.is_speaking = True
        await session.process_frame(make_raw_bytes())

        # Send more frames
        vad.process_frame.return_value = None
        await session.process_frame(make_raw_bytes())

        # Wait for receiver task to process the transcript.final
        await asyncio.sleep(0.05)

        # Assert: fence advanced to total_written
        assert rb.read_fence == rb.total_written
        assert rb.uncommitted_bytes == 0

        # Verify that transcript.final was emitted
        final_calls = [
            call
            for call in on_event.call_args_list
            if isinstance(call.args[0], TranscriptFinalEvent)
        ]
        assert len(final_calls) == 1

        await session.close()

    async def test_force_commit_triggers_session_commit(self) -> None:
        """Force commit from ring buffer (>90%) triggers session.commit()."""
        # Arrange: ring buffer pequeno para facilitar atingir 90%
        # 0.01s * 16000 * 2 = 320 bytes de capacidade
        rb = RingBuffer(duration_s=0.01, sample_rate=16000, bytes_per_sample=2)

        stream_handle = _make_stream_handle_mock()
        grpc_client = _make_grpc_client_mock(stream_handle)
        vad = make_vad_mock()

        session = StreamingSession(
            session_id="test_session",
            preprocessor=make_preprocessor_mock(),
            vad=vad,
            grpc_client=grpc_client,
            postprocessor=_make_postprocessor_mock(),
            on_event=AsyncMock(),
            ring_buffer=rb,
        )

        # Verify that the callback was wired
        assert rb._on_force_commit is not None

        # Trigger speech_start
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START,
            timestamp_ms=0,
        )
        vad.is_speaking = True

        # Use small frames to control the exact size.
        # Each default frame = 1024 * 2 = 2048 bytes.
        # rb.capacity = 320 bytes. First frame already exceeds capacity,
        # but fence=0 and total_written=0 so uncommitted=0 and write is free.
        # After the write, uncommitted = 2048 > 320 * 0.9 = 288 -> force commit.

        # The force commit sets _force_commit_pending = True
        # And at the end of process_frame, commit() is called.
        await session.process_frame(make_raw_bytes())

        # After process_frame with force commit:
        # - commit() closes the stream handle and waits for receiver
        # - segment_id increments
        assert session.segment_id == 1  # Incremented because of the commit

        await session.close()

    async def test_ring_buffer_callback_wired_in_init(self) -> None:
        """StreamingSession configures on_force_commit on the ring buffer when created."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        assert rb._on_force_commit is None  # Before creating session

        session = StreamingSession(
            session_id="test_session",
            preprocessor=make_preprocessor_mock(),
            vad=make_vad_mock(),
            grpc_client=_make_grpc_client_mock(),
            postprocessor=_make_postprocessor_mock(),
            on_event=AsyncMock(),
            ring_buffer=rb,
        )

        # After creating session, callback is wired
        assert rb._on_force_commit is not None
        assert rb._on_force_commit == session._metrics.on_ring_buffer_force_commit

        await session.close()

    async def test_force_commit_flag_reset_after_commit(self) -> None:
        """Force commit flag is consumed after process_frame executes commit."""
        rb = RingBuffer(duration_s=0.01, sample_rate=16000, bytes_per_sample=2)

        session = StreamingSession(
            session_id="test_session",
            preprocessor=make_preprocessor_mock(),
            vad=make_vad_mock(),
            grpc_client=_make_grpc_client_mock(),
            postprocessor=_make_postprocessor_mock(),
            on_event=AsyncMock(),
            ring_buffer=rb,
        )

        # Trigger force commit via callback
        session._metrics.on_ring_buffer_force_commit(0)

        vad = session._vad
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START,
            timestamp_ms=0,
        )
        vad.is_speaking = True

        # process_frame will: preprocess, VAD, send to worker, check flag
        await session.process_frame(make_raw_bytes())

        # Flag consumed by process_frame via consume_force_commit()
        assert session._metrics.consume_force_commit() is False

        await session.close()
