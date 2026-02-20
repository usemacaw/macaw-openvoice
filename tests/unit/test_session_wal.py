"""Tests for In-Memory WAL (Write-Ahead Log) for session recovery.

Covers:
- Initialization with zero values
- record_checkpoint updates all fields atomically
- Individual access properties (segment_id, buffer_offset, timestamp_ms)
- Multiple checkpoints: each overwrites the previous one
- WALCheckpoint is frozen (immutable)
- Integration: StreamingSession records checkpoint after transcript.final
"""

from __future__ import annotations

import asyncio
import dataclasses
from unittest.mock import AsyncMock, Mock

import pytest

from macaw._types import TranscriptSegment
from macaw.server.models.events import TranscriptFinalEvent
from macaw.session.ring_buffer import RingBuffer
from macaw.session.streaming import StreamingSession
from macaw.session.wal import SessionWAL, WALCheckpoint
from macaw.vad.detector import VADEvent, VADEventType
from tests.helpers import AsyncIterFromList, make_preprocessor_mock, make_raw_bytes, make_vad_mock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stream_handle_mock(events: list | None = None) -> Mock:
    """Create StreamHandle mock."""
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
    mock.process.side_effect = lambda text, **kwargs: text
    return mock


# ---------------------------------------------------------------------------
# SessionWAL: initialization and properties
# ---------------------------------------------------------------------------


class TestSessionWALInit:
    """Tests for initialization and initial state of WAL."""

    def test_wal_initializes_with_zero_values(self) -> None:
        """WAL starts with segment_id=0, buffer_offset=0, timestamp_ms=0."""
        wal = SessionWAL()
        assert wal.last_committed_segment_id == 0
        assert wal.last_committed_buffer_offset == 0
        assert wal.last_committed_timestamp_ms == 0

    def test_wal_checkpoint_initializes_correctly(self) -> None:
        """Checkpoint property returns WALCheckpoint with initial values."""
        wal = SessionWAL()
        cp = wal.checkpoint
        assert isinstance(cp, WALCheckpoint)
        assert cp.segment_id == 0
        assert cp.buffer_offset == 0
        assert cp.timestamp_ms == 0


# ---------------------------------------------------------------------------
# SessionWAL: record_checkpoint and access
# ---------------------------------------------------------------------------


class TestSessionWALRecordCheckpoint:
    """Tests for record_checkpoint and value reading."""

    def test_record_checkpoint_updates_all_fields(self) -> None:
        """record_checkpoint updates segment_id, buffer_offset and timestamp_ms."""
        wal = SessionWAL()
        wal.record_checkpoint(segment_id=3, buffer_offset=4096, timestamp_ms=12345)

        assert wal.last_committed_segment_id == 3
        assert wal.last_committed_buffer_offset == 4096
        assert wal.last_committed_timestamp_ms == 12345

    def test_last_committed_segment_id_after_checkpoint(self) -> None:
        """last_committed_segment_id returns correct value after checkpoint."""
        wal = SessionWAL()
        wal.record_checkpoint(segment_id=7, buffer_offset=0, timestamp_ms=0)
        assert wal.last_committed_segment_id == 7

    def test_last_committed_buffer_offset_after_checkpoint(self) -> None:
        """last_committed_buffer_offset returns correct value after checkpoint."""
        wal = SessionWAL()
        wal.record_checkpoint(segment_id=0, buffer_offset=99999, timestamp_ms=0)
        assert wal.last_committed_buffer_offset == 99999

    def test_last_committed_timestamp_ms_after_checkpoint(self) -> None:
        """last_committed_timestamp_ms returns correct value after checkpoint."""
        wal = SessionWAL()
        wal.record_checkpoint(segment_id=0, buffer_offset=0, timestamp_ms=987654)
        assert wal.last_committed_timestamp_ms == 987654

    def test_multiple_checkpoints_overwrite_previous(self) -> None:
        """Each checkpoint overwrites the previous one (not append-only)."""
        wal = SessionWAL()

        wal.record_checkpoint(segment_id=1, buffer_offset=1000, timestamp_ms=100)
        assert wal.last_committed_segment_id == 1

        wal.record_checkpoint(segment_id=2, buffer_offset=2000, timestamp_ms=200)
        assert wal.last_committed_segment_id == 2
        assert wal.last_committed_buffer_offset == 2000
        assert wal.last_committed_timestamp_ms == 200

        # Previous checkpoint is not accessible
        wal.record_checkpoint(segment_id=5, buffer_offset=8000, timestamp_ms=500)
        assert wal.last_committed_segment_id == 5
        assert wal.last_committed_buffer_offset == 8000
        assert wal.last_committed_timestamp_ms == 500

    def test_checkpoint_property_returns_current_checkpoint(self) -> None:
        """Checkpoint property returns the most recent WALCheckpoint."""
        wal = SessionWAL()
        wal.record_checkpoint(segment_id=4, buffer_offset=5000, timestamp_ms=300)
        cp = wal.checkpoint
        assert cp.segment_id == 4
        assert cp.buffer_offset == 5000
        assert cp.timestamp_ms == 300


# ---------------------------------------------------------------------------
# WALCheckpoint: immutability
# ---------------------------------------------------------------------------


class TestWALCheckpointImmutability:
    """Tests for WALCheckpoint immutability."""

    def test_wal_checkpoint_is_frozen(self) -> None:
        """WALCheckpoint is frozen (immutable) -- assignment raises FrozenInstanceError."""
        cp = WALCheckpoint(segment_id=1, buffer_offset=100, timestamp_ms=50)
        with pytest.raises(dataclasses.FrozenInstanceError):
            cp.segment_id = 99  # type: ignore[misc]

    def test_wal_checkpoint_uses_slots(self) -> None:
        """WALCheckpoint uses __slots__ for memory savings."""
        cp = WALCheckpoint(segment_id=1, buffer_offset=100, timestamp_ms=50)
        assert hasattr(cp, "__slots__")
        # frozen + slots: assigning a nonexistent attribute raises error
        # (FrozenInstanceError intercepts before slots AttributeError)
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
            cp.extra_field = "should not work"  # type: ignore[attr-defined]

    def test_wal_checkpoint_equality(self) -> None:
        """Two WALCheckpoints with same values are equal (dataclass)."""
        cp1 = WALCheckpoint(segment_id=1, buffer_offset=100, timestamp_ms=50)
        cp2 = WALCheckpoint(segment_id=1, buffer_offset=100, timestamp_ms=50)
        assert cp1 == cp2

    def test_wal_checkpoint_inequality(self) -> None:
        """WALCheckpoints with different values are not equal."""
        cp1 = WALCheckpoint(segment_id=1, buffer_offset=100, timestamp_ms=50)
        cp2 = WALCheckpoint(segment_id=2, buffer_offset=100, timestamp_ms=50)
        assert cp1 != cp2


# ---------------------------------------------------------------------------
# Integration: StreamingSession + WAL
# ---------------------------------------------------------------------------


class TestStreamingSessionWAL:
    """Integration tests for StreamingSession with SessionWAL."""

    async def test_session_creates_default_wal(self) -> None:
        """StreamingSession creates default WAL if none provided."""
        session = StreamingSession(
            session_id="test",
            preprocessor=make_preprocessor_mock(),
            vad=make_vad_mock(),
            grpc_client=_make_grpc_client_mock(),
            postprocessor=_make_postprocessor_mock(),
            on_event=AsyncMock(),
        )
        assert isinstance(session.wal, SessionWAL)
        assert session.wal.last_committed_segment_id == 0
        await session.close()

    async def test_session_uses_injected_wal(self) -> None:
        """StreamingSession uses injected WAL instead of creating default."""
        custom_wal = SessionWAL()
        custom_wal.record_checkpoint(segment_id=42, buffer_offset=0, timestamp_ms=0)

        session = StreamingSession(
            session_id="test",
            preprocessor=make_preprocessor_mock(),
            vad=make_vad_mock(),
            grpc_client=_make_grpc_client_mock(),
            postprocessor=_make_postprocessor_mock(),
            on_event=AsyncMock(),
            wal=custom_wal,
        )
        assert session.wal is custom_wal
        assert session.wal.last_committed_segment_id == 42
        await session.close()

    async def test_session_records_wal_checkpoint_after_transcript_final(self) -> None:
        """StreamingSession records WAL checkpoint after transcript.final."""
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
        wal = SessionWAL()

        session = StreamingSession(
            session_id="test_session",
            preprocessor=make_preprocessor_mock(),
            vad=vad,
            grpc_client=grpc_client,
            postprocessor=_make_postprocessor_mock(),
            on_event=on_event,
            enable_itn=False,
            ring_buffer=rb,
            wal=wal,
        )

        # Trigger speech_start
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

        # Assert: WAL checkpoint recorded
        assert wal.last_committed_segment_id == 0  # segment_id at time of final
        assert wal.last_committed_buffer_offset == rb.total_written
        assert wal.last_committed_timestamp_ms > 0  # monotonic timestamp

        # Verify that transcript.final was also emitted
        final_calls = [
            call
            for call in on_event.call_args_list
            if isinstance(call.args[0], TranscriptFinalEvent)
        ]
        assert len(final_calls) == 1

        await session.close()

    async def test_session_records_wal_without_ring_buffer(self) -> None:
        """StreamingSession records WAL checkpoint even without ring buffer."""
        final_segment = TranscriptSegment(
            text="sem ring buffer",
            is_final=True,
            segment_id=0,
            start_ms=0,
            end_ms=500,
            language="pt",
            confidence=0.9,
        )

        stream_handle = _make_stream_handle_mock(events=[final_segment])
        grpc_client = _make_grpc_client_mock(stream_handle)
        vad = make_vad_mock()
        wal = SessionWAL()

        session = StreamingSession(
            session_id="test_session",
            preprocessor=make_preprocessor_mock(),
            vad=vad,
            grpc_client=grpc_client,
            postprocessor=_make_postprocessor_mock(),
            on_event=AsyncMock(),
            enable_itn=False,
            ring_buffer=None,
            wal=wal,
        )

        # Trigger speech_start
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START,
            timestamp_ms=0,
        )
        vad.is_speaking = True
        await session.process_frame(make_raw_bytes())

        # Wait for receiver task
        await asyncio.sleep(0.05)

        # WAL checkpoint with buffer_offset=0 (no ring buffer)
        assert wal.last_committed_segment_id == 0
        assert wal.last_committed_buffer_offset == 0
        assert wal.last_committed_timestamp_ms > 0

        await session.close()

    async def test_session_wal_multiple_finals_overwrite(self) -> None:
        """Multiple transcript.final overwrite previous WAL checkpoint."""
        final1 = TranscriptSegment(
            text="primeiro",
            is_final=True,
            segment_id=0,
            start_ms=0,
            end_ms=500,
        )
        final2 = TranscriptSegment(
            text="segundo",
            is_final=True,
            segment_id=0,
            start_ms=500,
            end_ms=1000,
        )

        rb = RingBuffer(duration_s=5.0, sample_rate=16000, bytes_per_sample=2)
        stream_handle = _make_stream_handle_mock(events=[final1, final2])
        grpc_client = _make_grpc_client_mock(stream_handle)
        vad = make_vad_mock()
        wal = SessionWAL()

        session = StreamingSession(
            session_id="test_session",
            preprocessor=make_preprocessor_mock(),
            vad=vad,
            grpc_client=grpc_client,
            postprocessor=_make_postprocessor_mock(),
            on_event=AsyncMock(),
            enable_itn=False,
            ring_buffer=rb,
            wal=wal,
        )

        # Trigger speech_start and send frames
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START,
            timestamp_ms=0,
        )
        vad.is_speaking = True
        await session.process_frame(make_raw_bytes())
        vad.process_frame.return_value = None
        await session.process_frame(make_raw_bytes())

        # Wait for both finals to be processed
        await asyncio.sleep(0.05)

        # WAL should have the checkpoint from the last final
        assert wal.last_committed_segment_id == 0  # same segment, 2 finals
        assert wal.last_committed_buffer_offset == rb.total_written
        # Second checkpoint has timestamp >= first (monotonic)
        assert wal.last_committed_timestamp_ms > 0

        await session.close()
