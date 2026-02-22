"""Unit tests for StreamingGRPCClient and StreamHandle.

All tests use mocks for gRPC -- no real server is started.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import grpc.aio
import pytest

from macaw.exceptions import WorkerCrashError, WorkerTimeoutError
from macaw.proto.stt_worker_pb2 import AudioFrame, TranscriptEvent, Word
from macaw.scheduler.streaming import (
    StreamHandle,
    StreamingGRPCClient,
    _proto_event_to_transcript_segment,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from macaw._types import TranscriptSegment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class AsyncIterableFromList:
    """Wrapper that makes a list iterable with `async for`."""

    def __init__(self, items: Sequence[object]) -> None:
        self._items = items
        self._index = 0

    def __aiter__(self) -> AsyncIterableFromList:
        return self

    async def __anext__(self) -> object:
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        return item


def _make_async_iterable_call(
    mock_call: AsyncMock,
    events: Sequence[object],
) -> None:
    """Configures mock_call to be async-iterable over the list of events."""
    ait = AsyncIterableFromList(events)
    mock_call.__aiter__ = MagicMock(return_value=ait)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_call() -> AsyncMock:
    """Creates a mock of grpc.aio.StreamStreamCall."""
    call = AsyncMock(spec_set=["write", "done_writing", "cancel", "__aiter__", "__anext__"])
    call.write = AsyncMock()
    call.done_writing = AsyncMock()
    call.cancel = MagicMock()
    # Default: empty iterator
    call.__aiter__ = MagicMock(return_value=call)
    call.__anext__ = AsyncMock(side_effect=StopAsyncIteration)
    return call


@pytest.fixture
def stream_handle(mock_call: AsyncMock) -> StreamHandle:
    """Creates a StreamHandle with mock call."""
    return StreamHandle(session_id="sess_test_001", call=mock_call)


# ---------------------------------------------------------------------------
# StreamHandle — send_frame
# ---------------------------------------------------------------------------


class TestStreamHandleSendFrame:
    """Tests for StreamHandle.send_frame()."""

    async def test_send_frame_creates_correct_audio_frame(
        self,
        stream_handle: StreamHandle,
        mock_call: AsyncMock,
    ) -> None:
        """send_frame should create AudioFrame with correct fields and call write."""
        pcm_data = b"\x00\x01" * 160  # 20ms a 16kHz

        await stream_handle.send_frame(
            pcm_data=pcm_data,
            initial_prompt="Contexto anterior",
            hot_words=["PIX", "TED"],
        )

        mock_call.write.assert_called_once()
        frame: AudioFrame = mock_call.write.call_args[0][0]
        assert frame.session_id == "sess_test_001"
        assert frame.data == pcm_data
        assert frame.is_last is False
        assert frame.initial_prompt == "Contexto anterior"
        assert list(frame.hot_words) == ["PIX", "TED"]

    async def test_send_frame_defaults_empty_prompt_and_hot_words(
        self,
        stream_handle: StreamHandle,
        mock_call: AsyncMock,
    ) -> None:
        """send_frame without prompt and hot_words should use empty defaults."""
        await stream_handle.send_frame(pcm_data=b"\x00" * 320)

        frame: AudioFrame = mock_call.write.call_args[0][0]
        assert frame.initial_prompt == ""
        assert list(frame.hot_words) == []

    async def test_send_frame_on_closed_stream_raises_worker_crash(
        self,
        stream_handle: StreamHandle,
    ) -> None:
        """send_frame on an already closed stream should raise WorkerCrashError."""
        await stream_handle.close()
        assert stream_handle.is_closed is True

        with pytest.raises(WorkerCrashError):
            await stream_handle.send_frame(pcm_data=b"\x00" * 320)

    async def test_send_frame_grpc_error_raises_worker_crash_and_marks_closed(
        self,
        stream_handle: StreamHandle,
        mock_call: AsyncMock,
    ) -> None:
        """gRPC error during write should raise WorkerCrashError and mark closed."""
        mock_call.write.side_effect = grpc.aio.AioRpcError(
            code=grpc.StatusCode.UNAVAILABLE,
            initial_metadata=grpc.aio.Metadata(),
            trailing_metadata=grpc.aio.Metadata(),
            details="Connection refused",
            debug_error_string=None,
        )

        with pytest.raises(WorkerCrashError):
            await stream_handle.send_frame(pcm_data=b"\x00" * 320)

        assert stream_handle.is_closed is True


# ---------------------------------------------------------------------------
# StreamHandle — receive_events
# ---------------------------------------------------------------------------


class TestStreamHandleReceiveEvents:
    """Tests for StreamHandle.receive_events()."""

    async def test_receive_events_converts_proto_to_transcript_segment(
        self,
        stream_handle: StreamHandle,
        mock_call: AsyncMock,
    ) -> None:
        """receive_events should convert TranscriptEvent proto to TranscriptSegment."""
        event = TranscriptEvent(
            session_id="sess_test_001",
            event_type="partial",
            text="ola como",
            segment_id=0,
            start_ms=1500,
            end_ms=2000,
            language="pt",
            confidence=0.85,
        )
        _make_async_iterable_call(mock_call, [event])

        segments: list[TranscriptSegment] = []
        async for seg in stream_handle.receive_events():
            segments.append(seg)

        assert len(segments) == 1
        assert segments[0].text == "ola como"
        assert segments[0].is_final is False
        assert segments[0].segment_id == 0
        assert segments[0].start_ms == 1500
        assert segments[0].end_ms == 2000
        assert segments[0].language == "pt"
        assert segments[0].confidence == pytest.approx(0.85)

    async def test_receive_events_final_event(
        self,
        stream_handle: StreamHandle,
        mock_call: AsyncMock,
    ) -> None:
        """receive_events with event_type 'final' yields TranscriptSegment is_final=True."""
        event = TranscriptEvent(
            session_id="sess_test_001",
            event_type="final",
            text="ola como posso ajudar",
            segment_id=0,
            start_ms=1500,
            end_ms=4000,
            language="pt",
            confidence=0.95,
            words=[
                Word(word="ola", start=1.5, end=2.0, probability=0.99),
                Word(word="como", start=2.1, end=2.4, probability=0.97),
            ],
        )
        _make_async_iterable_call(mock_call, [event])

        segments: list[TranscriptSegment] = []
        async for seg in stream_handle.receive_events():
            segments.append(seg)

        assert len(segments) == 1
        seg = segments[0]
        assert seg.is_final is True
        assert seg.text == "ola como posso ajudar"
        assert seg.words is not None
        assert len(seg.words) == 2
        assert seg.words[0].word == "ola"
        assert seg.words[0].probability == pytest.approx(0.99)
        assert seg.words[1].word == "como"

    async def test_receive_events_multiple_events(
        self,
        stream_handle: StreamHandle,
        mock_call: AsyncMock,
    ) -> None:
        """receive_events should iterate over multiple events."""
        events = [
            TranscriptEvent(event_type="partial", text="ola", segment_id=0),
            TranscriptEvent(event_type="partial", text="ola como", segment_id=0),
            TranscriptEvent(event_type="final", text="ola como vai", segment_id=0),
        ]
        _make_async_iterable_call(mock_call, events)

        segments: list[TranscriptSegment] = []
        async for seg in stream_handle.receive_events():
            segments.append(seg)

        assert len(segments) == 3
        assert segments[0].is_final is False
        assert segments[1].is_final is False
        assert segments[2].is_final is True

    async def test_receive_events_grpc_error_raises_worker_crash(
        self,
        stream_handle: StreamHandle,
        mock_call: AsyncMock,
    ) -> None:
        """gRPC error during iteration should raise WorkerCrashError."""

        # Simulate async generator that raises error before yield
        async def _error_iter() -> AsyncIterator[TranscriptEvent]:
            raise grpc.aio.AioRpcError(
                code=grpc.StatusCode.UNAVAILABLE,
                initial_metadata=grpc.aio.Metadata(),
                trailing_metadata=grpc.aio.Metadata(),
                details="Worker crashed",
                debug_error_string=None,
            )
            yield TranscriptEvent()  # pragma: no cover — unreachable, needed for generator type

        mock_call.__aiter__ = MagicMock(return_value=_error_iter())

        with pytest.raises(WorkerCrashError):
            async for _ in stream_handle.receive_events():
                pass  # pragma: no cover

        assert stream_handle.is_closed is True

    async def test_receive_events_deadline_exceeded_raises_timeout(
        self,
        stream_handle: StreamHandle,
        mock_call: AsyncMock,
    ) -> None:
        """DEADLINE_EXCEEDED during iteration should raise WorkerTimeoutError."""

        # Simulate async generator that raises DEADLINE_EXCEEDED before yield
        async def _timeout_iter() -> AsyncIterator[TranscriptEvent]:
            raise grpc.aio.AioRpcError(
                code=grpc.StatusCode.DEADLINE_EXCEEDED,
                initial_metadata=grpc.aio.Metadata(),
                trailing_metadata=grpc.aio.Metadata(),
                details="Deadline exceeded",
                debug_error_string=None,
            )
            yield TranscriptEvent()  # pragma: no cover — unreachable, needed for generator type

        mock_call.__aiter__ = MagicMock(return_value=_timeout_iter())

        with pytest.raises(WorkerTimeoutError):
            async for _ in stream_handle.receive_events():
                pass  # pragma: no cover

        assert stream_handle.is_closed is True

    async def test_receive_events_empty_stream(
        self,
        stream_handle: StreamHandle,
        mock_call: AsyncMock,
    ) -> None:
        """Empty stream should complete without events."""
        _make_async_iterable_call(mock_call, [])

        segments: list[TranscriptSegment] = []
        async for seg in stream_handle.receive_events():
            segments.append(seg)  # pragma: no cover

        assert len(segments) == 0


# ---------------------------------------------------------------------------
# StreamHandle — close
# ---------------------------------------------------------------------------


class TestStreamHandleClose:
    """Tests for StreamHandle.close()."""

    async def test_close_sends_is_last_and_done_writing(
        self,
        stream_handle: StreamHandle,
        mock_call: AsyncMock,
    ) -> None:
        """close() should send frame with is_last=True and call done_writing."""
        await stream_handle.close()

        assert stream_handle.is_closed is True

        # Verify that write was called with is_last=True
        mock_call.write.assert_called_once()
        frame: AudioFrame = mock_call.write.call_args[0][0]
        assert frame.is_last is True
        assert frame.data == b""
        assert frame.session_id == "sess_test_001"

        mock_call.done_writing.assert_called_once()

    async def test_close_idempotent_on_already_closed(
        self,
        stream_handle: StreamHandle,
        mock_call: AsyncMock,
    ) -> None:
        """close() on an already closed stream should be a no-op."""
        await stream_handle.close()
        mock_call.write.reset_mock()
        mock_call.done_writing.reset_mock()

        # Second call should do nothing
        await stream_handle.close()

        mock_call.write.assert_not_called()
        mock_call.done_writing.assert_not_called()

    async def test_close_handles_grpc_error_gracefully(
        self,
        stream_handle: StreamHandle,
        mock_call: AsyncMock,
    ) -> None:
        """close() should handle gRPC error without propagating exception."""
        mock_call.write.side_effect = grpc.aio.AioRpcError(
            code=grpc.StatusCode.CANCELLED,
            initial_metadata=grpc.aio.Metadata(),
            trailing_metadata=grpc.aio.Metadata(),
            details="Already cancelled",
            debug_error_string=None,
        )

        # Should not raise an exception
        await stream_handle.close()

        assert stream_handle.is_closed is True


# ---------------------------------------------------------------------------
# StreamHandle — cancel
# ---------------------------------------------------------------------------


class TestStreamHandleCancel:
    """Tests for StreamHandle.cancel()."""

    async def test_cancel_calls_cancel_and_marks_closed(
        self,
        stream_handle: StreamHandle,
        mock_call: AsyncMock,
    ) -> None:
        """cancel() should call call.cancel() and mark closed."""
        await stream_handle.cancel()

        mock_call.cancel.assert_called_once()
        assert stream_handle.is_closed is True


# ---------------------------------------------------------------------------
# StreamHandle — properties
# ---------------------------------------------------------------------------


class TestStreamHandleProperties:
    """Tests for StreamHandle properties."""

    async def test_session_id_returns_correct_value(
        self,
        stream_handle: StreamHandle,
    ) -> None:
        """session_id should return the session ID."""
        assert stream_handle.session_id == "sess_test_001"

    async def test_is_closed_initially_false(
        self,
        stream_handle: StreamHandle,
    ) -> None:
        """is_closed should be False initially."""
        assert stream_handle.is_closed is False


# ---------------------------------------------------------------------------
# StreamingGRPCClient
# ---------------------------------------------------------------------------


class TestStreamingGRPCClient:
    """Tests for StreamingGRPCClient."""

    async def test_open_stream_without_connect_raises_worker_crash(self) -> None:
        """open_stream without connect() should raise WorkerCrashError."""
        client = StreamingGRPCClient("localhost:50051")

        with pytest.raises(WorkerCrashError):
            await client.open_stream("sess_test_001")

    async def test_connect_creates_channel_and_stub(self) -> None:
        """connect() should create gRPC channel and stub."""
        client = StreamingGRPCClient("localhost:50051")

        with patch("macaw.scheduler.streaming.grpc.aio.insecure_channel") as mock_channel_fn:
            mock_channel = MagicMock()
            mock_channel_fn.return_value = mock_channel

            await client.connect()

            mock_channel_fn.assert_called_once()
            assert client._channel is not None
            assert client._stub is not None

        # Cleanup
        client._channel = None
        client._stub = None

    async def test_open_stream_returns_stream_handle(self) -> None:
        """open_stream after connect should return StreamHandle."""
        client = StreamingGRPCClient("localhost:50051")

        mock_call = AsyncMock()
        mock_stub = MagicMock()
        mock_stub.TranscribeStream.return_value = mock_call

        # Inject stub directly
        client._stub = mock_stub
        client._channel = MagicMock()

        handle = await client.open_stream("sess_test_002")

        assert isinstance(handle, StreamHandle)
        assert handle.session_id == "sess_test_002"
        assert handle.is_closed is False

        # Cleanup
        client._channel = None
        client._stub = None

    async def test_close_closes_channel(self) -> None:
        """close() should close the gRPC channel and clear references."""
        client = StreamingGRPCClient("localhost:50051")

        mock_channel = AsyncMock()
        client._channel = mock_channel
        client._stub = MagicMock()

        await client.close()

        mock_channel.close.assert_called_once()
        assert client._channel is None
        assert client._stub is None

    async def test_close_idempotent_when_not_connected(self) -> None:
        """close() without connection should be a no-op without error."""
        client = StreamingGRPCClient("localhost:50051")
        await client.close()  # Should not raise an exception


# ---------------------------------------------------------------------------
# _proto_event_to_transcript_segment (pure function)
# ---------------------------------------------------------------------------


class TestProtoEventToTranscriptSegment:
    """Tests for proto -> domain conversion function."""

    def test_converts_final_event(self) -> None:
        """'final' event should produce TranscriptSegment with is_final=True."""
        event = TranscriptEvent(
            session_id="sess_001",
            event_type="final",
            text="ola mundo",
            segment_id=3,
            start_ms=1000,
            end_ms=2500,
            language="pt",
            confidence=0.92,
        )

        result = _proto_event_to_transcript_segment(event)

        assert result.text == "ola mundo"
        assert result.is_final is True
        assert result.segment_id == 3
        assert result.start_ms == 1000
        assert result.end_ms == 2500
        assert result.language == "pt"
        assert result.confidence == pytest.approx(0.92)
        assert result.words is None

    def test_converts_partial_event(self) -> None:
        """'partial' event should produce TranscriptSegment with is_final=False."""
        event = TranscriptEvent(
            session_id="sess_001",
            event_type="partial",
            text="ola",
            segment_id=0,
        )

        result = _proto_event_to_transcript_segment(event)

        assert result.text == "ola"
        assert result.is_final is False
        assert result.segment_id == 0
        assert result.start_ms == 0  # Proto default 0 preserved (valid timestamp)
        assert result.end_ms == 0
        assert result.language is None  # Proto default "" -> None
        assert result.confidence is None  # Proto default 0.0 -> None

    def test_converts_event_with_words(self) -> None:
        """Event with words should convert to tuple of WordTimestamp."""
        event = TranscriptEvent(
            session_id="sess_001",
            event_type="final",
            text="PIX transferencia",
            segment_id=1,
            start_ms=500,
            end_ms=2000,
            words=[
                Word(word="PIX", start=0.5, end=0.8, probability=0.99),
                Word(word="transferencia", start=0.9, end=2.0, probability=0.85),
            ],
        )

        result = _proto_event_to_transcript_segment(event)

        assert result.words is not None
        assert len(result.words) == 2

        assert result.words[0].word == "PIX"
        assert result.words[0].start == pytest.approx(0.5)
        assert result.words[0].end == pytest.approx(0.8)
        assert result.words[0].probability == pytest.approx(0.99)

        assert result.words[1].word == "transferencia"
        assert result.words[1].start == pytest.approx(0.9)
        assert result.words[1].end == pytest.approx(2.0)
        assert result.words[1].probability == pytest.approx(0.85)

    def test_word_probability_zero_becomes_none(self) -> None:
        """Probability 0.0 from proto (default) should become None in domain."""
        event = TranscriptEvent(
            session_id="sess_001",
            event_type="final",
            text="teste",
            segment_id=0,
            words=[
                Word(word="teste", start=0.0, end=1.0, probability=0.0),
            ],
        )

        result = _proto_event_to_transcript_segment(event)

        assert result.words is not None
        assert result.words[0].probability is None

    def test_confidence_zero_becomes_none(self) -> None:
        """Confidence 0.0 from proto (default) should become None in domain."""
        event = TranscriptEvent(
            session_id="sess_001",
            event_type="final",
            text="teste",
            segment_id=0,
            confidence=0.0,
        )

        result = _proto_event_to_transcript_segment(event)

        assert result.confidence is None

    def test_empty_language_becomes_none(self) -> None:
        """Empty language from proto (default) should become None in domain."""
        event = TranscriptEvent(
            session_id="sess_001",
            event_type="partial",
            text="teste",
            segment_id=0,
            language="",
        )

        result = _proto_event_to_transcript_segment(event)

        assert result.language is None

    def test_result_is_immutable(self) -> None:
        """Returned TranscriptSegment should be frozen (immutable)."""
        event = TranscriptEvent(
            event_type="final",
            text="teste",
            segment_id=0,
        )

        result = _proto_event_to_transcript_segment(event)

        with pytest.raises(AttributeError):
            result.text = "modificado"  # type: ignore[misc]
