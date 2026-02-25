"""Tests for CrossSegmentContext.

Validates that the cross-segment context correctly stores the last
N tokens from transcript.final for conditioning the next segment.

Tests are deterministic, with no external dependencies.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

from macaw._types import TranscriptSegment
from macaw.session.cross_segment import CrossSegmentContext
from macaw.session.streaming import StreamingSession
from macaw.vad.detector import VADEvent, VADEventType
from tests.helpers import AsyncIterFromList, make_float32_frame, make_raw_bytes

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


# ---------------------------------------------------------------------------
# CrossSegmentContext unit tests
# ---------------------------------------------------------------------------


class TestCrossSegmentContext:
    """Unit tests for CrossSegmentContext in isolation."""

    def test_initial_state_empty(self) -> None:
        """get_prompt() returns None when no context has been registered."""
        # Arrange
        ctx = CrossSegmentContext()

        # Act & Assert
        assert ctx.get_prompt() is None

    def test_update_stores_text(self) -> None:
        """After update(), get_prompt() returns the stored text."""
        # Arrange
        ctx = CrossSegmentContext()

        # Act
        ctx.update("hello world")

        # Assert
        assert ctx.get_prompt() == "hello world"

    def test_update_truncates_to_max_tokens(self) -> None:
        """Text with more words than max_tokens is truncated from the beginning."""
        # Arrange
        ctx = CrossSegmentContext(max_tokens=3)
        text = "one two three four five"

        # Act
        ctx.update(text)

        # Assert: kept the last 3 words
        assert ctx.get_prompt() == "three four five"

    def test_update_overwrites_previous(self) -> None:
        """Second call to update() replaces the previous context."""
        # Arrange
        ctx = CrossSegmentContext()
        ctx.update("first text")

        # Act
        ctx.update("second text")

        # Assert
        assert ctx.get_prompt() == "second text"

    def test_reset_clears_context(self) -> None:
        """After reset(), get_prompt() returns None."""
        # Arrange
        ctx = CrossSegmentContext()
        ctx.update("some text")

        # Act
        ctx.reset()

        # Assert
        assert ctx.get_prompt() is None

    def test_max_tokens_custom(self) -> None:
        """Custom max_tokens truncates correctly."""
        # Arrange
        ctx = CrossSegmentContext(max_tokens=5)
        text = "a b c d e f g h i j"

        # Act
        ctx.update(text)

        # Assert: last 5 words
        assert ctx.get_prompt() == "f g h i j"

    def test_empty_text_update(self) -> None:
        """update('') results in get_prompt() returning None."""
        # Arrange
        ctx = CrossSegmentContext()
        ctx.update("previous context")

        # Act
        ctx.update("")

        # Assert
        assert ctx.get_prompt() is None

    def test_whitespace_only_text_update(self) -> None:
        """update with whitespace only results in None."""
        # Arrange
        ctx = CrossSegmentContext()
        ctx.update("previous context")

        # Act
        ctx.update("   ")

        # Assert
        assert ctx.get_prompt() is None

    def test_exact_max_tokens_no_truncation(self) -> None:
        """Text with exactly max_tokens words is not truncated."""
        # Arrange
        ctx = CrossSegmentContext(max_tokens=4)
        text = "one two three four"

        # Act
        ctx.update(text)

        # Assert
        assert ctx.get_prompt() == "one two three four"

    def test_single_word_text(self) -> None:
        """Text with a single word is stored correctly."""
        # Arrange
        ctx = CrossSegmentContext()

        # Act
        ctx.update("hello")

        # Assert
        assert ctx.get_prompt() == "hello"

    def test_default_max_tokens_is_224(self) -> None:
        """Default max_tokens is 224 (half of Whisper's context window)."""
        # Arrange
        ctx = CrossSegmentContext()

        # Act: text with 250 words
        words = [f"word{i}" for i in range(250)]
        ctx.update(" ".join(words))

        # Assert: kept last 224 words
        result = ctx.get_prompt()
        assert result is not None
        assert len(result.split()) == 224
        assert result.endswith("word249")

    def test_default_uses_named_constant(self) -> None:
        """Default max_tokens matches _DEFAULT_CROSS_SEGMENT_MAX_TOKENS constant."""
        from macaw.session.cross_segment import _DEFAULT_CROSS_SEGMENT_MAX_TOKENS

        ctx = CrossSegmentContext()
        words = [f"w{i}" for i in range(_DEFAULT_CROSS_SEGMENT_MAX_TOKENS + 10)]
        ctx.update(" ".join(words))
        result = ctx.get_prompt()
        assert result is not None
        assert len(result.split()) == _DEFAULT_CROSS_SEGMENT_MAX_TOKENS


# ---------------------------------------------------------------------------
# Integration with StreamingSession
# ---------------------------------------------------------------------------


class TestStreamingSessionCrossSegment:
    """Integration tests for CrossSegmentContext with StreamingSession."""

    async def test_streaming_session_updates_context(self) -> None:
        """StreamingSession updates cross-segment context after transcript.final."""
        # Arrange
        final_segment = TranscriptSegment(
            text="ola como posso ajudar",
            is_final=True,
            segment_id=0,
            start_ms=1000,
            end_ms=2000,
            language="pt",
            confidence=0.95,
        )

        stream_handle = _make_stream_handle_mock(events=[final_segment])
        grpc_client = AsyncMock()
        grpc_client.open_stream = AsyncMock(return_value=stream_handle)

        vad = Mock()
        vad.process_frame.return_value = None
        vad.is_speaking = False
        vad.reset.return_value = None

        preprocessor = Mock()
        preprocessor.process_frame.return_value = make_float32_frame()

        cross_ctx = CrossSegmentContext(max_tokens=224)
        on_event = AsyncMock()

        session = StreamingSession(
            session_id="test_session",
            preprocessor=preprocessor,
            vad=vad,
            grpc_client=grpc_client,
            postprocessor=None,
            on_event=on_event,
            enable_itn=False,
            cross_segment_context=cross_ctx,
        )

        # Act: trigger speech_start -> receiver processes final
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START,
            timestamp_ms=1000,
        )
        await session.process_frame(make_raw_bytes())

        # Give time for receiver task to process the transcript.final
        await asyncio.sleep(0.05)

        # Assert: cross-segment context was updated with the final text
        assert cross_ctx.get_prompt() == "ola como posso ajudar"

        # Cleanup
        await session.close()

    async def test_streaming_session_sends_initial_prompt_with_context(self) -> None:
        """StreamingSession sends initial_prompt with cross-segment context."""
        # Arrange: first segment emits transcript.final
        final_segment = TranscriptSegment(
            text="primeiro segmento",
            is_final=True,
            segment_id=0,
            start_ms=0,
            end_ms=1000,
            language="pt",
            confidence=0.9,
        )

        stream_handle1 = _make_stream_handle_mock(events=[final_segment])
        stream_handle2 = _make_stream_handle_mock()

        grpc_client = AsyncMock()
        grpc_client.open_stream = AsyncMock(side_effect=[stream_handle1, stream_handle2])

        vad = Mock()
        vad.process_frame.return_value = None
        vad.is_speaking = False
        vad.reset.return_value = None

        preprocessor = Mock()
        preprocessor.process_frame.return_value = make_float32_frame()

        cross_ctx = CrossSegmentContext(max_tokens=224)
        on_event = AsyncMock()

        session = StreamingSession(
            session_id="test_session",
            preprocessor=preprocessor,
            vad=vad,
            grpc_client=grpc_client,
            postprocessor=None,
            on_event=on_event,
            enable_itn=False,
            cross_segment_context=cross_ctx,
        )

        # First segment: speech_start -> speech_end
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START,
            timestamp_ms=0,
        )
        await session.process_frame(make_raw_bytes())
        await asyncio.sleep(0.05)

        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_END,
            timestamp_ms=1000,
        )
        vad.is_speaking = False
        await session.process_frame(make_raw_bytes())

        # Context now has "primeiro segmento"
        assert cross_ctx.get_prompt() == "primeiro segmento"

        # Second segment: speech_start -> send frame
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START,
            timestamp_ms=2000,
        )
        vad.is_speaking = True
        await session.process_frame(make_raw_bytes())

        # Assert: first frame of second segment should have initial_prompt
        calls = stream_handle2.send_frame.call_args_list
        assert len(calls) >= 1

        first_call = calls[0]
        assert first_call.kwargs.get("initial_prompt") == "primeiro segmento"

        # Cleanup
        await session.close()

    async def test_streaming_session_initial_prompt_combines_hot_words_and_context(
        self,
    ) -> None:
        """initial_prompt combines hot words and cross-segment context."""
        # Arrange: pre-seed context
        cross_ctx = CrossSegmentContext(max_tokens=224)
        cross_ctx.update("contexto anterior")

        stream_handle = _make_stream_handle_mock()
        grpc_client = AsyncMock()
        grpc_client.open_stream = AsyncMock(return_value=stream_handle)

        vad = Mock()
        vad.process_frame.return_value = None
        vad.is_speaking = False
        vad.reset.return_value = None

        preprocessor = Mock()
        preprocessor.process_frame.return_value = make_float32_frame()

        session = StreamingSession(
            session_id="test_session",
            preprocessor=preprocessor,
            vad=vad,
            grpc_client=grpc_client,
            postprocessor=None,
            on_event=AsyncMock(),
            hot_words=["PIX", "TED"],
            enable_itn=False,
            cross_segment_context=cross_ctx,
        )

        # Act: speech_start -> send frame (first frame of the segment)
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START,
            timestamp_ms=0,
        )
        vad.is_speaking = True
        await session.process_frame(make_raw_bytes())

        # Assert: initial_prompt combines hot words and context
        calls = stream_handle.send_frame.call_args_list
        assert len(calls) >= 1

        first_call = calls[0]
        prompt = first_call.kwargs.get("initial_prompt")
        assert prompt == "Terms: PIX, TED. contexto anterior"

        # hot_words also sent
        assert first_call.kwargs.get("hot_words") == ["PIX", "TED"]

        # Cleanup
        await session.close()

    async def test_streaming_session_initial_prompt_hot_words_only(self) -> None:
        """initial_prompt with hot words but without cross-segment context."""
        # Arrange: no cross-segment context
        stream_handle = _make_stream_handle_mock()
        grpc_client = AsyncMock()
        grpc_client.open_stream = AsyncMock(return_value=stream_handle)

        vad = Mock()
        vad.process_frame.return_value = None
        vad.is_speaking = False
        vad.reset.return_value = None

        preprocessor = Mock()
        preprocessor.process_frame.return_value = make_float32_frame()

        session = StreamingSession(
            session_id="test_session",
            preprocessor=preprocessor,
            vad=vad,
            grpc_client=grpc_client,
            postprocessor=None,
            on_event=AsyncMock(),
            hot_words=["PIX", "Selic"],
            enable_itn=False,
        )

        # Act
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START,
            timestamp_ms=0,
        )
        vad.is_speaking = True
        await session.process_frame(make_raw_bytes())

        # Assert: initial_prompt has only hot words
        calls = stream_handle.send_frame.call_args_list
        first_call = calls[0]
        prompt = first_call.kwargs.get("initial_prompt")
        assert prompt == "Terms: PIX, Selic."

        # Cleanup
        await session.close()

    async def test_streaming_session_no_context_no_hot_words_no_prompt(self) -> None:
        """Without cross-segment context and without hot words, initial_prompt is None."""
        # Arrange
        stream_handle = _make_stream_handle_mock()
        grpc_client = AsyncMock()
        grpc_client.open_stream = AsyncMock(return_value=stream_handle)

        vad = Mock()
        vad.process_frame.return_value = None
        vad.is_speaking = False
        vad.reset.return_value = None

        preprocessor = Mock()
        preprocessor.process_frame.return_value = make_float32_frame()

        session = StreamingSession(
            session_id="test_session",
            preprocessor=preprocessor,
            vad=vad,
            grpc_client=grpc_client,
            postprocessor=None,
            on_event=AsyncMock(),
            enable_itn=False,
        )

        # Act
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START,
            timestamp_ms=0,
        )
        vad.is_speaking = True
        await session.process_frame(make_raw_bytes())

        # Assert: initial_prompt is None
        calls = stream_handle.send_frame.call_args_list
        first_call = calls[0]
        assert first_call.kwargs.get("initial_prompt") is None

        # Cleanup
        await session.close()

    async def test_streaming_session_context_only_no_hot_words(self) -> None:
        """Cross-segment context without hot words -> initial_prompt is only the context."""
        # Arrange
        cross_ctx = CrossSegmentContext()
        cross_ctx.update("contexto do segmento anterior")

        stream_handle = _make_stream_handle_mock()
        grpc_client = AsyncMock()
        grpc_client.open_stream = AsyncMock(return_value=stream_handle)

        vad = Mock()
        vad.process_frame.return_value = None
        vad.is_speaking = False
        vad.reset.return_value = None

        preprocessor = Mock()
        preprocessor.process_frame.return_value = make_float32_frame()

        session = StreamingSession(
            session_id="test_session",
            preprocessor=preprocessor,
            vad=vad,
            grpc_client=grpc_client,
            postprocessor=None,
            on_event=AsyncMock(),
            enable_itn=False,
            cross_segment_context=cross_ctx,
        )

        # Act
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START,
            timestamp_ms=0,
        )
        vad.is_speaking = True
        await session.process_frame(make_raw_bytes())

        # Assert
        calls = stream_handle.send_frame.call_args_list
        first_call = calls[0]
        assert first_call.kwargs.get("initial_prompt") == "contexto do segmento anterior"
        assert first_call.kwargs.get("hot_words") is None

        # Cleanup
        await session.close()

    async def test_streaming_session_context_uses_postprocessed_text(self) -> None:
        """Cross-segment context stores post-processed text (with ITN)."""
        # Arrange
        final_segment = TranscriptSegment(
            text="dois mil e vinte e cinco",
            is_final=True,
            segment_id=0,
            start_ms=0,
            end_ms=1000,
            language="pt",
            confidence=0.9,
        )

        stream_handle = _make_stream_handle_mock(events=[final_segment])
        grpc_client = AsyncMock()
        grpc_client.open_stream = AsyncMock(return_value=stream_handle)

        vad = Mock()
        vad.process_frame.return_value = None
        vad.is_speaking = False
        vad.reset.return_value = None

        preprocessor = Mock()
        preprocessor.process_frame.return_value = make_float32_frame()

        postprocessor = Mock()
        postprocessor.process.side_effect = lambda text, **kwargs: "2025"

        cross_ctx = CrossSegmentContext()
        on_event = AsyncMock()

        session = StreamingSession(
            session_id="test_session",
            preprocessor=preprocessor,
            vad=vad,
            grpc_client=grpc_client,
            postprocessor=postprocessor,
            on_event=on_event,
            enable_itn=True,
            cross_segment_context=cross_ctx,
        )

        # Act: trigger speech_start
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START,
            timestamp_ms=0,
        )
        await session.process_frame(make_raw_bytes())
        await asyncio.sleep(0.05)

        # Assert: context stores post-processed text (ITN applied)
        assert cross_ctx.get_prompt() == "2025"

        # Cleanup
        await session.close()

    async def test_prompt_not_sent_on_subsequent_frames(self) -> None:
        """initial_prompt is only sent on the first frame of the segment."""
        # Arrange
        cross_ctx = CrossSegmentContext()
        cross_ctx.update("contexto")

        stream_handle = _make_stream_handle_mock()
        grpc_client = AsyncMock()
        grpc_client.open_stream = AsyncMock(return_value=stream_handle)

        vad = Mock()
        vad.process_frame.return_value = None
        vad.is_speaking = False
        vad.reset.return_value = None

        preprocessor = Mock()
        preprocessor.process_frame.return_value = make_float32_frame()

        session = StreamingSession(
            session_id="test_session",
            preprocessor=preprocessor,
            vad=vad,
            grpc_client=grpc_client,
            postprocessor=None,
            on_event=AsyncMock(),
            enable_itn=False,
            cross_segment_context=cross_ctx,
        )

        # Act: speech_start + 3 frames
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START,
            timestamp_ms=0,
        )
        vad.is_speaking = True
        await session.process_frame(make_raw_bytes())

        vad.process_frame.return_value = None
        await session.process_frame(make_raw_bytes())
        await session.process_frame(make_raw_bytes())

        # Assert: first frame has prompt, subsequent frames do not
        calls = stream_handle.send_frame.call_args_list
        assert len(calls) == 3

        assert calls[0].kwargs.get("initial_prompt") == "contexto"
        assert calls[1].kwargs.get("initial_prompt") is None
        assert calls[2].kwargs.get("initial_prompt") is None

        # Cleanup
        await session.close()
