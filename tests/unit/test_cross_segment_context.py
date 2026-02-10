"""Testes do CrossSegmentContext.

Valida que o contexto cross-segment armazena corretamente os ultimos
N tokens do transcript.final para conditioning do proximo segmento.

Testes sao deterministicos, sem dependencias externas.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

import numpy as np

from macaw._types import TranscriptSegment
from macaw.session.cross_segment import CrossSegmentContext
from macaw.session.streaming import StreamingSession
from macaw.vad.detector import VADEvent, VADEventType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FRAME_SIZE = 1024


def _make_raw_bytes(n_samples: int = _FRAME_SIZE) -> bytes:
    """Gera bytes PCM int16 (zeros) com n_samples amostras."""
    return np.zeros(n_samples, dtype=np.int16).tobytes()


def _make_float32_frame(n_samples: int = _FRAME_SIZE) -> np.ndarray:
    """Gera frame float32 (zeros) para mock de preprocessor."""
    return np.zeros(n_samples, dtype=np.float32)


class _AsyncIterFromList:
    """Async iterator que yield items de uma lista."""

    def __init__(self, items: list) -> None:
        self._items = list(items)
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        if isinstance(item, Exception):
            raise item
        return item


def _make_stream_handle_mock(events: list | None = None) -> Mock:
    """Cria mock de StreamHandle."""
    handle = Mock()
    handle.is_closed = False
    handle.session_id = "test_session"
    if events is None:
        events = []
    handle.receive_events.return_value = _AsyncIterFromList(events)
    handle.send_frame = AsyncMock()
    handle.close = AsyncMock()
    handle.cancel = AsyncMock()
    return handle


# ---------------------------------------------------------------------------
# CrossSegmentContext unit tests
# ---------------------------------------------------------------------------


class TestCrossSegmentContext:
    """Testes unitarios do CrossSegmentContext isolado."""

    def test_initial_state_empty(self) -> None:
        """get_prompt() retorna None quando nenhum contexto foi registrado."""
        # Arrange
        ctx = CrossSegmentContext()

        # Act & Assert
        assert ctx.get_prompt() is None

    def test_update_stores_text(self) -> None:
        """Apos update(), get_prompt() retorna o texto armazenado."""
        # Arrange
        ctx = CrossSegmentContext()

        # Act
        ctx.update("hello world")

        # Assert
        assert ctx.get_prompt() == "hello world"

    def test_update_truncates_to_max_tokens(self) -> None:
        """Texto com mais palavras que max_tokens e truncado do inicio."""
        # Arrange
        ctx = CrossSegmentContext(max_tokens=3)
        text = "one two three four five"

        # Act
        ctx.update(text)

        # Assert: manteve as ultimas 3 palavras
        assert ctx.get_prompt() == "three four five"

    def test_update_overwrites_previous(self) -> None:
        """Segunda chamada a update() substitui o contexto anterior."""
        # Arrange
        ctx = CrossSegmentContext()
        ctx.update("first text")

        # Act
        ctx.update("second text")

        # Assert
        assert ctx.get_prompt() == "second text"

    def test_reset_clears_context(self) -> None:
        """Apos reset(), get_prompt() retorna None."""
        # Arrange
        ctx = CrossSegmentContext()
        ctx.update("some text")

        # Act
        ctx.reset()

        # Assert
        assert ctx.get_prompt() is None

    def test_context_text_property(self) -> None:
        """Property context_text retorna o mesmo que get_prompt()."""
        # Arrange
        ctx = CrossSegmentContext()

        # Assert: vazio
        assert ctx.context_text is None

        # Act
        ctx.update("hello")

        # Assert: com valor
        assert ctx.context_text == "hello"
        assert ctx.context_text == ctx.get_prompt()

    def test_max_tokens_custom(self) -> None:
        """max_tokens customizado trunca corretamente."""
        # Arrange
        ctx = CrossSegmentContext(max_tokens=5)
        text = "a b c d e f g h i j"

        # Act
        ctx.update(text)

        # Assert: ultimas 5 palavras
        assert ctx.get_prompt() == "f g h i j"

    def test_empty_text_update(self) -> None:
        """update('') resulta em get_prompt() retornando None."""
        # Arrange
        ctx = CrossSegmentContext()
        ctx.update("previous context")

        # Act
        ctx.update("")

        # Assert
        assert ctx.get_prompt() is None

    def test_whitespace_only_text_update(self) -> None:
        """update com apenas espacos resulta em None."""
        # Arrange
        ctx = CrossSegmentContext()
        ctx.update("previous context")

        # Act
        ctx.update("   ")

        # Assert
        assert ctx.get_prompt() is None

    def test_exact_max_tokens_no_truncation(self) -> None:
        """Texto com exatamente max_tokens palavras nao e truncado."""
        # Arrange
        ctx = CrossSegmentContext(max_tokens=4)
        text = "one two three four"

        # Act
        ctx.update(text)

        # Assert
        assert ctx.get_prompt() == "one two three four"

    def test_single_word_text(self) -> None:
        """Texto com uma unica palavra e armazenado corretamente."""
        # Arrange
        ctx = CrossSegmentContext()

        # Act
        ctx.update("hello")

        # Assert
        assert ctx.get_prompt() == "hello"

    def test_default_max_tokens_is_224(self) -> None:
        """Valor default de max_tokens e 224 (metade do context window do Whisper)."""
        # Arrange
        ctx = CrossSegmentContext()

        # Act: texto com 250 palavras
        words = [f"word{i}" for i in range(250)]
        ctx.update(" ".join(words))

        # Assert: manteve ultimas 224 palavras
        result = ctx.get_prompt()
        assert result is not None
        assert len(result.split()) == 224
        assert result.endswith("word249")


# ---------------------------------------------------------------------------
# Integration with StreamingSession
# ---------------------------------------------------------------------------


class TestStreamingSessionCrossSegment:
    """Testes de integracao do CrossSegmentContext com StreamingSession."""

    async def test_streaming_session_updates_context(self) -> None:
        """StreamingSession atualiza cross-segment context apos transcript.final."""
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
        preprocessor.process_frame.return_value = _make_float32_frame()

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

        # Act: trigger speech_start -> receiver processa final
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START,
            timestamp_ms=1000,
        )
        await session.process_frame(_make_raw_bytes())

        # Dar tempo para receiver task processar o transcript.final
        await asyncio.sleep(0.05)

        # Assert: cross-segment context foi atualizado com texto do final
        assert cross_ctx.get_prompt() == "ola como posso ajudar"

        # Cleanup
        await session.close()

    async def test_streaming_session_sends_initial_prompt_with_context(self) -> None:
        """StreamingSession envia initial_prompt com cross-segment context."""
        # Arrange: primeiro segmento emite transcript.final
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
        preprocessor.process_frame.return_value = _make_float32_frame()

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

        # Primeiro segmento: speech_start -> speech_end
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START,
            timestamp_ms=0,
        )
        await session.process_frame(_make_raw_bytes())
        await asyncio.sleep(0.05)

        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_END,
            timestamp_ms=1000,
        )
        vad.is_speaking = False
        await session.process_frame(_make_raw_bytes())

        # Contexto agora tem "primeiro segmento"
        assert cross_ctx.get_prompt() == "primeiro segmento"

        # Segundo segmento: speech_start -> enviar frame
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START,
            timestamp_ms=2000,
        )
        vad.is_speaking = True
        await session.process_frame(_make_raw_bytes())

        # Assert: primeiro frame do segundo segmento deve ter initial_prompt
        calls = stream_handle2.send_frame.call_args_list
        assert len(calls) >= 1

        first_call = calls[0]
        assert first_call.kwargs.get("initial_prompt") == "primeiro segmento"

        # Cleanup
        await session.close()

    async def test_streaming_session_initial_prompt_combines_hot_words_and_context(
        self,
    ) -> None:
        """initial_prompt combina hot words e cross-segment context."""
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
        preprocessor.process_frame.return_value = _make_float32_frame()

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

        # Act: speech_start -> envia frame (primeiro frame do segmento)
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START,
            timestamp_ms=0,
        )
        vad.is_speaking = True
        await session.process_frame(_make_raw_bytes())

        # Assert: initial_prompt combina hot words e contexto
        calls = stream_handle.send_frame.call_args_list
        assert len(calls) >= 1

        first_call = calls[0]
        prompt = first_call.kwargs.get("initial_prompt")
        assert prompt == "Termos: PIX, TED. contexto anterior"

        # hot_words tambem enviados
        assert first_call.kwargs.get("hot_words") == ["PIX", "TED"]

        # Cleanup
        await session.close()

    async def test_streaming_session_initial_prompt_hot_words_only(self) -> None:
        """initial_prompt com hot words mas sem cross-segment context."""
        # Arrange: sem cross-segment context
        stream_handle = _make_stream_handle_mock()
        grpc_client = AsyncMock()
        grpc_client.open_stream = AsyncMock(return_value=stream_handle)

        vad = Mock()
        vad.process_frame.return_value = None
        vad.is_speaking = False
        vad.reset.return_value = None

        preprocessor = Mock()
        preprocessor.process_frame.return_value = _make_float32_frame()

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
        await session.process_frame(_make_raw_bytes())

        # Assert: initial_prompt tem apenas hot words
        calls = stream_handle.send_frame.call_args_list
        first_call = calls[0]
        prompt = first_call.kwargs.get("initial_prompt")
        assert prompt == "Termos: PIX, Selic."

        # Cleanup
        await session.close()

    async def test_streaming_session_no_context_no_hot_words_no_prompt(self) -> None:
        """Sem cross-segment context e sem hot words, initial_prompt e None."""
        # Arrange
        stream_handle = _make_stream_handle_mock()
        grpc_client = AsyncMock()
        grpc_client.open_stream = AsyncMock(return_value=stream_handle)

        vad = Mock()
        vad.process_frame.return_value = None
        vad.is_speaking = False
        vad.reset.return_value = None

        preprocessor = Mock()
        preprocessor.process_frame.return_value = _make_float32_frame()

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
        await session.process_frame(_make_raw_bytes())

        # Assert: initial_prompt e None
        calls = stream_handle.send_frame.call_args_list
        first_call = calls[0]
        assert first_call.kwargs.get("initial_prompt") is None

        # Cleanup
        await session.close()

    async def test_streaming_session_context_only_no_hot_words(self) -> None:
        """Cross-segment context sem hot words -> initial_prompt e apenas o contexto."""
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
        preprocessor.process_frame.return_value = _make_float32_frame()

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
        await session.process_frame(_make_raw_bytes())

        # Assert
        calls = stream_handle.send_frame.call_args_list
        first_call = calls[0]
        assert first_call.kwargs.get("initial_prompt") == "contexto do segmento anterior"
        assert first_call.kwargs.get("hot_words") is None

        # Cleanup
        await session.close()

    async def test_streaming_session_context_uses_postprocessed_text(self) -> None:
        """Cross-segment context armazena texto pos-processado (com ITN)."""
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
        preprocessor.process_frame.return_value = _make_float32_frame()

        postprocessor = Mock()
        postprocessor.process.side_effect = lambda text: "2025"

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
        await session.process_frame(_make_raw_bytes())
        await asyncio.sleep(0.05)

        # Assert: contexto armazena texto pos-processado (ITN aplicado)
        assert cross_ctx.get_prompt() == "2025"

        # Cleanup
        await session.close()

    async def test_prompt_not_sent_on_subsequent_frames(self) -> None:
        """initial_prompt so e enviado no primeiro frame do segmento."""
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
        preprocessor.process_frame.return_value = _make_float32_frame()

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
        await session.process_frame(_make_raw_bytes())

        vad.process_frame.return_value = None
        await session.process_frame(_make_raw_bytes())
        await session.process_frame(_make_raw_bytes())

        # Assert: primeiro frame tem prompt, demais nao
        calls = stream_handle.send_frame.call_args_list
        assert len(calls) == 3

        assert calls[0].kwargs.get("initial_prompt") == "contexto"
        assert calls[1].kwargs.get("initial_prompt") is None
        assert calls[2].kwargs.get("initial_prompt") is None

        # Cleanup
        await session.close()
