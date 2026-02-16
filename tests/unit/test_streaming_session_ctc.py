"""Testes do pipeline adaptativo StreamingSession por arquitetura.

Valida que StreamingSession com architecture=CTC:
- Nao usa cross-segment context (CTC nao suporta initial_prompt)
- Emite partials nativos do worker diretamente
- Emite transcript.final com ITN
- Mantem VAD, state machine, ring buffer, WAL funcionais
- Mantem comportamento encoder-decoder para backward compat
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, Mock

import numpy as np

from macaw._types import STTArchitecture, TranscriptSegment
from macaw.session.streaming import StreamingSession
from tests.helpers import AsyncIterFromList


def _make_stream_handle(
    events: list | None = None,
) -> Mock:
    """Cria mock de StreamHandle com async iterator correto."""
    handle = Mock()
    handle.is_closed = False
    handle.session_id = "test-ctc"

    if events is None:
        events = []
    handle.receive_events.return_value = AsyncIterFromList(events)

    handle.send_frame = AsyncMock()
    handle.close = AsyncMock()
    handle.cancel = AsyncMock()
    return handle


def _make_session(
    *,
    architecture: STTArchitecture = STTArchitecture.ENCODER_DECODER,
    engine_supports_hot_words: bool = False,
    hot_words: list[str] | None = None,
    enable_itn: bool = True,
    cross_segment_context: MagicMock | None = None,
    ring_buffer: MagicMock | None = None,
    postprocessor: MagicMock | None = None,
) -> StreamingSession:
    """Cria StreamingSession com mocks minimos."""
    preprocessor = MagicMock()
    preprocessor.process_frame.return_value = np.zeros(320, dtype=np.float32)

    vad = MagicMock()
    vad.process_frame.return_value = None
    vad.is_speaking = False

    grpc_client = MagicMock()
    on_event = AsyncMock()

    return StreamingSession(
        session_id="test-ctc",
        preprocessor=preprocessor,
        vad=vad,
        grpc_client=grpc_client,
        postprocessor=postprocessor,
        on_event=on_event,
        hot_words=hot_words,
        enable_itn=enable_itn,
        architecture=architecture,
        engine_supports_hot_words=engine_supports_hot_words,
        cross_segment_context=cross_segment_context,
        ring_buffer=ring_buffer,
    )


class TestCTCArchitectureParam:
    """StreamingSession aceita e armazena architecture."""

    def test_default_architecture_is_encoder_decoder(self) -> None:
        session = _make_session()
        assert session._architecture == STTArchitecture.ENCODER_DECODER

    def test_ctc_architecture_stored(self) -> None:
        session = _make_session(architecture=STTArchitecture.CTC)
        assert session._architecture == STTArchitecture.CTC

    def test_encoder_decoder_architecture_stored(self) -> None:
        session = _make_session(architecture=STTArchitecture.ENCODER_DECODER)
        assert session._architecture == STTArchitecture.ENCODER_DECODER


class TestCTCCrossSegmentContext:
    """CTC nao usa cross-segment context (nao suporta initial_prompt)."""

    def test_ctc_build_prompt_skips_context(self) -> None:
        """CTC: cross-segment context NAO incluido no prompt."""
        context = MagicMock()
        context.get_prompt.return_value = "contexto anterior"

        session = _make_session(
            architecture=STTArchitecture.CTC,
            cross_segment_context=context,
        )
        prompt = session._build_initial_prompt()
        # CTC: context should be skipped
        assert prompt is None

    def test_encoder_decoder_build_prompt_uses_context(self) -> None:
        """Encoder-decoder: cross-segment context incluido no prompt."""
        context = MagicMock()
        context.get_prompt.return_value = "contexto anterior"

        session = _make_session(
            architecture=STTArchitecture.ENCODER_DECODER,
            cross_segment_context=context,
        )
        prompt = session._build_initial_prompt()
        assert prompt is not None
        assert "contexto anterior" in prompt

    def test_ctc_with_hot_words_no_native_still_in_prompt(self) -> None:
        """CTC sem suporte nativo: hot words injetadas no prompt (workaround)."""
        session = _make_session(
            architecture=STTArchitecture.CTC,
            engine_supports_hot_words=False,
            hot_words=["PIX", "TED"],
        )
        prompt = session._build_initial_prompt()
        assert prompt is not None
        assert "PIX" in prompt
        assert "TED" in prompt

    def test_ctc_with_hot_words_native_no_prompt(self) -> None:
        """CTC com suporte nativo: hot words NAO no prompt."""
        session = _make_session(
            architecture=STTArchitecture.CTC,
            engine_supports_hot_words=True,
            hot_words=["PIX", "TED"],
        )
        prompt = session._build_initial_prompt()
        assert prompt is None

    def test_ctc_with_context_and_hot_words_no_native(self) -> None:
        """CTC sem suporte nativo: hot words no prompt, context ignorado."""
        context = MagicMock()
        context.get_prompt.return_value = "contexto anterior"

        session = _make_session(
            architecture=STTArchitecture.CTC,
            engine_supports_hot_words=False,
            hot_words=["PIX"],
            cross_segment_context=context,
        )
        prompt = session._build_initial_prompt()
        # Hot words should be in prompt, context should NOT
        assert prompt is not None
        assert "PIX" in prompt
        assert "contexto anterior" not in prompt

    def test_encoder_decoder_with_context_and_hot_words_no_native(self) -> None:
        """Encoder-decoder: hot words E context no prompt."""
        context = MagicMock()
        context.get_prompt.return_value = "contexto anterior"

        session = _make_session(
            architecture=STTArchitecture.ENCODER_DECODER,
            engine_supports_hot_words=False,
            hot_words=["PIX"],
            cross_segment_context=context,
        )
        prompt = session._build_initial_prompt()
        assert prompt is not None
        assert "PIX" in prompt
        assert "contexto anterior" in prompt


class TestCTCReceiveWorkerEvents:
    """CTC: partials nativos do worker emitidos diretamente."""

    async def test_ctc_partial_emitted_directly(self) -> None:
        """CTC: TranscriptSegment(is_final=False) -> transcript.partial."""
        session = _make_session(architecture=STTArchitecture.CTC)
        on_event = session._on_event

        partial_segment = TranscriptSegment(
            text="ola",
            is_final=False,
            segment_id=0,
            start_ms=100,
        )

        mock_handle = _make_stream_handle(events=[partial_segment])
        session._stream_handle = mock_handle

        await session._receive_worker_events()

        on_event.assert_called_once()
        event = on_event.call_args[0][0]
        assert event.type == "transcript.partial"
        assert event.text == "ola"

    async def test_ctc_final_with_itn(self) -> None:
        """CTC: TranscriptSegment(is_final=True) -> ITN -> transcript.final."""
        mock_postprocessor = MagicMock()
        mock_postprocessor.process.return_value = "2025"

        session = _make_session(
            architecture=STTArchitecture.CTC,
            postprocessor=mock_postprocessor,
            enable_itn=True,
        )
        on_event = session._on_event

        final_segment = TranscriptSegment(
            text="dois mil e vinte e cinco",
            is_final=True,
            segment_id=0,
            start_ms=0,
            end_ms=2000,
            language="pt",
            confidence=0.95,
        )

        mock_handle = _make_stream_handle(events=[final_segment])
        session._stream_handle = mock_handle

        await session._receive_worker_events()

        mock_postprocessor.process.assert_called_once_with(
            "dois mil e vinte e cinco",
        )
        on_event.assert_called_once()
        event = on_event.call_args[0][0]
        assert event.type == "transcript.final"
        assert event.text == "2025"

    async def test_ctc_final_does_not_update_cross_segment_context(self) -> None:
        """CTC: transcript.final NAO atualiza cross-segment context."""
        context = MagicMock()
        session = _make_session(
            architecture=STTArchitecture.CTC,
            cross_segment_context=context,
        )

        final_segment = TranscriptSegment(
            text="hello",
            is_final=True,
            segment_id=0,
            start_ms=0,
            end_ms=1000,
        )

        mock_handle = _make_stream_handle(events=[final_segment])
        session._stream_handle = mock_handle

        await session._receive_worker_events()

        # Cross-segment context should NOT be updated for CTC
        context.update.assert_not_called()

    async def test_encoder_decoder_final_updates_cross_segment_context(self) -> None:
        """Encoder-decoder: transcript.final atualiza cross-segment context."""
        context = MagicMock()
        session = _make_session(
            architecture=STTArchitecture.ENCODER_DECODER,
            cross_segment_context=context,
        )

        final_segment = TranscriptSegment(
            text="hello world",
            is_final=True,
            segment_id=0,
            start_ms=0,
            end_ms=1000,
        )

        mock_handle = _make_stream_handle(events=[final_segment])
        session._stream_handle = mock_handle

        await session._receive_worker_events()

        context.update.assert_called_once_with("hello world")

    async def test_ctc_partial_and_final_sequence(self) -> None:
        """CTC: sequencia partial -> final emitida corretamente."""
        session = _make_session(architecture=STTArchitecture.CTC)
        on_event = session._on_event

        segments = [
            TranscriptSegment(text="ola", is_final=False, segment_id=0, start_ms=100),
            TranscriptSegment(text="ola como", is_final=False, segment_id=0, start_ms=200),
            TranscriptSegment(
                text="ola como posso ajudar",
                is_final=True,
                segment_id=0,
                start_ms=0,
                end_ms=2000,
            ),
        ]

        mock_handle = _make_stream_handle(events=segments)
        session._stream_handle = mock_handle

        await session._receive_worker_events()

        assert on_event.call_count == 3
        events = [call[0][0] for call in on_event.call_args_list]
        assert events[0].type == "transcript.partial"
        assert events[0].text == "ola"
        assert events[1].type == "transcript.partial"
        assert events[1].text == "ola como"
        assert events[2].type == "transcript.final"
        assert events[2].text == "ola como posso ajudar"


class TestCTCStateAndVAD:
    """CTC: state machine e VAD funcionam normalmente."""

    async def test_ctc_session_starts_in_init(self) -> None:
        from macaw._types import SessionState

        session = _make_session(architecture=STTArchitecture.CTC)
        assert session.session_state == SessionState.INIT

    async def test_ctc_close_works(self) -> None:
        session = _make_session(architecture=STTArchitecture.CTC)
        await session.close()
        assert session.is_closed

    async def test_ctc_wal_checkpoint_on_final(self) -> None:
        """CTC: WAL checkpoint registrado apos transcript.final."""
        session = _make_session(architecture=STTArchitecture.CTC)

        final_segment = TranscriptSegment(
            text="test",
            is_final=True,
            segment_id=0,
            start_ms=0,
            end_ms=500,
        )

        mock_handle = _make_stream_handle(events=[final_segment])
        session._stream_handle = mock_handle

        await session._receive_worker_events()

        # WAL should have a checkpoint
        assert session.wal.last_committed_segment_id == 0

    async def test_ctc_ring_buffer_commit_on_final(self) -> None:
        """CTC: ring buffer commit apos transcript.final."""
        mock_rb = MagicMock()
        mock_rb.total_written = 16000

        session = _make_session(
            architecture=STTArchitecture.CTC,
            ring_buffer=mock_rb,
        )

        final_segment = TranscriptSegment(
            text="test",
            is_final=True,
            segment_id=0,
            start_ms=0,
            end_ms=500,
        )

        mock_handle = _make_stream_handle(events=[final_segment])
        session._stream_handle = mock_handle

        await session._receive_worker_events()

        mock_rb.commit.assert_called_once_with(16000)


class TestBackwardCompat:
    """Encoder-decoder: comportamento M6 mantido (regressao zero)."""

    def test_encoder_decoder_default_no_architecture_param(self) -> None:
        """Sem parametro architecture = encoder-decoder (backward compat)."""
        preprocessor = MagicMock()
        vad = MagicMock()
        grpc_client = MagicMock()
        on_event = AsyncMock()

        session = StreamingSession(
            session_id="test",
            preprocessor=preprocessor,
            vad=vad,
            grpc_client=grpc_client,
            postprocessor=None,
            on_event=on_event,
        )
        assert session._architecture == STTArchitecture.ENCODER_DECODER

    def test_encoder_decoder_builds_prompt_with_context(self) -> None:
        """Encoder-decoder: prompt com context funciona."""
        context = MagicMock()
        context.get_prompt.return_value = "previous text"

        session = _make_session(
            architecture=STTArchitecture.ENCODER_DECODER,
            cross_segment_context=context,
        )
        prompt = session._build_initial_prompt()
        assert prompt is not None
        assert "previous text" in prompt

    def test_encoder_decoder_hot_words_in_prompt_no_native(self) -> None:
        """Encoder-decoder sem suporte nativo: hot words no prompt."""
        session = _make_session(
            architecture=STTArchitecture.ENCODER_DECODER,
            engine_supports_hot_words=False,
            hot_words=["PIX"],
        )
        prompt = session._build_initial_prompt()
        assert prompt is not None
        assert "PIX" in prompt
