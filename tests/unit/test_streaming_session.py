"""Testes do StreamingSession.

Valida que o orquestrador de streaming coordena corretamente:
preprocessing -> VAD -> gRPC worker -> post-processing.

Todos os testes usam mocks para dependencias externas.
Testes sao deterministicos — sem dependencia de timing real.
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
    """Cria mock de StreamHandle.

    Args:
        events: Lista de TranscriptSegment ou Exception para o receive_events.
                Se None, retorna iterator vazio.
    """
    handle = Mock()
    handle.is_closed = False
    handle.session_id = "test_session"

    # receive_events retorna um async iterable
    if events is None:
        events = []
    handle.receive_events.return_value = AsyncIterFromList(events)

    handle.send_frame = AsyncMock()
    handle.close = AsyncMock()
    handle.cancel = AsyncMock()
    return handle


def _make_grpc_client_mock(stream_handle: Mock | None = None) -> AsyncMock:
    """Cria mock de StreamingGRPCClient."""
    client = AsyncMock()
    if stream_handle is None:
        stream_handle = _make_stream_handle_mock()
    client.open_stream = AsyncMock(return_value=stream_handle)
    client.close = AsyncMock()
    return client


def _make_postprocessor_mock() -> Mock:
    """Cria mock de PostProcessingPipeline."""
    mock = Mock()
    mock.process.side_effect = lambda text: f"ITN({text})"
    return mock


def _make_on_event() -> AsyncMock:
    """Cria callback on_event mock."""
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
    """Cria StreamingSession com mocks configurados.

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
    """VAD speech_start emite vad.speech_start via callback."""
    # Arrange
    vad = make_vad_mock()
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1500,
    )
    vad.is_speaking = False  # Antes do speech_start, nao esta falando

    session, _, _, _, _, on_event = _make_session(vad=vad)

    # Act
    await session.process_frame(make_raw_bytes())

    # Dar tempo para receiver task iniciar
    await asyncio.sleep(0.01)

    # Assert
    on_event.assert_any_call(
        VADSpeechStartEvent(timestamp_ms=1500),
    )

    # Cleanup
    await session.close()


async def test_speech_end_emits_vad_event():
    """VAD speech_end emite vad.speech_end via callback."""
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

    # Simular speech_start
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())
    await asyncio.sleep(0.01)

    # Simular speech_end
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
    """ITN e aplicado apenas em transcript.final."""
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

    # Trigger speech_start -> abre stream e inicia receiver
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())

    # Dar tempo para o receiver task processar o evento
    await asyncio.sleep(0.05)

    # Trigger speech_end para flush
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_END,
        timestamp_ms=2000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())

    # Assert: ITN foi aplicado ao texto final
    postprocessor.process.assert_called_once_with("dois mil e vinte e cinco")

    # Verificar que o evento final tem texto pos-processado
    final_calls = [
        call for call in on_event.call_args_list if isinstance(call.args[0], TranscriptFinalEvent)
    ]
    assert len(final_calls) == 1
    assert final_calls[0].args[0].text == "ITN(dois mil e vinte e cinco)"


async def test_partial_transcript_no_postprocessing():
    """Partial transcripts NAO sao processados por ITN."""
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

    # Dar tempo para receiver task
    await asyncio.sleep(0.05)

    # Assert: ITN NAO foi chamado
    postprocessor.process.assert_not_called()

    # Verificar que o partial foi emitido com texto original
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
    """Cada segmento de fala recebe segment_id incremental."""
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

    # Primeiro segmento: speech_start -> speech_end
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())
    await asyncio.sleep(0.01)

    # Novo stream_handle para proximo open_stream
    stream_handle2 = _make_stream_handle_mock()
    grpc_client.open_stream = AsyncMock(return_value=stream_handle2)

    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_END,
        timestamp_ms=2000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())

    assert session.segment_id == 1

    # Segundo segmento: speech_start -> speech_end
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
    """close() fecha gRPC stream e marca sessao como CLOSED."""
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

    # Abrir stream (trigger speech_start)
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
    """Frames recebidos apos close sao ignorados."""
    # Arrange
    preprocessor = make_preprocessor_mock()

    session, _, _, _, _, _ = _make_session(preprocessor=preprocessor)

    # Fechar sessao
    await session.close()
    assert session.is_closed

    # Resetar contadores do mock
    preprocessor.process_frame.reset_mock()

    # Act: tentar processar frame
    await session.process_frame(make_raw_bytes())

    # Assert: preprocessor NAO foi chamado
    preprocessor.process_frame.assert_not_called()


async def test_full_speech_cycle():
    """Ciclo completo: speech_start -> partials -> final -> speech_end em ordem."""
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

    # Dar tempo para receiver task processar ambos eventos
    await asyncio.sleep(0.05)

    # 2. Speech end
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_END,
        timestamp_ms=2000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())

    # Assert: eventos na ordem correta
    event_types = [type(call.args[0]).__name__ for call in on_event.call_args_list]

    assert "VADSpeechStartEvent" in event_types
    assert "TranscriptPartialEvent" in event_types
    assert "TranscriptFinalEvent" in event_types
    assert "VADSpeechEndEvent" in event_types

    # Verificar ordem: speech_start < partial < final < speech_end
    start_idx = event_types.index("VADSpeechStartEvent")
    partial_idx = event_types.index("TranscriptPartialEvent")
    final_idx = event_types.index("TranscriptFinalEvent")
    end_idx = event_types.index("VADSpeechEndEvent")

    assert start_idx < partial_idx < final_idx < end_idx


async def test_hot_words_sent_only_on_first_frame():
    """Hot words sao enviados ao worker apenas no primeiro frame do segmento."""
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
    vad.is_speaking = True  # Apos speech_start, is_speaking = True
    await session.process_frame(make_raw_bytes())

    # Enviar mais frames durante fala
    vad.process_frame.return_value = None  # Sem transicao
    await session.process_frame(make_raw_bytes())
    await session.process_frame(make_raw_bytes())

    # Assert: hot_words enviados no primeiro frame, None nos seguintes
    calls = stream_handle.send_frame.call_args_list

    # Primeiro send_frame: deve ter hot_words
    assert calls[0].kwargs.get("hot_words") == ["PIX", "TED", "Selic"]

    # Segundo e terceiro: sem hot_words
    assert calls[1].kwargs.get("hot_words") is None
    assert calls[2].kwargs.get("hot_words") is None

    # Cleanup
    await session.close()


async def test_close_is_idempotent():
    """Chamar close() multiplas vezes nao causa erro."""
    # Arrange
    session, _, _, _, _, _ = _make_session()

    # Act
    await session.close()
    await session.close()
    await session.close()

    # Assert
    assert session.is_closed


async def test_no_postprocessor_skips_itn():
    """Se postprocessor e None, transcript.final e emitido sem ITN."""
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

    # Assert: texto sem ITN
    final_calls = [
        call for call in on_event.call_args_list if isinstance(call.args[0], TranscriptFinalEvent)
    ]
    assert len(final_calls) == 1
    assert final_calls[0].args[0].text == "texto cru"


async def test_itn_disabled_skips_postprocessing():
    """Se enable_itn=False, transcript.final e emitido sem ITN."""
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

    # Assert: postprocessor NAO foi chamado
    postprocessor.process.assert_not_called()

    # Texto original emitido
    final_calls = [
        call for call in on_event.call_args_list if isinstance(call.args[0], TranscriptFinalEvent)
    ]
    assert len(final_calls) == 1
    assert final_calls[0].args[0].text == "texto cru"


async def test_worker_crash_emits_error_event():
    """Worker crash durante streaming emite erro recuperavel."""
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

    # Trigger speech_start (inicia receiver task que vai crashar)
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())

    await asyncio.sleep(0.05)

    # Assert: evento de erro emitido
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
    """check_inactivity() retorna True apos timeout de INIT (30s default)."""
    # Arrange: usar clock controlavel na state machine
    current_time = [0.0]

    def fake_clock() -> float:
        return current_time[0]

    sm = SessionStateMachine(clock=fake_clock)
    session, _, _, _, _, _ = _make_session()
    # Injetar state machine com clock controlavel
    session._state_machine = sm

    # Recentemente criado: nao expirou
    assert not session.check_inactivity()

    # Simular 31s sem audio (INIT timeout = 30s -> CLOSED)
    current_time[0] = 31.0
    assert session.check_inactivity()


async def test_inactivity_reset_on_frame():
    """process_frame() reseta o timer de inatividade via state machine transition.

    Quando um frame com speech_start e processado, a state machine transita
    de INIT para ACTIVE (que nao tem timeout), impedindo que check_inactivity()
    retorne True. Frames em INIT (sem fala) nao resetam o timer da state machine,
    mas o estado INIT tem timeout de 30s para CLOSED.
    """
    # Arrange: usar clock controlavel na state machine
    current_time = [0.0]

    def fake_clock() -> float:
        return current_time[0]

    sm = SessionStateMachine(clock=fake_clock)
    session, _, vad, _, _, _ = _make_session()
    session._state_machine = sm

    # Verificar que em INIT com 25s passados, nao expirou
    current_time[0] = 25.0
    assert not session.check_inactivity()

    # Simular speech_start -> transita para ACTIVE (sem timeout)
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = True
    current_time[0] = 26.0
    await session.process_frame(make_raw_bytes())
    await asyncio.sleep(0.01)

    # ACTIVE nao tem timeout, logo check_inactivity retorna False
    current_time[0] = 200.0
    assert not session.check_inactivity()

    # Cleanup
    await session.close()


async def test_word_timestamps_in_final():
    """Final transcript com word timestamps e corretamente convertido."""
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
        enable_itn=False,  # Sem ITN para simplificar
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
    """Falha ao abrir gRPC stream emite erro recuperavel."""
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

    # Trigger speech_start -> open_stream vai falhar
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())

    # Assert: erro emitido
    error_calls = [
        call for call in on_event.call_args_list if isinstance(call.args[0], StreamingErrorEvent)
    ]
    assert len(error_calls) == 1
    assert error_calls[0].args[0].code == "worker_crash"
    assert error_calls[0].args[0].recoverable is True

    # Cleanup
    await session.close()


async def test_frames_sent_during_speech():
    """Frames de audio sao enviados ao worker durante fala."""
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

    # Mais frames durante fala
    vad.process_frame.return_value = None
    await session.process_frame(make_raw_bytes())
    await session.process_frame(make_raw_bytes())

    # Assert: 3 frames enviados ao worker (1 do speech_start + 2)
    assert stream_handle.send_frame.call_count == 3

    # Cleanup
    await session.close()


async def test_no_frames_sent_during_silence():
    """Frames NAO sao enviados ao worker quando nao ha fala."""
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

    # Processar frames em silencio
    vad.process_frame.return_value = None
    await session.process_frame(make_raw_bytes())
    await session.process_frame(make_raw_bytes())

    # Assert: nenhum frame enviado
    stream_handle.send_frame.assert_not_called()


class TestEventOrdering:
    """Testes de ordenacao de eventos: transcript.final ANTES de vad.speech_end."""

    async def test_transcript_final_emitted_before_speech_end(self) -> None:
        """transcript.final do worker e emitido ANTES de vad.speech_end.

        Garante que a semantica do protocolo WebSocket e respeitada:
        o ultimo transcript.final de um segmento vem antes do vad.speech_end.
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

        # Act: simular SPEECH_START -> frames -> SPEECH_END
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START,
            timestamp_ms=0,
        )
        await session.process_frame(make_raw_bytes())

        # Enviar frames durante fala
        vad.process_frame.return_value = None
        vad.is_speaking = True
        await session.process_frame(make_raw_bytes())

        # Aguardar receiver task processar o final (dar tempo ao event loop)
        await asyncio.sleep(0.05)

        # Emitir SPEECH_END
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_END,
            timestamp_ms=2000,
        )
        vad.is_speaking = False
        await session.process_frame(make_raw_bytes())

        # Assert: verificar ordenacao
        event_types = [type(e).__name__ for e in events_emitted]

        # Deve ter: speech_start, final, speech_end (nessa ordem)
        assert "VADSpeechStartEvent" in event_types
        assert "TranscriptFinalEvent" in event_types
        assert "VADSpeechEndEvent" in event_types

        final_idx = event_types.index("TranscriptFinalEvent")
        end_idx = event_types.index("VADSpeechEndEvent")
        assert final_idx < end_idx, (
            f"transcript.final (idx={final_idx}) deve vir antes de "
            f"vad.speech_end (idx={end_idx}). Ordem: {event_types}"
        )

        await session.close()
