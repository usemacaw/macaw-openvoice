"""Testes do StreamingSession.commit() — manual commit de segmento.

Valida que o commit manual fecha o stream gRPC atual, aguarda o worker
retornar transcript.final, incrementa segment_id e reseta estado para
que o proximo audio abra novo stream com hot words reenviados.

Todos os testes usam mocks para dependencias externas.
Testes sao deterministicos — sem dependencia de timing real.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

from macaw._types import TranscriptSegment
from macaw.server.models.events import (
    TranscriptFinalEvent,
)
from macaw.session.streaming import StreamingSession
from macaw.vad.detector import VADEvent, VADEventType
from tests.helpers import AsyncIterFromList, make_preprocessor_mock, make_raw_bytes, make_vad_mock


def _make_stream_handle_mock(
    events: list | None = None,
) -> Mock:
    """Cria mock de StreamHandle."""
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
    mock.process.side_effect = lambda text, **kwargs: f"ITN({text})"
    return mock


def _make_on_event() -> AsyncMock:
    """Cria callback on_event mock."""
    return AsyncMock()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_commit_during_speech_closes_stream_and_increments_segment():
    """commit() durante fala fecha o stream e incrementa segment_id."""
    # Arrange
    final_segment = TranscriptSegment(
        text="ola mundo",
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
    on_event = _make_on_event()

    session = StreamingSession(
        session_id="test_session",
        preprocessor=make_preprocessor_mock(),
        vad=vad,
        grpc_client=grpc_client,
        postprocessor=_make_postprocessor_mock(),
        on_event=on_event,
        enable_itn=False,
    )

    assert session.segment_id == 0

    # Trigger speech_start -> abre stream
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = True
    await session.process_frame(make_raw_bytes())
    await asyncio.sleep(0.05)  # Dar tempo para receiver task processar

    # Act: manual commit
    await session.commit()

    # Assert
    assert session.segment_id == 1
    stream_handle.close.assert_called_once()  # Stream foi fechado

    # Cleanup
    await session.close()


async def test_commit_during_silence_is_noop():
    """commit() durante silencio (sem stream handle) e no-op, sem erro."""
    # Arrange
    vad = make_vad_mock(is_speaking=False)
    on_event = _make_on_event()

    session = StreamingSession(
        session_id="test_session",
        preprocessor=make_preprocessor_mock(),
        vad=vad,
        grpc_client=_make_grpc_client_mock(),
        postprocessor=_make_postprocessor_mock(),
        on_event=on_event,
    )

    assert session.segment_id == 0

    # Act: commit sem stream ativo
    await session.commit()

    # Assert: no-op, segment_id nao muda
    assert session.segment_id == 0
    assert not session.is_closed


async def test_commit_on_closed_session_is_noop():
    """commit() em sessao fechada e no-op, sem erro."""
    # Arrange
    session = StreamingSession(
        session_id="test_session",
        preprocessor=make_preprocessor_mock(),
        vad=make_vad_mock(),
        grpc_client=_make_grpc_client_mock(),
        postprocessor=_make_postprocessor_mock(),
        on_event=_make_on_event(),
    )

    await session.close()
    assert session.is_closed

    # Act: commit em sessao fechada
    await session.commit()

    # Assert: no-op, sem excecao
    assert session.is_closed
    assert session.segment_id == 0


async def test_commit_resets_hot_words_for_next_segment():
    """Apos commit, hot words sao reenviados no proximo segmento."""
    # Arrange
    stream_handle1 = _make_stream_handle_mock()
    grpc_client = _make_grpc_client_mock(stream_handle1)
    vad = make_vad_mock()
    preprocessor = make_preprocessor_mock()

    session = StreamingSession(
        session_id="test_session",
        preprocessor=preprocessor,
        vad=vad,
        grpc_client=grpc_client,
        postprocessor=_make_postprocessor_mock(),
        on_event=_make_on_event(),
        hot_words=["PIX", "TED"],
    )

    # Trigger speech_start
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = True
    await session.process_frame(make_raw_bytes())

    # Primeiro frame: hot words enviados
    calls_before_commit = stream_handle1.send_frame.call_args_list
    assert calls_before_commit[0].kwargs.get("hot_words") == ["PIX", "TED"]

    # Enviar segundo frame: hot words NAO enviados
    vad.process_frame.return_value = None
    await session.process_frame(make_raw_bytes())
    assert stream_handle1.send_frame.call_args_list[1].kwargs.get("hot_words") is None

    # Act: manual commit
    await session.commit()

    # Preparar novo stream para proximo segmento
    stream_handle2 = _make_stream_handle_mock()
    grpc_client.open_stream = AsyncMock(return_value=stream_handle2)

    # Trigger novo speech_start -> abre novo stream
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=3000,
    )
    vad.is_speaking = True
    await session.process_frame(make_raw_bytes())

    # Assert: hot words enviados novamente no primeiro frame do novo segmento
    new_calls = stream_handle2.send_frame.call_args_list
    assert new_calls[0].kwargs.get("hot_words") == ["PIX", "TED"]

    # Cleanup
    await session.close()


async def test_commit_produces_transcript_final_from_worker():
    """commit() faz o worker emitir transcript.final para o audio acumulado."""
    # Arrange
    final_segment = TranscriptSegment(
        text="resultado do commit",
        is_final=True,
        segment_id=0,
        start_ms=1000,
        end_ms=3000,
        language="pt",
        confidence=0.9,
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
        enable_itn=True,
    )

    # Trigger speech_start
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = True
    await session.process_frame(make_raw_bytes())
    await asyncio.sleep(0.05)  # Receiver task processa o final_segment

    # Act: commit
    await session.commit()

    # Assert: transcript.final emitido via on_event
    final_calls = [
        call for call in on_event.call_args_list if isinstance(call.args[0], TranscriptFinalEvent)
    ]
    assert len(final_calls) == 1
    assert final_calls[0].args[0].text == "ITN(resultado do commit)"

    # Cleanup
    await session.close()


async def test_commit_opens_new_stream_for_subsequent_audio():
    """Apos commit, proximo speech_start abre novo stream com open_stream."""
    # Arrange
    stream_handle1 = _make_stream_handle_mock()
    grpc_client = _make_grpc_client_mock(stream_handle1)
    vad = make_vad_mock()

    session = StreamingSession(
        session_id="test_session",
        preprocessor=make_preprocessor_mock(),
        vad=vad,
        grpc_client=grpc_client,
        postprocessor=_make_postprocessor_mock(),
        on_event=_make_on_event(),
    )

    # Trigger speech_start -> abre stream 1
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = True
    await session.process_frame(make_raw_bytes())
    await asyncio.sleep(0.01)

    # open_stream chamado 1 vez
    assert grpc_client.open_stream.call_count == 1

    # Commit
    await session.commit()

    # Preparar novo stream handle
    stream_handle2 = _make_stream_handle_mock()
    grpc_client.open_stream = AsyncMock(return_value=stream_handle2)

    # Trigger novo speech_start -> deve abrir stream 2
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=3000,
    )
    vad.is_speaking = True
    await session.process_frame(make_raw_bytes())
    await asyncio.sleep(0.01)

    # Assert: open_stream chamado novamente
    grpc_client.open_stream.assert_called_once_with("test_session")

    # Cleanup
    await session.close()


async def test_commit_with_already_closed_stream_handle():
    """commit() funciona mesmo se stream handle ja esta fechado."""
    # Arrange
    stream_handle = _make_stream_handle_mock()
    stream_handle.is_closed = True  # Stream ja fechado
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

    # Trigger speech_start para ter stream_handle setado
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())
    await asyncio.sleep(0.01)

    initial_segment_id = session.segment_id

    # Act: commit com stream ja fechado (nao deve chamar close novamente)
    await session.commit()

    # Assert: segment_id incrementou, sem erro
    assert session.segment_id == initial_segment_id + 1
    stream_handle.close.assert_not_called()  # Nao fechou porque ja estava fechado

    # Cleanup
    await session.close()
