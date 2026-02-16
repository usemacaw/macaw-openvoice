"""Testes de hot words por sessao na StreamingSession.

Valida que hot words sao:
- Armazenados corretamente no init e via update_hot_words().
- Enviados ao worker apenas no primeiro frame de cada segmento de fala.
- Atualizados dinamicamente e usados no proximo segmento.

Todos os testes usam mocks para dependencias externas.
Testes sao deterministicos — sem dependencia de timing real.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

from macaw.session.streaming import StreamingSession
from macaw.vad.detector import VADEvent, VADEventType
from tests.helpers import make_float32_frame, make_raw_bytes


def _make_stream_handle_mock() -> Mock:
    """Cria mock de StreamHandle."""
    handle = Mock()
    handle.is_closed = False
    handle.session_id = "test_session"

    # receive_events retorna async iterator vazio
    async def _empty_iter():
        return
        yield  # necessario para tornar um async generator

    handle.receive_events.return_value = _empty_iter()
    handle.send_frame = AsyncMock()
    handle.close = AsyncMock()
    handle.cancel = AsyncMock()
    return handle


def _make_session(
    hot_words: list[str] | None = None,
) -> tuple[StreamingSession, Mock, Mock, AsyncMock, AsyncMock]:
    """Cria StreamingSession com mocks configurados.

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
# Tests: armazenamento de hot words
# ---------------------------------------------------------------------------


async def test_session_created_with_hot_words():
    """Hot words fornecidos no init sao armazenados corretamente."""
    # Arrange & Act
    session, _, _, _, _ = _make_session(hot_words=["PIX", "TED", "Selic"])

    # Assert
    assert session._hot_words == ["PIX", "TED", "Selic"]

    # Cleanup
    await session.close()


async def test_session_created_without_hot_words():
    """Sessao criada sem hot words tem None como default."""
    # Arrange & Act
    session, _, _, _, _ = _make_session(hot_words=None)

    # Assert
    assert session._hot_words is None

    # Cleanup
    await session.close()


async def test_update_hot_words():
    """update_hot_words() altera os hot words armazenados."""
    # Arrange
    session, _, _, _, _ = _make_session(hot_words=["PIX"])

    # Act
    session.update_hot_words(["TED", "Selic", "CDI"])

    # Assert
    assert session._hot_words == ["TED", "Selic", "CDI"]

    # Cleanup
    await session.close()


async def test_update_hot_words_to_none():
    """update_hot_words(None) limpa os hot words."""
    # Arrange
    session, _, _, _, _ = _make_session(hot_words=["PIX", "TED"])

    # Act
    session.update_hot_words(None)

    # Assert
    assert session._hot_words is None

    # Cleanup
    await session.close()


# ---------------------------------------------------------------------------
# Tests: envio de hot words ao worker
# ---------------------------------------------------------------------------


async def test_hot_words_sent_on_first_frame():
    """Hot words sao enviados ao worker no primeiro frame do segmento."""
    # Arrange
    session, _, vad, stream_handle, _ = _make_session(
        hot_words=["PIX", "TED", "Selic"],
    )

    # Act: trigger speech_start
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = True  # Apos speech_start
    await session.process_frame(make_raw_bytes())

    # Assert: primeiro send_frame deve incluir hot_words
    assert stream_handle.send_frame.call_count == 1
    first_call = stream_handle.send_frame.call_args_list[0]
    assert first_call.kwargs.get("hot_words") == ["PIX", "TED", "Selic"]

    # Cleanup
    await session.close()


async def test_hot_words_not_sent_on_subsequent_frames():
    """Hot words NAO sao enviados nos frames subsequentes do mesmo segmento."""
    # Arrange
    session, _, vad, stream_handle, _ = _make_session(
        hot_words=["PIX", "TED"],
    )

    # Act: trigger speech_start + enviar 2 frames adicionais
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = True
    await session.process_frame(make_raw_bytes())

    vad.process_frame.return_value = None  # Sem transicao
    await session.process_frame(make_raw_bytes())
    await session.process_frame(make_raw_bytes())

    # Assert: 3 frames enviados
    assert stream_handle.send_frame.call_count == 3

    # Segundo e terceiro frames: hot_words deve ser None
    second_call = stream_handle.send_frame.call_args_list[1]
    third_call = stream_handle.send_frame.call_args_list[2]
    assert second_call.kwargs.get("hot_words") is None
    assert third_call.kwargs.get("hot_words") is None

    # Cleanup
    await session.close()


async def test_hot_words_reset_on_new_segment():
    """Hot words sao reenviados no primeiro frame de um novo segmento."""
    # Arrange
    session, _, vad, _, _ = _make_session(
        hot_words=["PIX"],
    )
    grpc_client = session._grpc_client

    # Primeiro segmento: speech_start -> speech_end
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = True
    await session.process_frame(make_raw_bytes())
    await asyncio.sleep(0.01)

    # Preparar novo stream_handle para proximo segmento
    stream_handle_2 = _make_stream_handle_mock()
    grpc_client.open_stream = AsyncMock(return_value=stream_handle_2)

    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_END,
        timestamp_ms=2000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())

    # Segundo segmento: speech_start
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=3000,
    )
    vad.is_speaking = True
    await session.process_frame(make_raw_bytes())

    # Assert: hot_words enviados no primeiro frame do segundo segmento
    assert stream_handle_2.send_frame.call_count == 1
    first_call_seg2 = stream_handle_2.send_frame.call_args_list[0]
    assert first_call_seg2.kwargs.get("hot_words") == ["PIX"]

    # Cleanup
    await session.close()


async def test_updated_hot_words_used_in_next_segment():
    """Apos update_hot_words(), os NOVOS hot words sao usados no proximo segmento."""
    # Arrange
    session, _, vad, stream_handle, _ = _make_session(
        hot_words=["PIX", "TED"],
    )
    grpc_client = session._grpc_client

    # Primeiro segmento: speech_start -> frame -> speech_end
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = True
    await session.process_frame(make_raw_bytes())
    await asyncio.sleep(0.01)

    # Verificar que os hot words originais foram enviados
    first_call = stream_handle.send_frame.call_args_list[0]
    assert first_call.kwargs.get("hot_words") == ["PIX", "TED"]

    # Preparar novo stream_handle para proximo segmento
    stream_handle_2 = _make_stream_handle_mock()
    grpc_client.open_stream = AsyncMock(return_value=stream_handle_2)

    # Speech end: fecha primeiro segmento
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_END,
        timestamp_ms=2000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())

    # Act: atualizar hot words entre segmentos
    session.update_hot_words(["Selic", "CDI", "IPCA"])

    # Segundo segmento: speech_start
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=3000,
    )
    vad.is_speaking = True
    await session.process_frame(make_raw_bytes())

    # Assert: hot words ATUALIZADOS enviados no primeiro frame do segundo segmento
    assert stream_handle_2.send_frame.call_count == 1
    first_call_seg2 = stream_handle_2.send_frame.call_args_list[0]
    assert first_call_seg2.kwargs.get("hot_words") == ["Selic", "CDI", "IPCA"]

    # Cleanup
    await session.close()


async def test_no_hot_words_sent_when_none():
    """Quando hot_words e None, nenhum hot word e enviado ao worker."""
    # Arrange
    session, _, vad, stream_handle, _ = _make_session(hot_words=None)

    # Act: trigger speech_start + frame
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = True
    await session.process_frame(make_raw_bytes())

    # Assert: send_frame chamado sem hot_words
    assert stream_handle.send_frame.call_count == 1
    first_call = stream_handle.send_frame.call_args_list[0]
    assert first_call.kwargs.get("hot_words") is None

    # Cleanup
    await session.close()


async def test_update_hot_words_empty_list_sent_as_none():
    """Lista vazia de hot words resulta em None enviado ao worker.

    A logica em _send_frame_to_worker verifica `if self._hot_words` (truthy),
    entao lista vazia e tratada como falsy e nao e enviada.
    """
    # Arrange
    session, _, vad, stream_handle, _ = _make_session(hot_words=["PIX"])

    # Act: atualizar para lista vazia
    session.update_hot_words([])

    # Trigger speech_start
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = True
    await session.process_frame(make_raw_bytes())

    # Assert: hot_words nao enviado (lista vazia e falsy)
    assert stream_handle.send_frame.call_count == 1
    first_call = stream_handle.send_frame.call_args_list[0]
    assert first_call.kwargs.get("hot_words") is None

    # Cleanup
    await session.close()
