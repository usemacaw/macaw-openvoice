"""Testes de integracao: SessionStateMachine + StreamingSession.

Valida que a maquina de estados (M6) esta corretamente integrada ao
orquestrador de streaming. Foco em:
- Transicoes de estado disparadas por eventos VAD
- Timeouts por estado com clock controlavel
- Comportamento dependente de estado (frames em HOLD nao enviados)
- Emissao de SessionHoldEvent
- session.configure propagando timeouts
- Estado inicial (INIT, nao ACTIVE)

Todos os testes sao deterministicos — usam clock injetavel.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

import numpy as np

from macaw._types import SessionState
from macaw.server.models.events import SessionHoldEvent
from macaw.session.state_machine import SessionStateMachine, SessionTimeouts
from macaw.session.streaming import StreamingSession
from macaw.vad.detector import VADEvent, VADEventType

# Frame size padrao: 1024 samples a 16kHz = 64ms
_FRAME_SIZE = 1024


def _make_raw_bytes(n_samples: int = _FRAME_SIZE) -> bytes:
    """Gera bytes PCM int16 (zeros) com n_samples amostras."""
    return np.zeros(n_samples, dtype=np.int16).tobytes()


def _make_float32_frame(n_samples: int = _FRAME_SIZE) -> np.ndarray:
    """Gera frame float32 (zeros) para mock de preprocessor."""
    return np.zeros(n_samples, dtype=np.float32)


def _make_preprocessor_mock() -> Mock:
    """Cria mock de StreamingPreprocessor."""
    mock = Mock()
    mock.process_frame.return_value = _make_float32_frame()
    return mock


def _make_vad_mock(*, is_speaking: bool = False) -> Mock:
    """Cria mock de VADDetector."""
    mock = Mock()
    mock.process_frame.return_value = None
    mock.is_speaking = is_speaking
    mock.reset.return_value = None
    return mock


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


def _controllable_clock() -> tuple[list[float], object]:
    """Cria clock controlavel para state machine.

    Returns:
        (time_ref, clock_fn) onde time_ref[0] controla o tempo
        e clock_fn pode ser passado como clock para SessionStateMachine.
    """
    time_ref = [0.0]

    def clock() -> float:
        return time_ref[0]

    return time_ref, clock


def _make_session_with_sm(
    *,
    timeouts: SessionTimeouts | None = None,
    clock_ref: list[float] | None = None,
    clock_fn: object | None = None,
    vad: Mock | None = None,
    grpc_client: AsyncMock | None = None,
    on_event: AsyncMock | None = None,
    hot_words: list[str] | None = None,
    enable_itn: bool = True,
) -> tuple[StreamingSession, SessionStateMachine, Mock, AsyncMock, AsyncMock]:
    """Cria StreamingSession com SessionStateMachine e clock controlavel.

    Returns:
        (session, state_machine, vad, grpc_client, on_event)
    """
    if clock_ref is not None and clock_fn is None:

        def _clock() -> float:
            return clock_ref[0]

        clock_fn = _clock

    if clock_fn is None:
        _ref, clock_fn = _controllable_clock()

    sm = SessionStateMachine(timeouts=timeouts, clock=clock_fn)

    _vad = vad or _make_vad_mock()
    _grpc_client = grpc_client or _make_grpc_client_mock()
    _on_event = on_event or AsyncMock()

    session = StreamingSession(
        session_id="test_session",
        preprocessor=_make_preprocessor_mock(),
        vad=_vad,
        grpc_client=_grpc_client,
        postprocessor=_make_postprocessor_mock(),
        on_event=_on_event,
        hot_words=hot_words,
        enable_itn=enable_itn,
        state_machine=sm,
    )

    return session, sm, _vad, _grpc_client, _on_event


# ---------------------------------------------------------------------------
# Tests: Estado Inicial
# ---------------------------------------------------------------------------


async def test_initial_state_is_init():
    """Estado inicial da sessao deve ser INIT, nao ACTIVE."""
    time_ref, _clock = _controllable_clock()
    session, sm, _, _, _ = _make_session_with_sm(clock_ref=time_ref)

    assert session.session_state == SessionState.INIT
    assert sm.state == SessionState.INIT


async def test_session_state_property_reflects_state_machine():
    """session_state deve refletir o estado da state machine."""
    time_ref, _clock = _controllable_clock()
    session, sm, _, _, _ = _make_session_with_sm(clock_ref=time_ref)

    assert session.session_state == SessionState.INIT

    # Transitar manualmente
    sm.transition(SessionState.ACTIVE)
    assert session.session_state == SessionState.ACTIVE


# ---------------------------------------------------------------------------
# Tests: Transicoes VAD
# ---------------------------------------------------------------------------


async def test_speech_start_transitions_init_to_active():
    """Primeiro frame com fala transita INIT -> ACTIVE."""
    time_ref, _clock = _controllable_clock()
    vad = _make_vad_mock()
    session, _sm, _, _, _ = _make_session_with_sm(clock_ref=time_ref, vad=vad)

    assert session.session_state == SessionState.INIT

    # Emitir speech_start
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = True
    await session.process_frame(_make_raw_bytes())
    await asyncio.sleep(0.01)

    assert session.session_state == SessionState.ACTIVE

    # Cleanup
    await session.close()


async def test_speech_end_transitions_active_to_silence():
    """VAD speech_end transita ACTIVE -> SILENCE."""
    time_ref, _clock = _controllable_clock()
    vad = _make_vad_mock()
    session, _sm, _, _, _ = _make_session_with_sm(clock_ref=time_ref, vad=vad)

    # Primeiro: INIT -> ACTIVE via speech_start
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = True
    await session.process_frame(_make_raw_bytes())
    await asyncio.sleep(0.01)
    assert session.session_state == SessionState.ACTIVE

    # Agora: ACTIVE -> SILENCE via speech_end
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_END,
        timestamp_ms=2000,
    )
    vad.is_speaking = False
    await session.process_frame(_make_raw_bytes())

    assert session.session_state == SessionState.SILENCE


async def test_speech_start_during_silence_transitions_to_active():
    """Nova fala durante SILENCE transita SILENCE -> ACTIVE."""
    time_ref, _clock = _controllable_clock()
    vad = _make_vad_mock()
    grpc_client = _make_grpc_client_mock()
    session, _sm, _, _, _ = _make_session_with_sm(
        clock_ref=time_ref,
        vad=vad,
        grpc_client=grpc_client,
    )

    # INIT -> ACTIVE
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = True
    await session.process_frame(_make_raw_bytes())
    await asyncio.sleep(0.01)

    # ACTIVE -> SILENCE
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_END,
        timestamp_ms=2000,
    )
    vad.is_speaking = False
    await session.process_frame(_make_raw_bytes())
    assert session.session_state == SessionState.SILENCE

    # Novo stream handle para proximo open_stream
    stream_handle2 = _make_stream_handle_mock()
    grpc_client.open_stream = AsyncMock(return_value=stream_handle2)

    # SILENCE -> ACTIVE (nova fala)
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=3000,
    )
    vad.is_speaking = True
    await session.process_frame(_make_raw_bytes())
    await asyncio.sleep(0.01)

    assert session.session_state == SessionState.ACTIVE

    # Cleanup
    await session.close()


async def test_speech_start_during_hold_transitions_to_active():
    """Fala durante HOLD transita HOLD -> ACTIVE."""
    time_ref = [0.0]
    vad = _make_vad_mock()
    grpc_client = _make_grpc_client_mock()
    session, sm, _, _, _ = _make_session_with_sm(
        clock_ref=time_ref,
        vad=vad,
        grpc_client=grpc_client,
    )

    # INIT -> ACTIVE -> SILENCE
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = True
    await session.process_frame(_make_raw_bytes())
    await asyncio.sleep(0.01)

    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_END,
        timestamp_ms=2000,
    )
    vad.is_speaking = False
    await session.process_frame(_make_raw_bytes())
    assert session.session_state == SessionState.SILENCE

    # SILENCE -> HOLD via timeout (30s default)
    time_ref[0] = 31.0
    sm.transition(SessionState.HOLD)
    assert session.session_state == SessionState.HOLD

    # Novo stream handle
    stream_handle2 = _make_stream_handle_mock()
    grpc_client.open_stream = AsyncMock(return_value=stream_handle2)

    # HOLD -> ACTIVE (fala detectada)
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=35000,
    )
    vad.is_speaking = True
    await session.process_frame(_make_raw_bytes())
    await asyncio.sleep(0.01)

    assert session.session_state == SessionState.ACTIVE

    # Cleanup
    await session.close()


# ---------------------------------------------------------------------------
# Tests: Timeouts
# ---------------------------------------------------------------------------


async def test_init_timeout_transitions_to_closed():
    """Timeout de INIT (30s default) transita para CLOSED."""
    time_ref = [0.0]
    session, _sm, _, _, _ = _make_session_with_sm(clock_ref=time_ref)

    assert session.session_state == SessionState.INIT

    # Simular 31s passados
    time_ref[0] = 31.0

    result = session.check_inactivity()
    assert result is True
    assert session.session_state == SessionState.CLOSED


async def test_silence_timeout_transitions_to_hold():
    """Timeout de SILENCE (30s default) transita para HOLD via check_timeout."""
    time_ref = [0.0]
    vad = _make_vad_mock()
    on_event = AsyncMock()
    session, _sm, _, _, _ = _make_session_with_sm(
        clock_ref=time_ref,
        vad=vad,
        on_event=on_event,
    )

    # INIT -> ACTIVE -> SILENCE
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = True
    time_ref[0] = 1.0
    await session.process_frame(_make_raw_bytes())
    await asyncio.sleep(0.01)

    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_END,
        timestamp_ms=2000,
    )
    vad.is_speaking = False
    time_ref[0] = 2.0
    await session.process_frame(_make_raw_bytes())
    assert session.session_state == SessionState.SILENCE

    # Simular 31s no estado SILENCE
    time_ref[0] = 33.0  # 2.0 + 31.0

    result = await session.check_timeout()
    assert result == SessionState.HOLD
    assert session.session_state == SessionState.HOLD


async def test_hold_timeout_transitions_to_closing():
    """Timeout de HOLD (5min default) transita para CLOSING via check_timeout."""
    time_ref = [0.0]
    vad = _make_vad_mock()
    session, sm, _, _, _ = _make_session_with_sm(clock_ref=time_ref, vad=vad)

    # INIT -> ACTIVE -> SILENCE -> HOLD
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = True
    time_ref[0] = 1.0
    await session.process_frame(_make_raw_bytes())
    await asyncio.sleep(0.01)

    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_END,
        timestamp_ms=2000,
    )
    vad.is_speaking = False
    time_ref[0] = 2.0
    await session.process_frame(_make_raw_bytes())

    # Transitar manualmente para HOLD
    time_ref[0] = 3.0
    sm.transition(SessionState.HOLD)
    assert session.session_state == SessionState.HOLD

    # Simular 301s no estado HOLD (>300s default)
    time_ref[0] = 304.0  # 3.0 + 301.0

    result = await session.check_timeout()
    assert result == SessionState.CLOSING


async def test_closing_timeout_transitions_to_closed():
    """Timeout de CLOSING (2s default) transita para CLOSED via check_timeout."""
    time_ref = [0.0]
    session, sm, _, _, _ = _make_session_with_sm(clock_ref=time_ref)

    # INIT -> ACTIVE -> CLOSING
    time_ref[0] = 1.0
    sm.transition(SessionState.ACTIVE)
    time_ref[0] = 2.0
    sm.transition(SessionState.CLOSING)
    assert session.session_state == SessionState.CLOSING

    # Simular 3s no estado CLOSING (>2s default)
    time_ref[0] = 5.0

    result = await session.check_timeout()
    assert result == SessionState.CLOSED
    assert session.is_closed


# ---------------------------------------------------------------------------
# Tests: SessionHoldEvent
# ---------------------------------------------------------------------------


async def test_silence_to_hold_emits_session_hold_event():
    """Transicao SILENCE -> HOLD via check_timeout emite SessionHoldEvent."""
    time_ref = [0.0]
    vad = _make_vad_mock()
    on_event = AsyncMock()
    session, _sm, _, _, _ = _make_session_with_sm(
        clock_ref=time_ref,
        vad=vad,
        on_event=on_event,
    )

    # INIT -> ACTIVE -> SILENCE
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = True
    time_ref[0] = 1.0
    await session.process_frame(_make_raw_bytes())
    await asyncio.sleep(0.01)

    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_END,
        timestamp_ms=2000,
    )
    vad.is_speaking = False
    time_ref[0] = 2.0
    await session.process_frame(_make_raw_bytes())
    assert session.session_state == SessionState.SILENCE

    # Limpar chamadas anteriores do on_event para isolar a verificacao
    on_event.reset_mock()

    # Simular timeout de SILENCE (31s > 30s default)
    time_ref[0] = 33.0

    await session.check_timeout()

    # Verificar que SessionHoldEvent foi emitido
    hold_calls = [
        call for call in on_event.call_args_list if isinstance(call.args[0], SessionHoldEvent)
    ]
    assert len(hold_calls) == 1

    hold_event = hold_calls[0].args[0]
    assert hold_event.type == "session.hold"
    assert hold_event.hold_timeout_ms == 300000  # 5min = 300s = 300000ms


async def test_session_hold_event_has_correct_hold_timeout_ms():
    """SessionHoldEvent contem hold_timeout_ms correto apos session.configure."""
    time_ref = [0.0]
    vad = _make_vad_mock()
    on_event = AsyncMock()
    custom_timeouts = SessionTimeouts(
        init_timeout_s=30.0,
        silence_timeout_s=5.0,  # Curto para teste
        hold_timeout_s=120.0,  # 2min em vez de 5min
        closing_timeout_s=2.0,
    )
    session, _sm, _, _, _ = _make_session_with_sm(
        clock_ref=time_ref,
        vad=vad,
        on_event=on_event,
        timeouts=custom_timeouts,
    )

    # INIT -> ACTIVE -> SILENCE
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = True
    time_ref[0] = 1.0
    await session.process_frame(_make_raw_bytes())
    await asyncio.sleep(0.01)

    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_END,
        timestamp_ms=2000,
    )
    vad.is_speaking = False
    time_ref[0] = 2.0
    await session.process_frame(_make_raw_bytes())

    on_event.reset_mock()

    # Timeout de SILENCE (6s > 5s custom)
    time_ref[0] = 8.0

    await session.check_timeout()

    hold_calls = [
        call for call in on_event.call_args_list if isinstance(call.args[0], SessionHoldEvent)
    ]
    assert len(hold_calls) == 1
    assert hold_calls[0].args[0].hold_timeout_ms == 120000  # 2min


# ---------------------------------------------------------------------------
# Tests: Comportamento por Estado
# ---------------------------------------------------------------------------


async def test_frames_in_hold_not_sent_to_worker():
    """Frames em HOLD nao sao enviados ao worker (economia de GPU)."""
    time_ref = [0.0]
    vad = _make_vad_mock()
    stream_handle = _make_stream_handle_mock()
    grpc_client = _make_grpc_client_mock(stream_handle)
    session, sm, _, _, _ = _make_session_with_sm(
        clock_ref=time_ref,
        vad=vad,
        grpc_client=grpc_client,
    )

    # INIT -> ACTIVE (abre stream)
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = True
    time_ref[0] = 1.0
    await session.process_frame(_make_raw_bytes())
    await asyncio.sleep(0.01)

    # ACTIVE -> SILENCE -> HOLD
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_END,
        timestamp_ms=2000,
    )
    vad.is_speaking = False
    time_ref[0] = 2.0
    await session.process_frame(_make_raw_bytes())

    time_ref[0] = 3.0
    sm.transition(SessionState.HOLD)
    assert session.session_state == SessionState.HOLD

    # Resetar contador de frames enviados
    stream_handle.send_frame.reset_mock()

    # Novo stream para simular que "esta falando" mas em HOLD
    stream_handle2 = _make_stream_handle_mock()
    grpc_client.open_stream = AsyncMock(return_value=stream_handle2)

    # Enviar frames em HOLD (vad.is_speaking=True simulado,
    # mas sem speech_start event - estado permanece HOLD)
    vad.process_frame.return_value = None
    vad.is_speaking = True  # VAD acha que esta falando
    await session.process_frame(_make_raw_bytes())
    await session.process_frame(_make_raw_bytes())

    # Assert: nenhum frame enviado ao worker (estado e HOLD, nao ACTIVE)
    stream_handle2.send_frame.assert_not_called()


async def test_frames_in_init_not_sent_to_worker():
    """Frames em INIT (antes de speech_start) nao sao enviados ao worker."""
    time_ref = [0.0]
    vad = _make_vad_mock(is_speaking=False)
    stream_handle = _make_stream_handle_mock()
    grpc_client = _make_grpc_client_mock(stream_handle)
    session, _sm, _, _, _ = _make_session_with_sm(
        clock_ref=time_ref,
        vad=vad,
        grpc_client=grpc_client,
    )

    assert session.session_state == SessionState.INIT

    # Enviar frames sem speech_start
    vad.process_frame.return_value = None
    await session.process_frame(_make_raw_bytes())
    await session.process_frame(_make_raw_bytes())

    # Assert: nenhum frame enviado
    stream_handle.send_frame.assert_not_called()


async def test_frames_in_closing_rejected():
    """Frames em CLOSING sao ignorados (nao aceita novos frames)."""
    time_ref = [0.0]
    vad = _make_vad_mock()
    preprocessor = _make_preprocessor_mock()
    session, sm, _, _, _ = _make_session_with_sm(
        clock_ref=time_ref,
        vad=vad,
    )
    # Substituir preprocessor para verificar que nao e chamado
    session._preprocessor = preprocessor

    # Transitar para CLOSING
    time_ref[0] = 1.0
    sm.transition(SessionState.ACTIVE)
    time_ref[0] = 2.0
    sm.transition(SessionState.CLOSING)
    assert session.session_state == SessionState.CLOSING

    preprocessor.process_frame.reset_mock()

    # Tentar processar frame
    await session.process_frame(_make_raw_bytes())

    # Assert: preprocessor nao chamado (frame rejeitado no inicio)
    preprocessor.process_frame.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: session.configure / timeouts
# ---------------------------------------------------------------------------


async def test_update_session_timeouts():
    """update_session_timeouts() atualiza timeouts da state machine."""
    time_ref = [0.0]
    session, sm, _, _, _ = _make_session_with_sm(clock_ref=time_ref)

    # Timeouts default: INIT=30s
    assert sm.timeouts.init_timeout_s == 30.0

    # Atualizar timeouts
    new_timeouts = SessionTimeouts(
        init_timeout_s=10.0,
        silence_timeout_s=15.0,
        hold_timeout_s=60.0,
        closing_timeout_s=3.0,
    )
    session.update_session_timeouts(new_timeouts)

    assert sm.timeouts.init_timeout_s == 10.0
    assert sm.timeouts.silence_timeout_s == 15.0
    assert sm.timeouts.hold_timeout_s == 60.0
    assert sm.timeouts.closing_timeout_s == 3.0


async def test_updated_timeout_affects_check_timeout():
    """Timeouts atualizados sao usados imediatamente em check_timeout."""
    time_ref = [0.0]
    session, _sm, _, _, _ = _make_session_with_sm(clock_ref=time_ref)

    # Default INIT timeout: 30s
    time_ref[0] = 15.0  # 15s < 30s
    assert not session.check_inactivity()  # Nao expirou

    # Atualizar para 10s
    new_timeouts = SessionTimeouts(
        init_timeout_s=10.0,
        silence_timeout_s=30.0,
        hold_timeout_s=300.0,
        closing_timeout_s=2.0,
    )
    session.update_session_timeouts(new_timeouts)

    # Agora 15s > 10s, deve expirar
    assert session.check_inactivity()
    assert session.session_state == SessionState.CLOSED


# ---------------------------------------------------------------------------
# Tests: close()
# ---------------------------------------------------------------------------


async def test_close_transitions_to_closing_then_closed():
    """close() transita para CLOSING -> CLOSED via state machine."""
    time_ref = [0.0]
    vad = _make_vad_mock()
    session, _sm, _, _, _ = _make_session_with_sm(clock_ref=time_ref, vad=vad)

    # INIT -> ACTIVE
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = True
    time_ref[0] = 1.0
    await session.process_frame(_make_raw_bytes())
    await asyncio.sleep(0.01)
    assert session.session_state == SessionState.ACTIVE

    # close()
    time_ref[0] = 2.0
    await session.close()

    assert session.session_state == SessionState.CLOSED
    assert session.is_closed


async def test_close_from_init_transitions_to_closed():
    """close() de INIT transita diretamente via CLOSING -> CLOSED."""
    time_ref = [0.0]
    session, _sm, _, _, _ = _make_session_with_sm(clock_ref=time_ref)

    assert session.session_state == SessionState.INIT

    await session.close()

    # INIT nao tem transicao direta para CLOSING na state machine,
    # mas close() tenta CLOSING e se falhar, vai direto para CLOSED
    assert session.is_closed


# ---------------------------------------------------------------------------
# Tests: check_timeout retorna None quando nao ha timeout
# ---------------------------------------------------------------------------


async def test_check_timeout_returns_none_when_active():
    """ACTIVE nao tem timeout, check_timeout retorna None."""
    time_ref = [0.0]
    vad = _make_vad_mock()
    session, _sm, _, _, _ = _make_session_with_sm(clock_ref=time_ref, vad=vad)

    # INIT -> ACTIVE
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = True
    time_ref[0] = 1.0
    await session.process_frame(_make_raw_bytes())
    await asyncio.sleep(0.01)
    assert session.session_state == SessionState.ACTIVE

    # Mesmo apos muito tempo, ACTIVE nao expira
    time_ref[0] = 999.0
    result = await session.check_timeout()
    assert result is None
    assert session.session_state == SessionState.ACTIVE

    # Cleanup
    await session.close()


async def test_check_timeout_returns_none_when_closed():
    """CLOSED retorna None de check_timeout."""
    time_ref = [0.0]
    session, _sm, _, _, _ = _make_session_with_sm(clock_ref=time_ref)

    await session.close()
    assert session.is_closed

    time_ref[0] = 999.0
    result = await session.check_timeout()
    assert result is None
