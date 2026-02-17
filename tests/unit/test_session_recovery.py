"""Testes de recovery de crash do worker (M6-07).

Cobre:
- recover() abre novo stream gRPC
- recover() reenvia dados nao commitados do ring buffer
- recover() restaura segment_id do WAL
- recover() inicia nova receiver task
- Timeout de recovery fecha sessao
- Prevencao de recursao durante recovery
- Integracao: WorkerCrashError em receive_events dispara recovery
- Recovery sem ring buffer (apenas reabre stream)
- Recovery com ring buffer parcialmente commitado
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, Mock

from macaw._types import SessionState
from macaw.exceptions import WorkerCrashError
from macaw.server.models.events import StreamingErrorEvent
from macaw.session.ring_buffer import RingBuffer
from macaw.session.streaming import StreamingSession
from macaw.session.wal import SessionWAL
from tests.helpers import (
    AsyncIterFromList,
    make_preprocessor_mock,
    make_vad_mock,
)

if TYPE_CHECKING:
    from macaw.session.state_machine import SessionStateMachine


def _make_stream_handle_mock(events: list | None = None) -> Mock:
    """Cria mock de StreamHandle."""
    handle = Mock()
    handle.is_closed = False
    handle.session_id = "test-recovery"
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
    mock.process.side_effect = lambda text, **kwargs: text
    return mock


def _create_recovery_session(
    *,
    ring_buffer: RingBuffer | None = None,
    wal: SessionWAL | None = None,
    grpc_client: AsyncMock | None = None,
    on_event: AsyncMock | None = None,
    recovery_timeout_s: float = 1.0,
    state_machine: SessionStateMachine | None = None,
) -> tuple[StreamingSession, AsyncMock, AsyncMock]:
    """Cria sessao com ring buffer e WAL para teste de recovery.

    Returns:
        (session, grpc_client, on_event)
    """
    _grpc_client = grpc_client or _make_grpc_client_mock()
    _on_event = on_event or AsyncMock()

    session = StreamingSession(
        session_id="test-recovery",
        preprocessor=make_preprocessor_mock(),
        vad=make_vad_mock(),
        grpc_client=_grpc_client,
        postprocessor=_make_postprocessor_mock(),
        on_event=_on_event,
        ring_buffer=ring_buffer,
        wal=wal or SessionWAL(),
        recovery_timeout_s=recovery_timeout_s,
        state_machine=state_machine,
    )

    return session, _grpc_client, _on_event


# ---------------------------------------------------------------------------
# Tests: recover() abre novo stream
# ---------------------------------------------------------------------------


async def test_recover_opens_new_stream():
    """recover() abre um novo stream gRPC via grpc_client.open_stream()."""
    # Arrange
    new_handle = _make_stream_handle_mock()
    grpc_client = _make_grpc_client_mock(new_handle)
    session, _, _ = _create_recovery_session(grpc_client=grpc_client)

    # Colocar em ACTIVE (recovery so faz sentido em sessao ativa)
    session._state_machine.transition(SessionState.ACTIVE)

    # Act
    result = await session.recover()

    # Assert
    assert result is True
    grpc_client.open_stream.assert_awaited_once_with("test-recovery")
    assert session._stream_handle is new_handle

    # Cleanup
    await session.close()


async def test_recover_resends_uncommitted_data():
    """recover() reenvia dados nao commitados do ring buffer ao novo worker."""
    # Arrange
    rb = RingBuffer(duration_s=5.0, sample_rate=16000, bytes_per_sample=2)
    new_handle = _make_stream_handle_mock()
    grpc_client = _make_grpc_client_mock(new_handle)
    session, _, _ = _create_recovery_session(
        ring_buffer=rb,
        grpc_client=grpc_client,
    )
    session._state_machine.transition(SessionState.ACTIVE)

    # Escrever dados no ring buffer SEM commitar
    test_data = b"\x01\x02" * 500  # 1000 bytes
    rb.write(test_data)
    assert rb.uncommitted_bytes == 1000

    # Act
    result = await session.recover()

    # Assert
    assert result is True
    new_handle.send_frame.assert_awaited_once()
    sent_data = new_handle.send_frame.call_args.kwargs["pcm_data"]
    assert sent_data == test_data
    assert len(sent_data) == 1000

    # Cleanup
    await session.close()


async def test_recover_with_no_uncommitted_data():
    """recover() com ring buffer vazio ou totalmente commitado nao reenvia dados."""
    # Arrange
    rb = RingBuffer(duration_s=5.0, sample_rate=16000, bytes_per_sample=2)
    new_handle = _make_stream_handle_mock()
    grpc_client = _make_grpc_client_mock(new_handle)
    session, _, _ = _create_recovery_session(
        ring_buffer=rb,
        grpc_client=grpc_client,
    )
    session._state_machine.transition(SessionState.ACTIVE)

    # Escrever e commitar dados (tudo commitado, nada para reenviar)
    test_data = b"\x01\x02" * 100
    rb.write(test_data)
    rb.commit(rb.total_written)
    assert rb.uncommitted_bytes == 0

    # Act
    result = await session.recover()

    # Assert
    assert result is True
    new_handle.send_frame.assert_not_awaited()

    # Cleanup
    await session.close()


async def test_recover_restores_segment_id_from_wal():
    """recover() restaura segment_id = WAL.last_committed_segment_id + 1."""
    # Arrange
    wal = SessionWAL()
    wal.record_checkpoint(segment_id=5, buffer_offset=10000, timestamp_ms=50000)

    session, _, _ = _create_recovery_session(wal=wal)
    session._state_machine.transition(SessionState.ACTIVE)

    # segment_id antes do recovery
    session._segment_id = 99  # Valor arbitrario pre-recovery

    # Act
    result = await session.recover()

    # Assert
    assert result is True
    assert session.segment_id == 6  # WAL.last_committed_segment_id (5) + 1

    # Cleanup
    await session.close()


async def test_recover_starts_new_receiver_task():
    """recover() inicia uma nova receiver task para consumir eventos do worker."""
    # Arrange
    session, _, _ = _create_recovery_session()
    session._state_machine.transition(SessionState.ACTIVE)

    assert session._receiver_task is None

    # Act
    result = await session.recover()

    # Assert
    assert result is True
    assert session._receiver_task is not None
    assert not session._receiver_task.done()

    # Cleanup
    await session.close()


async def test_recover_timeout_closes_session():
    """Se open_stream falha com timeout, sessao transita para CLOSED."""
    # Arrange
    grpc_client = AsyncMock()

    async def slow_open_stream(_session_id: str) -> Mock:
        await asyncio.sleep(10.0)  # Muito lento
        return _make_stream_handle_mock()

    grpc_client.open_stream = slow_open_stream

    session, _, _ = _create_recovery_session(
        grpc_client=grpc_client,
        recovery_timeout_s=0.1,  # Timeout rapido
    )
    session._state_machine.transition(SessionState.ACTIVE)

    # Act
    result = await session.recover()

    # Assert
    assert result is False
    assert session.session_state == SessionState.CLOSED


async def test_recover_emits_recoverable_error():
    """WorkerCrashError no receiver emite erro recoverable com resume_segment_id."""
    # Arrange
    crash_handle = _make_stream_handle_mock()
    crash_handle.receive_events.return_value = AsyncIterFromList(
        [WorkerCrashError("test-recovery")]
    )

    # open_stream e chamado pelo recover() -- deve retornar handle valido
    recovery_handle = _make_stream_handle_mock()
    grpc_client = AsyncMock()
    grpc_client.open_stream = AsyncMock(return_value=recovery_handle)
    on_event = AsyncMock()

    session, _, _ = _create_recovery_session(
        grpc_client=grpc_client,
        on_event=on_event,
    )
    session._state_machine.transition(SessionState.ACTIVE)

    # Abrir stream inicial (que vai crashar) -- atribuicao manual
    session._stream_handle = crash_handle
    session._receiver_task = asyncio.create_task(
        session._receive_worker_events(),
    )

    # Aguardar receiver processar o crash e disparar recovery
    await asyncio.sleep(0.1)

    # Assert: evento de erro recoverable emitido
    error_calls = [
        call for call in on_event.call_args_list if isinstance(call.args[0], StreamingErrorEvent)
    ]
    assert len(error_calls) >= 1

    recoverable_errors = [call for call in error_calls if call.args[0].recoverable is True]
    assert len(recoverable_errors) >= 1
    assert recoverable_errors[0].args[0].code == "worker_crash"
    assert "recovery" in recoverable_errors[0].args[0].message.lower()

    # Cleanup
    await session.close()


async def test_recover_prevents_recursion():
    """Se _recovering ja e True, recover() retorna False sem tentar novamente."""
    # Arrange
    session, _, _ = _create_recovery_session()
    session._state_machine.transition(SessionState.ACTIVE)

    # Simular recovery em andamento
    session._recovery._recovering = True

    # Act
    result = await session.recover()

    # Assert
    assert result is False
    # Flag permanece True (nao foi modificada)
    assert session._recovery._recovering is True

    # Reset para cleanup
    session._recovery._recovering = False
    await session.close()


async def test_recover_with_ring_buffer_data_partially_committed():
    """Recovery com ring buffer parcialmente commitado reenvia apenas uncommitted."""
    # Arrange
    rb = RingBuffer(duration_s=5.0, sample_rate=16000, bytes_per_sample=2)
    new_handle = _make_stream_handle_mock()
    grpc_client = _make_grpc_client_mock(new_handle)
    session, _, _ = _create_recovery_session(
        ring_buffer=rb,
        grpc_client=grpc_client,
    )
    session._state_machine.transition(SessionState.ACTIVE)

    # Escrever dados e commitar metade
    committed_data = b"\x01\x00" * 500  # 1000 bytes
    uncommitted_data = b"\x02\x00" * 300  # 600 bytes
    rb.write(committed_data)
    rb.commit(rb.total_written)  # Commitar os primeiros 1000 bytes
    rb.write(uncommitted_data)

    assert rb.uncommitted_bytes == 600

    # Act
    result = await session.recover()

    # Assert
    assert result is True
    new_handle.send_frame.assert_awaited_once()
    sent_data = new_handle.send_frame.call_args.kwargs["pcm_data"]
    assert len(sent_data) == 600
    assert sent_data == uncommitted_data

    # Cleanup
    await session.close()


async def test_recover_without_ring_buffer():
    """Recovery sem ring buffer apenas reabre stream (sem reenvio de dados)."""
    # Arrange
    new_handle = _make_stream_handle_mock()
    grpc_client = _make_grpc_client_mock(new_handle)
    session, _, _ = _create_recovery_session(
        ring_buffer=None,  # Sem ring buffer
        grpc_client=grpc_client,
    )
    session._state_machine.transition(SessionState.ACTIVE)

    # Act
    result = await session.recover()

    # Assert
    assert result is True
    grpc_client.open_stream.assert_awaited_once()
    new_handle.send_frame.assert_not_awaited()

    # Cleanup
    await session.close()


async def test_recover_resets_recovering_flag():
    """Flag _recovering e resetada para False apos recovery (sucesso ou falha)."""
    # Arrange: recovery bem-sucedido
    session, _, _ = _create_recovery_session()
    session._state_machine.transition(SessionState.ACTIVE)

    assert session._recovery._recovering is False

    result = await session.recover()
    assert result is True
    assert session._recovery._recovering is False

    # Cleanup
    await session.close()


async def test_recover_resets_recovering_flag_on_failure():
    """Flag _recovering e resetada para False mesmo quando recovery falha."""
    # Arrange: recovery que falha (timeout)
    grpc_client = AsyncMock()

    async def failing_open_stream(_session_id: str) -> Mock:
        raise WorkerCrashError("test-recovery")

    grpc_client.open_stream = failing_open_stream

    session, _, _ = _create_recovery_session(grpc_client=grpc_client)
    session._state_machine.transition(SessionState.ACTIVE)

    # Act
    result = await session.recover()

    # Assert
    assert result is False
    assert session._recovery._recovering is False


async def test_receiver_crash_triggers_recovery():
    """WorkerCrashError durante receive_events dispara recover() automaticamente."""
    # Arrange
    crash_handle = _make_stream_handle_mock()
    crash_handle.receive_events.return_value = AsyncIterFromList(
        [WorkerCrashError("test-recovery")]
    )

    recovery_handle = _make_stream_handle_mock()

    # open_stream e chamado pelo recover(), nao pela criacao da sessao.
    # A primeira chamada vem do recover() e deve retornar recovery_handle.
    grpc_client = AsyncMock()
    grpc_client.open_stream = AsyncMock(return_value=recovery_handle)
    on_event = AsyncMock()

    session, _, _ = _create_recovery_session(
        grpc_client=grpc_client,
        on_event=on_event,
    )
    session._state_machine.transition(SessionState.ACTIVE)

    # Abrir stream inicial (que vai crashar) -- atribuicao manual
    session._stream_handle = crash_handle
    session._receiver_task = asyncio.create_task(
        session._receive_worker_events(),
    )

    # Aguardar recovery acontecer
    await asyncio.sleep(0.2)

    # Assert: recovery foi chamado (novo stream aberto)
    grpc_client.open_stream.assert_awaited_once_with("test-recovery")
    # Session nao esta fechada (recovery bem-sucedido)
    assert session.session_state != SessionState.CLOSED

    # Cleanup
    await session.close()


async def test_recover_resets_hot_words_sent_flag():
    """recover() reseta _hot_words_sent_for_segment para reenviar hot words."""
    # Arrange
    session, _, _ = _create_recovery_session()
    session._state_machine.transition(SessionState.ACTIVE)
    session._hot_words_sent_for_segment = True

    # Act
    result = await session.recover()

    # Assert
    assert result is True
    assert session._hot_words_sent_for_segment is False

    # Cleanup
    await session.close()


async def test_recover_resend_failure_returns_false():
    """Se reenvio de dados uncommitted falha, recover() retorna False."""
    # Arrange
    rb = RingBuffer(duration_s=5.0, sample_rate=16000, bytes_per_sample=2)
    rb.write(b"\x01\x02" * 500)  # Dados uncommitted

    new_handle = _make_stream_handle_mock()
    new_handle.send_frame = AsyncMock(side_effect=WorkerCrashError("test-recovery"))
    grpc_client = _make_grpc_client_mock(new_handle)

    session, _, _ = _create_recovery_session(
        ring_buffer=rb,
        grpc_client=grpc_client,
    )
    session._state_machine.transition(SessionState.ACTIVE)

    # Act
    result = await session.recover()

    # Assert
    assert result is False
    assert session._stream_handle is None


async def test_recover_open_stream_crash_closes_session():
    """Se open_stream levanta WorkerCrashError, sessao transita para CLOSED."""
    # Arrange
    grpc_client = AsyncMock()
    grpc_client.open_stream = AsyncMock(
        side_effect=WorkerCrashError("test-recovery"),
    )

    session, _, _ = _create_recovery_session(grpc_client=grpc_client)
    session._state_machine.transition(SessionState.ACTIVE)

    # Act
    result = await session.recover()

    # Assert
    assert result is False
    assert session.session_state == SessionState.CLOSED


async def test_recovery_failed_emits_irrecoverable_error():
    """Quando recovery falha, erro irrecuperavel e emitido."""
    # Arrange
    crash_handle = _make_stream_handle_mock()
    crash_handle.receive_events.return_value = AsyncIterFromList(
        [WorkerCrashError("test-recovery")]
    )

    # open_stream e chamado pelo recover() -- a primeira chamada
    # vem do recover() e deve falhar para testar o caminho de falha.
    grpc_client = AsyncMock()
    grpc_client.open_stream = AsyncMock(
        side_effect=WorkerCrashError("test-recovery"),
    )
    on_event = AsyncMock()

    session, _, _ = _create_recovery_session(
        grpc_client=grpc_client,
        on_event=on_event,
    )
    session._state_machine.transition(SessionState.ACTIVE)

    # Abrir stream inicial (que vai crashar) -- atribuicao manual
    session._stream_handle = crash_handle
    session._receiver_task = asyncio.create_task(
        session._receive_worker_events(),
    )

    # Aguardar recovery falhar
    await asyncio.sleep(0.2)

    # Assert: erro irrecuperavel emitido
    error_calls = [
        call for call in on_event.call_args_list if isinstance(call.args[0], StreamingErrorEvent)
    ]
    irrecoverable_errors = [call for call in error_calls if call.args[0].recoverable is False]
    assert len(irrecoverable_errors) >= 1
    assert "Recovery failed" in irrecoverable_errors[0].args[0].message
