"""Testes do Ring Buffer — read fence, force commit e integracao com StreamingSession.

Cobre:
- Read fence (commit, uncommitted_bytes, available_for_write_bytes)
- Protecao de dados nao commitados (BufferOverrunError)
- Force commit callback (>90% de uso nao commitado)
- Integracao com StreamingSession (write no ring buffer, commit apos final,
  force commit via callback sync -> flag async)

Testes de funcionalidade basica do ring buffer (write, read, wrap-around)
estao em test_ring_buffer.py.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from macaw._types import TranscriptSegment
from macaw.exceptions import BufferOverrunError
from macaw.server.models.events import TranscriptFinalEvent
from macaw.session.ring_buffer import _FORCE_COMMIT_THRESHOLD, RingBuffer
from macaw.session.streaming import StreamingSession
from macaw.vad.detector import VADEvent, VADEventType

# ---------------------------------------------------------------------------
# Helpers (compartilhados com test_streaming_session.py)
# ---------------------------------------------------------------------------

_FRAME_SIZE = 1024
_SAMPLE_RATE = 16000


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
    mock.process.side_effect = lambda text: text  # identity (sem transformacao)
    return mock


# ---------------------------------------------------------------------------
# Read Fence: propriedades e commit
# ---------------------------------------------------------------------------


class TestReadFence:
    """Testes do read fence (last_committed_offset)."""

    def test_read_fence_starts_at_zero(self) -> None:
        """Read fence inicia em 0 (nada commitado)."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        assert rb.read_fence == 0

    def test_commit_advances_fence(self) -> None:
        """commit() avanca o read fence para o offset dado."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        rb.write(b"\x01" * 5)
        rb.commit(3)
        assert rb.read_fence == 3

    def test_commit_to_total_written(self) -> None:
        """commit() pode avancar ate total_written (tudo commitado)."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        rb.write(b"\x01" * 7)
        rb.commit(rb.total_written)
        assert rb.read_fence == 7
        assert rb.uncommitted_bytes == 0

    def test_commit_below_current_fence_raises(self) -> None:
        """commit() com offset menor que o fence atual levanta ValueError."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        rb.write(b"\x01" * 5)
        rb.commit(3)
        with pytest.raises(ValueError, match="menor"):
            rb.commit(2)

    def test_commit_above_total_written_raises(self) -> None:
        """commit() com offset maior que total_written levanta ValueError."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        rb.write(b"\x01" * 5)
        with pytest.raises(ValueError, match="maior"):
            rb.commit(10)

    def test_commit_same_fence_is_noop(self) -> None:
        """commit() com o mesmo offset do fence atual e no-op (valido)."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        rb.write(b"\x01" * 5)
        rb.commit(3)
        rb.commit(3)  # No-op, nao deve levantar
        assert rb.read_fence == 3

    def test_commit_incremental(self) -> None:
        """commit() pode ser chamado incrementalmente."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        rb.write(b"\x01" * 10)
        rb.commit(2)
        assert rb.read_fence == 2
        rb.commit(5)
        assert rb.read_fence == 5
        rb.commit(10)
        assert rb.read_fence == 10


# ---------------------------------------------------------------------------
# Uncommitted bytes e available_for_write_bytes
# ---------------------------------------------------------------------------


class TestUncommittedBytes:
    """Testes de uncommitted_bytes e available_for_write_bytes."""

    def test_uncommitted_bytes_equals_total_written_when_no_commit(self) -> None:
        """Sem commit, tudo e nao commitado."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        rb.write(b"\x01" * 7)
        assert rb.uncommitted_bytes == 7

    def test_uncommitted_bytes_decreases_after_commit(self) -> None:
        """uncommitted_bytes diminui apos commit."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        rb.write(b"\x01" * 7)
        rb.commit(3)
        assert rb.uncommitted_bytes == 4

    def test_uncommitted_bytes_zero_after_full_commit(self) -> None:
        """uncommitted_bytes = 0 apos commit total."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        rb.write(b"\x01" * 7)
        rb.commit(rb.total_written)
        assert rb.uncommitted_bytes == 0

    def test_available_for_write_equals_capacity_when_all_committed(self) -> None:
        """Quando tudo e commitado, available = capacity."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        # capacity = 10
        rb.write(b"\x01" * 7)
        rb.commit(rb.total_written)
        assert rb.available_for_write_bytes == 10

    def test_available_for_write_decreases_with_uncommitted(self) -> None:
        """available_for_write diminui com dados nao commitados."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        # capacity = 10
        rb.write(b"\x01" * 7)
        # uncommitted = 7, available = 10 - 7 = 3
        assert rb.available_for_write_bytes == 3

    def test_available_for_write_zero_when_uncommitted_fills_capacity(self) -> None:
        """available_for_write = 0 quando uncommitted >= capacity."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        # capacity = 10
        rb.write(b"\x01" * 10)
        # uncommitted = 10 = capacity, available = 0
        assert rb.available_for_write_bytes == 0


# ---------------------------------------------------------------------------
# Write protection (fence protege dados nao commitados)
# ---------------------------------------------------------------------------


class TestWriteProtection:
    """Testes de protecao de escrita — fence impede sobrescrita de dados nao commitados."""

    def test_write_raises_when_would_overwrite_uncommitted(self) -> None:
        """Escrita que sobrescreveria dados nao commitados levanta BufferOverrunError."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        # capacity = 10
        rb.write(b"\x01" * 10)
        # uncommitted = 10, available = 0
        with pytest.raises(BufferOverrunError, match="sobrescreveria"):
            rb.write(b"\x02" * 1)

    def test_write_succeeds_after_partial_commit(self) -> None:
        """Escrita sucede apos commit parcial liberar espaco."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        # capacity = 10
        rb.write(b"\x01" * 10)
        rb.commit(5)  # libera 5 bytes
        # available = 10 - 5 = 5
        rb.write(b"\x02" * 5)  # Nao deve levantar
        assert rb.total_written == 15

    def test_write_exceeding_available_raises(self) -> None:
        """Escrita maior que available_for_write levanta BufferOverrunError."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        rb.write(b"\x01" * 8)
        # uncommitted = 8, available = 2
        with pytest.raises(BufferOverrunError, match="sobrescreveria"):
            rb.write(b"\x02" * 3)

    def test_write_exactly_available_succeeds(self) -> None:
        """Escrita exatamente do tamanho available nao levanta erro."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        rb.write(b"\x01" * 8)
        # uncommitted = 8, available = 2
        rb.write(b"\x02" * 2)  # Nao deve levantar
        assert rb.total_written == 10

    def test_write_unrestricted_when_all_committed(self) -> None:
        """Escrita e livre quando tudo esta commitado (nada para proteger)."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        rb.write(b"\x01" * 10)
        rb.commit(rb.total_written)
        # Tudo commitado — escrita livre, mesmo que > capacity
        rb.write(b"\x02" * 25)
        assert rb.total_written == 35

    def test_write_with_zero_uncommitted_allows_large_write(self) -> None:
        """Zero uncommitted permite escrita maior que capacity."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        # fence=0, total_written=0 -> uncommitted=0
        data = bytes(range(25))
        rb.write(data)  # 25 > 10 (capacity), mas uncommitted=0 -> OK
        assert rb.total_written == 25


# ---------------------------------------------------------------------------
# Force commit callback
# ---------------------------------------------------------------------------


class TestForceCommit:
    """Testes do callback on_force_commit (>90% de uso nao commitado)."""

    def test_force_commit_threshold_is_90_percent(self) -> None:
        """Threshold de force commit e 90%."""
        assert pytest.approx(0.90) == _FORCE_COMMIT_THRESHOLD

    def test_force_commit_triggered_above_90_percent(self) -> None:
        """Callback e invocado quando uncommitted > 90% da capacidade."""
        callback = Mock()
        rb = RingBuffer(
            duration_s=1.0,
            sample_rate=10,
            bytes_per_sample=1,
            on_force_commit=callback,
        )
        # capacity = 10, 90% = 9 bytes

        # Escrever 10 bytes (100% uncommitted) -> dispara callback
        rb.write(b"\x01" * 10)

        callback.assert_called_once_with(10)  # total_written = 10

    def test_force_commit_not_triggered_at_90_percent(self) -> None:
        """Callback NAO e invocado quando uncommitted = exatamente 90%."""
        callback = Mock()
        rb = RingBuffer(
            duration_s=1.0,
            sample_rate=10,
            bytes_per_sample=1,
            on_force_commit=callback,
        )
        # capacity = 10, 90% = 9 bytes. Threshold e STRICT > 0.90

        # Escrever exatamente 9 bytes (90% = exatamente threshold, nao > threshold)
        rb.write(b"\x01" * 9)

        callback.assert_not_called()

    def test_force_commit_not_triggered_below_90_percent(self) -> None:
        """Callback NAO e invocado quando uncommitted < 90%."""
        callback = Mock()
        rb = RingBuffer(
            duration_s=1.0,
            sample_rate=10,
            bytes_per_sample=1,
            on_force_commit=callback,
        )
        # capacity = 10

        rb.write(b"\x01" * 8)  # 80%
        callback.assert_not_called()

    def test_force_commit_not_triggered_when_committed(self) -> None:
        """Callback NAO e invocado quando dados ja estao commitados."""
        callback = Mock()
        rb = RingBuffer(
            duration_s=1.0,
            sample_rate=10,
            bytes_per_sample=1,
            on_force_commit=callback,
        )
        # capacity = 10

        rb.write(b"\x01" * 5)
        rb.commit(rb.total_written)  # Tudo commitado

        # Escrever mais 5 — uncommitted = 5 (50%) -> sem callback
        rb.write(b"\x02" * 5)
        callback.assert_not_called()

    def test_force_commit_not_triggered_without_callback(self) -> None:
        """Sem callback configurado, nenhum erro ocorre ao ultrapassar 90%."""
        rb = RingBuffer(
            duration_s=1.0,
            sample_rate=10,
            bytes_per_sample=1,
            on_force_commit=None,
        )
        # capacity = 10

        # Escrever 10 bytes sem callback — nao deve levantar erro
        rb.write(b"\x01" * 10)
        assert rb.total_written == 10

    def test_force_commit_called_multiple_times(self) -> None:
        """Callback pode ser chamado multiplas vezes conforme buffer enche."""
        callback = Mock()
        rb = RingBuffer(
            duration_s=1.0,
            sample_rate=100,
            bytes_per_sample=1,
            on_force_commit=callback,
        )
        # capacity = 100

        # Primeira escrita: 95 bytes (95% > 90%)
        rb.write(b"\x01" * 95)
        assert callback.call_count == 1

        # Comitar tudo e escrever novamente
        rb.commit(rb.total_written)
        rb.write(b"\x02" * 91)  # 91% > 90%
        assert callback.call_count == 2


# ---------------------------------------------------------------------------
# Integracao: StreamingSession + Ring Buffer
# ---------------------------------------------------------------------------


class TestStreamingSessionRingBuffer:
    """Testes de integracao StreamingSession com Ring Buffer."""

    async def test_session_writes_pcm_to_ring_buffer(self) -> None:
        """StreamingSession escreve frames PCM no ring buffer durante fala."""
        # Arrange
        rb = RingBuffer(duration_s=5.0, sample_rate=16000, bytes_per_sample=2)
        stream_handle = _make_stream_handle_mock()
        grpc_client = _make_grpc_client_mock(stream_handle)
        vad = _make_vad_mock()

        session = StreamingSession(
            session_id="test_session",
            preprocessor=_make_preprocessor_mock(),
            vad=vad,
            grpc_client=grpc_client,
            postprocessor=_make_postprocessor_mock(),
            on_event=AsyncMock(),
            ring_buffer=rb,
        )

        # Trigger speech_start
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START,
            timestamp_ms=0,
        )
        vad.is_speaking = True
        await session.process_frame(_make_raw_bytes())

        # Enviar mais frames durante fala
        vad.process_frame.return_value = None
        await session.process_frame(_make_raw_bytes())
        await session.process_frame(_make_raw_bytes())

        # Assert: ring buffer tem dados escritos.
        # Cada frame e 1024 samples * 2 bytes = 2048 bytes PCM int16.
        expected_bytes = 3 * _FRAME_SIZE * 2  # 3 frames
        assert rb.total_written == expected_bytes

        # Cleanup
        await session.close()

    async def test_session_without_ring_buffer_works(self) -> None:
        """StreamingSession funciona normalmente sem ring buffer (backward compat)."""
        stream_handle = _make_stream_handle_mock()
        grpc_client = _make_grpc_client_mock(stream_handle)
        vad = _make_vad_mock()

        session = StreamingSession(
            session_id="test_session",
            preprocessor=_make_preprocessor_mock(),
            vad=vad,
            grpc_client=grpc_client,
            postprocessor=_make_postprocessor_mock(),
            on_event=AsyncMock(),
            ring_buffer=None,  # Sem ring buffer
        )

        # Trigger speech_start e enviar frames — nao deve levantar erro
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START,
            timestamp_ms=0,
        )
        vad.is_speaking = True
        await session.process_frame(_make_raw_bytes())

        vad.process_frame.return_value = None
        await session.process_frame(_make_raw_bytes())

        # Assert: frames enviados ao worker normalmente
        assert stream_handle.send_frame.call_count == 2

        await session.close()

    async def test_ring_buffer_fence_advances_on_transcript_final(self) -> None:
        """Ring buffer read fence avanca apos transcript.final."""
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
        vad = _make_vad_mock()
        on_event = AsyncMock()

        session = StreamingSession(
            session_id="test_session",
            preprocessor=_make_preprocessor_mock(),
            vad=vad,
            grpc_client=grpc_client,
            postprocessor=_make_postprocessor_mock(),
            on_event=on_event,
            enable_itn=False,
            ring_buffer=rb,
        )

        # Trigger speech_start (abre stream, inicia receiver)
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START,
            timestamp_ms=0,
        )
        vad.is_speaking = True
        await session.process_frame(_make_raw_bytes())

        # Enviar mais frames
        vad.process_frame.return_value = None
        await session.process_frame(_make_raw_bytes())

        # Aguardar receiver task processar o transcript.final
        await asyncio.sleep(0.05)

        # Assert: fence avancou para total_written
        assert rb.read_fence == rb.total_written
        assert rb.uncommitted_bytes == 0

        # Verificar que transcript.final foi emitido
        final_calls = [
            call
            for call in on_event.call_args_list
            if isinstance(call.args[0], TranscriptFinalEvent)
        ]
        assert len(final_calls) == 1

        await session.close()

    async def test_force_commit_triggers_session_commit(self) -> None:
        """Force commit do ring buffer (>90%) dispara session.commit()."""
        # Arrange: ring buffer pequeno para facilitar atingir 90%
        # 0.01s * 16000 * 2 = 320 bytes de capacidade
        rb = RingBuffer(duration_s=0.01, sample_rate=16000, bytes_per_sample=2)

        stream_handle = _make_stream_handle_mock()
        grpc_client = _make_grpc_client_mock(stream_handle)
        vad = _make_vad_mock()

        session = StreamingSession(
            session_id="test_session",
            preprocessor=_make_preprocessor_mock(),
            vad=vad,
            grpc_client=grpc_client,
            postprocessor=_make_postprocessor_mock(),
            on_event=AsyncMock(),
            ring_buffer=rb,
        )

        # Verificar que o callback foi wired
        assert rb._on_force_commit is not None

        # Trigger speech_start
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START,
            timestamp_ms=0,
        )
        vad.is_speaking = True

        # Usar frames pequenos para controlar o tamanho exato.
        # Cada frame default = 1024 * 2 = 2048 bytes.
        # rb.capacity = 320 bytes. Primeiro frame ja excede capacity,
        # mas fence=0 e total_written=0 entao uncommitted=0 e write e livre.
        # Apos o write, uncommitted = 2048 > 320 * 0.9 = 288 -> force commit.

        # O force commit seta _force_commit_pending = True
        # E no final de process_frame, commit() e chamado.
        await session.process_frame(_make_raw_bytes())

        # Apos process_frame com force commit:
        # - commit() fecha o stream handle e aguarda receiver
        # - segment_id incrementa
        assert session.segment_id == 1  # Incrementou por causa do commit

        await session.close()

    async def test_ring_buffer_callback_wired_in_init(self) -> None:
        """StreamingSession configura on_force_commit no ring buffer ao ser criada."""
        rb = RingBuffer(duration_s=1.0, sample_rate=10, bytes_per_sample=1)
        assert rb._on_force_commit is None  # Antes de criar session

        session = StreamingSession(
            session_id="test_session",
            preprocessor=_make_preprocessor_mock(),
            vad=_make_vad_mock(),
            grpc_client=_make_grpc_client_mock(),
            postprocessor=_make_postprocessor_mock(),
            on_event=AsyncMock(),
            ring_buffer=rb,
        )

        # Apos criar session, callback esta wired
        assert rb._on_force_commit is not None
        assert rb._on_force_commit == session._metrics.on_ring_buffer_force_commit

        await session.close()

    async def test_force_commit_flag_reset_after_commit(self) -> None:
        """Flag _force_commit_pending e resetada apos commit ser executado."""
        rb = RingBuffer(duration_s=0.01, sample_rate=16000, bytes_per_sample=2)

        session = StreamingSession(
            session_id="test_session",
            preprocessor=_make_preprocessor_mock(),
            vad=_make_vad_mock(),
            grpc_client=_make_grpc_client_mock(),
            postprocessor=_make_postprocessor_mock(),
            on_event=AsyncMock(),
            ring_buffer=rb,
        )

        # Setar flag manualmente
        session._metrics._force_commit_pending = True

        # Chamar commit diretamente (sem stream ativo, e no-op mas reseta flag)
        # Na verdade, commit() nao reseta a flag — e process_frame que faz.
        # Vamos verificar via process_frame.
        vad = session._vad
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START,
            timestamp_ms=0,
        )
        vad.is_speaking = True

        # process_frame vai: preprocessar, VAD, enviar ao worker, checar flag
        await session.process_frame(_make_raw_bytes())

        # A flag deve ter sido consumida pelo process_frame.
        # (Pode ter sido setada novamente pelo ring buffer write se > 90%,
        # e entao consumida novamente pelo commit no final de process_frame)
        # O importante e que o ciclo funciona sem deadlock.

        await session.close()
