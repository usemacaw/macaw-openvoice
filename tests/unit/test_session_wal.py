"""Testes do WAL In-Memory (Write-Ahead Log) para recovery de sessao.

Cobre:
- Inicializacao com valores zero
- record_checkpoint atualiza todos os campos atomicamente
- Properties de acesso individual (segment_id, buffer_offset, timestamp_ms)
- Multiplos checkpoints: cada sobrescreve o anterior
- WALCheckpoint e frozen (imutavel)
- Integracao: StreamingSession registra checkpoint apos transcript.final
"""

from __future__ import annotations

import asyncio
import dataclasses
from unittest.mock import AsyncMock, Mock

import pytest

from macaw._types import TranscriptSegment
from macaw.server.models.events import TranscriptFinalEvent
from macaw.session.ring_buffer import RingBuffer
from macaw.session.streaming import StreamingSession
from macaw.session.wal import SessionWAL, WALCheckpoint
from macaw.vad.detector import VADEvent, VADEventType
from tests.helpers import AsyncIterFromList, make_preprocessor_mock, make_raw_bytes, make_vad_mock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stream_handle_mock(events: list | None = None) -> Mock:
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
    mock.process.side_effect = lambda text, **kwargs: text
    return mock


# ---------------------------------------------------------------------------
# SessionWAL: inicializacao e propriedades
# ---------------------------------------------------------------------------


class TestSessionWALInit:
    """Testes de inicializacao e estado inicial do WAL."""

    def test_wal_initializes_with_zero_values(self) -> None:
        """WAL inicia com segment_id=0, buffer_offset=0, timestamp_ms=0."""
        wal = SessionWAL()
        assert wal.last_committed_segment_id == 0
        assert wal.last_committed_buffer_offset == 0
        assert wal.last_committed_timestamp_ms == 0

    def test_wal_checkpoint_initializes_correctly(self) -> None:
        """Property checkpoint retorna WALCheckpoint com valores iniciais."""
        wal = SessionWAL()
        cp = wal.checkpoint
        assert isinstance(cp, WALCheckpoint)
        assert cp.segment_id == 0
        assert cp.buffer_offset == 0
        assert cp.timestamp_ms == 0


# ---------------------------------------------------------------------------
# SessionWAL: record_checkpoint e acesso
# ---------------------------------------------------------------------------


class TestSessionWALRecordCheckpoint:
    """Testes de record_checkpoint e leitura de valores."""

    def test_record_checkpoint_updates_all_fields(self) -> None:
        """record_checkpoint atualiza segment_id, buffer_offset e timestamp_ms."""
        wal = SessionWAL()
        wal.record_checkpoint(segment_id=3, buffer_offset=4096, timestamp_ms=12345)

        assert wal.last_committed_segment_id == 3
        assert wal.last_committed_buffer_offset == 4096
        assert wal.last_committed_timestamp_ms == 12345

    def test_last_committed_segment_id_after_checkpoint(self) -> None:
        """last_committed_segment_id retorna valor correto apos checkpoint."""
        wal = SessionWAL()
        wal.record_checkpoint(segment_id=7, buffer_offset=0, timestamp_ms=0)
        assert wal.last_committed_segment_id == 7

    def test_last_committed_buffer_offset_after_checkpoint(self) -> None:
        """last_committed_buffer_offset retorna valor correto apos checkpoint."""
        wal = SessionWAL()
        wal.record_checkpoint(segment_id=0, buffer_offset=99999, timestamp_ms=0)
        assert wal.last_committed_buffer_offset == 99999

    def test_last_committed_timestamp_ms_after_checkpoint(self) -> None:
        """last_committed_timestamp_ms retorna valor correto apos checkpoint."""
        wal = SessionWAL()
        wal.record_checkpoint(segment_id=0, buffer_offset=0, timestamp_ms=987654)
        assert wal.last_committed_timestamp_ms == 987654

    def test_multiple_checkpoints_overwrite_previous(self) -> None:
        """Cada checkpoint sobrescreve o anterior (nao e append-only)."""
        wal = SessionWAL()

        wal.record_checkpoint(segment_id=1, buffer_offset=1000, timestamp_ms=100)
        assert wal.last_committed_segment_id == 1

        wal.record_checkpoint(segment_id=2, buffer_offset=2000, timestamp_ms=200)
        assert wal.last_committed_segment_id == 2
        assert wal.last_committed_buffer_offset == 2000
        assert wal.last_committed_timestamp_ms == 200

        # Checkpoint anterior nao e acessivel
        wal.record_checkpoint(segment_id=5, buffer_offset=8000, timestamp_ms=500)
        assert wal.last_committed_segment_id == 5
        assert wal.last_committed_buffer_offset == 8000
        assert wal.last_committed_timestamp_ms == 500

    def test_checkpoint_property_returns_current_checkpoint(self) -> None:
        """Property checkpoint retorna o WALCheckpoint mais recente."""
        wal = SessionWAL()
        wal.record_checkpoint(segment_id=4, buffer_offset=5000, timestamp_ms=300)
        cp = wal.checkpoint
        assert cp.segment_id == 4
        assert cp.buffer_offset == 5000
        assert cp.timestamp_ms == 300


# ---------------------------------------------------------------------------
# WALCheckpoint: imutabilidade
# ---------------------------------------------------------------------------


class TestWALCheckpointImmutability:
    """Testes de imutabilidade do WALCheckpoint."""

    def test_wal_checkpoint_is_frozen(self) -> None:
        """WALCheckpoint e frozen (imutavel) — atribuicao levanta FrozenInstanceError."""
        cp = WALCheckpoint(segment_id=1, buffer_offset=100, timestamp_ms=50)
        with pytest.raises(dataclasses.FrozenInstanceError):
            cp.segment_id = 99  # type: ignore[misc]

    def test_wal_checkpoint_uses_slots(self) -> None:
        """WALCheckpoint usa __slots__ para economia de memoria."""
        cp = WALCheckpoint(segment_id=1, buffer_offset=100, timestamp_ms=50)
        assert hasattr(cp, "__slots__")
        # frozen + slots: atribuicao de atributo inexistente levanta erro
        # (FrozenInstanceError intercepta antes do AttributeError de slots)
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
            cp.extra_field = "nao deve funcionar"  # type: ignore[attr-defined]

    def test_wal_checkpoint_equality(self) -> None:
        """Dois WALCheckpoints com mesmos valores sao iguais (dataclass)."""
        cp1 = WALCheckpoint(segment_id=1, buffer_offset=100, timestamp_ms=50)
        cp2 = WALCheckpoint(segment_id=1, buffer_offset=100, timestamp_ms=50)
        assert cp1 == cp2

    def test_wal_checkpoint_inequality(self) -> None:
        """WALCheckpoints com valores diferentes nao sao iguais."""
        cp1 = WALCheckpoint(segment_id=1, buffer_offset=100, timestamp_ms=50)
        cp2 = WALCheckpoint(segment_id=2, buffer_offset=100, timestamp_ms=50)
        assert cp1 != cp2


# ---------------------------------------------------------------------------
# Integracao: StreamingSession + WAL
# ---------------------------------------------------------------------------


class TestStreamingSessionWAL:
    """Testes de integracao StreamingSession com SessionWAL."""

    async def test_session_creates_default_wal(self) -> None:
        """StreamingSession cria WAL default se nenhum fornecido."""
        session = StreamingSession(
            session_id="test",
            preprocessor=make_preprocessor_mock(),
            vad=make_vad_mock(),
            grpc_client=_make_grpc_client_mock(),
            postprocessor=_make_postprocessor_mock(),
            on_event=AsyncMock(),
        )
        assert isinstance(session.wal, SessionWAL)
        assert session.wal.last_committed_segment_id == 0
        await session.close()

    async def test_session_uses_injected_wal(self) -> None:
        """StreamingSession usa WAL injetado em vez de criar default."""
        custom_wal = SessionWAL()
        custom_wal.record_checkpoint(segment_id=42, buffer_offset=0, timestamp_ms=0)

        session = StreamingSession(
            session_id="test",
            preprocessor=make_preprocessor_mock(),
            vad=make_vad_mock(),
            grpc_client=_make_grpc_client_mock(),
            postprocessor=_make_postprocessor_mock(),
            on_event=AsyncMock(),
            wal=custom_wal,
        )
        assert session.wal is custom_wal
        assert session.wal.last_committed_segment_id == 42
        await session.close()

    async def test_session_records_wal_checkpoint_after_transcript_final(self) -> None:
        """StreamingSession registra checkpoint no WAL apos transcript.final."""
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
        vad = make_vad_mock()
        on_event = AsyncMock()
        wal = SessionWAL()

        session = StreamingSession(
            session_id="test_session",
            preprocessor=make_preprocessor_mock(),
            vad=vad,
            grpc_client=grpc_client,
            postprocessor=_make_postprocessor_mock(),
            on_event=on_event,
            enable_itn=False,
            ring_buffer=rb,
            wal=wal,
        )

        # Trigger speech_start
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START,
            timestamp_ms=0,
        )
        vad.is_speaking = True
        await session.process_frame(make_raw_bytes())

        # Enviar mais frames
        vad.process_frame.return_value = None
        await session.process_frame(make_raw_bytes())

        # Aguardar receiver task processar o transcript.final
        await asyncio.sleep(0.05)

        # Assert: WAL checkpoint registrado
        assert wal.last_committed_segment_id == 0  # segment_id no momento do final
        assert wal.last_committed_buffer_offset == rb.total_written
        assert wal.last_committed_timestamp_ms > 0  # monotonic timestamp

        # Verificar que transcript.final tambem foi emitido
        final_calls = [
            call
            for call in on_event.call_args_list
            if isinstance(call.args[0], TranscriptFinalEvent)
        ]
        assert len(final_calls) == 1

        await session.close()

    async def test_session_records_wal_without_ring_buffer(self) -> None:
        """StreamingSession registra WAL checkpoint mesmo sem ring buffer."""
        final_segment = TranscriptSegment(
            text="sem ring buffer",
            is_final=True,
            segment_id=0,
            start_ms=0,
            end_ms=500,
            language="pt",
            confidence=0.9,
        )

        stream_handle = _make_stream_handle_mock(events=[final_segment])
        grpc_client = _make_grpc_client_mock(stream_handle)
        vad = make_vad_mock()
        wal = SessionWAL()

        session = StreamingSession(
            session_id="test_session",
            preprocessor=make_preprocessor_mock(),
            vad=vad,
            grpc_client=grpc_client,
            postprocessor=_make_postprocessor_mock(),
            on_event=AsyncMock(),
            enable_itn=False,
            ring_buffer=None,
            wal=wal,
        )

        # Trigger speech_start
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START,
            timestamp_ms=0,
        )
        vad.is_speaking = True
        await session.process_frame(make_raw_bytes())

        # Aguardar receiver task
        await asyncio.sleep(0.05)

        # WAL checkpoint com buffer_offset=0 (sem ring buffer)
        assert wal.last_committed_segment_id == 0
        assert wal.last_committed_buffer_offset == 0
        assert wal.last_committed_timestamp_ms > 0

        await session.close()

    async def test_session_wal_multiple_finals_overwrite(self) -> None:
        """Multiplos transcript.final sobrescrevem checkpoint anterior no WAL."""
        final1 = TranscriptSegment(
            text="primeiro",
            is_final=True,
            segment_id=0,
            start_ms=0,
            end_ms=500,
        )
        final2 = TranscriptSegment(
            text="segundo",
            is_final=True,
            segment_id=0,
            start_ms=500,
            end_ms=1000,
        )

        rb = RingBuffer(duration_s=5.0, sample_rate=16000, bytes_per_sample=2)
        stream_handle = _make_stream_handle_mock(events=[final1, final2])
        grpc_client = _make_grpc_client_mock(stream_handle)
        vad = make_vad_mock()
        wal = SessionWAL()

        session = StreamingSession(
            session_id="test_session",
            preprocessor=make_preprocessor_mock(),
            vad=vad,
            grpc_client=grpc_client,
            postprocessor=_make_postprocessor_mock(),
            on_event=AsyncMock(),
            enable_itn=False,
            ring_buffer=rb,
            wal=wal,
        )

        # Trigger speech_start e enviar frames
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START,
            timestamp_ms=0,
        )
        vad.is_speaking = True
        await session.process_frame(make_raw_bytes())
        vad.process_frame.return_value = None
        await session.process_frame(make_raw_bytes())

        # Aguardar ambos finals serem processados
        await asyncio.sleep(0.05)

        # WAL deve ter o checkpoint do ultimo final
        assert wal.last_committed_segment_id == 0  # mesmo segmento, 2 finals
        assert wal.last_committed_buffer_offset == rb.total_written
        # Segundo checkpoint tem timestamp >= primeiro (monotonic)
        assert wal.last_committed_timestamp_ms > 0

        await session.close()
