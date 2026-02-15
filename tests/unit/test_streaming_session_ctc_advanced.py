"""Testes avancados de CTC streaming — M7-08.

Valida interacoes CTC-especificas com componentes M6:
- CTC + state machine: transicoes identicas a encoder-decoder
- CTC + ring buffer: escrita, commit, read fence
- CTC + force commit: trigger em 90% de capacidade
- CTC + recovery: crash -> WAL -> resume sem duplicacao de segment_id
- CTC + backpressure: rate_limit e frames_dropped
- CTC sem LocalAgreement: sessao CTC funciona sem LocalAgreement
- CTC + cross-segment: context ignorado para CTC, usado para encoder-decoder
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, Mock

import numpy as np

from macaw._types import SessionState, STTArchitecture, TranscriptSegment
from macaw.exceptions import WorkerCrashError
from macaw.session.backpressure import (
    BackpressureController,
    FramesDroppedAction,
    RateLimitAction,
)
from macaw.session.cross_segment import CrossSegmentContext
from macaw.session.ring_buffer import RingBuffer
from macaw.session.streaming import StreamingSession
from macaw.session.wal import SessionWAL

if TYPE_CHECKING:
    from macaw.session.state_machine import SessionStateMachine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _AsyncIterFromList:
    """Async iterator que yield items de uma lista.

    Necessario porque Python resolve __aiter__/__anext__ na CLASSE,
    nao na instancia — entao precisamos de uma classe real.
    """

    def __init__(self, items: list) -> None:
        self._items = list(items)
        self._index = 0

    def __aiter__(self):  # type: ignore[no-untyped-def]
        return self

    async def __anext__(self):  # type: ignore[no-untyped-def]
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        if isinstance(item, Exception):
            raise item
        return item


def _make_stream_handle(events: list | None = None) -> Mock:
    """Cria mock de StreamHandle com async iterator correto."""
    handle = Mock()
    handle.is_closed = False
    handle.session_id = "test-ctc-adv"
    handle.receive_events.return_value = _AsyncIterFromList(events or [])
    handle.send_frame = AsyncMock()
    handle.close = AsyncMock()
    handle.cancel = AsyncMock()
    return handle


def _make_session(
    *,
    architecture: STTArchitecture = STTArchitecture.CTC,
    state_machine: SessionStateMachine | None = None,
    ring_buffer: RingBuffer | None = None,
    wal: SessionWAL | None = None,
    cross_segment_context: CrossSegmentContext | None = None,
    postprocessor: MagicMock | None = None,
    grpc_client: MagicMock | None = None,
    on_event: AsyncMock | None = None,
) -> StreamingSession:
    """Cria StreamingSession CTC com mocks minimos e componentes reais opcionais."""
    preprocessor = MagicMock()
    preprocessor.process_frame.return_value = np.zeros(320, dtype=np.float32)
    vad = MagicMock()
    vad.process_frame.return_value = None
    vad.is_speaking = False
    _grpc_client = grpc_client or MagicMock()
    _on_event = on_event or AsyncMock()
    return StreamingSession(
        session_id="test-ctc-adv",
        preprocessor=preprocessor,
        vad=vad,
        grpc_client=_grpc_client,
        postprocessor=postprocessor,
        on_event=_on_event,
        architecture=architecture,
        ring_buffer=ring_buffer,
        wal=wal,
        state_machine=state_machine,
        cross_segment_context=cross_segment_context,
    )


# ---------------------------------------------------------------------------
# Tests: CTC + State Machine
# ---------------------------------------------------------------------------


class TestCTCStateMachine:
    """CTC: state machine funciona de forma identica a encoder-decoder."""

    def test_ctc_init_to_active(self) -> None:
        """CTC: transicao INIT -> ACTIVE funciona corretamente."""
        session = _make_session()
        assert session.session_state == SessionState.INIT

        session._state_machine.transition(SessionState.ACTIVE)
        assert session.session_state == SessionState.ACTIVE

    def test_ctc_active_to_silence_to_active(self) -> None:
        """CTC: ACTIVE -> SILENCE -> ACTIVE (silencio seguido de nova fala)."""
        session = _make_session()
        session._state_machine.transition(SessionState.ACTIVE)
        assert session.session_state == SessionState.ACTIVE

        session._state_machine.transition(SessionState.SILENCE)
        assert session.session_state == SessionState.SILENCE

        session._state_machine.transition(SessionState.ACTIVE)
        assert session.session_state == SessionState.ACTIVE

    def test_ctc_silence_to_hold(self) -> None:
        """CTC: SILENCE -> HOLD (silencio prolongado)."""
        session = _make_session()
        session._state_machine.transition(SessionState.ACTIVE)
        session._state_machine.transition(SessionState.SILENCE)
        assert session.session_state == SessionState.SILENCE

        session._state_machine.transition(SessionState.HOLD)
        assert session.session_state == SessionState.HOLD

    def test_ctc_hold_to_active(self) -> None:
        """CTC: HOLD -> ACTIVE (fala retomada apos hold)."""
        session = _make_session()
        session._state_machine.transition(SessionState.ACTIVE)
        session._state_machine.transition(SessionState.SILENCE)
        session._state_machine.transition(SessionState.HOLD)
        assert session.session_state == SessionState.HOLD

        session._state_machine.transition(SessionState.ACTIVE)
        assert session.session_state == SessionState.ACTIVE


# ---------------------------------------------------------------------------
# Tests: CTC + Ring Buffer
# ---------------------------------------------------------------------------


class TestCTCRingBuffer:
    """CTC: ring buffer funciona corretamente para armazenamento de audio."""

    def test_ctc_ring_buffer_write_and_commit(self) -> None:
        """Escrita no ring buffer seguida de commit avanca o read fence."""
        rb = RingBuffer(duration_s=30.0, sample_rate=16000, bytes_per_sample=2)
        session = _make_session(ring_buffer=rb)

        # Escrever dados no ring buffer
        test_data = b"\x01\x02" * 500  # 1000 bytes
        rb.write(test_data)

        assert rb.total_written == 1000
        assert rb.read_fence == 0

        # Commit avanca o read fence
        rb.commit(rb.total_written)
        assert rb.read_fence == 1000
        assert rb.uncommitted_bytes == 0

        # Sessao CTC e funcional com ring buffer
        assert session._ring_buffer is rb

    def test_ctc_ring_buffer_uncommitted_after_write(self) -> None:
        """Escrita sem commit mantem uncommitted_bytes > 0."""
        rb = RingBuffer(duration_s=30.0, sample_rate=16000, bytes_per_sample=2)
        _session = _make_session(ring_buffer=rb)

        data = b"\x00\x01" * 250  # 500 bytes
        rb.write(data)

        assert rb.uncommitted_bytes == 500
        assert rb.total_written == 500
        assert rb.read_fence == 0

    async def test_ctc_ring_buffer_commit_on_final(self) -> None:
        """Ring buffer e commitado apos receive_worker_events processar transcript.final."""
        rb = RingBuffer(duration_s=30.0, sample_rate=16000, bytes_per_sample=2)
        session = _make_session(ring_buffer=rb)

        # Escrever dados no ring buffer (simula frames enviados ao worker)
        rb.write(b"\x00" * 3200)
        assert rb.uncommitted_bytes == 3200

        # Simular transcript.final vindo do worker
        final_segment = TranscriptSegment(
            text="teste ctc",
            is_final=True,
            segment_id=0,
            start_ms=0,
            end_ms=1000,
        )
        handle = _make_stream_handle(events=[final_segment])
        session._stream_handle = handle

        await session._receive_worker_events()

        # Apos transcript.final, ring buffer deve estar commitado
        assert rb.read_fence == rb.total_written
        assert rb.uncommitted_bytes == 0


# ---------------------------------------------------------------------------
# Tests: CTC + Force Commit
# ---------------------------------------------------------------------------


class TestCTCForceCommit:
    """CTC: force commit disparado quando ring buffer atinge 90%."""

    def test_ctc_force_commit_flag_set_at_90_percent(self) -> None:
        """Force commit pending when buffer > 90% uncommitted."""
        # Ring buffer pequeno para facilitar teste (1000 bytes)
        rb = RingBuffer(duration_s=0.03125, sample_rate=16000, bytes_per_sample=2)
        # capacity = 0.03125 * 16000 * 2 = 1000 bytes
        assert rb.capacity_bytes == 1000

        session = _make_session(ring_buffer=rb)

        # No pending force commit initially
        assert session._metrics.consume_force_commit() is False

        # Escrever 901 bytes (>90% de 1000) para disparar force commit
        rb.write(b"\x00" * 901)

        # Force commit should now be pending
        assert session._metrics.consume_force_commit() is True
        # Consumed — second call returns False
        assert session._metrics.consume_force_commit() is False

    async def test_ctc_force_commit_pending_consumed_on_process_frame(self) -> None:
        """process_frame() consumes force_commit_pending and executes commit."""
        rb = RingBuffer(duration_s=0.03125, sample_rate=16000, bytes_per_sample=2)
        session = _make_session(ring_buffer=rb)

        # Trigger force commit via ring buffer callback
        session._metrics.on_ring_buffer_force_commit(0)

        # Colocar sessao em ACTIVE (INIT rejeita frame processing com stream)
        session._state_machine.transition(SessionState.ACTIVE)

        # process_frame deve consumir a flag
        raw_frame = np.zeros(320, dtype=np.int16).tobytes()
        await session.process_frame(raw_frame)

        # Flag consumed by process_frame
        assert session._metrics.consume_force_commit() is False


# ---------------------------------------------------------------------------
# Tests: CTC + Recovery
# ---------------------------------------------------------------------------


class TestCTCRecovery:
    """CTC: recovery de crash restaura sessao sem duplicacao de segment_id."""

    async def test_ctc_recovery_restores_segment_id(self) -> None:
        """Apos crash e recovery, segment_id vem do WAL."""
        wal = SessionWAL()
        wal.record_checkpoint(segment_id=3, buffer_offset=5000, timestamp_ms=100)

        new_handle = _make_stream_handle()
        grpc_client = MagicMock()
        grpc_client.open_stream = AsyncMock(return_value=new_handle)

        session = _make_session(
            wal=wal,
            grpc_client=grpc_client,
        )
        session._state_machine.transition(SessionState.ACTIVE)

        # segment_id antes do recovery (valor arbitrario)
        session._segment_id = 99

        result = await session.recover()

        assert result is True
        # WAL.last_committed_segment_id (3) + 1 = 4
        assert session.segment_id == 4

        await session.close()

    async def test_ctc_recovery_resends_uncommitted(self) -> None:
        """Recovery reenvia dados uncommitted do ring buffer ao novo worker."""
        rb = RingBuffer(duration_s=5.0, sample_rate=16000, bytes_per_sample=2)
        wal = SessionWAL()

        # Escrever e commitar parte dos dados
        committed_data = b"\x01\x00" * 400  # 800 bytes
        rb.write(committed_data)
        rb.commit(rb.total_written)
        wal.record_checkpoint(segment_id=0, buffer_offset=rb.total_written, timestamp_ms=50)

        # Escrever dados uncommitted
        uncommitted_data = b"\x02\x00" * 200  # 400 bytes
        rb.write(uncommitted_data)
        assert rb.uncommitted_bytes == 400

        new_handle = _make_stream_handle()
        grpc_client = MagicMock()
        grpc_client.open_stream = AsyncMock(return_value=new_handle)

        session = _make_session(
            ring_buffer=rb,
            wal=wal,
            grpc_client=grpc_client,
        )
        session._state_machine.transition(SessionState.ACTIVE)

        result = await session.recover()

        assert result is True
        new_handle.send_frame.assert_awaited_once()
        sent_data = new_handle.send_frame.call_args.kwargs["pcm_data"]
        assert len(sent_data) == 400
        assert sent_data == uncommitted_data

        await session.close()

    async def test_ctc_recovery_failure_closes_session(self) -> None:
        """Se recovery falha (grpc open falha), sessao transita para CLOSED."""
        grpc_client = MagicMock()
        grpc_client.open_stream = AsyncMock(
            side_effect=WorkerCrashError("test-ctc-adv"),
        )

        session = _make_session(grpc_client=grpc_client)
        session._state_machine.transition(SessionState.ACTIVE)

        result = await session.recover()

        assert result is False
        assert session.session_state == SessionState.CLOSED


# ---------------------------------------------------------------------------
# Tests: CTC sem LocalAgreement
# ---------------------------------------------------------------------------


class TestCTCNoLocalAgreement:
    """CTC: sessao funciona sem LocalAgreement (partials nativos do worker)."""

    async def test_ctc_has_no_local_agreement(self) -> None:
        """Sessao CTC funciona sem LocalAgreement — partials emitidos diretamente."""
        session = _make_session(architecture=STTArchitecture.CTC)
        on_event = session._on_event

        # Sequencia de partials + final do worker (nativo CTC)
        segments = [
            TranscriptSegment(text="ola", is_final=False, segment_id=0, start_ms=100),
            TranscriptSegment(text="ola mundo", is_final=False, segment_id=0, start_ms=200),
            TranscriptSegment(
                text="ola mundo como vai",
                is_final=True,
                segment_id=0,
                start_ms=0,
                end_ms=3000,
                confidence=0.92,
            ),
        ]

        handle = _make_stream_handle(events=segments)
        session._stream_handle = handle

        await session._receive_worker_events()

        # Todos os 3 eventos emitidos sem filtragem/LocalAgreement
        assert on_event.call_count == 3
        events = [call.args[0] for call in on_event.call_args_list]

        assert events[0].type == "transcript.partial"
        assert events[0].text == "ola"
        assert events[1].type == "transcript.partial"
        assert events[1].text == "ola mundo"
        assert events[2].type == "transcript.final"
        assert events[2].text == "ola mundo como vai"


# ---------------------------------------------------------------------------
# Tests: CTC + Backpressure
# ---------------------------------------------------------------------------


class TestCTCBackpressure:
    """BackpressureController: rate_limit e frames_dropped."""

    def test_backpressure_rate_limit_action(self) -> None:
        """BackpressureController retorna RateLimitAction quando taxa excede threshold."""
        # Clock controlavel: avanca lentamente para simular envio rapido
        wall_time = 0.0

        def fake_clock() -> float:
            return wall_time

        bp = BackpressureController(
            sample_rate=16000,
            max_backlog_s=10.0,
            rate_limit_threshold=1.2,
            clock=fake_clock,
        )

        # Frame de 20ms = 640 bytes (320 samples * 2 bytes)
        frame_bytes = 640
        frame_duration_s = 0.02  # 20ms

        # Primeiro frame: inicializa (nunca dispara)
        action = bp.record_frame(frame_bytes)
        assert action is None

        # Enviar muitos frames com pouco avanco de wall clock
        # Para disparar rate_limit, precisamos:
        # - wall_elapsed >= 0.5s (MIN_WALL_FOR_RATE_CHECK_S)
        # - effective_audio / wall_elapsed > 1.2 (rate_limit_threshold)
        #
        # Estrategia: avancar wall 0.5s e enviar ~1.0s de audio (taxa ~2.0)
        wall_time = 0.5
        n_frames_for_1s = int(1.0 / frame_duration_s)  # 50 frames = 1s de audio

        last_action = None
        for _ in range(n_frames_for_1s):
            result = bp.record_frame(frame_bytes)
            if result is not None:
                last_action = result

        assert last_action is not None
        assert isinstance(last_action, RateLimitAction)
        assert last_action.delay_ms >= 1

    def test_backpressure_frames_dropped_action(self) -> None:
        """BackpressureController retorna FramesDroppedAction quando backlog > max_backlog_s."""
        wall_time = 0.0

        def fake_clock() -> float:
            return wall_time

        bp = BackpressureController(
            sample_rate=16000,
            max_backlog_s=1.0,  # Threshold baixo para teste
            rate_limit_threshold=1.2,
            clock=fake_clock,
        )

        # Frame de 20ms = 640 bytes
        frame_bytes = 640

        # Primeiro frame: inicializa
        action = bp.record_frame(frame_bytes)
        assert action is None

        # Enviar audio suficiente para exceder max_backlog_s (1.0s)
        # sem avancar wall clock (wall_time permanece 0.0)
        # Backlog = audio_total - wall_elapsed
        # Precisamos de >1.0s de audio com 0.0s de wall elapsed
        # 1.0s / 0.02s = 50 frames; +1 para exceder
        n_frames = 52
        last_action = None
        for _ in range(n_frames):
            result = bp.record_frame(frame_bytes)
            if isinstance(result, FramesDroppedAction):
                last_action = result
                break

        assert last_action is not None
        assert isinstance(last_action, FramesDroppedAction)
        assert last_action.dropped_ms > 0


# ---------------------------------------------------------------------------
# Tests: CTC + Cross-Segment Context
# ---------------------------------------------------------------------------


class TestCTCCrossSegmentAdvanced:
    """CTC: cross-segment context ignorado para CTC, usado para encoder-decoder."""

    async def test_ctc_final_does_not_update_context(self) -> None:
        """transcript.final em sessao CTC NAO atualiza CrossSegmentContext."""
        context = CrossSegmentContext(max_tokens=224)
        session = _make_session(
            architecture=STTArchitecture.CTC,
            cross_segment_context=context,
        )

        final_segment = TranscriptSegment(
            text="teste de contexto ctc",
            is_final=True,
            segment_id=0,
            start_ms=0,
            end_ms=1000,
        )
        handle = _make_stream_handle(events=[final_segment])
        session._stream_handle = handle

        await session._receive_worker_events()

        # Context NAO atualizado para CTC
        assert context.get_prompt() is None

    async def test_encoder_decoder_final_updates_context(self) -> None:
        """transcript.final em sessao encoder-decoder atualiza CrossSegmentContext."""
        context = CrossSegmentContext(max_tokens=224)
        session = _make_session(
            architecture=STTArchitecture.ENCODER_DECODER,
            cross_segment_context=context,
        )

        final_segment = TranscriptSegment(
            text="contexto para proximo segmento",
            is_final=True,
            segment_id=0,
            start_ms=0,
            end_ms=2000,
        )
        handle = _make_stream_handle(events=[final_segment])
        session._stream_handle = handle

        await session._receive_worker_events()

        # Context atualizado para encoder-decoder
        assert context.get_prompt() == "contexto para proximo segmento"

    async def test_ctc_build_prompt_ignores_context(self) -> None:
        """CTC: _build_initial_prompt() ignora cross-segment context."""
        context = CrossSegmentContext(max_tokens=224)
        context.update("texto do segmento anterior")

        session = _make_session(
            architecture=STTArchitecture.CTC,
            cross_segment_context=context,
        )

        prompt = session._build_initial_prompt()
        # CTC: sem context no prompt (mesmo com context disponivel)
        assert prompt is None

    async def test_ctc_wal_checkpoint_recorded_on_final(self) -> None:
        """CTC: WAL checkpoint registrado apos transcript.final."""
        wal = SessionWAL()
        session = _make_session(wal=wal)

        final_segment = TranscriptSegment(
            text="checkpoint ctc",
            is_final=True,
            segment_id=0,
            start_ms=0,
            end_ms=500,
            confidence=0.88,
        )
        handle = _make_stream_handle(events=[final_segment])
        session._stream_handle = handle

        await session._receive_worker_events()

        assert wal.last_committed_segment_id == 0
        assert wal.last_committed_timestamp_ms > 0
