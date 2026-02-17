"""Teste de estabilidade de 30 minutos com todos os componentes M6.

Valida que uma sessao de streaming com state machine, ring buffer, WAL e
cross-segment context nao apresenta degradacao de latencia, memory leak
nem erros inesperados durante 30 minutos simulados de conversacao.

Todos os componentes M6 sao reais (state machine, ring buffer, WAL,
cross-segment context). Apenas preprocessing, VAD, gRPC e postprocessing
sao mocked com classes leves (sem acumulacao de call history).

A simulacao e acelerada — 30 minutos de audio processados em segundos de
tempo real. O tempo e controlado via frames contados (cada frame = 64ms).

Executar com:
    python -m pytest tests/integration/test_m6_stability.py -v --tb=short
"""

from __future__ import annotations

import asyncio
import time
import tracemalloc
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from macaw._types import SessionState, TranscriptSegment
from macaw.server.models.events import SessionHoldEvent, StreamingErrorEvent
from macaw.session.cross_segment import CrossSegmentContext
from macaw.session.ring_buffer import RingBuffer
from macaw.session.state_machine import SessionStateMachine, SessionTimeouts
from macaw.session.streaming import StreamingSession
from macaw.session.wal import SessionWAL
from macaw.vad.detector import VADEvent, VADEventType

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = [pytest.mark.integration, pytest.mark.slow]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Frame size: 1024 samples at 16kHz = 64ms
_FRAME_SIZE = 1024
_SAMPLE_RATE = 16000
_FRAME_DURATION_MS = _FRAME_SIZE / _SAMPLE_RATE * 1000  # 64ms
_BYTES_PER_FRAME = _FRAME_SIZE * 2  # PCM 16-bit = 2 bytes/sample

# Sessao simulada de 30 minutos
_SESSION_DURATION_S = 30 * 60  # 1800 seconds
_SESSION_DURATION_MS = _SESSION_DURATION_S * 1000

# Ciclo de fala normal: 3s fala + 1s silencio
_SPEECH_DURATION_MS = 3000
_SILENCE_DURATION_MS = 1000
_CYCLE_DURATION_MS = _SPEECH_DURATION_MS + _SILENCE_DURATION_MS

# Ciclo de fala longo (para HOLD): 3s fala + 5s silencio
_LONG_SILENCE_DURATION_MS = 5000

# Frames por segundo: 16000 / 1024 = 15.625
_FRAMES_PER_SECOND = _SAMPLE_RATE / _FRAME_SIZE

# Memory growth limit: 10MB (em bytes)
_MAX_MEMORY_GROWTH_BYTES = 10 * 1024 * 1024

# TTFB degradation limit: final <= 1.2x initial
_MAX_TTFB_DEGRADATION_FACTOR = 1.2


# ---------------------------------------------------------------------------
# Lightweight mocks (no call history accumulation)
# ---------------------------------------------------------------------------


class _LightPreprocessor:
    """Preprocessor mock sem acumulacao de historico de chamadas."""

    def __init__(self) -> None:
        self._frame = np.zeros(_FRAME_SIZE, dtype=np.float32)
        self.call_count = 0

    def process_frame(self, raw_bytes: bytes) -> np.ndarray:
        self.call_count += 1
        return self._frame


class _LightVAD:
    """VAD mock sem acumulacao de historico de chamadas."""

    def __init__(self) -> None:
        self.is_speaking = False
        self._next_event: VADEvent | None = None

    def set_next_event(self, event: VADEvent | None) -> None:
        self._next_event = event

    def process_frame(self, frame: np.ndarray) -> VADEvent | None:
        event = self._next_event
        self._next_event = None
        return event

    def reset(self) -> None:
        pass


class _LightPostprocessor:
    """Postprocessor mock sem acumulacao de historico."""

    def process(self, text: str, **kwargs: object) -> str:
        return text


class _AsyncIterFromList:
    """Async iterator que yield items de uma lista."""

    def __init__(self, items: list[Any]) -> None:
        self._items = list(items)
        self._index = 0

    def __aiter__(self) -> _AsyncIterFromList:
        return self

    async def __anext__(self) -> Any:
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        if isinstance(item, Exception):
            raise item
        return item


class _LightStreamHandle:
    """Stream handle mock sem acumulacao de historico."""

    def __init__(self, events: list[Any]) -> None:
        self.is_closed = False
        self.session_id = "stability_m6"
        self._events = events

    def receive_events(self) -> _AsyncIterFromList:
        return _AsyncIterFromList(self._events)

    async def send_frame(
        self,
        *,
        pcm_data: bytes,
        initial_prompt: str | None = None,
        hot_words: list[str] | None = None,
    ) -> None:
        pass

    async def close(self) -> None:
        self.is_closed = True

    async def cancel(self) -> None:
        self.is_closed = True


class _LightGRPCClient:
    """gRPC client mock sem acumulacao de historico."""

    def __init__(self, handle_factory: Callable[[], _LightStreamHandle]) -> None:
        self._handle_factory = handle_factory

    async def open_stream(self, session_id: str) -> _LightStreamHandle:
        return self._handle_factory()

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_raw_bytes(n_samples: int = _FRAME_SIZE) -> bytes:
    """Gera bytes PCM int16 (zeros) com n_samples amostras."""
    return np.zeros(n_samples, dtype=np.int16).tobytes()


# ---------------------------------------------------------------------------
# Test 1: 30-minute stability with HOLD transitions
# ---------------------------------------------------------------------------


async def test_m6_stability_30_minutes() -> None:
    """Sessao de 30 minutos (simulada) com todos os componentes M6.

    Simula ciclos de fala/silencio (3s fala + 1s silencio) durante 30 minutos.
    A cada ~100 ciclos (~6.6 min) insere um gap de silencio longo (5s) para
    disparar transicoes SILENCE -> HOLD via timeout da state machine.

    Usa clock mockado na state machine para controlar timeouts sem esperar
    tempo real. O clock avanca conforme o tempo simulado da sessao.

    Verifica:
    - segment_count: ~450 segmentos em 30min com ciclos de 4s
    - hold_transitions: > 0 (pelo menos 1 transicao para HOLD)
    - errors_emitted: 0
    - Crescimento de memoria < 10MB (tracemalloc)
    - TTFB nao degrada (final <= 1.2x initial, se initial > 5ms)
    - ring_buffer.total_written > 0
    - WAL.last_committed_segment_id == numero de segmentos
    - Sessao fecha limpa
    """
    # --- Setup ---
    preprocessor = _LightPreprocessor()
    postprocessor = _LightPostprocessor()
    vad = _LightVAD()
    raw_frame = _make_raw_bytes()

    # Contadores
    ttfb_measurements: list[float] = []
    errors_emitted: list[StreamingErrorEvent] = []
    hold_transitions = 0
    segment_count = 0

    async def tracking_on_event(event: Any) -> None:
        nonlocal hold_transitions
        if isinstance(event, StreamingErrorEvent):
            errors_emitted.append(event)
        elif isinstance(event, SessionHoldEvent):
            hold_transitions += 1

    # Clock mockado: avanca com o tempo simulado
    simulated_clock_s = 0.0

    def mock_clock() -> float:
        return simulated_clock_s

    # Handle factory com segment counter
    current_segment_id = 0

    def _make_handle() -> _LightStreamHandle:
        nonlocal current_segment_id
        final_segment = TranscriptSegment(
            text=f"segmento {current_segment_id}",
            is_final=True,
            segment_id=current_segment_id,
            start_ms=int(simulated_clock_s * 1000),
            end_ms=int(simulated_clock_s * 1000) + _SPEECH_DURATION_MS,
            language="pt",
            confidence=0.95,
        )
        current_segment_id += 1
        return _LightStreamHandle(events=[final_segment])

    grpc_client = _LightGRPCClient(handle_factory=_make_handle)

    # Real M6 components
    state_machine = SessionStateMachine(
        timeouts=SessionTimeouts(
            init_timeout_s=30.0,
            silence_timeout_s=2.0,  # Short: triggers HOLD during 5s gaps
            hold_timeout_s=300.0,
            closing_timeout_s=2.0,
        ),
        clock=mock_clock,
    )
    ring_buffer = RingBuffer(
        duration_s=60.0,
        sample_rate=_SAMPLE_RATE,
        bytes_per_sample=2,
    )
    wal = SessionWAL()
    cross_segment = CrossSegmentContext(max_tokens=224)

    session = StreamingSession(
        session_id="stability_m6_30min",
        preprocessor=preprocessor,  # type: ignore[arg-type]
        vad=vad,  # type: ignore[arg-type]
        grpc_client=grpc_client,  # type: ignore[arg-type]
        postprocessor=postprocessor,  # type: ignore[arg-type]
        on_event=tracking_on_event,
        hot_words=["PIX", "TED"],
        enable_itn=True,
        state_machine=state_machine,
        ring_buffer=ring_buffer,
        wal=wal,
        cross_segment_context=cross_segment,
    )

    # --- Baseline de memoria via tracemalloc ---
    tracemalloc.start()
    snapshot_baseline = tracemalloc.take_snapshot()

    # --- Simulacao de 30 minutos ---
    simulated_time_ms = 0
    is_in_speech = False
    speech_start_time: float | None = None
    cycle_index = 0
    frame_index = 0

    while simulated_time_ms < _SESSION_DURATION_MS:
        # Determinar se este ciclo usa silencio longo (para HOLD)
        # A cada 100 ciclos (~400s / ~6.6min), inserir silencio de 5s
        use_long_silence = (cycle_index > 0) and (cycle_index % 100 == 0)
        current_silence_ms = (
            _LONG_SILENCE_DURATION_MS if use_long_silence else _SILENCE_DURATION_MS
        )
        current_cycle_ms = _SPEECH_DURATION_MS + current_silence_ms

        # Posicao dentro do ciclo atual
        cycle_start_ms = simulated_time_ms
        cycle_end_ms = cycle_start_ms + current_cycle_ms
        speech_end_in_cycle_ms = cycle_start_ms + _SPEECH_DURATION_MS

        # Processar frames deste ciclo
        while simulated_time_ms < cycle_end_ms and simulated_time_ms < _SESSION_DURATION_MS:
            # Atualizar clock simulado
            simulated_clock_s = simulated_time_ms / 1000.0

            if simulated_time_ms < speech_end_in_cycle_ms:
                # Parte de fala do ciclo
                if not is_in_speech:
                    vad.set_next_event(
                        VADEvent(
                            type=VADEventType.SPEECH_START,
                            timestamp_ms=simulated_time_ms,
                        )
                    )
                    vad.is_speaking = True
                    is_in_speech = True
                    speech_start_time = time.monotonic()
            else:
                # Parte de silencio do ciclo
                if is_in_speech:
                    vad.set_next_event(
                        VADEvent(
                            type=VADEventType.SPEECH_END,
                            timestamp_ms=simulated_time_ms,
                        )
                    )
                    vad.is_speaking = False
                    is_in_speech = False
                    segment_count += 1

                    # Medir TTFB
                    if speech_start_time is not None:
                        elapsed = time.monotonic() - speech_start_time
                        ttfb_measurements.append(elapsed)
                        speech_start_time = None

            # Processar frame
            await session.process_frame(raw_frame)

            # Verificar timeouts da state machine durante silencio
            # (simula o check_timeout que o WS handler faz periodicamente)
            if not is_in_speech:
                await session.check_timeout()

            # Dar tempo ao event loop periodicamente
            if frame_index % 100 == 0:
                await asyncio.sleep(0.001)

            # Avancar tempo simulado
            simulated_time_ms += int(_FRAME_DURATION_MS)
            frame_index += 1

        cycle_index += 1

    # Fechar ultimo segmento se estava falando
    if is_in_speech:
        simulated_clock_s = simulated_time_ms / 1000.0
        vad.set_next_event(
            VADEvent(
                type=VADEventType.SPEECH_END,
                timestamp_ms=simulated_time_ms,
            )
        )
        vad.is_speaking = False
        await session.process_frame(raw_frame)
        segment_count += 1
        await asyncio.sleep(0.01)

    # Fechar sessao
    await session.close()

    # --- Snapshot final de memoria ---
    snapshot_final = tracemalloc.take_snapshot()
    tracemalloc.stop()

    # Calcular crescimento de memoria
    stats = snapshot_final.compare_to(snapshot_baseline, "lineno")
    memory_growth = sum(stat.size_diff for stat in stats if stat.size_diff > 0)

    # --- Assertions ---

    # 1. Alocacoes Python nao cresceram mais que 10MB
    assert memory_growth < _MAX_MEMORY_GROWTH_BYTES, (
        f"Memoria Python cresceu {memory_growth / (1024 * 1024):.2f} MB, "
        f"limite e {_MAX_MEMORY_GROWTH_BYTES / (1024 * 1024):.0f} MB. "
        f"Top alocadores: {[(str(s), s.size_diff) for s in stats[:5]]}"
    )

    # 2. TTFB nao degradou
    if len(ttfb_measurements) >= 10:
        n_samples = max(1, len(ttfb_measurements) // 10)
        initial_ttfb = sum(ttfb_measurements[:n_samples]) / n_samples
        final_ttfb = sum(ttfb_measurements[-n_samples:]) / n_samples

        # So valida degradacao se tempos forem significativos (> 5ms)
        if initial_ttfb > 0.005:
            degradation_factor = final_ttfb / initial_ttfb
            assert degradation_factor <= _MAX_TTFB_DEGRADATION_FACTOR, (
                f"TTFB degradou: inicial={initial_ttfb * 1000:.2f}ms, "
                f"final={final_ttfb * 1000:.2f}ms, "
                f"fator={degradation_factor:.2f}x "
                f"(limite: {_MAX_TTFB_DEGRADATION_FACTOR}x)"
            )

    # 3. Zero erros inesperados
    assert len(errors_emitted) == 0, (
        f"Erros inesperados emitidos: {[e.message for e in errors_emitted]}"
    )

    # 4. Segmentos processados (~450 em 30min com ciclos de 4s)
    expected_min_segments = 400
    assert segment_count >= expected_min_segments, (
        f"Apenas {segment_count} segmentos processados, "
        f"esperado pelo menos {expected_min_segments}"
    )

    # 5. HOLD transitions ocorreram (silencio longo > silence_timeout_s=2s)
    # Com ~4 gaps longos em 30min (ciclos 100, 200, 300, 400), deve haver >= 1
    assert hold_transitions > 0, (
        "Nenhuma transicao para HOLD detectada. "
        "Esperado pelo menos 1 durante gaps de silencio longo."
    )

    # 6. Ring buffer teve dados escritos
    assert ring_buffer.total_written > 0, "Ring buffer nao recebeu nenhum dado"

    # 7. WAL registrou todos os segmentos
    # WAL.last_committed_segment_id e o ID do ultimo segmento WITHIN a sessao
    # (session.segment_id avanca apos cada speech_end e force commit)
    assert wal.last_committed_segment_id >= expected_min_segments - 5, (
        f"WAL ultimo segment_id={wal.last_committed_segment_id}, "
        f"esperado >= {expected_min_segments - 5}"
    )

    # 8. Sessao fechou limpa
    assert session.is_closed
    assert session.session_state == SessionState.CLOSED

    # 9. Cross-segment context tem texto (do ultimo segmento)
    assert cross_segment.get_prompt() is not None


# ---------------------------------------------------------------------------
# Test 2: Force commit during long speech
# ---------------------------------------------------------------------------


async def test_m6_force_commit_during_long_speech() -> None:
    """Fala continua de 65s — mais longa que o ring buffer de 60s.

    O ring buffer (60s / 1,920,000 bytes) dispara force commit quando
    dados nao commitados excedem 90% da capacidade (~54s / 1,728,000 bytes).

    Verifica:
    - Pelo menos 1 force commit ocorreu (segment_id incrementou mid-speech)
    - Nenhum erro emitido
    - Sessao fecha limpa
    """
    preprocessor = _LightPreprocessor()
    postprocessor = _LightPostprocessor()
    vad = _LightVAD()
    raw_frame = _make_raw_bytes()

    errors_emitted: list[StreamingErrorEvent] = []
    force_commit_segment_ids: list[int] = []

    async def tracking_on_event(event: Any) -> None:
        if isinstance(event, StreamingErrorEvent):
            errors_emitted.append(event)

    # Handle factory: each call creates a new handle with a transcript.final.
    # The first handle is for the speech opened by SPEECH_START.
    # Subsequent handles are opened by force commit -> next process_frame cycle.
    handle_counter = 0

    def _make_handle() -> _LightStreamHandle:
        nonlocal handle_counter
        final_segment = TranscriptSegment(
            text=f"force commit segment {handle_counter}",
            is_final=True,
            segment_id=handle_counter,
            start_ms=0,
            end_ms=3000,
            language="pt",
            confidence=0.92,
        )
        handle_counter += 1
        return _LightStreamHandle(events=[final_segment])

    grpc_client = _LightGRPCClient(handle_factory=_make_handle)

    # Real M6 components
    ring_buffer = RingBuffer(
        duration_s=60.0,
        sample_rate=_SAMPLE_RATE,
        bytes_per_sample=2,
    )
    wal = SessionWAL()

    session = StreamingSession(
        session_id="force_commit_long_speech",
        preprocessor=preprocessor,  # type: ignore[arg-type]
        vad=vad,  # type: ignore[arg-type]
        grpc_client=grpc_client,  # type: ignore[arg-type]
        postprocessor=postprocessor,  # type: ignore[arg-type]
        on_event=tracking_on_event,
        enable_itn=False,
        ring_buffer=ring_buffer,
        wal=wal,
    )

    # --- Speech start ---
    vad.set_next_event(
        VADEvent(type=VADEventType.SPEECH_START, timestamp_ms=0),
    )
    vad.is_speaking = True

    # 65s of continuous speech
    # Frames needed: 65 * 15.625 = 1015.625 -> 1016 frames
    total_frames = int(65 * _FRAMES_PER_SECOND) + 1
    last_known_segment_id = 0

    for i in range(total_frames):
        await session.process_frame(raw_frame)

        # Detect force commit: segment_id increases mid-speech
        if session.segment_id > last_known_segment_id:
            force_commit_segment_ids.append(session.segment_id)
            last_known_segment_id = session.segment_id

        # Give event loop time periodically
        if i % 100 == 0:
            await asyncio.sleep(0.001)

    # --- Speech end ---
    vad.set_next_event(
        VADEvent(type=VADEventType.SPEECH_END, timestamp_ms=65000),
    )
    vad.is_speaking = False
    await session.process_frame(raw_frame)
    await asyncio.sleep(0.01)

    # --- Close session ---
    await session.close()

    # --- Assertions ---

    # 1. At least 1 force commit happened mid-speech
    assert len(force_commit_segment_ids) >= 1, (
        f"Nenhum force commit detectado durante 65s de fala continua. "
        f"Ring buffer: {ring_buffer.used_bytes}/{ring_buffer.capacity_bytes} "
        f"({ring_buffer.usage_percent:.1f}%). "
        f"Segment ID final: {session.segment_id}"
    )

    # 2. Zero erros emitidos
    assert len(errors_emitted) == 0, f"Erros inesperados: {[e.message for e in errors_emitted]}"

    # 3. Sessao fechou limpa
    assert session.is_closed
    assert session.session_state == SessionState.CLOSED

    # 4. WAL registered checkpoint for the force-committed segment.
    # The force commit closes the gRPC stream, the receiver processes
    # transcript.final and records WAL with the current segment_id (0).
    # After commit(), segment_id advances to 1. Subsequent frames during
    # the same speech are not processed (no stream open until next
    # SPEECH_START). So WAL records segment 0's checkpoint.
    assert wal.last_committed_segment_id >= 0, (
        f"WAL nao registrou checkpoints de force commit. "
        f"last_committed_segment_id={wal.last_committed_segment_id}"
    )
    assert wal.last_committed_buffer_offset > 0, (
        f"WAL buffer_offset deveria ser > 0 apos force commit. "
        f"last_committed_buffer_offset={wal.last_committed_buffer_offset}"
    )

    # 5. Session segment_id should have advanced past 0 (force commit + speech_end)
    assert session.segment_id >= 2, (
        f"segment_id deveria ser >= 2 (force commit + speech_end). Atual: {session.segment_id}"
    )
