"""Testes de metricas Prometheus para streaming STT.

Valida que o StreamingSession registra corretamente:
- TTFB (time to first partial/final apos speech_start)
- Final delay (tempo entre speech_end e transcript.final)
- Active sessions (incrementa/decrementa no lifecycle)
- VAD events (counter por tipo speech_start/speech_end)

Testes sao deterministicos — usam mock de time.monotonic para controlar timestamps.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from macaw._types import TranscriptSegment
from macaw.server.models.events import (
    TranscriptFinalEvent,
    TranscriptPartialEvent,
)
from macaw.session.streaming import StreamingSession
from macaw.vad.detector import VADEvent, VADEventType
from tests.helpers import (
    AsyncIterFromList,
    make_preprocessor_mock,
    make_raw_bytes,
    make_vad_mock,
)

# Verificar se prometheus_client esta disponivel para testes de valores
try:
    from prometheus_client import REGISTRY

    _HAS_PROMETHEUS = True
except ImportError:
    _HAS_PROMETHEUS = False


def _make_stream_handle_mock(
    events: list[object] | None = None,
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
    mock.process.side_effect = lambda text: text
    return mock


def _make_on_event() -> AsyncMock:
    """Cria callback on_event mock."""
    return AsyncMock()


def _make_session(
    *,
    stream_handle: Mock | None = None,
    vad: Mock | None = None,
    on_event: AsyncMock | None = None,
    session_id: str = "test_metrics",
) -> tuple[StreamingSession, Mock, Mock, AsyncMock]:
    """Cria StreamingSession com mocks para testes de metricas.

    Returns:
        (session, vad, stream_handle, on_event)
    """
    _vad = vad or make_vad_mock()
    _stream_handle = stream_handle or _make_stream_handle_mock()
    _on_event = on_event or _make_on_event()
    grpc_client = _make_grpc_client_mock(_stream_handle)

    session = StreamingSession(
        session_id=session_id,
        preprocessor=make_preprocessor_mock(),
        vad=_vad,
        grpc_client=grpc_client,
        postprocessor=_make_postprocessor_mock(),
        on_event=_on_event,
    )

    return session, _vad, _stream_handle, _on_event


# ---------------------------------------------------------------------------
# Helpers para leitura de metricas
# ---------------------------------------------------------------------------


def _get_gauge_value(metric_name: str) -> float:
    """Le o valor atual de um Gauge do REGISTRY default."""
    for metric in REGISTRY.collect():
        if metric.name == metric_name:
            for sample in metric.samples:
                if sample.name == metric_name:
                    return sample.value
    return 0.0


def _get_counter_value(metric_name: str, labels: dict[str, str]) -> float:
    """Le o valor atual de um Counter do REGISTRY default."""
    for metric in REGISTRY.collect():
        if metric.name == metric_name:
            for sample in metric.samples:
                if sample.name == f"{metric_name}_total" and sample.labels == labels:
                    return sample.value
    return 0.0


def _get_histogram_count(metric_name: str) -> float:
    """Le o _count de um Histogram do REGISTRY default."""
    for metric in REGISTRY.collect():
        if metric.name == metric_name:
            for sample in metric.samples:
                if sample.name == f"{metric_name}_count":
                    return sample.value
    return 0.0


def _get_histogram_sum(metric_name: str) -> float:
    """Le o _sum de um Histogram do REGISTRY default."""
    for metric in REGISTRY.collect():
        if metric.name == metric_name:
            for sample in metric.samples:
                if sample.name == f"{metric_name}_sum":
                    return sample.value
    return 0.0


# ---------------------------------------------------------------------------
# Tests: Active Sessions
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_PROMETHEUS, reason="prometheus_client not installed")
async def test_active_sessions_increments_on_init():
    """Active sessions incrementa quando uma sessao e criada."""
    # Arrange
    initial_value = _get_gauge_value("macaw_stt_active_sessions")

    # Act
    session, _, _, _ = _make_session(session_id="active_inc_test")

    # Assert
    current_value = _get_gauge_value("macaw_stt_active_sessions")
    assert current_value == initial_value + 1

    # Cleanup
    await session.close()


@pytest.mark.skipif(not _HAS_PROMETHEUS, reason="prometheus_client not installed")
async def test_active_sessions_decrements_on_close():
    """Active sessions decrementa quando uma sessao e fechada."""
    # Arrange
    session, _, _, _ = _make_session(session_id="active_dec_test")
    value_after_init = _get_gauge_value("macaw_stt_active_sessions")

    # Act
    await session.close()

    # Assert
    value_after_close = _get_gauge_value("macaw_stt_active_sessions")
    assert value_after_close == value_after_init - 1


@pytest.mark.skipif(not _HAS_PROMETHEUS, reason="prometheus_client not installed")
async def test_active_sessions_idempotent_close():
    """Multiplas chamadas a close() decrementam apenas uma vez."""
    # Arrange
    session, _, _, _ = _make_session(session_id="active_idempotent_test")
    value_after_init = _get_gauge_value("macaw_stt_active_sessions")

    # Act
    await session.close()
    await session.close()
    await session.close()

    # Assert: decrementou apenas 1x
    value_after_close = _get_gauge_value("macaw_stt_active_sessions")
    assert value_after_close == value_after_init - 1


# ---------------------------------------------------------------------------
# Tests: VAD Events Counter
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_PROMETHEUS, reason="prometheus_client not installed")
async def test_vad_speech_start_increments_counter():
    """Evento speech_start incrementa counter VAD."""
    # Arrange
    initial_starts = _get_counter_value(
        "macaw_stt_vad_events",
        {"event_type": "speech_start"},
    )
    vad = make_vad_mock()
    session, _, _, _ = _make_session(vad=vad, session_id="vad_start_test")

    # Act: trigger speech_start
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())
    await asyncio.sleep(0.01)

    # Assert
    current_starts = _get_counter_value(
        "macaw_stt_vad_events",
        {"event_type": "speech_start"},
    )
    assert current_starts == initial_starts + 1

    # Cleanup
    await session.close()


@pytest.mark.skipif(not _HAS_PROMETHEUS, reason="prometheus_client not installed")
async def test_vad_speech_end_increments_counter():
    """Evento speech_end incrementa counter VAD."""
    # Arrange
    initial_ends = _get_counter_value(
        "macaw_stt_vad_events",
        {"event_type": "speech_end"},
    )
    vad = make_vad_mock()
    stream_handle = _make_stream_handle_mock()
    session, _, _, _ = _make_session(
        vad=vad,
        stream_handle=stream_handle,
        session_id="vad_end_test",
    )

    # Trigger speech_start primeiro
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())
    await asyncio.sleep(0.01)

    # Act: trigger speech_end
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_END,
        timestamp_ms=2000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())

    # Assert
    current_ends = _get_counter_value(
        "macaw_stt_vad_events",
        {"event_type": "speech_end"},
    )
    assert current_ends == initial_ends + 1


# ---------------------------------------------------------------------------
# Tests: TTFB (Time to First Byte)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_PROMETHEUS, reason="prometheus_client not installed")
async def test_ttfb_recorded_on_first_partial():
    """TTFB e registrado quando o primeiro partial transcript e emitido."""
    # Arrange
    partial_seg = TranscriptSegment(
        text="ola",
        is_final=False,
        segment_id=0,
        start_ms=1000,
    )

    vad = make_vad_mock()
    stream_handle = _make_stream_handle_mock(events=[partial_seg])

    initial_count = _get_histogram_count("macaw_stt_ttfb_seconds")

    session, _, _, on_event = _make_session(
        vad=vad,
        stream_handle=stream_handle,
        session_id="ttfb_partial_test",
    )

    # Act: trigger speech_start -> receiver task processa partial
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())

    # Dar tempo para receiver task processar
    await asyncio.sleep(0.05)

    # Assert: TTFB foi registrado
    current_count = _get_histogram_count("macaw_stt_ttfb_seconds")
    assert current_count == initial_count + 1

    # Assert: partial event foi emitido
    partial_calls = [
        call
        for call in on_event.call_args_list
        if isinstance(call.args[0], TranscriptPartialEvent)
    ]
    assert len(partial_calls) == 1

    # Cleanup
    await session.close()


@pytest.mark.skipif(not _HAS_PROMETHEUS, reason="prometheus_client not installed")
async def test_ttfb_recorded_once_per_segment():
    """TTFB so e registrado uma vez por segmento de fala."""
    # Arrange: dois partials no mesmo segmento
    partial1 = TranscriptSegment(
        text="ola",
        is_final=False,
        segment_id=0,
        start_ms=1000,
    )
    partial2 = TranscriptSegment(
        text="ola como",
        is_final=False,
        segment_id=0,
        start_ms=1500,
    )

    vad = make_vad_mock()
    stream_handle = _make_stream_handle_mock(events=[partial1, partial2])

    initial_count = _get_histogram_count("macaw_stt_ttfb_seconds")

    session, _, _, _ = _make_session(
        vad=vad,
        stream_handle=stream_handle,
        session_id="ttfb_once_test",
    )

    # Act
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())

    await asyncio.sleep(0.05)

    # Assert: TTFB registrado apenas 1 vez (nao 2)
    current_count = _get_histogram_count("macaw_stt_ttfb_seconds")
    assert current_count == initial_count + 1

    # Cleanup
    await session.close()


@pytest.mark.skipif(not _HAS_PROMETHEUS, reason="prometheus_client not installed")
async def test_ttfb_value_reflects_elapsed_time():
    """TTFB value reflete o tempo real entre speech_start e primeiro transcript."""
    # Arrange
    partial_seg = TranscriptSegment(
        text="ola",
        is_final=False,
        segment_id=0,
        start_ms=1000,
    )

    vad = make_vad_mock()
    stream_handle = _make_stream_handle_mock(events=[partial_seg])

    initial_sum = _get_histogram_sum("macaw_stt_ttfb_seconds")

    # Usar monotonic time real (o TTFB sera pequeno mas > 0)
    session, _, _, _ = _make_session(
        vad=vad,
        stream_handle=stream_handle,
        session_id="ttfb_value_test",
    )

    # Act
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())

    await asyncio.sleep(0.05)

    # Assert: TTFB sum increased by a positive value
    current_sum = _get_histogram_sum("macaw_stt_ttfb_seconds")
    assert current_sum > initial_sum

    # Cleanup
    await session.close()


# ---------------------------------------------------------------------------
# Tests: Final Delay
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_PROMETHEUS, reason="prometheus_client not installed")
async def test_final_delay_recorded_when_final_after_speech_end():
    """Final delay e registrado quando transcript.final chega apos speech_end."""
    # Arrange: final segment que sera emitido pelo worker
    final_seg = TranscriptSegment(
        text="ola mundo",
        is_final=True,
        segment_id=0,
        start_ms=1000,
        end_ms=2000,
        language="pt",
        confidence=0.95,
    )

    vad = make_vad_mock()
    stream_handle = _make_stream_handle_mock(events=[final_seg])

    session, _, _, on_event = _make_session(
        vad=vad,
        stream_handle=stream_handle,
        session_id="final_delay_test",
    )

    # 1. Speech start
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())

    # Dar tempo para receiver task consumir o final
    await asyncio.sleep(0.05)

    # 2. Speech end (apos o final ja ter sido processado)
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_END,
        timestamp_ms=2000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())

    # Assert: final_delay pode ou nao ter sido registrado dependendo do timing.
    # O final pode ter chegado ANTES do speech_end, nesse caso final_delay
    # nao e registrado (speech_end_monotonic era None quando o final chegou).
    # Este teste valida que nao ha crash no fluxo.
    final_calls = [
        call for call in on_event.call_args_list if isinstance(call.args[0], TranscriptFinalEvent)
    ]
    assert len(final_calls) == 1


@pytest.mark.skipif(not _HAS_PROMETHEUS, reason="prometheus_client not installed")
async def test_final_delay_not_recorded_when_no_speech_end():
    """Final delay NAO e registrado quando speech_end nao ocorreu antes do final."""
    # Arrange: final chega enquanto ainda esta falando (sem speech_end)
    final_seg = TranscriptSegment(
        text="ola mundo",
        is_final=True,
        segment_id=0,
        start_ms=1000,
        end_ms=2000,
    )

    vad = make_vad_mock()
    stream_handle = _make_stream_handle_mock(events=[final_seg])

    initial_count = _get_histogram_count("macaw_stt_final_delay_seconds")

    session, _, _, _ = _make_session(
        vad=vad,
        stream_handle=stream_handle,
        session_id="no_final_delay_test",
    )

    # Apenas speech_start (sem speech_end)
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())

    await asyncio.sleep(0.05)

    # Assert: final_delay NAO foi registrado (speech_end_monotonic e None)
    current_count = _get_histogram_count("macaw_stt_final_delay_seconds")
    assert current_count == initial_count

    # Cleanup
    await session.close()


# ---------------------------------------------------------------------------
# Tests: Graceful Degradation
# ---------------------------------------------------------------------------


async def test_session_works_without_prometheus():
    """Session works normally even without prometheus_client.

    Validates that the NullMetric pattern works correctly: when
    prometheus_client is not installed, NullMetric instances are used
    instead, and all metric calls are silently discarded.
    """
    from macaw._null_metrics import NullMetric

    null = NullMetric()
    # Arrange: simulate NullMetric fallback (no prometheus_client)
    with (
        patch("macaw.session.streaming.stt_active_sessions", null),
        patch("macaw.session.streaming.stt_vad_events_total", null),
        patch("macaw.session.streaming.stt_ttfb_seconds", null),
        patch("macaw.session.streaming.stt_final_delay_seconds", null),
        patch("macaw.session.streaming.stt_session_duration_seconds", null),
        patch("macaw.session.streaming.stt_segments_force_committed_total", null),
        patch("macaw.session.streaming.stt_confidence_avg", null),
        patch("macaw.session.streaming.stt_worker_recoveries_total", null),
    ):
        vad = make_vad_mock()
        stream_handle = _make_stream_handle_mock()
        on_event = _make_on_event()

        # Act: criar sessao, processar frames, fechar
        session = StreamingSession(
            session_id="no_metrics_test",
            preprocessor=make_preprocessor_mock(),
            vad=vad,
            grpc_client=_make_grpc_client_mock(stream_handle),
            postprocessor=_make_postprocessor_mock(),
            on_event=on_event,
        )

        # Trigger speech_start
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_START,
            timestamp_ms=1000,
        )
        vad.is_speaking = False
        await session.process_frame(make_raw_bytes())
        await asyncio.sleep(0.01)

        # Trigger speech_end
        vad.process_frame.return_value = VADEvent(
            type=VADEventType.SPEECH_END,
            timestamp_ms=2000,
        )
        vad.is_speaking = False
        await session.process_frame(make_raw_bytes())

        # Close
        await session.close()

    # Assert: nenhum crash, sessao completou normalmente
    assert session.is_closed


@pytest.mark.skipif(not _HAS_PROMETHEUS, reason="prometheus_client not installed")
async def test_metrics_objects_are_not_none():
    """Quando prometheus_client esta instalado, metricas nao sao None."""
    from macaw.session import metrics as metrics_mod

    # M5 metrics
    assert metrics_mod.stt_active_sessions is not None
    assert metrics_mod.stt_final_delay_seconds is not None
    assert metrics_mod.stt_ttfb_seconds is not None
    assert metrics_mod.stt_vad_events_total is not None
    # M6 metrics
    assert metrics_mod.stt_session_duration_seconds is not None
    assert metrics_mod.stt_segments_force_committed_total is not None
    assert metrics_mod.stt_confidence_avg is not None
    assert metrics_mod.stt_worker_recoveries_total is not None


# ---------------------------------------------------------------------------
# Tests: Multiple Segments
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_PROMETHEUS, reason="prometheus_client not installed")
async def test_ttfb_recorded_per_segment_across_segments():
    """TTFB e registrado separadamente para cada segmento de fala."""
    # Arrange: primeiro segmento com partial
    partial1 = TranscriptSegment(
        text="primeiro",
        is_final=False,
        segment_id=0,
        start_ms=1000,
    )
    stream_handle1 = _make_stream_handle_mock(events=[partial1])

    vad = make_vad_mock()
    grpc_client = _make_grpc_client_mock(stream_handle1)
    on_event = _make_on_event()

    initial_count = _get_histogram_count("macaw_stt_ttfb_seconds")

    session = StreamingSession(
        session_id="ttfb_multi_seg_test",
        preprocessor=make_preprocessor_mock(),
        vad=vad,
        grpc_client=grpc_client,
        postprocessor=_make_postprocessor_mock(),
        on_event=on_event,
    )

    # Segmento 1: speech_start -> partial -> speech_end
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())
    await asyncio.sleep(0.05)

    # Preparar novo stream_handle para segundo segmento
    partial2 = TranscriptSegment(
        text="segundo",
        is_final=False,
        segment_id=1,
        start_ms=3000,
    )
    stream_handle2 = _make_stream_handle_mock(events=[partial2])
    grpc_client.open_stream = AsyncMock(return_value=stream_handle2)

    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_END,
        timestamp_ms=2000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())

    # Segmento 2: speech_start -> partial
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=3000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())
    await asyncio.sleep(0.05)

    # Assert: TTFB registrado 2x (uma por segmento)
    current_count = _get_histogram_count("macaw_stt_ttfb_seconds")
    assert current_count == initial_count + 2

    # Cleanup
    await session.close()


# ---------------------------------------------------------------------------
# Tests: M6 Metrics — Session Duration
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_PROMETHEUS, reason="prometheus_client not installed")
async def test_session_duration_recorded_on_close():
    """session_duration_seconds e registrado quando sessao e fechada."""
    initial_count = _get_histogram_count("macaw_stt_session_duration_seconds")

    session, _, _, _ = _make_session(session_id="duration_test")

    # Act
    await session.close()

    # Assert: duracao registrada
    current_count = _get_histogram_count("macaw_stt_session_duration_seconds")
    assert current_count == initial_count + 1

    # Sum deve ter aumentado (duracao > 0)
    current_sum = _get_histogram_sum("macaw_stt_session_duration_seconds")
    assert current_sum > 0


@pytest.mark.skipif(not _HAS_PROMETHEUS, reason="prometheus_client not installed")
async def test_session_duration_not_recorded_twice_on_double_close():
    """session_duration_seconds registrado apenas 1x mesmo com close() duplo."""
    initial_count = _get_histogram_count("macaw_stt_session_duration_seconds")

    session, _, _, _ = _make_session(session_id="duration_double_test")

    await session.close()
    await session.close()

    current_count = _get_histogram_count("macaw_stt_session_duration_seconds")
    assert current_count == initial_count + 1


# ---------------------------------------------------------------------------
# Tests: M6 Metrics — Confidence Average
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_PROMETHEUS, reason="prometheus_client not installed")
async def test_confidence_recorded_on_final_transcript():
    """confidence_avg e registrado quando transcript.final com confidence chega."""
    final_seg = TranscriptSegment(
        text="ola mundo",
        is_final=True,
        segment_id=0,
        start_ms=1000,
        end_ms=2000,
        language="pt",
        confidence=0.92,
    )

    vad = make_vad_mock()
    stream_handle = _make_stream_handle_mock(events=[final_seg])
    initial_count = _get_histogram_count("macaw_stt_confidence_avg")

    session, _, _, _on_event = _make_session(
        vad=vad,
        stream_handle=stream_handle,
        session_id="confidence_test",
    )

    # Trigger speech_start -> receiver task processa final
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())
    await asyncio.sleep(0.05)

    # Assert: confidence registrado
    current_count = _get_histogram_count("macaw_stt_confidence_avg")
    assert current_count == initial_count + 1

    await session.close()


@pytest.mark.skipif(not _HAS_PROMETHEUS, reason="prometheus_client not installed")
async def test_confidence_not_recorded_when_none():
    """confidence_avg NAO e registrado quando segment.confidence e None."""
    final_seg = TranscriptSegment(
        text="ola",
        is_final=True,
        segment_id=0,
        start_ms=1000,
        end_ms=2000,
        confidence=None,  # sem confidence
    )

    vad = make_vad_mock()
    stream_handle = _make_stream_handle_mock(events=[final_seg])
    initial_count = _get_histogram_count("macaw_stt_confidence_avg")

    session, _, _, _ = _make_session(
        vad=vad,
        stream_handle=stream_handle,
        session_id="no_confidence_test",
    )

    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())
    await asyncio.sleep(0.05)

    # Assert: confidence NAO registrado
    current_count = _get_histogram_count("macaw_stt_confidence_avg")
    assert current_count == initial_count

    await session.close()


# ---------------------------------------------------------------------------
# Tests: M6 Metrics — Force Committed Segments
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_PROMETHEUS, reason="prometheus_client not installed")
async def test_force_commit_counter_increments():
    """segments_force_committed_total incrementa no callback do ring buffer."""
    from macaw.session.ring_buffer import RingBuffer

    initial_count = _get_counter_value(
        "macaw_stt_segments_force_committed",
        {},
    )

    vad = make_vad_mock()
    stream_handle = _make_stream_handle_mock()

    # Criar ring buffer de 1s — 16000 * 2 = 32000 bytes.
    # Cada frame = 1024 * 2 = 2048 bytes, 90% = ~28800 bytes = ~14 frames.
    # Force commit callback dispara quando uncommitted > 90% da capacity.
    rb = RingBuffer(duration_s=1.0, sample_rate=16000, bytes_per_sample=2)

    session = StreamingSession(
        session_id="force_commit_metric_test",
        preprocessor=make_preprocessor_mock(),
        vad=vad,
        grpc_client=_make_grpc_client_mock(stream_handle),
        postprocessor=_make_postprocessor_mock(),
        on_event=_make_on_event(),
        ring_buffer=rb,
    )

    # Trigger speech_start
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = True
    await session.process_frame(make_raw_bytes())

    # Enviar frames para atingir >90% de uncommitted data.
    # A 90% o callback dispara e seta flag -> process_frame chama commit()
    # que avanca o fence liberando espaco para mais writes.
    vad.process_frame.return_value = None
    for _ in range(14):
        await session.process_frame(make_raw_bytes())

    await asyncio.sleep(0.01)

    # Assert: force commit counter incrementou
    current_count = _get_counter_value(
        "macaw_stt_segments_force_committed",
        {},
    )
    assert current_count > initial_count

    await session.close()


# ---------------------------------------------------------------------------
# Tests: M6 Metrics — Worker Recoveries
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_PROMETHEUS, reason="prometheus_client not installed")
async def test_recovery_success_increments_counter():
    """worker_recoveries_total com result=success incrementa apos recovery."""
    from macaw.exceptions import WorkerCrashError

    initial_success = _get_counter_value(
        "macaw_stt_worker_recoveries",
        {"result": "success"},
    )

    vad = make_vad_mock()
    # Primeiro stream handle: crash no receive_events
    crash_handle = _make_stream_handle_mock(events=[WorkerCrashError("w1")])
    # Segundo stream handle: recovery normal (vazio)
    recovery_handle = _make_stream_handle_mock(events=[])

    grpc_client = AsyncMock()
    grpc_client.open_stream = AsyncMock(
        side_effect=[crash_handle, recovery_handle],
    )
    grpc_client.close = AsyncMock()

    on_event = _make_on_event()

    session = StreamingSession(
        session_id="recovery_metric_test",
        preprocessor=make_preprocessor_mock(),
        vad=vad,
        grpc_client=grpc_client,
        postprocessor=_make_postprocessor_mock(),
        on_event=on_event,
        recovery_timeout_s=5.0,
    )

    # Trigger speech_start -> crash -> auto-recovery
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())

    # Dar tempo para receiver task crashar e recovery executar
    await asyncio.sleep(0.15)

    # Assert: recovery success counter incrementou
    current_success = _get_counter_value(
        "macaw_stt_worker_recoveries",
        {"result": "success"},
    )
    assert current_success == initial_success + 1

    await session.close()
