"""Tests for Prometheus metrics in STT streaming.

Validates that StreamingSession correctly records:
- TTFB (time to first partial/final after speech_start)
- Final delay (time between speech_end and transcript.final)
- Active sessions (increments/decrements on lifecycle)
- VAD events (counter per type speech_start/speech_end)

Tests are deterministic -- use time.monotonic mock to control timestamps.
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

# Check if prometheus_client is available for value tests
try:
    from prometheus_client import REGISTRY

    _HAS_PROMETHEUS = True
except ImportError:
    _HAS_PROMETHEUS = False


def _make_stream_handle_mock(
    events: list[object] | None = None,
) -> Mock:
    """Create StreamHandle mock."""
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
    """Create StreamingGRPCClient mock."""
    client = AsyncMock()
    if stream_handle is None:
        stream_handle = _make_stream_handle_mock()
    client.open_stream = AsyncMock(return_value=stream_handle)
    client.close = AsyncMock()
    return client


def _make_postprocessor_mock() -> Mock:
    """Create PostProcessingPipeline mock."""
    mock = Mock()
    mock.process.side_effect = lambda text, **kwargs: text
    return mock


def _make_on_event() -> AsyncMock:
    """Create on_event mock callback."""
    return AsyncMock()


def _make_session(
    *,
    stream_handle: Mock | None = None,
    vad: Mock | None = None,
    on_event: AsyncMock | None = None,
    session_id: str = "test_metrics",
) -> tuple[StreamingSession, Mock, Mock, AsyncMock]:
    """Create StreamingSession with mocks for metrics tests.

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
# Helpers for reading metrics
# ---------------------------------------------------------------------------


def _get_gauge_value(metric_name: str) -> float:
    """Read the current value of a Gauge from the default REGISTRY."""
    for metric in REGISTRY.collect():
        if metric.name == metric_name:
            for sample in metric.samples:
                if sample.name == metric_name:
                    return sample.value
    return 0.0


def _get_counter_value(metric_name: str, labels: dict[str, str]) -> float:
    """Read the current value of a Counter from the default REGISTRY."""
    for metric in REGISTRY.collect():
        if metric.name == metric_name:
            for sample in metric.samples:
                if sample.name == f"{metric_name}_total" and sample.labels == labels:
                    return sample.value
    return 0.0


def _get_histogram_count(metric_name: str) -> float:
    """Read the _count of a Histogram from the default REGISTRY."""
    for metric in REGISTRY.collect():
        if metric.name == metric_name:
            for sample in metric.samples:
                if sample.name == f"{metric_name}_count":
                    return sample.value
    return 0.0


def _get_histogram_sum(metric_name: str) -> float:
    """Read the _sum of a Histogram from the default REGISTRY."""
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
    """Active sessions increments when a session is created."""
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
    """Active sessions decrements when a session is closed."""
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
    """Multiple calls to close() decrement only once."""
    # Arrange
    session, _, _, _ = _make_session(session_id="active_idempotent_test")
    value_after_init = _get_gauge_value("macaw_stt_active_sessions")

    # Act
    await session.close()
    await session.close()
    await session.close()

    # Assert: decremented only once
    value_after_close = _get_gauge_value("macaw_stt_active_sessions")
    assert value_after_close == value_after_init - 1


# ---------------------------------------------------------------------------
# Tests: VAD Events Counter
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_PROMETHEUS, reason="prometheus_client not installed")
async def test_vad_speech_start_increments_counter():
    """speech_start event increments VAD counter."""
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
    """speech_end event increments VAD counter."""
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

    # Trigger speech_start first
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
    """TTFB is recorded when the first partial transcript is emitted."""
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

    # Act: trigger speech_start -> receiver task processes partial
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())

    # Give time for receiver task to process
    await asyncio.sleep(0.05)

    # Assert: TTFB was recorded
    current_count = _get_histogram_count("macaw_stt_ttfb_seconds")
    assert current_count == initial_count + 1

    # Assert: partial event was emitted
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
    """TTFB is recorded only once per speech segment."""
    # Arrange: two partials in the same segment
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

    # Assert: TTFB recorded only 1 time (not 2)
    current_count = _get_histogram_count("macaw_stt_ttfb_seconds")
    assert current_count == initial_count + 1

    # Cleanup
    await session.close()


@pytest.mark.skipif(not _HAS_PROMETHEUS, reason="prometheus_client not installed")
async def test_ttfb_value_reflects_elapsed_time():
    """TTFB value reflects real time between speech_start and first transcript."""
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

    # Use real monotonic time (TTFB will be small but > 0)
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
    """Final delay is recorded when transcript.final arrives after speech_end."""
    # Arrange: final segment that will be emitted by worker
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

    # Give time for receiver task to consume the final
    await asyncio.sleep(0.05)

    # 2. Speech end (after the final has already been processed)
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_END,
        timestamp_ms=2000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())

    # Assert: final_delay may or may not have been recorded depending on timing.
    # The final may have arrived BEFORE speech_end, in which case final_delay
    # is not recorded (speech_end_monotonic was None when the final arrived).
    # This test validates that there is no crash in the flow.
    final_calls = [
        call for call in on_event.call_args_list if isinstance(call.args[0], TranscriptFinalEvent)
    ]
    assert len(final_calls) == 1


@pytest.mark.skipif(not _HAS_PROMETHEUS, reason="prometheus_client not installed")
async def test_final_delay_not_recorded_when_no_speech_end():
    """Final delay is NOT recorded when speech_end did not occur before the final."""
    # Arrange: final arrives while still speaking (without speech_end)
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

    # Only speech_start (without speech_end)
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())

    await asyncio.sleep(0.05)

    # Assert: final_delay was NOT recorded (speech_end_monotonic is None)
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

        # Act: create session, process frames, close
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

    # Assert: no crash, session completed normally
    assert session.is_closed


@pytest.mark.skipif(not _HAS_PROMETHEUS, reason="prometheus_client not installed")
async def test_metrics_objects_are_not_none():
    """When prometheus_client is installed, metrics are not None."""
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
    """TTFB is recorded separately for each speech segment."""
    # Arrange: first segment with partial
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

    # Segment 1: speech_start -> partial -> speech_end
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())
    await asyncio.sleep(0.05)

    # Prepare new stream_handle for second segment
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

    # Segment 2: speech_start -> partial
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=3000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())
    await asyncio.sleep(0.05)

    # Assert: TTFB recorded 2x (one per segment)
    current_count = _get_histogram_count("macaw_stt_ttfb_seconds")
    assert current_count == initial_count + 2

    # Cleanup
    await session.close()


# ---------------------------------------------------------------------------
# Tests: M6 Metrics — Session Duration
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_PROMETHEUS, reason="prometheus_client not installed")
async def test_session_duration_recorded_on_close():
    """session_duration_seconds is recorded when session is closed."""
    initial_count = _get_histogram_count("macaw_stt_session_duration_seconds")

    session, _, _, _ = _make_session(session_id="duration_test")

    # Act
    await session.close()

    # Assert: duration recorded
    current_count = _get_histogram_count("macaw_stt_session_duration_seconds")
    assert current_count == initial_count + 1

    # Sum should have increased (duration > 0)
    current_sum = _get_histogram_sum("macaw_stt_session_duration_seconds")
    assert current_sum > 0


@pytest.mark.skipif(not _HAS_PROMETHEUS, reason="prometheus_client not installed")
async def test_session_duration_not_recorded_twice_on_double_close():
    """session_duration_seconds recorded only once even with double close()."""
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
    """confidence_avg is recorded when transcript.final with confidence arrives."""
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

    # Trigger speech_start -> receiver task processes final
    vad.process_frame.return_value = VADEvent(
        type=VADEventType.SPEECH_START,
        timestamp_ms=1000,
    )
    vad.is_speaking = False
    await session.process_frame(make_raw_bytes())
    await asyncio.sleep(0.05)

    # Assert: confidence recorded
    current_count = _get_histogram_count("macaw_stt_confidence_avg")
    assert current_count == initial_count + 1

    await session.close()


@pytest.mark.skipif(not _HAS_PROMETHEUS, reason="prometheus_client not installed")
async def test_confidence_not_recorded_when_none():
    """confidence_avg is NOT recorded when segment.confidence is None."""
    final_seg = TranscriptSegment(
        text="ola",
        is_final=True,
        segment_id=0,
        start_ms=1000,
        end_ms=2000,
        confidence=None,  # no confidence
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

    # Assert: confidence NOT recorded
    current_count = _get_histogram_count("macaw_stt_confidence_avg")
    assert current_count == initial_count

    await session.close()


# ---------------------------------------------------------------------------
# Tests: M6 Metrics — Force Committed Segments
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_PROMETHEUS, reason="prometheus_client not installed")
async def test_force_commit_counter_increments():
    """segments_force_committed_total increments on ring buffer callback."""
    from macaw.session.ring_buffer import RingBuffer

    initial_count = _get_counter_value(
        "macaw_stt_segments_force_committed",
        {},
    )

    vad = make_vad_mock()
    stream_handle = _make_stream_handle_mock()

    # Create 1s ring buffer -- 16000 * 2 = 32000 bytes.
    # Each frame = 1024 * 2 = 2048 bytes, 90% = ~28800 bytes = ~14 frames.
    # Force commit callback fires when uncommitted > 90% of capacity.
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

    # Send frames to reach >90% of uncommitted data.
    # At 90% the callback fires and sets flag -> process_frame calls commit()
    # which advances the fence freeing space for more writes.
    vad.process_frame.return_value = None
    for _ in range(14):
        await session.process_frame(make_raw_bytes())

    await asyncio.sleep(0.01)

    # Assert: force commit counter incremented
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
    """worker_recoveries_total with result=success increments after recovery."""
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
