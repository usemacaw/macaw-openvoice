"""Prometheus metrics for streaming STT.

Metrics are optional: if prometheus_client is not installed, the module
exports NullMetric instances that silently discard all observations.
Consumer code can call metric methods unconditionally (no guards needed).

Defined metrics (M5):
- macaw_stt_ttfb_seconds: Time to First Byte (first partial/final after speech start)
- macaw_stt_final_delay_seconds: Delay of final transcript after end of speech
- macaw_stt_active_sessions: Gauge of active WebSocket sessions
- macaw_stt_vad_events_total: Counter of VAD events by type (speech_start, speech_end)

Added metrics (M6):
- macaw_stt_session_duration_seconds: Total duration of closed sessions
- macaw_stt_segments_force_committed_total: Segments force committed (ring buffer >90%)
- macaw_stt_confidence_avg: Histogram of confidence for transcript.final
- macaw_stt_worker_recoveries_total: Counter of recovery attempts by result

Added metrics (M9):
- macaw_stt_muted_frames_total: Frames dropped by mute-on-speak (TTS active)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prometheus_client import Counter, Gauge, Histogram

try:
    from prometheus_client import Counter as _Counter
    from prometheus_client import Gauge as _Gauge
    from prometheus_client import Histogram as _Histogram

    stt_ttfb_seconds: Histogram = _Histogram(
        "macaw_stt_ttfb_seconds",
        "Time to first byte (first partial transcript after speech start)",
        buckets=(0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0),
    )

    stt_final_delay_seconds: Histogram = _Histogram(
        "macaw_stt_final_delay_seconds",
        "Delay of final transcript after end of speech",
        buckets=(0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0),
    )

    stt_active_sessions: Gauge = _Gauge(
        "macaw_stt_active_sessions",
        "Number of active WebSocket streaming sessions",
    )

    stt_vad_events_total: Counter = _Counter(
        "macaw_stt_vad_events_total",
        "Total VAD events by type",
        ["event_type"],
    )

    # M6 metrics

    stt_session_duration_seconds: Histogram = _Histogram(
        "macaw_stt_session_duration_seconds",
        "Total duration of completed streaming sessions",
        buckets=(1, 5, 10, 30, 60, 120, 300, 600, 1800),
    )

    stt_segments_force_committed_total: Counter = _Counter(
        "macaw_stt_segments_force_committed_total",
        "Segments force committed due to ring buffer exceeding 90% capacity",
    )

    stt_confidence_avg: Histogram = _Histogram(
        "macaw_stt_confidence_avg",
        "Confidence scores of final transcripts (proxy for WER)",
        buckets=(0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99),
    )

    stt_worker_recoveries_total: Counter = _Counter(
        "macaw_stt_worker_recoveries_total",
        "Worker crash recovery attempts by result",
        ["result"],
    )

    # M9 metrics

    stt_muted_frames_total: Counter = _Counter(
        "macaw_stt_muted_frames_total",
        "Frames discarded by mute-on-speak (TTS active)",
    )

    HAS_METRICS = True

except ImportError:
    from macaw._null_metrics import NullMetric as _Null

    stt_ttfb_seconds = _Null()  # type: ignore[assignment]
    stt_final_delay_seconds = _Null()  # type: ignore[assignment]
    stt_active_sessions = _Null()  # type: ignore[assignment]
    stt_vad_events_total = _Null()  # type: ignore[assignment]
    stt_session_duration_seconds = _Null()  # type: ignore[assignment]
    stt_segments_force_committed_total = _Null()  # type: ignore[assignment]
    stt_confidence_avg = _Null()  # type: ignore[assignment]
    stt_worker_recoveries_total = _Null()  # type: ignore[assignment]
    stt_muted_frames_total = _Null()  # type: ignore[assignment]

    HAS_METRICS = False
