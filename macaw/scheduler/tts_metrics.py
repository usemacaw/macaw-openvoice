"""Prometheus metrics for TTS.

Metrics are optional: if prometheus_client is not installed, the module
exports NullMetric instances that silently discard all observations.
Consumer code can call metric methods unconditionally (no guards needed).

Defined metrics (M9):
- macaw_tts_ttfb_seconds: Time to First Byte (first audio chunk after tts.speak)
- macaw_tts_synthesis_duration_seconds: Total synthesis duration (first to last chunk)
- macaw_tts_requests_total: Counter of TTS requests by status (ok/error/cancelled)
- macaw_tts_active_sessions: Gauge of sessions with active TTS
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prometheus_client import Counter, Gauge, Histogram

try:
    from prometheus_client import Counter as _Counter
    from prometheus_client import Gauge as _Gauge
    from prometheus_client import Histogram as _Histogram

    tts_ttfb_seconds: Histogram = _Histogram(
        "macaw_tts_ttfb_seconds",
        "Time to first audio byte after tts.speak command",
        buckets=(0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0),
    )

    tts_synthesis_duration_seconds: Histogram = _Histogram(
        "macaw_tts_synthesis_duration_seconds",
        "Total duration of TTS synthesis (first to last chunk)",
        buckets=(0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
    )

    tts_requests_total: Counter = _Counter(
        "macaw_tts_requests_total",
        "Total TTS requests by status",
        ["status"],
    )

    tts_active_sessions: Gauge = _Gauge(
        "macaw_tts_active_sessions",
        "Number of sessions with active TTS synthesis",
    )

except ImportError:
    from macaw._null_metrics import NullMetric as _Null

    tts_ttfb_seconds = _Null()  # type: ignore[assignment]
    tts_synthesis_duration_seconds = _Null()  # type: ignore[assignment]
    tts_requests_total = _Null()  # type: ignore[assignment]
    tts_active_sessions = _Null()  # type: ignore[assignment]
