"""Prometheus metrics for the batch Scheduler.

Metrics are optional: if prometheus_client is not installed, the module
exports NullMetric instances that silently discard all observations.
Consumer code can call metric methods unconditionally (no guards needed).

Defined metrics (M8):
- macaw_scheduler_queue_depth: Gauge of queue depth by priority
- macaw_scheduler_queue_wait_seconds: Histogram of time spent in the queue
- macaw_scheduler_grpc_duration_seconds: Histogram of gRPC call duration
- macaw_scheduler_cancel_latency_seconds: Histogram of cancel propagation latency
- macaw_scheduler_batch_size: Histogram of batch sizes dispatched
- macaw_scheduler_requests_total: Counter of requests by priority and status
- macaw_scheduler_aging_promotions_total: Counter of promotions by aging
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prometheus_client import Counter, Gauge, Histogram

try:
    from prometheus_client import Counter as _Counter
    from prometheus_client import Gauge as _Gauge
    from prometheus_client import Histogram as _Histogram

    scheduler_queue_depth: Gauge = _Gauge(
        "macaw_scheduler_queue_depth",
        "Number of requests pending in the scheduler queue",
        ["priority"],
    )

    scheduler_queue_wait_seconds: Histogram = _Histogram(
        "macaw_scheduler_queue_wait_seconds",
        "Time spent waiting in the scheduler queue",
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
    )

    scheduler_grpc_duration_seconds: Histogram = _Histogram(
        "macaw_scheduler_grpc_duration_seconds",
        "Duration of gRPC TranscribeFile calls to workers",
        buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
    )

    scheduler_cancel_latency_seconds: Histogram = _Histogram(
        "macaw_scheduler_cancel_latency_seconds",
        "Latency of cancel propagation to workers via gRPC",
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
    )

    scheduler_batch_size: Histogram = _Histogram(
        "macaw_scheduler_batch_size",
        "Number of requests per batch dispatch",
        buckets=(1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16),
    )

    scheduler_requests_total: Counter = _Counter(
        "macaw_scheduler_requests_total",
        "Total scheduler requests by priority and outcome",
        ["priority", "status"],
    )

    scheduler_aging_promotions_total: Counter = _Counter(
        "macaw_scheduler_aging_promotions_total",
        "Requests promoted from BATCH to REALTIME priority by aging",
    )

    HAS_METRICS = True

except ImportError:
    from macaw._null_metrics import NullMetric as _Null

    scheduler_queue_depth = _Null()  # type: ignore[assignment]
    scheduler_queue_wait_seconds = _Null()  # type: ignore[assignment]
    scheduler_grpc_duration_seconds = _Null()  # type: ignore[assignment]
    scheduler_cancel_latency_seconds = _Null()  # type: ignore[assignment]
    scheduler_batch_size = _Null()  # type: ignore[assignment]
    scheduler_requests_total = _Null()  # type: ignore[assignment]
    scheduler_aging_promotions_total = _Null()  # type: ignore[assignment]

    HAS_METRICS = False
