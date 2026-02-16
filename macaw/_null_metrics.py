"""Null Object metrics for when prometheus_client is not installed.

Provides no-op implementations of Histogram, Counter, and Gauge
that silently discard all observations. This eliminates the need
for ``if HAS_METRICS and metric is not None:`` guards at every
call site — consumers can always call metric methods unconditionally.
"""

from __future__ import annotations

from typing import Any


class NullMetric:
    """No-op metric that silently ignores all observations.

    Supports the full Histogram/Counter/Gauge interface so it can
    substitute any prometheus_client metric type.
    """

    def observe(self, amount: float = 0) -> None:
        """No-op observe (Histogram interface)."""

    def inc(self, amount: float = 1) -> None:
        """No-op increment (Counter/Gauge interface)."""

    def dec(self, amount: float = 1) -> None:
        """No-op decrement (Gauge interface)."""

    def set(self, value: float) -> None:
        """No-op set (Gauge interface)."""

    def labels(self, *args: Any, **kwargs: Any) -> NullMetric:
        """Return self so chained calls like `metric.labels(...).inc()` work."""
        return self
