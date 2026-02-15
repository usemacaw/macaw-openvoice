"""Tests for M8-07: LatencyTracker.

Tests cover: start, dequeue, grpc_started, complete, summary, cleanup,
discard, missing phases, TTL expiration, concurrent requests, integration
with Scheduler.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from macaw.scheduler.latency import LatencySummary, LatencyTracker

# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


class TestLatencyTrackerInit:
    def test_default_construction(self) -> None:
        """LatencyTracker creates with default TTL."""
        tracker = LatencyTracker()
        assert tracker.active_count == 0

    def test_custom_ttl(self) -> None:
        """LatencyTracker accepts custom TTL."""
        tracker = LatencyTracker(ttl_s=120.0)
        assert tracker.active_count == 0

    def test_invalid_ttl(self) -> None:
        """TTL <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="ttl_s"):
            LatencyTracker(ttl_s=0)

        with pytest.raises(ValueError, match="ttl_s"):
            LatencyTracker(ttl_s=-1)


# ---------------------------------------------------------------------------
# Phase tracking tests
# ---------------------------------------------------------------------------


class TestPhaseTracking:
    def test_start_increments_active_count(self) -> None:
        """start() increments active_count."""
        tracker = LatencyTracker()
        tracker.start("req_1")
        assert tracker.active_count == 1

    def test_start_multiple_requests(self) -> None:
        """Multiple start() calls track multiple requests."""
        tracker = LatencyTracker()
        tracker.start("req_1")
        tracker.start("req_2")
        tracker.start("req_3")
        assert tracker.active_count == 3

    def test_dequeued_on_unknown_request_is_safe(self) -> None:
        """dequeued() on unknown request does not raise."""
        tracker = LatencyTracker()
        tracker.dequeued("nonexistent")  # should not raise

    def test_grpc_started_on_unknown_request_is_safe(self) -> None:
        """grpc_started() on unknown request does not raise."""
        tracker = LatencyTracker()
        tracker.grpc_started("nonexistent")  # should not raise


# ---------------------------------------------------------------------------
# Complete and summary tests
# ---------------------------------------------------------------------------


class TestCompleteAndSummary:
    def test_complete_returns_summary(self) -> None:
        """complete() returns a LatencySummary with correct fields."""
        tracker = LatencyTracker()
        tracker.start("req_1")
        tracker.dequeued("req_1")
        tracker.grpc_started("req_1")

        summary = tracker.complete("req_1")

        assert summary is not None
        assert isinstance(summary, LatencySummary)
        assert summary.request_id == "req_1"
        assert summary.queue_wait >= 0
        assert summary.grpc_time >= 0
        assert summary.total_time >= 0
        assert summary.total_time >= summary.queue_wait
        assert summary.total_time >= summary.grpc_time

    def test_complete_removes_from_active(self) -> None:
        """complete() removes request from active tracking."""
        tracker = LatencyTracker()
        tracker.start("req_1")
        assert tracker.active_count == 1

        tracker.complete("req_1")
        assert tracker.active_count == 0

    def test_complete_unknown_returns_none(self) -> None:
        """complete() on unknown request returns None."""
        tracker = LatencyTracker()
        assert tracker.complete("nonexistent") is None

    def test_complete_with_missing_dequeue(self) -> None:
        """complete() works even if dequeued() was never called."""
        tracker = LatencyTracker()
        tracker.start("req_1")
        tracker.grpc_started("req_1")

        summary = tracker.complete("req_1")
        assert summary is not None
        # queue_wait should be 0 since dequeue_time was never set
        assert summary.queue_wait == 0.0

    def test_complete_with_missing_grpc_started(self) -> None:
        """complete() works even if grpc_started() was never called."""
        tracker = LatencyTracker()
        tracker.start("req_1")
        tracker.dequeued("req_1")

        summary = tracker.complete("req_1")
        assert summary is not None
        # grpc_time should be 0 since grpc_start_time was never set
        assert summary.grpc_time == 0.0

    def test_latency_values_are_monotonic(self) -> None:
        """Latency values reflect actual time passage."""
        tracker = LatencyTracker()

        t0 = time.monotonic()
        tracker.start("req_1")
        tracker.dequeued("req_1")
        tracker.grpc_started("req_1")
        summary = tracker.complete("req_1")

        assert summary is not None
        # total_time should be very small (microseconds) but non-negative
        assert summary.total_time >= 0
        assert summary.enqueue_time >= t0
        assert summary.complete_time >= summary.enqueue_time


# ---------------------------------------------------------------------------
# Discard tests
# ---------------------------------------------------------------------------


class TestDiscard:
    def test_discard_removes_active_entry(self) -> None:
        """discard() removes a tracked request."""
        tracker = LatencyTracker()
        tracker.start("req_1")
        assert tracker.active_count == 1

        tracker.discard("req_1")
        assert tracker.active_count == 0

    def test_discard_unknown_is_safe(self) -> None:
        """discard() on unknown request does not raise."""
        tracker = LatencyTracker()
        tracker.discard("nonexistent")  # should not raise


# ---------------------------------------------------------------------------
# Cleanup / TTL tests
# ---------------------------------------------------------------------------


class TestCleanup:
    def test_cleanup_removes_expired_entries(self) -> None:
        """cleanup() removes entries older than TTL."""
        tracker = LatencyTracker(ttl_s=1.0)

        # Fake old enqueue time
        with patch("macaw.scheduler.latency.time.monotonic", return_value=100.0):
            tracker.start("req_old")

        # Now at time 200 (100s later, way past 1s TTL)
        with patch("macaw.scheduler.latency.time.monotonic", return_value=200.0):
            removed = tracker.cleanup()

        assert removed == 1
        assert tracker.active_count == 0

    def test_cleanup_keeps_fresh_entries(self) -> None:
        """cleanup() does not remove entries within TTL."""
        tracker = LatencyTracker(ttl_s=60.0)
        tracker.start("req_fresh")

        removed = tracker.cleanup()
        assert removed == 0
        assert tracker.active_count == 1

    def test_cleanup_returns_count_of_expired_entries(self) -> None:
        """cleanup() returns count of expired active entries removed."""
        tracker = LatencyTracker(ttl_s=1.0)

        with patch("macaw.scheduler.latency.time.monotonic", return_value=100.0):
            tracker.start("req_1")
            tracker.start("req_2")

        with patch("macaw.scheduler.latency.time.monotonic", return_value=200.0):
            removed = tracker.cleanup()

        assert removed == 2
        assert tracker.active_count == 0


# ---------------------------------------------------------------------------
# Concurrent requests tests
# ---------------------------------------------------------------------------


class TestConcurrentRequests:
    def test_multiple_requests_independent(self) -> None:
        """Multiple requests are tracked independently."""
        tracker = LatencyTracker()

        tracker.start("req_1")
        tracker.start("req_2")

        tracker.dequeued("req_1")
        tracker.grpc_started("req_1")
        summary1 = tracker.complete("req_1")

        # req_2 is still active
        assert tracker.active_count == 1
        assert summary1 is not None
        assert summary1.request_id == "req_1"

        tracker.dequeued("req_2")
        tracker.grpc_started("req_2")
        summary2 = tracker.complete("req_2")

        assert tracker.active_count == 0
        assert summary2 is not None
        assert summary2.request_id == "req_2"

    def test_discard_one_keeps_others(self) -> None:
        """Discarding one request does not affect others."""
        tracker = LatencyTracker()
        tracker.start("req_1")
        tracker.start("req_2")

        tracker.discard("req_1")
        assert tracker.active_count == 1

        summary = tracker.complete("req_2")
        assert summary is not None
        assert summary.request_id == "req_2"


# ---------------------------------------------------------------------------
# LatencySummary frozen tests
# ---------------------------------------------------------------------------


class TestLatencySummary:
    def test_summary_is_frozen(self) -> None:
        """LatencySummary is immutable (frozen dataclass)."""
        tracker = LatencyTracker()
        tracker.start("req_1")
        summary = tracker.complete("req_1")
        assert summary is not None

        with pytest.raises(AttributeError):
            summary.queue_wait = 999.0  # type: ignore[misc]
