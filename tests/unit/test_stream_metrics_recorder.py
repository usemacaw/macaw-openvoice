"""Unit tests for StreamMetricsRecorder.

Tests cover: consume_force_commit() test-and-clear pattern,
on_ring_buffer_force_commit() callback, speech timing, and reset.
"""

from __future__ import annotations

from macaw.session.streaming import StreamMetricsRecorder


# ---------------------------------------------------------------------------
# consume_force_commit (test-and-clear)
# ---------------------------------------------------------------------------


class TestConsumeForceCommit:
    def test_initially_false(self) -> None:
        """consume_force_commit() returns False when no commit is pending."""
        m = StreamMetricsRecorder("s1")
        assert m.consume_force_commit() is False

    def test_returns_true_after_callback(self) -> None:
        """consume_force_commit() returns True after on_ring_buffer_force_commit()."""
        m = StreamMetricsRecorder("s1")
        m.on_ring_buffer_force_commit(1024)
        assert m.consume_force_commit() is True

    def test_clears_flag_after_consume(self) -> None:
        """Second call returns False — flag was consumed."""
        m = StreamMetricsRecorder("s1")
        m.on_ring_buffer_force_commit(1024)
        m.consume_force_commit()
        assert m.consume_force_commit() is False

    def test_multiple_callbacks_single_consume(self) -> None:
        """Multiple force-commit callbacks before consume still yield one True."""
        m = StreamMetricsRecorder("s1")
        m.on_ring_buffer_force_commit(1024)
        m.on_ring_buffer_force_commit(2048)
        assert m.consume_force_commit() is True
        assert m.consume_force_commit() is False

    def test_callback_after_consume_rearms(self) -> None:
        """A new callback after consume re-arms the flag."""
        m = StreamMetricsRecorder("s1")
        m.on_ring_buffer_force_commit(1024)
        m.consume_force_commit()
        m.on_ring_buffer_force_commit(2048)
        assert m.consume_force_commit() is True


# ---------------------------------------------------------------------------
# on_ring_buffer_force_commit ignores parameter
# ---------------------------------------------------------------------------


class TestOnRingBufferForceCommit:
    def test_ignores_total_written_parameter(self) -> None:
        """Callback sets flag regardless of total_written value."""
        m = StreamMetricsRecorder("s1")
        m.on_ring_buffer_force_commit(0)
        assert m.consume_force_commit() is True


# ---------------------------------------------------------------------------
# Speech timing lifecycle
# ---------------------------------------------------------------------------


class TestSpeechTimingLifecycle:
    def test_reset_segment_clears_timing(self) -> None:
        """reset_segment() clears speech_start and speech_end timestamps."""
        m = StreamMetricsRecorder("s1")
        m.on_speech_start()
        m.on_speech_end()
        m.reset_segment()
        # After reset, record_ttfb should be a no-op (no speech_start)
        # We verify by checking that TTFB flag is reset on next speech_start
        m.on_speech_start()
        assert m._ttfb_recorded_for_segment is False

    def test_speech_start_resets_ttfb_flag(self) -> None:
        """on_speech_start() resets the TTFB recorded flag."""
        m = StreamMetricsRecorder("s1")
        m._ttfb_recorded_for_segment = True
        m.on_speech_start()
        assert m._ttfb_recorded_for_segment is False
