"""Tests for BackpressureController.

Validates that the backpressure controller correctly detects:
- Audio sent at normal speed (1x) -> no events
- Sending faster than real-time -> RateLimitAction
- Excessive backlog -> FramesDroppedAction
- Counters for received and dropped frames
- Return to normal after slowdown

All tests use injectable clock for determinism.
"""

from __future__ import annotations

from macaw.session.backpressure import (
    BackpressureController,
    FramesDroppedAction,
    RateLimitAction,
)
from tests.helpers import SAMPLE_RATE

# PCM 16-bit mono 16kHz: 1 segundo = 32000 bytes
_BYTES_PER_SECOND = SAMPLE_RATE * 2  # 2 bytes per sample (int16)
_BYTES_PER_20MS = _BYTES_PER_SECOND // 50  # 640 bytes = 20ms frame


class _FakeClock:
    """Fake clock for deterministic tests.

    Allows manual time advancement without depending on time.monotonic().
    """

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        """Advance the clock by `seconds` seconds."""
        self._now += seconds


class TestBackpressureControllerNormalSpeed:
    """Tests with audio at normal speed (1x real-time)."""

    def test_first_frame_never_triggers(self) -> None:
        """First frame should never emit an action (no history)."""
        clock = _FakeClock()
        ctrl = BackpressureController(
            sample_rate=SAMPLE_RATE,
            clock=clock,
        )

        result = ctrl.record_frame(_BYTES_PER_20MS)

        assert result is None
        assert ctrl.frames_received == 1
        assert ctrl.frames_dropped == 0

    def test_normal_speed_no_events(self) -> None:
        """Audio at 1x real-time speed should not emit events."""
        clock = _FakeClock()
        ctrl = BackpressureController(
            sample_rate=SAMPLE_RATE,
            clock=clock,
        )

        # Send 100 frames of 20ms at real-time speed
        for _ in range(100):
            result = ctrl.record_frame(_BYTES_PER_20MS)
            assert result is None
            clock.advance(0.020)  # 20ms de wall-clock por frame

        assert ctrl.frames_received == 100
        assert ctrl.frames_dropped == 0

    def test_slightly_fast_below_threshold_no_events(self) -> None:
        """Audio at 1.1x (below threshold 1.2x) should not emit events."""
        clock = _FakeClock()
        ctrl = BackpressureController(
            sample_rate=SAMPLE_RATE,
            rate_limit_threshold=1.2,
            clock=clock,
        )

        # Send 20ms audio frames with 18.18ms wall-clock (~1.1x)
        for _ in range(100):
            result = ctrl.record_frame(_BYTES_PER_20MS)
            assert result is None
            clock.advance(0.01818)  # ~1.1x real-time

        assert ctrl.frames_dropped == 0


class TestBackpressureControllerRateLimit:
    """Tests for rate detection above threshold."""

    def test_fast_sending_triggers_rate_limit(self) -> None:
        """Audio at 2x real-time should emit RateLimitAction."""
        clock = _FakeClock()
        ctrl = BackpressureController(
            sample_rate=SAMPLE_RATE,
            rate_limit_threshold=1.2,
            max_backlog_s=10.0,
            clock=clock,
        )

        # Send 20ms audio frames with 10ms wall-clock (2x real-time)
        # Need wall_elapsed >= 0.5s for rate check to start,
        # so we send 100 frames (99 * 10ms = 0.99s wall-clock)
        actions: list[RateLimitAction] = []
        for _ in range(100):
            result = ctrl.record_frame(_BYTES_PER_20MS)
            if isinstance(result, RateLimitAction):
                actions.append(result)
            clock.advance(0.010)  # 10ms wall-clock = 2x real-time

        assert len(actions) >= 1, "Should emit at least 1 RateLimitAction"
        assert all(isinstance(a, RateLimitAction) for a in actions)
        assert all(a.delay_ms >= 1 for a in actions)

    def test_rate_limit_has_positive_delay(self) -> None:
        """RateLimitAction should have positive delay_ms."""
        clock = _FakeClock()
        ctrl = BackpressureController(
            sample_rate=SAMPLE_RATE,
            rate_limit_threshold=1.2,
            clock=clock,
        )

        # Send fast enough to trigger
        # wall_elapsed >= 0.5s needed: 200 frames * 5ms = 1.0s
        actions: list[RateLimitAction] = []
        for _ in range(200):
            result = ctrl.record_frame(_BYTES_PER_20MS)
            if isinstance(result, RateLimitAction):
                actions.append(result)
            clock.advance(0.005)  # 5ms wall-clock = 4x real-time

        assert len(actions) >= 1
        for action in actions:
            assert action.delay_ms >= 1

    def test_rate_returns_to_normal_after_slowdown(self) -> None:
        """After slowing down to 1x, should no longer emit rate_limit."""
        clock = _FakeClock()
        ctrl = BackpressureController(
            sample_rate=SAMPLE_RATE,
            rate_limit_threshold=1.2,
            max_backlog_s=100.0,  # backlog alto para nao dropar
            clock=clock,
        )

        # Phase 1: send fast (2x) for 2 seconds
        for _ in range(100):
            ctrl.record_frame(_BYTES_PER_20MS)
            clock.advance(0.010)  # 2x

        # Phase 2: advance clock so the sliding window "forgets" the burst
        clock.advance(6.0)

        # Phase 3: send at normal speed (1x) for 3 seconds
        actions_normal: list[RateLimitAction] = []
        for _ in range(150):
            result = ctrl.record_frame(_BYTES_PER_20MS)
            if isinstance(result, RateLimitAction):
                actions_normal.append(result)
            clock.advance(0.020)  # 1x real-time

        assert len(actions_normal) == 0, "Should not emit rate_limit after slowing down to 1x"


class TestBackpressureControllerFramesDrop:
    """Tests for frame drop due to excessive backlog."""

    def test_backlog_exceeds_max_triggers_drop(self) -> None:
        """Backlog > max_backlog_s should drop frames."""
        clock = _FakeClock()
        ctrl = BackpressureController(
            sample_rate=SAMPLE_RATE,
            max_backlog_s=1.0,  # apenas 1 segundo de backlog
            rate_limit_threshold=1.2,
            clock=clock,
        )

        # Send many frames instantaneously (wall-clock does not advance)
        # Each frame = 20ms of audio. 1s backlog = 50 frames without wall-clock.
        # After 50 frames (1s of audio) without advancing clock, backlog = 1s
        # First frame does not count (initializes), so 52 frames = ~1.02s
        drop_action = None
        for _ in range(100):
            result = ctrl.record_frame(_BYTES_PER_20MS)
            if isinstance(result, FramesDroppedAction):
                drop_action = result
                break
            # Do not advance clock! Simulates instantaneous burst

        assert drop_action is not None, "Should emit FramesDroppedAction"
        assert isinstance(drop_action, FramesDroppedAction)
        assert drop_action.dropped_ms > 0

    def test_dropped_frames_counter_incremented(self) -> None:
        """Dropped frames counter should be incremented correctly."""
        clock = _FakeClock()
        ctrl = BackpressureController(
            sample_rate=SAMPLE_RATE,
            max_backlog_s=0.5,  # meio segundo de backlog
            clock=clock,
        )

        # Send many frames without advancing clock
        total_drops = 0
        for _ in range(200):
            result = ctrl.record_frame(_BYTES_PER_20MS)
            if isinstance(result, FramesDroppedAction):
                total_drops += 1

        assert ctrl.frames_dropped == total_drops
        assert ctrl.frames_dropped > 0
        assert ctrl.frames_received == 200

    def test_dropped_ms_reflects_frame_duration(self) -> None:
        """dropped_ms should reflect the duration of the dropped frame."""
        clock = _FakeClock()
        ctrl = BackpressureController(
            sample_rate=SAMPLE_RATE,
            max_backlog_s=0.5,
            clock=clock,
        )

        # Fill backlog until overflow
        drop_action = None
        for _ in range(200):
            result = ctrl.record_frame(_BYTES_PER_20MS)
            if isinstance(result, FramesDroppedAction):
                drop_action = result
                break

        assert drop_action is not None
        # 20ms frame = 640 bytes = 320 samples = 20ms
        assert drop_action.dropped_ms == 20

    def test_backlog_within_limit_no_drop(self) -> None:
        """Backlog within limit should not drop frames."""
        clock = _FakeClock()
        ctrl = BackpressureController(
            sample_rate=SAMPLE_RATE,
            max_backlog_s=10.0,
            clock=clock,
        )

        # Send 5 seconds of audio instantaneously (5s < 10s backlog)
        for _ in range(250):  # 250 * 20ms = 5s
            result = ctrl.record_frame(_BYTES_PER_20MS)
            assert not isinstance(result, FramesDroppedAction)

        assert ctrl.frames_dropped == 0


class TestBackpressureControllerEdgeCases:
    """Tests for edge cases and boundary scenarios."""

    def test_single_frame_no_action(self) -> None:
        """A single frame should not emit any action."""
        clock = _FakeClock()
        ctrl = BackpressureController(
            sample_rate=SAMPLE_RATE,
            clock=clock,
        )

        result = ctrl.record_frame(_BYTES_PER_20MS)

        assert result is None

    def test_two_frames_same_time_no_crash(self) -> None:
        """Two frames at the same instant should not crash (division by zero)."""
        clock = _FakeClock()
        ctrl = BackpressureController(
            sample_rate=SAMPLE_RATE,
            max_backlog_s=10.0,
            clock=clock,
        )

        ctrl.record_frame(_BYTES_PER_20MS)
        result = ctrl.record_frame(_BYTES_PER_20MS)

        # Should not crash; result depends on backlog
        assert result is None or isinstance(result, RateLimitAction | FramesDroppedAction)

    def test_counters_start_at_zero(self) -> None:
        """Counters should start at zero."""
        ctrl = BackpressureController(sample_rate=SAMPLE_RATE)

        assert ctrl.frames_received == 0
        assert ctrl.frames_dropped == 0

    def test_large_frame_counted_correctly(self) -> None:
        """Large frames should have duration calculated correctly."""
        clock = _FakeClock()
        ctrl = BackpressureController(
            sample_rate=SAMPLE_RATE,
            max_backlog_s=2.0,
            clock=clock,
        )

        # Frame de 1 segundo = 32000 bytes
        one_sec_frame = _BYTES_PER_SECOND
        ctrl.record_frame(one_sec_frame)  # primeiro, inicializa
        clock.advance(0.001)  # quase instantaneo

        # Segundo frame de 1s: backlog sera ~2s em 0.001s de wall
        ctrl.record_frame(one_sec_frame)

        # Verify it did not crash and counters are correct
        assert ctrl.frames_received == 2

    def test_rate_limit_cooldown_prevents_spam(self) -> None:
        """Rate limit should respect 1s cooldown between emissions."""
        clock = _FakeClock()
        ctrl = BackpressureController(
            sample_rate=SAMPLE_RATE,
            rate_limit_threshold=1.2,
            max_backlog_s=100.0,  # alto para nao dropar
            clock=clock,
        )

        # Enviar a 3x real-time por 0.5s (25 frames de 20ms, 6.66ms wall cada)
        rate_limits = 0
        for _ in range(25):
            result = ctrl.record_frame(_BYTES_PER_20MS)
            if isinstance(result, RateLimitAction):
                rate_limits += 1
            clock.advance(0.00666)  # ~3x real-time

        # Cooldown de 1s: em 0.5s de wall-clock, deve emitir no maximo 1
        assert rate_limits <= 1

    def test_frames_dropped_priority_over_rate_limit(self) -> None:
        """FramesDroppedAction should have priority over RateLimitAction."""
        clock = _FakeClock()
        ctrl = BackpressureController(
            sample_rate=SAMPLE_RATE,
            max_backlog_s=0.5,
            rate_limit_threshold=1.2,
            clock=clock,
        )

        # Fill backlog quickly
        last_action = None
        for _ in range(100):
            result = ctrl.record_frame(_BYTES_PER_20MS)
            if result is not None:
                last_action = result
            # Do not advance clock

        # The last result after backlog overflow should be FramesDroppedAction
        # (backlog is checked before rate_limit in the code)
        assert isinstance(last_action, FramesDroppedAction)
