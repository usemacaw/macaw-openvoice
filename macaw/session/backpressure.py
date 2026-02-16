"""BackpressureController -- controls audio ingestion rate.

Monitors the rate of incoming audio frames relative to real time
(wall-clock). If the client sends audio faster than real-time, emits
rate_limit or frames_dropped actions to the WebSocket handler.

Behavior:
- Rate > rate_limit_threshold (default 1.2x): emit RateLimitAction
- Accumulated backlog > max_backlog_s (default 10s): emit FramesDroppedAction
- Audio at normal speed (1x) NEVER triggers events
- First frame never triggers (no history to compare)

The rate calculation uses accumulated backlog (audio_total - wall_elapsed)
as the main indicator. If backlog grows above the threshold
(rate_limit_threshold - 1.0) * wall_elapsed, the client is sending
faster than allowed.

Additionally, a 5s sliding window is used to check recent bursts
without penalizing older history.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from macaw._audio_constants import BYTES_PER_SAMPLE_INT16, STT_SAMPLE_RATE

if TYPE_CHECKING:
    from collections.abc import Callable

# Sliding window size for rate calculation (seconds)
_SLIDING_WINDOW_S = 5.0

# Minimum wall-clock time before starting rate checks (seconds)
# Avoids false positives in the first moments
_MIN_WALL_FOR_RATE_CHECK_S = 0.5

# Minimum interval between RateLimitAction emissions (seconds)
_RATE_LIMIT_COOLDOWN_S = 1.0


@dataclass(frozen=True, slots=True)
class RateLimitAction:
    """Action: client should slow down audio sending.

    Emitted when send rate exceeds rate_limit_threshold.
    The delay_ms field suggests how long the client should wait
    before sending the next frame.
    """

    delay_ms: int


@dataclass(frozen=True, slots=True)
class FramesDroppedAction:
    """Action: frames were dropped due to excessive backlog.

    Emitted when accumulated unprocessed audio exceeds max_backlog_s.
    The dropped_ms field indicates how much audio (in ms) was dropped.
    """

    dropped_ms: int


BackpressureAction = RateLimitAction | FramesDroppedAction


class BackpressureController:
    """Control backpressure for streaming audio ingestion.

    Monitors frame receive rate relative to real time.
    Uses accumulated backlog and sliding window to detect sending
    faster than real-time.

    Args:
        sample_rate: Audio sample rate in Hz (default: 16000).
        max_backlog_s: Maximum accumulated audio seconds before
            dropping frames (default: 10.0).
        rate_limit_threshold: Factor above which to emit rate_limit.
            1.2 means 120% of real-time (default: 1.2).
        clock: Clock function for test injection.
            Default: time.monotonic.
    """

    def __init__(
        self,
        sample_rate: int = STT_SAMPLE_RATE,
        max_backlog_s: float = 10.0,
        rate_limit_threshold: float = 1.2,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._sample_rate = sample_rate
        self._max_backlog_s = max_backlog_s
        self._rate_limit_threshold = rate_limit_threshold
        self._clock: Callable[[], float] = clock if clock is not None else time.monotonic

        # Counters
        self._frames_received = 0
        self._frames_dropped = 0

        # Sliding window: deque of (wall_time, audio_duration_s)
        self._window: deque[tuple[float, float]] = deque()

        # Start time and total accumulated audio
        self._start_time: float | None = None
        self._total_audio_s: float = 0.0

        # First frame duration (for rate correction)
        self._first_frame_duration_s: float = 0.0

        # Cooldown for rate_limit (-inf ensures first trigger works)
        self._last_rate_limit_time: float = float("-inf")

    @property
    def frames_received(self) -> int:
        """Total frames received (including dropped)."""
        return self._frames_received

    @property
    def frames_dropped(self) -> int:
        """Total frames dropped due to excessive backlog."""
        return self._frames_dropped

    def record_frame(self, frame_bytes: int) -> BackpressureAction | None:
        """Record receipt of an audio frame.

        Calculates frame duration in seconds based on byte size
        and sample_rate. Checks backlog and rate to decide whether
        backpressure is needed.

        Args:
            frame_bytes: Frame size in bytes (16-bit PCM mono).

        Returns:
            BackpressureAction if backpressure is needed, None otherwise.
            - RateLimitAction: rate above threshold, suggests delay_ms
            - FramesDroppedAction: backlog exceeded, frames dropped
        """
        now: float = self._clock()
        self._frames_received += 1

        # Frame duration in seconds
        n_samples = frame_bytes / BYTES_PER_SAMPLE_INT16
        frame_duration_s = n_samples / self._sample_rate

        # First frame: initialize and return (no history to compare)
        if self._start_time is None:
            self._start_time = now
            self._total_audio_s = frame_duration_s
            self._first_frame_duration_s = frame_duration_s
            self._window.append((now, frame_duration_s))
            return None

        # Update accumulators
        self._total_audio_s += frame_duration_s
        wall_elapsed = now - self._start_time
        self._window.append((now, frame_duration_s))

        # Prune sliding window: remove entries older than _SLIDING_WINDOW_S
        cutoff = now - _SLIDING_WINDOW_S
        while self._window and self._window[0][0] < cutoff:
            self._window.popleft()

        # Check backlog (total audio - elapsed wall time)
        # For the first frame, wall_elapsed=0 and total_audio=frame_duration,
        # so backlog = frame_duration (~0.02s). This is normal.
        backlog_s = self._total_audio_s - wall_elapsed
        if backlog_s > self._max_backlog_s:
            # Drop this frame
            self._frames_dropped += 1
            # Reduce accumulated audio by dropped amount
            self._total_audio_s -= frame_duration_s
            dropped_ms = int(frame_duration_s * 1000)
            return FramesDroppedAction(dropped_ms=dropped_ms)

        # Check rate only after minimum wall_elapsed
        # Avoids false positives early on (few frames,
        # imprecise ratio)
        if wall_elapsed < _MIN_WALL_FOR_RATE_CHECK_S:
            return None

        # Calculate rate: audio_total / wall_elapsed
        # For 1x audio: audio ~= wall, rate ~= 1.0
        # For 2x audio: audio ~= 2*wall, rate ~= 2.0
        # The first frame contributes frame_duration of audio "for free"
        # (no corresponding wall-time), so subtract the duration of the
        # first frame (constant) to correct.
        effective_audio = self._total_audio_s - self._first_frame_duration_s
        if wall_elapsed <= 0:
            return None
        rate = effective_audio / wall_elapsed

        if rate > self._rate_limit_threshold:
            # Cooldown: do not emit rate_limit too frequently
            if (now - self._last_rate_limit_time) < _RATE_LIMIT_COOLDOWN_S:
                return None

            self._last_rate_limit_time = now

            # Suggest delay for the client to return to real-time
            excess_rate = rate - 1.0
            delay_ms = max(1, int(excess_rate * frame_duration_s * 1000))
            return RateLimitAction(delay_ms=delay_ms)

        return None
