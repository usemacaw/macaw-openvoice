"""SessionStateMachine — state machine for STT streaming sessions.

Implements 6 states with valid transitions, configurable per-state timeouts,
and on_enter/on_exit callbacks. Pure, synchronous component — does not know
gRPC, WebSocket, or asyncio. The caller (StreamingSession) is responsible for
calling transition() and check_timeout() at the right times.

States:
    INIT -> ACTIVE -> SILENCE -> HOLD -> CLOSING -> CLOSED

Rules:
- CLOSED is terminal: no transitions are accepted from CLOSED.
- Any state can transition to CLOSED (unrecoverable error).
- Invalid transitions raise InvalidTransitionError.
- Timeouts are checked via check_timeout() (O(1), no timers/tasks).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from macaw._types import SessionState
from macaw.exceptions import InvalidTransitionError

if TYPE_CHECKING:
    from collections.abc import Callable

# Valid transitions: {current_state: {allowed_target_states}}
_VALID_TRANSITIONS: dict[SessionState, frozenset[SessionState]] = {
    SessionState.INIT: frozenset({SessionState.ACTIVE, SessionState.CLOSED}),
    SessionState.ACTIVE: frozenset(
        {SessionState.SILENCE, SessionState.CLOSING, SessionState.CLOSED}
    ),
    SessionState.SILENCE: frozenset(
        {SessionState.ACTIVE, SessionState.HOLD, SessionState.CLOSING, SessionState.CLOSED}
    ),
    SessionState.HOLD: frozenset({SessionState.ACTIVE, SessionState.CLOSING, SessionState.CLOSED}),
    SessionState.CLOSING: frozenset({SessionState.CLOSED}),
    SessionState.CLOSED: frozenset(),
}

# Timeouts: {state -> target_state_when_expires}
_TIMEOUT_TARGETS: dict[SessionState, SessionState] = {
    SessionState.INIT: SessionState.CLOSED,
    SessionState.SILENCE: SessionState.HOLD,
    SessionState.HOLD: SessionState.CLOSING,
    SessionState.CLOSING: SessionState.CLOSED,
}


_MIN_TIMEOUT_S = 1.0


@dataclass(frozen=True, slots=True)
class SessionTimeouts:
    """Configurable timeouts per state (in seconds).

    PRD defaults:
        INIT: 30s (no audio -> CLOSED)
        SILENCE: 30s (no speech -> HOLD)
        HOLD: 5min (prolonged silence -> CLOSING)
        CLOSING: 2s (flush pending -> CLOSED)

    Raises:
        ValueError: If any timeout is less than 1.0s.
    """

    init_timeout_s: float = 30.0
    silence_timeout_s: float = 30.0
    hold_timeout_s: float = 300.0
    closing_timeout_s: float = 2.0

    def __post_init__(self) -> None:
        for field_name in (
            "init_timeout_s",
            "silence_timeout_s",
            "hold_timeout_s",
            "closing_timeout_s",
        ):
            value = getattr(self, field_name)
            if value < _MIN_TIMEOUT_S:
                msg = f"Timeout '{field_name}' must be >= {_MIN_TIMEOUT_S}s, got {value}s"
                raise ValueError(msg)

    def get_timeout_for_state(self, state: SessionState) -> float | None:
        """Return the timeout in seconds for the state, or None if no timeout."""
        state_timeout_map = {
            SessionState.INIT: self.init_timeout_s,
            SessionState.SILENCE: self.silence_timeout_s,
            SessionState.HOLD: self.hold_timeout_s,
            SessionState.CLOSING: self.closing_timeout_s,
        }
        return state_timeout_map.get(state)


def timeouts_from_configure_command(
    current: SessionTimeouts,
    *,
    silence_timeout_ms: int | None = None,
    hold_timeout_ms: int | None = None,
) -> SessionTimeouts:
    """Create updated SessionTimeouts from SessionConfigureCommand fields.

    Converts millisecond values (WebSocket protocol) to seconds (state machine).
    None fields keep the current value.

    Args:
        current: Current timeouts.
        silence_timeout_ms: Silence timeout in ms (None = keep current).
        hold_timeout_ms: Hold timeout in ms (None = keep current).

    Returns:
        New SessionTimeouts with updated values.

    Raises:
        ValueError: If any converted timeout is less than 1.0s.
    """
    return SessionTimeouts(
        init_timeout_s=current.init_timeout_s,
        silence_timeout_s=(
            silence_timeout_ms / 1000.0
            if silence_timeout_ms is not None
            else current.silence_timeout_s
        ),
        hold_timeout_s=(
            hold_timeout_ms / 1000.0 if hold_timeout_ms is not None else current.hold_timeout_s
        ),
        closing_timeout_s=current.closing_timeout_s,
    )


class SessionStateMachine:
    """State machine for STT streaming sessions.

    Manages 6 states with valid transitions, per-state timeouts,
    and on_enter/on_exit callbacks.

    Args:
        timeouts: Configurable timeouts per state.
        on_enter: Callbacks called upon ENTERING a state.
        on_exit: Callbacks called upon EXITING a state.
        clock: Function that returns a monotonic timestamp (for deterministic tests).
    """

    def __init__(
        self,
        timeouts: SessionTimeouts | None = None,
        on_enter: dict[SessionState, Callable[[], None]] | None = None,
        on_exit: dict[SessionState, Callable[[], None]] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._state = SessionState.INIT
        self._timeouts = timeouts or _default_session_timeouts()
        self._on_enter = on_enter or {}
        self._on_exit = on_exit or {}
        self._clock = clock or time.monotonic
        self._state_entered_at = self._clock()

    @property
    def state(self) -> SessionState:
        """Current session state."""
        return self._state

    @property
    def elapsed_in_state_ms(self) -> int:
        """Time (ms) the session has been in the current state."""
        elapsed_s = self._clock() - self._state_entered_at
        return int(elapsed_s * 1000)

    @property
    def timeouts(self) -> SessionTimeouts:
        """Configured timeouts."""
        return self._timeouts

    def transition(self, target: SessionState) -> None:
        """Transition to the target state.

        Args:
            target: Target state for the transition.

        Raises:
            InvalidTransitionError: If the transition is invalid.
        """
        if target not in _VALID_TRANSITIONS[self._state]:
            raise InvalidTransitionError(self._state.value, target.value)

        previous = self._state

        # on_exit callback for previous state
        exit_cb = self._on_exit.get(previous)
        if exit_cb is not None:
            exit_cb()

        # Transition
        self._state = target
        self._state_entered_at = self._clock()

        # on_enter callback for new state
        enter_cb = self._on_enter.get(target)
        if enter_cb is not None:
            enter_cb()

    def check_timeout(self) -> SessionState | None:
        """Check whether the current state's timeout has expired.

        Returns:
            Target state if timeout expired, None otherwise.
            The caller is responsible for calling transition() with the result.

        Complexity: O(1) — just compare elapsed vs timeout.
        """
        timeout_s = self._timeouts.get_timeout_for_state(self._state)
        if timeout_s is None:
            return None

        elapsed_s = self._clock() - self._state_entered_at
        if elapsed_s >= timeout_s:
            return _TIMEOUT_TARGETS.get(self._state)

        return None

    def update_timeouts(self, timeouts: SessionTimeouts) -> None:
        """Update state machine timeouts.

        Affects the current state: if the current state's timeout changed, the new
        value is considered immediately on the next check_timeout() call.

        Args:
            timeouts: New timeouts.
        """
        self._timeouts = timeouts


def _default_session_timeouts() -> SessionTimeouts:
    """Build SessionTimeouts from environment-configurable settings.

    Falls back to dataclass defaults if settings are unavailable
    (e.g., in unit tests that don't initialize MacawSettings).
    """
    try:
        from macaw.config.settings import get_settings

        s = get_settings().session
        return SessionTimeouts(
            init_timeout_s=s.init_timeout_s,
            silence_timeout_s=s.silence_timeout_s,
            hold_timeout_s=s.hold_timeout_s,
            closing_timeout_s=s.closing_timeout_s,
        )
    except ImportError:
        return SessionTimeouts()
    except Exception:
        import logging

        logging.getLogger(__name__).warning(
            "Failed to load session timeouts from settings, using defaults",
            exc_info=True,
        )
        return SessionTimeouts()
