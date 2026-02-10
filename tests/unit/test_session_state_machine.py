"""Testes unitarios para SessionStateMachine.

Cobre: transicoes validas, transicoes invalidas, timeouts de cada estado,
callbacks on_enter/on_exit, estado CLOSED terminal, clock injetavel, e
elapsed_in_state_ms.
"""

from __future__ import annotations

import pytest

from macaw._types import SessionState
from macaw.exceptions import InvalidTransitionError
from macaw.session.state_machine import SessionStateMachine, SessionTimeouts

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeClock:
    """Clock determinisico para testes."""

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


# ---------------------------------------------------------------------------
# Estado inicial
# ---------------------------------------------------------------------------


class TestInitialState:
    def test_initial_state_is_init(self) -> None:
        sm = SessionStateMachine()
        assert sm.state == SessionState.INIT

    def test_elapsed_starts_at_zero(self) -> None:
        clock = FakeClock()
        sm = SessionStateMachine(clock=clock)
        assert sm.elapsed_in_state_ms == 0


# ---------------------------------------------------------------------------
# Transicoes validas
# ---------------------------------------------------------------------------


class TestValidTransitions:
    def test_init_to_active(self) -> None:
        sm = SessionStateMachine()
        sm.transition(SessionState.ACTIVE)
        assert sm.state == SessionState.ACTIVE

    def test_init_to_closed(self) -> None:
        sm = SessionStateMachine()
        sm.transition(SessionState.CLOSED)
        assert sm.state == SessionState.CLOSED

    def test_active_to_silence(self) -> None:
        sm = SessionStateMachine()
        sm.transition(SessionState.ACTIVE)
        sm.transition(SessionState.SILENCE)
        assert sm.state == SessionState.SILENCE

    def test_active_to_closing(self) -> None:
        sm = SessionStateMachine()
        sm.transition(SessionState.ACTIVE)
        sm.transition(SessionState.CLOSING)
        assert sm.state == SessionState.CLOSING

    def test_active_to_closed(self) -> None:
        sm = SessionStateMachine()
        sm.transition(SessionState.ACTIVE)
        sm.transition(SessionState.CLOSED)
        assert sm.state == SessionState.CLOSED

    def test_silence_to_active(self) -> None:
        sm = SessionStateMachine()
        sm.transition(SessionState.ACTIVE)
        sm.transition(SessionState.SILENCE)
        sm.transition(SessionState.ACTIVE)
        assert sm.state == SessionState.ACTIVE

    def test_silence_to_hold(self) -> None:
        sm = SessionStateMachine()
        sm.transition(SessionState.ACTIVE)
        sm.transition(SessionState.SILENCE)
        sm.transition(SessionState.HOLD)
        assert sm.state == SessionState.HOLD

    def test_silence_to_closing(self) -> None:
        sm = SessionStateMachine()
        sm.transition(SessionState.ACTIVE)
        sm.transition(SessionState.SILENCE)
        sm.transition(SessionState.CLOSING)
        assert sm.state == SessionState.CLOSING

    def test_silence_to_closed(self) -> None:
        sm = SessionStateMachine()
        sm.transition(SessionState.ACTIVE)
        sm.transition(SessionState.SILENCE)
        sm.transition(SessionState.CLOSED)
        assert sm.state == SessionState.CLOSED

    def test_hold_to_active(self) -> None:
        sm = SessionStateMachine()
        sm.transition(SessionState.ACTIVE)
        sm.transition(SessionState.SILENCE)
        sm.transition(SessionState.HOLD)
        sm.transition(SessionState.ACTIVE)
        assert sm.state == SessionState.ACTIVE

    def test_hold_to_closing(self) -> None:
        sm = SessionStateMachine()
        sm.transition(SessionState.ACTIVE)
        sm.transition(SessionState.SILENCE)
        sm.transition(SessionState.HOLD)
        sm.transition(SessionState.CLOSING)
        assert sm.state == SessionState.CLOSING

    def test_hold_to_closed(self) -> None:
        sm = SessionStateMachine()
        sm.transition(SessionState.ACTIVE)
        sm.transition(SessionState.SILENCE)
        sm.transition(SessionState.HOLD)
        sm.transition(SessionState.CLOSED)
        assert sm.state == SessionState.CLOSED

    def test_closing_to_closed(self) -> None:
        sm = SessionStateMachine()
        sm.transition(SessionState.ACTIVE)
        sm.transition(SessionState.CLOSING)
        sm.transition(SessionState.CLOSED)
        assert sm.state == SessionState.CLOSED


# ---------------------------------------------------------------------------
# Transicoes invalidas
# ---------------------------------------------------------------------------


class TestInvalidTransitions:
    def test_init_to_silence_raises(self) -> None:
        sm = SessionStateMachine()
        with pytest.raises(InvalidTransitionError) as exc_info:
            sm.transition(SessionState.SILENCE)
        assert exc_info.value.from_state == "init"
        assert exc_info.value.to_state == "silence"

    def test_init_to_hold_raises(self) -> None:
        sm = SessionStateMachine()
        with pytest.raises(InvalidTransitionError):
            sm.transition(SessionState.HOLD)

    def test_init_to_closing_raises(self) -> None:
        sm = SessionStateMachine()
        with pytest.raises(InvalidTransitionError):
            sm.transition(SessionState.CLOSING)

    def test_active_to_hold_raises(self) -> None:
        sm = SessionStateMachine()
        sm.transition(SessionState.ACTIVE)
        with pytest.raises(InvalidTransitionError):
            sm.transition(SessionState.HOLD)

    def test_active_to_init_raises(self) -> None:
        sm = SessionStateMachine()
        sm.transition(SessionState.ACTIVE)
        with pytest.raises(InvalidTransitionError):
            sm.transition(SessionState.INIT)

    def test_silence_to_init_raises(self) -> None:
        sm = SessionStateMachine()
        sm.transition(SessionState.ACTIVE)
        sm.transition(SessionState.SILENCE)
        with pytest.raises(InvalidTransitionError):
            sm.transition(SessionState.INIT)

    def test_hold_to_silence_raises(self) -> None:
        sm = SessionStateMachine()
        sm.transition(SessionState.ACTIVE)
        sm.transition(SessionState.SILENCE)
        sm.transition(SessionState.HOLD)
        with pytest.raises(InvalidTransitionError):
            sm.transition(SessionState.SILENCE)

    def test_hold_to_init_raises(self) -> None:
        sm = SessionStateMachine()
        sm.transition(SessionState.ACTIVE)
        sm.transition(SessionState.SILENCE)
        sm.transition(SessionState.HOLD)
        with pytest.raises(InvalidTransitionError):
            sm.transition(SessionState.INIT)

    def test_closing_to_active_raises(self) -> None:
        sm = SessionStateMachine()
        sm.transition(SessionState.ACTIVE)
        sm.transition(SessionState.CLOSING)
        with pytest.raises(InvalidTransitionError):
            sm.transition(SessionState.ACTIVE)

    def test_closing_to_init_raises(self) -> None:
        sm = SessionStateMachine()
        sm.transition(SessionState.ACTIVE)
        sm.transition(SessionState.CLOSING)
        with pytest.raises(InvalidTransitionError):
            sm.transition(SessionState.INIT)


# ---------------------------------------------------------------------------
# Estado CLOSED e terminal
# ---------------------------------------------------------------------------


class TestClosedIsTerminal:
    def test_closed_to_init_raises(self) -> None:
        sm = SessionStateMachine()
        sm.transition(SessionState.CLOSED)
        with pytest.raises(InvalidTransitionError):
            sm.transition(SessionState.INIT)

    def test_closed_to_active_raises(self) -> None:
        sm = SessionStateMachine()
        sm.transition(SessionState.CLOSED)
        with pytest.raises(InvalidTransitionError):
            sm.transition(SessionState.ACTIVE)

    def test_closed_to_silence_raises(self) -> None:
        sm = SessionStateMachine()
        sm.transition(SessionState.CLOSED)
        with pytest.raises(InvalidTransitionError):
            sm.transition(SessionState.SILENCE)

    def test_closed_to_hold_raises(self) -> None:
        sm = SessionStateMachine()
        sm.transition(SessionState.CLOSED)
        with pytest.raises(InvalidTransitionError):
            sm.transition(SessionState.HOLD)

    def test_closed_to_closing_raises(self) -> None:
        sm = SessionStateMachine()
        sm.transition(SessionState.CLOSED)
        with pytest.raises(InvalidTransitionError):
            sm.transition(SessionState.CLOSING)

    def test_closed_to_closed_raises(self) -> None:
        sm = SessionStateMachine()
        sm.transition(SessionState.CLOSED)
        with pytest.raises(InvalidTransitionError):
            sm.transition(SessionState.CLOSED)


# ---------------------------------------------------------------------------
# Qualquer estado -> CLOSED (erro irrecuperavel)
# ---------------------------------------------------------------------------


class TestAnyStateToClosed:
    def test_init_to_closed(self) -> None:
        sm = SessionStateMachine()
        sm.transition(SessionState.CLOSED)
        assert sm.state == SessionState.CLOSED

    def test_active_to_closed(self) -> None:
        sm = SessionStateMachine()
        sm.transition(SessionState.ACTIVE)
        sm.transition(SessionState.CLOSED)
        assert sm.state == SessionState.CLOSED

    def test_silence_to_closed(self) -> None:
        sm = SessionStateMachine()
        sm.transition(SessionState.ACTIVE)
        sm.transition(SessionState.SILENCE)
        sm.transition(SessionState.CLOSED)
        assert sm.state == SessionState.CLOSED

    def test_hold_to_closed(self) -> None:
        sm = SessionStateMachine()
        sm.transition(SessionState.ACTIVE)
        sm.transition(SessionState.SILENCE)
        sm.transition(SessionState.HOLD)
        sm.transition(SessionState.CLOSED)
        assert sm.state == SessionState.CLOSED

    def test_closing_to_closed(self) -> None:
        sm = SessionStateMachine()
        sm.transition(SessionState.ACTIVE)
        sm.transition(SessionState.CLOSING)
        sm.transition(SessionState.CLOSED)
        assert sm.state == SessionState.CLOSED


# ---------------------------------------------------------------------------
# Timeouts
# ---------------------------------------------------------------------------


class TestTimeouts:
    def test_init_timeout_returns_closed(self) -> None:
        clock = FakeClock()
        sm = SessionStateMachine(clock=clock)
        assert sm.check_timeout() is None
        clock.advance(30.0)
        assert sm.check_timeout() == SessionState.CLOSED

    def test_silence_timeout_returns_hold(self) -> None:
        clock = FakeClock()
        sm = SessionStateMachine(clock=clock)
        sm.transition(SessionState.ACTIVE)
        sm.transition(SessionState.SILENCE)
        assert sm.check_timeout() is None
        clock.advance(30.0)
        assert sm.check_timeout() == SessionState.HOLD

    def test_hold_timeout_returns_closing(self) -> None:
        clock = FakeClock()
        sm = SessionStateMachine(clock=clock)
        sm.transition(SessionState.ACTIVE)
        sm.transition(SessionState.SILENCE)
        sm.transition(SessionState.HOLD)
        assert sm.check_timeout() is None
        clock.advance(300.0)
        assert sm.check_timeout() == SessionState.CLOSING

    def test_closing_timeout_returns_closed(self) -> None:
        clock = FakeClock()
        sm = SessionStateMachine(clock=clock)
        sm.transition(SessionState.ACTIVE)
        sm.transition(SessionState.CLOSING)
        assert sm.check_timeout() is None
        clock.advance(2.0)
        assert sm.check_timeout() == SessionState.CLOSED

    def test_active_has_no_timeout(self) -> None:
        clock = FakeClock()
        sm = SessionStateMachine(clock=clock)
        sm.transition(SessionState.ACTIVE)
        clock.advance(9999.0)
        assert sm.check_timeout() is None

    def test_closed_has_no_timeout(self) -> None:
        clock = FakeClock()
        sm = SessionStateMachine(clock=clock)
        sm.transition(SessionState.CLOSED)
        clock.advance(9999.0)
        assert sm.check_timeout() is None

    def test_timeout_not_triggered_before_expiry(self) -> None:
        clock = FakeClock()
        sm = SessionStateMachine(clock=clock)
        clock.advance(29.9)
        assert sm.check_timeout() is None

    def test_timeout_triggered_at_exact_boundary(self) -> None:
        clock = FakeClock()
        sm = SessionStateMachine(clock=clock)
        clock.advance(30.0)
        assert sm.check_timeout() == SessionState.CLOSED

    def test_custom_timeouts(self) -> None:
        clock = FakeClock()
        timeouts = SessionTimeouts(init_timeout_s=5.0)
        sm = SessionStateMachine(timeouts=timeouts, clock=clock)
        clock.advance(4.9)
        assert sm.check_timeout() is None
        clock.advance(0.1)
        assert sm.check_timeout() == SessionState.CLOSED

    def test_timeout_resets_on_transition(self) -> None:
        clock = FakeClock()
        sm = SessionStateMachine(clock=clock)
        clock.advance(25.0)  # 25s em INIT (nao expirou)
        sm.transition(SessionState.ACTIVE)
        sm.transition(SessionState.SILENCE)
        # Agora em SILENCE, timeout e 30s a partir de agora
        clock.advance(25.0)
        assert sm.check_timeout() is None
        clock.advance(5.0)  # 30s em SILENCE -> expirou
        assert sm.check_timeout() == SessionState.HOLD


# ---------------------------------------------------------------------------
# Callbacks on_enter / on_exit
# ---------------------------------------------------------------------------


class TestCallbacks:
    def test_on_enter_called_on_transition(self) -> None:
        entered: list[str] = []
        sm = SessionStateMachine(
            on_enter={SessionState.ACTIVE: lambda: entered.append("active")},
        )
        sm.transition(SessionState.ACTIVE)
        assert entered == ["active"]

    def test_on_exit_called_on_transition(self) -> None:
        exited: list[str] = []
        sm = SessionStateMachine(
            on_exit={SessionState.INIT: lambda: exited.append("init")},
        )
        sm.transition(SessionState.ACTIVE)
        assert exited == ["init"]

    def test_both_callbacks_called_in_order(self) -> None:
        calls: list[str] = []
        sm = SessionStateMachine(
            on_exit={SessionState.INIT: lambda: calls.append("exit_init")},
            on_enter={SessionState.ACTIVE: lambda: calls.append("enter_active")},
        )
        sm.transition(SessionState.ACTIVE)
        assert calls == ["exit_init", "enter_active"]

    def test_callback_not_called_for_unregistered_state(self) -> None:
        calls: list[str] = []
        sm = SessionStateMachine(
            on_enter={SessionState.HOLD: lambda: calls.append("enter_hold")},
        )
        sm.transition(SessionState.ACTIVE)
        assert calls == []

    def test_callbacks_called_on_each_transition(self) -> None:
        entered: list[str] = []
        sm = SessionStateMachine(
            on_enter={
                SessionState.ACTIVE: lambda: entered.append("active"),
                SessionState.SILENCE: lambda: entered.append("silence"),
            },
        )
        sm.transition(SessionState.ACTIVE)
        sm.transition(SessionState.SILENCE)
        sm.transition(SessionState.ACTIVE)
        assert entered == ["active", "silence", "active"]

    def test_callbacks_not_called_on_invalid_transition(self) -> None:
        calls: list[str] = []
        sm = SessionStateMachine(
            on_enter={SessionState.HOLD: lambda: calls.append("enter_hold")},
            on_exit={SessionState.INIT: lambda: calls.append("exit_init")},
        )
        with pytest.raises(InvalidTransitionError):
            sm.transition(SessionState.HOLD)
        assert calls == []


# ---------------------------------------------------------------------------
# Clock injetavel e elapsed_in_state_ms
# ---------------------------------------------------------------------------


class TestClockAndElapsed:
    def test_elapsed_reflects_clock_advance(self) -> None:
        clock = FakeClock()
        sm = SessionStateMachine(clock=clock)
        clock.advance(1.5)
        assert sm.elapsed_in_state_ms == 1500

    def test_elapsed_resets_on_transition(self) -> None:
        clock = FakeClock()
        sm = SessionStateMachine(clock=clock)
        clock.advance(10.0)
        sm.transition(SessionState.ACTIVE)
        assert sm.elapsed_in_state_ms == 0
        clock.advance(2.5)
        assert sm.elapsed_in_state_ms == 2500

    def test_elapsed_truncates_to_int(self) -> None:
        clock = FakeClock()
        sm = SessionStateMachine(clock=clock)
        clock.advance(0.9999)
        assert sm.elapsed_in_state_ms == 999


# ---------------------------------------------------------------------------
# SessionTimeouts
# ---------------------------------------------------------------------------


class TestSessionTimeouts:
    def test_defaults(self) -> None:
        t = SessionTimeouts()
        assert t.init_timeout_s == 30.0
        assert t.silence_timeout_s == 30.0
        assert t.hold_timeout_s == 300.0
        assert t.closing_timeout_s == 2.0

    def test_custom_values(self) -> None:
        t = SessionTimeouts(
            init_timeout_s=10.0,
            silence_timeout_s=5.0,
            hold_timeout_s=60.0,
            closing_timeout_s=1.0,
        )
        assert t.init_timeout_s == 10.0
        assert t.silence_timeout_s == 5.0
        assert t.hold_timeout_s == 60.0
        assert t.closing_timeout_s == 1.0

    def test_get_timeout_for_state_returns_correct_values(self) -> None:
        t = SessionTimeouts(init_timeout_s=10.0, silence_timeout_s=20.0)
        assert t.get_timeout_for_state(SessionState.INIT) == 10.0
        assert t.get_timeout_for_state(SessionState.SILENCE) == 20.0
        assert t.get_timeout_for_state(SessionState.HOLD) == 300.0
        assert t.get_timeout_for_state(SessionState.CLOSING) == 2.0

    def test_get_timeout_for_active_returns_none(self) -> None:
        t = SessionTimeouts()
        assert t.get_timeout_for_state(SessionState.ACTIVE) is None

    def test_get_timeout_for_closed_returns_none(self) -> None:
        t = SessionTimeouts()
        assert t.get_timeout_for_state(SessionState.CLOSED) is None

    def test_frozen_immutable(self) -> None:
        t = SessionTimeouts()
        with pytest.raises(AttributeError):
            t.init_timeout_s = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# update_timeouts
# ---------------------------------------------------------------------------


class TestUpdateTimeouts:
    def test_update_changes_timeouts(self) -> None:
        clock = FakeClock()
        sm = SessionStateMachine(clock=clock)
        new_timeouts = SessionTimeouts(init_timeout_s=5.0)
        sm.update_timeouts(new_timeouts)
        assert sm.timeouts.init_timeout_s == 5.0

    def test_update_affects_current_state_check(self) -> None:
        clock = FakeClock()
        sm = SessionStateMachine(clock=clock)
        clock.advance(6.0)
        # Com timeout default (30s), nao expirou
        assert sm.check_timeout() is None
        # Atualizar para 5s -> ja expirou
        sm.update_timeouts(SessionTimeouts(init_timeout_s=5.0))
        assert sm.check_timeout() == SessionState.CLOSED


# ---------------------------------------------------------------------------
# Fluxo completo (integracao)
# ---------------------------------------------------------------------------


class TestFullLifecycle:
    def test_full_lifecycle_init_to_closed_via_hold(self) -> None:
        """Simula ciclo completo: INIT -> ACTIVE -> SILENCE -> HOLD -> CLOSING -> CLOSED."""
        clock = FakeClock()
        sm = SessionStateMachine(clock=clock)

        assert sm.state == SessionState.INIT
        sm.transition(SessionState.ACTIVE)
        assert sm.state == SessionState.ACTIVE
        sm.transition(SessionState.SILENCE)
        assert sm.state == SessionState.SILENCE
        sm.transition(SessionState.HOLD)
        assert sm.state == SessionState.HOLD
        sm.transition(SessionState.CLOSING)
        assert sm.state == SessionState.CLOSING
        sm.transition(SessionState.CLOSED)
        assert sm.state == SessionState.CLOSED

    def test_full_lifecycle_with_timeouts(self) -> None:
        """Simula ciclo via timeouts: INIT -> ACTIVE -> SILENCE --(30s)--> HOLD --(5min)--> CLOSING --(2s)--> CLOSED."""
        clock = FakeClock()
        sm = SessionStateMachine(clock=clock)

        sm.transition(SessionState.ACTIVE)
        sm.transition(SessionState.SILENCE)

        # 30s em SILENCE -> timeout -> HOLD
        clock.advance(30.0)
        target = sm.check_timeout()
        assert target == SessionState.HOLD
        sm.transition(target)
        assert sm.state == SessionState.HOLD

        # 5min em HOLD -> timeout -> CLOSING
        clock.advance(300.0)
        target = sm.check_timeout()
        assert target == SessionState.CLOSING
        sm.transition(target)
        assert sm.state == SessionState.CLOSING

        # 2s em CLOSING -> timeout -> CLOSED
        clock.advance(2.0)
        target = sm.check_timeout()
        assert target == SessionState.CLOSED
        sm.transition(target)
        assert sm.state == SessionState.CLOSED

    def test_silence_to_active_resets_timeout(self) -> None:
        """Simula: SILENCE (25s) -> fala detectada -> ACTIVE -> SILENCE (novo timeout)."""
        clock = FakeClock()
        sm = SessionStateMachine(clock=clock)

        sm.transition(SessionState.ACTIVE)
        sm.transition(SessionState.SILENCE)

        clock.advance(25.0)  # 25s em SILENCE, nao expirou
        assert sm.check_timeout() is None

        # Nova fala detectada
        sm.transition(SessionState.ACTIVE)
        sm.transition(SessionState.SILENCE)

        # 25s de novo: total desde inicio 50s, mas timeout resetou
        clock.advance(25.0)
        assert sm.check_timeout() is None

        clock.advance(5.0)  # 30s em SILENCE -> expirou
        assert sm.check_timeout() == SessionState.HOLD
