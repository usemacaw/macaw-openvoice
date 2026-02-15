"""Testes dedicados para SessionTimeouts e timeouts_from_configure_command.

Cobre: defaults do PRD, override de valores, validacao de minimo (1s),
conversao ms -> s, mudanca em runtime afetando estado atual, e boundaries.
"""

from __future__ import annotations

import pytest

from macaw._types import SessionState
from macaw.session.state_machine import (
    _MIN_TIMEOUT_S,
    SessionStateMachine,
    SessionTimeouts,
    timeouts_from_configure_command,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeClock:
    """Clock deterministico para testes."""

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


# ---------------------------------------------------------------------------
# SessionTimeouts defaults
# ---------------------------------------------------------------------------


class TestSessionTimeoutsDefaults:
    def test_defaults_match_prd(self) -> None:
        """Defaults devem ser: INIT=30s, SILENCE=30s, HOLD=300s, CLOSING=2s."""
        t = SessionTimeouts()
        assert t.init_timeout_s == 30.0
        assert t.silence_timeout_s == 30.0
        assert t.hold_timeout_s == 300.0
        assert t.closing_timeout_s == 2.0

    def test_override_specific_values(self) -> None:
        """Override parcial de valores deve preservar defaults dos demais."""
        t = SessionTimeouts(silence_timeout_s=5.0, hold_timeout_s=60.0)
        assert t.init_timeout_s == 30.0  # default mantido
        assert t.silence_timeout_s == 5.0  # override
        assert t.hold_timeout_s == 60.0  # override
        assert t.closing_timeout_s == 2.0  # default mantido


# ---------------------------------------------------------------------------
# Validacao de minimo
# ---------------------------------------------------------------------------


class TestMinimumTimeoutValidation:
    def test_init_timeout_below_minimum_raises(self) -> None:
        with pytest.raises(ValueError, match="init_timeout_s"):
            SessionTimeouts(init_timeout_s=0.5)

    def test_silence_timeout_below_minimum_raises(self) -> None:
        with pytest.raises(ValueError, match="silence_timeout_s"):
            SessionTimeouts(silence_timeout_s=0.1)

    def test_hold_timeout_below_minimum_raises(self) -> None:
        with pytest.raises(ValueError, match="hold_timeout_s"):
            SessionTimeouts(hold_timeout_s=0.0)

    def test_closing_timeout_below_minimum_raises(self) -> None:
        with pytest.raises(ValueError, match="closing_timeout_s"):
            SessionTimeouts(closing_timeout_s=0.999)

    def test_exactly_minimum_accepted(self) -> None:
        """Timeout exatamente igual ao minimo (1.0s) deve ser aceito."""
        t = SessionTimeouts(
            init_timeout_s=_MIN_TIMEOUT_S,
            silence_timeout_s=_MIN_TIMEOUT_S,
            hold_timeout_s=_MIN_TIMEOUT_S,
            closing_timeout_s=_MIN_TIMEOUT_S,
        )
        assert t.init_timeout_s == _MIN_TIMEOUT_S
        assert t.silence_timeout_s == _MIN_TIMEOUT_S
        assert t.hold_timeout_s == _MIN_TIMEOUT_S
        assert t.closing_timeout_s == _MIN_TIMEOUT_S

    def test_just_below_minimum_rejected(self) -> None:
        """Timeout 0.999s deve ser rejeitado (< 1.0s)."""
        with pytest.raises(ValueError):
            SessionTimeouts(silence_timeout_s=0.999)

    def test_negative_timeout_rejected(self) -> None:
        """Timeout negativo deve ser rejeitado."""
        with pytest.raises(ValueError):
            SessionTimeouts(init_timeout_s=-1.0)


# ---------------------------------------------------------------------------
# Conversao ms -> s (timeouts_from_configure_command)
# ---------------------------------------------------------------------------


class TestTimeoutsFromConfigureCommand:
    def test_convert_silence_timeout_ms_to_s(self) -> None:
        current = SessionTimeouts()
        updated = timeouts_from_configure_command(current, silence_timeout_ms=5000)
        assert updated.silence_timeout_s == 5.0

    def test_convert_hold_timeout_ms_to_s(self) -> None:
        current = SessionTimeouts()
        updated = timeouts_from_configure_command(current, hold_timeout_ms=60000)
        assert updated.hold_timeout_s == 60.0

    def test_none_fields_preserve_current(self) -> None:
        """Campos None devem manter os valores atuais."""
        current = SessionTimeouts(silence_timeout_s=15.0, hold_timeout_s=120.0)
        updated = timeouts_from_configure_command(current)
        assert updated.silence_timeout_s == 15.0
        assert updated.hold_timeout_s == 120.0
        assert updated.init_timeout_s == current.init_timeout_s
        assert updated.closing_timeout_s == current.closing_timeout_s

    def test_convert_both_silence_and_hold(self) -> None:
        current = SessionTimeouts()
        updated = timeouts_from_configure_command(
            current, silence_timeout_ms=10000, hold_timeout_ms=180000
        )
        assert updated.silence_timeout_s == 10.0
        assert updated.hold_timeout_s == 180.0

    def test_init_and_closing_timeouts_preserved(self) -> None:
        """init e closing timeouts nao sao afetados por configure command."""
        current = SessionTimeouts(init_timeout_s=10.0, closing_timeout_s=3.0)
        updated = timeouts_from_configure_command(current, silence_timeout_ms=5000)
        assert updated.init_timeout_s == 10.0
        assert updated.closing_timeout_s == 3.0

    def test_conversion_below_minimum_raises(self) -> None:
        """Conversao que resulte em timeout < 1s deve levantar ValueError."""
        current = SessionTimeouts()
        with pytest.raises(ValueError, match="silence_timeout_s"):
            timeouts_from_configure_command(current, silence_timeout_ms=500)

    def test_conversion_exactly_1000ms_accepted(self) -> None:
        """1000ms = 1.0s deve ser aceito (boundary minimo)."""
        current = SessionTimeouts()
        updated = timeouts_from_configure_command(current, silence_timeout_ms=1000)
        assert updated.silence_timeout_s == 1.0


# ---------------------------------------------------------------------------
# Runtime update afeta estado atual
# ---------------------------------------------------------------------------


class TestRuntimeUpdateAffectsCurrentState:
    def test_reducing_timeout_triggers_expiry(self) -> None:
        """Se em SILENCE ha 6s e timeout muda de 30s para 5s, deve expirar."""
        clock = FakeClock()
        sm = SessionStateMachine(clock=clock)
        sm.transition(SessionState.ACTIVE)
        sm.transition(SessionState.SILENCE)
        clock.advance(6.0)
        # Com default (30s), nao expirou
        assert sm.check_timeout() is None
        # Reduzir timeout para 5s -> ja expirou
        sm.update_timeouts(SessionTimeouts(silence_timeout_s=5.0))
        assert sm.check_timeout() == SessionState.HOLD

    def test_increasing_timeout_prevents_expiry(self) -> None:
        """Se em INIT ha 25s e timeout muda de 30s para 60s, nao deve expirar."""
        clock = FakeClock()
        sm = SessionStateMachine(clock=clock)
        clock.advance(25.0)
        # Com default (30s), faltam 5s
        assert sm.check_timeout() is None
        # Aumentar para 60s
        sm.update_timeouts(SessionTimeouts(init_timeout_s=60.0))
        clock.advance(10.0)  # 35s total, mas timeout agora e 60s
        assert sm.check_timeout() is None
        clock.advance(25.0)  # 60s total -> expirou
        assert sm.check_timeout() == SessionState.CLOSED

    def test_update_hold_timeout_while_in_hold(self) -> None:
        """Atualizar hold timeout enquanto em HOLD afeta imediatamente."""
        clock = FakeClock()
        sm = SessionStateMachine(clock=clock)
        sm.transition(SessionState.ACTIVE)
        sm.transition(SessionState.SILENCE)
        sm.transition(SessionState.HOLD)
        clock.advance(10.0)  # 10s em HOLD (default 300s)
        assert sm.check_timeout() is None
        # Reduzir para 5s -> ja expirou
        sm.update_timeouts(SessionTimeouts(hold_timeout_s=5.0))
        assert sm.check_timeout() == SessionState.CLOSING

    def test_update_via_configure_command_conversion(self) -> None:
        """Fluxo completo: session.configure com ms -> update_timeouts com s."""
        clock = FakeClock()
        sm = SessionStateMachine(clock=clock)
        sm.transition(SessionState.ACTIVE)
        sm.transition(SessionState.SILENCE)

        # Simular session.configure com silence_timeout_ms=5000
        new_timeouts = timeouts_from_configure_command(sm.timeouts, silence_timeout_ms=5000)
        sm.update_timeouts(new_timeouts)

        clock.advance(4.9)
        assert sm.check_timeout() is None
        clock.advance(0.1)  # 5.0s -> expirou com novo timeout
        assert sm.check_timeout() == SessionState.HOLD
