"""Testes do MuteController.

Valida:
- mute() / unmute() / is_muted property
- Idempotencia: mute duplicado e no-op
- Idempotencia: unmute duplicado e no-op
- Estado inicial e unmuted
"""

from __future__ import annotations

from macaw.session.mute import MuteController


class TestMuteControllerInitialState:
    def test_starts_unmuted(self) -> None:
        ctrl = MuteController(session_id="s1")
        assert ctrl.is_muted is False

    def test_default_session_id(self) -> None:
        ctrl = MuteController()
        assert ctrl.is_muted is False


class TestMute:
    def test_mute_sets_flag(self) -> None:
        ctrl = MuteController(session_id="s1")
        ctrl.mute()
        assert ctrl.is_muted is True

    def test_mute_idempotent(self) -> None:
        ctrl = MuteController(session_id="s1")
        ctrl.mute()
        ctrl.mute()  # second call is no-op
        assert ctrl.is_muted is True

    def test_mute_after_unmute(self) -> None:
        ctrl = MuteController(session_id="s1")
        ctrl.mute()
        ctrl.unmute()
        ctrl.mute()
        assert ctrl.is_muted is True


class TestUnmute:
    def test_unmute_clears_flag(self) -> None:
        ctrl = MuteController(session_id="s1")
        ctrl.mute()
        ctrl.unmute()
        assert ctrl.is_muted is False

    def test_unmute_idempotent(self) -> None:
        ctrl = MuteController(session_id="s1")
        ctrl.unmute()  # already unmuted
        ctrl.unmute()
        assert ctrl.is_muted is False

    def test_unmute_after_mute(self) -> None:
        ctrl = MuteController(session_id="s1")
        ctrl.mute()
        assert ctrl.is_muted is True
        ctrl.unmute()
        assert ctrl.is_muted is False


class TestMuteUnmuteSequence:
    def test_alternating_mute_unmute(self) -> None:
        ctrl = MuteController(session_id="s1")
        for _ in range(5):
            ctrl.mute()
            assert ctrl.is_muted is True
            ctrl.unmute()
            assert ctrl.is_muted is False

    def test_rapid_mute_unmute(self) -> None:
        """Simula cenario rapido de TTS start/stop."""
        ctrl = MuteController(session_id="s1")
        ctrl.mute()
        ctrl.unmute()
        ctrl.mute()
        ctrl.unmute()
        assert ctrl.is_muted is False
