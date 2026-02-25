"""Tests for MuteController.

Validates:
- mute() / unmute() / is_muted property
- Reference counting: each mute() increments depth
- Underflow guard: unmute() below 0 is safe
- mute_depth property for debugging
- Multi-context lifecycle
- Backward compatibility (single mute/unmute identical to boolean)
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

    def test_initial_depth_is_zero(self) -> None:
        ctrl = MuteController(session_id="s1")
        assert ctrl.mute_depth == 0


class TestMute:
    def test_mute_sets_flag(self) -> None:
        ctrl = MuteController(session_id="s1")
        ctrl.mute()
        assert ctrl.is_muted is True

    def test_double_mute_increments_depth(self) -> None:
        ctrl = MuteController(session_id="s1")
        ctrl.mute()
        ctrl.mute()
        assert ctrl.is_muted is True
        assert ctrl.mute_depth == 2

    def test_mute_after_unmute(self) -> None:
        ctrl = MuteController(session_id="s1")
        ctrl.mute()
        ctrl.unmute()
        ctrl.mute()
        assert ctrl.is_muted is True
        assert ctrl.mute_depth == 1


class TestUnmute:
    def test_unmute_clears_flag(self) -> None:
        ctrl = MuteController(session_id="s1")
        ctrl.mute()
        ctrl.unmute()
        assert ctrl.is_muted is False

    def test_unmute_underflow_guard(self) -> None:
        """Unmuting when already at 0 stays at 0 (no negative depth)."""
        ctrl = MuteController(session_id="s1")
        ctrl.unmute()
        ctrl.unmute()
        assert ctrl.is_muted is False
        assert ctrl.mute_depth == 0

    def test_unmute_after_mute(self) -> None:
        ctrl = MuteController(session_id="s1")
        ctrl.mute()
        assert ctrl.is_muted is True
        ctrl.unmute()
        assert ctrl.is_muted is False


class TestMuteDepthProperty:
    def test_depth_tracks_increments(self) -> None:
        ctrl = MuteController(session_id="s1")
        ctrl.mute()
        assert ctrl.mute_depth == 1
        ctrl.mute()
        assert ctrl.mute_depth == 2
        ctrl.mute()
        assert ctrl.mute_depth == 3

    def test_depth_tracks_decrements(self) -> None:
        ctrl = MuteController(session_id="s1")
        ctrl.mute()
        ctrl.mute()
        ctrl.mute()
        ctrl.unmute()
        assert ctrl.mute_depth == 2
        ctrl.unmute()
        assert ctrl.mute_depth == 1
        ctrl.unmute()
        assert ctrl.mute_depth == 0


class TestDoubleMuseSingleUnmute:
    def test_double_mute_single_unmute_still_muted(self) -> None:
        """Two contexts mute, one unmutes -> still muted (depth=1)."""
        ctrl = MuteController(session_id="s1")
        ctrl.mute()  # context A
        ctrl.mute()  # context B
        ctrl.unmute()  # context A finishes
        assert ctrl.is_muted is True
        assert ctrl.mute_depth == 1

    def test_double_mute_double_unmute_unmuted(self) -> None:
        """Both contexts unmute -> fully unmuted (depth=0)."""
        ctrl = MuteController(session_id="s1")
        ctrl.mute()
        ctrl.mute()
        ctrl.unmute()
        ctrl.unmute()
        assert ctrl.is_muted is False
        assert ctrl.mute_depth == 0


class TestMultiContextIndependent:
    def test_multi_context_independent_mute_unmute(self) -> None:
        """Simulates ADR-008 multi-context lifecycle from docstring."""
        ctrl = MuteController(session_id="s1")

        # Context A starts TTS
        ctrl.mute()
        assert ctrl.mute_depth == 1
        assert ctrl.is_muted is True

        # Context B starts TTS
        ctrl.mute()
        assert ctrl.mute_depth == 2
        assert ctrl.is_muted is True

        # Context A finishes
        ctrl.unmute()
        assert ctrl.mute_depth == 1
        assert ctrl.is_muted is True  # B still active

        # Context B finishes
        ctrl.unmute()
        assert ctrl.mute_depth == 0
        assert ctrl.is_muted is False

    def test_three_contexts_interleaved(self) -> None:
        """Three TTS contexts overlapping."""
        ctrl = MuteController(session_id="s1")

        ctrl.mute()  # A
        ctrl.mute()  # B
        ctrl.mute()  # C
        assert ctrl.mute_depth == 3

        ctrl.unmute()  # B finishes
        assert ctrl.mute_depth == 2
        assert ctrl.is_muted is True

        ctrl.unmute()  # A finishes
        assert ctrl.mute_depth == 1
        assert ctrl.is_muted is True

        ctrl.unmute()  # C finishes
        assert ctrl.mute_depth == 0
        assert ctrl.is_muted is False


class TestMuteUnmuteSequence:
    def test_alternating_mute_unmute(self) -> None:
        """Single context: alternating mute/unmute (backward-compatible)."""
        ctrl = MuteController(session_id="s1")
        for _ in range(5):
            ctrl.mute()
            assert ctrl.is_muted is True
            assert ctrl.mute_depth == 1
            ctrl.unmute()
            assert ctrl.is_muted is False
            assert ctrl.mute_depth == 0

    def test_rapid_mute_unmute_with_ref_counting(self) -> None:
        """Rapid start/stop of TTS contexts."""
        ctrl = MuteController(session_id="s1")
        ctrl.mute()
        ctrl.unmute()
        ctrl.mute()
        ctrl.unmute()
        assert ctrl.is_muted is False
        assert ctrl.mute_depth == 0


class TestBackwardCompatibility:
    def test_single_mute_unmute_identical_to_boolean(self) -> None:
        """Single mute/unmute (current usage) behaves identically to old bool."""
        ctrl = MuteController(session_id="s1")
        assert ctrl.is_muted is False
        ctrl.mute()
        assert ctrl.is_muted is True
        ctrl.unmute()
        assert ctrl.is_muted is False
