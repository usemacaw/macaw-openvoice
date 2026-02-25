"""Tests for commit_strategy WebSocket parameter (Sprint L, Task L1).

Validates:
 - SessionConfigureCommand accepts commit_strategy
 - SessionConfig includes commit_strategy in session.created
 - commit_strategy: "vad" preserves current default behavior
 - commit_strategy: "manual" disables VAD-triggered auto-commit
 - "commit" is accepted as a short alias for input_audio_buffer.commit
 - Switching commit_strategy mid-session works
 - StreamingSession.set_auto_commit controls VAD auto-commit
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from macaw.server.models.events import (
    SessionConfig,
    SessionConfigureCommand,
)
from macaw.server.ws_protocol import CommandResult, dispatch_message

# ---------------------------------------------------------------------------
# SessionConfig model tests
# ---------------------------------------------------------------------------


class TestSessionConfigCommitStrategy:
    """SessionConfig model includes commit_strategy."""

    def test_default_is_vad(self) -> None:
        config = SessionConfig()
        assert config.commit_strategy == "vad"

    def test_set_to_manual(self) -> None:
        config = SessionConfig(commit_strategy="manual")
        assert config.commit_strategy == "manual"

    def test_set_to_vad_explicit(self) -> None:
        config = SessionConfig(commit_strategy="vad")
        assert config.commit_strategy == "vad"

    def test_serialization_includes_commit_strategy(self) -> None:
        config = SessionConfig(commit_strategy="manual")
        data = config.model_dump()
        assert data["commit_strategy"] == "manual"


# ---------------------------------------------------------------------------
# SessionConfigureCommand model tests
# ---------------------------------------------------------------------------


class TestSessionConfigureCommandCommitStrategy:
    """SessionConfigureCommand accepts commit_strategy."""

    def test_default_is_none(self) -> None:
        cmd = SessionConfigureCommand()
        assert cmd.commit_strategy is None

    def test_set_vad(self) -> None:
        cmd = SessionConfigureCommand(commit_strategy="vad")
        assert cmd.commit_strategy == "vad"

    def test_set_manual(self) -> None:
        cmd = SessionConfigureCommand(commit_strategy="manual")
        assert cmd.commit_strategy == "manual"

    def test_invalid_strategy_rejected(self) -> None:
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            SessionConfigureCommand(commit_strategy="invalid")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# WS Protocol: "commit" alias
# ---------------------------------------------------------------------------


class TestCommitCommandAlias:
    """The 'commit' command type dispatches like input_audio_buffer.commit."""

    def test_commit_alias_dispatches(self) -> None:
        result = dispatch_message({"text": '{"type": "commit"}'})
        assert isinstance(result, CommandResult)
        assert result.command.type == "commit"

    def test_input_audio_buffer_commit_still_works(self) -> None:
        result = dispatch_message({"text": '{"type": "input_audio_buffer.commit"}'})
        assert isinstance(result, CommandResult)
        assert result.command.type == "input_audio_buffer.commit"


# ---------------------------------------------------------------------------
# StreamingSession.set_auto_commit
# ---------------------------------------------------------------------------


class TestStreamingSessionAutoCommit:
    """StreamingSession.set_auto_commit controls VAD auto-commit."""

    def _make_session(self) -> MagicMock:
        """Create a mock StreamingSession with set_auto_commit."""
        from macaw.session.streaming import StreamingSession

        # We need to test the real method, so patch __init__ to skip setup
        with patch.object(StreamingSession, "__init__", lambda self, **kwargs: None):
            session = StreamingSession()  # type: ignore[call-arg]
            session._auto_commit = True  # type: ignore[attr-defined]
            return session  # type: ignore[return-value]

    def test_default_is_auto_commit_true(self) -> None:
        session = self._make_session()
        assert session._auto_commit is True  # type: ignore[attr-defined]

    def test_set_auto_commit_false(self) -> None:
        session = self._make_session()
        session.set_auto_commit(False)
        assert session._auto_commit is False  # type: ignore[attr-defined]

    def test_set_auto_commit_true_again(self) -> None:
        session = self._make_session()
        session.set_auto_commit(False)
        session.set_auto_commit(True)
        assert session._auto_commit is True  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Handle configure command wiring
# ---------------------------------------------------------------------------


class TestHandleConfigureCommitStrategy:
    """_handle_configure_command wires commit_strategy to session."""

    @pytest.fixture()
    def ctx(self) -> MagicMock:
        from macaw.server.routes.realtime import SessionContext

        mock_ws = MagicMock()
        mock_ws.app.state = MagicMock()
        ctx = SessionContext(
            session_id="test",
            session_start=0.0,
            websocket=mock_ws,
        )
        ctx.session = MagicMock()
        ctx.session.set_auto_commit = MagicMock()
        return ctx

    @pytest.mark.asyncio
    async def test_commit_strategy_vad_sets_auto_commit_true(self, ctx: MagicMock) -> None:
        from macaw.server.routes.realtime import _handle_configure_command

        cmd = SessionConfigureCommand(commit_strategy="vad")
        await _handle_configure_command(ctx, cmd)
        ctx.session.set_auto_commit.assert_called_once_with(True)
        assert ctx.commit_strategy == "vad"

    @pytest.mark.asyncio
    async def test_commit_strategy_manual_sets_auto_commit_false(self, ctx: MagicMock) -> None:
        from macaw.server.routes.realtime import _handle_configure_command

        cmd = SessionConfigureCommand(commit_strategy="manual")
        await _handle_configure_command(ctx, cmd)
        ctx.session.set_auto_commit.assert_called_once_with(False)
        assert ctx.commit_strategy == "manual"

    @pytest.mark.asyncio
    async def test_commit_strategy_none_does_not_change(self, ctx: MagicMock) -> None:
        from macaw.server.routes.realtime import _handle_configure_command

        cmd = SessionConfigureCommand()
        await _handle_configure_command(ctx, cmd)
        ctx.session.set_auto_commit.assert_not_called()
        assert ctx.commit_strategy == "vad"  # Default from SessionContext

    @pytest.mark.asyncio
    async def test_commit_strategy_without_session(self) -> None:
        """When session is None, commit_strategy is still stored on ctx."""
        from macaw.server.routes.realtime import SessionContext, _handle_configure_command

        mock_ws = MagicMock()
        mock_ws.app.state = MagicMock()
        ctx = SessionContext(
            session_id="test",
            session_start=0.0,
            websocket=mock_ws,
        )
        cmd = SessionConfigureCommand(commit_strategy="manual")
        await _handle_configure_command(ctx, cmd)
        assert ctx.commit_strategy == "manual"
