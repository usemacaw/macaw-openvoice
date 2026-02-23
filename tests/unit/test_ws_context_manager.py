"""Tests for WSContextManager — multi-context TTS state management."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from macaw.server.ws_context_manager import (
    ContextState,
    TTSContext,
    WSContextManager,
)

# ---------------------------------------------------------------------------
# TTSContext dataclass
# ---------------------------------------------------------------------------


class TestTTSContext:
    def test_defaults(self) -> None:
        ctx = TTSContext(context_id="ctx-1")
        assert ctx.context_id == "ctx-1"
        assert ctx.state == ContextState.ACTIVE
        assert ctx.tts_task is None
        assert ctx.is_active is True
        assert ctx.text_buffer is not None
        assert ctx.cancel_event is not None
        assert not ctx.cancel_event.is_set()

    def test_touch_updates_last_activity(self) -> None:
        ctx = TTSContext(context_id="ctx-1")
        old_time = ctx.last_activity
        # Advance monotonic clock
        with patch("macaw.server.ws_context_manager.time") as mock_time:
            mock_time.monotonic.return_value = old_time + 5.0
            ctx.touch()
        assert ctx.last_activity == old_time + 5.0

    def test_is_active_false_when_closed(self) -> None:
        ctx = TTSContext(context_id="ctx-1")
        ctx.state = ContextState.CLOSED
        assert ctx.is_active is False

    def test_is_active_true_when_flushing(self) -> None:
        ctx = TTSContext(context_id="ctx-1")
        ctx.state = ContextState.FLUSHING
        assert ctx.is_active is True

    def test_cancel_event_independent_per_context(self) -> None:
        ctx1 = TTSContext(context_id="ctx-1")
        ctx2 = TTSContext(context_id="ctx-2")
        ctx1.cancel_event.set()
        assert ctx1.cancel_event.is_set()
        assert not ctx2.cancel_event.is_set()

    def test_text_buffer_independent_per_context(self) -> None:
        ctx1 = TTSContext(context_id="ctx-1")
        ctx2 = TTSContext(context_id="ctx-2")
        ctx1.text_buffer.append("Hello", request_id="r1")
        assert ctx1.text_buffer.pending_text == "Hello"
        assert ctx2.text_buffer.pending_text == ""


# ---------------------------------------------------------------------------
# WSContextManager — creation
# ---------------------------------------------------------------------------


class TestContextCreation:
    def test_create_context(self) -> None:
        mgr = WSContextManager(max_contexts=5)
        ctx = mgr.create_context("ctx-1")
        assert ctx.context_id == "ctx-1"
        assert ctx.state == ContextState.ACTIVE
        assert mgr.active_count == 1

    def test_create_multiple_contexts(self) -> None:
        mgr = WSContextManager(max_contexts=5)
        mgr.create_context("ctx-1")
        mgr.create_context("ctx-2")
        mgr.create_context("ctx-3")
        assert mgr.active_count == 3
        assert sorted(mgr.context_ids) == ["ctx-1", "ctx-2", "ctx-3"]

    def test_create_duplicate_raises(self) -> None:
        mgr = WSContextManager(max_contexts=5)
        mgr.create_context("ctx-1")
        with pytest.raises(ValueError, match="already exists"):
            mgr.create_context("ctx-1")

    def test_create_over_closed_context_succeeds(self) -> None:
        mgr = WSContextManager(max_contexts=5)
        mgr.create_context("ctx-1")
        mgr.close_context("ctx-1")
        # Should be able to re-create
        ctx = mgr.create_context("ctx-1")
        assert ctx.state == ContextState.ACTIVE

    def test_create_exceeds_max_raises(self) -> None:
        mgr = WSContextManager(max_contexts=2)
        mgr.create_context("ctx-1")
        mgr.create_context("ctx-2")
        with pytest.raises(ValueError, match="Maximum contexts"):
            mgr.create_context("ctx-3")

    def test_create_after_closing_frees_slot(self) -> None:
        mgr = WSContextManager(max_contexts=2)
        mgr.create_context("ctx-1")
        mgr.create_context("ctx-2")
        mgr.close_context("ctx-1")
        # Now there's room
        ctx = mgr.create_context("ctx-3")
        assert ctx.state == ContextState.ACTIVE
        assert mgr.active_count == 2


# ---------------------------------------------------------------------------
# WSContextManager — get
# ---------------------------------------------------------------------------


class TestContextGet:
    def test_get_existing(self) -> None:
        mgr = WSContextManager()
        mgr.create_context("ctx-1")
        assert mgr.get_context("ctx-1") is not None

    def test_get_nonexistent_returns_none(self) -> None:
        mgr = WSContextManager()
        assert mgr.get_context("nope") is None

    def test_get_active_excludes_closed(self) -> None:
        mgr = WSContextManager()
        mgr.create_context("ctx-1")
        mgr.close_context("ctx-1")
        assert mgr.get_context("ctx-1") is not None  # still visible
        assert mgr.get_active_context("ctx-1") is None  # filtered out

    def test_get_active_includes_flushing(self) -> None:
        mgr = WSContextManager()
        mgr.create_context("ctx-1")
        mgr.flush_context("ctx-1")
        assert mgr.get_active_context("ctx-1") is not None


# ---------------------------------------------------------------------------
# WSContextManager — close
# ---------------------------------------------------------------------------


class TestContextClose:
    def test_close_sets_state_and_cancel(self) -> None:
        mgr = WSContextManager()
        ctx = mgr.create_context("ctx-1")
        result = mgr.close_context("ctx-1")
        assert result is True
        assert ctx.state == ContextState.CLOSED
        assert ctx.cancel_event.is_set()

    def test_close_nonexistent_returns_false(self) -> None:
        mgr = WSContextManager()
        assert mgr.close_context("nope") is False

    def test_close_already_closed_returns_false(self) -> None:
        mgr = WSContextManager()
        mgr.create_context("ctx-1")
        assert mgr.close_context("ctx-1") is True
        assert mgr.close_context("ctx-1") is False

    def test_close_cancels_tts_task(self) -> None:
        mgr = WSContextManager()
        ctx = mgr.create_context("ctx-1")
        # Simulate active task with a mock
        mock_task = MagicMock()
        mock_task.done.return_value = False
        ctx.tts_task = mock_task
        mgr.close_context("ctx-1")
        mock_task.cancel.assert_called_once()

    def test_close_all(self) -> None:
        mgr = WSContextManager()
        mgr.create_context("ctx-1")
        mgr.create_context("ctx-2")
        mgr.create_context("ctx-3")
        closed = mgr.close_all()
        assert sorted(closed) == ["ctx-1", "ctx-2", "ctx-3"]
        assert mgr.active_count == 0

    def test_close_all_empty(self) -> None:
        mgr = WSContextManager()
        assert mgr.close_all() == []


# ---------------------------------------------------------------------------
# WSContextManager — flush
# ---------------------------------------------------------------------------


class TestContextFlush:
    def test_flush_returns_text_and_transitions_state(self) -> None:
        mgr = WSContextManager()
        ctx = mgr.create_context("ctx-1")
        ctx.text_buffer.append("Hello world.", request_id="r1")
        # Manually set buffer without split (use "none" strategy)
        ctx.text_buffer.split_strategy = "none"
        ctx.text_buffer.clear()
        ctx.text_buffer.append("Final text.", request_id="r1")

        text = mgr.flush_context("ctx-1")
        assert text == "Final text."
        assert ctx.state == ContextState.FLUSHING

    def test_flush_empty_buffer_returns_none(self) -> None:
        mgr = WSContextManager()
        mgr.create_context("ctx-1")
        text = mgr.flush_context("ctx-1")
        assert text is None

    def test_flush_nonexistent_returns_none(self) -> None:
        mgr = WSContextManager()
        assert mgr.flush_context("nope") is None

    def test_flush_closed_returns_none(self) -> None:
        mgr = WSContextManager()
        ctx = mgr.create_context("ctx-1")
        ctx.text_buffer.append("text", request_id="r1")
        mgr.close_context("ctx-1")
        assert mgr.flush_context("ctx-1") is None


# ---------------------------------------------------------------------------
# WSContextManager — inactivity
# ---------------------------------------------------------------------------


class TestInactivity:
    def test_get_inactive_context_ids(self) -> None:
        mgr = WSContextManager(inactivity_timeout_s=10.0)
        ctx = mgr.create_context("ctx-1")
        # Simulate context being old
        ctx.last_activity = time.monotonic() - 15.0
        inactive = mgr.get_inactive_context_ids()
        assert inactive == ["ctx-1"]

    def test_active_context_not_inactive(self) -> None:
        mgr = WSContextManager(inactivity_timeout_s=10.0)
        mgr.create_context("ctx-1")
        inactive = mgr.get_inactive_context_ids()
        assert inactive == []

    def test_closed_context_not_in_inactive(self) -> None:
        mgr = WSContextManager(inactivity_timeout_s=10.0)
        ctx = mgr.create_context("ctx-1")
        ctx.last_activity = time.monotonic() - 15.0
        mgr.close_context("ctx-1")
        # Closed contexts should not be reported as inactive
        inactive = mgr.get_inactive_context_ids()
        assert inactive == []

    def test_touch_resets_inactivity(self) -> None:
        mgr = WSContextManager(inactivity_timeout_s=10.0)
        ctx = mgr.create_context("ctx-1")
        ctx.last_activity = time.monotonic() - 15.0
        mgr.touch("ctx-1")
        inactive = mgr.get_inactive_context_ids()
        assert inactive == []

    def test_touch_nonexistent_is_noop(self) -> None:
        mgr = WSContextManager()
        mgr.touch("nope")  # should not raise


# ---------------------------------------------------------------------------
# WSContextManager — properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_max_contexts(self) -> None:
        mgr = WSContextManager(max_contexts=42)
        assert mgr.max_contexts == 42

    def test_inactivity_timeout(self) -> None:
        mgr = WSContextManager(inactivity_timeout_s=15.0)
        assert mgr.inactivity_timeout_s == 15.0

    def test_active_count_excludes_closed(self) -> None:
        mgr = WSContextManager()
        mgr.create_context("ctx-1")
        mgr.create_context("ctx-2")
        mgr.close_context("ctx-1")
        assert mgr.active_count == 1

    def test_context_ids_includes_closed(self) -> None:
        mgr = WSContextManager()
        mgr.create_context("ctx-1")
        mgr.create_context("ctx-2")
        mgr.close_context("ctx-1")
        assert sorted(mgr.context_ids) == ["ctx-1", "ctx-2"]


# ---------------------------------------------------------------------------
# MultiContextSettings
# ---------------------------------------------------------------------------


class TestMultiContextSettings:
    def test_defaults(self) -> None:
        from macaw.config.settings import MultiContextSettings

        settings = MultiContextSettings()
        assert settings.max_contexts == 10
        assert settings.inactivity_timeout_s == 20.0
        assert settings.max_concurrent_tts == 4

    def test_env_override(self) -> None:
        from macaw.config.settings import MultiContextSettings

        settings = MultiContextSettings(
            max_contexts=20,
            inactivity_timeout_s=30.0,
            max_concurrent_tts=8,
        )
        assert settings.max_contexts == 20
        assert settings.inactivity_timeout_s == 30.0
        assert settings.max_concurrent_tts == 8

    def test_validation_min(self) -> None:
        from pydantic import ValidationError

        from macaw.config.settings import MultiContextSettings

        with pytest.raises(ValidationError):
            MultiContextSettings(max_contexts=0)

    def test_on_macaw_settings(self) -> None:
        from macaw.config.settings import MacawSettings

        settings = MacawSettings()
        assert settings.multi_context.max_contexts == 10
