"""Tests for incremental TTS commands: tts.append, tts.flush, tts.clear.

Covers Pydantic model validation, ws_protocol dispatch, and union types.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from macaw.server.models.events import (
    ClientCommand,
    ServerEvent,
    SessionConfigureCommand,
    TTSAppendCommand,
    TTSBufferFlushedEvent,
    TTSCancelCommand,
    TTSClearCommand,
    TTSFlushCommand,
    TTSSpeakCommand,
)
from macaw.server.ws_protocol import (
    CommandResult,
    ErrorResult,
    dispatch_message,
)

# ---------------------------------------------------------------------------
# TTSAppendCommand model validation
# ---------------------------------------------------------------------------


class TestTTSAppendCommand:
    """Validation for tts.append command."""

    def test_valid_minimal(self) -> None:
        cmd = TTSAppendCommand(text="Hello")
        assert cmd.type == "tts.append"
        assert cmd.text == "Hello"
        assert cmd.request_id is None

    def test_valid_with_request_id(self) -> None:
        cmd = TTSAppendCommand(text="Hello", request_id="req_1")
        assert cmd.request_id == "req_1"

    def test_empty_text_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TTSAppendCommand(text="")

    def test_text_min_length(self) -> None:
        cmd = TTSAppendCommand(text="a")
        assert cmd.text == "a"

    def test_frozen(self) -> None:
        cmd = TTSAppendCommand(text="Hello")
        with pytest.raises(ValidationError):
            cmd.text = "world"  # type: ignore[misc]

    def test_type_literal(self) -> None:
        cmd = TTSAppendCommand(text="Hello")
        assert cmd.type == "tts.append"

    def test_model_dump(self) -> None:
        cmd = TTSAppendCommand(text="Hello", request_id="req_1")
        data = cmd.model_dump()
        assert data["type"] == "tts.append"
        assert data["text"] == "Hello"
        assert data["request_id"] == "req_1"


# ---------------------------------------------------------------------------
# TTSFlushCommand model validation
# ---------------------------------------------------------------------------


class TestTTSFlushCommand:
    """Validation for tts.flush command."""

    def test_valid_minimal(self) -> None:
        cmd = TTSFlushCommand()
        assert cmd.type == "tts.flush"
        assert cmd.request_id is None

    def test_valid_with_request_id(self) -> None:
        cmd = TTSFlushCommand(request_id="req_flush")
        assert cmd.request_id == "req_flush"

    def test_frozen(self) -> None:
        cmd = TTSFlushCommand()
        with pytest.raises(ValidationError):
            cmd.request_id = "new"  # type: ignore[misc]

    def test_model_dump(self) -> None:
        cmd = TTSFlushCommand(request_id="r1")
        data = cmd.model_dump()
        assert data["type"] == "tts.flush"
        assert data["request_id"] == "r1"


# ---------------------------------------------------------------------------
# TTSClearCommand model validation
# ---------------------------------------------------------------------------


class TestTTSClearCommand:
    """Validation for tts.clear command."""

    def test_valid_minimal(self) -> None:
        cmd = TTSClearCommand()
        assert cmd.type == "tts.clear"
        assert cmd.request_id is None

    def test_valid_with_request_id(self) -> None:
        cmd = TTSClearCommand(request_id="req_clear")
        assert cmd.request_id == "req_clear"

    def test_frozen(self) -> None:
        cmd = TTSClearCommand()
        with pytest.raises(ValidationError):
            cmd.request_id = "new"  # type: ignore[misc]

    def test_model_dump(self) -> None:
        cmd = TTSClearCommand()
        data = cmd.model_dump()
        assert data["type"] == "tts.clear"


# ---------------------------------------------------------------------------
# TTSBufferFlushedEvent model validation
# ---------------------------------------------------------------------------


class TestTTSBufferFlushedEvent:
    """Validation for tts.buffer_flushed server event."""

    def test_valid_manual(self) -> None:
        event = TTSBufferFlushedEvent(
            request_id="req_1",
            text="Hello world.",
            trigger="manual",
        )
        assert event.type == "tts.buffer_flushed"
        assert event.request_id == "req_1"
        assert event.text == "Hello world."
        assert event.trigger == "manual"

    def test_valid_auto_split(self) -> None:
        event = TTSBufferFlushedEvent(
            request_id="req_1",
            text="First sentence.",
            trigger="auto_split",
        )
        assert event.trigger == "auto_split"

    def test_valid_auto_timeout(self) -> None:
        event = TTSBufferFlushedEvent(
            request_id="req_1",
            text="Timed out text.",
            trigger="auto_timeout",
        )
        assert event.trigger == "auto_timeout"

    def test_valid_new_request_id(self) -> None:
        event = TTSBufferFlushedEvent(
            request_id="req_old",
            text="Old text.",
            trigger="new_request_id",
        )
        assert event.trigger == "new_request_id"

    def test_invalid_trigger_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TTSBufferFlushedEvent(
                request_id="req_1",
                text="text",
                trigger="invalid_trigger",  # type: ignore[arg-type]
            )

    def test_frozen(self) -> None:
        event = TTSBufferFlushedEvent(request_id="req_1", text="text", trigger="manual")
        with pytest.raises(ValidationError):
            event.text = "new"  # type: ignore[misc]

    def test_model_dump(self) -> None:
        event = TTSBufferFlushedEvent(request_id="req_1", text="Hello.", trigger="manual")
        data = event.model_dump()
        assert data["type"] == "tts.buffer_flushed"
        assert data["request_id"] == "req_1"
        assert data["text"] == "Hello."
        assert data["trigger"] == "manual"


# ---------------------------------------------------------------------------
# SessionConfigureCommand new fields
# ---------------------------------------------------------------------------


class TestSessionConfigureNewFields:
    """SessionConfigureCommand with tts_split_strategy and tts_flush_timeout_ms."""

    def test_default_values(self) -> None:
        cmd = SessionConfigureCommand()
        assert cmd.tts_split_strategy is None
        assert cmd.tts_flush_timeout_ms is None

    def test_set_split_strategy(self) -> None:
        cmd = SessionConfigureCommand(tts_split_strategy="paragraph")
        assert cmd.tts_split_strategy == "paragraph"

    def test_set_flush_timeout(self) -> None:
        cmd = SessionConfigureCommand(tts_flush_timeout_ms=3000)
        assert cmd.tts_flush_timeout_ms == 3000

    def test_flush_timeout_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            SessionConfigureCommand(tts_flush_timeout_ms=0)

    def test_flush_timeout_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SessionConfigureCommand(tts_flush_timeout_ms=-1)

    def test_invalid_split_strategy_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SessionConfigureCommand(tts_split_strategy="word")  # type: ignore[arg-type]

    def test_valid_split_strategies(self) -> None:
        for strategy in ("sentence", "paragraph", "none"):
            cmd = SessionConfigureCommand(tts_split_strategy=strategy)  # type: ignore[arg-type]
            assert cmd.tts_split_strategy == strategy

    def test_model_dump_includes_new_fields(self) -> None:
        cmd = SessionConfigureCommand(
            tts_split_strategy="sentence",
            tts_flush_timeout_ms=5000,
        )
        data = cmd.model_dump()
        assert data["tts_split_strategy"] == "sentence"
        assert data["tts_flush_timeout_ms"] == 5000


# ---------------------------------------------------------------------------
# ws_protocol dispatch
# ---------------------------------------------------------------------------


class TestWSProtocolDispatch:
    """ws_protocol.dispatch_message recognizes new commands."""

    def test_dispatch_tts_append(self) -> None:
        msg = {"text": json.dumps({"type": "tts.append", "text": "hello"})}
        result = dispatch_message(msg)
        assert isinstance(result, CommandResult)
        assert isinstance(result.command, TTSAppendCommand)
        assert result.command.text == "hello"

    def test_dispatch_tts_flush(self) -> None:
        msg = {"text": json.dumps({"type": "tts.flush"})}
        result = dispatch_message(msg)
        assert isinstance(result, CommandResult)
        assert isinstance(result.command, TTSFlushCommand)

    def test_dispatch_tts_clear(self) -> None:
        msg = {"text": json.dumps({"type": "tts.clear"})}
        result = dispatch_message(msg)
        assert isinstance(result, CommandResult)
        assert isinstance(result.command, TTSClearCommand)

    def test_dispatch_tts_append_with_request_id(self) -> None:
        msg = {"text": json.dumps({"type": "tts.append", "text": "token", "request_id": "r1"})}
        result = dispatch_message(msg)
        assert isinstance(result, CommandResult)
        assert isinstance(result.command, TTSAppendCommand)
        assert result.command.request_id == "r1"

    def test_dispatch_tts_append_empty_text_error(self) -> None:
        msg = {"text": json.dumps({"type": "tts.append", "text": ""})}
        result = dispatch_message(msg)
        assert isinstance(result, ErrorResult)
        assert result.event.code == "validation_error"

    def test_dispatch_tts_flush_with_request_id(self) -> None:
        msg = {"text": json.dumps({"type": "tts.flush", "request_id": "flush_r1"})}
        result = dispatch_message(msg)
        assert isinstance(result, CommandResult)
        assert isinstance(result.command, TTSFlushCommand)
        assert result.command.request_id == "flush_r1"

    def test_dispatch_tts_clear_with_request_id(self) -> None:
        msg = {"text": json.dumps({"type": "tts.clear", "request_id": "clear_r1"})}
        result = dispatch_message(msg)
        assert isinstance(result, CommandResult)
        assert isinstance(result.command, TTSClearCommand)
        assert result.command.request_id == "clear_r1"


# ---------------------------------------------------------------------------
# Union types
# ---------------------------------------------------------------------------


class TestUnionTypes:
    """ClientCommand and ServerEvent union types include new members."""

    def test_tts_append_in_client_command(self) -> None:
        cmd = TTSAppendCommand(text="hello")
        # Verify it's a valid member of ClientCommand union
        assert isinstance(cmd, TTSAppendCommand)
        # ClientCommand is a type alias (Union), verify at type level
        # We test by confirming the type is listed in get_args
        import typing

        args = typing.get_args(ClientCommand)
        assert TTSAppendCommand in args

    def test_tts_flush_in_client_command(self) -> None:
        import typing

        args = typing.get_args(ClientCommand)
        assert TTSFlushCommand in args

    def test_tts_clear_in_client_command(self) -> None:
        import typing

        args = typing.get_args(ClientCommand)
        assert TTSClearCommand in args

    def test_buffer_flushed_in_server_event(self) -> None:
        import typing

        args = typing.get_args(ServerEvent)
        assert TTSBufferFlushedEvent in args

    def test_existing_commands_still_in_union(self) -> None:
        """Ensure we did not accidentally remove existing union members."""
        import typing

        args = typing.get_args(ClientCommand)
        assert TTSSpeakCommand in args
        assert TTSCancelCommand in args
        assert SessionConfigureCommand in args
