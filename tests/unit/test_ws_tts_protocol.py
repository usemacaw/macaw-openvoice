"""Testes do protocolo WebSocket Full-Duplex TTS.

Valida:
- TTSSpeakCommand e TTSCancelCommand (Pydantic models, dispatch)
- TTSSpeakingStartEvent e TTSSpeakingEndEvent (serialization)
- SessionConfigureCommand com model_tts
- SessionConfig com model_tts
- dispatch_message() para novos comandos TTS
- Union types ServerEvent e ClientCommand incluem novos tipos
"""

from __future__ import annotations

import json
import uuid

from macaw.server.models.events import (
    ClientCommand,
    ServerEvent,
    SessionConfig,
    SessionConfigureCommand,
    SessionCreatedEvent,
    TTSCancelCommand,
    TTSSpeakCommand,
    TTSSpeakingEndEvent,
    TTSSpeakingStartEvent,
)
from macaw.server.ws_protocol import (
    CommandResult,
    ErrorResult,
    dispatch_message,
)

# ─── TTSSpeakCommand ───


class TestTTSSpeakCommand:
    def test_type_literal(self) -> None:
        cmd = TTSSpeakCommand(text="Hello")
        assert cmd.type == "tts.speak"

    def test_defaults(self) -> None:
        cmd = TTSSpeakCommand(text="Hello")
        assert cmd.voice == "default"
        assert cmd.request_id is None

    def test_custom_values(self) -> None:
        cmd = TTSSpeakCommand(
            text="Hello world",
            voice="alloy",
            request_id="req-123",
        )
        assert cmd.text == "Hello world"
        assert cmd.voice == "alloy"
        assert cmd.request_id == "req-123"

    def test_frozen(self) -> None:
        cmd = TTSSpeakCommand(text="Hello")
        try:
            cmd.text = "Changed"  # type: ignore[misc]
            raised = False
        except Exception:
            raised = True
        assert raised

    def test_serialization_roundtrip(self) -> None:
        cmd = TTSSpeakCommand(text="Ola", voice="default", request_id="r1")
        data = cmd.model_dump()
        assert data["type"] == "tts.speak"
        assert data["text"] == "Ola"
        restored = TTSSpeakCommand.model_validate(data)
        assert restored == cmd


# ─── TTSCancelCommand ───


class TestTTSCancelCommand:
    def test_type_literal(self) -> None:
        cmd = TTSCancelCommand()
        assert cmd.type == "tts.cancel"

    def test_request_id_default_none(self) -> None:
        cmd = TTSCancelCommand()
        assert cmd.request_id is None

    def test_with_request_id(self) -> None:
        cmd = TTSCancelCommand(request_id="req-456")
        assert cmd.request_id == "req-456"

    def test_serialization(self) -> None:
        cmd = TTSCancelCommand(request_id="r2")
        data = cmd.model_dump()
        assert data["type"] == "tts.cancel"
        assert data["request_id"] == "r2"


# ─── TTSSpeakingStartEvent ───


class TestTTSSpeakingStartEvent:
    def test_type_literal(self) -> None:
        event = TTSSpeakingStartEvent(request_id="r1", timestamp_ms=1000)
        assert event.type == "tts.speaking_start"

    def test_fields(self) -> None:
        event = TTSSpeakingStartEvent(request_id="req-abc", timestamp_ms=5000)
        assert event.request_id == "req-abc"
        assert event.timestamp_ms == 5000

    def test_serialization(self) -> None:
        event = TTSSpeakingStartEvent(request_id="r1", timestamp_ms=100)
        data = event.model_dump()
        assert data["type"] == "tts.speaking_start"
        assert data["request_id"] == "r1"
        assert data["timestamp_ms"] == 100


# ─── TTSSpeakingEndEvent ───


class TestTTSSpeakingEndEvent:
    def test_type_literal(self) -> None:
        event = TTSSpeakingEndEvent(request_id="r1", timestamp_ms=2000, duration_ms=1000)
        assert event.type == "tts.speaking_end"

    def test_defaults(self) -> None:
        event = TTSSpeakingEndEvent(request_id="r1", timestamp_ms=2000, duration_ms=1000)
        assert event.cancelled is False

    def test_cancelled_flag(self) -> None:
        event = TTSSpeakingEndEvent(
            request_id="r1",
            timestamp_ms=2000,
            duration_ms=500,
            cancelled=True,
        )
        assert event.cancelled is True

    def test_serialization(self) -> None:
        event = TTSSpeakingEndEvent(
            request_id="r1",
            timestamp_ms=5000,
            duration_ms=3000,
            cancelled=False,
        )
        data = event.model_dump()
        assert data["type"] == "tts.speaking_end"
        assert data["duration_ms"] == 3000
        assert data["cancelled"] is False


# ─── SessionConfigureCommand with model_tts ───


class TestSessionConfigureModelTts:
    def test_model_tts_default_none(self) -> None:
        cmd = SessionConfigureCommand()
        assert cmd.model_tts is None

    def test_model_tts_set(self) -> None:
        cmd = SessionConfigureCommand(model_tts="kokoro-v1")
        assert cmd.model_tts == "kokoro-v1"

    def test_model_tts_in_serialization(self) -> None:
        cmd = SessionConfigureCommand(model_tts="kokoro-v1")
        data = cmd.model_dump()
        assert data["model_tts"] == "kokoro-v1"


# ─── SessionConfig with model_tts ───


class TestSessionConfigModelTts:
    def test_model_tts_default_none(self) -> None:
        config = SessionConfig()
        assert config.model_tts is None

    def test_model_tts_set(self) -> None:
        config = SessionConfig(model_tts="kokoro-v1")
        assert config.model_tts == "kokoro-v1"

    def test_session_created_includes_model_tts(self) -> None:
        config = SessionConfig(model_tts="kokoro-v1")
        event = SessionCreatedEvent(
            session_id="sess-1",
            model="faster-whisper-tiny",
            config=config,
        )
        data = event.model_dump()
        assert data["config"]["model_tts"] == "kokoro-v1"


# ─── dispatch_message for TTS commands ───


class TestDispatchTTSCommands:
    def test_dispatch_tts_speak(self) -> None:
        msg = {"text": json.dumps({"type": "tts.speak", "text": "Hello"})}
        result = dispatch_message(msg)
        assert isinstance(result, CommandResult)
        assert isinstance(result.command, TTSSpeakCommand)
        assert result.command.text == "Hello"

    def test_dispatch_tts_speak_with_all_fields(self) -> None:
        msg = {
            "text": json.dumps(
                {
                    "type": "tts.speak",
                    "text": "Hello world",
                    "voice": "alloy",
                    "request_id": "req-123",
                }
            )
        }
        result = dispatch_message(msg)
        assert isinstance(result, CommandResult)
        cmd = result.command
        assert isinstance(cmd, TTSSpeakCommand)
        assert cmd.text == "Hello world"
        assert cmd.voice == "alloy"
        assert cmd.request_id == "req-123"

    def test_dispatch_tts_cancel(self) -> None:
        msg = {"text": json.dumps({"type": "tts.cancel"})}
        result = dispatch_message(msg)
        assert isinstance(result, CommandResult)
        assert isinstance(result.command, TTSCancelCommand)
        assert result.command.request_id is None

    def test_dispatch_tts_cancel_with_request_id(self) -> None:
        msg = {"text": json.dumps({"type": "tts.cancel", "request_id": "req-789"})}
        result = dispatch_message(msg)
        assert isinstance(result, CommandResult)
        cmd = result.command
        assert isinstance(cmd, TTSCancelCommand)
        assert cmd.request_id == "req-789"

    def test_dispatch_tts_speak_missing_text(self) -> None:
        msg = {"text": json.dumps({"type": "tts.speak"})}
        result = dispatch_message(msg)
        assert isinstance(result, ErrorResult)
        assert result.event.code == "validation_error"

    def test_dispatch_session_configure_with_model_tts(self) -> None:
        msg = {
            "text": json.dumps(
                {
                    "type": "session.configure",
                    "model_tts": "kokoro-v1",
                }
            )
        }
        result = dispatch_message(msg)
        assert isinstance(result, CommandResult)
        cmd = result.command
        assert isinstance(cmd, SessionConfigureCommand)
        assert cmd.model_tts == "kokoro-v1"


# ─── Union types include TTS ───


class TestUnionTypesIncludeTTS:
    def test_server_event_includes_tts_speaking_start(self) -> None:
        event: ServerEvent = TTSSpeakingStartEvent(request_id="r1", timestamp_ms=100)
        assert event.type == "tts.speaking_start"

    def test_server_event_includes_tts_speaking_end(self) -> None:
        event: ServerEvent = TTSSpeakingEndEvent(
            request_id="r1", timestamp_ms=200, duration_ms=100
        )
        assert event.type == "tts.speaking_end"

    def test_client_command_includes_tts_speak(self) -> None:
        cmd: ClientCommand = TTSSpeakCommand(text="Hello")
        assert cmd.type == "tts.speak"

    def test_client_command_includes_tts_cancel(self) -> None:
        cmd: ClientCommand = TTSCancelCommand()
        assert cmd.type == "tts.cancel"


# ─── Request ID auto-generation (consumer responsibility, but test the None default) ───


class TestRequestIdHandling:
    def test_speak_request_id_optional(self) -> None:
        cmd = TTSSpeakCommand(text="Hello")
        assert cmd.request_id is None

    def test_speak_request_id_preserves_value(self) -> None:
        rid = str(uuid.uuid4())
        cmd = TTSSpeakCommand(text="Hello", request_id=rid)
        assert cmd.request_id == rid

    def test_cancel_request_id_optional(self) -> None:
        cmd = TTSCancelCommand()
        assert cmd.request_id is None
