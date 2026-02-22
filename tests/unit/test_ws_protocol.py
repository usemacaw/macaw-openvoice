"""Unit tests for the WebSocket protocol handler."""

from __future__ import annotations

import json

from macaw.server.ws_protocol import (
    AudioFrameResult,
    CommandResult,
    ErrorResult,
    dispatch_message,
)


class TestBinaryFrameDispatch:
    """Dispatch of binary frames (audio)."""

    def test_binary_message_returns_audio_frame_result(self) -> None:
        """Message with bytes returns AudioFrameResult with the data."""
        audio_data = b"\x00\x01\x02\x03" * 100
        result = dispatch_message({"bytes": audio_data})

        assert isinstance(result, AudioFrameResult)
        assert result.data == audio_data

    def test_empty_bytes_returns_audio_frame_result(self) -> None:
        """Empty bytes are still a valid frame."""
        result = dispatch_message({"bytes": b""})

        assert isinstance(result, AudioFrameResult)
        assert result.data == b""


class TestCommandDispatch:
    """Dispatch of JSON commands."""

    def test_session_configure_parses_correctly(self) -> None:
        """session.configure is parsed to SessionConfigureCommand."""
        msg = {"text": json.dumps({"type": "session.configure", "language": "pt"})}
        result = dispatch_message(msg)

        assert isinstance(result, CommandResult)
        assert result.command.type == "session.configure"
        assert result.command.language == "pt"  # type: ignore[union-attr]

    def test_session_configure_with_vad_sensitivity(self) -> None:
        """session.configure accepts vad_sensitivity."""
        msg = {
            "text": json.dumps(
                {
                    "type": "session.configure",
                    "vad_sensitivity": "high",
                    "silence_timeout_ms": 500,
                }
            )
        }
        result = dispatch_message(msg)

        assert isinstance(result, CommandResult)
        assert result.command.type == "session.configure"
        assert result.command.vad_sensitivity.value == "high"  # type: ignore[union-attr]

    def test_session_cancel_parses_correctly(self) -> None:
        """session.cancel is parsed to SessionCancelCommand."""
        msg = {"text": json.dumps({"type": "session.cancel"})}
        result = dispatch_message(msg)

        assert isinstance(result, CommandResult)
        assert result.command.type == "session.cancel"

    def test_session_close_parses_correctly(self) -> None:
        """session.close is parsed to SessionCloseCommand."""
        msg = {"text": json.dumps({"type": "session.close"})}
        result = dispatch_message(msg)

        assert isinstance(result, CommandResult)
        assert result.command.type == "session.close"

    def test_input_audio_buffer_commit_parses_correctly(self) -> None:
        """input_audio_buffer.commit is parsed to InputAudioBufferCommitCommand."""
        msg = {"text": json.dumps({"type": "input_audio_buffer.commit"})}
        result = dispatch_message(msg)

        assert isinstance(result, CommandResult)
        assert result.command.type == "input_audio_buffer.commit"


class TestErrorHandling:
    """Parsing and validation errors."""

    def test_malformed_json_returns_error_result(self) -> None:
        """Invalid JSON returns ErrorResult with recoverable=True."""
        msg = {"text": "this is not json {{{"}
        result = dispatch_message(msg)

        assert isinstance(result, ErrorResult)
        assert result.event.code == "malformed_json"
        assert result.event.recoverable is True

    def test_json_array_returns_error_result(self) -> None:
        """JSON that is not an object returns ErrorResult."""
        msg = {"text": "[1, 2, 3]"}
        result = dispatch_message(msg)

        assert isinstance(result, ErrorResult)
        assert result.event.code == "malformed_json"
        assert result.event.recoverable is True

    def test_unknown_command_type_returns_error_result(self) -> None:
        """Unknown command type returns ErrorResult with recoverable=True."""
        msg = {"text": json.dumps({"type": "unknown.command"})}
        result = dispatch_message(msg)

        assert isinstance(result, ErrorResult)
        assert result.event.code == "unknown_command"
        assert "unknown.command" in result.event.message
        assert result.event.recoverable is True

    def test_missing_type_field_returns_error_result(self) -> None:
        """JSON without type field returns ErrorResult with recoverable=True."""
        msg = {"text": json.dumps({"language": "pt"})}
        result = dispatch_message(msg)

        assert isinstance(result, ErrorResult)
        assert result.event.code == "unknown_command"
        assert "type" in result.event.message.lower()
        assert result.event.recoverable is True

    def test_validation_error_returns_error_result(self) -> None:
        """Field with invalid value returns ErrorResult with recoverable=True."""
        # VADSensitivity is an enum with valid values: high, normal, low.
        # A value outside the enum causes a Pydantic ValidationError.
        msg = {
            "text": json.dumps(
                {
                    "type": "session.configure",
                    "vad_sensitivity": "INVALID_SENSITIVITY",
                }
            )
        }
        result = dispatch_message(msg)

        assert isinstance(result, ErrorResult)
        assert result.event.code == "validation_error"
        assert result.event.recoverable is True


class TestEdgeCases:
    """Special dispatch cases."""

    def test_message_with_neither_bytes_nor_text_returns_none(self) -> None:
        """Message with neither bytes nor text returns None."""
        result = dispatch_message({"type": "websocket.disconnect"})
        assert result is None

    def test_message_with_none_bytes_and_none_text(self) -> None:
        """Message with bytes=None and text=None returns None."""
        result = dispatch_message({"bytes": None, "text": None})
        assert result is None

    def test_empty_dict_returns_none(self) -> None:
        """Empty dict returns None."""
        result = dispatch_message({})
        assert result is None

    def test_bytes_takes_priority_over_text(self) -> None:
        """If message has both bytes and text, bytes takes priority."""
        audio_data = b"\x00\x01"
        result = dispatch_message(
            {
                "bytes": audio_data,
                "text": json.dumps({"type": "session.close"}),
            }
        )

        assert isinstance(result, AudioFrameResult)
        assert result.data == audio_data
