"""Testes unitarios do protocol handler WebSocket."""

from __future__ import annotations

import json

from macaw.server.ws_protocol import (
    AudioFrameResult,
    CommandResult,
    ErrorResult,
    dispatch_message,
)


class TestBinaryFrameDispatch:
    """Dispatch de frames binarios (audio)."""

    def test_binary_message_returns_audio_frame_result(self) -> None:
        """Mensagem com bytes retorna AudioFrameResult com os dados."""
        audio_data = b"\x00\x01\x02\x03" * 100
        result = dispatch_message({"bytes": audio_data})

        assert isinstance(result, AudioFrameResult)
        assert result.data == audio_data

    def test_empty_bytes_returns_audio_frame_result(self) -> None:
        """Bytes vazios ainda sao um frame valido."""
        result = dispatch_message({"bytes": b""})

        assert isinstance(result, AudioFrameResult)
        assert result.data == b""


class TestCommandDispatch:
    """Dispatch de comandos JSON."""

    def test_session_configure_parses_correctly(self) -> None:
        """session.configure e parseado para SessionConfigureCommand."""
        msg = {"text": json.dumps({"type": "session.configure", "language": "pt"})}
        result = dispatch_message(msg)

        assert isinstance(result, CommandResult)
        assert result.command.type == "session.configure"
        assert result.command.language == "pt"  # type: ignore[union-attr]

    def test_session_configure_with_vad_sensitivity(self) -> None:
        """session.configure aceita vad_sensitivity."""
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
        """session.cancel e parseado para SessionCancelCommand."""
        msg = {"text": json.dumps({"type": "session.cancel"})}
        result = dispatch_message(msg)

        assert isinstance(result, CommandResult)
        assert result.command.type == "session.cancel"

    def test_session_close_parses_correctly(self) -> None:
        """session.close e parseado para SessionCloseCommand."""
        msg = {"text": json.dumps({"type": "session.close"})}
        result = dispatch_message(msg)

        assert isinstance(result, CommandResult)
        assert result.command.type == "session.close"

    def test_input_audio_buffer_commit_parses_correctly(self) -> None:
        """input_audio_buffer.commit e parseado para InputAudioBufferCommitCommand."""
        msg = {"text": json.dumps({"type": "input_audio_buffer.commit"})}
        result = dispatch_message(msg)

        assert isinstance(result, CommandResult)
        assert result.command.type == "input_audio_buffer.commit"


class TestErrorHandling:
    """Erros de parsing e validacao."""

    def test_malformed_json_returns_error_result(self) -> None:
        """JSON invalido retorna ErrorResult com recoverable=True."""
        msg = {"text": "this is not json {{{"}
        result = dispatch_message(msg)

        assert isinstance(result, ErrorResult)
        assert result.event.code == "malformed_json"
        assert result.event.recoverable is True

    def test_json_array_returns_error_result(self) -> None:
        """JSON que nao e objeto retorna ErrorResult."""
        msg = {"text": "[1, 2, 3]"}
        result = dispatch_message(msg)

        assert isinstance(result, ErrorResult)
        assert result.event.code == "malformed_json"
        assert result.event.recoverable is True

    def test_unknown_command_type_returns_error_result(self) -> None:
        """Tipo de comando desconhecido retorna ErrorResult com recoverable=True."""
        msg = {"text": json.dumps({"type": "unknown.command"})}
        result = dispatch_message(msg)

        assert isinstance(result, ErrorResult)
        assert result.event.code == "unknown_command"
        assert "unknown.command" in result.event.message
        assert result.event.recoverable is True

    def test_missing_type_field_returns_error_result(self) -> None:
        """JSON sem campo type retorna ErrorResult com recoverable=True."""
        msg = {"text": json.dumps({"language": "pt"})}
        result = dispatch_message(msg)

        assert isinstance(result, ErrorResult)
        assert result.event.code == "unknown_command"
        assert "type" in result.event.message.lower()
        assert result.event.recoverable is True

    def test_validation_error_returns_error_result(self) -> None:
        """Campo com valor invalido retorna ErrorResult com recoverable=True."""
        # VADSensitivity e um enum com valores validos: high, normal, low.
        # Um valor fora do enum causa ValidationError no Pydantic.
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
    """Casos especiais de dispatch."""

    def test_message_with_neither_bytes_nor_text_returns_none(self) -> None:
        """Mensagem sem bytes nem text retorna None."""
        result = dispatch_message({"type": "websocket.disconnect"})
        assert result is None

    def test_message_with_none_bytes_and_none_text(self) -> None:
        """Mensagem com bytes=None e text=None retorna None."""
        result = dispatch_message({"bytes": None, "text": None})
        assert result is None

    def test_empty_dict_returns_none(self) -> None:
        """Dict vazio retorna None."""
        result = dispatch_message({})
        assert result is None

    def test_bytes_takes_priority_over_text(self) -> None:
        """Se mensagem tem bytes e text, bytes tem prioridade."""
        audio_data = b"\x00\x01"
        result = dispatch_message(
            {
                "bytes": audio_data,
                "text": json.dumps({"type": "session.close"}),
            }
        )

        assert isinstance(result, AudioFrameResult)
        assert result.data == audio_data
