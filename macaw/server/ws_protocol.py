"""Protocol handler para dispatch de mensagens WebSocket.

Recebe mensagens raw do WebSocket (dict com 'bytes' ou 'text') e retorna
um resultado tipado: audio bytes, comando parseado, ou evento de erro.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pydantic

from macaw.logging import get_logger
from macaw.server.models.events import (
    InputAudioBufferCommitCommand,
    SessionCancelCommand,
    SessionCloseCommand,
    SessionConfigureCommand,
    StreamingErrorEvent,
    TTSCancelCommand,
    TTSSpeakCommand,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from macaw.server.models.events import ClientCommand

logger = get_logger("server.ws_protocol")

# Mapeamento de type -> classe de comando
_COMMAND_TYPES: dict[str, type[ClientCommand]] = {
    "session.configure": SessionConfigureCommand,
    "session.cancel": SessionCancelCommand,
    "session.close": SessionCloseCommand,
    "input_audio_buffer.commit": InputAudioBufferCommitCommand,
    "tts.speak": TTSSpeakCommand,
    "tts.cancel": TTSCancelCommand,
}


@dataclass(frozen=True, slots=True)
class AudioFrameResult:
    """Resultado de dispatch: frame de audio binario."""

    data: bytes


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Resultado de dispatch: comando JSON parseado."""

    command: ClientCommand


@dataclass(frozen=True, slots=True)
class ErrorResult:
    """Resultado de dispatch: erro de parsing/validacao."""

    event: StreamingErrorEvent


# Union type para resultado de dispatch
DispatchResult = AudioFrameResult | CommandResult | ErrorResult


def dispatch_message(message: Mapping[str, Any]) -> DispatchResult | None:
    """Dispatch de mensagem WebSocket raw para resultado tipado.

    Args:
        message: Dict raw do ``websocket.receive()`` com chaves 'bytes' ou 'text'.

    Returns:
        ``AudioFrameResult`` para frames binarios, ``CommandResult`` para JSON
        parseado, ``ErrorResult`` para erros, ou ``None`` se a mensagem nao
        contem bytes nem text.
    """
    # Binary frame: audio data
    raw_bytes = message.get("bytes")
    if raw_bytes is not None:
        if not isinstance(raw_bytes, bytes):
            return ErrorResult(
                event=StreamingErrorEvent(
                    code="invalid_frame",
                    message="Binary frame data is not bytes",
                    recoverable=True,
                ),
            )
        return AudioFrameResult(data=raw_bytes)

    # Text frame: JSON command
    raw_text = message.get("text")
    if raw_text is not None:
        return _parse_command(str(raw_text))

    return None


def _parse_command(raw_text: str) -> CommandResult | ErrorResult:
    """Parseia texto JSON em um comando tipado.

    Fluxo:
        1. Deserializa JSON.
        2. Extrai campo ``type``.
        3. Valida contra o modelo Pydantic correto.

    Erros sao retornados como ``ErrorResult`` com ``recoverable=True``
    (conexao nao deve ser fechada).
    """
    # 1. Parse JSON
    try:
        data = json.loads(raw_text)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("malformed_json", error=str(exc), raw=raw_text[:200])
        return ErrorResult(
            event=StreamingErrorEvent(
                code="malformed_json",
                message=f"Invalid JSON: {exc}",
                recoverable=True,
            ),
        )

    if not isinstance(data, dict):
        logger.warning("invalid_command_format", raw=raw_text[:200])
        return ErrorResult(
            event=StreamingErrorEvent(
                code="malformed_json",
                message="Expected JSON object, got " + type(data).__name__,
                recoverable=True,
            ),
        )

    # 2. Extrair type
    command_type = data.get("type")
    if command_type is None:
        logger.warning("missing_type_field", data_keys=list(data.keys()))
        return ErrorResult(
            event=StreamingErrorEvent(
                code="unknown_command",
                message="Missing required field: 'type'",
                recoverable=True,
            ),
        )

    # 3. Lookup command class
    command_class = _COMMAND_TYPES.get(command_type)
    if command_class is None:
        logger.warning("unknown_command_type", command_type=command_type)
        return ErrorResult(
            event=StreamingErrorEvent(
                code="unknown_command",
                message=f"Unknown command type: '{command_type}'",
                recoverable=True,
            ),
        )

    # 4. Validate with Pydantic model
    try:
        command = command_class.model_validate(data)
    except pydantic.ValidationError as exc:
        logger.warning(
            "command_validation_error",
            command_type=command_type,
            error=str(exc),
        )
        return ErrorResult(
            event=StreamingErrorEvent(
                code="validation_error",
                message=f"Validation error for '{command_type}': {exc}",
                recoverable=True,
            ),
        )

    return CommandResult(command=command)
