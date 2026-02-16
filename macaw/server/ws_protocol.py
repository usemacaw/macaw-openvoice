"""Protocol handler for WebSocket message dispatch.

Receives raw WebSocket messages (dict with 'bytes' or 'text') and returns
a typed result: audio bytes, parsed command, or error event.
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

# Mapping of type -> command class
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
    """Dispatch result: binary audio frame."""

    data: bytes


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Dispatch result: parsed JSON command."""

    command: ClientCommand


@dataclass(frozen=True, slots=True)
class ErrorResult:
    """Dispatch result: parsing/validation error."""

    event: StreamingErrorEvent


# Union type for dispatch result
DispatchResult = AudioFrameResult | CommandResult | ErrorResult


def dispatch_message(message: Mapping[str, Any]) -> DispatchResult | None:
    """Dispatch raw WebSocket message to typed result.

    Args:
        message: Raw dict from ``websocket.receive()`` with 'bytes' or 'text' keys.

    Returns:
        ``AudioFrameResult`` for binary frames, ``CommandResult`` for parsed JSON,
        ``ErrorResult`` for errors, or ``None`` if the message contains
        neither bytes nor text.
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
    """Parse JSON text into a typed command.

    Flow:
        1. Deserialize JSON.
        2. Extract ``type`` field.
        3. Validate against the correct Pydantic model.

    Errors are returned as ``ErrorResult`` with ``recoverable=True``
    (connection should not be closed).
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

    # 2. Extract type
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
