"""Pydantic models for the multi-context TTS WebSocket protocol.

Client -> Server messages and Server -> Client events for the
``/v1/text-to-speech/{voice_id}/multi-stream-input`` endpoint.
See ADR-008 for protocol design.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Client -> Server messages
# ---------------------------------------------------------------------------


class InitializeConnection(BaseModel):
    """Open the multi-context connection."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    type: Literal["initialize_connection"] = "initialize_connection"
    model_tts: str | None = Field(
        default=None,
        description="Override TTS model (uses default if omitted).",
    )
    voice_settings: dict[str, object] | None = Field(
        default=None,
        description="Voice settings (stability, similarity_boost, etc).",
    )


class SendText(BaseModel):
    """Send text to a specific context."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    type: Literal["send_text"] = "send_text"
    context_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Unique identifier for the TTS context.",
    )
    text: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="Text chunk to append to the context buffer.",
    )


class FlushContext(BaseModel):
    """Flush remaining text in a context buffer and synthesize."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    type: Literal["flush_context"] = "flush_context"
    context_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
    )


class CloseContext(BaseModel):
    """Close a specific context."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    type: Literal["close_context"] = "close_context"
    context_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
    )


class CloseSocket(BaseModel):
    """Close all contexts and disconnect."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    type: Literal["close_socket"] = "close_socket"


class KeepAlive(BaseModel):
    """Reset inactivity timeout for the connection."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    type: Literal["keep_alive"] = "keep_alive"


# Union of all client messages
MultiContextClientMessage = (
    InitializeConnection | SendText | FlushContext | CloseContext | CloseSocket | KeepAlive
)

# Mapping of type -> model class for dispatch
MULTI_CONTEXT_MESSAGE_TYPES: dict[str, type[MultiContextClientMessage]] = {
    "initialize_connection": InitializeConnection,
    "send_text": SendText,
    "flush_context": FlushContext,
    "close_context": CloseContext,
    "close_socket": CloseSocket,
    "keep_alive": KeepAlive,
}

# ---------------------------------------------------------------------------
# Server -> Client events
# ---------------------------------------------------------------------------


class ConnectionInitializedEvent(BaseModel):
    """Acknowledgement after connection initialization."""

    type: Literal["connection_initialized"] = "connection_initialized"
    model_tts: str
    max_contexts: int


class AudioEvent(BaseModel):
    """Audio chunk from a specific TTS context."""

    type: Literal["audio"] = "audio"
    context_id: str
    audio: str = Field(description="Base64-encoded audio bytes.")


class ContextFlushedEvent(BaseModel):
    """Acknowledgement that a context's buffer was flushed."""

    type: Literal["context_flushed"] = "context_flushed"
    context_id: str


class ContextClosedEvent(BaseModel):
    """Acknowledgement that a context was closed."""

    type: Literal["context_closed"] = "context_closed"
    context_id: str


class IsFinalEvent(BaseModel):
    """Context synthesis complete."""

    type: Literal["is_final"] = "is_final"
    context_id: str


class MultiContextErrorEvent(BaseModel):
    """Error event, optionally scoped to a context."""

    type: Literal["error"] = "error"
    message: str
    context_id: str | None = None
    code: str = "internal_error"
    recoverable: bool = True


# Union of all server events
MultiContextServerEvent = (
    ConnectionInitializedEvent
    | AudioEvent
    | ContextFlushedEvent
    | ContextClosedEvent
    | IsFinalEvent
    | MultiContextErrorEvent
)
