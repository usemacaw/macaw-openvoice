"""Typed exceptions for Macaw OpenVoice.

Hierarchy:
    MacawError (base)
    +-- ServiceNotConfiguredError
    +-- ConfigError
    |   +-- ManifestParseError
    |   +-- ManifestValidationError
    +-- ModelError
    |   +-- ModelNotFoundError
    |   +-- ModelLoadError
    +-- WorkerError
    |   +-- WorkerCrashError
    |   +-- WorkerTimeoutError
    |   +-- WorkerUnavailableError
    +-- AudioError
    |   +-- AudioFormatError
    |   +-- AudioTooLargeError
    +-- SessionError
    |   +-- SessionNotFoundError
    |   +-- SessionClosedError
    |   +-- InvalidTransitionError
    |   +-- BufferOverrunError
    +-- VoiceNotFoundError
    +-- TTSError
    |   +-- TTSSynthesisError (client input errors -> INVALID_ARGUMENT)
    |   +-- TTSEngineError   (server engine errors -> INTERNAL)
    +-- InvalidRequestError
"""

from __future__ import annotations


class MacawError(Exception):
    """Base for all Macaw OpenVoice exceptions."""


class ServiceNotConfiguredError(MacawError):
    """A required service (Registry, Scheduler, etc.) was not configured at startup.

    Raised by FastAPI dependencies when app.state is missing a required component.
    Maps to HTTP 503 (Service Unavailable) in error handlers.
    """

    def __init__(self, service_name: str) -> None:
        self.service_name = service_name
        super().__init__(
            f"{service_name} not configured. Pass {service_name.lower()}= in create_app()."
        )


# --- Configuration ---


class ConfigError(MacawError):
    """Runtime configuration error."""


class ManifestParseError(ConfigError):
    """Failed to parse macaw.yaml file."""

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"Failed to parse manifest '{path}': {reason}")


class ManifestValidationError(ConfigError):
    """Invalid macaw.yaml manifest (missing required fields, wrong types)."""

    def __init__(self, path: str, errors: list[str]) -> None:
        self.path = path
        self.errors = errors
        detail = "; ".join(errors)
        super().__init__(f"Manifest '{path}' is invalid: {detail}")


# --- Models ---


class ModelError(MacawError):
    """Model-related error."""


class ModelNotFoundError(ModelError):
    """Model not found in the registry."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        super().__init__(f"Model '{model_name}' not found in the registry")


class ModelLoadError(ModelError):
    """Failed to load model into memory."""

    def __init__(self, model_name: str, reason: str) -> None:
        self.model_name = model_name
        self.reason = reason
        super().__init__(f"Failed to load model '{model_name}': {reason}")


# --- Worker ---


class WorkerError(MacawError):
    """Worker-related error (gRPC subprocesses)."""


class WorkerCrashError(WorkerError):
    """Worker crashed during operation."""

    def __init__(self, worker_id: str, exit_code: int | None = None) -> None:
        self.worker_id = worker_id
        self.exit_code = exit_code
        msg = f"Worker '{worker_id}' crashed"
        if exit_code is not None:
            msg += f" (exit code: {exit_code})"
        super().__init__(msg)


class WorkerTimeoutError(WorkerError):
    """Worker did not respond within the timeout."""

    def __init__(self, worker_id: str, timeout_seconds: float) -> None:
        self.worker_id = worker_id
        self.timeout_seconds = timeout_seconds
        super().__init__(f"Worker '{worker_id}' did not respond within {timeout_seconds}s")


class WorkerUnavailableError(WorkerError):
    """No worker available to serve the request."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        super().__init__(f"No worker available for model '{model_name}'")


# --- Audio ---


class AudioError(MacawError):
    """Audio processing error."""


class AudioFormatError(AudioError):
    """Unsupported or invalid audio format."""

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(f"Invalid audio format: {detail}")


class AudioTooLargeError(AudioError):
    """Audio file exceeds the allowed limit."""

    def __init__(self, size_bytes: int, max_bytes: int) -> None:
        self.size_bytes = size_bytes
        self.max_bytes = max_bytes
        size_mb = size_bytes / (1024 * 1024)
        max_mb = max_bytes / (1024 * 1024)
        super().__init__(f"Audio file ({size_mb:.1f}MB) exceeds the {max_mb:.1f}MB limit")


# --- Session ---


class SessionError(MacawError):
    """Streaming session error."""


class SessionNotFoundError(SessionError):
    """Session not found."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"Session '{session_id}' not found")


class SessionClosedError(SessionError):
    """Operation attempted on a closed session."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"Session '{session_id}' is already closed")


class InvalidTransitionError(SessionError):
    """Invalid state transition in the session state machine."""

    def __init__(self, from_state: str, to_state: str) -> None:
        self.from_state = from_state
        self.to_state = to_state
        super().__init__(f"Invalid transition: {from_state} -> {to_state}")


class BufferOverrunError(SessionError):
    """Attempt to read data already overwritten or beyond the ring buffer write."""


# --- Request ---


class InvalidRequestError(MacawError):
    """Invalid request parameter."""

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


# --- TTS ---


class VoiceNotFoundError(MacawError):
    """Referenced saved voice not found in VoiceStore."""

    def __init__(self, voice_id: str) -> None:
        self.voice_id = voice_id
        super().__init__(f"Saved voice '{voice_id}' not found")


class TTSError(MacawError):
    """Speech synthesis (TTS) error."""


class TTSSynthesisError(TTSError):
    """Client-side synthesis failure (bad input, missing params).

    Maps to gRPC INVALID_ARGUMENT in the TTS servicer.
    """

    def __init__(self, model_name: str, reason: str) -> None:
        self.model_name = model_name
        self.reason = reason
        super().__init__(f"TTS synthesis failed for model '{model_name}': {reason}")


class TTSEngineError(TTSError):
    """Server-side engine failure (GPU OOM, empty audio output, unexpected crash).

    Maps to gRPC INTERNAL in the TTS servicer (falls through to the
    generic except branch, which already uses INTERNAL).
    """

    def __init__(self, model_name: str, reason: str) -> None:
        self.model_name = model_name
        self.reason = reason
        super().__init__(f"TTS engine error for model '{model_name}': {reason}")
