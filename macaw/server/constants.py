"""Shared server constants.

Configurable values (``MAX_FILE_SIZE_BYTES``, ``TTS_GRPC_TIMEOUT``, etc.)
are resolved lazily from ``macaw.config.settings`` via PEP 562 ``__getattr__``.
All existing ``from macaw.server.constants import X`` imports keep working.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from macaw._audio_constants import TTS_DEFAULT_SAMPLE_RATE as TTS_DEFAULT_SAMPLE_RATE

# --- Static constants (not configurable via env) ---

ALLOWED_AUDIO_CONTENT_TYPES = frozenset(
    {
        "audio/wav",
        "audio/x-wav",
        "audio/wave",
        "audio/mpeg",
        "audio/mp3",
        "audio/flac",
        "audio/ogg",
        "audio/webm",
        "audio/x-flac",
        "application/octet-stream",  # generic fallback
    }
)

# Saved voice prefix convention.
SAVED_VOICE_PREFIX = "voice_"

# Default voice name sent to engine when a saved voice is resolved.
DEFAULT_VOICE_NAME = "default"

# --- Validation constants (shared between models and routes) ---

# Maximum model name length for form validation.
MODEL_NAME_MAX_LENGTH = 256

# Temperature defaults for STT inference.
DEFAULT_TEMPERATURE: float = 0.0
MAX_TEMPERATURE: float = 2.0

# --- WebSocket close codes (RFC 6455) ---
WS_CLOSE_NORMAL = 1000
WS_CLOSE_POLICY_VIOLATION = 1008


# --- Configurable via environment / .env (delegated to Settings) ---

# Mapping from module-level attribute names to Settings paths.
_SETTINGS_MAP: dict[str, tuple[str, str]] = {
    "MAX_FILE_SIZE_BYTES": ("server", "max_file_size_bytes"),
    "TTS_GRPC_TIMEOUT": ("tts", "grpc_timeout_s"),
    "TTS_LIST_VOICES_TIMEOUT": ("tts", "list_voices_timeout_s"),
    "TTS_MAX_TEXT_LENGTH": ("tts", "max_text_length"),
    "RETRY_AFTER_SECONDS": ("server", "retry_after_s"),
    "MAX_WS_FRAME_SIZE": ("server", "ws_max_frame_size_bytes"),
    "WS_INACTIVITY_TIMEOUT_S": ("server", "ws_inactivity_timeout_s"),
    "WS_HEARTBEAT_INTERVAL_S": ("server", "ws_heartbeat_interval_s"),
    "WS_CHECK_INTERVAL_S": ("server", "ws_check_interval_s"),
}


def __getattr__(name: str) -> object:
    """PEP 562 — resolve configurable constants from Settings on first access."""
    if name not in _SETTINGS_MAP:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)

    from macaw.config.settings import get_settings

    settings = get_settings()
    section_name, field_name = _SETTINGS_MAP[name]
    section = getattr(settings, section_name)
    return getattr(section, field_name)


if TYPE_CHECKING:
    # Static declarations so mypy / IDE autocomplete sees the types.
    MAX_FILE_SIZE_BYTES: int
    TTS_GRPC_TIMEOUT: float
    TTS_LIST_VOICES_TIMEOUT: float
    TTS_MAX_TEXT_LENGTH: int
    RETRY_AFTER_SECONDS: str
    MAX_WS_FRAME_SIZE: int
    WS_INACTIVITY_TIMEOUT_S: float
    WS_HEARTBEAT_INTERVAL_S: float
    WS_CHECK_INTERVAL_S: float
