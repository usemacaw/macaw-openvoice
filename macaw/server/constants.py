"""Shared server constants."""

from __future__ import annotations

import os

from macaw._audio_constants import TTS_DEFAULT_SAMPLE_RATE as TTS_DEFAULT_SAMPLE_RATE

# --- Configurable via environment variables ---

# Maximum audio file upload size (bytes). Default: 25MB.
MAX_FILE_SIZE_BYTES = int(os.environ.get("MACAW_MAX_FILE_SIZE_MB", "25")) * 1024 * 1024

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

# Timeout for the gRPC Synthesize RPC (seconds). Default: 60s.
TTS_GRPC_TIMEOUT = float(os.environ.get("MACAW_TTS_GRPC_TIMEOUT_S", "60.0"))

# Timeout for the gRPC ListVoices RPC (seconds). Default: 10s.
TTS_LIST_VOICES_TIMEOUT = float(os.environ.get("MACAW_TTS_LIST_VOICES_TIMEOUT_S", "10.0"))

# Retry-After header value for 503/502 responses (seconds).
RETRY_AFTER_SECONDS = os.environ.get("MACAW_RETRY_AFTER_S", "5")

# Maximum WebSocket binary frame size (bytes). Default: 1MB.
MAX_WS_FRAME_SIZE = int(os.environ.get("MACAW_MAX_WS_FRAME_SIZE_BYTES", "1048576"))

# WebSocket session timeouts (seconds).
WS_INACTIVITY_TIMEOUT_S = float(os.environ.get("MACAW_WS_INACTIVITY_TIMEOUT_S", "60.0"))
WS_HEARTBEAT_INTERVAL_S = float(os.environ.get("MACAW_WS_HEARTBEAT_INTERVAL_S", "10.0"))
WS_CHECK_INTERVAL_S = float(os.environ.get("MACAW_WS_CHECK_INTERVAL_S", "5.0"))

# Saved voice prefix convention.
SAVED_VOICE_PREFIX = "voice_"

# Default voice name sent to engine when a saved voice is resolved.
DEFAULT_VOICE_NAME = "default"

# --- Validation constants (shared between models and routes) ---

# Maximum text length for TTS synthesis (shared by SpeechRequest and TTSSpeakCommand).
TTS_MAX_TEXT_LENGTH = 4096

# Maximum model name length for form validation.
MODEL_NAME_MAX_LENGTH = 256

# Temperature defaults for STT inference.
DEFAULT_TEMPERATURE: float = 0.0
MAX_TEMPERATURE: float = 2.0

# --- WebSocket close codes (RFC 6455) ---
WS_CLOSE_NORMAL = 1000
WS_CLOSE_POLICY_VIOLATION = 1008
