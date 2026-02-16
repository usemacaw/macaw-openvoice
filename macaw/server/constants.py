"""Constantes compartilhadas do server."""

from __future__ import annotations

MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25MB

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
        "application/octet-stream",  # fallback generico
    }
)

# gRPC channel options for TTS workers (30MB max message size)
TTS_GRPC_CHANNEL_OPTIONS = [
    ("grpc.max_send_message_length", 30 * 1024 * 1024),
    ("grpc.max_receive_message_length", 30 * 1024 * 1024),
]

# Timeout for the gRPC Synthesize call (seconds)
TTS_GRPC_TIMEOUT = 60.0

# Default sample rate for TTS engines (24kHz is the standard for Kokoro, Qwen3-TTS, etc.)
TTS_DEFAULT_SAMPLE_RATE: int = 24000
